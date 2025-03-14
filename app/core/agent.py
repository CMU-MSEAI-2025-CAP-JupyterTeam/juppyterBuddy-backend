"""
JupyterBuddy Agent Module

This module defines the core LLM-based agent that interacts with JupyterLab notebooks.
1. The LLM decides which tools to call based on user input.
2. If tools are needed, it **forwards the request to the frontend for execution**.
3. Tool execution results/errors are sent back to the LLM for handling.
4. If no tools are needed, the LLM **immediately responds to the user**.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, TypedDict, Callable

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# Local imports
from app.core.llm import get_llm
from app.models.conversationModel import Conversation, MessageRole

# Set up logging
logger = logging.getLogger(__name__)

# Type definitions for state management
class AgentState(TypedDict):
    """State structure for the agent's execution loop."""
    notebook_context: Optional[Dict[str, Any]]
    messages: List[BaseMessage]
    output_to_user: Optional[str]
    actions: Optional[List[Dict[str, Any]]]
    error: Optional[Dict[str, str]]  # { "error_message": "msg" }
    waiting_for_frontend: bool  # True when waiting for frontend response
    end_agent_execution: bool  # True when LLM sends final response

# Define system prompt template for the agent
SYSTEM_PROMPT = """You are JupyterBuddy, an intelligent assistant integrated in JupyterLab.
You help users with their Jupyter notebooks by answering questions and taking actions.

You can:
1. Create new code or markdown cells.
2. Execute cells.
3. Update existing cells.
4. Get notebook metadata.

When working with users:
- Be concise and helpful.
- Explain your reasoning.
- Suggest best practices for coding and data science.

Handle errors by:
- Explaining issues in clear language.
- Suggesting solutions when possible.
- Continuing execution if errors are recoverable.

Current notebook context:
{notebook_context}
"""

class JupyterBuddyAgent:
    """
    Handles interaction with the LLM and manages execution loops based on tool calls.
    """
    
    def __init__(self, 
                 send_response_callback: Callable[[Dict[str, Any]], None],
                 send_action_callback: Callable[[Dict[str, Any]], None]):
        """
        Initializes the agent with WebSocket callbacks.
        
        Args:
            send_response_callback: Sends messages back to the user.
            send_action_callback: Sends actions to the frontend.
        """
        self.llm = get_llm()
        self.send_response = send_response_callback
        self.send_action = send_action_callback
        
        # Store conversation history
        self.latest_state = None  # Store state across calls

        # Create execution graph
        self.create_agent_graph()
        
    def create_agent_graph(self):
        """Creates the structured execution graph."""
        workflow = StateGraph(AgentState)

        # LLM Response Processing Node
        workflow.add_node("llm_node", self.llm_node)

        # Conditional Edges
        workflow.add_conditional_edges(
            "llm_node",
            self.should_wait_for_frontend,
            {
                True: END,  # Wait for frontend execution
                False: END  # End execution if no tool was called
            }
        )

        workflow.set_entry_point("llm_node")
        self.graph = workflow.compile()

    def llm_node(self, state: AgentState) -> AgentState:
        """
        Processes LLM output and determines next steps.
        
        Args:
            state: The current agent execution state.
            
        Returns:
            Updated agent state.
        """
        messages = state["messages"]
        response = self.llm.invoke(messages)

        # Extract tool calls
        tool_calls = response.additional_kwargs.get("tool_calls", [])

        # Update conversation history
        updated_messages = messages + [response]

        if tool_calls:
            actions = [{"tool_name": call["name"], "parameters": call["args"]} for call in tool_calls]
            updated_state = {
                **state,
                "messages": updated_messages,
                "actions": actions,
                "waiting_for_frontend": True,
                "end_agent_execution": False,
                "error": None  # Reset error on new action execution
            }
            self.send_action({"message": response.content, "actions": actions})
            return updated_state

        # If no tools are called, respond to user
        output_to_user = response.content
        if output_to_user and output_to_user.strip():
            self.send_response({"message": output_to_user, "actions": None})

        return {
            **state,
            "messages": updated_messages,
            "actions": None,
            "waiting_for_frontend": False,
            "end_agent_execution": True,
            "error": None
        }

    def should_wait_for_frontend(self, state: AgentState) -> bool:
        """Determines if execution should pause for frontend feedback."""
        return state["waiting_for_frontend"]
    
    def handle_agent_input(self, session_id: str, data: Dict[str, Any]):
        """
        Handles both **user messages** and **frontend execution results**.
        
        Args:
            session_id: The user session ID.
            data: Input message or frontend action result.
        """
        # **Check if existing state exists**
        if self.latest_state:
            current_state = self.latest_state.copy()
        else:
            current_state = {
                "notebook_context": None,
                "messages": [],
                "output_to_user": None,
                "actions": None,
                "error": None,
                "waiting_for_frontend": False,
                "end_agent_execution": False
            }

        if data.get("type") == "user_message":
            user_message = data["data"]
            notebook_context = data.get("notebook_context")
            
            # Append user message to conversation
            current_state["messages"].append(HumanMessage(content=user_message))
            current_state["notebook_context"] = notebook_context

            # Start execution
            self.latest_state = self.graph.invoke(current_state)

        elif data.get("type") == "action_result":
            action_result = data["data"]

            # Reset waiting state and handle errors if any
            current_state["waiting_for_frontend"] = False

            if action_result.get("error"):
                current_state["error"] = {"error_message": action_result["error"]}
            else:
                current_state["error"] = None

            # Continue execution with updated state
            self.latest_state = self.graph.invoke(current_state)
