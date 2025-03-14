"""
JupyterBuddy Agent Module

This module defines the core LLM-based agent that can take actions in JupyterLab notebooks
through a conversational interface. The agent follows a structured execution approach:
1. The LLM decides which tools to call based on user intent.
2. If tools are called, they are **forwarded to the frontend for execution**.
3. Tool outputs (or errors) are sent back to the LLM, allowing it to retry or generate a response.
4. The LLM responds to the user **only when no further tool execution is needed**.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, TypedDict, Callable

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# Local imports
from app.core.llm import get_llm
from app.models.conversation import Conversation, MessageRole

# Set up logging
logger = logging.getLogger(__name__)

# Type definitions for state management
class AgentState(TypedDict):
    """Type definition for the agent's state"""
    notebook_context: Optional[Dict[str, Any]]
    messages: List[BaseMessage]
    output_to_user: Optional[str]
    actions: Optional[List[Dict[str, Any]]]
    error: Optional[Dict[str, str]]  # { "error_message": "msg" }

# Define system prompt template for the agent
SYSTEM_PROMPT = """You are JupyterBuddy, an intelligent assistant integrated directly in JupyterLab.
You help users by answering questions about their Jupyter notebooks and taking actions on their behalf.

You have the ability to:
1. Create new code or markdown cells.
2. Execute cells.
3. Update existing cells.
4. Get information about the current notebook.

When working with users, follow these guidelines:
- Be concise and helpful.
- Understand both code and data science concepts.
- Explain your thinking clearly.
- When suggesting code, ensure it's correct, efficient, and follows best practices.
- When you make changes to the notebook, explain what you did.

When handling system errors:
- Translate technical errors into user-friendly language.
- Focus on solutions rather than problems.
- Suggest next steps the user can take.

Current notebook context:
{notebook_context}
"""

class JupyterBuddyAgent:
    """
    Main agent class that handles the conversation loop with the user and takes actions
    on the notebook based on the conversation.
    """
    
    def __init__(self, 
                 send_response_callback: Callable[[Dict[str, Any]], None],
                 send_action_callback: Callable[[Dict[str, Any]], None]):
        """
        Initialize the agent with callbacks for sending responses to the user
        and sending actions to the frontend.
        
        Args:
            send_response_callback: Function to send structured responses back to the user.
            send_action_callback: Function to send actions directly to the frontend.
        """
        # Initialize the LLM with tool awareness
        self.llm = get_llm()

        # Set up callbacks for communication
        self.send_response = send_response_callback
        self.send_action = send_action_callback
        
        # Store conversation history
        self.latest_conversation = {"messages": []}  # Always accessible
        
        # Create the agent graph
        self.create_agent_graph()
        
    def create_agent_graph(self):
        """Create the LangGraph state machine for agent execution."""
        workflow = StateGraph(AgentState)

        # LLM Response Processing Node
        workflow.add_node("process_llm_response", self.process_llm_response_node)

        # Set decision edges
        workflow.add_conditional_edges(
            "process_llm_response",
            self.should_wait_for_frontend,
            {
                True: END,  # If waiting for frontend, pause execution
                False: END  # Otherwise, stop execution and send response
            }
        )

        # Set the entry point
        workflow.set_entry_point("process_llm_response")

        # Compile the graph
        self.graph = workflow.compile()

    def process_llm_response_node(self, state: AgentState) -> AgentState:
        """
        Handles the LLM reasoning, decision making, and detecting tool calls.
        
        Args:
            state: The current state of the agent
            
        Returns:
            Updated state after processing
        """
        messages = state["messages"]

        # Call the LLM
        response = self.llm.invoke(messages)

        # Extract tool calls
        tool_calls = response.additional_kwargs.get("tool_calls", [])

        # Update conversation history before sending response
        self.latest_conversation["messages"] = messages + [response]

        if tool_calls:
            # Send tool execution request to frontend
            actions = [{"tool_name": call["name"], "parameters": call["args"]} for call in tool_calls]

            # Update state and wait for frontend response
            updated_state = {
                **state,
                "messages": self.latest_conversation["messages"],
                "actions": actions,
                "error": None  # Reset error on new action execution
            }
            self.send_action({"message": response.content, "actions": actions})
            return updated_state

        # Otherwise, stop execution and send final user response
        output_to_user = response.content
        if output_to_user and output_to_user.strip():
            self.send_response({"message": output_to_user, "actions": None})

        return {
            **state,
            "messages": self.latest_conversation["messages"],
            "actions": None,
            "error": None,  # Reset error state
            "output_to_user": output_to_user
        }

    def should_wait_for_frontend(self, state: AgentState) -> bool:
        """Determine if the agent should wait for frontend response (i.e., actions were sent)."""
        return state["actions"] is not None
    
    def handle_agent_input(self, session_id: str, data: Dict[str, Any]):
        """
        Handles new user messages OR frontend execution results.
        
        Args:
            session_id: The current user session ID.
            data: The input data, which can be a user message OR an action result.
        """
        if data.get("type") == "user_message":
            user_message = data["data"]
            formatted_context = "No active notebook" if data.get("notebook_context") is None else json.dumps(data["notebook_context"], indent=2)

            # Update conversation history
            self.latest_conversation["messages"].append(HumanMessage(content=user_message))

            initial_state = {
                "notebook_context": data.get("notebook_context"),
                "messages": self.latest_conversation["messages"],
                "output_to_user": None,
                "actions": None,
                "error": None
            }

            # Start agent execution
            self.graph.invoke(initial_state)

        elif data.get("type") == "action_result":
            # Handle frontend response
            action_result = data["data"]

            if action_result.get("error"):
                # Update state with error
                self.graph.invoke({
                    "error": {"error_message": action_result["error"]}
                })
            else:
                # Continue processing with updated notebook context
                self.graph.invoke({
                    "notebook_context": action_result.get("notebook_context"),
                    "error": None  # Reset error on success
                })
