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

def update_state(state: AgentState, **kwargs) -> AgentState:
    """
    Helper function to update state with new values.
    
    Args:
        state: Current agent state
        **kwargs: Key-value pairs to update in the state
        
    Returns:
        Updated agent state
    """
    updated_state = state.copy()
    for key, value in kwargs.items():
        if key in updated_state:
            updated_state[key] = value
    return updated_state

class LLMNode:
    """
    Responsible for processing user input through the LLM and determining actions.
    """
    
    def __init__(self, llm):
        """
        Initialize with an LLM instance.
        
        Args:
            llm: Language model instance
        """
        self.llm = llm
    
    def invoke(self, state: AgentState) -> AgentState:
        """
        Process messages through the LLM and determine next actions.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with LLM response
        """
        messages = state["messages"]
        
        # If the first message doesn't contain a system prompt, add one
        if not messages or not isinstance(messages[0], SystemMessage):
            system_message = SystemMessage(
                content=SYSTEM_PROMPT.format(
                    notebook_context=json.dumps(state["notebook_context"], indent=2)
                )
            )
            messages = [system_message] + messages
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        
        # Update messages with LLM response
        updated_messages = messages + [response]
        
        return update_state(
            state,
            messages=updated_messages,
            output_to_user=response.content
        )

class ToolExecutionerNode:
    """
    Handles tool execution decision-making and action generation.
    """
    
    def __init__(self, send_response_callback, send_action_callback):
        """
        Initialize with callbacks for sending responses and actions.
        
        Args:
            send_response_callback: Function to send responses to the user
            send_action_callback: Function to send actions to the frontend
        """
        self.send_response = send_response_callback
        self.send_action = send_action_callback
    
    def invoke(self, state: AgentState) -> AgentState:
        """
        Process LLM response to determine if tools need to be executed.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with action decisions
        """
        # Extract the last message (LLM response)
        messages = state["messages"]
        if not messages:
            return state
            
        last_message = messages[-1]
        
        # Extract tool calls if any
        tool_calls = getattr(last_message, "additional_kwargs", {}).get("tool_calls", [])
        
        if tool_calls:
            # Format actions for frontend execution
            actions = [
                {"tool_name": call["name"], "parameters": call["args"]} 
                for call in tool_calls
            ]
            
            # Send action request to frontend
            self.send_action({"message": last_message.content, "actions": actions})
            
            # Update state to wait for frontend
            return update_state(
                state,
                actions=actions,
                waiting_for_frontend=True,
                end_agent_execution=False,
                error=None  # Reset error on new action execution
            )
        else:
            # No tools called - send direct response to user
            if last_message.content and last_message.content.strip():
                self.send_response({"message": last_message.content, "actions": None})
            
            # Mark execution as complete
            return update_state(
                state,
                actions=None,
                waiting_for_frontend=False,
                end_agent_execution=True,
                error=None
            )

class JupyterBuddyAgent:
    """
    Main agent class that orchestrates the workflow between LLM decisions and tool execution.
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
        
        # Initialize nodes
        self.llm_node = LLMNode(self.llm)
        self.tool_executor = ToolExecutionerNode(send_response_callback, send_action_callback)
        
        # Store latest state
        self.latest_state = None
        
        # Create execution graph
        self.create_agent_graph()
        
    def create_agent_graph(self):
        """Creates the structured execution graph with LLM and ToolExecutioner nodes."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("llm_node", self.llm_node.invoke)
        workflow.add_node("tools_execution", self.tool_executor.invoke)
        
        # Add edges
        workflow.add_edge("llm_node", "tools_execution")
        
        # Add conditional edges from tools_execution
        workflow.add_conditional_edges(
            "tools_execution",
            self.should_wait_for_frontend,
            {
                True: END,  # Wait for frontend execution
                False: END  # End execution if no tool was called
            }
        )
        
        workflow.set_entry_point("llm_node")
        self.graph = workflow.compile()
    
    def should_wait_for_frontend(self, state: AgentState) -> bool:
        """Determines if execution should pause for frontend feedback."""
        return state["waiting_for_frontend"]
    
    def create_initial_state(self) -> AgentState:
        """Creates an initial empty state."""
        return {
            "notebook_context": None,
            "messages": [],
            "output_to_user": None,
            "actions": None,
            "error": None,
            "waiting_for_frontend": False,
            "end_agent_execution": False
        }
    
    def handle_agent_input(self, session_id: str, data: Dict[str, Any]):
        """
        Handles both **user messages** and **frontend execution results**.
        
        Args:
            session_id: The user session ID.
            data: Input message or frontend action result.
        """
        # Initialize or get existing state
        current_state = self.latest_state.copy() if self.latest_state else self.create_initial_state()
        
        if data.get("type") == "user_message":
            # Process new user message
            user_message = data["data"]
            notebook_context = data.get("notebook_context")
            
            logger.info(f"Received user message: {user_message}")
            
            # Append user message to conversation
            current_state["messages"].append(HumanMessage(content=user_message))
            current_state["notebook_context"] = notebook_context
            
            # Start execution
            self.latest_state = self.graph.invoke(current_state)
            
        elif data.get("type") == "action_result":
            # Process frontend action result
            action_result = data["data"]
            
            logger.info(f"Received action result: {json.dumps(action_result)}")
            
            # Update state based on execution result
            if action_result.get("error"):
                current_state = update_state(
                    current_state,
                    error={"error_message": action_result["error"]},
                    waiting_for_frontend=False
                )
            else:
                current_state = update_state(
                    current_state,
                    error=None,
                    waiting_for_frontend=False,
                    notebook_context=data.get("notebook_context", current_state["notebook_context"])
                )
            
            # Continue execution with updated state
            self.latest_state = self.graph.invoke(current_state)