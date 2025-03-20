"""
JupyterBuddy Agent Module with simplified tool call tracking.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, TypedDict, Callable, Set

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, END

# Local imports
from app.core.prompts import JUPYTERBUDDY_SYSTEM_PROMPT

# Set up logging
logger = logging.getLogger(__name__)

# Global session storage - keeping state outside the agent instance
global_session_states = {}


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
    pending_tool_calls: Dict[str, Dict]  # Map of tool_call_id -> tool details that need responses


def update_state(state: AgentState, **kwargs) -> AgentState:
    """Helper function to update state with new values."""
    updated_state = state.copy()
    for key, value in kwargs.items():
        if key in updated_state:
            updated_state[key] = value
    return updated_state


def get_session_state(session_id: str) -> AgentState:
    """Get the state for a session, creating a new one if it doesn't exist."""
    if session_id not in global_session_states:
        global_session_states[session_id] = {
            "notebook_context": None,
            "messages": [],
            "output_to_user": None,
            "actions": None,
            "error": None,
            "waiting_for_frontend": False,
            "end_agent_execution": False,
            "pending_tool_calls": {}  # Initialize empty pending tool calls
        }
    return global_session_states[session_id]


def update_session_state(session_id: str, state: AgentState) -> None:
    """Update the global state for a session."""
    global_session_states[session_id] = state


def fix_missing_tool_responses(messages: List[BaseMessage], pending_tool_calls: Dict[str, Dict]) -> List[BaseMessage]:
    """Add tool messages for any pending tool calls that haven't been responded to."""
    if not pending_tool_calls:
        return messages  # No pending tool calls, nothing to fix
    
    # Create a copy of messages
    updated_messages = messages.copy()
    
    # For each pending tool call, add a placeholder tool message
    for tool_call_id, tool_info in pending_tool_calls.items():
        logger.warning(f"Adding placeholder response for pending tool call: {tool_call_id}")
        
        tool_message = ToolMessage(
            content="No response received yet from the frontend.",
            tool_call_id=tool_call_id,
            name=tool_info.get("name", "unknown_tool"),
            status="error"
        )
        
        # Add the tool message to the end of the conversation
        updated_messages.append(tool_message)
    
    return updated_messages


class LLMNode:
    """Responsible for processing user input through the LLM and determining actions."""

    def __init__(self, llm):
        """Initialize with an LLM instance."""
        self.llm = llm

    async def invoke(self, state: AgentState) -> AgentState:
        """Process messages through the LLM and determine next actions."""
        # If there are pending tool calls, add placeholder responses
        messages = state["messages"]
        if state["pending_tool_calls"]:
            messages = fix_missing_tool_responses(messages, state["pending_tool_calls"])
            # Clear pending tool calls since we've added responses
            state = update_state(state, pending_tool_calls={})
        
        # Format conversation history (last few exchanges)
        conversation_summary = self._format_conversation_history(
            messages[-6:] if len(messages) > 6 else messages
        )

        # Format pending actions
        pending_actions = (
            "No pending actions."
            if not state["waiting_for_frontend"]
            else "Waiting for frontend to complete the previous action."
        )

        # Format error state
        error_state = (
            "No errors detected."
            if not state["error"]
            else f"ERROR: {state['error'].get('error_message', 'Unknown error')}\nThis error must be addressed before proceeding."
        )

        # Create system prompt with all context
        system_content = JUPYTERBUDDY_SYSTEM_PROMPT.format(
            notebook_context=json.dumps(state["notebook_context"], indent=2),
            conversation_history=conversation_summary,
            pending_actions=pending_actions,
            error_state=error_state,
        )

        system_message = SystemMessage(content=system_content)

        # Replace or add system message
        if messages and isinstance(messages[0], SystemMessage):
            messages = [system_message] + messages[1:]
        else:
            messages = [system_message] + messages

        try:
            # Get response from LLM
            response = self.llm.invoke(messages)
            
            # Update messages with LLM response
            updated_messages = messages + [response]
            
            # Track any new tool calls in the response
            pending_tool_calls = state["pending_tool_calls"].copy()
            if hasattr(response, "additional_kwargs"):
                tool_calls = response.additional_kwargs.get("tool_calls", [])
                for call in tool_calls:
                    if "id" in call:
                        tool_name = "unknown_tool"
                        if "function" in call:
                            tool_name = call.get("function", {}).get("name", "unknown_tool")
                        else:
                            tool_name = call.get("name", "unknown_tool")
                            
                        pending_tool_calls[call["id"]] = {
                            "name": tool_name,
                            "timestamp": None  # Could add timestamp if needed
                        }
            
            return update_state(
                state, 
                messages=updated_messages, 
                output_to_user=response.content,
                pending_tool_calls=pending_tool_calls
            )
        except Exception as e:
            logger.error(f"Error during LLM invocation: {str(e)}")
            
            # Add error information to state
            return update_state(
                state,
                error={"error_message": f"LLM invocation failed: {str(e)}"},
                messages=state["messages"]
            )

    def _format_conversation_history(self, messages):
        """Format conversation history for system prompt."""
        if not messages:
            return "No previous conversation."

        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                history.append(f"Assistant: {msg.content}")

        return "\n".join(history)


class ToolExecutionerNode:
    """Handles tool execution decision-making and action generation."""

    def __init__(self, send_response_callback, send_action_callback, session_id: str):
        """Initialize with callbacks for sending responses and actions."""
        self.send_response = send_response_callback
        self.send_action = send_action_callback
        self.session_id = session_id

    async def invoke(self, state: AgentState) -> AgentState:
        """Process LLM response to determine if tools need to be executed."""
        # Extract the last message (LLM response)
        messages = state["messages"]
        if not messages:
            return state

        # Check if there was an error in the LLM node
        if state["error"]:
            # If there was an error, send it to the user
            await self.send_response({
                "message": f"An error occurred: {state['error'].get('error_message', 'Unknown error')}",
                "actions": None,
                "session_id": self.session_id
            })
            
            # Mark execution as complete
            return update_state(
                state,
                waiting_for_frontend=False,
                end_agent_execution=True,
            )

        last_message = messages[-1]

        # Extract tool calls if any
        tool_calls = getattr(last_message, "additional_kwargs", {}).get(
            "tool_calls", []
        )

        if tool_calls:
            # Format actions for frontend execution
            actions = []
            for call in tool_calls:
                # Handle the OpenAI function calling format
                if "function" in call:
                    function_data = call.get("function", {})
                    name = function_data.get("name")
                    # Parse the arguments JSON string
                    try:
                        arguments = function_data.get("arguments", "{}")
                        args = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse arguments JSON: {arguments}")
                        args = {}

                    actions.append({
                        "tool_name": name, 
                        "parameters": args, 
                        "tool_call_id": call.get("id")
                    })
                else:
                    # Fallback for backward compatibility or other formats
                    actions.append(
                        {
                            "tool_name": call.get("name", ""),
                            "parameters": call.get("args", {}),
                            "tool_call_id": call.get("id", "")
                        }
                    )

            # Send action request to frontend
            await self.send_action({
                "message": last_message.content, 
                "actions": actions,
                "session_id": self.session_id
            })

            # Update state to wait for frontend
            return update_state(
                state,
                actions=actions,
                waiting_for_frontend=True,
                end_agent_execution=False,
                error=None,  # Reset error on new action execution
            )
        else:
            # No tools called - send direct response to user
            if last_message.content and last_message.content.strip():
                await self.send_response({
                    "message": last_message.content, 
                    "actions": None,
                    "session_id": self.session_id
                })

            # Mark execution as complete
            return update_state(
                state,
                actions=None,
                waiting_for_frontend=False,
                end_agent_execution=True,
                error=None,
            )


class JupyterBuddyAgent:
    """Main agent class that orchestrates the workflow between LLM decisions and tool execution."""

    def __init__(
        self,
        llm,
        send_response_callback,
        send_action_callback,
        session_id: str
    ):
        """Initializes the agent with WebSocket callbacks."""
        self.llm = llm
        self.send_response = send_response_callback
        self.send_action = send_action_callback
        self.session_id = session_id

        # Initialize nodes
        self.llm_node = LLMNode(self.llm)
        self.tool_executor = ToolExecutionerNode(
            send_response_callback, send_action_callback, session_id
        )
        
        # Graph will be created in initialize()
        self.graph = None

    @classmethod
    async def create(
        cls,
        llm,
        send_response_callback,
        send_action_callback,
        session_id: str
    ):
        """Factory method to create and initialize the agent asynchronously."""
        agent = cls(llm, send_response_callback, send_action_callback, session_id)
        await agent.initialize()
        return agent

    async def initialize(self):
        """Complete the initialization with async operations."""
        await self.create_agent_graph()
        return self

    async def create_agent_graph(self):
        """Creates the structured execution graph with LLM and ToolExecutioner nodes."""
        workflow = StateGraph(AgentState)

        # Add nodes with async support
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
                False: END,  # End execution if no tool was called
            },
        )

        workflow.set_entry_point("llm_node")
        self.graph = workflow.compile()

    def should_wait_for_frontend(self, state: AgentState) -> bool:
        """Determines if execution should pause for frontend feedback."""
        return state["waiting_for_frontend"]

    async def handle_agent_input(self, session_id: str, data: Dict[str, Any]):
        """Handles both user messages and frontend execution results."""
        # Get current state for session from global store
        current_state = get_session_state(session_id)
        # Make a copy to work with
        current_state = current_state.copy()

        if data.get("type") == "user_message":
            # Process new user message
            user_message = data["data"]
            notebook_context = data.get("notebook_context")

            logger.info(
                f"Received user message from session {session_id}: {user_message}"
            )

            # Append user message to conversation
            current_state["messages"].append(HumanMessage(content=user_message))
            current_state["notebook_context"] = notebook_context

            # Start execution - now using ainvoke for async
            updated_state = await self.graph.ainvoke(current_state)

            # Save updated state to global store
            update_session_state(session_id, updated_state)

        elif data.get("type") == "action_result":
            # Process frontend action result
            action_result = data["data"]
            action_list = current_state.get("actions", [])
            
            logger.info(f"Received action result from session {session_id}")
            
            # Add ToolMessage responses for each tool call
            if action_list:
                # Get the pending tool calls
                pending_tool_calls = current_state.get("pending_tool_calls", {}).copy()
                
                for action in action_list:
                    tool_call_id = action.get("tool_call_id")
                    tool_name = action.get("tool_name")
                    
                    # Skip if we don't have a tool call ID
                    if not tool_call_id:
                        continue
                    
                    # Create appropriate tool message content
                    if action_result.get("error"):
                        tool_content = action_result["error"]
                        status = "error"
                    else:
                        tool_content = action_result.get("result", "Action completed successfully")
                        status = "success"
                    
                    # Add the tool message to the conversation history
                    tool_message = ToolMessage(
                        content=tool_content,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status=status
                    )
                    current_state["messages"].append(tool_message)
                    logger.info(f"Added tool message for tool call {tool_call_id}")
                    
                    # Remove this tool call from pending list since we've handled it
                    if tool_call_id in pending_tool_calls:
                        del pending_tool_calls[tool_call_id]
                
                # Update the pending tool calls in the state
                current_state["pending_tool_calls"] = pending_tool_calls
            
            # Update state based on execution result
            if action_result.get("error"):
                current_state = update_state(
                    current_state,
                    error={"error_message": action_result["error"]},
                    waiting_for_frontend=False,
                )
            else:
                current_state = update_state(
                    current_state,
                    error=None,
                    waiting_for_frontend=False,
                    notebook_context=data.get(
                        "notebook_context", current_state["notebook_context"]
                    ),
                )

            # Continue execution with updated state
            updated_state = await self.graph.ainvoke(current_state)

            # Save updated state to global store
            update_session_state(session_id, updated_state)