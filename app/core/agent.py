"""
JupyterBuddy Agent Module - Simplified with Messages-Only Approach
"""

import json
import logging
from typing import Dict, List, Any, Optional, TypedDict

# LangChain imports
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
    ToolMessage,
)
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
    messages: List[BaseMessage]  # Contains all conversation and notebook context
    output_to_user: Optional[str]
    current_action: Optional[Dict[str, Any]]
    waiting_for_frontend: bool  # True when waiting for frontend response
    end_agent_execution: bool  # True when LLM sends final response
    first_message: bool  # Flag to track if this is the first message (for notebook context)


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
            "messages": [],
            "output_to_user": None,
            "current_action": None,
            "waiting_for_frontend": False,
            "end_agent_execution": False,
            "first_message": True
        }
    return global_session_states[session_id]


def update_session_state(session_id: str, state: AgentState) -> None:
    """Update the global state for a session."""
    global_session_states[session_id] = state


class LLMNode:
    """Responsible for processing user input through the LLM and determining actions."""

    def __init__(self, llm):
        """Initialize with an LLM instance."""
        self.llm = llm

    async def invoke(self, state: AgentState) -> AgentState:
        """Process messages through the LLM and determine next actions."""
        try:
            messages = state["messages"]
            
            tool_call_ids = set()
            
            # Check if there are any tool calls in the most recent message
            # We need this to determine if _select_messages might clip off responses
            if messages and isinstance(messages[-1], AIMessage):
                last_message = messages[-1]
                last_tool_calls = getattr(last_message, "additional_kwargs", {}).get("tool_calls", [])
                
                # Extract tool_call_ids from the most recent message
                if last_tool_calls:
                    for tc in last_tool_calls:
                        if isinstance(tc, dict) and "id" in tc:
                            tool_call_ids.add(tc.get("id"))
                            
            # Check if any tool calls in the most recent message are missing responses
            missing_responses = False
            if tool_call_ids:
                # Check if we have responses for all tool_call_ids
                for tc_id in tool_call_ids:
                    # Look for a ToolMessage with matching tool_call_id
                    found = False
                    for msg in messages:
                        if isinstance(msg, ToolMessage) and msg.tool_call_id == tc_id:
                            found = True
                            break
                    
                    if not found:
                        missing_responses = True
                        break
                
            # If there are missing responses, we should NOT send the last message to the LLM
            # This prevents sending an assistant message with tool_calls but no responses
            if missing_responses:
                logger.warning("Detected assistant message with tool calls but missing tool responses")
                # Remove the last message from messages_to_use if it has missing responses
                if len(messages) > 1:
                    messages_to_use = self._select_messages(messages[:-1])
                else:
                    # If there's only one message and it has missing responses, don't proceed
                    # This is an edge case that shouldn't happen in practice
                    logger.error("Cannot proceed: first message has tool calls with no responses")
                    messages_to_use = []
            else:
                # Normal case - select messages as usual
                messages_to_use = self._select_messages(messages)

            try:
                # Get response from LLM
                response = self.llm.invoke(messages_to_use)

                # Update full messages list with LLM response
                updated_messages = messages + [response]

                return update_state(
                    state,
                    messages=updated_messages,
                    output_to_user=response.content,
                )

            except Exception as e:
                logger.error(f"Error during LLM invocation: {str(e)}")
                # Create an AIMessage for error reporting
                error_message = AIMessage(
                    content="JupyterBuddy encountered an issue. Please try again."
                )
                return update_state(
                    state,
                    messages=messages + [error_message],
                    end_agent_execution=True,
                )

        except Exception as outer_e:
            # Catch any other exceptions in the method
            logger.error(f"Unexpected error in LLM node: {str(outer_e)}")
            # Create an AIMessage for error reporting
            error_message = AIMessage(
                content="JupyterBuddy service is temporarily unavailable."
            )
            return update_state(
                state,
                messages=state["messages"] + [error_message],
                end_agent_execution=True,
            )

    def _select_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Select the appropriate subset of messages to send to the LLM.
        
        This method ensures:
        1. Most recent system message is always included (if any exists)
        2. Assistant messages with tool_calls are always paired with their ToolMessage responses
        3. Recent conversation context is included within token limits
        """
        if not messages:
            return []
            
        if len(messages) <= 5:
            return messages
        
        # Find the most recent system message
        most_recent_system_msg = None
        for msg in reversed(messages):
            if isinstance(msg, SystemMessage):
                most_recent_system_msg = msg
                break
        
        # Start with most recent non-system messages
        # Filter system messages out for now
        non_system_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        recent_messages = non_system_messages[-3:] if non_system_messages else []
        
        # Scan backward through messages to find any unpaired tool calls
        # This ensures we include all tool call pairs in the context
        tool_call_pairs = []
        i = len(non_system_messages) - 1
        
        while i >= 0:
            msg = non_system_messages[i]
            
            # Check if this is an assistant message with tool calls
            tool_calls = getattr(msg, "additional_kwargs", {}).get("tool_calls", [])
            
            if isinstance(msg, AIMessage) and tool_calls:
                # Get all tool_call_ids from this message
                tool_call_ids = [tc.get("id") for tc in tool_calls 
                               if isinstance(tc, dict) and "id" in tc]
                
                # If no ids, continue
                if not tool_call_ids:
                    i -= 1
                    continue
                
                # Find the corresponding tool response messages
                # We need to search forward from this message
                assistant_idx = i
                response_messages = []
                
                # Check if the tool responses are already in our recent messages
                j = assistant_idx + 1
                remaining_ids = tool_call_ids.copy()
                
                while j < len(non_system_messages) and remaining_ids:
                    response = non_system_messages[j]
                    if (isinstance(response, ToolMessage) and 
                        response.tool_call_id in remaining_ids):
                        response_messages.append(response)
                        remaining_ids.remove(response.tool_call_id)
                    j += 1
                
                # If we found all responses, add the pair
                if not remaining_ids:
                    # Only add if not already in recent messages
                    if assistant_idx < len(non_system_messages) - 3:
                        tool_call_pairs.append([msg] + response_messages)
            
            i -= 1
        
        # Flatten tool call pairs
        flattened_pairs = []
        for pair in tool_call_pairs:
            flattened_pairs.extend(pair)
            
        # Combine all messages, ensuring no duplicates
        # 1. Start with most recent system message if present
        result = []
        if most_recent_system_msg:
            result.append(most_recent_system_msg)
            
        # 2. Add tool call pairs (ensuring no duplicates with recent messages)
        recent_ids = {id(msg) for msg in recent_messages}
        for msg in flattened_pairs:
            if id(msg) not in recent_ids:
                result.append(msg)
                
        # 3. Add recent messages
        result.extend(recent_messages)
        
        return result


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

        last_message = messages[-1]

        # If the last message is already an AIMessage but not from the LLM (e.g., error message)
        # just send it to the user and end execution
        if isinstance(last_message, AIMessage) and last_message.content and "encountered an issue" in last_message.content:
            await self.send_response(
                {
                    "message": last_message.content,
                    "actions": None,
                    "session_id": self.session_id,
                }
            )
            return update_state(
                state,
                waiting_for_frontend=False,
                end_agent_execution=True,
            )

        # Extract tool calls if any
        tool_calls = getattr(last_message, "additional_kwargs", {}).get("tool_calls", [])

        # Check if there are multiple tool calls
        if len(tool_calls) > 1:
            logger.info(f"Multiple tool calls detected ({len(tool_calls)}). Instructing LLM to use one tool at a time.")
            
            # Create ToolMessage responses for each tool call instead of a single AIMessage
            feedback_messages = []
            for tool_call in tool_calls:
                tool_call_id = tool_call.get("id", "")
                if tool_call_id:
                    tool_message = ToolMessage(
                        content="Please call only one tool at a time. Process the results of each tool before making additional tool calls.",
                        tool_call_id=tool_call_id,
                        name="system_feedback",
                    )
                    feedback_messages.append(tool_message)
            
            # Add the messages to the state
            updated_messages = state["messages"].copy() + feedback_messages
            
            # Return to the LLM node without executing any tools
            return update_state(
                state,
                messages=updated_messages,
                waiting_for_frontend=False,
                end_agent_execution=False,  # Signal to continue to LLM node
                current_action=None,
            )

        if len(tool_calls) == 1:
            try:
                # Process the single tool call
                tool_call = tool_calls[0]
                
                # Handle the OpenAI function calling format
                if "function" in tool_call:
                    function_data = tool_call.get("function", {})
                    name = function_data.get("name")
                    
                    # Parse the arguments JSON string
                    try:
                        arguments = function_data.get("arguments", "{}")
                        args = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse arguments JSON: {arguments}")
                        args = {}

                    action = {
                        "tool_name": name,
                        "parameters": args,
                        "tool_call_id": tool_call.get("id"),
                    }
                else:
                    # Fallback for backward compatibility or other formats
                    action = {
                        "tool_name": tool_call.get("name", ""),
                        "parameters": tool_call.get("args", {}),
                        "tool_call_id": tool_call.get("id", ""),
                    }

                # Send action request to frontend
                await self.send_action(
                    {
                        "message": last_message.content,
                        "actions": [action],
                        "session_id": self.session_id,
                    }
                )

                # Update state to wait for frontend
                return update_state(
                    state,
                    current_action=action,
                    waiting_for_frontend=True,
                    end_agent_execution=False,
                )
            except Exception as e:
                logger.error(f"Error executing tool: {str(e)}")
                
                # Create ToolMessage for the error response
                # We need to respond to the actual tool_call_id
                tool_call_id = None
                tool_name = "unknown_tool"
                
                # Extract the tool_call_id and name from the current tool call
                if tool_calls and len(tool_calls) > 0:
                    tool_call = tool_calls[0]
                    if isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id")
                        if "function" in tool_call:
                            tool_name = tool_call.get("function", {}).get("name", tool_name)
                        else:
                            tool_name = tool_call.get("name", tool_name)
                
                if tool_call_id:
                    error_tool_message = ToolMessage(
                        content=f"Error executing tool: {str(e)}",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error"
                    )
                    
                    await self.send_response(
                        {
                            "message": f"JupyterBuddy encountered an issue while executing commands: {str(e)}",
                            "actions": None,
                            "session_id": self.session_id,
                        }
                    )
                    
                    return update_state(
                        state,
                        messages=state["messages"] + [error_tool_message],
                        waiting_for_frontend=False,
                        end_agent_execution=False,  # Continue to LLM node
                    )
                else:
                    # If we don't have a tool_call_id, use an AIMessage as a fallback
                    error_message = AIMessage(
                        content=f"JupyterBuddy encountered an issue while executing commands: {str(e)}"
                    )
                    await self.send_response(
                        {
                            "message": f"JupyterBuddy encountered an issue while executing commands: {str(e)}",
                            "actions": None,
                            "session_id": self.session_id,
                        }
                    )
                    return update_state(
                        state,
                        messages=state["messages"] + [error_message],
                        waiting_for_frontend=False,
                        end_agent_execution=True,
                    )
        else:
            try:
                # No tools called - send direct response to user
                if last_message.content and last_message.content.strip():
                    await self.send_response(
                        {
                            "message": last_message.content,
                            "actions": None,
                            "session_id": self.session_id,
                        }
                    )

                # Mark execution as complete
                return update_state(
                    state,
                    current_action=None,
                    waiting_for_frontend=False,
                    end_agent_execution=True,
                )
            except Exception as e:
                logger.error(f"Error sending response: {str(e)}")
                # Create an AIMessage for error reporting
                error_message = AIMessage(
                    content=f"Error sending response: {str(e)}"
                )
                return update_state(
                    state,
                    messages=state["messages"] + [error_message],
                    waiting_for_frontend=False,
                    end_agent_execution=True,
                    current_action=None,  # Ensure current_action is cleared
                )


class JupyterBuddyAgent:
    """Main agent class that orchestrates the workflow between LLM decisions and tool execution."""

    def __init__(
        self, llm, send_response_callback, send_action_callback, session_id: str
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
        cls, llm, send_response_callback, send_action_callback, session_id: str
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

        # Add standard edge from LLM to tools
        workflow.add_edge("llm_node", "tools_execution")
        
        # Add conditional edges from tools_execution
        workflow.add_conditional_edges(
            "tools_execution",
            self.should_continue_execution,
            {
                True: "llm_node",  # Go back to LLM if we need to continue
                False: END,  # End execution if we're done
            },
        )

        workflow.set_entry_point("llm_node")
        self.graph = workflow.compile()

    def should_continue_execution(self, state: AgentState) -> bool:
        """Determine if we should continue execution by going back to the LLM."""
        # If we're waiting for frontend, we should end and wait for result
        if state["waiting_for_frontend"]:
            return False
        # If end_agent_execution is True, we should end
        if state["end_agent_execution"]:
            return False
        # Otherwise, we should continue by going back to the LLM
        # This handles the case of multiple tool calls
        return True

    async def handle_agent_input(self, session_id: str, data: Dict[str, Any]):
        """Handles both user messages and frontend execution results."""
        # Get current state for session from global store
        current_state = get_session_state(session_id)
        # Make a copy to work with
        current_state = current_state.copy()

        if data.get("type") == "user_message":
            # Check if we're still waiting for a tool response
            if current_state.get("waiting_for_frontend"):
                # Send a message to the user to wait
                await self.send_response(
                    {
                        "message": "Please wait for the current operation to complete before continuing.",
                        "actions": None,
                        "session_id": session_id,
                    }
                )
                return  # Don't proceed with the LLM invocation

            # Process new user message
            user_message = data["data"]
            notebook_context = data.get("notebook_context")

            logger.info(f"Received user message from session {session_id}: {user_message}")

            # If this is the first message and we have a notebook context, add it as a system message
            if current_state["first_message"] and notebook_context is not None:
                # Create a system message with notebook context included
                system_content = JUPYTERBUDDY_SYSTEM_PROMPT.format(
                    notebook_context=json.dumps(notebook_context, indent=2),
                    conversation_history="No previous conversation.",
                    pending_actions="No pending actions.",
                    error_state="No errors detected.",
                )
                
                current_state["messages"].append(SystemMessage(content=system_content))
                current_state["first_message"] = False
                logger.info(f"Added initial notebook context as system message for session {session_id}")

            # Append user message to conversation
            current_state["messages"].append(HumanMessage(content=user_message))

            # Start execution - now using ainvoke for async
            updated_state = await self.graph.ainvoke(current_state)

            # Save updated state to global store
            update_session_state(session_id, updated_state)

        elif data.get("type") == "action_result":
            # Process frontend action result
            action_result = data["data"]
            current_action = current_state.get("current_action")
            action_results = action_result.get("results", [])

            logger.info(f"Received action result from session {session_id}")

            # Verify we have results
            if not action_results or len(action_results) == 0:
                logger.warning("Received empty action results")
                # Send direct notification to user
                await self.send_response(
                    {
                        "message": "No results returned for tool execution",
                        "actions": None,
                        "session_id": session_id,
                    }
                )
                current_state["waiting_for_frontend"] = False
                current_state["current_action"] = None  # Clear current_action
                update_session_state(session_id, current_state)
                return

            # Get the result (should be only one)
            result = action_results[0]
            tool_success = result.get("success", False)
            tool_error = result.get("error")
            
            # Create content for the tool message
            if not tool_success:
                tool_content = tool_error or "Tool execution failed"
                status = "error"
            else:
                # Convert result to string
                result_content = result.get("result", {})
                tool_content = (
                    json.dumps(result_content)
                    if isinstance(result_content, dict)
                    else str(result_content)
                )
                status = "success"

            # Add the tool message to conversation - ONLY place we create ToolMessage
            current_action = current_state.get("current_action")
            if current_action:
                tool_message = ToolMessage(
                    content=tool_content,
                    tool_call_id=current_action.get("tool_call_id", ""),
                    name=current_action.get("tool_name", ""),
                    status=status,
                )
                current_state["messages"].append(tool_message)

            # Update state
            current_state["waiting_for_frontend"] = False
            current_state["current_action"] = None  # Clear current_action
            
            # Continue execution with updated state
            updated_state = await self.graph.ainvoke(current_state)
            update_session_state(session_id, updated_state)
