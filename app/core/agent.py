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
                logger.debug(f"[LLMNode] Messages before LLM invoke:\n{[str(m) for m in state['messages']]}")
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
        self.send_response = send_response_callback
        self.send_action = send_action_callback
        self.session_id = session_id

    async def invoke(self, state: AgentState) -> AgentState:
        messages = state["messages"]
        if not messages:
            return state

        last = messages[-1]

        # Gracefully return if it's an error AI message (not from OpenAI LLM)
        if isinstance(last, AIMessage) and "encountered an issue" in last.content:
            await self.send_response({
                "message": last.content,
                "actions": None,
                "session_id": self.session_id
            })
            return update_state(state, end_agent_execution=True, waiting_for_frontend=False)

        tool_calls = getattr(last, "additional_kwargs", {}).get("tool_calls", [])

        # 🛑 Case 1: No tools called → just send assistant response
        if not tool_calls:
            if last.content.strip():
                await self.send_response({
                    "message": last.content,
                    "actions": None,
                    "session_id": self.session_id
                })
            return update_state(state, end_agent_execution=True)

        # 🚫 Case 2: Multiple tool calls → instruct LLM to retry with one
        if len(tool_calls) > 1:
            logger.info(f"Multiple tool calls detected ({len(tool_calls)}). Instructing LLM to use one tool at a time.")
            feedback_msgs = [
                ToolMessage(
                    content="Please call only one tool at a time.",
                    tool_call_id=tc["id"],
                    name="system_feedback"
                )
                for tc in tool_calls if "id" in tc
            ]
            return update_state(
                state,
                messages=state["messages"] + feedback_msgs,
                end_agent_execution=False,
                waiting_for_frontend=False,
                current_action=None
            )

        # ✅ Case 3: Exactly one tool call → parse it
        try:
            tool_call = tool_calls[0]
            function_data = tool_call.get("function", {})
            tool_name = function_data.get("name", "")
            arguments = function_data.get("arguments", "{}")
            tool_call_id = tool_call.get("id", "")

            args = json.loads(arguments)

            action = {
                "tool_name": tool_name,
                "parameters": args,
                "tool_call_id": tool_call_id
            }

            await self.send_action({
                "message": last.content,
                "actions": [action],
                "session_id": self.session_id
            })

            # Update state and wait for frontend
            return update_state(
                state,
                current_action=action,
                waiting_for_frontend=True,
                end_agent_execution=False
            )

        except Exception as e:
            logger.exception(f"Error parsing tool call: {str(e)}")
            fallback_msg = AIMessage(content=f"Tool execution failed: {str(e)}")
            await self.send_response({
                "message": fallback_msg.content,
                "actions": None,
                "session_id": self.session_id
            })
            return update_state(
                state,
                messages=state["messages"] + [fallback_msg],
                end_agent_execution=True
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
        """
        Handles both user messages and frontend action results.
        This is the entry point for agent control loop execution.
        """
        # ⚠️ Use reference, not copy (CRITICAL!)
        current_state = get_session_state(session_id)

        # 🧠 Handle user messages from the frontend
        if data.get("type") == "user_message":
            if current_state.get("waiting_for_frontend"):
                await self.send_response({
                    "message": "Please wait for the current operation to complete.",
                    "actions": None,
                    "session_id": session_id
                })
                return

            user_message = data["data"]
            notebook_context = data.get("notebook_context")

            logger.info(f"Received user message from session {session_id}: {user_message}")

            # 👋 Add initial system message with notebook context (first message only)
            if current_state["first_message"] and notebook_context is not None:
                from app.core.prompts import JUPYTERBUDDY_SYSTEM_PROMPT

                system_content = JUPYTERBUDDY_SYSTEM_PROMPT.format(
                    notebook_context=json.dumps(notebook_context, indent=2),
                    conversation_history="No previous conversation.",
                    pending_actions="No pending actions.",
                    error_state="No errors detected.",
                )

                current_state["messages"].append(SystemMessage(content=system_content))
                current_state["first_message"] = False
                logger.info(f"Added initial notebook context as system message for session {session_id}")

            # ➕ Add the user message to message history
            current_state["messages"].append(HumanMessage(content=user_message))

            # ▶️ Start the LangGraph execution loop
            updated_state = await self.graph.ainvoke(current_state)

            # 💾 Update state in global store
            update_session_state(session_id, updated_state)

        # ⚙️ Handle action results from the frontend
        elif data.get("type") == "action_result":
            logger.debug(f"[Agent] Current state before updating:\n{json.dumps(current_state, indent=2, default=str)}")
            action_result = data["data"]
            current_action = current_state.get("current_action")
            action_results = action_result.get("results", [])

            logger.info(f"Received action result from session {session_id}")

            # 🛑 Check for missing or empty results
            if not action_results:
                logger.warning("No tool result received")
                await self.send_response({
                    "message": "No result returned from tool execution.",
                    "actions": None,
                    "session_id": session_id
                })
                current_state["waiting_for_frontend"] = False
                current_state["current_action"] = None
                update_session_state(session_id, current_state)
                return

            # ✅ Use only the first result (for now)
            result = action_results[0]
            tool_success = result.get("success", False)
            tool_error = result.get("error")

            # 🛠️ Create ToolMessage to send to LLM
            tool_message = ToolMessage(
                content=tool_error or json.dumps(result.get("result", {})),
                tool_call_id=current_action.get("tool_call_id", ""),
                name=current_action.get("tool_name", ""),
                status="error" if not tool_success else "success"
            )

            # ➕ Add the ToolMessage to the live state
            current_state["messages"].append(tool_message)

            # 🔄 Reset control flags
            current_state["waiting_for_frontend"] = False
            current_state["current_action"] = None

            # ▶️ Continue the LangGraph loop
            updated_state = await self.graph.ainvoke(current_state)

            # 💾 Update state again after round-trip
            update_session_state(session_id, updated_state)
