import json
import logging
from typing import Dict, List, Any, Optional, Callable, TypedDict

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

# AgentState definition for LangGraph
AgentState = TypedDict(
    "AgentState",
    {
        "messages": List[BaseMessage],
        "llm_response": Optional[AIMessage],
        "current_action": Optional[Dict],
        "waiting_for_frontend": bool,
        "end_agent_execution": bool,
        "first_message": bool,
        "single_tool_call_requests": int,
    },
)

# Set up logging
logger = logging.getLogger(__name__)


def update_state(state: AgentState, **kwargs) -> AgentState:
    """Return a new state object with updated keys."""
    new_state = state.copy()
    new_state.update(kwargs)
    return new_state


class JupyterBuddyAgent:
    """
    Asynchronous LLM agent class for managing LLM-tool interaction and notebook session state.
    Maintains internal LangGraph with tool calling support.
    """

    def __init__(
        self,
        session_id: str,
        llm: BaseChatModel,
        send_response_callback: Callable,
        send_action_callback: Callable,
    ):
        self.session_id = session_id
        self.llm = llm
        self.send_response = send_response_callback
        self.send_action = send_action_callback
        self.graph = None
        self.tools = []

    @classmethod
    async def create(
        cls,
        llm: BaseChatModel,
        session_id: str,
        send_response_callback: Callable,
        send_action_callback: Callable,
    ) -> "JupyterBuddyAgent":
        """Factory method to initialize the agent and LangGraph."""
        self = cls(session_id, llm, send_response_callback, send_action_callback)
        await self._create_graph()
        return self

    async def _create_graph(self):
        """Define LangGraph with LLM and Tool execution flow."""
        # create graph
        builder = StateGraph(AgentState)

        # add nodes
        builder.add_node("llm", self._llm_node)
        builder.add_node("tool_call_validator", self._tool_call_validator_node)
        builder.add_node("tools", self._tool_node)

        # add edges
        builder.set_entry_point("llm")
        builder.add_edge("llm", "tool_call_validator")
        builder.add_conditional_edges(
            "tool_call_validator",
            self._decide_next_step,
            {"no_tool": END, "one_tool": "tools", "retry": "llm"},
        )

        self.graph = builder.compile()

    async def _llm_node(self, state: AgentState) -> AgentState:
        """Run LLM and return updated state with assistant response."""
        # Filter messages: remove any assistant message with missing tool response
        relevant_history = self.state["messages"]

        # Get a version of the LLM with tools bound to it
        llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

        # Call LLM
        response = await llm_with_tools.ainvoke(relevant_history)

        updated = update_state(state, llm_response=response, end_agent_execution=False)
        logger.info(f"[LLM Node] Updated state: {updated}")
        return updated

    async def _tool_call_validator_node(self, state: AgentState) -> AgentState:
        llm_msg = state["llm_response"]
        tool_calls = llm_msg.additional_kwargs.get("tool_calls", []) if llm_msg else []

        if not tool_calls:
            # No tool call → send message and end
            await self.send_response(
                {
                    "message": llm_msg.content,
                    "actions": None,
                    "session_id": state["session_id"],
                }
            )
            return update_state(
                state,
                messages=state["messages"] + [llm_msg],
                llm_response=None,
                end_agent_execution=True,
            )

        elif len(tool_calls) == 1:
            # LLM returned exactly one tool call — we're ready to proceed

            # 🔍 Check if the last message was a temporary retry instruction
            last_msg = state["messages"][-1] if state["messages"] else None

            is_retry_msg = (
                isinstance(last_msg, SystemMessage) and
                last_msg.metadata and
                last_msg.metadata.get("retry_type") == "single_tool_call"
            )

            # If it was a retry message, remove it from the message history
            if is_retry_msg:
                pruned_messages = state["messages"][:-1]
            else:
                pruned_messages = state["messages"]

            # Commit the valid assistant message and reset the retry counter
            return update_state(
                state,
                messages=pruned_messages + [llm_msg],
                llm_response=None,
                single_tool_call_requests=0
            )


        else:
            retry_count = state.get("single_tool_call_requests", 0)

            if retry_count == 0:
                # First time asking for correction
                guidance = SystemMessage(
                    content="Respond with a single tool for the previous instruction.",
                    metadata={"retry_type": "single_tool_call"},
                )
                return update_state(
                    state,
                    messages=state["messages"] + [guidance],
                    llm_response=None,
                    single_tool_call_requests=1,
                )
            else:
                # Already asked — don't add another message
                # Retry LLM with existing message history
                return update_state(
                    state,
                    llm_response=None,
                    # single_tool_call_requests stays the same
                )

    async def _tool_node(self, state: AgentState) -> AgentState:
        """Extract tool call and send action to frontend."""
        messages = state["messages"]
        last = messages[-1] if messages else None
        tool_calls = getattr(last, "additional_kwargs", {}).get("tool_calls", [])

        if not tool_calls:
            # No tool call, just send response
            await self.send_response(
                {
                    "message": last.content if last else "",
                    "actions": None,
                    "session_id": self.session_id,
                }
            )
            return update_state(state, end_agent_execution=True)

        # Process the first tool call only
        tool_call = tool_calls[0]

        try:
            # Parse arguments as JSON
            args = json.loads(tool_call["function"].get("arguments", "{}"))

            # Create action payload
            action = {
                "tool_name": tool_call["function"]["name"],
                "parameters": args,
                "tool_call_id": tool_call["id"],
            }

            # Send action to frontend
            await self.send_action(
                {
                    "message": last.content,
                    "actions": [action],
                    "session_id": self.session_id,
                }
            )

            return update_state(
                state,
                current_action=action,
                waiting_for_frontend=True,
                end_agent_execution=False,
            )
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            # Add error message to conversation and end execution
            return update_state(
                state,
                messages=state["messages"]
                + [SystemMessage(content=f"Error: {str(e)}")],
                end_agent_execution=True,
            )

    def _should_continue(self, state: AgentState) -> bool:
        """Determine if the agent should continue processing or end execution."""
        if state.get("waiting_for_frontend", False) or state.get(
            "end_agent_execution", False
        ):
            return False
        return True

    def _decide_next_step(state: AgentState) -> str:
        llm_msg = state.get("llm_response")
        tool_calls = llm_msg.additional_kwargs.get("tool_calls", []) if llm_msg else []

        if not tool_calls:
            return "no_tool"
        elif len(tool_calls) == 1:
            return "one_tool"
        else:
            return "retry"

    async def handle_user_message(
        self, state: AgentState, user_input: str, notebook_context: Optional[Dict]
    ) -> AgentState:
        """Process user message and run LangGraph."""
        if state["waiting_for_frontend"]:
            await self.send_response(
                {
                    "message": "Please wait for the previous operation to complete.",
                    "actions": None,
                    "session_id": self.session_id,
                }
            )
            return state

        if state["first_message"]:
            from app.core.prompts import JUPYTERBUDDY_SYSTEM_PROMPT

            sys_msg = SystemMessage(
                content=JUPYTERBUDDY_SYSTEM_PROMPT.format(
                    notebook_context=json.dumps(notebook_context or {}, indent=2),
                    conversation_history="None",
                    pending_actions="None",
                    error_state="None",
                )
            )
            state["messages"].append(sys_msg)
            state["first_message"] = False

        state["messages"].append(HumanMessage(content=user_input))
        return await self.graph.ainvoke(state)

    async def handle_tool_result(
        self, state: AgentState, result_data: Dict[str, Any]
    ) -> AgentState:
        """Receive result from frontend tool execution."""
        current_action = state.get("current_action")

        if not current_action:
            logger.warning("Received tool result but no current action is pending")
            return state

        # Extract the result from results array
        results = result_data.get("results", [])
        result = results[0] if results else {"success": False, "error": "Empty result"}

        # Create tool message
        content = (
            json.dumps(result.get("result", {}))
            if result.get("success")
            else result.get("error", "Unknown error")
        )

        tool_message = ToolMessage(
            content=content,
            tool_call_id=current_action["tool_call_id"],
            name=current_action["tool_name"],
        )

        # Update state and continue execution
        updated_state = update_state(
            state,
            messages=state["messages"] + [tool_message],
            current_action=None,
            waiting_for_frontend=False,
        )

        return await self.graph.ainvoke(updated_state)

    def _filter_tool_call_mismatch(
        self, messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """Avoid sending LLM assistant messages with unresponded tool_calls."""
        if not messages:
            return []

        last = messages[-1]
        if isinstance(last, AIMessage):
            tool_calls = getattr(last, "additional_kwargs", {}).get("tool_calls", [])
            for tc in tool_calls:
                if not any(
                    isinstance(m, ToolMessage) and m.tool_call_id == tc["id"]
                    for m in messages
                ):
                    logger.warning(
                        "Filtered out assistant message with unresolved tool calls"
                    )
                    return messages[:-1]  # drop last assistant message
        return messages
