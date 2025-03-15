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
from typing import Dict, List, Any, Optional, TypedDict, Callable

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from app.core.llm import get_llm
from app.services.conversationService import ConversationService

# Set up logging
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """State structure for the agent's execution loop."""
    notebook_context: Optional[Dict[str, Any]]  # Stores the current notebook metadata
    messages: List[BaseMessage]  # List of conversation messages (User & AI)
    output_to_user: Optional[str]  # Final output message to send to user
    actions: Optional[List[Dict[str, Any]]]  # List of actions for the frontend
    error: Optional[Dict[str, str]]  # Stores error messages encountered
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
    
    def __init__(self, session_id: str, send_response_callback: Callable, send_action_callback: Callable):
        """
        Initializes the agent with WebSocket callbacks and loads existing state.
        
        Args:
            session_id: Unique identifier for the conversation.
            send_response_callback: Sends messages back to the user.
            send_action_callback: Sends actions to the frontend.
        """
        self.session_id = session_id
        self.llm = get_llm()
        self.send_response = send_response_callback
        self.send_action = send_action_callback
        self.conversation_service = ConversationService(session_id)
        self.latest_state = self.load_existing_state()  # Restore previous session state
        self.create_agent_graph()

    def load_existing_state(self) -> AgentState:
        """
        Loads previous conversation history and actions to restore the last known state.
        """
        messages = self.conversation_service.get_messages()
        last_action = self.conversation_service.get_last_action()
        return {
            "notebook_context": None,  # Notebook state is dynamically updated
            "messages": [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in messages],
            "output_to_user": None,
            "actions": [last_action] if last_action else None,  # Restore last executed action
            "error": None,
            "waiting_for_frontend": False,
            "end_agent_execution": False
        }

    def create_agent_graph(self):
        """Creates the structured execution graph for processing agent tasks."""
        workflow = StateGraph(AgentState)
        workflow.add_node("llm_node", self.llm_node)
        workflow.add_conditional_edges("llm_node", self.should_wait_for_frontend, {True: END, False: END})
        workflow.set_entry_point("llm_node")
        self.graph = workflow.compile()

    def update_state(self, state: AgentState, **updates) -> AgentState:
        """
        Updates the execution state with new values.

        Args:
            state: The current agent state.
            updates: Key-value pairs to update in the state.

        Returns:
            Updated agent state.
        """
        new_state = {**state, **updates}
        self.latest_state = new_state
        
        # Persist messages to database
        if updates.get("messages"):
            for msg in updates["messages"]:
                self.conversation_service.add_message("user" if isinstance(msg, HumanMessage) else "assistant", msg.content)
        
        # Persist tool execution actions to database
        if updates.get("actions"):
            for action in updates["actions"]:
                self.conversation_service.add_action(action["tool_name"], action["parameters"])
        
        return new_state

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
            self.send_action({"message": response.content, "actions": actions})  # Notify frontend of actions
            return self.update_state(state, messages=updated_messages, actions=actions, waiting_for_frontend=True, end_agent_execution=False, error=None)
        
        # If no tool execution required, send response directly to user
        output_to_user = response.content
        if output_to_user.strip():
            self.send_response({"message": output_to_user, "actions": None})
        
        return self.update_state(state, messages=updated_messages, actions=None, waiting_for_frontend=False, end_agent_execution=True, error=None)

    def should_wait_for_frontend(self, state: AgentState) -> bool:
        """Determines if execution should pause for frontend feedback or continue."""
        return state["waiting_for_frontend"]

    def handle_agent_input(self, session_id: str, data: Dict[str, Any]):
        """
        Handles both **user messages** and **frontend execution results**.
        """
        current_state = self.latest_state.copy() if self.latest_state else self.load_existing_state()
        
        if data.get("type") == "user_message":
            # Start execution
            # Append user message to conversation
            user_message = data["data"]
            notebook_context = data.get("notebook_context")
            current_state["messages"].append(HumanMessage(content=user_message))
            current_state["notebook_context"] = notebook_context
            self.latest_state = self.graph.invoke(current_state)  # Invoke LLM processing
        
        elif data.get("type") == "action_result":
            # Continue execution with updated state
            # Reset waiting state and handle errors if any
            action_result = data["data"]
            updated_state = self.update_state(current_state, waiting_for_frontend=False)
            
            if action_result.get("error"):
                updated_state = self.update_state(updated_state, error={"error_message": action_result["error"]})
            else:
                updated_state = self.update_state(updated_state, error=None)
            
            self.latest_state = self.graph.invoke(updated_state)  # Continue execution
