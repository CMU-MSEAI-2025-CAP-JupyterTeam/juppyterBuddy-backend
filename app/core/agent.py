"""
JupyterBuddy Agent Module

This module defines the core LLM-based agent that can take actions in JupyterLab notebooks
through a conversational interface. The agent uses a state machine approach where:
1. The LLM decides which tools to call based on user intent.
2. Tools execute independently with structured inputs.
3. Tool outputs are sent directly to the frontend without LLM modification.
4. Conversation history is retained across messages for better decision-making.
5. Tool execution is detected dynamically, ensuring robust execution tracking.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, TypedDict, Callable

# LangChain imports
from langchain_core.messages import SystemMessage, HumanMessage, FunctionMessage, AIMessage, BaseMessage
from langchain.callbacks import StdOutCallbackHandler
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# Local imports
from app.core.llm import get_llm
from app.core.tools import (
    create_cell_tool,
    update_cell_tool,
    execute_cell_tool,
    get_notebook_info_tool
)

# Set up logging
logger = logging.getLogger(__name__)

# Type definitions for state management
class AgentState(TypedDict):
    """Type definition for the agent's state"""
    notebook_context: Optional[Dict[str, Any]]
    messages: List[BaseMessage]
    output_to_user: Optional[str]
    should_use_tool: bool

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
- Don't blame the user for system or API issues.
- If appropriate, offer to try an alternative approach.

Current notebook context:
{notebook_context}

Remember: Your role is to decide which actions to take on behalf of the user.
The tools will execute independently, and their outputs will go directly to the frontend.
"""

class ToolExecutionTracker(StdOutCallbackHandler):
    """Tracks whether a tool was executed in the current execution step."""
    def __init__(self):
        self.tool_was_executed = False  # Reset every execution
    
    def on_tool_end(self, *args, **kwargs):
        """Mark that a tool was executed."""
        self.tool_was_executed = True

class JupyterBuddyAgent:
    """
    Main agent class that handles the conversation loop with the user and takes actions
    on the notebook based on the conversation.
    """
    
    def __init__(self, 
                 send_response_callback: Callable[[str], None],
                 send_action_callback: Callable[[Dict[str, Any]], None]):
        """
        Initialize the agent with callbacks for sending responses to the user
        and sending actions to the frontend.
        
        Args:
            send_response_callback: Function to send messages back to the user.
            send_action_callback: Function to send actions directly to the frontend.
        """
        # Set up tools
        self.tools = [
            create_cell_tool,
            update_cell_tool,
            execute_cell_tool,
            get_notebook_info_tool
        ]
        
        # Initialize the LLM with tools bound directly
        self.llm = get_llm(tools=self.tools)
        
        # Create tool executor
        self.tool_executor = ToolExecutor(self.tools)
        
        # Set up callbacks for communication
        self.send_response = send_response_callback
        self.send_action = send_action_callback
        
        # Store conversation history
        self.latest_conversation = {"messages": []}  # ✅ Always accessible
        
        # Create the agent graph
        self.create_agent_graph()
        
    def create_agent_graph(self):
        """Create the LangGraph state machine for agent execution."""
        workflow = StateGraph(AgentState)

        # LLM node
        workflow.add_node("agent", self.agent_node)

        # Continue looping if a tool was executed
        workflow.add_conditional_edges(
            "agent",
            self.should_use_tool,
            {
                True: "agent",  # 🔄 Keep looping if a tool was used
                False: END      # ❌ Stop if no more tools needed
            }
        )

        # Set the entry point
        workflow.set_entry_point("agent")

        # Compile the graph
        self.graph = workflow.compile()

    def agent_node(self, state: AgentState) -> AgentState:
        """
        Handles the LLM reasoning, decision making, and tool execution.
        
        Args:
            state: The current state of the agent
            
        Returns:
            Updated state after processing
        """
        messages = state["messages"]

        # ✅ Create tracker to detect tool execution
        tracker = ToolExecutionTracker()

        # 🧠 Call the LLM (which now directly calls tools)
        response = self.llm.invoke(messages, callbacks=[tracker])

        # ✅ Update conversation history **before** sending response
        self.latest_conversation["messages"] = messages + [response]

        # 🔄 If a tool was executed, loop back
        if tracker.tool_was_executed:
            return {
                **state,
                "messages": self.latest_conversation["messages"],
                "should_use_tool": True,
                "output_to_user": None
            }

        # ❌ Otherwise, stop execution
        output_to_user = response.content
        if output_to_user and output_to_user.strip():
            self.send_response(output_to_user)

        return {
            **state,
            "messages": self.latest_conversation["messages"],
            "should_use_tool": False,
            "output_to_user": output_to_user
        }

    def should_use_tool(self, state: AgentState) -> bool:
        """Check if a tool was executed."""
        return state.get("should_use_tool", False)
    
    def handle_message(self, user_message: str, notebook_context: Optional[Dict[str, Any]] = None):
        """Handles new user messages and runs the agent graph."""
        
        formatted_context = "No active notebook" if notebook_context is None else json.dumps(notebook_context, indent=2)
        system_message = SystemMessage(content=SYSTEM_PROMPT.format(notebook_context=formatted_context))

        # ✅ Ensure we update the stored conversation
        self.latest_conversation["messages"].append(HumanMessage(content=user_message))

        initial_state = {
            "notebook_context": notebook_context,
            "messages": self.latest_conversation["messages"],
            "output_to_user": None,
            "should_use_tool": False
        }

        # Start agent execution
        self.graph.invoke(initial_state)
