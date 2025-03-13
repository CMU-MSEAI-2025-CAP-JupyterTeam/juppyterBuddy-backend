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
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, FunctionMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_core.runnables import RunnableLambda
from langchain.callbacks import StdOutCallbackHandler
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# Local imports
from app.core.llm import get_llm
from app.models.conversation import Conversation, Message, MessageRole
from app.schemas.message import NotebookContext
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
    conversation: Conversation
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
        super().__init__()
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
        self.latest_conversation = {"messages": []}
        
        # Create the agent graph
        self.create_agent_graph()
        
    def create_agent_graph(self):
        """
        Create the state graph for the agent using LangGraph.
        This defines the flow between responding to users and taking actions.
        """
        # Define workflow states
        workflow = StateGraph(AgentState)
        
        # Node for generating LLM response
        workflow.add_node("agent", self.agent_node)
        
        # Add conditional edges for the agent to keep looping through tools
        workflow.add_conditional_edges(
            "agent",
            self.should_use_tool,
            {
                True: "agent",  # Continue if a tool was used
                False: END      # Stop if we're done with tools
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
        # Get the current conversation history
        messages = state["messages"]
        
        # Create a fresh tracker for this execution
        tracker = ToolExecutionTracker()
        
        # Call the LLM with the tracker to detect tool usage
        response = self.llm.invoke(messages, callbacks=[tracker])
        
        # If a tool was executed, continue the loop
        if tracker.tool_was_executed:
            return {
                **state,
                "messages": messages + [response],
                "should_use_tool": True,
                "output_to_user": None
            }
        
        # No tool was executed, just respond to the user
        output_to_user = response.content
        
        # Update the conversation history
        self.latest_conversation["messages"] = messages + [response]
        
        # Send the response to the user if it's not empty
        if output_to_user and output_to_user.strip():
            self.send_response(output_to_user)
        
        # Return updated state with completion flag
        return {
            **state,
            "messages": messages + [response],
            "should_use_tool": False,
            "output_to_user": output_to_user
        }
    
    def should_use_tool(self, state: AgentState) -> bool:
        """
        Determine if the agent should continue processing or finish.
        
        Args:
            state: The current state of the agent
            
        Returns:
            True if tools should be used, False if LLM should respond to user
        """
        return state.get("should_use_tool", False)
    
    def handle_message(self, user_message: str, notebook_context: Optional[Dict[str, Any]] = None):
        """
        Handle a new message from the user.
        
        Args:
            user_message: The message text from the user
            notebook_context: The current notebook context (cells, etc.)
        """
        logger.info(f"Handling user message: {user_message[:50]}...")
        
        # Format notebook context for prompt
        formatted_context = "No active notebook" if notebook_context is None else json.dumps(notebook_context, indent=2)
        
        # Create system message
        system_message = SystemMessage(content=SYSTEM_PROMPT.format(notebook_context=formatted_context))
        
        # Get previous conversation history
        previous_messages = self.latest_conversation.get("messages", [])
        
        # Decide whether to use history or start fresh
        if previous_messages and len(previous_messages) > 0:
            # Add new user message to existing history
            messages = previous_messages + [HumanMessage(content=user_message)]
        else:
            # Start a new conversation
            messages = [
                system_message,
                HumanMessage(content=user_message)
            ]
        
        # Create initial state
        initial_state = {
            "conversation": Conversation(messages=[
                Message(role=MessageRole.USER, content=user_message)
            ]),
            "notebook_context": notebook_context,
            "messages": messages,
            "output_to_user": None,
            "should_use_tool": False
        }
        
        # Run the agent graph
        self.graph.invoke(initial_state)
    
    def handle_action_result(self, result: Dict[str, Any], user_message: str, notebook_context: Optional[Dict[str, Any]] = None):
        """
        Handle the result of an action performed on the notebook.
        In our architecture, this is mainly needed for error handling.
        
        Args:
            result: The result of the action from the frontend
            user_message: The original user message
            notebook_context: The current notebook context
        """
        # Only process system errors or other special cases that need LLM attention
        if result.get("action_type") == "SYSTEM_ERROR":
            # Format notebook context for prompt
            formatted_context = "No active notebook" if notebook_context is None else json.dumps(notebook_context, indent=2)
            
            # Create system message
            system_message = SystemMessage(content=SYSTEM_PROMPT.format(notebook_context=formatted_context))
            
            # Get previous conversation history
            previous_messages = self.latest_conversation.get("messages", [])
            
            # Create a function message for the error
            error_info = result.get("result", {})
            error_type = error_info.get("error_type", "unknown")
            error_message = error_info.get("message", "An unknown error occurred")
            
            function_msg = FunctionMessage(
                name="system_error",
                content=json.dumps({
                    "error_type": error_type,
                    "message": error_message,
                    "context": "This is a system error that occurred during notebook operation. " +
                              "Please inform the user in a friendly way and suggest what they might do next."
                })
            )
            
            # Combine previous messages with the error message
            if previous_messages and len(previous_messages) > 0:
                messages = previous_messages + [function_msg]
            else:
                messages = [
                    system_message,
                    HumanMessage(content=user_message),
                    function_msg
                ]
            
            # Create state for error handling
            state = {
                "conversation": Conversation(messages=[
                    Message(role=MessageRole.USER, content=user_message),
                    Message(role=MessageRole.SYSTEM, content=json.dumps(result))
                ]),
                "notebook_context": notebook_context,
                "messages": messages,
                "output_to_user": None,
                "should_use_tool": False
            }
            
            # Process the error through the agent
            self.graph.invoke(state)
        else:
            # For normal action results, we don't need to do anything in our architecture
            logger.debug(f"Received action result: {result.get('action_type')} (no LLM processing needed)")