"""
JupyterBuddy Agent Module

This module defines the core LLM-based agent that can take actions in JupyterLab notebooks
through a conversational interface. The agent uses a state machine approach where:
1. The LLM decides which tools to call based on user intent
2. Tools execute independently with structured inputs
3. Tool outputs are sent directly to the frontend without LLM modification
"""
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, TypedDict, Callable

# LangChain imports
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import JsonOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, FunctionMessage, AIMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_core.tools import BaseTool, StructuredTool, tool

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolNode

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
    messages: List[Union[SystemMessage, HumanMessage, AIMessage, FunctionMessage]]
    current_tool_calls: List[Dict[str, Any]]
    output_to_user: Optional[str]
    should_use_tool: bool

# Define system prompt template for the agent
SYSTEM_PROMPT = """You are JupyterBuddy, an intelligent assistant integrated directly in JupyterLab.
You help users by answering questions about their Jupyter notebooks and taking actions on their behalf.

You have the ability to:
1. Create new code or markdown cells
2. Execute cells
3. Update existing cells
4. Get information about the current notebook

When working with users, follow these guidelines:
- Be concise and helpful
- Understand both code and data science concepts
- Explain your thinking clearly
- When suggesting code, ensure it's correct, efficient, and follows best practices
- When you make changes to the notebook, explain what you did

When handling system errors:
- Translate technical errors into user-friendly language
- Focus on solutions rather than problems
- Suggest next steps the user can take
- Don't blame the user for system or API issues
- If appropriate, offer to try an alternative approach

Current notebook context:
{notebook_context}

Remember: Your role is to decide which actions to take on behalf of the user.
The tools will execute independently, and their outputs will go directly to the frontend.
"""

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
            send_response_callback: Function to send messages back to the user
            send_action_callback: Function to send actions directly to the frontend
        """
        self.llm = get_llm()
        self.send_response = send_response_callback
        self.send_action = send_action_callback
        
        # Set up tools
        self.tools = [
            create_cell_tool,
            update_cell_tool,
            execute_cell_tool,
            get_notebook_info_tool
        ]
        
        # Create tool executor
        self.tool_executor = ToolExecutor(self.tools)
        
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
        
        # Node for executing tools directly
        workflow.add_node("direct_tool_executor", self.direct_tool_executor)
        
        # Edge from agent to END or tool executor based on decision
        workflow.add_conditional_edges(
            "agent",
            self.should_use_tool,
            {
                True: "direct_tool_executor",  # If the LLM decides to use a tool
                False: END                   # If the LLM just wants to respond to the user
            }
        )
        
        # Edge from tool executor to END (tool results go directly to frontend)
        workflow.add_edge("direct_tool_executor", END)
        
        # Set the entry point
        workflow.set_entry_point("agent")
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def agent_node(self, state: AgentState) -> AgentState:
        """
        Node that handles the LLM reasoning and decision making.
        
        Args:
            state: The current state of the agent
            
        Returns:
            Updated state with the LLM's response and tool calls
        """
        # Get the current conversation history
        messages = state["messages"]
        
        # Call LLM with the current conversation
        response = self.llm.invoke(messages)
        
        # Determine if there are any actions to take
        tool_calls = getattr(response, "tool_calls", None)
        
        if tool_calls:
            # LLM wants to use tools
            current_tool_calls = []
            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.args)
                current_tool_calls.append({"name": tool_name, "args": tool_args})
            
            # Add the LLM message to conversation
            messages.append(response)
            
            # Update state with tool calls
            return {
                **state,
                "messages": messages,
                "current_tool_calls": current_tool_calls,
                "should_use_tool": True,
                "output_to_user": response.content
            }
        else:
            # LLM just wants to respond to the user
            output_to_user = response.content
            
            # Add the LLM message to conversation
            messages.append(response)
            
            # Update state with user output
            return {
                **state,
                "messages": messages,
                "current_tool_calls": [],
                "should_use_tool": False,
                "output_to_user": output_to_user
            }
    
    def direct_tool_executor(self, state: AgentState) -> AgentState:
        """
        Node that directly executes tools and sends results to frontend.
        
        Args:
            state: The current state of the agent
            
        Returns:
            Updated state after tool execution
        """
        # Get the tool calls from the state
        tool_calls = state.get("current_tool_calls", [])
        
        # Execute each tool and send results directly to frontend
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            try:
                # Execute the tool (this returns a response but we don't need it
                # for the agent decision-making, only for debugging)
                tool_response = self.tool_executor.execute(
                    tool_name=tool_name,
                    tool_input=tool_args
                )
                
                # Log the tool response for debugging
                logger.debug(f"Tool response: {tool_response}")
                
                # Convert to the expected action format for frontend
                action_payload = {
                    "action_type": tool_name.upper(),
                    "payload": tool_args
                }
                
                # Send action directly to frontend
                self.send_action(action_payload)
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {e}")
                # Even on error, we continue with other tools
        
        # If LLM provided a message for the user along with the tool calls, send it
        if state.get("output_to_user"):
            self.send_response(state["output_to_user"])
        
        # Return state (no modifications needed as tools execute independently)
        return state
    
    def should_use_tool(self, state: AgentState) -> bool:
        """
        Determine if the agent should use tools or just respond to user.
        
        Args:
            state: The current state of the agent
            
        Returns:
            True if there are tools to use, False if just responding to user
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
        
        # Create initial state
        initial_state = {
            "conversation": Conversation(messages=[
                Message(role=MessageRole.USER, content=user_message)
            ]),
            "notebook_context": notebook_context,
            "messages": [
                system_message,
                HumanMessage(content=user_message)
            ],
            "current_tool_calls": [],
            "output_to_user": None,
            "should_use_tool": False
        }
        
        # Start the agent graph
        self._run_agent_loop(initial_state)
    
    def _run_agent_loop(self, state: AgentState):
        """
        Run the agent loop to take actions and generate responses.
        
        Args:
            state: The initial state for the agent
        """
        # Run the graph with the initial state
        result = self.graph.invoke(state)
        
        # Check if there's a response to send to the user
        output_to_user = result.get("output_to_user")
        if output_to_user and not result.get("should_use_tool", False):
            # If there's a user output and no tools were used, send the response
            self.send_response(output_to_user)
    
    def handle_action_result(self, result: Dict[str, Any], user_message: str, notebook_context: Optional[Dict[str, Any]] = None):
        """
        Handle the result of an action performed on the notebook.
        
        Args:
            result: The result of the action from the frontend
            user_message: The original user message
            notebook_context: The current notebook context
        """
        # Format notebook context for prompt
        formatted_context = "No active notebook" if notebook_context is None else json.dumps(notebook_context, indent=2)
        
        # Create system message
        system_message = SystemMessage(content=SYSTEM_PROMPT.format(notebook_context=formatted_context))
        
        # Create user message
        user_msg = HumanMessage(content=user_message)
        
        # Handle SYSTEM_ERROR specially with additional context
        if result.get("action_type") == "SYSTEM_ERROR":
            error_info = result.get("result", {})
            error_type = error_info.get("error_type", "unknown")
            error_message = error_info.get("message", "An unknown error occurred")
            
            # Create a specific function message for system errors
            function_msg = FunctionMessage(
                name="system_error",
                content=json.dumps({
                    "error_type": error_type,
                    "message": error_message,
                    "context": "This is a system error that occurred while processing a message. " +
                              "Please inform the user in a friendly way and suggest what they might do next."
                })
            )
        else:
            # Standard function message for normal action results
            function_msg = FunctionMessage(
                name=result.get("action_type", "UNKNOWN_ACTION").lower(),
                content=json.dumps(result.get("result", {}))
            )
        
        # Create initial state with the action result
        state = {
            "conversation": Conversation(messages=[
                Message(role=MessageRole.USER, content=user_message),
                Message(role=MessageRole.SYSTEM, content=json.dumps(result))
            ]),
            "notebook_context": notebook_context,
            "messages": [
                system_message,
                user_msg,
                function_msg
            ],
            "current_tool_calls": [],
            "output_to_user": None,
            "should_use_tool": False
        }
        
        # Run the agent loop with the new state
        self._run_agent_loop(state)