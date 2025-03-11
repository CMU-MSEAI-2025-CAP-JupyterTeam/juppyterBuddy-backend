"""
JupyterBuddy Agent Module

This module defines the core LLM-based agent that can take actions in JupyterLab notebooks
through a conversational interface. The agent uses a state machine approach to iterate between
responding to the user and taking actions on the notebook.
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
    next_steps: List[Dict[str, Any]]
    tool_result: Optional[str]
    output_to_user: Optional[str]
    should_continue: bool
    
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

Current notebook context:
{notebook_context}

Remember: You can either respond to the user directly or take actions on their notebook.
If you need to take an action, use the appropriate tool and then communicate with the user about what you did.
"""

class JupyterBuddyAgent:
    """
    Main agent class that handles the conversation loop with the user and takes actions
    on the notebook based on the conversation.
    """
    
    def __init__(self, 
                 send_response_callback: Callable[[str, Optional[List[Dict[str, Any]]]], None],
                 send_action_result_callback: Callable[[Dict[str, Any]], None]):
        """
        Initialize the agent with callbacks for sending responses to the user
        and receiving action results from the frontend.
        
        Args:
            send_response_callback: Function to send messages back to the user
            send_action_result_callback: Function to send the results of actions back to the agent
        """
        self.llm = get_llm()
        self.send_response = send_response_callback
        self.send_action_result = send_action_result_callback
        self.create_agent_graph()
        
    def create_agent_graph(self):
        """
        Create the state graph for the agent using LangGraph.
        This defines the flow between responding to users and taking actions.
        """
        # Define tools for interacting with the notebook
        tools = [
            create_cell_tool,
            update_cell_tool,
            execute_cell_tool,
            get_notebook_info_tool
        ]
        
        # Create tool executor
        tool_executor = ToolExecutor(tools)
        
        # Define workflow states
        workflow = StateGraph(AgentState)
        
        # Node for generating LLM response
        workflow.add_node("agent", self.agent_node)
        
        # Node for executing tools
        workflow.add_node("action", ToolNode(tool_executor))
        
        # Edge from agent to END if no action needed
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                True: "action",  # If the LLM decides to take an action
                False: END       # If the LLM just wants to respond to the user
            }
        )
        
        # Edge from action back to agent for continuation
        workflow.add_edge("action", "agent")
        
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
            Updated state with the LLM's response and next steps
        """
        # Get the current conversation history
        messages = state["messages"]
        
        # Call LLM with the current conversation
        response = self.llm.invoke(messages)
        
        # Determine if there are any actions to take
        tool_calls = getattr(response, "tool_calls", None)
        
        if tool_calls:
            # LLM wants to use tools
            next_steps = []
            for tool_call in tool_calls:
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.args)
                next_steps.append({"name": tool_name, "args": tool_args})
            
            # Add the LLM message to conversation
            messages.append(response)
            
            # Update state with next steps
            return {
                **state,
                "messages": messages,
                "next_steps": next_steps,
                "should_continue": True,
                "output_to_user": None
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
                "next_steps": [],
                "should_continue": False,
                "output_to_user": output_to_user
            }
    
    def should_continue(self, state: AgentState) -> bool:
        """
        Determine if the agent should continue with actions or just respond to user.
        
        Args:
            state: The current state of the agent
            
        Returns:
            True if there are actions to take, False if just responding to user
        """
        return len(state.get("next_steps", [])) > 0
    
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
            "next_steps": [],
            "tool_result": None,
            "output_to_user": None,
            "should_continue": True
        }
        
        # Start the agent graph
        self._run_agent_loop(initial_state)
    
    def _run_agent_loop(self, state: AgentState):
        """
        Run the agent loop to take actions and generate responses.
        
        Args:
            state: The initial state for the agent
        """
        # Create event stream from graph
        for event in self.graph.stream(state):
            # Get the event type and state
            event_type = event["type"]
            current_state = event["state"]
            
            # If the agent wants to take an action
            if event_type == "agent" and current_state.get("should_continue", False):
                next_steps = current_state.get("next_steps", [])
                if next_steps:
                    # Loop through all requested actions
                    for step in next_steps:
                        tool_name = step["name"]
                        tool_args = step["args"]
                        
                        # Convert to the expected action format for frontend
                        action_payload = {
                            "action_type": tool_name.upper(),
                            "payload": tool_args
                        }
                        
                        # Send action to frontend
                        self.send_action_result(action_payload)
            
            # If the agent has output for the user
            if event_type == "agent" and current_state.get("output_to_user"):
                output = current_state["output_to_user"]
                
                # Check if there are next steps for actions to include
                actions = current_state.get("next_steps", [])
                
                # Send response to user
                self.send_response(output, actions if actions else None)
    
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
        
        # Create function message with the result
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
            "next_steps": [],
            "tool_result": json.dumps(result),
            "output_to_user": None,
            "should_continue": True
        }
        
        # Continue the agent loop
        self._run_agent_loop(state)