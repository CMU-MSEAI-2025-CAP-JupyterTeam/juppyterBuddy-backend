"""
JupyterBuddy WebSocket Communication Module

This module handles real-time communication between the JupyterLab frontend
and backend via WebSockets. It routes messages between the frontend and the
JupyterBuddy Agent, handling session management and tool registration.
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from fastapi import WebSocket, WebSocketDisconnect

# Import the Agent class and LLM
from app.core.agent import JupyterBuddyAgent
from app.core.llm import get_llm

# Import LangChain tools
from langchain_core.tools import StructuredTool, BaseTool, Tool
from langchain_core.pydantic_v1 import Field, create_model

# Set up logging
logger = logging.getLogger(__name__)

class WebSocketManager:
    """
    Manages WebSocket connections and message routing for JupyterBuddy.
    """
    
    def __init__(self):
        """Initialize the WebSocket manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_agents: Dict[str, JupyterBuddyAgent] = {}
        self.session_tools: Dict[str, List[BaseTool]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Handle new WebSocket connection."""
        await websocket.accept()
        logger.info(f"WebSocket connection established for session {session_id}")
        self.active_connections[session_id] = websocket
        # Agent will be created when tools are registered
    
    async def disconnect(self, session_id: str):
        """Handle WebSocket disconnection."""
        if session_id in self.active_connections:
            logger.info(f"WebSocket disconnected for session {session_id}")
            del self.active_connections[session_id]
        
        # Clean up session resources
        if session_id in self.session_agents:
            del self.session_agents[session_id]
        if session_id in self.session_tools:
            del self.session_tools[session_id]
    
    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Send a message to a specific WebSocket client."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))
    
    def _convert_to_structured_tools(self, tools_json: str) -> List[BaseTool]:
        """Convert JSON tool definitions to LangChain structured tools."""
        tools = []
        tools_data = json.loads(tools_json)
        
        for tool_def in tools_data:
            tool_name = tool_def["name"]
            tool_description = tool_def.get("description", "")
            
            # Create parameter fields dictionary
            param_fields = {}
            required_params = tool_def.get("parameters", {}).get("required", [])
            
            for name, prop in tool_def.get("parameters", {}).get("properties", {}).items():
                # Determine the parameter type
                param_type = str
                if prop.get("type") == "integer":
                    param_type = int
                elif prop.get("type") == "boolean":
                    param_type = bool
                
                # Determine if parameter is required
                is_required = name in required_params
                
                # Add field to parameters dictionary
                param_fields[name] = (
                    Optional[param_type] if not is_required else param_type,
                    Field(description=prop.get("description", ""))
                )
            
            # Create a Pydantic model for the tool's parameters
            if param_fields:
                param_model = create_model(
                    f"{tool_name.capitalize()}Parameters",
                    **param_fields
                )
                
                # Create a function that will handle this tool
                def create_tool_func(tool_name=tool_name):
                    def tool_func(**kwargs) -> Dict[str, Any]:
                        return {
                            "tool_name": tool_name,
                            "parameters": kwargs
                        }
                    
                    tool_func.__name__ = tool_name
                    tool_func.__doc__ = tool_description
                    
                    return tool_func
                
                # Create the structured tool
                structured_tool = StructuredTool.from_function(
                    func=create_tool_func(),
                    name=tool_name,
                    description=tool_description,
                    args_schema=param_model,
                    return_direct=False
                )
                
                tools.append(structured_tool)
            else:
                # Simple tool with no parameters
                simple_tool = Tool(
                    name=tool_name,
                    description=tool_description,
                    func=lambda: {"tool_name": tool_name, "parameters": {}}
                )
                tools.append(simple_tool)
        
        return tools
    
    async def process_register_tools(self, session_id: str, data: Dict[str, Any]):
        """Handle tool registration message from frontend."""
        tools_json = data.get("data")
        if not tools_json:
            return
        
        try:
            # Convert JSON to structured tools
            structured_tools = self._convert_to_structured_tools(tools_json)
            
            # Store tools for this session
            self.session_tools[session_id] = structured_tools
            
            # Get storage directory from config or env var, or use default
            state_storage_dir = os.environ.get("AGENT_STATE_DIR", "agent_states")
            
            # Create the agent with the LLM that has tools bound to it
            llm = get_llm(tools=structured_tools)
            self.session_agents[session_id] = JupyterBuddyAgent(
                llm=llm,
                send_response_callback=lambda msg: self.send_message(session_id, msg),
                send_action_callback=lambda msg: self.send_message(session_id, msg),
                state_storage_dir=state_storage_dir
            )
            logger.info(f"Created agent for session {session_id} with {len(structured_tools)} tools")
        except Exception as e:
            logger.exception(f"Error registering tools: {str(e)}")
    
    async def process_user_message(self, session_id: str, data: Dict[str, Any]):
        """Handle user message from frontend."""
        if session_id not in self.session_agents:
            logger.error(f"No agent found for session {session_id}")
            await self.send_message(session_id, {
                "message": "Error: System initialization incomplete. Please refresh the page."
            })
            return
        
        # Process message with the agent
        agent = self.session_agents[session_id]
        await agent.handle_agent_input(session_id, {
            "type": "user_message",
            "data": data.get("data", ""),
            "notebook_context": data.get("notebook_context")
        })
    
    async def process_action_result(self, session_id: str, data: Dict[str, Any]):
        """Handle action result from frontend."""
        if session_id not in self.session_agents:
            logger.error(f"No agent found for session {session_id}")
            return
        
        # Get the data with results array
        action_data = data.get("data", {})
        action_results = action_data.get("results", [])
        
        # Check if any results have errors
        errors = [result.get("error") for result in action_results if result.get("error")]
        
        # Pass action result to agent for further processing
        agent = self.session_agents[session_id]
        
        # Create agent input with the complete results and any errors
        agent_input = {
            "type": "action_result",
            "data": action_data,
            "notebook_context": action_data.get("notebook_context")
        }
        
        # If there are errors, include them in the state update
        if errors:
            error_message = "; ".join(errors)
            agent_input["error"] = {"error_message": error_message}
        
        await agent.handle_agent_input(session_id, agent_input)
    
    # Main message handler
    async def handle_message(self, session_id: str, message: str):
        """Route incoming WebSocket messages based on type."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "register_tools":
                await self.process_register_tools(session_id, data)
            elif message_type == "user_message":
                await self.process_user_message(session_id, data)
            elif message_type == "action_result":
                await self.process_action_result(session_id, data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
        except Exception as e:
            logger.exception(f"Error processing message: {str(e)}")
            await self.send_message(session_id, {
                "message": f"An error occurred: {str(e)}"
            })

# Create a singleton instance
connection_manager = WebSocketManager()

# FastAPI WebSocket endpoint handler
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handle WebSocket connections and messages."""
    await connection_manager.connect(websocket, session_id)
    
    try:
        while True:
            message = await websocket.receive_text()
            await connection_manager.handle_message(session_id, message)
    except WebSocketDisconnect:
        await connection_manager.disconnect(session_id)
    except Exception as e:
        logger.exception(f"WebSocket error: {str(e)}")
        await connection_manager.disconnect(session_id)