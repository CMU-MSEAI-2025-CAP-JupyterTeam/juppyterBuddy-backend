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
        self.session_tools: Dict[str, List[Dict[str, Any]]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Handle new WebSocket connection."""
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
    
    async def create_callback_for_session(self, session_id: str):
        """Create an async callback function for sending messages to a specific session."""
        async def callback(msg: Dict[str, Any]):
            await self.send_message(session_id, msg)
        return callback
    
    def _prepare_openai_tools(self, tools_json: str) -> List[Dict[str, Any]]:
        """
        Parse the tools JSON from frontend which is already in OpenAI format.
        No conversion needed since frontend now sends in the correct format.
        """
        try:
            return json.loads(tools_json)
        except Exception as e:
            logger.exception(f"Error parsing tools JSON: {e}")
            return []

    async def process_register_tools(self, session_id: str, data: Dict[str, Any]):
        """Handle tool registration message from frontend."""
        tools_json = data.get("data")
        if not tools_json:
            return
        
        try:
            # Parse tools directly - already in OpenAI format
            openai_tools = self._prepare_openai_tools(tools_json)
            
            # Store tools for this session
            self.session_tools[session_id] = openai_tools
            
            # Create the agent with the LLM that has tools bound to it
            llm = get_llm(tools=openai_tools)
            
            # Create async callback for this session
            message_callback = await self.create_callback_for_session(session_id)
            
            # Use the async factory method to create and initialize the agent
            # Pass session_id to the agent constructor
            self.session_agents[session_id] = await JupyterBuddyAgent.create(
                llm=llm,
                send_response_callback=message_callback,
                send_action_callback=message_callback,
                session_id=session_id  # Pass session_id here
            )
            
            logger.info(f"Created agent for session {session_id} with {len(openai_tools)} tools")
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
        
        # Log information about received results
        logger.info(f"Received {len(action_results)} action results from session {session_id}")
        
        # Check if any results have errors
        errors = [result.get("error") for result in action_results if result.get("error")]
        
        # Create agent input with the complete results and any errors
        agent_input = {
            "type": "action_result",
            "data": {
                "results": action_results,  # Pass all results to the agent
                "notebook_context": action_data.get("notebook_context")
            }
        }
        
        # If there are errors, include them in the state update
        if errors:
            error_message = "; ".join(errors)
            agent_input["error"] = {"error_message": error_message}
        
        # Pass action result to agent for further processing
        agent = self.session_agents[session_id]
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
    try:
        logger.info(f"WebSocket connection attempt for session {session_id}")
        
        # Accept the WebSocket connection
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for session {session_id}")
        
        # Let connection manager know about the connection
        await connection_manager.connect(websocket, session_id)
        
        # Start message loop
        while True:
            logger.info(f"Waiting for message from session {session_id}")
            message = await websocket.receive_text()
            logger.info(f"Received message from session {session_id}")
            await connection_manager.handle_message(session_id, message)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        await connection_manager.disconnect(session_id)
    except Exception as e:
        logger.exception(f"WebSocket error for session {session_id}: {str(e)}")
        try:
            await connection_manager.disconnect(session_id)
        except Exception as inner_e:
            logger.exception(f"Error during disconnect cleanup: {str(inner_e)}")
            
            