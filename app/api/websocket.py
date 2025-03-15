"""
JupyterBuddy WebSocket Communication Module

This module handles real-time communication between the JupyterLab frontend
and backend via WebSockets. It routes messages between the frontend and the
JupyterBuddy Agent, handling session management and tool registration.
"""

import json
import logging
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect

# Import the Agent class
from app.core.agent import JupyterBuddyAgent

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
        self.session_tools: Dict[str, str] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Handle new WebSocket connection."""
        await websocket.accept()
        logger.info(f"WebSocket connection established for session {session_id}")
        self.active_connections[session_id] = websocket
        
        # Create a new agent for this session
        self.session_agents[session_id] = JupyterBuddyAgent(
            send_response_callback=lambda msg: self.send_message(session_id, msg),
            send_action_callback=lambda msg: self.send_message(session_id, msg)
        )
    
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
    
    async def process_user_message(self, session_id: str, data: Dict[str, Any]):
        """Handle user message from frontend."""
        if session_id not in self.session_agents:
            logger.error(f"No agent found for session {session_id}")
            await self.send_message(session_id, {
                "message": "Error: Session not initialized properly."
            })
            return
        
        # Pass message to agent for processing
        agent = self.session_agents[session_id]
        await agent.handle_agent_input(session_id, {
            "type": "user_message",
            "data": data.get("data", ""),
            "notebook_context": data.get("notebook_context")
        })
    
    async def process_register_tools(self, session_id: str, data: Dict[str, Any]):
        """Handle tool registration message from frontend."""
        tools_json = data.get("data")
        if not tools_json:
            return
        
        # Store tools for this session
        self.session_tools[session_id] = tools_json
        logger.info(f"Tools registered for session {session_id}")
        
        # Create or update agent with tools
        if session_id in self.session_agents:
            self.session_agents[session_id] = JupyterBuddyAgent(
                send_response_callback=lambda msg: self.send_message(session_id, msg),
                send_action_callback=lambda msg: self.send_message(session_id, msg),
                tools_json=tools_json
            )
    
    async def process_action_result(self, session_id: str, data: Dict[str, Any]):
        """Handle action result from frontend."""
        if session_id not in self.session_agents:
            logger.error(f"No agent found for session {session_id}")
            return
        
        # Pass action result to agent for further processing
        agent = self.session_agents[session_id]
        await agent.handle_agent_input(session_id, {
            "type": "action_result",
            "data": data.get("data", {})
        })
    
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