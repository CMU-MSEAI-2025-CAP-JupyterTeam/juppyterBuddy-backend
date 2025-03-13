"""
JupyterBuddy Conversation Model

This module defines the models for conversations, messages, and message roles.
Each Jupyter Notebook instance is treated as a separate user session, ensuring 
that no conversation history is persisted beyond active notebook sessions.

A new session ID is assigned each time a notebook is opened, and the system 
analyzes the notebook state at that moment to determine context.

The `ConnectionManager` follows the Singleton pattern to:
- Prevent multiple notebooks from creating duplicate connection instances.
- Ensure that all active WebSocket connections are tracked globally.
- Avoid memory leaks by properly closing WebSockets when notebooks are closed.
"""

import json
import logging
import asyncio
from typing import Dict, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Local imports
from app.core.agent import JupyterBuddyAgent
from app.schemas.message import NotebookContext

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# WebSocket connection manager
class ConnectionManager:
    """
    Manages active WebSocket connections and provides methods for sending messages.
    Implements the Singleton pattern for a global connection manager.
    """

    _instance = None  # Singleton instance

    def __new__(cls):
        """Ensures only one instance of ConnectionManager exists."""
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
            cls._instance.active_connections = {}
            cls._instance.user_agents = {}
            cls._instance.latest_messages = {}
            cls._instance.notebook_contexts = {}
        return cls._instance

    async def connect(self, websocket: WebSocket, session_id: str):
        """Handles new WebSocket client connections."""
        await websocket.accept()
        self.active_connections[session_id] = websocket

        # Create callbacks for agent communication
        def send_response_callback(content: str):
            asyncio.create_task(self.send_assistant_message(session_id, content))

        def send_action_callback(action: Dict[str, Any]):
            asyncio.create_task(self.send_action(session_id, action))

        # Initialize the agent for this session
        self.user_agents[session_id] = JupyterBuddyAgent(
            send_response_callback=send_response_callback,
            send_action_callback=send_action_callback
        )

        logger.info(f"Client connected: {session_id}")
        await self.send_system_message(session_id, "Connected to JupyterBuddy. How can I help you today?")

    async def disconnect(self, session_id: str):
        """Handles WebSocket disconnection and cleans up session data."""
        self.active_connections.pop(session_id, None)
        self.user_agents.pop(session_id, None)
        self.latest_messages.pop(session_id, None)
        self.notebook_contexts.pop(session_id, None)

        logger.info(f"Client disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """Sends a message to the specified client."""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

    async def send_system_message(self, session_id: str, content: str):
        """Sends a system message to the client."""
        await self.send_message(session_id, {"type": "system", "content": content})

    async def send_assistant_message(self, session_id: str, content: str):
        """Sends an assistant message to the client."""
        await self.send_message(session_id, {"type": "assistant", "content": content})

    async def send_action(self, session_id: str, action: Dict[str, Any]):
        """Sends an action request to the frontend."""
        await self.send_message(session_id, {"type": "action", "action": action})

    def process_user_message(self, session_id: str, content: str, notebook_context: Optional[Dict[str, Any]] = None):
        """Processes a user message and forwards it to the agent."""
        self.latest_messages[session_id] = content
        if notebook_context:
            self.notebook_contexts[session_id] = notebook_context

        agent = self.user_agents.get(session_id)
        if agent:
            agent.handle_message(content, notebook_context)

    def process_action_result(self, session_id: str, result: Dict[str, Any]):
        """Processes the result of an action from the frontend."""
        agent = self.user_agents.get(session_id)
        if agent:
            latest_message = self.latest_messages.get(session_id, "")
            notebook_context = self.notebook_contexts.get(session_id)
            agent.handle_action_result(result, latest_message, notebook_context)


# Create a singleton connection manager
connection_manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication."""
    await connection_manager.connect(websocket, session_id)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message_data = json.loads(data)

                if "content" in message_data:
                    content = message_data["content"]
                    notebook_context = message_data.get("notebook_context")

                    logger.info(f"Received user message from {session_id}: {content[:50]}...")
                    connection_manager.process_user_message(session_id, content, notebook_context)

                elif "action_result" in message_data:
                    result = message_data["action_result"]

                    logger.info(f"Received action result from {session_id}: {result.get('action_type', 'UNKNOWN')}")
                    connection_manager.process_action_result(session_id, result)

                else:
                    logger.warning(f"Received unknown message type from {session_id}")
                    await connection_manager.send_message(session_id, {
                        "type": "error",
                        "message": "Received an unknown message format."
                    })

            except json.JSONDecodeError:
                logger.error(f"Received invalid JSON from {session_id}")
                await connection_manager.send_message(session_id, {
                    "type": "error",
                    "message": "Invalid JSON format."
                })

    except WebSocketDisconnect:
        await connection_manager.disconnect(session_id)
    except Exception as e:
        logger.exception(f"WebSocket error with {session_id}: {e}")
        await connection_manager.disconnect(session_id)
