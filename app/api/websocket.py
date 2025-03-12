"""
JupyterBuddy Conversation Model

This module defines the models for conversations, messages, and message roles.
Instead of persisting conversation history across notebook restarts, JupyterBuddy
treats each Jupyter Notebook instance as a separate user session.

A new session ID is assigned each time a notebook is opened, and the system
analyzes the notebook state at that moment to determine context. When the notebook
is closed, the session is discarded, ensuring a stateless approach beyond active sessions.

The `ConnectionManager` follows the Singleton pattern, ensuring that there is only
one global instance responsible for managing all active WebSocket connections.

Even though each notebook session is independent, the Singleton is necessary to:
- Prevent multiple notebooks from creating duplicate `ConnectionManager` instances.
- Ensure that all active WebSocket connections are tracked in one place.
- Avoid memory leaks by properly closing WebSockets when notebooks are closed.

Without the Singleton, each notebook would create its own `ConnectionManager`,
leading to duplicate state tracking, inefficient memory use, and WebSockets
staying open indefinitely. By having a single instance, JupyterBuddy ensures
efficient session management while maintaining user independence.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import asyncio
from contextlib import asynccontextmanager

# Local imports
from app.core.agent import JupyterBuddyAgent
from app.schemas.message import MessageSchema, NotebookContext
from app.models.conversation import Conversation, Message, MessageRole

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

    # Singleton instance
    _instance = None

    def __new__(cls):
        """Create a new singleton instance if none exists."""
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
            # Initialize instance attributes without inline type annotations
            cls._instance.active_connections = {}
            cls._instance.user_agents = {}
            cls._instance.latest_messages = {}
            cls._instance.notebook_contexts = {}

        return cls._instance

    async def connect(self, websocket: WebSocket, session_id: str):
        """
        Connect a new WebSocket client.

        Args:
            websocket: The WebSocket connection
            session_id: Unique identifier for the session
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket

        # Create a new agent for this session
        def send_response_callback(
            content: str, actions: Optional[List[Dict[str, Any]]] = None
        ):
            """Callback for the agent to send responses to the user."""
            asyncio.create_task(
                self.send_assistant_message(session_id, content, actions)
            )

        def send_action_result_callback(action: Dict[str, Any]):
            """Callback for the agent to send action requests to the frontend."""
            asyncio.create_task(self.send_action(session_id, action))

        # Create the agent with the callbacks
        self.user_agents[session_id] = JupyterBuddyAgent(
            send_response_callback=send_response_callback,
            send_action_result_callback=send_action_result_callback,
        )

        logger.info(f"Client connected: {session_id}")
        await self.send_system_message(
            session_id, "Connected to JupyterBuddy. How can I help you today?"
        )

    async def disconnect(self, session_id: str):
        """
        Disconnect a WebSocket client.

        Args:
            session_id: Unique identifier for the session to disconnect
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.user_agents:
            del self.user_agents[session_id]
        if session_id in self.latest_messages:
            del self.latest_messages[session_id]
        if session_id in self.notebook_contexts:
            del self.notebook_contexts[session_id]

        logger.info(f"Client disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        """
        Send a message to a specific client.

        Args:
            session_id: Unique identifier for the session
            message: The message to send
        """
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

    async def send_system_message(self, session_id: str, content: str):
        """
        Send a system message to a specific client.

        Args:
            session_id: Unique identifier for the session
            content: The message content
        """
        await self.send_message(session_id, {"type": "system", "content": content})

    async def send_assistant_message(
        self,
        session_id: str,
        content: str,
        actions: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Send an assistant message to a specific client.

        Args:
            session_id: Unique identifier for the session
            content: The message content
            actions: Optional list of actions for the frontend to perform
        """
        await self.send_message(
            session_id, {"type": "assistant", "content": content, "actions": actions}
        )

    async def send_action(self, session_id: str, action: Dict[str, Any]):
        """
        Send an action request to the frontend.

        Args:
            session_id: Unique identifier for the session
            action: The action payload
        """
        await self.send_message(session_id, {"type": "action", "action": action})

    def process_user_message(
        self,
        session_id: str,
        content: str,
        notebook_context: Optional[Dict[str, Any]] = None,
    ):
        """
        Process a user message by sending it to the agent.

        Args:
            session_id: Unique identifier for the session
            content: The message content
            notebook_context: Optional notebook context data
        """
        # Store the latest message and notebook context
        self.latest_messages[session_id] = content
        if notebook_context:
            self.notebook_contexts[session_id] = notebook_context

        # Get the agent for this session
        agent = self.user_agents.get(session_id)
        if agent:
            # Process the message with the agent
            agent.handle_message(content, notebook_context)

    def process_action_result(self, session_id: str, result: Dict[str, Any]):
        """
        Process the result of an action from the frontend.

        Args:
            session_id: Unique identifier for the session
            result: The action result
        """
        # Get the agent for this session
        agent = self.user_agents.get(session_id)
        if agent:
            # Get the latest message and notebook context
            latest_message = self.latest_messages.get(session_id, "")
            notebook_context = self.notebook_contexts.get(session_id)

            # Process the action result with the agent
            agent.handle_action_result(result, latest_message, notebook_context)


# Create a singleton connection manager
connection_manager = ConnectionManager()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time communication.

    Args:
        websocket: The WebSocket connection
        session_id: Unique identifier for the session
    """
    # Connect the websocket
    await connection_manager.connect(websocket, session_id)

    try:
        # Process messages until disconnect
        while True:
            # Wait for a message from the client
            data = await websocket.receive_text()

            try:
                # Parse the message
                message_data = json.loads(data)

                # Handle different message types
                if "content" in message_data:
                    # User message
                    content = message_data.get("content", "")
                    notebook_context = message_data.get("notebook_context")

                    # Log the message
                    logger.info(
                        f"Received user message from {session_id}: {content[:50]}..."
                    )

                    # Process the message
                    connection_manager.process_user_message(
                        session_id, content, notebook_context
                    )

                elif "action_result" in message_data:
                    # Action result from frontend
                    result = message_data.get("action_result", {})

                    # Log the action result
                    logger.info(
                        f"Received action result from {session_id}: {result.get('action_type', 'UNKNOWN')}"
                    )

                    # Process the action result
                    connection_manager.process_action_result(session_id, result)

                else:
                    # Unknown message type
                    logger.warning(f"Received unknown message type from {session_id}")
                    # Send error to LLM to handle instead of direct system message
                    await connection_manager.send_message(
                        session_id,
                        {
                            "type": "action_result",
                            "action_result": {
                                "action_type": "SYSTEM_ERROR",
                                "result": {
                                    "error_type": "unknown_message",
                                    "message": "Received an unknown message format",
                                },
                                "success": False,
                            },
                        },
                    )

            except json.JSONDecodeError:
                # Invalid JSON
                logger.error(f"Received invalid JSON from {session_id}")
                # Send error to LLM to handle instead of direct system message
                await connection_manager.send_message(
                    session_id,
                    {
                        "type": "action_result",
                        "action_result": {
                            "action_type": "SYSTEM_ERROR",
                            "result": {
                                "error_type": "invalid_json",
                                "message": "Received invalid JSON format",
                            },
                            "success": False,
                        },
                    },
                )

            except Exception as e:
                # Other errors
                logger.exception(f"Error processing message from {session_id}: {e}")
                # Send error to LLM to handle instead of direct system message
                await connection_manager.send_message(
                    session_id,
                    {
                        "type": "action_result",
                        "action_result": {
                            "action_type": "SYSTEM_ERROR",
                            "result": {
                                "error_type": "general_error",
                                "message": str(e),
                            },
                            "success": False,
                        },
                    },
                )

    except WebSocketDisconnect:
        # Client disconnected
        await connection_manager.disconnect(session_id)
    except Exception as e:
        # Other errors
        logger.exception(f"WebSocket error with {session_id}: {e}")
        await connection_manager.disconnect(session_id)
