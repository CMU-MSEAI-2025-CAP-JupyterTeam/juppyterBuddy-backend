"""
WebSocket communication module for JupyterBuddy

Handles session-based communication, tool registration, and agent lifecycle.
"""

import json
import logging
from typing import Dict, Any, List
from fastapi import WebSocket, WebSocketDisconnect

from app.core.agent import JupyterBuddyAgent, get_session_state
from app.core.llm import get_llm

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_agents: Dict[str, JupyterBuddyAgent] = {}
        self.session_tools: Dict[str, List[Dict[str, Any]]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"✅ WebSocket connected: {session_id}")

    async def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_agents:
            del self.session_agents[session_id]
        if session_id in self.session_tools:
            del self.session_tools[session_id]
        logger.info(f"🔌 WebSocket disconnected: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(json.dumps(message))

    async def _callback_for(self, session_id: str):
        async def callback(msg: Dict[str, Any]):
            await self.send_message(session_id, msg)
        return callback

    async def handle_register_tools(self, session_id: str, data: Dict[str, Any]):
        tools_json = data.get("data")
        if not tools_json:
            return

        try:
            parsed_tools = json.loads(tools_json)
            self.session_tools[session_id] = parsed_tools
            logger.info(f"🛠️ Registered {len(parsed_tools)} tools for session {session_id}")

            llm = get_llm(tools=parsed_tools)
            callback = await self._callback_for(session_id)

            self.session_agents[session_id] = await JupyterBuddyAgent.create(
                llm=llm,
                send_response_callback=callback,
                send_action_callback=callback,
                session_id=session_id
            )
            logger.info(f"🤖 Agent created for session {session_id}")
        except Exception as e:
            logger.exception(f"❌ Failed to register tools for session {session_id}: {e}")
            await self.send_message(session_id, {
                "message": f"Error registering tools: {str(e)}"
            })

    async def handle_user_message(self, session_id: str, data: Dict[str, Any]):
        if session_id not in self.session_agents:
            logger.warning(f"No agent found for session {session_id}")
            await self.send_message(session_id, {
                "message": "System not ready. Please refresh the page."
            })
            return

        session_state = get_session_state(session_id)
        if session_state.get("waiting_for_frontend"):
            await self.send_message(session_id, {
                "message": "Please wait for the current operation to complete."
            })
            return

        await self.session_agents[session_id].handle_agent_input(session_id, {
            "type": "user_message",
            "data": data.get("data", ""),
            "notebook_context": data.get("notebook_context")
        })

    async def handle_action_result(self, session_id: str, data: Dict[str, Any]):
        if session_id not in self.session_agents:
            logger.warning(f"No agent found for session {session_id}")
            return

        await self.session_agents[session_id].handle_agent_input(session_id, {
            "type": "action_result",
            "data": data.get("data", {})
        })

    async def route_message(self, session_id: str, raw_msg: str):
        try:
            data = json.loads(raw_msg)
            message_type = data.get("type")

            if message_type == "register_tools":
                await self.handle_register_tools(session_id, data)
            elif message_type == "user_message":
                await self.handle_user_message(session_id, data)
            elif message_type == "action_result":
                await self.handle_action_result(session_id, data)
            else:
                logger.warning(f"Unknown message type from session {session_id}: {message_type}")
        except Exception as e:
            logger.exception(f"Error handling message from {session_id}: {e}")
            await self.send_message(session_id, {
                "message": f"An error occurred while processing your request: {str(e)}"
            })

# Singleton instance
connection_manager = WebSocketManager()

# FastAPI WebSocket endpoint
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    try:
        await connection_manager.connect(websocket, session_id)
        while True:
            raw_msg = await websocket.receive_text()
            await connection_manager.route_message(session_id, raw_msg)
    except WebSocketDisconnect:
        await connection_manager.disconnect(session_id)
    except Exception as e:
        logger.exception(f"Unhandled WebSocket error: {e}")
        await connection_manager.disconnect(session_id)
