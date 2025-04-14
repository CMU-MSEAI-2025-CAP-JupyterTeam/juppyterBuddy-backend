# websocket.py
import json
import logging
from typing import Dict, Any
import base64

from fastapi import WebSocket, WebSocketDisconnect
from app.core.agent import JupyterBuddyAgent, AgentState
from app.core.llm import get_llm

from app.services.rag import rag_store
from app.utils.parsers import extract_text_from_pdf  # we'll add this helper


# Set up logging
logger = logging.getLogger(__name__)

# In-memory storage
active_connections: Dict[str, WebSocket] = {}
session_states: Dict[str, AgentState] = {}
agent_instances: Dict[str, JupyterBuddyAgent] = {}


# send_json function to send JSON messages to the WebSocket client
async def send_json(session_id: str, message: Dict[str, Any]):
    """Send JSON message to WebSocket client."""
    websocket = active_connections.get(session_id)
    if websocket:
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"WebSocket send failed for {session_id}: {e}")


# Called when a new WebSocket connection is established
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """Handle WebSocket connection and messages."""
    await websocket.accept()
    active_connections[session_id] = websocket
    logger.info(f"WebSocket connected: {session_id}")

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "register_tools":
                await handle_register_tools(session_id, data)
            elif msg_type == "user_message":
                await handle_user_message(session_id, data)
            elif msg_type == "action_result":
                await handle_action_result(session_id, data)
            else:
                logger.warning(f"Unknown message type from {session_id}: {msg_type}")
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        # Clean up resources
        active_connections.pop(session_id, None)
        agent_instances.pop(session_id, None)
        session_states.pop(session_id, None)
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket handler: {e}")
        try:
            # Try to close the connection cleanly
            await websocket.close(code=1011)
        except:
            pass
        # Clean up resources
        active_connections.pop(session_id, None)
        agent_instances.pop(session_id, None)
        session_states.pop(session_id, None)


async def handle_register_tools(session_id: str, data: Dict[str, Any]):
    """Handle tool registration message."""
    try:
        tools = json.loads(data.get("data", "[]"))

        # Initialize LLM
        llm = get_llm()

        # Create callback functions for the agent
        async def send_response(
            payload,
        ):  # called when the LLM produces a response for the user.
            await send_json(session_id, payload)  # send_json(session_id, payload)

        async def send_action(payload):
            await send_json(session_id, payload)

        # Create the agent instance
        agent = await JupyterBuddyAgent.create(
            llm=llm,
            session_id=session_id,
            send_response_callback=send_response,
            send_action_callback=send_action,
        )

        # Store tools directly in the agent instance
        agent.tools = tools

        # Initialize session state
        agent_instances[session_id] = agent
        session_states[session_id] = {
            "messages": [],
            "llm_response": None,
            "current_action": None,
            "waiting_for_tool_response": False,
            "end_agent_execution": False,
            "first_message": True,
            "multiple_tool_call_requests": 0,
            "session_id": session_id,
        }

        logger.info(
            f"Agent initialized for session {session_id} with {len(tools)} tools"
        )

    except Exception as e:
        logger.exception(f"Tool registration failed: {e}")
        await send_json(
            session_id,
            {
                "message": f"Tool registration failed: {str(e)}",
                "session_id": session_id,
                "actions": None,
            },
        )


# Update handle_user_message to support context files
async def handle_user_message(session_id: str, data: Dict[str, Any]):
    agent = agent_instances.get(session_id)
    if not agent:
        logger.warning(f"Agent not ready for session {session_id}")
        return

    state = session_states.get(session_id)
    user_input = data.get("data")
    notebook_ctx = data.get("notebook_context")
    context_files = data.get("context", [])  # Optional

    try:
        for file in context_files:
            filename = file.get("filename", "unknown")
            mime = file.get("type", "text/plain")
            content = file.get("content", "")

            if not content:
                logger.warning(f"Empty context file skipped: {filename}")
                continue

            # Determine how to extract content based on MIME
            # MIME stands for Multipurpose Internet Mail Extensions (file type)
            if mime.startswith("text/"):
                logger.info(f"[RAG] Ingesting plain text: {filename}")
                rag_store.add_context(session_id, content)

            elif mime == "application/pdf":
                logger.info(f"[RAG] Ingesting PDF file: {filename}")
                try:
                    binary_data = base64.b64decode(content)
                    extracted = extract_text_from_pdf(binary_data)
                    rag_store.add_context(session_id, extracted)
                except Exception as e:
                    logger.warning(f"Failed to process PDF {filename}: {e}")
            else:
                logger.warning(f"Unsupported MIME type in context: {mime} ({filename})")

    except Exception as e:
        logger.exception(f"Error during RAG context processing: {e}")

    # Continue with normal agent flow
    updated_state = await agent.handle_user_message(state, user_input, notebook_ctx)
    session_states[session_id] = updated_state


async def handle_action_result(session_id: str, data: Dict[str, Any]):
    """Handle action result from frontend."""
    agent = agent_instances.get(session_id)
    if not agent:
        logger.warning(f"Agent not ready for session {session_id}")
        return

    state = session_states.get(session_id)
    result_payload = data.get("data", {})

    # Process the tool result
    updated_state = await agent.handle_tool_result(state, result_payload)
    session_states[session_id] = updated_state
