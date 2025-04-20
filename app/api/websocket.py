# websocket.py
import json
import logging
from typing import Dict, Any

from fastapi import WebSocket, WebSocketDisconnect
from app.core.agent import JupyterBuddyAgent, AgentState
from app.core.llm import get_llm

from app.services.rag import rag_store
from app.utils.parsers import extract_text

# Set up logging
logger = logging.getLogger(__name__)

# In-memory storage
active_connections: Dict[str, WebSocket] = {}
session_states: Dict[str, AgentState] = {}
agent_instances: Dict[str, JupyterBuddyAgent] = {}


# Internal tools
from app.core.internalTools import retrieve_context

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
                await socket_handle_user_message(session_id, data)
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


# Handle tool registration message
async def handle_register_tools(session_id: str, data: Dict[str, Any]):
    """Handle tool registration message."""
    try:
        
        # Step 1: Load internal + frontend tools
        internal_tools = [retrieve_context]
        external_tools = json.loads(data.get("data", "[]"))
        tools = external_tools + internal_tools

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


# Update socket_handle_user_message to support context files
async def socket_handle_user_message(session_id: str, data: Dict[str, Any]):
    agent = agent_instances.get(session_id)
    if not agent:
        logger.warning(f"Agent not ready for session {session_id}")
        return
    
    # Extract user input and context files
    state = session_states.get(session_id)
    user_input = data.get("data", "").strip()
    notebook_ctx = data.get("notebook_context")
    context_files = data.get("context", [])  # Optional
    successful_files = []
    failed_files = []

    try:
        for file in context_files:
            filename = file.get("filename", "unknown")
            mime = file.get("type", "text/plain")
            content = file.get("content", "")
            if not content:
                logger.warning(f"[RAG] Skipped empty file: {filename}")
                failed_files.append(filename)
                continue

            extracted_text = extract_text(content, mime)
            logger.info(f"[RAG] Extracted text from {filename}: {extracted_text[:100]}...")

            if extracted_text:
                logger.info(f"[RAG] Ingested: {filename} (type: {mime})")
                # print session_id, extracted_text
                logger.info(f"[RAG] Adding context for session {session_id}")
                # ✅ Updated line inside socket_handle_user_message
                rag_store.add_context(session_id, extracted_text, filename=filename)

                successful_files.append(filename)
            else:
                logger.warning(f"[RAG] Skipped unsupported or unreadable file: {filename} ({mime})")
                failed_files.append(filename)
    except Exception as e:
        logger.exception(f"Error during RAG context processing: {e}")

    # NEW: If message is empty but context was uploaded
    if not user_input:
        if successful_files:
            success_msg = f"📜 Saved {len(successful_files)} Instruction file(s): {', '.join(successful_files)}."
            if failed_files:
                success_msg += f" ⚠️ Skipped {len(failed_files)} unsupported or unreadable file(s): {', '.join(failed_files)}."
        elif failed_files:
            success_msg = f"⚠️ All {len(failed_files)} uploaded file(s) failed to process: {', '.join(failed_files)}."
        else:
            success_msg = "⚠️ No message received and no context provided. Please enter a message or upload a document."

        await send_json(
            session_id,
            {
                "message": success_msg,
                "actions": None,
                "session_id": session_id,
            },
        )
        return

    # 👇 Otherwise, continue with normal LLM execution
    updated_state = await agent.handle_user_message(state, user_input, notebook_ctx)
    session_states[session_id] = updated_state

# Handle action result from frontend
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