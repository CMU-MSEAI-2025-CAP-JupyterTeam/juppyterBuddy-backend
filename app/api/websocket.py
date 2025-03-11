# app/api/websocket.py
import json
import logging
import asyncio
from typing import Dict, Any, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import uuid4

from app.core.llm import LLMService

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active connections
active_connections: Dict[str, WebSocket] = {}

# Store conversation history for each session
conversation_history: Dict[str, List[Dict[str, str]]] = {}

# Initialize LLM service
llm_service = LLMService()

async def process_message(message: Dict[str, Any], session_id: str, notebook_context: Dict[str, Any] = None):
    """
    Process a message through the LLM and return a response.
    """
    content = message.get("content", "")
    
    # Get or create conversation history for this session
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Add user message to history
    conversation_history[session_id].append({
        "role": "user",
        "content": content
    })
    
    # Create messages list for LLM
    messages = conversation_history[session_id].copy()
    
    # Add notebook context if available
    if notebook_context:
        # Format notebook context as a system message
        context_message = f"Current notebook: {notebook_context.get('title', 'Untitled')}\n"
        
        # Include information about cells
        cells = notebook_context.get('cells', [])
        if cells:
            context_message += "Current notebook cells:\n"
            for i, cell in enumerate(cells):
                # Only include a subset of cells to avoid token limits
                if i > 5:
                    context_message += f"...and {len(cells) - 5} more cells\n"
                    break
                cell_type = cell.get('type', 'unknown')
                content_preview = cell.get('content', '')
                if len(content_preview) > 100:
                    content_preview = content_preview[:100] + "..."
                context_message += f"Cell {i} ({cell_type}): {content_preview}\n"
        
        # Add active cell info
        active_cell = notebook_context.get('activeCell')
        if active_cell is not None:
            context_message += f"Active cell: {active_cell}\n"
        
        # Insert context as a system message at the beginning
        messages.insert(0, {
            "role": "system",
            "content": context_message
        })
    
    # Generate response using LLM
    llm_response = await llm_service.generate_response(messages)
    
    # Add assistant response to conversation history
    conversation_history[session_id].append({
        "role": "assistant",
        "content": llm_response["content"]
    })
    
    # Format the response
    response = {
        "type": "assistant",
        "content": llm_response["content"],
        "id": str(uuid4())
    }
    
    # Add actions if any were identified
    if llm_response["actions"]:
        response["actions"] = llm_response["actions"]
    
    return response

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        # Send connection acknowledgment
        await websocket.send_json({
            "type": "system",
            "content": "Connected to JupyterBuddy backend",
            "id": str(uuid4())
        })
        
        # Main message loop
        while True:
            data = await websocket.receive_json()
            content = data.get("content")
            notebook_context = data.get("notebook_context")
            
            # Process the message
            response = await process_message(data, session_id, notebook_context)
            
            # Send response
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"Client {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "system",
                "content": f"Error: {str(e)}",
                "id": str(uuid4())
            })
        except:
            pass
    finally:
        # Cleanup on disconnect
        if session_id in active_connections:
            del active_connections[session_id]