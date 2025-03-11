# app/api/websocket.py
import json
import logging
import asyncio # Used to handle asynchronous tasks
from typing import Dict, Any # Import Dict and Any types
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import uuid4 # Generate unique IDs for messages

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active connections
active_connections: Dict[str, WebSocket] = {}

# Simple rule-based message processing for now
async def process_message(message: Dict[str, Any], notebook_context: Dict[str, Any] = None):
    """
    Process a message and return a response.
    This is a simplified implementation without using the LLM yet.
    """
    content = message.get("content", "").lower() # Return empty string if content is not present
    
    # Handle create cell commands
    if "create" in content and "cell" in content:
        cell_type = "code"
        if "markdown" in content:
            cell_type = "markdown"
            content_text = "## This is a new markdown cell\nAdd your markdown content here."
        else:
            content_text = "# Add your code here\nprint('Hello from JupyterBuddy!')"
        
        return {
            "type": "assistant",
            "content": f"I've created a new {cell_type} cell for you.",
            "id": str(uuid4()),
            "actions": [
                {
                    "action_type": "CREATE_CELL",
                    "payload": {
                        "cell_type": cell_type,
                        "content": content_text,
                        "position": "end"
                    }
                }
            ]
        }
    
    # Handle run cell commands
    elif "run" in content and "cell" in content:
        cell_index = notebook_context.get("activeCell") if notebook_context else None
        
        return {
            "type": "assistant",
            "content": "I've executed the active cell for you.",
            "id": str(uuid4()),
            "actions": [
                {
                    "action_type": "EXECUTE_CELL",
                    "payload": {
                        "cell_index": cell_index
                    }
                }
            ]
        }
    
    # Handle update cell commands
    elif "update" in content and "cell" in content:
        cell_index = notebook_context.get("activeCell") if notebook_context else 0
        
        # Extract content to add (a simple implementation)
        content_parts = content.split("with")
        new_content = "# Updated cell content"
        if len(content_parts) > 1:
            new_content = content_parts[1].strip()
        
        return {
            "type": "assistant",
            "content": f"I've updated cell #{cell_index} with the new content.",
            "id": str(uuid4()),
            "actions": [
                {
                    "action_type": "UPDATE_CELL",
                    "payload": {
                        "cell_index": cell_index,
                        "content": new_content
                    }
                }
            ]
        }
    
    # Default response for other queries
    return {
        "type": "assistant",
        "content": "I'm here to help with your notebook. You can ask me to create cells, run code, or update cells.",
        "id": str(uuid4())
    }
    
# Define WebSocket endpoint
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
            response = await process_message(data, notebook_context)
            
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