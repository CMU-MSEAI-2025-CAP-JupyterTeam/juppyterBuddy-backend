# app/api/websocket.py
import json
import logging
import asyncio
from typing import Dict, Any, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import uuid4

# Import the LLM service
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
    
    Args:
        message: The user message
        session_id: Unique session identifier
        notebook_context: Optional context about the current notebook state
        
    Returns:
        Formatted response to send back to the client
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
    
    # Generate response using LLM service
    llm_response = await llm_service.generate_response(messages)
    
    # Add assistant response to conversation history
    conversation_history[session_id].append({
        "role": "assistant",
        "content": llm_response["content"]
    })
    
    # Keep conversation history to a reasonable size (last 20 messages)
    if len(conversation_history[session_id]) > 20:
        conversation_history[session_id] = conversation_history[session_id][-20:]
    
    # Format the response
    response = {
        "type": "assistant",
        "content": llm_response["content"],
        "id": str(uuid4())
    }
    
    # Add actions if any were identified
    if llm_response.get("actions"):
        response["actions"] = llm_response["actions"]
    
    return response

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time communication.
    
    Args:
        websocket: WebSocket connection
        session_id: Unique session identifier
    """
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        # Send connection acknowledgment
        await websocket.send_json({
            "type": "system",
            "content": "Connected to JupyterBuddy backend",
            "id": str(uuid4())
        })
        
        # Send information about available models
        available_models = llm_service.get_available_models()
        if available_models:
            models_list = ", ".join(available_models)
            await websocket.send_json({
                "type": "system",
                "content": f"Available models: {models_list}",
                "id": str(uuid4())
            })
        
        # Main message loop
        while True:
            data = await websocket.receive_json()
            content = data.get("content")
            notebook_context = data.get("notebook_context")
            
            # Check if this is a special command (e.g., to switch models)
            if content.startswith("/model "):
                # Extract model name from command
                requested_model = content.replace("/model ", "").strip()
                if requested_model in available_models:
                    # Process the model switch request
                    await websocket.send_json({
                        "type": "system",
                        "content": f"Switched to model: {requested_model}",
                        "id": str(uuid4())
                    })
                    continue
                else:
                    await websocket.send_json({
                        "type": "system",
                        "content": f"Model '{requested_model}' is not available. Available models: {models_list}",
                        "id": str(uuid4())
                    })
                    continue
            
            # Check if notebook is required but not present
            if any(cmd in content.lower() for cmd in ["create cell", "run cell", "execute"]) and not notebook_context:
                await websocket.send_json({
                    "type": "assistant",
                    "content": "I notice you're trying to work with a notebook, but there's no active notebook. Please open a notebook first.",
                    "id": str(uuid4())
                })
                continue
            
            # Send processing notification
            await websocket.send_json({
                "type": "status",
                "content": "Thinking...",
                "id": str(uuid4())
            })
            
            # Process the message
            try:
                response = await process_message(data, session_id, notebook_context)
                
                # Send response
                await websocket.send_json(response)
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await websocket.send_json({
                    "type": "system",
                    "content": f"Error processing your request: {str(e)}",
                    "id": str(uuid4())
                })
            
    except WebSocketDisconnect:
        logger.info(f"Client {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
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
        logger.info(f"Connection closed for session {session_id}")