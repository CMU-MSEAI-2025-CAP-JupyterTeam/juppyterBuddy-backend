"""
WebSocket handler for JupyterBuddy

This module provides the WebSocket endpoint for real-time communication
between the JupyterBuddy frontend and backend.
"""

import json
import logging
from typing import Dict, Any

from fastapi import WebSocket, WebSocketDisconnect
from langchain_openai import ChatOpenAI

# Import the agent module components
from app.core.agent import JupyterBuddyAgent, get_session_state

# Set up logging
logger = logging.getLogger(__name__)

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}
# Store agent instances
agent_instances: Dict[str, JupyterBuddyAgent] = {}

async def send_response(data: Dict[str, Any]):
    """Send response to client via WebSocket."""
    session_id = data.get("session_id")
    if session_id in active_connections:
        websocket = active_connections[session_id]
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")

async def send_action(data: Dict[str, Any]):
    """Send action to client via WebSocket."""
    session_id = data.get("session_id")
    if session_id in active_connections:
        websocket = active_connections[session_id]
        try:
            # Extract the actions from data
            actions = data.get("actions", [])
            
            # Ensure we're only sending one action at a time
            if len(actions) > 1:
                logger.warning(f"Received multiple actions ({len(actions)}), only sending the first one")
                # Only keep the first action
                data["actions"] = [actions[0]]
                
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Error sending action: {str(e)}")

async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication with the frontend."""
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        logger.info(f"WebSocket connection established for session {session_id}")
        
        # Create LLM instance
        llm = ChatOpenAI(
            temperature=0.2,
            model="gpt-3.5-turbo-0125",  # Adjust model as needed
            streaming=False
        )
        
        # Create agent for this session
        agent = await JupyterBuddyAgent.create(
            llm=llm,
            send_response_callback=send_response,
            send_action_callback=send_action,
            session_id=session_id
        )
        
        agent_instances[session_id] = agent
        logger.info(f"Created new agent for session {session_id}")
        
        # Message handling loop
        while True:
            data_text = await websocket.receive_text()
            
            try:
                data = json.loads(data_text)
                message_type = data.get("type")
                logger.info(f"Received {message_type} message from session {session_id}")
                
                if message_type == "register_tools":
                    # Store tool definitions for OpenAI function calling
                    tools_json = data.get("data")
                    logger.info(f"Registered tools for session {session_id}")
                
                elif message_type == "user_message":
                    # Check if the agent is already waiting for a tool response
                    session_state = get_session_state(session_id)
                    if session_state.get("waiting_for_frontend"):
                        # Send response to wait
                        await send_response({
                            "message": "Please wait for the current operation to complete before continuing.",
                            "actions": None,
                            "session_id": session_id
                        })
                    else:
                        # Process the user message
                        await agent.handle_agent_input(session_id, data)
                
                elif message_type == "action_result":
                    # Make sure results array has exactly one result
                    action_data = data.get("data", {})
                    action_results = action_data.get("results", [])
                    
                    if len(action_results) > 1:
                        logger.warning(f"Received multiple results ({len(action_results)}), only processing the first one")
                        # Only keep the first result
                        action_data["results"] = [action_results[0]]
                        data["data"] = action_data
                    
                    # Forward to agent for processing
                    await agent.handle_agent_input(session_id, data)
                
                else:
                    logger.warning(f"Unknown message type: {message_type}")
            
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from session {session_id}")
                await websocket.send_json({
                    "message": "Error: Invalid JSON format",
                    "actions": None,
                    "session_id": session_id
                })
            except Exception as e:
                logger.exception(f"Error processing message: {str(e)}")
                await websocket.send_json({
                    "message": f"Error processing message: {str(e)}",
                    "actions": None,
                    "session_id": session_id
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
        # Clean up resources
        if session_id in active_connections:
            del active_connections[session_id]
    
    except Exception as e:
        logger.exception(f"WebSocket error for session {session_id}: {str(e)}")
        try:
            if session_id in active_connections:
                await active_connections[session_id].send_json({
                    "message": f"Server error: {str(e)}",
                    "actions": None,
                    "session_id": session_id
                })
        except:
            pass