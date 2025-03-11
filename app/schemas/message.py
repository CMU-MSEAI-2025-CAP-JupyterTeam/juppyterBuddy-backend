"""
JupyterBuddy Message Schemas

This module defines the schemas for messages and notebook context data
exchanged between the frontend and backend.
"""
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum

class MessageType(str, Enum):
    """Types of messages in the system."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    
class ActionType(str, Enum):
    """Types of actions that can be performed on notebooks."""
    CREATE_CELL = "CREATE_CELL"
    UPDATE_CELL = "UPDATE_CELL"
    EXECUTE_CELL = "EXECUTE_CELL"
    GET_NOTEBOOK_INFO = "GET_NOTEBOOK_INFO"

class CellData(BaseModel):
    """Schema for notebook cell data."""
    index: int
    type: str  # 'code' or 'markdown'
    content: str

class NotebookContext(BaseModel):
    """Schema for notebook context data."""
    path: Optional[str] = None
    title: Optional[str] = None
    cells: List[CellData] = []
    activeCell: Optional[int] = None

class Action(BaseModel):
    """Schema for notebook actions."""
    action_type: ActionType
    payload: Dict[str, Any]

class MessageSchema(BaseModel):
    """Schema for messages exchanged with the frontend."""
    content: str
    notebook_context: Optional[NotebookContext] = None

class ResponseSchema(BaseModel):
    """Schema for responses sent to the frontend."""
    type: MessageType
    content: str
    actions: Optional[List[Action]] = None

class ActionResultSchema(BaseModel):
    """Schema for action results from the frontend."""
    action_type: ActionType
    result: Dict[str, Any]
    success: bool
    error: Optional[str] = None