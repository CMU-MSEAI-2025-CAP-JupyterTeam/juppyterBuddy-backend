"""
JupyterBuddy Conversation Model

This module defines the models for conversations, messages, and message roles.
These models are used to store and track the conversation state between the user and the assistant.
"""
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class MessageRole(str, Enum):
    """Enumeration of possible message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

class Message(BaseModel):
    """Model for individual messages in a conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class NotebookAction(BaseModel):
    """Model for actions performed on notebooks."""
    action_type: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)
    result: Optional[Dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None

class Conversation(BaseModel):
    """Model for conversations between users and the assistant."""
    id: str = Field(default_factory=lambda: f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    messages: List[Message] = Field(default_factory=list)
    actions: List[NotebookAction] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a new message to the conversation.
        
        Args:
            role: The role of the message sender
            content: The message content
            metadata: Optional metadata for the message
            
        Returns:
            The newly created message
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def add_action(self, action_type: str, payload: Dict[str, Any]) -> NotebookAction:
        """
        Add a new notebook action to the conversation.
        
        Args:
            action_type: The type of action
            payload: The action payload
            
        Returns:
            The newly created action
        """
        action = NotebookAction(
            action_type=action_type,
            payload=payload
        )
        self.actions.append(action)
        self.updated_at = datetime.now()
        return action
    
    def update_action_result(self, action_index: int, result: Dict[str, Any], success: bool, error: Optional[str] = None) -> Optional[NotebookAction]:
        """
        Update an action with its result.
        
        Args:
            action_index: The index of the action to update
            result: The result of the action
            success: Whether the action was successful
            error: Optional error message if the action failed
            
        Returns:
            The updated action or None if the action index is out of bounds
        """
        if 0 <= action_index < len(self.actions):
            action = self.actions[action_index]
            action.result = result
            action.success = success
            action.error = error
            self.updated_at = datetime.now()
            return action
        return None
    
    def get_last_message(self) -> Optional[Message]:
        """
        Get the last message in the conversation.
        
        Returns:
            The last message or None if there are no messages
        """
        if self.messages:
            return self.messages[-1]
        return None
    
    def get_last_action(self) -> Optional[NotebookAction]:
        """
        Get the last action in the conversation.
        
        Returns:
            The last action or None if there are no actions
        """
        if self.actions:
            return self.actions[-1]
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the conversation to a dictionary.
        
        Returns:
            Dictionary representation of the conversation
        """
        return {
            "id": self.id,
            "messages": [msg.dict() for msg in self.messages],
            "actions": [action.dict() for action in self.actions],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata
        }