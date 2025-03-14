# app/models/conversationModel.py
"""
JupyterBuddy Conversation Model with Database Storage

This module defines database models for conversations, messages, and actions
to store conversation state persistently.

Uses SQLAlchemy for database integration, supporting SQLite/PostgreSQL.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from enum import Enum
from app.core.database import Base

# Enum for message roles
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

# Database Model for Messages
class MessageDB(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("conversations.session_id"))
    role = Column(String, index=True)
    content = Column(String)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

# Database Model for Notebook Actions
class NotebookActionDB(Base):
    __tablename__ = "notebook_actions"

    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("conversations.session_id"))
    action_type = Column(String)
    payload = Column(JSON)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    result = Column(JSON, nullable=True)
    success = Column(Boolean, default=False)
    error = Column(String, nullable=True)

# Database Model for Conversations
class ConversationDB(Base):
    __tablename__ = "conversations"

    session_id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    messages = relationship(
        "MessageDB", backref="conversation", cascade="all, delete-orphan"
    )
    actions = relationship(
        "NotebookActionDB", backref="conversation", cascade="all, delete-orphan"
    )
