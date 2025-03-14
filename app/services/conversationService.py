# app/services/convesationService.py 
from sqlalchemy.orm import Session
from models.conversationModel import ConversationDB, MessageDB
from datetime import datetime, timezone

# Conversation Handler Class
class Conversation:
    """
    Handles storing and retrieving conversation history from the database.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.db = SessionLocal()

        # Ensure conversation exists
        if not self.db.query(ConversationDB).filter_by(session_id=session_id).first():
            new_conversation = ConversationDB(session_id=session_id)
            self.db.add(new_conversation)
            self.db.commit()

    def add_message(self, role: MessageRole, content: str):
        """Add a new message to the conversation."""
        message = MessageDB(session_id=self.session_id, role=role, content=content)
        self.db.add(message)
        self.db.commit()

    def add_action(self, action_type: str, payload: Dict[str, Any]):
        """Add a new notebook action."""
        action = NotebookActionDB(
            session_id=self.session_id, action_type=action_type, payload=payload
        )
        self.db.add(action)
        self.db.commit()

    def update_action_result(
        self,
        action_id: str,
        result: Dict[str, Any],
        success: bool,
        error: Optional[str] = None,
    ):
        """Update an action with its result."""
        action = (
            self.db.query(NotebookActionDB)
            .filter_by(id=action_id, session_id=self.session_id)
            .first()
        )
        if action:
            action.result = result
            action.success = success
            action.error = error
            self.db.commit()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Retrieve all messages for the conversation."""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in self.db.query(MessageDB)
            .filter_by(session_id=self.session_id)
            .all()
        ]

    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the last message in the conversation."""
        msg = (
            self.db.query(MessageDB)
            .filter_by(session_id=self.session_id)
            .order_by(MessageDB.timestamp.desc())
            .first()
        )
        return (
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            if msg
            else None
        )

    def get_last_action(self) -> Optional[Dict[str, Any]]:
        """Get the last action in the conversation."""
        action = (
            self.db.query(NotebookActionDB)
            .filter_by(session_id=self.session_id)
            .order_by(NotebookActionDB.timestamp.desc())
            .first()
        )
        return (
            {
                "action_type": action.action_type,
                "payload": action.payload,
                "success": action.success,
                "error": action.error,
            }
            if action
            else None
        )

    def close(self):
        """Close the database session."""
        self.db.close()

