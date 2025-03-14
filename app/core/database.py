from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Database Configuration
DATABASE_URL = "sqlite:///./jupyterbuddy.db"  # Change to PostgreSQL if needed
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})

# Create Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define Base for ORM Models
Base = declarative_base()

# Create Tables (Avoid circular import issue)
def init_db():
    """Ensure all tables are created on startup."""
    from app.models.conversationModel import ConversationDB, MessageDB, NotebookActionDB
    Base.metadata.create_all(bind=engine)
