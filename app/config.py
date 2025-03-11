# app/config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the application."""
    
    # LLM settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
    
    # API settings
    API_PREFIX = "/api/v1"
    
    # WebSocket settings
    WS_PING_INTERVAL = 30  # seconds
    
    # Chat settings
    MAX_CONVERSATION_HISTORY = 20  # messages