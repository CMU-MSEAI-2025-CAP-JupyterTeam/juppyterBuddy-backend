# app/config.py
import os
from pydantic_settings import BaseSettings  # Changed import

class Settings(BaseSettings):
    """Application settings."""
    APP_NAME: str = "JupyterBuddy"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # Add more settings as needed
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow" # Allow extra fields in the settings

# Singleton instance of settings
_settings = None

def get_settings():
    """Get application settings."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings