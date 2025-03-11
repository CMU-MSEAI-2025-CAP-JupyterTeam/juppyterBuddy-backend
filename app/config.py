"""
JupyterBuddy Configuration Module

This module handles the application configuration, including environment variables,
settings, and default values. It uses Pydantic for settings validation.
"""
import os
import logging
from typing import Optional, Dict, Any, Union, List
from functools import lru_cache
from pydantic import BaseSettings, Field

# Set up logging
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """
    Application settings class using Pydantic for validation.
    
    This class automatically loads environment variables and provides default values
    for configuration settings.
    """
    # Application settings
    APP_NAME: str = "JupyterBuddy"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "A conversational assistant for JupyterLab"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]  # Allow all origins in development
    
    # LLM settings
    LLM_PROVIDER: str = Field(default="openai", env="LLM_PROVIDER")
    LLM_MODEL_NAME: Optional[str] = Field(default=None, env="LLM_MODEL_NAME")
    LLM_TEMPERATURE: float = Field(default=0.7, env="LLM_TEMPERATURE")
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    # Security settings
    API_KEY_ENCRYPTION_KEY: str = Field(
        default="default_encryption_key_change_in_production",
        env="API_KEY_ENCRYPTION_KEY"
    )
    
    class Config:
        """Pydantic configuration for environment variable loading."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        # Allow environment variables to override fields
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with LRU cache to avoid reloading for each request.
    
    Returns:
        Settings instance with loaded configuration
    """
    settings = Settings()
    logger.debug(f"Loaded settings with LLM provider: {settings.LLM_PROVIDER}")
    return settings