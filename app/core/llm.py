"""
JupyterBuddy LLM Integration Module

This module handles the integration with various LLM providers and models.
It provides a unified interface for working with different LLMs and implements
a factory pattern for easy swapping between different providers.
"""
import os
import logging
from typing import Dict, Any, Optional, List, Union, Literal
from enum import Enum
from functools import lru_cache

# LangChain imports for LLM integration
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, FunctionMessage

# Local imports
from app.config import get_settings
from app.core.security import get_api_key

# Set up logging
logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"  # For future local model support

class LLMFactory:
    """
    Factory class for creating LLM instances based on configuration.
    Implements the Factory design pattern for LLM creation.
    """
    
    @staticmethod
    def create_llm(
        provider: LLMProvider, 
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        streaming: bool = False,
        tools: Optional[List[BaseTool]] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the specified provider and parameters.
        
        Args:
            provider: The LLM provider to use
            model_name: The specific model to use (or None for default)
            temperature: The temperature setting for the LLM
            streaming: Whether to enable streaming responses
            tools: Optional list of tools to make available to the LLM
            **kwargs: Additional provider-specific parameters
            
        Returns:
            An instance of the specified LLM
            
        Raises:
            ValueError: If an unsupported provider is specified
        """
        settings = get_settings()
        
        # Get the API key for the specified provider
        api_key = get_api_key(provider.value)
        
        if provider == LLMProvider.OPENAI:
            # Default model if not specified
            if model_name is None:
                model_name = "gpt-4-turbo-preview"
                
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                streaming=streaming,
                api_key=api_key,
                **kwargs
            )
        
        elif provider == LLMProvider.ANTHROPIC:
            # Default model if not specified
            if model_name is None:
                model_name = "claude-3-opus-20240229"
                
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                streaming=streaming,
                anthropic_api_key=api_key,
                **kwargs
            )
        
        elif provider == LLMProvider.GOOGLE:
            # Default model if not specified
            if model_name is None:
                model_name = "gemini-1.5-pro"
                
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                convert_system_message_to_human=True,  # Needed for Gemini
                google_api_key=api_key,
                **kwargs
            )
        
        elif provider == LLMProvider.LOCAL:
            # Placeholder for future local model support
            # This could be implemented using LlamaCpp, Ollama, etc.
            raise NotImplementedError("Local model support is not yet implemented")
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

@lru_cache()
def get_llm() -> BaseChatModel:
    """
    Get the LLM instance based on configuration settings.
    Uses LRU cache to avoid recreating the LLM for each request.
    
    Returns:
        An instance of the configured LLM
    """
    settings = get_settings()
    
    # Get LLM configuration from settings
    provider = getattr(settings, "LLM_PROVIDER", LLMProvider.OPENAI)
    model_name = getattr(settings, "LLM_MODEL_NAME", None)
    temperature = getattr(settings, "LLM_TEMPERATURE", 0.7)
    
    # Convert string provider to enum if needed
    if isinstance(provider, str):
        provider = LLMProvider(provider)
    
    # Create the LLM using the factory
    try:
        llm = LLMFactory.create_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature
        )
        logger.info(f"Using LLM provider: {provider} with model: {model_name or 'default'}")
        return llm
    except Exception as e:
        logger.error(f"Error creating LLM: {e}")
        # Fallback to OpenAI if there's an error with the configured provider
        logger.warning("Falling back to default OpenAI model")
        return LLMFactory.create_llm(
            provider=LLMProvider.OPENAI,
            temperature=temperature
        )