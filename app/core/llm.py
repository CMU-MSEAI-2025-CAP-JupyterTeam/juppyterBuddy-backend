"""
JupyterBuddy LLM Integration Module

This module handles the integration with various LLM providers and models.
It provides a unified interface for working with different LLMs and implements
a factory pattern for easy swapping between different providers.
"""
import os
import logging
from typing import Optional, List
from enum import Enum

# LangChain imports for LLM integration
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import BaseTool

# Local imports
from app.config import get_settings

# Set up logging
logger = logging.getLogger(__name__)

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"  # Placeholder for future local model support

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
        tools: Optional[List[BaseTool]] = None,  # Allow tool binding
        **kwargs
    ) -> BaseChatModel:
        """
        Create an LLM instance based on the specified provider and parameters.
        
        Args:
            provider: The LLM provider to use
            model_name: The specific model to use (or None for default)
            temperature: The temperature setting for the LLM
            streaming: Whether to enable streaming responses
            tools: Optional list of tools to bind to the LLM
            **kwargs: Additional provider-specific parameters
            
        Returns:
            An instance of the specified LLM
            
        Raises:
            ValueError: If an unsupported provider is specified
        """
        settings = get_settings()

        # Use explicit API keys instead of dynamic lookup
        api_keys = {
            LLMProvider.OPENAI: os.environ.get("OPENAI_API_KEY"),
            LLMProvider.ANTHROPIC: os.environ.get("ANTHROPIC_API_KEY"),
            LLMProvider.GOOGLE: os.environ.get("GOOGLE_API_KEY"),  # Corrected
        }
        api_key = api_keys.get(provider)

        if provider == LLMProvider.OPENAI:
            if model_name is None:
                model_name = "gpt-4-turbo-preview"
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                streaming=streaming,
                api_key=api_key,
                tools=tools,  # Pass tools here
                **kwargs
            )

        elif provider == LLMProvider.ANTHROPIC:
            if model_name is None:
                model_name = "claude-3-opus-20240229"
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                streaming=streaming,
                anthropic_api_key=api_key,
                tools=tools,  # Pass tools here
                **kwargs
            )

        elif provider == LLMProvider.GOOGLE:
            if model_name is None:
                model_name = "gemini-1.5-pro"
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                convert_system_message_to_human=True,  # Needed for Gemini
                google_api_key=api_key,
                tools=tools,  # Pass tools here
                **kwargs
            )

        elif provider == LLMProvider.LOCAL:
            raise NotImplementedError("Local model support is not yet implemented")

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

def get_llm(tools: Optional[List[BaseTool]] = None) -> BaseChatModel:
    """
    Get the LLM instance based on configuration settings.
    
    Args:
        tools: Optional list of tools to bind to the LLM
        
    Returns:
        An instance of the configured LLM
    """
    provider_str = os.environ.get("LLM_PROVIDER", "openai").lower()
    model_name = os.environ.get("LLM_MODEL_NAME", None)
    temperature = float(os.environ.get("LLM_TEMPERATURE", "0.7"))

    # Convert string provider to enum
    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        logger.warning(f"Invalid provider: {provider_str}, falling back to OpenAI")
        provider = LLMProvider.OPENAI

    # Pass tools explicitly to the factory
    try:
        llm = LLMFactory.create_llm(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            tools=tools  # Ensure tools are properly passed
        )
        logger.info(f"Using LLM provider: {provider} with model: {model_name or 'default'}")
        return llm
    except Exception as e:
        logger.error(f"Error creating LLM: {e}")
        logger.warning("Falling back to default OpenAI model")
        return LLMFactory.create_llm(
            provider=LLMProvider.OPENAI,
            temperature=temperature,
            tools=tools  # Ensure tools are properly passed
        )
