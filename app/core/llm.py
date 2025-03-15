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
from typing import Optional, List, Dict, Any

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
            LLMProvider.GOOGLE: os.environ.get("GOOGLE_API_KEY"),
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

def get_llm(tools: Optional[List[Dict[str, Any]]] = None) -> BaseChatModel:
    """
    Get the LLM instance based on configuration settings.
    
    Args:
        tools: Optional list of tools in OpenAI format
        
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

    # Use explicit API keys instead of dynamic lookup
    api_keys = {
        LLMProvider.OPENAI: os.environ.get("OPENAI_API_KEY"),
        LLMProvider.ANTHROPIC: os.environ.get("ANTHROPIC_API_KEY"),
        LLMProvider.GOOGLE: os.environ.get("GOOGLE_API_KEY"),
    }
    api_key = api_keys.get(provider)
    
    # Log the provider and model being used
    logger.info(f"Using LLM provider: {provider} with model: {model_name or 'default'}")

    # Create and return the appropriate LLM
    if provider == LLMProvider.OPENAI:
        if model_name is None:
            model_name = "gpt-4-turbo-preview"
        
        # For OpenAI, we can use the tools directly with tool_choice="auto"
        if tools:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                tools=tools,
                tool_choice="auto"  # Enable automatic tool selection
            )
        else:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key
            )

    elif provider == LLMProvider.ANTHROPIC:
        if model_name is None:
            model_name = "claude-3-opus-20240229"
        
        # Anthropic has a different format for tools
        if tools:
            # Convert OpenAI format to Anthropic format if needed
            anthropic_tools = []
            for tool in tools:
                if "function" in tool:
                    # Extract from OpenAI format
                    function_data = tool["function"]
                    anthropic_tool = {
                        "name": function_data.get("name", ""),
                        "description": function_data.get("description", ""),
                        "input_schema": function_data.get("parameters", {})
                    }
                    anthropic_tools.append(anthropic_tool)
            
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                anthropic_api_key=api_key,
                tools=anthropic_tools
            )
        else:
            return ChatAnthropic(
                model=model_name,
                temperature=temperature,
                anthropic_api_key=api_key
            )

    elif provider == LLMProvider.GOOGLE:
        if model_name is None:
            model_name = "gemini-1.5-pro"
        
        # Google may require specific tool formatting as well
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            convert_system_message_to_human=True,  # Needed for Gemini
            google_api_key=api_key,
            # Note: Tool support may vary with Google models
            # Include tools if the model supports it
        )

    elif provider == LLMProvider.LOCAL:
        raise NotImplementedError("Local model support is not yet implemented")

    else:
        # This should not happen as we already handle invalid providers
        # But adding a fallback to OpenAI just in case
        logger.warning(f"Unknown provider {provider}, falling back to OpenAI")
        if model_name is None:
            model_name = "gpt-4-turbo-preview"
            
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_keys.get(LLMProvider.OPENAI)
        )