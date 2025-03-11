# app/core/llm.py
import os
import logging
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
import asyncio
import re

# LangChain imports
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for interacting with Language Models using LangChain.
    
    This service provides a unified interface to different LLM providers through
    LangChain's abstractions, making it easy to switch between providers or use
    multiple providers in the same application.
    """
    
    def __init__(self):
        """Initialize the LLM service with available models from environment variables."""
        # Initialize available models dict
        self.available_models = {}
        self.model_instances = {}
        
        # Configure OpenAI models if API key is available
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.available_models.update({
                "gpt-3.5-turbo": "openai",
                "gpt-4": "openai",
                "gpt-4-turbo": "openai"
            })
            logger.info("OpenAI models registered")
        else:
            logger.warning("OpenAI API key not found in environment variables")
        
        # Configure Anthropic models if API key is available
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            self.available_models.update({
                "claude-2": "anthropic",
                "claude-instant-1": "anthropic",
                "claude-3-opus": "anthropic",
                "claude-3-sonnet": "anthropic"
            })
            logger.info("Anthropic models registered")
        else:
            logger.warning("Anthropic API key not found in environment variables")
        
        # Default model
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
        if self.default_model not in self.available_models:
            if self.available_models:
                # Use the first available model as default
                self.default_model = next(iter(self.available_models))
                logger.warning(f"Default model not available, using {self.default_model} instead")
            else:
                logger.error("No models available. Check API keys.")
    
    def _get_langchain_messages(self, messages: List[Dict[str, str]]) -> List[BaseMessage]:
        """
        Convert dict messages to LangChain message objects.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            List of LangChain message objects
        """
        langchain_messages = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))
            else:
                logger.warning(f"Unknown message role: {role}")
        
        return langchain_messages
    
    def _get_model_instance(self, model_name: str):
        """
        Get or create a LangChain model instance for the specified model.
        
        Args:
            model_name: Name of the model to use
            
        Returns:
            LangChain model instance
        """
        # Return cached instance if available
        if model_name in self.model_instances:
            return self.model_instances[model_name]
        
        # Check if model is available
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available")
        
        provider = self.available_models[model_name]
        
        # Create model instance based on provider
        if provider == "openai":
            # Create OpenAI model
            model = ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                streaming=False
            )
        elif provider == "anthropic":
            # Create Anthropic model
            model = ChatAnthropic(
                model=model_name,
                temperature=0.7,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            raise ValueError(f"Unknown provider {provider}")
        
        # Cache and return the model instance
        self.model_instances[model_name] = model
        return model
    
    async def generate_response(self, 
                               messages: List[Dict[str, str]], 
                               model: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            model: Optional model name to override the default
            
        Returns:
            Dictionary containing content and extracted actions
        """
        # Use default model if none specified
        model = model or self.default_model
        
        # Ensure we have a system message
        if not any(msg.get("role") == "system" for msg in messages):
            messages.insert(0, {
                "role": "system",
                "content": self._get_default_system_prompt()
            })
        
        try:
            # Convert to LangChain message format
            langchain_messages = self._get_langchain_messages(messages)
            
            # Get model instance
            llm = self._get_model_instance(model)
            
            # Call the model (run in executor to keep things async-friendly)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: llm.invoke(langchain_messages)
            )
            
            # Extract content from response
            content = response.content
            
            # Extract actions from content
            actions = self._extract_actions(content)
            
            return {
                "content": content,
                "actions": actions
            }
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return {
                "content": f"I encountered an error while processing your request: {str(e)}",
                "actions": []
            }
    
    def _extract_actions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract actions from the content.
        
        Args:
            content: Response content from LLM
            
        Returns:
            List of action objects
        """
        actions = []
        
        # Extract code blocks
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL)
        
        for code in code_blocks:
            # Skip empty code blocks
            if not code.strip():
                continue
                
            actions.append({
                "action_type": "CREATE_CELL",
                "payload": {
                    "cell_type": "code",
                    "content": code.strip(),
                    "position": "end"
                }
            })
        
        return actions
    
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for JupyterBuddy.
        
        Returns:
            System prompt string
        """
        return """You are JupyterBuddy, an AI assistant integrated into JupyterLab.
You can help users by creating, executing, and updating notebook cells.
When a user asks you to create, run, or manipulate notebook cells, respond with the appropriate actions.

When users ask for code:
1. Include the code in ```python code blocks```
2. Provide clear explanations of what the code does
3. Use best practices and modern libraries (pandas, numpy, matplotlib, etc.)

For example, if a user asks to create a pandas dataframe, respond with a message explaining
what you're doing and include the Python code in a code block."""
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available models.
        
        Returns:
            List of available model names
        """
        return list(self.available_models.keys())