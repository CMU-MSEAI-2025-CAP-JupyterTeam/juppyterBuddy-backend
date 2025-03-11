# app/core/llm.py
import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import openai
import requests
import json

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

class LLMService:
    """Service to handle interactions with different LLM providers."""
    
    def __init__(self):
        """Initialize the LLM service with API keys from environment variables."""
        # OpenAI configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("OpenAI API key loaded")
        else:
            logger.warning("OpenAI API key not found in environment variables")
        
        # Other LLM providers could be added here
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.anthropic_api_key:
            logger.info("Anthropic API key loaded")
        
        # Default model to use
        self.default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              model: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message objects with 'role' and 'content' keys
            model: Optional model name to override the default
            
        Returns:
            Dictionary containing the LLM response
        """
        model = model or self.default_model
        
        # Determine which provider to use based on model name or config
        if model.startswith("gpt-") and self.openai_api_key:
            return await self._generate_openai_response(messages, model)
        elif model.startswith("claude-") and self.anthropic_api_key:
            return await self._generate_anthropic_response(messages, model)
        else:
            logger.error(f"Unsupported model {model} or missing API key")
            return {
                "content": "I apologize, but I'm unable to process your request at the moment due to configuration issues.",
                "actions": []
            }
    
    async def _generate_openai_response(self, messages: List[Dict[str, str]], 
                                       model: str) -> Dict[str, Any]:
        """Generate a response using OpenAI's API."""
        try:
            # Define system message for notebook commands
            system_message = {
                "role": "system",
                "content": """You are JupyterBuddy, an AI assistant integrated into JupyterLab.
You can help users by creating, executing, and updating notebook cells.
When a user asks you to create, run, or manipulate notebook cells, respond with the appropriate actions.
You can:
1. CREATE_CELL - Create new code or markdown cells
2. EXECUTE_CELL - Execute existing cells
3. UPDATE_CELL - Update the content of cells

For example, if a user asks to create a pandas dataframe, respond with a message and include code to create it."""
            }
            
            # Ensure system message is first in the list
            if messages and messages[0].get("role") != "system":
                messages = [system_message] + messages
            
            # Call OpenAI API
            response = await openai.ChatCompletion.acreate(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract actions based on content (a simple heuristic approach)
            content = response.choices[0].message.content
            actions = self._extract_actions_from_content(content)
            
            return {
                "content": content,
                "actions": actions
            }
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {e}")
            return {
                "content": f"I encountered an error while processing your request: {str(e)}",
                "actions": []
            }
    
    async def _generate_anthropic_response(self, messages: List[Dict[str, str]], 
                                          model: str) -> Dict[str, Any]:
        """Generate a response using Anthropic's API."""
        try:
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_to_anthropic_format(messages)
            
            # API endpoint
            url = "https://api.anthropic.com/v1/messages"
            
            # Request headers
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.anthropic_api_key,
                "anthropic-version": "2023-06-01"
            }
            
            # Request body
            data = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": 1000
            }
            
            # Call Anthropic API
            response = requests.post(url, headers=headers, json=data)
            response_data = response.json()
            
            # Extract content
            content = response_data.get("content", [{"text": "No response"}])[0]["text"]
            
            # Extract actions
            actions = self._extract_actions_from_content(content)
            
            return {
                "content": content,
                "actions": actions
            }
        except Exception as e:
            logger.error(f"Error generating Anthropic response: {e}")
            return {
                "content": f"I encountered an error while processing your request: {str(e)}",
                "actions": []
            }
    
    def _convert_to_anthropic_format(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Convert standard message format to Anthropic's format."""
        anthropic_messages = []
        
        for msg in messages:
            role = msg["role"]
            if role == "system":
                # Handle system message differently for Anthropic
                continue
            
            anthropic_role = "user" if role == "user" else "assistant"
            anthropic_messages.append({
                "role": anthropic_role,
                "content": msg["content"]
            })
        
        return anthropic_messages
    
    def _extract_actions_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract actions from the LLM response content.
        This is a simple implementation that looks for code blocks and creates actions.
        A more sophisticated implementation would parse the response for specific commands.
        """
        actions = []
        
        # Look for Python code blocks
        import re
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', content, re.DOTALL)
        
        for i, code in enumerate(code_blocks):
            # Create a cell action for each code block
            actions.append({
                "action_type": "CREATE_CELL",
                "payload": {
                    "cell_type": "code",
                    "content": code.strip(),
                    "position": "end"
                }
            })
        
        return actions