"""OpenAI provider for Model Context Protocol."""

import os
import json
from typing import Dict, List, Optional, Any, Union
import asyncio

from ..mcp.base import MCPBaseProvider


class OpenaiLLMProvider:
    """OpenAI LLM provider implementation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI provider.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY environment variable)
            model: Model to use (default: "gpt-4o")
            base_url: Base URL for the OpenAI API (optional)
            **kwargs: Additional parameters for the provider
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment")
        
        self.model = model
        self.base_url = base_url or "https://api.openai.com"
        self.client = MCPBaseProvider(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            **kwargs
        )
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the OpenAI API.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the OpenAI API
            
        Returns:
            The generated response
        """
        # Format prompt for OpenAI
        messages = [{"role": "user", "content": prompt}]
        
        # Use chat endpoint instead of completions for OpenAI
        try:
            return await self.chat(messages, **kwargs)
        except Exception as e:
            raise ValueError(f"Error generating response from OpenAI: {str(e)}")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat response using the OpenAI API.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            **kwargs: Additional parameters for the OpenAI API
            
        Returns:
            The generated response
        """
        # Prepare request data
        request_data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000),
            "top_p": kwargs.get("top_p", 1.0),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
            "presence_penalty": kwargs.get("presence_penalty", 0.0),
        }
        
        # Make the API call through the MCP client
        try:
            response = await self.client._ensure_session()
            async with self.client.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=request_data
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Error from OpenAI API: {error_text}")
                
                response_data = await response.json()
                
                # Extract the completion from the response
                completion = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not completion:
                    raise ValueError("No completion in response from OpenAI API")
                
                return completion
        except Exception as e:
            raise ValueError(f"Error communicating with OpenAI API: {str(e)}")
        finally:
            await self.client._close_session()