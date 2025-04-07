"""Anthropic provider for Model Context Protocol."""

import os
import json
from typing import Dict, List, Optional, Any, Union
import asyncio

from ..mcp.base import MCPBaseProvider


class AnthropicLLMProvider:
    """Anthropic LLM provider implementation."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Anthropic provider.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)
            model: Model to use (default: "claude-3-opus-20240229")
            base_url: Base URL for the Anthropic API (optional)
            **kwargs: Additional parameters for the provider
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided and not found in environment")
        
        self.model = model
        self.base_url = base_url or "https://api.anthropic.com"
        self.client = MCPBaseProvider(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            **kwargs
        )
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using the Anthropic API.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the Anthropic API
            
        Returns:
            The generated response
        """
        # Format prompt for Anthropic
        messages = [{"role": "user", "content": prompt}]
        
        # Use messages API for Anthropic
        try:
            return await self.chat(messages, **kwargs)
        except Exception as e:
            raise ValueError(f"Error generating response from Anthropic: {str(e)}")
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat response using the Anthropic API.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            **kwargs: Additional parameters for the Anthropic API
            
        Returns:
            The generated response
        """
        # Prepare request data for Anthropic
        # Map roles from the generic format to Anthropic format
        anthropic_messages = []
        for msg in messages:
            role = msg["role"]
            if role == "user":
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": msg["content"]})
            elif role == "system":
                # System messages are handled separately in Anthropic
                continue
        
        # Extract system message if present
        system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        
        # Prepare request data
        request_data = {
            "model": kwargs.get("model", self.model),
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        # Add system message if present
        if system_message:
            request_data["system"] = system_message
        
        # Make the API call through the MCP client
        try:
            await self.client._ensure_session()
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01"
            }
            async with self.client.session.post(
                f"{self.base_url}/v1/messages",
                json=request_data,
                headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Error from Anthropic API: {error_text}")
                
                response_data = await response.json()
                
                # Extract the completion from the response
                completion = response_data.get("content", [{}])[0].get("text", "")
                if not completion:
                    raise ValueError("No completion in response from Anthropic API")
                
                return completion
        except Exception as e:
            raise ValueError(f"Error communicating with Anthropic API: {str(e)}")
        finally:
            await self.client._close_session()

       