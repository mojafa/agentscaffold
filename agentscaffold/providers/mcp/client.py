import os
import json
import asyncio
import subprocess
from typing import Dict, List, Optional, Any, Union
import aiohttp
from pathlib import Path


class MCPClient:
    """Client for Model Context Protocol servers."""
    
    def __init__(
        self,
        server_config: Dict[str, Any],
        timeout: int = 60,
    ):
        """
        Initialize the MCP client.
        
        Args:
            server_config: Server configuration dictionary
            timeout: Request timeout in seconds
        """
        self.server_config = server_config
        self.timeout = timeout
        self.server_type = server_config.get("type", "stdio")
        self.process = None
        self.session = None
        
        # Set environment variables if specified
        if "env" in server_config:
            for key, value in server_config["env"].items():
                os.environ[key] = value
    
    async def _ensure_session(self):
        """Ensure that an HTTP session exists."""
        if self.session is None:
            headers = {}
            # Add API key if available
            if "env" in self.server_config and "API_KEY" in self.server_config["env"]:
                headers["Authorization"] = f"Bearer {self.server_config['env']['API_KEY']}"
            
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
    
    async def _close_session(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def _start_stdio_process(self):
        """Start the stdio MCP process."""
        if self.process is None:
            command = self.server_config.get("command")
            args = self.server_config.get("args", [])
            
            if not command:
                raise ValueError("No command specified for stdio MCP server")
            
            cmd = [command] + args
            env = os.environ.copy()
            if "env" in self.server_config:
                env.update(self.server_config["env"])
            
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
    
    async def _stop_stdio_process(self):
        """Stop the stdio MCP process."""
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except:
                pass
            self.process = None
    
    async def call(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP endpoint.
        
        Args:
            endpoint: The endpoint to call (e.g., "v1/completions")
            data: The request data
            
        Returns:
            The response data
        """
        if self.server_type == "http":
            return await self._http_call(endpoint, data)
        else:
            return await self._stdio_call(data)
    
    async def _http_call(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP call to the MCP server."""
        await self._ensure_session()
        try:
            url = f"{self.server_config['url']}/{endpoint}"
            async with self.session.post(url, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Error from MCP server: {error_text}")
                
                return await response.json()
        except Exception as e:
            raise ValueError(f"Error communicating with MCP server: {str(e)}")
        finally:
            await self._close_session()
    
    async def _stdio_call(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a stdio call to the MCP server."""
        await self._start_stdio_process()
        try:
            # Send the request
            request_json = json.dumps(data) + "\n"
            self.process.stdin.write(request_json.encode())
            await self.process.stdin.drain()
            
            # Read the response
            response_line = await self.process.stdout.readline()
            response_text = response_line.decode().strip()
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON response from MCP server: {response_text}")
        except Exception as e:
            raise ValueError(f"Error communicating with stdio MCP server: {str(e)}")
        finally:
            await self._stop_stdio_process()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The prompt to send to the model
            **kwargs: Additional parameters for the LLM
            
        Returns:
            The generated response
        """
        # Prepare request data
        request_data = {
            "prompt": prompt,
            **kwargs
        }
        
        # Set default model if not provided
        if "model" not in request_data:
            request_data["model"] = self.server_config.get("model", "default")
        
        # Make the API call
        response = await self.call("v1/completions", request_data)
        
        # Extract completion from response
        completion = response.get("completion", "")
        if not completion:
            raise ValueError("No completion in response from MCP server")
        
        return completion
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a chat response from the LLM.
        
        Args:
            messages: List of message dictionaries with "role" and "content" keys
            **kwargs: Additional parameters for the LLM
            
        Returns:
            The generated response
        """
        # Prepare request data
        request_data = {
            "messages": messages,
            **kwargs
        }
        
        # Set default model if not provided
        if "model" not in request_data:
            request_data["model"] = self.server_config.get("model", "default")
        
        # Make the API call
        response = await self.call("v1/chat", request_data)
        
        # Extract completion from response
        completion = response.get("completion", "")
        if not completion:
            raise ValueError("No completion in response from MCP server")
        
        return completion


def load_mcp_servers(location=None) -> List[Dict[str, Any]]:
    """
    Load MCP server configurations.
    
    Args:
        location: Path to look for configuration (default: current directory and user home)
        
    Returns:
        List of server configurations
    """
    config_paths = []
    
    # Add location-specific config path
    if location:
        config_paths.append(Path(location) / ".mcp_config.json")
    
    # Add user config path
    config_paths.append(Path.home() / ".mcp_config.json")
    
    # Add global config path
    global_config_dir = Path("/etc/agentscaffold") if os.name != "nt" else Path(os.environ.get("PROGRAMDATA", "C:\\ProgramData")) / "AgentScaffold"
    config_paths.append(global_config_dir / "mcp_config.json")
    
    # Load from first existing config file
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in {config_path}")
                return []
    
    return []


def save_mcp_servers(servers: List[Dict[str, Any]], location=None) -> bool:
    """
    Save MCP server configurations.
    
    Args:
        servers: List of server configurations
        location: Path to save configuration (default: user home)
        
    Returns:
        True if successful, False otherwise
    """
    if location:
        config_path = Path(location) / ".mcp_config.json"
    else:
        config_path = Path.home() / ".mcp_config.json"
    
    try:
        with open(config_path, "w") as f:
            json.dump(servers, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving configuration: {e}")
        return False


async def _test_connection(server_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test connection to an MCP server.
    
    Args:
        server_config: Server configuration
        
    Returns:
        Dictionary with test results
    """
    try:
        client = MCPClient(server_config, timeout=10)
        
        if server_config.get("type") == "http":
            await client._ensure_session()
            url = f"{server_config['url']}/v1/capability"
            
            async with client.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "capabilities": data}
                else:
                    error_text = await response.text()
                    return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
        else:
            # For stdio servers, we test by sending a ping
            response = await client.call("v1/capability", {})
            return {"success": True, "capabilities": response}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if server_config.get("type") == "http" and client and client.session:
            await client._close_session()


def test_mcp_connection(server_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test connection to an MCP server (synchronous wrapper).
    
    Args:
        server_config: Server configuration
        
    Returns:
        Dictionary with test results
    """
    return asyncio.run(_test_connection(server_config))


def get_provider_by_name(provider_name: str, provider_type: str = "llm", **kwargs) -> Any:
    """
    Get a provider instance by name.
    
    Args:
        provider_name: Name of the provider
        provider_type: Type of provider (llm, memory, search, logging)
        **kwargs: Additional parameters for the provider
        
    Returns:
        Provider instance
    """
    # Try to import providers defined in agentscaffold first, then try standard libraries
    if provider_type == "llm":
        if provider_name.lower() == "openai":
            try:
                from ..llm.openai import OpenaiLLMProvider
                return OpenaiLLMProvider(**kwargs)
            except ImportError:
                import agentscaffold.providers.llm.openai as openai
                return openai.OpenAI(**kwargs)
        elif provider_name.lower() == "anthropic":
            try:
                from ..llm.anthropic import AnthropicLLMProvider
                return AnthropicLLMProvider(**kwargs)
            except ImportError:
                import agentscaffold.providers.llm.anthropic as anthropic
                return anthropic.Anthropic(**kwargs)
        elif provider_name.lower() == "cohere":
            try:
                from .cohere import CohereLLMProvider
                return CohereLLMProvider(**kwargs)
            except ImportError:
                import cohere
                return cohere.Client(**kwargs)
    
    # Try to find MCP server configuration
    servers = load_mcp_servers()
    server = next((srv for srv in servers if srv.get("id") == provider_name), None)
    
    if server:
        return MCPClient(server, **kwargs)
    
    raise ValueError(f"Provider '{provider_name}' not found for type '{provider_type}'")


async def get_mcp_client(name: str, location=None) -> Optional[MCPClient]:
    """
    Get an MCP client for a named server.
    
    Args:
        name: Name of the MCP server
        location: Path to look for configuration
        
    Returns:
        MCPClient instance or None if not found
    """
    servers = load_mcp_servers(location)
    server = next((srv for srv in servers if srv.get("id") == name), None)
    
    if not server:
        return None
    
    return MCPClient(server)