"""
Base agent implementation for AgentScaffold.

This module provides the core agent functionality, with support for:
- LLM providers via Model Context Protocol (MCP)
- Memory storage and retrieval
- Search capabilities
- Logging and observability
- Secure execution in Daytona runtime environments
"""

import asyncio
import inspect
import logging
import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, ClassVar, Type, Callable
from pathlib import Path

# Optional imports - used if available
try:
    from pydantic import BaseModel, Field, ConfigDict
except ImportError:
    # Fallback minimal BaseModel implementation
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)
    
    def Field(*args, **kwargs):
        return None
    
    ConfigDict = dict


class AgentInput(BaseModel):
    """Base class for agent inputs."""
    message: str = Field(..., description="Input message for the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class AgentOutput(BaseModel):
    """Base class for agent outputs."""
    response: str = Field(..., description="Response from the agent")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseAgent:
    """Base agent class that all scaffolded agents inherit from."""
    
    def __init__(
        self,
        name: str = "Base Agent",
        template_dir: str = "flask",
        description: str = "A base agent with minimal functionality",
        llm_provider: Optional[str] = None,
        memory_provider: Optional[str] = None,
        search_provider: Optional[str] = None,
        logging_provider: Optional[str] = None,
        daytona_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            template_dir: Template directory to use
            description: Description of the agent
            llm_provider: LLM provider name (default: None)
            memory_provider: Memory provider name (default: None)
            search_provider: Search provider name (default: None)
            logging_provider: Logging provider name (default: None)
            daytona_enabled: Whether to use Daytona runtime (default: True)
            **kwargs: Additional configuration options
        """
        self.name = name
        self.description = description
        self.config = kwargs
        self.template_dir = template_dir
        self.daytona_enabled = daytona_enabled
        
        # Setup logging
        self._setup_logging(**kwargs)
        
        # Load environment variables
        self._load_env_vars()
        
        # Initialize specific providers
        self.llm_provider = self._init_provider("llm", llm_provider)
        self.search_provider = self._init_provider("search", search_provider)
        self.memory_provider = self._init_provider("memory", memory_provider)
        self.logging_provider = self._init_provider("logging", logging_provider)
        
        # Initialize MCP provider (for Model Context Protocol)
        self.mcp_provider = self._init_mcp()
        
        # Set input and output classes
        self.input_class = AgentInput
        self.output_class = AgentOutput
        
        # Internal state
        self.silent_mode = False
        self.conversation_history = []
        
        # Initialize DaytonaRuntime if enabled
        self._runtime = DaytonaRuntime() if daytona_enabled else None
        
        self.logger.info(f"Initialized {self.__class__.__name__} agent")
    
    def _setup_logging(self, log_level: str = "INFO", **kwargs):
        """Set up logging for the agent."""
        if not hasattr(self, 'logger'):
            self.logger = logging.getLogger(f"agent.{self.__class__.__name__}")
            level = getattr(logging, log_level.upper(), logging.INFO)
            self.logger.setLevel(level)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
    
    def _load_env_vars(self):
        """Load environment variables from .env files."""
        try:
            from dotenv import load_dotenv
            for env_path in ['.env', '../.env']:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    self.logger.info(f"Loaded environment variables from {env_path}")
                    break
        except ImportError:
            self.logger.info("dotenv package not available, trying manual env loading")
            for env_path in ['.env', '../.env']:
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#') and '=' in line:
                                key, value = line.strip().split('=', 1)
                                os.environ[key.strip()] = value.strip()
                    self.logger.info(f"Manually loaded environment variables from {env_path}")
                    break
    
    def _init_provider(self, provider_type: str, provider_name: Optional[str] = None):
        """
        Initialize a provider of the specified type.
        
        Args:
            provider_type: Type of provider (llm, memory, search, logging)
            provider_name: Name of the provider
            
        Returns:
            Provider instance or None
        """
        if not provider_name:
            self.logger.info(f"No {provider_type} provider specified")
            return None
        
        # Try to load provider dynamically
        try:
            # First, try to import from project-specific providers
            try:
                module_name = f".providers.{provider_type}.{provider_name.lower()}"
                class_name = f"{provider_name.capitalize()}{provider_type.capitalize()}Provider"
                
                # Get the current module
                current_module = sys.modules[self.__class__.__module__]
                
                # Get the package name from the module
                package_name = current_module.__package__
                
                # Import the provider module
                import importlib
                provider_module = importlib.import_module(module_name, package=package_name)
                
                # Get the provider class
                provider_class = getattr(provider_module, class_name)
                
                # Initialize the provider
                provider = provider_class()
                self.logger.info(f"Initialized {provider_type} provider: {provider_name} (project)")
                return provider
            except (ImportError, AttributeError) as e:
                self.logger.debug(f"Could not import project {provider_type} provider: {e}")
                
                # Next, try to import from agentscaffold
                try:
                    from importlib import import_module
                    
                    # For LLM providers, try MCP first
                    if provider_type == "llm":
                        try:
                            module_name = f"agentscaffold.providers.mcp.{provider_name.lower()}"
                            class_name = f"{provider_name.capitalize()}LLMProvider"
                            provider_module = import_module(module_name)
                            provider_class = getattr(provider_module, class_name)
                            provider = provider_class()
                            self.logger.info(f"Initialized {provider_type} provider: {provider_name} (MCP)")
                            return provider
                        except (ImportError, AttributeError) as e:
                            self.logger.debug(f"Could not import MCP {provider_type} provider: {e}")
                    
                    # Try regular provider
                    module_name = f"agentscaffold.providers.{provider_type}.{provider_name.lower()}"
                    class_name = f"{provider_name.capitalize()}{provider_type.capitalize()}Provider"
                    provider_module = import_module(module_name)
                    provider_class = getattr(provider_module, class_name)
                    provider = provider_class()
                    self.logger.info(f"Initialized {provider_type} provider: {provider_name} (agentscaffold)")
                    return provider
                except (ImportError, AttributeError) as e:
                    self.logger.debug(f"Could not import agentscaffold {provider_type} provider: {e}")
                    
                    # Finally, try to import directly
                    try:
                        module_name = provider_name.lower()
                        provider_module = import_module(module_name)
                        # For standard libraries, use their default client class
                        if module_name == "openai":
                            provider = provider_module.OpenAI()
                        elif module_name == "anthropic":
                            provider = provider_module.Anthropic()
                        else:
                            # Try to find a class that matches the provider name
                            for attr_name in dir(provider_module):
                                if attr_name.lower() == provider_name.lower() or attr_name.lower().startswith(provider_name.lower()):
                                    provider_class = getattr(provider_module, attr_name)
                                    if isinstance(provider_class, type):
                                        provider = provider_class()
                                        break
                            else:
                                raise AttributeError(f"Could not find provider class in {module_name}")
                        
                        self.logger.info(f"Initialized {provider_type} provider: {provider_name} (direct)")
                        return provider
                    except (ImportError, AttributeError) as e:
                        self.logger.warning(f"Could not import {provider_type} provider '{provider_name}': {e}")
                        return None
        except Exception as e:
            self.logger.error(f"Error initializing {provider_type} provider '{provider_name}': {e}")
            return None
    
    def _init_mcp(self):
        """Initialize MCP provider for interoperability."""
        try:
            # Try to import client from agentscaffold
            try:
                from agentscaffold.providers.mcp import get_mcp_client
                return get_mcp_client
            except ImportError:
                self.logger.debug("Could not import agentscaffold MCP client")
                
                # Try to import from relative path
                try:
                    # Get the current module
                    current_module = sys.modules[self.__class__.__module__]
                    
                    # Get the package name from the module
                    package_name = current_module.__package__
                    
                    # Import the MCP module
                    import importlib
                    mcp_module = importlib.import_module(f".providers.mcp", package=package_name)
                    
                    # Get the client function
                    get_mcp_client = getattr(mcp_module, "get_mcp_client")
                    
                    self.logger.info("Initialized MCP provider (project)")
                    return get_mcp_client
                except (ImportError, AttributeError) as e:
                    self.logger.debug(f"Could not import project MCP client: {e}")
                    return None
        except Exception as e:
            self.logger.error(f"Error initializing MCP provider: {e}")
            return None
    
    def set_silent_mode(self, silent: bool = True):
        """Set the agent to silent mode (less verbose output)."""
        self.silent_mode = silent
        if hasattr(self, "_runtime") and self._runtime and hasattr(self._runtime, "set_silent_mode"):
            self._runtime.set_silent_mode(silent)
    
    @property
    def runtime(self):
        """Get the DaytonaRuntime instance."""
        return self._runtime
    
    def get_flask_templates_path(self, template_name=None):
        """
        Get path to Flask template files.
        
        Args:
            template_name: Optional specific template name
            
        Returns:
            Path to template directory or specific template file
        """
        template_dir = Path(__file__).parent / "templates" / "flask"
        
        if template_name:
            return template_dir / template_name
        
        return template_dir
    
    async def run(self, input_data: Union[str, Dict[str, Any], AgentInput]) -> Dict[str, Any]:
        """
        Run the agent with the given input.
        
        This is the main entry point for agent execution. It handles:
        - Input validation and normalization
        - Exit commands
        - Execution in Daytona runtime if available
        - Fallback to local execution if needed
        
        Args:
            input_data: Input message or data dictionary
            
        Returns:
            Agent response with metadata
        """
        # Start timing
        start_time = time.time()
        
        # Normalize input
        if isinstance(input_data, str):
            input_data = {"message": input_data}
        elif isinstance(input_data, AgentInput):
            input_data = {"message": input_data.message, "context": input_data.context}
            
        # Check for exit command
        if isinstance(input_data, dict) and input_data.get('message', '').lower().strip() in ['exit', 'quit', 'bye']:
            self.logger.info("Received exit command, cleaning up resources...")
            if hasattr(self, "_runtime") and self._runtime and hasattr(self._runtime, "_cleanup_workspace"):
                self._runtime._cleanup_workspace()
            return {"response": "Session ended. All resources have been cleaned up.", 
                    "metadata": {"exited": True, "processing_time": time.time() - start_time}}
        
        # Validate input
        if not isinstance(input_data, dict):
            self.logger.error(f"Invalid input type: {type(input_data)}")
            return {"response": "Error: Invalid input type", 
                    "metadata": {"error": True, "processing_time": time.time() - start_time}}
        
        if "message" not in input_data:
            self.logger.error("No message in input")
            return {"response": "Error: No message provided", 
                    "metadata": {"error": True, "processing_time": time.time() - start_time}}
        
        # Prepare to run
        message = input_data.get("message", "")
        
        # Log this input to the conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        try:
            # Try Daytona runtime first if enabled
            if self._runtime and self.daytona_enabled:
                self.logger.info("Running agent in Daytona environment")
                try:
                    # Get the agent directory
                    agent_module = inspect.getmodule(self.__class__)
                    if agent_module is None:
                        raise RuntimeError("Cannot determine agent module")
                    agent_file = agent_module.__file__
                    if agent_file is None:
                        raise RuntimeError("Cannot determine agent file path")
                    agent_dir = os.path.dirname(os.path.abspath(agent_file))
                    self.logger.info(f"Using agent directory: {agent_dir}")
                    
                    # Execute in Daytona
                    result = await self._runtime.execute(agent_dir, input_data)
                    
                    # Store response in conversation history
                    if isinstance(result, dict) and "response" in result:
                        self.conversation_history.append({"role": "assistant", "content": result["response"]})
                    
                    # Add processing time
                    processing_time = time.time() - start_time
                    if isinstance(result, dict) and "metadata" in result:
                        result["metadata"]["processing_time"] = processing_time
                    
                    return result
                except Exception as e:
                    self.logger.error(f"❌ Daytona execution error: {e}")
                    self.logger.info("Falling back to local execution due to Daytona error")
            
            # Local execution fallback
            result = await self.process(input_data)
            
            # Store response in conversation history
            if isinstance(result, dict) and "response" in result:
                self.conversation_history.append({"role": "assistant", "content": result["response"]})
                await self.remember(result["response"], {"timestamp": datetime.now().isoformat(), "type": "assistant_message"})

            # Add processing time
            processing_time = time.time() - start_time
            if isinstance(result, dict) and "metadata" in result:
                result["metadata"]["processing_time"] = processing_time
            
            return result
            
        except Exception as e:
            # Handle any uncaught exceptions
            self.logger.error(f"Error in run: {e}", exc_info=True)
            return {"response": f"Error: {str(e)}", 
                    "metadata": {"error": True, "processing_time": time.time() - start_time}}
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input and generate a response locally.
        
        This is the main method to override in subclasses to implement agent-specific logic.
        The default implementation uses the LLM provider if available.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Response dictionary with response text and metadata
        """
        message = input_data.get("message", "")
        context = input_data.get("context", {})
        
        # If we have an LLM provider, use it
        if self.llm_provider:
            try:
                # Prepare prompt with history
                messages = []
                
                # Add system prompt if provided in context
                system_prompt = context.get("system_prompt", f"You are {self.name}, {self.description}")
                messages.append({"role": "system", "content": system_prompt})
                
                # Add conversation history (up to the last 10 exchanges)
                for entry in self.conversation_history[-20:]:
                    messages.append(entry)
                
                # Get response from LLM
                if hasattr(self.llm_provider, "chat"):
                    # Use chat API if available
                    response = await self.llm_provider.chat(messages)
                else:
                    # Fall back to completion API
                    full_prompt = "\n\n".join([
                        f"System: {system_prompt}",
                        *[f"{entry['role'].title()}: {entry['content']}" for entry in self.conversation_history[-20:]]
                    ])
                    response = await self.llm_provider.generate(full_prompt)
                
                return {"response": response, "metadata": {"provider": "llm"}}
            except Exception as e:
                self.logger.error(f"Error in LLM processing: {e}")
                return {"response": f"Error processing with LLM: {str(e)}", "metadata": {"error": True}}
        
        # Fallback to basic echo response
        return {"response": f"Received: {message}", "metadata": {"default": True}}
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a search using the configured search provider.
        
        Args:
            query: Search query
            
        Returns:
            List of search results
        """
        self.logger.info(f"Searching for: {query}")
        
        # If we have a search provider, use it
        if self.search_provider:
            try:
                if hasattr(self.search_provider, "search"):
                    results = await self.search_provider.search(query)
                    return results
                else:
                    self.logger.warning("Search provider does not have search method")
            except Exception as e:
                self.logger.error(f"Error in search: {e}")
        
        # If we have a runtime, try using its search capability
        if self._runtime:
            try:
                results = await self._runtime.search(query)
                return results
            except Exception as e:
                self.logger.error(f"Error in runtime search: {e}")
        
        self.logger.info(f"No search results for: {query}")
        return []
    
    async def remember(self, query: str = "") -> str:
        """
        Retrieve information from memory based on a query.
        
        Args:
            query: Memory retrieval query
            
        Returns:
            Memory content as string
        """
        self.logger.info(f"Retrieving memory for: {query}")
        
        # If we have a memory provider, use it
        if self.memory_provider:
            try:
                if hasattr(self.memory_provider, "retrieve"):
                    content = await self.memory_provider.retrieve(query)
                    return content
                else:
                    self.logger.warning("Memory provider does not have retrieve method")
            except Exception as e:
                self.logger.error(f"Error retrieving memory: {e}")
        
        # If we have a runtime, try using its memory capability
        if self._runtime:
            try:
                results = await self._runtime.retrieve_memory(query)
                if results:
                    # Format results as a string
                    memory_text = "\n\n".join([
                        f"Memory {i+1}:\n{json.dumps(entry.get('data', {}), indent=2)}"
                        for i, entry in enumerate(results)
                    ])
                    return memory_text
            except Exception as e:
                self.logger.error(f"Error in runtime memory retrieval: {e}")
        
        # Return empty string if no memory found
        return ""
    
    async def store_memory(self, data: Dict[str, Any]) -> bool:
        """
        Store data in memory.
        
        Args:
            data: Data to store
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Storing memory: {json.dumps(data)[:100]}...")
        
        # If we have a memory provider, use it
        if self.memory_provider:
            try:
                if hasattr(self.memory_provider, "store"):
                    success = await self.memory_provider.store(data)
                    return success
                else:
                    self.logger.warning("Memory provider does not have store method")
            except Exception as e:
                self.logger.error(f"Error storing memory: {e}")
        
        # If we have a runtime, try using its memory capability
        if self._runtime:
            try:
                success = await self._runtime.store_memory(data)
                return success
            except Exception as e:
                self.logger.error(f"Error in runtime memory storage: {e}")
        
        return False
    
    async def log_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """
        Log an event.
        
        Args:
            event_type: Type of event
            data: Event data
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Logging event: {event_type}")
        
        # If we have a logging provider, use it
        if self.logging_provider:
            try:
                if hasattr(self.logging_provider, "log"):
                    success = await self.logging_provider.log(event_type, data)
                    return success
                else:
                    self.logger.warning("Logging provider does not have log method")
            except Exception as e:
                self.logger.error(f"Error logging event: {e}")
        
        # If we have a runtime, try using its logging capability
        if self._runtime:
            try:
                success = await self._runtime.log_event(event_type, data)
                return success
            except Exception as e:
                self.logger.error(f"Error in runtime logging: {e}")
        
        # Always log to local logger as fallback
        if isinstance(data, dict):
            self.logger.info(f"Event {event_type}: {json.dumps(data)[:200]}...")
        else:
            self.logger.info(f"Event {event_type}: {str(data)[:200]}...")
        
        return True
    
    async def invoke_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a tool by name with the given input.
        
        Args:
            tool_name: Name of the tool to invoke
            tool_input: Input data for the tool
            
        Returns:
            Tool output
        """
        self.logger.info(f"Invoking tool: {tool_name}")
        
        # Check if method exists on the agent
        tool_method = getattr(self, f"tool_{tool_name}", None)
        if tool_method and callable(tool_method):
            try:
                # Check if it's an async method
                if asyncio.iscoroutinefunction(tool_method):
                    result = await tool_method(**tool_input)
                else:
                    result = tool_method(**tool_input)
                return {"status": "success", "result": result}
            except Exception as e:
                self.logger.error(f"Error invoking tool {tool_name}: {e}")
                return {"status": "error", "message": str(e)}
        
        # If MCP provider is available, try it
        if self.mcp_provider:
            try:
                client = await self.mcp_provider(tool_name)
                if client:
                    result = await client.call("v1/tool", {"name": tool_name, "input": tool_input})
                    return result
            except Exception as e:
                self.logger.error(f"Error invoking MCP tool {tool_name}: {e}")
        
        # If runtime is available, try using its MCP capability
        if self._runtime:
            try:
                result = await self._runtime.invoke_mcp(tool_name, tool_input)
                return result
            except Exception as e:
                self.logger.error(f"Error in runtime MCP invocation: {e}")
        
        return {"status": "error", "message": f"Tool {tool_name} not found"}


class DaytonaRuntime:
    """
    Daytona runtime for agent execution with persistent workspace and capabilities.
    
    This class handles:
    - Creating and managing a secure Daytona workspace
    - Executing agent code in the workspace
    - Providing search, memory, and logging capabilities
    - Managing API keys and environment variables
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Daytona runtime.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        self.workspace = None
        self._daytona = None
        self._CreateWorkspaceParams = None
        self._is_initialized = False
        self._agent_dir = None
        self._conversation_id = None
        self._api_keys = {}
        self.silent_mode = False
        
        self._load_env_vars()
        if not os.environ.get("DAYTONA_API_KEY"):
            print("⚠️ DAYTONA_API_KEY environment variable is required for Daytona integration")
        else:
            print(f"✅ Loaded Daytona API key (***{os.environ.get('DAYTONA_API_KEY')[-4:] if len(os.environ.get('DAYTONA_API_KEY', '')) > 4 else ''})")
        
        if not os.environ.get("DAYTONA_API_URL"):
            print("⚠️ DAYTONA_API_URL environment variable not found, using default")
        
        self._collect_api_keys()
        self._load_daytona_sdk()
        
    def set_silent_mode(self, silent: bool = True):
        """Set the runtime to silent mode (less verbose output)."""
        self.silent_mode = silent

    def _print(self, message: str, end="\n", flush=False):
        """Print a message if not in silent mode."""
        if not self.silent_mode:
            print(message, end=end, flush=flush)
    
    def _load_env_vars(self):
        """Load environment variables from .env files."""
        try:
            from dotenv import load_dotenv
            for env_path in ['.env', '../.env']:
                if os.path.exists(env_path):
                    load_dotenv(env_path)
                    break
        except ImportError:
            for env_path in ['.env', '../.env']:
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if line.strip() and not line.startswith('#') and '=' in line:
                                key, value = line.strip().split('=', 1)
                                os.environ[key.strip()] = value.strip()
    
    def _collect_api_keys(self):
        """Collect API keys from environment variables."""
        # Collect common API keys
        api_keys = {
            "OPENAI_API_KEY": "OpenAI",
            "ANTHROPIC_API_KEY": "Anthropic",
            "BRAVE_API_KEY": "Brave Search",
            "LOGFIRE_API_KEY": "LogFire",
        }
        
        for key, name in api_keys.items():
            value = os.environ.get(key)
            if value:
                self._api_keys[key] = value
                self._print(f"✅ Found {name} API key (length: {len(value)})")
    
    def _load_daytona_sdk(self):
        """Load Daytona SDK if available."""
        try:
            from daytona_sdk import Daytona, CreateWorkspaceParams, DaytonaConfig
            api_key = os.environ.get("DAYTONA_API_KEY", self.config.get("api_key"))
            server_url = os.environ.get("DAYTONA_API_URL", self.config.get("server_url"))
            target = os.environ.get("DAYTONA_TARGET", self.config.get("target", "us"))
            if not api_key:
                self._print("❌ No Daytona API key found")
                return
            daytona_config = DaytonaConfig(api_key=api_key, server_url=server_url, target=target)
            self._daytona = Daytona(daytona_config)
            self._CreateWorkspaceParams = CreateWorkspaceParams
            self._print("✅ Initialized Daytona SDK")
        except ImportError as e:
            self._print(f"❌ daytona-sdk package not installed: {e}")
        except Exception as e:
            self._print(f"❌ Daytona initialization failed: {e}")
    
    def _init_workspace(self, agent_dir: str) -> bool:
        """
        Initialize a Daytona workspace.
        
        Args:
            agent_dir: Agent code directory
            
        Returns:
            True if successful, False otherwise
        """
        if self.workspace is not None:
            self._print("🔄 Reusing existing workspace")
            return True
        if not self._daytona:
            self._print("❌ Daytona SDK not initialized")
            return False
        try:
            params = self._CreateWorkspaceParams(language="python")
            self.workspace = self._daytona.create(params)
            self._print(f"🔧 Created workspace ID: {self.workspace.id}")
            if not self._conversation_id:
                self._conversation_id = self._get_persistent_conversation_id(agent_dir)
            return True
        except Exception as e:
            self._print(f"❌ Error creating workspace: {e}")
            return False
    
    def _upload_agent_code(self, agent_dir: str) -> None:
        """
        Upload agent code to the Daytona workspace.
        
        Args:
            agent_dir: Path to agent code directory
        """
        if self._is_initialized and self._agent_dir == agent_dir:
            self._print("🔄 Reusing existing code upload")
            return
        
        self._agent_dir = agent_dir
        try:
            # Create agent directory in workspace
            self.workspace.process.exec("mkdir -p /home/daytona/agent")
            
            # Upload .env file with API keys
            self._print("Creating .env file with API keys...")
            env_content = ""
            for key, value in self._api_keys.items():
                env_content += f"{key}={value}\n"
            self.workspace.fs.upload_file("/home/daytona/agent/.env", env_content.encode('utf-8'))
            self._print("Successfully uploaded .env file with API keys")
            
            # Upload agent code
            self._print(f"Uploading agent code from {agent_dir}...")
            
            # Walk through agent directory and upload files
            for root, dirs, files in os.walk(agent_dir):
                # Skip __pycache__ directories
                if "__pycache__" in root:
                    continue
                
                # Get relative path
                rel_path = os.path.relpath(root, agent_dir)
                if rel_path == ".":
                    rel_path = ""
                
                # Create directory in workspace
                if rel_path:
                    self.workspace.process.exec(f"mkdir -p /home/daytona/agent/{rel_path}")
                
                # Upload files
                for file in files:
                    # Skip compiled Python files
                    if file.endswith(".pyc") or file.endswith(".pyo"):
                        continue
                    
                    src_path = os.path.join(root, file)
                    dst_path = f"/home/daytona/agent/{rel_path}/{file}" if rel_path else f"/home/daytona/agent/{file}"
                    
                    with open(src_path, "rb") as f:
                        content = f.read()
                        self.workspace.fs.upload_file(dst_path, content)
            
            self._print("Successfully uploaded agent code")
            self._is_initialized = True
        except Exception as e:
            self._print(f"Error uploading agent code: {e}")
    
    def _cleanup_workspace(self):
        """Clean up the Daytona workspace."""
        if not self.workspace:
            return
        try:
            self._print("🧹 Cleaning up workspace...")
            self._daytona.remove(self.workspace)
            self.workspace = None
            self._is_initialized = False
        except Exception as e:
            self._print(f"❌ Error cleaning up workspace: {e}")
    
    def _get_persistent_conversation_id(self, agent_dir: str) -> str:
        """
        Get a persistent conversation ID across sessions.
        
        Args:
            agent_dir: Agent code directory
            
        Returns:
            Conversation ID
        """
        # Use a file to store the conversation ID
        conversation_file = os.path.join(agent_dir, ".conversation_id")
        
        if os.path.exists(conversation_file):
            try:
                with open(conversation_file, 'r') as f:
                    conversation_id = f.read().strip()
                    if conversation_id:
                        self._print(f"🔄 Using existing conversation ID: {conversation_id}")
                        return conversation_id
            except Exception as e:
                self._print(f"Error reading conversation ID: {e}")
        
        # If no existing conversation ID, generate a new one
        conversation_id = str(uuid.uuid4())
        try:
            with open(conversation_file, 'w') as f:
                f.write(conversation_id)
            self._print(f"🆕 Created new conversation ID: {conversation_id}")
        except Exception as e:
            self._print(f"Error saving conversation ID: {e}")
        
        return conversation_id
    
    async def execute(self, agent_dir: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent code in the Daytona workspace.
        
        Args:
            agent_dir: Directory containing agent code
            input_data: Input data for the agent
            
        Returns:
            Agent response
        """
        if not self._daytona:
            self._print("❌ Daytona SDK not initialized")
            return {"response": "Error: Daytona SDK not initialized", "metadata": {"error": True}}
        
        try:
            # Initialize workspace if needed
            if not self._init_workspace(agent_dir):
                return {"response": "Error: Failed to initialize workspace", "metadata": {"error": True}}
            
            # Upload agent code
            self._upload_agent_code(agent_dir)
            
            # Ensure conversation ID is set
            if not self._conversation_id:
                self._conversation_id = self._get_persistent_conversation_id(agent_dir)
            
            # Prepare input data
            import base64, json
            input_json = json.dumps(input_data)
            input_base64 = base64.b64encode(input_json.encode('utf-8')).decode('utf-8')
            
            # Generate execution code
            execution_code = self._prepare_execution_code(input_base64, self._conversation_id)
            
            # Execute agent code
            self._print("🚀 Executing agent in Daytona workspace...")
            
            # Create temporary Python file with execution code
            execution_file = "/home/daytona/agent/execute_agent.py"
            self.workspace.fs.upload_file(execution_file, execution_code.encode('utf-8'))
            
            # Set environment variables for execution
            env_vars = " ".join([f"{key}='{value}'" for key, value in self._api_keys.items()])
            
            # Execute the file in the workspace
            response = self.workspace.process.exec(
                f"cd /home/daytona/agent && {env_vars} python3 execute_agent.py"
            )
            
            # Process response
            try:
                # Extract JSON response from output (may contain other print statements)
                import re
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    return result
                else:
                    # If no JSON found, return as plain text
                    return {"response": response, "metadata": {"raw_response": True}}
            except json.JSONDecodeError:
                # If not valid JSON, return as plain text
                return {"response": response, "metadata": {"raw_response": True}}
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self._print(f"❌ Error executing agent: {e}")
            self._print(error_traceback)
            return {"response": f"Error executing agent: {str(e)}", "metadata": {"error": True, "traceback": error_traceback}}
    
    def _prepare_execution_code(self, input_base64, conversation_id):
        """
        Generate optimized execution code with memory, logging, and MCP capabilities.
        This code is uploaded and executed in the Daytona workspace.
        
        Args:
            input_base64: Base64-encoded input data
            conversation_id: Conversation ID
            
        Returns:
            Python code to execute in the workspace
        """
        return f"""
import sys, os, json, base64, traceback, asyncio
from typing import Dict, Any, Optional, List
import time, uuid, urllib.request, urllib.parse

# Ensure agent directory exists
if not os.path.exists('/home/daytona/agent'):
    print("Creating agent directory...")
    os.makedirs('/home/daytona/agent', exist_ok=True)

sys.path.append('/home/daytona/agent')
sys.path.append('/home/daytona')
os.chdir('/home/daytona/agent')

# Load environment variables
try:
    from dotenv import load_dotenv
    if os.path.exists('/home/daytona/agent/.env'):
        load_dotenv('/home/daytona/agent/.env')
        print("✅ Loaded environment variables in Daytona")
    else:
        print("No .env file found in agent directory")
except ImportError:
    if os.path.exists('/home/daytona/agent/.env'):
        with open('/home/daytona/agent/.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#') and '=' in line:
                    key, value = line.strip().split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print("✅ Manually loaded environment variables in Daytona")
    else:
        print("No .env file found in agent directory")

print("Current directory contents:")
try:
    print(os.listdir('.'))
except Exception as e:
    print(f"Error listing directory: {{e}}")

# Check for API keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    print(f"Found OpenAI API key in Daytona (length: {{len(openai_api_key)}})")
anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
if anthropic_api_key:
    print(f"Found Anthropic API key in Daytona (length: {{len(anthropic_api_key)}})")
brave_api_key = os.environ.get("BRAVE_API_KEY")
if brave_api_key:
    print(f"Found Brave API key in Daytona (length: {{len(brave_api_key)}})")
logfire_api_key = os.environ.get("LOGFIRE_API_KEY")
if logfire_api_key:
    print(f"Found LogFire API key in Daytona (length: {{len(logfire_api_key)}})")

CONVERSATION_ID = "{conversation_id}"
print(f"🔄 Processing message in conversation: {{CONVERSATION_ID}}")

input_base64 = "{input_base64}"
input_json = base64.b64decode(input_base64).decode('utf-8')
input_data = json.loads(input_json)
message = input_data.get('message', '')
print(f"📩 Received message: '{{message}}'")

# --- MEMORY IMPLEMENTATION ---
conversation_history = []

def read_conversation_history():
    history_file = '/home/daytona/agent/conversation_history.json'
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading conversation history: {{e}}")
    return []

def save_conversation_history(history):
    history_file = '/home/daytona/agent/conversation_history.json'
    try:
        with open(history_file, 'w') as f:
            json.dump(history, f)
        print(f"Saved {{len(history)}} conversation entries to file")
    except Exception as e:
        print(f"Error saving conversation history: {{e}}")

conversation_history = read_conversation_history()
print(f"Loaded {{len(conversation_history)}} conversation entries from history")

def add_to_memory(text, metadata=None):
    metadata = metadata or {{}}
    entry = {{
        "id": str(uuid.uuid4()),
        "text": text,
        "timestamp": time.time(),
        "metadata": metadata
    }}
    conversation_history.append(entry)
    save_conversation_history(conversation_history)
    return entry["id"]

def get_context(query, n_results=3):
    if not conversation_history:
        return ""
    relevant = []
    for entry in conversation_history:
        if query.lower() in entry.get('text', '').lower():
            relevant.append(entry)
    if not relevant:
        relevant = sorted(conversation_history, key=lambda x: x.get('timestamp', 0), reverse=True)[:n_results]
    else:
        relevant = relevant[:n_results]
    if relevant:
        return "\\n\\n".join([f"Memory {{i+1}}:\\n{{m.get('text', '')}}" for i, m in enumerate(relevant)])
    return ""

# --- SEARCH IMPLEMENTATION ---
def search_with_brave(query):
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return {{"status": "error", "message": "No Brave API key provided"}}
         
    params = urllib.parse.urlencode({{"q": query, "count": 3}})
    url = f"https://api.search.brave.com/res/v1/web/search?{{params}}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("X-Subscription-Token", api_key)
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            results = []
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"][:3]:
                    results.append({{
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "snippet": result.get("description", "")
                    }})
            return {{"status": "success", "results": results}}
    except Exception as e:
        return {{"status": "error", "message": str(e), "results": []}}

# --- LOGGING IMPLEMENTATION ---
def log_to_logfire(event_type, data):
    api_key = os.environ.get("LOGFIRE_API_KEY")
    if not api_key:
        return {{"status": "error", "message": "No LogFire API key provided"}}
    
    try:
        # Prepare log entry
        log_entry = {{
            "timestamp": time.time(),
            "event": event_type,
            "data": data
        }}
        
        # Log to LogFire API or to local file if API not available
        try:
            url = "https://in.logfire.dev/ingest"
            req = urllib.request.Request(url)
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", f"Bearer {{api_key}}")
            
            log_data = json.dumps([log_entry]).encode('utf-8')
            
            with urllib.request.urlopen(req, log_data, timeout=10) as response:
                if response.status == 200:
                    return {{"status": "success"}}
                else:
                    raise Exception(f"LogFire API returned status {{response.status}}")
        except Exception as api_error:
            # Fallback to local logging
            log_file = '/home/daytona/agent/agent_logs.jsonl'
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\\n')
            return {{"status": "success", "message": "Logged locally (API error: {{str(api_error)}})"}}
    except Exception as e:
        return {{"status": "error", "message": str(e)}}

# --- MCP IMPLEMENTATION ---
def invoke_mcp(server_id, input_data):
    print(f"Invoking MCP server: {{server_id}}")
    
    if server_id == "brave-search" or "search" in server_id:
        return search_with_brave(input_data.get("query", ""))
    elif server_id == "memory" or "memory" in server_id:
        operation = input_data.get("operation", "retrieve")
        if operation == "store":
            data = input_data.get("data", {{}})
            return store_in_memory(data)
        else:
            query = input_data.get("query", "")
            return {{"status": "success", "context": get_context(query)}}
    elif server_id == "logfire" or "logging" in server_id:
        event = input_data.get("event", "custom_event")
        data = input_data.get("data", {{}})
        return log_to_logfire(event, data)
    else:
        return {{"status": "error", "message": f"Unknown MCP server: {{server_id}}"}}

def store_in_memory(data):
    if isinstance(data, str):
        data = {{"content": data}}
    
    memory_id = data.get("id", f"mem_{{int(time.time())}}")
    content = data.get("content", str(data))
    
    # Store in conversation history
    memory_entry = f"Memory: {{content}}"
    add_to_memory(memory_entry, {{"type": "stored_memory", "data": data}})
    
    return {{"status": "success", "message": "Stored in memory", "id": memory_id}}

# --- MAIN EXECUTION ---
try:
    # Add user input to conversation history
    add_to_memory(f"User: {{message}}", {{"type": "user_message"}})
    
    # Try to run the main agent process
    print("Loading agent module...")
    try:
        if os.path.exists('agent.py'):
            from agent import BaseAgent
            agent = BaseAgent(name="Daytona Agent")
            print("Agent module loaded successfully")
            
            # Process the input
            result = asyncio.run(agent.process(input_data))
            
            # Check if result is a dictionary
            if isinstance(result, dict) and "response" in result:
                response = result["response"]
                # Add response to conversation history
                add_to_memory(f"Agent: {{response}}", {{"type": "agent_response"}})
                print(f"Response: {{response[:100]}}...")
                print(json.dumps(result))
            else:
                # If not a dictionary, convert to standard format
                print(f"Unexpected result format: {{type(result)}}")
                result = {{"response": str(result), "metadata": {{"raw_result": True}}}}
                print(json.dumps(result))
        else:
            # If no agent.py, use fallback processing
            print("No agent.py found, using fallback processing")
            
            # Basic fallback response
            response = f"Processed message: {{message}}"
            # Add to conversation history
            add_to_memory(f"Agent: {{response}}", {{"type": "fallback_response"}})
            
            result = {{"response": response, "metadata": {{"fallback": True}}}}
            print(json.dumps(result))
    except Exception as e:
        error_message = f"Error processing with agent: {{str(e)}}"
        traceback_str = traceback.format_exc()
        print(f"Error: {{error_message}}")
        print(f"Traceback: {{traceback_str}}")
        
        # Add error to conversation history
        add_to_memory(f"Error: {{error_message}}", {{"type": "error", "traceback": traceback_str}})
        
        result = {{"response": f"Error: {{error_message}}", "metadata": {{"error": True, "traceback": traceback_str}}}}
        print(json.dumps(result))
        
error_message = f"Unexpected error: {{str(e)}}"
    traceback_str = traceback.format_exc()
    print(f"Critical error: {{error_message}}")
    print(f"Traceback: {{traceback_str}}")
    
    result = {{"response": f"Critical error: {{error_message}}", "metadata": {{"critical_error": True, "traceback": traceback_str}}}}
    print(json.dumps(result))
"""