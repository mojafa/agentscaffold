"""Scaffolding functionality for creating agents."""

import os
import shutil
import jinja2
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Set
import json
import logging

# Configure logging
logger = logging.getLogger("agentscaffold.scaffold")

# Try to import Typer-related components for CLI prompts
try:
    import typer
    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False
    # Fallback to input() for prompts

# Path to templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Config file for saved defaults
CONFIG_FILE = Path.home() / ".agentscaffold.json"

# Provider Configuration Dictionary
# Using a nested dictionary structure to keep all provider configs in one place
PROVIDERS = {
    "llm": {
        "openai": {
            "env_vars": ["OPENAI_API_KEY"],
            "package": "openai>=1.0.0",
            "description": "OpenAI API (gpt-4o, etc.)"
        },
        "anthropic": {
            "env_vars": ["ANTHROPIC_API_KEY"],
            "package": "anthropic>=0.5.0",
            "description": "Anthropic API (Claude)"
        },
        "none": {
            "env_vars": [],
            "package": None,
            "description": "No LLM provider"
        }
    },
    "search": {
        "brave": {
            "env_vars": ["BRAVE_API_KEY"],
            "package": "httpx>=0.24.0",
            "description": "Brave Search API"
        },
        "none": {
            "env_vars": [],
            "package": None,
            "description": "No search provider"
        }
    },
    "memory": {
        "chromadb": {
            "env_vars": ["OPENAI_API_KEY"],
            "package": "chromadb>=0.4.0",
            "description": "ChromaDB local vector database"
        },
        "none": {
            "env_vars": [],
            "package": None,
            "description": "No memory provider"
        }
    },
    "logging": {
        "logfire": {
            "env_vars": ["LOGFIRE_API_KEY"],
            "package": "logfire>=0.9.0",
            "description": "LogFire observability platform"
        },
        "none": {
            "env_vars": [],
            "package": None,
            "description": "No logging provider"
        }
    }
}

# Utility packages
UTILITY_PACKAGES = {
    "pypeteer": {
        "package": "pydantic-ai>=0.1.0",
        "description": "Pydantic-based AI model integration"
    },
    "tiktoken": {
        "package": "tiktoken>=0.4.0",
        "description": "Fast BPE tokenizer from OpenAI"
    },
    "jinja2": {
        "package": "jinja2>=3.0.0",
        "description": "Template engine for Python"
    },
    "fastapi": {
        "package": "fastapi>=0.100.0 uvicorn>=0.20.0",
        "description": "FastAPI web framework for building APIs"
    },
    "httpx": {
        "package": "httpx>=0.24.1",
        "description": "HTTP client required for MCP integration"
    }
}

# Default Daytona configuration
DAYTONA_CONFIG = {
    "env_vars": ["DAYTONA_API_KEY", "DAYTONA_API_URL", "DAYTONA_TARGET"],
    "description": "Daytona API for secure execution"
}


class Scaffold:
    """
    Main scaffolding class for creating and managing agent projects.
    """
    
    def __init__(self, templates_dir: Optional[Path] = None, config_file: Optional[Path] = None):
        """
        Initialize the scaffold with template and config paths.
        
        Args:
            templates_dir: Path to templates directory (default: package templates)
            config_file: Path to config file (default: ~/.agentscaffold.json)
        """
        self.templates_dir = templates_dir or TEMPLATES_DIR
        self.config_file = config_file or CONFIG_FILE
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load saved configuration defaults."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error loading config file: {e}")
                return {}
        return {}
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save configuration defaults.
        
        Args:
            config: Configuration to save
        """
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {self.config_file}")
    
    def prompt_choice(self, message: str, choices: List[str], default: Optional[str] = None) -> str:
        """
        Prompt for a choice from a list of options.
        
        Args:
            message: Prompt message
            choices: List of choices
            default: Default choice
            
        Returns:
            Selected choice
        """
        if HAS_TYPER:
            formatted_choices = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])
            choice_prompt = f"{message}\n{formatted_choices}\nChoose an option [1-{len(choices)}]"
            
            default_idx = choices.index(default) + 1 if default in choices else None
            default_prompt = f" (default: {default_idx})" if default_idx else ""
            
            while True:
                result = typer.prompt(f"{choice_prompt}{default_prompt}", default=str(default_idx) if default_idx else None)
                try:
                    idx = int(result)
                    if 1 <= idx <= len(choices):
                        return choices[idx-1]
                    else:
                        typer.echo("Invalid choice. Please try again.")
                except ValueError:
                    typer.echo("Invalid input. Please enter a number.")
        else:
            # Fallback to regular input
            print(message)
            for i, choice in enumerate(choices):
                print(f"{i+1}. {choice}")
            
            default_prompt = f" (default: {choices.index(default) + 1})" if default in choices else ""
            while True:
                try:
                    result = input(f"Choose an option [1-{len(choices)}]{default_prompt}: ")
                    if not result and default in choices:
                        return default
                    idx = int(result)
                    if 1 <= idx <= len(choices):
                        return choices[idx-1]
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
    
    def prompt_multiple_choice(self, message: str, choices: List[str], defaults: Optional[List[str]] = None) -> List[str]:
        """
        Prompt for multiple choices from a list of options.
        
        Args:
            message: Prompt message
            choices: List of choices
            defaults: Default choices
            
        Returns:
            List of selected choices
        """
        defaults = defaults or []
        
        if not HAS_TYPER:
            # Simple fallback for non-typer environments
            print(message)
            for i, choice in enumerate(choices):
                print(f"{i+1}. {choice}")
            
            defaults_str = ", ".join(str(choices.index(d) + 1) for d in defaults) if defaults else ""
            default_prompt = f" (defaults: {defaults_str})" if defaults else ""
            
            while True:
                try:
                    result = input(f"Choose options (comma-separated) [1-{len(choices)}]{default_prompt}: ")
                    if not result and defaults:
                        return defaults
                    
                    selected = []
                    for item in result.split(","):
                        idx = int(item.strip())
                        if 1 <= idx <= len(choices):
                            selected.append(choices[idx-1])
                        else:
                            print(f"Invalid choice: {idx}. Skipping.")
                    
                    if selected:
                        return selected
                    else:
                        print("No valid choices selected. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter comma-separated numbers.")
        else:
            # Use typer for multiple choice
            choices_with_descriptions = [f"{i+1}. {choice}" for i, choice in enumerate(choices)]
            choices_display = "\n".join(choices_with_descriptions)
            
            defaults_str = ", ".join(str(choices.index(d) + 1) for d in defaults) if defaults else ""
            default_prompt = f" (defaults: {defaults_str})" if defaults else ""
            
            prompt = f"{message}\n{choices_display}\nChoose options (comma-separated)"
            
            while True:
                result = typer.prompt(f"{prompt}{default_prompt}")
                
                if not result and defaults:
                    return defaults
                
                try:
                    selected = []
                    for item in result.split(","):
                        idx = int(item.strip())
                        if 1 <= idx <= len(choices):
                            selected.append(choices[idx-1])
                        else:
                            typer.echo(f"Invalid choice: {idx}. Skipping.")
                    
                    if selected:
                        return selected
                    else:
                        typer.echo("No valid choices selected. Please try again.")
                except ValueError:
                    typer.echo("Invalid input. Please enter comma-separated numbers.")
    
    def prompt_yes_no(self, message: str, default: bool = False) -> bool:
        """
        Prompt for a yes/no answer.
        
        Args:
            message: Prompt message
            default: Default value
            
        Returns:
            True for yes, False for no
        """
        if HAS_TYPER:
            return typer.confirm(message, default=default)
        else:
            default_str = "Y/n" if default else "y/N"
            response = input(f"{message} [{default_str}]: ").strip().lower()
            if not response:
                return default
            return response.startswith('y')
    
    def prompt_text(self, message: str, default: Optional[str] = None, validator: Optional[Callable[[str], bool]] = None) -> str:
        """
        Prompt for text input with optional validation.
        
        Args:
            message: Prompt message
            default: Default value
            validator: Validation function
            
        Returns:
            User input
        """
        if HAS_TYPER:
            while True:
                result = typer.prompt(message, default=default)
                if validator is None or validator(result):
                    return result
                typer.echo("Invalid input. Please try again.")
        else:
            default_prompt = f" (default: {default})" if default else ""
            while True:
                result = input(f"{message}{default_prompt}: ")
                if not result and default:
                    return default
                if validator is None or validator(result):
                    return result
                print("Invalid input. Please try again.")
    
    def get_agent_settings(self, 
                      name: Optional[str] = None, 
                      llm_provider: Optional[str] = None,
                      search_provider: Optional[str] = None,
                      memory_provider: Optional[str] = None,
                      logging_provider: Optional[str] = None,
                      utilities: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get agent settings through interactive prompts or defaults.
        
        Args:
            name: Agent name (optional, will prompt if not provided)
            llm_provider: LLM provider name (optional, will prompt if not provided)
            search_provider: Search provider name (optional, will prompt if not provided)
            memory_provider: Memory provider name (optional, will prompt if not provided)
            logging_provider: Logging provider name (optional, will prompt if not provided)
            utilities: List of utility packages (optional, will prompt if not provided)
            
        Returns:
            Dictionary of agent settings
        """
        # Load saved defaults
        defaults = self.config.get('defaults', {})
        
        # Agent name - only prompt if not provided
        if name is None:
            name = self.prompt_text("Enter agent name", validator=lambda s: bool(s.strip()) and not os.path.exists(s))
        
        # Convert kebab-case to snake_case for the package name
        package_name = name.replace("-", "_")
        package_name = self.prompt_text("Enter package name (for Python imports)", default=package_name, 
                                    validator=lambda s: bool(s.strip()) and s.isidentifier())
        
        # Agent description
        description = self.prompt_text("Enter agent description", default=f"An AI agent for {name}")
        
        # LLM Provider
        if llm_provider is None:
            llm_provider_choices = list(PROVIDERS["llm"].keys())
            llm_default = defaults.get('llm_provider', 'openai')
            llm_provider = self.prompt_choice("Select LLM provider:", llm_provider_choices, default=llm_default)
        
        # Prompt for API keys based on provider
        api_keys = {}
        
        # If OpenAI is selected, prompt for API key
        if llm_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            if HAS_TYPER:
                openai_api_key = typer.prompt("Enter your OpenAI API key", hide_input=True)
                typer.echo(f"OpenAI API key received (length: {len(openai_api_key)})")
                api_keys["OPENAI_API_KEY"] = openai_api_key
            else:
                openai_api_key = input("Enter your OpenAI API key: ")
                print(f"OpenAI API key received (length: {len(openai_api_key)})")
                api_keys["OPENAI_API_KEY"] = openai_api_key
        
        # Similarly for Anthropic, if selected
        if llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            if HAS_TYPER:
                anthropic_api_key = typer.prompt("Enter your Anthropic API key", hide_input=True)
                typer.echo(f"Anthropic API key received (length: {len(anthropic_api_key)})")
                api_keys["ANTHROPIC_API_KEY"] = anthropic_api_key
            else:
                anthropic_api_key = input("Enter your Anthropic API key: ")
                print(f"Anthropic API key received (length: {len(anthropic_api_key)})")
                api_keys["ANTHROPIC_API_KEY"] = anthropic_api_key
        
        # Search Provider
        if search_provider is None:
            search_provider_choices = list(PROVIDERS["search"].keys())
            search_default = defaults.get('search_provider', 'none')
            search_provider = self.prompt_choice("Select search provider:", search_provider_choices, default=search_default)
        
        # Memory Provider
        if memory_provider is None:
            memory_provider_choices = list(PROVIDERS["memory"].keys())
            memory_default = defaults.get('memory_provider', 'none')
            memory_provider = self.prompt_choice("Select memory provider:", memory_provider_choices, default=memory_default)
        
        # Logging Provider
        if logging_provider is None:
            logging_provider_choices = list(PROVIDERS["logging"].keys())
            logging_default = defaults.get('logging_provider', 'none')
            logging_provider = self.prompt_choice("Select logging provider:", logging_provider_choices, default=logging_default)
        
        # Utility packages
        if utilities is None:
            utility_choices = list(UTILITY_PACKAGES.keys())
            utility_defaults = defaults.get('utilities', [])
            utilities = self.prompt_multiple_choice("Select utility packages:", utility_choices, defaults=utility_defaults)
        
        # Save as defaults?
        save_as_defaults = self.prompt_yes_no("Save these choices as defaults for future projects?", default=False)
        if save_as_defaults:
            new_defaults = {
                'llm_provider': llm_provider,
                'search_provider': search_provider,
                'memory_provider': memory_provider,
                'logging_provider': logging_provider,
                'utilities': utilities
            }
            self.config['defaults'] = new_defaults
            self.save_config(self.config)
            if HAS_TYPER:
                typer.echo("✅ Settings saved as defaults")
            else:
                print("✅ Settings saved as defaults")
        
        # Gather all settings
        settings = {
            "agent_name": name,
            "project_name": name,  # Add this line to ensure project_name is defined
            "package_name": package_name,
            "agent_class_name": "".join(x.capitalize() for x in package_name.split("_")),
            "description": description,
            "llm_provider": llm_provider,
            "search_provider": search_provider,
            "memory_provider": memory_provider,
            "logging_provider": logging_provider,
            "utilities": utilities,
            "dependencies": self.generate_dependencies(llm_provider, search_provider, memory_provider, logging_provider, utilities),
            "env_vars": self.generate_env_vars(llm_provider, search_provider, memory_provider, logging_provider),
            "api_keys": api_keys
        }

        if isinstance(settings["env_vars"], list):
            settings["env_vars"] = {var: "" for var in settings["env_vars"]}
        
        return settings
            
    def generate_dependencies(self, 
                             llm_provider: str,
                             search_provider: str,
                             memory_provider: str,
                             logging_provider: str,
                             utilities: List[str]) -> List[str]:
        """
        Generate list of dependencies based on selected providers.
        
        Args:
            llm_provider: LLM provider
            search_provider: Search provider
            memory_provider: Memory provider
            logging_provider: Logging provider
            utilities: List of utility packages
            
        Returns:
            List of dependencies
        """
        dependencies = [
            "pydantic>=2.0.0",
            "agentscaffold",
            "pydantic-ai",
            "daytona-sdk>=0.1.0",  # Always include daytona-sdk
        ]
        
        # Add LLM provider package
        if llm_provider in PROVIDERS["llm"] and PROVIDERS["llm"][llm_provider]["package"]:
            dependencies.append(PROVIDERS["llm"][llm_provider]["package"])
        
        # Add Search provider package
        if search_provider != "none" and search_provider in PROVIDERS["search"] and PROVIDERS["search"][search_provider]["package"]:
            dependencies.append(PROVIDERS["search"][search_provider]["package"])
        
        # Add Memory provider package
        if memory_provider != "none" and memory_provider in PROVIDERS["memory"] and PROVIDERS["memory"][memory_provider]["package"]:
            dependencies.append(PROVIDERS["memory"][memory_provider]["package"])
        
        # Add Logging provider package
        if logging_provider != "none" and logging_provider in PROVIDERS["logging"] and PROVIDERS["logging"][logging_provider]["package"]:
            dependencies.append(PROVIDERS["logging"][logging_provider]["package"])
        
        # Add utility packages
        for util in utilities:
            if util in UTILITY_PACKAGES:
                dependencies.append(UTILITY_PACKAGES[util]["package"])
        
        # Deduplicate dependencies
        return list(set(dependencies))
    
    def generate_env_vars(self,
                        llm_provider: str,
                        search_provider: str,
                        memory_provider: str,
                        logging_provider: str) -> Dict[str, str]:
        """
        Generate environment variables based on selected providers.
        Always includes Daytona environment variables.
        
        Args:
            llm_provider: LLM provider
            search_provider: Search provider
            memory_provider: Memory provider
            logging_provider: Logging provider
            
        Returns:
            Dictionary of environment variables with empty string values
        """
        env_vars = {}
        
        # Add LLM provider env vars
        if llm_provider == "openai":
            env_vars["OPENAI_API_KEY"] = ""
        elif llm_provider == "anthropic":
            env_vars["ANTHROPIC_API_KEY"] = ""
        elif llm_provider == "cohere":
            env_vars["COHERE_API_KEY"] = ""
        
        # Add search provider env vars
        if search_provider == "brave":
            env_vars["BRAVE_API_KEY"] = ""
        elif search_provider == "google":
            env_vars["GOOGLE_API_KEY"] = ""
        
        # Add memory provider env vars
        if memory_provider == "pinecone":
            env_vars["PINECONE_API_KEY"] = ""
            env_vars["PINECONE_ENVIRONMENT"] = ""
        elif memory_provider == "supabase":
            env_vars["SUPABASE_URL"] = ""
            env_vars["SUPABASE_KEY"] = ""
        
        # Add logging provider env vars
        if logging_provider == "langfuse":
            env_vars["LANGFUSE_PUBLIC_KEY"] = ""
            env_vars["LANGFUSE_SECRET_KEY"] = ""
        elif logging_provider == "logfire":
            env_vars["LOGFIRE_API_KEY"] = ""
        
        # Always include Daytona environment variables
        for var in DAYTONA_CONFIG["env_vars"]:
            env_vars[var] = ""
        
        # Add the FORCE_DAYTONA flag
        env_vars["FORCE_DAYTONA"] = "true"
        
        return env_vars
    
    def _render_template_file(self, template_file_path, output_dir, settings, is_package_file=False):
        """
        Render a Jinja template file with the given settings.
        
        Args:
            template_file_path: Path to the template file
            output_dir: Directory to output the rendered file
            settings: Template variables
            is_package_file: Whether this is a package file (affects output path)
        """
        # Read template content
        with open(template_file_path, 'r') as f:
            template_content = f.read()
        
        # Create Jinja environment
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.dirname(template_file_path)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Create template from string
        template = env.from_string(template_content)
        
        # Render template with settings
        rendered_content = template.render(**settings)
        
        # Determine output file name (remove .jinja extension)
        file_name = os.path.basename(template_file_path).replace('.jinja', '')
        
        # Handle package file templates that use {{package_name}} placeholder
        if is_package_file:
            file_name = file_name.replace('{{package_name}}', settings['package_name'])
        
        # Create output file path
        output_file_path = os.path.join(output_dir, file_name)
        
        # Write rendered content to output file
        with open(output_file_path, 'w') as f:
            f.write(rendered_content)
        
        # Log creation
        if HAS_TYPER:
            typer.echo(f"Created {output_file_path}")
        else:
            logger.info(f"Created {output_file_path}")
            print(f"Created {output_file_path}")
        
        return output_file_path
    
    def create_project(self, 
                      name: str,
                      template: str = "basic",
                      output_dir: Optional[str] = None,
                      providers: Optional[Dict[str, str]] = None,
                      settings: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new agent project with the specified name and template.
    
        Args:
            name: Name of the agent to create
            template: Template to use (default: basic)
            output_dir: Directory to output the agent (default: current directory)
            providers: Dictionary of provider selections (llm, memory, search, logging)
            settings: Optional pre-defined settings (if None, will derive from name and providers)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if output_dir is None:
                output_dir = os.getcwd()
    
            # Generate settings
            if settings is None:
                # Convert providers to individual provider selections
                llm_provider = providers.get("llm") if providers else None
                search_provider = providers.get("search") if providers else None
                memory_provider = providers.get("memory") if providers else None
                logging_provider = providers.get("logging") if providers else None
                
                # Get the settings, passing the name and providers to avoid duplicate prompts
                settings = self.get_agent_settings(name, llm_provider, search_provider, memory_provider, logging_provider)
            
            # Create agent directory
            agent_dir = os.path.join(output_dir, name)
            os.makedirs(agent_dir, exist_ok=True)
    
            # Create package directory
            package_dir = os.path.join(agent_dir, settings["package_name"])
            os.makedirs(package_dir, exist_ok=True)
    
            # Copy and render template files from the template directory
            template_dir = self.templates_dir / template
            if not template_dir.exists():
                raise ValueError(f"Template '{template}' not found in {self.templates_dir}")
    
            # Define template files to render
            template_files = [
                "README.md.jinja",
                "main.py.jinja",
                "requirements.in.jinja", 
                "pyproject.toml.jinja",
                ".env.example.jinja"
            ]
            
            # Try to render each template file
            for file_name in template_files:
                template_file_path = os.path.join(template_dir, file_name)
                if os.path.exists(template_file_path):
                    self._render_template_file(template_file_path, agent_dir, settings)
            
            # Handle package files
            pkg_template_dir = template_dir / "{{package_name}}"
            if pkg_template_dir.exists():
                for file_name in os.listdir(pkg_template_dir):
                    if file_name.endswith(".jinja"):
                        pkg_file_path = os.path.join(pkg_template_dir, file_name)
                        self._render_template_file(pkg_file_path, package_dir, settings, is_package_file=True)
            
            # Create necessary directories
            provider_dirs = ["llm", "memory", "search", "logging"]
            for pdir in provider_dirs:
                os.makedirs(os.path.join(package_dir, "providers", pdir), exist_ok=True)
                # Create __init__.py files
                with open(os.path.join(package_dir, "providers", pdir, "__init__.py"), "w") as f:
                    f.write(f"# {pdir.capitalize()} providers\n")
            
            # Create the main providers/__init__.py
            with open(os.path.join(package_dir, "providers", "__init__.py"), "w") as f:
                f.write("# Provider modules\n")
    
            # Create .env.example file if the template doesn't exist
            env_example_path = os.path.join(agent_dir, '.env.example')
            if not os.path.exists(env_example_path):
                # Create a default .env.example file
                env_example_content = """# Environment variables for {}
                
            # Add environment variables for selected providers
            """.format(settings["agent_name"])
                env_vars = settings["env_vars"]
                if isinstance(env_vars, list):
                    # Handle list of env vars
                    env_vars_dict = {var: "" for var in env_vars}
                    for env_var in env_vars:
                        env_example_content += f"{env_var}=\n"
                elif isinstance(env_vars, dict):
                    # Handle dict of env vars
                    env_vars_dict = env_vars
                    for env_var, value in env_vars_dict.items():
                        env_example_content += f"{env_var}={value}\n"
                            
                            # Always ensure Daytona environment variables are included
                env_example_content += """
            # Daytona configuration (required for secure execution)
            DAYTONA_API_KEY=your-daytona-api-key
            DAYTONA_API_URL=your-daytona-server-url
            DAYTONA_TARGET=us

            # Force Daytona execution (true/false)
            FORCE_DAYTONA=true
            """
                
                with open(env_example_path, 'w') as f:
                    f.write(env_example_content)

    
            # Create .env file with API keys if provided
            env_path = os.path.join(agent_dir, '.env')
            if "api_keys" in settings and settings["api_keys"]:
                # Create a .env file with the provided API keys
                env_content = """# Environment variables for {}
# Generated with actual API keys during setup

""".format(settings["agent_name"])
            
                # Add API keys from settings
                for key, value in settings["api_keys"].items():
                    if value:  # Only add if the value is not empty
                        env_content += f"{key}={value}\n"
                
                # Add other environment variables without values
                daytona_vars_added = False

                env_vars = settings["env_vars"]
                if isinstance(env_vars, list):
                    for env_var in env_vars:
                        if env_var not in settings["api_keys"]:  # Skip if already added as an API key
                            env_content += f"{env_var}=\n"
                            # Mark if we've added any Daytona variables
                            if env_var.startswith("DAYTONA_"):
                                daytona_vars_added = True
                elif isinstance(env_vars, dict):
                    for env_var, value in env_vars.items():
                        if env_var not in settings["api_keys"]:  # Skip if already added as an API key
                            env_content += f"{env_var}={value}\n"
                            # Mark if we've added any Daytona variables
                            if env_var.startswith("DAYTONA_"):
                                daytona_vars_added = True
                # Only add Daytona section if not already added
                if not daytona_vars_added:
                    env_content += """
# Daytona configuration (required for secure execution)
DAYTONA_API_KEY=
DAYTONA_API_URL=
DAYTONA_TARGET=us

# Force Daytona execution (true/false)
FORCE_DAYTONA=true
"""
                
                with open(env_path, 'w') as f:
                    f.write(env_content)
                
                if HAS_TYPER:
                    typer.echo(f"Created {env_path} with API keys")
                else:
                    print(f"Created {env_path} with API keys")
            else:
                # No API keys provided, copy .env.example as .env
                if os.path.exists(env_example_path) and not os.path.exists(env_path):
                    shutil.copy(env_example_path, env_path)
                    if HAS_TYPER:
                        typer.echo(f"Created {env_path} (copied from .env.example)")
                    else:
                        print(f"Created {env_path} (copied from .env.example)")
            
            # Add .gitignore if it doesn't exist
            gitignore_path = os.path.join(agent_dir, '.gitignore')
            if not os.path.exists(gitignore_path):
                gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
.env
.venv/
venv/
ENV/
env/

# Distribution / packaging
dist/
build/
*.egg-info/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
logs/
*.log

# Local development files
.DS_Store
"""
                with open(gitignore_path, 'w') as f:
                    f.write(gitignore_content)
                if HAS_TYPER:
                    typer.echo(f"Created {gitignore_path}")
                else:
                    print(f"Created {gitignore_path}")
            
            # Generate MCP integration files
            self.generate_mcp_integration(agent_dir, settings)
            
            # Create Daytona workspace configuration
            self.create_daytona_config(agent_dir, settings)
    
            # Special handling for Flask template
            if template == "flask":
                # Add Flask-specific dependencies
                dependencies = settings.get("dependencies", [])
                for dep in ["flask", "python-dotenv"]:
                    if dep not in dependencies:
                        dependencies.append(dep)
                settings["dependencies"] = dependencies
                
                # Create app.py in root directory
                app_template = os.path.join(template_dir, "app.py.jinja")
                if os.path.exists(app_template):
                    self._render_template_file(app_template, agent_dir, settings)
            

            logger.info(f"Successfully created project {name} in {agent_dir}")

        except Exception as e:
            logger.error(f"Error creating project: {e}", exc_info=True)
            if HAS_TYPER:
                typer.echo(f"Error creating project: {e}")
            else:
                print(f"Error creating project: {e}")
            return False
        logger.info(f"Successfully created project {name} in {agent_dir}")
        return True
    def _add_provider_to_mcp_servers(self, mcp_servers: Dict[str, Dict[str, Any]], 
                                    provider_type: str, provider_name: Optional[str], 
                                    settings: Dict[str, Any]) -> None:
        """
        Add a provider configuration to MCP servers.
        
        Args:
            mcp_servers: Dictionary of MCP server configurations
            provider_type: Type of provider (llm, search, memory, logging)
            provider_name: Name of the provider
            settings: Agent settings
        """
        if not provider_name or provider_name == "none":
            return
        
        # Define provider configurations
        provider_configs = {
            "llm": {
                "openai": {
                    "type": "http",
                    "url": "https://api.openai.com",
                    "capability": "llm",
                    "env": {
                        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
                    }
                },
                "anthropic": {
                    "type": "http",
                    "url": "https://api.anthropic.com",
                    "capability": "llm",
                    "env": {
                        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
                    }
                },
                "ollama": {
                    "type": "http",
                    "url": "${OLLAMA_BASE_URL:-http://localhost:11434}",
                    "capability": "llm"
                }
            },
            "search": {
                "brave": {
                    "type": "http",
                    "url": "https://api.search.brave.com",
                    "capability": "search",
                    "env": {
                        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
                    }
                },
                "browserbase": {
                    "type": "http",
                    "url": "https://api.browserbase.com",
                    "capability": "search",
                    "env": {
                        "BROWSERBASE_API_KEY": "${BROWSERBASE_API_KEY}"
                    }
                },
                "google": {
                    "type": "http",
                    "url": "https://www.googleapis.com",
                    "capability": "search",
                    "env": {
                        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}",
                        "GOOGLE_CSE_ID": "${GOOGLE_CSE_ID}"
                    }
                }
            },
            "memory": {
                "chromadb": {
                    "type": "http",
                    "url": "http://localhost:8000",
                    "capability": "memory",
                    "config": {
                        "client_type": "persistent",
                        "data_dir": "${CHROMADB_PERSISTENCE_DIR:-~/.agentscaffold/chroma}"
                    }
                },
                "pinecone": {
                    "type": "http",
                    "url": "https://api.pinecone.io",
                    "capability": "memory",
                    "env": {
                        "PINECONE_API_KEY": "${PINECONE_API_KEY}",
                        "PINECONE_ENVIRONMENT": "${PINECONE_ENVIRONMENT}"
                    }
                },
                "supabase": {
                    "type": "http",
                    "url": "${SUPABASE_URL}",
                    "capability": "memory",
                    "env": {
                        "SUPABASE_KEY": "${SUPABASE_KEY}"
                    }
                }
            },
            "logging": {
                "logfire": {
                    "type": "http",
                    "url": "https://api.logfire.dev",
                    "capability": "logging",
                    "env": {
                        "LOGFIRE_API_KEY": "${LOGFIRE_API_KEY}"
                    }
                },
                "langfuse": {
                    "type": "http",
                    "url": "https://api.langfuse.com",
                    "capability": "logging",
                    "env": {
                        "LANGFUSE_PUBLIC_KEY": "${LANGFUSE_PUBLIC_KEY}",
                        "LANGFUSE_SECRET_KEY": "${LANGFUSE_SECRET_KEY}"
                    }
                }
            }
        }
        
        # Get provider configuration
        if provider_type in provider_configs and provider_name in provider_configs[provider_type]:
            config = provider_configs[provider_type][provider_name].copy()
            
            # Update env vars with actual values from settings if available
            if "env" in config and "api_keys" in settings:
                for env_var, template_value in config["env"].items():
                    if env_var in settings["api_keys"]:
                        # Use actual API key if provided
                        config["env"][env_var] = settings["api_keys"][env_var]
                    
            # Add to MCP servers
            mcp_servers[f"{provider_name}-{provider_type}"] = config
    
    def create_daytona_config(self, agent_dir, settings):
        """Create the Daytona.io configuration file."""
        daytona_config = {
            "defaultEnv": "dev",
            "environments": {
                "dev": {
                    "template": "static",
                    "services": {
                        "app": {
                            "type": "static",
                            "buildpack": {
                                "builder": "heroku/buildpacks:20",
                            },
                            "envSecretMounts": []
                        }
                    }
                }
            }
        }
        
        # Handle env_vars properly, checking if it's a dict or list
        env_vars = settings.get("env_vars", {})
        if isinstance(env_vars, dict):
            allowed_env = list(env_vars.keys())
        elif isinstance(env_vars, list):
            allowed_env = env_vars
        else:
            allowed_env = []
        
        daytona_config["environments"]["dev"]["allowed_env"] = allowed_env
        
        # Write the configuration file
        with open(os.path.join(agent_dir, ".daytona.json"), "w") as f:
            json.dump(daytona_config, f, indent=2)
    
    def generate_mcp_integration(self, agent_dir: str, settings: Dict[str, Any]) -> None:
        """
        Generate MCP integration files with improved configuration.
        
        Args:
            agent_dir: Path to the agent directory
            settings: Template variables
        """
        # Create MCP configuration file
        mcp_config = {
            "name": settings["agent_name"],
            "description": settings["description"],
            "version": "0.1.0",
            "integration_type": "agent",
            "capabilities": {
                "text": True,
                "image": False,
                "audio": False,
                "video": False,
                "file": False
            },
            "auth": {
                "type": "api_key",
                "header_name": "X-MCP-Auth"
            },
            "endpoints": {
                "invoke": {
                    "path": "/api/invoke",
                    "method": "POST"
                }
            },
            "settings": {}
        }
        
        # Define mcpServers configuration
        mcp_servers = {}
        
        # Add provider configurations to MCP servers - enhanced with better defaults
        llm_provider = settings.get("llm_provider")
        if llm_provider == "openai":
            mcp_servers["openai-llm"] = {
                "type": "http",
                "url": "https://api.openai.com",
                "capability": "llm",
                "env": {
                    "OPENAI_API_KEY": "${OPENAI_API_KEY}"
                },
                "auth": {
                    "type": "bearer",
                    "env_var": "OPENAI_API_KEY"
                },
                "endpoints": {
                    "chat": "/v1/chat/completions",
                }
            }
            # Add API key settings
            mcp_config["settings"]["openai-api-key"] = {
                "type": "string",
                "description": "OpenAI API Key",
                "required": True,
                "env_var": "OPENAI_API_KEY"
            }
        elif llm_provider == "anthropic":
            mcp_servers["anthropic-llm"] = {
                "type": "http",
                "url": "https://api.anthropic.com",
                "capability": "llm",
                "env": {
                    "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"
                },
                "endpoints": {
                    "chat": {
                        "path": "/v1/messages",
                        "method": "POST"
                    }
                }
            }
            # Add API key settings
            mcp_config["settings"]["anthropic-api-key"] = {
                "type": "string",
                "description": "Anthropic API Key",
                "required": True,
                "env_var": "ANTHROPIC_API_KEY"
            }
        
        # Add search provider if specified
        search_provider = settings.get("search_provider")
        if search_provider == "brave":
            mcp_servers["search"] = {
                "type": "http",
                "url": "https://api.search.brave.com",
                "capability": "search",
                "env": {
                    "BRAVE_API_KEY": "${BRAVE_API_KEY}"
                },
                 "auth": {
                    "type": "api_key",
                    "header_name": "X-Subscription-Token",
                    "env_var": "BRAVE_API_KEY"
                },
                "endpoints": {
                    "search": "/res/v1/web/search",   
                    }
                }
            
            # Add API key settings
            mcp_config["settings"]["brave-api-key"] = {
                "type": "string",
                "description": "Brave Search API Key",
                "required": True,
                "env_var": "BRAVE_API_KEY"
            }
        
        # Add memory provider if specified
        memory_provider = settings.get("memory_provider")
        if memory_provider == "chromadb":
            mcp_servers["chromadb"] = {
                "type": "http",
                "url": "http://localhost:8000",
                "capability": "memory",
                "endpoints": {
                    "add": {
                        "path": "/api/v1/collections/{collection_id}/add",
                        "method": "POST"
                    },
                    "query": {
                        "path": "/api/v1/collections/{collection_id}/query",
                        "method": "POST"
                    }
                }
            }
        
        # Add logging provider if specified
        logging_provider = settings.get("logging_provider")
        if logging_provider == "logfire":
            mcp_servers["logfire"] = {
                "type": "http",
                "url": "https://api.logfire.dev",
                "capability": "logging",
                "env": {
                    "LOGFIRE_API_KEY": "${LOGFIRE_API_KEY}"
                },
                "endpoints": {
                    "log": {
                        "path": "/ingest",
                        "method": "POST"
                    }
                }
            }
            # Add API key settings
            mcp_config["settings"]["logfire-api-key"] = {
                "type": "string",
                "description": "LogFire API Key",
                "required": True,
                "env_var": "LOGFIRE_API_KEY"
            }
        
        # Add environment variables as settings for any other vars
        env_vars = settings.get("env_vars", {})
        if isinstance(env_vars, dict):
            for env_var, _ in env_vars.items():
                # Skip vars already added and Daytona vars
                if (env_var.startswith("DAYTONA_") or 
                    env_var == "FORCE_DAYTONA" or
                    env_var in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BRAVE_API_KEY", "LOGFIRE_API_KEY"]):
                    continue
                
                # Format the setting name
                setting_name = env_var.lower().replace("_", "-")
                mcp_config["settings"][setting_name] = {
                    "type": "string",
                    "description": f"Value for {env_var}",
                    "required": True,
                    "env_var": env_var
                }
        
        # Add mcpServers to config if any were defined
        if mcp_servers:
            mcp_config["mcpServers"] = mcp_servers
        
        # Write MCP configuration file
        mcp_config_path = os.path.join(agent_dir, ".mcp.json")
        with open(mcp_config_path, 'w') as f:
            json.dump(mcp_config, f, indent=2)
        
        if HAS_TYPER:
            typer.echo(f"Created {mcp_config_path}")
        else:
            print(f"Created {mcp_config_path}")
            
    def create_daytona_config(self, agent_dir, settings):
        """Create the Daytona.io configuration file with improved settings."""
        daytona_config = {
            "defaultEnv": "dev",
            "environments": {
                "dev": {
                    "template": "static",
                    "services": {
                        "app": {
                            "type": "static",
                            "buildpack": {
                                "builder": "heroku/buildpacks:20",
                            },
                            "envSecretMounts": []
                        }
                    }
                }
            }
        }
        
        # Handle env_vars properly, checking if it's a dict or list
        env_vars = settings.get("env_vars", {})
        if isinstance(env_vars, dict):
            allowed_env = list(env_vars.keys())
        elif isinstance(env_vars, list):
            allowed_env = env_vars
        else:
            allowed_env = []
        
        # Always include these critical environment variables
        critical_env_vars = [
            "DAYTONA_API_KEY", 
            "DAYTONA_API_URL", 
            "DAYTONA_TARGET",
            "FORCE_DAYTONA"
        ]
        
        for var in critical_env_vars:
            if var not in allowed_env:
                allowed_env.append(var)
        
        daytona_config["environments"]["dev"]["allowed_env"] = allowed_env
        
        # Write the configuration file
        with open(os.path.join(agent_dir, ".daytona.json"), "w") as f:
            json.dump(daytona_config, f, indent=2)
    
    def create_flask_builder(self, name: str, description: str, output_dir: str) -> Dict[str, Any]:
            """
            Create a Flask application with builder pattern using existing templates.
            
            Args:
                name: Name of the application
                description: Description of the application
                output_dir: Directory to create the application in
                
            Returns:
                Dict with result status and message
            """
            try:
                # Get the templates directory
                templates_source_dir = self.templates_dir / "flask"
                if not templates_source_dir.exists():
                    return {
                        "status": "error",
                        "message": f"Flask templates directory not found at {templates_source_dir}"
                    }
                
                # Create project directory
                project_dir = Path(output_dir) / name
                os.makedirs(project_dir, exist_ok=True)
                
                # Create required directories
                for directory in ["templates", "static/css", "static/js"]:
                    os.makedirs(project_dir / directory, exist_ok=True)
                
                # Setup Jinja environment for template rendering
                env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(templates_source_dir),
                    autoescape=jinja2.select_autoescape(['html', 'xml']),
                    undefined=jinja2.StrictUndefined  # Raise errors for undefined variables
                )
                
                # Define template variables
                template_vars = {
                    "app_name": name,
                    "description": description,
                    "package_name": name.replace("-", "_")
                }
                
                # Render Jinja template files
                for template_file in templates_source_dir.glob("*.jinja"):
                    try:
                        # Read template
                        with open(template_file, 'r') as f:
                            template_content = f.read()
                        
                        # Create template from string
                        template = env.from_string(template_content)
                        
                        # Render template with detailed error handling
                        try:
                            rendered_content = template.render(**template_vars)
                        except jinja2.TemplateError as render_error:
                            logger.error(f"Error rendering template {template_file}: {render_error}")
                            logger.debug(f"Template content:\n{template_content}")
                            logger.debug(f"Template variables: {template_vars}")
                            raise
                        
                        # Get output filename (remove .jinja extension)
                        output_filename = template_file.name.replace('.jinja', '')
                        
                        # Write rendered content
                        with open(project_dir / output_filename, 'w') as f:
                            f.write(rendered_content)
                        
                        if HAS_TYPER:
                            typer.echo(f"Created {output_filename}")
                        else:
                            print(f"Created {output_filename}")
                    
                    except Exception as e:
                        logger.error(f"Failed to process template {template_file}: {e}")
                        # Continue with other templates
                
                # Add Flask-specific routes for builder pattern
                app_py_content = """
        from flask import Flask, request, jsonify, render_template
        import os
        from dotenv import load_dotenv

        # Load environment variables
        load_dotenv()

        app = Flask(__name__)

        @app.route('/')
        def index():
            return render_template('index.html')

        @app.route('/api/build', methods=['POST'])
        def build_component():
            data = request.json
            component_type = data.get('type')
            description = data.get('description')
            
            # Here we'll integrate with the agent to generate components
            try:
                from agent import Agent
                agent = Agent()
                result = agent.build_component(component_type, description)
                return jsonify(result)
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        @app.route('/api/preview', methods=['POST'])
        def preview_component():
            data = request.json
            html_content = data.get('html', '')
            
            # Store temporarily for preview
            with open('templates/preview.html', 'w') as f:
                f.write(html_content)
            
            return jsonify({"success": True, "preview_url": "/preview"})

        @app.route('/preview')
        def show_preview():
            return render_template('preview.html')

        if __name__ == '__main__':
            app.run(debug=True)
        """
                
                with open(project_dir / "app.py", 'w') as f:
                    f.write(app_py_content)
                
                # Create an index.html template
                index_html = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ app_name }} - Web Builder</title>
            <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
        </head>
        <body>
            <header>
                <h1>{{ app_name }}</h1>
                <p>{{ description }}</p>
            </header>
            
            <main>
                <div class="builder-container">
                    <div class="component-selector">
                        <h2>Build a Component</h2>
                        <select id="component-type">
                            <option value="form">Form</option>
                            <option value="navbar">Navigation Bar</option>
                            <option value="card">Card</option>
                            <option value="hero">Hero Section</option>
                            <option value="footer">Footer</option>
                            <option value="sidebar">Sidebar</option>
                            <option value="table">Data Table</option>
                        </select>
                        
                        <textarea id="component-description" placeholder="Describe the component you want to build..."></textarea>
                        
                        <button id="build-button">Build Component</button>
                    </div>
                    
                    <div class="preview-container">
                        <h2>Component Preview</h2>
                        <div id="preview-area">
                            <p>Your component will appear here</p>
                        </div>
                        <div class="code-container">
                            <h3>Generated Code</h3>
                            <pre id="code-preview"></pre>
                            <button id="copy-code-button">Copy Code</button>
                        </div>
                    </div>
                </div>
            </main>
            
            <footer>
                <p>Built with AgentScaffold</p>
            </footer>
            
            <script src="{{ url_for('static', filename='js/builder.js') }}"></script>
        </body>
        </html>"""

                os.makedirs(project_dir / "templates", exist_ok=True)
                with open(project_dir / "templates" / "index.html", 'w') as f:
                    f.write(index_html)
                
                # Create a basic CSS file
                css_content = """/* Basic Styles */
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f9f9f9;
        }

        header {
            background-color: #2c3e50;
            color: white;
            padding: 2rem 1rem;
            text-align: center;
        }

        header h1 {
            margin-bottom: 0.5rem;
        }

        main {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 1rem;
        }

        .builder-container {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        @media (min-width: 768px) {
            .builder-container {
                flex-direction: row;
            }

            .component-selector,
            .preview-container {
                flex: 1;
            }
        }

        .component-selector {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .component-selector h2 {
            margin-bottom: 1rem;
            color: #2c3e50;
        }

        select, textarea, button {
            width: 100%;
            padding: 0.8rem;
            margin-bottom: 1rem;
            border-radius: 4px;
            border: 1px solid #ddd;
        }

        textarea {
            height: 150px;
            resize: vertical;
        }

        button {
            background-color: #3498db;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #2980b9;
        }

        .preview-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .preview-container h2 {
            margin-bottom: 1rem;
            color: #2c3e50;
        }

        #preview-area {
            min-height: 200px;
            border: 1px dashed #ddd;
            padding: 1rem;
            margin-bottom: 1.5rem;
            border-radius: 4px;
        }

        .code-container {
            margin-top: 1.5rem;
        }

        .code-container h3 {
            margin-bottom: 0.5rem;
            color: #2c3e50;
        }

        #code-preview {
            background-color: #f5f5f5;
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
            min-height: 100px;
        }

        #copy-code-button {
            margin-top: 0.5rem;
        }

        footer {
            text-align: center;
            padding: 1.5rem;
            background-color: #2c3e50;
            color: white;
            margin-top: 2rem;
        }
        """
                
                os.makedirs(project_dir / "static" / "css", exist_ok=True)
                with open(project_dir / "static" / "css" / "styles.css", 'w') as f:
                    f.write(css_content)
                
                # Create a JavaScript file for the builder
                js_content = """// Builder JS
        document.addEventListener('DOMContentLoaded', function() {
            const buildButton = document.getElementById('build-button');
            const componentType = document.getElementById('component-type');
            const componentDescription = document.getElementById('component-description');
            const previewArea = document.getElementById('preview-area');
            const codePreview = document.getElementById('code-preview');
            const copyCodeButton = document.getElementById('copy-code-button');
            
            // Build button click handler
            buildButton.addEventListener('click', async function() {
                const type = componentType.value;
                const description = componentDescription.value;
                
                if (!description) {
                    alert('Please describe the component you want to build');
                    return;
                }
                
                buildButton.disabled = true;
                buildButton.textContent = 'Building...';
                previewArea.innerHTML = '<p>Generating your component...</p>';
                
                try {
                    const response = await fetch('/api/build', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            type: type,
                            description: description
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.error) {
                        previewArea.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                        codePreview.textContent = '';
                    } else {
                        // Show the generated HTML
                        previewArea.innerHTML = data.html || 'No HTML was generated';
                        codePreview.textContent = data.html || '';
                        
                        // Send to preview if successful
                        await fetch('/api/preview', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                html: data.html
                            })
                        });
                    }
                } catch (error) {
                    previewArea.innerHTML = `<p class="error">Error: ${error.message}</p>`;
                    codePreview.textContent = '';
                } finally {
                    buildButton.disabled = false;
                    buildButton.textContent = 'Build Component';
                }
            });
            
            // Copy code button handler
            copyCodeButton.addEventListener('click', function() {
                const code = codePreview.textContent;
                if (!code) return;
                
                navigator.clipboard.writeText(code).then(() => {
                    const originalText = copyCodeButton.textContent;
                    copyCodeButton.textContent = 'Copied!';
                    setTimeout(() => {
                        copyCodeButton.textContent = originalText;
                    }, 1500);
                });
            });
        });
        """
                
                os.makedirs(project_dir / "static" / "js", exist_ok=True)
                with open(project_dir / "static" / "js" / "builder.js", 'w') as f:
                    f.write(js_content)
                
                # Create a placeholder preview.html
                preview_html = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Component Preview</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    padding: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .preview-header {
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                .preview-header button {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .preview-container {
                    border: 1px solid #ddd;
                    padding: 20px;
                    border-radius: 4px;
                }
            </style>
        </head>
        <body>
            <div class="preview-header">
                <h1>Component Preview</h1>
                <button onclick="window.close()">Close Preview</button>
            </div>
            <div class="preview-container">
                <!-- Component will be rendered here -->
                <p>No component to preview yet.</p>
            </div>
        </body>
        </html>"""
                
                with open(project_dir / "templates" / "preview.html", 'w') as f:
                    f.write(preview_html)
                
                return {
                    "status": "success",
                    "message": f"Created Flask builder application in {project_dir} using templates",
                    "path": str(project_dir)
                }
                
            except Exception as e:
                logger.error(f"Error creating Flask builder: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"Failed to create Flask builder: {str(e)}"
            }