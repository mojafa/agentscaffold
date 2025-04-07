#!/usr/bin/env python
"""
Command-line interface for AgentScaffold.
"""

import os
import sys
import subprocess
import json
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# Use try/except for all imports that might not be available
try:
    import typer
    from typing_extensions import Annotated
except ImportError:
    sys.exit("Error: typer package is required. Install with: pip install typer typing-extensions")

# Optional fancy UI components
try:
    from halo import Halo
    HAS_HALO = True
except ImportError:
    HAS_HALO = False

    class Halo:
        def __init__(self, text=None, spinner=None):
            self.text = text

        def start(self):
            print(f"{self.text}...")
            return self

        def succeed(self, text):
            print(f"✅ {text}")

        def fail(self, text):
            print(f"❌ {text}")

# Define provider dictionaries with descriptions
LLM_PROVIDERS = {
    "openai": {"description": "OpenAI API (GPT models)", "package": "openai"},
    "anthropic": {"description": "Anthropic API (Claude models)", "package": "anthropic"},
    "none": {"description": "No LLM provider", "package": None},
}

SEARCH_PROVIDERS = {
    "brave": {"description": "Brave Search API", "package": "brave-search"},
    "google": {"description": "Google Search API", "package": "google-search-results"},
    "browserbase": {"description": "Browser-based search (no API needed)", "package": "pyppeteer"},
    "none": {"description": "No search provider", "package": None},
}

MEMORY_PROVIDERS = {
    "chromadb": {"description": "ChromaDB vector database", "package": "chromadb"},
    "pinecone": {"description": "Pinecone vector database", "package": "pinecone-client"},
    "supabase": {"description": "Supabase vector storage", "package": "supabase"},
    "none": {"description": "No memory provider", "package": None},
}

LOGGING_PROVIDERS = {
    "langfuse": {"description": "Langfuse observability platform", "package": "langfuse"},
    "logfire": {"description": "Logfire logging service", "package": "logfire"},
    "prometheus": {"description": "Prometheus metrics", "package": "prometheus-client"},
    "none": {"description": "No logging provider", "package": None},
}

UTILITY_PACKAGES = {
    "dotenv": {"description": "Environment variable management", "package": "python-dotenv"},
    "pyppeteer": {"description": "Headless browser automation", "package": "pyppeteer"},
    "pydantic": {"description": "Data validation and settings management", "package": "pydantic"},
    "fastapi": {"description": "Fast API framework (for API servers)", "package": "fastapi"},
}

# Templates directory path determination
TEMPLATES_DIR = Path(__file__).parent / "templates"

app = typer.Typer(help="AgentScaffold CLI for creating and managing AI agents")

def run_command_with_spinner(command, cwd, start_message, success_message, error_message):
    """Run a command using a spinner for feedback."""
    if HAS_HALO:
        spinner = Halo(text=start_message, spinner='dots')
        spinner.start()
    else:
        print(start_message)
        spinner = Halo(text=start_message)
    try:
        subprocess.run(command, cwd=cwd, check=True, capture_output=True)
        spinner.succeed(success_message)
    except subprocess.CalledProcessError as e:
        spinner.fail(error_message)
        typer.echo(f"Error details: {e.stderr.decode() if e.stderr else e}")
        raise typer.Exit(1)
    except FileNotFoundError:
        spinner.fail(f"Error: Command not found: {command[0]}")
        raise typer.Exit(1)

def show_provider_options():
    """Display available providers."""
    typer.echo("\nAvailable LLM providers:")
    for key, details in LLM_PROVIDERS.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")

    typer.echo("\nAvailable search providers:")
    for key, details in SEARCH_PROVIDERS.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")

    typer.echo("\nAvailable memory providers:")
    for key, details in MEMORY_PROVIDERS.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")

    typer.echo("\nAvailable logging providers:")
    for key, details in LOGGING_PROVIDERS.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")

    typer.echo("\nAvailable utility packages:")
    for key, details in UTILITY_PACKAGES.items():
        typer.echo(f"  • {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")

def create_new_agent(name, template, output_dir, settings):
    """Create a new agent project."""
    if not TEMPLATES_DIR.exists():
        typer.echo(f"Error: Templates directory not found: {TEMPLATES_DIR}")
        raise typer.Exit(1)
    
    # Check if template exists
    template_dir = TEMPLATES_DIR / template
    if not template_dir.exists():
        typer.echo(f"Error: Template '{template}' not found in {TEMPLATES_DIR}")
        raise typer.Exit(1)
    
    # Create project directory
    project_dir = Path(output_dir) / name
    if project_dir.exists():
        if not typer.confirm(f"Directory {project_dir} already exists. Overwrite?"):
            raise typer.Exit(1)
    
    # Create the project using the scaffold module
    from agentscaffold.scaffold import Scaffold
    
    scaffold = Scaffold()
    providers = {
        "llm": settings.get("llm_provider", "none"),
        "memory": settings.get("memory_provider", "none"),
        "search": settings.get("search_provider", "none"),
        "logging": settings.get("logging_provider", "none"),
    }
    
    # Filter out 'none' providers
    providers = {k: v for k, v in providers.items() if v != "none"}
    
    try:
        scaffold.create_project(
            name=name,
            template=template,
            providers=providers
        )
        
        # Create .env file if API keys are provided
        if settings.get("api_keys"):
            env_path = project_dir / ".env"
            with open(env_path, "w") as env_file:
                for key, value in settings["api_keys"].items():
                    env_file.write(f"{key}={value}\n")
            typer.echo(f"Created .env file with API keys at {env_path}")
        
        return True
    except Exception as e:
        typer.echo(f"Error creating project: {e}")
        return False

def generate_dependencies(llm_provider, search_provider, memory_provider, logging_provider, utilities):
    """Generate a list of dependencies based on selected providers."""
    dependencies = []
    
    # Add LLM provider dependency
    if llm_provider != "none" and llm_provider in LLM_PROVIDERS:
        pkg = LLM_PROVIDERS[llm_provider].get("package")
        if pkg:
            dependencies.append(pkg)
    
    # Add search provider dependency
    if search_provider != "none" and search_provider in SEARCH_PROVIDERS:
        pkg = SEARCH_PROVIDERS[search_provider].get("package")
        if pkg:
            dependencies.append(pkg)
    
    # Add memory provider dependency
    if memory_provider != "none" and memory_provider in MEMORY_PROVIDERS:
        pkg = MEMORY_PROVIDERS[memory_provider].get("package")
        if pkg:
            dependencies.append(pkg)
    
    # Add logging provider dependency
    if logging_provider != "none" and logging_provider in LOGGING_PROVIDERS:
        pkg = LOGGING_PROVIDERS[logging_provider].get("package")
        if pkg:
            dependencies.append(pkg)
    
    # Add utility packages
    for util in utilities:
        if util in UTILITY_PACKAGES:
            pkg = UTILITY_PACKAGES[util].get("package")
            if pkg:
                dependencies.append(pkg)
    
    return dependencies

def generate_env_vars(llm_provider, search_provider, memory_provider, logging_provider):
    """Generate a list of environment variables needed for the selected providers."""
    env_vars = []
    
    # Add LLM provider env vars
    if llm_provider == "openai":
        env_vars.append("OPENAI_API_KEY")
    elif llm_provider == "anthropic":
        env_vars.append("ANTHROPIC_API_KEY")
    
    
    # Add search provider env vars
    if search_provider == "brave":
        env_vars.append("BRAVE_API_KEY")
  
    
    # Add memory provider env vars
   
    # Add logging provider env vars
    
    if logging_provider == "logfire":
        env_vars.append("LOGFIRE_API_KEY")
    
    return env_vars

def get_agent_settings(name: str) -> Dict[str, Any]:
    """Get agent settings through interactive prompts."""
    package_name = name.replace("-", "_")
    settings = {
        "agent_name": name,
        "package_name": package_name,
        "agent_class_name": "".join(x.capitalize() for x in package_name.split("_")),
    }
    
    # Get description
    settings["description"] = typer.prompt(
        "Enter a brief description for this agent",
        default=f"An AI agent for {name}"
    )
    
    # LLM provider
    typer.echo("\nAvailable LLM providers:")
    for i, (key, details) in enumerate(LLM_PROVIDERS.items(), 1):
        typer.echo(f"  {i}. {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")
    
    while True:
        llm_choice = typer.prompt(
            "Select LLM provider",
            default="1"
        )
        try:
            llm_idx = int(llm_choice) - 1
            if 0 <= llm_idx < len(LLM_PROVIDERS):
                settings["llm_provider"] = list(LLM_PROVIDERS.keys())[llm_idx]
                break
        except ValueError:
            pass
        typer.echo(f"Please enter a number between 1 and {len(LLM_PROVIDERS)}")
    
    # If an LLM provider is selected (not "none"), prompt for API key
    if settings["llm_provider"] != "none":
        env_var = ""
        if settings["llm_provider"] == "openai":
            env_var = "OPENAI_API_KEY"
        elif settings["llm_provider"] == "anthropic":
            env_var = "ANTHROPIC_API_KEY"
        
        
        if env_var:
            api_key = typer.prompt(
                f"Enter your {settings['llm_provider']} API key (or leave blank to set later)",
                default="",
                hide_input=True
            )
            if api_key:
                settings["api_keys"] = {env_var: api_key}
    
    # Search provider
    typer.echo("\nAvailable search providers:")
    for i, (key, details) in enumerate(SEARCH_PROVIDERS.items(), 1):
        typer.echo(f"  {i}. {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")
    
    while True:
        search_choice = typer.prompt(
            "Select search provider",
            default=str(len(SEARCH_PROVIDERS))  # Default to "none"
        )
        try:
            search_idx = int(search_choice) - 1
            if 0 <= search_idx < len(SEARCH_PROVIDERS):
                settings["search_provider"] = list(SEARCH_PROVIDERS.keys())[search_idx]
                break
        except ValueError:
            pass
        typer.echo(f"Please enter a number between 1 and {len(SEARCH_PROVIDERS)}")
    
    # Memory provider
    typer.echo("\nAvailable memory providers:")
    for i, (key, details) in enumerate(MEMORY_PROVIDERS.items(), 1):
        typer.echo(f"  {i}. {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")
    
    while True:
        memory_choice = typer.prompt(
            "Select memory provider",
            default=str(len(MEMORY_PROVIDERS))  # Default to "none"
        )
        try:
            memory_idx = int(memory_choice) - 1
            if 0 <= memory_idx < len(MEMORY_PROVIDERS):
                settings["memory_provider"] = list(MEMORY_PROVIDERS.keys())[memory_idx]
                break
        except ValueError:
            pass
        typer.echo(f"Please enter a number between 1 and {len(MEMORY_PROVIDERS)}")
    
    # Logging provider
    typer.echo("\nAvailable logging providers:")
    for i, (key, details) in enumerate(LOGGING_PROVIDERS.items(), 1):
        typer.echo(f"  {i}. {typer.style(key, fg=typer.colors.CYAN)}: {details['description']}")
    
    while True:
        logging_choice = typer.prompt(
            "Select logging provider",
            default=str(len(LOGGING_PROVIDERS))  # Default to "none"
        )
        try:
            logging_idx = int(logging_choice) - 1
            if 0 <= logging_idx < len(LOGGING_PROVIDERS):
                settings["logging_provider"] = list(LOGGING_PROVIDERS.keys())[logging_idx]
                break
        except ValueError:
            pass
        typer.echo(f"Please enter a number between 1 and {len(LOGGING_PROVIDERS)}")
    
    # Utility packages
    settings["utilities"] = []
    for key, details in UTILITY_PACKAGES.items():
        if typer.confirm(f"Include {key} ({details['description']})?", default=key == "dotenv"):
            settings["utilities"].append(key)
    
    # Generate dependencies and env vars
    settings["dependencies"] = generate_dependencies(
        settings["llm_provider"],
        settings["search_provider"],
        settings["memory_provider"],
        settings["logging_provider"],
        settings["utilities"],
    )
    
    settings["env_vars"] = generate_env_vars(
        settings["llm_provider"],
        settings["search_provider"],
        settings["memory_provider"],
        settings["logging_provider"],
    )
    
    return settings

# Enhanced CLI functions for better Daytona and MCP integration

@app.command()
def new(
    name: Annotated[Optional[str], typer.Argument(help="Name of the agent to create")] = None,
    template: Annotated[str, typer.Option(help="Template to use")] = "basic",
    output_dir: Annotated[Optional[str], typer.Option(help="Directory to output the agent (default: current directory)")] = None,
    skip_install: Annotated[bool, typer.Option(help="Skip installing dependencies")] = False,
    interactive: Annotated[bool, typer.Option("--interactive/--no-interactive", "-i", help="Interactive prompts for configuration")] = True,
    llm_provider: Annotated[Optional[str], typer.Option(help="LLM provider (e.g., openai, anthropic, none)")] = None,
    search_provider: Annotated[Optional[str], typer.Option(help="Search provider (e.g., brave, browserbase, none)")] = "none",
    memory_provider: Annotated[Optional[str], typer.Option(help="Memory provider (e.g., supabase, none)")] = "none",
    logging_provider: Annotated[Optional[str], typer.Option(help="Logging provider (e.g., logfire, none)")] = "none",
    utilities: Annotated[Optional[List[str]], typer.Option(help="Utility packages to include, comma-separated")] = ["dotenv"],
    list_providers: Annotated[bool, typer.Option("--list-providers", "-l", help="List available providers and exit")] = False,
    api_key: Annotated[Optional[str], typer.Option(help="API key for the selected LLM provider")] = None,
    enable_mcp: Annotated[bool, typer.Option(help="Enable MCP integration")] = True,
):
    """
    Create a new agent with the specified name and template.
    """
    if list_providers:
        show_provider_options()
        raise typer.Exit()

    if output_dir is None:
        output_dir = os.getcwd()

    if name is None:
        name = typer.prompt("Enter agent name")

    agent_dir = os.path.join(output_dir, name)
    styled_name = typer.style(name, fg=typer.colors.BRIGHT_WHITE, bold=True)
    styled_template = typer.style(template, fg=typer.colors.CYAN)
    typer.echo(f"✨ Creating new agent '{styled_name}' using template '{styled_template}'...")

    # Get settings either from CLI args or interactive prompts
    if not interactive:
        # Use CLI args directly
        package_name = name.replace("-", "_")
        
        # Validate providers
        if llm_provider and llm_provider not in LLM_PROVIDERS:
            typer.echo(f"Error: Invalid LLM provider '{llm_provider}'.")
            typer.echo("Use --list-providers to see available options.")
            raise typer.Exit(1)
        if search_provider and search_provider not in SEARCH_PROVIDERS:
            typer.echo(f"Error: Invalid search provider '{search_provider}'.")
            typer.echo("Use --list-providers to see available options.")
            raise typer.Exit(1)
        if memory_provider and memory_provider not in MEMORY_PROVIDERS:
            typer.echo(f"Error: Invalid memory provider '{memory_provider}'.")
            typer.echo("Use --list-providers to see available options.")
            raise typer.Exit(1)
        if logging_provider and logging_provider not in LOGGING_PROVIDERS:
            typer.echo(f"Error: Invalid logging provider '{logging_provider}'.")
            typer.echo("Use --list-providers to see available options.")
            raise typer.Exit(1)
        
        # Validate utilities
        for util in utilities:
            if util not in UTILITY_PACKAGES and util != "none":
                typer.echo(f"Warning: Unknown utility package '{util}'.")
        
        # Create settings dict
        settings = {
            "agent_name": name,
            "project_name": name,
            "package_name": package_name,
            "agent_class_name": "".join(x.capitalize() for x in package_name.split("_")),
            "description": f"An AI agent for {name}",
            "llm_provider": llm_provider or "none",
            "search_provider": search_provider,
            "memory_provider": memory_provider,
            "logging_provider": logging_provider,
            "utilities": utilities or ["dotenv"],
        }
        
        # Add API key if provided
        if api_key and llm_provider:
            if llm_provider == "openai":
                settings["api_keys"] = {"OPENAI_API_KEY": api_key}
                typer.echo(f"Using provided API key for {llm_provider} (length: {len(api_key)})")
            elif llm_provider == "anthropic":
                settings["api_keys"] = {"ANTHROPIC_API_KEY": api_key}
                typer.echo(f"Using provided API key for {llm_provider} (length: {len(api_key)})")
            
        
        # Generate dependencies and env vars
        settings["dependencies"] = generate_dependencies(
            settings["llm_provider"],
            settings["search_provider"],
            settings["memory_provider"],
            settings["logging_provider"],
            settings["utilities"],
        )
        settings["env_vars"] = generate_env_vars(
            settings["llm_provider"],
            settings["search_provider"],
            settings["memory_provider"],
            settings["logging_provider"],
        )
    else:
        # Use interactive prompts
        from agentscaffold.scaffold import Scaffold
        scaffold = Scaffold()
        settings = scaffold.get_agent_settings(name)
        
        # Ensure project_name is set
        if "project_name" not in settings:
            settings["project_name"] = name

    # Create project with the Scaffold class
    from agentscaffold.scaffold import Scaffold
    
    scaffold = Scaffold()
    # Create the provider dictionary
    providers = {
        "llm": settings.get("llm_provider"),
        "search": settings.get("search_provider"),
        "memory": settings.get("memory_provider"),
        "logging": settings.get("logging_provider"),
    }
    # Filter out 'none' providers
    providers = {k: v for k, v in providers.items() if v != "none"}
    
    # Call create_project with the settings we already prepared
    try:
        result = scaffold.create_project(
            name=name,
            template=template,
            output_dir=output_dir,
            providers=providers,
            settings=settings  # Pass the complete settings
        )
        
        if not result:
            typer.echo("Failed to create project.")
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error creating project: {e}")
        raise typer.Exit(1)

    # Setup MCP integration if enabled
    if enable_mcp:
        typer.echo("🔌 Setting up MCP integration...")
        # This automatically generates the .mcp.json file with proper configuration
        try:
            scaffold.generate_mcp_integration(agent_dir, settings)
            typer.echo("✅ MCP integration configured")
        except Exception as e:
            typer.echo(f"⚠️ Warning: Failed to configure MCP integration: {e}")
    
    # Setup Daytona configuration
    try:
        scaffold.create_daytona_config(agent_dir, settings)
        typer.echo("✅ Daytona configuration created")
    except Exception as e:
        typer.echo(f"⚠️ Warning: Failed to create Daytona configuration: {e}")

    # Setup virtual environment and install dependencies
    has_uv = False
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        has_uv = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    typer.echo("\n✅ Your agent has been created successfully!")
    typer.echo("👉 To get started, run:")
    typer.echo(f"   cd {name}")
    typer.echo("   uv pip install -e .")
    typer.echo("   agentscaffold run")

    
@app.command()
def run(
    agent_dir: Annotated[Optional[str], typer.Option(help="Directory containing the agent")] = ".",
    message: Annotated[Optional[str], typer.Option("--message", "-m", help="Message to process")] = None,
    interactive: Annotated[bool, typer.Option("--interactive/--no-interactive", "-i", help="Interactive prompts for configuration")] = True,
    search: Annotated[Optional[str], typer.Option("--search", "-s", help="Search query")] = None,
    context: Annotated[bool, typer.Option("--context", "-c", help="Retrieve context from memory")] = False,
    context_query: Annotated[Optional[str], typer.Option(help="Query for context retrieval")] = None,
    silent: Annotated[bool, typer.Option("--silent", help="Run with minimal console output")] = False,
    mcp_tool: Annotated[Optional[str], typer.Option("--mcp-tool", help="Specify MCP tool to use")] = None,
    mcp_query: Annotated[Optional[str], typer.Option("--mcp-query", help="Query for MCP tool")] = None,
):
    """
    Run an agent in the specified directory.
    """
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(agent_dir, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
            typer.echo(f"Loaded environment variables from {env_path}")
        else:
            typer.echo(f"Warning: No .env file found at {env_path}")
    except ImportError:
        typer.echo("Warning: python-dotenv not installed, using system environment variables only")
    main_py = os.path.join(agent_dir, "main.py")
    if not os.path.exists(main_py):
        typer.echo(f"Error: {typer.style(main_py, fg=typer.colors.RED)} not found.")
        if agent_dir == ".":
            typer.echo("\nYou seem to be running this command from the agentscaffold project directory.")
            typer.echo("The 'run' command needs to be executed from inside an agent project directory.")
            typer.echo("\nTry one of these instead:")
            typer.echo("  1. cd into an agent project directory first:")
            typer.echo("     cd your-agent-project")
            typer.echo("     agentscaffold run")
            typer.echo("  2. Or specify an agent directory:")
            typer.echo("     agentscaffold run --agent-dir your-agent-project")
        raise typer.Exit(1)
    typer.echo(f"🚀 Running agent in '{typer.style(agent_dir, bold=True)}'...")
    cmd = [sys.executable, main_py]
    if interactive:
        cmd.append("--interactive")
    if silent:
        cmd.append("--silent")
    if message:
        cmd.extend(["--message", message])
    if search:
        cmd.extend(["--search", search])
    if context:
        cmd.append("--context")
    if context_query:
        cmd.extend(["--context-query", context_query])
    if mcp_tool:
        cmd.extend(["--mcp-tool", mcp_tool])
    if mcp_query:
        cmd.extend(["--mcp-query", mcp_query])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        typer.echo(f"🔥 Error running agent: {typer.style(str(e), fg=typer.colors.RED)}")
        raise typer.Exit(1)

@app.command()
def version():
    """Display the current version of AgentScaffold."""
    try:
        from agentscaffold import __version__
        typer.echo(f"✨ AgentScaffold v{typer.style(__version__, fg=typer.colors.CYAN)}")
    except (ImportError, AttributeError):
        typer.echo("⚠️ Could not determine AgentScaffold version")

# Define MCP subcommand group
mcp_app = typer.Typer(help="Manage MCP (Model Context Protocol) integrations")
app.add_typer(mcp_app, name="mcp")

@mcp_app.command("add")
def mcp_add(
    provider: Annotated[str, typer.Argument(help="Provider to add (e.g., openai, anthropic, brave)")],
    key: Annotated[str, typer.Option("--key", "-k", help="API key for the provider")] = None,
    agent_dir: Annotated[str, typer.Option(help="Directory of the agent")] = ".",
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Default model to use")] = None,
):
    """
    Add an MCP provider to an agent.
    """
    # Map provider names to standard MCP server IDs and configurations
    provider_configs = {
        "openai": {
            "server_id": "openai",
            "url": "https://api.openai.com",
            "capability": "llm",
            "env_var": "OPENAI_API_KEY",
            "default_model": "gpt-4o-2024-05-13",
            "endpoints": {
                "chat": {
                    "path": "/v1/chat/completions",
                    "method": "POST"
                }
            }
        },
        "anthropic": {
            "server_id": "anthropic",
            "url": "https://api.anthropic.com",
            "capability": "llm",
            "env_var": "ANTHROPIC_API_KEY",
            "default_model": "claude-3-opus-20240229",
            "endpoints": {
                "chat": {
                    "path": "/v1/messages",
                    "method": "POST"
                }
            }
        },
        "brave": {
            "server_id": "brave",
            "url": "https://api.search.brave.com",
            "capability": "search",
            "env_var": "BRAVE_API_KEY",
            "endpoints": {
                "search": {
                    "path": "/res/v1/web/search",
                    "method": "GET"
                }
            }
        },
        "logfire": {
            "server_id": "logfire",
            "url": "https://in.logfire.dev",
            "capability": "logging",
            "env_var": "LOGFIRE_API_KEY",
            "endpoints": {
                "log": {
                    "path": "/ingest",
                    "method": "POST"
                }
            }
        },
        "chromadb": {
            "server_id": "chromadb",
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
    }
    
    if provider not in provider_configs:
        typer.echo(f"Error: Unsupported provider '{provider}'. Supported providers: {', '.join(provider_configs.keys())}")
        raise typer.Exit(1) 
    
    # Get provider configuration
    config = provider_configs[provider]
    
    # Prompt for API key if not provided and provider requires one
    if not key and "env_var" in config:
        key = typer.prompt(f"Enter your {provider} API key", hide_input=True)
    
    # Check if agent exists
    agent_dir_path = Path(agent_dir)
    if not agent_dir_path.exists() or not agent_dir_path.is_dir():
        typer.echo(f"Error: Agent directory '{agent_dir}' not found")
        raise typer.Exit(1)
    
    # Check for mcp.json file
    mcp_file = agent_dir_path / ".mcp.json"
    
    # Load existing or create new MCP config
    mcp_config = {}
    if mcp_file.exists():
        try:
            with open(mcp_file, "r") as f:
                mcp_config = json.load(f)
        except json.JSONDecodeError:
            typer.echo(f"Warning: Invalid MCP config file at {mcp_file}, creating new one")
            mcp_config = {}
    
    # Initialize mcpServers section if needed
    if "mcpServers" not in mcp_config:
        mcp_config["mcpServers"] = {}
    
    # Initialize settings section if needed
    if "settings" not in mcp_config:
        mcp_config["settings"] = {}
    
    # Set server ID
    server_id = config["server_id"]
    
    # Add MCP server configuration
    mcp_config["mcpServers"][server_id] = {
        "type": "http",
        "url": config["url"],
        "capability": config["capability"],
    }
    
    # Add environment variables if applicable
    if "env_var" in config:
        if "env" not in mcp_config["mcpServers"][server_id]:
            mcp_config["mcpServers"][server_id]["env"] = {}
        
        env_var = config["env_var"]
        mcp_config["mcpServers"][server_id]["env"][env_var] = "${" + env_var + "}"
        
        # Add setting for API key
        setting_name = f"{provider}-api-key"
        mcp_config["settings"][setting_name] = {
            "type": "string",
            "description": f"API key for {provider}",
            "env_var": env_var,
            "required": True
        }
    
    # Add endpoints if defined
    if "endpoints" in config:
        mcp_config["mcpServers"][server_id]["endpoints"] = config["endpoints"]
    
    # Add default model if provided or available
    if model:
        if "config" not in mcp_config["mcpServers"][server_id]:
            mcp_config["mcpServers"][server_id]["config"] = {}
        
        mcp_config["mcpServers"][server_id]["config"]["model"] = model
    elif "default_model" in config:
        if "config" not in mcp_config["mcpServers"][server_id]:
            mcp_config["mcpServers"][server_id]["config"] = {}
        
        mcp_config["mcpServers"][server_id]["config"]["model"] = config["default_model"]
    
    # Add basic capabilities if not present
    if "capabilities" not in mcp_config:
        mcp_config["capabilities"] = {
            "text": True,
            "image": False,
            "audio": False,
            "video": False,
            "file": False
        }
    
    # Write updated configuration
    with open(mcp_file, "w") as f:
        json.dump(mcp_config, f, indent=2)
    
    typer.echo(f"✅ Added {provider} to MCP configuration as '{server_id}'")
    
    # Update .env file with API key if provided
    if key and "env_var" in config:
        env_file = agent_dir_path / ".env"
        env_var = config["env_var"]
        
        env_lines = []
        env_key_exists = False
        
        # Read existing .env file
        if env_file.exists():
            with open(env_file, "r") as f:
                env_lines = f.readlines()
            
            # Check if key already exists
            for i, line in enumerate(env_lines):
                if line.startswith(f"{env_var}="):
                    env_lines[i] = f"{env_var}={key}\n"
                    env_key_exists = True
                    break
        
        # Add key if it doesn't exist
        if not env_key_exists:
            env_lines.append(f"{env_var}={key}\n")
        
        # Write updated .env file
        with open(env_file, "w") as f:
            f.writelines(env_lines)
        
        typer.echo(f"✅ Updated {env_var} in .env file")
    
    # Update agent.py if necessary - not shown here as we've improved the templates instead
    
    return True

def fix_agent_implementation(agent_py_path, provider, model=None):
    """Fix agent.py to properly use the specified provider."""
    try:
        with open(agent_py_path, "r") as f:
            agent_content = f.read()
        
        # Check if agent already has a working implementation
        if "self.llm =" in agent_content and "No LLM provider configured" not in agent_content:
            typer.echo("Agent implementation already has LLM configuration, skipping update")
            return False
        
        # Create new agent implementation
        import_section = """import os
import asyncio
from typing import Dict, Any, Optional, List
"""
        
        if provider == "openai":
            import_section += "from openai import OpenAI\n"
        elif provider == "anthropic":
            import_section += "from anthropic import Anthropic\n"
       
        init_method = """    def __init__(self, **kwargs):
        """
        
        if provider == "openai":
            default_model = model or "gpt-4"
            init_method += f"""
        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is not set in environment variables")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)
        self.model = os.environ.get("OPENAI_MODEL", "{default_model}")
        
        # Initialize agent state
        self.conversation_history = []
"""
        elif provider == "anthropic":
            default_model = model or "claude-3-opus-20240229"
            init_method += f"""
        # Get API key from environment
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is not set in environment variables")
        
        # Initialize Anthropic client
        self.client = Anthropic(api_key=api_key)
        self.model = os.environ.get("ANTHROPIC_MODEL", "{default_model}")
        
        # Initialize agent state
        self.conversation_history = []
"""
       
        
        run_method = """    async def run(self, message: str, **kwargs) -> str:
        """
        
        if provider == "openai":
            run_method += """
        # Add input to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Prepare messages for the API
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Generate response using OpenAI
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                temperature=0.7,
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            return f"Error generating response: {str(e)}"
"""
        elif provider == "anthropic":
            run_method += """
        # Add input to conversation history
        self.conversation_history.append({"role": "user", "content": message})
        
        # Prepare messages for the API
        messages = []
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)
        
        # Generate response using Anthropic
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=1000,
                temperature=0.7,
                messages=messages
            )
            
            # Extract the response text
            response_text = response.content[0].text
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
        except Exception as e:
            return f"Error generating response: {str(e)}"
"""
              
        
        # Find the class definition
        import re
        class_match = re.search(r'class\s+(\w+):', agent_content)
        if not class_match:
            typer.echo("Could not find agent class definition, skipping update")
            return False
        
        class_name = class_match.group(1)
        
        # Create new implementation
        new_agent_content = import_section + f"\n\nclass {class_name}:\n"
        new_agent_content += init_method + "\n"
        new_agent_content += run_method
        
        # Write updated agent.py
        with open(agent_py_path, "w") as f:
            f.write(new_agent_content)
        
        return True
    except Exception as e:
        typer.echo(f"Error updating agent implementation: {e}")
        return False

@mcp_app.command("list")
def mcp_list(
    agent_dir: Annotated[str, typer.Option(help="Directory of the agent")] = ".",
):
    """
    List MCP providers in an agent.
    """
    # Check if agent exists
    agent_dir_path = Path(agent_dir)
    if not agent_dir_path.exists() or not agent_dir_path.is_dir():
        typer.echo(f"Error: Agent directory '{agent_dir}' not found")
        raise typer.Exit(1)
    
    # Check for mcp.json file
    mcp_file = agent_dir_path / ".mcp.json"
    
    if not mcp_file.exists():
        typer.echo("No MCP configuration found")
        raise typer.Exit(1)
    
    # Load MCP config
    try:
        with open(mcp_file, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        typer.echo(f"Error: Invalid MCP config file at {mcp_file}")
        raise typer.Exit(1)
    
    # Print providers
    if "mcpServers" in config and config["mcpServers"]:
        typer.echo("MCP Providers:")
        for server_id, server_config in config["mcpServers"].items():
            provider_type = server_config.get("type", "unknown")
            capability = server_config.get("capability", "unknown")
            url = server_config.get("url", "unknown")
            typer.echo(f"  • {typer.style(server_id, fg=typer.colors.CYAN)} ({provider_type}, {capability}): {url}")
    else:
        typer.echo("No MCP providers configured")


@mcp_app.command("test")
def mcp_test(
    name: Annotated[str, typer.Argument(help="Name of the MCP server to test")],
    scope: Annotated[str, typer.Option("-s", "--scope", help="Configuration scope (local, user, or project)")] = "project",
):
    """
    Test connectivity of an MCP server.
    """
    
    # Import functions directly without dependency on module structure
    from pathlib import Path
    import json
    import httpx
    
    # Simple implementation of load_mcp_servers
    def load_local_mcp_config(dir_path="."):
        mcp_file = Path(dir_path) / ".mcp.json"
        if mcp_file.exists():
            try:
                with open(mcp_file, "r") as f:
                    config = json.load(f)
                servers = []
                if "mcpServers" in config:
                    for server_id, server_config in config["mcpServers"].items():
                        server_config["id"] = server_id
                        servers.append(server_config)
                return servers
            except Exception as e:
                typer.echo(f"Error loading MCP config: {e}")
        return []
    
    # Simple implementation of test_mcp_connection
    def test_local_mcp_connection(server):
        try:
            url = server.get("url")
            if not url:
                return {"success": False, "error": "No URL configured"}
            
            # Create request headers
            headers = {"Content-Type": "application/json"}
            
            # Add authorization if we have environment variables defined
            if "env" in server:
                import os
                for env_var, _ in server["env"].items():
                    api_key = os.environ.get(env_var)
                    if api_key and env_var.endswith("_API_KEY"):
                        if "openai" in server.get("id", "").lower():
                            headers["Authorization"] = f"Bearer {api_key}"
                        else:
                            headers[env_var.replace("_", "-")] = api_key
            
            # Simple connection test
            response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
            
            if response.status_code < 400:
                return {
                    "success": True,
                    "capabilities": server.get("capability", "unknown")
                }
            else:
                return {
                    "success": False, 
                    "error": f"HTTP status {response.status_code}"
                }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Use our local implementations
    servers = load_local_mcp_config()
    server = next((srv for srv in servers if srv.get("id") == name), None)
    
    if not server:
        typer.echo(f"No MCP server named '{name}' found.")
        raise typer.Exit(1)
    
    typer.echo(f"Testing MCP server '{name}'...")
    result = test_local_mcp_connection(server)
    
    if result.get("success"):
        typer.echo(f"✅ MCP server '{name}' is reachable. Capabilities: {result.get('capabilities')}")
    else:
        typer.echo(f"❌ MCP server '{name}' test failed. Error: {result.get('error')}")

@mcp_app.command("remove")
def mcp_remove(
    provider: Annotated[str, typer.Argument(help="Provider to remove (e.g., openai-llm)")],
    agent_dir: Annotated[str, typer.Option(help="Directory of the agent")] = ".",
):
    """
    Remove an MCP provider from an agent.
    """
    # Check if agent exists
    agent_dir_path = Path(agent_dir)
    if not agent_dir_path.exists() or not agent_dir_path.is_dir():
        typer.echo(f"Error: Agent directory '{agent_dir}' not found")
        raise typer.Exit(1)
    
    # Check for mcp.json file
    mcp_file = agent_dir_path / ".mcp.json"
    
    if not mcp_file.exists():
        typer.echo("No MCP configuration found")
        raise typer.Exit(1)
    
    # Load MCP config
    try:
        with open(mcp_file, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        typer.echo(f"Error: Invalid MCP config file at {mcp_file}")
        raise typer.Exit(1)
    
    # Check if provider exists
    if "mcpServers" not in config or provider not in config["mcpServers"]:
        typer.echo(f"Error: Provider '{provider}' not found in MCP configuration")
        raise typer.Exit(1)
    
    # Get environment variable to remove
    env_var = None
    if "env" in config["mcpServers"][provider]:
        env_vars = config["mcpServers"][provider]["env"]
        if env_vars:
            env_var = list(env_vars.keys())[0]
    
    # Remove provider from config
    del config["mcpServers"][provider]
    
    # Remove provider from settings if exists
    provider_base = provider.split("-")[0]
    setting_id = f"{provider_base}-api-key"
    if "settings" in config and setting_id in config["settings"]:
        del config["settings"][setting_id]
    
    # Write updated config
    with open(mcp_file, "w") as f:
        json.dump(config, f, indent=2)
    
    typer.echo(f"✅ Removed provider '{provider}' from MCP configuration")
    
    # Optionally remove from .env
    if env_var and typer.confirm(f"Remove {env_var} from .env file?"):
        env_file = agent_dir_path / ".env"
        if env_file.exists():
            with open(env_file, "r") as f:
                env_lines = f.readlines()
            
            # Filter out the environment variable
            env_lines = [line for line in env_lines if not line.startswith(f"{env_var}=")]
            
            # Write updated .env file
            with open(env_file, "w") as f:
                f.writelines(env_lines)
            
            typer.echo(f"✅ Removed {env_var} from .env file")