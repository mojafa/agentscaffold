"""
Model Context Protocol (MCP) providers for Agent Scaffold.

This module provides functionality for interacting with MCP servers,
which implement a protocol for LLM inference and other AI capabilities.
"""

from .client import (
    MCPClient,
    load_mcp_servers,
    save_mcp_servers,
    test_mcp_connection,
    get_mcp_client,
)

__all__ = [
    "MCPClient",
    "load_mcp_servers",
    "save_mcp_servers",
    "test_mcp_connection",
    "get_mcp_client",
]