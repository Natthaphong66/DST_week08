"""Configuration management for pycaret-mcp-server."""
import os
from typing import List
from mcp.server.fastmcp import FastMCP


def get_env_bool(env_var: str, default: bool = False) -> bool:
    """Get boolean value from environment variable."""
    value = os.getenv(env_var, '').lower()
    if value in ('true', '1', 'yes', 'on'):
        return True
    elif value in ('false', '0', 'no', 'off'):
        return False
    return default


def get_env_int(env_var: str, default: int) -> int:
    """Get integer value from environment variable."""
    try:
        return int(os.getenv(env_var, str(default)))
    except (ValueError, TypeError):
        return default


# Configuration
MODELS_DIR = os.getenv('PYCARET_MCP_MODELS_DIR', 
                       os.path.join(os.path.dirname(__file__), '..', 'models'))
os.makedirs(MODELS_DIR, exist_ok=True)

# Constants
MAX_FILE_SIZE = get_env_int('PYCARET_MCP_MAX_FILE_SIZE', 100 * 1024 * 1024)  # Default: 100MB

# Logging configuration
LOG_LEVEL = os.getenv('PYCARET_MCP_LOG_LEVEL', 'INFO').upper()
LOG_FILE = os.getenv('PYCARET_MCP_LOG_FILE', 
                     os.path.join(os.path.dirname(__file__), '..', 'logs', 'mcp_server.log'))
LOG_MAX_BYTES = get_env_int('PYCARET_MCP_LOG_MAX_BYTES', 5 * 1024 * 1024)  # Default: 5MB
LOG_BACKUP_COUNT = get_env_int('PYCARET_MCP_LOG_BACKUP_COUNT', 3)

# Performance settings
ENABLE_MEMORY_MONITORING = get_env_bool('PYCARET_MCP_ENABLE_MEMORY_MONITORING', True)
MEMORY_WARNING_THRESHOLD = get_env_int('PYCARET_MCP_MEMORY_WARNING_THRESHOLD', 500)  # MB

# MCP Server Initialization
mcp = FastMCP("PyCaret-MCP-Server")


def print_config():
    """Print current configuration (useful for debugging)."""
    print("=" * 60)
    print("PyCaret MCP Server Configuration")
    print("=" * 60)
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"MAX_FILE_SIZE: {MAX_FILE_SIZE / (1024*1024):.2f} MB")
    print(f"LOG_LEVEL: {LOG_LEVEL}")
    print(f"LOG_FILE: {LOG_FILE}")
    print(f"LOG_MAX_BYTES: {LOG_MAX_BYTES / (1024*1024):.2f} MB")
    print(f"LOG_BACKUP_COUNT: {LOG_BACKUP_COUNT}")
    print(f"ENABLE_MEMORY_MONITORING: {ENABLE_MEMORY_MONITORING}")
    print(f"MEMORY_WARNING_THRESHOLD: {MEMORY_WARNING_THRESHOLD} MB")
    print("=" * 60)
