"""
Simple MCP Client for Agents
This provides a direct HTTP interface to the MCP server for agents to use.
"""

import requests
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# MCP Server configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000")
DEBUG_MODE = os.getenv("MCP_DEBUG", "false").lower() == "true"

class MCPClient:
    """Simple HTTP client for MCP server communication"""
    
    def __init__(self, server_url: str = MCP_SERVER_URL):
        self.server_url = server_url.rstrip('/')
        self.timeout = 30
        
    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        url = f"{self.server_url}/mcp/tools/{tool_name}"
        
        if DEBUG_MODE:
            print(f"🔗 MCP Call: {tool_name} -> {url}")
            print(f"   📤 Args: {kwargs}")
        
        try:
            response = requests.post(
                url,
                json=kwargs,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            result = response.json()
            
            if DEBUG_MODE:
                print(f"   📥 Result: {type(result).__name__}")
                
            return result
            
        except requests.exceptions.RequestException as e:
            error_msg = f"MCP HTTP call failed: {str(e)}"
            if DEBUG_MODE:
                print(f"   ❌ Error: {error_msg}")
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"MCP call failed: {str(e)}"
            if DEBUG_MODE:
                print(f"   ❌ Error: {error_msg}")
            return {"status": "error", "message": error_msg}

# Global MCP client instance
mcp_client = MCPClient()
