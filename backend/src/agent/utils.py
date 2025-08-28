import os
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams, create_static_tool_filter
from agents.agent import MCPConfig
from typing import List


MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000/mcp")


def get_mcp_server(allowed_tool_names: list[str]) -> MCPServerStreamableHttp:
    """Create and return an MCP server with specified tools."""
    
    static_tool_filter = create_static_tool_filter(allowed_tool_names=allowed_tool_names)
    params = MCPServerStreamableHttpParams(url=MCP_SERVER_URL, timeout=30)
    
    return MCPServerStreamableHttp(
        params=params,
        cache_tools_list=True,
        tool_filter=static_tool_filter
    )