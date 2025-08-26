import os
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams, create_static_tool_filter
from agents.agent import MCPConfig
from typing import List


MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000")
TLS_KEY_FILE = os.getenv("TLS_KEY_FILE", "/app/certs/agentic-app.key")
TLS_CERT_FILE = os.getenv("TLS_CERT_FILE", "/app/certs/agentic-app.crt")
TLS_CA_FILE = os.getenv("TLS_CA_FILE", "/app/certs/ca.crt")


def get_mcp_server(allowed_tool_names: List[str], url: str = MCP_SERVER_URL, timeout: int = 30) -> MCPServerStreamableHttp:
    """Get MCP server connection with specified tools"""
    
    static_tool_filter = create_static_tool_filter(allowed_tool_names=allowed_tool_names)
    
    params = MCPServerStreamableHttpParams(url=url, timeout=timeout)
    
    mcp_server = MCPServerStreamableHttp(
        params=params,
        cache_tools_list=True,
        tool_filter=static_tool_filter
    )

    return mcp_server


def get_mcp_config() -> MCPConfig:
    """Get MCP configuration for TLS"""
    return MCPConfig(
        tls_key_file=TLS_KEY_FILE,
        tls_cert_file=TLS_CERT_FILE,
        tls_ca_file=TLS_CA_FILE,
        tls_enabled=False  # Disable TLS for internal Docker communication
    )
