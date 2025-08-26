# 🤖 Agentic Data Analysis App - MCP Architecture

A multi-agent data analysis system built with **Model Context Protocol (MCP)** for secure separation of concerns between AI agents and data operations.

## 🏗️ Architecture Overview

This application demonstrates a modern approach to agentic AI systems using:

- **🔒 MCP Server**: Secure data operations and tool execution
- **🤖 Agent Backend**: AI reasoning and orchestration  
- **🌐 Nginx Proxy**: TLS termination and routing
- **📊 Phoenix Tracing**: Observability and monitoring
- **🎨 Gradio UI**: User-friendly interface

### 🎯 Key Benefits

- **Security**: Data operations isolated in MCP server
- **Scalability**: Microservices architecture
- **Observability**: Full tracing with Phoenix
- **Flexibility**: Multiple interfaces (FastAPI + Gradio)

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Environment variables in `.env` file

### Run the Application

```bash
# Test and start all services
./test_setup.sh

# Or manually:
docker-compose up -d
```

### Access Points

- **🎨 Gradio UI**: http://localhost:7860
- **🤖 FastAPI Backend**: http://localhost:7001
- **📊 MCP Server**: http://localhost:8000  
- **🔍 Phoenix Tracing**: http://localhost:6006
- **🔒 Nginx Proxy**: https://localhost:8443

## 📋 Services

### 🛠️ MCP Server (`mcp-server`)

**Purpose**: Secure data operations and code execution

**Technologies**: 
- FastMCP for tool definitions
- Polars for data processing
- Isolated execution environment

**Tools Available**:
- `get_file_context()`: List CSV files and headers
- `code_executor(code)`: Execute Python data analysis code

**Port**: 8000

### 🤖 Agent Backend (`agentic-app`)

**Purpose**: AI reasoning and orchestration

**Technologies**:
- FastAPI for REST API
- Custom DataAnalysisAgent
- MCP client for tool calls

**Endpoints**:
- `POST /analyze`: Analyze data with natural language
- `GET /files`: Get available data files
- `GET /health`: Health check

**Port**: 7000

### 🎨 Gradio UI (`gradio-ui`)

**Purpose**: User-friendly chat interface

**Features**:
- Natural language queries
- Real-time responses
- Example questions
- Chat history

**Port**: 7860

### 🔒 Nginx Proxy (`nginx-proxy`)

**Purpose**: TLS termination and routing

**Features**:
- Mutual TLS authentication
- Route /mcp/ → MCP Server
- Route / → Agent Backend

**Port**: 443 (8443 externally)

### 📊 Phoenix Tracing (`phoenix`)

**Purpose**: Observability and monitoring

**Features**:
- OpenTelemetry tracing
- Performance monitoring
- Request/response logging

**Port**: 6006

## 🔧 Development

### Project Structure

```
.
├── mcp_server/           # MCP server for data operations
│   ├── src/api/         # FastMCP server
│   ├── data/            # CSV data files
│   └── Dockerfile
├── backend/             # Agent backend
│   ├── src/agent/       # Data analysis agent
│   ├── src/api/         # FastAPI application
│   └── Dockerfile
├── nginx/               # Reverse proxy
│   └── nginx-proxy.conf
├── cert-generator/      # TLS certificates
└── docker-compose.yml   # Full stack
```

### Adding New Tools

1. **Add to MCP Server** (`mcp_server/src/api/server.py`):
```python
@mcp.tool
async def your_new_tool(param: str) -> dict:
    """Your tool description"""
    # Implementation here
    return {"result": "success"}
```

2. **Update MCP Client** (`backend/src/agent/mcp_client.py`):
```python
async def call_your_tool(self, param: str) -> str:
    return await self.call_tool("your_new_tool", param=param)
```

3. **Use in Agent** (`backend/src/agent/data_analysis.py`):
```python
result = await mcp_client.call_your_tool("test")
```

### Environment Variables

Create `.env` file:
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_key_here

# Phoenix Configuration  
PHOENIX_ENDPOINT=http://localhost:6006
PHOENIX_PROJECT_NAME=agentic-app-quickstart

# MCP Configuration
MCP_SERVER_URL=http://mcp-server:8000
```

## 🧪 Testing

### Manual Testing

```bash
# Test MCP server directly
curl -X POST http://localhost:8000/call/get_file_context \
     -H "Content-Type: application/json" -d '{}'

# Test agent backend
curl -X POST http://localhost:7001/analyze \
     -H "Content-Type: application/json" \
     -d '{"query": "Show available data files"}'

# Test file listing
curl http://localhost:7001/files
```

### Integration Testing

```bash
# Run the full test suite
./test_setup.sh

# Check logs
docker-compose logs -f mcp-server
docker-compose logs -f agentic-app
```

## 🔍 Monitoring

### Phoenix Tracing

1. Visit http://localhost:6006
2. View traces and performance metrics
3. Monitor agent interactions

### Service Health

```bash
# Check all services
docker-compose ps

# Individual service logs
docker-compose logs -f [service-name]
```

## 🛡️ Security Features

- **Isolated Execution**: Data operations run in separate container
- **TLS Encryption**: All external communication encrypted
- **Mutual TLS**: Client certificate authentication
- **No Direct Database Access**: Agents only access data via MCP tools

## 🎯 Next Steps

1. **Add Security**: Implement AST analysis and restricted execution
2. **Add Visualization**: Matplotlib/Plotly integration in MCP server
3. **Add Authentication**: User management and session handling
4. **Add Persistence**: Database for conversation history
5. **Add Real MCP**: Upgrade to official MCP protocol

## 📚 Resources

- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [OpenAI Agents SDK](https://github.com/openai/agents-sdk)
- [Phoenix Tracing](https://docs.arize.com/phoenix)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.
