FROM python:3.13-slim

WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Copy pyproject.toml and README
COPY mcp_server/pyproject.toml /app/
COPY mcp_server/README.md /app/

# Install dependencies
RUN uv pip install --system -e .

# Copy source code
COPY mcp_server/src /app/src
COPY mcp_server/data /app/data

# Create charts directory for visualization outputs
RUN mkdir -p /app/charts

# Expose port
EXPOSE 8000

# Run the MCP server
CMD ["python", "-m", "src.api.server"]
