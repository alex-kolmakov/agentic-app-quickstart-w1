FROM python:3.13-slim

WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Copy pyproject.toml for dependencies
COPY multi_agent_data_app_unified/pyproject.toml /app/
COPY multi_agent_data_app_unified/README.md /app/

# Install dependencies
RUN uv pip install --system -e .

# Copy the entire multi_agent_data_app
COPY multi_agent_data_app /app/multi_agent_data_app
COPY mcp_server /app/mcp_server

# Copy environment file
COPY .env /app/.env

# Create charts directory for generated visualizations
RUN mkdir -p /app/charts

# Expose port for Gradio
EXPOSE 7860

# Set environment variables
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860
ENV PYTHONPATH=/app

# Run the unified application
CMD ["python", "-m", "multi_agent_data_app.main"]
