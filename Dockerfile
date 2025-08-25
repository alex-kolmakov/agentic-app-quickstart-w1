# Use Python 3.13 as base image
FROM python:3.13-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files and project structure needed for UV
COPY pyproject.toml README.md ./
COPY multi_agent_data_app/ ./multi_agent_data_app/

# Install the project and its dependencies using UV
RUN uv pip install --system --no-cache .

# Create directory for generated charts
RUN mkdir -p /app/charts

# Set environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Expose the port
EXPOSE 7860

# Run the application
CMD ["python", "multi_agent_data_app/main.py"]
