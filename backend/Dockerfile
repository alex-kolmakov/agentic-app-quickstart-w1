FROM python:3.13-slim

WORKDIR /app

# Install uv for faster package management
RUN pip install uv

# Copy pyproject.toml and README
COPY backend/pyproject.toml /app/
COPY backend/README.md /app/

# Install dependencies
RUN uv pip install --system -e .

# Copy source code
COPY backend/src /app/src

# Create certs directory for TLS
RUN mkdir -p /app/certs

# Expose ports
EXPOSE 7000
EXPOSE 7860

# Default to FastAPI, but can be overridden with INTERFACE=gradio
ENV INTERFACE=fastapi

# Run the appropriate interface
CMD if [ "$INTERFACE" = "gradio" ]; then \
        python -m src.gradio_app; \
    else \
        python -m src.api.main; \
    fi
