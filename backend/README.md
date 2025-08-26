# Agentic App Backend

Backend service for the multi-agent data analysis system using OpenAI Agents SDK with MCP support.

## Features

- 4 specialized AI agents
- Native MCP protocol support
- FastAPI REST API
- Gradio web interface
- Phoenix tracing integration

## Agents

- **DataLoaderAgent**: File operations and data preparation
- **AnalyticsAgent**: Statistical calculations and analysis  
- **VisualizationAgent**: Chart creation and visual insights
- **CommunicationAgent**: User-friendly response formatting

## Interfaces

- FastAPI: `python -m src.api.main`
- Gradio: `python -m src.gradio_app`
