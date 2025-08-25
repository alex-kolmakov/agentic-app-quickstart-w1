# Dockerized Multi-Agent Data Analysis Application

This application provides a multi-agent system for CSV data analysis with advanced visualization capabilities, now fully containerized with Docker and integrated with Phoenix Arize for tracing.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- `.env` file with your API keys (see Environment Variables section)

### Running the Application

1. **Start all services:**
   ```bash
   docker-compose up -d
   ```

2. **Access the applications:**
   - **Main Application**: http://localhost:7860
   - **Phoenix Tracing UI**: http://localhost:6006

3. **Stop the services:**
   ```bash
   docker-compose down
   ```

## 🔧 Environment Variables

Create a `.env` file in the root directory with your configuration:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Phoenix Configuration (optional - defaults provided)
PHOENIX_ENDPOINT=http://phoenix:6006
PHOENIX_PROJECT_NAME=agentic-app-quickstart

# Gradio Configuration (optional - defaults provided)
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

## 📊 Features

### Multi-Agent System
- **📁 DataLoaderAgent**: File operations and data preparation
- **📊 AnalyticsAgent**: Statistical calculations and analysis  
- **📈 VisualizationAgent**: Chart creation and visual insights
- **💬 CommunicationAgent**: User-friendly response formatting

### Visualization Capabilities
- 📈 Bar charts for categorical comparisons
- 🔍 Scatter plots for relationship exploration
- 📦 Box plots for outlier detection

### Tracing & Monitoring
- 🔍 Real-time tracing with Phoenix Arize
- 📊 Performance monitoring and debugging
- 🕒 Conversation history tracking

## 🛠️ Development

### Building the Application
```bash
# Build the application image
docker-compose build app

# Build with no cache
docker-compose build --no-cache app
```

### Viewing Logs
```bash
# View all logs
docker-compose logs

# View specific service logs
docker-compose logs app
docker-compose logs phoenix

# Follow logs in real-time
docker-compose logs -f app
```

### Development Mode
For development, you can mount your local code:

```yaml
# Add to docker-compose.yml under app service volumes:
volumes:
  - ./multi_agent_data_app:/app/multi_agent_data_app
  - ./charts:/app/charts
  - ./.env:/app/.env:ro
```

## 📁 Data Files

Place your CSV files in the `multi_agent_data_app/data/` directory. The application comes with sample files:
- `employee_data.csv`
- `sample_sales.csv` 
- `weather_data.csv`

## 🐛 Troubleshooting

### Phoenix Not Starting
```bash
# Check Phoenix logs
docker-compose logs phoenix

# Restart Phoenix service
docker-compose restart phoenix
```

### Application Connection Issues
```bash
# Check if Phoenix is healthy
curl http://localhost:6006/health

# Restart the application
docker-compose restart app
```

### Port Conflicts
If ports 7860 or 6006 are already in use, modify the port mappings in `docker-compose.yml`:

```yaml
services:
  app:
    ports:
      - "8080:7860"  # Change host port to 8080
  phoenix:
    ports:
      - "6007:6006"  # Change host port to 6007
```

## 🔄 Updates

To update the application:
1. Pull the latest code
2. Rebuild and restart:
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

## 📈 Monitoring

Access the Phoenix UI at http://localhost:6006 to:
- View real-time traces of agent interactions
- Monitor performance metrics
- Debug conversation flows
- Analyze system behavior
