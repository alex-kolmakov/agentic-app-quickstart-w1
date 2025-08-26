#!/bin/bash

echo "🚀 Testing MCP-based Agentic App Setup"
echo "======================================"

echo ""
echo "📋 Checking Docker Compose configuration..."
docker-compose config

echo ""
echo "🛠️ Building all services..."
docker-compose build

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 30

echo ""
echo "🔍 Checking service status..."
docker-compose ps

echo ""
echo "📊 Testing MCP server directly..."
curl -X POST http://localhost:8000/call/get_file_context \
     -H "Content-Type: application/json" \
     -d '{}' || echo "MCP server not ready yet"

echo ""
echo "🌐 Services should be available at:"
echo "   📊 MCP Server: http://localhost:8000"
echo "   🤖 FastAPI Backend: http://localhost:7001" 
echo "   🎨 Gradio UI: http://localhost:7860"
echo "   🔍 Phoenix Tracing: http://localhost:6006"
echo "   🔒 Nginx Proxy: https://localhost:8443"

echo ""
echo "🎉 Setup complete! Check the services above."
echo "💡 To view logs: docker-compose logs -f [service-name]"
echo "🛑 To stop: docker-compose down"
