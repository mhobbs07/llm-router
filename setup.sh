#!/bin/bash

# LLM Router Setup Script
# This script helps you get started with the LLM Router and Open WebUI

set -e

echo "🚀 LLM Router Setup"
echo "==================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file. Please edit it with your API keys."
    echo ""
    echo "Required API keys to set in .env:"
    echo "  - OPENAI_API_KEY"
    echo "  - ANTHROPIC_API_KEY"
    echo "  - GOOGLE_API_KEY"
    echo "  - XAI_API_KEY"
    echo ""
    echo "Optional settings:"
    echo "  - WEBUI_SECRET_KEY (for Open WebUI security)"
    echo ""
    read -p "Press Enter after you've edited the .env file..."
else
    echo "✅ .env file already exists"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

echo "🐳 Starting services..."
echo "This will start:"
echo "  - LLM Router (port 8000)"
echo "  - Open WebUI (port 3000)"
echo ""

# Start services
docker-compose up -d

echo ""
echo "🎉 Services started successfully!"
echo ""
echo "Access your services:"
echo "  🌐 Open WebUI: http://localhost:3000"
echo "  🔌 LLM Router API: http://localhost:8000"
echo "  💚 Health Check: http://localhost:8000/health"
echo ""
echo "Useful commands:"
echo "  docker-compose logs -f          # View all logs"
echo "  docker-compose logs -f llm-router    # View LLM Router logs"
echo "  docker-compose logs -f open-webui    # View Open WebUI logs"
echo "  docker-compose down             # Stop all services"
echo "  make health                    # Check if services are healthy"
echo ""
echo "Happy chatting! 🤖"