#!/bin/bash

echo "🚀 Starting iTrade.com Trading Platform..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️ .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys"
fi

# Start the application
echo "🌐 Starting Flask application..."
echo "📱 Open your browser and go to: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python app.py