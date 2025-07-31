#!/bin/bash

# iTrade.com Startup Script

echo "Starting iTrade.com - Professional Stock Trading Platform"
echo "=================================================="

# Activate virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your API keys before running again."
    exit 1
fi

# Run the application
echo "Starting Flask application..."
echo "Visit http://localhost:5000 to access iTrade.com"
echo "Press Ctrl+C to stop the server"
python app.py