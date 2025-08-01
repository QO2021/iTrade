#!/bin/bash

echo "🚀 Setting up iTrade.com Trading Platform..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 is available"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment. Installing python3-venv..."
        sudo apt-get update && sudo apt-get install -y python3-venv python3-full
        python3 -m venv venv
    fi
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating .env file..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys before running the application"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🔑 Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - FRED_API_KEY (free from https://fred.stlouisfed.org/docs/api/api_key.html)"
echo "   - OPENAI_API_KEY (from https://platform.openai.com/)"
echo "   - Email settings for password reset (optional)"
echo ""
echo "2. Start the application:"
echo "   ./start.sh"
echo ""
echo "3. Open your browser and go to: http://localhost:5000"
echo ""
echo "Happy Trading! 📈"