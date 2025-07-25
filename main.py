"""
iTrade - High-Frequency Trading Platform
Main application entry point with FastAPI web interface
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Set
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
from contextlib import asynccontextmanager

# iTrade modules
from config import Config
from data_sources.financial_data import MarketDataAggregator, create_data_provider
from data_sources.news_scraper import NewsAnalyzer, get_market_sentiment
from analytics.greeks_calculator import GreeksCalculator, compare_greeks
from strategies.trading_strategies import (
    create_strategy, create_default_ensemble, SignalType,
    MomentumStrategy, MeanReversionStrategy, BreakoutStrategy, 
    SentimentStrategy, GreeksBasedStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
config = Config()
data_provider = None
news_analyzer = None
greeks_calculator = None
trading_ensemble = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except:
                pass

manager = ConnectionManager()

# Pydantic models for API
class SymbolRequest(BaseModel):
    symbol: str

class StrategyRequest(BaseModel):
    strategy_type: str
    symbol: str
    params: Optional[Dict] = None

class PortfolioPosition(BaseModel):
    symbol: str
    quantity: float
    market_value: float

class WatchlistRequest(BaseModel):
    symbols: List[str]

# Global state
active_symbols: Set[str] = set()
portfolio_positions: List[Dict] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    global data_provider, news_analyzer, greeks_calculator, trading_ensemble
    
    logger.info("Starting iTrade platform...")
    
    # Initialize components
    data_provider = MarketDataAggregator()
    news_analyzer = NewsAnalyzer()
    greeks_calculator = GreeksCalculator()
    trading_ensemble = create_default_ensemble()
    
    # Start background tasks
    asyncio.create_task(market_data_monitor())
    asyncio.create_task(news_monitor())
    
    logger.info("iTrade platform started successfully!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down iTrade platform...")

# Create FastAPI app
app = FastAPI(
    title="iTrade - High-Frequency Trading Platform",
    description="Advanced algorithmic trading platform with real-time data and analytics",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Background monitoring tasks
async def market_data_monitor():
    """Monitor market data for active symbols"""
    while True:
        try:
            if active_symbols:
                for symbol in active_symbols:
                    # Get real-time data
                    data = await data_provider.get_best_price(symbol)
                    
                    if data:
                        # Broadcast to WebSocket clients
                        await manager.broadcast({
                            "type": "price_update",
                            "symbol": symbol,
                            "data": data
                        })
                
                await asyncio.sleep(config.DATA_REFRESH_INTERVAL)
            else:
                await asyncio.sleep(5)  # Wait longer if no active symbols
                
        except Exception as e:
            logger.error(f"Error in market data monitor: {e}")
            await asyncio.sleep(5)

async def news_monitor():
    """Monitor news sentiment"""
    while True:
        try:
            # Get market sentiment
            sentiment_data = await get_market_sentiment()
            
            if sentiment_data:
                await manager.broadcast({
                    "type": "market_sentiment",
                    "data": sentiment_data
                })
            
            await asyncio.sleep(config.NEWS_REFRESH_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in news monitor: {e}")
            await asyncio.sleep(config.NEWS_REFRESH_INTERVAL)

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>iTrade - High-Frequency Trading Platform</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body class="bg-gray-100">
        <div id="app" class="min-h-screen">
            <nav class="bg-blue-600 text-white p-4">
                <div class="container mx-auto flex justify-between items-center">
                    <h1 class="text-2xl font-bold">iTrade Platform</h1>
                    <div class="space-x-4">
                        <span id="market-status" class="bg-green-500 px-3 py-1 rounded">Live</span>
                        <span id="connection-status" class="bg-gray-500 px-3 py-1 rounded">Connecting...</span>
                    </div>
                </div>
            </nav>
            
            <div class="container mx-auto p-4">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- Symbol Input and Controls -->
                    <div class="lg:col-span-1 space-y-6">
                        <div class="bg-white p-6 rounded-lg shadow">
                            <h2 class="text-xl font-semibold mb-4">Add Symbol</h2>
                            <div class="flex space-x-2">
                                <input type="text" id="symbol-input" placeholder="AAPL" 
                                       class="flex-1 border border-gray-300 rounded px-3 py-2">
                                <button onclick="addSymbol()" 
                                        class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                                    Add
                                </button>
                            </div>
                        </div>
                        
                        <!-- Active Symbols -->
                        <div class="bg-white p-6 rounded-lg shadow">
                            <h2 class="text-xl font-semibold mb-4">Watchlist</h2>
                            <div id="watchlist" class="space-y-2">
                                <!-- Symbols will be added here -->
                            </div>
                        </div>
                        
                        <!-- Market Sentiment -->
                        <div class="bg-white p-6 rounded-lg shadow">
                            <h2 class="text-xl font-semibold mb-4">Market Sentiment</h2>
                            <div id="sentiment-display">
                                <div class="text-gray-500">Loading...</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Main Content Area -->
                    <div class="lg:col-span-2 space-y-6">
                        <!-- Price Chart -->
                        <div class="bg-white p-6 rounded-lg shadow">
                            <h2 class="text-xl font-semibold mb-4">Price Chart</h2>
                            <div id="price-chart" style="height: 400px;"></div>
                        </div>
                        
                        <!-- Trading Signals -->
                        <div class="bg-white p-6 rounded-lg shadow">
                            <h2 class="text-xl font-semibold mb-4">Trading Signals</h2>
                            <div id="signals-display" class="space-y-2">
                                <!-- Signals will be displayed here -->
                            </div>
                        </div>
                        
                        <!-- Greeks Dashboard -->
                        <div class="bg-white p-6 rounded-lg shadow">
                            <h2 class="text-xl font-semibold mb-4">Risk Metrics (Greeks)</h2>
                            <div id="greeks-display">
                                <!-- Greeks will be displayed here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // WebSocket connection
            let ws = null;
            let priceData = {};
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = function() {
                    document.getElementById('connection-status').textContent = 'Connected';
                    document.getElementById('connection-status').className = 'bg-green-500 px-3 py-1 rounded';
                };
                
                ws.onmessage = function(event) {
                    const message = JSON.parse(event.data);
                    handleWebSocketMessage(message);
                };
                
                ws.onclose = function() {
                    document.getElementById('connection-status').textContent = 'Disconnected';
                    document.getElementById('connection-status').className = 'bg-red-500 px-3 py-1 rounded';
                    setTimeout(connectWebSocket, 5000); // Reconnect after 5 seconds
                };
            }
            
            function handleWebSocketMessage(message) {
                switch(message.type) {
                    case 'price_update':
                        updatePrice(message.symbol, message.data);
                        break;
                    case 'market_sentiment':
                        updateSentiment(message.data);
                        break;
                    case 'trading_signal':
                        displaySignal(message.data);
                        break;
                }
            }
            
            function updatePrice(symbol, data) {
                priceData[symbol] = data;
                updateWatchlistDisplay();
                updateChart(symbol, data);
            }
            
            function updateSentiment(data) {
                const sentimentDisplay = document.getElementById('sentiment-display');
                const sentiment = data.market_sentiment;
                
                if (sentiment) {
                    const sentimentScore = sentiment.overall_sentiment || 0;
                    const color = sentimentScore > 0 ? 'text-green-600' : sentimentScore < 0 ? 'text-red-600' : 'text-gray-600';
                    
                    sentimentDisplay.innerHTML = `
                        <div class="space-y-2">
                            <div class="flex justify-between">
                                <span>Overall Sentiment:</span>
                                <span class="${color} font-semibold">${sentimentScore.toFixed(3)}</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Confidence:</span>
                                <span>${(sentiment.confidence || 0).toFixed(2)}</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Articles:</span>
                                <span>${sentiment.article_count || 0}</span>
                            </div>
                        </div>
                    `;
                }
            }
            
            function updateWatchlistDisplay() {
                const watchlist = document.getElementById('watchlist');
                watchlist.innerHTML = '';
                
                for (const [symbol, data] of Object.entries(priceData)) {
                    const price = data.price || 0;
                    const change = data.change || 0;
                    const changeClass = change >= 0 ? 'text-green-600' : 'text-red-600';
                    
                    const symbolDiv = document.createElement('div');
                    symbolDiv.className = 'flex justify-between items-center p-2 border rounded';
                    symbolDiv.innerHTML = `
                        <div class="font-semibold">${symbol}</div>
                        <div class="text-right">
                            <div>$${price.toFixed(2)}</div>
                            <div class="${changeClass} text-sm">${change >= 0 ? '+' : ''}${change.toFixed(2)}</div>
                        </div>
                    `;
                    watchlist.appendChild(symbolDiv);
                }
            }
            
            function updateChart(symbol, data) {
                // Simple price chart update
                // In a real implementation, you'd maintain historical data
                const trace = {
                    x: [new Date()],
                    y: [data.price],
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: symbol
                };
                
                const layout = {
                    title: `${symbol} Price`,
                    xaxis: { title: 'Time' },
                    yaxis: { title: 'Price ($)' }
                };
                
                Plotly.newPlot('price-chart', [trace], layout);
            }
            
            async function addSymbol() {
                const input = document.getElementById('symbol-input');
                const symbol = input.value.toUpperCase().trim();
                
                if (symbol) {
                    try {
                        const response = await fetch('/api/symbols/add', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ symbol: symbol })
                        });
                        
                        if (response.ok) {
                            input.value = '';
                            // Symbol will be added to watchlist via WebSocket updates
                        }
                    } catch (error) {
                        console.error('Error adding symbol:', error);
                    }
                }
            }
            
            // Initialize
            connectWebSocket();
            
            // Allow Enter key to add symbol
            document.getElementById('symbol-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    addSymbol();
                }
            });
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming WebSocket messages if needed
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/symbols/add")
async def add_symbol(request: SymbolRequest):
    """Add a symbol to the watchlist"""
    symbol = request.symbol.upper()
    active_symbols.add(symbol)
    
    # Get initial data
    try:
        data = await data_provider.get_best_price(symbol)
        await manager.broadcast({
            "type": "price_update",
            "symbol": symbol,
            "data": data
        })
        
        return {"message": f"Symbol {symbol} added successfully", "data": data}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/symbols/{symbol}")
async def remove_symbol(symbol: str):
    """Remove a symbol from the watchlist"""
    symbol = symbol.upper()
    active_symbols.discard(symbol)
    return {"message": f"Symbol {symbol} removed"}

@app.get("/api/symbols")
async def get_active_symbols():
    """Get list of active symbols"""
    return {"symbols": list(active_symbols)}

@app.post("/api/analysis/greeks")
async def get_greeks_analysis(request: SymbolRequest):
    """Get Greeks analysis for a symbol"""
    try:
        greeks = greeks_calculator.calculate_greeks(request.symbol.upper())
        return greeks.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analysis/signal")
async def get_trading_signal(request: StrategyRequest):
    """Get trading signal for a symbol using specified strategy"""
    try:
        symbol = request.symbol.upper()
        
        # Get historical data
        yahoo_provider = create_data_provider("yahoo")
        data = await yahoo_provider.get_historical_data(symbol, "1y")
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # Create strategy and generate signal
        strategy = create_strategy(request.strategy_type, request.params)
        signal = await strategy.generate_signal(symbol, data)
        
        # Broadcast signal
        await manager.broadcast({
            "type": "trading_signal",
            "data": {
                "symbol": signal.symbol,
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence,
                "reason": signal.reason,
                "strategy": signal.strategy_name,
                "timestamp": signal.timestamp.isoformat()
            }
        })
        
        return {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "confidence": signal.confidence,
            "price": signal.price,
            "reason": signal.reason,
            "strategy_name": signal.strategy_name,
            "timestamp": signal.timestamp.isoformat(),
            "metadata": signal.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analysis/ensemble")
async def get_ensemble_signal(request: SymbolRequest):
    """Get ensemble trading signal for a symbol"""
    try:
        symbol = request.symbol.upper()
        
        # Get historical data
        yahoo_provider = create_data_provider("yahoo")
        data = await yahoo_provider.get_historical_data(symbol, "1y")
        
        if data.empty:
            raise HTTPException(status_code=404, detail="No data found for symbol")
        
        # Generate ensemble signal
        signal = await trading_ensemble.generate_ensemble_signal(symbol, data)
        
        return {
            "symbol": signal.symbol,
            "signal_type": signal.signal_type.value,
            "confidence": signal.confidence,
            "price": signal.price,
            "reason": signal.reason,
            "strategy_name": signal.strategy_name,
            "timestamp": signal.timestamp.isoformat(),
            "metadata": signal.metadata
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/market/sentiment")
async def get_current_sentiment():
    """Get current market sentiment"""
    try:
        sentiment_data = await news_analyzer.analyze_market_news()
        return sentiment_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/market/data")
async def get_market_data(request: SymbolRequest):
    """Get comprehensive market data for a symbol"""
    try:
        symbol = request.symbol.upper()
        data = await data_provider.get_comprehensive_data(symbol)
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/strategies")
async def get_available_strategies():
    """Get list of available trading strategies"""
    return {
        "strategies": [
            {
                "type": "momentum",
                "name": "Momentum Strategy",
                "description": "Uses RSI, MACD, and volume indicators"
            },
            {
                "type": "mean_reversion",
                "name": "Mean Reversion Strategy", 
                "description": "Uses Bollinger Bands and moving averages"
            },
            {
                "type": "breakout",
                "name": "Breakout Strategy",
                "description": "Detects price breakouts with volume confirmation"
            },
            {
                "type": "sentiment",
                "name": "Sentiment Strategy",
                "description": "Combines news sentiment with price momentum"
            },
            {
                "type": "greeks",
                "name": "Greeks-Based Strategy",
                "description": "Uses risk metrics and Greeks for signals"
            }
        ]
    }

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio positions"""
    return {"positions": portfolio_positions}

@app.post("/api/portfolio/add")
async def add_position(position: PortfolioPosition):
    """Add a position to the portfolio"""
    portfolio_positions.append(position.dict())
    return {"message": "Position added successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_symbols": len(active_symbols),
        "portfolio_positions": len(portfolio_positions)
    }

@app.get("/api/system/status")
async def system_status():
    """Get system status and metrics"""
    return {
        "status": "running",
        "active_symbols": list(active_symbols),
        "data_providers": ["yahoo", "alpha_vantage", "polygon"],
        "strategies": ["momentum", "mean_reversion", "breakout", "sentiment", "greeks"],
        "websocket_connections": len(manager.active_connections),
        "market_hours": {
            "open": f"{config.MARKET_OPEN_HOUR}:{config.MARKET_OPEN_MINUTE:02d}",
            "close": f"{config.MARKET_CLOSE_HOUR}:{config.MARKET_CLOSE_MINUTE:02d}"
        },
        "refresh_intervals": {
            "data": f"{config.DATA_REFRESH_INTERVAL}s",
            "news": f"{config.NEWS_REFRESH_INTERVAL}s"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info"
    )