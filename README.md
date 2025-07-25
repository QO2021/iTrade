# iTrade - High-Frequency Trading Platform

A comprehensive algorithmic trading platform with real-time data ingestion, sentiment analysis, and advanced risk metrics calculation including Alpha, Beta, Sigma, and Gamma.

## Features

### 🚀 Core Trading Features
- **Real-time Market Data**: Integration with Yahoo Finance, Alpha Vantage, and Polygon.io
- **Multiple Trading Strategies**: Momentum, Mean Reversion, Breakout, Sentiment-based, and Greeks-based strategies
- **Strategy Ensemble**: Combine multiple strategies with weighted voting
- **Risk Management**: Comprehensive risk metrics and position sizing
- **Portfolio Management**: Track positions, P&L, and performance metrics

### 📊 Financial Analytics
- **Greeks Calculation**: Alpha, Beta, Gamma, Delta, Theta, Vega, Rho
- **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Maximum Drawdown, VaR, CVaR
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, and more
- **Options Analytics**: Black-Scholes model implementation

### 📰 News & Sentiment Analysis
- **Multi-source News Aggregation**: MSN, WSJ, Yahoo Finance, Bloomberg, CNBC, Reuters, FT, etc.
- **Real-time Sentiment Analysis**: TextBlob-based sentiment scoring
- **Market Impact Assessment**: Ticker extraction and relevance scoring
- **Economic Data Integration**: FRED API for economic indicators

### 🌐 Web Interface
- **Real-time Dashboard**: Live price updates via WebSocket
- **Interactive Charts**: Plotly.js-based price charting
- **Strategy Testing**: Test different strategies on historical data
- **Risk Dashboard**: Visual display of Greeks and risk metrics

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd iTrade
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Run the application**
```bash
python main.py
```

5. **Access the web interface**
Open your browser and navigate to `http://localhost:8000`

## Configuration

### API Keys Required

#### Financial Data APIs
- **Alpha Vantage**: Get free API key at https://www.alphavantage.co/support/#api-key
- **Polygon.io**: Get API key at https://polygon.io/ (for high-frequency data)
- **Yahoo Finance**: No API key required (using yfinance library)

#### News APIs (Optional)
- **News API**: Get API key at https://newsapi.org/
- **Twitter API**: For additional sentiment data

### Environment Variables

```bash
# API Keys
ALPHA_VANTAGE_API_KEY=your_api_key_here
POLYGON_API_KEY=your_polygon_key_here

# Database (Optional - uses SQLite by default)
DATABASE_URL=postgresql://username:password@localhost:5432/itrade_db

# Trading Parameters
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.1  # 10% max position size
STOP_LOSS_THRESHOLD=0.02  # 2% stop loss
TAKE_PROFIT_THRESHOLD=0.05  # 5% take profit

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

## Data Sources

### Financial News & Market Updates
- **MSN Money**: https://www.msn.com/en-us/money
- **Wall Street Journal**: https://wsj.com
- **Yahoo Finance**: https://finance.yahoo.com/
- **Bloomberg**: https://bloomberg.com
- **CNBC**: https://cnbc.com
- **Reuters Finance**: https://reuters.com/finance
- **Financial Times**: https://ft.com
- **Morningstar**: https://morningstar.com
- **Seeking Alpha**: https://seekingalpha.com
- **TradingView**: https://tradingview.com

### Personal Finance Resources
- **NerdWallet**: https://nerdwallet.com - Credit cards, loans, mortgages advice
- **Investopedia**: https://investopedia.com - Educational articles and financial dictionary
- **Mint**: https://mint.intuit.com - Budgeting and expense tracking

### Economic Data & Research
- **FRED**: https://fred.stlouisfed.org - Federal Reserve Economic Data
- **World Bank**: https://data.worldbank.org - Global economic indicators
- **SEC EDGAR**: https://sec.gov/edgar - Public company filings

## API Reference

### WebSocket Events

#### Price Updates
```json
{
  "type": "price_update",
  "symbol": "AAPL",
  "data": {
    "price": 150.25,
    "bid": 150.20,
    "ask": 150.30,
    "volume": 1000000,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Trading Signals
```json
{
  "type": "trading_signal",
  "data": {
    "symbol": "AAPL",
    "signal_type": "buy",
    "confidence": 0.75,
    "reason": "Bullish momentum with high volume",
    "strategy": "Momentum Strategy",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### REST API Endpoints

#### Symbols Management
- `POST /api/symbols/add` - Add symbol to watchlist
- `DELETE /api/symbols/{symbol}` - Remove symbol from watchlist
- `GET /api/symbols` - Get active symbols

#### Analysis
- `POST /api/analysis/greeks` - Get Greeks analysis for symbol
- `POST /api/analysis/signal` - Get trading signal using specific strategy
- `POST /api/analysis/ensemble` - Get ensemble signal from all strategies

#### Market Data
- `GET /api/market/sentiment` - Get current market sentiment
- `POST /api/market/data` - Get comprehensive market data

#### System
- `GET /health` - Health check
- `GET /api/system/status` - System status and metrics

## Trading Strategies

### 1. Momentum Strategy
- **Indicators**: RSI, MACD, Volume
- **Logic**: Identifies trending stocks with strong momentum
- **Parameters**: RSI periods, MACD settings, volume thresholds

### 2. Mean Reversion Strategy
- **Indicators**: Bollinger Bands, Moving Averages
- **Logic**: Identifies oversold/overbought conditions for reversal trades
- **Parameters**: Bollinger period, standard deviations, mean period

### 3. Breakout Strategy
- **Indicators**: Support/Resistance, ATR, Volume
- **Logic**: Detects price breakouts with volume confirmation
- **Parameters**: Lookback period, volume threshold, ATR multiplier

### 4. Sentiment Strategy
- **Data Sources**: News sentiment, social media
- **Logic**: Combines sentiment analysis with price momentum
- **Parameters**: Sentiment thresholds, confidence levels

### 5. Greeks-Based Strategy
- **Metrics**: Alpha, Beta, Sharpe Ratio, Max Drawdown
- **Logic**: Selects assets based on risk-adjusted returns
- **Parameters**: Beta range, minimum Sharpe ratio, max drawdown

## Greeks & Risk Metrics

### The Greeks
- **Alpha**: Excess return over benchmark (Jensen's Alpha)
- **Beta**: Sensitivity to market movements (systematic risk)
- **Gamma**: Rate of change of Delta (second-order derivative)
- **Delta**: Price sensitivity to underlying asset
- **Theta**: Time decay (for options)
- **Vega**: Volatility sensitivity
- **Rho**: Interest rate sensitivity

### Risk Metrics
- **Sigma**: Volatility (standard deviation of returns)
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Conditional VaR**: Expected loss beyond VaR threshold

## Architecture

```
iTrade/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration management
├── models.py              # SQLAlchemy database models
├── database.py            # Database connection management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── data_sources/         # Data ingestion modules
│   ├── financial_data.py # Market data providers
│   └── news_scraper.py   # News and sentiment analysis
├── analytics/            # Financial analytics
│   └── greeks_calculator.py # Greeks and risk metrics
├── strategies/           # Trading strategies
│   └── trading_strategies.py # Strategy implementations
└── static/              # Static web assets
```

## Development

### Adding New Data Sources
1. Create a new provider class inheriting from `FinancialDataProvider`
2. Implement required methods: `get_real_time_price`, `get_historical_data`
3. Add to the `MarketDataAggregator` providers list

### Adding New Strategies
1. Create a new strategy class inheriting from `BaseStrategy`
2. Implement `generate_signal` and `get_required_periods` methods
3. Add to the strategy factory in `create_strategy` function

### Adding New Risk Metrics
1. Add calculation method to `GreeksCalculator` class
2. Update `GreeksResult` dataclass with new metric
3. Ensure proper error handling and logging

## Performance Considerations

### High-Frequency Trading
- **Latency**: WebSocket connections for real-time updates
- **Throughput**: Async/await patterns for concurrent processing
- **Caching**: Redis integration for frequently accessed data
- **Database**: Connection pooling and async operations

### Scalability
- **Horizontal Scaling**: Stateless API design
- **Load Balancing**: FastAPI with multiple workers
- **Data Processing**: Celery for background tasks
- **Monitoring**: Health checks and system metrics

## Risk Disclaimer

⚠️ **Important**: This software is for educational and research purposes only. 

- **No Financial Advice**: This platform does not provide financial advice
- **Risk of Loss**: Trading involves substantial risk of loss
- **Paper Trading**: Use with paper trading accounts first
- **Due Diligence**: Always conduct your own research
- **Regulatory Compliance**: Ensure compliance with local regulations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, feature requests, or bug reports:
- Create an issue on GitHub
- Check the documentation
- Review the examples in the code

## Roadmap

### Phase 1 (Current)
- ✅ Basic trading strategies
- ✅ Real-time data integration
- ✅ Web interface
- ✅ Greeks calculation

### Phase 2 (Planned)
- [ ] Machine learning models
- [ ] Backtesting framework
- [ ] Advanced order types
- [ ] Portfolio optimization

### Phase 3 (Future)
- [ ] Multi-asset support (crypto, forex)
- [ ] Real broker integration
- [ ] Mobile application
- [ ] Advanced charting

## Acknowledgments

- **Data Providers**: Yahoo Finance, Alpha Vantage, Polygon.io
- **Technical Analysis**: TA-Lib library
- **Web Framework**: FastAPI and modern web technologies
- **Financial Libraries**: yfinance, pandas, numpy, scipy