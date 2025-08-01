# iportfolio.com - AI-Powered Portfolio Management Platform

A comprehensive portfolio management platform built with Flask, featuring real-time market data, AI-powered analysis, advanced portfolio optimization, and risk assessment tools.

## Features

### Core Portfolio Management
- **Portfolio Optimization**: Modern Portfolio Theory implementation with volatility targeting
- **Real-time Market Data**: Integration with Yahoo Finance API for live market data
- **Interactive Charts**: Beautiful candlestick charts and correlation matrices with Plotly.js
- **Trade Execution**: Simplified trading interface (mock implementation)
- **Portfolio Tracking**: Advanced portfolio analytics and performance tracking

### Advanced Financial Analytics
- **CAPM Calculations**: Capital Asset Pricing Model analysis
- **Sharpe Ratio**: Risk-adjusted return measurements
- **Beta Calculations**: Market sensitivity analysis
- **Volatility Analysis**: Comprehensive risk assessment
- **Correlation Analysis**: Asset correlation matrices and heatmaps
- **Fama-French Model**: Five-factor model implementation for turnover advice

### Economic Data Integration
- **Federal Reserve Data**: Integration with FRED API for economic indicators
- **Market Indicators**: CPI, unemployment rate, interest rates, bond yields, house prices
- **Commodity Prices**: Oil and gold price tracking
- **VIX Integration**: Market volatility and fear index
- **Economic Dashboard**: Real-time economic indicator visualization

### AI-Powered Analysis
- **News Sentiment Analysis**: OpenAI-powered analysis of financial news
- **Sector Analysis**: AI-driven sector trend analysis
- **Market Intelligence**: Political and economic impact assessment
- **FOMC Analysis**: Federal Reserve meeting minutes analysis
- **Congressional Trading**: Political trading activity monitoring
- **Risk Assessment**: AI-powered portfolio risk evaluation

### Portfolio Optimization Engine
- **Volatility Targeting**: Create portfolios with specific volatility levels (5%-95%)
- **Modern Portfolio Theory**: Efficient frontier optimization
- **Risk-Return Optimization**: Maximize returns for given risk levels
- **Asset Allocation**: Intelligent asset weight distribution
- **Rebalancing Advice**: Systematic portfolio rebalancing recommendations

### User Interface
- **Responsive Design**: Bootstrap 5 for mobile-friendly interface
- **Professional Theme**: Modern financial platform aesthetics
- **Interactive Elements**: Real-time updates and smooth animations
- **Advanced Visualizations**: Heatmaps, pie charts, and correlation matrices
- **Multiple Dashboards**: Specialized views for different use cases

## Technology Stack

### Backend
- **Flask**: Python web framework
- **SQLAlchemy**: Database ORM with advanced models
- **Flask-Login**: User session management
- **Flask-WTF**: Form handling and validation
- **Flask-Mail**: Email functionality
- **NumPy**: Numerical computing for financial calculations
- **Pandas**: Data manipulation and analysis
- **SciPy**: Scientific computing and optimization
- **Scikit-learn**: Machine learning for portfolio optimization

### Frontend
- **Bootstrap 5**: Advanced CSS framework
- **Plotly.js**: Interactive charts, heatmaps, and 3D visualizations
- **Font Awesome**: Professional icon library
- **Custom JavaScript**: Advanced client-side functionality

### APIs & Data Sources
- **Yahoo Finance**: Comprehensive stock market data via yfinance
- **FRED API**: Economic data from Federal Reserve Bank of St. Louis
- **News API**: Financial news aggregation and analysis
- **OpenAI API**: Advanced AI-powered analysis and insights

### Database
- **SQLite**: Development database with advanced schema
- **PostgreSQL**: Production database support
- **Advanced Models**: Portfolio, Holdings, and Analysis tables

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd iportfolio
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys and configuration
   ```

5. **Initialize database**
   ```bash
   python app.py
   # Database will be created automatically on first run
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

The application will be available at `http://localhost:5000`

## Configuration

### Required API Keys

1. **FRED API Key** (Free)
   - Visit: https://fred.stlouisfed.org/docs/api/api_key.html
   - Sign up for an account and get your API key
   - Add to `.env` file: `FRED_API_KEY=your_key_here`

2. **OpenAI API Key** (Paid)
   - Visit: https://platform.openai.com/api-keys
   - Create an account and get your API key
   - Add to `.env` file: `OPENAI_API_KEY=your_key_here`

3. **News API Key** (Free tier available)
   - Visit: https://newsapi.org
   - Register for an API key
   - Add to `.env` file: `NEWS_API_KEY=your_key_here`

4. **Email Configuration** (Optional, for password reset)
   - Configure SMTP settings in `.env`
   - For Gmail, use app-specific passwords

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
SECRET_KEY=your-secret-key-here-use-random-string
DATABASE_URL=sqlite:///iportfolio.db
FRED_API_KEY=your-fred-api-key-from-fred-website
OPENAI_API_KEY=your-openai-api-key-from-platform
NEWS_API_KEY=your-news-api-key-from-newsapi
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

## Usage

### Getting Started

1. **Register an Account**
   - Visit the homepage and click "Register"
   - Provide username, email, and secure password

2. **Explore the Dashboard**
   - View market overview and economic indicators
   - Check recent news and sentiment analysis
   - Browse portfolio analytics

3. **Portfolio Optimization**
   - Navigate to "Optimization" tab
   - Enter desired stocks and volatility target
   - Get AI-optimized portfolio weights
   - View Fama-French model recommendations

4. **Advanced Analytics**
   - Access correlation matrices
   - View Fama-French factors
   - Monitor economic indicators
   - Get AI-powered market analysis

5. **Market News & Intelligence**
   - Read curated financial news
   - Monitor FOMC developments
   - Track congressional trading activity
   - Analyze market sentiment

### Portfolio Optimization Features

- **Volatility Targeting**: Choose from 5% to 95% volatility with 10% increments
- **Modern Portfolio Theory**: Efficient frontier optimization
- **Risk Metrics**: Comprehensive risk assessment including Sharpe ratio
- **Asset Allocation**: Optimized weight distribution
- **Turnover Advice**: Fama-French model recommendations

### Advanced Analytics

- **CAPM Analysis**: Capital Asset Pricing Model calculations
- **Beta Analysis**: Market sensitivity measurements
- **Correlation Analysis**: Asset correlation matrices
- **Volatility Analysis**: Historical and implied volatility
- **Economic Indicators**: Real-time economic data dashboard

## API Endpoints

### Portfolio Management
- `/optimization`: Portfolio optimization interface
- `/analytics`: Advanced analytics dashboard
- `/market_news`: News and sentiment analysis

### API Routes
- `/api/stock_analysis/<symbol>`: Get comprehensive stock analysis
- `/api/stock_search`: Stock symbol search functionality

## Development

### Project Structure
```
iportfolio/
├── app.py                     # Main Flask application with advanced features
├── requirements.txt           # Python dependencies including ML libraries
├── .env.example              # Environment variables template
├── README.md                 # Comprehensive documentation
├── templates/                # HTML templates
│   ├── base.html             # Base template with navigation
│   ├── index.html            # Homepage
│   ├── login.html            # Login page
│   ├── register.html         # Registration page
│   ├── dashboard.html        # User dashboard
│   ├── optimization.html     # Portfolio optimization
│   ├── optimization_result.html # Optimization results
│   ├── analytics.html        # Advanced analytics
│   ├── market_news.html      # News and sentiment
│   ├── trade.html            # Trading interface
│   ├── portfolio.html        # Portfolio view
│   └── stock_detail.html     # Stock details
├── static/                   # Static files
│   ├── css/
│   │   └── style.css         # Custom styles
│   └── js/
│       └── app.js            # JavaScript functionality
└── instance/                 # Instance-specific files
    └── iportfolio.db         # SQLite database (auto-generated)
```

### Database Models

- **User**: User accounts and authentication
- **Trade**: Trading history and transactions
- **Portfolio**: Portfolio definitions and metadata
- **PortfolioHolding**: Individual portfolio holdings
- **MarketAnalysis**: Cached analysis results

### Advanced Features

#### Portfolio Optimization
- Modern Portfolio Theory implementation
- Scipy optimization for efficient frontiers
- Risk-return optimization
- Volatility targeting

#### Financial Calculations
- CAPM (Capital Asset Pricing Model)
- Sharpe ratio calculations
- Beta coefficient analysis
- Correlation matrix generation
- Volatility calculations

#### AI Integration
- OpenAI-powered sentiment analysis
- News impact assessment
- Sector trend analysis
- Market condition evaluation

#### Economic Data
- FRED API integration
- Real-time economic indicators
- VIX, oil, and gold price tracking
- Economic impact analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is for educational and demonstration purposes only. It provides simulated trading functionality and should not be used for actual financial transactions. The portfolio optimization and analysis tools are based on historical data and do not guarantee future performance. Always consult with qualified financial advisors before making investment decisions.

## Support

For support, questions, or feature requests, please open an issue on the GitHub repository.

## Acknowledgments

- Yahoo Finance for comprehensive market data
- Federal Reserve Economic Data (FRED) for economic indicators
- OpenAI for AI-powered analysis capabilities
- News API for financial news aggregation
- Bootstrap and Font Awesome for UI components
- Plotly.js for advanced data visualizations
- Scientific Python ecosystem (NumPy, Pandas, SciPy, Scikit-learn)