# iPortfolio.com - AI-Powered Portfolio Management Platform

A comprehensive financial portfolio management website built with Python Flask, featuring advanced portfolio optimization, real-time market data, AI-powered analysis, and modern portfolio theory implementation.

## 🚀 Key Features

### 🔐 User Authentication & Security
- Secure user registration with email verification
- Password reset functionality via email
- Session management with Flask-Login
- CSRF protection and SQL injection prevention

### 📊 Advanced Portfolio Management
- **Portfolio Optimization**: Modern Portfolio Theory implementation with volatility targeting
- **Risk-Based Allocation**: Create portfolios with target volatility from 5% to 95%
- **Sharpe Ratio Maximization**: Automatic optimization for risk-adjusted returns
- **Correlation Analysis**: Cross-asset correlation calculations
- **Rebalancing Tools**: Portfolio rebalancing recommendations

### 🧮 Financial Calculations & Metrics
- **CAPM (Capital Asset Pricing Model)**: Expected return calculations
- **Sharpe Ratio**: Risk-adjusted performance metrics
- **Beta Calculation**: Systematic risk measurement relative to market
- **Volatility Analysis**: Historical and implied volatility calculations
- **Maximum Drawdown**: Risk assessment metrics

### 📈 Comprehensive Market Data
- **Yahoo Finance Integration**: Real-time stock prices, charts, and company data
- **Federal Reserve (FRED) API**: Economic indicators including:
  - Consumer Price Index (CPI)
  - Federal funds rate and bond yields
  - Unemployment rates
  - House price index
  - Industrial production
  - Consumer sentiment
- **Market Indices**: S&P 500, Dow Jones, NASDAQ, VIX tracking
- **Commodities**: Gold, oil, and silver prices

### 🤖 AI-Powered Financial Analysis
- **OpenAI Integration**: Advanced sector and market analysis
- **News Sentiment Analysis**: Financial news impact assessment
- **FOMC Meeting Analysis**: Federal Reserve policy impact evaluation
- **Congressional Trading Data**: Insider trading activity monitoring
- **Sector Correlation Analysis**: Cross-sector relationship evaluation
- **Risk Assessment**: AI-powered portfolio risk evaluation

### 💼 Trading & Execution
- Simulated stock trading (buy/sell orders)
- Real-time price execution
- Trade history and analytics
- Portfolio performance tracking
- Position sizing recommendations

### 🎨 Modern UI/UX
- Responsive Bootstrap 5 design
- Interactive Plotly charts and visualizations
- Real-time market data updates
- Mobile-optimized interface
- Dark/light theme support

## 🛠 Technology Stack

- **Backend**: Python Flask with SQLAlchemy ORM
- **Database**: SQLite (development) / PostgreSQL (production)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Visualization**: Plotly.js for interactive charts
- **Financial Analysis**: NumPy, Pandas, SciPy for calculations
- **Portfolio Optimization**: CVXPY for convex optimization
- **APIs**: Yahoo Finance, FRED, OpenAI, Alpha Vantage
- **Email**: Flask-Mail for notifications

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd iportfolio-website
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` file with your API keys:
```env
SECRET_KEY=your-secret-key-here
FRED_API_KEY=your-fred-api-key
OPENAI_API_KEY=your-openai-api-key
ALPHA_VANTAGE_API_KEY=your-alpha-vantage-api-key
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

4. **Run the application**
```bash
python app.py
```

Visit `http://localhost:5000` to access iPortfolio.com

## 🔑 API Keys Setup

### Required APIs

#### Federal Reserve (FRED) API - **Required**
- Visit [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
- Create free account and generate API key
- Used for economic indicators and market data

#### OpenAI API - **Required for AI Analysis**
- Visit [OpenAI Platform](https://platform.openai.com/api-keys)
- Create account and generate API key
- Used for financial news analysis and sector insights

### Optional APIs

#### Alpha Vantage API - **Optional**
- Visit [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
- Free tier available for additional market data

#### Email Configuration (Gmail) - **For Password Reset**
- Enable 2-factor authentication
- Generate App Password
- Add credentials to `.env` file

## 📁 Project Structure

```
iportfolio-website/
├── app.py                     # Main Flask application with portfolio optimization
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # Project documentation
├── templates/                # Jinja2 templates
│   ├── base.html            # Updated base template with new navigation
│   ├── index.html           # Landing page
│   ├── dashboard.html       # Portfolio dashboard with market overview
│   ├── create_portfolio.html # Portfolio creation interface
│   ├── portfolio_detail.html # Individual portfolio analysis
│   ├── analysis.html        # Comprehensive market analysis
│   ├── trades.html          # Enhanced trade history
│   ├── stock_detail.html    # Stock analysis with financial metrics
│   └── auth/                # Authentication templates
├── static/                  # Static assets
│   ├── css/style.css       # Custom styles
│   └── js/app.js           # JavaScript functionality
└── instance/               # Instance-specific files
    └── iportfolio.db       # SQLite database
```

## 🎯 Core Features Explained

### Portfolio Optimization Engine
- **Modern Portfolio Theory**: Implements Markowitz portfolio optimization
- **Risk Targeting**: Create portfolios with specific volatility targets (5%-95%)
- **Efficient Frontier**: Calculate optimal risk-return combinations
- **Constraint Optimization**: Long-only positions with weight constraints

### Financial Metrics & Analysis
- **Sharpe Ratio**: Risk-adjusted return calculations
- **Beta Calculation**: Systematic risk relative to market (S&P 500)
- **CAPM Expected Returns**: Theoretical expected returns based on risk
- **Correlation Matrix**: Cross-asset correlation analysis
- **Volatility Modeling**: Historical and forward-looking volatility

### AI-Powered Insights
- **Sector Analysis**: Deep dive into sector performance and trends
- **News Sentiment**: Real-time news impact on market sectors
- **Economic Correlation**: Link between economic indicators and stock performance
- **Risk Assessment**: AI-evaluated portfolio risk levels

### Real-time Data Integration
- **Market Data**: Live prices, volume, and market statistics
- **Economic Indicators**: Fed rates, inflation, employment data
- **Volatility Index (VIX)**: Market fear gauge
- **Commodity Prices**: Gold, oil, and precious metals

## 🔒 Security & Performance

- **Password Hashing**: Werkzeug security for password protection
- **CSRF Protection**: Flask-WTF form protection
- **SQL Injection Prevention**: SQLAlchemy ORM queries
- **Session Security**: Secure session management
- **Input Validation**: Form validation and sanitization

## 🚀 Deployment Options

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn WSGI server
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker (create Dockerfile)
docker build -t iportfolio .
docker run -p 5000:5000 iportfolio
```

### Environment Variables for Production
```env
FLASK_ENV=production
DATABASE_URL=postgresql://user:pass@localhost/iportfolio
SECRET_KEY=generated-secure-key
```

## 📊 API Endpoints

### Core Application
- `/` - Landing page
- `/dashboard` - Portfolio dashboard
- `/analysis` - Market analysis page

### Portfolio Management
- `/portfolio/create` - Create optimized portfolio
- `/portfolio/<id>` - Portfolio details and metrics
- `/trade` - Trading interface
- `/trades` - Trade history

### Stock Analysis
- `/stock/<symbol>` - Detailed stock analysis
- `/api/stock_search` - Stock search API

### Authentication
- `/login`, `/register`, `/logout` - User authentication
- `/forgot_password`, `/reset_password/<token>` - Password reset

## 🧪 Testing Portfolio Optimization

### Example Portfolio Creation
1. Navigate to "Create Portfolio"
2. Choose target volatility (e.g., 25% for moderate risk)
3. Enter stock symbols: `AAPL, GOOGL, MSFT, AMZN, TSLA`
4. System calculates optimal weights for target volatility
5. View portfolio metrics: Sharpe ratio, expected return, risk

### Sample Calculations
- **Conservative Portfolio (15% volatility)**: Higher allocation to stable stocks
- **Aggressive Portfolio (65% volatility)**: Higher allocation to growth stocks
- **Balanced Portfolio (35% volatility)**: Diversified allocation across sectors

## 📈 Financial Models Implemented

### Modern Portfolio Theory
- **Mean-Variance Optimization**: Maximizes return for given risk level
- **Efficient Frontier**: Optimal portfolios for each risk level
- **Risk-Return Trade-off**: Mathematical optimization of portfolio weights

### Capital Asset Pricing Model (CAPM)
- **Expected Return**: Risk-free rate + Beta × Market risk premium
- **Beta Calculation**: Covariance with market / Market variance
- **Risk Assessment**: Systematic vs. unsystematic risk

### Performance Metrics
- **Sharpe Ratio**: (Return - Risk-free rate) / Volatility
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Correlation Analysis**: Asset correlation matrix

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/portfolio-optimization`
3. Implement changes with tests
4. Submit pull request with detailed description

## 📄 License

Educational and demonstration purposes. Ensure compliance with financial regulations for commercial use.

## ⚠️ Important Disclaimer

**Educational Use Only**: This platform is for learning portfolio management concepts. No real trading occurs. Consult financial professionals for actual investment decisions. Past performance does not guarantee future results.

## 🆘 Support & Documentation

- **Issues**: GitHub issues for bug reports
- **Features**: Feature requests welcome
- **Documentation**: Comprehensive inline code documentation
- **API Reference**: RESTful API documentation available

---

**Built with 🧠 Modern Portfolio Theory, 📊 Real-time Data, and 🤖 AI Analysis**

*Empowering intelligent investment decisions through technology and quantitative finance.*