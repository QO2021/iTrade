# iTrade.com - Professional Stock Trading Platform

## 🚀 Overview

iTrade.com is a comprehensive stock trading platform built with Python Flask, featuring real-time market data, AI-powered analysis, and professional-grade portfolio management tools.

## ✨ Features

### 🔐 User Authentication
- **Secure Registration & Login**: Email-based account creation
- **Password Recovery**: Forgot password functionality with email verification
- **Session Management**: Secure user sessions with Flask-Login

### 📊 Market Data & Analysis
- **Real-time Stock Data**: Powered by Yahoo Finance API
- **Economic Indicators**: Fed St. Louis FRED API integration
  - Consumer Price Index (CPI)
  - Federal Funds Interest Rate
  - 10-Year Treasury Bond Yield
  - House Price Index
  - Unemployment Rate
  - VIX Volatility Index

### 🤖 AI-Powered Insights
- **LLM Market Analysis**: OpenAI GPT integration for comprehensive market sentiment analysis
- **News Analysis**: AI-driven analysis of market trends and sector performance
- **Risk Assessment**: Automated risk evaluation based on economic indicators

### 📈 Advanced Analytics
- **Correlation Analysis**: Multi-stock correlation matrices for portfolio diversification
- **Performance Metrics**: Real-time P&L calculations and portfolio performance tracking
- **Interactive Charts**: Plotly-powered candlestick charts and visualizations

### 💼 Portfolio Management
- **Trade Execution**: Buy/Sell order management
- **Position Tracking**: Real-time position monitoring with gain/loss calculations
- **Trade History**: Comprehensive trading history with detailed records
- **Performance Dashboard**: Portfolio value, P&L, and allocation analysis

### 🎨 Modern UI/UX
- **Responsive Design**: Bootstrap 5 with custom styling
- **Dark Theme Support**: Professional trading interface
- **Interactive Components**: Real-time search and dynamic content updates
- **Mobile Optimized**: Full functionality across all devices

## 🛠️ Technology Stack

### Backend
- **Python 3.8+**
- **Flask** - Web framework
- **SQLAlchemy** - Database ORM
- **Flask-Login** - User session management
- **Flask-Mail** - Email functionality
- **pandas** - Data analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning utilities

### APIs & Data Sources
- **Yahoo Finance** (`yfinance`) - Stock market data
- **FRED API** (`fredapi`) - Economic indicators
- **OpenAI API** - AI-powered market analysis
- **Plotly** - Interactive charts and visualizations

### Frontend
- **Bootstrap 5** - UI framework
- **Font Awesome** - Icon library
- **JavaScript ES6** - Interactive functionality
- **Chart.js/Plotly** - Data visualization

### Database
- **SQLite** (Development)
- **PostgreSQL** (Production ready)

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd itrade
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

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` file with your API keys:
```env
SECRET_KEY=your-secret-key-here
FRED_API_KEY=your-fred-api-key
OPENAI_API_KEY=your-openai-api-key
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

5. **Initialize database**
```bash
python app.py
```

6. **Access the application**
Open your browser and navigate to: `http://localhost:5000`

## 🔑 API Keys Setup

### FRED API Key (Free)
1. Visit [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Request a free API key
3. Add to `.env` file

### OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and get API key
3. Add to `.env` file

### Email Configuration (Optional)
For password reset functionality:
1. Use Gmail with App Password
2. Configure SMTP settings in `.env`

## 📱 Usage Guide

### Getting Started
1. **Register**: Create an account with email verification
2. **Login**: Access your personal dashboard
3. **Explore**: Navigate through market data and analysis tools

### Trading Workflow
1. **Search Stocks**: Use the search bar to find stocks
2. **Analyze**: View detailed stock information and charts
3. **Trade**: Execute buy/sell orders
4. **Monitor**: Track your portfolio performance

### Advanced Features
1. **Market Analysis**: AI-powered comprehensive market insights
2. **Correlation Analysis**: Analyze relationships between stocks
3. **Portfolio Optimization**: Use correlation data for diversification

## 🏗️ Architecture

```
iTrade/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── templates/            # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Homepage
│   ├── dashboard.html    # Trading dashboard
│   ├── portfolio.html    # Portfolio management
│   ├── correlation.html  # Correlation analysis
│   └── market_analysis.html # AI market insights
├── static/               # Static assets
│   ├── css/             # Stylesheets
│   └── js/              # JavaScript files
└── instance/            # Database files
```

## 🔧 Configuration

### Environment Variables
| Variable | Description | Required |
|----------|-------------|----------|
| `SECRET_KEY` | Flask secret key | Yes |
| `DATABASE_URL` | Database connection string | Yes |
| `FRED_API_KEY` | FRED economic data API key | No |
| `OPENAI_API_KEY` | OpenAI API key for AI analysis | No |
| `MAIL_USERNAME` | Email for password reset | No |
| `MAIL_PASSWORD` | Email password/app password | No |

### Database Models
- **User**: User accounts and authentication
- **Trade**: Trading history and transactions
- **Portfolio**: Position tracking and performance

## 🚀 Deployment

### Production Deployment
1. **Use PostgreSQL database**
```bash
pip install psycopg2-binary
```

2. **Update environment variables**
```env
DATABASE_URL=postgresql://user:password@localhost/itrade
```

3. **Use production WSGI server**
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🛡️ Security

- Passwords are hashed using Werkzeug security
- CSRF protection with Flask-WTF
- SQL injection protection with SQLAlchemy ORM
- Session management with Flask-Login

## 📞 Support

For support and questions:
- Open an issue on GitHub
- Contact: [your-email@domain.com]

## 🔄 Updates & Roadmap

### Current Version: 1.0.0
- Full trading platform functionality
- AI-powered market analysis
- Correlation analysis tools
- Comprehensive portfolio management

### Planned Features
- Real-time notifications
- Advanced charting tools
- Social trading features
- Mobile app development
- Options trading support

---

**Disclaimer**: This platform is for educational and demonstration purposes. Always consult with financial advisors before making investment decisions.