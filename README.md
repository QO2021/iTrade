# iTrade.com - Professional Stock Trading Platform

A comprehensive stock trading website built with Python Flask, featuring real-time market data, AI-powered analysis, and modern web technologies.

## Features

### 🔐 User Authentication
- User registration with email verification
- Secure login/logout system
- Password reset functionality via email
- Session management with Flask-Login

### 📊 Market Data Integration
- **Yahoo Finance API**: Real-time stock prices, charts, and company information
- **Federal Reserve (FRED) API**: Economic indicators including:
  - Consumer Price Index (CPI)
  - Interest rates
  - Bond yields
  - House prices
  - Unemployment rates

### 🤖 AI-Powered Analysis
- OpenAI integration for sector analysis
- Financial and political news sentiment analysis
- Market trend predictions
- Risk assessment for stock sectors

### 💼 Trading Features
- Simulated stock trading (buy/sell)
- Portfolio management and tracking
- Trade history and performance analytics
- Real-time price charts with Plotly

### 🎨 Modern UI/UX
- Responsive Bootstrap 5 design
- Interactive charts and visualizations
- Real-time market status updates
- Mobile-friendly interface

## Technology Stack

- **Backend**: Python Flask
- **Database**: SQLAlchemy (SQLite/PostgreSQL)
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Charts**: Plotly.js
- **APIs**: Yahoo Finance, FRED, OpenAI
- **Email**: Flask-Mail

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd itrade-website
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
```

Edit `.env` file with your API keys and configuration:
```env
SECRET_KEY=your-super-secret-key-here
FRED_API_KEY=your-fred-api-key
OPENAI_API_KEY=your-openai-api-key
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
```

4. **Initialize the database**
```bash
python app.py
```

5. **Run the application**
```bash
python app.py
```

Visit `http://localhost:5000` to access the application.

## API Keys Setup

### Federal Reserve (FRED) API
1. Visit [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a free account
3. Generate an API key
4. Add to your `.env` file

### OpenAI API
1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create an account and generate an API key
3. Add to your `.env` file

### Email Configuration (Gmail)
1. Enable 2-factor authentication on your Gmail account
2. Generate an App Password
3. Use the app password in your `.env` file

## Project Structure

```
itrade-website/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # Project documentation
├── templates/            # Jinja2 templates
│   ├── base.html         # Base template
│   ├── index.html        # Landing page
│   ├── login.html        # Login page
│   ├── register.html     # Registration page
│   ├── dashboard.html    # Trading dashboard
│   ├── stock_detail.html # Stock information page
│   ├── trade.html        # Trading interface
│   └── portfolio.html    # Portfolio management
├── static/               # Static assets
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── app.js        # JavaScript functionality
└── instance/             # Instance-specific files
    └── itrade.db         # SQLite database (created automatically)
```

## Key Features Explained

### Real-time Market Data
The application fetches live stock data using the Yahoo Finance API (`yfinance` library), providing:
- Current stock prices
- Historical price charts
- Company information
- Market statistics

### Economic Indicators
Integration with the Federal Reserve's FRED API provides macroeconomic data:
- Inflation rates (CPI)
- Federal funds rate
- Treasury bond yields
- Housing market data
- Employment statistics

### AI Market Analysis
OpenAI's GPT models analyze:
- Sector performance trends
- Market sentiment from news
- Risk assessment
- Trading recommendations

### Simulated Trading
Users can practice trading with:
- Virtual $10,000 starting balance
- Real-time price execution
- Portfolio tracking
- Performance analytics

## Security Features

- Password hashing with Werkzeug
- CSRF protection with Flask-WTF
- Session management
- Email verification for password resets
- SQL injection prevention with SQLAlchemy

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
For production deployment, consider:
- Using PostgreSQL instead of SQLite
- Setting up a reverse proxy (nginx)
- Using WSGI server (Gunicorn)
- Implementing SSL/HTTPS
- Setting up monitoring and logging

## API Endpoints

- `/` - Landing page
- `/login` - User authentication
- `/register` - User registration
- `/dashboard` - Main trading dashboard
- `/stock/<symbol>` - Stock detail page
- `/trade` - Trading interface
- `/portfolio` - Portfolio management
- `/api/stock_search` - Stock search API

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational purposes. Please ensure compliance with all relevant financial regulations if adapting for real trading.

## Disclaimer

**Important**: This is a demo trading platform for educational purposes only. No real money is involved, and no actual trades are executed. Always consult with financial professionals before making real investment decisions.

## Support

For questions or issues, please open an issue on the repository or contact the development team.

---

Built with ❤️ using Python Flask and modern web technologies.