"""
iTrade Configuration Module
Contains all configuration settings for the high-frequency trading platform
"""

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    POLYGON_API_KEY = os.getenv('POLYGON_API_KEY', '')
    YAHOO_FINANCE_API_KEY = os.getenv('YAHOO_FINANCE_API_KEY', '')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', '')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET', '')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN', '')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET', '')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://username:password@localhost:5432/itrade_db')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Trading Configuration
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10% of portfolio
    STOP_LOSS_THRESHOLD = float(os.getenv('STOP_LOSS_THRESHOLD', '0.02'))  # 2%
    TAKE_PROFIT_THRESHOLD = float(os.getenv('TAKE_PROFIT_THRESHOLD', '0.05'))  # 5%
    
    # Risk Management
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', '0.05'))  # 5% daily loss limit
    POSITION_SIZING_METHOD = os.getenv('POSITION_SIZING_METHOD', 'kelly_criterion')
    
    # Data Sources
    DATA_SOURCES = {
        'financial_news': [
            'https://www.msn.com/en-us/money',
            'https://www.wsj.com',
            'https://finance.yahoo.com/',
            'https://www.bloomberg.com',
            'https://www.cnbc.com',
            'https://www.reuters.com/finance',
            'https://www.ft.com',
            'https://www.morningstar.com',
            'https://seekingalpha.com',
            'https://www.tradingview.com'
        ],
        'personal_finance': [
            'https://www.nerdwallet.com',
            'https://www.investopedia.com',
            'https://mint.intuit.com'
        ],
        'economic_data': [
            'https://fred.stlouisfed.org',
            'https://data.worldbank.org',
            'https://www.sec.gov/edgar'
        ]
    }
    
    # Trading Hours (NYSE)
    MARKET_OPEN_HOUR = 9
    MARKET_OPEN_MINUTE = 30
    MARKET_CLOSE_HOUR = 16
    MARKET_CLOSE_MINUTE = 0
    
    # Real-time Data Configuration
    WEBSOCKET_TIMEOUT = 30
    DATA_REFRESH_INTERVAL = 1  # seconds
    NEWS_REFRESH_INTERVAL = 300  # 5 minutes
    
    # Greeks Calculation Parameters
    RISK_FREE_RATE = 0.05  # 5% annual risk-free rate
    VOLATILITY_WINDOW = 30  # days for historical volatility calculation
    
    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '8000'))
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'