"""
iTrade Database Models
SQLAlchemy models for storing trading data, positions, and financial metrics
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Asset(Base):
    __tablename__ = 'assets'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    asset_type = Column(String(50), nullable=False)  # stock, option, crypto, forex
    exchange = Column(String(50))
    sector = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prices = relationship("Price", back_populates="asset")
    positions = relationship("Position", back_populates="asset")
    greeks = relationship("Greeks", back_populates="asset")

class Price(Base):
    __tablename__ = 'prices'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('assets.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float, nullable=False)
    high_price = Column(Float, nullable=False)
    low_price = Column(Float, nullable=False)
    close_price = Column(Float, nullable=False)
    volume = Column(Integer)
    bid = Column(Float)
    ask = Column(Float)
    spread = Column(Float)
    
    # Relationship
    asset = relationship("Asset", back_populates="prices")

class Greeks(Base):
    __tablename__ = 'greeks'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('assets.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # The Greeks
    alpha = Column(Float)  # Excess return relative to market
    beta = Column(Float)   # Sensitivity to market movements
    gamma = Column(Float)  # Rate of change of delta
    delta = Column(Float)  # Price sensitivity to underlying
    theta = Column(Float)  # Time decay
    vega = Column(Float)   # Volatility sensitivity
    rho = Column(Float)    # Interest rate sensitivity
    
    # Additional risk metrics
    sigma = Column(Float)  # Volatility
    sharpe_ratio = Column(Float)
    sortino_ratio = Column(Float)
    max_drawdown = Column(Float)
    var_95 = Column(Float)  # Value at Risk 95%
    cvar_95 = Column(Float)  # Conditional Value at Risk 95%
    
    # Relationship
    asset = relationship("Asset", back_populates="greeks")

class Position(Base):
    __tablename__ = 'positions'
    
    id = Column(Integer, primary_key=True)
    asset_id = Column(Integer, ForeignKey('assets.id'), nullable=False)
    strategy_id = Column(Integer, ForeignKey('strategies.id'))
    
    # Position details
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float)
    market_value = Column(Float)
    unrealized_pnl = Column(Float)
    realized_pnl = Column(Float)
    
    # Position management
    stop_loss = Column(Float)
    take_profit = Column(Float)
    position_type = Column(String(10))  # long, short
    status = Column(String(20), default='open')  # open, closed, pending
    
    # Timestamps
    opened_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    closed_at = Column(DateTime)
    
    # Relationships
    asset = relationship("Asset", back_populates="positions")
    strategy = relationship("Strategy", back_populates="positions")
    trades = relationship("Trade", back_populates="position")

class Strategy(Base):
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    algorithm_type = Column(String(50))  # momentum, mean_reversion, arbitrage, etc.
    
    # Strategy parameters
    parameters = Column(Text)  # JSON string of strategy parameters
    risk_level = Column(String(20))  # low, medium, high
    max_allocation = Column(Float)  # Maximum portfolio allocation
    
    # Performance metrics
    total_return = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    positions = relationship("Position", back_populates="strategy")

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey('positions.id'))
    
    # Trade details
    trade_type = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)
    
    # Order details
    order_type = Column(String(20))  # market, limit, stop, stop_limit
    time_in_force = Column(String(10))  # day, gtc, ioc, fok
    
    # Execution details
    executed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    execution_venue = Column(String(50))
    
    # Relationship
    position = relationship("Position", back_populates="trades")

class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    
    # Market indices
    sp500_price = Column(Float)
    nasdaq_price = Column(Float)
    dow_price = Column(Float)
    vix_price = Column(Float)
    
    # Economic indicators
    interest_rate = Column(Float)
    inflation_rate = Column(Float)
    unemployment_rate = Column(Float)
    gdp_growth = Column(Float)
    
    # Sentiment indicators
    fear_greed_index = Column(Float)
    put_call_ratio = Column(Float)
    market_sentiment = Column(String(20))  # bullish, bearish, neutral

class NewsArticle(Base):
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    content = Column(Text)
    source = Column(String(100))
    url = Column(String(1000))
    
    # Sentiment analysis
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_magnitude = Column(Float)  # 0 to 1
    
    # Impact assessment
    market_impact = Column(String(20))  # high, medium, low
    affected_symbols = Column(String(500))  # Comma-separated symbols
    
    # Timestamps
    published_at = Column(DateTime)
    scraped_at = Column(DateTime, default=datetime.utcnow)

class AlertRule(Base):
    __tablename__ = 'alert_rules'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    asset_id = Column(Integer, ForeignKey('assets.id'))
    
    # Alert conditions
    condition_type = Column(String(50))  # price_above, price_below, volume_spike, etc.
    threshold_value = Column(Float)
    comparison_operator = Column(String(10))  # >, <, >=, <=, ==
    
    # Alert settings
    is_active = Column(Boolean, default=True)
    notification_methods = Column(String(200))  # email, sms, push, discord
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_triggered = Column(DateTime)

class Portfolio(Base):
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    
    # Portfolio metrics
    total_value = Column(Float, nullable=False)
    cash_balance = Column(Float, nullable=False)
    invested_amount = Column(Float, nullable=False)
    total_return = Column(Float)
    daily_return = Column(Float)
    
    # Risk metrics
    portfolio_beta = Column(Float)
    portfolio_alpha = Column(Float)
    portfolio_sigma = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)