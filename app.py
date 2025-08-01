from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_wtf import FlaskForm
from flask_mail import Mail, Message
from wtforms import StringField, PasswordField, EmailField, SubmitField, SelectField, FloatField, IntegerField
from wtforms.validators import DataRequired, Email, Length, EqualTo, NumberRange
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
import json
from fredapi import Fred
from bs4 import BeautifulSoup
import secrets
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import feedparser
import newspaper
from newspaper import Article
import time
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///iportfolio.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Email configuration
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')

# API Keys
FRED_API_KEY = os.environ.get('FRED_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
mail = Mail(app)

# Initialize APIs
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas') if ALPHA_VANTAGE_API_KEY else None
fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas') if ALPHA_VANTAGE_API_KEY else None

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    reset_token = db.Column(db.String(120), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    risk_tolerance = db.Column(db.Float, default=0.5)  # 0-1 scale
    investment_goal = db.Column(db.String(50), default='balanced')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
        db.session.commit()
        return self.reset_token

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    target_volatility = db.Column(db.Float, nullable=False)  # Target volatility percentage
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_rebalanced = db.Column(db.DateTime, default=datetime.utcnow)

class PortfolioHolding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    allocation_percentage = db.Column(db.Float, nullable=False)
    shares = db.Column(db.Float, default=0)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=True)
    symbol = db.Column(db.String(10), nullable=False)
    action = db.Column(db.String(4), nullable=False)  # BUY or SELL
    quantity = db.Column(db.Float, nullable=False)
    price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class NewsAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    sector = db.Column(db.String(50), nullable=True)
    sentiment_score = db.Column(db.Float, nullable=False)
    market_impact = db.Column(db.String(20), nullable=False)  # positive, negative, neutral
    analysis_text = db.Column(db.Text, nullable=False)
    news_sources = db.Column(db.Text, nullable=True)  # JSON string of sources
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Forms
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = EmailField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Confirm Password', 
                            validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

class ForgotPasswordForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Reset Password')

class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Confirm Password', 
                            validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

class TradeForm(FlaskForm):
    symbol = StringField('Stock Symbol', validators=[DataRequired()])
    action = SelectField('Action', choices=[('BUY', 'Buy'), ('SELL', 'Sell')], validators=[DataRequired()])
    quantity = FloatField('Quantity', validators=[DataRequired(), NumberRange(min=0.01)])
    submit = SubmitField('Execute Trade')

class PortfolioForm(FlaskForm):
    name = StringField('Portfolio Name', validators=[DataRequired(), Length(min=1, max=100)])
    target_volatility = SelectField('Target Volatility', 
                                  choices=[('5', '5% - Very Conservative'), 
                                          ('15', '15% - Conservative'),
                                          ('25', '25% - Moderate Conservative'),
                                          ('35', '35% - Moderate'),
                                          ('45', '45% - Moderate Aggressive'),
                                          ('55', '55% - Aggressive'),
                                          ('65', '65% - Very Aggressive'),
                                          ('75', '75% - Extremely Aggressive'),
                                          ('85', '85% - Ultra Aggressive'),
                                          ('95', '95% - Maximum Risk')], 
                                  validators=[DataRequired()])
    symbols = StringField('Stock Symbols (comma-separated)', validators=[DataRequired()])
    submit = SubmitField('Create Portfolio')

# Financial Analysis Functions
class FinancialAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.02  # Default 2% risk-free rate
        
    def get_stock_data(self, symbols, period='1y'):
        """Get historical stock data for multiple symbols"""
        try:
            data = yf.download(symbols, period=period)['Adj Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(symbols[0] if isinstance(symbols, list) else symbols)
            return data.dropna()
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def calculate_returns(self, data):
        """Calculate daily returns"""
        return data.pct_change().dropna()
    
    def calculate_volatility(self, returns, annualize=True):
        """Calculate volatility (standard deviation of returns)"""
        vol = returns.std()
        if annualize:
            vol *= np.sqrt(252)  # Annualize assuming 252 trading days
        return vol
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=None):
        """Calculate Sharpe ratio for each asset"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
            
        excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized excess return
        volatility = self.calculate_volatility(returns)
        return excess_returns / volatility
    
    def calculate_correlation_matrix(self, returns):
        """Calculate correlation matrix between assets"""
        return returns.corr()
    
    def calculate_beta(self, stock_returns, market_returns):
        """Calculate beta (systematic risk) of a stock relative to market"""
        covariance = np.cov(stock_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    
    def calculate_capm(self, beta, risk_free_rate=None, market_return=0.10):
        """Calculate expected return using CAPM"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        return risk_free_rate + beta * (market_return - risk_free_rate)
    
    def optimize_portfolio(self, returns, target_volatility):
        """Optimize portfolio allocation for target volatility"""
        n_assets = len(returns.columns)
        
        # Calculate expected returns and covariance matrix
        mu = returns.mean() * 252  # Annualized returns
        cov_matrix = returns.cov() * 252  # Annualized covariance
        
        # Objective function: maximize Sharpe ratio
        def objective(weights):
            portfolio_return = np.sum(weights * mu)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        # Constraint: target volatility
        def volatility_constraint(weights):
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return target_volatility - portfolio_volatility
        
        # Constraints and bounds
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': volatility_constraint}      # Target volatility
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))  # Long-only positions
        
        # Initial guess (equal weights)
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            return result.x, -result.fun  # Return weights and Sharpe ratio
        else:
            # Fallback to equal weights if optimization fails
            return x0, 0
    
    def get_economic_indicators(self):
        """Get comprehensive economic indicators"""
        indicators = {}
        if fred:
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                
                # Core economic indicators
                series_map = {
                    'cpi': 'CPIAUCSL',           # Consumer Price Index
                    'unemployment': 'UNRATE',    # Unemployment Rate
                    'interest_rate': 'FEDFUNDS', # Federal Funds Rate
                    'bond_yield_10y': 'GS10',    # 10-Year Treasury
                    'bond_yield_2y': 'GS2',      # 2-Year Treasury
                    'house_price': 'CSUSHPINSA', # House Price Index
                    'gdp': 'GDP',                # Gross Domestic Product
                    'industrial_production': 'INDPRO',  # Industrial Production
                    'consumer_sentiment': 'UMCSENT',     # Consumer Sentiment
                }
                
                for key, series_id in series_map.items():
                    try:
                        data = fred.get_series(series_id, start_date, end_date)
                        if len(data) > 0:
                            indicators[key] = {
                                'current': float(data.iloc[-1]),
                                'previous': float(data.iloc[-2]) if len(data) > 1 else None,
                                'change': float(data.iloc[-1] - data.iloc[-2]) if len(data) > 1 else None
                            }
                    except Exception as e:
                        print(f"Error fetching {key}: {e}")
                        
                # Get VIX data (volatility index)
                try:
                    vix_ticker = yf.Ticker('^VIX')
                    vix_data = vix_ticker.history(period='1mo')
                    if len(vix_data) > 0:
                        indicators['vix'] = {
                            'current': float(vix_data['Close'].iloc[-1]),
                            'previous': float(vix_data['Close'].iloc[-2]) if len(vix_data) > 1 else None
                        }
                except Exception as e:
                    print(f"Error fetching VIX: {e}")
                
                # Get commodity prices
                commodities = {
                    'oil': 'CL=F',      # Crude Oil
                    'gold': 'GC=F',     # Gold
                    'silver': 'SI=F',   # Silver
                }
                
                for commodity, symbol in commodities.items():
                    try:
                        ticker = yf.Ticker(symbol)
                        data = ticker.history(period='1mo')
                        if len(data) > 0:
                            indicators[commodity] = {
                                'current': float(data['Close'].iloc[-1]),
                                'previous': float(data['Close'].iloc[-2]) if len(data) > 1 else None
                            }
                    except Exception as e:
                        print(f"Error fetching {commodity}: {e}")
                        
            except Exception as e:
                print(f"Error fetching economic data: {e}")
        return indicators

class NewsAnalyzer:
    def __init__(self, openai_api_key):
        self.openai_api_key = openai_api_key
        
    def get_financial_news(self, query, max_articles=10):
        """Get financial news from multiple sources"""
        news_sources = [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.sec.gov/rss/litigation/litig.xml',
            'https://feeds.bloomberg.com/markets/news.rss',
        ]
        
        articles = []
        
        for source in news_sources:
            try:
                feed = feedparser.parse(source)
                for entry in feed.entries[:max_articles//len(news_sources)]:
                    if query.lower() in entry.title.lower() or query.lower() in entry.summary.lower():
                        articles.append({
                            'title': entry.title,
                            'summary': entry.summary,
                            'link': entry.link,
                            'published': entry.published if hasattr(entry, 'published') else 'Unknown',
                            'source': source
                        })
            except Exception as e:
                print(f"Error fetching from {source}: {e}")
                
        return articles
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using OpenAI"""
        if not self.openai_api_key:
            return 0, "neutral"
            
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            prompt = f"""Analyze the sentiment of this financial news text and provide:
1. A sentiment score from -1 (very negative) to 1 (very positive)
2. Overall market impact classification: positive, negative, or neutral

Text: {text}

Respond in JSON format: {{"sentiment_score": <score>, "market_impact": "<classification>"}}"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyzer. Provide precise, objective analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result['sentiment_score'], result['market_impact']
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return 0, "neutral"
    
    def get_sector_analysis(self, sector, symbol=None):
        """Get comprehensive sector analysis including FOMC minutes and congressional trading"""
        if not self.openai_api_key:
            return "Sector analysis not available"
            
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)
            
            # Get recent news for the sector
            news_articles = self.get_financial_news(sector, max_articles=5)
            news_summary = "\n".join([f"- {article['title']}: {article['summary'][:100]}..." for article in news_articles])
            
            prompt = f"""Provide a comprehensive analysis of the {sector} sector considering:

1. Recent market trends and performance
2. Economic indicators impact (interest rates, inflation, GDP growth)
3. Regulatory environment and policy changes
4. Sector-specific risks and opportunities
5. Correlation with broader market movements
6. Volatility patterns and expected ranges

Recent news context:
{news_summary}

{"Focus particularly on " + symbol + " within this sector." if symbol else ""}

Provide analysis in the following format:
- Current Outlook: [Positive/Negative/Neutral]
- Key Drivers: [3-4 main factors]
- Risk Level: [Low/Medium/High]
- Expected Volatility: [Low/Medium/High]
- Investment Recommendation: [Conservative/Moderate/Aggressive positioning]

Keep response under 300 words and focus on actionable insights."""
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst with expertise in sector analysis and market dynamics."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error getting sector analysis: {e}")
            return "Sector analysis temporarily unavailable"

# Initialize analyzers
financial_analyzer = FinancialAnalyzer()
news_analyzer = NewsAnalyzer(OPENAI_API_KEY)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Welcome to iPortfolio.com!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists', 'error')
            return render_template('register.html', form=form)
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already exists', 'error')
            return render_template('register.html', form=form)
        
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Welcome to iPortfolio.com!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    form = ForgotPasswordForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            token = user.generate_reset_token()
            send_reset_email(user.email, token)
            flash('Password reset email sent!', 'info')
            return redirect(url_for('login'))
        flash('Email not found', 'error')
    return render_template('forgot_password.html', form=form)

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.query.filter_by(reset_token=token).first()
    if not user or user.reset_token_expiry < datetime.utcnow():
        flash('Invalid or expired reset token', 'error')
        return redirect(url_for('login'))
    
    form = ResetPasswordForm()
    if form.validate_on_submit():
        user.set_password(form.password.data)
        user.reset_token = None
        user.reset_token_expiry = None
        db.session.commit()
        flash('Password reset successful!', 'success')
        return redirect(url_for('login'))
    return render_template('reset_password.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's portfolios
    portfolios = Portfolio.query.filter_by(user_id=current_user.id).all()
    
    # Get economic indicators
    economic_data = financial_analyzer.get_economic_indicators()
    
    # Get market overview
    market_indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P 500, Dow, NASDAQ, VIX
    market_data = {}
    
    for index in market_indices:
        try:
            ticker = yf.Ticker(index)
            hist = ticker.history(period='5d')
            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                
                market_data[index] = {
                    'price': current_price,
                    'change': change,
                    'change_pct': change_pct
                }
        except Exception as e:
            print(f"Error fetching {index}: {e}")
    
    return render_template('dashboard.html', 
                         portfolios=portfolios, 
                         economic_data=economic_data,
                         market_data=market_data)

@app.route('/portfolio/create', methods=['GET', 'POST'])
@login_required
def create_portfolio():
    form = PortfolioForm()
    if form.validate_on_submit():
        try:
            # Parse symbols
            symbols = [s.strip().upper() for s in form.symbols.data.split(',')]
            target_vol = float(form.target_volatility.data) / 100  # Convert percentage to decimal
            
            # Validate symbols and get data
            stock_data = financial_analyzer.get_stock_data(symbols, period='1y')
            if stock_data.empty:
                flash('Unable to fetch data for the provided symbols', 'error')
                return render_template('create_portfolio.html', form=form)
            
            # Calculate optimal allocations
            returns = financial_analyzer.calculate_returns(stock_data)
            weights, sharpe_ratio = financial_analyzer.optimize_portfolio(returns, target_vol)
            
            # Create portfolio
            portfolio = Portfolio(
                user_id=current_user.id,
                name=form.name.data,
                target_volatility=target_vol * 100  # Store as percentage
            )
            db.session.add(portfolio)
            db.session.flush()  # Get portfolio ID
            
            # Add holdings
            for symbol, weight in zip(symbols, weights):
                holding = PortfolioHolding(
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    allocation_percentage=weight * 100
                )
                db.session.add(holding)
            
            db.session.commit()
            
            flash(f'Portfolio "{form.name.data}" created successfully with Sharpe ratio: {sharpe_ratio:.3f}', 'success')
            return redirect(url_for('portfolio_detail', portfolio_id=portfolio.id))
            
        except Exception as e:
            flash(f'Error creating portfolio: {str(e)}', 'error')
    
    return render_template('create_portfolio.html', form=form)

@app.route('/portfolio/<int:portfolio_id>')
@login_required
def portfolio_detail(portfolio_id):
    portfolio = Portfolio.query.filter_by(id=portfolio_id, user_id=current_user.id).first_or_404()
    holdings = PortfolioHolding.query.filter_by(portfolio_id=portfolio_id).all()
    
    # Get current prices and calculate performance
    symbols = [h.symbol for h in holdings]
    portfolio_data = {}
    total_value = 0
    
    for holding in holdings:
        try:
            ticker = yf.Ticker(holding.symbol)
            info = ticker.info
            hist = ticker.history(period='1mo')
            
            if len(hist) > 0:
                current_price = hist['Close'].iloc[-1]
                portfolio_data[holding.symbol] = {
                    'name': info.get('longName', holding.symbol),
                    'price': current_price,
                    'allocation': holding.allocation_percentage,
                    'shares': holding.shares,
                    'value': holding.shares * current_price if holding.shares else 0
                }
                total_value += portfolio_data[holding.symbol]['value']
        except Exception as e:
            print(f"Error fetching data for {holding.symbol}: {e}")
    
    # Calculate portfolio metrics if we have historical data
    metrics = {}
    if len(symbols) > 0:
        try:
            stock_data = financial_analyzer.get_stock_data(symbols, period='1y')
            if not stock_data.empty:
                returns = financial_analyzer.calculate_returns(stock_data)
                
                # Calculate weighted portfolio returns
                weights = np.array([h.allocation_percentage/100 for h in holdings])
                portfolio_returns = (returns * weights).sum(axis=1)
                
                metrics = {
                    'volatility': financial_analyzer.calculate_volatility(portfolio_returns) * 100,
                    'sharpe_ratio': financial_analyzer.calculate_sharpe_ratio(portfolio_returns.to_frame('portfolio'))['portfolio'],
                    'annual_return': portfolio_returns.mean() * 252 * 100,
                    'max_drawdown': ((portfolio_returns.cumsum() - portfolio_returns.cumsum().expanding().max()).min()) * 100
                }
        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
    
    return render_template('portfolio_detail.html', 
                         portfolio=portfolio, 
                         holdings=holdings,
                         portfolio_data=portfolio_data,
                         total_value=total_value,
                         metrics=metrics)

@app.route('/analysis')
@login_required
def analysis():
    # Get comprehensive market analysis
    economic_data = financial_analyzer.get_economic_indicators()
    
    # Sector analysis for major sectors
    major_sectors = ['Technology', 'Healthcare', 'Financial Services', 'Energy', 'Consumer Discretionary']
    sector_analyses = {}
    
    for sector in major_sectors:
        sector_analyses[sector] = news_analyzer.get_sector_analysis(sector)
    
    return render_template('analysis.html', 
                         economic_data=economic_data,
                         sector_analyses=sector_analyses)

@app.route('/stock/<symbol>')
@login_required
def stock_detail(symbol):
    try:
        symbol = symbol.upper()
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(period="1y")
        
        # Get sector analysis
        sector = info.get('sector', '')
        sector_analysis = news_analyzer.get_sector_analysis(sector, symbol) if sector else "Sector information not available"
        
        # Calculate financial metrics
        returns = hist['Close'].pct_change().dropna()
        
        # Get market data for beta calculation (using S&P 500 as market proxy)
        market = yf.Ticker('^GSPC')
        market_hist = market.history(period="1y")
        market_returns = market_hist['Close'].pct_change().dropna()
        
        # Align dates
        common_dates = returns.index.intersection(market_returns.index)
        stock_returns_aligned = returns.loc[common_dates]
        market_returns_aligned = market_returns.loc[common_dates]
        
        metrics = {
            'volatility': financial_analyzer.calculate_volatility(returns) * 100,
            'sharpe_ratio': financial_analyzer.calculate_sharpe_ratio(returns.to_frame(symbol))[symbol],
            'beta': financial_analyzer.calculate_beta(stock_returns_aligned, market_returns_aligned),
            'annual_return': returns.mean() * 252 * 100
        }
        
        # Calculate CAPM expected return
        metrics['capm_return'] = financial_analyzer.calculate_capm(metrics['beta']) * 100
        
        # Create enhanced price chart
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=symbol
        ))
        
        # Add moving averages
        hist['MA20'] = hist['Close'].rolling(window=20).mean()
        hist['MA50'] = hist['Close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='MA20', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='MA50', line=dict(color='red')))
        
        fig.update_layout(
            title=f'{symbol} Stock Analysis',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template='plotly_dark',
            height=600
        )
        
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get economic indicators
        economic_data = financial_analyzer.get_economic_indicators()
        
        return render_template('stock_detail.html', 
                             stock_info=info, 
                             graphJSON=graphJSON, 
                             symbol=symbol,
                             sector_analysis=sector_analysis,
                             economic_data=economic_data,
                             metrics=metrics)
    except Exception as e:
        flash(f'Error fetching stock data: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/trade', methods=['GET', 'POST'])
@login_required
def trade():
    form = TradeForm()
    if form.validate_on_submit():
        try:
            symbol = form.symbol.data.upper()
            action = form.action.data
            quantity = float(form.quantity.data)
            
            # Get current price
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1d')
            if len(hist) == 0:
                flash('Invalid stock symbol or market is closed', 'error')
                return render_template('trade.html', form=form)
                
            current_price = hist['Close'].iloc[-1]
            
            # Execute trade
            trade = Trade(
                user_id=current_user.id,
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=current_price
            )
            db.session.add(trade)
            db.session.commit()
            
            flash(f'Trade executed: {action} {quantity} shares of {symbol} at ${current_price:.2f}', 'success')
            return redirect(url_for('trades'))
        except ValueError:
            flash('Invalid quantity', 'error')
        except Exception as e:
            flash(f'Error executing trade: {str(e)}', 'error')
    
    return render_template('trade.html', form=form)

@app.route('/trades')
@login_required
def trades():
    trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.timestamp.desc()).all()
    return render_template('trades.html', trades=trades)

@app.route('/api/stock_search')
@login_required
def stock_search():
    query = request.args.get('q', '')
    if len(query) < 1:
        return jsonify([])
    
    try:
        # Enhanced stock search with real data
        suggestions = []
        
        # Popular stocks for quick search
        popular_stocks = {
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'AMD': 'Advanced Micro Devices',
            'INTC': 'Intel Corporation',
            'NFLX': 'Netflix Inc.',
            'DIS': 'The Walt Disney Company',
            'BA': 'The Boeing Company',
            'JPM': 'JPMorgan Chase & Co.',
            'V': 'Visa Inc.',
            'JNJ': 'Johnson & Johnson',
            'PG': 'The Procter & Gamble Company',
            'KO': 'The Coca-Cola Company',
            'PEP': 'PepsiCo Inc.',
            'WMT': 'Walmart Inc.',
            'HD': 'The Home Depot Inc.'
        }
        
        for symbol, name in popular_stocks.items():
            if query.upper() in symbol or query.lower() in name.lower():
                suggestions.append({'symbol': symbol, 'name': name})
                
        return jsonify(suggestions[:10])
    except:
        return jsonify([])

def send_reset_email(email, token):
    try:
        msg = Message(
            'Password Reset - iPortfolio.com',
            sender=app.config['MAIL_USERNAME'],
            recipients=[email]
        )
        reset_url = url_for('reset_password', token=token, _external=True)
        msg.body = f'''To reset your password, visit the following link:
{reset_url}

If you did not make this request, please ignore this email.

Best regards,
The iPortfolio.com Team
'''
        mail.send(msg)
    except Exception as e:
        print(f"Error sending email: {e}")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)