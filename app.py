from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from flask_wtf import FlaskForm
from flask_mail import Mail, Message
from wtforms import StringField, PasswordField, EmailField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
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
from sklearn.preprocessing import StandardScaler
import pandas_datareader as pdr
from newsapi import NewsApiClient

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
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
mail = Mail(app)

# Initialize APIs
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    reset_token = db.Column(db.String(120), nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_reset_token(self):
        self.reset_token = secrets.token_urlsafe(32)
        self.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
        db.session.commit()
        return self.reset_token

# Trade model
class Trade(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    action = db.Column(db.String(4), nullable=False)  # BUY or SELL
    quantity = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Portfolio model
class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    volatility_target = db.Column(db.Float, nullable=False)  # Target volatility percentage
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Portfolio Holdings model
class PortfolioHolding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    weight = db.Column(db.Float, nullable=False)  # Percentage weight in portfolio
    shares = db.Column(db.Integer, nullable=False)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

# Market Analysis model
class MarketAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symbol = db.Column(db.String(10), nullable=False)
    analysis_date = db.Column(db.DateTime, default=datetime.utcnow)
    sector_analysis = db.Column(db.Text)
    news_sentiment = db.Column(db.Float)  # Sentiment score -1 to 1
    volatility = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    beta = db.Column(db.Float)
    correlation_data = db.Column(db.Text)  # JSON string

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
    quantity = StringField('Quantity', validators=[DataRequired()])
    submit = SubmitField('Execute Trade')

class PortfolioOptimizationForm(FlaskForm):
    portfolio_name = StringField('Portfolio Name', validators=[DataRequired(), Length(min=3, max=100)])
    volatility_target = SelectField('Volatility Target', 
                                  choices=[(i, f'{i}%') for i in range(5, 100, 10)], 
                                  validators=[DataRequired()])
    symbols = StringField('Stock Symbols (comma-separated)', validators=[DataRequired()])
    submit = SubmitField('Optimize Portfolio')

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
            flash('Logged in successfully!', 'success')
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
        flash('Registration successful! Please log in.', 'success')
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
    return render_template('dashboard.html')

@app.route('/stock/<symbol>')
@login_required
def stock_detail(symbol):
    try:
        stock = yf.Ticker(symbol.upper())
        info = stock.info
        hist = stock.history(period="1mo")
        
        # Create price chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name=symbol.upper()
        ))
        fig.update_layout(
            title=f'{symbol.upper()} Stock Price',
            yaxis_title='Price ($)',
            xaxis_title='Date',
            template='plotly_dark'
        )
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        # Get sector analysis
        sector_analysis = get_sector_analysis(info.get('sector', ''))
        
        # Get economic indicators
        economic_data = get_economic_indicators()
        
        return render_template('stock_detail.html', 
                             stock_info=info, 
                             graphJSON=graphJSON, 
                             symbol=symbol.upper(),
                             sector_analysis=sector_analysis,
                             economic_data=economic_data)
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
            quantity = int(form.quantity.data)
            
            # Get current price
            stock = yf.Ticker(symbol)
            current_price = stock.info.get('currentPrice', 0)
            
            if current_price == 0:
                flash('Invalid stock symbol', 'error')
                return render_template('trade.html', form=form)
            
            # Execute trade (simplified - in real app, you'd integrate with broker API)
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
            return redirect(url_for('portfolio'))
        except ValueError:
            flash('Invalid quantity', 'error')
        except Exception as e:
            flash(f'Error executing trade: {str(e)}', 'error')
    
    return render_template('trade.html', form=form)

@app.route('/portfolio')
@login_required
def portfolio():
    trades = Trade.query.filter_by(user_id=current_user.id).order_by(Trade.timestamp.desc()).all()
    return render_template('portfolio.html', trades=trades)

@app.route('/optimization', methods=['GET', 'POST'])
@login_required
def optimization():
    form = PortfolioOptimizationForm()
    if form.validate_on_submit():
        try:
            symbols = [s.strip().upper() for s in form.symbols.data.split(',')]
            target_volatility = int(form.volatility_target.data)
            portfolio_name = form.portfolio_name.data
            
            # Get market data and optimize
            market_data = get_market_data_for_symbols(symbols)
            if market_data.empty:
                flash('Error fetching market data for symbols', 'error')
                return render_template('optimization.html', form=form)
            
            returns_data = calculate_returns(market_data)
            optimization_result = optimize_portfolio(symbols, target_volatility, returns_data)
            
            # Save portfolio to database
            portfolio = Portfolio(
                user_id=current_user.id,
                name=portfolio_name,
                volatility_target=target_volatility
            )
            db.session.add(portfolio)
            db.session.commit()
            
            # Save holdings
            for symbol, weight in optimization_result['weights'].items():
                holding = PortfolioHolding(
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    weight=weight * 100,  # Store as percentage
                    shares=0  # Would be calculated based on investment amount
                )
                db.session.add(holding)
            
            db.session.commit()
            flash('Portfolio optimized successfully!', 'success')
            return render_template('optimization_result.html', 
                                 result=optimization_result, 
                                 symbols=symbols,
                                 portfolio_name=portfolio_name)
        
        except Exception as e:
            flash(f'Error optimizing portfolio: {str(e)}', 'error')
    
    return render_template('optimization.html', form=form)

@app.route('/analytics')
@login_required
def analytics():
    # Get user's portfolios
    portfolios = Portfolio.query.filter_by(user_id=current_user.id).all()
    
    # Get economic indicators
    economic_data = get_economic_indicators_extended()
    
    # Get FF factors
    ff_factors = get_fama_french_factors()
    
    # Sample stocks for correlation analysis
    sample_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'META']
    correlation_data = analyze_correlation(sample_symbols)
    
    return render_template('analytics.html', 
                         portfolios=portfolios,
                         economic_data=economic_data,
                         ff_factors=ff_factors,
                         correlation_data=correlation_data,
                         sample_symbols=sample_symbols)

@app.route('/market_news')
@login_required
def market_news():
    # Get financial news
    general_news = get_financial_news("financial markets stock market", page_size=10)
    fomc_news = get_financial_news("FOMC Federal Reserve meeting", page_size=5)
    congress_news = get_financial_news("congress trading stocks", page_size=5)
    
    # Analyze sentiment
    general_sentiment = analyze_news_sentiment(general_news)
    
    return render_template('market_news.html',
                         general_news=general_news,
                         fomc_news=fomc_news,
                         congress_news=congress_news,
                         general_sentiment=general_sentiment)

@app.route('/api/stock_analysis/<symbol>')
@login_required
def stock_analysis_api(symbol):
    try:
        # Get stock data
        stock = yf.Ticker(symbol.upper())
        hist = stock.history(period="1y")
        
        if hist.empty:
            return jsonify({'error': 'No data found for symbol'})
        
        # Calculate metrics
        returns = calculate_returns(hist['Close'])
        volatility = calculate_volatility(returns)
        sharpe_ratio = calculate_sharpe_ratio(returns)
        
        # Get market data for beta calculation
        market = yf.Ticker("^GSPC")  # S&P 500
        market_hist = market.history(period="1y")
        market_returns = calculate_returns(market_hist['Close'])
        
        # Align dates
        aligned_data = pd.concat([returns, market_returns], axis=1, join='inner')
        if not aligned_data.empty:
            beta = calculate_beta(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
            capm_return = calculate_capm(beta)
        else:
            beta = 1.0
            capm_return = 0.08
        
        return jsonify({
            'symbol': symbol.upper(),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'beta': float(beta),
            'camp_return': float(capm_return),
            'current_price': float(hist['Close'].iloc[-1])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/stock_search')
@login_required
def stock_search():
    query = request.args.get('q', '')
    if len(query) < 1:
        return jsonify([])
    
    try:
        # Simple stock search (in production, use a proper stock search API)
        suggestions = []
        common_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'NFLX']
        for stock in common_stocks:
            if query.upper() in stock:
                suggestions.append({'symbol': stock, 'name': stock})
        return jsonify(suggestions[:5])
    except:
        return jsonify([])

def send_reset_email(email, token):
    try:
        msg = Message(
            'Password Reset - iportfolio.com',
            sender=app.config['MAIL_USERNAME'],
            recipients=[email]
        )
        reset_url = url_for('reset_password', token=token, _external=True)
        msg.body = f'''To reset your password, visit the following link:
{reset_url}

If you did not make this request, please ignore this email.
'''
        mail.send(msg)
    except Exception as e:
        print(f"Error sending email: {e}")

def get_economic_indicators():
    indicators = {}
    if fred:
        try:
            # Get recent economic data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            cpi_data = fred.get_series('CPIAUCSL', start_date, end_date)
            unemployment_data = fred.get_series('UNRATE', start_date, end_date)
            interest_rate_data = fred.get_series('FEDFUNDS', start_date, end_date)
            bond_yield_data = fred.get_series('GS10', start_date, end_date)
            house_price_data = fred.get_series('CSUSHPINSA', start_date, end_date)
            
            if len(cpi_data) > 0:
                indicators['cpi'] = cpi_data.iloc[-1]
            if len(unemployment_data) > 0:
                indicators['unemployment'] = unemployment_data.iloc[-1]
            if len(interest_rate_data) > 0:
                indicators['interest_rate'] = interest_rate_data.iloc[-1]
            if len(bond_yield_data) > 0:
                indicators['bond_yield'] = bond_yield_data.iloc[-1]
            if len(house_price_data) > 0:
                indicators['house_price'] = house_price_data.iloc[-1]
        except Exception as e:
            print(f"Error fetching economic data: {e}")
    return indicators

def get_sector_analysis(sector):
    if not OPENAI_API_KEY or not sector:
        return "Sector analysis not available"
    
    try:
        # Get recent news and analyze sector trends
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        prompt = f"""Analyze the current market sentiment and trends for the {sector} sector. 
        Consider recent news, market conditions, and provide a brief analysis of whether 
        this sector is likely to perform well in the short term. Keep response under 200 words."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst providing market insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting sector analysis: {e}")
        return "Sector analysis temporarily unavailable"

# Advanced Financial Calculations
def calculate_returns(prices):
    """Calculate returns from price series"""
    return prices.pct_change().dropna()

def calculate_volatility(returns, periods=252):
    """Calculate annualized volatility"""
    return returns.std() * np.sqrt(periods)

def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods=252):
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate/periods
    return (excess_returns.mean() * periods) / (returns.std() * np.sqrt(periods))

def calculate_beta(stock_returns, market_returns):
    """Calculate beta coefficient"""
    covariance = np.cov(stock_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    return covariance / market_variance if market_variance != 0 else 0

def calculate_capm(beta, risk_free_rate=0.02, market_return=0.10):
    """Calculate expected return using CAPM"""
    return risk_free_rate + beta * (market_return - risk_free_rate)

def get_fama_french_factors():
    """Get Fama-French factors from FRED or other sources"""
    try:
        if fred:
            # Get market risk premium (simplified)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*5)
            
            # Simplified factors - in real implementation, use proper FF data
            return {
                'market_premium': 0.06,
                'smb': 0.02,  # Small minus Big
                'hml': 0.03,  # High minus Low
                'rmw': 0.01,  # Robust minus Weak
                'cma': 0.01   # Conservative minus Aggressive
            }
    except Exception as e:
        print(f"Error fetching FF factors: {e}")
    
    return {
        'market_premium': 0.06,
        'smb': 0.02,
        'hml': 0.03,
        'rmw': 0.01,
        'cma': 0.01
    }

def optimize_portfolio(symbols, target_volatility, returns_data):
    """Optimize portfolio using Modern Portfolio Theory"""
    try:
        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean() * 252
        cov_matrix = returns_data.cov() * 252
        
        n_assets = len(symbols)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        def portfolio_return(weights):
            return np.sum(expected_returns * weights)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
            {'type': 'eq', 'fun': lambda x: portfolio_volatility(x) - target_volatility/100}  # target volatility
        ]
        
        # Bounds
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Maximize return for given volatility
        result = minimize(
            lambda x: -portfolio_return(x),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            weights = result.x
            expected_return = portfolio_return(weights)
            volatility = portfolio_volatility(weights)
            sharpe = (expected_return - 0.02) / volatility
            
            return {
                'weights': dict(zip(symbols, weights)),
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'success': True
            }
    except Exception as e:
        print(f"Error optimizing portfolio: {e}")
    
    # Fallback to equal weights
    equal_weight = 1.0 / len(symbols)
    return {
        'weights': {symbol: equal_weight for symbol in symbols},
        'expected_return': 0.08,
        'volatility': target_volatility/100,
        'sharpe_ratio': 0.4,
        'success': False
    }

def get_market_data_for_symbols(symbols, period='1y'):
    """Get historical market data for multiple symbols"""
    try:
        data = yf.download(symbols, period=period)
        if len(symbols) == 1:
            return data['Adj Close'].to_frame(symbols[0])
        else:
            return data['Adj Close']
    except Exception as e:
        print(f"Error fetching market data: {e}")
        return pd.DataFrame()

def analyze_correlation(symbols, period='1y'):
    """Calculate correlation matrix between symbols"""
    try:
        data = get_market_data_for_symbols(symbols, period)
        returns = calculate_returns(data)
        correlation_matrix = returns.corr()
        return correlation_matrix.to_dict()
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return {}

def get_financial_news(query="financial markets", language='en', page_size=10):
    """Get financial news using News API"""
    if not NEWS_API_KEY:
        return []
    
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        articles = newsapi.get_everything(
            q=query,
            language=language,
            sort_by='publishedAt',
            page_size=page_size
        )
        return articles.get('articles', [])
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def analyze_news_sentiment(articles):
    """Analyze sentiment of news articles using OpenAI"""
    if not OPENAI_API_KEY or not articles:
        return 0.0
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Combine article titles and descriptions
        text_to_analyze = " ".join([
            f"{article.get('title', '')} {article.get('description', '')}"
            for article in articles[:5]  # Analyze first 5 articles
        ])
        
        prompt = f"""Analyze the sentiment of the following financial news text and return a score between -1 (very negative) and 1 (very positive):

{text_to_analyze}

Return only a numerical score between -1 and 1."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial sentiment analyst. Return only numerical scores."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.3
        )
        
        try:
            sentiment_score = float(response.choices[0].message.content.strip())
            return max(-1, min(1, sentiment_score))  # Clamp between -1 and 1
        except ValueError:
            return 0.0
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 0.0

def get_economic_indicators_extended():
    """Get extended economic indicators including VIX, oil, gold"""
    indicators = get_economic_indicators()
    
    try:
        # Get VIX, oil, and gold prices
        vix = yf.Ticker("^VIX")
        oil = yf.Ticker("CL=F")  # Crude Oil Futures
        gold = yf.Ticker("GC=F")  # Gold Futures
        
        vix_data = vix.history(period="5d")
        oil_data = oil.history(period="5d")
        gold_data = gold.history(period="5d")
        
        if not vix_data.empty:
            indicators['vix'] = vix_data['Close'].iloc[-1]
        if not oil_data.empty:
            indicators['oil_price'] = oil_data['Close'].iloc[-1]
        if not gold_data.empty:
            indicators['gold_price'] = gold_data['Close'].iloc[-1]
            
    except Exception as e:
        print(f"Error fetching additional indicators: {e}")
    
    return indicators

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)