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

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///itrade.db')
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
            'Password Reset - iTrade.com',
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

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5000)