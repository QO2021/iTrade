"""
Greeks Calculator Module
Computes alpha, beta, sigma, gamma and other risk metrics for financial instruments
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GreeksResult:
    """Data class to hold Greeks calculation results"""
    symbol: str
    timestamp: datetime
    
    # The Greeks
    alpha: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0
    delta: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    
    # Additional risk metrics
    sigma: float = 0.0  # Volatility
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk 95%
    cvar_95: float = 0.0  # Conditional Value at Risk 95%
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'theta': self.theta,
            'vega': self.vega,
            'rho': self.rho,
            'sigma': self.sigma,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'cvar_95': self.cvar_95,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'annualized_volatility': self.annualized_volatility
        }

class GreeksCalculator:
    """Main class for calculating Greeks and risk metrics"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        
    def calculate_greeks(self, symbol: str, period: str = "1y", 
                        benchmark: str = "^GSPC") -> GreeksResult:
        """Calculate all Greeks and risk metrics for a symbol"""
        try:
            # Fetch price data
            stock_data = self._fetch_price_data(symbol, period)
            benchmark_data = self._fetch_price_data(benchmark, period)
            
            if stock_data.empty or benchmark_data.empty:
                logger.error(f"No data available for {symbol} or {benchmark}")
                return GreeksResult(symbol=symbol, timestamp=datetime.now())
            
            # Calculate returns
            stock_returns = stock_data['Close'].pct_change().dropna()
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            
            # Align the data
            aligned_data = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
            if aligned_data.empty:
                logger.error(f"No aligned data for {symbol} and {benchmark}")
                return GreeksResult(symbol=symbol, timestamp=datetime.now())
                
            stock_returns = aligned_data.iloc[:, 0]
            benchmark_returns = aligned_data.iloc[:, 1]
            
            # Calculate Greeks and metrics
            result = GreeksResult(symbol=symbol, timestamp=datetime.now())
            
            # Beta calculation
            result.beta = self._calculate_beta(stock_returns, benchmark_returns)
            
            # Alpha calculation  
            result.alpha = self._calculate_alpha(stock_returns, benchmark_returns, result.beta)
            
            # Volatility (Sigma)
            result.sigma = self._calculate_volatility(stock_returns)
            result.annualized_volatility = result.sigma * np.sqrt(252)
            
            # Performance metrics
            result.total_return = self._calculate_total_return(stock_data['Close'])
            result.annualized_return = self._calculate_annualized_return(stock_returns)
            
            # Risk metrics
            result.sharpe_ratio = self._calculate_sharpe_ratio(stock_returns)
            result.sortino_ratio = self._calculate_sortino_ratio(stock_returns)
            result.max_drawdown = self._calculate_max_drawdown(stock_data['Close'])
            result.var_95 = self._calculate_var(stock_returns, confidence=0.95)
            result.cvar_95 = self._calculate_cvar(stock_returns, confidence=0.95)
            
            # Options Greeks (if options data available)
            try:
                options_greeks = self._calculate_options_greeks(symbol, stock_data['Close'].iloc[-1])
                result.delta = options_greeks.get('delta', 0.0)
                result.gamma = options_greeks.get('gamma', 0.0)
                result.theta = options_greeks.get('theta', 0.0)
                result.vega = options_greeks.get('vega', 0.0)
                result.rho = options_greeks.get('rho', 0.0)
            except Exception as e:
                logger.warning(f"Could not calculate options Greeks for {symbol}: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Greeks for {symbol}: {e}")
            return GreeksResult(symbol=symbol, timestamp=datetime.now())
    
    def _fetch_price_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch historical price data"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta (systematic risk)"""
        try:
            covariance = np.cov(stock_returns, market_returns)[0][1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 0.0
                
            beta = covariance / market_variance
            return float(beta)
        except Exception as e:
            logger.error(f"Error calculating beta: {e}")
            return 0.0
    
    def _calculate_alpha(self, stock_returns: pd.Series, market_returns: pd.Series, 
                        beta: float) -> float:
        """Calculate alpha (excess return)"""
        try:
            stock_mean = stock_returns.mean() * 252  # Annualized
            market_mean = market_returns.mean() * 252  # Annualized
            
            # CAPM expected return
            expected_return = self.risk_free_rate + beta * (market_mean - self.risk_free_rate)
            
            # Alpha is actual return minus expected return
            alpha = stock_mean - expected_return
            return float(alpha)
        except Exception as e:
            logger.error(f"Error calculating alpha: {e}")
            return 0.0
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate volatility (standard deviation of returns)"""
        try:
            return float(returns.std())
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_total_return(self, prices: pd.Series) -> float:
        """Calculate total return over the period"""
        try:
            if len(prices) < 2:
                return 0.0
            return float((prices.iloc[-1] / prices.iloc[0]) - 1)
        except Exception as e:
            logger.error(f"Error calculating total return: {e}")
            return 0.0
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return"""
        try:
            cumulative_return = (1 + returns).prod() - 1
            days = len(returns)
            
            if days == 0:
                return 0.0
                
            annualized = (1 + cumulative_return) ** (252 / days) - 1
            return float(annualized)
        except Exception as e:
            logger.error(f"Error calculating annualized return: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        try:
            excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
            
            if returns.std() == 0:
                return 0.0
                
            sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
            return float(sharpe)
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (uses downside deviation)"""
        try:
            excess_returns = returns - (self.risk_free_rate / 252)
            downside_returns = excess_returns[excess_returns < 0]
            
            if len(downside_returns) == 0:
                return float('inf')
                
            downside_deviation = downside_returns.std()
            
            if downside_deviation == 0:
                return float('inf')
                
            sortino = excess_returns.mean() / downside_deviation * np.sqrt(252)
            return float(sortino)
        except Exception as e:
            logger.error(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            # Calculate running maximum
            running_max = prices.expanding().max()
            
            # Calculate drawdown
            drawdown = (prices - running_max) / running_max
            
            # Maximum drawdown
            max_dd = drawdown.min()
            return float(max_dd)
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            var = np.percentile(returns, (1 - confidence) * 100)
            return float(var)
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            var = self._calculate_var(returns, confidence)
            cvar = returns[returns <= var].mean()
            return float(cvar)
        except Exception as e:
            logger.error(f"Error calculating CVaR: {e}")
            return 0.0
    
    def _calculate_options_greeks(self, symbol: str, current_price: float) -> Dict[str, float]:
        """Calculate options Greeks using Black-Scholes model"""
        try:
            # This is a simplified implementation
            # In practice, you'd need actual options data
            
            # Fetch options data
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            
            if not options_dates:
                return {}
            
            # Get options chain for the nearest expiry
            exp_date = options_dates[0]
            options_chain = ticker.option_chain(exp_date)
            
            # Find at-the-money options
            calls = options_chain.calls
            puts = options_chain.puts
            
            if calls.empty:
                return {}
            
            # Find closest to current price
            atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
            
            if atm_call.empty:
                return {}
            
            strike = atm_call['strike'].iloc[0]
            implied_vol = atm_call['impliedVolatility'].iloc[0]
            
            # Calculate time to expiry
            exp_datetime = pd.to_datetime(exp_date)
            days_to_expiry = (exp_datetime - pd.Timestamp.now()).days
            time_to_expiry = days_to_expiry / 365.0
            
            if time_to_expiry <= 0:
                return {}
            
            # Black-Scholes Greeks calculation
            greeks = self._black_scholes_greeks(
                current_price, strike, time_to_expiry, 
                self.risk_free_rate, implied_vol
            )
            
            return greeks
            
        except Exception as e:
            logger.error(f"Error calculating options Greeks: {e}")
            return {}
    
    def _black_scholes_greeks(self, S: float, K: float, T: float, 
                             r: float, sigma: float) -> Dict[str, float]:
        """Calculate Black-Scholes Greeks"""
        try:
            # d1 and d2 calculations
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Greeks calculations
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            theta = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                     r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            
            return {
                'delta': float(delta),
                'gamma': float(gamma),
                'theta': float(theta),
                'vega': float(vega),
                'rho': float(rho)
            }
        except Exception as e:
            logger.error(f"Error in Black-Scholes calculation: {e}")
            return {}

class PortfolioGreeksCalculator:
    """Calculate Greeks for a portfolio of positions"""
    
    def __init__(self, risk_free_rate: float = 0.05):
        self.calculator = GreeksCalculator(risk_free_rate)
        
    def calculate_portfolio_greeks(self, positions: List[Dict]) -> Dict:
        """
        Calculate portfolio-level Greeks
        positions: List of dicts with 'symbol', 'quantity', 'weight' keys
        """
        try:
            portfolio_greeks = {
                'alpha': 0.0,
                'beta': 0.0,
                'sigma': 0.0,
                'total_weight': 0.0,
                'positions': []
            }
            
            total_value = sum(pos.get('market_value', 0) for pos in positions)
            
            for position in positions:
                symbol = position['symbol']
                quantity = position.get('quantity', 0)
                market_value = position.get('market_value', 0)
                weight = market_value / total_value if total_value > 0 else 0
                
                # Calculate Greeks for this position
                greeks = self.calculator.calculate_greeks(symbol)
                
                # Weight the Greeks by position size
                weighted_alpha = greeks.alpha * weight
                weighted_beta = greeks.beta * weight
                weighted_sigma = greeks.sigma * weight
                
                # Add to portfolio totals
                portfolio_greeks['alpha'] += weighted_alpha
                portfolio_greeks['beta'] += weighted_beta
                portfolio_greeks['sigma'] += weighted_sigma ** 2  # Variance
                portfolio_greeks['total_weight'] += weight
                
                # Store position details
                position_greeks = greeks.to_dict()
                position_greeks.update({
                    'quantity': quantity,
                    'market_value': market_value,
                    'weight': weight,
                    'weighted_alpha': weighted_alpha,
                    'weighted_beta': weighted_beta
                })
                
                portfolio_greeks['positions'].append(position_greeks)
            
            # Calculate portfolio volatility (sqrt of variance)
            portfolio_greeks['sigma'] = np.sqrt(portfolio_greeks['sigma'])
            
            # Calculate portfolio Sharpe ratio
            if portfolio_greeks['sigma'] > 0:
                portfolio_return = sum(pos['weighted_alpha'] for pos in portfolio_greeks['positions'])
                portfolio_greeks['sharpe_ratio'] = portfolio_return / portfolio_greeks['sigma']
            else:
                portfolio_greeks['sharpe_ratio'] = 0.0
            
            portfolio_greeks['timestamp'] = datetime.now()
            
            return portfolio_greeks
            
        except Exception as e:
            logger.error(f"Error calculating portfolio Greeks: {e}")
            return {}

class RealTimeGreeksMonitor:
    """Monitor Greeks in real-time for active positions"""
    
    def __init__(self, symbols: List[str], update_interval: int = 60):
        self.symbols = symbols
        self.update_interval = update_interval
        self.calculator = GreeksCalculator()
        self.latest_greeks = {}
        
    async def start_monitoring(self):
        """Start real-time monitoring of Greeks"""
        import asyncio
        
        while True:
            try:
                await self.update_greeks()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in Greeks monitoring: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def update_greeks(self):
        """Update Greeks for all monitored symbols"""
        for symbol in self.symbols:
            try:
                greeks = self.calculator.calculate_greeks(symbol)
                self.latest_greeks[symbol] = greeks
                logger.info(f"Updated Greeks for {symbol}: "
                          f"Alpha={greeks.alpha:.4f}, Beta={greeks.beta:.4f}, "
                          f"Sigma={greeks.sigma:.4f}")
            except Exception as e:
                logger.error(f"Error updating Greeks for {symbol}: {e}")
    
    def get_latest_greeks(self, symbol: str) -> Optional[GreeksResult]:
        """Get latest Greeks for a symbol"""
        return self.latest_greeks.get(symbol)
    
    def get_all_latest_greeks(self) -> Dict[str, GreeksResult]:
        """Get latest Greeks for all symbols"""
        return self.latest_greeks.copy()

# Utility functions
def calculate_single_greek(symbol: str, greek_type: str, **kwargs) -> float:
    """Calculate a single Greek for a symbol"""
    calculator = GreeksCalculator()
    greeks = calculator.calculate_greeks(symbol)
    return getattr(greeks, greek_type, 0.0)

def compare_greeks(symbols: List[str]) -> pd.DataFrame:
    """Compare Greeks across multiple symbols"""
    calculator = GreeksCalculator()
    results = []
    
    for symbol in symbols:
        greeks = calculator.calculate_greeks(symbol)
        results.append(greeks.to_dict())
    
    df = pd.DataFrame(results)
    df.set_index('symbol', inplace=True)
    return df

def screen_by_greeks(symbols: List[str], criteria: Dict[str, Tuple[float, float]]) -> List[str]:
    """Screen symbols based on Greeks criteria
    
    criteria: Dict with Greek name as key and (min, max) tuple as value
    Example: {'beta': (0.5, 1.5), 'sharpe_ratio': (1.0, float('inf'))}
    """
    calculator = GreeksCalculator()
    qualified_symbols = []
    
    for symbol in symbols:
        greeks = calculator.calculate_greeks(symbol)
        qualified = True
        
        for greek_name, (min_val, max_val) in criteria.items():
            greek_value = getattr(greeks, greek_name, 0.0)
            if not (min_val <= greek_value <= max_val):
                qualified = False
                break
        
        if qualified:
            qualified_symbols.append(symbol)
    
    return qualified_symbols