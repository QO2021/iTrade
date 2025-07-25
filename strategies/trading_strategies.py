"""
Trading Strategies Module
Contains various algorithmic trading strategies for high-frequency trading
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from enum import Enum

from analytics.greeks_calculator import GreeksCalculator
from data_sources.financial_data import MarketDataAggregator
from data_sources.news_scraper import get_ticker_sentiment

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

@dataclass
class TradingSignal:
    """Trading signal data structure"""
    symbol: str
    signal_type: SignalType
    confidence: float  # 0-1
    price: float
    timestamp: datetime
    strategy_name: str
    reason: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        self.name = name
        self.params = params or {}
        self.data_provider = MarketDataAggregator()
        self.greeks_calculator = GreeksCalculator()
        
    @abstractmethod
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate trading signal for a symbol"""
        pass
    
    @abstractmethod
    def get_required_periods(self) -> int:
        """Return minimum periods required for strategy"""
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data"""
        if data.empty or len(data) < self.get_required_periods():
            return False
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return all(col in data.columns for col in required_columns)

class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'volume_multiplier': 1.5
        }
        if params:
            default_params.update(params)
        super().__init__("Momentum Strategy", default_params)
    
    def get_required_periods(self) -> int:
        return max(self.params['rsi_period'], self.params['macd_slow']) + 10
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate momentum-based signal"""
        if not self.validate_data(data):
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0.0,
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason="Insufficient data"
            )
        
        try:
            # Calculate technical indicators
            close_prices = data['Close'].values
            volume = data['Volume'].values
            
            # RSI
            rsi = talib.RSI(close_prices, timeperiod=self.params['rsi_period'])
            current_rsi = rsi[-1]
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                close_prices,
                fastperiod=self.params['macd_fast'],
                slowperiod=self.params['macd_slow'],
                signalperiod=self.params['macd_signal']
            )
            
            # Volume analysis
            avg_volume = np.mean(volume[-20:])  # 20-day average
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume
            
            # Price momentum
            price_change = (close_prices[-1] - close_prices[-5]) / close_prices[-5]
            
            # Generate signal
            signal_type = SignalType.HOLD
            confidence = 0.5
            reason = "Neutral momentum"
            
            # Buy conditions
            if (current_rsi < self.params['rsi_oversold'] and
                macd[-1] > macd_signal[-1] and
                macd_hist[-1] > macd_hist[-2] and
                volume_ratio > self.params['volume_multiplier']):
                
                signal_type = SignalType.STRONG_BUY
                confidence = 0.8 + min(0.2, volume_ratio - 1)
                reason = f"Oversold RSI ({current_rsi:.1f}), bullish MACD crossover, high volume"
                
            elif (current_rsi < 50 and
                  macd[-1] > macd_signal[-1] and
                  price_change > 0.02):
                
                signal_type = SignalType.BUY
                confidence = 0.6 + min(0.2, price_change * 10)
                reason = f"Bullish momentum, RSI {current_rsi:.1f}, positive price change"
            
            # Sell conditions
            elif (current_rsi > self.params['rsi_overbought'] and
                  macd[-1] < macd_signal[-1] and
                  macd_hist[-1] < macd_hist[-2]):
                
                signal_type = SignalType.STRONG_SELL
                confidence = 0.8
                reason = f"Overbought RSI ({current_rsi:.1f}), bearish MACD crossover"
                
            elif (current_rsi > 60 and
                  macd[-1] < macd_signal[-1] and
                  price_change < -0.02):
                
                signal_type = SignalType.SELL
                confidence = 0.6
                reason = f"Bearish momentum, RSI {current_rsi:.1f}, negative price change"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=close_prices[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=reason,
                metadata={
                    'rsi': current_rsi,
                    'macd': macd[-1],
                    'macd_signal': macd_signal[-1],
                    'volume_ratio': volume_ratio,
                    'price_change': price_change
                }
            )
            
        except Exception as e:
            logger.error(f"Error in momentum strategy for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=f"Error: {str(e)}"
            )

class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'bollinger_period': 20,
            'bollinger_std': 2,
            'mean_period': 50,
            'oversold_threshold': 0.2,
            'overbought_threshold': 0.8
        }
        if params:
            default_params.update(params)
        super().__init__("Mean Reversion Strategy", default_params)
    
    def get_required_periods(self) -> int:
        return max(self.params['bollinger_period'], self.params['mean_period']) + 10
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate mean reversion signal"""
        if not self.validate_data(data):
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0.0,
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason="Insufficient data"
            )
        
        try:
            close_prices = data['Close'].values
            
            # Bollinger Bands
            upper_band, middle_band, lower_band = talib.BBANDS(
                close_prices,
                timeperiod=self.params['bollinger_period'],
                nbdevup=self.params['bollinger_std'],
                nbdevdn=self.params['bollinger_std']
            )
            
            # Moving average
            ma = talib.SMA(close_prices, timeperiod=self.params['mean_period'])
            
            # Current values
            current_price = close_prices[-1]
            current_upper = upper_band[-1]
            current_lower = lower_band[-1]
            current_middle = middle_band[-1]
            current_ma = ma[-1]
            
            # Calculate position within bands
            if current_upper != current_lower:
                band_position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                band_position = 0.5
            
            # Distance from mean
            mean_distance = (current_price - current_ma) / current_ma
            
            # Generate signal
            signal_type = SignalType.HOLD
            confidence = 0.5
            reason = "Price within normal range"
            
            # Buy signal (oversold)
            if band_position <= self.params['oversold_threshold']:
                if current_price < current_lower:
                    signal_type = SignalType.STRONG_BUY
                    confidence = 0.8 + min(0.2, (self.params['oversold_threshold'] - band_position) * 5)
                    reason = f"Price below lower Bollinger band, oversold condition"
                else:
                    signal_type = SignalType.BUY
                    confidence = 0.6
                    reason = f"Approaching lower Bollinger band, potential reversal"
            
            # Sell signal (overbought)
            elif band_position >= self.params['overbought_threshold']:
                if current_price > current_upper:
                    signal_type = SignalType.STRONG_SELL
                    confidence = 0.8 + min(0.2, (band_position - self.params['overbought_threshold']) * 5)
                    reason = f"Price above upper Bollinger band, overbought condition"
                else:
                    signal_type = SignalType.SELL
                    confidence = 0.6
                    reason = f"Approaching upper Bollinger band, potential reversal"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=reason,
                metadata={
                    'band_position': band_position,
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'middle_band': current_middle,
                    'mean_distance': mean_distance
                }
            )
            
        except Exception as e:
            logger.error(f"Error in mean reversion strategy for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=f"Error: {str(e)}"
            )

class BreakoutStrategy(BaseStrategy):
    """Breakout trading strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'lookback_period': 20,
            'volume_threshold': 1.5,
            'price_threshold': 0.02,
            'atr_period': 14,
            'atr_multiplier': 2.0
        }
        if params:
            default_params.update(params)
        super().__init__("Breakout Strategy", default_params)
    
    def get_required_periods(self) -> int:
        return max(self.params['lookback_period'], self.params['atr_period']) + 10
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate breakout signal"""
        if not self.validate_data(data):
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0.0,
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason="Insufficient data"
            )
        
        try:
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            volume = data['Volume'].values
            
            # Calculate support and resistance levels
            lookback = self.params['lookback_period']
            resistance = np.max(high_prices[-lookback:])
            support = np.min(low_prices[-lookback:])
            
            # Average True Range for volatility
            atr = talib.ATR(high_prices, low_prices, close_prices, 
                           timeperiod=self.params['atr_period'])
            current_atr = atr[-1]
            
            # Volume analysis
            avg_volume = np.mean(volume[-lookback:])
            current_volume = volume[-1]
            volume_ratio = current_volume / avg_volume
            
            # Current price
            current_price = close_prices[-1]
            previous_price = close_prices[-2]
            
            # Price change
            price_change = (current_price - previous_price) / previous_price
            
            # Generate signal
            signal_type = SignalType.HOLD
            confidence = 0.5
            reason = "No breakout detected"
            
            # Bullish breakout
            if (current_price > resistance and
                price_change > self.params['price_threshold'] and
                volume_ratio > self.params['volume_threshold']):
                
                signal_type = SignalType.STRONG_BUY
                confidence = 0.7 + min(0.3, (price_change * 10) + (volume_ratio - 1))
                reason = f"Bullish breakout above resistance ({resistance:.2f}), high volume"
                
            elif (current_price > resistance and
                  volume_ratio > 1.0):
                
                signal_type = SignalType.BUY
                confidence = 0.6
                reason = f"Price above resistance, moderate volume"
            
            # Bearish breakout
            elif (current_price < support and
                  abs(price_change) > self.params['price_threshold'] and
                  volume_ratio > self.params['volume_threshold']):
                
                signal_type = SignalType.STRONG_SELL
                confidence = 0.7 + min(0.3, (abs(price_change) * 10) + (volume_ratio - 1))
                reason = f"Bearish breakdown below support ({support:.2f}), high volume"
                
            elif (current_price < support and
                  volume_ratio > 1.0):
                
                signal_type = SignalType.SELL
                confidence = 0.6
                reason = f"Price below support, moderate volume"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=reason,
                metadata={
                    'resistance': resistance,
                    'support': support,
                    'atr': current_atr,
                    'volume_ratio': volume_ratio,
                    'price_change': price_change
                }
            )
            
        except Exception as e:
            logger.error(f"Error in breakout strategy for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=f"Error: {str(e)}"
            )

class SentimentStrategy(BaseStrategy):
    """News sentiment-based trading strategy"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'sentiment_threshold': 0.3,
            'confidence_threshold': 0.6,
            'price_momentum_weight': 0.4,
            'sentiment_weight': 0.6
        }
        if params:
            default_params.update(params)
        super().__init__("Sentiment Strategy", default_params)
    
    def get_required_periods(self) -> int:
        return 5  # Minimal price data needed
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate sentiment-based signal"""
        if not self.validate_data(data):
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0.0,
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason="Insufficient data"
            )
        
        try:
            # Get sentiment data
            sentiment_data = await get_ticker_sentiment(symbol)
            
            if not sentiment_data or 'sentiment' not in sentiment_data:
                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    price=data['Close'].iloc[-1],
                    timestamp=datetime.now(),
                    strategy_name=self.name,
                    reason="No sentiment data available"
                )
            
            sentiment = sentiment_data['sentiment']
            overall_sentiment = sentiment.get('overall_sentiment', 0.0)
            sentiment_confidence = sentiment.get('confidence', 0.0)
            article_count = sentiment.get('article_count', 0)
            
            # Price momentum
            close_prices = data['Close'].values
            short_ma = np.mean(close_prices[-5:])
            long_ma = np.mean(close_prices[-10:])
            momentum = (short_ma - long_ma) / long_ma
            
            # Combine sentiment and momentum
            sentiment_score = overall_sentiment * self.params['sentiment_weight']
            momentum_score = momentum * self.params['price_momentum_weight']
            combined_score = sentiment_score + momentum_score
            
            # Generate signal
            signal_type = SignalType.HOLD
            confidence = 0.5
            reason = "Neutral sentiment and momentum"
            
            if (overall_sentiment > self.params['sentiment_threshold'] and
                sentiment_confidence > self.params['confidence_threshold'] and
                article_count > 2):
                
                if momentum > 0:
                    signal_type = SignalType.STRONG_BUY
                    confidence = min(0.9, 0.6 + overall_sentiment + momentum)
                    reason = f"Positive sentiment ({overall_sentiment:.2f}) + bullish momentum"
                else:
                    signal_type = SignalType.BUY
                    confidence = min(0.8, 0.5 + overall_sentiment)
                    reason = f"Positive sentiment ({overall_sentiment:.2f}), mixed momentum"
                    
            elif (overall_sentiment < -self.params['sentiment_threshold'] and
                  sentiment_confidence > self.params['confidence_threshold'] and
                  article_count > 2):
                
                if momentum < 0:
                    signal_type = SignalType.STRONG_SELL
                    confidence = min(0.9, 0.6 + abs(overall_sentiment) + abs(momentum))
                    reason = f"Negative sentiment ({overall_sentiment:.2f}) + bearish momentum"
                else:
                    signal_type = SignalType.SELL
                    confidence = min(0.8, 0.5 + abs(overall_sentiment))
                    reason = f"Negative sentiment ({overall_sentiment:.2f}), mixed momentum"
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=close_prices[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=reason,
                metadata={
                    'sentiment': overall_sentiment,
                    'sentiment_confidence': sentiment_confidence,
                    'article_count': article_count,
                    'momentum': momentum,
                    'combined_score': combined_score
                }
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment strategy for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=f"Error: {str(e)}"
            )

class GreeksBasedStrategy(BaseStrategy):
    """Strategy based on Greeks and risk metrics"""
    
    def __init__(self, params: Dict[str, Any] = None):
        default_params = {
            'beta_min': 0.5,
            'beta_max': 1.5,
            'alpha_threshold': 0.05,
            'sharpe_min': 1.0,
            'max_drawdown_max': -0.15,
            'volatility_max': 0.3
        }
        if params:
            default_params.update(params)
        super().__init__("Greeks-Based Strategy", default_params)
    
    def get_required_periods(self) -> int:
        return 252  # One year of data for accurate Greeks calculation
    
    async def generate_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate signal based on Greeks and risk metrics"""
        if not self.validate_data(data):
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0.0,
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason="Insufficient data"
            )
        
        try:
            # Calculate Greeks
            greeks = self.greeks_calculator.calculate_greeks(symbol)
            
            current_price = data['Close'].iloc[-1]
            
            # Score based on Greeks criteria
            score = 0
            reasons = []
            
            # Beta check
            if self.params['beta_min'] <= greeks.beta <= self.params['beta_max']:
                score += 1
                reasons.append(f"Beta in range ({greeks.beta:.2f})")
            else:
                reasons.append(f"Beta out of range ({greeks.beta:.2f})")
            
            # Alpha check
            if greeks.alpha > self.params['alpha_threshold']:
                score += 2
                reasons.append(f"Positive alpha ({greeks.alpha:.3f})")
            elif greeks.alpha < -self.params['alpha_threshold']:
                score -= 1
                reasons.append(f"Negative alpha ({greeks.alpha:.3f})")
            
            # Sharpe ratio check
            if greeks.sharpe_ratio > self.params['sharpe_min']:
                score += 1
                reasons.append(f"Good Sharpe ratio ({greeks.sharpe_ratio:.2f})")
            else:
                reasons.append(f"Low Sharpe ratio ({greeks.sharpe_ratio:.2f})")
            
            # Max drawdown check
            if greeks.max_drawdown > self.params['max_drawdown_max']:
                score += 1
                reasons.append(f"Acceptable drawdown ({greeks.max_drawdown:.2f})")
            else:
                score -= 1
                reasons.append(f"High drawdown ({greeks.max_drawdown:.2f})")
            
            # Volatility check
            if greeks.annualized_volatility < self.params['volatility_max']:
                score += 1
                reasons.append(f"Low volatility ({greeks.annualized_volatility:.2f})")
            else:
                reasons.append(f"High volatility ({greeks.annualized_volatility:.2f})")
            
            # Generate signal based on score
            if score >= 4:
                signal_type = SignalType.STRONG_BUY
                confidence = 0.8 + min(0.2, (score - 4) * 0.1)
            elif score >= 2:
                signal_type = SignalType.BUY
                confidence = 0.6 + (score - 2) * 0.1
            elif score <= -2:
                signal_type = SignalType.SELL
                confidence = 0.6 + abs(score + 2) * 0.1
            else:
                signal_type = SignalType.HOLD
                confidence = 0.5
            
            reason = f"Greeks score: {score}/5. " + "; ".join(reasons[:3])
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=reason,
                metadata=greeks.to_dict()
            )
            
        except Exception as e:
            logger.error(f"Error in Greeks strategy for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1],
                timestamp=datetime.now(),
                strategy_name=self.name,
                reason=f"Error: {str(e)}"
            )

class StrategyEnsemble:
    """Ensemble of multiple strategies with weighted voting"""
    
    def __init__(self, strategies: List[Tuple[BaseStrategy, float]]):
        """
        Initialize with list of (strategy, weight) tuples
        Weights should sum to 1.0
        """
        self.strategies = strategies
        total_weight = sum(weight for _, weight in strategies)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Strategy weights must sum to 1.0, got {total_weight}")
    
    async def generate_ensemble_signal(self, symbol: str, data: pd.DataFrame) -> TradingSignal:
        """Generate ensemble signal from all strategies"""
        try:
            signals = []
            
            # Get signals from all strategies
            for strategy, weight in self.strategies:
                signal = await strategy.generate_signal(symbol, data)
                signals.append((signal, weight))
            
            # Calculate weighted scores
            buy_score = 0.0
            sell_score = 0.0
            hold_score = 0.0
            
            total_confidence = 0.0
            reasons = []
            
            for signal, weight in signals:
                weighted_confidence = signal.confidence * weight
                
                if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
                    multiplier = 2.0 if signal.signal_type == SignalType.STRONG_BUY else 1.0
                    buy_score += weighted_confidence * multiplier
                elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
                    multiplier = 2.0 if signal.signal_type == SignalType.STRONG_SELL else 1.0
                    sell_score += weighted_confidence * multiplier
                else:
                    hold_score += weighted_confidence
                
                total_confidence += weighted_confidence
                reasons.append(f"{signal.strategy_name}: {signal.signal_type.value}")
            
            # Determine ensemble signal
            max_score = max(buy_score, sell_score, hold_score)
            
            if max_score == buy_score and buy_score > sell_score * 1.2:
                if buy_score > 1.5:
                    signal_type = SignalType.STRONG_BUY
                else:
                    signal_type = SignalType.BUY
            elif max_score == sell_score and sell_score > buy_score * 1.2:
                if sell_score > 1.5:
                    signal_type = SignalType.STRONG_SELL
                else:
                    signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
            
            confidence = min(0.95, total_confidence / len(self.strategies))
            reason = f"Ensemble: {', '.join(reasons[:3])}"
            
            current_price = data['Close'].iloc[-1] if not data.empty else 0.0
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                price=current_price,
                timestamp=datetime.now(),
                strategy_name="Strategy Ensemble",
                reason=reason,
                metadata={
                    'buy_score': buy_score,
                    'sell_score': sell_score,
                    'hold_score': hold_score,
                    'individual_signals': [s.signal_type.value for s, _ in signals]
                }
            )
            
        except Exception as e:
            logger.error(f"Error in strategy ensemble for {symbol}: {e}")
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                price=data['Close'].iloc[-1] if not data.empty else 0.0,
                timestamp=datetime.now(),
                strategy_name="Strategy Ensemble",
                reason=f"Error: {str(e)}"
            )

# Factory function to create strategy instances
def create_strategy(strategy_type: str, params: Dict[str, Any] = None) -> BaseStrategy:
    """Factory function to create strategy instances"""
    strategies = {
        'momentum': MomentumStrategy,
        'mean_reversion': MeanReversionStrategy,
        'breakout': BreakoutStrategy,
        'sentiment': SentimentStrategy,
        'greeks': GreeksBasedStrategy
    }
    
    if strategy_type in strategies:
        return strategies[strategy_type](params)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

def create_default_ensemble() -> StrategyEnsemble:
    """Create a default ensemble with balanced strategies"""
    strategies = [
        (MomentumStrategy(), 0.25),
        (MeanReversionStrategy(), 0.25),
        (BreakoutStrategy(), 0.2),
        (SentimentStrategy(), 0.15),
        (GreeksBasedStrategy(), 0.15)
    ]
    return StrategyEnsemble(strategies)