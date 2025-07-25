"""
Financial Data Ingestion Module
Handles fetching real-time and historical market data from various sources
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import asyncio
import aiohttp
import websocket
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from polygon import RESTClient
import logging

from config import Config

logger = logging.getLogger(__name__)

class FinancialDataProvider:
    """Base class for financial data providers"""
    
    def __init__(self):
        self.config = Config()
        
    async def get_real_time_price(self, symbol: str) -> Dict:
        """Get real-time price data for a symbol"""
        raise NotImplementedError
        
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data"""
        raise NotImplementedError
        
    async def get_options_data(self, symbol: str) -> Dict:
        """Get options chain data"""
        raise NotImplementedError

class YahooFinanceProvider(FinancialDataProvider):
    """Yahoo Finance data provider"""
    
    async def get_real_time_price(self, symbol: str) -> Dict:
        """Get real-time price data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'price': info.get('currentPrice', info.get('regularMarketPrice')),
                'bid': info.get('bid'),
                'ask': info.get('ask'),
                'volume': info.get('volume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'beta': info.get('beta'),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_options_data(self, symbol: str) -> Dict:
        """Get options chain data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            options = ticker.options
            
            options_data = {}
            for expiry in options[:5]:  # Get first 5 expiry dates
                chain = ticker.option_chain(expiry)
                options_data[expiry] = {
                    'calls': chain.calls.to_dict('records'),
                    'puts': chain.puts.to_dict('records')
                }
            
            return options_data
        except Exception as e:
            logger.error(f"Error fetching options data for {symbol}: {e}")
            return {}

class AlphaVantageProvider(FinancialDataProvider):
    """Alpha Vantage data provider"""
    
    def __init__(self):
        super().__init__()
        self.api_key = self.config.ALPHA_VANTAGE_API_KEY
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.fd = FundamentalData(key=self.api_key, output_format='pandas')
        
    async def get_real_time_price(self, symbol: str) -> Dict:
        """Get real-time price data from Alpha Vantage"""
        try:
            url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    data = await response.json()
                    
                    quote = data.get('Global Quote', {})
                    return {
                        'symbol': symbol,
                        'price': float(quote.get('05. price', 0)),
                        'change': float(quote.get('09. change', 0)),
                        'change_percent': quote.get('10. change percent', '0%'),
                        'volume': int(quote.get('06. volume', 0)),
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return {}
    
    async def get_fundamental_data(self, symbol: str) -> Dict:
        """Get fundamental data from Alpha Vantage"""
        try:
            overview, _ = self.fd.get_company_overview(symbol)
            income_statement, _ = self.fd.get_income_statement_annual(symbol)
            balance_sheet, _ = self.fd.get_balance_sheet_annual(symbol)
            
            return {
                'overview': overview.to_dict('records')[0] if not overview.empty else {},
                'income_statement': income_statement.to_dict('records')[:5],  # Last 5 years
                'balance_sheet': balance_sheet.to_dict('records')[:5]
            }
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {symbol}: {e}")
            return {}

class PolygonProvider(FinancialDataProvider):
    """Polygon.io data provider for high-frequency data"""
    
    def __init__(self):
        super().__init__()
        self.api_key = self.config.POLYGON_API_KEY
        self.client = RESTClient(self.api_key) if self.api_key else None
        
    async def get_real_time_price(self, symbol: str) -> Dict:
        """Get real-time price data from Polygon"""
        if not self.client:
            return {}
            
        try:
            # Get last trade
            trades = self.client.get_last_trade(symbol)
            
            # Get quote data
            quote = self.client.get_last_quote(symbol)
            
            return {
                'symbol': symbol,
                'price': trades.price,
                'size': trades.size,
                'bid': quote.bid,
                'ask': quote.ask,
                'bid_size': quote.bid_size,
                'ask_size': quote.ask_size,
                'timestamp': datetime.fromtimestamp(trades.timestamp / 1000)
            }
        except Exception as e:
            logger.error(f"Error fetching Polygon data for {symbol}: {e}")
            return {}
    
    async def get_aggregates(self, symbol: str, timespan: str = "minute", 
                           multiplier: int = 1, from_date: str = None, 
                           to_date: str = None) -> pd.DataFrame:
        """Get aggregate bars (OHLCV) data"""
        if not self.client:
            return pd.DataFrame()
            
        try:
            if not from_date:
                from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
                
            aggs = self.client.get_aggs(
                ticker=symbol,
                multiplier=multiplier,
                timespan=timespan,
                from_=from_date,
                to=to_date
            )
            
            data = []
            for agg in aggs:
                data.append({
                    'timestamp': datetime.fromtimestamp(agg.timestamp / 1000),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching aggregates for {symbol}: {e}")
            return pd.DataFrame()

class WebSocketDataFeed:
    """Real-time WebSocket data feed"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.callbacks = {}
        self.ws = None
        
    def add_callback(self, event_type: str, callback):
        """Add callback for specific event type"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            event_type = data.get('ev', 'unknown')
            
            if event_type in self.callbacks:
                for callback in self.callbacks[event_type]:
                    callback(data)
                    
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.info("WebSocket connection closed")
    
    def start_feed(self):
        """Start the WebSocket feed"""
        # This is a placeholder for WebSocket implementation
        # In practice, you'd connect to your broker's WebSocket API
        pass

class MarketDataAggregator:
    """Aggregates data from multiple sources"""
    
    def __init__(self):
        self.providers = {
            'yahoo': YahooFinanceProvider(),
            'alpha_vantage': AlphaVantageProvider(),
            'polygon': PolygonProvider()
        }
        
    async def get_best_price(self, symbol: str) -> Dict:
        """Get best price from multiple sources"""
        tasks = []
        for name, provider in self.providers.items():
            tasks.append(self._get_price_with_source(name, provider, symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and empty results
        valid_results = [r for r in results if isinstance(r, dict) and r]
        
        if not valid_results:
            return {}
        
        # Return the most recent price
        return max(valid_results, key=lambda x: x.get('timestamp', datetime.min))
    
    async def _get_price_with_source(self, source_name: str, provider: FinancialDataProvider, symbol: str) -> Dict:
        """Get price with source identifier"""
        try:
            data = await provider.get_real_time_price(symbol)
            data['source'] = source_name
            return data
        except Exception as e:
            logger.error(f"Error getting price from {source_name} for {symbol}: {e}")
            return {}
    
    async def get_comprehensive_data(self, symbol: str) -> Dict:
        """Get comprehensive data for a symbol"""
        yahoo_provider = self.providers['yahoo']
        alpha_provider = self.providers['alpha_vantage']
        
        # Fetch data in parallel
        price_task = self.get_best_price(symbol)
        historical_task = yahoo_provider.get_historical_data(symbol, "1y")
        options_task = yahoo_provider.get_options_data(symbol)
        fundamental_task = alpha_provider.get_fundamental_data(symbol)
        
        price_data, historical_data, options_data, fundamental_data = await asyncio.gather(
            price_task, historical_task, options_task, fundamental_task,
            return_exceptions=True
        )
        
        return {
            'real_time': price_data if isinstance(price_data, dict) else {},
            'historical': historical_data if isinstance(historical_data, pd.DataFrame) else pd.DataFrame(),
            'options': options_data if isinstance(options_data, dict) else {},
            'fundamentals': fundamental_data if isinstance(fundamental_data, dict) else {}
        }

class EconomicDataProvider:
    """Provider for economic indicators from FRED and other sources"""
    
    def __init__(self):
        self.fred_base_url = "https://api.stlouisfed.org/fred/series/observations"
        
    async def get_fred_data(self, series_id: str, api_key: str = None) -> pd.DataFrame:
        """Get economic data from FRED API"""
        try:
            params = {
                'series_id': series_id,
                'file_type': 'json',
                'api_key': api_key or 'demo'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.fred_base_url, params=params) as response:
                    data = await response.json()
                    
                    observations = data.get('observations', [])
                    df = pd.DataFrame(observations)
                    
                    if not df.empty:
                        df['date'] = pd.to_datetime(df['date'])
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        
                    return df
        except Exception as e:
            logger.error(f"Error fetching FRED data for {series_id}: {e}")
            return pd.DataFrame()
    
    async def get_key_indicators(self) -> Dict:
        """Get key economic indicators"""
        indicators = {
            'gdp': 'GDP',
            'unemployment': 'UNRATE',
            'inflation': 'CPIAUCSL',
            'federal_funds': 'FEDFUNDS',
            'treasury_10y': 'GS10'
        }
        
        results = {}
        for name, series_id in indicators.items():
            data = await self.get_fred_data(series_id)
            if not data.empty:
                latest = data.iloc[-1]
                results[name] = {
                    'value': latest['value'],
                    'date': latest['date']
                }
        
        return results

# Factory function to create data provider instances
def create_data_provider(provider_type: str = "aggregator") -> FinancialDataProvider:
    """Factory function to create data provider instances"""
    providers = {
        'yahoo': YahooFinanceProvider,
        'alpha_vantage': AlphaVantageProvider,
        'polygon': PolygonProvider,
        'aggregator': MarketDataAggregator
    }
    
    if provider_type in providers:
        return providers[provider_type]()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")