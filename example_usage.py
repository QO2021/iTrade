#!/usr/bin/env python3
"""
iTrade Usage Examples
Demonstrates how to use the iTrade platform programmatically
"""

import asyncio
import pandas as pd
from datetime import datetime

# Import iTrade modules
from data_sources.financial_data import MarketDataAggregator, YahooFinanceProvider
from analytics.greeks_calculator import GreeksCalculator, compare_greeks
from strategies.trading_strategies import (
    create_strategy, create_default_ensemble,
    MomentumStrategy, MeanReversionStrategy
)
from data_sources.news_scraper import get_market_sentiment, get_ticker_sentiment

async def example_data_fetching():
    """Example: Fetching real-time and historical data"""
    print("=" * 50)
    print("1. Data Fetching Example")
    print("=" * 50)
    
    # Initialize data provider
    data_provider = MarketDataAggregator()
    
    # Get real-time price for Apple
    symbol = "AAPL"
    print(f"\nFetching real-time data for {symbol}...")
    
    real_time_data = await data_provider.get_best_price(symbol)
    if real_time_data:
        print(f"Current Price: ${real_time_data.get('price', 'N/A')}")
        print(f"Volume: {real_time_data.get('volume', 'N/A'):,}")
        print(f"Source: {real_time_data.get('source', 'N/A')}")
    
    # Get comprehensive data
    print(f"\nFetching comprehensive data for {symbol}...")
    comprehensive_data = await data_provider.get_comprehensive_data(symbol)
    
    if not comprehensive_data['historical'].empty:
        historical = comprehensive_data['historical']
        print(f"Historical data: {len(historical)} days")
        print(f"Price range: ${historical['Close'].min():.2f} - ${historical['Close'].max():.2f}")

def example_greeks_calculation():
    """Example: Calculating Greeks and risk metrics"""
    print("\n" + "=" * 50)
    print("2. Greeks Calculation Example")
    print("=" * 50)
    
    # Initialize Greeks calculator
    calculator = GreeksCalculator()
    
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    
    print(f"\nCalculating Greeks for: {', '.join(symbols)}")
    
    # Calculate Greeks for each symbol
    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        try:
            greeks = calculator.calculate_greeks(symbol)
            
            print(f"Alpha: {greeks.alpha:.4f}")
            print(f"Beta: {greeks.beta:.4f}")
            print(f"Sigma (Volatility): {greeks.sigma:.4f}")
            print(f"Sharpe Ratio: {greeks.sharpe_ratio:.4f}")
            print(f"Max Drawdown: {greeks.max_drawdown:.4f}")
            
        except Exception as e:
            print(f"Error calculating Greeks for {symbol}: {e}")
    
    # Compare Greeks across symbols
    print(f"\nComparing Greeks across symbols...")
    try:
        comparison_df = compare_greeks(symbols[:3])  # Compare first 3 symbols
        print(comparison_df[['alpha', 'beta', 'sigma', 'sharpe_ratio']].round(4))
    except Exception as e:
        print(f"Error in comparison: {e}")

async def example_trading_strategies():
    """Example: Using different trading strategies"""
    print("\n" + "=" * 50)
    print("3. Trading Strategies Example")
    print("=" * 50)
    
    symbol = "AAPL"
    
    # Get historical data
    yahoo_provider = YahooFinanceProvider()
    historical_data = await yahoo_provider.get_historical_data(symbol, "6m")
    
    if historical_data.empty:
        print(f"No historical data available for {symbol}")
        return
    
    print(f"\nTesting strategies on {symbol} with {len(historical_data)} days of data")
    
    # Test different strategies
    strategies = [
        ("momentum", "Momentum Strategy"),
        ("mean_reversion", "Mean Reversion Strategy"),
        ("breakout", "Breakout Strategy")
    ]
    
    for strategy_type, strategy_name in strategies:
        print(f"\n--- {strategy_name} ---")
        try:
            strategy = create_strategy(strategy_type)
            signal = await strategy.generate_signal(symbol, historical_data)
            
            print(f"Signal: {signal.signal_type.value.upper()}")
            print(f"Confidence: {signal.confidence:.2f}")
            print(f"Reason: {signal.reason}")
            print(f"Price: ${signal.price:.2f}")
            
        except Exception as e:
            print(f"Error with {strategy_name}: {e}")
    
    # Test ensemble strategy
    print(f"\n--- Ensemble Strategy ---")
    try:
        ensemble = create_default_ensemble()
        ensemble_signal = await ensemble.generate_ensemble_signal(symbol, historical_data)
        
        print(f"Ensemble Signal: {ensemble_signal.signal_type.value.upper()}")
        print(f"Confidence: {ensemble_signal.confidence:.2f}")
        print(f"Reason: {ensemble_signal.reason}")
        
    except Exception as e:
        print(f"Error with ensemble strategy: {e}")

async def example_sentiment_analysis():
    """Example: News sentiment analysis"""
    print("\n" + "=" * 50)
    print("4. Sentiment Analysis Example")
    print("=" * 50)
    
    print("\nFetching market sentiment...")
    try:
        market_sentiment = await get_market_sentiment()
        
        if market_sentiment.get('market_sentiment'):
            sentiment = market_sentiment['market_sentiment']
            print(f"Overall Market Sentiment: {sentiment.get('overall_sentiment', 0):.3f}")
            print(f"Confidence: {sentiment.get('confidence', 0):.2f}")
            print(f"Articles Analyzed: {sentiment.get('article_count', 0)}")
            
            # Show top mentioned tickers
            top_tickers = market_sentiment.get('top_tickers', [])
            if top_tickers:
                print(f"\nTop Mentioned Tickers:")
                for ticker, mentions in top_tickers[:5]:
                    print(f"  {ticker}: {mentions} mentions")
        
    except Exception as e:
        print(f"Error fetching market sentiment: {e}")
    
    # Get sentiment for specific ticker
    symbol = "AAPL"
    print(f"\nFetching sentiment for {symbol}...")
    try:
        ticker_sentiment = await get_ticker_sentiment(symbol)
        
        if ticker_sentiment.get('sentiment'):
            sentiment = ticker_sentiment['sentiment']
            print(f"{symbol} Sentiment: {sentiment.get('overall_sentiment', 0):.3f}")
            print(f"Articles Found: {len(ticker_sentiment.get('articles', []))}")
            
    except Exception as e:
        print(f"Error fetching {symbol} sentiment: {e}")

def example_risk_management():
    """Example: Risk management calculations"""
    print("\n" + "=" * 50)
    print("5. Risk Management Example")
    print("=" * 50)
    
    # Example portfolio positions
    portfolio_positions = [
        {"symbol": "AAPL", "quantity": 100, "market_value": 15000},
        {"symbol": "GOOGL", "quantity": 50, "market_value": 12000},
        {"symbol": "MSFT", "quantity": 75, "market_value": 8000},
    ]
    
    print("Portfolio Positions:")
    total_value = sum(pos["market_value"] for pos in portfolio_positions)
    
    for position in portfolio_positions:
        weight = position["market_value"] / total_value
        print(f"  {position['symbol']}: ${position['market_value']:,} ({weight:.1%})")
    
    print(f"\nTotal Portfolio Value: ${total_value:,}")
    
    # Calculate portfolio-level Greeks
    print("\nCalculating portfolio Greeks...")
    try:
        from analytics.greeks_calculator import PortfolioGreeksCalculator
        
        portfolio_calculator = PortfolioGreeksCalculator()
        portfolio_greeks = portfolio_calculator.calculate_portfolio_greeks(portfolio_positions)
        
        if portfolio_greeks:
            print(f"Portfolio Beta: {portfolio_greeks.get('beta', 0):.4f}")
            print(f"Portfolio Alpha: {portfolio_greeks.get('alpha', 0):.4f}")
            print(f"Portfolio Volatility: {portfolio_greeks.get('sigma', 0):.4f}")
            print(f"Portfolio Sharpe Ratio: {portfolio_greeks.get('sharpe_ratio', 0):.4f}")
        
    except Exception as e:
        print(f"Error calculating portfolio Greeks: {e}")

async def main():
    """Run all examples"""
    print("iTrade Platform Usage Examples")
    print("=" * 50)
    print("This script demonstrates various features of the iTrade platform")
    print("Note: Some examples require API keys to be configured in .env file")
    
    try:
        # Run examples
        await example_data_fetching()
        example_greeks_calculation()
        await example_trading_strategies()
        await example_sentiment_analysis()
        example_risk_management()
        
        print("\n" + "=" * 50)
        print("Examples completed!")
        print("=" * 50)
        print("\nTo use the full platform:")
        print("1. Set up your API keys in .env file")
        print("2. Run: python main.py")
        print("3. Open browser to http://localhost:8000")
        
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nError running examples: {e}")
        print("Make sure you have installed all dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())