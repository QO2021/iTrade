"""
News Scraping and Sentiment Analysis Module
Fetches financial news from multiple sources and performs sentiment analysis
"""

import asyncio
import aiohttp
import requests
import feedparser
import newspaper
from newspaper import Article
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from textblob import TextBlob
import nltk
import re
import logging
from urllib.parse import urljoin, urlparse

from config import Config

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class NewsSource:
    """Base class for news sources"""
    
    def __init__(self, name: str, base_url: str):
        self.name = name
        self.base_url = base_url
        
    async def fetch_articles(self, limit: int = 10) -> List[Dict]:
        """Fetch articles from the news source"""
        raise NotImplementedError

class RSSNewsSource(NewsSource):
    """RSS-based news source"""
    
    def __init__(self, name: str, base_url: str, rss_url: str):
        super().__init__(name, base_url)
        self.rss_url = rss_url
        
    async def fetch_articles(self, limit: int = 10) -> List[Dict]:
        """Fetch articles from RSS feed"""
        try:
            # Use requests in executor to avoid blocking
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(None, feedparser.parse, self.rss_url)
            
            articles = []
            for entry in feed.entries[:limit]:
                article = {
                    'title': entry.get('title', ''),
                    'url': entry.get('link', ''),
                    'published': self._parse_date(entry.get('published')),
                    'summary': entry.get('summary', ''),
                    'source': self.name
                }
                articles.append(article)
                
            return articles
        except Exception as e:
            logger.error(f"Error fetching RSS feed from {self.name}: {e}")
            return []
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object"""
        try:
            return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
        except:
            try:
                return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %Z')
            except:
                return datetime.now()

class WebScrapingNewsSource(NewsSource):
    """Web scraping-based news source"""
    
    def __init__(self, name: str, base_url: str, selectors: Dict[str, str]):
        super().__init__(name, base_url)
        self.selectors = selectors  # CSS selectors for different elements
        
    async def fetch_articles(self, limit: int = 10) -> List[Dict]:
        """Fetch articles by web scraping"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url) as response:
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            articles = []
            
            # Find article links
            article_elements = soup.select(self.selectors.get('article_links', 'a'))[:limit]
            
            for element in article_elements:
                url = element.get('href')
                if url and not url.startswith('http'):
                    url = urljoin(self.base_url, url)
                    
                title = element.get_text(strip=True)
                
                if url and title:
                    article = {
                        'title': title,
                        'url': url,
                        'published': datetime.now(),
                        'summary': '',
                        'source': self.name
                    }
                    articles.append(article)
                    
            return articles
        except Exception as e:
            logger.error(f"Error scraping {self.name}: {e}")
            return []

class NewsSourceManager:
    """Manages multiple news sources"""
    
    def __init__(self):
        self.sources = self._initialize_sources()
        
    def _initialize_sources(self) -> List[NewsSource]:
        """Initialize news sources with their configurations"""
        sources = []
        
        # RSS-based sources
        rss_sources = [
            {
                'name': 'Yahoo Finance',
                'base_url': 'https://finance.yahoo.com',
                'rss_url': 'https://finance.yahoo.com/news/rssindex'
            },
            {
                'name': 'Reuters Finance',
                'base_url': 'https://www.reuters.com/finance',
                'rss_url': 'https://feeds.reuters.com/reuters/businessNews'
            },
            {
                'name': 'MarketWatch',
                'base_url': 'https://www.marketwatch.com',
                'rss_url': 'http://feeds.marketwatch.com/marketwatch/marketpulse'
            }
        ]
        
        for config in rss_sources:
            sources.append(RSSNewsSource(**config))
        
        # Web scraping sources
        scraping_sources = [
            {
                'name': 'Bloomberg',
                'base_url': 'https://www.bloomberg.com/markets',
                'selectors': {
                    'article_links': 'a[data-module="HeadlineLink"]'
                }
            },
            {
                'name': 'CNBC',
                'base_url': 'https://www.cnbc.com/finance/',
                'selectors': {
                    'article_links': '.Card-titleContainer a'
                }
            },
            {
                'name': 'Financial Times',
                'base_url': 'https://www.ft.com/markets',
                'selectors': {
                    'article_links': '.o-teaser__heading a'
                }
            }
        ]
        
        for config in scraping_sources:
            sources.append(WebScrapingNewsSource(**config))
            
        return sources
    
    async def fetch_all_articles(self, limit_per_source: int = 5) -> List[Dict]:
        """Fetch articles from all sources"""
        tasks = []
        for source in self.sources:
            tasks.append(source.fetch_articles(limit_per_source))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_articles = []
        for articles in results:
            if isinstance(articles, list):
                all_articles.extend(articles)
                
        return all_articles

class ArticleProcessor:
    """Processes and enriches news articles"""
    
    def __init__(self):
        self.financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'stock', 'share', 'market', 'trading', 'investment', 'analyst',
            'upgrade', 'downgrade', 'buy', 'sell', 'hold', 'target price',
            'merger', 'acquisition', 'ipo', 'dividend', 'split', 'buyback'
        ]
        
    async def process_article(self, article_data: Dict) -> Dict:
        """Process a single article to extract content and metadata"""
        try:
            # Download and parse the full article
            article = Article(article_data['url'])
            await asyncio.get_event_loop().run_in_executor(None, article.download)
            await asyncio.get_event_loop().run_in_executor(None, article.parse)
            
            # Extract content
            article_data['content'] = article.text
            article_data['authors'] = article.authors
            article_data['keywords'] = article.keywords
            article_data['meta_keywords'] = article.meta_keywords
            
            # Perform sentiment analysis
            sentiment = self.analyze_sentiment(article.text)
            article_data.update(sentiment)
            
            # Extract mentioned tickers
            tickers = self.extract_tickers(article.text)
            article_data['mentioned_tickers'] = tickers
            
            # Calculate relevance score
            article_data['relevance_score'] = self.calculate_relevance(article.text)
            
            return article_data
            
        except Exception as e:
            logger.error(f"Error processing article {article_data.get('url', '')}: {e}")
            return article_data
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text using multiple methods"""
        if not text:
            return {'sentiment_score': 0.0, 'sentiment_label': 'neutral'}
            
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Determine sentiment label
        if polarity > 0.1:
            label = 'positive'
        elif polarity < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
            
        return {
            'sentiment_score': polarity,
            'sentiment_magnitude': subjectivity,
            'sentiment_label': label
        }
    
    def extract_tickers(self, text: str) -> List[str]:
        """Extract stock tickers from text"""
        # Pattern to match stock tickers (1-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)
        
        # Filter out common words that might match the pattern
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN',
            'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'HAS', 'HIS', 'WHO',
            'DID', 'GET', 'MAY', 'HIM', 'OLD', 'SEE', 'TWO', 'WAY', 'ITS',
            'NOW', 'NEW', 'USE', 'MAN', 'DAY', 'GOT', 'HOW', 'PUT', 'END',
            'WHY', 'LET', 'SAY', 'SHE', 'OWN', 'OFF', 'SET', 'RUN', 'TOO',
            'ANY', 'TOP', 'BIG', 'BAD', 'LOW', 'WIN', 'BUY', 'TRY', 'ASK'
        }
        
        # Filter tickers
        tickers = [ticker for ticker in potential_tickers 
                  if ticker not in common_words and len(ticker) <= 5]
        
        # Remove duplicates and return
        return list(set(tickers))
    
    def calculate_relevance(self, text: str) -> float:
        """Calculate relevance score based on financial keywords"""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in self.financial_keywords 
                          if keyword in text_lower)
        
        # Normalize by text length
        words = len(text.split())
        if words == 0:
            return 0.0
            
        relevance = (keyword_count / words) * 100
        return min(relevance, 10.0)  # Cap at 10.0

class SentimentAggregator:
    """Aggregates sentiment across multiple articles"""
    
    def __init__(self):
        self.processor = ArticleProcessor()
        
    def calculate_market_sentiment(self, articles: List[Dict]) -> Dict:
        """Calculate overall market sentiment from articles"""
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'confidence': 0.0,
                'article_count': 0
            }
        
        # Filter articles with sentiment data
        articles_with_sentiment = [
            article for article in articles 
            if 'sentiment_score' in article
        ]
        
        if not articles_with_sentiment:
            return {
                'overall_sentiment': 0.0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'confidence': 0.0,
                'article_count': 0
            }
        
        # Calculate weighted sentiment
        total_weight = 0
        weighted_sentiment = 0
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for article in articles_with_sentiment:
            weight = article.get('relevance_score', 1.0)
            sentiment = article.get('sentiment_score', 0.0)
            label = article.get('sentiment_label', 'neutral')
            
            weighted_sentiment += sentiment * weight
            total_weight += weight
            sentiment_counts[label] += 1
        
        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0.0
        
        # Calculate confidence based on agreement
        total_articles = len(articles_with_sentiment)
        max_count = max(sentiment_counts.values())
        confidence = max_count / total_articles if total_articles > 0 else 0.0
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': sentiment_counts,
            'confidence': confidence,
            'article_count': total_articles,
            'timestamp': datetime.now()
        }
    
    def get_ticker_sentiment(self, articles: List[Dict], ticker: str) -> Dict:
        """Get sentiment specific to a ticker"""
        ticker_articles = [
            article for article in articles 
            if ticker in article.get('mentioned_tickers', [])
        ]
        
        return self.calculate_market_sentiment(ticker_articles)

class NewsAnalyzer:
    """Main news analysis orchestrator"""
    
    def __init__(self):
        self.source_manager = NewsSourceManager()
        self.processor = ArticleProcessor()
        self.sentiment_aggregator = SentimentAggregator()
        
    async def analyze_market_news(self, limit_per_source: int = 5) -> Dict:
        """Analyze current market news and sentiment"""
        # Fetch articles from all sources
        articles = await self.source_manager.fetch_all_articles(limit_per_source)
        
        if not articles:
            return {
                'articles': [],
                'market_sentiment': self.sentiment_aggregator.calculate_market_sentiment([]),
                'top_tickers': [],
                'error': 'No articles found'
            }
        
        # Process articles to extract content and sentiment
        processed_articles = []
        for article in articles:
            processed_article = await self.processor.process_article(article)
            processed_articles.append(processed_article)
        
        # Calculate overall market sentiment
        market_sentiment = self.sentiment_aggregator.calculate_market_sentiment(processed_articles)
        
        # Find most mentioned tickers
        ticker_mentions = {}
        for article in processed_articles:
            for ticker in article.get('mentioned_tickers', []):
                ticker_mentions[ticker] = ticker_mentions.get(ticker, 0) + 1
        
        top_tickers = sorted(ticker_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'articles': processed_articles,
            'market_sentiment': market_sentiment,
            'top_tickers': top_tickers,
            'analysis_timestamp': datetime.now()
        }
    
    async def get_ticker_analysis(self, ticker: str, limit_per_source: int = 10) -> Dict:
        """Get news analysis specific to a ticker"""
        # Fetch more articles to increase chances of finding ticker mentions
        articles = await self.source_manager.fetch_all_articles(limit_per_source)
        
        # Process articles
        processed_articles = []
        for article in articles:
            processed_article = await self.processor.process_article(article)
            processed_articles.append(processed_article)
        
        # Get ticker-specific sentiment
        ticker_sentiment = self.sentiment_aggregator.get_ticker_sentiment(processed_articles, ticker)
        
        # Filter articles mentioning the ticker
        ticker_articles = [
            article for article in processed_articles 
            if ticker in article.get('mentioned_tickers', [])
        ]
        
        return {
            'ticker': ticker,
            'articles': ticker_articles,
            'sentiment': ticker_sentiment,
            'analysis_timestamp': datetime.now()
        }

# Utility functions
async def get_market_sentiment() -> Dict:
    """Quick function to get current market sentiment"""
    analyzer = NewsAnalyzer()
    return await analyzer.analyze_market_news()

async def get_ticker_sentiment(ticker: str) -> Dict:
    """Quick function to get sentiment for a specific ticker"""
    analyzer = NewsAnalyzer()
    return await analyzer.get_ticker_analysis(ticker)