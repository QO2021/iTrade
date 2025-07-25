"""
Database Connection and Initialization Module
Handles database setup, connections, and operations for iTrade platform
"""

import asyncio
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import StaticPool
import logging
from typing import AsyncGenerator

from config import Config
from models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database connection and session manager"""
    
    def __init__(self):
        self.config = Config()
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        
    def init_sync_db(self):
        """Initialize synchronous database connection"""
        try:
            # For demo purposes, use SQLite if PostgreSQL URL not provided
            if not self.config.DATABASE_URL or 'postgresql' not in self.config.DATABASE_URL:
                db_url = "sqlite:///./itrade.db"
                self.engine = create_engine(
                    db_url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool
                )
            else:
                self.engine = create_engine(self.config.DATABASE_URL)
            
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def init_async_db(self):
        """Initialize asynchronous database connection"""
        try:
            # For demo purposes, use SQLite if PostgreSQL URL not provided
            if not self.config.DATABASE_URL or 'postgresql' not in self.config.DATABASE_URL:
                db_url = "sqlite+aiosqlite:///./itrade.db"
            else:
                # Convert postgresql:// to postgresql+asyncpg://
                db_url = self.config.DATABASE_URL.replace('postgresql://', 'postgresql+asyncpg://')
            
            self.async_engine = create_async_engine(db_url)
            
            self.AsyncSessionLocal = sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create tables
            async with self.async_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Async database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing async database: {e}")
            raise
    
    def get_db_session(self):
        """Get synchronous database session"""
        if not self.SessionLocal:
            self.init_sync_db()
        
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    async def get_async_db_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get asynchronous database session"""
        if not self.AsyncSessionLocal:
            await self.init_async_db()
        
        async with self.AsyncSessionLocal() as session:
            try:
                yield session
            finally:
                await session.close()
    
    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
        if self.engine:
            self.engine.dispose()

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions
def get_db():
    """Get database session (for dependency injection)"""
    return db_manager.get_db_session()

async def get_async_db():
    """Get async database session (for dependency injection)"""
    async for session in db_manager.get_async_db_session():
        yield session

async def init_database():
    """Initialize database on startup"""
    await db_manager.init_async_db()

async def close_database():
    """Close database on shutdown"""
    await db_manager.close()