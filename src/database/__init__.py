from typing import Optional

# Type hints for lazy-loaded instances  
from .postgresql_connection import PostgreSQLConnection
from .mysql_frontend_connection import MySQLFrontendConnection
from .mysql_news_connection import MySQLNewsConnection

# Global instances (lazy initialized)
_postgres_db: Optional[PostgreSQLConnection] = None
_frontend_db: Optional[MySQLFrontendConnection] = None
_news_db: Optional[MySQLNewsConnection] = None


def get_postgres_db() -> PostgreSQLConnection:
    """Get PostgreSQL database connection (lazy singleton)"""
    global _postgres_db
    if _postgres_db is None:
        _postgres_db = PostgreSQLConnection()
    return _postgres_db


def get_frontend_db() -> MySQLFrontendConnection:
    """Get MySQL frontend database connection (lazy singleton)"""
    global _frontend_db
    if _frontend_db is None:
        _frontend_db = MySQLFrontendConnection()
    return _frontend_db


def get_news_db() -> MySQLNewsConnection:
    """Get MySQL news database connection (lazy singleton)"""
    global _news_db
    if _news_db is None:
        _news_db = MySQLNewsConnection()  
    return _news_db


# Dependencies for get_db FastAPI dependency injection
def get_db_dependency():
    """FastAPI dependency for getting PostgreSQL session"""
    session = get_postgres_db().get_session()
    try:
        yield session
    finally:
        session.close()


def get_connection_dependency():
    """FastAPI dependency for getting PostgreSQL connection"""
    conn = get_postgres_db().get_connection()
    try:
        yield conn
    finally:
        conn.close()


# Export public API
__all__ = [
    'get_postgres_db',
    'get_frontend_db', 
    'get_news_db',
    'get_db_dependency',
    'get_connection_dependency'
]