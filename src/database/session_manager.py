"""
Session Manager - Database connection and session management
Handles SQLAlchemy sessions for all database operations
"""

from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool, NullPool
import os

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from urllib.parse import quote_plus as urlquote
from src.utils.config_loader import ConfigReaderInstance

class SessionManager(LoggerMixin):
    """
    Manages database connections and sessions
    Uses SQLAlchemy with connection pooling
    """
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize SessionManager
        
        Args:
            database_url: Database connection string (optional)
        """
        super().__init__()
        
        # Get database URL from settings or environment
        self.database_url = database_url or self._get_database_url()
        
        # Create engine with connection pooling
        self.engine = self._create_engine()
        
        # Create session factory
        self.SessionLocal = scoped_session(
            sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
        )
        
        self.logger.info(f"[SESSION MANAGER] Initialized with database: {self._mask_db_url(self.database_url)}")
    
    def _get_database_url(self) -> str:
        """
        Get database URL from settings or environment variables
        
        Returns:
            Database connection string
        """
        
        try:
            db_config = ConfigReaderInstance.yaml.read_config_from_file(
                settings.DATABASE_CONFIG_FILENAME
            )
            postgres_config = db_config.get('POSTGRES')
            
            if postgres_config:
                return 'postgresql://{}:{}@{}:{}/{}'.format(
                    postgres_config['USER'],
                    urlquote(postgres_config['PASSWORD']),
                    postgres_config['HOST'],
                    postgres_config['PORT'],
                    postgres_config['DATABASE_NAME']
                )
        except Exception as e:
            self.logger.warning(f"[SESSION MANAGER] Failed to load config from file: {e}")
    
    def _create_engine(self):
        """
        Create SQLAlchemy engine with appropriate pooling
        Uses config from database_config.yaml if available
        
        Returns:
            SQLAlchemy Engine instance
        """
        # Load pool settings from config if available
        pool_size = 10
        max_overflow = 20
        pool_pre_ping = True
        echo = False
        
        try:
            db_config = ConfigReaderInstance.yaml.read_config_from_file(
                settings.DATABASE_CONFIG_FILENAME
            )
            postgres_config = db_config.get('POSTGRES', {})
            
            # Get pool settings from config if exists
            pool_size = int(postgres_config.get('POOL_SIZE', 10))
            max_overflow = int(postgres_config.get('MAX_OVERFLOW', 20))
            echo = postgres_config.get('SQLALCHEMY_ECHO', 'false').lower() == 'true'
            
        except Exception as e:
            self.logger.warning(f"[SESSION MANAGER] Using default pool settings: {e}")
        
        # Determine pool class based on environment
        env = os.getenv('ENV', 'development')
        if env == 'production':
            pool_class = QueuePool
        else:
            pool_class = QueuePool
        
        engine = create_engine(
            self.database_url,
            poolclass=pool_class,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,  # Test connections before using
            echo=False  # Set to True for SQL query logging
        )
        
        self.logger.info(f"[SESSION MANAGER] Engine created with pool_size={pool_size}, max_overflow={max_overflow}")
        
        return engine
    
    @contextmanager
    def create_session(self) -> Generator[Session, None, None]:
        """
        Create a database session with proper cleanup
        
        Yields:
            SQLAlchemy Session instance
            
        Example:
            with session_manager.create_session() as db:
                result = db.query(Model).all()
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"[SESSION MANAGER] Database error: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """
        Get a new session instance (manual management)
        Caller is responsible for closing the session
        
        Returns:
            SQLAlchemy Session instance
        """
        return self.SessionLocal()
    
    @contextmanager
    def transaction(self) -> Generator[Session, None, None]:
        """
        Create a transactional session
        Auto-commits on success, rolls back on failure
        
        Yields:
            SQLAlchemy Session instance
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
            self.logger.debug("[SESSION MANAGER] Transaction committed")
        except Exception as e:
            session.rollback()
            self.logger.error(f"[SESSION MANAGER] Transaction rolled back: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """
        Close all database connections
        Should be called on application shutdown
        """
        try:
            self.SessionLocal.remove()
            self.engine.dispose()
            self.logger.info("[SESSION MANAGER] All connections closed")
        except Exception as e:
            self.logger.error(f"[SESSION MANAGER] Error closing connections: {e}")
    
    def _mask_db_url(self, url: str) -> str:
        """
        Mask sensitive parts of database URL for logging
        
        Args:
            url: Database URL
            
        Returns:
            Masked URL safe for logging
        """
        if '@' in url:
            # postgresql://user:pass@host:port/db
            parts = url.split('@')
            protocol = parts[0].split('//')[0]
            host_db = parts[1] if len(parts) > 1 else ''
            return f"{protocol}//***:***@{host_db}"
        return url
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.create_session() as db:
                # Execute simple query
                db.execute("SELECT 1")
                self.logger.info("[SESSION MANAGER] Database connection test successful")
                return True
        except Exception as e:
            self.logger.error(f"[SESSION MANAGER] Database connection test failed: {e}")
            return False
    
    def get_connection_info(self) -> dict:
        """
        Get current connection pool statistics
        
        Returns:
            Dictionary with connection pool info
        """
        pool = self.engine.pool
        return {
            "size": pool.size() if hasattr(pool, 'size') else 'N/A',
            "checked_in": pool.checkedin() if hasattr(pool, 'checkedin') else 'N/A',
            "checked_out": pool.checkedout() if hasattr(pool, 'checkedout') else 'N/A',
            "overflow": pool.overflow() if hasattr(pool, 'overflow') else 'N/A',
            "total": pool.total() if hasattr(pool, 'total') else 'N/A'
        }


# Global instance (singleton pattern)
_session_manager_instance = None

def get_session_manager() -> SessionManager:
    """
    Get global SessionManager instance (singleton)
    
    Returns:
        SessionManager instance
    """
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager()
    return _session_manager_instance


# Dependency for FastAPI
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions
    
    Usage:
        @router.get("/items")
        async def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    session_manager = get_session_manager()
    with session_manager.create_session() as session:
        yield session