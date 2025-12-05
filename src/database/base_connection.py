from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Generator, Any

from src.utils.logger.custom_logging import LoggerMixin


class DatabaseConnectionBase(LoggerMixin, ABC):
    """Base class for all database connections"""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def get_connection(self) -> Any:
        """Get database connection"""
        pass
        
    @contextmanager
    @abstractmethod
    def connection_scope(self) -> Generator[Any, None, None]:
        """Connection context manager"""
        pass


class MySQLConnectionBase(DatabaseConnectionBase):
    """Base class for MySQL connections"""
    
    def __init__(self, db_config_key: str):
        super().__init__()
        self.db_config_key = db_config_key
        self.connection_params = self._load_connection_params()
        
    def _load_connection_params(self) -> dict:
        """Load MySQL connection parameters"""
        import pymysql
        from src.utils.config_loader import ConfigReaderInstance
        mysql_config = ConfigReaderInstance.yaml.read_config_from_file("database_config.yaml")
        config = mysql_config.get(self.db_config_key, {})
        
        return {
            'host': config.get('HOST', 'localhost'),
            'port': int(config.get('PORT', 3306)),
            'user': config.get('USER', 'root'),
            'password': config.get('PASSWORD', ''),
            'db': config.get('DATABASE'),
            'charset': config.get('CHARSET', 'utf8mb4'),
            'cursorclass': pymysql.cursors.DictCursor
        }
    
    def get_connection(self):
        """Get MySQL connection"""
        import pymysql
        try:
            connection = pymysql.connect(**self.connection_params)
            return connection
        except Exception as e:
            self.logger.error(f"Failed to connect to MySQL: {str(e)}")
            raise
    
    @contextmanager
    def connection_scope(self) -> Generator:
        """MySQL connection context manager"""
        connection = self.get_connection()
        try:
            yield connection
            connection.commit()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Error in MySQL connection scope: {str(e)}")
            raise
        finally:
            connection.close()
    
    def execute_query(self, query: str, params: tuple = None) -> Any:
        """Execute query and return results"""
        with self.connection_scope() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchall()
                return result
    
    def execute_scalar(self, query: str, params: tuple = None) -> Any:
        """Execute query and return single value"""
        with self.connection_scope() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchone()
                return result[0] if result else None


class PostgreSQLConnectionBase(DatabaseConnectionBase):
    """Base class for PostgreSQL connection"""
    
    def __init__(self):
        super().__init__()
        self.connection_string = self._get_connection_string()
        self._engine = None
        self._SessionLocal = None
        
    def _get_connection_string(self) -> str:
        """Get PostgreSQL connection string"""
        from urllib.parse import quote_plus as urlquote
        from src.utils.config import settings
        from src.utils.config_loader import ConfigReaderInstance
        
        db_config = ConfigReaderInstance.yaml.read_config_from_file(settings.DATABASE_CONFIG_FILENAME)
        postgres_config = db_config.get('POSTGRES')
        
        return 'postgresql://{}:{}@{}:{}/{}'.format(
            postgres_config['USER'],
            urlquote(postgres_config['PASSWORD']),
            postgres_config['HOST'],
            postgres_config['PORT'],
            postgres_config['DATABASE_NAME']
        )
    
    @property
    def engine(self):
        """Lazy initialization of SQLAlchemy engine"""
        if self._engine is None:
            from sqlalchemy import create_engine
            from src.utils.config import settings
            from src.utils.config_loader import ConfigReaderInstance
            
            db_config = ConfigReaderInstance.yaml.read_config_from_file(settings.DATABASE_CONFIG_FILENAME)
            postgres_config = db_config.get('POSTGRES')
            
            self._engine = create_engine(
                self.connection_string, 
                echo=False, # postgres_config.get('SQLALCHEMY_ECHO', 'false').lower() == 'true',
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
            )
        return self._engine
    
    @property
    def SessionLocal(self):
        """Lazy initialization of session factory"""
        if self._SessionLocal is None:
            from sqlalchemy.orm import sessionmaker
            self._SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        return self._SessionLocal
    
    def get_connection(self):
        """Get direct psycopg2 connection"""
        import psycopg2
        return psycopg2.connect(self.connection_string)
    
    def get_session(self):
        """Get SQLAlchemy session"""
        return self.SessionLocal()
    
    @contextmanager
    def connection_scope(self) -> Generator:
        """Connection context manager for psycopg2"""
        connection = self.get_connection()
        try:
            yield connection
            connection.commit()
        except Exception as e:
            connection.rollback()
            self.logger.error(f"Transaction rolled back due to error: {str(e)}")
            raise
        finally:
            connection.close()
    
    @contextmanager
    def session_scope(self) -> Generator:
        """Session context manager for SQLAlchemy"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Session database error: {str(e)}")
            raise
        finally:
            session.close()