from datetime import datetime
from sqlalchemy.exc import IntegrityError
from typing import List, Dict, Any

from src.database.models.stock_schemas import NewsSchema
from src.database import get_postgres_db
from src.utils.logger.custom_logging import LoggerMixin

class NewsRepository(LoggerMixin):
    """Repository for saving and retrieving news articles"""
    def __init__(self):
        super().__init__()
        self.db = get_postgres_db()


    async def save_articles(self, articles: List[Dict[str, Any]]) -> bool:
        try:
            with self.db.session_scope() as session:
                for article_data in articles:
                    # Validate with Pydantic
                    news_item = NewsSchema(**article_data)
                    
                    # Convert to SQLAlchemy model
                    db_news = NewsSchema(
                        title=news_item.title,
                        url=news_item.url,
                        content=news_item.content,
                        author=news_item.author,
                        created_at=news_item.created_at or datetime.utcnow()
                    )
                    session.add(db_news)
            return True
        except Exception as e:
            self.logger.error(f"Error saving news: {str(e)}")
            return False
            
    def _parse_datetime(self, date_str: str) -> datetime:
        """
        Parse a datetime string to datetime object, with fallback to current time.
        
        Args:
            date_str: Datetime string to parse
            
        Returns:
            datetime: Parsed datetime or current time
        """
        if not date_str:
            return datetime.utcnow()
            
        try:
            # Try different formats
            for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If no format matches, use current time
            return datetime.utcnow()
        except Exception:
            return datetime.utcnow()