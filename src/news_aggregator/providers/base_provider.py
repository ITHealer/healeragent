from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from ..schemas.unified_news import UnifiedNewsItem, NewsProvider
from ..schemas.fmp_news import NewsCategory

logger = logging.getLogger(__name__)


class BaseNewsProvider(ABC):
    """
    Abstract base class for news providers.
    
    Each provider must:
    1. Fetch news from their source
    2. Convert to UnifiedNewsItem format
    3. Handle errors gracefully
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    @abstractmethod
    def provider_name(self) -> NewsProvider:
        """Return the provider identifier"""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Provider priority for deduplication.
        Lower number = higher priority.
        When duplicate URLs found, keep the one from higher priority provider.
        """
        pass
    
    @abstractmethod
    async def fetch_news(
        self,
        categories: List[NewsCategory],
        page: int = 0,
        limit: int = 20,
        **kwargs
    ) -> List[UnifiedNewsItem]:
        """
        Fetch news and convert to unified format.
        
        Args:
            categories: List of news categories to fetch
            page: Page number for pagination
            limit: Number of items per page
            **kwargs: Provider-specific options
            
        Returns:
            List of UnifiedNewsItem
        """
        pass
    
    def _log_fetch_start(self, category: str, page: int, limit: int):
        """Log fetch operation start"""
        self.logger.info(f"[{self.provider_name.value}] Fetching {category} news - page={page}, limit={limit}")
    
    def _log_fetch_complete(self, category: str, count: int, time_ms: int):
        """Log fetch operation completion"""
        self.logger.info(f"[{self.provider_name.value}] Fetched {count} {category} articles in {time_ms}ms")
    
    def _log_fetch_error(self, category: str, error: str):
        """Log fetch error"""
        self.logger.error(f"[{self.provider_name.value}] Error fetching {category}: {error}")