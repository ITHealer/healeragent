"""
Base class for all fundamental analysis tools

Provides:
- Shared FMP API fetching logic
- Redis cache integration
- Common error handling
- TTL configuration
"""

import httpx
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from abc import ABC

class FinancialDataFetcher(ABC):
    """
    Base class for fetching financial data from FMP with cache
    
    Handles:
    - API requests with retry
    - Redis caching
    - Error handling
    - TTL management
    """
    
    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    
    # Cache TTLs (in seconds)
    CACHE_TTL_ANNUAL = 86400  # 24 hours
    CACHE_TTL_QUARTERLY = 21600  # 6 hours
    
    def __init__(self, api_key: str, logger: logging.Logger):
        """
        Initialize fetcher
        
        Args:
            api_key: FMP API key
            logger: Logger instance
        """
        self.api_key = api_key
        self.logger = logger
    
    async def fetch_from_fmp(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Optional[List[Dict]]:
        """
        Fetch data from FMP API
        
        Args:
            endpoint: API endpoint (e.g., "income-statement")
            params: Query parameters
            
        Returns:
            List of data or None if error
        """
        # Add API key to params
        params["apikey"] = self.api_key
        
        # Build URL
        url = f"{self.FMP_BASE_URL}/v3/{endpoint}"
        
        self.logger.debug(f"[FMP] Fetching from {url} with params: {params}")
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data or len(data) == 0:
                    self.logger.warning(f"[FMP] No data returned from {endpoint}")
                    return None
                
                self.logger.info(
                    f"[FMP] Successfully fetched {len(data)} records from {endpoint}"
                )
                
                return data
                
        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"[FMP] HTTP error {e.response.status_code}: {e.response.text}"
            )
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Request error for {endpoint}: {e}")
            return None
    
    def get_cache_ttl(self, period: str) -> int:
        """Get appropriate cache TTL based on period"""
        if period == "annual":
            return self.CACHE_TTL_ANNUAL
        else:
            return self.CACHE_TTL_QUARTERLY
    
    def build_cache_key(
        self,
        tool_name: str,
        symbol: str,
        period: str,
        limit: int
    ) -> str:
        """Build standardized cache key"""
        return f"{tool_name}_{symbol.upper()}_{period}_limit_{limit}"