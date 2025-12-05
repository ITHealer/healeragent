import httpx
from pydantic import ValidationError
from typing import Optional, List, Dict, Any
from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin

from src.models.company_search import (
    StockSymbolSearchItem,
    CompanyNameSearchItem,
    CIKSearchItem,
    CUSIPSearchItem,
    ISINSearchItem,
    StockScreenerItem,
    ExchangeVariantItem,
    StockScreenerFilters
)


class CompanySearchService(LoggerMixin):
    """
    Company Search operations.
    
    Features:
    - Symbol, name, CIK, CUSIP, ISIN search
    - Stock screener with multiple filters
    - Exchange variants lookup
    - Built-in error handling & logging
    """
    
    def __init__(self):
        """Initialize service with FMP configuration."""
        super().__init__()
        
        self.base_url = settings.FMP_URL_STABLE or "https://financialmodelingprep.com/stable"
        self.api_key = settings.FMP_API_KEY
        self.timeout = httpx.Timeout(30.0, connect=10.0) 
        
    async def _make_fmp_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Internal method to make FMP API requests.
        
        Args:
            endpoint: FMP endpoint path (e.g., "search-symbol")
            params: Query parameters dict
            
        Returns:
            List of dictionaries if successful, None if error
        """
        if not self.api_key or self.api_key == "YOUR_FMP_API_KEY_PLACEHOLDER":
            self.logger.error("FMP API key is not configured")
            return None
            
        # Build URL
        url = f"{self.base_url}/{endpoint}"
        
        # Add API key to params
        if params is None:
            params = {}
        params["apikey"] = self.api_key
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                self.logger.debug(f"FMP Request: {url} with params: {params}")
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # FMP returns list or dict with 'Error Message' key
                if isinstance(data, dict) and "Error Message" in data:
                    self.logger.error(f"FMP API Error: {data['Error Message']}")
                    return None
                    
                if not isinstance(data, list):
                    self.logger.warning(f"Unexpected FMP response format: {type(data)}")
                    return None
                    
                self.logger.debug(f"FMP Response: {len(data)} items returned")
                return data
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"HTTP error calling FMP API: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.TimeoutException:
            self.logger.error(f"Timeout calling FMP API endpoint: {endpoint}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error calling FMP API: {str(e)}", exc_info=True)
            return None
    
    # ========================================================================
    # 1. STOCK SYMBOL SEARCH
    # ========================================================================
    async def search_by_symbol(
        self,
        query: str,
        limit: Optional[int] = None,
        exchange: Optional[str] = None
    ) -> Optional[List[StockSymbolSearchItem]]:
        """
        Search stocks by symbol or partial symbol match.
        
        Args:
            query: Symbol search query (e.g., "AAPL", "AA")
            limit: Max number of results (optional)
            exchange: Filter by exchange (e.g., "NASDAQ", "NYSE")
            
        Returns:
            List of StockSymbolSearchItem or None if error
            
        Example:
            results = await service.search_by_symbol("AAPL", limit=10)
        """
        params = {"query": query}
        if limit:
            params["limit"] = limit
        if exchange:
            params["exchange"] = exchange
            
        data = await self._make_fmp_request("search-symbol", params)
        
        if data is None:
            return None
            
        try:
            # Transform list of dicts to list of StockSymbolSearchItem objects to quickly access fields. 
            # Example: obj.name, obj.symbol,...
            # ** is called "unpacking"
            return [StockSymbolSearchItem(**item) for item in data]
        except Exception as e:
            self.logger.error(f"Error parsing symbol search results: {str(e)}")
            return None
    
    # ========================================================================
    # 2. COMPANY NAME SEARCH
    # ========================================================================
    async def search_by_name(
        self,
        query: str,
        limit: Optional[int] = None,
        exchange: Optional[str] = None
    ) -> Optional[List[CompanyNameSearchItem]]:
        """
        Search companies by name or partial name match.
        
        Args:
            query: Company name query (e.g., "Apple", "Microsoft")
            limit: Max number of results
            exchange: Filter by exchange
            
        Returns:
            List of CompanyNameSearchItem or None if error
            
        Example:
            results = await service.search_by_name("Apple Inc", limit=5)
        """
        params = {"query": query}
        if limit:
            params["limit"] = limit
        if exchange:
            params["exchange"] = exchange
            
        data = await self._make_fmp_request("search-name", params)
        
        if data is None:
            return None
            
        try:
            return [CompanyNameSearchItem(**item) for item in data]
        except Exception as e:
            self.logger.error(f"Error parsing name search results: {str(e)}")
            return None
    
    # ========================================================================
    # 3. CIK SEARCH
    # ========================================================================
    async def search_by_cik(self, cik: str) -> Optional[List[CIKSearchItem]]:
        """
        Search company by CIK (Central Index Key).
        
        Args:
            cik: CIK number (e.g., "320193" or "0000320193")
            
        Returns:
            List of CIKSearchItem (usually 1 item) or None if error
            
        Example:
            results = await service.search_by_cik("320193")  # Apple
        """
        # Remove leading zeros if present for consistency
        cik_clean = cik.lstrip("0") if cik else ""
        
        if not cik_clean:
            self.logger.warning("Empty CIK provided")
            return None
            
        params = {"cik": cik_clean}
        data = await self._make_fmp_request("search-cik", params)
        
        if data is None:
            return None
            
        try:
            return [CIKSearchItem(**item) for item in data]
        except Exception as e:
            self.logger.error(f"Error parsing CIK search results: {str(e)}")
            return None
    
    # ========================================================================
    # 4. CUSIP SEARCH
    # ========================================================================
    async def search_by_cusip(self, cusip: str) -> Optional[List[CUSIPSearchItem]]:
        """
        Search security by CUSIP identifier.
        
        Args:
            cusip: 9-character CUSIP code (e.g., "037833100")
            
        Returns:
            List of CUSIPSearchItem or None if error
            
        Example:
            results = await service.search_by_cusip("037833100")  # Apple
        """
        if not cusip or len(cusip) != 9:
            self.logger.warning(f"Invalid CUSIP format: {cusip} (must be 9 characters)")
            return None
            
        params = {"cusip": cusip}
        data = await self._make_fmp_request("search-cusip", params)
        
        if data is None:
            return None
            
        try:
            return [CUSIPSearchItem(**item) for item in data]
        except Exception as e:
            self.logger.error(f"Error parsing CUSIP search results: {str(e)}")
            return None
    
    # ========================================================================
    # 5. ISIN SEARCH
    # ========================================================================
    async def search_by_isin(self, isin: str) -> Optional[List[ISINSearchItem]]:
        """
        Search security by ISIN (International Securities Identification Number).
        
        Args:
            isin: 12-character ISIN code (e.g., "US0378331005")
            
        Returns:
            List of ISINSearchItem hoặc None nếu error
            
        Example:
            results = await service.search_by_isin("US0378331005")  # Apple
        """
        if not isin or len(isin) != 12:
            self.logger.warning(f"Invalid ISIN format: {isin} (must be 12 characters)")
            return None
            
        params = {"isin": isin}
        data = await self._make_fmp_request("search-isin", params)
        
        if data is None:
            return None
            
        try:
            return [ISINSearchItem(**item) for item in data]
        except Exception as e:
            self.logger.error(f"Error parsing ISIN search results: {str(e)}")
            return None
    
    # ========================================================================
    # 6. STOCK SCREENER
    # ========================================================================
    async def screen_stocks(
        self,
        filters: Optional[StockScreenerFilters] = None,
        **kwargs
    ) -> Optional[List[StockScreenerItem]]:
        """
        Screen stocks with multiple filter criteria.
        
        Args:
            filters: StockScreenerFilters object with filter params
            **kwargs: Individual filter parameters (alternative to filters object)
            
        Returns:
            List of StockScreenerItem matching criteria, or None if error
            
        Example:
            # Using filters object
            filters = StockScreenerFilters(
                market_cap_more_than=1000000000,
                sector="Technology",
                limit=50
            )
            results = await service.screen_stocks(filters=filters)
            
            # Using kwargs
            results = await service.screen_stocks(
                marketCapMoreThan=1000000000,
                sector="Technology",
                limit=50
            )
        """
        # Build params from filters object hoặc kwargs
        params = {}
        
        if filters:
            # Convert filters object to dict, excluding None values
            params = filters.model_dump(
                by_alias=True,
                exclude_none=True
            )
        else:
            # Use kwargs directly
            params = {k: v for k, v in kwargs.items() if v is not None}
        
        # Set default limit if not provided
        if "limit" not in params:
            params["limit"] = 100
            
        data = await self._make_fmp_request("company-screener", params)
        
        if data is None:
            return None
            
        try:
            return [StockScreenerItem(**item) for item in data]
        except Exception as e:
            self.logger.error(f"Error parsing screener results: {str(e)}")
            return None
    
    # ========================================================================
    # 7. EXCHANGE VARIANTS
    # ========================================================================
    async def search_exchange_variants(
        self,
        symbol: str
    ) -> Optional[List[ExchangeVariantItem]]:
        """
        Search all exchanges where a symbol is listed.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            
        Returns:
            List of ExchangeVariantItem showing all exchanges, or None if error
            
        Example:
            results = await service.search_exchange_variants("AAPL")
            # Returns: [AAPL on NASDAQ, AAPL on XETRA, etc.]
        """
        if not symbol:
            self.logger.warning("Empty symbol provided for exchange variants search")
            return None
            
        params = {"symbol": symbol.upper()}
        data = await self._make_fmp_request("search-exchange-variants", params)
        
        if data is None:
            return None
            
        try:
            # return [ExchangeVariantItem(**item) for item in data]
            validated_results = []
            for item in data:
                try:
                    validated_item = ExchangeVariantItem.model_validate(item) 
                    validated_results.append(validated_item)
                except ValidationError as e:
                    self.logger.error(f"Error parsing exchange variants results: {e.errors()}")
                    continue 
                    
            return validated_results
        except Exception as e:
            self.logger.error(f"Error parsing exchange variants results: {str(e)}")
            return None
    
    # ========================================================================
    # UNIFIED SEARCH (Convenience Method)
    # ========================================================================
    async def unified_search(
        self,
        query: str,
        search_type: str = "auto",
        limit: Optional[int] = 10
    ) -> Optional[List[Any]]:
        """
        Unified search method tự động detect search type.
        
        Args:
            query: Search query string
            search_type: "auto", "symbol", "name", "cik", "cusip", "isin"
            limit: Max results
            
        Returns:
            List of appropriate search items based on detected type
            
        Example:
            # Auto-detect based on query format
            results = await service.unified_search("AAPL")      # Symbol search
            results = await service.unified_search("Apple Inc") # Name search
            results = await service.unified_search("320193")    # CIK search
        """
        query = query.strip()
        
        if search_type == "auto":
            # Auto-detect search type based on query pattern
            if len(query) == 9 and query.isalnum():
                search_type = "cusip"
            elif len(query) == 12 and query[:2].isalpha() and query[2:].isdigit():
                search_type = "isin"
            elif query.isdigit():
                search_type = "cik"
            elif len(query) <= 5 and query.isupper():
                search_type = "symbol"
            else:
                search_type = "name"
        
        self.logger.info(f"Unified search: query='{query}', type={search_type}")
        
        # Route to appropriate search method
        if search_type == "symbol":
            return await self.search_by_symbol(query, limit=limit)
        elif search_type == "name":
            return await self.search_by_name(query, limit=limit)
        elif search_type == "cik":
            return await self.search_by_cik(query)
        elif search_type == "cusip":
            return await self.search_by_cusip(query)
        elif search_type == "isin":
            return await self.search_by_isin(query)
        else:
            self.logger.warning(f"Unknown search type: {search_type}")
            return None


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================
# Create singleton instance for use across application
_company_search_service_instance = None

def get_company_search_service() -> CompanySearchService:
    """
    Get singleton instance của CompanySearchService.
    
    Returns:
        CompanySearchService instance
    """
    global _company_search_service_instance
    if _company_search_service_instance is None:
        _company_search_service_instance = CompanySearchService()
    return _company_search_service_instance