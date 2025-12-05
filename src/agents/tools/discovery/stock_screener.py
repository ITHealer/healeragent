"""
Stock Screener Tool - Find stocks by sector and criteria

Location: src/agents/tools/discovery/stock_screener.py
"""

import httpx
from typing import Optional, Any
from datetime import datetime

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
import logging
logger = logging.getLogger(__name__)


# ============================================================================
# FMP Sector Mapping
# ============================================================================

SECTOR_MAPPING = {
    # English variants
    "technology": "Technology",
    "tech": "Technology",
    "healthcare": "Healthcare",
    "health": "Healthcare",
    "financial": "Financial Services",
    "finance": "Financial Services",
    "financials": "Financial Services",
    "financial services": "Financial Services",
    "consumer cyclical": "Consumer Cyclical",
    "consumer": "Consumer Cyclical",
    "industrials": "Industrials",
    "industrial": "Industrials",
    "energy": "Energy",
    "utilities": "Utilities",
    "real estate": "Real Estate",
    "realestate": "Real Estate",
    "materials": "Basic Materials",
    "basic materials": "Basic Materials",
    "communication": "Communication Services",
    "communication services": "Communication Services",
    "consumer defensive": "Consumer Defensive",
    
    # Vietnamese variants
    "công nghệ": "Technology",
    "cong nghe": "Technology",
    "y tế": "Healthcare",
    "y te": "Healthcare",
    "chăm sóc sức khỏe": "Healthcare",
    "tài chính": "Financial Services",
    "tai chinh": "Financial Services",
    "dịch vụ tài chính": "Financial Services",
    "năng lượng": "Energy",
    "nang luong": "Energy",
    "công nghiệp": "Industrials",
    "cong nghiep": "Industrials",
    "tiện ích": "Utilities",
    "tien ich": "Utilities",
    "bất động sản": "Real Estate",
    "bat dong san": "Real Estate",
    "vật liệu": "Basic Materials",
    "vat lieu": "Basic Materials",
    "truyền thông": "Communication Services",
    "truyen thong": "Communication Services",
    "tiêu dùng": "Consumer Cyclical",
    "tieu dung": "Consumer Cyclical",
}


class StockScreenerTool(BaseTool, LoggerMixin):
    """
    Stock Screener Tool
    
    Find stocks based on multiple criteria including sector, market cap,
    price range, volume, etc.
    
    CRITICAL: This tool FINDS stocks - does NOT need symbol input!
    """
    
    CACHE_TTL = 600  # 10 minutes
    
    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key or settings.FMP_API_KEY
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
        # Define schema
        self.schema = ToolSchema(
            name="stockScreener",
            category="discovery",
            description=(
                "Find and filter stocks based on multiple criteria (sector, market cap, price, volume, etc.). "
                "This tool DISCOVERS stocks - it does NOT analyze specific symbols. "
                "Returns list of stock symbols matching ALL specified criteria."
            ),
            requires_symbol=False,
            capabilities=[
                "✅ Screen stocks by sector (Technology, Healthcare, Financial Services, etc.)",
                "✅ Filter by market cap, price range, volume, beta",
                "✅ Filter by exchange (NYSE, NASDAQ) and country",
                "✅ Return list of matching stock symbols for further analysis",
                "✅ Can be used without any symbol input - this tool FINDS symbols"
            ],
            limitations=[
                "❌ Cannot compute technical indicators (RSI, MACD, Bollinger Bands)",
                "❌ Cannot screen by technical conditions directly (must use in sequence)",
                "❌ Cannot analyze fundamentals - only returns symbols",
                "❌ Result count limited by 'limit' parameter (recommended: 5-10)",
                "❌ Does NOT provide detailed stock data - use other tools on results"
            ],
            usage_hints=[
                "→ User wants to FIND stocks: 'find tech stocks', 'tìm cổ phiếu công nghệ'",
                "→ User provides CRITERIA without symbol: 'stocks with high volume'",
                "→ User asks 'which stocks...': 'which stocks have low P/E in healthcare?'",
                "→ User uses search verbs: 'search', 'find', 'discover', 'tìm', 'show me'",
                "→ First step in screener queries - other tools analyze the results",
                "→ NEVER use when user mentions specific symbol/company (AAPL, Apple, Tesla)"
            ],
            parameters=[
                ToolParameter(
                    name="sector",
                    type="string",
                    description="Sector to filter (Technology, Healthcare, Financial Services, etc.)",
                    required=False,
                    enum=["Technology", "Healthcare", "Financial Services", "Consumer Cyclical", 
                        "Industrials", "Energy", "Utilities", "Real Estate", "Basic Materials", 
                        "Communication Services", "Consumer Defensive"]
                ),
                ToolParameter(
                    name="industry",
                    type="string",
                    description="Industry to filter",
                    required=False
                ),
                ToolParameter(
                    name="market_cap_more_than",
                    type="integer",
                    description="Minimum market cap in USD",
                    required=False
                ),
                ToolParameter(
                    name="market_cap_lower_than",
                    type="integer",
                    description="Maximum market cap in USD",
                    required=False
                ),
                ToolParameter(
                    name="price_more_than",
                    type="number",
                    description="Minimum stock price",
                    required=False
                ),
                ToolParameter(
                    name="price_lower_than",
                    type="number",
                    description="Maximum stock price",
                    required=False
                ),
                ToolParameter(
                    name="volume_more_than",
                    type="integer",
                    description="Minimum daily volume",
                    required=False
                ),
                ToolParameter(
                    name="beta_more_than",
                    type="number",
                    description="Minimum beta",
                    required=False
                ),
                ToolParameter(
                    name="beta_lower_than",
                    type="number",
                    description="Maximum beta",
                    required=False
                ),
                ToolParameter(
                    name="dividend_more_than",
                    type="number",
                    description="Minimum dividend yield",
                    required=False
                ),
                ToolParameter(
                    name="is_etf",
                    type="boolean",
                    description="Filter ETFs only",
                    required=False,
                    default=False
                ),
                ToolParameter(
                    name="is_actively_trading",
                    type="boolean",
                    description="Only actively trading stocks",
                    required=False,
                    default=True
                ),
                ToolParameter(
                    name="exchange",
                    type="string",
                    description="Exchange (NYSE, NASDAQ)",
                    required=False
                ),
                ToolParameter(
                    name="country",
                    type="string",
                    description="Country (US)",
                    required=False
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Number of stocks to return (RECOMMENDED: 5-10 for quality analysis)",
                    required=False,
                    default=10,
                    min_value=1,
                    max_value=100
                ),
            ],
            returns={
                "stocks": "array - List of stock objects with symbol, name, sector, etc.",
                "symbols": "array - List of ticker symbols",
                "count": "number - Number of results",
                "criteria": "object - Applied filter criteria"
            },
            typical_execution_time_ms=1500
        )
        
        logger.info("✅ Registered tool: stockScreener (category: discovery)")
    
    def _normalize_sector(self, sector: str) -> Optional[str]:
        """Normalize sector name to FMP format"""
        if not sector:
            return None
            
        sector_lower = sector.lower().strip()
        
        # Check mapping
        if sector_lower in SECTOR_MAPPING:
            return SECTOR_MAPPING[sector_lower]
        
        # Check if already in correct format
        valid_sectors = [
            "Technology", "Healthcare", "Financial Services",
            "Consumer Cyclical", "Industrials", "Energy",
            "Utilities", "Real Estate", "Basic Materials",
            "Communication Services", "Consumer Defensive"
        ]
        
        for valid in valid_sectors:
            if sector_lower == valid.lower():
                return valid
        
        # Return as-is with title case
        return sector.title()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[dict]:
        """Get cached result from Redis"""
        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                import json
                cached = await redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: dict) -> None:
        """Cache result to Redis"""
        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                import json
                await redis_client.set(cache_key, json.dumps(result), ex=self.CACHE_TTL)
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")
    
    async def execute(
        self,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        market_cap_more_than: Optional[int] = None,
        market_cap_lower_than: Optional[int] = None,
        price_more_than: Optional[float] = None,
        price_lower_than: Optional[float] = None,
        volume_more_than: Optional[int] = None,
        beta_more_than: Optional[float] = None,
        beta_lower_than: Optional[float] = None,
        dividend_more_than: Optional[float] = None,
        is_etf: Optional[bool] = False,
        is_actively_trading: Optional[bool] = True,
        exchange: Optional[str] = None,
        country: Optional[str] = None,
        limit: Optional[int] = 20
    ) -> ToolOutput:
        """
        Execute stock screener
        
        Args:
            sector: Sector to filter (Technology, Healthcare, etc.)
            industry: Industry to filter
            market_cap_more_than: Minimum market cap
            market_cap_lower_than: Maximum market cap
            price_more_than: Minimum price
            price_lower_than: Maximum price
            volume_more_than: Minimum volume
            beta_more_than: Minimum beta
            beta_lower_than: Maximum beta
            dividend_more_than: Minimum dividend yield
            is_etf: Filter ETFs only
            is_actively_trading: Only actively trading
            exchange: Exchange filter
            country: Country filter
            limit: Number of results (max 100)
            
        Returns:
            ToolOutput with list of matching stocks
        """
        start_time = datetime.now()
        
        try:
            # Build API params
            api_params = {"apikey": self.api_key}
            
            # Normalize and add sector
            if sector:
                normalized_sector = self._normalize_sector(sector)
                if normalized_sector:
                    api_params["sector"] = normalized_sector
            
            # Add other params
            if industry:
                api_params["industry"] = industry
            if market_cap_more_than is not None:
                api_params["marketCapMoreThan"] = market_cap_more_than
            if market_cap_lower_than is not None:
                api_params["marketCapLowerThan"] = market_cap_lower_than
            if price_more_than is not None:
                api_params["priceMoreThan"] = price_more_than
            if price_lower_than is not None:
                api_params["priceLowerThan"] = price_lower_than
            if volume_more_than is not None:
                api_params["volumeMoreThan"] = volume_more_than
            if beta_more_than is not None:
                api_params["betaMoreThan"] = beta_more_than
            if beta_lower_than is not None:
                api_params["betaLowerThan"] = beta_lower_than
            if dividend_more_than is not None:
                api_params["dividendMoreThan"] = dividend_more_than
            if is_etf is not None:
                api_params["isEtf"] = is_etf
            if is_actively_trading is not None:
                api_params["isActivelyTrading"] = is_actively_trading
            if exchange:
                api_params["exchange"] = exchange
            if country:
                api_params["country"] = country
            if limit:
                api_params["limit"] = min(limit, 100)
            
            # Log request (without apikey)
            log_params = {k: v for k, v in api_params.items() if k != "apikey"}
            self.logger.info(f"[{self.schema.name}] Screening stocks with: {log_params}")
            
            # Check cache
            cache_key = f"screener:{hash(str(sorted(log_params.items())))}"
            cached = await self._get_cached_result(cache_key)
            if cached:
                self.logger.info(f"[{self.schema.name}] Cache HIT")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached
                )
            
            # Make API request
            url = f"{self.base_url}/stock-screener"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=api_params)
                response.raise_for_status()
                data = response.json()
            
            if not isinstance(data, list):
                self.logger.warning(f"[{self.schema.name}] Unexpected response: {type(data)}")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error="Unexpected API response format"
                )
            
            # Process results
            stocks = []
            symbols = []
            for item in data:
                if isinstance(item, dict) and item.get("symbol"):
                    symbol = item.get("symbol")
                    symbols.append(symbol)
                    stocks.append({
                        "symbol": symbol,
                        "name": item.get("companyName"),
                        "sector": item.get("sector"),
                        "industry": item.get("industry"),
                        "market_cap": item.get("marketCap"),
                        "price": item.get("price"),
                        "volume": item.get("volume"),
                        "beta": item.get("beta"),
                        "exchange": item.get("exchangeShortName"),
                        "country": item.get("country")
                    })
            
            # Build result
            result = {
                "stocks": stocks,
                "symbols": symbols,
                "count": len(stocks),
                "criteria": log_params,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add summary
            if stocks:
                result["top_symbols"] = symbols[:10]
                result["summary"] = f"Found {len(stocks)} stocks"
                if api_params.get("sector"):
                    result["summary"] += f" in {api_params['sector']} sector"
            else:
                result["summary"] = "No stocks found matching criteria"
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: Found {len(stocks)} stocks ({execution_time}ms) List symbols {stocks}"
            )
            
            return ToolOutput(
                tool_name=self.schema.name,
                status="success",
                data=result
            )
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error: {e.response.status_code}"
            self.logger.error(f"[{self.schema.name}] {error_msg}")
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=error_msg
            )
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[{self.schema.name}] Error: {error_msg}", exc_info=True)
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=error_msg
            )