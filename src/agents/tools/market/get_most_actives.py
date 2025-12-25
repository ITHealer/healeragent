# src/agents/tools/market/get_most_actives.py
"""
GetMostActivesTool - Fetch most actively traded stocks

Dedicated tool for highest volume stocks - NO parameters needed.
This eliminates LLM enum selection errors.
"""

import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetMostActivesTool(BaseTool, LoggerMixin):
    """
    Tool: Get Most Actives
    
    Fetch most actively traded stocks by volume.
    NO SYMBOL REQUIRED - returns top ~20 stocks with highest trading volume.
    
    FMP API:
    - GET https://financialmodelingprep.com/stable/most-actives
    """

    CACHE_TTL = 300  # 5 minutes

    def __init__(self, api_key: str):
        """
        Initialize GetMostActivesTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"

        self.schema = ToolSchema(
            name="getMostActives",
            category="market",
            description=(
                "Fetch most actively traded stocks (highest volume today). "
                "NO SYMBOL REQUIRED - returns ~20 stocks with highest trading volume. "
                "Use when user asks about: most active, highest volume, most traded, "
                "khối lượng giao dịch lớn nhất, cổ phiếu giao dịch nhiều nhất."
            ),
            capabilities=[
                "✅ Top ~20 stocks with highest trading volume today",
                "✅ Real-time price, change amount and percentage",
                "✅ Volume data in shares",
                "✅ Company name and market cap"
            ],
            limitations=[
                "❌ Only current trading day data",
                "❌ Cannot filter by sector or price range",
                "❌ US market stocks only"
            ],
            usage_hints=[
                # English triggers
                "User asks: 'Most active stocks' → USE THIS",
                "User asks: 'Highest volume today' → USE THIS",
                "User asks: 'Most traded stocks' → USE THIS",
                "User asks: 'What stocks have the most activity?' → USE THIS",
                # Vietnamese triggers
                "User asks: 'Cổ phiếu giao dịch nhiều nhất' → USE THIS",
                "User asks: 'Khối lượng lớn nhất hôm nay' → USE THIS",
                "User asks: 'Most active hôm nay' → USE THIS",
                "User asks: 'Cổ phiếu sôi động nhất' → USE THIS",
                # When NOT to use
                "User asks about specific stock price → DO NOT USE (use getStockPrice)",
                "User asks about gainers → DO NOT USE (use getTopGainers)",
                "User asks about losers → DO NOT USE (use getTopLosers)"
            ],
            parameters=[],  # NO parameters - eliminates LLM errors
            returns={
                "stocks": "array - List of most active stocks with volume data",
                "count": "number - Total stocks returned",
                "timestamp": "string - Data timestamp"
            },
            typical_execution_time_ms=800,
            requires_symbol=False
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """
        Execute getMostActives
        
        No parameters needed - fetches current most active stocks.
        
        Returns:
            ToolOutput with most active stocks data
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"[{self.schema.name}] Fetching most active stocks")
            
            # Check cache first
            cache_key = "getMostActives_v2"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result
                )
            
            # Fetch from FMP API
            url = f"{self.base_url}/most-actives"
            params = {"apikey": self.api_key}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                stocks_data = response.json()
            
            # Validate response
            if not stocks_data or not isinstance(stocks_data, list):
                self.logger.error(f"[{self.schema.name}] Invalid API response")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error="Invalid response format from FMP API",
                    metadata={"response_type": type(stocks_data).__name__}
                )
            
            # Validate data integrity - actives should have volume > 0
            valid_stocks = []
            invalid_count = 0
            
            for stock in stocks_data:
                volume = stock.get("volume", 0)
                if volume and volume > 0:
                    valid_stocks.append(stock)
                else:
                    invalid_count += 1
            
            if invalid_count > 0:
                self.logger.warning(
                    f"[{self.schema.name}] Filtered {invalid_count} stocks "
                    f"with zero/missing volume"
                )
            
            # Sort by volume descending (ensure proper ordering)
            valid_stocks.sort(key=lambda x: x.get("volume", 0), reverse=True)
            
            # Build result
            result = {
                "mover_type": "actives",
                "stocks": valid_stocks,
                "count": len(valid_stocks),
                "timestamp": datetime.now().isoformat(),
                "data_source": "FMP"
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: {len(valid_stocks)} active stocks "
                f"({execution_time:.0f}ms)"
            )
            
            return ToolOutput(
                tool_name=self.schema.name,
                status="success",
                data=result
            )
                        
        except httpx.HTTPStatusError as e:
            self.logger.error(f"[{self.schema.name}] HTTP error: {e}")
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=f"FMP API error: {e.response.status_code}",
                metadata={"response": e.response.text[:200]}
            )
        except httpx.TimeoutException as e:
            self.logger.error(f"[{self.schema.name}] Timeout: {e}")
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error="Request timeout - FMP API not responding",
                metadata={"timeout": 30}
            )
        except Exception as e:
            self.logger.error(f"[{self.schema.name}] Error: {e}", exc_info=True)
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=str(e),
                metadata={"type": type(e).__name__}
            )

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result from Redis"""
        try:
            redis_client = await get_redis_client_llm()
            cached_bytes = await redis_client.get(cache_key)
            await redis_client.close()
            
            if cached_bytes:
                return json.loads(cached_bytes.decode('utf-8'))
            return None
        except Exception as e:
            self.logger.warning(f"[{self.schema.name}] Cache read error: {e}")
            return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result to Redis"""
        try:
            redis_client = await get_redis_client_llm()
            json_string = json.dumps(result, default=str)
            await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
            await redis_client.close()
        except Exception as e:
            self.logger.warning(f"[{self.schema.name}] Cache write error: {e}")