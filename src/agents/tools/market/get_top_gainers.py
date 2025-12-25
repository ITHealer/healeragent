# src/agents/tools/market/get_top_gainers.py
"""
GetTopGainersTool - Fetch top gaining stocks

Dedicated tool for biggest gainers - NO parameters needed.
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


class GetTopGainersTool(BaseTool, LoggerMixin):
    """
    Tool: Get Top Gainers
    
    Fetch biggest gaining stocks from market.
    NO SYMBOL REQUIRED - returns top ~20 stocks with largest % gains.
    
    FMP API:
    - GET https://financialmodelingprep.com/stable/biggest-gainers
    """

    CACHE_TTL = 300  # 5 minutes

    def __init__(self, api_key: str):
        """
        Initialize GetTopGainersTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"

        self.schema = ToolSchema(
            name="getTopGainers",
            category="market",
            description=(
                "Fetch top gaining stocks (biggest percentage increase today). "
                "NO SYMBOL REQUIRED - returns ~20 stocks with largest gains. "
                "Use when user asks about: top gainers, hot stocks, stocks going up, "
                "cổ phiếu tăng mạnh, tăng giá nhiều nhất."
            ),
            capabilities=[
                "✅ Top ~20 stocks with largest % gains today",
                "✅ Real-time price, change amount and percentage",
                "✅ Volume and market cap data",
                "✅ Company name included"
            ],
            limitations=[
                "❌ Only current trading day data",
                "❌ Cannot filter by sector or market cap",
                "❌ US market stocks only"
            ],
            usage_hints=[
                # English triggers
                "User asks: 'Top gainers today' → USE THIS",
                "User asks: 'Which stocks are up the most?' → USE THIS",
                "User asks: 'Hot stocks right now' → USE THIS",
                "User asks: 'Biggest winners today' → USE THIS",
                # Vietnamese triggers
                "User asks: 'Cổ phiếu tăng mạnh nhất' → USE THIS",
                "User asks: 'Top gainers hôm nay' → USE THIS",
                "User asks: 'Cổ phiếu nào tăng giá nhiều?' → USE THIS",
                # When NOT to use
                "User asks about specific stock price → DO NOT USE (use getStockPrice)",
                "User asks about losers → DO NOT USE (use getTopLosers)",
                "User asks about volume → DO NOT USE (use getMostActives)"
            ],
            parameters=[],  # NO parameters - eliminates LLM errors
            returns={
                "stocks": "array - List of top gaining stocks with price data",
                "count": "number - Total stocks returned",
                "timestamp": "string - Data timestamp"
            },
            typical_execution_time_ms=800,
            requires_symbol=False
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """
        Execute getTopGainers
        
        No parameters needed - fetches current top gainers.
        
        Returns:
            ToolOutput with top gaining stocks data
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"[{self.schema.name}] Fetching top gainers")
            
            # Check cache first
            cache_key = "getTopGainers_v2"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result
                )
            
            # Fetch from FMP API
            url = f"{self.base_url}/biggest-gainers"
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
            
            # Validate data integrity - gainers should have positive change
            valid_stocks = []
            invalid_count = 0
            
            for stock in stocks_data:
                change_pct = stock.get("changesPercentage", 0)
                if change_pct > 0:
                    valid_stocks.append(stock)
                else:
                    invalid_count += 1
            
            if invalid_count > 0:
                self.logger.warning(
                    f"[{self.schema.name}] Filtered {invalid_count} stocks "
                    f"with non-positive change"
                )
            
            # Build result
            result = {
                "mover_type": "gainers",
                "stocks": valid_stocks,
                "count": len(valid_stocks),
                "timestamp": datetime.now().isoformat(),
                "data_source": "FMP"
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: {len(valid_stocks)} gainers "
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