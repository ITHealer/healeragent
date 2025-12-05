# src/agents/tools/market/get_market_movers.py
"""
GetMarketMoversTool - FIXED VERSION

Fixes:
1. Replaced self.create_error_output() with ToolOutput directly
2. Fixed typo: "scuccess" -> "success"
3. Proper error handling
"""

import json
import time
from typing import Dict, Any, Literal
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetMarketMoversTool(BaseTool, LoggerMixin):
    """
    Tool 20: Get Market Movers
    
    Fetch top gainers, losers, or most active stocks
    
    FMP Stable APIs:
    - GET https://financialmodelingprep.com/stable/biggest-gainers
    - GET https://financialmodelingprep.com/stable/biggest-losers
    - GET https://financialmodelingprep.com/stable/most-actives
    """

    CACHE_TTL = 300  # 5 minutes

    def __init__(self, api_key: str):
        """
        Initialize GetMarketMoversTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"

        self.schema = ToolSchema(
            name="getMarketMovers",
            category="market",
            description=(
                "Fetch top market movers: biggest gainers, biggest losers, or most active stocks. "
                "NO SYMBOL REQUIRED - returns top stocks by category. "
                "Use when user asks about hot stocks, trending stocks, top gainers/losers, or most active."
            ),
            capabilities=[
                "✅ Top gainers (largest % gains)",
                "✅ Top losers (largest % losses)",
                "✅ Most active (highest volume)",
                "✅ Real-time price data for top stocks",
                "✅ Volume and market cap",
                "✅ 52-week high/low"
            ],
            limitations=[
                "❌ Limited to top movers only (~20 stocks)",
                "❌ No historical movers data",
                "❌ Cannot filter by sector or market cap"
            ],
            usage_hints=[
                # English
                "User asks: 'Top gainers today' → USE THIS with mover_type=gainers",
                "User asks: 'Biggest losers' → USE THIS with mover_type=losers",
                "User asks: 'Most active stocks' → USE THIS with mover_type=actives",
                "User asks: 'Hot stocks today' → USE THIS with mover_type=gainers",
                # Vietnamese
                "User asks: 'Cổ phiếu tăng mạnh nhất' → USE THIS with mover_type=gainers",
                "User asks: 'Top losers hôm nay' → USE THIS with mover_type=losers",
                "User asks: 'Khối lượng giao dịch lớn nhất' → USE THIS with mover_type=actives",
                # When NOT to use
                "User asks about specific stock → DO NOT USE (use getStockPrice)",
                "User asks about market overview → DO NOT USE (use getMarketIndices)"
            ],
            parameters=[
                ToolParameter(
                    name="mover_type",
                    type="string",
                    description="Type of movers to fetch",
                    required=False,
                    default="gainers",
                    enum=["gainers", "losers", "actives"]
                )
            ],
            returns={
                "mover_type": "string",
                "stocks": "array - List of top movers",
                "count": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=900,
            requires_symbol=False  # IMPORTANT: This tool does NOT require symbol
        )


    async def execute(self, mover_type: str = "gainers", **kwargs) -> ToolOutput:
        """
        Execute getMarketMovers
        
        Args:
            mover_type: Type of movers ("gainers", "losers", "actives")
            
        Returns:
            ToolOutput with market movers data
        """
        start_time = time.time()
        
        try:
            # Validate mover_type
            if mover_type not in ["gainers", "losers", "actives"]:
                # FIX: Use ToolOutput directly instead of self.create_error_output()
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error=f"Invalid mover_type: {mover_type}. Allowed values: gainers, losers, actives",
                    metadata={"allowed_values": ["gainers", "losers", "actives"]}
                )
            
            self.logger.info(f"[{self.schema.name}] Fetching market movers (type={mover_type})")
            
            # Check cache
            cache_key = f"getMarketMovers_{mover_type}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",  # FIX: "scuccess" -> "success"
                    data=cached_result
                )
            
            # Map mover_type to endpoint
            endpoint_map = {
                "gainers": "biggest-gainers",
                "losers": "biggest-losers",
                "actives": "most-actives"
            }
            
            # Fetch from FMP API
            url = f"{self.base_url}/{endpoint_map[mover_type]}"
            params = {"apikey": self.api_key}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                stocks_data = response.json()
            
            # Validate response
            if not stocks_data or not isinstance(stocks_data, list):
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error="Invalid response format from FMP API",
                    metadata={"response": stocks_data}
                )
                        
            if len(stocks_data) < 10:
                self.logger.warning(
                    f"[{self.schema.name}] Low stock count: {len(stocks_data)} (expected 10+)"
                )
            
            # Validate data integrity
            if mover_type == "gainers":
                invalid = [s for s in stocks_data if s.get("changesPercentage", 0) <= 0]
                if invalid:
                    self.logger.warning(
                        f"[{self.schema.name}] Found {len(invalid)} stocks with non-positive change in gainers"
                    )
            
            elif mover_type == "losers":
                invalid = [s for s in stocks_data if s.get("changesPercentage", 0) >= 0]
                if invalid:
                    self.logger.warning(
                        f"[{self.schema.name}] Found {len(invalid)} stocks with non-negative change in losers"
                    )
            
            # Build result
            result = {
                "mover_type": mover_type,
                "stocks": stocks_data,
                "count": len(stocks_data),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: {len(stocks_data)} {mover_type} "
                f"({execution_time:.0f}ms)"
            )
            
            return ToolOutput(
                tool_name=self.schema.name,
                status="success",  # FIX: "scuccess" -> "success"
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
        except Exception as e:
            self.logger.error(f"[{self.schema.name}] Error: {e}", exc_info=True)
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=str(e),
                metadata={"type": type(e).__name__}
            )

    async def _get_cached_result(self, cache_key: str) -> Dict[str, Any] | None:
        """Get cached result from Redis"""
        try:
            redis_client = await get_redis_client_llm()
            cached_bytes = await redis_client.get(cache_key)
            await redis_client.close()
            
            if cached_bytes:
                return json.loads(cached_bytes.decode('utf-8'))
            return None
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
            return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result to Redis"""
        try:
            redis_client = await get_redis_client_llm()
            json_string = json.dumps(result)
            await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
            await redis_client.close()
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")