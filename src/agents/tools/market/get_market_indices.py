# src/agents/tools/market/get_market_indices.py

import json
import time
from typing import Dict, Any
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetMarketIndicesTool(BaseTool, LoggerMixin):
    """
    Tool 18: Get Market Indices
    
    Fetch real-time quotes for major market indices (S&P 500, Nasdaq, Dow Jones, VIX, etc.)
    
    FMP Stable API:
    GET https://financialmodelingprep.com/stable/batch-index-quotes
    """

    CACHE_TTL = 300  # 5 minutes - indices update frequently

    def __init__(self, api_key: str):
        """
        Initialize GetMarketIndicesTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        
        self.schema = ToolSchema(
            name="getMarketIndices",
            category="market",
            description=(
                "Fetch real-time quotes for ALL major market indices without needing symbol. "
                "Returns S&P 500, Nasdaq, Dow Jones, VIX, and other global indices. "
                "NO SYMBOL REQUIRED - automatically returns all indices. "
                "Use when user asks about overall market, market sentiment, or major indices."
            ),
            capabilities=[
                "✅ All major US indices (S&P 500, Nasdaq, Dow Jones, VIX)",
                "✅ Real-time quotes (15-min delay on free tier)",
                "✅ Intraday price changes",
                "✅ 52-week high/low for each index",
                "✅ Market cap and volume",
                "✅ Quick access to major indices"
            ],
            limitations=[
                "❌ 5-minute delay on free tier",
                "❌ No historical index data",
                "❌ Cannot filter specific indices",
                "❌ Does NOT return list of stocks in a sector"
            ],
            usage_hints=[
                # English - Market overview
                "User asks: 'How is the market today?' → USE THIS (NO params)",
                "User asks: 'Show me market indices' → USE THIS (NO params)",
                "User asks: 'S&P 500 performance' → USE THIS (NO params)",
                "User asks: 'What's the VIX level?' → USE THIS (NO params)",
                "User asks: 'Market sentiment' → USE THIS (NO params)",
                # Vietnamese - Thị trường
                "User asks: 'Thị trường hôm nay thế nào?' → USE THIS (NO params)",
                "User asks: 'Các chỉ số thị trường' → USE THIS (NO params)",
                "User asks: 'Tình hình thị trường' → USE THIS (NO params)",
                # When NOT to use
                "User asks about SPECIFIC stock → DO NOT USE (use getStockPrice)",
                "User asks about SECTORS → DO NOT USE (use getSectorPerformance)",
                "User asks about top movers → DO NOT USE (use getMarketMovers)",
                "User wants LIST OF STOCKS in a sector → DO NOT USE (this returns indices only)"
            ],
            parameters=[],  # NO PARAMETERS NEEDED
            returns={
                "indices": "array - All index quotes",
                "index_count": "number",
                "major_indices": "object - S&P 500, Nasdaq, Dow Jones quick access",
                "timestamp": "string"
            },
            typical_execution_time_ms=800,
            requires_symbol=False  # KEY: No symbol required
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """
        Execute getMarketIndices
        
        Args:
            **kwargs: No parameters needed
            
        Returns:
            ToolOutput with index quotes
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"[{self.schema.name}] Fetching all market indices")
            
            # Check cache
            cache_key = f"getMarketIndices_all"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT")
                # ═══════════════════════════════════════════════════════════
                # FIX: status must be string "success", not int 200
                # ═══════════════════════════════════════════════════════════
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",  # ✅ FIXED: was 200 (int)
                    data=cached_result
                )
            
            # Fetch from FMP API
            url = f"{self.base_url}/batch-index-quotes"
            params = {"apikey": self.api_key}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                indices_data = response.json()
            
            # Validate response
            if not indices_data or not isinstance(indices_data, list):
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error="Invalid response format from FMP API",
                    metadata={"response": indices_data}
                )
            
            # Check major indices presence
            symbols = {idx.get("symbol") for idx in indices_data}
            major_indices = ["^GSPC", "^IXIC", "^DJI"]
            missing_major = [s for s in major_indices if s not in symbols]
            
            if len(missing_major) == len(major_indices):
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error="No major indices found in response",
                    metadata={"symbols_received": list(symbols)[:10]}
                )
            
            # Extract major indices for quick access
            major_dict = {}
            for idx in indices_data:
                symbol = idx.get("symbol", "")
                if symbol in major_indices:
                    major_dict[symbol] = {
                        "name": idx.get("name"),
                        "price": idx.get("price"),
                        "change": idx.get("change"),
                        "changes_percentage": idx.get("changesPercentage")
                    }
            
            # Build result
            result = {
                "indices": indices_data,
                "index_count": len(indices_data),
                "major_indices": major_dict,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: {len(indices_data)} indices "
                f"({execution_time:.0f}ms)"
            )
            
            # ═══════════════════════════════════════════════════════════
            # FIX: status must be string "success", not int 200
            # ═══════════════════════════════════════════════════════════
            return ToolOutput(
                tool_name=self.schema.name,
                status="success",  # ✅ FIXED: was 200 (int)
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