# src/agents/tools/market/get_market_breadth.py

import json
import time
from typing import Dict, Any
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetMarketBreadthTool(BaseTool, LoggerMixin):
    """
    Tool 21: Get Market Breadth
    
    Calculate market breadth metrics (advance/decline ratio, sector breadth, market sentiment)
    
    Custom calculation using:
    - GET https://financialmodelingprep.com/stable/sector-performance-snapshot
    """

    CACHE_TTL = 900  # 15 minutes

    def __init__(self, api_key: str):
        """
        Initialize GetMarketBreadthTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        
        self.schema = ToolSchema(
            name="getMarketBreadth",
            category="market",
            description=(
                "Calculate market breadth metrics: advance/decline ratio, sector breadth, market sentiment. "
                "NO SYMBOL REQUIRED - analyzes overall market internals. "
                "Use when user asks about market strength, market health, breadth indicators, or internal market analysis."
            ),
            capabilities=[
                "✅ Advance/decline ratio",
                "✅ Percentage of advancing/declining sectors",
                "✅ Market sentiment (bullish/bearish/neutral)",
                "✅ Sector breadth analysis",
                "✅ Market health assessment"
            ],
            limitations=[
                "❌ Based on sector data (not individual stocks)",
                "❌ Updated every 15 minutes",
                "❌ No historical breadth data"
            ],
            usage_hints=[
                # English
                "User asks: 'How strong is the market?' → USE THIS (NO params)",
                "User asks: 'Market breadth today' → USE THIS (NO params)",
                "User asks: 'Advance decline ratio' → USE THIS (NO params)",
                "User asks: 'Market internals' → USE THIS (NO params)",
                # Vietnamese
                "User asks: 'Độ rộng thị trường' → USE THIS (NO params)",
                "User asks: 'Thị trường mạnh không?' → USE THIS (NO params)",
                "User asks: 'Tình trạng thị trường' → USE THIS (NO params)",
                # When NOT to use
                "User asks about specific stock → DO NOT USE",
                "User asks about sectors → DO NOT USE (use getSectorPerformance)"
            ],
            parameters=[],  # NO PARAMETERS
            returns={
                "breadth": "object - Market breadth metrics",
                "advancers": "number",
                "decliners": "number",
                "advance_decline_ratio": "number",
                "market_sentiment": "string",
                "timestamp": "string"
            },
            typical_execution_time_ms=1000,
            requires_symbol=False
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """
        Execute getMarketBreadth
        
        Returns:
            ToolOutput with market breadth data
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"[{self.schema.name}] Calculating market breadth")
            
            # Check cache
            cache_key = "getMarketBreadth_latest"
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
            
            # Fetch sector performance
            url = f"{self.base_url}/sector-performance-snapshot"
            params = {"apikey": self.api_key}
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                sectors_data = response.json()
            
            # Validate response
            if not sectors_data or not isinstance(sectors_data, list):
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error="Invalid response format from FMP API",
                    metadata={"response": sectors_data}
                )
            
            # Calculate breadth metrics
            advancers = sum(1 for s in sectors_data if s.get("changePercent", 0) > 0)
            decliners = sum(1 for s in sectors_data if s.get("changePercent", 0) < 0)
            unchanged = len(sectors_data) - advancers - decliners
            total_sectors = len(sectors_data)
            
            # Calculate ratios
            advance_decline_ratio = advancers / max(decliners, 1)
            percent_advancers = (advancers / total_sectors * 100) if total_sectors > 0 else 0
            percent_decliners = (decliners / total_sectors * 100) if total_sectors > 0 else 0
            
            # Determine market sentiment
            if advance_decline_ratio > 1.5:
                market_sentiment = "bullish"
            elif advance_decline_ratio < 0.67:
                market_sentiment = "bearish"
            else:
                market_sentiment = "neutral"
            
            # Build breadth object
            breadth = {
                "advancers": advancers,
                "decliners": decliners,
                "unchanged": unchanged,
                "advance_decline_ratio": round(advance_decline_ratio, 2),
                "percent_advancers": round(percent_advancers, 2),
                "percent_decliners": round(percent_decliners, 2),
                "total_sectors": total_sectors,
                "market_sentiment": market_sentiment
            }
            
            # Build result
            result = {
                "breadth": breadth,
                "advancers": advancers,
                "decliners": decliners,
                "unchanged": unchanged,
                "advance_decline_ratio": round(advance_decline_ratio, 2),
                "percent_advancers": round(percent_advancers, 2),
                "market_sentiment": market_sentiment,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: {advancers}↑ / {decliners}↓ "
                f"(ratio={advance_decline_ratio:.2f}, sentiment={market_sentiment}) "
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