# src/agents/tools/market/get_sector_performance.py

import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetSectorPerformanceTool(BaseTool, LoggerMixin):
    """
    Tool 19: Get Sector Performance
    
    Fetch performance snapshot for all market sectors (Technology, Healthcare, Finance, etc.)
    
    FMP Stable API:
    GET https://financialmodelingprep.com/stable/sector-performance-snapshot
    """

    CACHE_TTL = 900  # 15 minutes

    def __init__(self, api_key: str):
        """
        Initialize GetSectorPerformanceTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        
        # Define schema
        self.schema = ToolSchema(
            name="getSectorPerformance",
            category="market",
            description=(
                "Fetch performance snapshot for ALL market sectors without needing symbol. "
                "Returns percentage change for Technology, Healthcare, Finance, Energy, and all other sectors. "
                "NO SYMBOL REQUIRED - automatically returns all sectors. "
                "Use when user asks about sector rotation, best/worst sectors, or sector analysis."
            ),
            capabilities=[
                "✅ All market sectors performance (11 sectors)",
                "✅ Sector percentage changes",
                "✅ Number of companies per sector",
                "✅ Best and worst performing sectors",
                "✅ Market sentiment by sector count",
                "✅ Historical date support (optional)"
            ],
            limitations=[
                "❌ Updated every 15 minutes",
                "❌ No intraday sector changes",
                "❌ No subsector breakdowns"
            ],
            usage_hints=[
                # English
                "User asks: 'Which sector is performing best?' → USE THIS (NO params)",
                "User asks: 'Show me sector performance' → USE THIS (NO params)",
                "User asks: 'Technology sector performance' → USE THIS (NO params)",
                "User asks: 'Sector rotation analysis' → USE THIS (NO params)",
                # Vietnamese
                "User asks: 'Ngành nào tốt nhất?' → USE THIS (NO params)",
                "User asks: 'Hiệu suất các sector' → USE THIS (NO params)",
                "User asks: 'Phân tích sector Technology' → USE THIS (NO params)",
                # When NOT to use
                "User asks about specific STOCK → DO NOT USE (use getStockPrice)",
                "User asks about MARKET INDICES → DO NOT USE (use getMarketIndices)"
            ],
            parameters=[
                ToolParameter(
                    name="date",
                    type="string",
                    description="Optional date in YYYY-MM-DD format. If not provided, returns latest",
                    required=False,
                    pattern=r"^\d{4}-\d{2}-\d{2}$"
                )
            ],
            returns={
                "sectors": "array - All sector performance data",
                "sector_count": "number",
                "date": "string",
                "best_sector": "string",
                "worst_sector": "string",
                "summary": "object - Market sentiment",
                "timestamp": "string"
            },
            typical_execution_time_ms=900,
            requires_symbol=False
        )
        
    async def execute(self, date: Optional[str] = None, **kwargs) -> ToolOutput:
        """
        Execute getSectorPerformance
        
        Args:
            date: Optional date in YYYY-MM-DD format
            
        Returns:
            ToolOutput with sector performance data
        """
        start_time = time.time()
        
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        try:
            self.logger.info(f"[{self.schema.name}] Fetching sector performance (date={date})")
            
            # Check cache
            cache_key = f"getSectorPerformance_{date or 'latest'}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result
                )
            
            
            # Fetch from FMP API
            url = f"{self.base_url}/sector-performance-snapshot"
            params = {"apikey": self.api_key}
            if date:
                params["date"] = date
            
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
            
            if len(sectors_data) < 8:
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="error",
                    error="Insufficient sectors data: expected at least 8, got {len(sectors_data)}",
                    metadata={"symbols_received": sectors_data}
                )
            
            # Find best and worst sectors
            sectors_sorted = sorted(
                sectors_data,
                key=lambda x: x.get("changePercent", 0),
                reverse=True
            )
            
            best_sector = sectors_sorted[0]["sector"] if sectors_sorted else "Unknown"
            worst_sector = sectors_sorted[-1]["sector"] if sectors_sorted else "Unknown"
            
            # Calculate summary
            changes = [s.get("changePercent", 0) for s in sectors_data]
            positive_count = sum(1 for c in changes if c > 0)
            negative_count = sum(1 for c in changes if c < 0)
            
            summary = {
                "positive_sectors": positive_count,
                "negative_sectors": negative_count,
                "neutral_sectors": len(sectors_data) - positive_count - negative_count,
                "avg_change": sum(changes) / len(changes) if changes else 0,
                "market_sentiment": "bullish" if positive_count > negative_count else "bearish" if negative_count > positive_count else "neutral"
            }
            
            # Build result
            result = {
                "sectors": sectors_data,
                "sector_count": len(sectors_data),
                "date": date or datetime.now().strftime("%Y-%m-%d"),
                "best_sector": best_sector,
                "worst_sector": worst_sector,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: {len(sectors_data)} sectors "
                f"(best={best_sector}, worst={worst_sector}) ({execution_time:.0f}ms)"
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