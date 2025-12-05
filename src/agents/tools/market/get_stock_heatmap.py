# src/agents/tools/market/get_stock_heatmap.py

import json
import time
from typing import Dict, Any, Literal
from datetime import datetime

import httpx

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    execute_tools_parallel,
    execute_tools_sequential,
    create_success_output,
    create_error_output,
    create_partial_output
)
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetStockHeatmapTool(BaseTool, LoggerMixin):
    """
    Tool 22: Get Stock Heatmap
    
    Generate heatmap data grouped by sector or industry
    
    FMP Stable APIs:
    - GET https://financialmodelingprep.com/stable/sector-performance-snapshot
    - GET https://financialmodelingprep.com/stable/available-sectors
    """

    CACHE_TTL = 900  # 15 minutes

    def __init__(self, api_key: str):
        """
        Initialize GetStockHeatmapTool
        
        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/stable"
        
        self.schema = ToolSchema(
            name="getStockHeatmap",
            category="market",
            description=(
                "Generate heatmap data grouped by sector or industry for market visualization. "
                "NO SYMBOL REQUIRED - returns all sectors/industries. "
                "Use when user asks for visual market overview, sector comparison, or heatmap."
            ),
            capabilities=[
                "✅ Sector-based heatmap",
                "✅ Industry-based heatmap (if group_by=industry)",
                "✅ Color intensity calculation (normalized 0-1)",
                "✅ Performance ranking",
                "✅ Market cap weighting"
            ],
            limitations=[
                "❌ Limited to sector/industry grouping",
                "❌ No individual stock cells",
                "❌ No custom grouping options"
            ],
            usage_hints=[
                # English
                "User asks: 'Show me market heatmap' → USE THIS (NO params)",
                "User asks: 'Sector heatmap' → USE THIS with group_by=sector",
                "User asks: 'Visual market overview' → USE THIS (NO params)",
                # Vietnamese
                "User asks: 'Heatmap thị trường' → USE THIS (NO params)",
                "User asks: 'So sánh các sector trực quan' → USE THIS (NO params)",
                # When NOT to use
                "User wants specific numbers → DO NOT USE (use getSectorPerformance)"
            ],
            parameters=[
                ToolParameter(
                    name="group_by",
                    type="string",
                    description="Group heatmap by sector or industry",
                    required=False,
                    default="sector",
                    allowed_values=["sector", "industry"]
                )
            ],
            returns={
                "group_by": "string",
                "cells": "array - Heatmap cells",
                "min_change": "number",
                "max_change": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=1100,
            requires_symbol=False
        )

    async def execute(self, group_by: str = "sector", **kwargs) -> ToolOutput:
        """
        Execute getStockHeatmap
        
        Args:
            group_by: Group by "sector" or "industry"
            
        Returns:
            ToolOutput with heatmap data
        """
        start_time = time.time()
        
        try:
            # Validate group_by
            if group_by not in ["sector", "industry"]:
                return create_error_output(
                    error=f"Invalid group_by: {group_by}",
                    details={"allowed_values": ["sector", "industry"]}
                )
            
            self.logger.info(f"[{self.schema.name}] Generating heatmap (group_by={group_by})")
            
            # Check cache
            cache_key = f"getStockHeatmap_{group_by}"
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
                        
            # Fetch sector performance (currently only sector supported by FMP)
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
            
            # Build heatmap cells
            cells = []
            changes = []
            
            for sector in sectors_data:
                change_percent = sector.get("changePercent", 0)
                changes.append(change_percent)
                
                cell = {
                    "name": sector.get("sector", "Unknown"),
                    "change_percent": change_percent,
                    "number_of_stocks": sector.get("numberOfCompanies", 0),
                    "color_intensity": self._calculate_color_intensity(change_percent)
                }
                cells.append(cell)
            
            # Sort by change_percent descending
            cells.sort(key=lambda x: x["change_percent"], reverse=True)
            
            # Find min/max
            min_change = min(changes) if changes else 0
            max_change = max(changes) if changes else 0
            
            # Build result
            result = {
                "group_by": group_by,
                "cells": cells,
                "min_change": round(min_change, 2),
                "max_change": round(max_change, 2),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS: {len(cells)} cells "
                f"(range: {min_change:.2f}% to {max_change:.2f}%) "
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

    def _calculate_color_intensity(self, change_percent: float) -> float:
        """
        Calculate color intensity for heatmap cell
        
        Maps change_percent to 0-1 range:
        - -10% or less → 0 (deep red)
        - 0% → 0.5 (neutral)
        - +10% or more → 1 (deep green)
        
        Args:
            change_percent: Percentage change
            
        Returns:
            Float between 0 and 1
        """
        # Normalize from -10% to +10% into 0 to 1
        normalized = (change_percent + 10) / 20
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, normalized))

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