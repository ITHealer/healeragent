"""
GetEarningsCalendarTool - FIXED with proper Redis cache pattern

Uses: src.helpers.redis_cache helpers
"""

import httpx
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    create_success_output,
    create_error_output
)

from src.helpers.redis_cache import get_redis_client_llm


class GetEarningsCalendarTool(BaseTool):
    """
    Atomic tool for fetching historical earnings reports
    
    Category: news
    Data Source: FMP /stable/earnings
    Cache: Uses aioredis via get_redis_client_llm()
    """
    
    FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
    CACHE_TTL = 3600  # 1 hour
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        
        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")
        
        if not api_key:
            raise ValueError("FMP_API_KEY required for GetEarningsCalendarTool")
        
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Define tool schema
        self.schema = ToolSchema(
            name="getEarningsCalendar",
            category="news",
            description=(
                "Fetch earnings calendar with upcoming and past earnings dates, "
                "actual vs expected results, and EPS surprises. "
                "Use when user asks about earnings date, earnings report, or earnings surprise."
            ),
            capabilities=[
                "✅ Next earnings date",
                "✅ Historical earnings dates",
                "✅ Actual vs expected EPS",
                "✅ EPS surprise percentage",
                "✅ Revenue vs estimates",
                "✅ Earnings call time",
                "✅ Past earnings performance"
            ],
            limitations=[
                "❌ Earnings dates can change",
                "❌ Estimates may vary by source",
                "❌ No real-time earnings data",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'When is Apple earnings?' → USE THIS with symbol=AAPL",
                "User asks: 'TSLA earnings date' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA earnings surprise' → USE THIS with symbol=NVDA",
                "User asks: 'Show me Microsoft earnings calendar' → USE THIS with symbol=MSFT",
                "User asks: 'Did Amazon beat estimates?' → USE THIS with symbol=AMZN",
                
                # Vietnamese
                "User asks: 'Apple báo cáo thu nhập khi nào?' → USE THIS with symbol=AAPL",
                "User asks: 'Ngày earnings của Tesla' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA có beat estimate không?' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks for INCOME statement → DO NOT USE (use getIncomeStatement)",
                "User asks about general NEWS → DO NOT USE (use getStockNews)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Number of earnings reports to return",
                    required=False,
                    default=4
                )
            ],
            returns={
                "symbol": "string",
                "next_earnings_date": "string",
                "earnings_history": "array",
                "eps_surprises": "array",
                "beat_rate": "number - Percentage of earnings beats",
                "avg_surprise": "number - Average EPS surprise %",
                "timestamp": "string"
            },
            typical_execution_time_ms=1000,
            requires_symbol=True
        )
            
    async def execute(
        self,
        symbol: str,
        limit: int = 8,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute earnings calendar fetch with Redis cache
        
        Args:
            symbol: Stock symbol
            limit: Number of historical earnings (default: 8, max: 20)
            
        Returns:
            ToolOutput with earnings history
        """
        start_time = datetime.now()
        symbol = symbol.upper()
        
        # Validate limit
        limit = min(max(1, limit), 20)
        
        self.logger.info(
            f"[getEarningsCalendar] Executing: symbol={symbol}, limit={limit}"
        )
        
        try:
            # Build cache key
            cache_key = f"getEarningsCalendar_{symbol}_{limit}"
            
            # Get Redis client
            redis_client = await get_redis_client_llm()
            
            # Try cache first
            cached_data = None
            if redis_client:
                try:
                    cached_bytes = await redis_client.get(cache_key)
                    if cached_bytes:
                        self.logger.info(f"[CACHE HIT] {cache_key}")
                        cached_data = json.loads(cached_bytes.decode('utf-8'))
                except Exception as e:
                    self.logger.warning(f"[CACHE] Error reading: {e}")
            
            if cached_data:
                # Return cached result
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.info(
                    f"[getEarningsCalendar] ✅ CACHED ({int(execution_time)}ms)"
                )
                
                return create_success_output(
                    tool_name="getEarningsCalendar",
                    data=cached_data,
                    metadata={
                        "symbol": symbol,
                        "execution_time_ms": int(execution_time),
                        "limit": limit,
                        "from_cache": True
                    }
                )
            
            # Fetch from API
            earnings_data = await self._fetch_earnings(symbol)
            
            if not earnings_data:
                return create_error_output(
                    tool_name="getEarningsCalendar",
                    error=f"No earnings data available for {symbol}",
                    metadata={
                        "symbol": symbol,
                        "limit": limit
                    }
                )
            
            # Format response
            result_data = self._format_earnings_data(earnings_data, symbol, limit)
            
            # Cache the result
            if redis_client:
                try:
                    json_string = json.dumps(result_data)
                    await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
                    self.logger.info(f"[CACHE SET] {cache_key} (TTL={self.CACHE_TTL}s)")
                except Exception as e:
                    self.logger.warning(f"[CACHE] Error writing: {e}")
            
            # Close Redis connection
            if redis_client:
                try:
                    await redis_client.close()
                except Exception as e:
                    self.logger.debug(f"[CACHE] Error closing Redis: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"[getEarningsCalendar] ✅ SUCCESS ({int(execution_time)}ms) - "
                f"{result_data['report_count']} reports"
            )
            
            return create_success_output(
                tool_name="getEarningsCalendar",
                data=result_data,
                metadata={
                    "symbol": symbol,
                    "execution_time_ms": int(execution_time),
                    "limit": limit,
                    "from_cache": False,
                    "report_count": result_data['report_count']
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[getEarningsCalendar] Error for {symbol}: {e}",
                exc_info=True
            )
            
            return create_error_output(
                tool_name="getEarningsCalendar",
                error=str(e),
                metadata={
                    "symbol": symbol,
                    "limit": limit
                }
            )
    
    async def _fetch_earnings(self, symbol: str) -> Optional[Any]:
        """Fetch earnings from FMP Stable API"""
        
        url = f"{self.FMP_STABLE_BASE}/earnings"
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        self.logger.info(f"[FMP] GET {url} with params: {params}")
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                
                if response.status_code != 200:
                    self.logger.error(
                        f"[FMP] HTTP {response.status_code}: {response.text[:200]}"
                    )
                    return None
                
                data = response.json()
                
                if isinstance(data, dict) and "Error Message" in data:
                    self.logger.error(f"[FMP] API Error: {data['Error Message']}")
                    return None
                
                self.logger.info(
                    f"[FMP] ✅ Success: {len(data) if isinstance(data, list) else 1} items"
                )
                
                return data
                
        except httpx.TimeoutException:
            self.logger.error(f"[FMP] Timeout fetching earnings")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error: {e}", exc_info=True)
            return None
    
    def _format_earnings_data(
        self,
        raw_data: Any,
        symbol: str,
        limit: int
    ) -> Dict[str, Any]:
        """Format earnings data to structured output"""
        
        if not isinstance(raw_data, list):
            raw_data = []
        
        # Sort by date descending (most recent first)
        sorted_data = sorted(
            raw_data,
            key=lambda x: x.get("date", ""),
            reverse=True
        )
        
        # Parse earnings reports
        reports = []
        for item in sorted_data[:limit]:
            eps_actual = item.get("epsActual")
            eps_estimated = item.get("epsEstimated")
            revenue_actual = item.get("revenueActual")
            revenue_estimated = item.get("revenueEstimated")
            
            # Calculate surprises
            eps_surprise = None
            eps_surprise_pct = None
            if eps_actual is not None and eps_estimated is not None:
                eps_surprise = round(eps_actual - eps_estimated, 4)
                if eps_estimated != 0:
                    eps_surprise_pct = round((eps_surprise / eps_estimated) * 100, 2)
            
            revenue_surprise = None
            revenue_surprise_pct = None
            if revenue_actual is not None and revenue_estimated is not None:
                revenue_surprise = revenue_actual - revenue_estimated
                if revenue_estimated != 0:
                    revenue_surprise_pct = round(
                        (revenue_surprise / revenue_estimated) * 100, 2
                    )
            
            report = {
                "date": item.get("date", ""),
                "eps_actual": eps_actual,
                "eps_estimated": eps_estimated,
                "eps_surprise": eps_surprise,
                "eps_surprise_pct": eps_surprise_pct,
                "revenue_actual": revenue_actual,
                "revenue_estimated": revenue_estimated,
                "revenue_surprise": revenue_surprise,
                "revenue_surprise_pct": revenue_surprise_pct,
                "last_updated": item.get("lastUpdated", "")
            }
            reports.append(report)
        
        # Calculate summary statistics
        summary = self._calculate_summary(reports)
        
        return {
            "symbol": symbol,
            "report_count": len(reports),
            "reports": reports,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_summary(self, reports: list) -> Dict[str, Any]:
        """Calculate summary statistics from earnings reports"""
        
        if not reports:
            return {}
        
        eps_surprises = [
            r['eps_surprise_pct'] for r in reports
            if r['eps_surprise_pct'] is not None
        ]
        
        revenue_surprises = [
            r['revenue_surprise_pct'] for r in reports
            if r['revenue_surprise_pct'] is not None
        ]
        
        # Count beats vs misses
        eps_beats = sum(1 for s in eps_surprises if s > 0)
        eps_misses = sum(1 for s in eps_surprises if s < 0)
        
        revenue_beats = sum(1 for s in revenue_surprises if s > 0)
        revenue_misses = sum(1 for s in revenue_surprises if s < 0)
        
        return {
            "eps_beat_rate": round(eps_beats / len(eps_surprises), 2) if eps_surprises else None,
            "eps_beats": eps_beats,
            "eps_misses": eps_misses,
            "avg_eps_surprise_pct": round(
                sum(eps_surprises) / len(eps_surprises), 2
            ) if eps_surprises else None,
            "revenue_beat_rate": round(
                revenue_beats / len(revenue_surprises), 2
            ) if revenue_surprises else None,
            "revenue_beats": revenue_beats,
            "revenue_misses": revenue_misses,
            "avg_revenue_surprise_pct": round(
                sum(revenue_surprises) / len(revenue_surprises), 2
            ) if revenue_surprises else None
        }


# Standalone test
if __name__ == "__main__":
    import asyncio
    import os
    
    async def test():
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            print("❌ FMP_API_KEY not set")
            return
        
        tool = GetEarningsCalendarTool(api_key=api_key)
        
        print("\n" + "="*60)
        print("Testing getEarningsCalendar Tool")
        print("="*60)
        
        # Test: NVDA earnings history
        print("\nTest: NVDA earnings history")
        result = await tool.execute(symbol="NVDA", limit=5)
        
        if result['status'] == 'success':
            data = result['data']
            print(f"✅ Success: {data['report_count']} reports")
            print(f"\nSummary:")
            print(f"  - EPS Beat Rate: {data['summary'].get('eps_beat_rate', 'N/A')}")
            print(f"  - Avg EPS Surprise: {data['summary'].get('avg_eps_surprise_pct', 'N/A')}%")
            
            print(f"\nRecent Reports:")
            for i, report in enumerate(data['reports'][:3], 1):
                print(f"\n  Report {i} ({report['date']}):")
                print(f"  - EPS: ${report['eps_actual']} vs ${report['eps_estimated']} (surprise: {report['eps_surprise_pct']}%)")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    asyncio.run(test())