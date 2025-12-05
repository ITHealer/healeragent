"""
GetCompanyEventsTool - FIXED with proper Redis cache pattern

Uses: src.helpers.redis_cache helpers
"""

import httpx
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    create_success_output,
    create_error_output
)

from src.helpers.redis_cache import get_redis_client_llm


class GetCompanyEventsTool(BaseTool):
    """
    Atomic tool for fetching dividend events
    
    Category: news
    Data Source: FMP /stable/dividends-calendar
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
            raise ValueError("FMP_API_KEY required for GetCompanyEventsTool")
        
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Define tool schema
        self.schema = ToolSchema(
            name="getCompanyEvents",
            category="news",
            description=(
                "Fetch upcoming and recent company events (earnings calls, conferences, "
                "dividends, splits, analyst days, etc.). "
                "Use when user asks about company events, upcoming dates, or corporate actions."
            ),
            capabilities=[
                "✅ Upcoming earnings calls",
                "✅ Dividend dates and amounts",
                "✅ Stock splits",
                "✅ Investor conferences",
                "✅ Product launches",
                "✅ Shareholder meetings",
                "✅ Event dates and details"
            ],
            limitations=[
                "❌ Event dates can change",
                "❌ Not all events are announced in advance",
                "❌ Limited to public announcements",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple upcoming events' → USE THIS with symbol=AAPL",
                "User asks: 'When is TSLA dividend?' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA stock split date' → USE THIS with symbol=NVDA",
                "User asks: 'Microsoft investor events' → USE THIS with symbol=MSFT",
                "User asks: 'Show me Amazon corporate calendar' → USE THIS with symbol=AMZN",
                
                # Vietnamese
                "User asks: 'Sự kiện sắp tới của Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Tesla có chia cổ tức không?' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA stock split' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks for EARNINGS results → DO NOT USE (use getEarningsCalendar)",
                "User asks about NEWS articles → DO NOT USE (use getStockNews)"
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
                    name="event_types",
                    type="array",
                    description="Types of events to fetch",
                    required=False,
                    default=["all"]
                ),
                ToolParameter(
                    name="from_date",
                    type="string",
                    description="Start date for events (YYYY-MM-DD)",
                    required=False
                )
            ],
            returns={
                "symbol": "string",
                "events": "array - Company events",
                "event_count": "number",
                "next_dividend": "object",
                "next_earnings_call": "string",
                "upcoming_events": "array",
                "timestamp": "string"
            },
            typical_execution_time_ms=1100,
            requires_symbol=True
        )
    
    async def execute(
        self,
        symbol: str,
        lookback_days: int = 365,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute company events (dividends) fetch with Redis cache
        
        Args:
            symbol: Stock symbol
            lookback_days: Days to look back (default: 365, max: 730)
            
        Returns:
            ToolOutput with dividend events
        """
        start_time = datetime.now()
        symbol = symbol.upper()
        
        # Validate lookback
        lookback_days = min(max(30, lookback_days), 730)
        
        self.logger.info(
            f"[getCompanyEvents] Executing: symbol={symbol}, lookback_days={lookback_days}"
        )
        
        try:
            # Calculate date range
            to_date = datetime.now().date()
            from_date = to_date - timedelta(days=lookback_days)
            
            # Build cache key
            cache_key = f"getCompanyEvents_{symbol}_{from_date}_{to_date}"
            
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
                    f"[getCompanyEvents] ✅ CACHED ({int(execution_time)}ms)"
                )
                
                return create_success_output(
                    tool_name="getCompanyEvents",
                    data=cached_data,
                    metadata={
                        "symbol": symbol,
                        "execution_time_ms": int(execution_time),
                        "lookback_days": lookback_days,
                        "from_cache": True
                    }
                )
            
            # Fetch from API
            events_data = await self._fetch_dividends(from_date, to_date)
            
            if not events_data:
                return create_error_output(
                    tool_name="getCompanyEvents",
                    error=f"No dividend data available for date range",
                    metadata={
                        "symbol": symbol,
                        "from_date": from_date.isoformat(),
                        "to_date": to_date.isoformat()
                    }
                )
            
            # Filter by symbol
            symbol_events = [
                item for item in events_data
                if item.get("symbol", "").upper() == symbol
            ]
            
            if not symbol_events:
                return create_error_output(
                    tool_name="getCompanyEvents",
                    error=f"No dividend events found for {symbol} in the last {lookback_days} days",
                    metadata={
                        "symbol": symbol,
                        "lookback_days": lookback_days,
                        "from_date": from_date.isoformat(),
                        "to_date": to_date.isoformat()
                    }
                )
            
            # Format response
            result_data = self._format_events_data(
                symbol_events,
                symbol,
                from_date,
                to_date
            )
            
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
                f"[getCompanyEvents] ✅ SUCCESS ({int(execution_time)}ms) - "
                f"{result_data['event_count']} events"
            )
            
            return create_success_output(
                tool_name="getCompanyEvents",
                data=result_data,
                metadata={
                    "symbol": symbol,
                    "execution_time_ms": int(execution_time),
                    "lookback_days": lookback_days,
                    "from_cache": False,
                    "event_count": result_data['event_count']
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[getCompanyEvents] Error for {symbol}: {e}",
                exc_info=True
            )
            
            return create_error_output(
                tool_name="getCompanyEvents",
                error=str(e),
                metadata={
                    "symbol": symbol,
                    "lookback_days": lookback_days
                }
            )
    
    async def _fetch_dividends(
        self,
        from_date: datetime.date,
        to_date: datetime.date
    ) -> Optional[Any]:
        """Fetch dividends calendar from FMP Stable API"""
        
        url = f"{self.FMP_STABLE_BASE}/dividends-calendar"
        params = {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
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
            self.logger.error(f"[FMP] Timeout fetching dividends")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error: {e}", exc_info=True)
            return None
    
    def _format_events_data(
        self,
        raw_data: list,
        symbol: str,
        from_date: datetime.date,
        to_date: datetime.date
    ) -> Dict[str, Any]:
        """Format dividend events data to structured output"""
        
        # Sort by date descending (most recent first)
        sorted_data = sorted(
            raw_data,
            key=lambda x: x.get("date", ""),
            reverse=True
        )
        
        # Parse dividend events
        events = []
        for item in sorted_data:
            event = {
                "ex_dividend_date": item.get("date", ""),
                "record_date": item.get("recordDate", ""),
                "payment_date": item.get("paymentDate", ""),
                "declaration_date": item.get("declarationDate", ""),
                "dividend": item.get("dividend"),
                "adj_dividend": item.get("adjDividend"),
                "yield": item.get("yield"),
                "frequency": item.get("frequency", "Unknown")
            }
            events.append(event)
        
        # Calculate summary statistics
        summary = self._calculate_dividend_summary(events)
        
        return {
            "symbol": symbol,
            "event_count": len(events),
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "events": events,
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_dividend_summary(self, events: list) -> Dict[str, Any]:
        """Calculate dividend summary statistics"""
        
        if not events:
            return {}
        
        # Extract dividends
        dividends = [
            e['dividend'] for e in events
            if e['dividend'] is not None
        ]
        
        if not dividends:
            return {}
        
        # Get most recent event
        recent_event = events[0]
        
        # Calculate total and average
        total_dividends = sum(dividends)
        avg_dividend = total_dividends / len(dividends)
        
        # Determine most common frequency
        frequencies = [e['frequency'] for e in events if e['frequency'] != "Unknown"]
        most_common_freq = max(set(frequencies), key=frequencies.count) if frequencies else "Unknown"
        
        return {
            "total_dividends": round(total_dividends, 4),
            "average_dividend": round(avg_dividend, 4),
            "payment_count": len(dividends),
            "most_recent_dividend": recent_event.get('dividend'),
            "most_recent_ex_date": recent_event.get('ex_dividend_date'),
            "most_recent_payment_date": recent_event.get('payment_date'),
            "most_recent_yield": recent_event.get('yield'),
            "frequency": most_common_freq
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
        
        tool = GetCompanyEventsTool(api_key=api_key)
        
        print("\n" + "="*60)
        print("Testing getCompanyEvents Tool")
        print("="*60)
        
        # Test: AAPL dividend history
        print("\nTest: AAPL dividend history (last 365 days)")
        result = await tool.execute(symbol="AAPL", lookback_days=365)
        
        if result['status'] == 'success':
            data = result['data']
            print(f"✅ Success: {data['event_count']} dividend events")
            print(f"\nSummary:")
            print(f"  - Total Dividends: ${data['summary'].get('total_dividends', 'N/A')}")
            print(f"  - Average: ${data['summary'].get('average_dividend', 'N/A')}")
            print(f"  - Frequency: {data['summary'].get('frequency', 'N/A')}")
            print(f"  - Most Recent: ${data['summary'].get('most_recent_dividend', 'N/A')} on {data['summary'].get('most_recent_ex_date', 'N/A')}")
            
            print(f"\nRecent Events:")
            for i, event in enumerate(data['events'][:3], 1):
                print(f"\n  Event {i}:")
                print(f"  - Ex-Date: {event['ex_dividend_date']}")
                print(f"  - Dividend: ${event['dividend']}")
                print(f"  - Yield: {event['yield']}%")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    asyncio.run(test())