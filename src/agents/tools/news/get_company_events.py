# # """
# # GetCompanyEventsTool - FIXED with proper Redis cache pattern

# # Uses: src.helpers.redis_cache helpers
# # """

# # import httpx
# # import json
# # import logging
# # from typing import Dict, Any, Optional, List
# # from datetime import datetime, timedelta

# # from src.agents.tools.base import (
# #     BaseTool,
# #     ToolSchema,
# #     ToolParameter,
# #     create_success_output,
# #     create_error_output
# # )

# # from src.helpers.redis_cache import get_redis_client_llm


# # class GetCompanyEventsTool(BaseTool):
# #     """
# #     Atomic tool for fetching dividend events
    
# #     Category: news
# #     Data Source: FMP /stable/dividends-calendar
# #     Cache: Uses aioredis via get_redis_client_llm()
# #     """
    
# #     FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
# #     CACHE_TTL = 3600  # 1 hour
    
# #     def __init__(self, api_key: Optional[str] = None):
# #         super().__init__()
        
# #         if api_key is None:
# #             import os
# #             api_key = os.environ.get("FMP_API_KEY")
        
# #         if not api_key:
# #             raise ValueError("FMP_API_KEY required for GetCompanyEventsTool")
        
# #         self.api_key = api_key
# #         self.logger = logging.getLogger(__name__)
        
# #         # Define tool schema
# #         self.schema = ToolSchema(
# #             name="getCompanyEvents",
# #             category="news",
# #             description=(
# #                 "Fetch upcoming and recent company events (earnings calls, conferences, "
# #                 "dividends, splits, analyst days, etc.). "
# #                 "Use when user asks about company events, upcoming dates, or corporate actions."
# #             ),
# #             capabilities=[
# #                 "✅ Upcoming earnings calls",
# #                 "✅ Dividend dates and amounts",
# #                 "✅ Stock splits",
# #                 "✅ Investor conferences",
# #                 "✅ Product launches",
# #                 "✅ Shareholder meetings",
# #                 "✅ Event dates and details"
# #             ],
# #             limitations=[
# #                 "❌ Event dates can change",
# #                 "❌ Not all events are announced in advance",
# #                 "❌ Limited to public announcements",
# #                 "❌ One symbol at a time"
# #             ],
# #             usage_hints=[
# #                 # English
# #                 "User asks: 'Apple upcoming events' → USE THIS with symbol=AAPL",
# #                 "User asks: 'When is TSLA dividend?' → USE THIS with symbol=TSLA",
# #                 "User asks: 'NVDA stock split date' → USE THIS with symbol=NVDA",
# #                 "User asks: 'Microsoft investor events' → USE THIS with symbol=MSFT",
# #                 "User asks: 'Show me Amazon corporate calendar' → USE THIS with symbol=AMZN",
                
# #                 # Vietnamese
# #                 "User asks: 'Sự kiện sắp tới của Apple' → USE THIS with symbol=AAPL",
# #                 "User asks: 'Tesla có chia cổ tức không?' → USE THIS with symbol=TSLA",
# #                 "User asks: 'NVDA stock split' → USE THIS with symbol=NVDA",
                
# #                 # When NOT to use
# #                 "User asks for EARNINGS results → DO NOT USE (use getEarningsCalendar)",
# #                 "User asks about NEWS articles → DO NOT USE (use getStockNews)"
# #             ],
# #             parameters=[
# #                 ToolParameter(
# #                     name="symbol",
# #                     type="string",
# #                     description="Stock ticker symbol",
# #                     required=True,
# #                     pattern="^[A-Z]{1,7}$"
# #                 ),
# #                 ToolParameter(
# #                     name="event_types",
# #                     type="array",
# #                     description="Types of events to fetch",
# #                     required=False,
# #                     default=["all"]
# #                 ),
# #                 ToolParameter(
# #                     name="from_date",
# #                     type="string",
# #                     description="Start date for events (YYYY-MM-DD)",
# #                     required=False
# #                 )
# #             ],
# #             returns={
# #                 "symbol": "string",
# #                 "events": "array - Company events",
# #                 "event_count": "number",
# #                 "next_dividend": "object",
# #                 "next_earnings_call": "string",
# #                 "upcoming_events": "array",
# #                 "timestamp": "string"
# #             },
# #             typical_execution_time_ms=1100,
# #             requires_symbol=True
# #         )
    
# #     async def execute(
# #         self,
# #         symbol: str,
# #         event_types: List[str] = None,  # ← THÊM
# #         from_date: str = None,          # ← THÊM  
# #         lookback_days: int = 365,
# #         **kwargs
# #     ) -> Dict[str, Any]:
# #         """
# #         Execute company events (dividends) fetch with Redis cache
        
# #         Args:
# #             symbol: Stock symbol
# #             lookback_days: Days to look back (default: 365, max: 730)
            
# #         Returns:
# #             ToolOutput with dividend events
# #         """
# #         start_time = datetime.now()
# #         symbol = symbol.upper()
        
# #         # Validate lookback
# #         lookback_days = min(max(30, lookback_days), 730)
        
# #         self.logger.info(
# #             f"[getCompanyEvents] Executing: symbol={symbol}, lookback_days={lookback_days}"
# #         )
        
# #         try:
# #             # Calculate date range
# #             to_date = datetime.now().date()
# #             from_date = to_date - timedelta(days=lookback_days)
            
# #             # Build cache key
# #             cache_key = f"getCompanyEvents_{symbol}_{from_date}_{to_date}"
            
# #             # Get Redis client
# #             redis_client = await get_redis_client_llm()
            
# #             # Try cache first
# #             cached_data = None
# #             if redis_client:
# #                 try:
# #                     cached_bytes = await redis_client.get(cache_key)
# #                     if cached_bytes:
# #                         self.logger.info(f"[CACHE HIT] {cache_key}")
# #                         cached_data = json.loads(cached_bytes.decode('utf-8'))
# #                 except Exception as e:
# #                     self.logger.warning(f"[CACHE] Error reading: {e}")
            
# #             if cached_data:
# #                 # Return cached result
# #                 execution_time = (datetime.now() - start_time).total_seconds() * 1000
# #                 self.logger.info(
# #                     f"[getCompanyEvents] ✅ CACHED ({int(execution_time)}ms)"
# #                 )
                
# #                 return create_success_output(
# #                     tool_name="getCompanyEvents",
# #                     data=cached_data,
# #                     metadata={
# #                         "symbol": symbol,
# #                         "execution_time_ms": int(execution_time),
# #                         "lookback_days": lookback_days,
# #                         "from_cache": True
# #                     }
# #                 )
            
# #             # Fetch from API
# #             events_data = await self._fetch_dividends(from_date, to_date)
            
# #             if not events_data:
# #                 return create_error_output(
# #                     tool_name="getCompanyEvents",
# #                     error=f"No dividend data available for date range",
# #                     metadata={
# #                         "symbol": symbol,
# #                         "from_date": from_date.isoformat(),
# #                         "to_date": to_date.isoformat()
# #                     }
# #                 )
            
# #             # Filter by symbol
# #             symbol_events = [
# #                 item for item in events_data
# #                 if item.get("symbol", "").upper() == symbol
# #             ]
            
# #             if not symbol_events:
# #                 return create_error_output(
# #                     tool_name="getCompanyEvents",
# #                     error=f"No dividend events found for {symbol} in the last {lookback_days} days",
# #                     metadata={
# #                         "symbol": symbol,
# #                         "lookback_days": lookback_days,
# #                         "from_date": from_date.isoformat(),
# #                         "to_date": to_date.isoformat()
# #                     }
# #                 )
            
# #             # Format response
# #             result_data = self._format_events_data(
# #                 symbol_events,
# #                 symbol,
# #                 from_date,
# #                 to_date
# #             )
            
# #             # Cache the result
# #             if redis_client:
# #                 try:
# #                     json_string = json.dumps(result_data)
# #                     await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
# #                     self.logger.info(f"[CACHE SET] {cache_key} (TTL={self.CACHE_TTL}s)")
# #                 except Exception as e:
# #                     self.logger.warning(f"[CACHE] Error writing: {e}")
            
# #             # Close Redis connection
# #             if redis_client:
# #                 try:
# #                     await redis_client.close()
# #                 except Exception as e:
# #                     self.logger.debug(f"[CACHE] Error closing Redis: {e}")
            
# #             execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
# #             self.logger.info(
# #                 f"[getCompanyEvents] ✅ SUCCESS ({int(execution_time)}ms) - "
# #                 f"{result_data['event_count']} events"
# #             )
            
# #             return create_success_output(
# #                 tool_name="getCompanyEvents",
# #                 data=result_data,
# #                 metadata={
# #                     "symbol": symbol,
# #                     "execution_time_ms": int(execution_time),
# #                     "lookback_days": lookback_days,
# #                     "from_cache": False,
# #                     "event_count": result_data['event_count']
# #                 }
# #             )
            
# #         except Exception as e:
# #             self.logger.error(
# #                 f"[getCompanyEvents] Error for {symbol}: {e}",
# #                 exc_info=True
# #             )
            
# #             return create_error_output(
# #                 tool_name="getCompanyEvents",
# #                 error=str(e),
# #                 metadata={
# #                     "symbol": symbol,
# #                     "lookback_days": lookback_days
# #                 }
# #             )
    
# #     async def _fetch_dividends(
# #         self,
# #         from_date: datetime.date,
# #         to_date: datetime.date
# #     ) -> Optional[Any]:
# #         """Fetch dividends calendar from FMP Stable API"""
        
# #         url = f"{self.FMP_STABLE_BASE}/dividends-calendar"
# #         params = {
# #             "from": from_date.isoformat(),
# #             "to": to_date.isoformat(),
# #             "apikey": self.api_key
# #         }
        
# #         self.logger.info(f"[FMP] GET {url} with params: {params}")
        
# #         try:
# #             async with httpx.AsyncClient(timeout=15.0) as client:
# #                 response = await client.get(url, params=params)
                
# #                 if response.status_code != 200:
# #                     self.logger.error(
# #                         f"[FMP] HTTP {response.status_code}: {response.text[:200]}"
# #                     )
# #                     return None
                
# #                 data = response.json()
                
# #                 if isinstance(data, dict) and "Error Message" in data:
# #                     self.logger.error(f"[FMP] API Error: {data['Error Message']}")
# #                     return None
                
# #                 self.logger.info(
# #                     f"[FMP] ✅ Success: {len(data) if isinstance(data, list) else 1} items"
# #                 )
                
# #                 return data
                
# #         except httpx.TimeoutException:
# #             self.logger.error(f"[FMP] Timeout fetching dividends")
# #             return None
# #         except Exception as e:
# #             self.logger.error(f"[FMP] Error: {e}", exc_info=True)
# #             return None
    
# #     def _format_events_data(
# #         self,
# #         raw_data: list,
# #         symbol: str,
# #         from_date: datetime.date,
# #         to_date: datetime.date
# #     ) -> Dict[str, Any]:
# #         """Format dividend events data to structured output"""
        
# #         # Sort by date descending (most recent first)
# #         sorted_data = sorted(
# #             raw_data,
# #             key=lambda x: x.get("date", ""),
# #             reverse=True
# #         )
        
# #         # Parse dividend events
# #         events = []
# #         for item in sorted_data:
# #             event = {
# #                 "ex_dividend_date": item.get("date", ""),
# #                 "record_date": item.get("recordDate", ""),
# #                 "payment_date": item.get("paymentDate", ""),
# #                 "declaration_date": item.get("declarationDate", ""),
# #                 "dividend": item.get("dividend"),
# #                 "adj_dividend": item.get("adjDividend"),
# #                 "yield": item.get("yield"),
# #                 "frequency": item.get("frequency", "Unknown")
# #             }
# #             events.append(event)
        
# #         # Calculate summary statistics
# #         summary = self._calculate_dividend_summary(events)
        
# #         return {
# #             "symbol": symbol,
# #             "event_count": len(events),
# #             "from_date": from_date.isoformat(),
# #             "to_date": to_date.isoformat(),
# #             "events": events,
# #             "summary": summary,
# #             "timestamp": datetime.now().isoformat()
# #         }
    
# #     def _calculate_dividend_summary(self, events: list) -> Dict[str, Any]:
# #         """Calculate dividend summary statistics"""
        
# #         if not events:
# #             return {}
        
# #         # Extract dividends
# #         dividends = [
# #             e['dividend'] for e in events
# #             if e['dividend'] is not None
# #         ]
        
# #         if not dividends:
# #             return {}
        
# #         # Get most recent event
# #         recent_event = events[0]
        
# #         # Calculate total and average
# #         total_dividends = sum(dividends)
# #         avg_dividend = total_dividends / len(dividends)
        
# #         # Determine most common frequency
# #         frequencies = [e['frequency'] for e in events if e['frequency'] != "Unknown"]
# #         most_common_freq = max(set(frequencies), key=frequencies.count) if frequencies else "Unknown"
        
# #         return {
# #             "total_dividends": round(total_dividends, 4),
# #             "average_dividend": round(avg_dividend, 4),
# #             "payment_count": len(dividends),
# #             "most_recent_dividend": recent_event.get('dividend'),
# #             "most_recent_ex_date": recent_event.get('ex_dividend_date'),
# #             "most_recent_payment_date": recent_event.get('payment_date'),
# #             "most_recent_yield": recent_event.get('yield'),
# #             "frequency": most_common_freq
# #         }


# # # Standalone test
# # if __name__ == "__main__":
# #     import asyncio
# #     import os
    
# #     async def test():
# #         api_key = os.getenv("FMP_API_KEY")
# #         if not api_key:
# #             print("❌ FMP_API_KEY not set")
# #             return
        
# #         tool = GetCompanyEventsTool(api_key=api_key)
        
# #         print("\n" + "="*60)
# #         print("Testing getCompanyEvents Tool")
# #         print("="*60)
        
# #         # Test: AAPL dividend history
# #         print("\nTest: AAPL dividend history (last 365 days)")
# #         result = await tool.execute(symbol="AAPL", lookback_days=365)
        
# #         if result['status'] == 'success':
# #             data = result['data']
# #             print(f"✅ Success: {data['event_count']} dividend events")
# #             print(f"\nSummary:")
# #             print(f"  - Total Dividends: ${data['summary'].get('total_dividends', 'N/A')}")
# #             print(f"  - Average: ${data['summary'].get('average_dividend', 'N/A')}")
# #             print(f"  - Frequency: {data['summary'].get('frequency', 'N/A')}")
# #             print(f"  - Most Recent: ${data['summary'].get('most_recent_dividend', 'N/A')} on {data['summary'].get('most_recent_ex_date', 'N/A')}")
            
# #             print(f"\nRecent Events:")
# #             for i, event in enumerate(data['events'][:3], 1):
# #                 print(f"\n  Event {i}:")
# #                 print(f"  - Ex-Date: {event['ex_dividend_date']}")
# #                 print(f"  - Dividend: ${event['dividend']}")
# #                 print(f"  - Yield: {event['yield']}%")
# #         else:
# #             print(f"❌ Error: {result.get('error')}")
    
# #     asyncio.run(test())




# # File: src/agents/tools/news/get_company_events.py
# """
# GetCompanyEventsTool - FIXED: Renamed to reflect actual functionality (Dividends only)

# FMP API: /stable/dividends-calendar
# Returns: Dividend events only (ex-date, payment date, amount, yield)

# NOTE: This tool fetches DIVIDENDS only, not general company events.
# For earnings, use getEarningsCalendar.
# """

# import httpx
# import json
# import logging
# from typing import Dict, Any, Optional, List
# from datetime import datetime, timedelta

# from src.agents.tools.base import (
#     BaseTool,
#     ToolSchema,
#     ToolParameter,
#     create_success_output,
#     create_error_output
# )

# from src.helpers.redis_cache import get_redis_client_llm


# class GetCompanyEventsTool(BaseTool):
#     """
#     Atomic tool for fetching dividend events
    
#     Category: news
#     Data Source: FMP /stable/dividends-calendar
#     Cache: Redis with 1 hour TTL
    
#     ⚠️ IMPORTANT: This tool ONLY returns dividend events.
#     FMP's "dividends-calendar" endpoint does not include:
#     - Earnings calls
#     - Stock splits
#     - Investor conferences
#     - Product launches
    
#     For earnings: Use getEarningsCalendar
#     """
    
#     FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
#     CACHE_TTL = 3600  # 1 hour
    
#     def __init__(self, api_key: Optional[str] = None):
#         super().__init__()
        
#         if api_key is None:
#             import os
#             api_key = os.environ.get("FMP_API_KEY")
        
#         if not api_key:
#             raise ValueError("FMP_API_KEY required for GetCompanyEventsTool")
        
#         self.api_key = api_key
#         self.logger = logging.getLogger(__name__)
        
#         # Define tool schema
#         self.schema = ToolSchema(
#             name="getCompanyEvents",
#             category="news",
#             description=(
#                 "Fetch dividend history and upcoming dividend dates for a stock. "
#                 "Returns ex-dividend dates, payment dates, dividend amounts, and yields. "
#                 "⚠️ Note: This tool returns DIVIDENDS only, not general company events."
#             ),
#             capabilities=[
#                 "✅ Dividend payment history",
#                 "✅ Ex-dividend dates",
#                 "✅ Payment dates and amounts",
#                 "✅ Dividend yields",
#                 "✅ Payment frequency",
#                 "✅ Historical dividend trends"
#             ],
#             limitations=[
#                 "❌ ONLY dividends - no earnings, splits, or conferences",
#                 "❌ Limited to companies that pay dividends",
#                 "❌ Historical data only (no future predictions)",
#                 "❌ One symbol at a time"
#             ],
#             usage_hints=[
#                 # English
#                 "User asks: 'Apple dividend' → USE THIS with symbol=AAPL",
#                 "User asks: 'When does TSLA pay dividend?' → USE THIS with symbol=TSLA",
#                 "User asks: 'Microsoft dividend history' → USE THIS with symbol=MSFT",
#                 "User asks: 'NVDA dividend yield' → USE THIS with symbol=NVDA",
                
#                 # Vietnamese
#                 "User asks: 'Cổ tức Apple' → USE THIS with symbol=AAPL",
#                 "User asks: 'Tesla có chia cổ tức không?' → USE THIS with symbol=TSLA",
#                 "User asks: 'Lịch sử cổ tức Microsoft' → USE THIS with symbol=MSFT",
                
#                 # When NOT to use
#                 "User asks for EARNINGS → DO NOT USE (use getEarningsCalendar)",
#                 "User asks for STOCK SPLITS → DO NOT USE (not available in FMP free tier)",
#                 "User asks for CONFERENCES → DO NOT USE (not available)",
#                 "User asks for NEWS → DO NOT USE (use getStockNews)"
#             ],
#             parameters=[
#                 ToolParameter(
#                     name="symbol",
#                     type="string",
#                     description="Stock ticker symbol",
#                     required=True,
#                     pattern="^[A-Z]{1,7}$"
#                 ),
#                 ToolParameter(
#                     name="lookback_days",
#                     type="integer",
#                     description="Days of dividend history to fetch (default: 365, max: 730)",
#                     required=False,
#                     default=365,
#                     min_value=30,
#                     max_value=730
#                 )
#             ],
#             returns={
#                 "symbol": "string",
#                 "event_count": "number",
#                 "from_date": "string",
#                 "to_date": "string",
#                 "events": "array - Dividend events with ex-date, payment date, amount, yield",
#                 "summary": "object - Total dividends, average, frequency, most recent",
#                 "timestamp": "string"
#             },
#             typical_execution_time_ms=1100,
#             requires_symbol=True
#         )
    
#     async def execute(
#         self,
#         symbol: str,
#         lookback_days: int = 365
#     ) -> Dict[str, Any]:
#         """
#         Execute dividend events fetch with Redis cache
        
#         Args:
#             symbol: Stock symbol (e.g., AAPL, MSFT)
#             lookback_days: Days of history (default: 365, max: 730)
            
#         Returns:
#             ToolOutput with dividend events
            
#         FMP API Response Format:
#         [
#           {
#             "date": "2024-11-08",          # Ex-dividend date
#             "label": "November 08, 24",
#             "adjDividend": 0.25,
#             "symbol": "AAPL",
#             "dividend": 0.25,
#             "recordDate": "2024-11-11",
#             "paymentDate": "2024-11-14",
#             "declarationDate": "2024-10-31",
#             "yield": 0.46,                  # Dividend yield %
#             "frequency": "Quarterly"
#           },
#           ...
#         ]
#         """
#         start_time = datetime.now()
#         symbol = symbol.upper()
        
#         # Validate lookback_days
#         lookback_days = min(max(30, lookback_days), 730)
        
#         self.logger.info(
#             f"[getCompanyEvents] Executing: symbol={symbol}, lookback_days={lookback_days}"
#         )
        
#         try:
#             # Calculate date range
#             to_date = datetime.now().date()
#             from_date = to_date - timedelta(days=lookback_days)
            
#             # Build cache key
#             cache_key = f"dividends_{symbol}_{from_date}_{to_date}"
            
#             # Get Redis client
#             redis_client = await get_redis_client_llm()
            
#             # Try cache first
#             cached_data = None
#             if redis_client:
#                 try:
#                     cached_bytes = await redis_client.get(cache_key)
#                     if cached_bytes:
#                         self.logger.info(f"[CACHE HIT] {cache_key}")
#                         cached_data = json.loads(cached_bytes.decode('utf-8'))
#                 except Exception as e:
#                     self.logger.warning(f"[CACHE] Error reading: {e}")
            
#             if cached_data:
#                 # Return cached result
#                 execution_time = (datetime.now() - start_time).total_seconds() * 1000
#                 self.logger.info(
#                     f"[getCompanyEvents] ✅ CACHED ({int(execution_time)}ms)"
#                 )
                
#                 return create_success_output(
#                     tool_name="getCompanyEvents",
#                     data=cached_data,
#                     metadata={
#                         "symbol": symbol,
#                         "execution_time_ms": int(execution_time),
#                         "lookback_days": lookback_days,
#                         "from_cache": True
#                     }
#                 )
            
#             # Fetch from API
#             events_data = await self._fetch_dividends(from_date, to_date)
            
#             if not events_data:
#                 return create_error_output(
#                     tool_name="getCompanyEvents",
#                     error=f"No dividend data available for date range {from_date} to {to_date}",
#                     metadata={
#                         "symbol": symbol,
#                         "from_date": from_date.isoformat(),
#                         "to_date": to_date.isoformat()
#                     }
#                 )
            
#             # Filter by symbol
#             symbol_events = [
#                 item for item in events_data
#                 if item.get("symbol", "").upper() == symbol
#             ]
            
#             if not symbol_events:
#                 # Check if symbol doesn't pay dividends
#                 return create_error_output(
#                     tool_name="getCompanyEvents",
#                     error=(
#                         f"No dividend events found for {symbol} in the last {lookback_days} days. "
#                         f"This stock may not pay dividends (e.g., growth stocks like TSLA, NVDA historically). "
#                         f"Try a dividend-paying stock like AAPL, MSFT, or KO."
#                     ),
#                     metadata={
#                         "symbol": symbol,
#                         "lookback_days": lookback_days,
#                         "from_date": from_date.isoformat(),
#                         "to_date": to_date.isoformat(),
#                         "suggestion": "Use dividend-paying stocks: AAPL, MSFT, JNJ, KO, PG"
#                     }
#                 )
            
#             # Format response
#             result_data = self._format_events_data(
#                 symbol_events,
#                 symbol,
#                 from_date,
#                 to_date
#             )
            
#             # Cache the result
#             if redis_client:
#                 try:
#                     json_string = json.dumps(result_data)
#                     await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
#                     self.logger.info(f"[CACHE SET] {cache_key} (TTL={self.CACHE_TTL}s)")
#                 except Exception as e:
#                     self.logger.warning(f"[CACHE] Error writing: {e}")
            
#             # Close Redis connection
#             if redis_client:
#                 try:
#                     await redis_client.close()
#                 except Exception as e:
#                     self.logger.debug(f"[CACHE] Error closing Redis: {e}")
            
#             execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
#             self.logger.info(
#                 f"[getCompanyEvents] ✅ SUCCESS ({int(execution_time)}ms) - "
#                 f"{result_data['event_count']} dividend events"
#             )
            
#             return create_success_output(
#                 tool_name="getCompanyEvents",
#                 data=result_data,
#                 metadata={
#                     "symbol": symbol,
#                     "execution_time_ms": int(execution_time),
#                     "lookback_days": lookback_days,
#                     "from_cache": False,
#                     "event_count": result_data['event_count']
#                 }
#             )
            
#         except Exception as e:
#             self.logger.error(
#                 f"[getCompanyEvents] Error for {symbol}: {e}",
#                 exc_info=True
#             )
            
#             return create_error_output(
#                 tool_name="getCompanyEvents",
#                 error=str(e),
#                 metadata={
#                     "symbol": symbol,
#                     "lookback_days": lookback_days
#                 }
#             )
    
#     async def _fetch_dividends(
#         self,
#         from_date: datetime.date,
#         to_date: datetime.date
#     ) -> Optional[List[Dict[str, Any]]]:
#         """
#         Fetch dividends calendar from FMP Stable API
        
#         FMP Endpoint: GET /stable/dividends-calendar
#         Params:
#           - from: YYYY-MM-DD (start date)
#           - to: YYYY-MM-DD (end date)
#           - apikey: Your API key
          
#         Response: Array of dividend objects
#         """
        
#         url = f"{self.FMP_STABLE_BASE}/dividends-calendar"
#         params = {
#             "from": from_date.isoformat(),
#             "to": to_date.isoformat(),
#             "apikey": self.api_key
#         }
        
#         self.logger.debug(f"[FMP] GET {url}")
#         self.logger.debug(f"[FMP] Params: from={from_date}, to={to_date}")
        
#         try:
#             async with httpx.AsyncClient(timeout=15.0) as client:
#                 response = await client.get(url, params=params)
                
#                 if response.status_code != 200:
#                     self.logger.error(
#                         f"[FMP] HTTP {response.status_code}: {response.text[:200]}"
#                     )
#                     return None
                
#                 data = response.json()
                
#                 # Check for API errors
#                 if isinstance(data, dict) and "Error Message" in data:
#                     self.logger.error(f"[FMP] API Error: {data['Error Message']}")
#                     return None
                
#                 if not isinstance(data, list):
#                     self.logger.error(f"[FMP] Unexpected response format: {type(data)}")
#                     return None
                
#                 self.logger.info(f"[FMP] ✅ Success: {len(data)} dividend events total")
                
#                 return data
                
#         except httpx.TimeoutException:
#             self.logger.error(f"[FMP] Timeout fetching dividends")
#             return None
#         except Exception as e:
#             self.logger.error(f"[FMP] Error: {e}", exc_info=True)
#             return None
    
#     def _format_events_data(
#         self,
#         raw_data: List[Dict[str, Any]],
#         symbol: str,
#         from_date: datetime.date,
#         to_date: datetime.date
#     ) -> Dict[str, Any]:
#         """
#         Format dividend events data to structured output
        
#         Args:
#             raw_data: Raw dividend objects from FMP
#             symbol: Stock symbol
#             from_date: Query start date
#             to_date: Query end date
            
#         Returns:
#             Structured dividend data with summary stats
#         """
        
#         # Sort by ex-dividend date descending (most recent first)
#         sorted_data = sorted(
#             raw_data,
#             key=lambda x: x.get("date", ""),
#             reverse=True
#         )
        
#         # Parse dividend events
#         events = []
#         for item in sorted_data:
#             event = {
#                 "ex_dividend_date": item.get("date", ""),
#                 "record_date": item.get("recordDate", ""),
#                 "payment_date": item.get("paymentDate", ""),
#                 "declaration_date": item.get("declarationDate", ""),
#                 "dividend": item.get("dividend"),           # Regular dividend amount
#                 "adj_dividend": item.get("adjDividend"),     # Adjusted for splits
#                 "yield": item.get("yield"),                  # Dividend yield %
#                 "frequency": item.get("frequency", "Unknown"),
#                 "label": item.get("label", "")
#             }
#             events.append(event)
        
#         # Calculate summary statistics
#         summary = self._calculate_dividend_summary(events)
        
#         return {
#             "symbol": symbol,
#             "event_count": len(events),
#             "from_date": from_date.isoformat(),
#             "to_date": to_date.isoformat(),
#             "events": events,
#             "summary": summary,
#             "timestamp": datetime.now().isoformat()
#         }
    
#     def _calculate_dividend_summary(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Calculate dividend summary statistics
        
#         Args:
#             events: List of dividend events
            
#         Returns:
#             Summary with totals, averages, frequency, most recent
#         """
        
#         if not events:
#             return {
#                 "message": "No dividend events found",
#                 "total_dividends": 0,
#                 "payment_count": 0
#             }
        
#         # Extract dividends (using adj_dividend if available)
#         dividends = []
#         for e in events:
#             div = e.get('adj_dividend') or e.get('dividend')
#             if div is not None:
#                 try:
#                     dividends.append(float(div))
#                 except (ValueError, TypeError):
#                     continue
        
#         if not dividends:
#             return {
#                 "message": "Dividend amounts not available",
#                 "total_dividends": 0,
#                 "payment_count": 0
#             }
        
#         # Get most recent event
#         recent_event = events[0]
        
#         # Calculate statistics
#         total_dividends = sum(dividends)
#         avg_dividend = total_dividends / len(dividends)
        
#         # Determine most common frequency
#         frequencies = [e['frequency'] for e in events if e['frequency'] != "Unknown"]
#         most_common_freq = (
#             max(set(frequencies), key=frequencies.count) 
#             if frequencies else "Unknown"
#         )
        
#         # Extract recent yield
#         recent_yield = recent_event.get('yield')
#         try:
#             recent_yield_val = float(recent_yield) if recent_yield is not None else None
#         except (ValueError, TypeError):
#             recent_yield_val = None
        
#         return {
#             "total_dividends": round(total_dividends, 4),
#             "average_dividend": round(avg_dividend, 4),
#             "payment_count": len(dividends),
#             "most_recent_dividend": recent_event.get('dividend'),
#             "most_recent_ex_date": recent_event.get('ex_dividend_date'),
#             "most_recent_payment_date": recent_event.get('payment_date'),
#             "most_recent_yield": recent_yield_val,
#             "frequency": most_common_freq,
#             "annualized_estimate": round(avg_dividend * self._frequency_to_count(most_common_freq), 4)
#         }
    
#     def _frequency_to_count(self, frequency: str) -> int:
#         """Convert frequency string to annual count"""
#         freq_map = {
#             "Quarterly": 4,
#             "Monthly": 12,
#             "Semi-Annual": 2,
#             "Annual": 1,
#             "Unknown": 4  # Default to quarterly
#         }
#         return freq_map.get(frequency, 4)


# File: src/agents/tools/news/get_company_events.py
"""
GetCompanyEventsTool - FIXED with proper symbol matching

Uses: FMP /stable/dividends-calendar API
Cache: Redis via get_redis_client_llm()

FIXED:
- Multiple field name matching (symbol, ticker, Symbol, Ticker)
- Better debug logging for API response
- Graceful handling when symbol not found
"""

import httpx
import json
import logging
from typing import Dict, Any, Optional, List
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
                "User asks: 'Apple upcoming events' → USE THIS with symbol=AAPL",
                "User asks: 'When is TSLA dividend?' → USE THIS with symbol=TSLA",
                "User asks: 'NVDA stock split date' → USE THIS with symbol=NVDA",
                "User asks: 'Microsoft investor events' → USE THIS with symbol=MSFT",
                "User asks: 'Sự kiện sắp tới của Apple' → USE THIS with symbol=AAPL",
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
                    name="lookback_days",
                    type="integer",
                    description="Days to look back (default: 365)",
                    required=False,
                    default=365
                )
            ],
            returns={
                "symbol": "string",
                "events": "array - Company events",
                "event_count": "number",
                "summary": "object - Dividend summary",
                "timestamp": "string"
            },
            typical_execution_time_ms=1100,
            requires_symbol=True
        )
    
    def _get_symbol_from_item(self, item: Dict) -> str:
        """
        Extract symbol from item using multiple possible field names
        
        FMP API might use different field names:
        - symbol
        - ticker
        - Symbol
        - Ticker
        """
        # Try different field names (case variations)
        for field in ['symbol', 'ticker', 'Symbol', 'Ticker', 'SYMBOL', 'TICKER']:
            value = item.get(field)
            if value:
                return str(value).upper().strip()
        
        return ""
    
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
        symbol = symbol.upper().strip()
        
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
                    error=f"No dividend data available from FMP API",
                    metadata={
                        "symbol": symbol,
                        "from_date": from_date.isoformat(),
                        "to_date": to_date.isoformat()
                    }
                )
            
            # Debug: Log sample item structure
            if events_data and len(events_data) > 0:
                sample_item = events_data[0]
                self.logger.debug(
                    f"[getCompanyEvents] Sample item fields: {list(sample_item.keys())}"
                )
                self.logger.debug(
                    f"[getCompanyEvents] Sample item: {json.dumps(sample_item, default=str)[:500]}"
                )
            
            # Filter by symbol using multiple field names
            symbol_events = [
                item for item in events_data
                if self._get_symbol_from_item(item) == symbol
            ]
            
            self.logger.info(
                f"[getCompanyEvents] Filtered: {len(symbol_events)}/{len(events_data)} "
                f"events match symbol={symbol}"
            )
            
            # If no events found, check if it's a data issue
            if not symbol_events:
                # Get unique symbols in response for debugging
                unique_symbols = set()
                for item in events_data[:100]:  # Check first 100 items
                    sym = self._get_symbol_from_item(item)
                    if sym:
                        unique_symbols.add(sym)
                
                self.logger.warning(
                    f"[getCompanyEvents] No events for {symbol}. "
                    f"Sample symbols in response: {list(unique_symbols)[:10]}"
                )
                
                # Return empty success instead of error (no dividends is valid)
                result_data = {
                    "symbol": symbol,
                    "events": [],
                    "event_count": 0,
                    "summary": {
                        "total_dividends": 0,
                        "average_dividend": 0,
                        "payment_count": 0,
                        "frequency": "None",
                        "message": f"No dividend events found for {symbol} in the last {lookback_days} days"
                    },
                    "from_date": from_date.isoformat(),
                    "to_date": to_date.isoformat(),
                    "timestamp": datetime.now().isoformat()
                }
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Return success with empty events (not an error)
                return create_success_output(
                    tool_name="getCompanyEvents",
                    data=result_data,
                    formatted_context=(
                        f"📅 COMPANY EVENTS for {symbol}:\n"
                        f"No dividend events found in the last {lookback_days} days.\n"
                        f"This could mean the stock doesn't pay dividends or "
                        f"dividend dates are outside the search range."
                    ),
                    metadata={
                        "symbol": symbol,
                        "execution_time_ms": int(execution_time),
                        "lookback_days": lookback_days,
                        "from_cache": False,
                        "event_count": 0
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
            
            # Build formatted context
            formatted_context = self._build_formatted_context(result_data)
            
            return create_success_output(
                tool_name="getCompanyEvents",
                data=result_data,
                formatted_context=formatted_context,
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
        from_date,
        to_date
    ) -> Optional[List[Dict]]:
        """Fetch dividends calendar from FMP Stable API"""
        
        url = f"{self.FMP_STABLE_BASE}/dividends-calendar"
        params = {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "apikey": self.api_key
        }
        
        self.logger.info(f"[FMP] GET {url}")
        self.logger.debug(f"[FMP] Params: from={from_date}, to={to_date}")
        
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
                
                if isinstance(data, list):
                    self.logger.info(
                        f"[FMP] ✅ Success: {len(data)} dividend events total"
                    )
                    return data
                else:
                    self.logger.warning(f"[FMP] Unexpected response type: {type(data)}")
                    return None
                
        except httpx.TimeoutException:
            self.logger.error(f"[FMP] Timeout fetching dividends")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error: {e}", exc_info=True)
            return None
    
    def _format_events_data(
        self,
        events: List[Dict],
        symbol: str,
        from_date,
        to_date
    ) -> Dict[str, Any]:
        """Format dividend events into structured response"""
        
        # Sort by date (most recent first)
        sorted_events = sorted(
            events,
            key=lambda x: x.get('date', x.get('exDividendDate', x.get('ex_dividend_date', ''))),
            reverse=True
        )
        
        # Format each event
        formatted_events = []
        for event in sorted_events:
            formatted_event = {
                "ex_dividend_date": event.get('date', event.get('exDividendDate', '')),
                "payment_date": event.get('paymentDate', event.get('payment_date', '')),
                "record_date": event.get('recordDate', event.get('record_date', '')),
                "declaration_date": event.get('declarationDate', event.get('declaration_date', '')),
                "dividend": event.get('dividend', event.get('adjDividend', 0)),
                "yield": event.get('yield', event.get('dividendYield', 0)),
                "frequency": event.get('frequency', 'Unknown')
            }
            formatted_events.append(formatted_event)
        
        # Calculate summary
        summary = self._calculate_summary(formatted_events)
        
        return {
            "symbol": symbol,
            "events": formatted_events,
            "event_count": len(formatted_events),
            "summary": summary,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_summary(self, events: List[Dict]) -> Dict[str, Any]:
        """Calculate dividend summary statistics"""
        
        if not events:
            return {
                "total_dividends": 0,
                "average_dividend": 0,
                "payment_count": 0,
                "frequency": "None"
            }
        
        dividends = [e.get('dividend', 0) for e in events if e.get('dividend')]
        
        if not dividends:
            return {
                "total_dividends": 0,
                "average_dividend": 0,
                "payment_count": len(events),
                "frequency": "Unknown"
            }
        
        total_dividends = sum(dividends)
        avg_dividend = total_dividends / len(dividends) if dividends else 0
        
        # Get most recent event info
        recent_event = events[0] if events else {}
        
        # Determine frequency
        frequencies = [e.get('frequency', 'Unknown') for e in events if e.get('frequency')]
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
    
    def _build_formatted_context(self, result_data: Dict) -> str:
        """Build human-readable formatted context for LLM"""
        
        symbol = result_data.get('symbol', 'Unknown')
        event_count = result_data.get('event_count', 0)
        summary = result_data.get('summary', {})
        events = result_data.get('events', [])
        
        lines = [f"📅 COMPANY EVENTS for {symbol}:"]
        
        if event_count == 0:
            lines.append("No dividend events found in the search period.")
            return '\n'.join(lines)
        
        # Summary
        lines.append(f"\n📊 Dividend Summary:")
        lines.append(f"  - Total Payments: {summary.get('payment_count', 0)}")
        lines.append(f"  - Total Dividends: ${summary.get('total_dividends', 0):.4f}")
        lines.append(f"  - Average Dividend: ${summary.get('average_dividend', 0):.4f}")
        lines.append(f"  - Frequency: {summary.get('frequency', 'Unknown')}")
        
        if summary.get('most_recent_dividend'):
            lines.append(f"\n📌 Most Recent:")
            lines.append(f"  - Dividend: ${summary.get('most_recent_dividend', 0):.4f}")
            lines.append(f"  - Ex-Date: {summary.get('most_recent_ex_date', 'N/A')}")
            lines.append(f"  - Payment Date: {summary.get('most_recent_payment_date', 'N/A')}")
        
        # Recent events (max 5)
        if events:
            lines.append(f"\n📋 Recent Events (showing {min(5, len(events))} of {len(events)}):")
            for i, event in enumerate(events[:5], 1):
                lines.append(
                    f"  {i}. Ex-Date: {event.get('ex_dividend_date', 'N/A')} | "
                    f"Dividend: ${event.get('dividend', 0):.4f} | "
                    f"Payment: {event.get('payment_date', 'N/A')}"
                )
        
        return '\n'.join(lines)


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
        
        print(f"Status: {result.status}")
        if result.status == 'success':
            data = result.data
            print(f"✅ Success: {data.get('event_count', 0)} dividend events")
            print(f"\nFormatted Context:")
            print(result.formatted_context or "N/A")
        else:
            print(f"❌ Error: {result.error}")
    
    asyncio.run(test())