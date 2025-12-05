"""
News & Events Tools Package - FIXED VERSION

Uses proper Redis cache pattern with aioredis
"""

from src.agents.tools.news.get_stock_news import GetStockNewsTool
from src.agents.tools.news.get_earnings_calendar import GetEarningsCalendarTool
from src.agents.tools.news.get_company_events import GetCompanyEventsTool

__all__ = [
    "GetStockNewsTool",
    "GetEarningsCalendarTool",
    "GetCompanyEventsTool"
]