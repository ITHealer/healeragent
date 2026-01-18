"""
Finance Guru - FMP Data Service

Provides Financial Modeling Prep (FMP) API data fetching for Finance Guru tools.
Handles API calls, caching, and fallback to mock data when API is unavailable.

Author: HealerAgent Development Team
"""

import os
import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class FMPService:
    """
    Financial Modeling Prep API Service.

    Provides methods to fetch financial data from FMP API.
    Falls back to mock data when API key is not available.
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    DEFAULT_TIMEOUT = 15.0

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP Service.

        Args:
            api_key: FMP API key. If not provided, uses FMP_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        self.logger = logger

        if not self.api_key:
            self.logger.warning(
                "FMP_API_KEY not available - will use mock data"
            )

    async def _fetch(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Fetch data from FMP API.

        Args:
            endpoint: API endpoint path (e.g., "v3/quote/AAPL")
            params: Optional query parameters

        Returns:
            API response data or None if error
        """
        if not self.api_key:
            return None

        params = params or {}
        params["apikey"] = self.api_key

        url = f"{self.FMP_BASE_URL}/{endpoint}"

        try:
            async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"[FMP] HTTP error {e.response.status_code} for {endpoint}"
            )
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Request error for {endpoint}: {e}")
            return None

    # =========================================================================
    # COMPANY DATA
    # =========================================================================

    async def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company profile.

        Args:
            symbol: Stock symbol

        Returns:
            Company profile data or mock data
        """
        data = await self._fetch(f"v3/profile/{symbol.upper()}")

        if data and isinstance(data, list) and len(data) > 0:
            return data[0]

        # Return mock data
        return self._mock_company_profile(symbol)

    async def get_key_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get key financial metrics.

        Args:
            symbol: Stock symbol

        Returns:
            Key metrics data or mock data
        """
        data = await self._fetch(f"v3/key-metrics/{symbol.upper()}")

        if data and isinstance(data, list) and len(data) > 0:
            return data[0]

        return self._mock_key_metrics(symbol)

    async def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get income statement.

        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            Income statement data or mock data
        """
        data = await self._fetch(
            f"v3/income-statement/{symbol.upper()}",
            params={"period": period, "limit": limit},
        )

        if data and isinstance(data, list) and len(data) > 0:
            return data[0]

        return self._mock_income_statement(symbol)

    async def get_cash_flow_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cash flow statement.

        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            Cash flow data or mock data
        """
        data = await self._fetch(
            f"v3/cash-flow-statement/{symbol.upper()}",
            params={"period": period, "limit": limit},
        )

        if data and isinstance(data, list) and len(data) > 0:
            return data[0]

        return self._mock_cash_flow(symbol)

    # =========================================================================
    # PRICE DATA
    # =========================================================================

    async def get_historical_price(
        self,
        symbol: str,
        days: int = 365,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical daily price data.

        Args:
            symbol: Stock symbol
            days: Number of days of history

        Returns:
            List of dicts with date, open, high, low, close, volume
            (FMP API format - newest first)
        """
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        data = await self._fetch(
            f"v3/historical-price-full/{symbol.upper()}",
            params={
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
        )

        if data and "historical" in data and len(data["historical"]) > 0:
            # Return raw FMP format (newest first)
            return data["historical"]

        return self._mock_historical_price(symbol, days)

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote.

        Args:
            symbol: Stock symbol

        Returns:
            Quote data or mock data
        """
        data = await self._fetch(f"v3/quote/{symbol.upper()}")

        if data and isinstance(data, list) and len(data) > 0:
            return data[0]

        return self._mock_quote(symbol)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _parse_date(self, date_str: str) -> date:
        """Parse date string to date object."""
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            return date.today()

    # =========================================================================
    # MOCK DATA
    # =========================================================================

    def _mock_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Generate mock company profile."""
        profiles = {
            "AAPL": {
                "symbol": "AAPL",
                "companyName": "Apple Inc.",
                "price": 185.0,
                "mktCap": 2850000000000,
                "sharesOutstanding": 15400000000,
                "cash": 62000000000,
                "totalDebt": 109000000000,
                "industry": "Consumer Electronics",
                "sector": "Technology",
            },
            "MSFT": {
                "symbol": "MSFT",
                "companyName": "Microsoft Corporation",
                "price": 380.0,
                "mktCap": 2830000000000,
                "sharesOutstanding": 7450000000,
                "cash": 104000000000,
                "totalDebt": 47000000000,
                "industry": "Software",
                "sector": "Technology",
            },
            "GOOGL": {
                "symbol": "GOOGL",
                "companyName": "Alphabet Inc.",
                "price": 140.0,
                "mktCap": 1750000000000,
                "sharesOutstanding": 12500000000,
                "cash": 118000000000,
                "totalDebt": 14000000000,
                "industry": "Internet Content & Information",
                "sector": "Communication Services",
            },
        }

        return profiles.get(symbol.upper(), {
            "symbol": symbol.upper(),
            "companyName": f"{symbol.upper()} Inc.",
            "price": 100.0,
            "mktCap": 100000000000,
            "sharesOutstanding": 1000000000,
            "cash": 10000000000,
            "totalDebt": 5000000000,
            "industry": "Technology",
            "sector": "Technology",
        })

    def _mock_key_metrics(self, symbol: str) -> Dict[str, Any]:
        """Generate mock key metrics."""
        metrics = {
            "AAPL": {
                "symbol": "AAPL",
                "eps": 6.15,
                "bookValuePerShare": 4.25,
                "revenuePerShare": 24.35,
                "dividendPerShare": 0.96,
                "peRatio": 30.0,
                "pbRatio": 43.5,
            },
            "MSFT": {
                "symbol": "MSFT",
                "eps": 11.05,
                "bookValuePerShare": 28.50,
                "revenuePerShare": 29.80,
                "dividendPerShare": 3.00,
                "peRatio": 34.4,
                "pbRatio": 13.3,
            },
        }

        return metrics.get(symbol.upper(), {
            "symbol": symbol.upper(),
            "eps": 5.0,
            "bookValuePerShare": 20.0,
            "revenuePerShare": 30.0,
            "dividendPerShare": 1.0,
            "peRatio": 20.0,
            "pbRatio": 5.0,
        })

    def _mock_income_statement(self, symbol: str) -> Dict[str, Any]:
        """Generate mock income statement."""
        return {
            "symbol": symbol.upper(),
            "revenue": 100000000000,
            "grossProfit": 40000000000,
            "operatingIncome": 25000000000,
            "netIncome": 20000000000,
            "eps": 5.0,
            "epsdiluted": 4.95,
        }

    def _mock_cash_flow(self, symbol: str) -> Dict[str, Any]:
        """Generate mock cash flow statement."""
        cash_flows = {
            "AAPL": {"freeCashFlow": 99000000000},
            "MSFT": {"freeCashFlow": 59000000000},
        }

        return cash_flows.get(symbol.upper(), {"freeCashFlow": 5000000000})

    def _mock_quote(self, symbol: str) -> Dict[str, Any]:
        """Generate mock quote."""
        return {
            "symbol": symbol.upper(),
            "price": 100.0,
            "change": 1.5,
            "changesPercentage": 1.52,
            "volume": 50000000,
        }

    def _mock_historical_price(
        self,
        symbol: str,
        days: int,
    ) -> List[Dict[str, Any]]:
        """Generate mock historical price data (FMP API format - newest first)."""
        import random

        base_price = 100.0
        historical = []
        today = date.today()

        # Generate data from oldest to newest
        for i in range(days):
            d = today - timedelta(days=days - i - 1)

            # Generate random but realistic price movement
            daily_return = random.gauss(0.0005, 0.02)  # ~0.05% mean, 2% std
            base_price = base_price * (1 + daily_return)

            open_price = base_price * random.uniform(0.99, 1.01)
            close_price = base_price * random.uniform(0.99, 1.01)
            high_price = max(open_price, close_price) * random.uniform(1.0, 1.02)
            low_price = min(open_price, close_price) * random.uniform(0.98, 1.0)
            volume = int(random.uniform(20000000, 100000000))

            historical.append({
                "date": d.isoformat(),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": volume,
            })

        # FMP returns newest first, so reverse
        return list(reversed(historical))
