"""
Finance Guru - FMP Data Service

Production-ready Financial Modeling Prep (FMP) API service for Finance Guru tools.
Follows the same patterns as existing tools in the codebase.

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

    Production-ready service for fetching financial data from FMP API.
    Follows the same patterns as existing tools (GetStockPriceTool, GetTechnicalIndicatorsTool, etc.)
    """

    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP Service.

        Args:
            api_key: FMP API key. If not provided, uses FMP_API_KEY env var.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        self.logger = logger

        if not self.api_key:
            raise ValueError(
                "FMP_API_KEY is required. "
                "Provide api_key parameter or set FMP_API_KEY environment variable."
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

        Raises:
            httpx.HTTPStatusError: On HTTP errors
        """
        params = params or {}
        params["apikey"] = self.api_key

        url = f"{self.FMP_BASE_URL}/{endpoint}"

        self.logger.debug(f"[FMP] Fetching from {url}")

        async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if not data:
                self.logger.warning(f"[FMP] No data returned from {endpoint}")
                return None

            return data

    # =========================================================================
    # COMPANY DATA
    # =========================================================================

    async def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company profile from FMP API.

        Endpoint: GET /v3/profile/{symbol}

        Args:
            symbol: Stock symbol

        Returns:
            Company profile data dict or None
        """
        try:
            data = await self._fetch(f"v3/profile/{symbol.upper()}")

            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info(f"[FMP] Got company profile for {symbol}")
                return data[0]

            return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching profile for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching profile for {symbol}: {e}")
            return None

    async def get_key_metrics(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get key financial metrics from FMP API.

        Endpoint: GET /v3/key-metrics/{symbol}

        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            Key metrics data dict or None
        """
        try:
            data = await self._fetch(
                f"v3/key-metrics/{symbol.upper()}",
                params={"period": period, "limit": limit}
            )

            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info(f"[FMP] Got key metrics for {symbol}")
                return data[0]

            return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching key metrics for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching key metrics for {symbol}: {e}")
            return None

    async def get_income_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get income statement from FMP API.

        Endpoint: GET /v3/income-statement/{symbol}

        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            Income statement data dict or None
        """
        try:
            data = await self._fetch(
                f"v3/income-statement/{symbol.upper()}",
                params={"period": period, "limit": limit}
            )

            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info(f"[FMP] Got income statement for {symbol}")
                return data[0]

            return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching income statement for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching income statement for {symbol}: {e}")
            return None

    async def get_cash_flow_statement(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cash flow statement from FMP API.

        Endpoint: GET /v3/cash-flow-statement/{symbol}

        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            Cash flow data dict or None
        """
        try:
            data = await self._fetch(
                f"v3/cash-flow-statement/{symbol.upper()}",
                params={"period": period, "limit": limit}
            )

            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info(f"[FMP] Got cash flow statement for {symbol}")
                return data[0]

            return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching cash flow for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching cash flow for {symbol}: {e}")
            return None

    async def get_balance_sheet(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get balance sheet from FMP API.

        Endpoint: GET /v3/balance-sheet-statement/{symbol}

        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            Balance sheet data dict or None
        """
        try:
            data = await self._fetch(
                f"v3/balance-sheet-statement/{symbol.upper()}",
                params={"period": period, "limit": limit}
            )

            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info(f"[FMP] Got balance sheet for {symbol}")
                return data[0]

            return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching balance sheet for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching balance sheet for {symbol}: {e}")
            return None

    # =========================================================================
    # PRICE DATA
    # =========================================================================

    async def get_historical_price(
        self,
        symbol: str,
        days: int = 365,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical daily price data from FMP API.

        Endpoint: GET /v3/historical-price-full/{symbol}

        Following the same pattern as GetTechnicalIndicatorsTool._fetch_historical_data()

        Args:
            symbol: Stock symbol
            days: Number of days of history (timeseries parameter)

        Returns:
            List of dicts with date, open, high, low, close, volume
            (FMP API format - newest first)

        Raises:
            ValueError: If insufficient data returned
        """
        try:
            url = f"{self.FMP_BASE_URL}/v3/historical-price-full/{symbol.upper()}"
            params = {"apikey": self.api_key, "timeseries": days}

            async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()

                data = response.json()
                historical = data.get("historical", [])

                if historical:
                    self.logger.info(
                        f"[FMP] Got {len(historical)} days of historical data for {symbol}"
                    )
                    return historical

                self.logger.warning(f"[FMP] No historical data returned for {symbol}")
                return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching historical data for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching historical data for {symbol}: {e}")
            return None

    async def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote from FMP API.

        Endpoint: GET /v3/quote/{symbol}

        Following the same pattern as GetStockPriceTool._fetch_from_fmp()

        Args:
            symbol: Stock symbol

        Returns:
            Quote data dict or None
        """
        try:
            data = await self._fetch(f"v3/quote/{symbol.upper()}")

            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info(f"[FMP] Got quote for {symbol}")
                return data[0]
            elif data and isinstance(data, dict):
                return data

            return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching quote for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching quote for {symbol}: {e}")
            return None

    # =========================================================================
    # RATIOS & METRICS
    # =========================================================================

    async def get_financial_ratios(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get financial ratios from FMP API.

        Endpoint: GET /v3/ratios/{symbol}

        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            Financial ratios data dict or None
        """
        try:
            data = await self._fetch(
                f"v3/ratios/{symbol.upper()}",
                params={"period": period, "limit": limit}
            )

            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info(f"[FMP] Got financial ratios for {symbol}")
                return data[0]

            return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching ratios for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching ratios for {symbol}: {e}")
            return None

    async def get_enterprise_value(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Get enterprise value from FMP API.

        Endpoint: GET /v3/enterprise-values/{symbol}

        Args:
            symbol: Stock symbol
            period: "annual" or "quarter"
            limit: Number of periods

        Returns:
            Enterprise value data dict or None
        """
        try:
            data = await self._fetch(
                f"v3/enterprise-values/{symbol.upper()}",
                params={"period": period, "limit": limit}
            )

            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info(f"[FMP] Got enterprise value for {symbol}")
                return data[0]

            return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching enterprise value for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching enterprise value for {symbol}: {e}")
            return None

    async def get_dcf(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get DCF valuation from FMP API.

        Endpoint: GET /v3/discounted-cash-flow/{symbol}

        Args:
            symbol: Stock symbol

        Returns:
            DCF data dict or None
        """
        try:
            data = await self._fetch(f"v3/discounted-cash-flow/{symbol.upper()}")

            if data and isinstance(data, list) and len(data) > 0:
                self.logger.info(f"[FMP] Got DCF for {symbol}")
                return data[0]
            elif data and isinstance(data, dict):
                return data

            return None

        except httpx.HTTPStatusError as e:
            self.logger.error(f"[FMP] HTTP error fetching DCF for {symbol}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[FMP] Error fetching DCF for {symbol}: {e}")
            return None
