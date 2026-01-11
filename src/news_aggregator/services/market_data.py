"""
Market Data Service
===================

Fetches market data (quotes, historical prices) from FMP API.

Provides:
- Current price and quote data
- Historical prices for change calculations (24h, 7d, 30d)
- Support for stocks, crypto, ETFs, forex

Endpoints used:
- /stable/quote?symbol=TSLA,NVDA,PLTR
- /stable/historical-price-full/{symbol}?timeseries=30
- /stable/crypto/quote?symbol=BTCUSD,ETHUSD

Usage:
    service = MarketDataService()
    data = await service.get_market_data(["TSLA", "BTC", "NVDA"])
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from src.news_aggregator.schemas.task import (
    MarketData,
    PriceChange,
    SymbolType,
)

# Initialize logger
logger = logging.getLogger(__name__)


# Known crypto symbols
CRYPTO_SYMBOLS = {
    "BTC", "ETH", "DOGE", "XRP", "SOL", "ADA", "DOT", "AVAX", "MATIC", "LINK",
    "LTC", "UNI", "ATOM", "XLM", "ALGO", "FIL", "VET", "THETA", "AAVE", "EOS",
    "BTCUSD", "ETHUSD", "DOGEUSD", "XRPUSD", "SOLUSD",
}


class MarketDataService:
    """
    Service for fetching market data from FMP API.

    Supports:
    - Stock quotes (TSLA, NVDA, AAPL, etc.)
    - Crypto quotes (BTC, ETH, etc.)
    - Historical data for change calculations

    Note: FMP uses different URL patterns for different endpoints:
    - /stable/ for quotes: https://financialmodelingprep.com/stable/quote
    - /api/v3/ for historical: https://financialmodelingprep.com/api/v3/historical-price-full
    """

    BASE_URL = "https://financialmodelingprep.com/stable"
    BASE_URL_V3 = "https://financialmodelingprep.com/api/v3"  # For historical data
    DEFAULT_TIMEOUT = 30.0

    def __init__(self, api_key: Optional[str] = None, timeout: float = None):
        """
        Initialize market data service.

        Args:
            api_key: FMP API key. Falls back to FMP_API_KEY env var.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        self.timeout = timeout or self.DEFAULT_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
        self.logger = logger

        if not self.api_key:
            self.logger.warning("[MarketData] FMP_API_KEY not set")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _classify_symbol(self, symbol: str) -> SymbolType:
        """Classify symbol type (stock, crypto, etc.)."""
        symbol_upper = symbol.upper().replace("-USD", "").replace("USD", "")

        if symbol_upper in CRYPTO_SYMBOLS or symbol.upper().endswith("USD"):
            return SymbolType.CRYPTO
        elif symbol.upper().endswith(".X"):
            return SymbolType.FOREX
        else:
            return SymbolType.STOCK

    def _normalize_crypto_symbol(self, symbol: str) -> str:
        """Normalize crypto symbol for FMP API."""
        symbol_upper = symbol.upper()
        if symbol_upper in CRYPTO_SYMBOLS and not symbol_upper.endswith("USD"):
            return f"{symbol_upper}USD"
        return symbol_upper

    async def _fetch_endpoint(
        self,
        endpoint: str,
        params: Dict[str, Any] = None,
        use_v3: bool = False,
    ) -> Optional[Any]:
        """
        Fetch data from FMP endpoint.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            use_v3: Use /api/v3/ base URL (required for historical endpoints)

        Returns:
            JSON response or None on error
        """
        if not self.api_key:
            return None

        client = await self._get_client()
        base_url = self.BASE_URL_V3 if use_v3 else self.BASE_URL
        url = f"{base_url}/{endpoint}"

        # Add API key to params
        params = params or {}
        params["apikey"] = self.api_key

        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            self.logger.error(f"[MarketData] HTTP error {e.response.status_code}: {endpoint}")
            return None
        except Exception as e:
            self.logger.error(f"[MarketData] Request error: {e}")
            return None

    async def get_stock_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for stock symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            Dict mapping symbol to quote data
        """
        if not symbols:
            return {}

        symbols_str = ",".join(symbols)
        self.logger.debug(f"[MarketData] Fetching stock quotes: {symbols_str}")

        data = await self._fetch_endpoint("quote", {"symbol": symbols_str})

        if not data or not isinstance(data, list):
            return {}

        result = {}
        for item in data:
            symbol = item.get("symbol", "")
            if symbol:
                result[symbol] = item

        return result

    async def get_crypto_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get quotes for crypto symbols.

        Args:
            symbols: List of crypto symbols (e.g., BTC, ETH, BTCUSD)

        Returns:
            Dict mapping symbol to quote data
        """
        if not symbols:
            return {}

        # Normalize crypto symbols
        normalized = [self._normalize_crypto_symbol(s) for s in symbols]
        symbols_str = ",".join(normalized)

        self.logger.debug(f"[MarketData] Fetching crypto quotes: {symbols_str}")

        data = await self._fetch_endpoint("crypto/quote", {"symbol": symbols_str})

        if not data or not isinstance(data, list):
            return {}

        result = {}
        for item in data:
            symbol = item.get("symbol", "")
            if symbol:
                # Map back to original symbol format
                original = symbol.replace("USD", "") if symbol.endswith("USD") else symbol
                result[original] = item
                result[symbol] = item  # Also store normalized version

        return result

    async def get_historical_prices(
        self,
        symbol: str,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get historical daily prices for a symbol.

        Args:
            symbol: Stock or crypto symbol
            days: Number of days of history

        Returns:
            List of daily price data (most recent first)
        """
        symbol_type = self._classify_symbol(symbol)

        if symbol_type == SymbolType.CRYPTO:
            endpoint = f"historical-price-full/{self._normalize_crypto_symbol(symbol)}"
        else:
            endpoint = f"historical-price-full/{symbol}"

        # Historical data requires /api/v3/ endpoint (not /stable/)
        data = await self._fetch_endpoint(endpoint, {"timeseries": days}, use_v3=True)

        if not data:
            return []

        # Handle different response formats
        if isinstance(data, list):
            return data[:days]
        elif isinstance(data, dict):
            historical = data.get("historical", [])
            return historical[:days] if isinstance(historical, list) else []

        return []

    def _calculate_change(
        self,
        current_price: float,
        historical: List[Dict[str, Any]],
        days: int,
    ) -> Optional[PriceChange]:
        """
        Calculate price change for a period.

        Args:
            current_price: Current price
            historical: Historical price data
            days: Number of days for the period

        Returns:
            PriceChange object or None
        """
        if not historical or len(historical) < days:
            return None

        try:
            # Get price from N days ago
            past_data = historical[min(days - 1, len(historical) - 1)]
            past_price = past_data.get("close") or past_data.get("price")

            if not past_price or past_price == 0:
                return None

            change_value = current_price - past_price
            change_percent = (change_value / past_price) * 100

            period_map = {1: "24h", 7: "7d", 30: "30d"}

            return PriceChange(
                period=period_map.get(days, f"{days}d"),
                change_percent=round(change_percent, 2),
                change_value=round(change_value, 2),
                start_price=round(past_price, 2),
                end_price=round(current_price, 2),
            )
        except Exception as e:
            self.logger.warning(f"[MarketData] Error calculating change: {e}")
            return None

    async def get_market_data(
        self,
        symbols: List[str],
        include_historical: bool = True,
    ) -> Dict[str, MarketData]:
        """
        Get complete market data for multiple symbols.

        Args:
            symbols: List of symbols (stocks and/or crypto)
            include_historical: Include 24h, 7d, 30d changes

        Returns:
            Dict mapping symbol to MarketData
        """
        if not symbols:
            return {}

        start_time = time.time()
        self.logger.info(f"[MarketData] Fetching data for {len(symbols)} symbols")

        # Classify symbols
        stock_symbols = []
        crypto_symbols = []

        for symbol in symbols:
            symbol_type = self._classify_symbol(symbol)
            if symbol_type == SymbolType.CRYPTO:
                crypto_symbols.append(symbol)
            else:
                stock_symbols.append(symbol)

        # Fetch quotes concurrently
        tasks = []
        if stock_symbols:
            tasks.append(self.get_stock_quotes(stock_symbols))
        if crypto_symbols:
            tasks.append(self.get_crypto_quotes(crypto_symbols))

        quote_results = await asyncio.gather(*tasks)

        # Combine quote results
        all_quotes = {}
        for result in quote_results:
            all_quotes.update(result)

        # Fetch historical data if needed
        historical_data = {}
        if include_historical:
            historical_tasks = []
            for symbol in symbols:
                historical_tasks.append(self.get_historical_prices(symbol, days=30))

            historical_results = await asyncio.gather(*historical_tasks)
            for symbol, history in zip(symbols, historical_results):
                historical_data[symbol] = history

        # Build MarketData objects
        result = {}
        for symbol in symbols:
            quote = all_quotes.get(symbol) or all_quotes.get(symbol.upper())
            if not quote:
                # Try normalized crypto symbol
                normalized = self._normalize_crypto_symbol(symbol)
                quote = all_quotes.get(normalized)

            if not quote:
                self.logger.warning(f"[MarketData] No quote data for {symbol}")
                continue

            current_price = quote.get("price", 0)
            if not current_price:
                continue

            symbol_type = self._classify_symbol(symbol)

            # Calculate changes
            changes = []
            if include_historical and symbol in historical_data:
                history = historical_data[symbol]
                for days in [1, 7, 30]:
                    change = self._calculate_change(current_price, history, days)
                    if change:
                        changes.append(change)

            # Create MarketData
            market_data = MarketData(
                symbol=symbol,
                symbol_type=symbol_type,
                current_price=round(current_price, 2),
                currency="USD",
                volume=quote.get("volume"),
                market_cap=quote.get("marketCap"),
                changes=changes,
                last_updated=datetime.utcnow(),
            )
            result[symbol] = market_data

        elapsed_ms = int((time.time() - start_time) * 1000)
        self.logger.info(
            f"[MarketData] Fetched {len(result)}/{len(symbols)} symbols ({elapsed_ms}ms)"
        )

        return result


# Singleton instance
_market_data_service: Optional[MarketDataService] = None


def get_market_data_service() -> MarketDataService:
    """Get singleton market data service instance."""
    global _market_data_service
    if _market_data_service is None:
        _market_data_service = MarketDataService()
    return _market_data_service
