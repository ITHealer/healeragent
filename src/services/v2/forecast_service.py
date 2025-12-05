# src/services/v2/forecast_service.py

import logging
from typing import Any, Dict, List, Optional

import httpx

from src.utils.config import settings
from src.utils.logger.set_up_log_dataFMP import setup_logger
from src.models.equity_forecast import (
    FMPPriceTargetConsensusItem,
    FMPPriceTargetSummaryItem,
    FMPAnalystEstimateItem,
    FMPRatingSnapshotItem,
    FMPFinancialScoreItem,
    FMPDiscountedCashFlowItem,
)

logger = setup_logger(__name__, log_level=logging.INFO)

FMP_API_KEY = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL
FMP_URL_STABLE = settings.FMP_URL_STABLE


class ForecastService:
    """Service for fetching forecast/price target/rating/score data from FMP"""

    def __init__(self) -> None:
        self._timeout: float = 10.0

    async def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if params is None:
            params = {}
        if FMP_API_KEY:
            params.setdefault("apikey", FMP_API_KEY)

        logger.debug(f"[ForecastService] GET {url} params={params}")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url, params=params)

            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(f"[ForecastService] HTTP {e.response.status_code}: {e}")
                raise

            try:
                return resp.json()
            except Exception as e:
                logger.error(f"[ForecastService] JSON parse error: {e}")
                raise

    # NEW: Get current quote
    async def get_current_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time stock quote for current price.
        Endpoint: /v3/quote/{symbol}
        """
        url = f"{BASE_FMP_URL}/v3/quote/{symbol}"
        raw = await self._get_json(url)
        
        if isinstance(raw, list) and len(raw) > 0:
            return raw[0]
        elif isinstance(raw, dict):
            return raw
        
        logger.warning(f"[ForecastService] No quote data for {symbol}")
        return None

    async def get_price_target_consensus(
        self,
        symbol: str,
    ) -> Optional[FMPPriceTargetConsensusItem]:
        """
        /stable/price-target-consensus?symbol=XXX
        Returns latest price target consensus
        """
        url = f"{FMP_URL_STABLE}/price-target-consensus"
        raw = await self._get_json(url, {"symbol": symbol})

        if isinstance(raw, list):
            if not raw:
                return None
            try:
                raw_sorted = sorted(
                    raw,
                    key=lambda x: (x.get("lastUpdated") or x.get("date") or ""),
                    reverse=True,
                )
                return FMPPriceTargetConsensusItem.model_validate(raw_sorted[0])
            except Exception as e:
                logger.warning(f"Sort error: {e}, using first element")
                return FMPPriceTargetConsensusItem.model_validate(raw[0])

        if isinstance(raw, dict):
            return FMPPriceTargetConsensusItem.model_validate(raw)

        logger.warning(f"Unexpected type for price-target-consensus: {type(raw)}")
        return None

    async def get_price_target_summary(
        self,
        symbol: str,
    ) -> Optional[List[FMPPriceTargetSummaryItem]]:
        """
        /stable/price-target-summary?symbol=XXX
        """
        url = f"{FMP_URL_STABLE}/price-target-summary"
        raw = await self._get_json(url, {"symbol": symbol})

        if isinstance(raw, list):
            return [FMPPriceTargetSummaryItem.model_validate(item) for item in raw]

        if isinstance(raw, dict):
            return [FMPPriceTargetSummaryItem.model_validate(raw)]

        logger.warning(f"Unexpected type for price-target-summary: {type(raw)}")
        return None

    async def get_analyst_estimates(
        self,
        symbol: str,
        period: str = "annual",
        limit: int = 4,
    ) -> Optional[List[FMPAnalystEstimateItem]]:
        """
        /stable/analyst-estimates?symbol=AAPL&period=annual&limit=4
        """
        url = f"{FMP_URL_STABLE}/analyst-estimates"
        params = {
            "symbol": symbol,
            "period": period,
            "limit": limit,
        }
        raw = await self._get_json(url, params)

        if isinstance(raw, list):
            return [FMPAnalystEstimateItem.model_validate(item) for item in raw]

        if isinstance(raw, dict):
            return [FMPAnalystEstimateItem.model_validate(raw)]

        logger.warning(f"Unexpected type for analyst-estimates: {type(raw)}")
        return None

    async def get_ratings_snapshot(
        self,
        symbol: str,
    ) -> Optional[List[FMPRatingSnapshotItem]]:
        """
        /stable/ratings-snapshot?symbol=XXX
        """
        url = f"{FMP_URL_STABLE}/ratings-snapshot"
        raw = await self._get_json(url, {"symbol": symbol})

        if isinstance(raw, list):
            return [FMPRatingSnapshotItem.model_validate(item) for item in raw]

        if isinstance(raw, dict):
            return [FMPRatingSnapshotItem.model_validate(raw)]

        logger.warning(f"Unexpected type for ratings-snapshot: {type(raw)}")
        return None

    async def get_financial_score(
        self,
        symbol: str,
    ) -> Optional[List[FMPFinancialScoreItem]]:
        """
        /v4/score?symbol=XXX
        """
        url = f"{BASE_FMP_URL}/v4/score"
        raw = await self._get_json(url, {"symbol": symbol})

        if isinstance(raw, list):
            return [FMPFinancialScoreItem.model_validate(item) for item in raw]

        if isinstance(raw, dict):
            return [FMPFinancialScoreItem.model_validate(raw)]

        logger.warning(f"Unexpected type for financial score: {type(raw)}")
        return None

    async def get_discounted_cash_flow(
        self,
        symbol: str,
    ) -> Optional[List[FMPDiscountedCashFlowItem]]:
        """
        /v3/discounted-cash-flow/{symbol}
        """
        url = f"{BASE_FMP_URL}/v3/discounted-cash-flow/{symbol}"
        raw = await self._get_json(url)

        if isinstance(raw, list):
            return [FMPDiscountedCashFlowItem.model_validate(item) for item in raw]

        if isinstance(raw, dict):
            return [FMPDiscountedCashFlowItem.model_validate(raw)]

        logger.warning(f"Unexpected type for discounted-cash-flow: {type(raw)}")
        return None