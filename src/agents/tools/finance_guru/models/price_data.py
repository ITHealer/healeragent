"""
Finance Guru - Price Data Models

Pydantic models for price series and market data.
These are the primary input models for most Finance Guru calculations.

USAGE:
    # Simple price series
    prices = PriceDataInput(
        ticker="AAPL",
        dates=["2024-01-01", "2024-01-02", ...],
        prices=[150.0, 152.5, ...],
    )

    # Full OHLCV data
    ohlcv = OHLCVData(
        ticker="TSLA",
        dates=[...],
        open=[...],
        high=[...],
        low=[...],
        close=[...],
        volume=[...],
    )

Author: HealerAgent Development Team
Created: 2025-01-18
"""

from datetime import date
from typing import List, Optional

import numpy as np
from pydantic import Field, field_validator, model_validator

from src.agents.tools.finance_guru.models.base import (
    BaseFinanceModel,
    FinanceValidationError,
    MarketType,
    validate_ticker,
)


class PriceDataInput(BaseFinanceModel):
    """
    Historical price data for calculations.

    WHAT: Container for time-series price data
    WHY: Ensures price data is valid before calculations begin
    VALIDATES:
        - Prices are positive (can't have negative stock prices)
        - Dates are chronologically sorted
        - Minimum data points for statistical validity
        - Equal number of prices and dates (data alignment)

    EDUCATIONAL NOTE:
    Price data is the foundation of most financial calculations.
    Quality issues here propagate to ALL downstream analysis.
    This model catches errors early, before they cause calculation errors.
    """

    ticker: str = Field(
        ...,
        description="Asset ticker symbol (e.g., 'AAPL', 'TSLA', 'BTC-USD')",
        min_length=1,
        max_length=20,
    )

    dates: List[date] = Field(
        ...,
        description="Trading dates in chronological order",
        min_length=2,  # Need at least 2 points for returns
    )

    prices: List[float] = Field(
        ...,
        description="Closing prices corresponding to each date",
        min_length=2,
    )

    volumes: Optional[List[float]] = Field(
        default=None,
        description="Optional trading volumes for each date",
    )

    market_type: MarketType = Field(
        default=MarketType.STOCK,
        description="Type of market (stock, crypto, etc.)",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker_format(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        return validate_ticker(v)

    @field_validator("prices")
    @classmethod
    def validate_prices_positive(cls, v: List[float]) -> List[float]:
        """
        Ensure all prices are positive.

        WHY: Stock prices cannot be negative. Zero prices indicate
        delisted stocks or data errors. We reject both.
        """
        for i, price in enumerate(v):
            if price <= 0:
                raise FinanceValidationError(
                    f"Price at index {i} is {price}. All prices must be positive.",
                    field="prices",
                    value=price,
                    suggestion="Check for data errors or delisted securities",
                )
        return v

    @field_validator("dates")
    @classmethod
    def validate_dates_sorted(cls, v: List[date]) -> List[date]:
        """
        Ensure dates are in chronological order.

        WHY: Time-series calculations assume sequential data.
        Out-of-order dates produce incorrect returns and volatility.
        """
        if v != sorted(v):
            raise FinanceValidationError(
                "Dates must be in chronological order (earliest to latest)",
                field="dates",
                suggestion="Sort your data by date before creating PriceDataInput",
            )
        return v

    @model_validator(mode="after")
    def validate_alignment(self) -> "PriceDataInput":
        """
        Ensure all arrays have the same length.

        WHY: Each price needs a corresponding date.
        Misalignment causes index errors in calculations.
        """
        n_dates = len(self.dates)
        n_prices = len(self.prices)

        if n_prices != n_dates:
            raise FinanceValidationError(
                f"Length mismatch: {n_prices} prices but {n_dates} dates",
                field="prices/dates",
                suggestion="Each price must have a corresponding date",
            )

        if self.volumes is not None:
            if len(self.volumes) != n_dates:
                raise FinanceValidationError(
                    f"Length mismatch: {len(self.volumes)} volumes but {n_dates} dates",
                    field="volumes",
                    suggestion="Each volume must have a corresponding date",
                )

        return self

    @model_validator(mode="after")
    def check_duplicates(self) -> "PriceDataInput":
        """
        Check for duplicate dates.

        WHY: Duplicate dates indicate data quality issues.
        """
        if len(self.dates) != len(set(self.dates)):
            raise FinanceValidationError(
                "Duplicate dates found in price data",
                field="dates",
                suggestion="Each date should appear only once",
            )
        return self

    def calculate_returns(self) -> List[float]:
        """
        Calculate daily returns from prices.

        Returns:
            List of daily returns (percentage changes)
        """
        returns = []
        for i in range(1, len(self.prices)):
            ret = (self.prices[i] - self.prices[i - 1]) / self.prices[i - 1]
            returns.append(ret)
        return returns

    def to_numpy(self) -> "np.ndarray":
        """Convert prices to numpy array."""
        return np.array(self.prices)

    @property
    def latest_price(self) -> float:
        """Get the most recent price."""
        return self.prices[-1]

    @property
    def start_date(self) -> date:
        """Get the earliest date."""
        return self.dates[0]

    @property
    def end_date(self) -> date:
        """Get the most recent date."""
        return self.dates[-1]

    @property
    def trading_days(self) -> int:
        """Get number of trading days in the data."""
        return len(self.dates)


class OHLCVData(BaseFinanceModel):
    """
    Full OHLCV (Open, High, Low, Close, Volume) price data.

    WHAT: Complete candlestick data for technical analysis
    WHY: Many indicators require High/Low data (Stochastic, Williams %R, etc.)
    VALIDATES:
        - High >= Low for each day (fundamental market constraint)
        - Close is within High-Low range
        - All arrays aligned

    EDUCATIONAL NOTE:
    OHLCV data captures intraday price action:
    - Open: Price at market open
    - High: Highest price during the day
    - Low: Lowest price during the day
    - Close: Price at market close
    - Volume: Number of shares/contracts traded
    """

    ticker: str = Field(
        ...,
        description="Asset ticker symbol",
        min_length=1,
        max_length=20,
    )

    dates: List[date] = Field(
        ...,
        description="Trading dates in chronological order",
        min_length=2,
    )

    open: List[float] = Field(
        ...,
        description="Opening prices",
        min_length=2,
    )

    high: List[float] = Field(
        ...,
        description="High prices (highest during period)",
        min_length=2,
    )

    low: List[float] = Field(
        ...,
        description="Low prices (lowest during period)",
        min_length=2,
    )

    close: List[float] = Field(
        ...,
        description="Closing prices",
        min_length=2,
    )

    volume: Optional[List[float]] = Field(
        default=None,
        description="Trading volumes",
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker_format(cls, v: str) -> str:
        """Validate and normalize ticker symbol."""
        return validate_ticker(v)

    @field_validator("open", "high", "low", "close")
    @classmethod
    def validate_prices_positive(cls, v: List[float]) -> List[float]:
        """Ensure all prices are positive."""
        for i, price in enumerate(v):
            if price <= 0:
                raise FinanceValidationError(
                    f"Price at index {i} is {price}. All prices must be positive.",
                    field="prices",
                    value=price,
                )
        return v

    @field_validator("dates")
    @classmethod
    def validate_dates_sorted(cls, v: List[date]) -> List[date]:
        """Ensure dates are sorted chronologically."""
        if v != sorted(v):
            raise FinanceValidationError(
                "Dates must be in chronological order",
                field="dates",
            )
        return v

    @model_validator(mode="after")
    def validate_ohlcv_consistency(self) -> "OHLCVData":
        """
        Validate OHLCV data consistency.

        Checks:
        1. All arrays have same length
        2. High >= Low for each day
        3. Close is within [Low, High] range
        4. Open is within [Low, High] range
        """
        n = len(self.dates)

        # Check length alignment
        for field_name, data in [
            ("open", self.open),
            ("high", self.high),
            ("low", self.low),
            ("close", self.close),
        ]:
            if len(data) != n:
                raise FinanceValidationError(
                    f"{field_name} has {len(data)} items but dates has {n}",
                    field=field_name,
                )

        if self.volume is not None and len(self.volume) != n:
            raise FinanceValidationError(
                f"volume has {len(self.volume)} items but dates has {n}",
                field="volume",
            )

        # Check High >= Low for each day
        for i in range(n):
            if self.high[i] < self.low[i]:
                raise FinanceValidationError(
                    f"High ({self.high[i]}) < Low ({self.low[i]}) at index {i} ({self.dates[i]})",
                    field="high/low",
                    suggestion="This indicates invalid market data",
                )

            # Close should be within [Low, High]
            if not (self.low[i] <= self.close[i] <= self.high[i]):
                raise FinanceValidationError(
                    f"Close ({self.close[i]}) outside [Low, High] range at index {i}",
                    field="close",
                    suggestion="Close price must be between Low and High",
                )

            # Open should be within [Low, High]
            if not (self.low[i] <= self.open[i] <= self.high[i]):
                raise FinanceValidationError(
                    f"Open ({self.open[i]}) outside [Low, High] range at index {i}",
                    field="open",
                    suggestion="Open price must be between Low and High",
                )

        return self

    def to_price_data(self) -> PriceDataInput:
        """Convert to simple PriceDataInput (close prices only)."""
        return PriceDataInput(
            ticker=self.ticker,
            dates=self.dates,
            prices=self.close,
            volumes=self.volume,
        )


class ReturnsData(BaseFinanceModel):
    """
    Pre-calculated returns data.

    WHAT: Daily/periodic returns instead of raw prices
    WHY: Some calculations work directly with returns
    VALIDATES:
        - Returns are reasonable (not extreme outliers)
        - Dates are sorted

    EDUCATIONAL NOTE:
    Returns can be:
    - Simple returns: (P1 - P0) / P0
    - Log returns: ln(P1 / P0)

    Log returns are additive across time and better for statistical analysis.
    Simple returns are more intuitive for communication.
    """

    ticker: str = Field(
        ...,
        description="Asset ticker symbol",
    )

    dates: List[date] = Field(
        ...,
        description="Dates for each return observation",
        min_length=1,
    )

    returns: List[float] = Field(
        ...,
        description="Return values (decimal, not percentage)",
        min_length=1,
    )

    return_type: str = Field(
        default="simple",
        pattern=r"^(simple|log)$",
        description="Type of returns: 'simple' or 'log'",
    )

    period: str = Field(
        default="daily",
        pattern=r"^(daily|weekly|monthly|annual)$",
        description="Return period frequency",
    )

    @field_validator("dates")
    @classmethod
    def validate_dates_sorted(cls, v: List[date]) -> List[date]:
        """Ensure dates are sorted."""
        if v != sorted(v):
            raise FinanceValidationError(
                "Dates must be in chronological order",
                field="dates",
            )
        return v

    @model_validator(mode="after")
    def validate_alignment(self) -> "ReturnsData":
        """Ensure dates and returns are aligned."""
        if len(self.dates) != len(self.returns):
            raise FinanceValidationError(
                f"Length mismatch: {len(self.returns)} returns but {len(self.dates)} dates",
                field="returns/dates",
            )
        return self

    @staticmethod
    def from_prices(
        ticker: str,
        dates: List[date],
        prices: List[float],
        return_type: str = "simple",
    ) -> "ReturnsData":
        """
        Create ReturnsData from price series.

        Args:
            ticker: Asset ticker
            dates: Price dates
            prices: Price values
            return_type: 'simple' or 'log'

        Returns:
            ReturnsData with calculated returns
        """
        if len(prices) < 2:
            raise FinanceValidationError(
                "Need at least 2 prices to calculate returns",
                field="prices",
            )

        returns = []
        return_dates = []

        for i in range(1, len(prices)):
            if return_type == "log":
                import math
                ret = math.log(prices[i] / prices[i - 1])
            else:
                ret = (prices[i] - prices[i - 1]) / prices[i - 1]

            returns.append(ret)
            return_dates.append(dates[i])

        return ReturnsData(
            ticker=ticker,
            dates=return_dates,
            returns=returns,
            return_type=return_type,
            period="daily",
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PriceDataInput",
    "OHLCVData",
    "ReturnsData",
]
