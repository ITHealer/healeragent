"""
Finance Guru - Base Models

Core Pydantic models and types used across all Finance Guru modules.
These provide the foundation for type-safe financial calculations.

DESIGN PRINCIPLES:
1. Immutability: Models are immutable after creation (use frozen=True where needed)
2. Validation: All business rules validated on instantiation
3. Documentation: Every field has clear description
4. Serialization: All models support JSON serialization for API responses

Author: HealerAgent Development Team
Created: 2025-01-18
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class FinanceValidationError(ValueError):
    """
    Custom exception for finance-specific validation errors.

    Provides clear error messages with context about what went wrong
    and how to fix it.

    Attributes:
        field: The field that failed validation
        value: The invalid value
        reason: Why the validation failed
        suggestion: How to fix the issue
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        reason: Optional[str] = None,
        suggestion: Optional[str] = None,
    ):
        self.field = field
        self.value = value
        self.reason = reason
        self.suggestion = suggestion

        # Build detailed message
        parts = [message]
        if field:
            parts.append(f"Field: {field}")
        if value is not None:
            parts.append(f"Value: {value}")
        if reason:
            parts.append(f"Reason: {reason}")
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")

        super().__init__(" | ".join(parts))


# =============================================================================
# ENUMS
# =============================================================================

class MarketType(str, Enum):
    """Type of market for asset classification."""
    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"


class Currency(str, Enum):
    """Supported currencies."""
    USD = "USD"
    VND = "VND"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"
    BTC = "BTC"  # For crypto
    ETH = "ETH"


class SignalType(str, Enum):
    """Trading signal types."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    OVERBOUGHT = "overbought"
    OVERSOLD = "oversold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


class TrendDirection(str, Enum):
    """Trend direction for technical analysis."""
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Severity level for warnings and anomalies."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# BASE MODELS
# =============================================================================

class BaseFinanceModel(BaseModel):
    """
    Base class for all Finance Guru models.

    Provides common configuration and utilities:
    - Strict validation by default
    - JSON serialization support
    - Immutable after creation (optional)
    - Extra fields forbidden to catch typos

    All Finance Guru models should inherit from this class.
    """

    model_config = ConfigDict(
        # Validate default values
        validate_default=True,
        # Forbid extra attributes (catch typos)
        extra="forbid",
        # Use enum values in serialization
        use_enum_values=True,
        # Populate by field name
        populate_by_name=True,
        # Validate on assignment
        validate_assignment=True,
        # JSON serialization settings
        ser_json_timedelta="iso8601",
        ser_json_bytes="base64",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for API responses."""
        return self.model_dump(mode="json")

    def to_json(self) -> str:
        """Convert model to JSON string."""
        return self.model_dump_json()


class BaseCalculationResult(BaseFinanceModel):
    """
    Base class for calculation results.

    All calculation outputs should inherit from this class.
    Provides common fields for tracking calculation metadata.
    """

    # Identification
    ticker: Optional[str] = Field(
        default=None,
        description="Asset ticker symbol (e.g., 'AAPL', 'BTC-USD')",
        pattern=r"^[A-Z0-9\-\.]{1,20}$",
    )

    # Timing
    calculation_date: date = Field(
        default_factory=date.today,
        description="Date when calculation was performed",
    )

    calculated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when calculation was performed (UTC)",
    )

    # Metadata
    calculation_method: Optional[str] = Field(
        default=None,
        description="Method or algorithm used for calculation",
    )

    data_points_used: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of data points used in calculation",
    )

    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings generated during calculation",
    )

    calculation_time_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Calculation time in milliseconds",
    )


# =============================================================================
# COMMON DATA TYPES
# =============================================================================

class PricePoint(BaseFinanceModel):
    """
    Single price point with date.

    Used for time series data where each point has a date and price.
    """

    point_date: date = Field(..., description="Date of the price point", alias="date")
    price: float = Field(..., gt=0, description="Price value (must be positive)")
    volume: Optional[float] = Field(
        default=None,
        ge=0,
        description="Trading volume (optional)",
    )

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Ensure price is a valid positive number."""
        if v <= 0:
            raise FinanceValidationError(
                "Price must be positive",
                field="price",
                value=v,
                suggestion="Check for data errors or delisted securities",
            )
        return v


class TimeSeries(BaseFinanceModel):
    """
    Time series data container.

    Generic container for any time-indexed financial data.
    Ensures dates are sorted and aligned with values.
    """

    dates: List[date] = Field(
        ...,
        min_length=1,
        description="Dates in chronological order",
    )

    values: List[float] = Field(
        ...,
        min_length=1,
        description="Values corresponding to each date",
    )

    name: Optional[str] = Field(
        default=None,
        description="Name/label for this time series",
    )

    @model_validator(mode="after")
    def validate_alignment(self) -> "TimeSeries":
        """Ensure dates and values are aligned and sorted."""
        if len(self.dates) != len(self.values):
            raise FinanceValidationError(
                f"Length mismatch: {len(self.dates)} dates but {len(self.values)} values",
                field="dates/values",
                suggestion="Ensure each date has a corresponding value",
            )

        # Check dates are sorted
        if self.dates != sorted(self.dates):
            raise FinanceValidationError(
                "Dates must be in chronological order",
                field="dates",
                suggestion="Sort your data by date before creating TimeSeries",
            )

        return self

    def __len__(self) -> int:
        """Return number of data points."""
        return len(self.dates)


class FinancialPeriod(BaseFinanceModel):
    """
    Represents a financial reporting period.

    Used for quarterly/annual financial data.
    """

    period_type: str = Field(
        ...,
        pattern=r"^(Q[1-4]|FY|TTM|annual|quarterly)$",
        description="Period type: Q1, Q2, Q3, Q4, FY (fiscal year), TTM (trailing 12 months)",
    )

    year: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="Fiscal year",
    )

    end_date: date = Field(
        ...,
        description="Period end date",
    )

    @property
    def period_label(self) -> str:
        """Human-readable period label like 'Q3 2024' or 'FY 2023'."""
        return f"{self.period_type} {self.year}"


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_positive(value: float, field_name: str) -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise FinanceValidationError(
            f"{field_name} must be positive",
            field=field_name,
            value=value,
        )
    return value


def validate_percentage(value: float, field_name: str, allow_negative: bool = True) -> float:
    """Validate that a value is a reasonable percentage."""
    if not allow_negative and value < 0:
        raise FinanceValidationError(
            f"{field_name} cannot be negative",
            field=field_name,
            value=value,
        )

    # Warn if percentage seems unreasonable (>500% or <-100%)
    if value > 5.0:  # 500%
        raise FinanceValidationError(
            f"{field_name} of {value:.1%} seems unreasonably high",
            field=field_name,
            value=value,
            suggestion="Verify this is correct. Percentages should typically be between -100% and 100%",
        )

    if value < -1.0:  # -100%
        raise FinanceValidationError(
            f"{field_name} of {value:.1%} indicates total loss",
            field=field_name,
            value=value,
            suggestion="Returns below -100% are impossible for long positions",
        )

    return value


def validate_date_range(
    start_date: date,
    end_date: date,
    min_days: int = 1,
    max_days: int = 3650,  # 10 years
) -> Tuple[date, date]:
    """Validate a date range is sensible."""
    if start_date >= end_date:
        raise FinanceValidationError(
            "Start date must be before end date",
            field="date_range",
            value=f"{start_date} to {end_date}",
        )

    days = (end_date - start_date).days

    if days < min_days:
        raise FinanceValidationError(
            f"Date range too short: {days} days (minimum: {min_days})",
            field="date_range",
        )

    if days > max_days:
        raise FinanceValidationError(
            f"Date range too long: {days} days (maximum: {max_days})",
            field="date_range",
            suggestion="Consider using a shorter time period for analysis",
        )

    return start_date, end_date


def validate_ticker(ticker: str) -> str:
    """Validate ticker symbol format."""
    import re

    # Pattern: 1-10 uppercase letters/numbers, optional suffix like .TO, -USD
    pattern = r"^[A-Z0-9]{1,10}(\.[A-Z]{1,2}|-[A-Z]{2,4})?$"

    if not re.match(pattern, ticker.upper()):
        raise FinanceValidationError(
            f"Invalid ticker format: {ticker}",
            field="ticker",
            value=ticker,
            suggestion="Ticker should be uppercase letters/numbers (e.g., AAPL, BTC-USD)",
        )

    return ticker.upper()


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Numeric types that can be used for financial values
NumericType = Union[int, float, Decimal]

# Type variable for generic models
T = TypeVar("T", bound=BaseFinanceModel)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "FinanceValidationError",
    # Enums
    "MarketType",
    "Currency",
    "SignalType",
    "TrendDirection",
    "Severity",
    # Base Models
    "BaseFinanceModel",
    "BaseCalculationResult",
    # Data Types
    "PricePoint",
    "TimeSeries",
    "FinancialPeriod",
    # Validators
    "validate_positive",
    "validate_percentage",
    "validate_date_range",
    "validate_ticker",
    # Type Aliases
    "NumericType",
    "T",
]
