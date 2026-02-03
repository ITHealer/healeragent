"""
Finance Guru - Valuation Models (Phase 1)

Layer 1: Pydantic models for intrinsic value calculations.

This module provides validated data structures for:
- DCF (Discounted Cash Flow) valuation
- Graham Formula valuation
- DDM (Dividend Discount Model) valuation
- Comparable company analysis

EDUCATIONAL NOTES:
- DCF: Values a company based on projected future cash flows discounted to present value
- Graham Formula: Benjamin Graham's simplified intrinsic value calculation
- DDM: Values stocks based on future dividend payments
- All methods attempt to find "fair value" vs current market price

Author: HealerAgent Development Team
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, field_validator, model_validator

from src.agents.tools.finance_guru.models.base import (
    BaseFinanceModel,
    BaseCalculationResult,
)


# =============================================================================
# ENUMS
# =============================================================================


class ValuationMethod(str, Enum):
    """Valuation methods available.

    DCF: Discounted Cash Flow - most comprehensive, requires projections
    GRAHAM: Graham Formula - simple, uses EPS and growth
    DDM: Dividend Discount Model - for dividend-paying stocks
    COMPARABLE: Relative valuation using peer multiples
    """
    DCF = "dcf"
    GRAHAM = "graham"
    DDM = "ddm"
    COMPARABLE = "comparable"


class DDMType(str, Enum):
    """Types of Dividend Discount Models.

    GORDON: Gordon Growth Model - assumes constant dividend growth
    TWO_STAGE: Two-stage model - high growth then stable growth
    H_MODEL: H-model - declining growth rate over time
    THREE_STAGE: Three-stage - growth, transition, mature phases
    """
    GORDON = "gordon"
    TWO_STAGE = "two_stage"
    H_MODEL = "h_model"
    THREE_STAGE = "three_stage"


class GrowthAssumption(str, Enum):
    """Growth rate assumptions for projections.

    CONSERVATIVE: Below historical average
    MODERATE: Near historical average
    AGGRESSIVE: Above historical average
    CUSTOM: User-specified growth rates
    """
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


# =============================================================================
# INPUT MODELS - DCF
# =============================================================================


class DCFInputData(BaseFinanceModel):
    """Input data for DCF valuation.

    DCF values a company by projecting future free cash flows and
    discounting them back to present value.

    Attributes:
        symbol: Stock ticker symbol
        current_fcf: Current year free cash flow (millions)
        growth_rates: Projected growth rates for each year
        terminal_growth: Long-term growth rate (typically 2-3%)
        discount_rate: WACC or required return (typically 8-12%)
        shares_outstanding: Number of shares (millions)
        current_price: Current stock price for comparison
        cash: Cash and equivalents (millions)
        debt: Total debt (millions)

    EDUCATIONAL NOTE:
    - FCF = Operating Cash Flow - Capital Expenditures
    - Terminal growth should not exceed GDP growth long-term
    - Discount rate reflects risk - higher risk = higher rate
    """
    symbol: str = Field(..., min_length=1, max_length=20, description="Stock symbol")
    current_fcf: float = Field(..., description="Current FCF (millions)")
    growth_rates: List[float] = Field(
        ..., min_length=1, max_length=10,
        description="Yearly growth rates (e.g., [0.15, 0.12, 0.10])"
    )
    terminal_growth: float = Field(
        0.025, ge=0.0, le=0.05,
        description="Terminal growth rate (2.5% default)"
    )
    discount_rate: float = Field(
        0.10, ge=0.05, le=0.25,
        description="Discount rate/WACC (10% default)"
    )
    shares_outstanding: float = Field(..., gt=0, description="Shares outstanding (millions)")
    current_price: Optional[float] = Field(None, gt=0, description="Current stock price")
    cash: float = Field(0.0, ge=0, description="Cash & equivalents (millions)")
    debt: float = Field(0.0, ge=0, description="Total debt (millions)")

    @field_validator("growth_rates")
    @classmethod
    def validate_growth_rates(cls, v: List[float]) -> List[float]:
        """Ensure growth rates are reasonable."""
        for rate in v:
            if rate < -0.5 or rate > 1.0:
                raise ValueError(f"Growth rate {rate} is unrealistic (should be -50% to +100%)")
        return v


class DCFConfig(BaseFinanceModel):
    """Configuration for DCF calculation.

    Attributes:
        projection_years: Number of years to project (5-10 typical)
        margin_of_safety: Safety margin for conservative estimate (e.g., 0.25 = 25%)
        sensitivity_range: Range for sensitivity analysis (±%)
    """
    projection_years: int = Field(5, ge=3, le=10, description="Projection period")
    margin_of_safety: float = Field(0.25, ge=0.0, le=0.5, description="Margin of safety")
    sensitivity_range: float = Field(0.20, ge=0.05, le=0.50, description="Sensitivity range")


# =============================================================================
# INPUT MODELS - GRAHAM
# =============================================================================


class GrahamInputData(BaseFinanceModel):
    """Input data for Graham Formula valuation.

    Benjamin Graham's formula: V = EPS × (8.5 + 2g) × (4.4/Y)

    Where:
    - V = Intrinsic value
    - EPS = Earnings per share (trailing 12 months)
    - g = Expected annual growth rate (5-year)
    - 8.5 = P/E base for no-growth company
    - Y = Current yield on AAA corporate bonds

    Attributes:
        symbol: Stock ticker
        eps: Earnings per share (trailing 12 months)
        growth_rate: Expected 5-year growth rate
        current_price: Current stock price
        aaa_yield: AAA corporate bond yield (default 4.4%)
    """
    symbol: str = Field(..., min_length=1, max_length=20, description="Stock symbol")
    eps: float = Field(..., description="EPS (trailing 12 months)")
    growth_rate: float = Field(..., ge=0, le=0.50, description="5-year growth rate")
    current_price: float = Field(..., gt=0, description="Current stock price")
    aaa_yield: float = Field(0.044, ge=0.01, le=0.15, description="AAA bond yield")

    @field_validator("eps")
    @classmethod
    def validate_eps(cls, v: float) -> float:
        """Graham formula requires positive earnings."""
        if v <= 0:
            raise ValueError("Graham formula requires positive EPS")
        return v


class GrahamConfig(BaseFinanceModel):
    """Configuration for Graham calculation.

    Attributes:
        pe_base: P/E ratio for no-growth company (Graham used 8.5)
        growth_multiplier: Multiplier for growth rate (Graham used 2)
        margin_of_safety: Safety margin (Graham recommended 33%)
    """
    pe_base: float = Field(8.5, ge=5.0, le=15.0, description="Base P/E ratio")
    growth_multiplier: float = Field(2.0, ge=1.0, le=3.0, description="Growth multiplier")
    margin_of_safety: float = Field(0.33, ge=0.0, le=0.5, description="Margin of safety")


# =============================================================================
# INPUT MODELS - DDM
# =============================================================================


class DDMInputData(BaseFinanceModel):
    """Input data for Dividend Discount Model valuation.

    DDM values a stock as the present value of all future dividend payments.

    Gordon Growth Model: P = D1 / (r - g)
    Where:
    - P = Stock price
    - D1 = Expected dividend next year
    - r = Required return
    - g = Dividend growth rate (must be < r)

    Attributes:
        symbol: Stock ticker
        current_dividend: Annual dividend per share
        dividend_growth: Expected dividend growth rate
        required_return: Required rate of return
        current_price: Current stock price for comparison
        payout_ratio: Dividend payout ratio (optional, for growth estimation)
    """
    symbol: str = Field(..., min_length=1, max_length=20, description="Stock symbol")
    current_dividend: float = Field(..., gt=0, description="Current annual dividend")
    dividend_growth: float = Field(..., ge=0, le=0.20, description="Dividend growth rate")
    required_return: float = Field(..., ge=0.04, le=0.25, description="Required return")
    current_price: float = Field(..., gt=0, description="Current stock price")
    payout_ratio: Optional[float] = Field(None, ge=0, le=1.0, description="Payout ratio")

    @model_validator(mode="after")
    def validate_growth_vs_return(self) -> "DDMInputData":
        """Ensure growth rate is less than required return."""
        if self.dividend_growth >= self.required_return:
            raise ValueError(
                f"Dividend growth ({self.dividend_growth:.2%}) must be less than "
                f"required return ({self.required_return:.2%})"
            )
        return self


class DDMConfig(BaseFinanceModel):
    """Configuration for DDM calculation.

    Attributes:
        model_type: Type of DDM model to use
        high_growth_years: Years of high growth (for multi-stage)
        high_growth_rate: High growth phase rate
        stable_growth_rate: Stable/terminal growth rate
    """
    model_type: DDMType = Field(DDMType.GORDON, description="DDM model type")
    high_growth_years: int = Field(5, ge=3, le=10, description="High growth period")
    high_growth_rate: Optional[float] = Field(
        None, ge=0, le=0.30,
        description="High growth rate"
    )
    stable_growth_rate: float = Field(0.03, ge=0, le=0.05, description="Stable growth rate")


# =============================================================================
# INPUT MODELS - COMPARABLE
# =============================================================================


class ComparableCompany(BaseFinanceModel):
    """Data for a comparable company.

    Attributes:
        symbol: Company ticker
        pe_ratio: Price-to-Earnings ratio
        pb_ratio: Price-to-Book ratio
        ps_ratio: Price-to-Sales ratio
        ev_ebitda: Enterprise Value / EBITDA
        market_cap: Market capitalization (millions)
    """
    symbol: str = Field(..., description="Company symbol")
    pe_ratio: Optional[float] = Field(None, gt=0, description="P/E ratio")
    pb_ratio: Optional[float] = Field(None, gt=0, description="P/B ratio")
    ps_ratio: Optional[float] = Field(None, gt=0, description="P/S ratio")
    ev_ebitda: Optional[float] = Field(None, gt=0, description="EV/EBITDA")
    market_cap: Optional[float] = Field(None, gt=0, description="Market cap (millions)")


class ComparableInputData(BaseFinanceModel):
    """Input data for comparable company valuation.

    Relative valuation compares a target company to similar peers
    using valuation multiples.

    Attributes:
        symbol: Target company ticker
        eps: Target company EPS
        book_value_per_share: Book value per share
        revenue_per_share: Revenue per share
        ebitda_per_share: EBITDA per share
        current_price: Current stock price
        peers: List of comparable companies
    """
    symbol: str = Field(..., description="Target company symbol")
    eps: float = Field(..., description="Target EPS")
    book_value_per_share: float = Field(..., gt=0, description="Book value per share")
    revenue_per_share: float = Field(..., gt=0, description="Revenue per share")
    ebitda_per_share: Optional[float] = Field(None, gt=0, description="EBITDA per share")
    current_price: float = Field(..., gt=0, description="Current price")
    peers: List[ComparableCompany] = Field(
        ..., min_length=2, max_length=20,
        description="Comparable companies"
    )


# =============================================================================
# OUTPUT MODELS
# =============================================================================


class DCFProjection(BaseFinanceModel):
    """Single year DCF projection.

    Attributes:
        year: Projection year number
        fcf: Projected free cash flow
        growth_rate: Growth rate applied
        discount_factor: Discount factor for this year
        present_value: Discounted value of FCF
    """
    year: int = Field(..., description="Year number")
    fcf: float = Field(..., description="Projected FCF (millions)")
    growth_rate: float = Field(..., description="Growth rate applied")
    discount_factor: float = Field(..., description="Discount factor")
    present_value: float = Field(..., description="Present value (millions)")


class DCFOutput(BaseCalculationResult):
    """DCF valuation result.

    Attributes:
        symbol: Stock symbol
        projections: Year-by-year projections
        sum_of_pv_fcf: Sum of discounted cash flows
        terminal_value: Terminal value
        pv_terminal_value: Present value of terminal value
        enterprise_value: Total enterprise value
        equity_value: Value to equity holders
        intrinsic_value_per_share: Fair value per share
        current_price: Current market price
        upside_potential: Percentage upside/downside
        margin_of_safety_price: Price with margin of safety
        verdict: "undervalued", "overvalued", or "fairly_valued"

        # Enhanced fields for comprehensive analysis
        fcf_source: Source of FCF data (API, manual, normalized)
        discount_rate: WACC/discount rate used
        terminal_growth: Terminal growth rate used
        implied_growth_rate: Reverse DCF implied growth (given current price)
        sensitivity_2d: 2D sensitivity matrix (WACC × Terminal Growth)
        validation_warnings: Any validation warnings
    """
    symbol: str = Field(..., description="Stock symbol")
    projections: List[DCFProjection] = Field(..., description="Cash flow projections")
    sum_of_pv_fcf: float = Field(..., description="Sum of PV of FCFs")
    terminal_value: float = Field(..., description="Terminal value")
    pv_terminal_value: float = Field(..., description="PV of terminal value")
    enterprise_value: float = Field(..., description="Enterprise value")
    equity_value: float = Field(..., description="Equity value")
    intrinsic_value_per_share: float = Field(..., description="Intrinsic value/share")
    current_price: Optional[float] = Field(None, description="Current price")
    upside_potential: Optional[float] = Field(None, description="Upside potential %")
    margin_of_safety_price: float = Field(..., description="Price with safety margin")
    verdict: str = Field(..., description="Valuation verdict")

    # Input parameters (for transparency)
    discount_rate: Optional[float] = Field(None, description="WACC/discount rate used")
    terminal_growth: Optional[float] = Field(None, description="Terminal growth rate used")
    fcf_source: Optional[str] = Field(None, description="Source of FCF data")
    shares_outstanding: Optional[float] = Field(None, description="Shares outstanding (M)")
    cash: Optional[float] = Field(None, description="Cash & equivalents (M)")
    debt: Optional[float] = Field(None, description="Total debt (M)")

    # Enhanced analysis
    implied_growth_rate: Optional[float] = Field(
        None,
        description="Reverse DCF: implied perpetual growth rate given current price"
    )
    tv_as_pct_of_ev: Optional[float] = Field(
        None,
        description="Terminal value as % of enterprise value"
    )

    # Sensitivity analysis (1D)
    sensitivity: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Sensitivity analysis results"
    )

    # 2D Sensitivity Matrix (WACC × Terminal Growth)
    sensitivity_2d: Optional[Dict[str, Any]] = Field(
        None,
        description="2D sensitivity matrix: wacc_rates, growth_rates, and values grid"
    )

    # Validation
    validation_warnings: Optional[List[str]] = Field(
        None,
        description="Validation warnings or flags"
    )


class GrahamOutput(BaseCalculationResult):
    """Graham Formula valuation result.

    Attributes:
        symbol: Stock symbol
        eps: EPS used in calculation
        growth_rate: Growth rate used
        aaa_yield: Bond yield used
        graham_value: Raw Graham formula value
        intrinsic_value: Adjusted intrinsic value
        margin_of_safety_price: Price with safety margin
        current_price: Current market price
        upside_potential: Percentage upside/downside
        verdict: Valuation verdict
    """
    symbol: str = Field(..., description="Stock symbol")
    eps: float = Field(..., description="EPS used")
    growth_rate: float = Field(..., description="Growth rate")
    aaa_yield: float = Field(..., description="AAA bond yield")
    graham_value: float = Field(..., description="Graham formula result")
    intrinsic_value: float = Field(..., description="Intrinsic value")
    margin_of_safety_price: float = Field(..., description="Safety margin price")
    current_price: float = Field(..., description="Current price")
    upside_potential: float = Field(..., description="Upside potential %")
    verdict: str = Field(..., description="Valuation verdict")

    # Components breakdown
    pe_no_growth: float = Field(..., description="P/E for no-growth (8.5)")
    growth_premium: float = Field(..., description="Growth premium (2g)")
    bond_adjustment: float = Field(..., description="Bond yield adjustment")


class DDMOutput(BaseCalculationResult):
    """DDM valuation result.

    Attributes:
        symbol: Stock symbol
        model_type: DDM model used
        current_dividend: Current dividend
        dividend_growth: Dividend growth rate
        required_return: Required return used
        intrinsic_value: Calculated intrinsic value
        current_price: Current market price
        dividend_yield: Current dividend yield
        upside_potential: Percentage upside/downside
        verdict: Valuation verdict
    """
    symbol: str = Field(..., description="Stock symbol")
    model_type: DDMType = Field(..., description="DDM model used")
    current_dividend: float = Field(..., description="Current dividend")
    expected_dividend: float = Field(..., description="Next year expected dividend")
    dividend_growth: float = Field(..., description="Dividend growth rate")
    required_return: float = Field(..., description="Required return")
    intrinsic_value: float = Field(..., description="Intrinsic value")
    current_price: float = Field(..., description="Current price")
    dividend_yield: float = Field(..., description="Current yield")
    upside_potential: float = Field(..., description="Upside potential %")
    verdict: str = Field(..., description="Valuation verdict")

    # For multi-stage models
    stages: Optional[List[Dict[str, Any]]] = Field(None, description="Stage breakdown")


class MultipleValuation(BaseFinanceModel):
    """Valuation using a single multiple.

    Attributes:
        multiple_name: Name of the multiple (P/E, P/B, etc.)
        peer_average: Average multiple from peers
        peer_median: Median multiple from peers
        implied_value: Implied value using average
        implied_value_median: Implied value using median
    """
    multiple_name: str = Field(..., description="Multiple name")
    peer_average: float = Field(..., description="Peer average")
    peer_median: float = Field(..., description="Peer median")
    implied_value: float = Field(..., description="Implied value (average)")
    implied_value_median: float = Field(..., description="Implied value (median)")


class ComparableOutput(BaseCalculationResult):
    """Comparable company valuation result.

    Attributes:
        symbol: Target company symbol
        peers: Peer companies used
        valuations: Valuation by each multiple
        average_intrinsic_value: Average across all methods
        median_intrinsic_value: Median across all methods
        current_price: Current market price
        upside_potential: Percentage upside/downside
        verdict: Valuation verdict
    """
    symbol: str = Field(..., description="Target symbol")
    peers: List[str] = Field(..., description="Peer symbols")
    valuations: List[MultipleValuation] = Field(..., description="Multiple valuations")
    average_intrinsic_value: float = Field(..., description="Average intrinsic value")
    median_intrinsic_value: float = Field(..., description="Median intrinsic value")
    current_price: float = Field(..., description="Current price")
    upside_potential: float = Field(..., description="Upside potential")
    verdict: str = Field(..., description="Valuation verdict")

    # Peer statistics
    peer_stats: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Statistics for each multiple"
    )


class ValuationSummary(BaseCalculationResult):
    """Combined valuation summary across multiple methods.

    Attributes:
        symbol: Stock symbol
        methods_used: Valuation methods applied
        valuations: Dict mapping method to intrinsic value
        average_value: Average intrinsic value
        median_value: Median intrinsic value
        range: (min, max) valuation range
        current_price: Current market price
        overall_verdict: Overall recommendation
        confidence: Confidence level based on method agreement
    """
    symbol: str = Field(..., description="Stock symbol")
    methods_used: List[ValuationMethod] = Field(..., description="Methods used")
    valuations: Dict[str, float] = Field(..., description="Value by method")
    average_value: float = Field(..., description="Average intrinsic value")
    median_value: float = Field(..., description="Median intrinsic value")
    range: Tuple[float, float] = Field(..., description="(min, max) range")
    current_price: float = Field(..., description="Current price")
    overall_verdict: str = Field(..., description="Overall verdict")
    confidence: str = Field(..., description="Confidence: high/medium/low")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
