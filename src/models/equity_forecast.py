# src/models/equity_forecast.py

import datetime as dt
from typing import Any, Dict, List, Optional, TypeVar, Generic
from pydantic import BaseModel, Field, ConfigDict

# ============================================================
# GENERIC API RESPONSE WRAPPER
# ============================================================

DataType = TypeVar('DataType')

class APIResponseData(BaseModel, Generic[DataType]):
    """
    Simple wrapper for data payload.
    Supports both single item and list of items.
    """
    data: DataType = Field(..., description="Data payload")


class APIResponse(BaseModel, Generic[DataType]):
    """
    Standard API response wrapper used across the application.
    Provides consistent structure for all endpoints.
    """
    message: str = Field("OK", description="Response message")
    status: str = Field("200", description="HTTP status code as string")
    provider_used: Optional[str] = Field(None, description="Data provider used")
    
    data: Optional[APIResponseData[DataType]] = Field(
        None, 
        description="Response data wrapper"
    )
    
    class Config:
        populate_by_name = True

# ============================================================
# RAW FMP MODELS
# ============================================================

class FMPPriceTargetConsensusItem(BaseModel):
    """
    Raw model for FMP /stable/price-target-consensus endpoint.
    Maps directly to API response fields.
    """
    model_config = ConfigDict(extra="allow")

    symbol: str
    date: Optional[dt.date] = Field(default=None)
    lastUpdated: Optional[dt.datetime] = Field(default=None)
    targetHigh: Optional[float] = Field(default=None)
    targetLow: Optional[float] = Field(default=None)
    targetMedian: Optional[float] = Field(default=None)
    targetConsensus: Optional[float] = Field(default=None)
    targetAvg: Optional[float] = Field(default=None)
    consensus: Optional[float] = Field(default=None)
    numberOfAnalysts: Optional[int] = Field(default=None)
    analystCount: Optional[int] = Field(default=None)
    numAnalysts: Optional[int] = Field(default=None)
    currency: Optional[str] = Field(default=None)
    targetCurrency: Optional[str] = Field(default=None)


class FMPPriceTargetSummaryItem(BaseModel):
    """Raw model for FMP /stable/price-target-summary endpoint"""
    model_config = ConfigDict(extra="allow")

    symbol: Optional[str] = None
    date: Optional[dt.date] = None
    period: Optional[str] = None
    targetHigh: Optional[float] = None
    targetLow: Optional[float] = None
    targetMedian: Optional[float] = None
    targetConsensus: Optional[float] = None
    numberOfAnalysts: Optional[int] = None


class FMPAnalystEstimateItem(BaseModel):
    """Raw model for FMP /stable/analyst-estimates endpoint"""
    model_config = ConfigDict(extra="allow")

    symbol: Optional[str] = None
    date: Optional[dt.date] = None
    revenueLow: Optional[float] = None
    revenueHigh: Optional[float] = None
    revenueAvg: Optional[float] = None
    ebitdaLow: Optional[float] = None
    ebitdaHigh: Optional[float] = None
    ebitdaAvg: Optional[float] = None
    ebitLow: Optional[float] = None
    ebitHigh: Optional[float] = None
    ebitAvg: Optional[float] = None
    netIncomeLow: Optional[float] = None
    netIncomeHigh: Optional[float] = None
    netIncomeAvg: Optional[float] = None
    sgaExpenseLow: Optional[float] = None
    sgaExpenseHigh: Optional[float] = None
    sgaExpenseAvg: Optional[float] = None
    epsAvg: Optional[float] = None
    epsHigh: Optional[float] = None
    epsLow: Optional[float] = None
    numAnalystsRevenue: Optional[int] = None
    numAnalystsEps: Optional[int] = None


class FMPRatingSnapshotItem(BaseModel):
    """Raw model for FMP /stable/ratings-snapshot endpoint"""
    model_config = ConfigDict(extra="allow")

    symbol: Optional[str] = None
    date: Optional[dt.date] = None
    ratingScore: Optional[float] = None
    ratingRecommendation: Optional[str] = None
    ratingText: Optional[str] = None


class FMPFinancialScoreItem(BaseModel):
    """Raw model for FMP Financial Score API"""
    model_config = ConfigDict(extra="allow")

    symbol: Optional[str] = None
    date: Optional[dt.date] = None
    score: Optional[float] = None
    piotroskiScore: Optional[float] = None


class FMPDiscountedCashFlowItem(BaseModel):
    """Raw model for FMP Discounted Cash Flow API"""
    model_config = ConfigDict(extra="allow")

    symbol: Optional[str] = None
    date: Optional[dt.date] = None
    dcf: Optional[float] = None
    stockPrice: Optional[float] = None


# ============================================================
# API RESPONSE MODELS
# ============================================================

class PriceForecastBand(BaseModel):
    """
    Lightweight model for chart rendering.
    Contains only essential fields for drawing forecast zone.
    """
    symbol: str
    as_of: Optional[str] = None
    target_low: Optional[float] = None
    target_high: Optional[float] = None
    target_mean: Optional[float] = None
    target_median: Optional[float] = None
    number_of_analysts: Optional[int] = None
    currency: Optional[str] = None


class PriceForecastContext(BaseModel):
    """
    Complete forecast context with all available data sources.
    Used for detailed tooltip display and AI analysis input.
    """
    symbol: str
    price_target_band: Optional[PriceForecastBand] = None
    price_target_consensus: Optional[FMPPriceTargetConsensusItem] = None
    price_target_summary: Optional[List[FMPPriceTargetSummaryItem]] = None
    analyst_estimates: Optional[List[FMPAnalystEstimateItem]] = None
    ratings_snapshot: Optional[List[FMPRatingSnapshotItem]] = None
    financial_scores: Optional[List[FMPFinancialScoreItem]] = None
    dcf_items: Optional[List[FMPDiscountedCashFlowItem]] = None


# ============================================================
# STRUCTURED AI ANALYSIS OUTPUT
# ============================================================

class PriceForecastSummary(BaseModel):
    """Quick overview of current price vs forecast range"""
    current_price: Optional[float] = Field(
        None,
        description="Current market price if provided"
    )
    target_low: Optional[float] = Field(
        None,
        description="Analyst target low price"
    )
    target_high: Optional[float] = Field(
        None,
        description="Analyst target high price"
    )
    target_consensus: Optional[float] = Field(
        None,
        description="Analyst consensus/mean target"
    )
    number_of_analysts: Optional[int] = Field(
        None,
        description="Number of analysts providing estimates"
    )
    price_vs_target_pct: Optional[float] = Field(
        None,
        description="Current price vs consensus target as percentage"
    )
    position_analysis: str = Field(
        ...,
        description="Brief explanation of where current price sits relative to forecast band"
    )


class ValuationInsights(BaseModel):
    """Deep dive into what drives the price forecast"""
    revenue_growth_outlook: str = Field(
        ...,
        description="Analysis of revenue growth expectations based on analyst estimates"
    )
    earnings_trajectory: str = Field(
        ...,
        description="EPS growth projections and quality assessment"
    )
    profitability_health: str = Field(
        ...,
        description="EBITDA, operating margins, and profitability trends"
    )
    dcf_valuation_gap: Optional[str] = Field(
        None,
        description="DCF intrinsic value vs market price analysis (if available)"
    )
    financial_strength_score: Optional[str] = Field(
        None,
        description="Piotroski score and overall financial health (if available)"
    )
    analyst_rating_consensus: Optional[str] = Field(
        None,
        description="Buy/Hold/Sell consensus and rating score interpretation (if available)"
    )


class RiskFactors(BaseModel):
    """Comprehensive risk assessment and uncertainties"""
    forecast_reliability: str = Field(
        ...,
        description="Assessment of forecast quality based on analyst coverage and data recency"
    )
    market_risks: str = Field(
        ...,
        description="Macro risks that could invalidate forecasts (recession, policy changes, etc)"
    )
    company_specific_risks: str = Field(
        ...,
        description="Business-specific risks based on fundamentals and industry position"
    )
    valuation_risks: str = Field(
        ...,
        description="Risks related to current valuation levels (overvalued/undervalued dynamics)"
    )
    data_gaps: Optional[str] = Field(
        None,
        description="What critical data is missing that limits analysis confidence"
    )


class InvestmentGuidance(BaseModel):
    """Actionable framework for using this forecast"""
    how_to_interpret: str = Field(
        ...,
        description="Practical explanation of what this forecast means for investors"
    )
    suitable_investor_profiles: str = Field(
        ...,
        description="Which investor types and timeframes this forecast is most relevant for"
    )
    key_monitoring_metrics: str = Field(
        ...,
        description="What metrics to watch to validate or invalidate this forecast"
    )
    complementary_analysis: str = Field(
        ...,
        description="What other analysis should be combined with this forecast"
    )
    important_disclaimers: str = Field(
        ...,
        description="Critical caveats about forecast limitations and proper usage"
    )


class ForecastAnalysisMetadata(BaseModel):
    """Tracking information about the analysis generation"""
    analysis_timestamp: dt.datetime = Field(
        default_factory=dt.datetime.now,
        description="When analysis was generated"
    )
    model_used: str = Field(..., description="LLM model name")
    provider_used: str = Field(..., description="LLM provider")
    data_sources_available: List[str] = Field(
        default_factory=list,
        description="Which FMP data sources were available for analysis"
    )
    data_sources_missing: List[str] = Field(
        default_factory=list,
        description="Which expected data sources were not available"
    )
    language: str = Field("vi", description="Analysis output language")
    data_freshness: Optional[str] = Field(
        None,
        description="How recent the underlying forecast data is"
    )


class ForecastAIAnalysisResponse(BaseModel):
    """
    Simplified AI analysis - markdown-first approach.
    Structure is extracted from markdown, not generated by LLM.
    """
    symbol: str
    current_price: Optional[float] = None
    target_mean: Optional[float] = None
    
    # Main content - markdown format (easy for LLM)
    analysis_markdown: str = Field(
        ...,
        description="Complete analysis in markdown with clear sections"
    )
    
    # Metadata
    metadata: ForecastAnalysisMetadata
    
    # Structured fields (extracted from markdown, not LLM-generated)
    summary: Optional[PriceForecastSummary] = None
    valuation_insights: Optional[ValuationInsights] = None
    risk_factors: Optional[RiskFactors] = None
    investment_guidance: Optional[InvestmentGuidance] = None