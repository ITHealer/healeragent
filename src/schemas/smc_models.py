from enum import Enum
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# ENUM DEFINITIONS
# ============================================================================

class Direction(str, Enum):
    """Market direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class StructureType(str, Enum):
    """Market structure event types"""
    BOS = "BOS"
    CHOCH = "CHoCH"


class StructureCategory(str, Enum):
    """Structure timeframe category"""
    INTERNAL = "internal"
    SWING = "swing"


class ZoneType(str, Enum):
    """Liquidity zone types"""
    EQH = "EQH"
    EQL = "EQL"


class SignalStrength(str, Enum):
    """Trading signal strength"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class ConfidenceLevel(str, Enum):
    """Analysis confidence level"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PriceZone(str, Enum):
    """Price position relative to range"""
    PREMIUM = "premium"
    EQUILIBRIUM = "equilibrium"
    DISCOUNT = "discount"


# ============================================================================
# SMC DATA MODELS - Nested structures for smcData field
# ============================================================================

class OrderBlockData(BaseModel):
    """Order Block data from indicator"""
    type: str = Field(..., description="bullish or bearish")
    time: int = Field(..., description="Unix timestamp")
    high: float = Field(..., description="OB high price")
    low: float = Field(..., description="OB low price")
    open: float = Field(..., description="OB open price")
    close: float = Field(..., description="OB close price")
    startIndex: int = Field(..., description="Candle index when OB formed")
    endIndex: Optional[int] = Field(None, description="Candle index when mitigated")
    mitigated: bool = Field(False, description="Whether OB has been tested")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "bullish",
                "time": 1733655600,
                "high": 97850.00,
                "low": 97200.00,
                "open": 97300.00,
                "close": 97750.00,
                "startIndex": 145,
                "endIndex": None,
                "mitigated": False
            }
        }


class StructureEventData(BaseModel):
    """Market structure event (BOS/CHoCH)"""
    time: int = Field(..., description="Unix timestamp")
    type: str = Field(..., description="BOS or CHoCH")
    direction: str = Field(..., description="bullish or bearish")
    level: float = Field(..., description="Price level of structure break")
    fromIndex: int = Field(..., description="Starting candle index")
    toIndex: int = Field(..., description="Ending candle index")
    category: str = Field("swing", description="internal or swing")


class LiquidityZoneData(BaseModel):
    """Liquidity zone (EQH/EQL)"""
    time: int = Field(..., description="Unix timestamp")
    type: str = Field(..., description="EQH or EQL")
    level: float = Field(..., description="Price level")
    fromIndex: int = Field(..., description="Starting candle index")
    toIndex: int = Field(..., description="Ending candle index")
    category: Optional[str] = Field("swing", description="Structure category")


class FairValueGapData(BaseModel):
    """Fair Value Gap (imbalance zone)"""
    type: str = Field(..., description="bullish or bearish")
    startTime: int = Field(..., description="Unix timestamp")
    startIndex: int = Field(..., description="Candle index")
    topPrice: float = Field(..., description="Upper boundary")
    bottomPrice: float = Field(..., description="Lower boundary")
    mitigated: bool = Field(False, description="Whether FVG has been filled")
    mitigatedIndex: Optional[int] = Field(None, description="Index when filled")


class PremiumDiscountData(BaseModel):
    """Premium/Discount zone data"""
    fromIndex: int = Field(..., description="Starting candle index")
    toIndex: int = Field(..., description="Ending candle index")
    midpoint: float = Field(..., description="Equilibrium price level")


class SwingPointData(BaseModel):
    """Swing high/low point"""
    time: int = Field(..., description="Unix timestamp")
    price: float = Field(..., description="Swing price level")
    type: str = Field(..., description="high or low")
    index: int = Field(..., description="Candle index")


class SMCIndicatorData(BaseModel):
    """Complete SMC indicator data structure"""
    orderBlocks: List[OrderBlockData] = Field(default_factory=list)
    structureEvents: List[StructureEventData] = Field(default_factory=list)
    liquidityZones: List[LiquidityZoneData] = Field(default_factory=list)
    fairValueGaps: List[FairValueGapData] = Field(default_factory=list)
    premiumDiscount: List[PremiumDiscountData] = Field(default_factory=list)
    swingsInternal: List[SwingPointData] = Field(default_factory=list)
    swingsSwing: List[SwingPointData] = Field(default_factory=list)


class SMCMetadata(BaseModel):
    """Metadata summary from indicator - many fields can be null"""
    totalOrderBlocks: Optional[int] = Field(0)
    activeOrderBlocks: Optional[int] = Field(0)
    mitigatedOrderBlocks: Optional[int] = Field(0)
    totalStructureEvents: Optional[int] = Field(0)
    bosCount: Optional[int] = Field(0)
    chochCount: Optional[int] = Field(0)
    totalLiquidityZones: Optional[int] = Field(0)
    eqhCount: Optional[int] = Field(0)
    eqlCount: Optional[int] = Field(0)
    totalFairValueGaps: Optional[int] = Field(0)
    activeFvgCount: Optional[int] = Field(0)
    mitigatedFvgCount: Optional[int] = Field(0)
    currentTrend: Optional[str] = Field("neutral")
    lastStructureType: Optional[str] = Field(None)
    priceInPremium: Optional[bool] = Field(False)
    nearestSupport: Optional[float] = Field(None)
    nearestResistance: Optional[float] = Field(None)


class TradingSignalData(BaseModel):
    """Pre-computed trading signals - all fields can be null"""
    bias: Optional[str] = Field(None)
    strength: Optional[str] = Field(None)
    entryZones: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    stopLoss: Optional[float] = Field(None)
    targets: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    riskRewardRatio: Optional[float] = Field(None)


# ============================================================================
# API REQUEST MODELS - Flat structure for Swagger visibility
# ============================================================================

class SMCAnalyzeRequest(BaseModel):
    """
    Request model for SMC Analysis API
    
    All top-level fields are visible in Swagger for easy testing.
    """
    # === SESSION & LLM CONFIG (TOP) ===
    session_id: Optional[str] = Field(
        default=None, 
        description="Chat session ID for conversation context and memory"
    )
    question_input: Optional[str] = Field(
        default=None, 
        description="Additional user question about the analysis (optional)"
    )
    target_language: str = Field(
        default="en", 
        description="Response language code (en, vi, zh)",
        json_schema_extra={"example": "en"}
    )
    model_name: str = Field(
        default="gpt-4.1-nano", 
        description="LLM model name for analysis interpretation",
        json_schema_extra={"example": "gpt-4.1-nano"}
    )
    provider_type: str = Field(
        default="openai", 
        description="LLM provider type",
        json_schema_extra={"example": "openai"}
    )
    collection_name: Optional[str] = Field(
        default=None, 
        description="RAG collection name for additional context (optional)",
        json_schema_extra={"example": None}
    )
    
    # === SYMBOL INFO ===
    symbol: str = Field(
        ..., 
        description="Trading symbol (e.g., BTCUSDT, ETHUSDT)",
        json_schema_extra={"example": "BTCUSDT"}
    )
    interval: str = Field(
        ..., 
        description="Timeframe interval (e.g., 1h, 4h, 1d)",
        json_schema_extra={"example": "1h"}
    )
    mode: str = Field(
        default="swing", 
        description="Analysis mode: swing or internal",
        json_schema_extra={"example": "swing"}
    )
    timestamp: int = Field(
        ..., 
        description="Data timestamp in milliseconds",
        json_schema_extra={"example": 1733673600000}
    )
    currentPrice: float = Field(
        ..., 
        description="Current market price",
        json_schema_extra={"example": 98750.50}
    )
    
    # === SMC DATA ===
    smcData: SMCIndicatorData = Field(
        ..., 
        description="Complete SMC indicator data"
    )
    metadata: SMCMetadata = Field(
        ..., 
        description="Summary metadata from SMC indicator"
    )
    tradingSignals: Optional[TradingSignalData] = Field(
        default=None, 
        description="Pre-computed trading signals (optional)"
    )

# ============================================================================
# OUTPUT MODELS - Analysis Results
# ============================================================================

class TrendAnalysisResult(BaseModel):
    """Market trend analysis result"""
    direction: str = Field(..., description="bullish, bearish, or neutral")
    strength: str = Field(..., description="strong, moderate, or weak")
    last_structure_event: str = Field(..., description="BOS or CHoCH")
    confirmation_level: str = Field(..., description="high, medium, or low")
    reasoning: str = Field(..., description="Explanation of trend determination")


class StructureAnalysisResult(BaseModel):
    """Market structure analysis"""
    swing_highs: List[float] = Field(default_factory=list)
    swing_lows: List[float] = Field(default_factory=list)
    bos_events: int = Field(0)
    choch_events: int = Field(0)
    last_bos_direction: Optional[str] = Field(None)
    last_choch_direction: Optional[str] = Field(None)
    structure_bias: str = Field("neutral")
    reasoning: str = Field("")


class OrderBlockAnalysisResult(BaseModel):
    """Order block analysis result"""
    active_bullish_obs: List[Dict[str, Any]] = Field(default_factory=list)
    active_bearish_obs: List[Dict[str, Any]] = Field(default_factory=list)
    total_active: int = Field(0)
    total_mitigated: int = Field(0)
    strongest_demand_zone: Optional[Dict[str, Any]] = Field(None)
    strongest_supply_zone: Optional[Dict[str, Any]] = Field(None)
    reasoning: str = Field("")


class LiquidityAnalysisResult(BaseModel):
    """Liquidity zone analysis"""
    buy_side_liquidity: List[Dict[str, Any]] = Field(default_factory=list)
    sell_side_liquidity: List[Dict[str, Any]] = Field(default_factory=list)
    nearest_eqh: Optional[float] = Field(None)
    nearest_eql: Optional[float] = Field(None)
    potential_sweep_targets: List[float] = Field(default_factory=list)
    reasoning: str = Field("")


class FVGAnalysisResult(BaseModel):
    """Fair Value Gap analysis"""
    active_bullish_fvgs: List[Dict[str, Any]] = Field(default_factory=list)
    active_bearish_fvgs: List[Dict[str, Any]] = Field(default_factory=list)
    total_active: int = Field(0)
    total_mitigated: int = Field(0)
    nearest_unfilled_fvg: Optional[Dict[str, Any]] = Field(None)
    reasoning: str = Field("")


class PremiumDiscountResult(BaseModel):
    """Premium/Discount zone analysis"""
    current_zone: str = Field("equilibrium")
    equilibrium_price: float = Field(0.0)
    premium_threshold: float = Field(0.0)
    discount_threshold: float = Field(0.0)
    distance_to_equilibrium_pct: float = Field(0.0)
    trade_recommendation: str = Field("")
    reasoning: str = Field("")


class EntryZoneResult(BaseModel):
    """Recommended entry zone"""
    zone_type: str = Field(...)
    price_range: Dict[str, float] = Field(...)
    confidence: str = Field("medium")
    reasons: List[str] = Field(default_factory=list)


class TargetLevelResult(BaseModel):
    """Take profit target level"""
    price: float = Field(...)
    target_type: str = Field(...)
    reasoning: str = Field("")


class TradingPlanResult(BaseModel):
    """Complete trading plan with timeframe context"""
    bias: str = Field(...)
    signal_strength: str = Field(...)
    recommended_action: str = Field(...)
    entry_zones: List[EntryZoneResult] = Field(default_factory=list)
    stop_loss: Optional[float] = Field(None)  # Can be null if no valid SL
    stop_loss_reasoning: str = Field("")
    targets: List[TargetLevelResult] = Field(default_factory=list)
    risk_reward_ratio: Optional[float] = Field(None)  # Can be null if can't calculate
    invalidation_level: Optional[float] = Field(None)  # Can be null
    key_warnings: List[str] = Field(default_factory=list)
    # Timeframe context
    timeframe_type: Optional[str] = Field(None, description="scalping, intraday, day_trading, swing_trading")
    hold_duration: Optional[str] = Field(None, description="Expected hold duration")


class SMCAnalysisResult(BaseModel):
    """Complete SMC analysis result"""
    symbol: str
    interval: str
    mode: str = Field("swing", description="Analysis mode: swing or internal")
    timestamp: str
    current_price: float
    
    # Timeframe classification
    timeframe_type: str = Field("swing_trading", description="scalping, intraday, day_trading, swing_trading")
    is_short_term: bool = Field(False, description="True if short-term trading")
    
    trend_analysis: TrendAnalysisResult
    structure_analysis: StructureAnalysisResult
    order_block_analysis: OrderBlockAnalysisResult
    liquidity_analysis: LiquidityAnalysisResult
    fvg_analysis: FVGAnalysisResult
    premium_discount_analysis: PremiumDiscountResult
    trading_plan: TradingPlanResult
    
    executive_summary: str
    analysis_confidence: str
    data_quality_score: float


class SMCAnalyzeResponse(BaseModel):
    """API response model for SMC analysis"""
    status: str = Field("success")
    message: str = Field("")
    session_id: Optional[str] = Field(None)
    data: Optional[SMCAnalysisResult] = Field(None, description="Structured analysis data")
    llm_interpretation: Optional[str] = Field(None, description="AI-generated analysis interpretation")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class SMCStreamChunk(BaseModel):
    """Streaming response chunk"""
    type: str = Field(..., description="analysis, trading_plan, interpretation, complete, error")
    content: Any = Field(None)
    session_id: Optional[str] = Field(None)