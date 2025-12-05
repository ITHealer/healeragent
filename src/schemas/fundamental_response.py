from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class KeyInsights(BaseModel):
    revenue_trend: str = Field(..., description="Revenue growth classification")
    profitability_trend: str = Field(..., description="Profitability trend")
    cash_flow_health: str = Field(..., description="Cash flow status")
    debt_management: str = Field(..., description="Debt management status")
    growth_quality_score: float = Field(..., description="Growth quality score (0-100)")

class FundamentalAnalysisResult(BaseModel):
    symbol: str
    period: str
    latest_date: str
    raw_data: List[Dict[str, Any]]
    analysis: str = Field(..., description="AI-generated analysis text")
    key_insights: KeyInsights

class FundamentalAnalysisResponse(BaseModel):
    message: str
    status: str
    symbol: str
    period: str
    model_used: Optional[str] = None
    provider_used: Optional[str] = None
    analysis_result: Optional[FundamentalAnalysisResult] = None