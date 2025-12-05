from fastapi import APIRouter, HTTPException, Query, Path
from typing import List, Optional

from src.schemas.response import ComprehensiveAnalysisRequest
from src.handlers.comprehensive_analysis_handler import ComprehensiveAnalysisHandler


router = APIRouter(prefix="/stock-analysis")
comprehensive_handler = ComprehensiveAnalysisHandler()

@router.post("/comprehensive-analysis")
async def get_comprehensive_analysis(request: ComprehensiveAnalysisRequest):
    """
    Get aggregate analysis for stock codes.
    """
    try:
        result = await comprehensive_handler.perform_comprehensive_analysis(
            symbol=request.symbol,
            lookback_days=request.lookback_days,
            analyses=request.analyses
        )
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error analyzing {request.symbol}: {str(e)}",
            "data": None
        }
    

@router.get("/comprehensive-analysis/{symbol}")
async def get_comprehensive_analysis_for_symbol(
    symbol: str = Path(..., description="Stock symbol"),
    lookback_days: int = Query(252, description="Number of historical days"),
    analyses: Optional[str] = Query(None, description="List of comma separated analysis")
):
    try:
        analyses_list = None
        if analyses:
            analyses_list = [a.strip() for a in analyses.split(",")]
            
        result = await comprehensive_handler.perform_comprehensive_analysis(
            symbol=symbol,
            lookback_days=lookback_days,
            analyses=analyses_list
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))