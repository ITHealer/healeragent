import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.news_aggregator.schemas.request import NewsAggregatorRequest, DigestRequest
from src.news_aggregator.schemas.response import NewsAggregatorResponse, DigestResponse
from src.news_aggregator.services.aggregator_service import NewsAggregatorService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news-aggregator")

class NewsAggregatorRequestAPI(BaseModel):
    # Filter criteria
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to match in title/content"
    )
    symbols: list[str] = Field(
        default_factory=list,
        description="Stock/crypto symbols to filter"
    )
    categories: list[str] = Field(
        default=["stock", "crypto", "general"],
        description="News categories: stock, crypto, forex, general, press_release, fmp_article"
    )
    
    # Time range
    time_range_hours: int = Field(default=24, ge=1, le=168)
    
    # Pagination
    max_articles: int = Field(default=50, ge=10, le=200)
    page: int = Field(default=0, ge=0)
    limit_per_category: int = Field(default=30, ge=10, le=100)
    
    # Options
    generate_digest: bool = Field(default=False)
    target_language: str = Field(default="en")
    include_full_content: bool = Field(default=False)
    
    # Provider options
    use_tavily: bool = Field(default=True)
    tavily_max_results: int = Field(default=10, ge=1, le=20)
    
    # LLM options
    model_name: str = Field(default="gpt-4.1-nano")
    provider_type: str = Field(default="openai")


class DigestRequestAPI(BaseModel):
    """Request to generate digest from already-fetched articles"""
    article_ids: list[str] = Field(..., min_length=1)
    target_language: str = Field(default="en")
    model_name: str = Field(default="gpt-4.1-nano")
    provider_type: str = Field(default="openai")
    max_top_stories: int = Field(default=3, ge=3, le=10)


aggregator_service = NewsAggregatorService()

@router.post("/aggregate")
async def aggregate_news(
    request: NewsAggregatorRequestAPI,
    # api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)  # Uncomment for auth
) -> Dict[str, Any]:
    """
    Main endpoint - Fetch, dedupe, match, and optionally summarize news.
    
    Web BE gọi endpoint này với keywords/symbols của user.
    AI xử lý và trả về kết quả.
    
    **Request Example:**
    ```json
    {
        "keywords": ["AAPL", "Apple", "iPhone"],
        "symbols": ["AAPL", "NVDA"],
        "categories": ["stock", "general"],
        "time_range_hours": 24,
        "max_articles": 50,
        "generate_digest": true,
        "target_language": "vi"
    }
    ```
    
    **Response:**
    - status: success/error
    - articles: List of matched articles sorted by relevance
    - digest: LLM-generated summary (if generate_digest=true)
    - metadata: Processing stats for monitoring
    """
    try:
        internal_request = NewsAggregatorRequest(
            keywords=request.keywords,
            symbols=request.symbols,
            categories=request.categories,
            time_range_hours=request.time_range_hours,
            max_articles=request.max_articles,
            page=request.page,
            limit_per_category=request.limit_per_category,
            generate_digest=request.generate_digest,
            target_language=request.target_language,
            include_full_content=request.include_full_content,
            use_tavily=request.use_tavily,
            tavily_max_results=request.tavily_max_results,
            model_name=request.model_name,
            provider_type=request.provider_type,
        )
        
        # Execute aggregation
        result = await aggregator_service.aggregate(internal_request)
        
        return result.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Router] Aggregation error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Aggregation failed: {str(e)}"
        )


@router.post("/digest")
async def generate_digest_only(
    request: DigestRequestAPI,
) -> Dict[str, Any]:
    """
    Generate digest from list articles.
    
    Use case: Web BE đã có articles (từ call trước),
    chỉ muốn generate summary mới.
    
    Note: This endpoint requires article data to be cached or re-fetched.
    For now, returns error as articles need to be in memory.
    """
    # This would require a cache or storage of articles
    # For stateless design, recommend using /aggregate with generate_digest=true
    
    return {
        "status": "error",
        "message": "Use /aggregate endpoint with generate_digest=true instead. "
                   "This endpoint requires article caching which is not implemented "
                   "in stateless mode.",
        "recommendation": "Call /aggregate with your filters and generate_digest=true"
    }


# @router.get("/categories")
# async def list_categories() -> Dict[str, Any]:
#     """
#     List available news categories.
#     """
#     return {
#         "categories": [
#             {"id": "stock", "name": "Stock News", "description": "Stock market news from FMP"},
#             {"id": "crypto", "name": "Crypto News", "description": "Cryptocurrency news"},
#             {"id": "forex", "name": "Forex News", "description": "Foreign exchange news"},
#             {"id": "general", "name": "General News", "description": "General financial news"},
#             {"id": "press_release", "name": "Press Releases", "description": "Company press releases"},
#             {"id": "fmp_article", "name": "FMP Articles", "description": "Articles by FMP analysts"},
#         ]
#     }