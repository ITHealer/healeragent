import json
from fastapi import APIRouter, Depends, HTTPException, Request, status
from typing import Optional, Dict, Any, Generic, List, TypeVar
from pydantic import BaseModel, Field

from src.schemas.question_suggestion import (
    QuestionSuggestionRequest,
    QuestionSuggestionRequest_v2,
    SuggestedQuestion_v2,
    SuggestedQuestion,
    UserLevel,
    AssetType
)

from src.models.equity import APIResponse, APIResponseData
from src.services.question_suggestion_service import QuestionSuggestionService, QuestionType, UserLevel
from src.services.news_service import NewsService
from src.helpers.chat_management_helper import ChatService
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.redis_cache import get_redis_client
from src.utils.config import settings
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.handlers.api_key_authenticator_handler import APIKeyAuth

import logging
import aioredis

logger = logging.getLogger(__name__)

router = APIRouter()
api_key_auth = APIKeyAuth()

# Initialize services
news_service = NewsService()
chat_service = ChatService() 
llm_service = LLMGeneratorProvider()

question_suggestion_service = QuestionSuggestionService(
    news_service=news_service,
    chat_service=chat_service,
    llm_service=llm_service
)

DataType = TypeVar('DataType')

class APIResponse(BaseModel, Generic[DataType]): 
    message: str = Field("OK", description="General notice stating the result of the request")
    status: str = Field("200", description="HTTP status code as a string")
    provider_used: Optional[str] = Field(None, description="The final data provider used for the request")

    data: List[DataType] = Field(..., description="List of data items of the specified type")

    class Config:
        populate_by_name = True 


@router.post("/question-suggestions",
             response_model=APIResponse[SuggestedQuestion],
             summary="Suggest questions based on conversation context")
async def get_question_suggestions(
    request: QuestionSuggestionRequest,
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Enhanced question suggestions using conversation summary
    """
    try:
        logger.info(f"Getting question suggestions for session: {request.session_id}")
        
        # Use enhanced method if session_id is provided
        if request.session_id:
            suggestions = await question_suggestion_service.generate_question_suggestions_with_context(
                data=request
            )
        else:
            # Fallback to basic generation without context
            suggestions = await question_suggestion_service.generate_question_suggestions(
                request=request
            )
        
        if not suggestions:
            raise HTTPException(
                status_code=500,
                detail="Unable to create suggestion questions"
            )
        
        api_response = APIResponse[SuggestedQuestion](
            message="OK",
            status="200",
            provider_used="internal",
            data=suggestions
        )
        
        return api_response
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error creating suggestions: {str(e)}"
        )


@router.post("/question-suggestions-cards",
             response_model=APIResponse[SuggestedQuestion_v2],
             summary="Generate question suggestions based on market context and user preferences")
async def get_question_suggestions_v2(
    request: QuestionSuggestionRequest_v2,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Generate intelligent question suggestions for users based on:
    
    - **Asset Type**: Focus on specific market
      - `stock`: Stock market questions only
      - `crypto`: Cryptocurrency questions only
    
    - **Question Type**: The type of analysis the user is interested in
      - `technical_expert`: Technical analysis questions (charts, indicators, patterns)
      - `crypto_analyst`: Cryptocurrency-focused questions
      - `fundamental_guru`: Fundamental analysis questions (financials, valuations)
      - `sentiment_analyzer`: Market sentiment and news-based questions
    
    - **User Level**: Complexity of questions based on user expertise
      - `beginner`: Simple, educational questions
      - `intermediate`: More detailed analysis questions
      - `advanced`: Complex, sophisticated market questions
    
    - **Market Context**: Latest news and events from the market
      - Stock news for stock asset type
      - Crypto news for crypto asset type
      - Market events and press releases
    
    Returns 8 contextually relevant questions with symbols and relevance reasons.
    """
    
    try:
        # Validate asset type
        valid_asset_types = [AssetType.STOCK, AssetType.CRYPTO]
        if request.asset_type not in valid_asset_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid asset_type. Must be one of: {', '.join(valid_asset_types)}"
            )
        
        # Validate question type
        valid_types = [QuestionType.TECHNICAL, QuestionType.CRYPTO, 
                      QuestionType.FUNDAMENTAL, QuestionType.SENTIMENT]
        if request.question_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid question_type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Validate user level
        valid_levels = [UserLevel.BEGINNER, UserLevel.INTERMEDIATE, UserLevel.ADVANCED]
        if request.user_level not in valid_levels:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid user_level. Must be one of: {', '.join(valid_levels)}"
            )
        
        logger.info(f"Generating question suggestions for: asset_type={request.asset_type}, type={request.question_type}, level={request.user_level}")
        
        # Check cache first - include asset_type in cache key
        cache_key = f"question_suggestions:{request.asset_type}:{request.question_type}:{request.user_level}"
        
        if redis_client:
            try:
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    logger.info(f"Cache HIT for question suggestions: {cache_key}")
                    suggestions = [SuggestedQuestion_v2(**item) for item in json.loads(cached_data)]
                    
                    return APIResponse[SuggestedQuestion_v2](
                        message="OK (cached)",
                        status="200",
                        provider_used="cached",
                        data=suggestions
                    )
            except Exception as cache_error:
                logger.error(f"Cache error: {cache_error}")
        
        logger.info(f"Cache MISS for question suggestions: {cache_key}. Generating new suggestions.")
        
        # Create schema request object
        schema_request = QuestionSuggestionRequest_v2(
            asset_type=request.asset_type,
            question_type=request.question_type,
            user_level=request.user_level,
            model_name=request.model_name,
            provider_type=request.provider_type
        )
        
        # Generate suggestions
        suggestions = await question_suggestion_service.generate_question_suggestions_v2(
            request=schema_request,
            k=8  # Generate 8 questions
        )
        
        if not suggestions:
            raise HTTPException(
                status_code=500,
                detail="Unable to generate question suggestions"
            )
        
        # Cache the results
        if redis_client and suggestions:
            try:
                cache_data = json.dumps([s.dict() for s in suggestions])
                # Use CACHE_TTL_NEWS or default 10 minutes (600 seconds)
                cache_ttl = getattr(settings, 'CACHE_TTL_SUGGESTIONS', getattr(settings, 'CACHE_TTL_NEWS', 600))
                await redis_client.setex(
                    cache_key,
                    cache_ttl,
                    cache_data
                )
                logger.info(f"Cached question suggestions for {cache_key}")
            except Exception as cache_error:
                logger.error(f"Error caching suggestions: {cache_error}")
        
        # Build response
        api_response = APIResponse[SuggestedQuestion_v2](
            message=f"Successfully generated {len(suggestions)} question suggestions for {request.asset_type}",
            status="200",
            provider_used=f"{request.provider_type}:{request.model_name}",
            data=suggestions
        )
        
        # Log context for debugging
        context = await question_suggestion_service._build_context_v2(schema_request)
        logger.debug(f"Context used: asset_type={request.asset_type}, {len(context.get('news', []))} news items, {len(context.get('symbols', []))} symbols")
        
        return api_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating question suggestions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
