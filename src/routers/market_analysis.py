from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any, AsyncGenerator
import datetime
import json
import os
import httpx
from src.schemas.response import (
    MarketMoversAnalysisRequest, 
    MarketMoversAnalysisResponse,
    DiscoveryItemOutput
)
from src.services.discovery_service import DiscoveryService
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.chat_management_helper import ChatService
from src.helpers.redis_cache import get_redis_client
from src.utils.logger.custom_logging import LoggerMixin
import aioredis
from src.providers.provider_factory import ProviderType, ModelProviderFactory
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.helpers.llm_chat_helper import (
    stream_with_heartbeat,
    analyze_conversation_importance,
    SSE_HEADERS,
    DEFAULT_HEARTBEAT_SEC,
    sse_error,
    sse_done
)
from src.agents.memory.memory_manager import get_memory_manager
from src.helpers.language_detector import language_detector, DetectionMethod
from src.services.background_tasks import trigger_summary_update_nowait

router = APIRouter(prefix="/market-analysis")

api_key_auth = APIKeyAuth()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger

discovery_service = DiscoveryService()
llm_generator = LLMGeneratorProvider()
chat_service = ChatService()
chat_service = ChatService()
memory_manager = get_memory_manager()


async def analyze_market_movers_with_llm(
    data: List[DiscoveryItemOutput],
    movers_type: str,  # "gainers" or "losers"
    question_input: str,
    model_name: str,
    provider_type: str
) -> str:
    """
    Analyze market movers data using LLM
    """
    # Convert DiscoveryItemOutput objects to dictionaries
    data_summary = []
    for item in data[:10]:  # Limit to top 10 for LLM context
        # Convert object to dictionary safely
        item_dict = {
            "symbol": getattr(item, 'symbol', 'N/A'),
            "name": getattr(item, 'name', 'N/A'),
            "price": float(getattr(item, 'price', 0)) if getattr(item, 'price', None) else 0,
            "change": float(getattr(item, 'change', 0)) if getattr(item, 'change', None) else 0,
            "changePercent": float(getattr(item, 'percent_change', 0)) if getattr(item, 'percent_change', None) else 0,
            "volume": int(getattr(item, 'volume', 0)) if getattr(item, 'volume', None) else 0,
        }
        
        # Add any additional fields that might exist
        for attr in ['market_cap', 'pe_ratio', 'week_52_high', 'week_52_low']:
            if hasattr(item, attr):
                value = getattr(item, attr)
                if value is not None:
                    item_dict[attr] = value
        
        data_summary.append(item_dict)
    
    # Create analysis prompt
    prompt = f"""You are an expert financial analyst. Analyze the following top {movers_type} in the market and provide actionable insights.
ALWAYS begin with: "I'm your ToponeLogic assistant, As your financial market analyst, I've reviewed the current market data and here are my insights:"

Market {movers_type.upper()} Data:
{json.dumps(data_summary, indent=2)}

User Question: {question_input}

Please provide a comprehensive analysis that includes:
1. Key trends and patterns among these {movers_type}
2. Trading volume analysis - which stocks are seeing unusual volume?
3. Price movement patterns - are these moves sustainable?
4. Notable opportunities or concerns for investors
5. Specific recommendations based on the data
6. Answer any specific questions from the user

Format your response in a clear, structured manner that would be helpful for both novice and experienced investors."""

    try:
        # Get LLM response
        messages = [
            {"role": "system", "content": "You are a professional financial analyst providing market insights."},
            {"role": "user", "content": prompt}
        ]
        
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)

        # Use appropriate provider
        if provider_type == ProviderType.OLLAMA:
            base_url = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "No response.")
        else:
            llm_generator = LLMGeneratorProvider()
            response = await llm_generator.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            )
            return response.get("content", "No response.")
        
    except Exception as e:
        logger.error(f"Error analyzing with LLM: {str(e)}")
        return f"Error generating analysis: {str(e)}"


async def stream_market_movers_with_llm(
    data: List[DiscoveryItemOutput],
    movers_type: str,  # "gainers" or "losers"
    question_input: str,
    target_language: str,
    model_name: str,
    provider_type: str,
    memory_context: str = ""
) -> AsyncGenerator[str, None]:
    """
    Stream market movers analysis using LLM
    """
    detection_method = ""
    if len(question_input.split()) < 2:
        detection_method = DetectionMethod.LLM
    else:
        detection_method = DetectionMethod.LIBRARY

    # Language detection
    language_info = await language_detector.detect(
        text=question_input,
        method=detection_method,
        system_language=target_language,
        model_name=model_name,
        provider_type=provider_type,
        api_key=api_key
    )

    detected_language = language_info["detected_language"]

    if detected_language:
        lang_name = {
            "en": "English",
            "vi": "Vietnamese", 
            "zh": "Chinese",
            "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
            "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
        }.get(detected_language, "the detected language")
        
    language_instruction = f"""
    CRITICAL LANGUAGE REQUIREMENT:
    You MUST respond ENTIRELY in {lang_name} language.
    - ALL text, explanations, and analysis must be in {lang_name}
    - Use appropriate financial terminology for {lang_name}
    - Format numbers and dates according to {lang_name} conventions
    """
    
    # Convert DiscoveryItemOutput objects to dictionaries
    data_summary = []
    for item in data[:10]:  # Limit to top 10 for LLM context
        # Convert object to dictionary safely
        item_dict = {
            "symbol": getattr(item, 'symbol', 'N/A'),
            "name": getattr(item, 'name', 'N/A'),
            "price": float(getattr(item, 'price', 0)) if getattr(item, 'price', None) else 0,
            "change": float(getattr(item, 'change', 0)) if getattr(item, 'change', None) else 0,
            "changePercent": float(getattr(item, 'percent_change', 0)) if getattr(item, 'percent_change', None) else 0,
            "volume": int(getattr(item, 'volume', 0)) if getattr(item, 'volume', None) else 0,
        }
        
        # Add any additional fields that might exist
        for attr in ['market_cap', 'pe_ratio', 'week_52_high', 'week_52_low']:
            if hasattr(item, attr):
                value = getattr(item, attr)
                if value is not None:
                    item_dict[attr] = value
        
        data_summary.append(item_dict)
    
    # Create analysis prompt
    prompt = f"""You are an expert financial analyst. Analyze the following top {movers_type} in the market and provide actionable insights.

Market {movers_type.upper()} Data:
{json.dumps(data_summary, indent=2)}

User Question: {question_input}

Please provide a comprehensive analysis that includes:
1. Key trends and patterns among these {movers_type}
2. Trading volume analysis - which stocks are seeing unusual volume?
3. Price movement patterns - are these moves sustainable?
4. Notable opportunities or concerns for investors
5. Specific recommendations based on the data
6. Answer any specific questions from the user

Format your response in a clear, structured manner that would be helpful for both novice and experienced investors."""

    if memory_context:
        prompt = f"""Previous market analyses and insights:
{memory_context}

Current market {movers_type} analysis:
{prompt}

Consider historical patterns and previous market movements when providing analysis."""
    else:
        prompt = prompt
    
    try:
        # Get LLM response
        messages = [
            {"role": "system", "content": f"You are a professional financial analyst providing market insights.\n {language_instruction} \n"},
            {"role": "user", "content": prompt}
        ]
        
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)

        # Stream response
        llm_generator = LLMGeneratorProvider()
        async for chunk in llm_generator.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            clean_thinking=True
        ):
            yield chunk
        
    except Exception as e:
        logger.error(f"Error streaming analysis with LLM: {str(e)}")
        yield f"Error generating analysis: {str(e)}"


@router.post("/gainers", response_model=MarketMoversAnalysisResponse)
async def analyze_market_gainers(
    request: Request,
    analysis_request: MarketMoversAnalysisRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get top market gainers and analyze them using LLM
    """
    try:
        user_id = getattr(request.state, "user_id", None)
        
        # Config 
        limit = 10

        # Save user question to chat history
        question_id = None
        if analysis_request.session_id and user_id:
            try:
                chat_service = ChatService()
                question_id = chat_service.save_user_question(
                    session_id=analysis_request.session_id,
                    created_at=datetime.datetime.now(),
                    created_by=user_id,
                    content=f"Analyze market gainers: {analysis_request.question_input}"
                )
            except Exception as e:
                logger.error(f"Error saving user question: {str(e)}")
        
        # Get gainers data from discovery service
        logger.info(f"Fetching top {limit} gainers")
        gainers_data = None
        
        try:
            gainers_data = await discovery_service.get_gainers(
                limit=limit,
                redis_client=redis_client
            )
        except Exception as e:
            logger.error(f"Error fetching gainers data: {str(e)}")
            # Try without Redis if error
            gainers_data = await discovery_service.get_gainers(
                limit=limit,
                redis_client=None
            )
        
        if not gainers_data:
            return MarketMoversAnalysisResponse(
                status="error",
                message="No gainers data available",
                data=None,
                # raw_data=None
            )
        
        # Analyze with LLM
        logger.info(f"Analyzing {len(gainers_data)} gainers with {analysis_request.model_name}")
        analysis = await analyze_market_movers_with_llm(
            data=gainers_data,
            movers_type="gainers",
            question_input=analysis_request.question_input,
            model_name=analysis_request.model_name,
            provider_type=analysis_request.provider_type
        )
        
        # Save assistant response to chat history
        if analysis_request.session_id and user_id and question_id:
            try:
                chat_service = ChatService()
                chat_service.save_assistant_response(
                    session_id=analysis_request.session_id,
                    created_at=datetime.datetime.now(),
                    question_id=question_id,
                    content=analysis,
                    response_time=0.1
                )
            except Exception as e:
                logger.error(f"Error saving assistant response: {str(e)}")
        
        # Convert DiscoveryItemOutput objects to dictionaries for serialization
        raw_data_dicts = []
        if gainers_data:
            for item in gainers_data:
                item_dict = {}
                # Convert all attributes to dictionary
                for attr in dir(item):
                    if not attr.startswith('_'):
                        value = getattr(item, attr)
                        # Skip methods
                        if not callable(value):
                            # Handle datetime objects
                            if hasattr(value, 'isoformat'):
                                item_dict[attr] = value.isoformat()
                            # Handle other non-serializable types
                            elif isinstance(value, (str, int, float, bool, type(None))):
                                item_dict[attr] = value
                            else:
                                # Try to convert to string for other types
                                try:
                                    item_dict[attr] = str(value)
                                except:
                                    item_dict[attr] = None
                raw_data_dicts.append(item_dict)
        
        return MarketMoversAnalysisResponse(
            status="success",
            message=f"Successfully analyzed top {len(gainers_data)} market gainers",
            data=analysis,
            # raw_data=raw_data_dicts  # Use converted dictionaries
        )
        
    except Exception as e:
        logger.error(f"Error in analyze_market_gainers: {str(e)}", exc_info=True)
        return MarketMoversAnalysisResponse(
            status="error",
            message=f"Error analyzing market gainers: {str(e)}",
            data=None,
            # raw_data=None
        )


@router.post("/gainers/stream")
async def analyze_market_gainers_stream(
    request: Request,
    analysis_request: MarketMoversAnalysisRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get top market gainers and stream analysis using LLM with memory integration
    """
    user_id = getattr(request.state, "user_id", None)

    async def generate_stream():
        full_response = []
        gainers_data = None
        
        try:
            chat_history = ""
            if analysis_request.session_id:
                try:
                    chat_history = ChatMessageHistory.string_message_chat_history(
                        session_id=analysis_request.session_id
                    )
                except Exception as e:
                    logger.error(f"Error fetching history: {e}")

            context = ""
            memory_stats = {}
            if analysis_request.session_id and user_id:
                try:
                    query_text = analysis_request.question_input or "Analyze market gainers"
                    context, memory_stats, _ = await memory_manager.get_relevant_context(
                        session_id=analysis_request.session_id,
                        user_id=user_id,
                        current_query=query_text,
                        llm_provider=llm_generator,
                        max_short_term=5,
                        max_long_term=3
                    )
                except Exception as e:
                    logger.error(f"Error getting memory context: {e}")
            
            enhanced_history = ""
            if context:
                enhanced_history = f"{context}\n\n"
            if chat_history:
                enhanced_history += f"[Conversation History]\n{chat_history}"
        
            # Config 
            limit = 10
            
            # Get gainers data
            logger.info(f"Fetching top {limit} gainers")
            
            try:
                gainers_data = await discovery_service.get_gainers(
                    limit=limit,
                    redis_client=redis_client
                )
            except Exception as e:
                logger.error(f"Error fetching gainers data: {e}")
                gainers_data = await discovery_service.get_gainers(
                    limit=limit,
                    redis_client=None
                )
            
            if not gainers_data:
                yield sse_error("No gainers data available")
                yield sse_done()
                return
            
            # Convert data to dicts 
            raw_data_dicts = []
            for item in gainers_data:
                item_dict = {}
                for attr in dir(item):
                    if not attr.startswith('_'):
                        value = getattr(item, attr)
                        if not callable(value):
                            if hasattr(value, 'isoformat'):
                                item_dict[attr] = value.isoformat()
                            elif isinstance(value, (str, int, float, bool, type(None))):
                                item_dict[attr] = value
                            else:
                                try:
                                    item_dict[attr] = str(value)
                                except:
                                    item_dict[attr] = None
                raw_data_dicts.append(item_dict)
            
            # Save user question
            question_id = None
            if analysis_request.session_id and user_id:
                try:
                    question_id = chat_service.save_user_question(
                        session_id=analysis_request.session_id,
                        created_at=datetime.datetime.now(),
                        created_by=user_id,
                        content=f"Analyze market gainers: {analysis_request.question_input}"
                    )
                except Exception as e:
                    logger.error(f"Error saving question: {e}")
            
            logger.info(f"Streaming analysis of {len(gainers_data)} gainers with {analysis_request.model_name}")
            
            llm_gen = stream_market_movers_with_llm(
                data=gainers_data,
                movers_type="gainers",
                question_input=analysis_request.question_input,
                target_language=analysis_request.target_language,
                model_name=analysis_request.model_name,
                provider_type=analysis_request.provider_type,
                memory_context=enhanced_history
            )
            
            async for event in stream_with_heartbeat(llm_gen, DEFAULT_HEARTBEAT_SEC):
                if event["type"] == "content":
                    full_response.append(event["chunk"])
                    yield f"{json.dumps({'content': event['chunk']}, ensure_ascii=False)}\n\n"
                elif event["type"] == "heartbeat":
                    yield ": heartbeat\n\n"
                elif event["type"] == "error":
                    yield sse_error(event["error"])
                    break
                elif event["type"] == "done":
                    break

            # Join full response
            analysis = ''.join(full_response)
            
            # Analyze conversation importance 
            importance_score = 0.5
            if analysis_request.session_id and user_id and analysis:
                try:
                    analysis_model = "gpt-4.1-nano" if analysis_request.provider_type == ProviderType.OPENAI else analysis_request.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=analysis_request.question_input or "Analyze market gainers",
                        response=analysis,
                        llm_provider=llm_generator,
                        model_name=analysis_model,
                        provider_type=analysis_request.provider_type
                    )
                    
                    if raw_data_dicts:
                        avg_gain = sum(float(d.get('changePercent', 0)) for d in raw_data_dicts[:5]) / 5
                        if avg_gain > 10:
                            importance_score = min(1.0, importance_score + 0.15)
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")
            
            # Store in memory system 
            if analysis_request.session_id and user_id and analysis:
                try:
                    top_movers = [
                        {"symbol": item.get("symbol"), "change_percent": item.get("changePercent"), "volume": item.get("volume")}
                        for item in raw_data_dicts[:5]
                    ]
                    
                    await memory_manager.store_conversation_turn(
                        session_id=analysis_request.session_id,
                        user_id=user_id,
                        query=f"Analyze market gainers: {analysis_request.question_input}",
                        response=analysis,
                        metadata={
                            "type": "market_movers_analysis",
                            "movers_type": "gainers",
                            "count": len(gainers_data),
                            "top_movers": top_movers,
                            "avg_gain": sum(float(d.get('changePercent', 0)) for d in raw_data_dicts) / len(raw_data_dicts) if raw_data_dicts else 0
                        },
                        importance_score=importance_score
                    )
                    
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=analysis_request.session_id,
                            created_at=datetime.datetime.now(),
                            question_id=question_id,
                            content=analysis,
                            response_time=0.1
                        )

                    trigger_summary_update_nowait(
                        session_id=analysis_request.session_id,
                        user_id=user_id
                    )
                except Exception as e:
                    logger.error(f"Error saving to memory: {e}")
            
            yield sse_done()
            
        except Exception as e:
            logger.error(f"Error in analyze_market_gainers_stream: {e}", exc_info=True)
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )

@router.post("/losers", response_model=MarketMoversAnalysisResponse)
async def analyze_market_losers(
    request: Request,
    analysis_request: MarketMoversAnalysisRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get top market losers and analyze them using LLM
    """
    try:
        user_id = getattr(request.state, "user_id", None)
        
        # Config
        limit = 10

        # Save user question to chat history
        question_id = None
        if analysis_request.session_id and user_id:
            try:
                chat_service = ChatService()
                question_id = chat_service.save_user_question(
                    session_id=analysis_request.session_id,
                    created_at=datetime.datetime.now(),
                    created_by=user_id,
                    content=f"Analyze market losers: {analysis_request.question_input}"
                )
            except Exception as e:
                logger.error(f"Error saving user question: {str(e)}")
        
        # Get losers data from discovery service
        logger.info(f"Fetching top {limit} losers")
        losers_data = None
        
        try:
            losers_data = await discovery_service.get_losers(
                limit=limit,
                redis_client=redis_client
            )
        except Exception as e:
            logger.error(f"Error fetching losers data: {str(e)}")
            # Try without Redis if error
            losers_data = await discovery_service.get_losers(
                limit=limit,
                redis_client=None
            )
        
        if not losers_data:
            return MarketMoversAnalysisResponse(
                status="error",
                message="No losers data available",
                data=None,
                # raw_data=None
            )
        
        # Analyze with LLM
        logger.info(f"Analyzing {len(losers_data)} losers with {analysis_request.model_name}")
        analysis = await analyze_market_movers_with_llm(
            data=losers_data,
            movers_type="losers",
            question_input=analysis_request.question_input,
            model_name=analysis_request.model_name,
            provider_type=analysis_request.provider_type
        )
        
        # Save assistant response to chat history
        if analysis_request.session_id and user_id and question_id:
            try:
                chat_service = ChatService()
                chat_service.save_assistant_response(
                    session_id=analysis_request.session_id,
                    created_at=datetime.datetime.now(),
                    question_id=question_id,
                    content=analysis,
                    response_time=0.1
                )
            except Exception as e:
                logger.error(f"Error saving assistant response: {str(e)}")
        
        # Convert DiscoveryItemOutput objects to dictionaries for serialization
        raw_data_dicts = []
        if losers_data:
            for item in losers_data:
                item_dict = {}
                # Convert all attributes to dictionary
                for attr in dir(item):
                    if not attr.startswith('_'):
                        value = getattr(item, attr)
                        # Skip methods
                        if not callable(value):
                            # Handle datetime objects
                            if hasattr(value, 'isoformat'):
                                item_dict[attr] = value.isoformat()
                            # Handle other non-serializable types
                            elif isinstance(value, (str, int, float, bool, type(None))):
                                item_dict[attr] = value
                            else:
                                # Try to convert to string for other types
                                try:
                                    item_dict[attr] = str(value)
                                except:
                                    item_dict[attr] = None
                raw_data_dicts.append(item_dict)
        
        return MarketMoversAnalysisResponse(
            status="success",
            message=f"Successfully analyzed top {len(losers_data)} market losers",
            data=analysis,
            # raw_data=raw_data_dicts  # Use converted dictionaries
        )
        
    except Exception as e:
        logger.error(f"Error in analyze_market_losers: {str(e)}", exc_info=True)
        return MarketMoversAnalysisResponse(
            status="error",
            message=f"Error analyzing market losers: {str(e)}",
            data=None,
            # raw_data=None
        )
    

@router.post("/losers/stream")
async def analyze_market_losers_stream(
    request: Request,
    analysis_request: MarketMoversAnalysisRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client)
):
    """
    Get top market losers and stream analysis using LLM
    """
    user_id = getattr(request.state, "user_id", None)
    
    async def generate_stream():
        full_response = []
        losers_data = None

        try:
            chat_history = ""
            if analysis_request.session_id:
                try:
                    chat_history = ChatMessageHistory.string_message_chat_history(
                        session_id=analysis_request.session_id
                    )
                except Exception as e:
                    logger.error(f"Error fetching history: {e}")

            context = ""
            memory_stats = {}
            if analysis_request.session_id and user_id:
                try:
                    query_text = analysis_request.question_input or "Analyze market losers"
                    context, memory_stats, _ = await memory_manager.get_relevant_context(
                        session_id=analysis_request.session_id,
                        user_id=user_id,
                        current_query=query_text,
                        llm_provider=llm_generator,
                        max_short_term=5,
                        max_long_term=3
                    )
                except Exception as e:
                    logger.error(f"Error getting memory context: {e}")
            
            enhanced_history = ""
            if context:
                enhanced_history = f"{context}\n\n"
            if chat_history:
                enhanced_history += f"[Conversation History]\n{chat_history}"

            # Config
            limit = 10

            # Get losers data
            logger.info(f"Fetching top {limit} losers")
            
            try:
                losers_data = await discovery_service.get_losers(
                    limit=limit,
                    redis_client=redis_client
                )
            except Exception as e:
                logger.error(f"Error fetching losers data: {e}")
                losers_data = await discovery_service.get_losers(
                    limit=limit,
                    redis_client=None
                )
            
            if not losers_data:
                yield sse_error("No losers data available")
                yield sse_done()
                return
            
            # Convert data to dicts
            raw_data_dicts = []
            for item in losers_data:
                item_dict = {}
                for attr in dir(item):
                    if not attr.startswith('_'):
                        value = getattr(item, attr)
                        if not callable(value):
                            if hasattr(value, 'isoformat'):
                                item_dict[attr] = value.isoformat()
                            elif isinstance(value, (str, int, float, bool, type(None))):
                                item_dict[attr] = value
                            else:
                                try:
                                    item_dict[attr] = str(value)
                                except:
                                    item_dict[attr] = None
                raw_data_dicts.append(item_dict)
            
            # Save user question
            question_id = None
            if analysis_request.session_id and user_id:
                try:
                    question_id = chat_service.save_user_question(
                        session_id=analysis_request.session_id,
                        created_at=datetime.datetime.now(),
                        created_by=user_id,
                        content=f"Analyze market losers: {analysis_request.question_input}"
                    )
                except Exception as e:
                    logger.error(f"Error saving question: {e}")
            
            logger.info(f"Streaming analysis of {len(losers_data)} losers with {analysis_request.model_name}")
            
            llm_gen = stream_market_movers_with_llm(
                data=losers_data,
                movers_type="losers",
                question_input=analysis_request.question_input,
                target_language=analysis_request.target_language,
                model_name=analysis_request.model_name,
                provider_type=analysis_request.provider_type,
                memory_context=enhanced_history
            )
            
            async for event in stream_with_heartbeat(llm_gen, DEFAULT_HEARTBEAT_SEC):
                if event["type"] == "content":
                    full_response.append(event["chunk"])
                    yield f"{json.dumps({'content': event['chunk']}, ensure_ascii=False)}\n\n"
                elif event["type"] == "heartbeat":
                    yield ": heartbeat\n\n"
                elif event["type"] == "error":
                    yield sse_error(event["error"])
                    break
                elif event["type"] == "done":
                    break
            
            # Join full response
            analysis = ''.join(full_response)
            
            # Analyze conversation importance
            importance_score = 0.5
            if analysis_request.session_id and user_id and analysis:
                try:
                    analysis_model = "gpt-4.1-nano" if analysis_request.provider_type == ProviderType.OPENAI else analysis_request.model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=analysis_request.question_input or "Analyze market losers",
                        response=analysis,
                        llm_provider=llm_generator,
                        model_name=analysis_model,
                        provider_type=analysis_request.provider_type
                    )
                    
                    if raw_data_dicts:
                        avg_loss = sum(float(d.get('changePercent', 0)) for d in raw_data_dicts[:5]) / 5
                        if avg_loss < -10:
                            importance_score = min(1.0, importance_score + 0.15)
                except Exception as e:
                    logger.error(f"Error analyzing importance: {e}")
            
            # Store in memory system 
            if analysis_request.session_id and user_id and analysis:
                try:
                    top_movers = [
                        {"symbol": item.get("symbol"), "change_percent": item.get("changePercent"), "volume": item.get("volume")}
                        for item in raw_data_dicts[:5]
                    ]
                    
                    await memory_manager.store_conversation_turn(
                        session_id=analysis_request.session_id,
                        user_id=user_id,
                        query=f"Analyze market losers: {analysis_request.question_input}",
                        response=analysis,
                        metadata={
                            "type": "market_movers_analysis",
                            "movers_type": "losers",
                            "count": len(losers_data),
                            "top_movers": top_movers,
                            "avg_loss": sum(float(d.get('changePercent', 0)) for d in raw_data_dicts) / len(raw_data_dicts) if raw_data_dicts else 0
                        },
                        importance_score=importance_score
                    )
                    
                    if question_id:
                        chat_service.save_assistant_response(
                            session_id=analysis_request.session_id,
                            created_at=datetime.datetime.now(),
                            question_id=question_id,
                            content=analysis,
                            response_time=0.1
                        )

                    trigger_summary_update_nowait(
                        session_id=analysis_request.session_id,
                        user_id=user_id
                    )
                except Exception as e:
                    logger.error(f"Error saving to memory: {e}")
            
            yield sse_done()
            
        except Exception as e:
            logger.error(f"Error in analyze_market_losers_stream: {e}", exc_info=True)
            yield sse_error(str(e))
            yield sse_done()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS
    )