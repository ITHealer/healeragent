# from fastapi import APIRouter, HTTPException, Request, Depends, status
# from typing import Dict, Any, Optional
# from pydantic import BaseModel, Field
# from fastapi.responses import StreamingResponse
# import json
# import datetime

# from src.schemas.response import BasicResponse
# from src.handlers.api_key_authenticator_handler import APIKeyAuth
# from src.handlers.options_strategy_handler import OptionsStrategyHandler
# from src.providers.provider_factory import ProviderType
# from src.utils.logger.custom_logging import LoggerMixin
# from src.providers.provider_factory import ModelProviderFactory
# from src.helpers.llm_helper import LLMGeneratorProvider
# from src.agents.memory.memory_manager import MemoryManager
# from src.handlers.llm_chat_handler import ChatHandler, ChatMessageHistory, ChatService
# from src.helpers.language_detector import language_detector, DetectionMethod
# from src.routers.llm_chat import analyze_conversation_importance
# from src.services.background_tasks import trigger_summary_update_nowait

# router = APIRouter(prefix="/options")
# api_key_auth = APIKeyAuth()
# logger_mixin = LoggerMixin()
# logger = logger_mixin.logger
# llm_provider = LLMGeneratorProvider()
# memory_manager = MemoryManager()
# chat_service = ChatService()

# # Request/Response Models
# class OptionsStrategyRequest(BaseModel):
#     session_id: Optional[str]
#     symbol: str = Field(..., description="Stock symbol to analyze")
#     question_input: Optional[str] = Field(None, description="Query from user")
#     target_language: Optional[str]
#     model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
#     provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")
    
# class MarketSentimentRequest(BaseModel):
#     """Request model for market sentiment analysis"""
#     symbol: str = Field(..., description="Stock symbol to analyze")
#     use_llm: bool = Field(False, description="Use LLM for detailed analysis (default: False)")
#     model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
#     provider_type: str = Field("openai", description="Provider type: ollama, openai, gemini")
#     # lookback_days: int = Field(20, description="Number of days to analyze")
    
# class OptionsStrategyResponse(BaseModel):
#     status: str
#     message: str
#     data: Optional[Dict[str, Any]] = None

# class MarketSentimentResponse(BaseModel):
#     status: str
#     message: str
#     data: Optional[Dict[str, Any]] = None


# @router.post("/strategy/analyze", response_model=OptionsStrategyResponse)
# async def analyze_options_strategy(
#     request: OptionsStrategyRequest,
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ) -> OptionsStrategyResponse:
#     """
#     Analyze a stock symbol and recommend options trading strategies.
    
#     This endpoint:
#     1. Fetches market data from FMP
#     2. Calculates technical indicators
#     3. Uses LLM to generate strategy recommendations
#     """
#     try:
#         handler = OptionsStrategyHandler()
        
#         # Get API Key
#         api_key = ModelProviderFactory._get_api_key(request.provider_type)
        
#         result = await handler.analyze_strategy(
#             symbol=request.symbol,
#             model_name=request.model_name,
#             provider_type=request.provider_type,
#             api_key=api_key
#         )
        
#         return OptionsStrategyResponse(
#             status="success",
#             message=f"Options strategy analysis completed for {request.symbol}",
#             data=result
#         )
        
#     except ValueError as e:
#         logger.error(f"Validation error in options strategy analysis: {str(e)}")
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
#     except Exception as e:
#         logger.error(f"Error in options strategy analysis: {str(e)}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# @router.post("/strategy/analyze/stream")
# async def analyze_options_strategy_stream(
#     request_header: Request,
#     request: OptionsStrategyRequest,
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     """
#     Analyze a stock symbol and stream options trading strategy recommendations.
    
#     This endpoint:
#     1. Fetches market data from FMP
#     2. Calculates technical indicators (sent immediately)
#     3. Streams LLM strategy recommendations in real-time
#     """
#     async def format_sse():
#         try:
#             handler = OptionsStrategyHandler()
#             start_time = datetime.datetime.now()
#             user_id = getattr(request_header.state, "user_id", None)
        
#             question_id = None
#             if request.session_id and user_id:
#                 try:
#                     # Create question from symbol and user input
#                     question_content = request.question_input if request.question_input else f"Analyze options strategy for {request.symbol}"
                    
#                     question_id = chat_service.save_user_question(
#                         session_id=request.session_id,
#                         created_at=datetime.datetime.now(),
#                         created_by=user_id,
#                         content=question_content
#                     )
#                     logger.info(f"Saved user question with ID: {question_id}")
#                 except Exception as save_error:
#                     logger.error(f"Error saving user question: {str(save_error)}")
            
#             # 2. Get chat history
#             chat_history = ""
#             if request.session_id:
#                 try:
#                     chat_history = ChatMessageHistory.string_message_chat_history(
#                         session_id=request.session_id
#                     )
#                     logger.info(f"Retrieved chat history for session: {request.session_id}")
#                 except Exception as e:
#                     logger.error(f"Error fetching history: {e}")
            
#             # 3. Get memory context (similar to general/stream)
#             context = ""
#             memory_stats = {}
#             document_references = []
#             if request.session_id and user_id:
#                 try:
#                     # Use question_input or create one from symbol
#                     query_for_memory = request.question_input if request.question_input else f"Options strategy analysis for {data.symbol}"
                    
#                     context, memory_stats, document_references = await memory_manager.get_relevant_context(
#                         session_id=request.session_id,
#                         user_id=user_id,
#                         current_query=query_for_memory,
#                         llm_provider=llm_provider,
#                         max_short_term=5,  # Top 5 relevant recent conversations
#                         max_long_term=3    # Top 3 important past conversations
#                     )
#                     logger.info(f"Retrieved memory context: {memory_stats}")
#                 except Exception as e:
#                     logger.error(f"Error getting memory context: {e}")
            
#             # 4. Build enhanced history with context
#             enhanced_history = ""
#             if context:
#                 enhanced_history = f"[Relevant Context from Memory]\n{context}\n\n"
#             if chat_history:
#                 enhanced_history += f"[Conversation History]\n{chat_history}"
            
#             # Get API Key
#             api_key = ModelProviderFactory._get_api_key(request.provider_type)
            
#             interpretation_chunks = []
#             initial_data = None
            
#             detection_method = ""
#             if len(request.question_input.split()) < 2:
#                 detection_method = DetectionMethod.LLM
#             else:
#                 detection_method = DetectionMethod.LIBRARY

#             # Language detection
#             language_info = await language_detector.detect(
#                 text=request.question_input,
#                 method=detection_method,
#                 system_language=request.target_language,
#                 model_name=request.model_name,
#                 provider_type=request.provider_type,
#                 api_key=api_key
#             )

#             detected_language = language_info["detected_language"]

#             if detected_language:
#                 lang_name = {
#                     "en": "English",
#                     "vi": "Vietnamese", 
#                     "zh": "Chinese",
#                     "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
#                     "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
#                 }.get(detected_language, "the detected language")
                
#             language_instruction = f"""
#             CRITICAL LANGUAGE REQUIREMENT:
#             You MUST respond ENTIRELY in {lang_name} language.
#             - ALL text, explanations, and analysis must be in {lang_name}
#             - Use appropriate financial terminology for {lang_name}
#             - Format numbers and dates according to {lang_name} conventions
#             """

#             # Stream analysis results
#             async for result in handler.stream_strategy_analysis(
#                 symbol=request.symbol,
#                 model_name=request.model_name,
#                 provider_type=request.provider_type,
#                 api_key=api_key, 
#                 language_instruction=language_instruction
#             ):
#                 if result["type"] == "initial_data":
#                     # Send initial technical analysis data
#                     initial_data = result["data"]
#                     yield f"{json.dumps({'initial_data': initial_data})}\n\n"
                    
#                 elif result["type"] == "recommendation_chunk":
#                     # Stream LLM recommendation chunks
#                     chunk = result["content"]
#                     interpretation_chunks.append(chunk)
#                     yield f"{json.dumps({'content': chunk})}\n\n"
                    
#                 elif result["type"] == "final_strategies":
#                     # Send final enhanced strategies
#                     yield f"{json.dumps({'final_strategies': result['data']})}\n\n"
                    
#                 elif result["type"] == "error":
#                     # Handle errors
#                     yield f"{json.dumps({'error': result['error']})}\n\n"
            
#             # Prepare complete response for logging/caching
#             complete_interpretation = ''.join(interpretation_chunks)
            
#             # 8. Analyze conversation importance (like general/stream)
#             importance_score = 0.5  # Default
            
#             if request.session_id and user_id and complete_interpretation:
#                 try:
#                     analysis_model = "gpt-4.1-nano" if request.provider_type == ProviderType.OPENAI else request.model_name
                    
#                     # Analyze importance of options strategy discussion
#                     importance_score = await analyze_conversation_importance(
#                         query=request.question_input if request.question_input else f"Options strategy for {request.symbol}",
#                         response=complete_interpretation,
#                         llm_provider=llm_provider,
#                         model_name=analysis_model,
#                         provider_type=request.provider_type
#                     )
                    
#                     logger.info(f"Strategy analysis importance score: {importance_score}")
                    
#                 except Exception as e:
#                     logger.error(f"Error analyzing conversation importance: {e}")
            
#             # 9. Store conversation in memory system (like general/stream)
#             if request.session_id and user_id and complete_interpretation:
#                 try:
#                     # Store in memory
#                     await memory_manager.store_conversation_turn(
#                         session_id=request.session_id,
#                         user_id=user_id,
#                         query=request.question_input if request.question_input else f"Options strategy analysis for {request.symbol}",
#                         response=complete_interpretation,
#                         metadata={
#                             "type": "options_strategy",
#                             "symbol": request.symbol,
#                             "initial_data": initial_data  # Store technical data as metadata
#                         },
#                         importance_score=importance_score
#                     )
                    
#                     # Calculate response time
#                     response_time = (datetime.datetime.now() - start_time).total_seconds()
                    
#                     # Save assistant response to history
#                     if question_id:
#                         chat_service.save_assistant_response(
#                             session_id=request.session_id,
#                             created_at=datetime.datetime.now(),
#                             question_id=question_id,
#                             content=complete_interpretation,
#                             response_time=response_time
#                         )
#                         logger.info(f"Saved strategy analysis to history")
                    
#                     trigger_summary_update_nowait(session_id=request.session_id, user_id=user_id)

#                 except Exception as save_error:
#                     logger.error(f"Error saving to memory/history: {str(save_error)}")
            
#             # 10. Log completion
#             if initial_data and complete_interpretation:
#                 logger.info(f"Completed streaming options strategy analysis for {request.symbol}")

#             # 11. Send completion signal
#             yield "[DONE]\n\n"
            
#         except Exception as e:
#             logger.error(f"Error in options strategy streaming: {str(e)}")
#             yield f"{json.dumps({'error': str(e)})}\n\n"
#             yield "[DONE]\n\n"
    
#     # Return streaming response with proper headers
#     return StreamingResponse(
#         format_sse(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "X-Content-Type-Options": "nosniff",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no"  # Disable Nginx buffering
#         }
#     )


# @router.post("/sentiment/analyze", response_model=MarketSentimentResponse)
# async def analyze_market_sentiment(
#     request: MarketSentimentRequest,
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ) -> MarketSentimentResponse:
#     """
#     Analyze market sentiment (bullish/bearish) for a stock symbol.
    
#     This endpoint:
#     1. Fetches recent price and volume data
#     2. Calculates momentum indicators
#     3. Determines sentiment using technical analysis
#     4. Optionally uses LLM for detailed explanation (if use_llm=True)
    
#     Default behavior (use_llm=False):
#     - Fast technical analysis based on indicators
#     - Returns sentiment with confidence score
#     - No detailed explanation
    
#     With LLM (use_llm=True):
#     - Enhanced analysis with detailed reasoning
#     - Natural language explanation
#     - More nuanced sentiment assessment
#     """
#     try:
#         handler = OptionsStrategyHandler()
        
#         # Get API Key
#         api_key = ModelProviderFactory._get_api_key(request.provider_type)
        
#         result = await handler.analyze_sentiment(
#             symbol=request.symbol,
#             use_llm=request.use_llm,
#             model_name=request.model_name,
#             provider_type=request.provider_type,
#             api_key=api_key,
#             # lookback_days=request.lookback_days
#         )
        
#         return MarketSentimentResponse(
#             status="success",
#             message=f"Market sentiment analysis completed for {request.symbol}",
#             data=result
#         )
        
#     except ValueError as e:
#         logger.error(f"Validation error in sentiment analysis: {str(e)}")
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
#     except Exception as e:
#         logger.error(f"Error in sentiment analysis: {str(e)}")
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))