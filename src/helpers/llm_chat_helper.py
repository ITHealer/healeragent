import asyncio
import os
import json
import httpx
import traceback
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, Any, Optional, List

from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.agents.memory.memory_manager import get_memory_manager
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod
from src.handlers.llm_chat_handler import ChatService, ChatMessageHistory, ConversationAnalysis
from src.services.news_service import NewsService
from src.models.equity import NewsItemOutput, APIResponse, APIResponseData
from src.handlers.comprehensive_analysis_handler import ComprehensiveAnalysisHandler
from src.helpers.redis_cache import get_cache, set_cache, get_redis_client_llm


logger_mixin = LoggerMixin()
logger = logger_mixin.logger

# Constants
DEFAULT_HEARTBEAT_SEC = 15

# SSE Headers
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "X-Content-Type-Options": "nosniff"
}

async def stream_with_heartbeat(
    llm_generator: AsyncGenerator[str, None],
    heartbeat_sec: int = DEFAULT_HEARTBEAT_SEC
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Heartbeat streaming generator.
    """
    pending_task: Optional[asyncio.Task] = None
    
    try:
        while True:
            if pending_task is None:
                pending_task = asyncio.create_task(anext(llm_generator))
            
            done, _ = await asyncio.wait({pending_task}, timeout=heartbeat_sec)
            
            if done:
                try:
                    chunk = pending_task.result()
                except StopAsyncIteration:
                    yield {"type": "done"}
                    break
                except Exception as e:
                    logger.error(f"LLM generator error: {e}")
                    yield {"type": "error", "error": str(e)}
                    break
                finally:
                    pending_task = None
                
                if chunk:
                    yield {"type": "content", "chunk": chunk}
                    await asyncio.sleep(0)
            else:
                yield {"type": "heartbeat"}
                
    except asyncio.CancelledError:
        logger.debug("Stream cancelled by client disconnect")
        raise
    finally:
        # Cleanup pending task
        if pending_task is not None and not pending_task.done():
            pending_task.cancel()
            try:
                await pending_task
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
        
        # Close generator
        if llm_generator is not None and hasattr(llm_generator, 'aclose'):
            try:
                await llm_generator.aclose()
            except Exception:
                pass


def sse_error(message: str) -> str:
    """Format SSE error message with type field for FE consistency."""
    return f"{json.dumps({'type': 'error', 'error': message}, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    """Format SSE done message with type field for FE consistency."""
    return f"{json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


def sse_content(content: str) -> str:
    """Format SSE content message with type field for FE consistency."""
    return f"{json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"


def sse_heartbeat() -> str:
    """Format SSE heartbeat comment to keep connection alive."""
    return ": heartbeat\n\n"


@dataclass
class StreamResult:
    """
    Container for streaming result metadata.
    
    Tracks accumulated response, timing, and errors during streaming.
    """
    accumulated_response: str = ""
    chunk_count: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    importance_score: float = 0.5
    error: Optional[str] = None
    
    @property
    def response_time(self) -> float:
        """Calculate total response time in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    def append_chunk(self, chunk: str) -> None:
        """Append chunk to accumulated response."""
        self.accumulated_response += chunk
        self.chunk_count += 1


async def analyze_conversation_importance(
    query: str,
    response: str,
    llm_provider: LLMGeneratorProvider,
    model_name: str,
    provider_type: str
) -> float:
    """
    Use LLM to analyze conversation importance and extract metadata for financial chatbot.
    
    Args:
        query: User's question
        response: Assistant's response
        llm_provider: LLM provider instance
        model_name: Model to use for analysis
        provider_type: Provider type (openai, ollama, etc.)
        
    Returns:
        Importance score between 0.0 and 1.0
    """
    try:
        # Normalize provider type
        if isinstance(provider_type, str):
            provider_type_lower = provider_type.lower()
            if provider_type_lower == "openai":
                provider_enum = ProviderType.OPENAI
            elif provider_type_lower == "ollama":
                provider_enum = ProviderType.OLLAMA
            else:
                provider_enum = ProviderType.OPENAI  # Default fallback
        else:
            provider_enum = provider_type
        
        # Build analysis prompt with explicit JSON instruction
        analysis_prompt = f"""Analyze this financial conversation and provide importance score (0.0-1.0):

User Query: {query}
Assistant Response: {response[:1000]}...

Score based on:
- Financial relevance and complexity
- Actionable insights provided
- Educational/strategic value
- Whether it's follow-up or references context

Focus on: trading, investment, market analysis, portfolio, risk management.

IMPORTANT: Respond with ONLY a valid JSON object in this exact format (no markdown, no explanation):
{{"importance_score": 0.8, "reasoning": "Brief explanation"}}"""

        messages = [
            {
                "role": "system", 
                "content": "You are a financial conversation analyst. Always respond with valid JSON only."
            },
            {
                "role": "user", 
                "content": analysis_prompt
            }
        ]
        
        # Get API key
        api_key = ModelProviderFactory._get_api_key(provider_enum)
        
        # Prepare structured output format (only for OpenAI)
        structured_format = None
        if provider_enum == ProviderType.OPENAI:
            structured_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "importance_analysis",
                    "schema": ConversationAnalysis.model_json_schema(),
                    "strict": False 
                }
            }
        
        # Call LLM
        llm_response = await llm_provider.generate_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_enum,
            api_key=api_key,
            temperature=0.1,
            response_format=structured_format
        )
        
        # Get content from response
        content = llm_response.get("content", "")
        
        # Check if content is empty
        if not content or not content.strip():
            logger.warning("Empty response from LLM for importance analysis")
            return 0.5  # Default medium importance
        
        # Clean content (remove markdown code blocks if present)
        content_cleaned = content.strip()
        if content_cleaned.startswith("```json"):
            content_cleaned = content_cleaned.replace("```json", "").replace("```", "").strip()
        elif content_cleaned.startswith("```"):
            content_cleaned = content_cleaned.replace("```", "").strip()
        
        # Try to parse as JSON
        try:
            parsed_data = json.loads(content_cleaned)
            
            # Validate with Pydantic if possible
            try:
                analysis = ConversationAnalysis.model_validate(parsed_data)
                importance_score = analysis.importance_score
            except Exception:
                # Fallback to direct dict access
                importance_score = float(parsed_data.get("importance_score", 0.5))
            
            # Validate score range
            importance_score = max(0.0, min(1.0, importance_score))
            
            logger.info(f"Analyzed importance: {importance_score}")
            return importance_score
            
        except json.JSONDecodeError as je:
            # JSON parsing failed - try regex fallback
            logger.warning(f"JSON decode failed for content: {content_cleaned[:100]}... Error: {je}")
            
            # Try to extract score using regex
            import re
            score_match = re.search(r'"importance_score"\s*:\s*([0-9.]+)', content_cleaned)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))
                    logger.info(f"Extracted importance score via regex: {score}")
                    return score
                except ValueError:
                    pass
            
            # Final fallback based on response length and keywords
            score = 0.5  # Default medium
            
            # Simple heuristic: check for financial keywords
            financial_keywords = ['trade', 'invest', 'portfolio', 'stock', 'market', 'risk', 'analysis', 'buy', 'sell']
            keyword_count = sum(1 for keyword in financial_keywords if keyword.lower() in response.lower())
            
            if keyword_count >= 3:
                score = 0.7
            elif keyword_count >= 1:
                score = 0.6
            
            logger.info(f"Using fallback importance score: {score}")
            return score
        
    except Exception as e:
        logger.error(f"Error analyzing conversation importance with LLM: {e}")
        # Return default score instead of failing
        return 0.5
    

class LLMStreamingService(LoggerMixin):
    """
    Unified service for all streaming operations with FULL PROMPTS.
    """
    
    def __init__(self):
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()
        self.memory_manager = get_memory_manager()
        self.chat_service = ChatService()
        self.news_service = NewsService()
        self.analysis_handler = ComprehensiveAnalysisHandler()
    
    # =========================================================================
    # CORE: Streaming Loop with Heartbeat
    # =========================================================================
    
    async def _stream_with_heartbeat_core(
        self,
        llm_generator: AsyncGenerator[str, None],
        result: StreamResult,
        heartbeat_sec: int = DEFAULT_HEARTBEAT_SEC
    ) -> AsyncGenerator[str, None]:
        """
        Core streaming loop with safe heartbeat pattern.
        """
        pending_task: Optional[asyncio.Task] = None
        
        try:
            while True:
                if pending_task is None:
                    pending_task = asyncio.create_task(anext(llm_generator))
                
                done, _ = await asyncio.wait({pending_task}, timeout=heartbeat_sec)
                
                if done:
                    try:
                        chunk = pending_task.result()
                    except StopAsyncIteration:
                        break
                    except Exception as e:
                        self.logger.error(f"LLM generator error: {e}")
                        result.error = str(e)
                        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
                        break
                    finally:
                        pending_task = None
                    
                    if chunk:
                        result.append_chunk(chunk)
                        yield f"{json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                        # yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0)
                else:
                    yield ": heartbeat\n\n"
                    
        except asyncio.CancelledError:
            self.logger.debug("Stream core cancelled by client disconnect")
            raise
        finally:
            if pending_task is not None and not pending_task.done():
                pending_task.cancel()
                try:
                    await pending_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
    
    async def _close_generator_safe(
        self, 
        llm_generator: Optional[AsyncGenerator]
    ) -> None:
        """Safely close an async generator."""
        if llm_generator is not None and hasattr(llm_generator, 'aclose'):
            try:
                await llm_generator.aclose()
                self.logger.debug("LLM generator closed successfully")
            except Exception as e:
                self.logger.warning(f"Error closing LLM generator: {e}")
    
    async def _get_enhanced_context(
        self,
        session_id: str,
        user_id: int,
        question_input: str,
        timeout_seconds: float = 5.0  # Default 5 second timeout
    ) -> str:
        """
        Get chat history + memory context with timeout protection.

        This method has a timeout to prevent blocking the main request
        if memory retrieval is slow or hangs.
        """
        enhanced_history = ""

        chat_history = ""
        if session_id:
            try:
                chat_history = ChatMessageHistory.string_message_chat_history(
                    session_id=session_id
                )
            except Exception as e:
                self.logger.error(f"Error fetching chat history: {e}")

        memory_context = ""
        if session_id and user_id:
            try:
                # Add timeout to prevent blocking if memory retrieval is slow
                memory_context, _, _ = await asyncio.wait_for(
                    self.memory_manager.get_relevant_context(
                        session_id=session_id,
                        user_id=user_id,
                        current_query=question_input,
                        llm_provider=self.llm_provider,
                        max_short_term=5,
                        max_long_term=3
                    ),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Memory context retrieval timed out after {timeout_seconds}s")
                # Continue without memory context rather than blocking
            except Exception as e:
                self.logger.error(f"Error getting memory context: {e}")

        if memory_context:
            enhanced_history = f"[Memory Context]\n{memory_context}\n\n"
        if chat_history:
            enhanced_history += f"[Conversation History]\n{chat_history}"

        return enhanced_history
    
    # =========================================================================
    # SSE Event Helpers
    # =========================================================================
    # Standard format
    # def _sse_status(self, message: str) -> str:
    #     return f"data: {json.dumps({'type': 'status', 'content': message}, ensure_ascii=False)}\n\n"
    
    # def _sse_error(self, message: str) -> str:
    #     return f"data: {json.dumps({'type': 'error', 'content': message}, ensure_ascii=False)}\n\n"
    
    # def _sse_done(self) -> str:
    #     return "data: [DONE]\n\n"
    
    # def _sse_retry(self, ms: int = 1000) -> str:
    #     return f"retry: {ms}\n\n"

    # Custom format
    def _sse_status(self, message: str) -> str:
        return "" 
    
    def _sse_error(self, message: str) -> str:
        return f"{json.dumps({'error': message}, ensure_ascii=False)}\n\n"

    def _sse_done(self) -> str:
        return "[DONE]\n\n"

    def _sse_retry(self, ms: int = 1000) -> str:
        return ""
    
    # =========================================================================
    # News Fetching for Stock Analysis
    # =========================================================================
    
    async def _get_news_context(
        self,
        symbol: str,
        limit: int = 5,
        lookback_days: int = 7
    ) -> str:
        """Fetch relevant news for a stock symbol."""
        news_context = ""
        
        try:
            to_date = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            redis_client = await get_redis_client_llm()
            cache_key = f"company_news_{symbol.upper()}_limit_{limit}"
            
            cached_response = await get_cache(redis_client, cache_key, APIResponse[NewsItemOutput])
            
            if cached_response and cached_response.data and cached_response.data.data:
                news_items = cached_response.data.data
            else:
                news_items = await self.news_service.get_company_news(
                    symbol.upper(), limit, from_date=from_date, to_date=to_date
                )
                
                if news_items:
                    response_data_payload = APIResponseData[NewsItemOutput](data=news_items)
                    api_response = APIResponse[NewsItemOutput](
                        message="OK",
                        status="200",
                        provider_used="fmp",
                        data=response_data_payload
                    )
                    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_CHART)
            
            if news_items:
                news_context = "RELEVANT NEWS:\n\n"
                for i, news_item in enumerate(news_items[:5], 1):
                    if news_item.description:
                        news_context += f"{i}. {news_item.title}\n"
                        news_context += f"   Date: {news_item.date}\n"
                        news_context += f"   Source: {news_item.source_site or 'Unknown'}\n"
                        news_context += f"   Description: {news_item.description}\n\n"
            
            if redis_client:
                await redis_client.close()
                
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")
        
        return news_context
    
    # =========================================================================
    # PUBLIC: General Chat Streaming
    # =========================================================================
    
    async def stream_general_chat(
        self,
        content: str,
        session_id: str,
        user_id: int,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        target_language: str = None,
        enable_thinking: bool = True,
        heartbeat_sec: int = DEFAULT_HEARTBEAT_SEC,
        question_id: int = None,
        store_to_memory: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream general chat response."""
        result = StreamResult()
        llm_generator: Optional[AsyncGenerator] = None
        
        try:
            yield self._sse_retry(1000)
            yield self._sse_status("Initializing...")
            
            yield self._sse_status("Loading context...")
            chat_history = await self._get_enhanced_context(
                session_id=session_id,
                user_id=user_id,
                question_input=content
            )
            
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            detection_method = (
                DetectionMethod.LLM if len(content.split()) < 2 
                else DetectionMethod.LIBRARY
            )
            language_info = await language_detector.detect(
                text=content,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )
            detected_language = language_info["detected_language"]
            
            system_message = self._build_general_chat_system_message(
                enable_thinking, model_name, detected_language
            )
            user_content = self._build_general_chat_user_content(content, chat_history)
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            yield self._sse_status("Generating response...")
            
            if provider_type == ProviderType.OLLAMA:
                async for chunk in self._stream_ollama(model_name, messages):
                    result.append_chunk(chunk)
                    yield f"{json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
                    # yield f"data: {json.dumps({'type': 'content', 'content': chunk}, ensure_ascii=False)}\n\n"
            else:
                llm_generator = self.llm_provider.stream_response(
                    model_name=model_name,
                    messages=messages,
                    provider_type=provider_type,
                    api_key=api_key,
                    clean_thinking=True,
                    enable_thinking=enable_thinking
                )
                
                async for sse_chunk in self._stream_with_heartbeat_core(
                    llm_generator, result, heartbeat_sec
                ):
                    yield sse_chunk
            
            if store_to_memory and session_id and user_id and result.accumulated_response:
                await self._store_conversation(
                    session_id=session_id,
                    user_id=user_id,
                    query=content,
                    response=result.accumulated_response,
                    question_id=question_id,
                    model_name=model_name,
                    provider_type=provider_type,
                    response_time=result.response_time,
                    metadata={"type": "general_chat_stream"}
                )
            
            yield self._sse_done()
            
        except asyncio.CancelledError:
            self.logger.info(f"General chat cancelled: session={session_id}")
            raise
        except Exception as e:
            self.logger.error(f"General chat error: {e}")
            yield self._sse_error(str(e))
            yield self._sse_done()
        finally:
            await self._close_generator_safe(llm_generator)
    
    # =========================================================================
    # PUBLIC: Market Overview Streaming
    # =========================================================================
    
    async def stream_market_overview(
        self,
        market_data: List[Dict[str, Any]],
        question_input: str,
        session_id: str,
        user_id: int,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        target_language: str = None,
        heartbeat_sec: int = DEFAULT_HEARTBEAT_SEC,
        question_id: int = None,
        store_to_memory: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream market overview analysis."""
        result = StreamResult()
        llm_generator: Optional[AsyncGenerator] = None
        
        try:
            yield self._sse_retry(1000)
            yield self._sse_status("Analyzing market data...")
            
            chat_history = await self._get_enhanced_context(
                session_id=session_id,
                user_id=user_id,
                question_input=question_input
            )
            
            api_key = ModelProviderFactory._get_api_key(provider_type)
            symbols_str = self._format_market_data(market_data)
            
            detection_method = (
                DetectionMethod.LLM if len(question_input.split()) < 2 
                else DetectionMethod.LIBRARY
            )
            language_info = await language_detector.detect(
                text=question_input,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )
            detected_language = language_info["detected_language"]
            
            system_message = self._build_market_overview_system_message(detected_language)
            user_content = self._build_market_overview_user_content(
                question_input, symbols_str, chat_history
            )
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            yield self._sse_status("Generating analysis...")
            
            if provider_type == ProviderType.OLLAMA:
                async for chunk in self._stream_ollama(model_name, messages):
                    result.append_chunk(chunk)
                    yield f"{json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            else:
                llm_generator = self.llm_provider.stream_response(
                    model_name=model_name,
                    messages=messages,
                    provider_type=provider_type,
                    api_key=api_key,
                    clean_thinking=True
                )
                
                async for sse_chunk in self._stream_with_heartbeat_core(
                    llm_generator, result, heartbeat_sec
                ):
                    yield sse_chunk
            
            if store_to_memory and session_id and user_id and result.accumulated_response:
                await self._store_conversation(
                    session_id=session_id,
                    user_id=user_id,
                    query=question_input,
                    response=result.accumulated_response,
                    question_id=question_id,
                    model_name=model_name,
                    provider_type=provider_type,
                    response_time=result.response_time,
                    metadata={"type": "market_overview_stream"}
                )
            
            yield self._sse_done()
            
        except asyncio.CancelledError:
            self.logger.info(f"Market overview cancelled: session={session_id}")
            raise
        except Exception as e:
            self.logger.error(f"Market overview error: {e}")
            yield self._sse_error(str(e))
            yield self._sse_done()
        finally:
            await self._close_generator_safe(llm_generator)
    
    # =========================================================================
    # PUBLIC: Stock Analysis Streaming
    # =========================================================================
    
    async def stream_stock_analysis(
        self,
        analysis_data: Dict[str, Any],
        user_query: str,
        session_id: str,
        user_id: int,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        target_language: str = None,
        heartbeat_sec: int = DEFAULT_HEARTBEAT_SEC,
        question_id: int = None,
        store_to_memory: bool = True,
        include_news: bool = True,
        news_lookback_days: int = 7
    ) -> AsyncGenerator[str, None]:
        """Stream comprehensive stock analysis with news integration."""
        result = StreamResult()
        llm_generator: Optional[AsyncGenerator] = None
        
        try:
            symbol = analysis_data.get('symbol', 'stock')
            yield self._sse_retry(1000)
            yield self._sse_status(f"Analyzing {symbol}...")
            
            chat_history = await self._get_enhanced_context(
                session_id=session_id,
                user_id=user_id,
                question_input=user_query
            )
            
            # Fetch news
            news_context = ""
            if include_news and symbol:
                yield self._sse_status(f"Fetching news for {symbol}...")
                news_context = await self._get_news_context(
                    symbol=symbol,
                    limit=5,
                    lookback_days=news_lookback_days
                )
            
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            extracted_data = self.analysis_handler.extract_summaries(analysis_data)
            summaries = extracted_data["summaries"]
            key_metrics = extracted_data["key_metrics"]
            
            detection_method = (
                DetectionMethod.LLM if len(user_query.split()) < 2 
                else DetectionMethod.LIBRARY
            )
            language_info = await language_detector.detect(
                text=user_query,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )
            detected_language = language_info["detected_language"]
            
            system_message = self._build_stock_analysis_system_message(detected_language)
            user_content = self._build_stock_analysis_user_content(
                analysis_data, summaries, key_metrics, user_query, chat_history, news_context
            )
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            yield self._sse_status("Generating investment analysis...")
            
            if provider_type == ProviderType.OLLAMA:
                async for chunk in self._stream_ollama(model_name, messages):
                    result.append_chunk(chunk)
                    yield f"{json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            else:
                llm_generator = self.llm_provider.stream_response(
                    model_name=model_name,
                    messages=messages,
                    provider_type=provider_type,
                    api_key=api_key,
                    clean_thinking=True
                )
                
                async for sse_chunk in self._stream_with_heartbeat_core(
                    llm_generator, result, heartbeat_sec
                ):
                    yield sse_chunk
            
            if store_to_memory and session_id and user_id and result.accumulated_response:
                await self._store_conversation(
                    session_id=session_id,
                    user_id=user_id,
                    query=user_query,
                    response=result.accumulated_response,
                    question_id=question_id,
                    model_name=model_name,
                    provider_type=provider_type,
                    response_time=result.response_time,
                    metadata={
                        "type": "stock_analysis_stream",
                        "symbol": symbol
                    }
                )
            
            yield self._sse_done()
            
        except asyncio.CancelledError:
            self.logger.info(f"Stock analysis cancelled: session={session_id}")
            raise
        except Exception as e:
            self.logger.error(f"Stock analysis error: {e}")
            yield self._sse_error(str(e))
            yield self._sse_done()
        finally:
            await self._close_generator_safe(llm_generator)
    
    # =========================================================================
    # PUBLIC: Trending Analysis Streaming
    # =========================================================================
    
    async def stream_trending_analysis(
        self,
        gainers: List[Dict],
        losers: List[Dict],
        actives: List[Dict],
        question_input: str,
        session_id: str,
        user_id: int,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        target_language: str = None,
        heartbeat_sec: int = DEFAULT_HEARTBEAT_SEC,
        question_id: int = None,
        store_to_memory: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream trending stocks analysis."""
        result = StreamResult()
        llm_generator: Optional[AsyncGenerator] = None
        
        try:
            yield self._sse_retry(1000)
            yield self._sse_status("Analyzing trending stocks...")
            
            chat_history = await self._get_enhanced_context(
                session_id=session_id,
                user_id=user_id,
                question_input=question_input
            )
            
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            gainers_str = self._format_movers_data(gainers, "gainers")
            losers_str = self._format_movers_data(losers, "losers")
            actives_str = self._format_movers_data(actives, "actives")
            
            detection_method = (
                DetectionMethod.LLM if len(question_input.split()) < 2 
                else DetectionMethod.LIBRARY
            )
            language_info = await language_detector.detect(
                text=question_input,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )
            detected_language = language_info["detected_language"]
            
            system_message = self._build_trending_system_message(detected_language)
            user_content = self._build_trending_user_content(
                question_input, gainers_str, losers_str, actives_str, chat_history
            )
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            yield self._sse_status("Generating market insights...")
            
            if provider_type == ProviderType.OLLAMA:
                async for chunk in self._stream_ollama(model_name, messages):
                    result.append_chunk(chunk)
                    yield f"{json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            else:
                llm_generator = self.llm_provider.stream_response(
                    model_name=model_name,
                    messages=messages,
                    provider_type=provider_type,
                    api_key=api_key,
                    clean_thinking=True
                )
                
                async for sse_chunk in self._stream_with_heartbeat_core(
                    llm_generator, result, heartbeat_sec
                ):
                    yield sse_chunk
            
            if store_to_memory and session_id and user_id and result.accumulated_response:
                all_symbols = []
                for item in (gainers or []) + (losers or []) + (actives or []):
                    if isinstance(item, dict) and 'symbol' in item:
                        all_symbols.append(item['symbol'])
                
                await self._store_conversation(
                    session_id=session_id,
                    user_id=user_id,
                    query=question_input,
                    response=result.accumulated_response,
                    question_id=question_id,
                    model_name=model_name,
                    provider_type=provider_type,
                    response_time=result.response_time,
                    metadata={
                        "type": "trending_analysis_stream",
                        "symbols": list(set(all_symbols))[:20]
                    }
                )
            
            yield self._sse_done()
            
        except asyncio.CancelledError:
            self.logger.info(f"Trending analysis cancelled: session={session_id}")
            raise
        except Exception as e:
            self.logger.error(f"Trending analysis error: {e}")
            yield self._sse_error(str(e))
            yield self._sse_done()
        finally:
            await self._close_generator_safe(llm_generator)
    
    # =========================================================================
    # PUBLIC: Heatmap Analysis Streaming
    # =========================================================================
    
    async def stream_heatmap_analysis(
        self,
        heatmap_data: List[Dict[str, Any]],
        question_input: str,
        session_id: str,
        user_id: int,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        target_language: str = None,
        heartbeat_sec: int = DEFAULT_HEARTBEAT_SEC,
        question_id: int = None,
        store_to_memory: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream S&P 500 heatmap analysis."""
        result = StreamResult()
        llm_generator: Optional[AsyncGenerator] = None
        
        try:
            yield self._sse_retry(1000)
            yield self._sse_status("Processing heatmap data...")
            
            chat_history = await self._get_enhanced_context(
                session_id=session_id,
                user_id=user_id,
                question_input=question_input
            )
            
            api_key = ModelProviderFactory._get_api_key(provider_type)
            formatted_data = self._format_heatmap_data(heatmap_data)
            
            detection_method = (
                DetectionMethod.LLM if len(question_input.split()) < 2 
                else DetectionMethod.LIBRARY
            )
            language_info = await language_detector.detect(
                text=question_input,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )
            detected_language = language_info["detected_language"]
            
            system_message = self._build_heatmap_system_message(detected_language)
            user_content = self._build_heatmap_user_content(
                question_input, formatted_data, chat_history
            )
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
            
            yield self._sse_status("Generating heatmap analysis...")
            
            if provider_type == ProviderType.OLLAMA:
                async for chunk in self._stream_ollama(model_name, messages):
                    result.append_chunk(chunk)
                    yield f"{json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"
            else:
                llm_generator = self.llm_provider.stream_response(
                    model_name=model_name,
                    messages=messages,
                    provider_type=provider_type,
                    api_key=api_key,
                    clean_thinking=True,
                    enable_thinking=False
                )
                
                async for sse_chunk in self._stream_with_heartbeat_core(
                    llm_generator, result, heartbeat_sec
                ):
                    yield sse_chunk
            
            if store_to_memory and session_id and user_id and result.accumulated_response:
                await self._store_conversation(
                    session_id=session_id,
                    user_id=user_id,
                    query=question_input,
                    response=result.accumulated_response,
                    question_id=question_id,
                    model_name=model_name,
                    provider_type=provider_type,
                    response_time=result.response_time,
                    metadata={"type": "heatmap_analysis_stream"}
                )
            
            yield self._sse_done()
            
        except asyncio.CancelledError:
            self.logger.info(f"Heatmap analysis cancelled: session={session_id}")
            raise
        except Exception as e:
            self.logger.error(f"Heatmap analysis error: {e}")
            yield self._sse_error(str(e))
            yield self._sse_done()
        finally:
            await self._close_generator_safe(llm_generator)
    
    # =========================================================================
    # Ollama Streaming Helper
    # =========================================================================
    
    async def _stream_ollama(
        self,
        model_name: str,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream from Ollama API."""
        base_url = os.getenv('OLLAMA_HOST')
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": True
                }
            )
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith('data: '):
                    line = line[6:]
                if line == '[DONE]':
                    break
                try:
                    chunk = json.loads(line)
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue
    
    # =========================================================================
    # Memory Storage
    # =========================================================================
    
    async def _store_conversation(
        self,
        session_id: str,
        user_id: int,
        query: str,
        response: str,
        question_id: int,
        model_name: str,
        provider_type: str,
        response_time: float,
        metadata: Dict[str, Any] = None,
        skip_importance_analysis: bool = True  # Skip by default to avoid blocking
    ) -> None:
        """Store conversation to memory and chat history."""
        try:
            importance_score = 0.5

            # Skip importance analysis by default to avoid extra LLM call that blocks
            # Can be enabled for important conversations
            if not skip_importance_analysis:
                try:
                    analysis_model = "gpt-4.1-nano" if provider_type == ProviderType.OPENAI else model_name
                    # Add timeout to prevent blocking
                    importance_score = await asyncio.wait_for(
                        analyze_conversation_importance(
                            query=query,
                            response=response,
                            llm_provider=self.llm_provider,
                            model_name=analysis_model,
                            provider_type=provider_type
                        ),
                        timeout=5.0  # 5 second timeout
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Importance analysis timed out, using default score")
                except Exception as e:
                    self.logger.warning(f"Importance analysis failed: {e}")
            
            store_metadata = metadata or {}
            store_metadata.update({
                "response_time": response_time,
                "importance_score": importance_score,
                "model": model_name,
                "provider": provider_type,
                "timestamp": datetime.now().isoformat()
            })
            
            await self.memory_manager.store_conversation_turn(
                session_id=session_id,
                user_id=user_id,
                query=query,
                response=response,
                metadata=store_metadata,
                importance_score=importance_score
            )
            
            if question_id:
                self.chat_service.save_assistant_response(
                    session_id=session_id,
                    created_at=datetime.now(),
                    question_id=question_id,
                    content=response,
                    response_time=response_time
                )
            
            try:
                from src.services.background_tasks import trigger_summary_update_nowait
                trigger_summary_update_nowait(session_id=session_id, user_id=user_id)
            except ImportError:
                pass
            
            self.logger.debug(f"Stored: session={session_id}, importance={importance_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error storing conversation: {e}")
    

    # =========================================================================
    # Data Formatters
    # =========================================================================
    def _format_market_data(self, market_data: List[Dict]) -> str:
        """Format market data"""
        if not market_data:
            return "No market data available."
        
        symbol_analyses = []
        for item in market_data:
            if hasattr(item, 'dict'):
                item = item.dict()
            elif not isinstance(item, dict):
                continue
            
            symbol = item.get("symbol", "Unknown")
            name = item.get("name", "Unknown")
            currency = item.get("currency", "USD")
            
            last_price = item.get("last_price")
            change = item.get("change")
            change_percent = item.get("change_percent")
            
            volume = item.get("volume")
            volume_avg = item.get("volume_average")
            volume_avg_10d = item.get("volume_average_10d")
            
            ma_50d = item.get("ma_50d")
            ma_200d = item.get("ma_200d")
            
            open_price = item.get("open")
            high = item.get("high")
            low = item.get("low")
            prev_close = item.get("prev_close")
            year_high = item.get("year_high")
            year_low = item.get("year_low")
            
            symbol_analysis = f"""**{symbol} ({name})**:
- Last Price: {last_price} {currency}
- Change: {change} ({change_percent}%)
- Volume: {volume}
- 50-Day Moving Average: {ma_50d}
- 200-Day Moving Average: {ma_200d}
- Open: {open_price}
- High: {high}
- Low: {low}
- Previous Close: {prev_close}
- 52-Week High: {year_high}
- 52-Week Low: {year_low}
- Average Volume: {volume_avg}
- 10-Day Average Volume: {volume_avg_10d}
"""
            symbol_analyses.append(symbol_analysis)
        
        return "\n".join(symbol_analyses) if symbol_analyses else "No symbol data available."
    
    def _format_movers_data(self, movers: List[Dict], category: str) -> str:
        """Format gainers/losers/actives data."""
        if not movers:
            return f"No {category} data available."
        
        lines = []
        for item in movers:
            symbol = item.get('symbol', 'Unknown')
            name = item.get('name', 'Unknown')
            price = item.get('price', 'N/A')
            change = item.get('change', 'N/A')
            pct = item.get('percent_change', 'N/A')
            volume = item.get('volume', 'N/A')
            
            lines.append(
                f"**{symbol}** - {name}: ${price}, "
                f"Change: {change} ({pct}%), Volume: {volume}"
            )
        
        return "\n".join(lines)
    
    def _format_movers_data(self, movers: List[Dict], category: str) -> str:
        """Format gainers/losers/actives data."""
        if not movers:
            return f"No {category} data available."
        
        lines = []
        for item in movers:
            symbol = item.get('symbol', 'Unknown')
            name = item.get('name', 'Unknown')
            price = item.get('price', 'N/A')
            change = item.get('change', 'N/A')
            pct = item.get('percent_change', 'N/A')
            volume = item.get('volume', 'N/A')
            
            lines.append(
                f"**{symbol}** - {name}:\n"
                f"Price: {price}, Change: {change} ({pct}%), Volume: {volume}"
            )
        
        return "\n".join(lines)

    def _format_heatmap_data(self, heatmap_data: List[Dict[str, Any]]) -> str:
        """Format heatmap data for LLM analysis."""
        if not heatmap_data:
            return "No heatmap data available"
        
        # Group by sector
        sectors_data: Dict[str, List[Dict]] = {}
        for item in heatmap_data:
            sector = item.get('sector', 'Unknown')
            if sector not in sectors_data:
                sectors_data[sector] = []
            sectors_data[sector].append(item)
        
        # Build formatted output
        formatted = "📊 S&P 500 HEATMAP ANALYSIS DATA\n\n"
        
        # Overall statistics
        total_stocks = len(heatmap_data)
        gainers = [s for s in heatmap_data if s.get('changesPercentage', 0) > 0]
        strong_gainers = [s for s in heatmap_data if s.get('changesPercentage', 0) > 3]
        losers = [s for s in heatmap_data if s.get('changesPercentage', 0) < 0]
        strong_losers = [s for s in heatmap_data if s.get('changesPercentage', 0) < -3]
        
        # Market breadth analysis
        advancement_ratio = len(gainers) / total_stocks if total_stocks > 0 else 0
        market_sentiment = "🟢 Risk-On" if advancement_ratio > 0.6 else "🔴 Risk-Off" if advancement_ratio < 0.4 else "🟡 Mixed"
        
        formatted += f"📈 **MARKET OVERVIEW** ({market_sentiment}): {total_stocks} stocks analyzed\n"
        formatted += f"✅ Advancing: {len(gainers)} ({advancement_ratio*100:.1f}%) | Strong: {len(strong_gainers)}\n"
        formatted += f"❌ Declining: {len(losers)} ({(1-advancement_ratio)*100:.1f}%) | Weak: {len(strong_losers)}\n"
        formatted += f"📊 Market Breadth Ratio: {len(gainers)}/{len(losers)} = {'Bullish' if advancement_ratio > 0.55 else 'Bearish' if advancement_ratio < 0.45 else 'Neutral'}\n\n"
        
        # Enhanced top movers
        top_gainers = sorted(heatmap_data, key=lambda x: x.get('changesPercentage', 0), reverse=True)[:8]
        top_losers = sorted(heatmap_data, key=lambda x: x.get('changesPercentage', 0))[:8]
        
        formatted += "🚀 **TOP GAINERS** (Momentum Leaders):\n"
        for i, stock in enumerate(top_gainers, 1):
            volume_indicator = "🔥" if stock.get('volume', 0) > stock.get('avgVolume', 1) * 2 else "📈"
            formatted += f"  {i}. {stock['symbol']} ({stock.get('name', 'N/A')}): {stock['changesPercentage']:.2f}% | ${stock.get('price', 0):.2f} {volume_indicator}\n"
        
        formatted += "\n📉 **TOP LOSERS** (Pressure Points):\n"
        for i, stock in enumerate(top_losers, 1):
            volume_indicator = "⚠️" if stock.get('volume', 0) > stock.get('avgVolume', 1) * 2 else "📉"
            formatted += f"  {i}. {stock['symbol']} ({stock.get('name', 'N/A')}): {stock['changesPercentage']:.2f}% | ${stock.get('price', 0):.2f} {volume_indicator}\n"
        
        # Enhanced sector analysis
        formatted += "\n📊 **SECTOR PERFORMANCE MATRIX**:\n"
        sector_performance = []
        
        for sector, stocks in sectors_data.items():
            avg_change = sum(s.get('changesPercentage', 0) for s in stocks) / len(stocks)
            total_market_cap = sum(s.get('marketCap', 0) for s in stocks)
            total_volume = sum(s.get('volume', 0) for s in stocks)
            sector_gainers = len([s for s in stocks if s.get('changesPercentage', 0) > 0])
            sector_breadth = sector_gainers / len(stocks) if stocks else 0
            
            if avg_change > 1.5:
                performance_status = "🟢 STRONG"
            elif avg_change > 0:
                performance_status = "🟡 POSITIVE"
            elif avg_change > -1.5:
                performance_status = "🟠 WEAK"
            else:
                performance_status = "🔴 DECLINING"
            
            sector_performance.append({
                'sector': sector,
                'avg_change': avg_change,
                'status': performance_status,
                'breadth': sector_breadth,
                'stocks': stocks,
                'market_cap': total_market_cap,
                'volume': total_volume
            })
        
        sector_performance.sort(key=lambda x: x['avg_change'], reverse=True)
        
        for sector_data in sector_performance:
            breadth_indicator = "💪" if sector_data['breadth'] > 0.7 else "⚖️" if sector_data['breadth'] > 0.4 else "🔻"
            
            formatted += f"\n{sector_data['status']} **{sector_data['sector']}** {breadth_indicator}\n"
            formatted += f"  - Performance: {sector_data['avg_change']:+.2f}% | Breadth: {sector_data['breadth']*100:.0f}% advancing\n"
            formatted += f"  - Stocks: {len(sector_data['stocks'])} | Market Cap: ${sector_data['market_cap']/1e9:.1f}B\n"
            formatted += f"  - Key Players: {', '.join([s['symbol'] for s in sorted(sector_data['stocks'], key=lambda x: abs(x.get('changesPercentage', 0)), reverse=True)[:4]])}\n"
        
        # Institutional activity signals
        high_volume_stocks = [s for s in heatmap_data if s.get('volume', 0) > s.get('avgVolume', 1) * 2]
        institutional_activity = [s for s in high_volume_stocks if s.get('marketCap', 0) > 10e9]
        
        formatted += "\n💼 **INSTITUTIONAL ACTIVITY SIGNALS**:\n"
        formatted += f"  - High Volume Stocks: {len(high_volume_stocks)} (>200% avg volume)\n"
        formatted += f"  - Large Cap Activity: {len(institutional_activity)} stocks showing institutional interest\n"
        
        # Market cap distribution analysis
        mega_caps = [s for s in heatmap_data if s.get('marketCap', 0) > 200e9]
        large_caps = [s for s in heatmap_data if 50e9 <= s.get('marketCap', 0) <= 200e9]
        mid_caps = [s for s in heatmap_data if 10e9 <= s.get('marketCap', 0) < 50e9]
        
        mega_performance = sum(s.get('changesPercentage', 0) for s in mega_caps) / len(mega_caps) if mega_caps else 0
        large_performance = sum(s.get('changesPercentage', 0) for s in large_caps) / len(large_caps) if large_caps else 0
        
        formatted += "\n💰 **MARKET CAP PERFORMANCE ANALYSIS**:\n"
        formatted += f"  - Mega Cap (>$200B): {len(mega_caps)} stocks | Avg: {mega_performance:+.2f}%\n"
        formatted += f"  - Large Cap ($50B-$200B): {len(large_caps)} stocks | Avg: {large_performance:+.2f}%\n"
        formatted += f"  - Mid Cap ($10B-$50B): {len(mid_caps)} stocks\n"
        formatted += f"  - Size Factor: {'Large Cap Leading' if mega_performance > large_performance else 'Small Cap Outperforming'}\n"
        
        return formatted


    # Message Builders
    def _get_language_instruction(self, detected_language: str) -> str:
        """Get language instruction for system prompt."""
        lang_map = {
            "en": "English",
            "vi": "Vietnamese",
            "zh": "Chinese",
            "zh-cn": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
        }
        lang_name = lang_map.get(detected_language, "the detected language")
        
        return f"""CRITICAL LANGUAGE REQUIREMENT:
You MUST respond ENTIRELY in {lang_name}.
- ALL text, explanations, and analysis must be in {lang_name}
- Use appropriate financial terminology for {lang_name}
- Format numbers and dates according to {lang_name} conventions"""
    
    def _build_general_chat_system_message(
        self, enable_thinking: bool, model_name: str, detected_language: str
    ) -> str:
        """Build system message for general chat."""
        language_instruction = self._get_language_instruction(detected_language)
        
        return f"""You are TopOneLogic Assistant, a professional financial assistant specializing in stocks and cryptocurrency analysis.

{language_instruction}

LANGUAGE ADAPTATION:
- Detect and respond in the user's language automatically
- Use culturally appropriate financial terms and formats
- Maintain professional tone across all languages

CRITICAL INSTRUCTIONS:
1. ALWAYS carefully read and use the conversation history provided below
2. When answering questions, FIRST check the conversation history for relevant information
3. If the user asks about something mentioned in history, you MUST recall and use that information
4. Use specific numbers, dates, and metrics from context
5. Never invent financial data or make unsupported predictions
6. Adapt currency symbols, date formats, and number formats to user's region

RESPONSE STRUCTURE:
1. Answer the question general helpfully
2. For financial topics use context data to support your analysis if relevant and provide clear, actionable financial insights:
   - Analyze price trends, volume, fundamentals & technical indicators (stocks).  
   - Evaluate market cap, trading volume, price action & volatility (crypto).  
   - Highlight risks, opportunities and broader market context. 

TONE & STYLE:
- Professional yet accessible in user's native language
- Data-driven and objective across all languages
- Include relevant disclaimers using appropriate legal language
- Structure information clearly with local formatting preferences
- Use familiar financial terminology for the user's market/region

Remember: Think step-by-step detect language → analyze context → identify key insights → provide localized structured response."""

    def _build_general_chat_user_content(self, content: str, chat_history: str) -> str:
        """Build user content for general chat."""
        if chat_history and chat_history.strip():
            return f"""=== CURRENT QUESTION (Please respond in this language) ===
{content}

=== CONVERSATION HISTORY (For context only) ===
{chat_history}

Instructions: Answer the CURRENT QUESTION above in the SAME LANGUAGE it was asked, using relevant information from the history if needed."""
        return content
    
    def _build_market_overview_system_message(self, detected_language: str) -> str:
        """Build system message for market overview."""
        language_instruction = self._get_language_instruction(detected_language)
        
        return f"""You are TopOneLogic Assistant, a professional financial advisor specializing in market analysis and investment guidance.

{language_instruction}

CONTEXT MANAGEMENT RULES
**CRITICAL**: Before answering, analyze if the user's question relates to previous conversation:
1. **USE PREVIOUS CONTEXT when user:**
   - Uses pronouns referring to previous topics ("it", "this stock", "that company")
   - Asks follow-up questions ("what about...", "and how about...", "additionally...")
   - References previous analysis ("as you mentioned", "you said earlier")
   - Compares with previous discussion ("compared to the previous stock")
2. **IGNORE PREVIOUS CONTEXT when user:**
   - Asks about a completely NEW stock symbol
   - Starts a new topic unrelated to previous discussion
   - Asks a standalone question with complete information
   - Explicitly requests fresh analysis ("analyze AAPL" when previous was about TSLA)
3. **CONTEXT DECISION PROCESS:**
   - First: Identify if current question is self-contained
   - Second: Check if it references previous conversation
   - Third: Only use relevant parts of history, not everything
   - Fourth: If unsure, prioritize the current question's direct needs

MARKET ANALYSIS FRAMEWORK:
When analyzing market data (indices, prices, volumes, changes):

1. COMPARATIVE ANALYSIS
   - Compare global markets performance (US, EU, Asia)
   - Identify relative strength/weakness between regions
   - Highlight sector rotation patterns

2. PATTERN RECOGNITION  
   - Spot common trends across markets
   - Note significant divergences or correlations
   - Identify unusual percentage movements (>2%)

3. VOLATILITY ASSESSMENT
   - Analyze intraday ranges (high-low spreads)
   - Compare current volatility to historical norms
   - Assess market sentiment (risk-on vs risk-off)

4. ACTIONABLE INSIGHTS
   - Recommend specific sectors/regions for opportunities
   - Suggest portfolio adjustments based on trends
   - Provide risk management guidance

RESPONSE STRUCTURE:
- Lead with key market takeaway
- Support with specific data points
- End with 2-3 targeted follow-up questions about:
  • Specific indices/regions
  • Sector impact analysis  
  • Investment strategy implications
  • Historical context

TONE: Data-driven, concise, investor-focused. Use specific numbers and percentages to support analysis."""
    
    def _build_market_overview_user_content(
        self, question: str, symbols_str: str, chat_history: str
    ) -> str:
        """Build user content for market overview."""
        if chat_history and chat_history.strip():
            return f"""=== USER'S QUESTION (Respond in this language) ===
{question}

=== MARKET DATA TO ANALYZE ===
{symbols_str}

=== CONVERSATION HISTORY (For context) ===
{chat_history}

Please provide comprehensive analysis addressing the user's question."""
        else:
            return f"""=== USER'S QUESTION ===
{question}

=== MARKET DATA ===
{symbols_str}

Please provide comprehensive analysis."""
    
    def _build_trending_system_message(self, detected_language: str) -> str:
        """Build system message for trending analysis."""
        language_instruction = self._get_language_instruction(detected_language)

        return f"""You are TopOneLogic Assistant, a professional financial market analyst with expertise in stock market analysis.

{language_instruction}

CRITICAL INSTRUCTIONS:
1. RESPONSE REQUIREMENTS:
   - Use clear, simple language suitable for investors of all levels
   - Format numbers appropriately (e.g., $1,234.56 for English, 1.234,56 for Vietnamese)
   - Include emojis sparingly for better readability (📈 for gains, 📉 for losses, 💰 for opportunities)

2. ANALYSIS FOCUS:
   - Key metrics: price changes, volume patterns, volatility
   - Market sentiment and momentum indicators
   - Risk assessment and opportunity identification
   - Actionable insights for investment decisions

# CONTEXT MANAGEMENT RULES
**CRITICAL**: Before answering, analyze if the user's question relates to previous conversation:
1. **USE PREVIOUS CONTEXT when user:**
   - Uses pronouns referring to previous topics ("it", "this stock", "that company")
   - Asks follow-up questions ("what about...", "and how about...", "additionally...")
   - References previous analysis ("as you mentioned", "you said earlier")
   - Compares with previous discussion ("compared to the previous stock")
2. **IGNORE PREVIOUS CONTEXT when user:**
   - Asks about a completely NEW stock symbol
   - Starts a new topic unrelated to previous discussion
   - Asks a standalone question with complete information
   - Explicitly requests fresh analysis ("analyze AAPL" when previous was about TSLA)
3. **CONTEXT DECISION PROCESS:**
   - First: Identify if current question is self-contained
   - Second: Check if it references previous conversation
   - Third: Only use relevant parts of history, not everything
   - Fourth: If unsure, prioritize the current question's direct needs"""
    
    def _build_trending_user_content(
        self, question: str, gainers_str: str, losers_str: str, 
        actives_str: str, chat_history: str
    ) -> str:
        """Build user content for trending analysis."""
        if chat_history and chat_history.strip():
            return f"""=== USER'S QUESTION (Respond in this language) ===
{question}

=== TRENDING STOCKS DATA ===
Please analyze the following stock market data. Provide comprehensive insights about market trends, risks, and opportunities.

**📈 GAINERS:**
{gainers_str}

**📉 LOSERS:**
{losers_str}

**🔥 MOST ACTIVE:**
{actives_str}

Structure your response with these sections:
1. **Market Overview** - Key trends and market sentiment
2. **Top Gainers Analysis** - Why these stocks are rising, opportunities
3. **Top Losers Analysis** - Risk factors and warning signs
4. **Most Active Stocks** - Volume analysis and implications
5. **Investment Recommendations** - Actionable advice with risk levels

=== PREVIOUS CONTEXT ===
{chat_history}

Analyze the trending stocks and provide insights."""
        else:
            return f"""=== USER'S QUESTION ===
{question}

=== TRENDING STOCKS DATA ===

**📈 GAINERS:**
{gainers_str}

**📉 LOSERS:**
{losers_str}

**🔥 MOST ACTIVE:**
{actives_str}

Analyze and provide insights."""
    
    def _build_stock_analysis_system_message(self, detected_language: str) -> str:
        """Build system message for stock analysis."""
        language_instruction = self._get_language_instruction(detected_language)
        
        return f"""You are TopOneLogic Assistant, an expert financial analyst using ReAct (Reasoning and Acting) with Chain of Thought methodology.

{language_instruction}

# CONTEXT MANAGEMENT RULES
**CRITICAL**: Before answering, analyze if the user's question relates to previous conversation:
1. **USE PREVIOUS CONTEXT when user:**
   - Uses pronouns referring to previous topics ("it", "this stock", "that company")
   - Asks follow-up questions ("what about...", "and how about...", "additionally...")
   - References previous analysis ("as you mentioned", "you said earlier")
   - Compares with previous discussion ("compared to the previous stock")
2. **IGNORE PREVIOUS CONTEXT when user:**
   - Asks about a completely NEW stock symbol
   - Starts a new topic unrelated to previous discussion
   - Asks a standalone question with complete information
   - Explicitly requests fresh analysis ("analyze AAPL" when previous was about TSLA)
3. **CONTEXT DECISION PROCESS:**
   - First: Identify if current question is self-contained
   - Second: Check if it references previous conversation
   - Third: Only use relevant parts of history, not everything
   - Fourth: If unsure, prioritize the current question's direct needs

# DATA TRANSPARENCY RULES
**MANDATORY**: When presenting data, you MUST:
1. If a date/timestamp is provided with any metric, ALWAYS include it
2. Format: "[Metric]: [Value] (as of [Date])" or "[Metric] ([Date]): [Value]"
3. For news items: Always show the publication date
4. If NO date is provided in the data, state "current data" or "latest available"
5. Never invent or assume dates that aren't in the source data

Examples:
- "Current price: $150.25 (as of 2024-03-15)" ✓
- "RSI: 65 (calculated on 2024-03-14)" ✓  
- "Current price: $150.25" ✗ (missing date if available)

# ANALYSIS FRAMEWORK
## Step 1: Reasoning Process - INTERNAL ANALYSIS (do NOT reveal to user)
Analyze systematically:
1. TREND ASSESSMENT:
   - Price vs Moving Averages (20/50/200 SMA)
   - Trend strength and direction
   
2. MOMENTUM EVALUATION:
   - RSI position (oversold <30, neutral 30-70, overbought >70)
   - MACD signal (bullish/bearish crossover)
   
3. RISK ANALYSIS:
   - Support/Resistance levels
   - Stop loss positioning (ATR vs percentage)
   - Risk/Reward ratio calculation
   
4. VOLUME & PATTERNS:
   - Volume profile (accumulation/distribution)
   - Chart patterns (reliability assessment)
   
5. MARKET STRENGTH:
   - Relative strength vs benchmark
   - Sector performance context
   
6. NEWS IMPACT:
   - Fundamental catalysts
   - Sentiment shifts
   - Time sensitivity of news

## Step 2: Decision Matrix
Based on analysis, calculate signal strength:
- Count bullish vs bearish indicators
- Weight by reliability (Price action > Indicators > Patterns)
- Factor in news sentiment

## Step 3: Investment Recommendation
### For STRONG BUY (>70% bullish signals):
**🟢 STRONG BUY RECOMMENDATION**
- Entry: [specific price or condition]
- Target 1: [based on resistance]
- Target 2: [based on pattern]
- Stop Loss: [based on ATR or support]
- Position Size: [% of portfolio based on risk]
- Reasoning: [3-4 key factors]

### For MODERATE BUY (50-70% bullish):
**🟡 CONDITIONAL BUY**
- Wait for: [specific trigger]
- Entry zones: [price ranges]
- Risk factors to monitor: [list]
- Alternative strategy: [DCA approach]

### For NEUTRAL/WAIT (40-60% mixed):
**⚪ HOLD/WAIT**
- Current situation: [analysis]
- What to watch: [key levels/indicators]
- Decision triggers: [specific conditions]

### For SELL/AVOID (<40% bullish):
**🔴 AVOID/SELL RECOMMENDATION**
- Key concerns: [major risks]
- If holding: [exit strategy]
- Better alternatives: [suggestions]

# DECISION CRITERIA
Your recommendation MUST be based on:
1. **Technical Weight (60%)**:
   - Price action relative to MAs
   - Momentum indicators alignment
   - Volume confirmation
   
2. **Risk Assessment (25%)**:
   - Clear stop loss levels
   - Risk/Reward ratio > 1.5:1
   - Position sizing rules
   
3. **Market Context (15%)**:
   - Relative strength
   - News/Fundamental factors
   - Overall market conditions

## Summary
[1-2 sentences with clear action plan]

# OUTPUT REQUIREMENTS
1. Base recommendations on actual data values
2. Provide ONE clear recommendation (BUY/CONDITIONAL BUY/WAIT/AVOID)
3. Include SPECIFIC entry, stop, and target prices
4. Explain your reasoning with exact indicator values
5. Address the user's specific question if provided
6. Use simple language but be technically accurate
7. Be decisive - no wishy-washy recommendations
8. **DATE TRANSPARENCY**: Always include dates/timestamps when they exist in the source data

Remember: Investors need actionable advice with clear reasoning. Your analysis should give them confidence in their decision."""
    
    def _build_stock_analysis_user_content(
        self,
        analysis_data: Dict,
        summaries: Dict,
        key_metrics: Dict,
        user_query: str,
        chat_history: str,
        news_context: str = ""
    ) -> str:
        """Build FULL user content for stock analysis (matching original)."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        symbol = analysis_data.get("symbol", "Unknown")
        
        data_analysis = f"""
Analyze the stock with symbol {symbol}. Use the provided STRUCTURED data below instead of trying to parse the entire JSON.

TECHNICAL ANALYSIS SUMMARY:
{summaries.get("technical_analysis", "N/A")}

RISK ANALYSIS SUMMARY:
{summaries.get("risk_analysis", "N/A")}

VOLUME PROFILE SUMMARY:
{summaries.get("volume_profile", "N/A")}

PATTERN RECOGNITION SUMMARY:
{summaries.get("pattern_recognition", "N/A")}

RELATIVE STRENGTH SUMMARY:
{summaries.get("relative_strength", "N/A")}

KEY METRICS:
- Current price: ${key_metrics.get("price", "N/A")}
- RSI: {key_metrics.get("rsi", "N/A")}
- MACD bullish: {key_metrics.get("macd_bullish", "N/A")}
- SMA 20: ${key_metrics.get("moving_averages", {}).get("sma_20", "N/A")}
- SMA 50: ${key_metrics.get("moving_averages", {}).get("sma_50", "N/A")}
- Stop levels (ATR 2x): ${key_metrics.get("stop_levels", {}).get("atr_2x", "N/A")}
- Stop levels (5%): ${key_metrics.get("stop_levels", {}).get("percent_5", "N/A")}
- Recent swing low: ${key_metrics.get("stop_levels", {}).get("recent_swing", "N/A")}

REMINDER: When presenting these metrics, include the date ({current_date}) to show data freshness.
"""
        
        if user_query:
            if chat_history and chat_history.strip():
                final_context = f"""=== USER'S QUESTION (Respond in this language) ===
{user_query}

=== COMPREHENSIVE ANALYSIS DATA ===
{data_analysis}

=== CONVERSATION HISTORY (For reference) ===
{chat_history}

=== CRITICAL DATE TRANSPARENCY REMINDER ===
You MUST include dates/timestamps whenever they appear in the data above.
- For prices and indicators: State the date they were calculated/retrieved
- For news: Always mention publication dates
- This ensures users understand data freshness and can make informed decisions

Please provide investment analysis addressing the user's question."""
            else:
                final_context = f"""=== USER'S QUESTION (Respond in this language) ===
{user_query}

=== COMPREHENSIVE ANALYSIS DATA ===
{data_analysis}

=== CRITICAL DATE TRANSPARENCY REMINDER ===
You MUST include dates/timestamps whenever they appear in the data above.
- For prices and indicators: State the date they were calculated/retrieved  
- For news: Always mention publication dates
- This ensures users understand data freshness and can make informed decisions

Please provide investment analysis."""
        else:
            final_context = f"""=== ANALYSIS REQUEST ===
Please analyze this stock data comprehensively.

=== DATA ===
{data_analysis}

=== CRITICAL DATE TRANSPARENCY REMINDER ===
You MUST include dates/timestamps whenever they appear in the data above.
- For prices and indicators: State the date they were calculated/retrieved
- For news: Always mention publication dates  
- This ensures users understand data freshness and can make informed decisions
"""
        
        if news_context:
            final_context += f"\n\n{news_context}"
        
        return final_context
    
    def _build_heatmap_system_message(self, detected_language: str) -> str:
        """Build system message for heatmap analysis."""
        language_instruction = self._get_language_instruction(detected_language)
        
        return f"""You are TopOneLogic Assistant, an expert in the stock market analyst specializing in heatmap visualization and market trends analysis multi-language.

{language_instruction}

=== CONTEXT MANAGEMENT RULES === 
**CRITICAL**: Before answering, analyze if the user's question relates to previous conversation:
1. **USE PREVIOUS CONTEXT when user:**
   - Uses pronouns referring to previous topics ("it", "this stock", "that company")
   - Asks follow-up questions ("what about...", "and how about...", "additionally...")
   - References previous analysis ("as you mentioned", "you said earlier")
   - Compares with previous discussion ("compared to the previous stock")
2. **IGNORE PREVIOUS CONTEXT when user:**
   - Asks about a completely NEW stock symbol
   - Starts a new topic unrelated to previous discussion
   - Asks a standalone question with complete information
   - Explicitly requests fresh analysis ("analyze AAPL" when previous was about TSLA)
3. **CONTEXT DECISION PROCESS:**
   - First: Identify if current question is self-contained
   - Second: Check if it references previous conversation
   - Third: Only use relevant parts of history, not everything
   - Fourth: If unsure, prioritize the current question's direct needs

=== CRITICAL INSTRUCTIONS ===:
When provided with previous tool responses or conversation history, you should:
1. Reference and build upon previous heatmap analyses when relevant
2. Highlight changes in sector performance if previously analyzed
3. Update any previous market assessments based on new data
    
Your task is to analyze S&P 500 heatmap data and provide key insights:
1. Identify sectors/industries performing well or poorly
2. Highlight notable stocks (top gainers/losers)
3. Comment on overall market trends and sentiment
4. Provide insights on market cap distribution, P/E ratios, and volume patterns
5. Alert on potential risks or investment opportunities
6. Explain what the heatmap reveals about market dynamics

Use clear, concise language and focus on actionable insights."""
    
    def _build_heatmap_user_content(
        self, question: str, formatted_data: str, chat_history: str
    ) -> str:
        """Build user content for heatmap analysis."""
        if chat_history and chat_history.strip():
            return f"""=== USER'S QUESTION (Respond in this language) ===
{question}

=== S&P 500 HEATMAP DATA ===
{formatted_data}

=== CONVERSATION HISTORY ===
{chat_history}

Analyze the heatmap and address the user's question."""
        else:
            return f"""=== USER'S QUESTION (Respond in this language) ===
{question}

=== S&P 500 HEATMAP DATA ===
{formatted_data}

Analyze and provide insights."""


# =============================================================================
# Singleton Instance
# =============================================================================
llm_streaming_service = LLMStreamingService()