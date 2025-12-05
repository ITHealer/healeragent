import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from src.helpers.redis_cache import get_redis_client
from src.helpers.chat_management_helper import ChatService
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory

logger = logging.getLogger(__name__)

class SessionSummaryService:
    """Service to manage conversation summaries for better context"""
    
    CACHE_PREFIX = "session_summary:"
    CACHE_TTL = 3600  # 1 hour default TTL
    SUMMARY_WINDOW_SIZE = 5  # Number of messages to summarize
    
    def __init__(self, chat_service: ChatService, llm_service: LLMGeneratorProvider):
        self.chat_service = chat_service
        self.llm_service = llm_service
        self.redis_client = None
        self._background_tasks = set()
        
    async def _get_redis(self):
        """Get redis client lazily"""
        if not self.redis_client:
            # Fix: get_redis_client là async generator, cần iterate để lấy client
            try:
                async for client in get_redis_client():
                    self.redis_client = client
                    break
            except Exception as e:
                logger.warning(f"Could not get Redis client: {e}")
                self.redis_client = None
        return self.redis_client
    
    async def get_or_create_summary(
        self, 
        session_id: str,
        force_refresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Get cached summary or create new one"""
        
        if not session_id:
            return None
            
        try:
            redis = await self._get_redis()
            cache_key = f"{self.CACHE_PREFIX}{session_id}"
            
            # Check cache first if not forcing refresh
            if not force_refresh and redis:
                try:
                    cached = await redis.get(cache_key)
                    if cached:
                        logger.info(f"Using cached summary for session {session_id}")
                        return json.loads(cached)
                except Exception as cache_error:
                    logger.warning(f"Cache read error: {cache_error}")
            
            # Generate new summary
            summary = await self._generate_summary(session_id)
            
            # Cache the summary if redis available
            if summary and redis:
                try:
                    await redis.setex(
                        cache_key,
                        self.CACHE_TTL,
                        json.dumps(summary)
                    )
                    logger.info(f"Cached new summary for session {session_id}")
                except Exception as cache_error:
                    logger.warning(f"Cache write error: {cache_error}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting/creating summary: {e}")
            # Return empty summary on error
            return {
                "summary": "",
                "key_points": [],
                "topics": [],
                "unresolved": []
            }
    
    async def _generate_summary(self, session_id: str) -> Dict[str, Any]:
        """Generate summary from recent messages"""
        
        try:
            # Get recent messages (excluding the latest one)
            history = self.chat_service.get_chat_history(
                session_id=session_id,
                limit=self.SUMMARY_WINDOW_SIZE + 1  # Get 6 to skip latest
            )
            
            if not history or len(history) < 2:
                return {
                    "summary": "",
                    "key_points": [],
                    "topics": [],
                    "unresolved": []
                }
            
            # Skip the latest message and use previous 5
            messages_to_summarize = history[1:self.SUMMARY_WINDOW_SIZE + 1]
            
            # Prepare conversation text
            conversation_text = ""
            for content, role in messages_to_summarize:
                conversation_text += f"{role}: {content[:500]}\n\n"
            
            # Generate summary using LLM
            summary_data = await self._call_llm_for_summary(conversation_text)
            
            return summary_data
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "summary": "",
                "key_points": [],
                "topics": [],
                "unresolved": []
            }
    
    async def _call_llm_for_summary(self, conversation_text: str) -> Dict[str, Any]:
        """Call LLM to generate structured summary"""
        
        try:
            # Get LLM instance
            api_key = ModelProviderFactory._get_api_key("openai")
            llm = await self.llm_service.get_llm(
                model_name="gpt-4.1-nano",
                provider_type="openai",
                api_key=api_key
            )
            
            messages = [
                {
                    "role": "system",
                    "content": """You are a conversation analyzer. Summarize the key points from a financial discussion.
Focus on extracting:
1. Main topics discussed
2. Key decisions or conclusions
3. Unresolved questions or points needing clarification
4. Important symbols, metrics, or data points mentioned"""
                },
                {
                    "role": "user",
                    "content": f"""Analyze this conversation and provide a structured summary:

{conversation_text}

Output format (JSON):
{{
    "summary": "Brief 2-3 sentence summary of the conversation",
    "key_points": ["point 1", "point 2", "point 3"],
    "topics": ["topic1", "topic2"],
    "unresolved": ["question or unclear point 1", "question 2"],
    "mentioned_symbols": ["AAPL", "TSLA"],
    "mentioned_metrics": ["P/E ratio", "RSI"]
}}"""
                }
            ]
            
            # Call LLM
            response = await llm.generate(messages)
            
            # Parse response
            response_content = ""
            if hasattr(response, 'content'):
                response_content = response.content
            elif isinstance(response, dict):
                response_content = response.get('content', '')
            else:
                response_content = str(response)
            
            # Clean and parse JSON
            json_start = response_content.find('{')
            json_end = response_content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_content[json_start:json_end]
                return json.loads(json_str)
            
            return json.loads(response_content)
            
        except Exception as e:
            logger.error(f"Error calling LLM for summary: {e}")
            return {
                "summary": "",
                "key_points": [],
                "topics": [],
                "unresolved": []
            }
    
    async def trigger_background_update(self, session_id: str):
        """Trigger background summary update when new message arrives"""
        
        task = asyncio.create_task(self._background_update(session_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        
    async def _background_update(self, session_id: str):
        """Background task to update summary"""
        
        try:
            logger.info(f"Starting background summary update for session {session_id}")
            
            # Small delay to ensure message is saved
            await asyncio.sleep(0.5)
            
            # Force refresh the summary
            await self.get_or_create_summary(session_id, force_refresh=True)
            
            logger.info(f"Completed background summary update for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error in background summary update: {e}")
    
    async def invalidate_cache(self, session_id: str):
        """Invalidate cached summary for a session"""
        
        try:
            redis = await self._get_redis()
            if redis:
                cache_key = f"{self.CACHE_PREFIX}{session_id}"
                await redis.delete(cache_key)
                logger.info(f"Invalidated cache for session {session_id}")
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")