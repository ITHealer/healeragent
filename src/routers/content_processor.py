from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, Dict, Any, List, AsyncGenerator
import os
import uuid
import json
import asyncio
import logging
from datetime import datetime

from src.agents.memory.memory_manager import MemoryManager
from src.handlers.content_processor import ContentProcessor, ContentTypeDetector
from src.media.handlers.content_processor_manager import processor_manager
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.chat_management_helper import ChatService
from src.helpers.llm_helper import LLMGeneratorProvider
from src.routers.llm_chat import analyze_conversation_importance
from src.services.background_tasks import trigger_summary_update_nowait
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.providers.provider_factory import ProviderType

# Configure logging
logger = LoggerMixin().logger

# API Router
router = APIRouter(prefix="/summarizer-url")

memory_manager = MemoryManager()
llm_provider = LLMGeneratorProvider()
chat_service = ChatService()
api_key_auth = APIKeyAuth()


# Request/Response Models
class StandardResponse(BaseModel):
    status: str = Field(..., description="Success or Error")
    message: str = Field("", description="Response message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    
class ContentProcessRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL of video or article to process")
    platforms: Optional[str] = Field(None, description="List of supported video platforms (just only for video)")
    target_language: Optional[str] = Field(
        None, 
        description="Target language code (en, vi, zh, ja, ko, etc.). If not provided, keeps source language"
    )
    include_original: Optional[bool] = Field(
        True,
        description="Include original content in response"
    )
    model_name: Optional[str] = Field(
        "gpt-4.1-nano",
        description="Model name for processing"
    )
    provider_type: Optional[str] = Field(
        "openai",
        description="Provider type (openai, anthropic, etc.)"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation context"
    )

class ContentProcessResponse(BaseModel):
    """Response model for content processing"""
    status: str = Field(..., description="Processing status (success/error)")
    content_type: str = Field(..., description="Content type (video/article)")
    url: str = Field(..., description="Original URL")
    source_language: Optional[str] = Field(None, description="Detected source language")
    target_language: Optional[str] = Field(None, description="Target language for output")
    summary: Optional[str] = Field(None, description="Generated summary")
    original_content: Optional[str] = Field(None, description="Original transcript or article text")
    translation: Optional[str] = Field(None, description="Translated content if applicable")
    translation_needed: Optional[bool] = Field(False, description="Whether translation was performed")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    context_used: Optional[bool] = None
    memory_stats: Optional[Dict[str, Any]] = None

class BatchProcessRequest(BaseModel):
    """Request model for batch processing"""
    session_id: Optional[str] = None
    urls: List[HttpUrl] = Field(..., description="List of URLs to process")
    target_language: Optional[str] = Field(None, description="Target language for all URLs")
    model_name: Optional[str] = Field("gpt-4.1-nano", description="Model name")
    provider_type: Optional[str] = Field("openai", description="Provider type")
    
    class Config:
        json_schema_extra = {
            "example": {
                "urls": [
                    "https://www.youtube.com/watch?v=example1",
                    "https://www.bbc.com/news/article-example"
                ],
                "target_language": "en"
            }
        }


# Dependency to get processor instance
def get_processor(model_name: str, provider_type: str) -> ContentProcessor:
    """
    Get configured content processor
    """
    # Validate provider type
    if provider_type.lower() not in ProviderType.list():
        raise HTTPException(
            status_code=400,
            detail=f"Invalid provider type. Must be one of: {ProviderType.list()}"
        )
    
    # Get API key based on provider
    api_key = None
    if provider_type.lower() == ProviderType.OLLAMA:
        api_key = None  # Ollama doesn't need API key
    else:
        # Get API key from environment
        if provider_type.lower() == ProviderType.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider_type.lower() == ProviderType.GEMINI:
            api_key = os.getenv("GEMINI_API_KEY")
        else:
            api_key = None
        
        if not api_key:
            raise HTTPException(
                status_code=500,
                detail=f"API key not configured for {provider_type}. Please set OPENAI_API_KEY environment variable."
            )
        
    # Get cached processor from manager
    try:
        processor = processor_manager.get_processor(
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key
        )
        return processor
        
    except Exception as e:
        logger.error(f"Failed to get processor: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize processor: {str(e)}"
        )
    
# Memory helper functions
async def get_memory_context(
    session_id: str,
    user_id: str,
    query: str,
    provider_type: str = "openai"
) -> tuple[str, Dict[str, Any]]:
    """
    Get relevant context from memory system
    
    Returns:
        tuple: (context string, memory stats)
    """
    if not session_id or not user_id:
        return "", {}
    
    try:
        context, memory_stats, document_references = await memory_manager.get_relevant_context(
            session_id=session_id,
            user_id=user_id,
            current_query=query,
            llm_provider=llm_provider,
            max_short_term=5,
            max_long_term=3
        )
        logger.info(f"Retrieved memory context for content processing: {memory_stats}")
        return context, memory_stats
    except Exception as e:
        logger.error(f"Error getting memory context: {e}")
        return "", {}

async def save_content_to_memory(
    session_id: str,
    user_id: str,
    model_name: str,
    provider_type: str,
    url: str,
    content_type: str,
    summary: str,
    original_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save processed content to memory system
    """
    if not session_id or not user_id:
        return
    
    try:
        # Create a structured query representing the content processing request
        query = f"Process {content_type} from URL: {url}"
        
        # Create comprehensive response with all extracted information
        response_parts = [f"**{content_type.title()} Summary:**\n{summary}"]
        
        if metadata:
            if content_type == "video" and metadata.get("video_info"):
                response_parts.insert(0, f"**Title:** {metadata['video_info'].get('title', 'N/A')}")
            response_parts.append(f"\n**Source:** {url}")
        
        response = "\n\n".join(response_parts)
        
        # Analyze importance based on content
        importance_score = await analyze_conversation_importance(
            query=query,
            response=response,
            llm_provider=llm_provider,
            model_name=model_name,
            provider_type=provider_type
        )
        
        # Store in memory
        await memory_manager.store_conversation_turn(
            session_id=session_id,
            user_id=user_id,
            query=query,
            response=response,
            metadata={
                "type": "content_processing",
                "content_type": content_type,
                "url": url,
                "source_language": metadata.get("source_language"),
                "target_language": metadata.get("target_language"),
                **metadata
            },
            importance_score=importance_score
        )
        
        # Trigger background summary update
        trigger_summary_update_nowait(session_id=session_id, user_id=user_id)
        
        logger.info(f"Saved content to memory: {content_type} from {url} with importance {importance_score}")
        
    except Exception as e:
        logger.error(f"Error saving content to memory: {e}")


# ======================= API ENDPOINTS =======================

@router.post("/process", response_model=StandardResponse)
async def process_content(
    request: Request,
    data: ContentProcessRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Enhanced content processing with standard response format
    
    Returns:
        StandardResponse with all data in 'data' field
    """
    try:
        start_time = datetime.now()
        user_id = getattr(request.state, "user_id", None)
        
        logger.info(f"Processing content: {data.url}")
        
        # Get memory context if enabled
        context = ""
        memory_stats = {}
        if data.session_id and user_id:
            context_query = f"Processing content from {data.url}. Target language: {data.target_language or 'auto'}"
            context, memory_stats = await get_memory_context(
                session_id=data.session_id,
                user_id=user_id,
                query=context_query,
                provider_type=data.provider_type
            )
        
        # Get processor
        processor = get_processor(
            model_name=data.model_name,
            provider_type=data.provider_type
        )
        
        # Process URL
        result = await processor.process_url(
            url=str(data.url),
            target_language=data.target_language,
            print_progress=False
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Handle error case
        if result.get("status") == "error":
            logger.error(f"Processing failed: {result.get('error')}")
            return StandardResponse(
                status="error",
                message=result.get("error", "Processing failed"),
                data={
                    "url": str(data.url),
                    "content_type": result.get("type", "unknown"),
                    "processing_time": processing_time,
                    "error_details": result.get("error")
                }
            )
        
        # Build response data
        response_data = {
            "url": str(data.url),
            "content_type": result.get("type", "unknown"),
            "source_language": result.get("source_language"),
            "target_language": result.get("target_language"),
            "summary": result.get("summary"),
            "translation_needed": result.get("translation_needed", False),
            "processing_time": processing_time,
            "cached_processor": True
        }
        
        # Add metadata
        if result.get("type") == "video":
            response_data["video_info"] = result.get("video_info", {})
            response_data["transcript_length"] = len(result.get("transcript", ""))
        elif result.get("type") == "article":
            response_data["original_length"] = result.get("original_length", 0)
            response_data["chunks_created"] = result.get("chunks_created", 1)
        
        # Include original content if requested
        if data.include_original:
            if result.get("type") == "video":
                response_data["original_content"] = result.get("transcript")
            else:
                response_data["original_content"] = result.get("original_text")
        
        # Include translation if exists
        if result.get("translation"):
            response_data["translation"] = result.get("translation")
        
        # Add memory info if used
        response_data["context_used"] = bool(context)
        response_data["memory_stats"] = memory_stats
    
        # Save to memory if enabled
        if data.session_id and user_id:
            await save_content_to_memory(
                session_id=data.session_id,
                user_id=user_id,
                model_name=data.model_name,
                provider_type=data.provider_type,
                url=str(data.url),
                content_type=result.get("type", "unknown"),
                summary=result.get("summary", ""),
                original_text=result.get("transcript") or result.get("original_text"),
                metadata={
                    "source_language": result.get("source_language"),
                    "target_language": result.get("target_language")
                }
            )
        
        # Save to chat history
        if data.session_id and user_id:
            try:
                question_id = chat_service.save_user_question(
                    session_id=data.session_id,
                    created_at=datetime.now(),
                    created_by=user_id,
                    content=f"Process {result.get('type', 'content')} from: {data.url}"
                )
                
                chat_service.save_assistant_response(
                    session_id=data.session_id,
                    created_at=datetime.now(),
                    question_id=question_id,
                    content=result.get("summary", ""),
                    response_time=processing_time
                )
            except Exception as e:
                logger.error(f"Error saving to chat history: {e}")
        
        # Return standard response
        message = f"Successfully processed {result.get('type', 'content')} in {processing_time:.2f} seconds"
        
        return StandardResponse(
            status="success",
            message=message,
            data=response_data
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return StandardResponse(
            status="error",
            message=f"Failed to process content: {str(e)}",
            data={
                "url": str(data.url),
                "error_details": str(e)
            }
        )


@router.post("/process-stream")
async def process_content_stream(
    request: Request,
    data: ContentProcessRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Stream only the summary content progressively
    
    Returns:
        SSE stream with summary chunks only
    """
    user_id = getattr(request.state, "user_id", None)
    
    async def generate_summary_stream() -> AsyncGenerator[str, None]:
        try:
            # Get processor
            processor = get_processor(data.model_name, data.provider_type)
            
            # Process the URL
            result = await processor.process_url(
                url=str(data.url),
                target_language=data.target_language,
                print_progress=False
            )
            
            # Check for errors
            if result.get("status") == "error":
                yield f"{json.dumps({'error': result.get('error', 'Processing failed')})}\n\n"
                yield "[DONE]\n\n"
                return
            
            # Get the summary
            summary = result.get("summary", "")
            
            if not summary:
                yield f"{json.dumps({'error': 'No summary generated'})}\n\n"
                yield "[DONE]\n\n"
                return
            
            # Stream summary in chunks (simulate progressive generation)
            # Split by sentences for natural streaming
            import re
            sentences = re.split(r'(?<=[.!?])\s+', summary)
            
            current_chunk = ""
            for sentence in sentences:
                current_chunk = sentence + " "
                
                # Stream each sentence
                yield f"{json.dumps({'content': current_chunk})}\n\n"
                
                # Small delay for natural streaming effect
                await asyncio.sleep(0.1)
            
            # Save to memory if enabled (after streaming complete)
            if data.session_id and user_id:
                try:
                    await save_content_to_memory(
                        session_id=data.session_id,
                        user_id=user_id,
                        model_name=data.model_name,
                        provider_type=data.provider_type,
                        url=str(data.url),
                        content_type=result.get("type", "unknown"),
                        summary=summary,
                        metadata={
                            "source_language": result.get("source_language"),
                            "target_language": result.get("target_language"),
                            "streamed": True
                        }
                    )
                except Exception as e:
                    logger.error(f"Error saving to memory: {e}")
            
            # Send completion signal
            yield "[DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"{json.dumps({'error': str(e)})}\n\n"
            yield "[DONE]\n\n"
    
    return StreamingResponse(
        generate_summary_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*"
        }
    )

# Admin endpoints for cache management
# @router.post("/admin/clear-cache", response_model=StandardResponse)
# async def clear_processor_cache(
#     cache_key: Optional[str] = None,
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     """
#     Clear processor cache (admin only).
#     Useful for forcing re-initialization or freeing memory.
#     """
#     try:
#         processor_manager.clear_cache(cache_key)
        
#         return StandardResponse(
#             status="success",
#             message=f"Cache cleared: {cache_key if cache_key else 'all'}",
#             data=processor_manager.get_cache_stats()
#         )
#     except Exception as e:
#         return StandardResponse(
#             status="error",
#             message=f"Failed to clear cache: {str(e)}",
#             data={}
#         )


# @router.get("/admin/cache-stats", response_model=StandardResponse)
# async def get_cache_stats(
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     """Get processor cache statistics"""
#     return StandardResponse(
#         status="success",
#         message="Cache statistics",
#         data=processor_manager.get_cache_stats()
#     )

@router.post("/batch-process", response_model=StandardResponse)
async def batch_process_content(
    request: Request,
    data: BatchProcessRequest,
    background_tasks: BackgroundTasks,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Batch process multiple URLs with standard response format
    """
    try:
        user_id = getattr(request.state, "user_id", None)
        task_id = str(uuid.uuid4())
        
        # Initialize batch tracking
        batch_info = {
            "task_id": task_id,
            "status": "processing",
            "total": len(data.urls),
            "completed": 0,
            "results": [],
            "memory_enabled": bool(data.session_id)
        }
        
        async def process_batch():
            """Background task to process all URLs"""
            processor = get_processor(data.model_name, data.provider_type)
            
            for i, url in enumerate(data.urls):
                try:
                    result = await processor.process_url(
                        url=str(url),
                        target_language=data.target_language,
                        print_progress=False
                    )
                    
                    # Save to memory if enabled
                    if data.session_id and user_id and result.get("status") == "success":
                        await save_content_to_memory(
                            session_id=data.session_id,
                            user_id=user_id,
                            model_name=data.model_name,
                            provider_type=data.provider_type,
                            url=str(url),
                            content_type=result.get("type", "unknown"),
                            summary=result.get("summary", ""),
                            metadata={"batch_id": task_id, "index": i}
                        )
                    
                    batch_info["results"].append({
                        "url": str(url),
                        "status": result.get("status", "success"),
                        "summary": result.get("summary"),
                        "content_type": result.get("type")
                    })
                    
                except Exception as e:
                    batch_info["results"].append({
                        "url": str(url),
                        "status": "error",
                        "error": str(e)
                    })
                
                batch_info["completed"] = i + 1
            
            batch_info["status"] = "completed"
            
            # Trigger memory update if used
            if data.session_id and user_id:
                trigger_summary_update_nowait(session_id=data.session_id, user_id=user_id)
        
        # Add to background tasks
        background_tasks.add_task(process_batch)
        
        return StandardResponse(
            status="success",
            message=f"Batch processing started for {len(data.urls)} URLs",
            data={
                "task_id": task_id,
                "total_urls": len(data.urls),
                "check_status_url": f"/content/batch-status/{task_id}",
                "memory_enabled": batch_info["memory_enabled"]
            }
        )
        
    except Exception as e:
        logger.error(f"Error starting batch process: {str(e)}")
        return StandardResponse(
            status="error",
            message=f"Failed to start batch processing: {str(e)}",
            data={"urls": [str(url) for url in data.urls]}
        )


# @router.get("/health", response_model=StandardResponse)
# async def health_check():
#     """Health check with standard response format"""
#     try:
#         api_key = os.getenv("OPENAI_API_KEY")
        
#         # Check video support
#         try:
#             from src.media.video_processor import VideoProcessor
#             video_support = True
#         except:
#             video_support = False
        
#         health_data = {
#             "service": "Universal Content Processor",
#             "status": "healthy" if api_key else "unhealthy",
#             "capabilities": {
#                 "video_processing": video_support,
#                 "article_processing": True,
#                 "translation": True,
#                 "summarization": True
#             },
#             "supported_languages": ["en", "vi", "zh", "ja", "ko", "fr", "de", "es"],
#             "timestamp": datetime.now().isoformat()
#         }
        
#         if not api_key:
#             return StandardResponse(
#                 status="error",
#                 message="API key not configured",
#                 data=health_data
#             )
        
#         return StandardResponse(
#             status="success",
#             message="Service is healthy",
#             data=health_data
#         )
        
#     except Exception as e:
#         return StandardResponse(
#             status="error",
#             message=f"Health check failed: {str(e)}",
#             data={"timestamp": datetime.now().isoformat()}
#         )

# @router.get("/supported-platforms")
# async def get_supported_platforms():
#     """Get list of supported video platforms and content types"""
#     return {
#         "video_platforms": list(ContentTypeDetector.VIDEO_PLATFORMS.keys()),
#         "video_extensions": ContentTypeDetector.VIDEO_EXTENSIONS,
#         "article_processing": "All standard web articles and blog posts",
#         "translation_support": {
#             "supported_languages": ["en", "vi", "zh", "ja", "ko", "fr", "de", "es", "pt", "ru", "ar"],
#             "auto_detection": True,
#             "conditional_translation": "Automatically translates when target language differs from source"
#         },
#         "features": {
#             "video": [
#                 "Audio extraction with yt-dlp",
#                 "Speech-to-text with Faster-Whisper",
#                 "Automatic language detection",
#                 "AI-powered transcript optimization",
#                 "Multi-language summarization",
#                 "Conditional translation"
#             ],
#             "article": [
#                 "Dynamic content extraction with Selenium",
#                 "Smart text chunking for long content",
#                 "Language detection with lingua",
#                 "Context-aware summarization",
#                 "Long-form content support",
#                 "Translation support"
#             ]
#         },
#         "api_endpoints": {
#             "single_process": "/content/process",
#             "stream_process": "/content/process-stream",
#             "batch_process": "/content/batch-process",
#             "health_check": "/content/health",
#             "supported_platforms": "/content/supported-platforms"
#         }
#     }


# Integration with app.py
def get_router():
    """Return router for inclusion in main app"""
    return router