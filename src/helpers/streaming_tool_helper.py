import json
from typing import AsyncGenerator, Dict, Any, Optional
from collections.abc import AsyncGenerator

from src.utils.logger.custom_logging import LoggerMixin


class StreamingToolHelper(LoggerMixin):
    """Helper class for streaming tool responses with memory accumulation"""
    
    @staticmethod
    async def format_sse_with_accumulation(
        stream_generator: AsyncGenerator[str, None],
        session_id: str
    ) -> AsyncGenerator[str, None]:
        """
        Format streaming response as SSE while accumulating full response
        
        Args:
            stream_generator: Generator yielding response chunks
            session_id: Session ID for tracking
            
        Yields:
            SSE formatted chunks
            
        Returns via side effect:
            Accumulated full response stored in generator
        """
        full_response = []
        
        async for chunk in stream_generator:
            if chunk:
                full_response.append(chunk)
                # Format as SSE
                event_data = {
                    "session_id": session_id,
                    "type": "chunk",
                    "data": chunk
                }
                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        
        # Store full response for later access
        stream_generator.accumulated_response = ''.join(full_response)
        
        # Send completion event
        completion_data = {
            "session_id": session_id,
            "type": "completion",
            "data": "[DONE]"
        }
        yield f"data: {json.dumps(completion_data)}\n\n"