import asyncio
import logging
from typing import Optional
from datetime import datetime, timedelta
from src.services.session_summary_service import SessionSummaryService
from src.helpers.chat_management_helper import ChatService
from src.helpers.llm_helper import LLMGeneratorProvider

logger = logging.getLogger(__name__)

class BackgroundTaskManager:
    """Manages background tasks for session summaries"""
    
    def __init__(self):
        self.chat_service = ChatService()
        self.llm_service = LLMGeneratorProvider()
        self.summary_service = SessionSummaryService(
            chat_service=self.chat_service,
            llm_service=self.llm_service
        )
        self.running_tasks = {}
        
    def schedule_summary_update_nowait(
        self, 
        session_id: str,
        delay_seconds: int = 30
    ):
        """
        Schedule summary update WITHOUT waiting (fire-and-forget)
        This returns immediately without blocking the API
        """
        try:
            # Cancel existing task if any
            if session_id in self.running_tasks:
                self.running_tasks[session_id].cancel()
            
            # Create task WITHOUT await - runs in background
            task = asyncio.create_task(
                self._delayed_summary_update(session_id, delay_seconds)
            )
            self.running_tasks[session_id] = task
            
            # Add callback to handle exceptions
            task.add_done_callback(
                lambda t: self._handle_task_exception(t, session_id)
            )
            
        except Exception as e:
            logger.error(f"Error scheduling summary update: {e}")
    
    def _handle_task_exception(self, task: asyncio.Task, session_id: str):
        """Handle exceptions from background tasks"""
        try:
            if task.exception():
                logger.error(f"Background task failed for session {session_id}: {task.exception()}")
        except asyncio.CancelledError:
            pass
        finally:
            if session_id in self.running_tasks:
                del self.running_tasks[session_id]
    
    async def _delayed_summary_update(
        self, 
        session_id: str, 
        delay_seconds: int
    ):
        """Execute summary update after delay"""
        try:
            # Wait for delay (allows batching multiple messages)
            await asyncio.sleep(delay_seconds)
            
            # Update summary
            logger.info(f"Updating summary for session {session_id}")
            await self.summary_service.get_or_create_summary(
                session_id=session_id,
                force_refresh=True
            )
                
        except asyncio.CancelledError:
            logger.info(f"Summary update cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Error updating summary for session {session_id}: {e}")

# Global instance
background_task_manager = BackgroundTaskManager()

def trigger_summary_update_nowait(
    session_id: str,
    user_id: Optional[str] = None
):
    """
    COMPLETELY NON-BLOCKING trigger for summary update
    This function returns immediately without any await
    """
    try:
        # Direct call without await - completely non-blocking
        background_task_manager.schedule_summary_update_nowait(
            session_id=session_id,
            delay_seconds=30
        )
    except Exception as e:
        logger.error(f"Error triggering summary update: {e}")

# Alternative: Using asyncio.ensure_future for even more control
def trigger_summary_update_deferred(
    session_id: str,
    user_id: Optional[str] = None
):
    """
    Alternative non-blocking approach using ensure_future
    """
    async def _update_task():
        try:
            await asyncio.sleep(30)  # Delay
            summary_service = SessionSummaryService(
                chat_service=ChatService(),
                llm_service=LLMGeneratorProvider()
            )
            await summary_service.get_or_create_summary(
                session_id=session_id,
                force_refresh=True
            )
        except Exception as e:
            logger.error(f"Deferred summary update failed: {e}")
    
    # Schedule without waiting
    asyncio.ensure_future(_update_task())