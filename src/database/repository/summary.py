# """
# Repository for Recursive Summaries
# Handles CRUD operations for conversation summaries
# """

# from typing import Optional, Dict, Any, List
# from datetime import datetime
# import uuid

# from src.database import get_postgres_db
# from src.database.models.recursive_summaries import RecursiveSummaries
# from src.utils.logger.custom_logging import LoggerMixin


# class SummaryRepository(LoggerMixin):
#     """Repository for managing recursive summaries"""
    
#     def __init__(self):
#         super().__init__()
#         self.db = get_postgres_db()
    
    
#     def get_active_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
#         """
#         Get the current active summary for a session
        
#         Args:
#             session_id: Session identifier
            
#         Returns:
#             Dict with summary data or None
#         """
#         try:
#             with self.db.session_scope() as session:
#                 summary = session.query(RecursiveSummaries).filter(
#                     RecursiveSummaries.session_id == session_id,
#                     RecursiveSummaries.is_active == True
#                 ).order_by(RecursiveSummaries.version.desc()).first()
                
#                 if summary:
#                     return {
#                         'id': str(summary.id),
#                         'session_id': str(summary.session_id),
#                         'user_id': summary.user_id,
#                         'version': summary.version,
#                         'summary_text': summary.summary_text,
#                         'token_count': summary.token_count,
#                         'message_count': summary.message_count,
#                         'total_messages_in_session': summary.total_messages_in_session,
#                         'model_name': summary.model_name,
#                         'created_at': summary.created_at,
#                         'updated_at': summary.updated_at
#                     }
                
#                 return None
                
#         except Exception as e:
#             self.logger.error(f"Error getting active summary for session {session_id}: {e}")
#             return None
    
    
#     def create_summary(
#         self,
#         session_id: str,
#         user_id: str,
#         summary_text: str,
#         version: int,
#         token_count: int,
#         message_count: int,
#         total_messages: int,
#         model_name: str,
#         organization_id: Optional[str] = None,
#         messages_start_id: Optional[str] = None,
#         messages_end_id: Optional[str] = None,
#         previous_summary_tokens: Optional[int] = None
#     ) -> Optional[str]:
#         """
#         Create a new summary version
        
#         Args:
#             session_id: Session identifier
#             user_id: User identifier
#             summary_text: The summary content
#             version: Summary version number
#             token_count: Number of tokens in summary
#             message_count: Number of messages summarized in this version
#             total_messages: Total messages in session at time of summary
#             model_name: LLM model used for summarization
#             organization_id: Organization ID (optional)
#             messages_start_id: First message ID in summary
#             messages_end_id: Last message ID in summary
#             previous_summary_tokens: Token count from previous summary
            
#         Returns:
#             Summary ID or None
#         """
#         try:
#             with self.db.session_scope() as session:
#                 # Deactivate previous summaries for this session
#                 session.query(RecursiveSummaries).filter(
#                     RecursiveSummaries.session_id == session_id,
#                     RecursiveSummaries.is_active == True
#                 ).update({'is_active': False})
                
#                 # Create new summary
#                 summary_id = uuid.uuid4()
#                 new_summary = RecursiveSummaries(
#                     id=summary_id,
#                     session_id=session_id,
#                     user_id=user_id,
#                     organization_id=organization_id,
#                     version=version,
#                     summary_text=summary_text,
#                     messages_start_id=messages_start_id,
#                     messages_end_id=messages_end_id,
#                     message_count=message_count,
#                     total_messages_in_session=total_messages,
#                     token_count=token_count,
#                     previous_summary_tokens=previous_summary_tokens,
#                     model_name=model_name,
#                     is_active=True,
#                     created_at=datetime.utcnow()
#                 )
                
#                 session.add(new_summary)
                
#                 self.logger.info(
#                     f"Created summary v{version} for session {session_id}: "
#                     f"{token_count} tokens, {message_count} messages"
#                 )
                
#                 return str(summary_id)
                
#         except Exception as e:
#             self.logger.error(f"Error creating summary: {e}")
#             return None
    
    
#     def get_summary_history(
#         self,
#         session_id: str,
#         limit: int = 10
#     ) -> List[Dict[str, Any]]:
#         """
#         Get summary version history for a session
        
#         Args:
#             session_id: Session identifier
#             limit: Maximum number of versions to return
            
#         Returns:
#             List of summary dicts (newest first)
#         """
#         try:
#             with self.db.session_scope() as session:
#                 summaries = session.query(RecursiveSummaries).filter(
#                     RecursiveSummaries.session_id == session_id
#                 ).order_by(RecursiveSummaries.version.desc()).limit(limit).all()
                
#                 return [{
#                     'id': str(s.id),
#                     'version': s.version,
#                     'summary_text': s.summary_text,
#                     'token_count': s.token_count,
#                     'message_count': s.message_count,
#                     'total_messages': s.total_messages_in_session,
#                     'is_active': s.is_active,
#                     'created_at': s.created_at
#                 } for s in summaries]
                
#         except Exception as e:
#             self.logger.error(f"Error getting summary history: {e}")
#             return []
    
    
#     def get_summary_stats(self, session_id: str) -> Dict[str, Any]:
#         """
#         Get statistics about summaries for a session
        
#         Args:
#             session_id: Session identifier
            
#         Returns:
#             Dict with summary statistics
#         """
#         try:
#             with self.db.session_scope() as session:
#                 summaries = session.query(RecursiveSummaries).filter(
#                     RecursiveSummaries.session_id == session_id
#                 ).all()
                
#                 if not summaries:
#                     return {
#                         'has_summary': False,
#                         'version_count': 0
#                     }
                
#                 active_summary = next((s for s in summaries if s.is_active), None)
                
#                 return {
#                     'has_summary': True,
#                     'version_count': len(summaries),
#                     'current_version': active_summary.version if active_summary else 0,
#                     'current_tokens': active_summary.token_count if active_summary else 0,
#                     'total_messages_summarized': active_summary.total_messages_in_session if active_summary else 0,
#                     'last_updated': active_summary.updated_at or active_summary.created_at if active_summary else None
#                 }
                
#         except Exception as e:
#             self.logger.error(f"Error getting summary stats: {e}")
#             return {'has_summary': False, 'error': str(e)}
    
    
#     def delete_session_summaries(self, session_id: str) -> bool:
#         """
#         Delete all summaries for a session
        
#         Args:
#             session_id: Session identifier
            
#         Returns:
#             Success status
#         """
#         try:
#             with self.db.session_scope() as session:
#                 deleted_count = session.query(RecursiveSummaries).filter(
#                     RecursiveSummaries.session_id == session_id
#                 ).delete()
                
#                 self.logger.info(f"Deleted {deleted_count} summaries for session {session_id}")
#                 return True
                
#         except Exception as e:
#             self.logger.error(f"Error deleting summaries: {e}")
#             return False


"""
Repository for Recursive Summaries
Handles CRUD operations for conversation summaries

PRODUCTION NOTES:
- Use get_summary_repository() singleton to avoid memory leaks
- Use async methods (get_active_summary_async, etc.) to avoid blocking event loop
- Sync methods kept for backward compatibility but should be migrated
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from src.database import get_postgres_db
from src.database.models.recursive_summaries import RecursiveSummaries
from src.utils.logger.custom_logging import LoggerMixin


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
_summary_repository_instance: Optional['SummaryRepository'] = None
_summary_repository_lock = asyncio.Lock()

# Thread pool for running sync DB operations without blocking event loop
_db_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="summary_repo_")


def get_summary_repository() -> 'SummaryRepository':
    """
    Get singleton instance of SummaryRepository.

    Use this instead of SummaryRepository() to prevent memory leaks
    when handling thousands of concurrent requests.

    Returns:
        SummaryRepository singleton instance
    """
    global _summary_repository_instance

    if _summary_repository_instance is None:
        _summary_repository_instance = SummaryRepository()

    return _summary_repository_instance


class SummaryRepository(LoggerMixin):
    """
    Repository for managing recursive summaries.

    IMPORTANT: Use get_summary_repository() singleton instead of direct instantiation.

    For async contexts (FastAPI routes, async handlers), use the async methods:
    - get_active_summary_async()
    - create_summary_async()
    - get_summary_stats_async()
    - delete_session_summaries_async()

    Sync methods are kept for backward compatibility but will block the event loop.
    """

    def __init__(self):
        super().__init__()
        self.db = get_postgres_db()
        self._executor = _db_executor  # Shared thread pool
    
    
    def get_active_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current active summary for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with summary data or None
        """
        try:
            with self.db.session_scope() as session:
                summary = session.query(RecursiveSummaries).filter(
                    RecursiveSummaries.session_id == session_id,
                    RecursiveSummaries.is_active == True
                ).order_by(RecursiveSummaries.version.desc()).first()
                
                if summary:
                    return {
                        'id': str(summary.id),
                        'session_id': str(summary.session_id),
                        'user_id': summary.user_id,
                        'version': summary.version,
                        'summary_text': summary.summary_text,
                        'token_count': summary.token_count,
                        'message_count': summary.message_count,
                        'total_messages_in_session': summary.total_messages_in_session,
                        'model_name': summary.model_name,
                        'created_at': summary.created_at,
                        'updated_at': summary.updated_at
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting active summary for session {session_id}: {e}")
            return None
    
    
    def create_summary(
        self,
        session_id: str,
        user_id: str,
        summary_text: str,
        version: int,
        token_count: int,
        message_count: int,
        total_messages: int,
        model_name: str,
        organization_id: Optional[str] = None,
        messages_start_id: Optional[str] = None,
        messages_end_id: Optional[str] = None,
        previous_summary_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Create a new summary version
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            summary_text: The summary content
            version: Summary version number
            token_count: Number of tokens in summary
            message_count: Number of messages summarized in this version
            total_messages: Total messages in session at time of summary
            model_name: LLM model used for summarization
            organization_id: Organization ID (optional)
            messages_start_id: First message ID in summary
            messages_end_id: Last message ID in summary
            previous_summary_tokens: Token count from previous summary
            
        Returns:
            Summary ID or None
        """
        try:
            with self.db.session_scope() as session:
                # Deactivate previous summaries for this session
                session.query(RecursiveSummaries).filter(
                    RecursiveSummaries.session_id == session_id,
                    RecursiveSummaries.is_active == True
                ).update({'is_active': False})
                
                # Create new summary
                summary_id = uuid.uuid4()
                new_summary = RecursiveSummaries(
                    id=summary_id,
                    session_id=session_id,
                    user_id=user_id,
                    organization_id=organization_id,
                    version=version,
                    summary_text=summary_text,
                    messages_start_id=messages_start_id,
                    messages_end_id=messages_end_id,
                    message_count=message_count,
                    total_messages_in_session=total_messages,
                    token_count=token_count,
                    previous_summary_tokens=previous_summary_tokens,
                    model_name=model_name,
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                
                session.add(new_summary)
                
                self.logger.info(
                    f"Created summary v{version} for session {session_id}: "
                    f"{token_count} tokens, {message_count} messages"
                )
                
                return str(summary_id)
                
        except Exception as e:
            self.logger.error(f"Error creating summary: {e}")
            return None
    
    
    def get_summary_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get summary version history for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of versions to return
            
        Returns:
            List of summary dicts (newest first)
        """
        try:
            with self.db.session_scope() as session:
                summaries = session.query(RecursiveSummaries).filter(
                    RecursiveSummaries.session_id == session_id
                ).order_by(RecursiveSummaries.version.desc()).limit(limit).all()
                
                return [{
                    'id': str(s.id),
                    'version': s.version,
                    'summary_text': s.summary_text,
                    'token_count': s.token_count,
                    'message_count': s.message_count,
                    'total_messages': s.total_messages_in_session,
                    'is_active': s.is_active,
                    'created_at': s.created_at
                } for s in summaries]
                
        except Exception as e:
            self.logger.error(f"Error getting summary history: {e}")
            return []
    
    
    def get_summary_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics about summaries for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with summary statistics
        """
        try:
            with self.db.session_scope() as session:
                summaries = session.query(RecursiveSummaries).filter(
                    RecursiveSummaries.session_id == session_id
                ).all()
                
                if not summaries:
                    return {
                        'has_summary': False,
                        'version_count': 0
                    }
                
                active_summary = next((s for s in summaries if s.is_active), None)
                
                return {
                    'has_summary': True,
                    'version_count': len(summaries),
                    'current_version': active_summary.version if active_summary else 0,
                    'current_tokens': active_summary.token_count if active_summary else 0,
                    'total_messages_summarized': active_summary.total_messages_in_session if active_summary else 0,
                    'last_updated': active_summary.updated_at or active_summary.created_at if active_summary else None
                }
                
        except Exception as e:
            self.logger.error(f"Error getting summary stats: {e}")
            return {'has_summary': False, 'error': str(e)}
    
    
    def delete_session_summaries(self, session_id: str) -> bool:
        """
        Delete all summaries for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        try:
            with self.db.session_scope() as session:
                deleted_count = session.query(RecursiveSummaries).filter(
                    RecursiveSummaries.session_id == session_id
                ).delete()

                self.logger.info(f"Deleted {deleted_count} summaries for session {session_id}")
                return True

        except Exception as e:
            self.logger.error(f"Error deleting summaries: {e}")
            return False

    # =========================================================================
    # ASYNC WRAPPER METHODS (NON-BLOCKING)
    # Use these in async contexts to avoid blocking the event loop
    # =========================================================================

    async def get_active_summary_async(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Async version of get_active_summary.

        Runs the sync database operation in a thread pool to avoid
        blocking the event loop. Safe for production with high concurrency.

        Args:
            session_id: Session identifier

        Returns:
            Dict with summary data or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_active_summary,
            session_id
        )

    async def create_summary_async(
        self,
        session_id: str,
        user_id: str,
        summary_text: str,
        version: int,
        token_count: int,
        message_count: int,
        total_messages: int,
        model_name: str,
        organization_id: Optional[str] = None,
        messages_start_id: Optional[str] = None,
        messages_end_id: Optional[str] = None,
        previous_summary_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Async version of create_summary.

        Runs the sync database operation in a thread pool to avoid
        blocking the event loop.

        Args:
            session_id: Session identifier
            user_id: User identifier
            summary_text: The summary content
            version: Summary version number
            token_count: Number of tokens in summary
            message_count: Number of messages summarized
            total_messages: Total messages in session
            model_name: LLM model used
            organization_id: Organization ID (optional)
            messages_start_id: First message ID
            messages_end_id: Last message ID
            previous_summary_tokens: Previous summary token count

        Returns:
            Summary ID or None
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.create_summary(
                session_id=session_id,
                user_id=user_id,
                summary_text=summary_text,
                version=version,
                token_count=token_count,
                message_count=message_count,
                total_messages=total_messages,
                model_name=model_name,
                organization_id=organization_id,
                messages_start_id=messages_start_id,
                messages_end_id=messages_end_id,
                previous_summary_tokens=previous_summary_tokens
            )
        )

    async def get_summary_stats_async(self, session_id: str) -> Dict[str, Any]:
        """
        Async version of get_summary_stats.

        Args:
            session_id: Session identifier

        Returns:
            Dict with summary statistics
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.get_summary_stats,
            session_id
        )

    async def delete_session_summaries_async(self, session_id: str) -> bool:
        """
        Async version of delete_session_summaries.

        Args:
            session_id: Session identifier

        Returns:
            Success status
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.delete_session_summaries,
            session_id
        )

    async def get_summary_history_async(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Async version of get_summary_history.

        Args:
            session_id: Session identifier
            limit: Maximum number of versions to return

        Returns:
            List of summary dicts (newest first)
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.get_summary_history(session_id, limit)
        )