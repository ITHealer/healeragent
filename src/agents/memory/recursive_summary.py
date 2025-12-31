"""
Recursive Summary Manager for MemGPT-style Memory System

PRODUCTION NOTES:
- Uses singleton patterns for SummaryRepository and ChatService
- All database operations are async (non-blocking)
- Optimized thresholds for production workloads
"""

from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.database.repository.summary import get_summary_repository
from src.services.summary_generation_service import SummaryGenerationService
from src.helpers.chat_management_helper import get_chat_service
from src.providers.provider_factory import ProviderType


class RecursiveSummaryManager(LoggerMixin):
    """
    Manages recursive conversation summaries

    Pattern:
    - Messages 1-15 → Summary v1
    - Messages 16-30 + Summary v1 → Summary v2
    - Messages 31-45 + Summary v2 → Summary v3

    Key Features:
    - Progressive summarization to handle long conversations
    - Preserves important context while reducing token usage
    - Integrates with context compaction for memory efficiency
    - Non-blocking async database operations for production
    """

    # ==========================================================================
    # PRODUCTION CONFIGURATION
    # ==========================================================================
    MESSAGE_THRESHOLD = 15          # Create summary every N messages (was 10)
    MAX_SUMMARY_TOKENS = 1500       # Keep summaries concise (was 1000)
    MIN_NEW_MESSAGES_FOR_RECURSIVE = 8  # Min new messages for recursive (was 5)

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI
    ):
        """
        Initialize RecursiveSummaryManager

        Uses singleton instances to prevent memory leaks in production.

        Args:
            model_name: LLM model for summarization
            provider_type: LLM provider
        """
        super().__init__()

        # Use singletons instead of creating new instances
        self.summary_repo = get_summary_repository()
        self.chat_service = get_chat_service()

        # SummaryGenerationService is lightweight, can create per instance
        self.summary_service = SummaryGenerationService(model_name=model_name)
        self.provider_type = provider_type

        self.logger.info(
            f"[RECURSIVE-SUMMARY] Initialized with threshold={self.MESSAGE_THRESHOLD}, "
            f"model={model_name}"
        )
    
    
    async def get_active_summary(self, session_id: str) -> Optional[str]:
        """
        Get current active summary for a session

        Uses async database call to avoid blocking event loop.

        Args:
            session_id: Session identifier

        Returns:
            Summary text or None
        """
        try:
            # Use async method to avoid blocking event loop
            summary = await self.summary_repo.get_active_summary_async(session_id)

            if summary:
                self.logger.debug(
                    f"[SUMMARY] Loaded v{summary['version']} for session {session_id[:8]}"
                )
                return summary['summary_text']

            return None

        except Exception as e:
            self.logger.error(f"[SUMMARY] Error getting active summary: {e}")
            return None
    
    
    async def check_and_create_summary(
        self,
        session_id: str,
        user_id: str,
        organization_id: Optional[str] = None,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if summary should be created and create it
        
        Decision Logic:
        1. No existing summary + messages >= threshold → Create initial
        2. Has summary + new_messages >= threshold → Create recursive
        3. Otherwise → Skip
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            organization_id: Organization ID (optional)
            model_name: LLM model (optional override)
            provider_type: LLM provider (optional override)
            
        Returns:
            Dict with creation status and metadata
        """
        try:
            self.logger.info(
                f"[SUMMARY CHECK] Session: {session_id[:8]}..., User: {user_id}"
            )

            # Step 1: Get message count (async)
            message_count = await self._get_session_message_count_async(session_id)

            self.logger.info(
                f"[SUMMARY CHECK] Total messages: {message_count}, "
                f"Threshold: {self.MESSAGE_THRESHOLD}"
            )

            # Step 2: Get existing summary (async)
            existing_summary = await self.summary_repo.get_active_summary_async(session_id)
            
            # Step 3: Determine if summary is needed
            should_create, reason = self._should_create_summary(
                message_count=message_count,
                existing_summary=existing_summary
            )
            
            if not should_create:
                self.logger.info(f"[SUMMARY CHECK] SKIP - {reason}")
                return {
                    'created': False,
                    'reason': reason,
                    'message_count': message_count,
                    'threshold': self.MESSAGE_THRESHOLD
                }
            
            # Step 4: Create summary
            self.logger.info(f"[SUMMARY CREATE] Starting: {reason}")
            
            result = await self._create_new_summary(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                message_count=message_count,
                existing_summary=existing_summary,
                model_name=model_name,
                provider_type=provider_type
            )
            
            # Log result
            if result.get('created'):
                self.logger.info(
                    f"[SUMMARY CREATE] SUCCESS - v{result['version']}: "
                    f"{result['token_count']} tokens, "
                    f"summarized {result['messages_summarized']} messages"
                )
            else:
                self.logger.warning(
                    f"[SUMMARY CREATE] FAILED - {result.get('error', 'Unknown')}"
                )

            return result
            
        except Exception as e:
            self.logger.error(f"[SUMMARY] Error in check_and_create_summary: {e}")
            return {
                'created': False,
                'error': str(e)
            }
    
    
    def _should_create_summary(
        self,
        message_count: int,
        existing_summary: Optional[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """
        Determine if a summary should be created
        
        FIXED: Clear, deterministic logic
        
        Args:
            message_count: Total messages in session
            existing_summary: Existing summary data or None
            
        Returns:
            (should_create: bool, reason: str)
        """
        # Case 1: Not enough messages yet
        if message_count < self.MESSAGE_THRESHOLD:
            return False, f"Not enough messages ({message_count} < {self.MESSAGE_THRESHOLD})"
        
        # Case 2: No existing summary - create initial
        if not existing_summary:
            return True, f"Creating INITIAL summary for {message_count} messages"
        
        # Case 3: Has existing summary - check if new messages warrant recursive
        previous_total = existing_summary.get('total_messages_in_session', 0)
        new_message_count = message_count - previous_total
        
        # Already summarized up to this point
        if new_message_count <= 0:
            return False, f"No new messages since last summary (v{existing_summary['version']})"
        
        # Not enough new messages for recursive summary
        if new_message_count < self.MIN_NEW_MESSAGES_FOR_RECURSIVE:
            return False, (
                f"Not enough new messages for recursive "
                f"({new_message_count} < {self.MIN_NEW_MESSAGES_FOR_RECURSIVE})"
            )
        
        # Check if we've accumulated enough new messages
        # Create recursive summary when new messages reach threshold
        if new_message_count >= self.MESSAGE_THRESHOLD:
            return True, (
                f"Creating RECURSIVE summary: v{existing_summary['version']} + "
                f"{new_message_count} new messages → v{existing_summary['version'] + 1}"
            )
        
        # Optional: Create recursive at intervals (every N new messages)
        # This handles gradual accumulation
        if message_count % self.MESSAGE_THRESHOLD == 0:
            return True, (
                f"Creating RECURSIVE summary at interval: "
                f"{message_count} total messages"
            )
        
        return False, (
            f"Waiting for more messages ({new_message_count} new, "
            f"need {self.MESSAGE_THRESHOLD})"
        )
    
    
    async def _create_new_summary(
        self,
        session_id: str,
        user_id: str,
        organization_id: Optional[str],
        message_count: int,
        existing_summary: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new summary (initial or recursive)
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            organization_id: Organization ID
            message_count: Current message count
            existing_summary: Previous summary (if any)
            model_name: LLM model override
            provider_type: LLM provider override
            
        Returns:
            Dict with creation result
        """
        try:
            if existing_summary:
                # Recursive summarization
                result = await self._create_recursive_summary(
                    session_id=session_id,
                    user_id=user_id,
                    organization_id=organization_id,
                    existing_summary=existing_summary,
                    message_count=message_count,
                    model_name=model_name,
                    provider_type=provider_type
                )
            else:
                # Initial summarization
                result = await self._create_initial_summary(
                    session_id=session_id,
                    user_id=user_id,
                    organization_id=organization_id,
                    message_count=message_count,
                    model_name=model_name,
                    provider_type=provider_type
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"[SUMMARY] Error creating new summary: {e}")
            return {
                'created': False,
                'error': str(e)
            }
    
    
    async def _create_initial_summary(
        self,
        session_id: str,
        user_id: str,
        organization_id: Optional[str],
        message_count: int,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create first summary from scratch
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            organization_id: Organization ID
            message_count: Current message count
            model_name: LLM model override
            provider_type: LLM provider override
            
        Returns:
            Dict with creation result
        """
        try:
            # Get messages to summarize (async)
            messages = await self._get_messages_for_summary_async(
                session_id=session_id,
                start_index=0,
                count=message_count
            )

            if not messages:
                return {
                    'created': False,
                    'error': 'No messages to summarize'
                }

            self.logger.debug(
                f"[SUMMARY] Generating initial summary from {len(messages)} messages"
            )

            # Generate summary
            summary_result = await self.summary_service.generate_initial_summary(
                messages=messages,
                model_name=model_name,
                provider_type=provider_type
            )

            if not summary_result.get('summary_text'):
                return {
                    'created': False,
                    'error': 'Summary generation failed - no text returned'
                }

            # Save to database (async)
            summary_id = await self.summary_repo.create_summary_async(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                summary_text=summary_result['summary_text'],
                version=1,
                token_count=summary_result['token_count'],
                message_count=len(messages),
                total_messages=message_count,
                model_name=model_name or self.summary_service.model_name,
                messages_start_id=messages[0].get('id') if messages else None,
                messages_end_id=messages[-1].get('id') if messages else None
            )
            
            if summary_id:
                return {
                    'created': True,
                    'summary_id': summary_id,
                    'version': 1,
                    'token_count': summary_result['token_count'],
                    'messages_summarized': len(messages)
                }
            else:
                return {
                    'created': False,
                    'error': 'Failed to save summary to database'
                }
                
        except Exception as e:
            self.logger.error(f"[SUMMARY] Error creating initial summary: {e}")
            return {
                'created': False,
                'error': str(e)
            }
    
    
    async def _create_recursive_summary(
        self,
        session_id: str,
        user_id: str,
        organization_id: Optional[str],
        existing_summary: Dict[str, Any],
        message_count: int,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create new summary by combining existing summary with new messages
        This implements the RECURSIVE pattern
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            organization_id: Organization ID
            existing_summary: Previous summary data
            message_count: Current message count
            model_name: LLM model override
            provider_type: LLM provider override
            
        Returns:
            Dict with creation result
        """
        try:
            # Get new messages (after last summary)
            previous_total = existing_summary['total_messages_in_session']
            new_message_count = message_count - previous_total
            
            if new_message_count <= 0:
                return {
                    'created': False,
                    'reason': 'no_new_messages'
                }

            # Get new messages (async)
            new_messages = await self._get_messages_for_summary_async(
                session_id=session_id,
                start_index=previous_total,
                count=new_message_count
            )

            if not new_messages:
                return {
                    'created': False,
                    'error': 'Could not retrieve new messages'
                }

            self.logger.debug(
                f"[SUMMARY] Generating recursive summary: "
                f"v{existing_summary['version']} + {len(new_messages)} new messages"
            )

            # Generate recursive summary
            summary_result = await self.summary_service.generate_recursive_summary(
                previous_summary=existing_summary['summary_text'],
                new_messages=new_messages,
                model_name=model_name,
                provider_type=provider_type
            )

            if not summary_result.get('summary_text'):
                return {
                    'created': False,
                    'error': 'Recursive summary generation failed'
                }

            # Save new version (async)
            new_version = existing_summary['version'] + 1

            summary_id = await self.summary_repo.create_summary_async(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                summary_text=summary_result['summary_text'],
                version=new_version,
                token_count=summary_result['token_count'],
                message_count=len(new_messages),
                total_messages=message_count,
                model_name=model_name or self.summary_service.model_name,
                messages_start_id=new_messages[0].get('id') if new_messages else None,
                messages_end_id=new_messages[-1].get('id') if new_messages else None,
                previous_summary_tokens=existing_summary['token_count']
            )
            
            if summary_id:
                return {
                    'created': True,
                    'summary_id': summary_id,
                    'version': new_version,
                    'previous_version': existing_summary['version'],
                    'token_count': summary_result['token_count'],
                    'messages_summarized': len(new_messages),
                    'total_messages': message_count
                }
            else:
                return {
                    'created': False,
                    'error': 'Failed to save recursive summary'
                }
                
        except Exception as e:
            self.logger.error(f"[SUMMARY] Error creating recursive summary: {e}")
            return {
                'created': False,
                'error': str(e)
            }
    
    
    def _get_session_message_count(self, session_id: str) -> int:
        """Get total message count for a session (sync - legacy)"""
        try:
            history = self.chat_service.get_chat_history(
                session_id=session_id,
                limit=1000  # Get large number to count
            )
            return len(history)
        except Exception as e:
            self.logger.error(f"[SUMMARY] Error getting message count: {e}")
            return 0

    async def _get_session_message_count_async(self, session_id: str) -> int:
        """
        Get total message count for a session (async - non-blocking).

        Args:
            session_id: Session identifier

        Returns:
            Number of messages in the session
        """
        try:
            history = await self.chat_service.get_chat_history_async(
                session_id=session_id,
                limit=1000  # Get large number to count
            )
            return len(history)
        except Exception as e:
            self.logger.error(f"[SUMMARY] Error getting message count: {e}")
            return 0

    def _get_messages_for_summary(
        self,
        session_id: str,
        start_index: int,
        count: int
    ) -> List[Dict[str, str]]:
        """
        Get messages for summarization (sync - legacy)

        Args:
            session_id: Session identifier
            start_index: Starting message index
            count: Number of messages to get

        Returns:
            List of message dicts with 'role', 'content', 'id'
        """
        try:
            # Get all messages
            all_messages = self.chat_service.get_chat_history(
                session_id=session_id,
                limit=1000
            )

            # Reverse to chronological order (oldest first)
            all_messages.reverse()

            # Slice to get target range
            target_messages = all_messages[start_index:start_index + count]

            # Convert to format for LLM
            formatted_messages = []
            for msg in target_messages:
                content, role = msg  # (content, role) tuple

                # Map role
                if role == 'user':
                    llm_role = 'user'
                elif role == 'assistant':
                    llm_role = 'assistant'
                else:
                    llm_role = 'user'  # Default

                formatted_messages.append({
                    'role': llm_role,
                    'content': content
                })

            return formatted_messages

        except Exception as e:
            self.logger.error(f"[SUMMARY] Error getting messages for summary: {e}")
            return []

    async def _get_messages_for_summary_async(
        self,
        session_id: str,
        start_index: int,
        count: int
    ) -> List[Dict[str, str]]:
        """
        Get messages for summarization (async - non-blocking).

        Args:
            session_id: Session identifier
            start_index: Starting message index
            count: Number of messages to get

        Returns:
            List of message dicts with 'role', 'content'
        """
        try:
            # Get all messages (async)
            all_messages = await self.chat_service.get_chat_history_async(
                session_id=session_id,
                limit=1000
            )

            # Reverse to chronological order (oldest first)
            all_messages.reverse()

            # Slice to get target range
            target_messages = all_messages[start_index:start_index + count]

            # Convert to format for LLM
            formatted_messages = []
            for msg in target_messages:
                content, role = msg  # (content, role) tuple

                # Map role
                if role == 'user':
                    llm_role = 'user'
                elif role == 'assistant':
                    llm_role = 'assistant'
                else:
                    llm_role = 'user'  # Default

                formatted_messages.append({
                    'role': llm_role,
                    'content': content
                })

            return formatted_messages

        except Exception as e:
            self.logger.error(f"[SUMMARY] Error getting messages for summary: {e}")
            return []
    
    
    def format_summary_for_context(self, summary_text: str) -> str:
        """
        Format summary for inclusion in context window
        
        Args:
            summary_text: Summary content
            
        Returns:
            Formatted summary string
        """
        if not summary_text:
            return ""
        
        formatted = f"""### CONVERSATION SUMMARY (Earlier Messages)

{summary_text}

---
"""
        return formatted
    
    
    async def get_summary_stats(self, session_id: str) -> Dict[str, Any]:
        """
        Get statistics about summaries for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict with summary statistics
        """
        try:
            stats = self.summary_repo.get_summary_stats(session_id)
            return stats
        except Exception as e:
            self.logger.error(f"[SUMMARY] Error getting summary stats: {e}")
            return {
                'has_summary': False,
                'error': str(e)
            }
    
    
    async def delete_session_summaries(self, session_id: str) -> bool:
        """
        Delete all summaries for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Success status
        """
        try:
            return self.summary_repo.delete_session_summaries(session_id)
        except Exception as e:
            self.logger.error(f"[SUMMARY] Error deleting summaries: {e}")
            return False
    
    
    async def force_create_summary(
        self,
        session_id: str,
        user_id: str,
        organization_id: Optional[str] = None,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Force create a summary regardless of thresholds
        Useful for manual triggering or testing
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            organization_id: Organization ID (optional)
            model_name: LLM model (optional)
            provider_type: LLM provider (optional)
            
        Returns:
            Dict with creation result
        """
        try:
            message_count = self._get_session_message_count(session_id)
            
            if message_count < 2:
                return {
                    'created': False,
                    'error': 'Need at least 2 messages to create summary'
                }
            
            existing_summary = self.summary_repo.get_active_summary(session_id)
            
            result = await self._create_new_summary(
                session_id=session_id,
                user_id=user_id,
                organization_id=organization_id,
                message_count=message_count,
                existing_summary=existing_summary,
                model_name=model_name,
                provider_type=provider_type
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"[SUMMARY] Force create failed: {e}")
            return {
                'created': False,
                'error': str(e)
            }