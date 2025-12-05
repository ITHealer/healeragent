"""
Session Repository - Database operations for chat sessions
Works with existing ChatSessions schema
Fixed: Use sender_role field instead of role (Messages schema)
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from uuid import UUID
from sqlalchemy import and_, or_, desc, func
from sqlalchemy.orm import Session

from src.database.models.schemas import ChatSessions, Messages
from src.database.session_manager import SessionManager
from src.utils.logger.custom_logging import LoggerMixin


class SessionRepository(LoggerMixin):
    """
    Repository for session-related database operations
    Uses existing ChatSessions and Messages schemas
    """
    
    def __init__(self):
        super().__init__()
        self.session_manager = SessionManager()
    
    
    async def get_session_messages(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get messages from a session
        
        Args:
            session_id: Session UUID
            limit: Max messages to return
            offset: Skip first N messages
            
        Returns:
            List of message dicts
        """
        try:
            with self.session_manager.create_session() as db:
                messages = db.query(Messages).filter(
                    Messages.session_id == session_id
                ).order_by(
                    Messages.created_at.desc()
                ).limit(limit).offset(offset).all()
                
                # Convert to dict format
                result = []
                for msg in messages:
                    result.append({
                        'id': str(msg.id),
                        'session_id': str(msg.session_id),
                        'role': msg.sender_role if msg.sender_role else 'user',  # Fixed: use sender_role
                        'content': msg.content,
                        'created_at': msg.created_at.isoformat() if msg.created_at else None,
                        'created_by': msg.created_by
                    })
                
                self.logger.info(
                    f"[SESSION REPO] Retrieved {len(result)} messages "
                    f"from session {session_id[:8]}..."
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting session messages: {e}")
            return []
    
    
    async def get_session_messages_by_date_range(
        self,
        session_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get messages from session within date range
        
        Args:
            session_id: Session UUID
            start_date: Start of date range
            end_date: End of date range
            limit: Max messages
            
        Returns:
            List of message dicts
        """
        try:
            with self.session_manager.create_session() as db:
                messages = db.query(Messages).filter(
                    and_(
                        Messages.session_id == session_id,
                        Messages.created_at >= start_date,
                        Messages.created_at <= end_date
                    )
                ).order_by(
                    Messages.created_at.desc()
                ).limit(limit).all()
                
                # Convert to dict
                result = []
                for msg in messages:
                    result.append({
                        'id': str(msg.id),
                        'session_id': str(msg.session_id),
                        'role': msg.sender_role if msg.sender_role else 'user',  # Fixed: use sender_role
                        'content': msg.content,
                        'created_at': msg.created_at.isoformat() if msg.created_at else None,
                        'created_by': msg.created_by
                    })
                
                self.logger.info(
                    f"[SESSION REPO] Found {len(result)} messages "
                    f"between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting messages by date: {e}")
            return []
    
    
    async def search_messages_by_content(
        self,
        session_id: str,
        search_terms: List[str],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search messages containing specific terms
        
        Args:
            session_id: Session UUID
            search_terms: List of terms to search for
            limit: Max results
            
        Returns:
            List of matching messages
        """
        try:
            with self.session_manager.create_session() as db:
                # Build search condition
                search_conditions = []
                for term in search_terms:
                    search_conditions.append(
                        func.lower(Messages.content).contains(term.lower())
                    )
                
                messages = db.query(Messages).filter(
                    and_(
                        Messages.session_id == session_id,
                        or_(*search_conditions)
                    )
                ).order_by(
                    Messages.created_at.desc()
                ).limit(limit).all()
                
                # Convert to dict
                result = []
                for msg in messages:
                    result.append({
                        'id': str(msg.id),
                        'session_id': str(msg.session_id),
                        'role': msg.sender_role if msg.sender_role else 'user',  # Fixed: use sender_role
                        'content': msg.content,
                        'created_at': msg.created_at.isoformat() if msg.created_at else None,
                        'created_by': msg.created_by
                    })
                
                self.logger.info(
                    f"[SESSION REPO] Search found {len(result)} messages "
                    f"matching terms: {search_terms}"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error searching messages: {e}")
            return []
    
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session metadata
        
        Args:
            session_id: Session UUID
            
        Returns:
            Session info dict or None
        """
        try:
            with self.session_manager.create_session() as db:
                session = db.query(ChatSessions).filter(
                    ChatSessions.id == session_id
                ).first()
                
                if session:
                    return {
                        'id': str(session.id),
                        'title': session.title,
                        'user_id': session.user_id,
                        'organization_id': session.organization_id,
                        'start_date': session.start_date.isoformat() if session.start_date else None,
                        'end_date': session.end_date.isoformat() if session.end_date else None,
                        'state': session.state
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting session info: {e}")
            return None
    
    
    async def get_message_count(self, session_id: str) -> int:
        """
        Get total message count for session
        
        Args:
            session_id: Session UUID
            
        Returns:
            Message count
        """
        try:
            with self.session_manager.create_session() as db:
                count = db.query(func.count(Messages.id)).filter(
                    Messages.session_id == session_id
                ).scalar()
                
                return count or 0
                
        except Exception as e:
            self.logger.error(f"Error getting message count: {e}")
            return 0
    
    
    async def get_recent_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get user's recent chat sessions
        
        Args:
            user_id: User ID
            limit: Max sessions to return
            
        Returns:
            List of session info dicts
        """
        try:
            with self.session_manager.create_session() as db:
                sessions = db.query(ChatSessions).filter(
                    ChatSessions.user_id == user_id
                ).order_by(
                    ChatSessions.start_date.desc()
                ).limit(limit).all()
                
                result = []
                for session in sessions:
                    # Get message count for each session
                    msg_count = db.query(func.count(Messages.id)).filter(
                        Messages.session_id == session.id
                    ).scalar()
                    
                    result.append({
                        'id': str(session.id),
                        'title': session.title,
                        'start_date': session.start_date.isoformat() if session.start_date else None,
                        'message_count': msg_count or 0
                    })
                
                self.logger.info(
                    f"[SESSION REPO] Found {len(result)} recent sessions for user {user_id}"
                )
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error getting recent sessions: {e}")
            return []