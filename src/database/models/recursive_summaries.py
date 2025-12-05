"""
Database model for Recursive Summaries
Stores hierarchical conversation summaries for memory system
"""

import uuid
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime

from src.database.models.base import Base


class RecursiveSummaries(Base):
    """
    Recursive summary storage for chat sessions
    
    Each session can have multiple summary versions as conversation grows:
    - Version 1: Summary of messages 1-20
    - Version 2: Summary of (Summary v1 + messages 21-40)
    - Version 3: Summary of (Summary v2 + messages 41-60)
    
    Pattern: New Summary = Previous Summary + New Messages
    """
    __tablename__ = "recursive_summaries"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    # Session identification
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    user_id = Column(String(50), nullable=False, index=True)
    organization_id = Column(String(50), nullable=True, index=True)
    
    # Summary versioning
    version = Column(Integer, nullable=False, default=1)
    
    # Summary content
    summary_text = Column(Text, nullable=False)
    
    # Metadata about summarized content
    messages_start_id = Column(UUID(as_uuid=True), nullable=True)  # First message ID in this summary
    messages_end_id = Column(UUID(as_uuid=True), nullable=True)    # Last message ID in this summary
    message_count = Column(Integer, nullable=False, default=0)     # Total messages summarized
    total_messages_in_session = Column(Integer, nullable=False, default=0)  # Session message count at time of summary
    
    # Token management
    token_count = Column(Integer, nullable=False, default=0)
    previous_summary_tokens = Column(Integer, nullable=True)  # Tokens from previous summary version
    
    # Model used for summarization
    model_name = Column(String(100), nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)  # Current active summary for session
    
    # Timestamps
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<RecursiveSummary(id={self.id}, session={self.session_id}, v{self.version}, tokens={self.token_count})>"