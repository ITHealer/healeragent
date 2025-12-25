# File: src/database/models/procedural_memory.py
"""
Database models for Procedural Memory System
Stores learned patterns: tool sequences, error avoidance, query mappings

Table: procedural_memory_patterns
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, Text, Boolean, JSON
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY

from src.database.models.base import Base


class ProceduralMemoryPattern(Base):
    """
    Stores learned patterns from successful executions
    
    Pattern Types:
    - tool_sequence: Successful tool chains
    - error_avoidance: Errors to avoid
    - query_mapping: Query type → tools mapping
    """
    __tablename__ = "procedural_memory_patterns"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    
    # Pattern identification
    pattern_hash = Column(String(32), nullable=False, index=True, unique=True)
    pattern_type = Column(String(50), nullable=False, index=True)
    # Values: 'tool_sequence', 'error_avoidance', 'query_mapping'
    
    # Scope
    user_id = Column(String(50), nullable=True, index=True)
    # NULL = global pattern, value = user-specific
    organization_id = Column(String(50), nullable=True, index=True)
    
    # ========================================================================
    # TOOL SEQUENCE Fields (pattern_type = 'tool_sequence')
    # ========================================================================
    query_intent = Column(String(255), nullable=True, index=True)
    tool_sequence = Column(ARRAY(String), nullable=True)
    tool_params_template = Column(JSON, nullable=True)
    symbols = Column(ARRAY(String), nullable=True)
    
    # Performance metrics
    success_count = Column(Integer, default=1, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)
    avg_execution_time_ms = Column(Float, default=0.0, nullable=False)
    
    # ========================================================================
    # ERROR AVOIDANCE Fields (pattern_type = 'error_avoidance')
    # ========================================================================
    tool_name = Column(String(100), nullable=True, index=True)
    error_type = Column(String(255), nullable=True)
    error_context = Column(Text, nullable=True)
    avoidance_strategy = Column(Text, nullable=True)
    alternative_tool = Column(String(100), nullable=True)
    occurrence_count = Column(Integer, default=1, nullable=False)
    last_occurrence = Column(DateTime, nullable=True)
    
    # ========================================================================
    # QUERY MAPPING Fields (pattern_type = 'query_mapping')
    # ========================================================================
    query_keywords = Column(ARRAY(String), nullable=True)
    recommended_tools = Column(ARRAY(String), nullable=True)
    execution_strategy = Column(String(50), nullable=True)
    # Values: 'parallel', 'sequential'
    language = Column(String(10), nullable=True)
    usage_count = Column(Integer, default=1, nullable=False)
    satisfaction_score = Column(Float, default=0.8, nullable=True)
    
    # ========================================================================
    # Common metadata
    # ========================================================================
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime, nullable=True, default=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True, onupdate=datetime.utcnow)
    
    # Additional metadata as JSON
    # metadata = Column(JSON, nullable=True)
    pattern_metadata = Column(JSON, nullable=True)

    
    def __repr__(self):
        return (
            f"<ProceduralMemoryPattern("
            f"id={self.id}, type={self.pattern_type}, "
            f"hash={self.pattern_hash[:8]}..., "
            f"success={self.success_count}, failure={self.failure_count})>"
        )
    
    def get_success_rate(self) -> float:
        """Calculate success rate for tool_sequence patterns"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'id': str(self.id),
            'pattern_hash': self.pattern_hash,
            'pattern_type': self.pattern_type,
            'user_id': self.user_id,
            'query_intent': self.query_intent,
            'tool_sequence': self.tool_sequence,
            'tool_params_template': self.tool_params_template,
            'symbols': self.symbols,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'avg_execution_time_ms': self.avg_execution_time_ms,
            'tool_name': self.tool_name,
            'error_type': self.error_type,
            'error_context': self.error_context,
            'avoidance_strategy': self.avoidance_strategy,
            'alternative_tool': self.alternative_tool,
            'occurrence_count': self.occurrence_count,
            'query_keywords': self.query_keywords,
            'recommended_tools': self.recommended_tools,
            'execution_strategy': self.execution_strategy,
            'language': self.language,
            'usage_count': self.usage_count,
            'satisfaction_score': self.satisfaction_score,
            'is_active': self.is_active,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'success_rate': self.get_success_rate() if self.pattern_type == 'tool_sequence' else None,
            'pattern_metadata': self.pattern_metadata
        }