# File: src/database/repository/procedural_memory_repository.py
"""
Procedural Memory Repository - Database operations for learned patterns

Handles CRUD operations for:
- Tool sequence patterns (successful workflows)
- Error avoidance patterns 
- Query mapping patterns

Uses PostgreSQL with SQLAlchemy ORM following existing codebase patterns.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select, update, delete, func, and_, or_, desc
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.utils.logger.custom_logging import LoggerMixin
from src.database.models.procedural_memory import ProceduralMemoryPattern


class ProceduralMemoryRepository(LoggerMixin):
    """
    Repository for Procedural Memory Pattern operations
    
    Features:
    - Upsert patterns (ON CONFLICT DO UPDATE)
    - Fast lookup with indexes
    - Transaction support
    - Bulk operations
    """
    
    def __init__(self, session: Session):
        """
        Initialize repository with SQLAlchemy session
        
        Args:
            session: SQLAlchemy Session
        """
        super().__init__()
        self.session = session
    
    # ========================================================================
    # CREATE / UPSERT Operations
    # ========================================================================
    
    def upsert_pattern(self, pattern_data: Dict[str, Any]) -> Optional[str]:
        """
        Insert or update a pattern using PostgreSQL upsert
        
        Args:
            pattern_data: Dict with pattern fields including 'pattern_hash'
            
        Returns:
            Pattern ID (UUID string) if successful
        """
        try:
            pattern_hash = pattern_data.get('pattern_hash')
            if not pattern_hash:
                self.logger.error("pattern_hash is required for upsert")
                return None
            
            # Build insert statement
            stmt = insert(ProceduralMemoryPattern).values(**pattern_data)
            
            # Define columns to update on conflict
            update_columns = {
                'success_count': stmt.excluded.success_count,
                'failure_count': stmt.excluded.failure_count,
                'avg_execution_time_ms': stmt.excluded.avg_execution_time_ms,
                'occurrence_count': stmt.excluded.occurrence_count,
                'usage_count': stmt.excluded.usage_count,
                'last_used': stmt.excluded.last_used,
                'updated_at': datetime.utcnow(),
                'symbols': stmt.excluded.symbols,
                'satisfaction_score': stmt.excluded.satisfaction_score
            }
            
            # Filter out None values
            update_columns = {k: v for k, v in update_columns.items() if v is not None}
            
            # On conflict, update existing
            stmt = stmt.on_conflict_do_update(
                index_elements=['pattern_hash'],
                set_=update_columns
            ).returning(ProceduralMemoryPattern.id)
            
            result = self.session.execute(stmt)
            self.session.commit()
            
            row = result.fetchone()
            if row:
                pattern_id = str(row[0])
                self.logger.info(f"[UPSERT] Pattern {pattern_hash[:8]}... -> {pattern_id[:8]}...")
                return pattern_id
            
            return None
            
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error upserting pattern: {e}")
            return None
    
    def create_pattern(self, pattern: ProceduralMemoryPattern) -> Optional[str]:
        """
        Create a new pattern
        
        Args:
            pattern: ProceduralMemoryPattern instance
            
        Returns:
            Pattern ID if successful
        """
        try:
            self.session.add(pattern)
            self.session.commit()
            self.logger.info(f"[CREATE] Pattern {pattern.pattern_hash[:8]}... created")
            return str(pattern.id)
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error creating pattern: {e}")
            return None
    
    # ========================================================================
    # READ Operations
    # ========================================================================
    
    def get_by_hash(self, pattern_hash: str) -> Optional[ProceduralMemoryPattern]:
        """
        Get pattern by hash
        
        Args:
            pattern_hash: Unique pattern hash
            
        Returns:
            Pattern or None
        """
        try:
            stmt = select(ProceduralMemoryPattern).where(
                ProceduralMemoryPattern.pattern_hash == pattern_hash
            )
            result = self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting pattern by hash: {e}")
            return None
    
    def get_by_id(self, pattern_id: str) -> Optional[ProceduralMemoryPattern]:
        """
        Get pattern by ID
        
        Args:
            pattern_id: UUID string
            
        Returns:
            Pattern or None
        """
        try:
            stmt = select(ProceduralMemoryPattern).where(
                ProceduralMemoryPattern.id == pattern_id
            )
            result = self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            self.logger.error(f"Error getting pattern by ID: {e}")
            return None
    
    def get_tool_sequences_by_intent(
        self,
        query_intent: str,
        user_id: Optional[str] = None,
        min_success_rate: float = 0.6,
        min_usage: int = 2,
        limit: int = 10
    ) -> List[ProceduralMemoryPattern]:
        """
        Get tool sequence patterns matching query intent
        
        Args:
            query_intent: Query intent to match
            user_id: Optional user ID filter
            min_success_rate: Minimum required success rate
            min_usage: Minimum usage count
            limit: Max results
            
        Returns:
            List of matching patterns
        """
        try:
            # Build conditions
            conditions = [
                ProceduralMemoryPattern.pattern_type == 'tool_sequence',
                ProceduralMemoryPattern.is_active == True,
                ProceduralMemoryPattern.query_intent.ilike(f'%{query_intent}%')
            ]
            
            # User scope: include global (user_id IS NULL) and user-specific
            if user_id:
                conditions.append(
                    or_(
                        ProceduralMemoryPattern.user_id == None,
                        ProceduralMemoryPattern.user_id == user_id
                    )
                )
            else:
                conditions.append(ProceduralMemoryPattern.user_id == None)
            
            stmt = select(ProceduralMemoryPattern).where(
                and_(*conditions)
            ).order_by(
                desc(ProceduralMemoryPattern.success_count),
                desc(ProceduralMemoryPattern.last_used)
            ).limit(limit)
            
            result = self.session.execute(stmt)
            patterns = result.scalars().all()
            
            # Filter by success rate in Python (more flexible)
            filtered = []
            for p in patterns:
                total = p.success_count + p.failure_count
                if total >= min_usage:
                    rate = p.success_count / total
                    if rate >= min_success_rate:
                        filtered.append(p)
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Error getting tool sequences: {e}")
            return []
    
    def get_error_patterns_for_tool(
        self,
        tool_name: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[ProceduralMemoryPattern]:
        """
        Get error avoidance patterns for a tool
        
        Args:
            tool_name: Tool to check
            user_id: Optional user filter
            limit: Max results
            
        Returns:
            List of error patterns
        """
        try:
            conditions = [
                ProceduralMemoryPattern.pattern_type == 'error_avoidance',
                ProceduralMemoryPattern.is_active == True,
                ProceduralMemoryPattern.tool_name == tool_name
            ]
            
            if user_id:
                conditions.append(
                    or_(
                        ProceduralMemoryPattern.user_id == None,
                        ProceduralMemoryPattern.user_id == user_id
                    )
                )
            
            stmt = select(ProceduralMemoryPattern).where(
                and_(*conditions)
            ).order_by(
                desc(ProceduralMemoryPattern.occurrence_count)
            ).limit(limit)
            
            result = self.session.execute(stmt)
            return result.scalars().all()
            
        except Exception as e:
            self.logger.error(f"Error getting error patterns: {e}")
            return []
    
    def get_query_mappings(
        self,
        keywords: List[str],
        language: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[ProceduralMemoryPattern]:
        """
        Get query mapping patterns matching keywords
        
        Args:
            keywords: Keywords to match
            language: Optional language filter
            user_id: Optional user filter
            limit: Max results
            
        Returns:
            List of query mappings
        """
        try:
            conditions = [
                ProceduralMemoryPattern.pattern_type == 'query_mapping',
                ProceduralMemoryPattern.is_active == True
            ]
            
            # Language filter
            if language:
                conditions.append(
                    or_(
                        ProceduralMemoryPattern.language == language,
                        ProceduralMemoryPattern.language == 'auto'
                    )
                )
            
            # User scope
            if user_id:
                conditions.append(
                    or_(
                        ProceduralMemoryPattern.user_id == None,
                        ProceduralMemoryPattern.user_id == user_id
                    )
                )
            
            stmt = select(ProceduralMemoryPattern).where(
                and_(*conditions)
            ).order_by(
                desc(ProceduralMemoryPattern.usage_count),
                desc(ProceduralMemoryPattern.satisfaction_score)
            ).limit(limit * 3)  # Get more, filter in Python
            
            result = self.session.execute(stmt)
            patterns = result.scalars().all()
            
            # Score by keyword match
            scored = []
            for p in patterns:
                if not p.query_keywords:
                    continue
                matches = sum(1 for kw in keywords if kw.lower() in [qk.lower() for qk in p.query_keywords])
                if matches > 0:
                    scored.append((p, matches))
            
            # Sort by matches descending
            scored.sort(key=lambda x: x[1], reverse=True)
            
            return [p for p, _ in scored[:limit]]
            
        except Exception as e:
            self.logger.error(f"Error getting query mappings: {e}")
            return []
    
    def get_top_patterns(
        self,
        pattern_type: str,
        limit: int = 100,
        user_id: Optional[str] = None
    ) -> List[ProceduralMemoryPattern]:
        """
        Get top patterns by type for cache warming
        
        Args:
            pattern_type: Type of pattern
            limit: Max results
            user_id: Optional user filter
            
        Returns:
            List of top patterns
        """
        try:
            conditions = [
                ProceduralMemoryPattern.pattern_type == pattern_type,
                ProceduralMemoryPattern.is_active == True
            ]
            
            if user_id:
                conditions.append(
                    or_(
                        ProceduralMemoryPattern.user_id == None,
                        ProceduralMemoryPattern.user_id == user_id
                    )
                )
            
            # Order by relevance metric based on type
            if pattern_type == 'tool_sequence':
                order_col = desc(ProceduralMemoryPattern.success_count)
            elif pattern_type == 'error_avoidance':
                order_col = desc(ProceduralMemoryPattern.occurrence_count)
            else:
                order_col = desc(ProceduralMemoryPattern.usage_count)
            
            stmt = select(ProceduralMemoryPattern).where(
                and_(*conditions)
            ).order_by(order_col).limit(limit)
            
            result = self.session.execute(stmt)
            return result.scalars().all()
            
        except Exception as e:
            self.logger.error(f"Error getting top patterns: {e}")
            return []
    
    # ========================================================================
    # UPDATE Operations
    # ========================================================================
    
    def increment_success(self, pattern_hash: str) -> bool:
        """Increment success count for a pattern"""
        try:
            stmt = update(ProceduralMemoryPattern).where(
                ProceduralMemoryPattern.pattern_hash == pattern_hash
            ).values(
                success_count=ProceduralMemoryPattern.success_count + 1,
                last_used=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            self.session.execute(stmt)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error incrementing success: {e}")
            return False
    
    def increment_failure(self, pattern_hash: str) -> bool:
        """Increment failure count for a pattern"""
        try:
            stmt = update(ProceduralMemoryPattern).where(
                ProceduralMemoryPattern.pattern_hash == pattern_hash
            ).values(
                failure_count=ProceduralMemoryPattern.failure_count + 1,
                updated_at=datetime.utcnow()
            )
            self.session.execute(stmt)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error incrementing failure: {e}")
            return False
    
    def update_execution_time(
        self,
        pattern_hash: str,
        new_time_ms: float
    ) -> bool:
        """Update average execution time"""
        try:
            # Get current pattern
            pattern = self.get_by_hash(pattern_hash)
            if not pattern:
                return False
            
            # Calculate new average
            total_executions = pattern.success_count
            if total_executions > 0:
                old_total = pattern.avg_execution_time_ms * (total_executions - 1)
                new_avg = (old_total + new_time_ms) / total_executions
            else:
                new_avg = new_time_ms
            
            stmt = update(ProceduralMemoryPattern).where(
                ProceduralMemoryPattern.pattern_hash == pattern_hash
            ).values(
                avg_execution_time_ms=new_avg,
                updated_at=datetime.utcnow()
            )
            self.session.execute(stmt)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error updating execution time: {e}")
            return False
    
    def add_symbols(self, pattern_hash: str, new_symbols: List[str]) -> bool:
        """Add symbols to a pattern's symbol list"""
        try:
            pattern = self.get_by_hash(pattern_hash)
            if not pattern:
                return False
            
            existing = set(pattern.symbols or [])
            existing.update(new_symbols)
            
            stmt = update(ProceduralMemoryPattern).where(
                ProceduralMemoryPattern.pattern_hash == pattern_hash
            ).values(
                symbols=list(existing),
                updated_at=datetime.utcnow()
            )
            self.session.execute(stmt)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error adding symbols: {e}")
            return False
    
    # ========================================================================
    # DELETE Operations
    # ========================================================================
    
    def soft_delete(self, pattern_hash: str) -> bool:
        """Soft delete a pattern (set is_active=False)"""
        try:
            stmt = update(ProceduralMemoryPattern).where(
                ProceduralMemoryPattern.pattern_hash == pattern_hash
            ).values(
                is_active=False,
                updated_at=datetime.utcnow()
            )
            self.session.execute(stmt)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error soft deleting pattern: {e}")
            return False
    
    def hard_delete(self, pattern_hash: str) -> bool:
        """Hard delete a pattern"""
        try:
            stmt = delete(ProceduralMemoryPattern).where(
                ProceduralMemoryPattern.pattern_hash == pattern_hash
            )
            self.session.execute(stmt)
            self.session.commit()
            return True
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error hard deleting pattern: {e}")
            return False
    
    def cleanup_old_patterns(self, days_threshold: int = 90) -> int:
        """Delete patterns not used in X days"""
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=days_threshold)
            
            stmt = delete(ProceduralMemoryPattern).where(
                and_(
                    ProceduralMemoryPattern.last_used < cutoff,
                    ProceduralMemoryPattern.is_active == True
                )
            )
            result = self.session.execute(stmt)
            self.session.commit()
            
            deleted = result.rowcount
            self.logger.info(f"[CLEANUP] Deleted {deleted} old patterns")
            return deleted
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error cleaning up patterns: {e}")
            return 0
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about patterns"""
        try:
            conditions = [ProceduralMemoryPattern.is_active == True]
            if user_id:
                conditions.append(
                    or_(
                        ProceduralMemoryPattern.user_id == None,
                        ProceduralMemoryPattern.user_id == user_id
                    )
                )
            
            # Count by type
            stmt = select(
                ProceduralMemoryPattern.pattern_type,
                func.count(ProceduralMemoryPattern.id)
            ).where(
                and_(*conditions)
            ).group_by(
                ProceduralMemoryPattern.pattern_type
            )
            
            result = self.session.execute(stmt)
            counts = {row[0]: row[1] for row in result}
            
            return {
                'tool_sequences': counts.get('tool_sequence', 0),
                'error_patterns': counts.get('error_avoidance', 0),
                'query_mappings': counts.get('query_mapping', 0),
                'total': sum(counts.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}