import json
import hashlib
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType


class ConsolidationAction(str, Enum):
    """Actions the consolidation agent can take"""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MERGE = "merge"
    NOOP = "noop"


class MemoryEntry(BaseModel):
    """Represents a single memory entry"""
    model_config = ConfigDict(extra='forbid')
    
    id: str = Field(description="Unique identifier")
    content: str = Field(description="Memory content")
    category: str = Field(default="general", description="Category: portfolio, preference, fact, etc.")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence score")
    source: str = Field(default="conversation", description="Source of memory")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: Optional[str] = Field(default=None)
    access_count: int = Field(default=0, description="How often this memory was accessed")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConsolidationDecision(BaseModel):
    """Decision made by consolidation agent"""
    model_config = ConfigDict(extra='forbid')
    
    action: ConsolidationAction = Field(description="Action to take")
    target_id: Optional[str] = Field(default=None, description="Target memory ID for UPDATE/DELETE/MERGE")
    new_content: Optional[str] = Field(default=None, description="New/updated content")
    category: str = Field(default="general", description="Memory category")
    confidence: float = Field(default=0.8, description="Decision confidence")
    reasoning: str = Field(default="", description="Why this decision")
    merge_ids: List[str] = Field(default_factory=list, description="IDs to merge (for MERGE action)")


class ConsolidationResult(BaseModel):
    """Result of consolidation process"""
    model_config = ConfigDict(extra='forbid')
    
    success: bool = Field(default=True)
    action_taken: ConsolidationAction = Field(default=ConsolidationAction.NOOP)
    affected_ids: List[str] = Field(default_factory=list)
    message: str = Field(default="")
    error: Optional[str] = Field(default=None)


# ============================================================================
# Consolidation Agent
# ============================================================================

class MemoryConsolidationAgent(LoggerMixin):
    """
    Intelligent Memory Consolidation Agent
    
    Compares new information with existing memories and decides:
    - ADD: Information is new and valuable
    - UPDATE: Information updates existing memory
    - DELETE: Information contradicts/obsoletes existing
    - MERGE: Information should be combined with existing
    - NOOP: Information already exists or not worth storing
    """
    
    # Similarity threshold for duplicate detection
    SIMILARITY_THRESHOLD = 0.75
    
    # Categories that support consolidation
    CONSOLIDATABLE_CATEGORIES = [
        "portfolio",
        "watchlist", 
        "preference",
        "risk_tolerance",
        "trading_style",
        "investment_goal",
        "personal_info"
    ]
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI
    ):
        """
        Initialize Consolidation Agent
        
        Args:
            model_name: LLM model for semantic analysis
            provider_type: LLM provider
        """
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()
        self.model_name = model_name
        self.provider_type = provider_type
        
        self.logger.info("[MEMORY-CONSOLIDATION] Initialized successfully")
    

    async def consolidate(
        self,
        new_info: str,
        existing_memories: List[Dict[str, Any]],
        category: str = "general",
        user_context: Optional[Dict] = None
    ) -> ConsolidationDecision:
        """
        Main consolidation logic with ENHANCED LOGGING
        """

        self.logger.info(f"[MEMORY-INPUT] Analyzing '{category}': {new_info}")
        
        # Quick validation
        if not new_info or len(new_info.strip()) < 3:
            return ConsolidationDecision(action=ConsolidationAction.NOOP, reasoning="Input empty")
        
        # Step 1: Check exact duplicate
        exact_match = self._find_exact_duplicate(new_info, existing_memories)
        if exact_match:
            self.logger.info(f"⏭ [MEMORY-SKIP] Exact duplicate found (ID: {exact_match.get('id')})")
            return ConsolidationDecision(
                action=ConsolidationAction.NOOP,
                target_id=exact_match.get('id'),
                reasoning="Exact duplicate"
            )
        
        # Step 2: Find similar
        similar_memories = self._find_similar_memories(new_info, existing_memories, category)
        
        # Step 3: Semantic Decision
        decision = None
        if similar_memories:
            self.logger.info(f"[MEMORY-CHECK] Found {len(similar_memories)} similar entries. Thinking...")
            decision = await self._semantic_consolidation(
                new_info=new_info,
                similar_memories=similar_memories,
                category=category,
                user_context=user_context
            )
        else:
            # Step 4: Add New
            decision = ConsolidationDecision(
                action=ConsolidationAction.ADD,
                new_content=new_info,
                category=category,
                confidence=0.9,
                reasoning="Information is new (no overlap found)"
            )
        
        # Step 5: Log
        self._log_decision(decision, new_info)
        
        return decision

    def _log_decision(self, decision: ConsolidationDecision, new_info: str):
        """Helper to print beautiful logs for memory actions"""
        action = decision.action
        reason = decision.reasoning
        conf = f"{decision.confidence:.2f}"
        
        if action == ConsolidationAction.ADD:
            self.logger.info(
                f"[MEMORY-ADD] (Conf: {conf}) \n"
                f"   New: '{new_info}'\n"
                f"   Reason: {reason}"
            )
        elif action == ConsolidationAction.UPDATE:
            self.logger.info(
                f"[MEMORY-UPDATE] Target ID: {decision.target_id} (Conf: {conf}) \n"
                f"   Change: '{new_info}'\n"
                f"   Reason: {reason}"
            )
        elif action == ConsolidationAction.DELETE:
            self.logger.info(
                f"[MEMORY-DELETE] Target ID: {decision.target_id} (Conf: {conf}) \n"
                f"   Obsolete Info: '{new_info}'\n"
                f"   Reason: {reason}"
            )
        elif action == ConsolidationAction.MERGE:
            self.logger.info(
                f"[MEMORY-MERGE] IDs: {decision.merge_ids} (Conf: {conf}) \n"
                f"   Reason: {reason}"
            )
        elif action == ConsolidationAction.NOOP:
            self.logger.info(f"[MEMORY-NOOP] No action needed. Reason: {reason}")

    def _find_exact_duplicate(
        self,
        new_info: str,
        existing_memories: List[Dict[str, Any]]
    ) -> Optional[Dict]:
        """
        Fast exact duplicate detection
        """
        # Normalize new info
        normalized_new = self._normalize_text(new_info)
        new_hash = hashlib.md5(normalized_new.encode()).hexdigest()
        
        for mem in existing_memories:
            content = mem.get('content', '')
            normalized_existing = self._normalize_text(content)
            existing_hash = hashlib.md5(normalized_existing.encode()).hexdigest()
            
            # 1. Hash Comparison
            if new_hash == existing_hash:
                return mem
            
            # 2. Compare strings (Only check 'in' if lengths are equal)
            len_new = len(normalized_new)
            len_exist = len(normalized_existing)
            
            if len_new > 0 and len_exist > 0:
                # Only considered duplicates if the length difference is no more than 10%
                ratio = min(len_new, len_exist) / max(len_new, len_exist)
                
                if ratio > 0.9: 
                    if normalized_new in normalized_existing or normalized_existing in normalized_new:
                        return mem
        
        return None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text:
            return ""
        # Lowercase, remove extra spaces, basic normalization
        return ' '.join(text.lower().split())
    
    # ========================================================================
    # Similarity Detection
    # ========================================================================
    
    def _find_similar_memories(
        self,
        new_info: str,
        existing_memories: List[Dict[str, Any]],
        category: str
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar memories using keyword overlap
        
        Args:
            new_info: New information
            existing_memories: Existing memory entries
            category: Category to filter by
            
        Returns:
            List of similar memories with similarity scores
        """
        similar = []
        new_keywords = self._extract_keywords(new_info)
        
        if not new_keywords:
            return similar
        
        for mem in existing_memories:
            content = mem.get('content', '')
            mem_category = mem.get('category', 'general')
            
            # Category match bonus
            category_match = mem_category == category
            
            # Calculate keyword overlap
            mem_keywords = self._extract_keywords(content)
            if not mem_keywords:
                continue
            
            overlap = len(new_keywords & mem_keywords)
            total = len(new_keywords | mem_keywords)
            
            if total > 0:
                similarity = overlap / total
                
                # Boost similarity if same category
                if category_match:
                    similarity = min(1.0, similarity * 1.2)
                
                if similarity >= self.SIMILARITY_THRESHOLD:
                    similar.append({
                        **mem,
                        '_similarity': similarity
                    })
        
        # Sort by similarity descending
        similar.sort(key=lambda x: x.get('_similarity', 0), reverse=True)
        
        return similar[:5]  # Top 5 most similar
    
    def _extract_keywords(self, text: str) -> set:
        """
        Extract meaningful keywords from text
        
        Args:
            text: Input text
            
        Returns:
            Set of keywords
        """
        if not text:
            return set()
        
        # Simple keyword extraction (stopword removal)
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
            'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
            'if', 'or', 'because', 'until', 'while', 'about', 'against',
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
            'am', 'user', 'like', 'want', 'prefer', 'interested'
        }
        
        words = text.lower().split()
        keywords = set()
        
        for word in words:
            # Clean word
            clean = ''.join(c for c in word if c.isalnum())
            if clean and len(clean) > 2 and clean not in stopwords:
                keywords.add(clean)
        
        return keywords
    
    # ========================================================================
    # Semantic Consolidation
    # ========================================================================
    
    async def _semantic_consolidation(
        self,
        new_info: str,
        similar_memories: List[Dict[str, Any]],
        category: str,
        user_context: Optional[Dict] = None
    ) -> ConsolidationDecision:
        """
        Use LLM to decide consolidation action based on semantic understanding
        
        Args:
            new_info: New information
            similar_memories: Similar existing memories
            category: Category of new info
            user_context: Optional user context
            
        Returns:
            ConsolidationDecision
        """
        
        # Build context for LLM
        existing_context = "\n".join([
            f"- ID: {m.get('id', 'unknown')}, Content: {m.get('content', '')[:200]}, "
            f"Category: {m.get('category', 'general')}, Similarity: {m.get('_similarity', 0):.2f}"
            for m in similar_memories[:3]
        ])
        
        prompt = f"""You are a memory consolidation agent. Analyze the NEW information and decide the best action.

NEW INFORMATION:
"{new_info}"

CATEGORY: {category}

EXISTING SIMILAR MEMORIES:
{existing_context}

AVAILABLE ACTIONS:
1. ADD - New information is unique and should be added
2. UPDATE - New information updates/corrects an existing memory (specify which ID)
3. DELETE - New information makes existing memory obsolete (specify which ID)
4. MERGE - New information should be combined with existing (specify which IDs)
5. NOOP - Information already exists or is not worth storing

DECISION CRITERIA:
- ADD: Info is new, relevant, and not redundant
- UPDATE: Info corrects/updates existing (e.g., new portfolio position replaces old)
- DELETE: Info contradicts existing (e.g., "I sold AAPL" when memory says "I hold AAPL")
- MERGE: Multiple memories about same topic should be combined
- NOOP: Info is duplicate, trivial, or already covered

Respond with ONLY valid JSON:
{{
    "action": "add|update|delete|merge|noop",
    "target_id": "ID of existing memory (for update/delete) or null",
    "merge_ids": ["list", "of", "IDs"] (for merge) or [],
    "new_content": "Final content to store (for add/update/merge)" or null,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation"
}}"""

        try:
            response = await self.llm_provider.generate_response(
                messages=[{"role": "user", "content": prompt}],
                model_name=self.model_name,
                provider_type=self.provider_type,
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse response
            response_text = response.get('content', '') if isinstance(response, dict) else str(response)
            
            # Extract JSON
            decision_data = self._parse_json_response(response_text)
            
            if decision_data:
                return ConsolidationDecision(
                    action=ConsolidationAction(decision_data.get('action', 'noop')),
                    target_id=decision_data.get('target_id'),
                    new_content=decision_data.get('new_content'),
                    category=category,
                    confidence=float(decision_data.get('confidence', 0.7)),
                    reasoning=decision_data.get('reasoning', ''),
                    merge_ids=decision_data.get('merge_ids', [])
                )
            
        except Exception as e:
            self.logger.error(f"[CONSOLIDATE] LLM error: {e}")
        
        # Fallback: If similar and can't decide, default to UPDATE most similar
        if similar_memories:
            most_similar = similar_memories[0]
            return ConsolidationDecision(
                action=ConsolidationAction.UPDATE,
                target_id=most_similar.get('id'),
                new_content=new_info,
                category=category,
                confidence=0.6,
                reasoning="Fallback: Updating most similar memory"
            )
        
        # Final fallback: ADD
        return ConsolidationDecision(
            action=ConsolidationAction.ADD,
            new_content=new_info,
            category=category,
            confidence=0.5,
            reasoning="Fallback: Adding as new"
        )
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON from LLM response"""
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    # ========================================================================
    # Batch Consolidation
    # ========================================================================
    
    async def consolidate_batch(
        self,
        new_items: List[Dict[str, str]],
        existing_memories: List[Dict[str, Any]],
        user_context: Optional[Dict] = None
    ) -> List[ConsolidationDecision]:
        """
        Consolidate multiple items in batch
        
        Args:
            new_items: List of {content, category} dicts
            existing_memories: Existing memories
            user_context: Optional user context
            
        Returns:
            List of decisions
        """
        decisions = []
        
        # Process sequentially to avoid conflicts
        current_memories = existing_memories.copy()
        
        for item in new_items:
            content = item.get('content', '')
            category = item.get('category', 'general')
            
            decision = await self.consolidate(
                new_info=content,
                existing_memories=current_memories,
                category=category,
                user_context=user_context
            )
            
            decisions.append(decision)
            
            # Update working memory set based on decision
            if decision.action == ConsolidationAction.ADD:
                # Add new entry to working set
                current_memories.append({
                    'id': hashlib.md5(content.encode()).hexdigest()[:8],
                    'content': decision.new_content or content,
                    'category': category
                })
            elif decision.action == ConsolidationAction.DELETE:
                # Remove deleted entry from working set
                current_memories = [
                    m for m in current_memories 
                    if m.get('id') != decision.target_id
                ]
        
        return decisions
    
    # ========================================================================
    # Relevance Decay
    # ========================================================================
    
    def check_relevance_decay(
        self,
        memory: Dict[str, Any],
        current_time: Optional[datetime] = None,
        decay_days: int = 90
    ) -> bool:
        """
        Check if a memory has decayed in relevance
        
        Args:
            memory: Memory entry
            current_time: Current timestamp (default: now)
            decay_days: Days after which memory may be obsolete
            
        Returns:
            True if memory should be considered for deletion
        """
        current = current_time or datetime.now()
        
        # Get memory timestamps
        created_str = memory.get('created_at', '')
        updated_str = memory.get('updated_at', '')
        access_count = memory.get('access_count', 0)
        
        try:
            # Use updated_at if available, else created_at
            last_active_str = updated_str or created_str
            if last_active_str:
                last_active = datetime.fromisoformat(last_active_str.replace('Z', '+00:00'))
                days_since = (current - last_active.replace(tzinfo=None)).days
                
                # High access count reduces decay
                effective_days = decay_days + (access_count * 10)
                
                if days_since > effective_days:
                    return True
        except Exception as e:
            self.logger.warning(f"Error checking decay: {e}")
        
        return False
    
    async def cleanup_decayed_memories(
        self,
        memories: List[Dict[str, Any]],
        decay_days: int = 90
    ) -> List[str]:
        """
        Identify memories that should be cleaned up due to decay
        
        Args:
            memories: List of memory entries
            decay_days: Threshold for decay
            
        Returns:
            List of memory IDs to delete
        """
        to_delete = []
        
        for mem in memories:
            if self.check_relevance_decay(mem, decay_days=decay_days):
                mem_id = mem.get('id')
                if mem_id:
                    to_delete.append(mem_id)
                    self.logger.info(
                        f"[DECAY] Memory {mem_id} marked for cleanup"
                    )
        
        return to_delete


# ============================================================================
# Factory Function
# ============================================================================

def create_consolidation_agent(
    model_name: str = "gpt-4.1-nano",
    provider_type: str = ProviderType.OPENAI
) -> MemoryConsolidationAgent:
    """
    Create a configured MemoryConsolidationAgent instance
    
    Args:
        model_name: LLM model
        provider_type: LLM provider
        
    Returns:
        Configured agent instance
    """
    return MemoryConsolidationAgent(
        model_name=model_name,
        provider_type=provider_type
    )