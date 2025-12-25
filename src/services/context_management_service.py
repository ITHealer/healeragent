import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

from src.agents.memory.context_compressor import (
    ContextCompressor,
    CompactionConfig,
    CompactionStrategy,
    CompactionResult,
    ContextStats
)
from src.agents.memory.core_memory import CoreMemory
from src.agents.memory.recursive_summary import RecursiveSummaryManager
from src.helpers.token_counter import TokenCounter


@dataclass
class PreparedContext:
    """Result of context preparation"""
    
    # Messages
    messages: List[Dict[str, str]]
    system_prompt: str
    
    # Memory content
    core_memory: str
    summary: Optional[str]
    
    # Metadata
    total_tokens: int
    was_compacted: bool
    compaction_result: Optional[CompactionResult] = None
    
    # Timing
    preparation_time_ms: int = 0


class ContextManagementService:
    """
    Unified Context Management Service
    
    Responsibilities:
    1. Load and assemble context from all sources
    2. Monitor token usage
    3. Trigger compaction when needed
    4. Provide context statistics
    
    Architecture:
    ┌──────────────────────────────────────────────┐
    │           Context Management                  │
    ├──────────────────────────────────────────────┤
    │  ┌────────┐ ┌─────────┐ ┌──────────────┐    │
    │  │ Core   │ │ Summary │ │   History    │    │
    │  │ Memory │ │ Manager │ │  Compressor  │    │
    │  └───┬────┘ └────┬────┘ └──────┬───────┘    │
    │      │           │             │             │
    │      └───────────┴─────────────┘             │
    │                  │                           │
    │           ┌──────▼──────┐                   │
    │           │  Assembled  │                   │
    │           │   Context   │                   │
    │           └─────────────┘                   │
    └──────────────────────────────────────────────┘
    """
    
    # Context window budget (tokens)
    MAX_CONTEXT_TOKENS = 180000
    SYSTEM_BUDGET = 1000
    CORE_MEMORY_BUDGET = 2000
    SUMMARY_BUDGET = 2000
    HISTORY_BUDGET = 50000
    TOOLS_BUDGET = 5000
    RESPONSE_RESERVE = 4000
    
    def __init__(
        self,
        enable_compaction: bool = True,
        compaction_threshold: int = 100000,
        compaction_strategy: str = "smart_summary"
    ):
        """
        Initialize Context Management Service
        
        Args:
            enable_compaction: Whether to auto-compact
            compaction_threshold: Token threshold for compaction
            compaction_strategy: Strategy for compaction
        """
        self.logger = logging.getLogger(__name__)
        
        self.enable_compaction = enable_compaction
        self.compaction_threshold = compaction_threshold
        
        # Initialize components
        self.core_memory = CoreMemory()
        self.summary_manager = RecursiveSummaryManager()
        self.token_counter = TokenCounter()
        
        # Initialize compressor
        strategy_map = {
            "keep_last_n": CompactionStrategy.KEEP_LAST_N,
            "token_truncation": CompactionStrategy.TOKEN_TRUNCATION,
            "smart_summary": CompactionStrategy.SMART_SUMMARY,
            "recursive_summary": CompactionStrategy.RECURSIVE_SUMMARY
        }
        
        compaction_config = CompactionConfig(
            token_threshold=compaction_threshold,
            strategy=strategy_map.get(compaction_strategy, CompactionStrategy.SMART_SUMMARY)
        )
        
        self.compressor = ContextCompressor(config=compaction_config)
        
        # Stats
        self._last_stats: Optional[ContextStats] = None
        self._compaction_history: List[CompactionResult] = []
        
        self.logger.info(
            f"[CONTEXT MANAGER SERVICE] Initialized with compaction={enable_compaction}, "
            f"threshold={compaction_threshold} tokens"
        )
    
    # ========================================================================
    # MAIN PREPARATION METHOD
    # ========================================================================
    
    async def prepare_context(
        self,
        messages: List[Dict[str, str]],
        session_id: str,
        user_id: str,
        system_prompt: str = "",
        force_compaction: bool = False
    ) -> PreparedContext:
        """
        Prepare context for LLM call
        
        This method:
        1. Loads core memory
        2. Gets recursive summary (if exists)
        3. Checks token usage
        4. Compacts if needed
        5. Returns assembled context
        
        Args:
            messages: Conversation messages
            session_id: Session ID
            user_id: User ID
            system_prompt: Base system prompt
            force_compaction: Force compaction regardless of threshold
            
        Returns:
            PreparedContext with all assembled components
        """
        start_time = datetime.now()
        
        try:
            # ================================================================
            # STEP 1: Load Core Memory
            # ================================================================
            core_memory_data = await self.core_memory.load_core_memory(user_id)
            core_memory_str = self.core_memory.format_for_context(core_memory_data)
            
            self.logger.info(f"[CONTEXT MGR] Loaded core memory for user {user_id}")
            
            # ================================================================
            # STEP 2: Get Recursive Summary
            # ================================================================
            summary = await self.summary_manager.get_active_summary(session_id)
            summary_str = ""
            
            if summary:
                summary_str = self.summary_manager.format_summary_for_context(summary)
                self.logger.info(f"[CONTEXT MGR] Found existing summary for session")
            
            # ================================================================
            # STEP 3: Check Token Usage
            # ================================================================
            needs_compaction, stats = self.compressor.should_compact(
                messages=messages,
                system_prompt=system_prompt + core_memory_str + summary_str
            )
            
            self._last_stats = stats
            
            # ================================================================
            # STEP 4: Compact if Needed
            # ================================================================
            compaction_result = None
            final_messages = messages
            was_compacted = False
            
            if (needs_compaction or force_compaction) and self.enable_compaction:
                self.logger.info(f"[CONTEXT MGR] Triggering compaction...")
                
                # Extract symbols to preserve
                symbols = self.compressor.extract_symbols_from_messages(messages)
                
                compaction_result = await self.compressor.compact(
                    messages=messages,
                    preserve_keywords=symbols,
                    system_prompt=system_prompt
                )
                
                if compaction_result.success:
                    final_messages = compaction_result.preserved_messages or messages
                    was_compacted = True
                    self._compaction_history.append(compaction_result)
                    
                    self.logger.info(
                        f"[CONTEXT MGR] ✅ Compacted: "
                        f"{compaction_result.original_tokens} → {compaction_result.final_tokens} tokens"
                    )
                else:
                    self.logger.warning("[CONTEXT MGR] Compaction failed, using original messages")
            
            # ================================================================
            # STEP 5: Assemble Final Context
            # ================================================================
            # Build enhanced system prompt
            enhanced_system = self._build_enhanced_system_prompt(
                base_prompt=system_prompt,
                core_memory=core_memory_str,
                summary=summary_str
            )
            
            # Calculate final token count
            total_tokens = self._count_total_tokens(
                system=enhanced_system,
                messages=final_messages
            )
            
            # Calculate timing
            prep_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return PreparedContext(
                messages=final_messages,
                system_prompt=enhanced_system,
                core_memory=core_memory_str,
                summary=summary_str,
                total_tokens=total_tokens,
                was_compacted=was_compacted,
                compaction_result=compaction_result,
                preparation_time_ms=prep_time
            )
            
        except Exception as e:
            self.logger.error(f"[CONTEXT MGR] Error preparing context: {e}", exc_info=True)
            
            # Return basic context on error
            return PreparedContext(
                messages=messages,
                system_prompt=system_prompt,
                core_memory="",
                summary=None,
                total_tokens=0,
                was_compacted=False,
                preparation_time_ms=0
            )
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _build_enhanced_system_prompt(
        self,
        base_prompt: str,
        core_memory: str,
        summary: str
    ) -> str:
        """Build enhanced system prompt with memory context"""
        
        parts = [base_prompt]
        
        if core_memory:
            parts.append(f"\n{core_memory}")
        
        if summary:
            parts.append(f"\n{summary}")
        
        return "\n".join(parts)
    
    def _count_total_tokens(
        self,
        system: str,
        messages: List[Dict[str, str]]
    ) -> int:
        """Count total tokens in context"""
        total = self.token_counter.count_tokens(system)
        
        for msg in messages:
            content = msg.get("content", "")
            total += self.token_counter.count_tokens(content)
        
        return total
    
    # ========================================================================
    # MANUAL COMPACTION
    # ========================================================================
    
    async def compact_now(
        self,
        messages: List[Dict[str, str]],
        preserve_keywords: List[str] = None,
        strategy: str = None
    ) -> CompactionResult:
        """
        Manually trigger compaction
        
        Args:
            messages: Messages to compact
            preserve_keywords: Keywords to preserve
            strategy: Override strategy
            
        Returns:
            CompactionResult
        """
        strategy_enum = None
        if strategy:
            strategy_map = {
                "keep_last_n": CompactionStrategy.KEEP_LAST_N,
                "token_truncation": CompactionStrategy.TOKEN_TRUNCATION,
                "smart_summary": CompactionStrategy.SMART_SUMMARY,
                "recursive_summary": CompactionStrategy.RECURSIVE_SUMMARY
            }
            strategy_enum = strategy_map.get(strategy)
        
        result = await self.compressor.compact(
            messages=messages,
            strategy=strategy_enum,
            preserve_keywords=preserve_keywords
        )
        
        if result.success:
            self._compaction_history.append(result)
        
        return result
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get current context statistics"""
        return {
            "last_stats": {
                "total_tokens": self._last_stats.total_tokens if self._last_stats else 0,
                "usage_percent": self._last_stats.usage_percent if self._last_stats else 0,
                "needs_compaction": self._last_stats.needs_compaction if self._last_stats else False
            } if self._last_stats else None,
            "compaction_history_count": len(self._compaction_history),
            "total_tokens_saved": sum(r.tokens_saved for r in self._compaction_history),
            "enable_compaction": self.enable_compaction,
            "threshold": self.compaction_threshold
        }
    
    def get_compaction_history(self) -> List[Dict[str, Any]]:
        """Get history of compaction operations"""
        return [
            {
                "strategy": r.strategy_used,
                "original_tokens": r.original_tokens,
                "final_tokens": r.final_tokens,
                "tokens_saved": r.tokens_saved,
                "compression_ratio": r.compression_ratio,
                "timestamp": r.timestamp
            }
            for r in self._compaction_history
        ]
    
    def reset_stats(self):
        """Reset statistics for new session"""
        self._last_stats = None
        self._compaction_history = []
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    def set_compaction_enabled(self, enabled: bool):
        """Enable or disable automatic compaction"""
        self.enable_compaction = enabled
        self.logger.info(f"[CONTEXT MGR] Compaction {'enabled' if enabled else 'disabled'}")
    
    def set_compaction_threshold(self, threshold: int):
        """Set token threshold for compaction"""
        self.compaction_threshold = threshold
        self.compressor.config.token_threshold = threshold
        self.logger.info(f"[CONTEXT MGR] Compaction threshold set to {threshold}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_context_manager(
    enable_compaction: bool = True,
    threshold: int = 100000,
    strategy: str = "smart_summary"
) -> ContextManagementService:
    """Factory function to create ContextManagementService"""
    return ContextManagementService(
        enable_compaction=enable_compaction,
        compaction_threshold=threshold,
        compaction_strategy=strategy
    )