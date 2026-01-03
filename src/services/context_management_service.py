"""
Context Management Service for MemGPT-style Memory System

PRODUCTION NOTES:
- Uses percentage-based triggering for flexible memory management
- Auto-calculate threshold from max_context_tokens * trigger_percent
- Enhanced logging with percentage info
"""

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
from src.agents.memory.core_memory import get_core_memory
from src.agents.memory.recursive_summary import get_recursive_summary_manager
from src.helpers.token_counter import TokenCounter


# ============================================================================
# DATA CLASSES
# ============================================================================
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

    # Percentage info
    usage_percent: float = 0.0

    # Timing
    preparation_time_ms: int = 0


# ============================================================================
# CONTEXT MANAGEMENT SERVICE
# ============================================================================
class ContextManagementService:
    """
    Unified Context Management Service with Percentage-based Auto-Trigger

    Features:
    - Percentage-based triggering instead of absolute threshold
    - Auto-calculate threshold from max_context_tokens * trigger_percent
    - Enhanced logging with percentage info
    - Strategy selection based on context

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
    │           │  CHECK %    │                   │
    │           │  THRESHOLD  │                   │
    │           └──────┬──────┘                   │
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
        max_context_tokens: int = 180000,
        trigger_percent: float = 80.0,
        compaction_strategy: str = "smart_summary"
    ):
        """
        Initialize Context Management Service with percentage-based triggering

        Args:
            enable_compaction: Whether to auto-compact when threshold reached
            max_context_tokens: Maximum context window size (default: 180K)
            trigger_percent: Percentage threshold for compaction (default: 80%)
            compaction_strategy: Strategy for compaction (smart_summary, keep_last_n, etc.)
        """
        self.logger = logging.getLogger(__name__)

        # Store configuration
        self.enable_compaction = enable_compaction
        self.max_context_tokens = max_context_tokens
        self.trigger_percent = trigger_percent

        # Calculate threshold from percentage
        self.compaction_threshold = int(max_context_tokens * (trigger_percent / 100))

        # Initialize components
        self.core_memory = get_core_memory()
        self.summary_manager = get_recursive_summary_manager()
        self.token_counter = TokenCounter()

        # Strategy mapping
        strategy_map = {
            "keep_last_n": CompactionStrategy.KEEP_LAST_N,
            "token_truncation": CompactionStrategy.TOKEN_TRUNCATION,
            "smart_summary": CompactionStrategy.SMART_SUMMARY,
            "recursive_summary": CompactionStrategy.RECURSIVE_SUMMARY
        }

        # Updated config with percentage
        compaction_config = CompactionConfig(
            token_threshold=self.compaction_threshold,
            strategy=strategy_map.get(compaction_strategy, CompactionStrategy.SMART_SUMMARY)
        )

        self.compressor = ContextCompressor(config=compaction_config)

        # Stats tracking
        self._last_stats: Optional[ContextStats] = None
        self._compaction_history: List[CompactionResult] = []

        # Enhanced logging
        self.logger.info(
            f"[CONTEXT MANAGER] Initialized: "
            f"compaction={enable_compaction}, "
            f"trigger={trigger_percent}% ({self.compaction_threshold:,} tokens), "
            f"strategy={compaction_strategy}"
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
        Prepare context for LLM call with auto-compaction

        This method:
        1. Loads core memory
        2. Gets recursive summary (if exists)
        3. Checks token usage (percentage-based)
        4. Auto-compacts if threshold exceeded
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
            # STEP 3: Check Token Usage (Percentage-based)
            # ================================================================
            # Combine all additional context for accurate token counting
            additional_context = core_memory_str + summary_str

            needs_compaction, stats = self.compressor.should_compact(
                messages=messages,
                system_prompt=system_prompt + additional_context
            )

            self._last_stats = stats

            # Log percentage usage
            usage_percent = (stats.total_tokens / self.max_context_tokens) * 100

            self.logger.info(
                f"[CONTEXT MGR] Token usage: {usage_percent:.1f}% "
                f"({stats.total_tokens:,}/{self.max_context_tokens:,} tokens)"
            )

            # ================================================================
            # STEP 4: Auto-Compact if Threshold Exceeded
            # ================================================================
            compaction_result = None
            final_messages = messages
            was_compacted = False

            if (needs_compaction or force_compaction) and self.enable_compaction:
                self.logger.warning(
                    f"[CONTEXT MGR] ⚠️ Context at {usage_percent:.1f}% - "
                    f"AUTO-COMPACTING..."
                )

                # Extract symbols to preserve
                symbols = self.compressor.extract_symbols_from_messages(messages)

                # Select strategy based on context
                strategy = self._select_compaction_strategy(
                    message_count=len(messages),
                    has_financial_data=len(symbols) > 0
                )

                compaction_result = await self.compressor.compact(
                    messages=messages,
                    preserve_keywords=symbols,
                    system_prompt=system_prompt,
                    strategy=strategy
                )

                if compaction_result.success:
                    final_messages = compaction_result.preserved_messages or messages
                    was_compacted = True
                    self._compaction_history.append(compaction_result)

                    self.logger.info(
                        f"[CONTEXT MGR] ✅ Compacted: "
                        f"{compaction_result.original_tokens:,} → "
                        f"{compaction_result.final_tokens:,} tokens "
                        f"(saved {compaction_result.tokens_saved:,})"
                    )
                else:
                    self.logger.warning(
                        "[CONTEXT MGR] ⚠️ Compaction failed, using original messages"
                    )

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

            self.logger.info(
                f"[CONTEXT MGR] Context prepared in {prep_time}ms - "
                f"Compacted: {was_compacted}"
            )

            return PreparedContext(
                messages=final_messages,
                system_prompt=enhanced_system,
                core_memory=core_memory_str,
                summary=summary_str,
                total_tokens=total_tokens,
                was_compacted=was_compacted,
                compaction_result=compaction_result,
                usage_percent=usage_percent,
                preparation_time_ms=prep_time
            )

        except Exception as e:
            self.logger.error(f"[CONTEXT MGR] Error preparing context: {e}")

            # Return basic context on error
            return PreparedContext(
                messages=messages,
                system_prompt=system_prompt,
                core_memory="",
                summary=None,
                total_tokens=0,
                was_compacted=False,
                usage_percent=0.0
            )

    # ========================================================================
    # STRATEGY SELECTION
    # ========================================================================

    def _select_compaction_strategy(
        self,
        message_count: int,
        has_financial_data: bool = False
    ) -> CompactionStrategy:
        """
        Select compaction strategy based on context characteristics

        Args:
            message_count: Number of messages in conversation
            has_financial_data: Whether conversation contains financial symbols

        Returns:
            CompactionStrategy enum
        """
        # Very long conversations -> recursive summary (most aggressive)
        if message_count > 100:
            self.logger.info("[CONTEXT MGR] Strategy: recursive_summary (100+ messages)")
            return CompactionStrategy.RECURSIVE_SUMMARY

        # Financial data or medium conversations -> smart summary (preserves keywords)
        if has_financial_data or message_count > 20:
            self.logger.info(
                f"[CONTEXT MGR] Strategy: smart_summary "
                f"(financial={has_financial_data}, messages={message_count})"
            )
            return CompactionStrategy.SMART_SUMMARY

        # Short conversations -> keep last N (simple, preserves recent context)
        self.logger.info("[CONTEXT MGR] Strategy: keep_last_n (short conversation)")
        return CompactionStrategy.KEEP_LAST_N

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _build_enhanced_system_prompt(
        self,
        base_prompt: str,
        core_memory: str,
        summary: str
    ) -> str:
        """Build enhanced system prompt with memory components"""
        parts = [base_prompt]

        if core_memory:
            parts.append(f"\n\n<core_memory>\n{core_memory}\n</core_memory>")

        if summary:
            parts.append(f"\n\n<conversation_summary>\n{summary}\n</conversation_summary>")

        return "".join(parts)

    def _count_total_tokens(
        self,
        system: str,
        messages: List[Dict[str, str]]
    ) -> int:
        """Count total tokens in context"""
        total = self.token_counter.count_tokens(system) if system else 0

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
            preserve_keywords: Keywords to preserve (e.g., stock symbols)
            strategy: Override strategy (keep_last_n, smart_summary, etc.)

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
            "config": {
                "enable_compaction": self.enable_compaction,
                "max_context_tokens": self.max_context_tokens,
                "trigger_percent": self.trigger_percent,
                "threshold_tokens": self.compaction_threshold
            }
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
        self.logger.info(
            f"[CONTEXT MGR] Compaction {'enabled' if enabled else 'disabled'}"
        )

    def set_trigger_percent(self, percent: float):
        """Set percentage threshold for compaction"""
        if 0 < percent <= 100:
            self.trigger_percent = percent
            self.compaction_threshold = int(self.max_context_tokens * (percent / 100))
            self.compressor.config.token_threshold = self.compaction_threshold

            self.logger.info(
                f"[CONTEXT MGR] Trigger set to {percent}% "
                f"({self.compaction_threshold:,} tokens)"
            )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_context_manager(
    enable_compaction: bool = True,
    max_context_tokens: int = 180000,
    trigger_percent: float = 80.0,
    strategy: str = "smart_summary"
) -> ContextManagementService:
    """
    Factory function to create ContextManagementService

    Args:
        enable_compaction: Whether to enable auto-compaction
        max_context_tokens: Maximum context window size
        trigger_percent: Percentage threshold (default 80%)
        strategy: Compaction strategy

    Returns:
        Configured ContextManagementService
    """
    return ContextManagementService(
        enable_compaction=enable_compaction,
        max_context_tokens=max_context_tokens,
        trigger_percent=trigger_percent,
        compaction_strategy=strategy
    )


# Singleton instance
_context_manager_instance: Optional[ContextManagementService] = None


def get_context_manager(
    enable_compaction: bool = True,
    max_context_tokens: int = 180000,
    trigger_percent: float = 80.0,
    strategy: str = "smart_summary"
) -> ContextManagementService:
    """
    Get singleton ContextManagementService instance

    Args:
        enable_compaction: Whether to auto-compact
        max_context_tokens: Maximum context window size
        trigger_percent: Percentage threshold for compaction
        strategy: Compaction strategy

    Returns:
        ContextManagementService singleton instance
    """
    global _context_manager_instance

    if _context_manager_instance is None:
        _context_manager_instance = ContextManagementService(
            enable_compaction=enable_compaction,
            max_context_tokens=max_context_tokens,
            trigger_percent=trigger_percent,
            compaction_strategy=strategy
        )

    return _context_manager_instance


def reset_context_manager():
    """Reset singleton instance (for testing)"""
    global _context_manager_instance
    _context_manager_instance = None