"""
ContextBuilder Service - Unified Context Assembly

Centralizes all context loading for the AI chatbot:
- Core Memory (user profile, preferences)
- Conversation Summary (compressed old messages)
- Recent Messages (raw recent K messages)
- Working Memory Symbols (session-scoped symbols)

PROBLEM SOLVED:
Previously, context was loaded differently in each phase:
- Intent Classification: loaded history but NOT summary
- Agent Loop: loaded summary but unclear on history
- Direct Response: inconsistent

SOLUTION:
Single service that loads ALL context ONCE, then provides
phase-specific views based on configuration.

Usage:
    builder = get_context_builder()
    context = await builder.build_context(
        session_id=session_id,
        user_id=user_id,
        phase="intent_classification"
    )

    # Use context in LLM calls
    system_prompt = context.to_system_prompt(base_prompt)
"""

import asyncio
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# CONFIGURATION
# ============================================================================

class ContextPhase(str, Enum):
    """Phases that require context"""
    INTENT_CLASSIFICATION = "intent_classification"
    AGENT_LOOP = "agent_loop"
    DIRECT_RESPONSE = "direct_response"
    SYNTHESIS = "synthesis"


@dataclass
class ContextConfig:
    """Configuration for context loading per phase"""
    include_core_memory: bool = True
    include_summary: bool = True
    recent_k: int = 5
    include_wm_symbols: bool = True
    include_tool_descriptions: bool = False
    max_tokens: int = 100_000


# Phase-specific configurations
PHASE_CONFIGS: Dict[str, ContextConfig] = {
    ContextPhase.INTENT_CLASSIFICATION.value: ContextConfig(
        include_core_memory=True,
        include_summary=True,      # Critical for context continuity
        recent_k=3,                # Fast classification - fewer messages
        include_wm_symbols=True,
        include_tool_descriptions=False,
    ),
    ContextPhase.AGENT_LOOP.value: ContextConfig(
        include_core_memory=True,
        include_summary=True,
        recent_k=5,                # Balanced - more context for reasoning
        include_wm_symbols=True,
        include_tool_descriptions=True,
    ),
    ContextPhase.DIRECT_RESPONSE.value: ContextConfig(
        include_core_memory=True,
        include_summary=True,
        recent_k=5,
        include_wm_symbols=True,
        include_tool_descriptions=False,
    ),
    ContextPhase.SYNTHESIS.value: ContextConfig(
        include_core_memory=True,
        include_summary=True,
        recent_k=3,                # Less context needed for final synthesis
        include_wm_symbols=False,
        include_tool_descriptions=False,
    ),
}


# ============================================================================
# ASSEMBLED CONTEXT
# ============================================================================

@dataclass
class AssembledContext:
    """
    Complete assembled context for a request.

    Contains all context components loaded from various sources,
    ready to be injected into LLM prompts.
    """
    # Core Memory (user profile from YAML/DB)
    core_memory: Optional[str] = None

    # Conversation Summary (compressed old messages)
    summary: Optional[str] = None

    # Recent Messages (raw messages for immediate context)
    recent_messages: List[Dict[str, str]] = field(default_factory=list)

    # Working Memory Symbols (session-scoped symbols from previous turns)
    wm_symbols: List[str] = field(default_factory=list)

    # Metadata
    total_tokens: int = 0
    was_compacted: bool = False
    loaded_at: datetime = field(default_factory=datetime.utcnow)
    phase: str = "unknown"
    session_id: Optional[str] = None
    user_id: Optional[int] = None

    # Loading status
    core_memory_loaded: bool = False
    summary_loaded: bool = False
    history_loaded: bool = False
    wm_loaded: bool = False

    def to_system_prompt(self, base_prompt: str) -> str:
        """
        Build system prompt with all context injected.

        Args:
            base_prompt: The base system prompt to augment

        Returns:
            Complete system prompt with context tags
        """
        parts = [base_prompt]

        if self.core_memory:
            parts.append(f"\n\n<USER_PROFILE>\n{self.core_memory}\n</USER_PROFILE>")

        if self.summary:
            parts.append(f"\n\n<CONVERSATION_SUMMARY>\n{self.summary}\n</CONVERSATION_SUMMARY>")

        if self.wm_symbols:
            symbols_str = ", ".join(self.wm_symbols)
            parts.append(f"\n\n<SYMBOLS_HINT>Recently discussed: {symbols_str}</SYMBOLS_HINT>")

        return "".join(parts)

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of loaded context for logging/debugging"""
        return {
            "phase": self.phase,
            "core_memory_chars": len(self.core_memory) if self.core_memory else 0,
            "summary_chars": len(self.summary) if self.summary else 0,
            "recent_messages_count": len(self.recent_messages),
            "wm_symbols": self.wm_symbols,
            "total_tokens": self.total_tokens,
            "was_compacted": self.was_compacted,
            "loaded_at": self.loaded_at.isoformat(),
        }


# ============================================================================
# CONTEXT BUILDER SERVICE
# ============================================================================

class ContextBuilder(LoggerMixin):
    """
    Unified context assembly service.

    Loads context ONCE, provides phase-specific views.
    Eliminates inconsistency between phases.
    """

    def __init__(self):
        super().__init__()
        self._cache: Dict[str, AssembledContext] = {}
        self._cache_ttl_seconds = 30  # Cache for 30 seconds

    async def build_context(
        self,
        session_id: str,
        user_id: int,
        phase: str = ContextPhase.AGENT_LOOP.value,
        force_reload: bool = False,
    ) -> AssembledContext:
        """
        Build complete context for a phase.

        Args:
            session_id: Chat session ID
            user_id: User ID for profile loading
            phase: Phase name (intent_classification, agent_loop, etc.)
            force_reload: Force reload even if cached

        Returns:
            AssembledContext with all components loaded
        """
        # Get config for phase
        config = PHASE_CONFIGS.get(phase, PHASE_CONFIGS[ContextPhase.AGENT_LOOP.value])

        # Check cache
        cache_key = f"{session_id}_{phase}"
        if not force_reload and cache_key in self._cache:
            cached = self._cache[cache_key]
            age_seconds = (datetime.utcnow() - cached.loaded_at).total_seconds()
            if age_seconds < self._cache_ttl_seconds:
                self.logger.debug(f"[CONTEXT_BUILDER] Using cached context (age={age_seconds:.1f}s)")
                return cached

        self.logger.info(f"[CONTEXT_BUILDER] Building context for phase={phase}")

        # Initialize context
        context = AssembledContext(
            phase=phase,
            session_id=session_id,
            user_id=user_id,
        )

        # Load components in parallel
        tasks = []

        if config.include_core_memory:
            tasks.append(self._load_core_memory(user_id, context))

        if config.include_summary:
            tasks.append(self._load_summary(session_id, context))

        tasks.append(self._load_recent_messages(session_id, config.recent_k, context))

        if config.include_wm_symbols:
            tasks.append(self._load_wm_symbols(session_id, context))

        # Execute all loads in parallel
        await asyncio.gather(*tasks, return_exceptions=True)

        # Log summary
        summary = context.get_context_summary()
        self.logger.info(
            f"[CONTEXT_BUILDER] ✅ Context built: "
            f"core_memory={summary['core_memory_chars']}chars, "
            f"summary={summary['summary_chars']}chars, "
            f"messages={summary['recent_messages_count']}, "
            f"wm_symbols={summary['wm_symbols']}"
        )

        # Cache the context
        self._cache[cache_key] = context

        return context

    async def _load_core_memory(self, user_id: int, context: AssembledContext) -> None:
        """Load core memory (user profile)"""
        try:
            from src.agents.memory.core_memory import get_core_memory

            core_memory = get_core_memory()
            cm_data = await core_memory.load_core_memory(str(user_id))
            human_block = cm_data.get("human", "")

            if human_block and len(human_block) > 20:
                context.core_memory = human_block
                context.core_memory_loaded = True
                self.logger.debug(f"[CONTEXT_BUILDER] Core Memory loaded: {len(human_block)} chars")
            else:
                self.logger.debug(f"[CONTEXT_BUILDER] Core Memory empty/minimal for user {user_id}")

        except Exception as e:
            self.logger.warning(f"[CONTEXT_BUILDER] Core Memory load failed: {e}")

    async def _load_summary(self, session_id: str, context: AssembledContext) -> None:
        """Load conversation summary"""
        try:
            from src.agents.memory.recursive_summary import get_recursive_summary_manager

            summary_manager = get_recursive_summary_manager()
            summary = await summary_manager.get_active_summary(session_id)

            if summary:
                context.summary = summary
                context.summary_loaded = True
                self.logger.debug(f"[CONTEXT_BUILDER] Summary loaded: {len(summary)} chars")
            else:
                self.logger.debug(f"[CONTEXT_BUILDER] No active summary for session")

        except Exception as e:
            self.logger.warning(f"[CONTEXT_BUILDER] Summary load failed: {e}")

    async def _load_recent_messages(
        self,
        session_id: str,
        limit: int,
        context: AssembledContext
    ) -> None:
        """Load recent conversation messages"""
        try:
            from src.db.repositories.chat_repository import ChatRepository

            chat_repo = ChatRepository()
            messages = await chat_repo.get_conversation_history(
                session_id=session_id,
                limit=limit
            )

            if messages:
                context.recent_messages = messages
                context.history_loaded = True
                self.logger.debug(f"[CONTEXT_BUILDER] History loaded: {len(messages)} messages")

        except Exception as e:
            self.logger.warning(f"[CONTEXT_BUILDER] History load failed: {e}")

    async def _load_wm_symbols(self, session_id: str, context: AssembledContext) -> None:
        """Load working memory symbols"""
        try:
            from src.agents.memory.working_memory_integration import (
                setup_working_memory_for_request
            )

            wm_integration = setup_working_memory_for_request(
                session_id=session_id,
                user_id=context.user_id or 0,
                flow_id=f"ctx_{session_id[:8]}",
            )

            symbols = wm_integration.get_current_symbols()
            if symbols:
                context.wm_symbols = symbols
                context.wm_loaded = True
                self.logger.debug(f"[CONTEXT_BUILDER] WM symbols loaded: {symbols}")

        except Exception as e:
            self.logger.warning(f"[CONTEXT_BUILDER] WM symbols load failed: {e}")

    def clear_cache(self, session_id: Optional[str] = None) -> None:
        """Clear context cache"""
        if session_id:
            keys_to_remove = [k for k in self._cache if k.startswith(session_id)]
            for key in keys_to_remove:
                del self._cache[key]
            self.logger.debug(f"[CONTEXT_BUILDER] Cleared cache for session {session_id}")
        else:
            self._cache.clear()
            self.logger.debug("[CONTEXT_BUILDER] Cleared all cache")


# ============================================================================
# SINGLETON
# ============================================================================

_context_builder_instance: Optional[ContextBuilder] = None


def get_context_builder() -> ContextBuilder:
    """Get singleton ContextBuilder instance"""
    global _context_builder_instance

    if _context_builder_instance is None:
        _context_builder_instance = ContextBuilder()

    return _context_builder_instance


def reset_context_builder() -> None:
    """Reset singleton (for testing)"""
    global _context_builder_instance
    _context_builder_instance = None
