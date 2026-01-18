"""
Conversation Compactor Service

Auto-compress long conversations when token usage exceeds threshold.
Designed for V3/V4 agentic workflows with GPT-4.1-mini.

Key Features:
- Token-based threshold monitoring
- LLM-powered smart summarization
- Symbol and context preservation
- Clean integration with agent loops

Usage:
    compactor = get_conversation_compactor()

    # Check if compaction needed
    if compactor.should_compact(messages, system_prompt):
        result = await compactor.compact(messages, symbols=["NVDA", "GOOGL"])
        messages = result.compacted_messages
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.token_counter import TokenCounter, get_token_counter


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CompactorConfig:
    """
    Configuration for conversation compaction.

    Designed for GPT-4.1-mini with ~128K context window.
    Default threshold: 100K tokens (78% of context).
    """

    # Token thresholds
    max_context_tokens: int = 128000      # GPT-4.1-mini context window
    trigger_threshold: int = 100000       # Trigger compaction at this token count
    target_tokens: int = 20000            # Target tokens after compaction

    # Retention settings
    retention_messages: int = 4           # Keep last N messages unchanged
    preserve_system: bool = True          # Always preserve system prompt

    # Summary settings
    max_summary_tokens: int = 2000        # Max tokens for summary
    summary_model: str = "gpt-4.1-nano"   # Fast model for summarization

    # Advanced
    response_reserve: int = 4000          # Reserve for LLM response

    def __post_init__(self):
        """Validate configuration."""
        if self.trigger_threshold >= self.max_context_tokens:
            self.trigger_threshold = int(self.max_context_tokens * 0.78)

        if self.target_tokens >= self.trigger_threshold:
            self.target_tokens = int(self.trigger_threshold * 0.2)


# ============================================================================
# RESULT MODELS
# ============================================================================

@dataclass
class CompactionStats:
    """Statistics about current context."""

    total_tokens: int
    message_count: int
    system_tokens: int
    history_tokens: int
    usage_percent: float
    needs_compaction: bool
    threshold: int


@dataclass
class CompactionResult:
    """Result of compaction operation."""

    success: bool
    compacted_messages: List[Dict[str, str]]
    summary: Optional[str]

    # Metrics
    original_tokens: int
    final_tokens: int
    tokens_saved: int
    compression_ratio: float
    messages_removed: int

    # Metadata
    strategy: str
    execution_time_ms: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Preserved context
    preserved_symbols: List[str] = field(default_factory=list)


# ============================================================================
# SUMMARY PROMPT TEMPLATE
# ============================================================================

SUMMARY_PROMPT_TEMPLATE = """You are summarizing a conversation for continuation in a new context window.

<conversation_to_summarize>
{conversation}
</conversation_to_summarize>

<symbols_to_preserve>
{symbols}
</symbols_to_preserve>

Create a focused continuation summary that preserves:

1. **Task Overview**
   - The user's core request and what they're trying to accomplish
   - Any specific requirements or constraints mentioned

2. **Completed Work**
   - What has been done so far
   - Key findings or data retrieved (especially for financial symbols)
   - Important numbers, prices, or metrics discussed

3. **Key Discoveries**
   - Technical constraints or limitations found
   - Decisions made and their reasoning
   - Any errors encountered and how they were handled

4. **Current State**
   - Where we are in the workflow
   - What the user just asked or is waiting for

5. **Next Steps**
   - What needs to happen next
   - Any pending questions or clarifications needed

6. **Symbol Context** (CRITICAL)
   - For each financial symbol mentioned, preserve:
     - Current price/data if discussed
     - Analysis or insights provided
     - User's interest or questions about it

IMPORTANT:
- Be concise but complete - enable immediate resumption
- Preserve ALL financial data, prices, and symbol information
- Keep the summary under {max_tokens} tokens
- Write in a way that allows seamless continuation

Wrap your summary in <summary></summary> tags."""


# ============================================================================
# CONVERSATION COMPACTOR SERVICE
# ============================================================================

class ConversationCompactor(LoggerMixin):
    """
    Service for auto-compacting long conversations.

    Monitors token usage and compresses conversation history when threshold
    is exceeded, preserving critical context and symbols.
    """

    def __init__(self, config: Optional[CompactorConfig] = None):
        """
        Initialize Conversation Compactor.

        Args:
            config: Compaction configuration (uses defaults if None)
        """
        super().__init__()

        self.config = config or CompactorConfig()
        self._token_counter: Optional[TokenCounter] = None

        # Metrics tracking
        self._compaction_count = 0
        self._total_tokens_saved = 0

        self.logger.info(
            f"[COMPACTOR] Initialized: threshold={self.config.trigger_threshold:,} tokens, "
            f"target={self.config.target_tokens:,} tokens"
        )

    # ========================================================================
    # TOKEN COUNTING
    # ========================================================================

    def _get_token_counter(self) -> TokenCounter:
        """Get or create token counter instance."""
        if self._token_counter is None:
            self._token_counter = get_token_counter()
        return self._token_counter

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        return self._get_token_counter().count_tokens(text)

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count total tokens in message list."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.count_tokens(content)
            elif isinstance(content, list):
                # Handle content blocks
                for block in content:
                    if isinstance(block, dict):
                        total += self.count_tokens(str(block))
        return total

    # ========================================================================
    # COMPACTION CHECK
    # ========================================================================

    def get_stats(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        additional_context: str = ""
    ) -> CompactionStats:
        """
        Get current context statistics.

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            additional_context: Additional context (core memory, etc.)

        Returns:
            CompactionStats with token counts and compaction status
        """
        system_tokens = self.count_tokens(system_prompt)
        context_tokens = self.count_tokens(additional_context)
        history_tokens = self.count_messages_tokens(messages)

        total_tokens = system_tokens + context_tokens + history_tokens
        usage_percent = (total_tokens / self.config.max_context_tokens) * 100

        needs_compaction = total_tokens > self.config.trigger_threshold

        return CompactionStats(
            total_tokens=total_tokens,
            message_count=len(messages),
            system_tokens=system_tokens + context_tokens,
            history_tokens=history_tokens,
            usage_percent=round(usage_percent, 2),
            needs_compaction=needs_compaction,
            threshold=self.config.trigger_threshold
        )

    def should_compact(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        additional_context: str = ""
    ) -> bool:
        """
        Check if compaction is needed.

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            additional_context: Additional context

        Returns:
            True if compaction should be triggered
        """
        stats = self.get_stats(messages, system_prompt, additional_context)

        if stats.needs_compaction:
            self.logger.warning(
                f"[COMPACTOR] ⚠️ COMPACTION NEEDED: "
                f"{stats.usage_percent:.1f}% usage "
                f"({stats.total_tokens:,}/{self.config.trigger_threshold:,} tokens)"
            )
        else:
            self.logger.debug(
                f"[COMPACTOR] Context OK: {stats.usage_percent:.1f}% "
                f"({stats.total_tokens:,} tokens)"
            )

        return stats.needs_compaction

    # ========================================================================
    # COMPACTION EXECUTION
    # ========================================================================

    async def compact(
        self,
        messages: List[Dict[str, str]],
        symbols: List[str] = None,
        additional_context: str = ""
    ) -> CompactionResult:
        """
        Compact conversation history using LLM summarization.

        Args:
            messages: Messages to compact
            symbols: Financial symbols to preserve
            additional_context: Additional context to consider

        Returns:
            CompactionResult with compacted messages and metrics
        """
        start_time = datetime.now()
        symbols = symbols or []

        # Get initial stats
        initial_tokens = self.count_messages_tokens(messages)

        self.logger.info(
            f"[COMPACTOR] 🔄 Starting compaction: "
            f"{len(messages)} messages, {initial_tokens:,} tokens"
        )

        try:
            # Split messages: to_summarize vs to_keep
            retention = self.config.retention_messages

            if len(messages) <= retention:
                # Not enough messages to compact
                return CompactionResult(
                    success=True,
                    compacted_messages=messages,
                    summary=None,
                    original_tokens=initial_tokens,
                    final_tokens=initial_tokens,
                    tokens_saved=0,
                    compression_ratio=0.0,
                    messages_removed=0,
                    strategy="no_change",
                    execution_time_ms=0,
                    preserved_symbols=symbols
                )

            to_summarize = messages[:-retention]
            to_keep = messages[-retention:]

            # Generate summary using LLM
            summary = await self._generate_summary(
                to_summarize,
                symbols,
                additional_context
            )

            # Create compacted message list
            compacted_messages = []

            # Add summary as system context
            if summary:
                compacted_messages.append({
                    "role": "user",
                    "content": f"<conversation_summary>\n{summary}\n</conversation_summary>\n\nPlease continue from where we left off."
                })
                compacted_messages.append({
                    "role": "assistant",
                    "content": "I understand the context from our previous conversation. I'll continue helping you with the analysis. What would you like to focus on next?"
                })

            # Add retained messages
            compacted_messages.extend(to_keep)

            # Calculate metrics
            final_tokens = self.count_messages_tokens(compacted_messages)
            tokens_saved = initial_tokens - final_tokens
            compression_ratio = tokens_saved / initial_tokens if initial_tokens > 0 else 0

            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

            # Update internal metrics
            self._compaction_count += 1
            self._total_tokens_saved += tokens_saved

            self.logger.info(
                f"[COMPACTOR] ✅ Compaction complete: "
                f"{initial_tokens:,} → {final_tokens:,} tokens "
                f"(saved {tokens_saved:,}, {compression_ratio:.1%} reduction) "
                f"in {execution_time}ms"
            )

            return CompactionResult(
                success=True,
                compacted_messages=compacted_messages,
                summary=summary,
                original_tokens=initial_tokens,
                final_tokens=final_tokens,
                tokens_saved=tokens_saved,
                compression_ratio=compression_ratio,
                messages_removed=len(to_summarize),
                strategy="smart_summary",
                execution_time_ms=execution_time,
                preserved_symbols=symbols
            )

        except Exception as e:
            self.logger.error(f"[COMPACTOR] ❌ Error during compaction: {e}", exc_info=True)

            # Fallback: simple truncation
            return await self._fallback_truncation(messages, symbols, initial_tokens, start_time)

    # ========================================================================
    # SUMMARY GENERATION
    # ========================================================================

    async def _generate_summary(
        self,
        messages: List[Dict[str, str]],
        symbols: List[str],
        additional_context: str = ""
    ) -> str:
        """
        Generate LLM-powered summary of messages.

        Args:
            messages: Messages to summarize
            symbols: Symbols to preserve
            additional_context: Additional context

        Returns:
            Summary string
        """
        # Format messages for summary
        conversation_text = self._format_messages_for_summary(messages)
        symbols_text = ", ".join(symbols) if symbols else "None mentioned"

        # Build prompt
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            conversation=conversation_text,
            symbols=symbols_text,
            max_tokens=self.config.max_summary_tokens
        )

        # Call LLM for summarization
        try:
            from src.helpers.llm_helper import LLMGeneratorProvider

            llm_provider = LLMGeneratorProvider()

            response = await llm_provider.generate_text_simple(
                prompt=prompt,
                model_name=self.config.summary_model,
                max_tokens=self.config.max_summary_tokens,
                temperature=0.3,  # Low temperature for consistent summaries
            )

            # Extract summary from tags if present
            summary = self._extract_summary(response)

            self.logger.debug(
                f"[COMPACTOR] Generated summary: {len(summary)} chars, "
                f"~{self.count_tokens(summary)} tokens"
            )

            return summary

        except Exception as e:
            self.logger.error(f"[COMPACTOR] Summary generation failed: {e}")

            # Fallback: simple extraction
            return self._create_simple_summary(messages, symbols)

    def _format_messages_for_summary(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into readable text for summarization."""
        parts = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if isinstance(content, str):
                # Truncate very long messages
                if len(content) > 2000:
                    content = content[:2000] + "... [truncated]"
                parts.append(f"[{role.upper()}]: {content}")
            elif isinstance(content, list):
                # Handle content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    combined = " ".join(text_parts)
                    if len(combined) > 2000:
                        combined = combined[:2000] + "... [truncated]"
                    parts.append(f"[{role.upper()}]: {combined}")

        return "\n\n".join(parts)

    def _extract_summary(self, response: str) -> str:
        """Extract summary from LLM response."""
        import re

        # Try to extract from <summary> tags
        match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: return entire response
        return response.strip()

    def _create_simple_summary(
        self,
        messages: List[Dict[str, str]],
        symbols: List[str]
    ) -> str:
        """Create simple summary without LLM (fallback)."""
        parts = [
            f"Previous conversation summary ({len(messages)} messages):",
        ]

        if symbols:
            parts.append(f"Symbols discussed: {', '.join(symbols)}")

        # Extract key user queries
        user_queries = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) < 200:
                    user_queries.append(content)

        if user_queries:
            parts.append("Key user queries:")
            for q in user_queries[-3:]:  # Last 3 queries
                parts.append(f"  - {q}")

        return "\n".join(parts)

    # ========================================================================
    # FALLBACK TRUNCATION
    # ========================================================================

    async def _fallback_truncation(
        self,
        messages: List[Dict[str, str]],
        symbols: List[str],
        initial_tokens: int,
        start_time: datetime
    ) -> CompactionResult:
        """Fallback truncation when LLM summarization fails."""

        retention = self.config.retention_messages
        kept_messages = messages[-retention:] if len(messages) > retention else messages

        # Add truncation notice
        truncation_notice = {
            "role": "user",
            "content": f"[Previous {len(messages) - len(kept_messages)} messages truncated due to context limits. Symbols: {', '.join(symbols) if symbols else 'None'}]"
        }

        compacted = [truncation_notice] + kept_messages
        final_tokens = self.count_messages_tokens(compacted)

        execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return CompactionResult(
            success=True,
            compacted_messages=compacted,
            summary=None,
            original_tokens=initial_tokens,
            final_tokens=final_tokens,
            tokens_saved=initial_tokens - final_tokens,
            compression_ratio=(initial_tokens - final_tokens) / initial_tokens if initial_tokens > 0 else 0,
            messages_removed=len(messages) - len(kept_messages),
            strategy="fallback_truncation",
            execution_time_ms=execution_time,
            preserved_symbols=symbols
        )

    # ========================================================================
    # METRICS
    # ========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get compaction metrics."""
        return {
            "compaction_count": self._compaction_count,
            "total_tokens_saved": self._total_tokens_saved,
            "config": {
                "trigger_threshold": self.config.trigger_threshold,
                "target_tokens": self.config.target_tokens,
                "max_context": self.config.max_context_tokens,
                "retention_messages": self.config.retention_messages
            }
        }

    def reset_metrics(self):
        """Reset metrics counters."""
        self._compaction_count = 0
        self._total_tokens_saved = 0


# ============================================================================
# SINGLETON & FACTORY
# ============================================================================

_compactor_instance: Optional[ConversationCompactor] = None


def get_conversation_compactor(
    trigger_threshold: int = 100000,
    target_tokens: int = 20000,
    max_context: int = 128000
) -> ConversationCompactor:
    """
    Get singleton ConversationCompactor instance.

    Args:
        trigger_threshold: Token count to trigger compaction (default: 100K)
        target_tokens: Target tokens after compaction (default: 20K)
        max_context: Max context window size (default: 128K for GPT-4.1-mini)

    Returns:
        ConversationCompactor singleton instance
    """
    global _compactor_instance

    if _compactor_instance is None:
        config = CompactorConfig(
            max_context_tokens=max_context,
            trigger_threshold=trigger_threshold,
            target_tokens=target_tokens
        )
        _compactor_instance = ConversationCompactor(config=config)

    return _compactor_instance


def reset_compactor():
    """Reset singleton instance (for testing)."""
    global _compactor_instance
    _compactor_instance = None


# ============================================================================
# CONVENIENCE FUNCTIONS FOR AGENT LOOPS
# ============================================================================

async def check_and_compact_if_needed(
    messages: List[Dict[str, str]],
    system_prompt: str = "",
    symbols: List[str] = None,
    additional_context: str = ""
) -> tuple[List[Dict[str, str]], Optional[CompactionResult]]:
    """
    Convenience function: Check if compaction needed and compact if so.

    Use this in agent loops to auto-manage context.

    Args:
        messages: Conversation messages
        system_prompt: System prompt
        symbols: Symbols to preserve
        additional_context: Additional context

    Returns:
        Tuple of (potentially_compacted_messages, compaction_result_or_none)

    Example:
        messages, compaction_result = await check_and_compact_if_needed(
            messages=conversation_history,
            system_prompt=system_prompt,
            symbols=classification.symbols
        )

        if compaction_result:
            logger.info(f"Compacted: saved {compaction_result.tokens_saved} tokens")
    """
    compactor = get_conversation_compactor()

    if compactor.should_compact(messages, system_prompt, additional_context):
        result = await compactor.compact(
            messages=messages,
            symbols=symbols or [],
            additional_context=additional_context
        )
        return result.compacted_messages, result

    return messages, None
