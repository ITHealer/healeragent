import functools
import re
import json
from typing import Dict, List, Tuple, Optional, Any, Literal
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType

# Token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# ============================================================================
# TOKEN COUNTER
# ============================================================================

@functools.lru_cache(maxsize=10)
def get_encoding(model_name: str):
    """Get tiktoken encoding for model (cached)"""
    if not TIKTOKEN_AVAILABLE:
        return None
    
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens in text"""
    if not text:
        return 0
    
    encoding = get_encoding(model_name)
    if encoding:
        return len(encoding.encode(text))
    
    # Fallback: approximate 4 chars per token
    return len(text) // 4


# ============================================================================
# DATA MODELS
# ============================================================================

class CompactionStrategy(Enum):
    """Available compaction strategies"""
    KEEP_LAST_N = "keep_last_n"
    TOKEN_TRUNCATION = "token_truncation"
    SMART_SUMMARY = "smart_summary"
    RECURSIVE_SUMMARY = "recursive_summary"


@dataclass
class CompactionConfig:
    """Configuration for context compaction"""
    
    # Thresholds
    token_threshold: int = 100000          # Trigger at this token count
    token_target: int = 50000              # Target after compaction
    message_threshold: int = 50            # Trigger at this message count
    
    # Strategy
    strategy: CompactionStrategy = CompactionStrategy.SMART_SUMMARY
    
    # Retention
    retention_window: int = 6              # Messages to keep unchanged
    preserve_system: bool = True           # Always keep system prompt
    
    # Summary settings
    max_summary_tokens: int = 2000         # Max tokens for summary
    summary_model: str = "gpt-4.1-nano"    # Model for summarization
    
    # Advanced
    preserve_keywords: List[str] = field(default_factory=list)  # Keywords to preserve


@dataclass
class CompactionResult:
    """Result of compaction operation"""
    
    success: bool
    strategy_used: str
    
    # Token metrics
    original_tokens: int
    final_tokens: int
    tokens_saved: int
    compression_ratio: float
    
    # Message metrics
    original_messages: int
    final_messages: int
    messages_summarized: int
    
    # Content
    summary_text: Optional[str] = None
    preserved_messages: Optional[List[Dict]] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time_ms: int = 0


@dataclass
class ContextStats:
    """Statistics about current context"""
    
    total_tokens: int
    message_count: int
    system_tokens: int
    history_tokens: int
    
    # Percentages
    usage_percent: float
    
    # Thresholds
    threshold_tokens: int
    needs_compaction: bool
    
    # Details
    token_breakdown: Dict[str, int] = field(default_factory=dict)


# ============================================================================
# COMPACTION PROMPTS
# ============================================================================

class CompactionPrompts:
    """Prompts for context summarization"""
    
    @staticmethod
    def get_financial_summary_prompt(
        preserve_keywords: List[str] = None
    ) -> str:
        """
        Summary prompt optimized for financial conversations
        
        Based on Claude Code's compaction prompt structure
        """
        preserve_section = ""
        if preserve_keywords:
            preserve_section = f"""
6. PRESERVE these specific items if mentioned:
   {', '.join(preserve_keywords)}
"""
        
        return f"""Analyze this financial conversation and create a structured summary.

SUMMARIZATION RULES:
1. Primary Objective: What was the user trying to accomplish?
2. Stocks/Symbols Discussed: List ALL tickers mentioned with context
3. Key Data Points: Important numbers, ratios, prices mentioned
4. Decisions Made: Any conclusions or actions decided
5. Current State: Where the conversation left off
{preserve_section}
CRITICAL REQUIREMENTS:
- Keep ALL stock symbols (AAPL, NVDA, etc.) mentioned
- Preserve specific numbers and ratios (RSI=28, P/E=15.2)
- Include any watchlist or portfolio mentions
- Note user preferences discovered

FORMAT:
<summary>
## Objective
[User's main goal]

## Symbols Discussed
- [SYMBOL]: [Brief context]

## Key Data
- [Metric]: [Value] - [Interpretation if any]

## Decisions
- [Any conclusions reached]

## Current State
[Where conversation left off]

## User Preferences
[Any preferences learned]
</summary>

Keep under 2000 tokens. Be concise but preserve critical financial data.
"""

    @staticmethod
    def get_general_summary_prompt() -> str:
        """General purpose summary prompt"""
        return """Analyze this conversation and create a structured summary.

Create a summary that preserves:
1. Main topics discussed
2. Key decisions made
3. Important facts and numbers
4. Current state/where we left off
5. Any user preferences learned

Format output in <summary>...</summary> tags.
Keep under 1500 tokens.
"""


# ============================================================================
# CONTEXT COMPRESSOR
# ============================================================================

class ContextCompressor(LoggerMixin):
    """
    Advanced Context Compressor with Multi-Strategy Support
    
    Features:
    - Multiple compaction strategies
    - Smart summarization with LLM
    - Token monitoring and metrics
    - Preservation of critical content
    
    Usage:
        compressor = ContextCompressor()
        
        # Check if compaction needed
        stats = compressor.get_context_stats(messages)
        
        if stats.needs_compaction:
            result = await compressor.compact(
                messages=messages,
                strategy=CompactionStrategy.SMART_SUMMARY
            )
    """
    
    def __init__(
        self,
        config: Optional[CompactionConfig] = None
    ):
        """
        Initialize Context Compressor
        
        Args:
            config: Compaction configuration (uses defaults if None)
        """
        super().__init__()
        
        self.config = config or CompactionConfig()
        self.llm_provider = LLMGeneratorProvider()
        
        self.logger.info(
            f"[CONTEXT COMPRESSOR] Initialized with strategy={self.config.strategy.value}, "
            f"threshold={self.config.token_threshold} tokens"
        )
    
    # ========================================================================
    # CONTEXT ANALYSIS
    # ========================================================================
    
    def get_context_stats(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = ""
    ) -> ContextStats:
        """
        Analyze current context and return statistics
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt if separate
            
        Returns:
            ContextStats with detailed metrics
        """
        # Count tokens by category
        system_tokens = count_tokens(system_prompt)
        
        token_breakdown = {"system": system_tokens}
        history_tokens = 0
        
        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            role = msg.get("role", "unknown")
            msg_tokens = count_tokens(content)
            
            history_tokens += msg_tokens
            token_breakdown[f"{role}_{i}"] = msg_tokens
        
        total_tokens = system_tokens + history_tokens
        usage_percent = (total_tokens / self.config.token_threshold) * 100
        
        # Check thresholds
        needs_compaction = (
            total_tokens > self.config.token_threshold or
            len(messages) > self.config.message_threshold
        )
        
        return ContextStats(
            total_tokens=total_tokens,
            message_count=len(messages),
            system_tokens=system_tokens,
            history_tokens=history_tokens,
            usage_percent=round(usage_percent, 2),
            threshold_tokens=self.config.token_threshold,
            needs_compaction=needs_compaction,
            token_breakdown=token_breakdown
        )
    
    def should_compact(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = ""
    ) -> Tuple[bool, ContextStats]:
        """
        Check if compaction is needed
        
        Returns:
            (needs_compaction, stats)
        """
        stats = self.get_context_stats(messages, system_prompt)
        
        if stats.needs_compaction:
            self.logger.warning(
                f"[CONTEXT COMPRESSOR] ⚠️ Compaction needed: "
                f"{stats.usage_percent:.1f}% usage, "
                f"{stats.message_count} messages"
            )
        else:
            self.logger.info(
                f"[CONTEXT COMPRESSOR] ✅ Context OK: "
                f"{stats.usage_percent:.1f}% usage"
            )
        
        return stats.needs_compaction, stats
    
    # ========================================================================
    # COMPACTION METHODS
    # ========================================================================
    
    async def compact(
        self,
        messages: List[Dict[str, str]],
        strategy: Optional[CompactionStrategy] = None,
        preserve_keywords: List[str] = None,
        system_prompt: str = ""
    ) -> CompactionResult:
        """
        Compact messages using specified strategy
        
        Args:
            messages: Messages to compact
            strategy: Strategy to use (default from config)
            preserve_keywords: Keywords to preserve in summary
            system_prompt: System prompt for context
            
        Returns:
            CompactionResult with compacted messages
        """
        start_time = datetime.now()
        strategy = strategy or self.config.strategy
        
        # Get initial stats
        initial_stats = self.get_context_stats(messages, system_prompt)
        
        self.logger.info(
            f"[CONTEXT COMPRESSOR] Starting compaction: "
            f"strategy={strategy.value}, "
            f"messages={len(messages)}, "
            f"tokens={initial_stats.total_tokens}"
        )
        
        try:
            # Execute strategy
            if strategy == CompactionStrategy.KEEP_LAST_N:
                result = await self._compact_keep_last_n(messages)
            
            elif strategy == CompactionStrategy.TOKEN_TRUNCATION:
                result = await self._compact_token_truncation(messages)
            
            elif strategy == CompactionStrategy.SMART_SUMMARY:
                result = await self._compact_smart_summary(
                    messages,
                    preserve_keywords or self.config.preserve_keywords
                )
            
            elif strategy == CompactionStrategy.RECURSIVE_SUMMARY:
                result = await self._compact_recursive(
                    messages,
                    preserve_keywords or self.config.preserve_keywords
                )
            
            else:
                # Fallback to keep_last_n
                result = await self._compact_keep_last_n(messages)
            
            # Calculate execution time
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            result.execution_time_ms = execution_time
            
            self.logger.info(
                f"[CONTEXT COMPRESSOR] ✅ Compaction complete: "
                f"{initial_stats.total_tokens} → {result.final_tokens} tokens "
                f"({result.compression_ratio:.1%} reduction) "
                f"in {execution_time}ms"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"[CONTEXT COMPRESSOR] Error: {e}", exc_info=True)
            
            # Fallback to simple truncation
            return await self._compact_keep_last_n(messages)
    
    # ========================================================================
    # STRATEGY IMPLEMENTATIONS
    # ========================================================================
    
    async def _compact_keep_last_n(
        self,
        messages: List[Dict[str, str]]
    ) -> CompactionResult:
        """
        Strategy 1: Keep last N messages
        
        Fast and simple, good for specific tasks
        """
        n = self.config.retention_window
        original_count = len(messages)
        original_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
        
        # Keep last N messages
        preserved = messages[-n:] if len(messages) > n else messages
        final_tokens = sum(count_tokens(m.get("content", "")) for m in preserved)
        
        return CompactionResult(
            success=True,
            strategy_used="keep_last_n",
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            tokens_saved=original_tokens - final_tokens,
            compression_ratio=(original_tokens - final_tokens) / original_tokens if original_tokens > 0 else 0,
            original_messages=original_count,
            final_messages=len(preserved),
            messages_summarized=original_count - len(preserved),
            preserved_messages=preserved
        )
    
    async def _compact_token_truncation(
        self,
        messages: List[Dict[str, str]]
    ) -> CompactionResult:
        """
        Strategy 2: Token-based truncation
        
        Keep messages from end until token budget exhausted
        """
        original_count = len(messages)
        original_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
        
        target = self.config.token_target
        preserved = []
        current_tokens = 0
        
        # Work backwards
        for msg in reversed(messages):
            msg_tokens = count_tokens(msg.get("content", ""))
            
            if current_tokens + msg_tokens > target:
                break
            
            preserved.insert(0, msg)
            current_tokens += msg_tokens
        
        # Ensure at least retention_window messages
        if len(preserved) < self.config.retention_window:
            preserved = messages[-self.config.retention_window:]
            current_tokens = sum(count_tokens(m.get("content", "")) for m in preserved)
        
        return CompactionResult(
            success=True,
            strategy_used="token_truncation",
            original_tokens=original_tokens,
            final_tokens=current_tokens,
            tokens_saved=original_tokens - current_tokens,
            compression_ratio=(original_tokens - current_tokens) / original_tokens if original_tokens > 0 else 0,
            original_messages=original_count,
            final_messages=len(preserved),
            messages_summarized=original_count - len(preserved),
            preserved_messages=preserved
        )
    
    async def _compact_smart_summary(
        self,
        messages: List[Dict[str, str]],
        preserve_keywords: List[str] = None
    ) -> CompactionResult:
        """
        Strategy 3: LLM-powered smart summarization
        
        Best for preserving context while reducing tokens
        """
        original_count = len(messages)
        original_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
        
        # Split into messages to summarize and messages to preserve
        retention = self.config.retention_window
        
        if len(messages) <= retention:
            # Not enough messages to summarize
            return CompactionResult(
                success=True,
                strategy_used="smart_summary",
                original_tokens=original_tokens,
                final_tokens=original_tokens,
                tokens_saved=0,
                compression_ratio=0,
                original_messages=original_count,
                final_messages=original_count,
                messages_summarized=0,
                preserved_messages=messages
            )
        
        # Messages to summarize (older ones)
        to_summarize = messages[:-retention]
        to_preserve = messages[-retention:]
        
        # Format conversation for summarization
        conversation_text = self._format_messages_for_summary(to_summarize)
        
        # Get summary prompt
        summary_prompt = CompactionPrompts.get_financial_summary_prompt(preserve_keywords)
        
        # Generate summary
        summary_text = await self._generate_summary(
            conversation_text=conversation_text,
            summary_prompt=summary_prompt
        )
        
        if not summary_text:
            # Fallback to token truncation
            return await self._compact_token_truncation(messages)
        
        # Create compacted message list
        summary_message = {
            "role": "user",
            "content": f"<conversation_summary>\n{summary_text}\n</conversation_summary>"
        }
        
        final_messages = [summary_message] + to_preserve
        final_tokens = sum(count_tokens(m.get("content", "")) for m in final_messages)
        
        return CompactionResult(
            success=True,
            strategy_used="smart_summary",
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            tokens_saved=original_tokens - final_tokens,
            compression_ratio=(original_tokens - final_tokens) / original_tokens if original_tokens > 0 else 0,
            original_messages=original_count,
            final_messages=len(final_messages),
            messages_summarized=len(to_summarize),
            summary_text=summary_text,
            preserved_messages=final_messages
        )
    
    async def _compact_recursive(
        self,
        messages: List[Dict[str, str]],
        preserve_keywords: List[str] = None
    ) -> CompactionResult:
        """
        Strategy 4: Recursive summarization
        
        For very long conversations, summarize in chunks
        Pattern: Messages 1-20 → Summary v1
                 Messages 21-40 + Summary v1 → Summary v2
        """
        original_count = len(messages)
        original_tokens = sum(count_tokens(m.get("content", "")) for m in messages)
        
        chunk_size = 20
        retention = self.config.retention_window
        
        if len(messages) <= retention + chunk_size:
            # Use smart summary for smaller conversations
            return await self._compact_smart_summary(messages, preserve_keywords)
        
        # Keep recent messages
        to_preserve = messages[-retention:]
        to_summarize = messages[:-retention]
        
        # Recursive summarization
        accumulated_summary = ""
        
        for i in range(0, len(to_summarize), chunk_size):
            chunk = to_summarize[i:i + chunk_size]
            
            if not chunk:
                break
            
            # Include previous summary in context
            if accumulated_summary:
                context_text = f"Previous context:\n{accumulated_summary}\n\n"
            else:
                context_text = ""
            
            conversation_text = self._format_messages_for_summary(chunk)
            full_text = context_text + conversation_text
            
            # Generate summary for this chunk
            chunk_summary = await self._generate_summary(
                conversation_text=full_text,
                summary_prompt=CompactionPrompts.get_financial_summary_prompt(preserve_keywords)
            )
            
            if chunk_summary:
                accumulated_summary = chunk_summary
        
        if not accumulated_summary:
            return await self._compact_token_truncation(messages)
        
        # Create final compacted messages
        summary_message = {
            "role": "user",
            "content": f"<conversation_summary>\n{accumulated_summary}\n</conversation_summary>"
        }
        
        final_messages = [summary_message] + to_preserve
        final_tokens = sum(count_tokens(m.get("content", "")) for m in final_messages)
        
        return CompactionResult(
            success=True,
            strategy_used="recursive_summary",
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            tokens_saved=original_tokens - final_tokens,
            compression_ratio=(original_tokens - final_tokens) / original_tokens if original_tokens > 0 else 0,
            original_messages=original_count,
            final_messages=len(final_messages),
            messages_summarized=len(to_summarize),
            summary_text=accumulated_summary,
            preserved_messages=final_messages
        )
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _format_messages_for_summary(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
        """Format messages into readable text for summarization"""
        lines = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            
            # Truncate very long messages
            if len(content) > 1000:
                content = content[:1000] + "..."
            
            lines.append(f"[{role}]: {content}")
        
        return "\n\n".join(lines)
    
    async def _generate_summary(
        self,
        conversation_text: str,
        summary_prompt: str
    ) -> Optional[str]:
        """Generate summary using LLM"""
        
        try:
            full_prompt = f"{summary_prompt}\n\nCONVERSATION:\n{conversation_text}"
            
            response = await self.llm_provider.generate_response(
                model_name=self.config.summary_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a conversation summarization expert. Create concise, accurate summaries that preserve critical information."
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                temperature=0.3,
                max_tokens=self.config.max_summary_tokens,
                provider_type=ProviderType.OPENAI
            )
            
            # Extract content
            if isinstance(response, dict):
                content = response.get("content", "")
            else:
                content = str(response)
            
            # Extract from <summary> tags if present
            match = re.search(r'<summary>(.*?)</summary>', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            return content.strip()
            
        except Exception as e:
            self.logger.error(f"[CONTEXT COMPRESSOR] Summary generation error: {e}")
            return None
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def extract_symbols_from_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> List[str]:
        """
        Extract stock symbols from messages
        
        Used to ensure symbols are preserved during compaction
        """
        # symbol_pattern = r'\b[A-Z]{1,5}\b'
        symbol_pattern = r'\b[A-Z][A-Z0-9]{0,14}\b'
        symbols = set()
        
        # Common words to exclude
        exclude = {
            'I', 'A', 'THE', 'AND', 'OR', 'NOT', 'IS', 'ARE', 'WAS', 'BE',
            'TO', 'OF', 'IN', 'FOR', 'ON', 'WITH', 'AS', 'AT', 'BY', 'AN',
            'IT', 'IF', 'SO', 'UP', 'DO', 'NO', 'HE', 'WE', 'MY', 'OK',
            'API', 'LLM', 'AI', 'ML', 'USD', 'EUR', 'JPY', 'ETF', 'IPO'
        }
        
        for msg in messages:
            content = msg.get("content", "")
            matches = re.findall(symbol_pattern, content)
            
            for match in matches:
                if match not in exclude and len(match) >= 2:
                    symbols.add(match)
        
        return list(symbols)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_context_compressor(
    token_threshold: int = 100000,
    strategy: str = "smart_summary"
) -> ContextCompressor:
    """
    Factory function to create ContextCompressor
    
    Args:
        token_threshold: When to trigger compaction
        strategy: One of: keep_last_n, token_truncation, smart_summary, recursive_summary
    """
    strategy_map = {
        "keep_last_n": CompactionStrategy.KEEP_LAST_N,
        "token_truncation": CompactionStrategy.TOKEN_TRUNCATION,
        "smart_summary": CompactionStrategy.SMART_SUMMARY,
        "recursive_summary": CompactionStrategy.RECURSIVE_SUMMARY
    }
    
    config = CompactionConfig(
        token_threshold=token_threshold,
        strategy=strategy_map.get(strategy, CompactionStrategy.SMART_SUMMARY)
    )
    
    return ContextCompressor(config=config)