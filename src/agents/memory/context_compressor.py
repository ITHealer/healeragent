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
    
    max_context_tokens: int = 80000       # Max context window size
    trigger_percent: float = 80.0         # Trigger compaction at this percentage

    # Thresholds
    token_threshold: int = 80000           # Trigger at this token count
    token_target: int = 20000              # Target after compaction
    message_threshold: int = 10            # Trigger at this message count
    
    # Strategy
    strategy: CompactionStrategy = CompactionStrategy.SMART_SUMMARY
    
    # Retention
    retention_window: int = 4              # Keep last N messages unchanged during compaction
    preserve_system: bool = True           # Always keep system prompt
    
    # Summary settings
    max_summary_tokens: int = 1200         # Max tokens for summary
    summary_model: str = "gpt-4.1-nano"    # Model for summarization
    
    # Advanced
    preserve_keywords: List[str] = field(default_factory=list)  # Keywords to preserve
    response_reserve: int = 3000           # Reserve tokens for LLM response

    def __post_init__(self):
        # Auto-calculate threshold based on percentage
        calculated_threshold = int(self.max_context_tokens * (self.trigger_percent / 100))

        # Only update if using default value (80000)
        if self.token_threshold == 80000:
            self.token_threshold = calculated_threshold

            
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
    usage_percent: float
    
    # Percentages
    usage_percent: float
    
    # Thresholds
    threshold_tokens: int
    needs_compaction: bool
    
    # Details
    token_breakdown: Dict[str, int] = field(default_factory=dict)


# Common words to exclude from symbol detection
SYMBOL_EXCLUSIONS = {
    # Common English words (1-5 chars, uppercase)
    'I', 'A', 'THE', 'AND', 'OR', 'NOT', 'IS', 'ARE', 'WAS', 'BE',
    'TO', 'OF', 'IN', 'FOR', 'ON', 'WITH', 'AS', 'AT', 'BY', 'AN',
    'IT', 'IF', 'SO', 'UP', 'DO', 'NO', 'HE', 'WE', 'MY', 'OK',
    'AM', 'PM', 'VS', 'RE', 'FYI', 'TBD', 'NA', 'NB',
    
    # Tech/Finance acronyms (not stock symbols)
    'API', 'LLM', 'AI', 'ML', 'NLP', 'GPT', 'CPU', 'GPU', 'RAM',
    'USD', 'EUR', 'JPY', 'GBP', 'CNY', 'KRW', 'VND', 'THB',
    'CEO', 'CFO', 'CTO', 'COO', 'CMO', 'VP', 'EVP', 'SVP',
    'ETF', 'IPO', 'ESG', 'ROI', 'ROE', 'ROA', 'EPS', 'PE',
    'YTD', 'QTD', 'MTD', 'YOY', 'MOM', 'QOQ',
    'NYSE', 'NASDAQ', 'SEC', 'FINRA', 'FDIC',
    'BTC', 'ETH', 'USDT', 'USDC',  # Crypto symbols usually handled separately
    
    # Common abbreviations
    'USA', 'UK', 'EU', 'UN', 'NATO', 'WHO', 'IMF',
    'LLC', 'INC', 'LTD', 'PLC', 'AG', 'SA', 'NV',
    
    # HTML/Code related
    'HTML', 'CSS', 'JS', 'SQL', 'JSON', 'XML', 'HTTP', 'HTTPS',
    'GET', 'POST', 'PUT', 'DELETE', 'PATCH',
}

# Pattern for potential stock symbols (1-5 uppercase alphanumeric, starts with letter)
SYMBOL_PATTERN = re.compile(r'\b[A-Z][A-Z0-9]{0,14}\b') # r'\b[A-Z][A-Z0-9]{0,14}\b'


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
        self._token_counter = None

        self.logger.info(
            f"[CONTEXT COMPRESSOR] Initialized with strategy={self.config.strategy.value}, "
            f"threshold={self.config.token_threshold} tokens"
        )
    
    # ========================================================================
    # CONTEXT ANALYSIS
    # ========================================================================
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        if not text:
            return 0
        
        if self._token_counter is None:
            try:
                from src.helpers.token_counter import TokenCounter
                self._token_counter = TokenCounter()
            except ImportError:
                pass
        
        if self._token_counter:
            return self._token_counter.count_tokens(text)
        
        # Fallback: ~4 chars per token
        return len(text) // 4

    def get_context_stats(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        additional_context: str = ""
    ) -> ContextStats:
        """
        Calculate context statistics with percentage usage
        
        Args:
            messages: Conversation messages
            system_prompt: System prompt
            additional_context: Core memory + summary + working memory
            
        Returns:
            ContextStats with percentage usage
        """
        # Count system prompt tokens
        system_tokens = self._count_tokens(system_prompt)
        
        # Count additional context tokens
        additional_tokens = self._count_tokens(additional_context)
        
        # Count message tokens
        history_tokens = 0
        token_breakdown = {}
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            msg_tokens = self._count_tokens(content)
            history_tokens += msg_tokens
            token_breakdown[f"{role}_{i}"] = msg_tokens
        
        # Total tokens
        total_tokens = system_tokens + additional_tokens + history_tokens
        
        # Calculate percentage usage
        usage_percent = (total_tokens / self.config.max_context_tokens) * 100
        
        # Check against percentage threshold
        needs_compaction = (
            total_tokens > self.config.token_threshold or  # Over token limit
            (len(messages) > self.config.message_threshold and usage_percent >= 50)  # Many messages + moderate usage
        )
        # needs_compaction = (
        #     usage_percent >= self.config.trigger_percent or
        #     len(messages) > self.config.message_threshold
        # )
        
        return ContextStats(
            total_tokens=total_tokens,
            message_count=len(messages),
            system_tokens=system_tokens + additional_tokens,
            history_tokens=history_tokens,
            usage_percent=round(usage_percent, 2),
            threshold_tokens=self.config.token_threshold,
            needs_compaction=needs_compaction,
            token_breakdown=token_breakdown
        )
    
    
    def should_compact(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = "",
        additional_context: str = ""
    ) -> Tuple[bool, ContextStats]:
        """
        Check if compaction is needed based on 90% threshold

        Args:
            messages: Conversation messages
            system_prompt: System prompt
            additional_context: Core memory + summary + working memory
            
        Returns:
            (needs_compaction, stats)
        """
        # Get stats with percentage
        stats = self.get_context_stats(
            messages=messages,
            system_prompt=system_prompt,
            additional_context=additional_context
        )
        
        # Log stats
        if stats.needs_compaction:
            self.logger.warning(
                f"[CONTEXT COMPRESSOR] COMPACTION NEEDED: "
                f"{stats.usage_percent:.1f}% usage "
                f"({stats.total_tokens:,}/{self.config.max_context_tokens:,} tokens), "
                f"{stats.message_count} messages"
            )
        else:
            self.logger.info(
                f"[CONTEXT COMPRESSOR] Context OK: "
                f"{stats.usage_percent:.1f}% usage "
                f"({stats.total_tokens:,} tokens)"
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
                f"[CONTEXT COMPRESSOR] Compaction complete: "
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
        messages: List[Dict[str, str]],
        preserve_keywords: List[str] = None
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Keep last N messages strategy"""
        n = self.config.retention_window
        
        if len(messages) <= n:
            return messages, None
        
        # Keep last N messages
        kept_messages = messages[-n:]
        
        # Create brief summary of removed messages
        removed_count = len(messages) - n
        summary = f"[{removed_count} earlier messages summarized]"
        
        if preserve_keywords:
            summary += f" Keywords preserved: {', '.join(preserve_keywords[:5])}"
        
        return kept_messages, summary
    
    async def _compact_token_truncation(
        self,
        messages: List[Dict[str, str]],
        preserve_keywords: List[str] = None
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Token truncation strategy"""
        target_tokens = self.config.token_threshold // 2
        
        result_messages = []
        current_tokens = 0
        
        # Process from newest to oldest
        for msg in reversed(messages):
            content = msg.get("content", "")
            msg_tokens = self._count_tokens(content)
            
            if current_tokens + msg_tokens <= target_tokens:
                result_messages.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        summary = f"[Context truncated to fit {target_tokens:,} tokens]"
        
        return result_messages, summary
    
    async def _compact_smart_summary(
        self,
        messages: List[Dict[str, str]],
        preserve_keywords: List[str] = None
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        Smart summary strategy using LLM
        
        This creates an intelligent summary of older messages
        while keeping recent ones intact.
        """
        # Keep retention window messages unchanged
        retention = self.config.retention_window
        
        if len(messages) <= retention:
            return messages, None
        
        # Messages to summarize vs keep
        to_summarize = messages[:-retention]
        to_keep = messages[-retention:]
        
        # Build summary (would use LLM in production)
        # For now, create a structured summary
        summary_parts = []
        
        # Extract key information
        user_queries = []
        assistant_responses = []
        
        for msg in to_summarize:
            role = msg.get("role", "")
            content = msg.get("content", "")[:200]  # Truncate long content
            
            if role == "user":
                user_queries.append(content)
            elif role == "assistant":
                assistant_responses.append(content[:100])
        
        if user_queries:
            summary_parts.append(f"User discussed: {len(user_queries)} topics")
        
        if preserve_keywords:
            summary_parts.append(f"Symbols mentioned: {', '.join(preserve_keywords[:10])}")
        
        summary = " | ".join(summary_parts) if summary_parts else "[Earlier context summarized]"
        
        # Prepend summary as system context
        summary_message = {
            "role": "system",
            "content": f"<conversation_summary>\n{summary}\n</conversation_summary>"
        }
        
        result = [summary_message] + to_keep
        
        return result, summary
    
    async def _compact_recursive_summary(
        self,
        messages: List[Dict[str, str]],
        preserve_keywords: List[str] = None
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """Recursive summary strategy for very long conversations"""
        # Similar to smart_summary but with multiple levels
        return await self._compact_smart_summary(messages, preserve_keywords)
    
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
        all_symbols = []
        
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue
            
            # Find all potential symbols using pattern
            matches = SYMBOL_PATTERN.findall(content)
            
            for match in matches:
                # Skip exclusions (common words, acronyms)
                if match in SYMBOL_EXCLUSIONS:
                    continue
                
                # Skip if too short (likely not a symbol)
                if len(match) < 2:
                    continue
                
                all_symbols.append(match)
        
        # Deduplicate while preserving order
        seen = set()
        unique_symbols = []
        for sym in all_symbols:
            if sym not in seen:
                seen.add(sym)
                unique_symbols.append(sym)
        
        if unique_symbols:
            self.logger.info(
                f"[CONTEXT COMPRESSOR] Extracted {len(unique_symbols)} symbols: "
                f"{unique_symbols[:10]}{'...' if len(unique_symbols) > 10 else ''}"
            )
        
        return unique_symbols


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