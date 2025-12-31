"""
Token Counter Utility
Provides accurate token counting for different LLM models

PRODUCTION NOTES:
- Use get_token_counter() singleton to avoid memory leaks
- TokenCounter caches tiktoken encoding for efficiency
"""

import tiktoken
from typing import List, Dict, Optional
from src.utils.logger.custom_logging import LoggerMixin


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
_token_counter_instance: Optional['TokenCounter'] = None


def get_token_counter() -> 'TokenCounter':
    """
    Get singleton instance of TokenCounter.

    TokenCounter uses tiktoken which caches encodings, so sharing
    a single instance is more efficient for production.

    Returns:
        TokenCounter singleton instance
    """
    global _token_counter_instance

    if _token_counter_instance is None:
        _token_counter_instance = TokenCounter()

    return _token_counter_instance


class TokenCounter(LoggerMixin):
    """Token counter supporting multiple encoding schemes"""
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter
        
        Args:
            model: Model name for determining encoding (default: gpt-4)
        """
        super().__init__()
        self.model = model
        
        # Get appropriate encoding
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base (used by GPT-4, GPT-3.5-turbo)
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.logger.warning(
                f"Model {model} not found in tiktoken, using cl100k_base encoding"
            )
    
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string
        
        Args:
            text: Input text
            
        Returns:
            int: Number of tokens
        """
        if not text:
            return 0
        
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            self.logger.error(f"Error counting tokens: {e}")
            # Rough approximation fallback
            return int(len(text.split()) * 1.3)
    
    
    def count_messages_tokens(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None
    ) -> int:
        """
        Count tokens in a list of messages (OpenAI chat format)
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (optional, uses self.model if not provided)
            
        Returns:
            int: Total token count for messages
        """
        if not messages:
            return 0
        
        model = model or self.model
        
        try:
            # Tokens per message overhead
            # GPT-4/GPT-3.5-turbo: every message has overhead
            if "gpt-4" in model or "gpt-3.5" in model:
                tokens_per_message = 3
                tokens_per_name = 1
            else:
                tokens_per_message = 3
                tokens_per_name = 1
            
            num_tokens = 0
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += self.count_tokens(str(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            
            # Every reply is primed with assistant message
            num_tokens += 3
            
            return num_tokens
            
        except Exception as e:
            self.logger.error(f"Error counting message tokens: {e}")
            # Fallback: sum all content tokens
            return sum(self.count_tokens(msg.get('content', '')) for msg in messages)
    
    
    def truncate_to_token_limit(
        self, 
        text: str, 
        max_tokens: int, 
        suffix: str = "..."
    ) -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            suffix: Suffix to add when truncated (default: "...")
            
        Returns:
            str: Truncated text
        """
        if not text:
            return text
        
        current_tokens = self.count_tokens(text)
        
        if current_tokens <= max_tokens:
            return text
        
        # Reserve tokens for suffix
        suffix_tokens = self.count_tokens(suffix)
        target_tokens = max_tokens - suffix_tokens
        
        if target_tokens <= 0:
            return suffix
        
        # Binary search for optimal truncation point
        tokens = self.encoding.encode(text)
        truncated_tokens = tokens[:target_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)
        
        return truncated_text + suffix
    
    
    def estimate_cost(
        self, 
        input_tokens: int, 
        output_tokens: int,
        model: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Estimate API cost based on token counts
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model name (optional, uses self.model if not provided)
            
        Returns:
            Dict with cost breakdown
        """
        model = model or self.model
        
        # Pricing per 1M tokens (as of 2025)
        pricing = {
            "gpt-4": {"input": 30.0, "output": 60.0},
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        }
        
        # Find matching pricing
        model_pricing = None
        for key in pricing.keys():
            if key in model.lower():
                model_pricing = pricing[key]
                break
        
        if not model_pricing:
            # Default fallback
            model_pricing = {"input": 10.0, "output": 30.0}
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
        output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "model": model
        }
    
    
    def get_context_window_size(self, model: Optional[str] = None) -> int:
        """
        Get context window size for a model
        
        Args:
            model: Model name (optional, uses self.model if not provided)
            
        Returns:
            int: Context window size in tokens
        """
        model = model or self.model
        
        context_windows = {
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 16385,
            "claude-3": 200000,
            "claude-2": 100000,
        }
        
        for key, size in context_windows.items():
            if key in model.lower():
                return size
        
        # Default fallback
        return 8192
    
    
    def check_context_fit(
        self, 
        texts: List[str], 
        model: Optional[str] = None,
        reserve_tokens: int = 1000
    ) -> Dict:
        """
        Check if multiple texts fit within context window
        
        Args:
            texts: List of text strings
            model: Model name (optional, uses self.model if not provided)
            reserve_tokens: Tokens to reserve for response (default: 1000)
            
        Returns:
            Dict with fit analysis
        """
        model = model or self.model
        context_window = self.get_context_window_size(model)
        available_tokens = context_window - reserve_tokens
        
        token_counts = [self.count_tokens(text) for text in texts]
        total_tokens = sum(token_counts)
        
        return {
            "fits": total_tokens <= available_tokens,
            "total_tokens": total_tokens,
            "available_tokens": available_tokens,
            "context_window": context_window,
            "reserve_tokens": reserve_tokens,
            "usage_percent": round((total_tokens / available_tokens) * 100, 2),
            "individual_counts": token_counts,
            "overflow_tokens": max(0, total_tokens - available_tokens)
        }