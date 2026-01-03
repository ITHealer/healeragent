"""
Summary Generation Service
Uses LLM to create recursive summaries of conversations
Specialized for Stock/Crypto domain
"""

from typing import Optional, Dict, List, Any
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.token_counter import TokenCounter
from src.providers.provider_factory import ModelProviderFactory, ProviderType


class SummaryGenerationService(LoggerMixin):
    """
    Service for generating conversation summaries using LLM
    Implements recursive summarization pattern
    """
    
    # Use cheap, fast model for summarization
    MAX_SUMMARY_TOKENS = 1000  # Target summary length
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None
    ):
        """
        Initialize summary generation service
        
        Args:
            model_name: LLM model to use (default: gpt-4o-mini)
        """
        from src.utils.config import settings

        super().__init__()
        self.model_name = model_name or settings.SUMMARY_MODEL or "gpt-4.1-nano"
        self.provider_type = provider_type or settings.SUMMARY_PROVIDER or ProviderType.OPENAI
        self.llm_provider = LLMGeneratorProvider()
        self.token_counter = TokenCounter()
    
    
    async def generate_initial_summary(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Generate first summary from a batch of messages
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            provider_type: LLM provider
            
        Returns:
            Dict with summary_text and token_count
        """
        try:
            # Use defaults if not provided
            effective_model = model_name or self.model_name
            effective_provider = provider_type or self.provider_type

            # Build conversation text
            conversation_text = self._format_messages_for_summary(messages)

            # Create summarization prompt
            prompt = self._build_initial_summary_prompt(conversation_text)

            # Generate summary
            api_key = ModelProviderFactory._get_api_key(effective_provider)

            summary_messages = [
                {"role": "system", "content": self._get_summarization_system_prompt()},
                {"role": "user", "content": prompt}
            ]

            response = await self.llm_provider.generate_response(
                model_name=effective_model,
                messages=summary_messages,
                provider_type=effective_provider,
                api_key=api_key
            )
            
            summary_text = response.get('content', '')
            
            # Count tokens
            token_count = self.token_counter.count_tokens(summary_text)
            
            self.logger.info(
                f"Generated initial summary: {token_count} tokens from {len(messages)} messages"
            )
            
            return {
                'summary_text': summary_text,
                'token_count': token_count,
                'messages_summarized': len(messages)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating initial summary: {e}")
            return {
                'summary_text': '',
                'token_count': 0,
                'error': str(e)
            }
    
    
    async def generate_recursive_summary(
        self,
        previous_summary: str,
        new_messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate new summary by combining previous summary with new messages
        This is the RECURSIVE part
        
        Args:
            previous_summary: Existing summary text
            new_messages: New messages to add
            provider_type: LLM provider
            
        Returns:
            Dict with summary_text and token_count
        """
        try:
            # Use defaults if not provided
            effective_model = model_name or self.model_name
            effective_provider = provider_type or self.provider_type

            # Format new messages
            new_conversation = self._format_messages_for_summary(new_messages)

            # Create recursive summarization prompt
            prompt = self._build_recursive_summary_prompt(
                previous_summary,
                new_conversation
            )

            # Generate summary
            api_key = ModelProviderFactory._get_api_key(effective_provider)

            summary_messages = [
                {"role": "system", "content": self._get_summarization_system_prompt()},
                {"role": "user", "content": prompt}
            ]

            response = await self.llm_provider.generate_response(
                model_name=effective_model,
                messages=summary_messages,
                provider_type=effective_provider,
                api_key=api_key
            )
            
            summary_text = response.get('content', '')
            token_count = self.token_counter.count_tokens(summary_text)
            
            self.logger.info(
                f"Generated recursive summary: {token_count} tokens "
                f"(prev: {self.token_counter.count_tokens(previous_summary)}, "
                f"new msgs: {len(new_messages)})"
            )
            
            return {
                'summary_text': summary_text,
                'token_count': token_count,
                'messages_summarized': len(new_messages)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recursive summary: {e}")
            return {
                'summary_text': previous_summary,  # Fallback to previous
                'token_count': self.token_counter.count_tokens(previous_summary),
                'error': str(e)
            }
    
    def _get_summarization_system_prompt(self) -> str:
        """System prompt for summarization"""
        return """You are an AI assistant specialized in summarizing financial conversations about stocks and cryptocurrencies.

Your task is to create CONCISE, INFORMATIVE summaries that preserve:
1. **Key Facts**: Stock symbols, prices, quantities, dates
2. **User Preferences**: Investment style, risk tolerance, favorite assets
3. **Portfolio Information**: Holdings, entry prices, positions
4. **Analysis Discussions**: Technical/fundamental insights shared
5. **Decisions & Plans**: Trading decisions, future plans

CRITICAL RULES:
- Keep summaries under 1000 tokens
- Use bullet points for clarity
- Preserve NUMBERS and SYMBOLS exactly (VNM, 78000, 100 shares)
- Focus on ACTIONABLE information
- Maintain user's language preference (EN/VI)
- Remove redundant pleasantries and confirmations

OUTPUT FORMAT:
Use structured sections:
📊 **Market Discussion**: [Stocks/crypto discussed]
💼 **Portfolio Updates**: [Holdings, positions]
🎯 **User Preferences**: [Risk, style, interests]
📈 **Key Insights**: [Analysis, decisions]
🗣️ **Language**: [User's preferred language]"""
    
    
    def _build_initial_summary_prompt(self, conversation_text: str) -> str:
        """Build prompt for initial summarization"""
        return f"""Summarize this financial conversation concisely, preserving all important details:

{conversation_text}

Create a structured summary following the format in your system instructions.
Focus on facts, numbers, and user preferences."""
    
    
    def _build_recursive_summary_prompt(
        self,
        previous_summary: str,
        new_conversation: str
    ) -> str:
        """Build prompt for recursive summarization"""
        return f"""You have an EXISTING SUMMARY of earlier conversation:

--- PREVIOUS SUMMARY ---
{previous_summary}

--- NEW CONVERSATION ---
{new_conversation}

Task: Create an UPDATED SUMMARY that:
1. Keeps all important information from the previous summary
2. Adds new information from the new conversation
3. Removes any redundant or outdated information
4. Maintains the same structured format
5. Stays under 1000 tokens

If new information contradicts old information (e.g., portfolio update), keep the NEWER information."""
    
    
    def _format_messages_for_summary(
        self,
        messages: List[Dict[str, str]]
    ) -> str:
        """
        Format messages into readable conversation text
        
        Args:
            messages: List of message dicts
            
        Returns:
            Formatted conversation string
        """
        conversation_lines = []
        
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            
            if role == 'user':
                conversation_lines.append(f"User: {content}")
            elif role == 'assistant':
                conversation_lines.append(f"Assistant: {content}")
            elif role == 'system':
                continue  # Skip system messages
            else:
                conversation_lines.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(conversation_lines)
    
    
    def estimate_summary_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        return self.token_counter.count_tokens(text)
    
    
    def should_create_summary(
        self,
        message_count: int,
        threshold: int = 10
    ) -> bool:
        """
        Check if summary should be created
        
        Args:
            message_count: Current message count in session
            threshold: Message threshold for summarization
            
        Returns:
            Boolean indicating if summary should be created
        """
        return message_count >= threshold and message_count % threshold == 0