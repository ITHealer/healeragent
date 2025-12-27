"""
Simple Mode Handler - Fast Path for Conversational Queries

Bypasses the full planning pipeline for simple queries:
- Greetings, thanks, goodbye
- Simple acknowledgments
- Definition/explanation questions
- General knowledge questions

Performance Target: < 1-2 seconds response time
LLM Calls: 1 (direct response)
Tools: 0
"""

from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType


# ============================================================================
# DATA CLASSES
# ============================================================================

class SimpleResponseType(Enum):
    """Types of simple responses"""
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    ACKNOWLEDGMENT = "acknowledgment"
    DEFINITION = "definition"
    GENERAL_KNOWLEDGE = "general_knowledge"


@dataclass
class SimpleResponse:
    """Response from Simple Mode handler"""
    content: str
    response_type: SimpleResponseType
    tokens_used: int
    response_time_ms: int
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "response_type": self.response_type.value,
            "tokens_used": self.tokens_used,
            "response_time_ms": self.response_time_ms,
            "language": self.language
        }


# ============================================================================
# RESPONSE TEMPLATES
# ============================================================================

class SimpleResponseTemplates:
    """
    Quick response templates for deterministic replies

    For greetings/thanks/goodbye, we can respond without LLM call
    for maximum speed. Only use LLM for knowledge questions.
    """

    GREETINGS = {
        "vi": [
            "Xin chào! Tôi là trợ lý tài chính AI của bạn. Tôi có thể giúp bạn phân tích cổ phiếu, xem giá, chỉ báo kỹ thuật và tin tức thị trường. Bạn cần hỗ trợ gì hôm nay?",
            "Chào bạn! Tôi sẵn sàng hỗ trợ bạn với các phân tích tài chính. Hãy cho tôi biết mã cổ phiếu hoặc câu hỏi của bạn nhé!",
        ],
        "en": [
            "Hello! I'm your AI financial assistant. I can help you analyze stocks, check prices, technical indicators, and market news. What can I help you with today?",
            "Hi there! I'm ready to assist you with financial analysis. Let me know the stock symbol or your question!",
        ],
        "zh": [
            "您好！我是您的AI金融助手。我可以帮助您分析股票、查看价格、技术指标和市场新闻。今天有什么可以帮您的？",
        ]
    }

    THANKS = {
        "vi": [
            "Không có gì! Nếu bạn cần thêm thông tin về thị trường hoặc cổ phiếu nào, cứ hỏi tôi nhé!",
            "Rất vui được giúp đỡ! Hãy liên hệ nếu cần thêm phân tích nào.",
        ],
        "en": [
            "You're welcome! If you need more information about any stock or market, just let me know!",
            "Happy to help! Feel free to ask if you need more analysis.",
        ],
        "zh": [
            "不客气！如果您需要更多股票或市场信息，随时问我！",
        ]
    }

    GOODBYE = {
        "vi": [
            "Tạm biệt! Chúc bạn giao dịch thành công. Hẹn gặp lại!",
            "Chào bạn! Nếu cần hỗ trợ thêm, đừng ngại quay lại nhé!",
        ],
        "en": [
            "Goodbye! Wishing you successful trading. See you next time!",
            "Take care! Feel free to come back if you need more help!",
        ],
        "zh": [
            "再见！祝您交易顺利。下次见！",
        ]
    }

    ACKNOWLEDGMENT = {
        "vi": [
            "Được rồi! Bạn cần tôi giúp gì tiếp theo?",
            "OK! Còn điều gì khác tôi có thể hỗ trợ không?",
        ],
        "en": [
            "Got it! What else can I help you with?",
            "OK! Is there anything else you'd like to know?",
        ],
        "zh": [
            "好的！还有什么我可以帮您的吗？",
        ]
    }


# ============================================================================
# SIMPLE MODE HANDLER
# ============================================================================

class SimpleModeHandler(LoggerMixin):
    """
    Fast-path handler for simple/conversational queries

    Bypasses full planning pipeline for maximum speed:
    - Deterministic responses for greetings/thanks (no LLM)
    - Single LLM call for knowledge questions

    Usage:
        handler = SimpleModeHandler()

        # For streaming
        async for chunk in handler.handle_stream(query, hint="greeting"):
            yield chunk

        # For non-streaming
        response = await handler.handle(query, hint="definition")
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        provider_type: ProviderType = ProviderType.OPENAI
    ):
        """
        Initialize Simple Mode Handler

        Args:
            model_name: Model for knowledge questions (small/fast model preferred)
            provider_type: LLM provider
        """
        super().__init__()

        self.model_name = model_name
        self.provider_type = provider_type

        self.llm_provider = LLMGeneratorProvider()
        self.api_key = ModelProviderFactory._get_api_key(provider_type)

        self.logger.info(f"[SIMPLE:INIT] SimpleModeHandler initialized with model={model_name}")

    # ========================================================================
    # MAIN HANDLER METHODS
    # ========================================================================

    async def handle(
        self,
        query: str,
        hint: Optional[str] = None,
        language: str = "en",
        core_memory: Optional[Dict[str, Any]] = None
    ) -> SimpleResponse:
        """
        Handle simple query (non-streaming)

        Args:
            query: User query
            hint: Response type hint from router (greeting, thanks, etc.)
            language: Detected language
            core_memory: Optional user context

        Returns:
            SimpleResponse with content and metadata
        """
        start_time = datetime.now()

        self.logger.info(f"[SIMPLE] Handling query with hint={hint}, lang={language}")

        try:
            response_type = self._parse_hint(hint)

            # Deterministic responses (no LLM needed)
            if response_type in [
                SimpleResponseType.GREETING,
                SimpleResponseType.THANKS,
                SimpleResponseType.GOODBYE,
                SimpleResponseType.ACKNOWLEDGMENT
            ]:
                content = self._get_template_response(response_type, language)
                return SimpleResponse(
                    content=content,
                    response_type=response_type,
                    tokens_used=0,  # No LLM call
                    response_time_ms=self._elapsed_ms(start_time),
                    language=language
                )

            # LLM-based responses for knowledge questions
            content, tokens = await self._generate_llm_response(
                query=query,
                response_type=response_type,
                language=language,
                core_memory=core_memory
            )

            return SimpleResponse(
                content=content,
                response_type=response_type,
                tokens_used=tokens,
                response_time_ms=self._elapsed_ms(start_time),
                language=language
            )

        except Exception as e:
            self.logger.error(f"[SIMPLE] Error handling query: {e}", exc_info=True)

            # Fallback response
            return SimpleResponse(
                content=self._get_error_response(language),
                response_type=SimpleResponseType.GENERAL_KNOWLEDGE,
                tokens_used=0,
                response_time_ms=self._elapsed_ms(start_time),
                language=language
            )

    async def handle_stream(
        self,
        query: str,
        hint: Optional[str] = None,
        language: str = "en",
        core_memory: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle simple query with streaming

        Yields:
            Stream events: {type: "text_delta", content: "..."}
        """
        start_time = datetime.now()
        response_type = self._parse_hint(hint)

        self.logger.info(f"[SIMPLE:STREAM] Handling with hint={hint}, lang={language}")

        # Deterministic responses (instant, no streaming needed)
        if response_type in [
            SimpleResponseType.GREETING,
            SimpleResponseType.THANKS,
            SimpleResponseType.GOODBYE,
            SimpleResponseType.ACKNOWLEDGMENT
        ]:
            content = self._get_template_response(response_type, language)

            # Emit single chunk
            yield {
                "type": "text_delta",
                "content": content,
                "is_final": True
            }

            yield {
                "type": "done",
                "response_type": response_type.value,
                "tokens_used": 0,
                "response_time_ms": self._elapsed_ms(start_time)
            }
            return

        # LLM streaming for knowledge questions
        async for event in self._stream_llm_response(
            query=query,
            response_type=response_type,
            language=language,
            core_memory=core_memory
        ):
            yield event

        yield {
            "type": "done",
            "response_type": response_type.value,
            "response_time_ms": self._elapsed_ms(start_time)
        }

    # ========================================================================
    # LLM RESPONSE GENERATION
    # ========================================================================

    async def _generate_llm_response(
        self,
        query: str,
        response_type: SimpleResponseType,
        language: str,
        core_memory: Optional[Dict[str, Any]]
    ) -> tuple[str, int]:
        """Generate response using LLM"""

        system_prompt = self._build_system_prompt(language, core_memory)
        user_prompt = self._build_user_prompt(query, response_type)

        response = await self.llm_provider.generate_response(
            model_name=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            provider_type=self.provider_type,
            api_key=self.api_key
        )

        content = response.get("content", "") if isinstance(response, dict) else str(response)
        tokens = response.get("usage", {}).get("total_tokens", 0) if isinstance(response, dict) else 0

        return content, tokens

    async def _stream_llm_response(
        self,
        query: str,
        response_type: SimpleResponseType,
        language: str,
        core_memory: Optional[Dict[str, Any]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response using LLM"""

        system_prompt = self._build_system_prompt(language, core_memory)
        user_prompt = self._build_user_prompt(query, response_type)

        try:
            async for chunk in self.llm_provider.generate_response_stream(
                model_name=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
                provider_type=self.provider_type,
                api_key=self.api_key
            ):
                if isinstance(chunk, dict):
                    content = chunk.get("content", "")
                else:
                    content = str(chunk)

                if content:
                    yield {
                        "type": "text_delta",
                        "content": content,
                        "is_final": False
                    }

        except Exception as e:
            self.logger.error(f"[SIMPLE:STREAM] Error: {e}")
            yield {
                "type": "error",
                "content": self._get_error_response(language)
            }

    # ========================================================================
    # PROMPT BUILDING
    # ========================================================================

    def _build_system_prompt(
        self,
        language: str,
        core_memory: Optional[Dict[str, Any]]
    ) -> str:
        """Build system prompt for simple mode"""

        lang_instruction = {
            "vi": "Trả lời hoàn toàn bằng tiếng Việt.",
            "en": "Respond entirely in English.",
            "zh": "完全用中文回复。"
        }.get(language, "Respond in the same language as the user's query.")

        memory_context = ""
        if core_memory:
            user_info = core_memory.get("human", "")
            if user_info:
                memory_context = f"""
<user_context>
{user_info[:500]}
</user_context>
"""

        return f"""<role>
You are a helpful AI financial assistant. You provide clear, accurate, and concise information.
</role>

<language_instruction>
{lang_instruction}
</language_instruction>
{memory_context}
<guidelines>
1. Be concise but informative
2. Use bullet points for lists
3. Include relevant examples when helpful
4. For financial concepts, explain in simple terms
5. If unsure, acknowledge limitations
6. Keep responses under 200 words for simple questions
</guidelines>"""

    def _build_user_prompt(
        self,
        query: str,
        response_type: SimpleResponseType
    ) -> str:
        """Build user prompt based on response type"""

        if response_type == SimpleResponseType.DEFINITION:
            return f"""<task>
Explain this financial concept clearly and concisely.
</task>

<query>
{query}
</query>

<format>
- Start with a clear definition
- Add 1-2 practical examples if helpful
- Keep it under 150 words
</format>"""

        # General knowledge
        return f"""<task>
Answer this question helpfully and concisely.
</task>

<query>
{query}
</query>

<format>
- Direct answer first
- Brief explanation if needed
- Keep it concise
</format>"""

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _parse_hint(self, hint: Optional[str]) -> SimpleResponseType:
        """Parse hint string to response type"""
        hint_map = {
            "greeting": SimpleResponseType.GREETING,
            "thanks": SimpleResponseType.THANKS,
            "goodbye": SimpleResponseType.GOODBYE,
            "acknowledgment": SimpleResponseType.ACKNOWLEDGMENT,
            "definition": SimpleResponseType.DEFINITION,
        }
        return hint_map.get(hint, SimpleResponseType.GENERAL_KNOWLEDGE)

    def _get_template_response(
        self,
        response_type: SimpleResponseType,
        language: str
    ) -> str:
        """Get template response for deterministic replies"""
        import random

        templates = {
            SimpleResponseType.GREETING: SimpleResponseTemplates.GREETINGS,
            SimpleResponseType.THANKS: SimpleResponseTemplates.THANKS,
            SimpleResponseType.GOODBYE: SimpleResponseTemplates.GOODBYE,
            SimpleResponseType.ACKNOWLEDGMENT: SimpleResponseTemplates.ACKNOWLEDGMENT,
        }

        template_dict = templates.get(response_type, {})
        responses = template_dict.get(language, template_dict.get("en", ["Hello!"]))

        return random.choice(responses)

    def _get_error_response(self, language: str) -> str:
        """Get error fallback response"""
        error_responses = {
            "vi": "Xin lỗi, tôi gặp sự cố khi xử lý yêu cầu. Vui lòng thử lại.",
            "en": "Sorry, I encountered an issue processing your request. Please try again.",
            "zh": "抱歉，处理您的请求时出现问题。请重试。"
        }
        return error_responses.get(language, error_responses["en"])

    def _elapsed_ms(self, start_time: datetime) -> int:
        """Calculate elapsed time in milliseconds"""
        return int((datetime.now() - start_time).total_seconds() * 1000)


# ============================================================================
# SINGLETON & FACTORY
# ============================================================================

_handler_instance: Optional[SimpleModeHandler] = None


def get_simple_mode_handler(
    model_name: str = "gpt-4.1-nano",
    provider_type: ProviderType = ProviderType.OPENAI
) -> SimpleModeHandler:
    """
    Get singleton SimpleModeHandler instance

    Args:
        model_name: Model for knowledge questions
        provider_type: LLM provider

    Returns:
        SimpleModeHandler singleton instance
    """
    global _handler_instance

    if _handler_instance is None:
        _handler_instance = SimpleModeHandler(
            model_name=model_name,
            provider_type=provider_type
        )

    return _handler_instance
