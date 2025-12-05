from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from src.translation_agent.agents.agent import AgentConfig
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.providers.base_provider import ModelProvider
from src.translation_agent.models.translation_models import TokenUsage
from src.translation_agent.helpers.text_chunker import SmartTextChunker
from src.utils.logger.custom_logging import LoggerMixin

@dataclass(kw_only=True)
class TranslationAgentConfig(AgentConfig):
    provider_type: str = ProviderType.OPENAI
    
    def __post_init__(self):
        if self.provider_type not in ProviderType.list():
            raise ValueError(f"Invalid provider_type: {self.provider_type}")

class TranslationAgent(LoggerMixin):
    """
    Features:
    - 3-phase translation: Initial → Reflect → Improve
    - Context-aware
    - Token tracking
    - Multi-provider support
    """
    
    def __init__(self, config: TranslationAgentConfig):
        super().__init__()
        
        self.config = config
        self.chunker = SmartTextChunker()
        
        # Create provider
        api_key = ModelProviderFactory._get_api_key(config.provider_type)
        self.provider: ModelProvider = ModelProviderFactory.create_provider(
            provider_type=config.provider_type,
            model_name=config.model_id,
            api_key=api_key
        )
        
        # Token tracking
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cached_tokens = 0
        self.session_reasoning_tokens = 0
    
    def _extract_token_usage(self, response: Dict) -> TokenUsage:
        """Extract token usage từ provider response"""
        raw_response = response.get("raw_response")
        
        if not raw_response or not hasattr(raw_response, 'usage'):
            return TokenUsage()
        
        usage = raw_response.usage
        input_tokens = getattr(usage, 'prompt_tokens', 0)
        output_tokens = getattr(usage, 'completion_tokens', 0)
        
        cached_tokens = 0
        if hasattr(usage, 'prompt_tokens_details'):
            cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
        
        reasoning_tokens = 0
        if hasattr(usage, 'completion_tokens_details'):
            reasoning_tokens = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)
        
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cached_tokens=cached_tokens
        )
    
    async def initial_translation(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None,
        emotions: Optional[List[str]] = None
    ) -> Tuple[str, TokenUsage]:
        """Phase 1: Initial translation"""
        system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}."
        
        # Build user message
        parts = [f"Translate from {source_lang} to {target_lang}."]
        
        if context and context.strip():
            parts.append(f"\nCONTEXT: {context}\nConsider this for appropriate vocabulary and tone.")
        
        if emotions and len(emotions) > 0:
            emotion_list = ", ".join(emotions)
            parts.append(f"\nEMOTIONS: Convey {emotion_list} in the translation.")
        
        parts.append(f"\nSource text:\n{source_text}")
        parts.append(f"\nProvide ONLY the translation in {target_lang}, nothing else.")
        
        user_message = "".join(parts)
        
        try:
            await self.provider.initialize()
            
            response = await self.provider.generate(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.config.temperature
            )
            
            translation = response.get("content", "").strip()
            token_usage = self._extract_token_usage(response)
            
            self.session_input_tokens += token_usage.input_tokens
            self.session_output_tokens += token_usage.output_tokens
            self.session_cached_tokens += token_usage.cached_tokens
            self.session_reasoning_tokens += token_usage.reasoning_tokens
            
            return translation, token_usage
            
        except Exception as e:
            self.logger.error(f"Initial translation failed: {e}")
            raise
    
    async def reflect_on_translation(
        self,
        source_text: str,
        translation: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None,
        country: Optional[str] = None,
        emotions: Optional[List[str]] = None
    ) -> Tuple[str, TokenUsage]:
        """Phase 2: Reflection"""
        system_message = f"You are an expert translation reviewer for {source_lang} to {target_lang}."
        
        parts = ["Review this translation and suggest improvements."]
        
        if context and context.strip():
            parts.append(f"\nCONTEXT: {context}")
        
        if country:
            parts.append(f"\nMatch the style of {target_lang} spoken in {country}.")
        
        parts.append(f"\n\n<SOURCE>\n{source_text}\n</SOURCE>")
        parts.append(f"\n<TRANSLATION>\n{translation}\n</TRANSLATION>")
        
        parts.append("\n\nEvaluate:")
        parts.append("\n(i) Accuracy - errors, omissions")
        parts.append("\n(ii) Fluency - grammar, flow")
        parts.append("\n(iii) Style - cultural appropriateness")
        parts.append("\n(iv) Terminology - consistency")
        
        if emotions and len(emotions) > 0:
            emotion_list = ", ".join(emotions)
            parts.append(f"\n(v) Emotional tone (convey: {emotion_list})")
        
        parts.append(f"\n\nWrite suggestions in {target_lang}.")
        
        user_message = "".join(parts)
        
        try:
            response = await self.provider.generate(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.config.temperature
            )
            
            reflection = response.get("content", "").strip()
            token_usage = self._extract_token_usage(response)
            
            self.session_input_tokens += token_usage.input_tokens
            self.session_output_tokens += token_usage.output_tokens
            self.session_cached_tokens += token_usage.cached_tokens
            self.session_reasoning_tokens += token_usage.reasoning_tokens
            
            return reflection, token_usage
            
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            raise
    
    async def improve_translation(
        self,
        source_text: str,
        initial_translation: str,
        reflection: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None,
        emotions: Optional[List[str]] = None
    ) -> Tuple[str, TokenUsage]:
        """Phase 3: Improvement"""
        system_message = f"You are an expert translator for {source_lang} to {target_lang}."
        
        parts = ["Improve the translation using the suggestions."]
        
        if context and context.strip():
            parts.append(f"\nCONTEXT: {context}")
        
        if emotions and len(emotions) > 0:
            emotion_list = ", ".join(emotions)
            parts.append(f"\nEMOTIONS: Convey {emotion_list}.")
        
        parts.append(f"\n\n<SOURCE>\n{source_text}\n</SOURCE>")
        parts.append(f"\n<INITIAL_TRANSLATION>\n{initial_translation}\n</INITIAL_TRANSLATION>")
        parts.append(f"\n<SUGGESTIONS>\n{reflection}\n</SUGGESTIONS>")
        parts.append(f"\n\nOutput ONLY the improved translation in {target_lang}, nothing else.")
        
        user_message = "".join(parts)
        
        try:
            response = await self.provider.generate(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.config.temperature
            )
            
            improved = response.get("content", "").strip()
            token_usage = self._extract_token_usage(response)
            
            self.session_input_tokens += token_usage.input_tokens
            self.session_output_tokens += token_usage.output_tokens
            self.session_cached_tokens += token_usage.cached_tokens
            self.session_reasoning_tokens += token_usage.reasoning_tokens
            
            return improved, token_usage
            
        except Exception as e:
            self.logger.error(f"Improvement failed: {e}")
            raise
    
    async def translate_single_chunk(
        self,
        source_text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None,
        country: Optional[str] = None,
        emotions: Optional[List[str]] = None,
        enable_reflection: bool = True
    ) -> Tuple[str, str, str]:
        """Translate single chunk - Returns: (final, initial, reflection)"""
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cached_tokens = 0
        self.session_reasoning_tokens = 0
        
        # Phase 1
        initial, _ = await self.initial_translation(
            source_text, source_lang, target_lang, context, emotions
        )
        
        if not enable_reflection:
            return initial, initial, ""
        
        # Phase 2
        reflection, _ = await self.reflect_on_translation(
            source_text, initial, source_lang, target_lang, context, country, emotions
        )
        
        # Phase 3
        final, _ = await self.improve_translation(
            source_text, initial, reflection, source_lang, target_lang, context, emotions
        )
        
        return final, initial, reflection
    
    async def translate_multi_chunks(
        self,
        chunks: List[str],
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None,
        country: Optional[str] = None,
        emotions: Optional[List[str]] = None,
        enable_reflection: bool = True
    ) -> Tuple[List[str], List[str], List[str]]:
        """Translate multiple chunks - Returns: (finals, initials, reflections)"""
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_cached_tokens = 0
        self.session_reasoning_tokens = 0
        
        finals = []
        initials = []
        reflections = []
        
        total = len(chunks)
        
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Translating chunk {i+1}/{total}")
            
            # Enhanced context
            enhanced_context = context or ""
            
            if i > 0:
                prev = chunks[i-1][-200:] if len(chunks[i-1]) > 200 else chunks[i-1]
                enhanced_context += f"\n\nPrevious: ...{prev}"
            
            if i < total - 1:
                next_chunk = chunks[i+1][:200] if len(chunks[i+1]) > 200 else chunks[i+1]
                enhanced_context += f"\n\nNext: {next_chunk}..."
            
            final, initial, reflection = await self.translate_single_chunk(
                chunk, source_lang, target_lang, enhanced_context, country, emotions, enable_reflection
            )
            
            finals.append(final)
            initials.append(initial)
            reflections.append(reflection)
        
        return finals, initials, reflections
    
    def get_token_usage(self) -> TokenUsage:
        """Get token usage"""
        return TokenUsage(
            input_tokens=self.session_input_tokens,
            output_tokens=self.session_output_tokens,
            reasoning_tokens=self.session_reasoning_tokens,
            cached_tokens=self.session_cached_tokens
        )