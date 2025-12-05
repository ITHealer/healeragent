import time

from src.translation_agent.agents.translation_agent import TranslationAgent, TranslationAgentConfig
from src.translation_agent.models.translation_models import (
    TranslationRequest,
    TranslationResponse,
    TranslationMetadata
)
from src.translation_agent.helpers.text_chunker import SmartTextChunker
from src.utils.logger.custom_logging import LoggerMixin

class TranslationService(LoggerMixin):
    
    def __init__(self):
        super().__init__()
        self.chunker = SmartTextChunker()
    
    async def translate(
        self,
        request: TranslationRequest,
        provider_type: str,
        model_name: str,
        temperature: float = 0.3
    ) -> TranslationResponse:
        
        start_time = time.time()
        
        # Validate
        request.validate()
        self.logger.info(f"Translation: {request.source_lang} → {request.target_lang}")
        self.logger.info(f"Provider: {provider_type}, Model: {model_name}")
        
        # Create agent for this request
        agent_config = TranslationAgentConfig(
            provider_type=provider_type,
            base_url="", 
            model_id=model_name,
            temperature=temperature,
            concurrent=1,
            timeout=120,
            retry=2
        )
        agent = TranslationAgent(agent_config)
        
        # Smart chunking
        chunks, total_tokens = self.chunker.chunk_with_context_preservation(
            request.source_text,
            request.max_tokens_per_chunk
        )
        
        self.logger.info(f"Chunked: {len(chunks)} chunks, {total_tokens} tokens")
        
        # Translation
        if len(chunks) == 1:
            # Single chunk
            self.logger.info("Single chunk translation")
            final, initial, reflection = await agent.translate_single_chunk(
                chunks[0],
                request.source_lang,
                request.target_lang,
                request.context,
                request.country,
                request.emotions,
                request.enable_reflection
            )
            
            finals = [final]
            initials = [initial] if request.enable_reflection else None
            reflections = [reflection] if request.enable_reflection else None
            
        else:
            # Multi chunks
            self.logger.info(f"Multi-chunk translation: {len(chunks)} chunks")
            finals, initials, reflections = await agent.translate_multi_chunks(
                chunks,
                request.source_lang,
                request.target_lang,
                request.context,
                request.country,
                request.emotions,
                request.enable_reflection
            )
        
        # Aggregate results
        final_translation = "".join(finals)
        initial_translation = "".join(initials) if initials else None
        reflection_text = "\n\n".join([r for r in reflections if r]) if reflections else None
        
        # Get token usage
        token_usage = agent.get_token_usage()
        
        # Build metadata
        processing_time = time.time() - start_time
        metadata = TranslationMetadata(
            processing_time=processing_time,
            chunks_count=len(chunks),
            source_text_length=len(request.source_text),
            translated_text_length=len(final_translation)
        )
        
        self.logger.info(f"Complete in {processing_time:.2f}s")
        self.logger.info(f"Tokens: {token_usage.total_tokens} total ({token_usage.input_tokens} in, {token_usage.output_tokens} out)")
        
        return TranslationResponse(
            translated_text=final_translation,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            token_usage=token_usage,
            metadata=metadata,
            initial_translation=initial_translation,
            reflection=reflection_text
        )