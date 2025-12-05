from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@dataclass
class TranslationRequest:
    """Request model for translation API"""
    source_text: str
    source_lang: str
    target_lang: str
    context: Optional[str] = None
    country: Optional[str] = None
    emotions: Optional[List[str]] = None
    
    # Processing options
    enable_reflection: bool = True
    max_tokens_per_chunk: int = 1000
    
    def validate(self):
        """Validation"""
        if not self.source_text or not self.source_text.strip():
            raise ValueError("source_text cannot be empty")
        if self.source_lang == self.target_lang:
            raise ValueError("source_lang and target_lang must be different")

@dataclass
class TokenUsage:
    """Token usage tracking"""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    
    def __post_init__(self):
        self.total_tokens = self.input_tokens + self.output_tokens
    
    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens
        }

@dataclass
class TranslationMetadata:
    """Metadata"""
    processing_time: float
    chunks_count: int
    source_text_length: int
    translated_text_length: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "processing_time": self.processing_time,
            "chunks_count": self.chunks_count,
            "source_text_length": self.source_text_length,
            "translated_text_length": self.translated_text_length,
            "timestamp": self.timestamp
        }

@dataclass
class TranslationResponse:
    """Response"""
    translated_text: str
    source_lang: str
    target_lang: str
    token_usage: TokenUsage
    metadata: TranslationMetadata
    initial_translation: Optional[str] = None
    reflection: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "translated_text": self.translated_text,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "token_usage": self.token_usage.to_dict(),
            "metadata": self.metadata.to_dict(),
            "initial_translation": self.initial_translation,
            "reflection": self.reflection
        }