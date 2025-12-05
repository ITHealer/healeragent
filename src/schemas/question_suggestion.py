from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class UserLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class QuestionType(str, Enum):
    TECHNICAL = "technical_expert"
    CRYPTO = "crypto_analyst"
    FUNDAMENTAL = "fundamental_guru"
    SENTIMENT = "sentiment_analyzer"

class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"

class QuestionSuggestionRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID để lấy chat history")
    k: int = Field(5, ge=1, le=10, description="Số lượng câu hỏi cần gợi ý")
    user_level: UserLevel = Field(UserLevel.BEGINNER, description="Level kinh nghiệm của user")
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
    provider_type: str = Field("openai", description="Provider type: openai, ollama, gemini")
    include_chat_context: Optional[bool] = Field(True, description="Include chat history context")
    include_summary: Optional[bool] = Field(True, description="Include conversation summary")

class ConversationSummary(BaseModel):
    session_id: str
    summary: str
    key_topics: List[str]
    mentioned_symbols: List[str]
    last_updated: datetime
    message_count: int
    
class SuggestedQuestion(BaseModel):
    question: str = Field(..., description="Câu hỏi gợi ý")
    category: str = Field(..., description="Danh mục câu hỏi (technical, fundamental, news, etc.)")
    relevance_reason: str = Field(..., description="Lý do câu hỏi này phù hợp")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score for the suggestion")
    context_based: bool = Field(False, description="Whether question is based on conversation context")

class QuestionSuggestionResponse(BaseModel):
    questions: List[SuggestedQuestion]
    context_used: Dict[str, Any]
    summary_available: bool

class QuestionSuggestionRequest_v2(BaseModel):
    question_type: str = Field(
        ...,
        description="Type of analysis: technical_expert, crypto_analyst, fundamental_guru, sentiment_analyzer"
    )
    user_level: str = Field(
        default=UserLevel.BEGINNER,
        description="User expertise level: beginner, intermediate, advanced"
    )
    asset_type: str = Field(
        ...,
        description="Asset type filter: stock or crypto - determines which market to focus suggestions on"
    )
    model_name: str = Field(
        default="gpt-4.1-nano-2025-04-14",
        description="LLM model to use for generating suggestions"
    )
    provider_type: str = Field(
        default="openai",
        description="LLM provider: openai, gemini, ollama"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Chat session ID for context (optional)"
    )
    k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of questions to generate (1-20)"
    )


class SuggestedQuestion_v2(BaseModel):
    """Model for a suggested question"""
    question: str = Field(
        ...,
        description="The suggested question text"
    )
    category: str = Field(
        ...,
        description="Question category: technical, fundamental, sentiment, market, crypto, general"
    )
    relevance_reason: str = Field(
        ...,
        description="Why this question is relevant/suggested"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Stock/crypto symbol related to the question (if applicable)"
    )
    priority: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Priority score 1-10 (higher is more important)"
    )

# Category definitions for reference
QUESTION_CATEGORIES = {
    "technical": "Technical analysis (charts, indicators, patterns)",
    "fundamental": "Fundamental analysis (financials, valuations)",
    "sentiment": "Market sentiment and news analysis",
    "market": "Overall market conditions and trends",
    "crypto": "Cryptocurrency specific analysis",
    "general": "General investment questions"
}   