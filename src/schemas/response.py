from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from src.utils.config import settings
from src.providers.provider_factory import ProviderType
from src.utils.constants import APIModelName

class BasicResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict | list | str] = None

class ChatResponse(BaseModel):
    id: str
    role: str = "assistant"
    content: str

class BasicResponseDelete(BaseModel):
    Status: str
    Message: str
    Data: Optional[dict | list | str] = None

# =============================================================================
# region Schema for Tools
# =============================================================================
# General Chat
class GeneralChatBot(BaseModel):
    session_id: str
    question_input: str
    target_language: Optional[str] = None
    model_name: str = Field(APIModelName.GPT41Nano, description="Model name")
    provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")
    collection_name: str = ''
    use_multi_collection: bool = False
    enable_thinking: bool = Field(True, description="Enable thinking mode for qwen3 model")

# Stock Analysis
class ComprehensiveAnalysisRequest(BaseModel):
    symbol: str
    lookback_days: int = 252
    analyses: Optional[List[str]] = None

class AnalysisResponsePayload(BaseModel):
    session_id: Optional[str] = None
    symbol: str
    question_input: Optional[str]
    target_language: Optional[str] = None
    model_name: str = Field(APIModelName.GPT41Nano, description="Model name")
    provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")

# Trending 
class StockStringPayload(BaseModel):
    session_id: str
    question_input: Optional[str] = None
    target_language: Optional[str] = None
    gainers: List[Dict]
    losers: Optional[List[Dict]] = None
    actives: Optional[List[Dict]] = None
    model_name: str
    provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")

# Market Overview
class MarketOverviewPayload(BaseModel):
    session_id: Optional[str] = None
    question_input: Optional[str] = None
    target_language: Optional[str] = None
    data: List[Dict]
    model_name: str
    provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")

class PromptRequest(BaseModel):
    prompt: str
    model_name: str
    provider_type: str

# Heatmap
class StockHeatmapPayload(BaseModel):
    session_id: str = Field(None, description="Chat session ID")
    question_input: str = Field("Show me the stock heatmap", description="User's question")
    target_language: Optional[str] = None
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
    provider_type: str = Field(ProviderType.OPENAI, description="Provider type: ollama, openai, gemini")


class ChatRequest(BaseModel):
    session_id: str
    question_input: str
    model_name: str = Field(APIModelName.GPT41Nano, description="Model name")
    collection_name: str = settings.QDRANT_COLLECTION_NAME
    use_multi_collection: bool = False


class MarketMoversAnalysisRequest(BaseModel):
    """Request model for market movers analysis (gainers/losers)"""
    session_id: str = Field(..., description="Chat session ID for conversation history")
    question_input: str = Field(
        "Analyze this data and provide insights", 
        description="User's question or analysis request"
    )
    target_language: Optional[str] = None
    model_name: str = Field("gpt-4.1-nano-2025-04-14", description="LLM model to use")
    provider_type: str = Field("openai", description="Provider type: openai, ollama, gemini")
    # limit: int = Field(10, ge=1, le=30, description="Number of items to analyze")

class MarketMoversAnalysisResponse(BaseModel):
    """Response model for market movers analysis"""
    status: str
    message: str
    data: Optional[str] = None


class ChartDataItem(BaseModel):
    time: str = Field(..., description="Chuỗi datetime định dạng YYYY-MM-DD HH:MM:SS hoặc YYYY-MM-DDTHH:MM:SS")
    value: float = Field(..., description="Giá trị tại thời điểm tương ứng")

class DiscoveryItemOutput(BaseModel): 
    symbol: str
    name: Optional[str] = None
    url_logo: Optional[str] = None
    event_catalyst: Optional[str] = Field(None, description="Nguồn dữ liệu cho trường này chưa xác định, mặc định là None")
    price: Optional[float] = None
    change: Optional[float] = None
    percent_change: Optional[float] = None
    volume: Optional[float] = None
    chartData: List[ChartDataItem] = Field(default_factory=list, description="Dữ liệu biểu đồ lịch sử cho mã")
    @validator('price', 'change', 'percent_change', 'volume', pre=True)
    def validate_optional_floats(cls, v):
        if v == "":
            return None
        return v