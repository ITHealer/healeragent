"""
Task Schemas for News Analysis Task System
==========================================

Defines schemas for async task processing:
- TaskRequest: Job submission from BE .NET
- TaskStatus: Job status tracking
- TaskResult: Analysis result with citations

Similar to Grok Tasks - AI reads and analyzes news,
outputs structured reports per symbol with citations.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class TaskType(str, Enum):
    """Types of analysis tasks."""
    NEWS_ANALYSIS = "news_analysis"          # Deep news analysis with insights
    MARKET_SUMMARY = "market_summary"        # Quick market overview
    STOCK_PERFORMANCE = "stock_performance"  # Stock price & news analysis
    CRYPTO_ANALYSIS = "crypto_analysis"      # Crypto market analysis
    TECH_NEWS = "tech_news"                  # Tech industry news digest
    MARKET_OVERVIEW = "market_overview"      # General market overview
    CUSTOM = "custom"                        # Fully custom analysis


class TaskStatus(str, Enum):
    """Task processing status."""
    PENDING = "pending"          # In queue, waiting to process
    PROCESSING = "processing"    # Currently being processed
    COMPLETED = "completed"      # Successfully completed
    FAILED = "failed"            # Failed with error
    CANCELLED = "cancelled"      # Cancelled by user/system


class TaskPriority(int, Enum):
    """Task priority levels for queue ordering."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20


class SymbolType(str, Enum):
    """Asset type for symbol."""
    STOCK = "stock"
    CRYPTO = "crypto"
    ETF = "etf"
    FOREX = "forex"


# =============================================================================
# Request Models
# =============================================================================

class TaskRequest(BaseModel):
    """
    Task submission request from BE .NET.

    BE sends this to create a new analysis task.
    Task is queued and processed asynchronously.

    Example:
    {
        "request_id": 1792,
        "symbols": ["TSLA", "BTC", "NVDA"],
        "task_type": "news_analysis",
        "target_language": "vi",
        "prompt": "Phân tích từ góc độ nhà đầu tư dài hạn, tập trung vào AI và robo-taxi",
        "callback_url": "https://api.example.com/webhooks/task-complete",
        "priority": 10
    }
    """
    # BE reference
    request_id: int = Field(
        ...,
        description="BE .NET request ID for callback"
    )

    # Task configuration
    symbols: List[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Symbols to analyze (e.g., ['TSLA', 'BTC', 'NVDA'])"
    )
    task_type: TaskType = Field(
        default=TaskType.STOCK_PERFORMANCE,
        description="Type of analysis to perform"
    )

    # Output options
    target_language: str = Field(
        default="vi",
        description="Output language: en, vi, zh, ja, ko"
    )

    # User instructions/prompt (like Grok Tasks "Hướng dẫn")
    prompt: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="""Custom instructions to guide the AI analysis.
        Examples:
        - "Chỉ phân tích tin tức liên quan đến AI và robo-taxi"
        - "Phân tích từ góc độ nhà đầu tư dài hạn"
        - "So sánh Tesla với các đối thủ EV như Rivian, Lucid"
        - "Tập trung vào phân tích kỹ thuật và điểm hỗ trợ/kháng cự"
        - "Đánh giá rủi ro và các yếu tố tiêu cực cần lưu ý"
        - "Bitcoin có nên mua ở mức giá hiện tại không?"
        """
    )

    # Callback
    callback_url: Optional[str] = Field(
        default=None,
        description="URL to POST results when complete"
    )
    callback_secret: Optional[str] = Field(
        default=None,
        description="Secret for X-Webhook-Secret header"
    )

    # Queue options
    priority: int = Field(
        default=TaskPriority.NORMAL,
        ge=1,
        le=100,
        description="Task priority (higher = processed first)"
    )
    scheduled_at: Optional[datetime] = Field(
        default=None,
        description="When to process (None = immediately)"
    )

    # Processing options
    max_news_per_symbol: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Max news articles to fetch per symbol"
    )
    include_market_data: bool = Field(
        default=True,
        description="Include price and historical data"
    )
    extract_full_content: bool = Field(
        default=True,
        description="Extract full article content via Tavily"
    )

    def generate_job_id(self) -> str:
        """Generate unique job ID."""
        return f"task_{self.request_id}_{uuid.uuid4().hex[:8]}"


# =============================================================================
# Symbol Analysis Models
# =============================================================================

class PriceChange(BaseModel):
    """Price change data for a time period."""
    period: str = Field(..., description="Period: 24h, 7d, 30d")
    change_percent: float = Field(..., description="Percentage change")
    change_value: Optional[float] = Field(None, description="Absolute change")
    start_price: Optional[float] = Field(None, description="Price at period start")
    end_price: Optional[float] = Field(None, description="Current price")


class MarketData(BaseModel):
    """Market data for a symbol."""
    symbol: str
    symbol_type: SymbolType
    current_price: float
    currency: str = "USD"
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    changes: List[PriceChange] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class NewsSource(BaseModel):
    """Citation source for an insight."""
    index: int = Field(..., description="Reference number [1], [2], etc.")
    title: str
    url: str
    source: str = Field(..., description="Domain: reuters.com, cnbc.com, etc.")
    published_at: Optional[datetime] = None


class SymbolInsight(BaseModel):
    """Key insight about a symbol with citation."""
    text: str = Field(..., description="Insight text in target language")
    source_indices: List[int] = Field(
        ...,
        description="Reference numbers to NewsSource"
    )
    sentiment: Optional[str] = Field(
        None,
        description="bullish, bearish, neutral"
    )


class SymbolAnalysis(BaseModel):
    """
    Complete analysis for a single symbol.

    Includes:
    - Market data (price, changes)
    - Key insights with citations
    - Sentiment and outlook
    - All sources referenced
    """
    symbol: str
    symbol_type: SymbolType
    display_name: str = Field(..., description="Full name: Tesla Inc., Bitcoin, etc.")

    # Market data
    market_data: Optional[MarketData] = None

    # Analysis
    sentiment: str = Field(
        ...,
        description="Overall sentiment: BULLISH, BEARISH, NEUTRAL, MIXED"
    )
    sentiment_score: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score: -1 (bearish) to +1 (bullish)"
    )

    # Insights with citations
    key_insights: List[SymbolInsight] = Field(
        default_factory=list,
        description="Key insights about the symbol"
    )

    # Outlook
    short_term_outlook: Optional[str] = Field(
        None,
        description="Short-term prediction (1-7 days)"
    )
    long_term_outlook: Optional[str] = Field(
        None,
        description="Long-term prediction (1-3 months)"
    )

    # Risk factors
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Key risk factors to watch"
    )

    # Sources
    sources: List[NewsSource] = Field(
        default_factory=list,
        description="All news sources used for this analysis"
    )


# =============================================================================
# Task Result Models
# =============================================================================

class TaskResult(BaseModel):
    """
    Complete task result with all symbol analyses.

    This is what gets sent to the callback URL.
    """
    # Task identification
    job_id: str
    request_id: int
    status: TaskStatus

    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None

    # Results
    title: str = Field(..., description="Report title in target language")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    target_language: str
    prompt: Optional[str] = Field(None, description="User instructions used for this analysis")

    # Symbol analyses
    analyses: List[SymbolAnalysis] = Field(
        default_factory=list,
        description="Analysis per symbol"
    )

    # Overall summary
    overall_sentiment: str = Field(
        default="MIXED",
        description="Overall market sentiment"
    )
    key_themes: List[str] = Field(
        default_factory=list,
        description="Key themes across all symbols"
    )
    summary: Optional[str] = Field(
        None,
        description="Executive summary in target language"
    )

    # Error (if failed)
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processing metadata (timing, counts, etc.)"
    )


class TaskStatusResponse(BaseModel):
    """Response for task status query."""
    job_id: str
    request_id: int
    status: TaskStatus
    progress: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Progress percentage"
    )
    message: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None


class TaskSubmitResponse(BaseModel):
    """Response when task is submitted."""
    job_id: str
    request_id: int
    status: TaskStatus = TaskStatus.PENDING
    message: str = "Task queued successfully"
    queue_position: Optional[int] = None
    estimated_start: Optional[datetime] = None


# =============================================================================
# Callback Payload
# =============================================================================

class CallbackPayload(BaseModel):
    """
    Payload sent to BE .NET callback URL.

    Format compatible with BE API:
    POST /api/v1/user-task/submit-generation-result
    """
    requestId: int = Field(..., alias="request_id")
    content: str = Field(..., description="JSON string of TaskResult or formatted text")

    class Config:
        populate_by_name = True


# =============================================================================
# Internal Processing Models
# =============================================================================

class ArticleContent(BaseModel):
    """Extracted article content for processing."""
    url: str
    title: str
    content: str = Field(..., description="Full article text")
    snippet: str = Field(default="", description="Short snippet")
    source: str = Field(..., description="Source domain")
    published_at: Optional[datetime] = None
    symbol: Optional[str] = None
    extraction_success: bool = True
    extraction_method: str = Field(
        default="tavily",
        description="How content was extracted"
    )


class ProcessingContext(BaseModel):
    """Internal context during task processing."""
    job_id: str
    request: TaskRequest

    # Collected data
    articles: List[ArticleContent] = Field(default_factory=list)
    market_data: Dict[str, MarketData] = Field(default_factory=dict)

    # Processing state
    current_phase: str = "initialized"
    phases_completed: List[str] = Field(default_factory=list)

    # Timing
    phase_timings: Dict[str, int] = Field(default_factory=dict)

    # Errors (non-fatal)
    warnings: List[str] = Field(default_factory=list)
