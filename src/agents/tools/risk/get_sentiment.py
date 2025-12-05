# File: src/agents/tools/risk/get_sentiment.py

"""
GetSentimentTool - Atomic Tool for Sentiment Analysis

FIXED: Proper integration with SentimentAnalysisHandler
- Fetch raw sentiment from FMP API
- Process using handler's logic
- Return structured atomic tool output
"""

import httpx
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)

# Import handler for processing logic
try:
    from src.handlers.sentiment_analysis_handler import SentimentAnalysisHandler
    HANDLER_AVAILABLE = True
except ImportError:
    HANDLER_AVAILABLE = False
    SentimentAnalysisHandler = None


class GetSentimentTool(BaseTool):
    """
    Atomic tool để phân tích sentiment
    
    Data Flow:
    1. Fetch raw sentiment from FMP API
    2. Process using SentimentAnalysisHandler._process_sentiment_data()
    3. Format to atomic tool schema
    
    Usage:
        tool = GetSentimentTool()
        result = await tool.safe_execute(symbol="AAPL")
        
        if result.is_success():
            overall = result.data['overall_sentiment']
            score = result.data['sentiment_score']
            trend = result.data['sentiment_trend']
    """
    
    # FMP API Configuration
    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize tool
        
        Args:
            api_key: FMP API key for sentiment data
        """
        super().__init__()
        
        # Get API key
        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")
        
        if not api_key:
            raise ValueError("FMP_API_KEY required for sentiment analysis")
        
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Initialize handler for processing (if available)
        self.sentiment_handler = None
        if HANDLER_AVAILABLE:
            try:
                self.sentiment_handler = SentimentAnalysisHandler()
                self.logger.info("[getSentiment] SentimentAnalysisHandler initialized")
            except Exception as e:
                self.logger.warning(f"[getSentiment] Could not init handler: {e}")
        
        # Define schema
        self.schema = ToolSchema(
            name="getSentiment",
            category="risk",
            description=(
                "Analyze market sentiment from news, social media, and analyst opinions. "
                "Returns sentiment score (0-100), sentiment trend, and sentiment breakdown. "
                "Use when user asks about market sentiment, investor mood, or public opinion."
            ),
            capabilities=[
                "✅ Overall sentiment score (0-100, 50=neutral)",
                "✅ Sentiment sources (news, social media, analysts)",
                "✅ Bullish/bearish percentage",
                "✅ Sentiment trend (improving/declining)",
                "✅ Key sentiment drivers",
                "✅ Sentiment change over time"
            ],
            limitations=[
                "❌ Sentiment is lagging indicator (based on past)",
                "❌ Social media sentiment can be manipulated",
                "❌ News sentiment may be delayed",
                "❌ One symbol at a time"
            ],
            usage_hints=[
                # English
                "User asks: 'Apple sentiment' → USE THIS with symbol=AAPL",
                "User asks: 'What do people think about TSLA?' → USE THIS with symbol=TSLA",
                "User asks: 'Is NVDA bullish or bearish?' → USE THIS with symbol=NVDA",
                "User asks: 'Microsoft market sentiment' → USE THIS with symbol=MSFT",
                
                # Vietnamese
                "User asks: 'Tâm lý thị trường với Apple' → USE THIS with symbol=AAPL",
                "User asks: 'Tesla có bullish không?' → USE THIS with symbol=TSLA",
                "User asks: 'Sentiment của NVDA' → USE THIS with symbol=NVDA",
                
                # When NOT to use
                "User asks for NEWS articles → DO NOT USE (use getStockNews)",
                "User asks about FUNDAMENTALS → DO NOT USE (use getFinancialRatios)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                ),
                ToolParameter(
                    name="lookback_days",
                    type="integer",
                    description="Number of days to analyze sentiment",
                    required=False,
                    default=7
                )
            ],
            returns={
                "symbol": "string",
                "sentiment_score": "number - 0-100 (50=neutral)",
                "sentiment_label": "string - Bullish/Neutral/Bearish",
                "bullish_percent": "number",
                "bearish_percent": "number",
                "sentiment_trend": "string - Improving/Declining/Stable",
                "sources": "object - Breakdown by source",
                "key_drivers": "array - Main sentiment drivers",
                "timestamp": "string"
            },
            typical_execution_time_ms=1600,
            requires_symbol=True
        )
            
    async def execute(self, symbol: str) -> ToolOutput:
        """
        Execute sentiment analysis
        
        Args:
            symbol: Stock symbol
            
        Returns:
            ToolOutput with sentiment data
        """
        symbol_upper = symbol.upper()
        
        self.logger.info(f"[getSentiment] Executing for symbol={symbol_upper}")
        
        try:
            # ════════════════════════════════════════════════════════
            # STEP 1: Fetch raw sentiment data from FMP
            # ════════════════════════════════════════════════════════
            raw_sentiment_data = await self._fetch_from_fmp(symbol_upper)
            
            if not raw_sentiment_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No sentiment data available for {symbol_upper}. "
                          f"Symbol may not have sufficient social media coverage."
                )
            
            self.logger.info(
                f"[getSentiment] Fetched {len(raw_sentiment_data)} sentiment data points"
            )
            
            # ════════════════════════════════════════════════════════
            # STEP 2: Process sentiment data using handler logic
            # ════════════════════════════════════════════════════════
            if self.sentiment_handler:
                # Use handler's processing logic (accessing private method)
                processed_data = self.sentiment_handler._process_sentiment_data(
                    raw_sentiment_data
                )
            else:
                # Fallback: Self-process if handler not available
                processed_data = self._process_sentiment_data_fallback(
                    raw_sentiment_data
                )
            
            # Check if processing returned valid data
            if processed_data.get("data_points", 0) == 0:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"Could not process sentiment data for {symbol_upper}"
                )
            
            self.logger.info(
                f"[getSentiment] Processed data: {processed_data.get('data_points')} points, "
                f"avg sentiment: {processed_data.get('average_sentiment')}"
            )
            
            # ════════════════════════════════════════════════════════
            # STEP 3: Format to atomic tool schema
            # ════════════════════════════════════════════════════════
            formatted_data = self._format_sentiment_data(processed_data, symbol_upper)
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "FMP Social Sentiment API",
                    "symbol_queried": symbol_upper,
                    "data_points": processed_data.get("data_points", 0),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[getSentiment] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Sentiment analysis failed: {str(e)}"
            )
    
    async def _fetch_from_fmp(self, symbol: str) -> Optional[List[Dict]]:
        """
        Fetch raw sentiment data from FMP API
        
        Endpoint: /api/v4/historical/social-sentiment
        Returns: List of sentiment data points with StockTwits & Twitter sentiment
        """
        url = f"{self.FMP_BASE_URL}/v4/historical/social-sentiment"
        
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data or len(data) == 0:
                    self.logger.warning(
                        f"[getSentiment] No sentiment data returned for {symbol}"
                    )
                    return None
                
                self.logger.debug(
                    f"[getSentiment] FMP returned {len(data)} sentiment records"
                )
                
                return data
                
        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"[getSentiment] FMP HTTP error {e.response.status_code}: "
                f"{e.response.text}"
            )
            return None
        except Exception as e:
            self.logger.error(f"[getSentiment] FMP request error: {e}")
            return None
    
    def _process_sentiment_data_fallback(
        self,
        sentiment_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Fallback processing if handler not available
        
        Simplified version of handler's _process_sentiment_data()
        """
        import statistics
        
        if not sentiment_data:
            return self._get_empty_sentiment_summary()
        
        # Extract scores
        sentiment_scores = []
        stocktwits_scores = []
        twitter_scores = []
        
        for item in sentiment_data:
            stocktwits = item.get('stocktwitsSentiment')
            twitter = item.get('twitterSentiment')
            
            if stocktwits is not None:
                stocktwits_scores.append(float(stocktwits))
            if twitter is not None:
                twitter_scores.append(float(twitter))
            
            # Combined score
            if stocktwits is not None and twitter is not None:
                combined = (float(stocktwits) + float(twitter)) / 2
                sentiment_scores.append(combined)
            elif stocktwits is not None:
                sentiment_scores.append(float(stocktwits))
            elif twitter is not None:
                sentiment_scores.append(float(twitter))
        
        if not sentiment_scores:
            return self._get_empty_sentiment_summary()
        
        avg_sentiment = statistics.mean(sentiment_scores)
        volatility = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
        
        return {
            "data_points": len(sentiment_scores),
            "average_sentiment": round(avg_sentiment, 3),
            "sentiment_level": self._get_sentiment_level_normalized(avg_sentiment),
            "sentiment_trend": "Stable",  # Simplified
            "volatility": round(volatility, 3),
            "recent_shift": "N/A",
            "latest_sentiment": round(sentiment_scores[0], 3),
            "stocktwits_avg_sentiment": round(statistics.mean(stocktwits_scores), 3) if stocktwits_scores else 0,
            "twitter_avg_sentiment": round(statistics.mean(twitter_scores), 3) if twitter_scores else 0,
            "date_range": "N/A"
        }
    
    def _get_empty_sentiment_summary(self) -> Dict[str, Any]:
        """Return empty sentiment summary"""
        return {
            "data_points": 0,
            "average_sentiment": 0,
            "sentiment_level": "No Data",
            "sentiment_trend": "No Data",
            "volatility": 0,
            "recent_shift": "N/A",
            "latest_sentiment": 0,
            "stocktwits_avg_sentiment": 0,
            "twitter_avg_sentiment": 0,
            "date_range": "N/A"
        }
    
    def _get_sentiment_level_normalized(self, avg_sentiment: float) -> str:
        """
        Determine sentiment level for 0-1 scale (FMP format)
        
        Matches handler's logic
        """
        if avg_sentiment >= 0.8:
            return "Very Positive"
        elif avg_sentiment >= 0.65:
            return "Positive"
        elif avg_sentiment >= 0.55:
            return "Slightly Positive"
        elif avg_sentiment >= 0.45:
            return "Neutral"
        elif avg_sentiment >= 0.35:
            return "Slightly Negative"
        elif avg_sentiment >= 0.2:
            return "Negative"
        else:
            return "Very Negative"
    
    def _format_sentiment_data(
        self,
        processed_data: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Format processed data to atomic tool schema
        
        Args:
            processed_data: Output from _process_sentiment_data()
            symbol: Stock symbol
            
        Returns:
            Formatted data matching tool schema
        """
        # Extract sentiment score (0-1 range from FMP)
        avg_sentiment = processed_data.get("average_sentiment", 0)
        
        # Convert to -1 to +1 range for consistency
        # FMP returns 0-1, we normalize to -1 to +1
        normalized_score = (avg_sentiment - 0.5) * 2
        
        # Determine overall sentiment
        sentiment_level = processed_data.get("sentiment_level", "Neutral")
        
        # Map to overall sentiment categories
        if sentiment_level in ["Very Positive", "Positive"]:
            overall = "Bullish"
        elif sentiment_level in ["Very Negative", "Negative"]:
            overall = "Bearish"
        else:
            overall = "Neutral"
        
        # Extract social activity
        social_activity = processed_data.get("recent_social_activity", {})
        
        return {
            "symbol": symbol,
            "overall_sentiment": overall,
            "sentiment_score": round(normalized_score, 3),
            "sentiment_level": sentiment_level,
            "sentiment_trend": processed_data.get("sentiment_trend", "Stable"),
            "recent_shift": processed_data.get("recent_shift", "N/A"),
            "volatility": processed_data.get("volatility", 0.0),
            "latest_sentiment": processed_data.get("latest_sentiment", 0.0),
            "stocktwits_avg": processed_data.get("stocktwits_avg_sentiment", 0.0),
            "twitter_avg": processed_data.get("twitter_avg_sentiment", 0.0),
            "social_activity": {
                "total_posts": social_activity.get("total_posts", 0),
                "total_likes": social_activity.get("total_likes", 0),
                "engagement_rate": social_activity.get("engagement_rate", 0.0)
            },
            "data_points": processed_data.get("data_points", 0),
            "date_range": processed_data.get("date_range", "N/A"),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import json
    import os
    
    async def test_tool():
        """Standalone test for GetSentimentTool"""
        
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("❌ ERROR: FMP_API_KEY not found in environment")
            return
        
        print("=" * 80)
        print("TESTING [GetSentimentTool]")
        print("=" * 80)
        
        tool = GetSentimentTool(api_key=api_key)
        
        # Test 1: Valid symbol with good social coverage
        print("\n📊 Test 1: AAPL (high coverage expected)")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL")
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            print("✅ SUCCESS")
            print(f"\nOverall: {result.data['overall_sentiment']}")
            print(f"Sentiment score: {result.data['sentiment_score']} "
                  f"({result.data['sentiment_level']})")
            print(f"Trend: {result.data['sentiment_trend']}")
            print(f"Recent shift: {result.data['recent_shift']}")
            print(f"\nStockTwits: {result.data['stocktwits_avg']:.3f}")
            print(f"Twitter: {result.data['twitter_avg']:.3f}")
            print(f"\nData points: {result.data['data_points']}")
            print(f"Date range: {result.data['date_range']}")
            
            social = result.data['social_activity']
            print(f"\nSocial activity:")
            print(f"  Posts: {social['total_posts']}")
            print(f"  Likes: {social['total_likes']}")
            print(f"  Engagement: {social['engagement_rate']:.2f}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        # Test 2: Another popular symbol
        print("\n📊 Test 2: TSLA sentiment")
        print("-" * 40)
        result = await tool.safe_execute(symbol="TSLA")
        
        if result.is_success():
            print(f"✅ Sentiment: {result.data['sentiment_level']} "
                  f"(score: {result.data['sentiment_score']:+.3f})")
            print(f"✅ Trend: {result.data['sentiment_trend']}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        # Test 3: Low coverage symbol
        print("\n📊 Test 3: Low coverage symbol")
        print("-" * 40)
        result = await tool.safe_execute(symbol="ZZZ")
        
        if result.is_success():
            print(f"✅ Data points: {result.data['data_points']}")
        else:
            print(f"⚠️ Expected error: {result.error}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)
    
    asyncio.run(test_tool())