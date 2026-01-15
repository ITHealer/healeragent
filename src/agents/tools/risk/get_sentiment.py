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

try:
    from src.handlers.sentiment_analysis_handler import SentimentAnalysisHandler
    HANDLER_AVAILABLE = True
except ImportError:
    HANDLER_AVAILABLE = False
    SentimentAnalysisHandler = None


class GetSentimentTool(BaseTool):
    """
    Atomic tool for sentiment analysis
    
    Uses FMP /v4/historical/social-sentiment API
    Returns sentiment score with label (Bullish/Bearish/Neutral)
    """
    
    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        
        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")
        
        if not api_key:
            raise ValueError("FMP_API_KEY required")
        
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        self.sentiment_handler = None
        if HANDLER_AVAILABLE:
            try:
                self.sentiment_handler = SentimentAnalysisHandler()
            except Exception as e:
                self.logger.warning(f"[getSentiment] Handler init failed: {e}")
        
        self.schema = ToolSchema(
            name="getSentiment",
            category="risk",
            description=(
                "Analyze market sentiment from social media (Twitter, StockTwits, Reddit). "
                "Returns sentiment score and classification. "
                "Use when user asks about market sentiment, social media buzz, or investor mood."
            ),
            capabilities=[
                "✅ Combined sentiment score (0-100)",
                "✅ Sentiment classification (Bullish/Bearish/Neutral)",
                "✅ Sentiment trend analysis",
                "✅ Multi-platform aggregation (Twitter, StockTwits)"
            ],
            limitations=[
                "❌ Lagging indicator (not predictive)",
                "❌ Social media bias (retail sentiment)",
                "❌ Updated hourly (not real-time)"
            ],
            usage_hints=[
                "User asks: 'AAPL sentiment' → USE THIS",
                "User asks: 'What's the buzz on TSLA?' → USE THIS",
                "User asks: 'Tâm lý NVDA' → USE THIS",
                "User asks: 'Social media sentiment Microsoft' → USE THIS"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                ),
                ToolParameter(
                    name="lookback_days",
                    type="integer",
                    description="Days to analyze",
                    required=False,
                    default=7
                )
            ],
            returns={
                "symbol": "string",
                "sentiment_score": "number",
                "sentiment_label": "string",
                "sentiment_trend": "string"
            },
            typical_execution_time_ms=1600,
            requires_symbol=True
        )
    
    async def execute(self, symbol: str, lookback_days: int = 7) -> ToolOutput:
        """
        Execute sentiment analysis
        
        Args:
            symbol: Stock symbol
            lookback_days: Days to analyze (default: 7)
            
        Returns:
            ToolOutput with ALL required fields
        """
        start_time = datetime.now()
        symbol_upper = symbol.upper()
        lookback_days = int(max(1, min(30, lookback_days)))  # Ensure int for slicing

        self.logger.info(
            f"[getSentiment] Executing: symbol={symbol_upper}, lookback={lookback_days}"
        )
        
        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1: Fetch from FMP
            # ═══════════════════════════════════════════════════════════
            raw_data = await self._fetch_from_fmp(symbol_upper, lookback_days)
            
            if not raw_data:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No sentiment data available for {symbol_upper}"
                )
            
            # ═══════════════════════════════════════════════════════════
            # STEP 2: Process sentiment data
            # ═══════════════════════════════════════════════════════════
            if self.sentiment_handler:
                processed = self.sentiment_handler._process_sentiment_data(raw_data)
            else:
                processed = self._process_sentiment_data_fallback(raw_data)
            
            if processed.get("data_points", 0) == 0:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"Could not process sentiment data for {symbol_upper}"
                )
            
            # ═══════════════════════════════════════════════════════════
            # STEP 3: Format with ALL required fields
            # ═══════════════════════════════════════════════════════════
            formatted = self._format_sentiment_data(processed, symbol_upper)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"[getSentiment] ✅ SUCCESS ({int(execution_time)}ms) - "
                f"Score: {formatted['sentiment_score']:.3f}, "
                f"Label: {formatted['sentiment_label']}"
            )
            
            return create_success_output(
                tool_name=self.schema.name,
                data=formatted,
                metadata={
                    "source": "FMP Social Sentiment",
                    "symbol_queried": symbol_upper,
                    "lookback_days": lookback_days,
                    "data_points": processed.get("data_points", 0),
                    "execution_time_ms": int(execution_time),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"[getSentiment] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Sentiment analysis failed: {str(e)}"
            )
    
    async def _fetch_from_fmp(
        self,
        symbol: str,
        lookback_days: int
    ) -> Optional[List[Dict]]:
        """
        Fetch from FMP Social Sentiment API
        
        API: /v4/historical/social-sentiment
        
        Response format:
        [
            {
                "date": "2024-01-15",
                "symbol": "AAPL",
                "stocktwitsSentiment": 0.625,
                "twitterSentiment": 0.547,
                "twitterMentions": 1250,
                "stocktwitsMentions": 850
            },
            ...
        ]
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
                
                if not data:
                    self.logger.warning(f"[getSentiment] No data returned for {symbol}")
                    return None
                
                # Limit to lookback_days
                limited_data = data[:lookback_days] if len(data) > lookback_days else data
                
                self.logger.info(
                    f"[getSentiment] Fetched {len(limited_data)} data points for {symbol}"
                )
                
                return limited_data
                
        except httpx.HTTPStatusError as e:
            self.logger.error(
                f"[getSentiment] FMP HTTP error {e.response.status_code}"
            )
            return None
        except Exception as e:
            self.logger.error(f"[getSentiment] FMP error: {e}")
            return None
    
    def _process_sentiment_data_fallback(
        self,
        data: List[Dict]
    ) -> Dict[str, Any]:
        """
        Fallback processing when handler not available
        
        Aggregates StockTwits and Twitter sentiment scores
        """
        import statistics
        
        if not data:
            return {"data_points": 0, "average_sentiment": 0.5}
        
        scores = []
        stocktwits_scores = []
        twitter_scores = []
        
        for item in data:
            st = item.get('stocktwitsSentiment')
            tw = item.get('twitterSentiment')
            
            # Collect individual platform scores
            if st is not None:
                stocktwits_scores.append(float(st))
            if tw is not None:
                twitter_scores.append(float(tw))
            
            # Combined score (average of both platforms)
            if st is not None and tw is not None:
                scores.append((float(st) + float(tw)) / 2)
            elif st is not None:
                scores.append(float(st))
            elif tw is not None:
                scores.append(float(tw))
        
        if not scores:
            return {"data_points": 0, "average_sentiment": 0.5}
        
        avg_sentiment = statistics.mean(scores)
        
        return {
            "data_points": len(scores),
            "average_sentiment": round(avg_sentiment, 6),
            "sentiment_level": self._get_sentiment_level(avg_sentiment),
            "sentiment_trend": self._calculate_trend(scores),
            "volatility": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
            "latest_sentiment": round(scores[0], 4) if scores else 0.5,
            "stocktwits_avg": round(statistics.mean(stocktwits_scores), 4) if stocktwits_scores else None,
            "twitter_avg": round(statistics.mean(twitter_scores), 4) if twitter_scores else None
        }
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """Calculate sentiment trend from score history"""
        if len(scores) < 3:
            return "Stable"
        
        # Compare recent (first 3) vs older (last 3)
        recent_avg = sum(scores[:3]) / 3
        older_avg = sum(scores[-3:]) / 3
        
        diff = recent_avg - older_avg
        
        if diff > 0.1:
            return "Improving"
        elif diff < -0.1:
            return "Declining"
        else:
            return "Stable"
    
    def _get_sentiment_level(self, avg: float) -> str:
        """Map sentiment score to level description"""
        if avg >= 0.8:
            return "Very Positive"
        elif avg >= 0.65:
            return "Positive"
        elif avg >= 0.55:
            return "Slightly Positive"
        elif avg >= 0.45:
            return "Neutral"
        elif avg >= 0.35:
            return "Slightly Negative"
        elif avg >= 0.2:
            return "Negative"
        else:
            return "Very Negative"
    
    def _format_sentiment_data(
        self,
        processed: Dict,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Format to output schema
        
        ✅ Returns ALL required fields:
        - sentiment_score
        - sentiment_label (REQUIRED - was missing)
        - sentiment_trend
        """
        # Extract average sentiment (0-1 scale from FMP)
        avg = processed.get("average_sentiment", 0.5)
        
        # Normalize to -1 to +1 scale
        normalized = (avg - 0.5) * 2
        
        # Get sentiment level description
        level = processed.get("sentiment_level", "Neutral")
        
        # ═══════════════════════════════════════════════════════════
        # ✅ CRITICAL FIX: Map to sentiment_label (REQUIRED FIELD)
        # ═══════════════════════════════════════════════════════════
        if level in ["Very Positive", "Positive"]:
            sentiment_label = "BULLISH"
        elif level in ["Slightly Positive"]:
            sentiment_label = "SLIGHTLY_BULLISH"
        elif level in ["Very Negative", "Negative"]:
            sentiment_label = "BEARISH"
        elif level in ["Slightly Negative"]:
            sentiment_label = "SLIGHTLY_BEARISH"
        else:
            sentiment_label = "NEUTRAL"
        
        # ═══════════════════════════════════════════════════════════
        # Return ALL required fields from schema
        # ═══════════════════════════════════════════════════════════
        return {
            # ✅ Required fields
            "symbol": symbol,
            "sentiment_score": round(normalized, 3),
            "sentiment_label": sentiment_label,           # ← REQUIRED (was missing)
            "sentiment_trend": processed.get("sentiment_trend", "Stable"),
            
            # Additional fields (not in schema but useful)
            "overall_sentiment": sentiment_label.replace("_", " ").title(),
            "sentiment_level": level,
            "raw_score": round(avg, 4),  # Original 0-1 scale
            "volatility": processed.get("volatility", 0.0),
            "data_points": processed.get("data_points", 0),
            "latest_sentiment": processed.get("latest_sentiment"),
            "stocktwits_avg": processed.get("stocktwits_avg"),
            "twitter_avg": processed.get("twitter_avg"),
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import os
    
    async def test_tool():
        """Test GetSentimentTool with validation"""
        
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("❌ ERROR: FMP_API_KEY not found")
            return
        
        print("=" * 80)
        print("TESTING [GetSentimentTool] - FIXED VERSION")
        print("=" * 80)
        
        tool = GetSentimentTool(api_key=api_key)
        
        # Test 1: AAPL sentiment
        print("\n📊 Test 1: AAPL")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL", lookback_days=7)
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            # Verify ALL required fields
            required_fields = tool.schema.get_required_fields()
            missing = [f for f in required_fields if f not in result.data]
            
            if missing:
                print(f"⚠️  PARTIAL - Missing: {missing}")
            else:
                print(f"✅ ALL REQUIRED FIELDS PRESENT")
            
            print(f"\nData:")
            print(f"  Score: {result.data['sentiment_score']:.3f}")
            print(f"  Label: {result.data['sentiment_label']}")  # ← Required field
            print(f"  Trend: {result.data['sentiment_trend']}")
            print(f"  Data Points: {result.data['data_points']}")
            
            if result.data.get('stocktwits_avg'):
                print(f"  StockTwits: {result.data['stocktwits_avg']:.3f}")
            if result.data.get('twitter_avg'):
                print(f"  Twitter: {result.data['twitter_avg']:.3f}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        # Test 2: Another symbol
        print("\n📊 Test 2: TSLA")
        print("-" * 40)
        result2 = await tool.safe_execute(symbol="TSLA")
        
        if result2.is_success():
            print(f"✅ Label: {result2.data['sentiment_label']}")
            print(f"✅ Score: {result2.data['sentiment_score']:.3f}")
        
        print("\n" + "=" * 80)
    
    asyncio.run(test_tool())