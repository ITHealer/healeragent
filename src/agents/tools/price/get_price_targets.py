import httpx
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)
from src.helpers.redis_cache import get_redis_client_llm


class GetPriceTargetsTool(BaseTool):
    """
        Atomic tool for analyst price targets with ratings
        
        Data sources:
        - FMP /v4/price-target-consensus (price targets)
        - FMP /stable/grades-consensus (analyst ratings)
        
        Cache: 4 hours (analyst targets don't change frequently)
    """
    
    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"
    CACHE_TTL = 14400  # 4 hours
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        
        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")
        
        if not api_key:
            raise ValueError("FMP_API_KEY not provided")
        
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Define schema
        self.schema = ToolSchema(
            name="getPriceTargets",
            category="price",
            description=(
                "Get analyst price targets and recommendations from major financial institutions. "
                "Returns consensus price targets, analyst ratings, and price projections. "
                "Use when user asks about analyst opinions, price targets, or buy/sell ratings."
            ),
            capabilities=[
                "✅ Analyst consensus price target",
                "✅ High/low price target range",
                "✅ Number of analysts covering",
                "✅ Average rating (Buy/Hold/Sell)",
                "✅ Rating breakdown (Strong Buy/Buy/Hold/Sell)",
                "✅ Target vs current price comparison"
            ],
            limitations=[
                "❌ Targets may be outdated (updated quarterly)",
                "❌ Not all stocks have analyst coverage",
                "❌ No real-time target updates"
            ],
            usage_hints=[
                "User asks: 'Apple price target' → USE THIS",
                "User asks: 'What do analysts say about TSLA?' → USE THIS",
                "User asks: 'Should I buy Microsoft?' → USE THIS",
                "User asks: 'Mục tiêu giá của Amazon' → USE THIS",
                "User asks: 'Các nhà phân tích đánh giá NVDA như thế nào?' → USE THIS",
                "User wants technical analysis targets → DO NOT USE (use getTechnicalIndicators)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                )
            ],
            returns={
                "symbol": "string",
                "consensus_target": "number",
                "high_target": "number",
                "low_target": "number",
                "analyst_count": "number",
                "average_rating": "string",
                "timestamp": "string"
            },
            typical_execution_time_ms=1200,
            requires_symbol=True
        )
    
    async def execute(self, symbol: str) -> ToolOutput:
        """
        Execute price targets retrieval with analyst ratings
        
        Args:
            symbol: Stock symbol
            
        Returns:
            ToolOutput with ALL required fields
        """
        start_time = datetime.now()
        symbol_upper = symbol.upper()
        
        self.logger.info(f"[getPriceTargets] Executing for symbol={symbol_upper}")
        
        try:
            # ═══════════════════════════════════════════════════════════
            # STEP 1: Check Redis cache
            # ═══════════════════════════════════════════════════════════
            cache_key = f"price_targets_{symbol_upper}"
            redis_client = await get_redis_client_llm()
            
            if redis_client:
                try:
                    cached_bytes = await redis_client.get(cache_key)
                    if cached_bytes:
                        self.logger.info(f"[CACHE HIT] {cache_key}")
                        if isinstance(cached_bytes, bytes):
                            cached_data = json.loads(cached_bytes.decode('utf-8'))
                        else:
                            cached_data = json.loads(cached_bytes)

                        # cached_data = json.loads(cached_bytes.decode('utf-8'))
                        
                        execution_time = (datetime.now() - start_time).total_seconds() * 1000
                        
                        # Generate formatted context for LLM
                        formatted_context = self._generate_formatted_context(cached_data)

                        return create_success_output(
                            tool_name=self.schema.name,
                            data=cached_data,
                            metadata={
                                "source": "Redis cache",
                                "symbol_queried": symbol_upper,
                                "execution_time_ms": int(execution_time),
                                "from_cache": True,
                                "timestamp": datetime.now().isoformat()
                            },
                            formatted_context=formatted_context
                        )
                except Exception as e:
                    self.logger.warning(f"[CACHE] Error reading: {e}")
            
            # ═══════════════════════════════════════════════════════════
            # STEP 2: Fetch from both APIs in parallel
            # ═══════════════════════════════════════════════════════════
            price_targets_task = self._fetch_price_targets(symbol_upper)
            ratings_task = self._fetch_analyst_ratings(symbol_upper)
            
            results = await asyncio.gather(
                price_targets_task,
                ratings_task,
                return_exceptions=True
            )
            
            price_targets = results[0]
            ratings = results[1]
            
            # Handle errors
            if isinstance(price_targets, Exception):
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"Failed to fetch price targets: {str(price_targets)}"
                )
            
            if not price_targets:
                return create_error_output(
                    tool_name=self.schema.name,
                    error=f"No analyst price targets available for {symbol_upper}"
                )
            
            # ═══════════════════════════════════════════════════════════
            # STEP 3: Format with ALL required fields
            # ═══════════════════════════════════════════════════════════
            formatted_data = self._format_targets_data(
                price_targets,
                ratings if not isinstance(ratings, Exception) else None,
                symbol_upper
            )
            
            # ═══════════════════════════════════════════════════════════
            # STEP 4: Cache the result
            # ═══════════════════════════════════════════════════════════
            if redis_client:
                try:
                    json_string = json.dumps(formatted_data)
                    await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
                    self.logger.info(f"[CACHE SET] {cache_key} (TTL={self.CACHE_TTL}s)")
                except Exception as e:
                    self.logger.warning(f"[CACHE] Error writing: {e}")
            
            # Close Redis
            if redis_client:
                try:
                    await redis_client.close()
                except:
                    pass
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            self.logger.info(
                f"[getPriceTargets] ✅ SUCCESS ({int(execution_time)}ms) - "
                f"Analysts: {formatted_data['analyst_count']}, "
                f"Rating: {formatted_data['average_rating']}"
            )

            # Generate formatted context for LLM
            formatted_context = self._generate_formatted_context(formatted_data)

            return create_success_output(
                tool_name=self.schema.name,
                data=formatted_data,
                metadata={
                    "source": "FMP price-target-consensus + grades-consensus",
                    "symbol_queried": symbol_upper,
                    "execution_time_ms": int(execution_time),
                    "from_cache": False,
                    "cache_ttl": f"{self.CACHE_TTL}s",
                    "timestamp": datetime.now().isoformat()
                },
                formatted_context=formatted_context
            )
            
        except Exception as e:
            self.logger.error(
                f"[getPriceTargets] Error for {symbol_upper}: {e}",
                exc_info=True
            )
            return create_error_output(
                tool_name=self.schema.name,
                error=f"Failed to fetch price targets: {str(e)}"
            )
    
    async def _fetch_price_targets(self, symbol: str) -> Optional[Dict]:
        """Fetch price targets from FMP API"""
        url = f"{self.FMP_BASE_URL}/v4/price-target-consensus"
        
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # FMP returns list with single item or dict
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif isinstance(data, dict) and data:
                    return data
                
                return None
                
        except httpx.HTTPStatusError as e:
            self.logger.error(f"FMP HTTP error {e.response.status_code}")
            return None
        except Exception as e:
            self.logger.error(f"FMP request error: {e}")
            return None
    
    async def _fetch_analyst_ratings(self, symbol: str) -> Optional[Dict]:
        """
        Fetch analyst ratings from FMP grades-consensus API
        
        Response format:
        {
          "symbol": "AAPL",
          "strongBuy": 15,
          "buy": 10,
          "hold": 5,
          "sell": 1,
          "strongSell": 0
        }
        """
        url = f"{self.FMP_STABLE_BASE}/grades-consensus"
        
        params = {
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # FMP returns list or dict
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
                elif isinstance(data, dict) and data:
                    return data
                
                return None
                
        except Exception as e:
            self.logger.error(f"FMP grades-consensus error: {e}")
            return None
    
    def _format_targets_data(
        self,
        price_targets: Dict,
        ratings: Optional[Dict],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Format raw FMP responses to tool schema
        
        ✅ Returns ALL required fields:
        - consensus_target
        - high_target
        - low_target
        - analyst_count
        - average_rating
        """
        # ═══════════════════════════════════════════════════════════
        # Extract price target values
        # ═══════════════════════════════════════════════════════════
        target_high = self._safe_float(price_targets.get("targetHigh", 0.0))
        target_low = self._safe_float(price_targets.get("targetLow", 0.0))
        target_median = self._safe_float(price_targets.get("targetMedian", 0.0))
        target_consensus = self._safe_float(price_targets.get("targetConsensus", 0.0))
        
        # Number of analysts from price targets API
        num_analysts_from_targets = int(price_targets.get("numberOfAnalysts", 0))
        
        # Current price for comparison
        current_price = self._safe_float(price_targets.get("currentPrice", 0.0))
        
        # ═══════════════════════════════════════════════════════════
        # Calculate analyst metrics from ratings
        # ═══════════════════════════════════════════════════════════
        analyst_count = num_analysts_from_targets
        average_rating = "No Rating"
        rating_breakdown = {}
        
        if ratings and isinstance(ratings, dict):
            strong_buy = int(ratings.get("strongBuy", 0))
            buy = int(ratings.get("buy", 0))
            hold = int(ratings.get("hold", 0))
            sell = int(ratings.get("sell", 0))
            strong_sell = int(ratings.get("strongSell", 0))
            
            # Calculate total analysts from ratings
            total_from_ratings = strong_buy + buy + hold + sell + strong_sell
            
            # Use the larger count (more accurate)
            if total_from_ratings > analyst_count:
                analyst_count = total_from_ratings
            
            # Calculate weighted average rating
            if analyst_count > 0:
                # Scoring: Strong Buy=5, Buy=4, Hold=3, Sell=2, Strong Sell=1
                total_score = (
                    strong_buy * 5 +
                    buy * 4 +
                    hold * 3 +
                    sell * 2 +
                    strong_sell * 1
                )
                
                avg_score = total_score / analyst_count
                
                # Convert to label
                if avg_score >= 4.5:
                    average_rating = "Strong Buy"
                elif avg_score >= 3.5:
                    average_rating = "Buy"
                elif avg_score >= 2.5:
                    average_rating = "Hold"
                elif avg_score >= 1.5:
                    average_rating = "Sell"
                else:
                    average_rating = "Strong Sell"
            
            rating_breakdown = {
                "strong_buy": strong_buy,
                "buy": buy,
                "hold": hold,
                "sell": sell,
                "strong_sell": strong_sell
            }
        
        # ═══════════════════════════════════════════════════════════
        # Calculate additional metrics
        # ═══════════════════════════════════════════════════════════
        upside_pct = 0.0
        if current_price > 0 and target_consensus > 0:
            upside_pct = ((target_consensus - current_price) / current_price) * 100
        
        target_range_pct = 0.0
        if target_low > 0 and target_high > 0:
            target_range_pct = ((target_high - target_low) / target_low) * 100
        
        # ═══════════════════════════════════════════════════════════
        # Return ALL required fields (schema.returns)
        # ═══════════════════════════════════════════════════════════
        return {
            # ✅ Required fields from schema
            "symbol": symbol,
            "consensus_target": round(target_consensus, 2),
            "high_target": round(target_high, 2),
            "low_target": round(target_low, 2),
            "analyst_count": analyst_count,
            "average_rating": average_rating,
            "timestamp": datetime.now().isoformat(),
            
            # Additional fields (not in schema but useful)
            "median_target": round(target_median, 2),
            "rating_breakdown": rating_breakdown,
            "current_price": round(current_price, 2) if current_price > 0 else None,
            "upside_potential_pct": round(upside_pct, 2),
            "target_range_pct": round(target_range_pct, 2)
        }
    
    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    def _generate_formatted_context(self, data: Dict[str, Any]) -> str:
        """
        Generate human-readable formatted context for LLM consumption.

        Args:
            data: Formatted price targets data

        Returns:
            Human-readable string summary of analyst price targets
        """
        symbol = data.get("symbol", "Unknown")
        consensus = data.get("consensus_target", 0)
        high = data.get("high_target", 0)
        low = data.get("low_target", 0)
        analyst_count = data.get("analyst_count", 0)
        rating = data.get("average_rating", "No Rating")
        current_price = data.get("current_price")
        upside = data.get("upside_potential_pct", 0)
        breakdown = data.get("rating_breakdown", {})

        lines = [
            f"=== {symbol} ANALYST PRICE TARGETS ===",
            f"Consensus Target: ${consensus:,.2f}",
            f"Target Range: ${low:,.2f} - ${high:,.2f}",
            f"Analysts Covering: {analyst_count}",
            f"Average Rating: {rating}",
            ""
        ]

        # Current price comparison
        if current_price and current_price > 0:
            lines.append(f"Current Price: ${current_price:,.2f}")
            if upside >= 0:
                lines.append(f"Upside Potential: +{upside:.1f}%")
            else:
                lines.append(f"Downside Risk: {upside:.1f}%")
            lines.append("")

        # Rating breakdown
        if breakdown:
            lines.append("Rating Breakdown:")
            lines.append(f"  Strong Buy: {breakdown.get('strong_buy', 0)}")
            lines.append(f"  Buy: {breakdown.get('buy', 0)}")
            lines.append(f"  Hold: {breakdown.get('hold', 0)}")
            lines.append(f"  Sell: {breakdown.get('sell', 0)}")
            lines.append(f"  Strong Sell: {breakdown.get('strong_sell', 0)}")
            lines.append("")

        # Interpretation
        lines.append("Interpretation:")
        if rating in ["Strong Buy", "Buy"]:
            lines.append(f"  Analysts are BULLISH on {symbol}")
        elif rating in ["Strong Sell", "Sell"]:
            lines.append(f"  Analysts are BEARISH on {symbol}")
        else:
            lines.append(f"  Analysts have MIXED views on {symbol}")

        return "\n".join(lines)


# ============================================================================
# Standalone Testing
# ============================================================================

if __name__ == "__main__":
    import asyncio
    import os
    
    async def test_tool():
        """Test GetPriceTargetsTool with validation"""
        
        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            print("❌ ERROR: FMP_API_KEY not found")
            return
        
        print("=" * 80)
        print("TESTING [GetPriceTargetsTool] - FIXED VERSION")
        print("=" * 80)
        
        tool = GetPriceTargetsTool(api_key=api_key)
        
        # Test with AAPL
        print("\n📊 Test: AAPL")
        print("-" * 40)
        result = await tool.safe_execute(symbol="AAPL")
        
        print(f"Status: {result.status}")
        print(f"Execution time: {result.execution_time_ms}ms")
        
        if result.is_success():
            print("✅ SUCCESS")
            
            # Verify ALL required fields present
            required_fields = tool.schema.get_required_fields()
            missing = [f for f in required_fields if f not in result.data]
            
            if missing:
                print(f"\n⚠️  PARTIAL - Missing fields: {missing}")
            else:
                print(f"\n✅ ALL REQUIRED FIELDS PRESENT")
            
            print(f"\nData:")
            print(f"  Symbol: {result.data['symbol']}")
            print(f"  Consensus: ${result.data['consensus_target']:,.2f}")
            print(f"  High: ${result.data['high_target']:,.2f}")
            print(f"  Low: ${result.data['low_target']:,.2f}")
            print(f"  Analysts: {result.data['analyst_count']}")
            print(f"  Rating: {result.data['average_rating']}")
            
            if result.data.get('rating_breakdown'):
                bd = result.data['rating_breakdown']
                print(f"\n  Rating Breakdown:")
                print(f"    Strong Buy: {bd.get('strong_buy', 0)}")
                print(f"    Buy: {bd.get('buy', 0)}")
                print(f"    Hold: {bd.get('hold', 0)}")
                print(f"    Sell: {bd.get('sell', 0)}")
        else:
            print(f"❌ ERROR: {result.error}")
        
        print("\n" + "=" * 80)
    
    asyncio.run(test_tool())