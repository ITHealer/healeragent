"""
Market Scanner Handler

Provides comprehensive market analysis using improved tools.
Consolidates multiple analysis types into 5 high-quality steps:
1. Technical & Chart Analysis
2. Market Position (Relative Strength)
3. Risk Analysis
4. Sentiment & News
5. Fundamental Analysis

Each step saves results to chat session for context continuity.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.tools.technical.get_technical_indicators import GetTechnicalIndicatorsTool
from src.agents.tools.technical.get_relative_strength import GetRelativeStrengthTool
from src.agents.tools.risk.suggest_stop_loss import SuggestStopLossTool
from src.agents.tools.risk.get_sentiment import GetSentimentTool
from src.agents.tools.news.get_stock_news import GetStockNewsTool
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory


class MarketScannerHandler(LoggerMixin):
    """
    Handler for market scanning with 5 consolidated analysis steps.
    Uses improved tools for better quality analysis.
    """

    def __init__(self):
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()

        # Initialize tools (lazy loading)
        self._technical_tool = None
        self._rs_tool = None
        self._stop_loss_tool = None
        self._sentiment_tool = None
        self._news_tool = None

    @property
    def technical_tool(self) -> GetTechnicalIndicatorsTool:
        if self._technical_tool is None:
            self._technical_tool = GetTechnicalIndicatorsTool()
        return self._technical_tool

    @property
    def rs_tool(self) -> GetRelativeStrengthTool:
        if self._rs_tool is None:
            self._rs_tool = GetRelativeStrengthTool()
        return self._rs_tool

    @property
    def stop_loss_tool(self) -> SuggestStopLossTool:
        if self._stop_loss_tool is None:
            self._stop_loss_tool = SuggestStopLossTool()
        return self._stop_loss_tool

    @property
    def sentiment_tool(self) -> GetSentimentTool:
        if self._sentiment_tool is None:
            self._sentiment_tool = GetSentimentTool()
        return self._sentiment_tool

    @property
    def news_tool(self) -> GetStockNewsTool:
        if self._news_tool is None:
            self._news_tool = GetStockNewsTool()
        return self._news_tool

    # =========================================================================
    # STEP 1: Technical & Chart Analysis
    # =========================================================================
    async def get_technical_analysis(
        self,
        symbol: str,
        timeframe: str = "1Y"
    ) -> Dict[str, Any]:
        """
        Get comprehensive technical analysis using improved tool.

        Returns:
            - Timeframe context
            - Trend regime (bullish/bearish/ranging)
            - Momentum indicators
            - Volatility metrics
            - Volume confirmation
            - Support/Resistance levels
            - Trading setups (pullback/breakout)
            - Invalidation levels
        """
        try:
            # Use improved technical indicators tool
            result = await self.technical_tool.execute(
                symbol=symbol,
                timeframe=timeframe
            )

            if result.get("status") == "error":
                return {
                    "success": False,
                    "error": result.get("error", "Technical analysis failed"),
                    "symbol": symbol
                }

            data = result.get("data", {})

            # Extract key components for structured output
            formatted_result = self._format_technical_output(symbol, timeframe, data)

            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "analysis": formatted_result,
                "raw_data": data,
                "llm_summary": data.get("llm_summary", "")
            }

        except Exception as e:
            self.logger.error(f"[MarketScanner] Technical analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    def _format_technical_output(
        self,
        symbol: str,
        timeframe: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format technical data into user-friendly structure."""
        indicators = data.get("indicators", {})
        outlook = data.get("outlook", {})
        rec = data.get("trading_recommendation", {})
        sr = data.get("support_resistance", {})
        price_ctx = data.get("price_context", {})

        # 1. TIMEFRAME CONTEXT
        timeframe_ctx = {
            "period": timeframe,
            "days": data.get("analysis_period_days"),
            "date_range": data.get("date_range"),
            "current_price": data.get("current_price")
        }

        # 2. TREND REGIME
        ma_data = indicators.get("moving_averages", {})
        adx_data = indicators.get("adx", {})
        trend_regime = {
            "overall": outlook.get("outlook", "NEUTRAL"),
            "short_term": ma_data.get("trend", "N/A"),
            "strength": adx_data.get("trend_strength", "N/A"),
            "adx_value": adx_data.get("adx"),
            "direction": adx_data.get("direction", "N/A"),
            "price_vs_sma20": ma_data.get("price_position", {}).get("above_sma_20"),
            "price_vs_sma50": ma_data.get("price_position", {}).get("above_sma_50"),
            "price_vs_sma200": ma_data.get("price_position", {}).get("above_sma_200")
        }

        # 3. MOMENTUM
        rsi_data = indicators.get("rsi", {})
        macd_data = indicators.get("macd", {})
        stoch_data = indicators.get("stochastic", {})
        momentum = {
            "rsi": {
                "value": rsi_data.get("value"),
                "condition": rsi_data.get("condition"),
                "signal": rsi_data.get("signal"),
                "trend": rsi_data.get("trend")
            },
            "macd": {
                "signal": macd_data.get("signal"),
                "histogram": macd_data.get("histogram"),
                "histogram_trend": macd_data.get("histogram_trend"),
                "crossover": macd_data.get("crossover")
            },
            "stochastic": {
                "k": stoch_data.get("k"),
                "d": stoch_data.get("d"),
                "condition": stoch_data.get("condition"),
                "signal": stoch_data.get("signal")
            }
        }

        # 4. VOLATILITY
        vol_pack = indicators.get("volatility_pack", {})
        bb_data = indicators.get("bollinger_bands", {})
        atr_data = indicators.get("atr", {})
        volatility = {
            "regime": vol_pack.get("volatility_regime", "NORMAL"),
            "atr_pct": vol_pack.get("atr_pct"),
            "atr_value": atr_data.get("value"),
            "bb_width_pct": vol_pack.get("bb_width_pct"),
            "bb_squeeze": vol_pack.get("bb_squeeze", False),
            "bb_position": bb_data.get("position"),
            "risk_note": vol_pack.get("risk_framing", "")
        }

        # 5. VOLUME CONFIRMATION
        vol_metrics = rec.get("volume_metrics", {})
        volume = {
            "rvol": vol_metrics.get("rvol"),
            "trend": vol_metrics.get("volume_trend"),
            "confirms_price": vol_metrics.get("volume_confirms_price", False),
            "confirmation_note": vol_metrics.get("volume_confirmation_note", "")
        }

        # 6. LEVELS (Support/Resistance)
        levels = {
            "support": sr.get("support_levels", []),
            "resistance": sr.get("resistance_levels", []),
            "pivot_points": sr.get("pivot_points", {})
        }

        # 7. TRADING SETUPS
        setups = self._identify_setups(data)

        # 8. INVALIDATION
        invalidation = self._calculate_invalidation(data)

        # 9. ACTION RECOMMENDATION
        action = {
            "recommendation": rec.get("overall_action", "HOLD"),
            "strength": rec.get("action_strength", "NEUTRAL"),
            "bullish_indicators": rec.get("signal_breakdown", {}).get("bullish_indicators", []),
            "bearish_indicators": rec.get("signal_breakdown", {}).get("bearish_indicators", []),
            "note": rec.get("note", "")
        }

        return {
            "timeframe": timeframe_ctx,
            "trend_regime": trend_regime,
            "momentum": momentum,
            "volatility": volatility,
            "volume": volume,
            "levels": levels,
            "setups": setups,
            "invalidation": invalidation,
            "action": action
        }

    def _identify_setups(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify pullback and breakout trading setups."""
        indicators = data.get("indicators", {})
        outlook = data.get("outlook", {})
        rec = data.get("trading_recommendation", {})

        pullback_setup = None
        breakout_setup = None

        # Get key values
        rsi = indicators.get("rsi", {}).get("value")
        bb_position = indicators.get("bollinger_bands", {}).get("position")
        bb_squeeze = indicators.get("volatility_pack", {}).get("bb_squeeze", False)
        trend = outlook.get("outlook", "NEUTRAL")
        vol_confirms = rec.get("volume_metrics", {}).get("volume_confirms_price", False)
        rvol = rec.get("volume_metrics", {}).get("rvol", 0)

        ma_data = indicators.get("moving_averages", {})
        price_pos = ma_data.get("price_position", {})
        above_sma20 = price_pos.get("above_sma_20")
        above_sma50 = price_pos.get("above_sma_50")

        current_price = data.get("current_price", 0)
        sma20 = ma_data.get("sma", {}).get("sma_20", {}).get("value")
        sma50 = ma_data.get("sma", {}).get("sma_50", {}).get("value")

        # PULLBACK SETUP: Uptrend + RSI pullback to 40-50 + price near SMA20
        if trend in ["BULLISH", "SLIGHTLY_BULLISH"] and above_sma50:
            if rsi and 35 <= rsi <= 55:
                # Check if price is within 3% of SMA20
                if sma20 and current_price:
                    distance_to_sma20 = abs(current_price - sma20) / sma20 * 100
                    if distance_to_sma20 <= 3:
                        pullback_setup = {
                            "type": "PULLBACK_TO_SMA20",
                            "quality": "HIGH" if vol_confirms else "MODERATE",
                            "entry": f"Near SMA20 (${sma20:.2f})",
                            "stop": f"Below SMA50 (${sma50:.2f})" if sma50 else "Below recent swing low",
                            "reason": f"Uptrend pullback - RSI at {rsi:.1f}, price {distance_to_sma20:.1f}% from SMA20"
                        }

        # BREAKOUT SETUP: BB Squeeze + increasing volume
        if bb_squeeze:
            rvol_float = float(rvol) if rvol else 0
            if rvol_float >= 1.2:
                direction = "BULLISH" if trend in ["BULLISH", "SLIGHTLY_BULLISH"] else "BEARISH" if trend in ["BEARISH", "SLIGHTLY_BEARISH"] else "WATCH"
                breakout_setup = {
                    "type": "BB_SQUEEZE_BREAKOUT",
                    "quality": "HIGH" if vol_confirms and rvol_float >= 1.5 else "MODERATE",
                    "direction": direction,
                    "rvol": rvol_float,
                    "reason": f"BB Squeeze detected with RVOL {rvol_float:.2f}x - breakout imminent"
                }

        return {
            "pullback": pullback_setup,
            "breakout": breakout_setup,
            "notes": [
                "Pullback: Look for RSI 40-50 in uptrend near SMA20",
                "Breakout: BB Squeeze + RVOL >= 1.2x confirms breakout"
            ]
        }

    def _calculate_invalidation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate invalidation levels for trade management."""
        indicators = data.get("indicators", {})
        sr = data.get("support_resistance", {})
        outlook = data.get("outlook", {})

        current_price = data.get("current_price", 0)
        trend = outlook.get("outlook", "NEUTRAL")

        ma_data = indicators.get("moving_averages", {})
        sma20 = ma_data.get("sma", {}).get("sma_20", {}).get("value")
        sma50 = ma_data.get("sma", {}).get("sma_50", {}).get("value")
        sma200 = ma_data.get("sma", {}).get("sma_200", {}).get("value")

        support_levels = sr.get("support_levels", [])
        resistance_levels = sr.get("resistance_levels", [])

        # For LONG positions (bullish bias)
        long_invalidation = []
        if sma20:
            long_invalidation.append({
                "level": sma20,
                "type": "SMA20",
                "severity": "WARNING",
                "note": "Close below SMA20 weakens short-term bullish case"
            })
        if sma50:
            long_invalidation.append({
                "level": sma50,
                "type": "SMA50",
                "severity": "CRITICAL",
                "note": "Close below SMA50 invalidates medium-term uptrend"
            })
        if support_levels:
            nearest_support = support_levels[0] if isinstance(support_levels[0], (int, float)) else support_levels[0].get("price", 0)
            long_invalidation.append({
                "level": nearest_support,
                "type": "SUPPORT",
                "severity": "CRITICAL",
                "note": "Break below key support level"
            })

        # For SHORT positions (bearish bias)
        short_invalidation = []
        if sma20:
            short_invalidation.append({
                "level": sma20,
                "type": "SMA20",
                "severity": "WARNING",
                "note": "Close above SMA20 weakens short-term bearish case"
            })
        if sma50:
            short_invalidation.append({
                "level": sma50,
                "type": "SMA50",
                "severity": "CRITICAL",
                "note": "Close above SMA50 invalidates medium-term downtrend"
            })
        if resistance_levels:
            nearest_resistance = resistance_levels[0] if isinstance(resistance_levels[0], (int, float)) else resistance_levels[0].get("price", 0)
            short_invalidation.append({
                "level": nearest_resistance,
                "type": "RESISTANCE",
                "severity": "CRITICAL",
                "note": "Break above key resistance level"
            })

        return {
            "long_positions": long_invalidation,
            "short_positions": short_invalidation,
            "current_bias": "LONG" if trend in ["BULLISH", "SLIGHTLY_BULLISH"] else "SHORT" if trend in ["BEARISH", "SLIGHTLY_BEARISH"] else "NEUTRAL"
        }

    # =========================================================================
    # STEP 2: Market Position (Relative Strength)
    # =========================================================================
    async def get_market_position(
        self,
        symbol: str,
        benchmark: str = "SPY"
    ) -> Dict[str, Any]:
        """
        Get relative strength analysis vs benchmark.

        Returns:
            - RS score vs benchmark
            - Outperforming/Underperforming status
            - Sector context
        """
        try:
            result = await self.rs_tool.execute(
                symbol=symbol,
                benchmark=benchmark,
                lookback_periods=[21, 63, 126, 252]
            )

            if result.get("status") == "error":
                return {
                    "success": False,
                    "error": result.get("error", "RS analysis failed"),
                    "symbol": symbol
                }

            data = result.get("data", {})

            return {
                "success": True,
                "symbol": symbol,
                "benchmark": benchmark,
                "analysis": data,
                "llm_summary": data.get("llm_summary", result.get("formatted_context", ""))
            }

        except Exception as e:
            self.logger.error(f"[MarketScanner] RS analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    # =========================================================================
    # STEP 3: Risk Analysis
    # =========================================================================
    async def get_risk_analysis(
        self,
        symbol: str,
        entry_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get risk analysis with stop loss suggestions.

        Returns:
            - Stop loss levels (ATR-based, Support-based, Percentage)
            - Risk metrics
            - Position sizing guidance
        """
        try:
            result = await self.stop_loss_tool.execute(
                symbol=symbol,
                entry_price=entry_price,
                lookback_days=60
            )

            if result.get("status") == "error":
                return {
                    "success": False,
                    "error": result.get("error", "Risk analysis failed"),
                    "symbol": symbol
                }

            data = result.get("data", {})

            return {
                "success": True,
                "symbol": symbol,
                "analysis": data,
                "llm_summary": data.get("llm_summary", result.get("formatted_context", ""))
            }

        except Exception as e:
            self.logger.error(f"[MarketScanner] Risk analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    # =========================================================================
    # STEP 4: Sentiment & News
    # =========================================================================
    async def get_sentiment_news(
        self,
        symbol: str,
        news_limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get combined sentiment and news analysis.

        Returns:
            - Social sentiment data
            - Recent news with analysis
            - Combined sentiment score
        """
        try:
            # Get sentiment data
            sentiment_result = await self.sentiment_tool.execute(symbol=symbol)

            # Get news data
            news_result = await self.news_tool.execute(
                symbol=symbol,
                limit=news_limit
            )

            sentiment_data = sentiment_result.get("data", {}) if sentiment_result.get("status") != "error" else {}
            news_data = news_result.get("data", {}) if news_result.get("status") != "error" else {}

            return {
                "success": True,
                "symbol": symbol,
                "sentiment": sentiment_data,
                "news": news_data,
                "llm_summary": self._combine_sentiment_news_summary(sentiment_data, news_data)
            }

        except Exception as e:
            self.logger.error(f"[MarketScanner] Sentiment/News analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    def _combine_sentiment_news_summary(
        self,
        sentiment_data: Dict[str, Any],
        news_data: Dict[str, Any]
    ) -> str:
        """Combine sentiment and news into a summary."""
        lines = []

        # Sentiment section
        if sentiment_data:
            sentiment_score = sentiment_data.get("sentiment_score", 0)
            sentiment_label = "BULLISH" if sentiment_score > 0.3 else "BEARISH" if sentiment_score < -0.3 else "NEUTRAL"
            lines.append(f"SOCIAL SENTIMENT: {sentiment_label} (Score: {sentiment_score:.2f})")

        # News section
        if news_data:
            articles = news_data.get("articles", [])
            if articles:
                lines.append(f"\nRECENT NEWS: {len(articles)} articles")
                for i, article in enumerate(articles[:3], 1):
                    title = article.get("title", "No title")
                    lines.append(f"  {i}. {title[:80]}...")

        return "\n".join(lines) if lines else "No sentiment/news data available"

    # =========================================================================
    # LLM STREAMING HELPERS
    # =========================================================================
    def create_technical_prompt(
        self,
        symbol: str,
        analysis_data: Dict[str, Any],
        user_question: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> str:
        """Create prompt for technical analysis LLM response."""
        llm_summary = analysis_data.get("llm_summary", "")
        formatted = analysis_data.get("analysis", {})

        # Build prompt
        prompt_parts = [
            f"=== TECHNICAL ANALYSIS DATA FOR {symbol} ===",
            "",
            llm_summary if llm_summary else json.dumps(formatted, indent=2, default=str),
            "",
            "=== ANALYSIS REQUIREMENTS ===",
            "Based on the above technical data, provide a comprehensive analysis covering:",
            "1. **Trend Regime**: Current trend direction and strength",
            "2. **Momentum Status**: RSI, MACD, Stochastic readings",
            "3. **Volatility Assessment**: ATR, BB width, squeeze status",
            "4. **Volume Confirmation**: Does volume support price action?",
            "5. **Key Levels**: Important support/resistance",
            "6. **Trading Setups**: Any pullback or breakout opportunities?",
            "7. **Invalidation**: What would invalidate the current bias?",
            "8. **Action**: Clear recommendation with reasoning",
            ""
        ]

        if user_question:
            prompt_parts.append(f"User's specific question: {user_question}")
            prompt_parts.append("")

        if target_language:
            prompt_parts.append(f"IMPORTANT: Respond entirely in {target_language}.")

        return "\n".join(prompt_parts)

    async def stream_technical_analysis(
        self,
        symbol: str,
        timeframe: str,
        model_name: str,
        provider_type: str,
        api_key: str,
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        chat_history: str = ""
    ):
        """Stream technical analysis response from LLM."""
        # Get technical data
        analysis_result = await self.get_technical_analysis(symbol, timeframe)

        if not analysis_result.get("success"):
            yield f"Error: {analysis_result.get('error', 'Analysis failed')}"
            return

        # Create prompt
        prompt = self.create_technical_prompt(
            symbol=symbol,
            analysis_data=analysis_result,
            user_question=user_question,
            target_language=target_language
        )

        # Add chat history if available
        if chat_history:
            full_prompt = f"Previous context:\n{chat_history}\n\nCurrent analysis:\n{prompt}"
        else:
            full_prompt = prompt

        messages = [
            {
                "role": "system",
                "content": """You are a professional technical analyst. Analyze the provided data and give clear, actionable insights.

Key principles:
- Be specific with numbers and levels
- Explain the significance of each indicator
- Identify trading setups when present
- Always mention invalidation levels
- Use clear structure with headers
- Match the user's language if specified"""
            },
            {"role": "user", "content": full_prompt}
        ]

        # Stream response
        async for chunk in self.llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key
        ):
            yield chunk

    async def stream_market_position(
        self,
        symbol: str,
        benchmark: str,
        model_name: str,
        provider_type: str,
        api_key: str,
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        chat_history: str = ""
    ):
        """Stream market position analysis from LLM."""
        analysis_result = await self.get_market_position(symbol, benchmark)

        if not analysis_result.get("success"):
            yield f"Error: {analysis_result.get('error', 'Analysis failed')}"
            return

        llm_summary = analysis_result.get("llm_summary", "")

        prompt = f"""=== RELATIVE STRENGTH ANALYSIS: {symbol} vs {benchmark} ===

{llm_summary}

Provide analysis covering:
1. **Relative Performance**: How is {symbol} performing vs {benchmark}?
2. **Trend**: Is relative strength improving or declining?
3. **Sector Context**: How does this fit in the broader market?
4. **Implications**: What does this mean for positioning?
"""

        if user_question:
            prompt += f"\nUser question: {user_question}"
        if target_language:
            prompt += f"\n\nIMPORTANT: Respond in {target_language}."

        if chat_history:
            prompt = f"Previous context:\n{chat_history}\n\n{prompt}"

        messages = [
            {"role": "system", "content": "You are a professional market analyst specializing in relative strength analysis."},
            {"role": "user", "content": prompt}
        ]

        async for chunk in self.llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key
        ):
            yield chunk

    async def stream_risk_analysis(
        self,
        symbol: str,
        model_name: str,
        provider_type: str,
        api_key: str,
        entry_price: Optional[float] = None,
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        chat_history: str = ""
    ):
        """Stream risk analysis from LLM."""
        analysis_result = await self.get_risk_analysis(symbol, entry_price)

        if not analysis_result.get("success"):
            yield f"Error: {analysis_result.get('error', 'Analysis failed')}"
            return

        llm_summary = analysis_result.get("llm_summary", "")
        data = analysis_result.get("analysis", {})

        prompt = f"""=== RISK ANALYSIS: {symbol} ===

{llm_summary}

Raw data:
{json.dumps(data, indent=2, default=str)}

Provide analysis covering:
1. **Stop Loss Levels**: Recommended stop levels and reasoning
2. **Risk Assessment**: Current volatility and risk profile
3. **Position Sizing**: Guidance based on risk tolerance
4. **Risk/Reward**: Potential setups and their R:R ratios
"""

        if entry_price:
            prompt += f"\nEntry price provided: ${entry_price:.2f}"
        if user_question:
            prompt += f"\nUser question: {user_question}"
        if target_language:
            prompt += f"\n\nIMPORTANT: Respond in {target_language}."

        if chat_history:
            prompt = f"Previous context:\n{chat_history}\n\n{prompt}"

        messages = [
            {"role": "system", "content": "You are a professional risk manager. Provide clear risk analysis with specific levels and position sizing guidance."},
            {"role": "user", "content": prompt}
        ]

        async for chunk in self.llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key
        ):
            yield chunk

    async def stream_sentiment_news(
        self,
        symbol: str,
        model_name: str,
        provider_type: str,
        api_key: str,
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        chat_history: str = ""
    ):
        """Stream sentiment and news analysis from LLM."""
        analysis_result = await self.get_sentiment_news(symbol)

        if not analysis_result.get("success"):
            yield f"Error: {analysis_result.get('error', 'Analysis failed')}"
            return

        llm_summary = analysis_result.get("llm_summary", "")
        sentiment = analysis_result.get("sentiment", {})
        news = analysis_result.get("news", {})

        prompt = f"""=== SENTIMENT & NEWS ANALYSIS: {symbol} ===

{llm_summary}

Sentiment Data:
{json.dumps(sentiment, indent=2, default=str) if sentiment else "No sentiment data available"}

News Data:
{json.dumps(news, indent=2, default=str) if news else "No news data available"}

Provide analysis covering:
1. **Sentiment Overview**: Social media and news sentiment
2. **Key News Themes**: Important recent developments
3. **Market Impact**: How might this affect the stock?
4. **Trading Implications**: What should traders watch for?
"""

        if user_question:
            prompt += f"\nUser question: {user_question}"
        if target_language:
            prompt += f"\n\nIMPORTANT: Respond in {target_language}."

        if chat_history:
            prompt = f"Previous context:\n{chat_history}\n\n{prompt}"

        messages = [
            {"role": "system", "content": "You are a financial news analyst. Analyze sentiment and news data to provide actionable insights."},
            {"role": "user", "content": prompt}
        ]

        async for chunk in self.llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key
        ):
            yield chunk


# Singleton instance
market_scanner_handler = MarketScannerHandler()
