"""
Market Scanner Handler

Provides comprehensive market analysis using improved tools.
Consolidates multiple analysis types into 5 high-quality steps:
1. Technical & Chart Analysis
2. Market Position (Relative Strength)
3. Risk Analysis
4. Sentiment & News
5. Fundamental Analysis

Design principles (based on LLM optimization research):
- Use tool's llm_summary as PRIMARY source (optimized for LLM consumption)
- Keep raw_data for audit/verification only (not sent to LLM by default)
- Avoid derived logic that may conflict with tool's conclusions
- Clear facts hierarchy in prompts
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


# =============================================================================
# SYSTEM PROMPTS WITH FACTS HIERARCHY
# =============================================================================

TECHNICAL_SYSTEM_PROMPT = """You are a professional technical analyst providing clear, actionable insights.

## FACTS HIERARCHY (IMPORTANT)
When analyzing the data provided:
1. **PRIMARY SOURCE**: The tool's llm_summary is the authoritative analysis
2. **NUMBERS**: Always cite specific numbers from the data (prices, RSI, MACD values)
3. **CONFLICT RULE**: If you notice any inconsistency, trust the raw indicator values over interpretations

## OUTPUT STRUCTURE
Provide analysis in this order:
1. **TL;DR** (1-2 sentences summary with action)
2. **Trend Analysis** (short-term vs long-term)
3. **Momentum** (RSI, MACD, Stochastic readings)
4. **Volatility & Volume** (ATR%, BB squeeze, RVOL)
5. **Key Levels** (Support/Resistance with prices)
6. **Setup Opportunities** (only if clearly present in data)
7. **Invalidation** (what would invalidate the current bias)
8. **Action** (clear recommendation with reasoning)

## RULES
- Be specific: use exact numbers from the data
- Don't fabricate setups if data doesn't support them
- Always mention risk/invalidation levels
- Match user's language if specified"""

MARKET_POSITION_SYSTEM_PROMPT = """You are a professional market analyst specializing in relative strength analysis.

## FACTS HIERARCHY
1. **PRIMARY SOURCE**: The tool's RS scores and comparisons
2. **CONTEXT**: Use the multi-timeframe RS data to identify trends
3. **DON'T FABRICATE**: Only discuss what the data shows

## OUTPUT STRUCTURE
1. **Summary**: Is the stock outperforming or underperforming?
2. **Multi-timeframe**: 21d, 63d, 126d, 252d RS comparison
3. **Trend**: Is RS improving or declining?
4. **Implications**: What does this mean for positioning?"""

RISK_ANALYSIS_SYSTEM_PROMPT = """You are a professional risk manager providing clear risk analysis.

## FACTS HIERARCHY
1. **PRIMARY SOURCE**: Stop loss levels and volatility data from the tool
2. **POSITION SIZING**: Base on ATR% and volatility regime
3. **BE SPECIFIC**: Use exact price levels for stops

## OUTPUT STRUCTURE
1. **Volatility Assessment**: Current ATR%, volatility regime
2. **Stop Loss Levels**: ATR-based, Support-based, Percentage-based
3. **Position Sizing Guidance**: Based on risk tolerance
4. **Risk/Reward**: If entry price provided"""

SENTIMENT_NEWS_SYSTEM_PROMPT = """You are a financial analyst specializing in sentiment and news analysis.

## FACTS HIERARCHY
1. **SENTIMENT DATA**: Social sentiment scores and trends
2. **NEWS**: Recent headlines and their potential impact
3. **DON'T SPECULATE**: Base analysis on provided data only

## OUTPUT STRUCTURE
1. **Sentiment Overview**: Bullish/Bearish/Neutral with score
2. **Key News Themes**: Important recent developments
3. **Market Impact**: How might this affect the stock?
4. **Trading Implications**: What should traders watch for?"""


class MarketScannerHandler(LoggerMixin):
    """
    Handler for market scanning with 5 consolidated analysis steps.

    Design: Use tool's llm_summary directly (optimized for LLM),
    avoid derived logic that may conflict with tool conclusions.
    """

    def __init__(self):
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()

        # Initialize tools (lazy loading to avoid blocking)
        self._technical_tool = None
        self._rs_tool = None
        self._stop_loss_tool = None
        self._sentiment_tool = None
        self._news_tool = None

    # =========================================================================
    # LAZY TOOL INITIALIZATION (non-blocking)
    # =========================================================================
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

        Returns tool's llm_summary (optimized for LLM) as primary source.
        raw_data kept for audit/verification only.
        """
        try:
            result = await self.technical_tool.execute(
                symbol=symbol,
                timeframe=timeframe
            )

            if result.status == "error":
                return {
                    "success": False,
                    "error": result.error or "Technical analysis failed",
                    "symbol": symbol
                }

            data = result.data or {}

            # Return tool's output directly - llm_summary is already optimized
            return {
                "success": True,
                "symbol": symbol,
                "timeframe": timeframe,
                "llm_summary": data.get("llm_summary", ""),
                # Keep raw_data for audit only (not sent to LLM by default)
                "raw_data": data
            }

        except Exception as e:
            self.logger.error(f"[MarketScanner] Technical analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    # =========================================================================
    # STEP 2: Market Position (Relative Strength)
    # =========================================================================
    async def get_market_position(
        self,
        symbol: str,
        benchmark: str = "SPY"
    ) -> Dict[str, Any]:
        """Get relative strength analysis vs benchmark."""
        try:
            result = await self.rs_tool.execute(
                symbol=symbol,
                benchmark=benchmark,
                lookback_periods=[21, 63, 126, 252]
            )

            if result.status == "error":
                return {
                    "success": False,
                    "error": result.error or "RS analysis failed",
                    "symbol": symbol
                }

            data = result.data or {}

            return {
                "success": True,
                "symbol": symbol,
                "benchmark": benchmark,
                "llm_summary": data.get("llm_summary", result.formatted_context or ""),
                "raw_data": data
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
        """Get risk analysis with stop loss suggestions."""
        try:
            result = await self.stop_loss_tool.execute(
                symbol=symbol,
                entry_price=entry_price,
                lookback_days=60
            )

            if result.status == "error":
                return {
                    "success": False,
                    "error": result.error or "Risk analysis failed",
                    "symbol": symbol
                }

            data = result.data or {}

            return {
                "success": True,
                "symbol": symbol,
                "entry_price": entry_price,
                "llm_summary": data.get("llm_summary", result.formatted_context or ""),
                "raw_data": data
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
        """Get combined sentiment and news analysis."""
        try:
            # Get sentiment data
            sentiment_result = await self.sentiment_tool.execute(symbol=symbol)

            # Get news data
            news_result = await self.news_tool.execute(
                symbol=symbol,
                limit=news_limit
            )

            sentiment_data = (sentiment_result.data or {}) if sentiment_result.status != "error" else {}
            news_data = (news_result.data or {}) if news_result.status != "error" else {}

            # Combine summaries from both tools
            llm_summary = self._build_sentiment_news_summary(symbol, sentiment_data, news_data)

            return {
                "success": True,
                "symbol": symbol,
                "llm_summary": llm_summary,
                "raw_data": {
                    "sentiment": sentiment_data,
                    "news": news_data
                }
            }

        except Exception as e:
            self.logger.error(f"[MarketScanner] Sentiment/News analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    def _build_sentiment_news_summary(
        self,
        symbol: str,
        sentiment_data: Dict[str, Any],
        news_data: Dict[str, Any]
    ) -> str:
        """Build combined summary from sentiment and news tools."""
        lines = [f"=== SENTIMENT & NEWS ANALYSIS: {symbol} ===", ""]

        # Sentiment section
        if sentiment_data:
            sentiment_summary = sentiment_data.get("llm_summary", "")
            if sentiment_summary:
                lines.append("SOCIAL SENTIMENT:")
                lines.append(sentiment_summary)
            else:
                # Fallback to raw data
                score = sentiment_data.get("sentiment_score", 0)
                label = "BULLISH" if score > 0.3 else "BEARISH" if score < -0.3 else "NEUTRAL"
                lines.append(f"SOCIAL SENTIMENT: {label} (Score: {score:.2f})")
            lines.append("")

        # News section
        if news_data:
            news_summary = news_data.get("llm_summary", "")
            if news_summary:
                lines.append("RECENT NEWS:")
                lines.append(news_summary)
            else:
                # Fallback to raw articles
                articles = news_data.get("articles", [])
                if articles:
                    lines.append(f"RECENT NEWS: {len(articles)} articles")
                    for i, article in enumerate(articles[:5], 1):
                        title = article.get("title", "")[:100]
                        date = article.get("publishedDate", "")[:10]
                        lines.append(f"  {i}. [{date}] {title}")

        return "\n".join(lines) if len(lines) > 2 else f"No sentiment/news data available for {symbol}"

    # =========================================================================
    # LLM STREAMING - Using tool's llm_summary directly
    # =========================================================================
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
        """
        Stream technical analysis response from LLM.

        Uses tool's llm_summary directly (already optimized for LLM).
        """
        # Get technical data
        analysis_result = await self.get_technical_analysis(symbol, timeframe)

        if not analysis_result.get("success"):
            yield f"Error: {analysis_result.get('error', 'Analysis failed')}"
            return

        # Build prompt using tool's llm_summary (primary source)
        llm_summary = analysis_result.get("llm_summary", "")

        if not llm_summary:
            yield "Error: No analysis data available"
            return

        # Build user prompt
        prompt_parts = [
            f"=== TECHNICAL ANALYSIS DATA FOR {symbol} ({timeframe}) ===",
            "",
            llm_summary,
            "",
            "=== YOUR TASK ===",
            "Analyze the above data and provide insights following the output structure in your instructions.",
        ]

        if user_question:
            prompt_parts.extend([
                "",
                f"User's specific question: {user_question}",
                "Address this question while still providing the full analysis."
            ])

        if target_language:
            prompt_parts.extend([
                "",
                f"IMPORTANT: Respond entirely in {target_language}."
            ])

        prompt = "\n".join(prompt_parts)

        # Add chat history context if available
        if chat_history:
            prompt = f"[Previous conversation context]\n{chat_history}\n\n[Current analysis]\n{prompt}"

        messages = [
            {"role": "system", "content": TECHNICAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
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

        prompt_parts = [
            f"=== RELATIVE STRENGTH ANALYSIS: {symbol} vs {benchmark} ===",
            "",
            llm_summary if llm_summary else "No RS data available",
            "",
            "=== YOUR TASK ===",
            "Analyze the relative strength data following your output structure."
        ]

        if user_question:
            prompt_parts.extend(["", f"User question: {user_question}"])
        if target_language:
            prompt_parts.extend(["", f"IMPORTANT: Respond in {target_language}."])

        prompt = "\n".join(prompt_parts)

        if chat_history:
            prompt = f"[Previous context]\n{chat_history}\n\n{prompt}"

        messages = [
            {"role": "system", "content": MARKET_POSITION_SYSTEM_PROMPT},
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

        prompt_parts = [
            f"=== RISK ANALYSIS: {symbol} ===",
            ""
        ]

        if entry_price:
            prompt_parts.append(f"Entry Price: ${entry_price:.2f}")
            prompt_parts.append("")

        prompt_parts.extend([
            llm_summary if llm_summary else "No risk data available",
            "",
            "=== YOUR TASK ===",
            "Provide risk analysis following your output structure."
        ])

        if user_question:
            prompt_parts.extend(["", f"User question: {user_question}"])
        if target_language:
            prompt_parts.extend(["", f"IMPORTANT: Respond in {target_language}."])

        prompt = "\n".join(prompt_parts)

        if chat_history:
            prompt = f"[Previous context]\n{chat_history}\n\n{prompt}"

        messages = [
            {"role": "system", "content": RISK_ANALYSIS_SYSTEM_PROMPT},
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

        prompt_parts = [
            llm_summary if llm_summary else f"No sentiment/news data available for {symbol}",
            "",
            "=== YOUR TASK ===",
            "Analyze the sentiment and news data following your output structure."
        ]

        if user_question:
            prompt_parts.extend(["", f"User question: {user_question}"])
        if target_language:
            prompt_parts.extend(["", f"IMPORTANT: Respond in {target_language}."])

        prompt = "\n".join(prompt_parts)

        if chat_history:
            prompt = f"[Previous context]\n{chat_history}\n\n{prompt}"

        messages = [
            {"role": "system", "content": SENTIMENT_NEWS_SYSTEM_PROMPT},
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
