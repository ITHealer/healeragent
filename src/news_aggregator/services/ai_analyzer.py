"""
AI Analyzer Service
===================

Uses LLM to analyze news articles and market data per symbol.
Produces structured analysis with citations for each claim.

Features:
- Per-symbol analysis with market context
- Citation tracking for every insight
- Sentiment analysis (bullish/bearish/neutral)
- Short-term and long-term outlook
- Risk factor identification

Output follows the Grok Tasks style:
- Structured report per symbol
- All claims cite source URLs
- Executive summary across symbols

Usage:
    analyzer = AIAnalyzer()
    result = await analyzer.analyze(
        articles_by_symbol={"TSLA": [...], "BTC": [...]},
        market_data={"TSLA": MarketData(...), ...},
        target_language="vi"
    )
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.news_aggregator.schemas.task import (
    ArticleContent,
    MarketData,
    NewsSource,
    SymbolAnalysis,
    SymbolInsight,
    SymbolType,
    TaskResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Prompts
# =============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are a senior financial analyst providing detailed analysis reports.

For each symbol, you will receive:
1. Market data (current price, 24h/7d/30d changes)
2. News articles with full content and source URLs
3. Optional: User instructions for customizing the analysis

Your task:
1. Analyze the news and market data for each symbol
2. Extract key insights WITH CITATIONS to source URLs
3. Determine sentiment (BULLISH, BEARISH, NEUTRAL, MIXED)
4. Provide short-term (1-7 days) and long-term (1-3 months) outlook
5. Identify risk factors
6. **If user provides instructions, prioritize and follow them**

CRITICAL: For EVERY claim or insight, you MUST cite the source using [N] format where N is the article number.

Output format (JSON):
{
    "analyses": [
        {
            "symbol": "TSLA",
            "display_name": "Tesla Inc.",
            "sentiment": "BULLISH",
            "sentiment_score": 0.7,
            "key_insights": [
                {
                    "text": "Insight text here",
                    "source_indices": [1, 2],
                    "sentiment": "bullish"
                }
            ],
            "short_term_outlook": "Test vùng $500 trong tuần tới",
            "long_term_outlook": "Mục tiêu $600 nếu robo-taxi thành công",
            "risk_factors": ["Factor 1", "Factor 2"]
        }
    ],
    "overall_sentiment": "MIXED",
    "key_themes": ["AI boom", "Crypto consolidation", "Fed policy"],
    "summary": "Executive summary in target language (2-3 sentences)"
}

Guidelines:
- Be specific with numbers and percentages
- Always cite sources for claims
- sentiment_score: -1.0 (very bearish) to +1.0 (very bullish)
- Write in the specified target language
- Be natural and engaging, not robotic
- **Follow user instructions carefully if provided**
"""

ANALYSIS_USER_PROMPT_TEMPLATE = """Analyze the following market data and news articles.
Target language: {target_language}
Date: {current_date}
{user_instructions}
## MARKET DATA
{market_data_text}

## NEWS ARTICLES BY SYMBOL
{articles_text}

Generate analysis following the JSON format specified. Output ONLY valid JSON."""

# User instruction examples for reference
USER_INSTRUCTION_EXAMPLES = {
    "focus_topic": "Chỉ phân tích tin tức liên quan đến AI và robo-taxi",
    "investor_perspective": "Phân tích từ góc độ nhà đầu tư dài hạn",
    "comparison": "So sánh Tesla với các đối thủ EV như Rivian, Lucid",
    "technical_focus": "Tập trung vào phân tích kỹ thuật và điểm hỗ trợ/kháng cự",
    "risk_assessment": "Đánh giá rủi ro và các yếu tố tiêu cực cần lưu ý",
    "specific_question": "Bitcoin có nên mua ở mức giá hiện tại không?",
}


# =============================================================================
# Symbol Display Names
# =============================================================================

SYMBOL_NAMES = {
    # Stocks
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "PLTR": "Palantir Technologies",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel Corporation",
    # Crypto
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "DOGE": "Dogecoin",
    "XRP": "Ripple",
    "SOL": "Solana",
    "ADA": "Cardano",
    "DOT": "Polkadot",
    "AVAX": "Avalanche",
    "MATIC": "Polygon",
    "LINK": "Chainlink",
}


class AIAnalyzer:
    """
    AI-powered news and market data analyzer.

    Produces structured analysis reports with citations
    similar to Grok Tasks output format.
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        provider_type: str = "openai",
    ):
        """
        Initialize AI analyzer.

        Args:
            model_name: LLM model to use
            provider_type: LLM provider (openai, ollama, etc.)
        """
        self.model_name = model_name
        self.provider_type = provider_type
        self.logger = logger
        self._llm_provider = None

    def _get_llm_provider(self):
        """Lazy load LLM provider."""
        if self._llm_provider is None:
            try:
                from src.helpers.llm_helper import LLMGeneratorProvider
                self._llm_provider = LLMGeneratorProvider()
            except ImportError:
                self.logger.error("[AIAnalyzer] LLMGeneratorProvider not available")
                raise
        return self._llm_provider

    def _get_symbol_name(self, symbol: str) -> str:
        """Get display name for symbol."""
        return SYMBOL_NAMES.get(symbol.upper(), symbol.upper())

    def _format_market_data(self, market_data: Dict[str, MarketData]) -> str:
        """Format market data for LLM prompt."""
        lines = []

        for symbol, data in market_data.items():
            line = f"**{symbol}** ({self._get_symbol_name(symbol)}): ${data.current_price:,.2f}"

            # Add changes
            changes_str = []
            for change in data.changes:
                sign = "+" if change.change_percent >= 0 else ""
                changes_str.append(f"{change.period}: {sign}{change.change_percent:.1f}%")

            if changes_str:
                line += f" ({', '.join(changes_str)})"

            if data.volume:
                line += f" | Volume: {data.volume:,}"

            lines.append(line)

        return "\n".join(lines)

    def _format_articles(
        self,
        articles_by_symbol: Dict[str, List[ArticleContent]],
    ) -> str:
        """Format articles for LLM prompt with numbered citations."""
        lines = []
        article_index = 1

        for symbol, articles in articles_by_symbol.items():
            lines.append(f"\n### {symbol} ({self._get_symbol_name(symbol)})")

            for article in articles:
                lines.append(f"\n[{article_index}] **{article.title}**")
                lines.append(f"Source: {article.source}")
                if article.published_at:
                    lines.append(f"Date: {article.published_at.strftime('%Y-%m-%d')}")
                lines.append(f"URL: {article.url}")

                # Content (truncate if too long)
                content = article.content[:3000] if article.content else article.snippet
                if content:
                    lines.append(f"Content: {content}")

                article_index += 1

        return "\n".join(lines)

    def _build_source_mapping(
        self,
        articles_by_symbol: Dict[str, List[ArticleContent]],
    ) -> Dict[int, NewsSource]:
        """Build mapping from article index to NewsSource."""
        mapping = {}
        index = 1

        for symbol, articles in articles_by_symbol.items():
            for article in articles:
                mapping[index] = NewsSource(
                    index=index,
                    title=article.title,
                    url=article.url,
                    source=article.source,
                    published_at=article.published_at,
                )
                index += 1

        return mapping

    def _parse_llm_response(
        self,
        response: str,
        source_mapping: Dict[int, NewsSource],
        market_data: Dict[str, MarketData],
    ) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        # Clean up response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            self.logger.error(f"[AIAnalyzer] JSON parse error: {e}")
            self.logger.debug(f"[AIAnalyzer] Response: {response[:500]}...")
            return {"analyses": [], "error": str(e)}

        # Process analyses and attach sources
        analyses = []
        for analysis_data in data.get("analyses", []):
            symbol = analysis_data.get("symbol", "")

            # Get market data for this symbol
            md = market_data.get(symbol)

            # Build sources list for this symbol
            sources = []
            for insight in analysis_data.get("key_insights", []):
                for idx in insight.get("source_indices", []):
                    if idx in source_mapping and source_mapping[idx] not in sources:
                        sources.append(source_mapping[idx])

            # Create SymbolAnalysis
            analysis = SymbolAnalysis(
                symbol=symbol,
                symbol_type=md.symbol_type if md else SymbolType.STOCK,
                display_name=analysis_data.get("display_name", self._get_symbol_name(symbol)),
                market_data=md,
                sentiment=analysis_data.get("sentiment", "NEUTRAL"),
                sentiment_score=analysis_data.get("sentiment_score", 0.0),
                key_insights=[
                    SymbolInsight(
                        text=ins.get("text", ""),
                        source_indices=ins.get("source_indices", []),
                        sentiment=ins.get("sentiment"),
                    )
                    for ins in analysis_data.get("key_insights", [])
                ],
                short_term_outlook=analysis_data.get("short_term_outlook"),
                long_term_outlook=analysis_data.get("long_term_outlook"),
                risk_factors=analysis_data.get("risk_factors", []),
                sources=sources,
            )
            analyses.append(analysis)

        return {
            "analyses": analyses,
            "overall_sentiment": data.get("overall_sentiment", "MIXED"),
            "key_themes": data.get("key_themes", []),
            "summary": data.get("summary"),
        }

    def _format_user_instructions(self, prompt: Optional[str]) -> str:
        """Format user instructions for the LLM prompt."""
        if not prompt or not prompt.strip():
            return ""

        return f"""
## USER INSTRUCTIONS (IMPORTANT - Follow these carefully!)
{prompt.strip()}

"""

    async def analyze(
        self,
        articles_by_symbol: Dict[str, List[ArticleContent]],
        market_data: Dict[str, MarketData],
        target_language: str = "vi",
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze news and market data for multiple symbols.

        Args:
            articles_by_symbol: Dict mapping symbol to list of articles
            market_data: Dict mapping symbol to MarketData
            target_language: Output language
            prompt: Optional user instructions to guide the analysis
                    Examples:
                    - "Chỉ phân tích tin tức liên quan đến AI và robo-taxi"
                    - "Phân tích từ góc độ nhà đầu tư dài hạn"
                    - "So sánh Tesla với các đối thủ EV như Rivian, Lucid"

        Returns:
            Dict with analyses, overall_sentiment, key_themes, summary
        """
        if not articles_by_symbol:
            return {"analyses": [], "error": "No articles to analyze"}

        self.logger.info(
            f"[AIAnalyzer] Analyzing {len(articles_by_symbol)} symbols "
            f"with {sum(len(a) for a in articles_by_symbol.values())} articles"
            + (f" | prompt: {prompt[:50]}..." if prompt else "")
        )

        # Build source mapping
        source_mapping = self._build_source_mapping(articles_by_symbol)

        # Format prompts
        market_data_text = self._format_market_data(market_data)
        articles_text = self._format_articles(articles_by_symbol)
        user_instructions = self._format_user_instructions(prompt)

        user_prompt = ANALYSIS_USER_PROMPT_TEMPLATE.format(
            target_language=target_language.upper(),
            current_date=datetime.utcnow().strftime("%Y-%m-%d"),
            user_instructions=user_instructions,
            market_data_text=market_data_text,
            articles_text=articles_text,
        )

        try:
            # Call LLM
            llm_provider = self._get_llm_provider()

            # Build messages in the format expected by LLMGeneratorProvider
            messages = [
                {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            response_data = await llm_provider.generate_response(
                model_name=self.model_name,
                messages=messages,
                provider_type=self.provider_type,
            )

            # Extract text content from response
            # Response format depends on provider, but typically has 'content' or 'text'
            if isinstance(response_data, dict):
                response = response_data.get("content") or response_data.get("text") or str(response_data)
            else:
                response = str(response_data)

            if not response:
                return {"analyses": [], "error": "Empty LLM response"}

            # Parse response
            result = self._parse_llm_response(
                response=response,
                source_mapping=source_mapping,
                market_data=market_data,
            )

            self.logger.info(
                f"[AIAnalyzer] Generated {len(result.get('analyses', []))} symbol analyses"
            )

            return result

        except Exception as e:
            self.logger.error(f"[AIAnalyzer] Analysis error: {e}", exc_info=True)
            return {"analyses": [], "error": str(e)}

    async def generate_report_title(
        self,
        symbols: List[str],
        target_language: str = "vi",
    ) -> str:
        """Generate report title based on symbols and language."""
        symbols_str = ", ".join([f"${s}" for s in symbols[:5]])

        titles = {
            "vi": f"Thông Tin Cập Nhật Về Các Tài Sản ({datetime.utcnow().strftime('%d/%m/%Y')})",
            "en": f"Asset Update Report ({datetime.utcnow().strftime('%Y-%m-%d')})",
            "zh": f"资产更新报告 ({datetime.utcnow().strftime('%Y-%m-%d')})",
            "ja": f"資産更新レポート ({datetime.utcnow().strftime('%Y-%m-%d')})",
            "ko": f"자산 업데이트 보고서 ({datetime.utcnow().strftime('%Y-%m-%d')})",
        }

        return titles.get(target_language.lower(), titles["en"])


# Singleton instance
_ai_analyzer: Optional[AIAnalyzer] = None


def get_ai_analyzer(
    model_name: str = "gpt-4.1-mini",
    provider_type: str = "openai",
) -> AIAnalyzer:
    """Get AI analyzer instance."""
    global _ai_analyzer
    if _ai_analyzer is None:
        _ai_analyzer = AIAnalyzer(model_name=model_name, provider_type=provider_type)
    return _ai_analyzer
