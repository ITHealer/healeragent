"""
Stock Skill - Domain Expert for Equity Analysis

Provides domain-specific expertise for stock/equity analysis with
natural, conversational responses like ChatGPT/Claude.
"""

from typing import Dict, List

from src.agents.skills.skill_base import BaseSkill, SkillConfig


class StockSkill(BaseSkill):
    """
    Domain expert for stock/equity analysis.

    Designed for natural, conversational responses without rigid formatting.
    """

    def __init__(self):
        """Initialize StockSkill with predefined configuration."""
        config = SkillConfig(
            name="STOCK_ANALYST",
            description="Senior Equity Research Analyst with expertise in "
                        "fundamental and technical analysis",
            market_type="stock",
            preferred_tools=[
                "getStockPrice",
                "getStockPerformance",
                "getTechnicalIndicators",
                "detectChartPatterns",
                "getSupportResistance",
                "getIncomeStatement",
                "getBalanceSheet",
                "getCashFlow",
                "getFinancialRatios",
                "getKeyMetrics",
                "getPriceTargets",
                "getAnalystEstimates",
                "getInsiderTrading",
                "getInstitutionalHoldings",
                "getStockNews",
                "getSentiment",
                "assessRisk",
                "getVolumeProfile",
                "suggestStopLoss",
            ],
            version="2.0.0",
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        """Get stock analysis system prompt - concise and natural."""
        return """You are an experienced equity research analyst who provides clear, data-driven stock analysis.

## Your Expertise
- **Fundamental Analysis**: Valuation (P/E, P/B, EV/EBITDA), growth metrics, financial health
- **Technical Analysis**: Price trends, support/resistance, momentum indicators (RSI, MACD)
- **Market Context**: Sector comparisons, institutional activity, sentiment analysis

## Analysis Principles
1. **Data-First**: Always cite specific numbers from tools (prices, ratios, percentages)
2. **Balanced View**: Present both opportunities and risks objectively
3. **Context Matters**: Compare metrics to sector averages and historical ranges
4. **Clarity**: Explain technical terms when they add value to understanding
5. **Actionable**: Provide specific price levels when discussing support/resistance/targets

## Important Notes
- Use probabilistic language for predictions (markets are uncertain)
- Note data timestamps when relevant
- Distinguish between facts (data) and interpretation (analysis)"""

    def get_analysis_framework(self) -> str:
        """Get analysis guidelines - flexible, not prescriptive."""
        return """## Response Guidelines

**Adapt your response to the query:**
- Simple price check → Brief answer with key context
- Detailed analysis → Comprehensive breakdown with data
- Comparison request → Side-by-side metrics with interpretation

**Always include when relevant:**
- Current price and recent performance (%, timeframe)
- Key technical levels (support, resistance)
- Relevant fundamental ratios with context
- Clear risk factors

**Communication:**
- Respond in the same language as the user's query
- Use tables for comparing multiple metrics
- Be conversational and educational, not robotic
- End with 2-3 relevant follow-up questions the user might want to explore

**Remember:** You're a helpful analyst having a conversation, not filling out a form."""

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Get stock analysis examples."""
        return []  # Let the model respond naturally without rigid examples

    def get_terminology(self) -> Dict[str, Dict[str, str]]:
        """Get stock terminology translations."""
        return {
            "P/E Ratio": {"vi": "Hệ số P/E", "en": "P/E Ratio"},
            "EPS": {"vi": "Thu nhập trên cổ phiếu", "en": "Earnings Per Share"},
            "Market Cap": {"vi": "Vốn hóa thị trường", "en": "Market Capitalization"},
            "Support": {"vi": "Ngưỡng hỗ trợ", "en": "Support Level"},
            "Resistance": {"vi": "Ngưỡng kháng cự", "en": "Resistance Level"},
            "RSI": {"vi": "Chỉ số sức mạnh tương đối", "en": "Relative Strength Index"},
            "MACD": {"vi": "Đường MACD", "en": "Moving Average Convergence Divergence"},
        }
