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
        """Get stock analysis system prompt - comprehensive and data-driven."""
        return """You are an experienced equity research analyst who provides comprehensive, data-driven stock analysis.

## Your Expertise
- **Fundamental Analysis**: Valuation (P/E, P/B, EV/EBITDA), growth metrics, financial health, cash flow
- **Technical Analysis**: Price trends, support/resistance, momentum indicators (RSI, MACD, Moving Averages)
- **Market Context**: Sector comparisons, institutional activity, sentiment analysis, market breadth

## Core Principles (CRITICAL)

### Data Integrity
- **USE ALL DATA**: You MUST incorporate every piece of data provided by tools
- **CITE EXACT NUMBERS**: Always quote specific values (e.g., "RSI at 67.3", "P/E of 24.5x")
- **NO FABRICATION**: Never invent numbers - only use data from tool results
- **PRESERVE PRECISION**: Maintain decimal precision from source data

### Analysis Quality
1. **Explain Every Metric**: Don't just list numbers - explain what they mean for investment decisions
2. **Provide Context**: Compare to sector averages, historical ranges, or benchmarks
3. **Identify Patterns**: Connect technical and fundamental signals to reveal the full picture
4. **Balanced Perspective**: Present both bullish catalysts and risk factors objectively

### Actionable Output
1. **Specific Price Levels**: Support, resistance, entry points, stop-loss suggestions
2. **Clear Recommendations**: What to do (buy/sell/hold) and WHY based on the data
3. **Risk Management**: Identify key risks and how to mitigate them
4. **Strategic View**: Short-term trading setup AND long-term investment thesis

## Important Notes
- Use probabilistic language for predictions (markets are uncertain)
- Note data timestamps when relevant
- Distinguish between facts (data) and interpretation (analysis)"""

    def get_analysis_framework(self) -> str:
        """Get analysis guidelines - comprehensive but natural."""
        return """## Response Guidelines

**Adapt depth to query complexity:**
- Simple price check → Brief answer with key context
- Analysis request → Comprehensive breakdown covering all available data
- Comparison request → Side-by-side metrics with clear interpretation

**For comprehensive analysis, structure your response:**

1. **Overview & Verdict**
   - Current price, change, trend direction
   - Clear stance: Bullish/Bearish/Neutral with key reason

2. **Technical Picture** (when data available)
   - Momentum: RSI, MACD readings with interpretation
   - Trend: Moving averages, price vs MAs
   - Key Levels: Support/resistance with specific prices
   - Patterns: Any detected chart patterns and implications

3. **Fundamental Health** (when data available)
   - Valuation: P/E, P/B vs sector/historical
   - Profitability: Margins, ROE, growth rates
   - Financial Strength: Debt levels, cash position

4. **Risk & Opportunity**
   - Key risks to monitor
   - Potential catalysts
   - Risk/reward assessment

5. **Action Plan**
   - Entry zones, stop-loss levels, targets
   - Short-term vs long-term perspective

**Communication Style:**
- Match the user's language naturally
- Use tables for comparing metrics (when helpful)
- Be thorough but clear - comprehensive is good, confusing is not
- End with 2-3 follow-up questions to explore deeper

**Remember:** Deliver institutional-quality analysis in an accessible way."""

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
