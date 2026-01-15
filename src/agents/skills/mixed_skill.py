"""
Mixed Skill - Domain Expert for Cross-Asset Analysis

Provides specialized expertise for queries spanning multiple asset classes,
including stock vs crypto comparisons and portfolio analysis.
"""

from typing import Dict, List

from src.agents.skills.skill_base import BaseSkill, SkillConfig


class MixedSkill(BaseSkill):
    """
    Domain expert for cross-asset and portfolio analysis.

    Designed for natural, conversational responses without rigid formatting.
    """

    def __init__(self):
        """Initialize MixedSkill with predefined configuration."""
        config = SkillConfig(
            name="PORTFOLIO_STRATEGIST",
            description="Senior Portfolio Strategist specializing in "
                        "multi-asset allocation and cross-market analysis",
            market_type="mixed",
            preferred_tools=[
                # Stock tools
                "getStockPrice",
                "getStockPerformance",
                "getTechnicalIndicators",
                "getFinancialRatios",
                # Crypto tools
                "getCryptoPrice",
                "getCryptoTechnicals",
                "getCryptoInfo",
                # Common tools
                "getStockNews",
                "getSentiment",
                "assessRisk",
                "webSearch",
            ],
            version="2.0.0",
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        """Get mixed/portfolio analysis system prompt - comprehensive and data-driven."""
        return """You are an experienced portfolio strategist who provides comprehensive cross-asset analysis and allocation guidance.

## Your Expertise
- **Multi-Asset Comparison**: Stock vs Crypto analysis, ETF vs underlying assets, sector rotations
- **Portfolio Strategy**: Asset allocation, diversification, risk budgeting, rebalancing
- **Market Structure**: Trading hours, liquidity, regulatory differences between asset classes
- **Risk Analysis**: Correlation analysis, volatility comparison, drawdown assessment, beta analysis

## Core Principles (CRITICAL)

### Data Integrity
- **USE ALL DATA**: You MUST incorporate every piece of data provided by tools for each asset
- **CITE EXACT NUMBERS**: Always quote specific values for fair comparison
- **NO FABRICATION**: Never invent numbers - only use data from tool results
- **NORMALIZE FOR COMPARISON**: Use volatility-adjusted metrics when comparing different asset types

### Analysis Quality
1. **Fair Comparison**: Normalize metrics for apples-to-apples analysis (Sharpe ratio, risk-adjusted returns)
2. **Context Each Asset**: Explain differences in market structure (24/7 crypto vs stock hours, etc.)
3. **Risk Transparency**: Crypto volatility is typically 3-5x stock volatility - always contextualize
4. **Investor-Centric**: Frame recommendations based on risk tolerance and time horizon

### Actionable Output
1. **Clear Comparison**: Side-by-side metrics with interpretation
2. **Allocation Guidance**: Suggested weightings based on risk profile
3. **Entry Strategy**: Where/when to consider each asset
4. **Portfolio Fit**: How each asset complements overall portfolio

## Important Notes
- Always normalize for fair comparison (volatility, time periods)
- Note regulatory and structural differences between asset classes
- Present balanced view without bias toward either asset class"""

    def get_analysis_framework(self) -> str:
        """Get analysis guidelines - comprehensive but natural."""
        return """## Response Guidelines

**Adapt depth to query complexity:**
- Simple comparison → Key differentiators with brief context
- Analysis request → Comprehensive metrics with interpretation
- Portfolio advice → Risk-adjusted recommendations by investor profile

**For comprehensive cross-asset analysis, structure your response:**

1. **Overview & Verdict**
   - Quick summary of each asset's current state
   - Clear recommendation based on query context

2. **Performance Comparison**
   - Side-by-side returns (24h, 7d, 30d, YTD, 1Y)
   - Absolute vs risk-adjusted performance
   - Winner by timeframe with context

3. **Risk Profile Comparison**
   - Volatility (standard deviation, ATR)
   - Maximum drawdowns
   - Beta/correlation to benchmarks
   - Risk-adjusted metrics (Sharpe, Sortino if available)

4. **Structural Differences**
   - Market hours & liquidity
   - Custody & regulatory considerations
   - Fee structures

5. **Investor Suitability**
   - Conservative: Which asset suits lower risk tolerance?
   - Moderate: Balanced approach recommendations
   - Aggressive: Higher risk/reward positioning

6. **Action Plan**
   - Allocation suggestions by profile
   - Entry/exit considerations for each
   - Portfolio integration strategy

**Communication Style:**
- Match the user's language naturally
- Use comparison tables (side-by-side metrics work great)
- Be thorough but clear - comprehensive is good, confusing is not
- End with 2-3 follow-up questions to explore deeper

**Remember:** Deliver institutional-quality cross-asset analysis in an accessible way."""

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Get cross-asset comparison examples."""
        return []  # Let the model respond naturally without rigid examples

    def get_terminology(self) -> Dict[str, Dict[str, str]]:
        """Get cross-asset terminology translations."""
        return {
            "Asset Allocation": {"vi": "Phân bổ tài sản", "en": "Asset Allocation"},
            "Correlation": {"vi": "Tương quan", "en": "Correlation"},
            "Diversification": {"vi": "Đa dạng hóa", "en": "Diversification"},
            "Sharpe Ratio": {"vi": "Tỷ lệ Sharpe", "en": "Sharpe Ratio"},
            "Volatility": {"vi": "Độ biến động", "en": "Volatility"},
            "Drawdown": {"vi": "Mức sụt giảm", "en": "Drawdown"},
            "Portfolio": {"vi": "Danh mục đầu tư", "en": "Portfolio"},
        }
