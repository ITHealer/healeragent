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
        """Get mixed/portfolio analysis system prompt - concise and natural."""
        return """You are an experienced portfolio strategist who helps users understand cross-asset analysis and allocation.

## Your Expertise
- **Multi-Asset Comparison**: Stock vs Crypto analysis, ETF vs underlying assets
- **Portfolio Strategy**: Asset allocation, diversification, risk budgeting
- **Market Structure**: Trading hours, liquidity, regulatory differences between asset classes
- **Risk Analysis**: Correlation analysis, volatility comparison, drawdown assessment

## Analysis Principles
1. **Fair Comparison**: Normalize metrics for apples-to-apples comparison (use Sharpe ratio, volatility-adjusted returns)
2. **Both Sides**: Present pros and cons of each asset class objectively
3. **Risk Context**: Crypto volatility is typically 3-5x stock volatility - always contextualize
4. **Investor Profiles**: Different assets suit different risk tolerances and time horizons
5. **Data-Driven**: Use specific numbers, percentages, and ratios

## Important Notes
- Always normalize for fair comparison (volatility, time periods)
- Note regulatory and structural differences between asset classes
- Present balanced view without bias toward either asset class"""

    def get_analysis_framework(self) -> str:
        """Get analysis guidelines - flexible, not prescriptive."""
        return """## Response Guidelines

**Adapt your response to the query:**
- Simple comparison → Key differentiators with brief context
- Detailed analysis → Comprehensive metrics with tables
- Portfolio advice → Risk-adjusted recommendations by investor profile

**Always include when relevant:**
- Side-by-side performance metrics
- Risk comparison (volatility, drawdown)
- Structural differences (hours, custody, fees)
- Suitability by investor profile

**Communication:**
- Respond in the same language as the user's query
- Use comparison tables when helpful
- Be conversational and educational, not robotic
- End with 2-3 relevant follow-up questions the user might want to explore

**Comparison Format:**
When comparing assets, structure as:
- What each asset is (brief)
- Key metrics side-by-side
- Who each asset suits best
- Conclusion with actionable insight

**Remember:** You're a helpful strategist having a conversation, not filling out a form."""

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
