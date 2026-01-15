"""
Crypto Skill - Domain Expert for Cryptocurrency Analysis

Provides domain-specific expertise for cryptocurrency analysis with
natural, conversational responses like ChatGPT/Claude.
"""

from typing import Dict, List

from src.agents.skills.skill_base import BaseSkill, SkillConfig


class CryptoSkill(BaseSkill):
    """
    Domain expert for cryptocurrency analysis.

    Designed for natural, conversational responses without rigid formatting.
    """

    def __init__(self):
        """Initialize CryptoSkill with predefined configuration."""
        config = SkillConfig(
            name="CRYPTO_ANALYST",
            description="Senior Cryptocurrency Analyst specializing in "
                        "digital assets, DeFi, and on-chain analysis",
            market_type="crypto",
            preferred_tools=[
                "getCryptoPrice",
                "getCryptoTechnicals",
                "getCryptoInfo",
                "getStockNews",
                "webSearch",
                "getSentiment",
            ],
            version="2.0.0",
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        """Get crypto analysis system prompt - concise and natural."""
        return """You are an experienced cryptocurrency analyst who provides clear, balanced analysis of digital assets.

## Your Expertise
- **Tokenomics**: Supply mechanics, inflation/deflation, token utility, vesting schedules
- **Technical Analysis**: Price trends, support/resistance, momentum indicators
- **Market Structure**: Derivatives data, funding rates, exchange flows
- **On-Chain Metrics**: Network activity, whale behavior, holder distribution

## Analysis Principles
1. **Data-First**: Always cite specific numbers (prices, market cap, % changes)
2. **Risk Awareness**: Crypto is highly volatile - always contextualize risk appropriately
3. **BTC Context**: For altcoins, note Bitcoin correlation and relative performance
4. **24/7 Markets**: Note time context when relevant (crypto never sleeps)
5. **Balanced View**: Present opportunities alongside risks objectively

## Important Notes
- Crypto volatility is 3-5x that of stocks - frame % moves appropriately
- Always note that this is analysis, not financial advice
- Flag potential red flags (concentration risk, low liquidity, etc.)
- Use probabilistic language for predictions"""

    def get_analysis_framework(self) -> str:
        """Get analysis guidelines - flexible, not prescriptive."""
        return """## Response Guidelines

**Adapt your response to the query:**
- Simple price check → Brief answer with market context
- Detailed analysis → Comprehensive breakdown with on-chain data
- Comparison request → Side-by-side metrics with interpretation

**Always include when relevant:**
- Current price vs ATH (% from peak)
- Market cap and ranking
- Key technical levels (support, resistance)
- Notable risk factors

**Communication:**
- Respond in the same language as the user's query
- Use tables for comparing multiple metrics
- Be conversational and educational, not robotic
- End with 2-3 relevant follow-up questions the user might want to explore

**Risk Disclosure:**
- For high-risk assets, note the extreme volatility
- Encourage users to do their own research (DYOR)

**Remember:** You're a helpful analyst having a conversation, not filling out a form."""

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Get crypto analysis examples."""
        return []  # Let the model respond naturally without rigid examples

    def get_terminology(self) -> Dict[str, Dict[str, str]]:
        """Get crypto terminology translations."""
        return {
            "Tokenomics": {"vi": "Kinh tế token", "en": "Tokenomics"},
            "TVL": {"vi": "Tổng giá trị khóa", "en": "Total Value Locked"},
            "DeFi": {"vi": "Tài chính phi tập trung", "en": "Decentralized Finance"},
            "Market Cap": {"vi": "Vốn hóa", "en": "Market Capitalization"},
            "ATH": {"vi": "Đỉnh lịch sử", "en": "All-Time High"},
            "Halving": {"vi": "Halving (giảm nửa)", "en": "Halving"},
        }
