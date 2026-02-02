"""
Crypto Skill - Domain Expert for Cryptocurrency Analysis

Provides domain-specific expertise for cryptocurrency analysis with
natural, conversational responses like ChatGPT/Claude.

Supports hierarchical synthesis with crypto-specific phases.
"""

from typing import Dict, List, Optional

from src.agents.skills.skill_base import (
    BaseSkill,
    Phase,
    PhaseSummary,
    PhaseType,
    SkillConfig,
    SkillContext,
)


class CryptoSkill(BaseSkill):
    """
    Domain expert for cryptocurrency analysis.

    Supports hierarchical synthesis with 2 phases:
    1. Technical & Price: Price data, technical indicators, chart patterns
    2. Context & Fundamentals: Tokenomics, news, sentiment, market context
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
            version="3.0.0",
            enable_hierarchical_synthesis=True,
        )
        super().__init__(config)

    # ====================================================================
    # Hierarchical Synthesis: Phases
    # ====================================================================

    def get_phases(self, context: Optional[SkillContext] = None) -> List[Phase]:
        """Define crypto analysis execution phases."""
        phases = []

        # Phase 1: Technical & Price
        phases.append(Phase(
            name="technical",
            display_name="Technical & Price Analysis",
            phase_type=PhaseType.TECHNICAL,
            tools=["getCryptoPrice", "getCryptoTechnicals"],
            synthesis_focus=(
                "Summarize current price, 24h/7d/30d performance, "
                "technical indicators (RSI, MACD, MAs), support/resistance, "
                "volume analysis, and chart patterns with exact numbers."
            ),
            max_summary_tokens=500,
            priority=1,
        ))

        # Phase 2: Context & Fundamentals
        phases.append(Phase(
            name="context",
            display_name="Market Context & Fundamentals",
            phase_type=PhaseType.CONTEXT,
            tools=["getCryptoInfo", "getStockNews", "getSentiment", "webSearch"],
            synthesis_focus=(
                "Summarize tokenomics (supply, market cap, rank), "
                "recent news and catalysts, market sentiment, "
                "and any fundamental developments."
            ),
            max_summary_tokens=400,
            priority=2,
        ))

        return phases

    def get_phase_synthesis_prompt(
        self,
        phase: Phase,
        context: Optional[SkillContext] = None,
    ) -> str:
        """Get phase-specific synthesis prompt for crypto analysis."""
        phase_templates = {
            "technical": (
                "Create a structured crypto technical summary:\n"
                "1. Current Price (price, 24h change, market cap, rank)\n"
                "2. Performance (24h, 7d, 30d, YTD returns)\n"
                "3. Key Indicators (RSI, MACD with exact values)\n"
                "4. Support/Resistance (specific price levels)\n"
                "5. Volume (24h volume, vs average, buy/sell pressure)\n"
                "6. Technical Outlook (1-2 sentences)"
            ),
            "context": (
                "Create a structured market context summary:\n"
                "1. Tokenomics (circulating/total supply, inflation, utility)\n"
                "2. Recent News (top 2-3 material items with impact)\n"
                "3. Sentiment (market mood, social mentions, fear/greed)\n"
                "4. Catalysts & Risks (upcoming events, regulatory threats)\n"
                "5. Context Outlook (1-2 sentences)"
            ),
        }

        template = phase_templates.get(phase.name, "")
        base_prompt = super().get_phase_synthesis_prompt(phase, context)

        if template:
            return f"{base_prompt}\n\nStructure:\n{template}"
        return base_prompt

    def _get_final_sections(self, context: Optional[SkillContext] = None) -> List[str]:
        """Crypto-specific final report sections."""
        return [
            "TL;DR (2-3 sentences: price, verdict, key action)",
            "Market Context (BTC dominance, macro, risk sentiment)",
            "Technical Picture (momentum, trend, key levels)",
            "Fundamental & On-Chain (tokenomics, network health)",
            "Scenario Analysis (bull case with target, bear case with risk)",
            "Action Plan (entry/target/stop with position type)",
        ]

    def get_system_prompt(self) -> str:
        """Get crypto analysis system prompt - comprehensive and data-driven."""
        return """You are an experienced cryptocurrency analyst who provides comprehensive, data-driven analysis of digital assets.

## Your Expertise
- **Tokenomics**: Supply mechanics, inflation/deflation, token utility, vesting schedules
- **Technical Analysis**: Price trends, support/resistance, momentum indicators, chart patterns
- **Market Structure**: Derivatives data, funding rates, exchange flows, liquidity analysis
- **On-Chain Metrics**: Network activity, whale behavior, holder distribution, TVL trends

## Core Principles (CRITICAL)

### Data Integrity
- **USE ALL DATA**: You MUST incorporate every piece of data provided by tools
- **CITE EXACT NUMBERS**: Always quote specific values (e.g., "BTC at $67,432", "RSI at 58.2")
- **NO FABRICATION**: Never invent numbers or fake confidence percentages
- **SOURCE ATTRIBUTION**: State data source (e.g., "Source: CoinGecko API")

### Market Context (REQUIRED for comprehensive analysis)
1. **BTC Dominance**: How is BTC performing? Altcoin season or BTC season?
2. **Macro Correlation**: Risk-on/risk-off sentiment, correlation with stocks/DXY
3. **News & Catalysts**: Recent news, upcoming events (halvings, upgrades, regulations)

### Recommendation Logic (CRITICAL - NO CONTRADICTIONS)
1. **LONG Setup**: Entry < Target, Stop below Entry
   - Example: Entry $60,000, Target $75,000, Stop $55,000
2. **SHORT Setup**: Entry > Target, Stop above Entry
   - Example: Entry $60,000, Target $50,000, Stop $65,000
3. **ALWAYS** specify if recommendation is LONG or SHORT

### Scenario Analysis (REQUIRED)
1. **Bull Case**: Upside target + probability (e.g., "55% probability")
2. **Bear Case**: Downside target + probability (e.g., "45% probability")
3. **Invalidation**: What would invalidate each scenario

## Important Notes
- Crypto volatility is 3-5x that of stocks - frame % moves appropriately
- Always note that this is analysis, not financial advice
- Flag red flags (concentration risk, low liquidity, rug pull risks)
- Don't use fake "confidence %" for patterns without methodology"""

    def get_analysis_framework(self) -> str:
        """Get analysis guidelines - comprehensive but natural."""
        return """## Response Guidelines

**Adapt depth to query complexity:**
- Simple price check → Brief answer with market context
- Analysis request → Comprehensive breakdown covering all available data
- Comparison request → Side-by-side metrics with clear interpretation

**For comprehensive analysis, structure your response:**

### 0. **TL;DR / Executive Summary** (ALWAYS START WITH THIS)
   - 2-3 sentences: Current status, verdict, key action
   - Example: "BTC at $67K, consolidating after ATH. HOLD positions, accumulate on dips to $62K. Bull case $80K (60%), Bear case $55K (40%)."

### 1. **Market Context** (REQUIRED)
   - BTC Dominance: Is it BTC season or altcoin season?
   - Macro: Risk-on/risk-off sentiment, DXY, stock market correlation
   - News: Recent catalysts, upcoming events

### 2. **Technical Picture** (when data available)
   - Momentum: RSI, MACD with interpretation
   - Key Levels: Support/resistance with specific prices
   - Volume: Trading activity vs average
   - Patterns: Chart patterns (describe without fake confidence %)

### 3. **Fundamental & On-Chain** (when data available)
   - Tokenomics: Supply dynamics, inflation rate
   - Network health: Active addresses, transaction volume
   - DeFi metrics: TVL, protocol revenue (if applicable)

### 4. **Scenario Analysis** (REQUIRED)
   - **Bull Case (X% probability)**: Target, triggers, timeframe
   - **Bear Case (Y% probability)**: Target, triggers, invalidation
   - Be explicit about what changes the scenario

### 5. **Action Plan** (MUST BE LOGICALLY CONSISTENT)
   - **LONG Setup**: Entry < Target, Stop < Entry
   - **SHORT Setup**: Entry > Target, Stop > Entry
   - Specify position type (LONG/SHORT)
   - Risk/reward ratio

**Communication Style:**
- Write like you're explaining to a smart friend, not writing a textbook
- Use tables for comparing metrics when helpful
- Be thorough but clear - comprehensive is good, confusing is not

**Language:**
- Match the user's language naturally (Vietnamese → Vietnamese, English → English)
- Never switch languages mid-conversation unless user does first

**CRITICAL - Tool Transparency:**
- NEVER mention internal tool names in responses (e.g., DON'T say "getCryptoPrice shows...")
- Present data naturally: "BTC is at $67,432" NOT "The getCryptoPrice tool returned..."
- Reference sources generically: "On-chain data shows..." or "Market data indicates..."

**Risk Disclosure:** Emphasize crypto volatility (3-5x stocks) and DYOR.

**Remember:** Deliver institutional-quality analysis. No logical contradictions in recommendations."""

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
