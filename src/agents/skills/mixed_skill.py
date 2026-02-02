"""
Mixed Skill - Domain Expert for Cross-Asset Analysis

Provides specialized expertise for queries spanning multiple asset classes,
including stock vs crypto comparisons and portfolio analysis.

Supports hierarchical synthesis with cross-asset phases.
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


class MixedSkill(BaseSkill):
    """
    Domain expert for cross-asset and portfolio analysis.

    Supports hierarchical synthesis with 3 phases:
    1. Stock Data: Price, technicals, financials for equity assets
    2. Crypto Data: Price, technicals, on-chain for crypto assets
    3. Context & Comparison: News, sentiment, risk, cross-asset context
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
            version="3.0.0",
            enable_hierarchical_synthesis=True,
        )
        super().__init__(config)

    # ====================================================================
    # Hierarchical Synthesis: Phases
    # ====================================================================

    def get_phases(self, context: Optional[SkillContext] = None) -> List[Phase]:
        """Define cross-asset analysis execution phases."""
        phases = []

        # Phase 1: Stock Data Collection
        phases.append(Phase(
            name="stock_data",
            display_name="Stock/Equity Analysis",
            phase_type=PhaseType.TECHNICAL,
            tools=[
                "getStockPrice",
                "getStockPerformance",
                "getTechnicalIndicators",
                "getFinancialRatios",
            ],
            synthesis_focus=(
                "Summarize equity asset data: price, performance, "
                "technical indicators, and key financial ratios. "
                "Include exact numbers for comparison."
            ),
            max_summary_tokens=500,
            priority=1,
        ))

        # Phase 2: Crypto Data Collection
        phases.append(Phase(
            name="crypto_data",
            display_name="Crypto Asset Analysis",
            phase_type=PhaseType.MARKET_STRUCTURE,
            tools=[
                "getCryptoPrice",
                "getCryptoTechnicals",
                "getCryptoInfo",
            ],
            synthesis_focus=(
                "Summarize crypto asset data: price, performance, "
                "technical indicators, tokenomics, and market metrics. "
                "Include exact numbers for comparison."
            ),
            max_summary_tokens=500,
            priority=1,
        ))

        # Phase 3: Cross-Asset Context
        phases.append(Phase(
            name="comparison_context",
            display_name="Cross-Asset Context",
            phase_type=PhaseType.COMPARISON,
            tools=[
                "getStockNews",
                "getSentiment",
                "assessRisk",
                "webSearch",
            ],
            synthesis_focus=(
                "Summarize cross-asset context: news for both assets, "
                "comparative sentiment, risk assessment, and any macro "
                "factors affecting both asset classes."
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
        """Get phase-specific synthesis prompt for mixed analysis."""
        phase_templates = {
            "stock_data": (
                "Create a structured stock/equity summary:\n"
                "1. Price & Performance (current price, returns by timeframe)\n"
                "2. Technical Signals (RSI, MACD, trend direction)\n"
                "3. Valuation (P/E, P/B, key ratios vs sector)\n"
                "4. Stock Assessment (1-2 sentences)"
            ),
            "crypto_data": (
                "Create a structured crypto asset summary:\n"
                "1. Price & Performance (price, 24h/7d/30d returns, market cap)\n"
                "2. Technical Signals (RSI, MACD, trend direction)\n"
                "3. Tokenomics (supply, inflation, utility)\n"
                "4. Crypto Assessment (1-2 sentences)"
            ),
            "comparison_context": (
                "Create a structured cross-asset context summary:\n"
                "1. News Impact (key news for each asset class)\n"
                "2. Sentiment Comparison (sentiment for each asset)\n"
                "3. Risk Profile (volatility comparison, risk factors)\n"
                "4. Macro Context (how macro affects each asset differently)\n"
                "5. Correlation & Diversification (1-2 sentences)"
            ),
        }

        template = phase_templates.get(phase.name, "")
        base_prompt = super().get_phase_synthesis_prompt(phase, context)

        if template:
            return f"{base_prompt}\n\nStructure:\n{template}"
        return base_prompt

    def _get_final_sections(self, context: Optional[SkillContext] = None) -> List[str]:
        """Cross-asset final report sections."""
        return [
            "TL;DR (2-3 sentences: verdict on each asset, recommended allocation)",
            "Market Context (macro environment, correlation, risk sentiment)",
            "Performance Comparison (side-by-side returns with table)",
            "Risk Profile Comparison (volatility, drawdown, risk-adjusted metrics)",
            "Scenario Analysis (bull/bear case for each asset with probabilities)",
            "Action Plan (allocation by risk profile, entry/target/stop per asset)",
        ]

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
- **NO FABRICATION**: Never invent numbers or fake confidence percentages
- **NORMALIZE FOR COMPARISON**: Use volatility-adjusted metrics when comparing different asset types

### Market Context (REQUIRED)
1. **Macro Environment**: How do interest rates/Fed affect each asset class differently?
2. **Correlation Analysis**: Are stocks and crypto moving together or diverging?
3. **Risk-On/Risk-Off**: Current market sentiment and its implications

### Recommendation Logic (CRITICAL - NO CONTRADICTIONS)
1. **LONG Setup**: Entry < Target, Stop below Entry
2. **SHORT Setup**: Entry > Target, Stop above Entry
3. **ALWAYS** specify if recommendation is LONG or SHORT for each asset

### Scenario Analysis (REQUIRED)
1. **Bull Case**: Upside for each asset + probability
2. **Bear Case**: Downside for each asset + probability
3. **Correlation Scenario**: What if correlation breaks down?

## Important Notes
- Always normalize for fair comparison (volatility, time periods)
- Crypto volatility is typically 3-5x stock volatility - contextualize
- Present balanced view without bias toward either asset class"""

    def get_analysis_framework(self) -> str:
        """Get analysis guidelines - comprehensive but natural."""
        return """## Response Guidelines

**Adapt depth to query complexity:**
- Simple comparison → Key differentiators with brief context
- Analysis request → Comprehensive metrics with interpretation
- Portfolio advice → Risk-adjusted recommendations by investor profile

**For comprehensive cross-asset analysis, structure your response:**

### 0. **TL;DR / Executive Summary** (ALWAYS START WITH THIS)
   - 2-3 sentences: Quick verdict on each asset, recommended allocation
   - Example: "TSLA vs BTC: Both consolidating. For balanced portfolio: 60% TSLA, 40% BTC. TSLA offers lower vol, BTC higher upside potential."

### 1. **Market Context** (REQUIRED)
   - Macro: How do rates/Fed affect stocks vs crypto differently?
   - Correlation: Are they moving together or diverging?
   - Risk Sentiment: Risk-on or risk-off environment?

### 2. **Performance Comparison**
   - Side-by-side returns (24h, 7d, 30d, YTD, 1Y)
   - Risk-adjusted performance (Sharpe if available)
   - Winner by timeframe with context

### 3. **Risk Profile Comparison**
   - Volatility comparison (crypto typically 3-5x stock vol)
   - Maximum drawdowns
   - Correlation to benchmarks

### 4. **Scenario Analysis** (REQUIRED)
   - **Bull Case (X%)**: Each asset's upside + triggers
   - **Bear Case (Y%)**: Each asset's downside + triggers

### 5. **Action Plan** (MUST BE LOGICALLY CONSISTENT)
   - **LONG Setup**: Entry < Target, Stop < Entry
   - **SHORT Setup**: Entry > Target, Stop > Entry
   - Allocation by investor profile (Conservative/Moderate/Aggressive)
   - Risk/reward for each asset

**Communication Style:**
- Write like you're explaining to a smart friend, not writing a textbook
- Use comparison tables (side-by-side metrics work great)
- Be thorough but clear - comprehensive is good, confusing is not

**Language:**
- Match the user's language naturally (Vietnamese → Vietnamese, English → English)
- Never switch languages mid-conversation unless user does first

**CRITICAL - Tool Transparency:**
- NEVER mention internal tool names in responses (e.g., DON'T say "getStockPrice shows..." or "getCryptoPrice returned...")
- Present data naturally: "TSLA is at $350, BTC at $67K" NOT "The tools returned..."
- Reference sources generically: "Market data shows..." or "Real-time data indicates..."

**Remember:** Deliver institutional-quality analysis. No logical contradictions in recommendations."""

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
