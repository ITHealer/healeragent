"""
Stock Skill - Domain Expert for Equity Analysis

Provides domain-specific expertise for stock/equity analysis with
natural, conversational responses like ChatGPT/Claude.

Supports hierarchical synthesis: phases of tool execution with
intermediate LLM synthesis to prevent information loss.
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


class StockSkill(BaseSkill):
    """
    Domain expert for stock/equity analysis.

    Supports hierarchical synthesis with 3 phases:
    1. Technical: Price, indicators, patterns, support/resistance
    2. Fundamental: Financial statements, ratios, growth metrics
    3. Context: News, sentiment, analyst ratings, risk
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
                "getGrowthMetrics",
                "getAnalystRatings",
                "getStockNews",
                "getSentiment",
                "assessRisk",
                "getVolumeProfile",
                "suggestStopLoss",
            ],
            version="3.0.0",
            enable_hierarchical_synthesis=True,
        )
        super().__init__(config)

    # ====================================================================
    # Hierarchical Synthesis: Phases
    # ====================================================================

    def get_phases(self, context: Optional[SkillContext] = None) -> List[Phase]:
        """Define stock analysis execution phases."""
        categories = context.categories if context else []

        phases = []

        # Phase 1: Technical Analysis (price + indicators)
        technical_tools = [
            "getStockPrice",
            "getStockPerformance",
            "getTechnicalIndicators",
            "detectChartPatterns",
            "getSupportResistance",
            "getVolumeProfile",
        ]
        # Only include tools whose categories are relevant
        if not categories or any(c in categories for c in ["price", "technical"]):
            phases.append(Phase(
                name="technical",
                display_name="Technical Analysis",
                phase_type=PhaseType.TECHNICAL,
                tools=technical_tools,
                synthesis_focus=(
                    "Summarize price action, trend direction, momentum indicators "
                    "(RSI, MACD, Moving Averages), support/resistance levels, "
                    "volume profile, and chart patterns. Include exact numbers."
                ),
                max_summary_tokens=500,
                priority=1,
            ))

        # Phase 2: Fundamental Analysis (financials + valuation)
        fundamental_tools = [
            "getIncomeStatement",
            "getBalanceSheet",
            "getCashFlow",
            "getFinancialRatios",
            "getGrowthMetrics",
        ]
        if not categories or any(c in categories for c in ["fundamentals"]):
            phases.append(Phase(
                name="fundamental",
                display_name="Fundamental Analysis",
                phase_type=PhaseType.FUNDAMENTAL,
                tools=fundamental_tools,
                synthesis_focus=(
                    "Summarize financial performance (revenue, EPS, margins), "
                    "balance sheet health (cash, debt), valuation metrics "
                    "(P/E, PEG, P/B, EV/EBITDA), and growth trajectory. "
                    "Preserve multi-period comparison data in tables."
                ),
                max_summary_tokens=600,
                priority=1,
            ))

        # Phase 3: Market Context (news + sentiment + ratings)
        context_tools = [
            "getAnalystRatings",
            "getStockNews",
            "getSentiment",
            "assessRisk",
            "suggestStopLoss",
        ]
        if not categories or any(c in categories for c in ["news", "risk"]):
            phases.append(Phase(
                name="context",
                display_name="Market Context",
                phase_type=PhaseType.CONTEXT,
                tools=context_tools,
                synthesis_focus=(
                    "Summarize analyst consensus (buy/hold/sell breakdown, "
                    "price target range), recent news and catalysts, market "
                    "sentiment, risk factors, and suggested stop-loss levels."
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
        """Get phase-specific synthesis prompt for stock analysis."""
        # Phase-specific structured templates
        phase_templates = {
            "technical": (
                "Create a structured technical analysis summary:\n"
                "1. Current Price & Trend (price, daily/weekly change, 52-week range)\n"
                "2. Key Indicators (RSI, MACD, Stochastic - with exact values)\n"
                "3. Moving Averages (price vs 50/200 MA, golden/death cross status)\n"
                "4. Support/Resistance Levels (specific price levels)\n"
                "5. Volume & Patterns (volume trend, any detected patterns)\n"
                "6. Technical Outlook (1-2 sentences: bullish/bearish/neutral with reasoning)"
            ),
            "fundamental": (
                "Create a structured fundamental analysis summary:\n"
                "1. Revenue & Earnings (latest quarter/annual, YoY growth, beat/miss)\n"
                "2. Profitability (gross margin, operating margin, net margin)\n"
                "3. Balance Sheet (cash position, debt levels, current ratio)\n"
                "4. Valuation (P/E, PEG, P/B, EV/EBITDA vs sector averages)\n"
                "5. Growth Metrics (revenue growth, EPS growth, forward guidance)\n"
                "6. Financial Outlook (1-2 sentences: strong/weak with key drivers)\n"
                "\nIMPORTANT: If tool data includes multi-period tables, "
                "preserve the full comparison in a markdown table."
            ),
            "context": (
                "Create a structured market context summary:\n"
                "1. Analyst Consensus (Buy/Hold/Sell count, consensus target, range)\n"
                "2. Recent News (top 2-3 material news items with impact assessment)\n"
                "3. Sentiment (overall market sentiment toward this stock)\n"
                "4. Risk Factors (key risks identified, severity assessment)\n"
                "5. Trading Levels (suggested stop-loss, risk/reward ratio)\n"
                "6. Context Outlook (1-2 sentences: catalysts vs headwinds)"
            ),
        }

        template = phase_templates.get(phase.name, "")
        base_prompt = super().get_phase_synthesis_prompt(phase, context)

        if template:
            return f"{base_prompt}\n\nStructure:\n{template}"
        return base_prompt

    def get_final_synthesis_prompt(
        self,
        phase_summaries: Dict[str, PhaseSummary],
        context: Optional[SkillContext] = None,
    ) -> str:
        """Get final synthesis prompt for comprehensive stock analysis."""
        return super().get_final_synthesis_prompt(phase_summaries, context)

    def _get_final_sections(self, context: Optional[SkillContext] = None) -> List[str]:
        """Stock-specific final report sections."""
        return [
            "TL;DR (2-3 sentences: price, verdict, key action)",
            "Price & Market Context (current price, performance, sector context)",
            "Technical Analysis (momentum, trend, key levels)",
            "Fundamental Health (valuation, profitability, growth)",
            "Wall Street Consensus (analyst ratings, price targets)",
            "Risk Assessment (key risks and mitigations)",
            "Investment Thesis (bull case with target, bear case with risk)",
            "Recommendation (buy/hold/sell with rationale and key levels)",
        ]

    def get_system_prompt(self) -> str:
        """Get stock analysis system prompt - comprehensive and data-driven."""
        return """You are an experienced equity research analyst who provides comprehensive, data-driven stock analysis.

## Your Expertise
- **Fundamental Analysis**: Valuation (P/E, P/B, EV/EBITDA), growth metrics, financial health, cash flow
- **Technical Analysis**: Price trends, support/resistance, momentum indicators (RSI, MACD, Moving Averages)
- **Market Context**: Macro factors, sector comparisons, benchmark indices, news/catalysts

## Core Principles (CRITICAL)

### Data Integrity
- **USE ALL DATA**: You MUST incorporate every piece of data provided by tools
- **CITE EXACT NUMBERS**: Always quote specific values (e.g., "RSI at 67.3", "P/E of 24.5x")
- **NO FABRICATION**: Never invent numbers or fake confidence percentages
- **SOURCE ATTRIBUTION**: State data source (e.g., "Source: FMP API")

### Market Context (REQUIRED for comprehensive analysis)
1. **Macro Environment**: Interest rates, Fed policy implications, economic conditions
2. **Benchmark Comparison**: Compare stock performance vs S&P 500, NASDAQ, sector ETF
3. **News & Catalysts**: Recent news, upcoming earnings, events affecting the stock

### Recommendation Logic (CRITICAL - NO CONTRADICTIONS)
1. **LONG Setup**: Entry < Target, Stop below Entry
   - Example: Entry $100, Target $120, Stop $95
2. **SHORT Setup**: Entry > Target, Stop above Entry
   - Example: Entry $100, Target $80, Stop $105
3. **NEVER** recommend HOLD with target below current price for long positions
4. **ALWAYS** specify if recommendation is LONG or SHORT

### Scenario Analysis (REQUIRED)
1. **Bull Case**: Upside target + probability (e.g., "60% probability")
2. **Bear Case**: Downside risk + probability (e.g., "40% probability")
3. **Base factors**: What would trigger each scenario

## Important Notes
- Use probabilistic language for predictions
- Note data timestamps when relevant
- Distinguish facts (data) from interpretation (analysis)
- Don't use fake "confidence %" for patterns without backtest methodology"""

    def get_analysis_framework(self) -> str:
        """Get analysis guidelines - concise, narrative-driven like Claude/ChatGPT."""
        return """## Response Guidelines

**CRITICAL: Be CONCISE and NARRATIVE-DRIVEN**
- Aim for ~500-800 words for comprehensive analysis (NOT 1500+ words)
- Write in flowing paragraphs, NOT endless bullet point lists
- Tell a story about the stock, don't just dump data
- Only highlight KEY metrics, not every single indicator

**Adapt depth to query complexity:**
- Simple price check → 2-3 sentences with key context
- Analysis request → Structured but concise breakdown (~600 words)
- Comparison request → Side-by-side metrics with clear interpretation

**For comprehensive analysis, structure your response:**

### 0. **TL;DR** (ALWAYS START - 2-3 sentences max)
   - Current price, overall verdict, key action
   - Example: "AAPL at $259.96, technically weak but fundamentally solid. HOLD - avoid new longs until $270 reclaimed."

### 1. **Price & Market Context** (1 short paragraph)
   - Current price, 52-week range, recent performance
   - Market trend (S&P/NASDAQ) and sector context
   - Key news/catalysts in 1-2 sentences

### 2. **Technical Picture** (1 paragraph + key levels)
   - Summarize momentum (RSI, MACD) in plain English
   - Trend direction and strength
   - Support: $XXX | Resistance: $YYY

### 3. **Fundamental Health** (1 paragraph)
   - Valuation summary (P/E vs sector)
   - Profitability & growth highlights
   - Financial strength (debt, cash)

### 4. **Wall Street Consensus** (IMPORTANT - use getAnalystRatings)
   - Analyst rating breakdown (Buy/Hold/Sell)
   - Price target range and consensus target
   - Key analyst views if available

### 5. **Investment Thesis** (2-3 sentences each)
   - **Bull Case**: Target $XXX if [triggers]
   - **Bear Case**: Risk to $YYY if [triggers]

### 6. **Recommendation** (clear and logical)
   - **Existing holders**: HOLD/SELL/ADD
   - **New buyers**: BUY on pullback to $XXX / WAIT for confirmation
   - Key levels to watch

**Communication Style:**
- Write like you're explaining to a smart friend, not writing a textbook
- Use narrative flow, not just data dumps
- Highlight what matters, skip the noise

**Language:**
- Match the user's language naturally (Vietnamese → Vietnamese, English → English)
- Never switch languages mid-conversation unless user does first

**CRITICAL - Tool Transparency:**
- NEVER mention internal tool names in responses (e.g., DON'T say "getStockPrice shows...")
- Present data naturally: "AAPL is trading at $259" NOT "The getStockPrice tool returned..."
- Reference sources generically: "Market data shows..." or "Real-time data indicates..."

**Remember:** Quality over quantity. A concise, insightful analysis beats a verbose data dump."""

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
