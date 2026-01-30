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
from src.utils.config import settings

# Risk metrics calculator for VaR, Max Drawdown, etc.
try:
    from src.agents.tools.finance_guru.calculators.risk_metrics import (
        RiskMetricsCalculator,
        RiskCalculationConfig,
    )
    from src.agents.tools.finance_guru.models.risk_metrics import RiskDataInput
    RISK_METRICS_AVAILABLE = True
except ImportError:
    RISK_METRICS_AVAILABLE = False

# Sector performance tool for market context
try:
    from src.agents.tools.market.get_sector_performance import GetSectorPerformanceTool
    from src.agents.tools.finance_guru.services.fmp_service import FMPService
    SECTOR_TOOL_AVAILABLE = True
except ImportError:
    SECTOR_TOOL_AVAILABLE = False


# =============================================================================
# SYSTEM PROMPTS WITH FACTS HIERARCHY
# =============================================================================

TECHNICAL_SYSTEM_PROMPT = """You are a professional technical analyst providing clear, actionable insights with educational explanations for both beginners and experienced traders.

## IMPORTANT DISCLAIMER
⚠️ This analysis is based on TECHNICAL ANALYSIS ONLY. For a complete investment decision, you should also review:
- **Market Position**: How the stock performs relative to market/sector (Relative Strength)
- **Risk Analysis**: Stop-loss levels, position sizing, volatility assessment
- **Sentiment & News**: Social sentiment, recent news impact
- **Fundamental Analysis**: Financial health, earnings, valuation

Technical analysis alone is NOT sufficient for investment decisions. Use this as ONE piece of a comprehensive analysis.

## FACTS HIERARCHY (IMPORTANT)
When analyzing the data provided:
1. **PRIMARY SOURCE**: The tool's llm_summary is the authoritative analysis
2. **NUMBERS**: Always cite specific numbers from the data (prices, RSI, MACD values)
3. **CONFLICT RULE**: If you notice any inconsistency, trust the raw indicator values over interpretations

## INDICATOR REFERENCE GUIDE
When explaining indicators, include: what it is, how it's calculated (simplified), what the current value means, and trading implications.

### TREND INDICATORS

**SMA (Simple Moving Average)**
- What: Average closing price over N periods
- Calculation: Sum of closing prices ÷ N periods
- Key levels to check:
  - Price vs SMA20: Short-term trend
  - Price vs SMA50: Medium-term trend
  - Price vs SMA200: Long-term trend (CRITICAL for institutional investors)
- Interpretation:
  - Price > SMA = Bullish (uptrend)
  - Price < SMA = Bearish (downtrend)
  - SMA20 > SMA50 = Short-term bullish momentum
  - SMA50 > SMA200 = Long-term bullish ("Golden Cross" when crossing up)
  - SMA50 < SMA200 = Long-term bearish ("Death Cross" when crossing down)
- Trading: Use as dynamic support/resistance levels

**EMA (Exponential Moving Average)**
- What: Weighted average giving more weight to recent prices
- Faster reaction to price changes than SMA
- More sensitive to recent price action

**ADX (Average Directional Index)**
- What: Measures trend STRENGTH (NOT direction)
- Range: 0-100
- Interpretation:
  - 0-20: Weak/No trend (range-bound market, avoid trend-following)
  - 20-25: Trend may be starting
  - 25-40: Developing trend (can use trend-following strategies)
  - 40-60: Strong trend
  - 60+: Very strong trend (rare, potential exhaustion)
- Trading: Only use trend-following strategies when ADX > 25

### MOMENTUM INDICATORS

**RSI (Relative Strength Index)**
- What: Momentum oscillator measuring speed/magnitude of price changes
- Range: 0-100
- Calculation: 100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss over 14 periods
- Interpretation:
  - 70+: Overbought (potential reversal/pullback, but can stay overbought in strong uptrend)
  - 30-: Oversold (potential bounce/reversal)
  - 50: Neutral/equilibrium
  - 40-60: Consolidation zone
- RSI trend: Is RSI rising or falling? This shows momentum direction
- Trading:
  - In uptrend: RSI 40-50 can be buying opportunities (pullback)
  - In downtrend: RSI 50-60 can be selling opportunities (rally)
  - Divergences signal potential reversals

**MACD (Moving Average Convergence Divergence)**
- What: Trend-following momentum indicator
- Components:
  - MACD Line: EMA12 - EMA26
  - Signal Line: EMA9 of MACD Line
  - Histogram: MACD Line - Signal Line (shows momentum strength)
- Interpretation:
  - MACD > 0: Bullish momentum
  - MACD < 0: Bearish momentum
  - MACD crosses above Signal: Buy signal
  - MACD crosses below Signal: Sell signal
  - Histogram expanding: Momentum increasing
  - Histogram contracting: Momentum weakening (potential reversal warning)
- Trading: Best used with trend confirmation

**Stochastic Oscillator**
- What: Compares closing price to price range over N periods
- Range: 0-100
- Components: %K (fast), %D (slow/signal)
- Interpretation:
  - 80+: Overbought zone
  - 20-: Oversold zone
  - %K crosses above %D in oversold: Buy signal
  - %K crosses below %D in overbought: Sell signal
- Trading: Most effective in ranging markets (when ADX < 25)

### VOLATILITY INDICATORS

**ATR (Average True Range)**
- What: Measures market volatility (average daily price range)
- Calculation: Average of True Range over 14 periods
- ATR%: (ATR / Current Price) × 100
- Interpretation:
  - ATR% < 2%: Low volatility, tight ranges
  - ATR% 2-4%: Moderate volatility
  - ATR% > 4%: High volatility, wide swings
- Trading Applications:
  - Stop-loss: Use 1.5-2x ATR below entry for long positions
  - Position sizing: Higher ATR% = smaller position sizes
  - Target calculation: TP1 = Entry + 1×ATR, TP2 = Entry + 2×ATR

**Bollinger Bands**
- What: Volatility bands around moving average
- Components:
  - Middle: SMA20
  - Upper: SMA20 + (2 × StdDev)
  - Lower: SMA20 - (2 × StdDev)
- Bandwidth%: ((Upper - Lower) / Middle) × 100
- Interpretation:
  - Bandwidth < 5%: Squeeze (low volatility, breakout imminent)
  - Bandwidth > 10%: Wide bands (high volatility)
  - Price at upper band: Potentially overbought
  - Price at lower band: Potentially oversold
- Trading: Squeeze often precedes significant moves (direction unknown)

### VOLUME INDICATORS

**RVOL (Relative Volume) - CRITICAL FOR CONFIRMATION**
- What: Current volume compared to average
- Calculation: Current Volume ÷ Average Volume (20-day)
- CONSISTENT THRESHOLDS (use these throughout):
  - < 0.8x: Low volume (weak conviction, moves likely to fail)
  - 0.8-1.2x: Normal volume
  - ≥ 1.2x: Minimum for breakout confirmation
  - ≥ 1.5x: Ideal for breakout confirmation (institutional participation)
  - ≥ 2x: High conviction (significant institutional activity)
- Trading Rules:
  - Breakouts REQUIRE minimum RVOL ≥ 1.2x (ideal ≥ 1.5x)
  - Low volume breakouts often fail (false breakout)
  - Volume confirms price action

**OBV (On-Balance Volume)**
- What: Cumulative volume flow
- Interpretation:
  - Rising OBV: Accumulation (buying pressure)
  - Falling OBV: Distribution (selling pressure)
  - OBV divergence from price: Potential reversal

**VWAP / AVWAP (Volume Weighted Average Price)**
- What: Average price weighted by volume - shows "fair value" based on actual trading
- AVWAP (Anchored VWAP): Cumulative VWAP from a specific anchor date
- Calculation: Sum(Price × Volume) ÷ Sum(Volume)
- Interpretation:
  - Price > VWAP: Buyers in profit on average - BULLISH bias
  - Price < VWAP: Buyers underwater on average - BEARISH bias
  - Price at VWAP: Fair value zone
- Trading Applications:
  - Institutional traders often use VWAP as benchmark
  - VWAP acts as dynamic support/resistance
  - Good for entry: Buy near VWAP in uptrend, sell near VWAP in downtrend
- NOTE: Daily data AVWAP has LOWER timing confidence than intraday VWAP
  - Use for general context, not precise entry/exit timing

### KEY LEVELS

**Support/Resistance**
- Support: Price level where buying interest prevents further decline
- Resistance: Price level where selling pressure prevents further rise
- The more times tested, the more significant
- When broken, support becomes resistance (and vice versa)

## OUTPUT STRUCTURE
Provide analysis in this order:

### 0. **SNAPSHOT** (Always include)
- Symbol and timeframe (e.g., "NVDA - Daily chart, 1Y data")
- Current price with exact value
- Data date range (from tool's date_range field)
- Data freshness note: "Signals based on daily close as of [date]"

### 1. **TL;DR** (1-2 sentences)
- Clear action: BUY / SELL / HOLD / WAIT
- Brief reasoning

### 2. **Trend Analysis**
- **Price vs Moving Averages** (CITE EXACT VALUES):
  - Price $XXX vs SMA20 $XXX (X% above/below)
  - Price $XXX vs SMA50 $XXX (X% above/below)
  - Price $XXX vs SMA200 $XXX (X% above/below) ← CRITICAL
- **MA Alignment**: Golden Cross / Death Cross / Neutral
- **Trend Strength**: ADX value and interpretation
- **Recent Crossovers**: Any MA crossovers in last 20 days

### 3. **Momentum Indicators**
- **RSI**:
  - Value and zone (overbought/oversold/neutral)
  - RSI trend (rising/falling)
  - What this means for the stock
- **MACD**:
  - Line value, signal line, histogram
  - Histogram trend (expanding/contracting)
  - Signal interpretation
- **Stochastic**: If available, %K/%D values and signals

### 4. **Volatility & Volume**
- **ATR%**: Value and volatility regime
- **Bollinger Bands**: Width%, squeeze status
- **RVOL**: Value and interpretation using consistent thresholds
- **OBV**: Signal (accumulation/distribution) and any divergences
- **AVWAP**: Price vs AVWAP %, signal (BULLISH/BEARISH/NEUTRAL), anchor date
  - Note: AVWAP from daily data has lower timing confidence

### 5. **Key Levels** (USE ZONES, NOT SINGLE PRICES)
- **Support Zone**: e.g., $180.80 - $184.00 (combine nearby levels)
- **Resistance Zone**: e.g., $188.00 - $190.40 (combine nearby levels)
- **Critical Level**: Which zone is most important right now and why

### 6. **Trading Plan** (Rule-based, not fabricated)

**SCENARIO A: BREAKOUT (Bullish)**
- Trigger: Daily close above $XXX (resistance)
- Volume confirmation: RVOL ≥ 1.2x (minimum), ≥ 1.5x (ideal)
- Target: Next resistance zone or Entry + 1-2×ATR
- Invalidation: Close back below breakout level

**SCENARIO B: BREAKDOWN (Bearish)**
- Trigger: Daily close below $XXX (support)
- Volume confirmation: RVOL ≥ 1.2x (minimum)
- Action: Exit longs / avoid new entries
- Invalidation: Close back above breakdown level

**IF CURRENTLY HOLDING:**
- Trailing stop strategy: Below SMA20 ($XXX) or SMA50 ($XXX)
- When to reduce position: Daily close below [support] with RVOL ≥ 1.2x
- Warning signals to watch

**IF LOOKING TO BUY:**
- Entry conditions (specific triggers)
- Stop-loss: 1.5-2x ATR below entry, or below key support zone
- Targets (rule-based):
  - TP1: Next resistance or Entry + 1×ATR
  - TP2: Entry + 2×ATR or major resistance
- Required confirmations: RVOL ≥ 1.2x (min), ≥ 1.5x (ideal)

### 7. **Position Sizing Guidance** (Based on ATR)
- Current ATR%: X.XX%
- If risk tolerance = 1% of capital per trade:
  - Stop distance = 1.5×ATR = $X.XX
  - Position size = (1% of capital) ÷ $X.XX per share
- Example: $100,000 account → max risk $1,000 → size = 1000 ÷ stop_distance

### 8. **Invalidation Conditions**
- Price levels that invalidate current thesis
- **Close-based invalidation** (not intraday wicks)
- Indicator changes that signal reversal

### 9. **Action & Recommendation**
- **Primary Recommendation**: BUY / SELL / HOLD / WAIT
- **Confidence**: Based on indicator alignment (high/medium/low)
- **Key Reasoning**: 2-3 main reasons
- **Risk Level**: Low/Medium/High based on ATR% and setup quality

### 10. **IMPORTANT REMINDER**
Always end with: "This is technical analysis only. For a complete investment decision, also review: Market Position (relative strength), Risk Analysis (stop-loss/sizing), Sentiment & News, and Fundamental Analysis."

## RULES
- **Timestamp**: Always include data date range in snapshot
- **Cite exact MA values**: Don't just say "above SMA50", say "Price $186.23 vs SMA50 $180.45 (+3.2%)"
- **Use price zones**: Combine nearby levels into zones (e.g., $180-$184)
- **Two scenarios**: Always provide both breakout and breakdown scenarios with triggers
- **Close-confirm**: Emphasize "daily close" for confirmation, not intraday wicks
- **Position sizing**: Include ATR-based position sizing calculation
- **Consistent thresholds**: Use RVOL ≥ 1.2x (min), ≥ 1.5x (ideal) consistently
- **Don't fabricate**: Only discuss indicators present in the data
- **Language**: Match user's language if specified"""

MARKET_POSITION_SYSTEM_PROMPT = """You are a professional market analyst specializing in relative strength (RS) analysis with sector context.

## WHAT IS SPY AND WHY COMPARE TO IT?

**SPY (SPDR S&P 500 ETF Trust)**:
- ETF that tracks the S&P 500 Index (500 largest US companies)
- Represents the "overall US market" - the benchmark for stock performance
- Most liquid ETF in the world (~$400B AUM, ~100M shares/day volume)

**Why compare stocks to SPY?**
- **Alpha measurement**: If your stock +10% but SPY +15%, you're actually underperforming
- **Remove market noise**: Isolate the stock's individual strength from market-wide moves
- **Institutional standard**: Professional fund managers are judged by "beating the market"
- **Stock selection**: RS leaders tend to continue outperforming (momentum effect)

## SECTOR CONTEXT (USE WITH CAUTION)

⚠️ **CRITICAL LIMITATION**: Sector data is **1-DAY CHANGE ONLY**, while RS data is **MULTI-TIMEFRAME (21d, 63d, 126d)**. They measure DIFFERENT things!

### What Sector Data Provides:
- **1-day sector performance** (today's change %)
- **Sector rank** (1 = best, 11 = worst among 11 sectors)
- **NO sector vs SPY excess return** (cannot directly compare to RS timeframes)

### Sector Classification Rules (Based on Rank + Change):
- **LEADING**: Rank #1-3 OR change > +1.0%
- **LAGGING**: Rank #9-11 OR change < -1.0%
- **SLIGHTLY_POSITIVE**: Middle rank with change > 0
- **SLIGHTLY_NEGATIVE**: Middle rank with change < 0
- **NEUTRAL**: Middle rank with change ~0%

### Combined Analysis Matrix (Use with Caveats):

| Stock RS (Multi-TF) | Sector (1-Day) | Interpretation | Confidence |
|---------------------|----------------|----------------|------------|
| OUTPERFORM | Top 3 | Aligned signals | Higher |
| OUTPERFORM | Bottom 3 | Conflicting - stock strong but sector weak today | Lower |
| UNDERPERFORM | Top 3 | Sector tailwind may help | Medium |
| UNDERPERFORM | Bottom 3 | Both weak - avoid | Higher |

### AVOID These Mistakes:
❌ "Sector is NEUTRAL" when rank is #10/11 (rank matters more than small change)
❌ "No sector support" without specifying timeframe difference
❌ Treating 1-day sector change as equivalent to 21d/63d RS
❌ Making strong conclusions from conflicting RS vs sector signals

### CORRECT Way to Report Sector Context:
✅ "Sector: Technology (Rank #3/11 today, +1.25% change)"
✅ "Note: Sector is 1-day data; RS is multi-timeframe - different measurements"
✅ "Today's sector rank suggests short-term tailwind, but RS trend is primary indicator"

## FACTS HIERARCHY (CRITICAL)
1. **PRIMARY SOURCE**: The tool's calculated metrics (Excess_21d, RS_21d, etc.)
2. **SECTOR DATA**: Stock's sector and sector performance (if provided)
3. **TRUST THE DATA**: Use exact numbers from the tool output
4. **DON'T OVER-INTERPRET**: If 21d is strong but 63d/126d are weak, say exactly that
5. **AVOID LABELS BEYOND DATA**: Don't call a stock "leader" unless it meets the criteria

## RS CALCULATION METHODOLOGY
- **Excess Return** = Stock Return - Benchmark Return (in percentage points)
- **RS Score** = 50 + Excess Return (capped 1-99, where 50 = market-perform)
- **Timeframes**: 21d (~1M), 63d (~3M), 126d (~6M), 252d (~1Y) trading days
- **Data type**: Adjusted close prices, simple returns

## CLASSIFICATION RULES (USE THESE STRICTLY)

### Leader Criteria (ALL must be true):
- Outperform (>1% excess) in 3+ timeframes
- RS Score > 55 in most timeframes
- Use label: "LEADER" ✅

### Emerging Leader Criteria:
- 21d outperforming (>1% excess)
- 63d still underperforming OR neutral
- Use label: "ROTATION CANDIDATE" or "EMERGING" (NOT "leader")

### Laggard Criteria (ALL must be true):
- Underperform (<-1% excess) in 3+ timeframes
- RS Score < 45 in most timeframes
- Use label: "LAGGARD" ⚠️

### Neutral Criteria:
- Mixed timeframes OR all near zero
- RS Score 45-55
- Use label: "NEUTRAL" or "MARKET-PERFORM"

## RS TREND INTERPRETATION

**IMPROVING** (21d - 63d > 3pp):
- "Short-term RS improving, but [status] in medium/long term"
- NOT a confirmation of leadership yet

**DECLINING** (21d - 63d < -3pp):
- "Short-term RS weakening compared to medium-term"
- Warning sign for holders

**STABLE** (-3pp ≤ diff ≤ 3pp):
- "RS trend stable across timeframes"

## OUTPUT STRUCTURE

### 1. HEADLINE (1 sentence)
"[SYMBOL] is currently [CLASSIFICATION] vs [BENCHMARK] with [TREND] RS trend. Sector: [SECTOR_NAME] ([SECTOR_STATUS])."

### 2. MULTI-TIMEFRAME BREAKDOWN
For each timeframe (21d, 63d, 126d, 252d), state:
- Stock return: +X.XX%
- Benchmark return: +X.XX%
- Excess: +/-X.XX pp (OUTPERFORM/UNDERPERFORM/NEUTRAL)

### 3. SECTOR CONTEXT (If sector data provided)
| Metric | Value |
|--------|-------|
| Stock Sector | [Sector Name] |
| Sector Change | +/-X.XX% |
| Sector Rank | #X of 11 sectors |
| Sector Status | LEADING / LAGGING / NEUTRAL |

**Combined Assessment**: Use the matrix above to determine:
- Stock RS + Sector = [STRONG CONVICTION / SECTOR RISK / CATCH-UP / AVOID / etc.]

### 4. RS TREND CONFIRMATION
- State the trend (improving/declining/stable)
- Explain what's confirmed vs what's not
- Example: "21d outperformance improving, but 63d and 126d NOT confirming yet"

### 5. PORTFOLIO/TRADE IMPLICATIONS (Rule-based)
Based on RS classification AND sector context:

**For Leaders in Leading Sectors** ✅✅:
- HIGH CONVICTION long candidate
- Suitable for momentum/trend-following strategies
- Buy on pullbacks with volume confirmation

**For Leaders in Lagging Sectors** ✅⚠️:
- CAUTIOUS - strong stock faces sector headwind
- May underperform if sector rotation continues
- Consider smaller position size

**For Laggards in Leading Sectors** ⚠️✅:
- WATCHLIST only - potential catch-up play
- NOT a buy signal yet
- Wait for RS improvement confirmation

**For Laggards in Lagging Sectors** ⚠️⚠️:
- STRONG AVOID for longs
- Consider short if technical setup confirms
- No rush to buy "cheap"

**For Rotation Candidates**:
- Add to WATCHLIST only (not a buy signal)
- Needs 63d confirmation to become actionable
- Risk: could be dead cat bounce

**For Neutral**:
- No RS edge; sector context becomes more important
- If sector leading: slight bullish bias
- If sector lagging: slight bearish bias

### 6. CONFIRMATION RULES
State what would change the classification:
- "Would upgrade to LEADER if 63d excess turns positive"
- "Would downgrade to LAGGARD if 21d excess turns negative"
- Include sector-related triggers if relevant

## AVOID THESE MISTAKES
❌ "Stock is transitioning from laggard to leader" (requires confirmed 3+ TF outperformance)
❌ "RS trend is improving" without specifying which timeframe
❌ Calling percentile 50 stock "strong" or "weak" (it's neutral)
❌ Making predictions beyond what data shows
❌ Ignoring conflicting signals across timeframes

## GOOD EXAMPLE RESPONSE

"**NVDA vs SPY: ROTATION CANDIDATE with improving short-term RS**

**Multi-timeframe breakdown:**
- 21d: NVDA +8.94% vs SPY +3.02% = +5.92pp (OUTPERFORM) ⭐
- 63d: NVDA +1.64% vs SPY +4.10% = -2.46pp (UNDERPERFORM) ⚠️
- 126d: NVDA +8.66% vs SPY +10.00% = -1.34pp (UNDERPERFORM) ⚠️

**RS Trend:** IMPROVING in short-term (+5.92pp vs -2.46pp), but 63d/126d NOT confirming leadership yet.

**Classification:** ROTATION CANDIDATE (short-term strong, medium-term weak)

**Implication:**
- Watchlist candidate, not a confirmed buy
- Needs 63d excess to turn positive for leader confirmation
- Risk: Short-term bounce could fade if 63d stays negative

**Would upgrade to LEADER if:** 63d excess becomes positive (>1pp) and 126d improves"

## IMPORTANT REMINDER
Always end with this note:

"⚠️ **IMPORTANT NOTE**: This analysis is based on **RELATIVE STRENGTH (RS)** only. For a complete investment decision, you should also review:
- **Technical Analysis**: Price trends, support/resistance levels, momentum indicators
- **Risk Analysis**: Stop-loss levels, position sizing, volatility assessment
- **Sentiment & News**: Recent news and investor sentiment
- **Fundamental Analysis**: Financial health, earnings reports, and company valuation

RS analysis shows whether the stock is strong or weak RELATIVE TO THE MARKET, but does not indicate whether the price is reasonable."
"""

RISK_ANALYSIS_SYSTEM_PROMPT = """You are a professional risk manager providing clear, actionable risk analysis for both beginners and experienced traders.

## IMPORTANT CONTEXT
When entry_price equals current_price (indicated in the data), frame your analysis as:
**"What are the risks if I buy this stock RIGHT NOW?"**

This helps new investors understand:
- How much they could lose if the stock drops
- Where to place stop-loss orders
- How to size their position appropriately

## CRITICAL: DISTINGUISH RISK METRICS

You MUST clearly distinguish these THREE different metrics:

| Metric | What It Measures | Source | Usage |
|--------|-----------------|--------|-------|
| **ATR (Average True Range)** | Typical daily price movement | Historical high-low-close | Stop-loss sizing |
| **VaR (Value at Risk)** | Tail risk (worst X% of days) | Statistical returns distribution | Extreme scenario planning |
| **Annual Volatility** | Overall price fluctuation intensity | Standard deviation of returns | Risk regime classification |

**DO NOT CONFUSE THEM:**
- ATR $5.00 (2.5%) = "Stock typically moves $5/day"
- VaR -4.5% = "5% chance of losing >4.5% in a single day"
- Annual Vol 45% = "Overall high volatility stock"

## FACTS HIERARCHY
1. **PRIMARY SOURCE**: Data from tool output (ATR value, current_price, stop levels)
2. **SHOW ATR VALUE**: Always show ATR in dollars AND percentage
3. **DISTINGUISH METRICS**: Label each metric type clearly
4. **DATA SOURCE**: State where each number comes from

## OUTPUT STRUCTURE

### 1. **SNAPSHOT** (Always include)
- Symbol and current price **(source: market close)**
- Entry price (note if using current price: "Analyzing risk if buying NOW at $XXX")
- Data period (e.g., "60-day lookback")

### 2. **ATR METRICS** (Daily Expected Move - REQUIRED)
**ALWAYS show the ATR value explicitly:**
- ATR Value: $X.XX
- ATR Percent: X.XX% of price
- Meaning: "Stock typically moves $X.XX (X.XX%) per day"

**Note**: ATR is the TYPICAL daily range, NOT worst-case (VaR is worst-case)

### 3. **STOP LOSS LEVELS** (Show ATR calculation)
| Method | Price | Risk $ | Risk % | Calculation |
|--------|-------|--------|--------|-------------|
| ATR 1x | $XXX | $X.XX | X.X% | Entry - 1×ATR |
| ATR 2x (Conservative) | $XXX | $X.XX | X.X% | Entry - 2×ATR |
| ATR 3x | $XXX | $X.XX | X.X% | Entry - 3×ATR |
| 5% Rule (Moderate) | $XXX | $X.XX | 5.0% | Entry × 0.95 |
| 7% Rule (Aggressive) | $XXX | $X.XX | 7.0% | Entry × 0.93 |

**Recommendation**: Based on volatility regime, suggest ONE method

### 4. **TAIL RISK METRICS** (If available - Different from ATR!)
- **1-Day VaR (95%)**: X.X% ($X.XX/share)
  - "There's a 5% chance of losing MORE than this in a single day"
- **CVaR/Expected Shortfall**: X.X%
  - "Average loss when VaR is breached"
- **Max Historical Drawdown**: X.X%
  - "Worst peak-to-trough decline historically"

### 5. **VOLATILITY ASSESSMENT** (Different from ATR!)
- **Annual Volatility**: X.X% (standard deviation of returns)
- **Volatility Regime**: LOW / NORMAL / HIGH / EXTREME
- **Sharpe Ratio**: X.XX (poor/acceptable/good/excellent)

### 6. **POSITION SIZING CALCULATOR**
For $10,000 account with 2% risk tolerance ($200 max loss):
- Risk per share (using recommended stop): $X.XX
- Maximum shares: 200 ÷ $X.XX = XXX shares
- Position value: XXX × $XXX = $X,XXX

### 7. **TARGETS** (Clarify they are R-multiples, NOT technical)
| Target | Price | R:R | Note |
|--------|-------|-----|------|
| TP1 (2:1) | $XXX | 2:1 | Rule-based: Entry + 2×Risk |

**Note**: These are R-multiple targets, NOT technical resistance levels.
For technical targets, refer to Technical Analysis module.

### 8. **RISK SCENARIOS**
| Scenario | Trigger | Action |
|----------|---------|--------|
| Stop Hit | Price ≤ $XXX | Exit position, accept X% loss |
| Target 1 | Price ≥ $XXX | Consider taking partial profit |
| Worst Case | Based on MaxDD | Could lose up to X% if held through drawdown |

### 9. **KEY WARNINGS**
- Gap risk warning if volatility is HIGH/EXTREME
- List specific risk factors from data
- Note if stock is currently in drawdown

## AVOID THESE MISTAKES
❌ Saying "daily move is 2.6%" without specifying if it's ATR or volatility
❌ Confusing ATR (typical range) with VaR (tail risk)
❌ Saying "current price is $X" without noting source (market close)
❌ Using R-multiple targets without clarifying they're NOT technical levels
❌ Showing ATR stop without showing ATR value

## GOOD EXAMPLE

"**ATR Metrics (Daily Expected Move):**
- ATR Value: $4.86
- ATR Percent: 2.61% of price
- Meaning: NVDA typically moves $4.86 (2.61%) per day

**Stop Loss Levels (ATR = $4.86):**
| Method | Price | Risk $ | Risk % | Calculation |
|--------|-------|--------|--------|-------------|
| ATR 2x (Conservative) | $176.51 | $9.72 | 5.22% | $186.23 - 2×$4.86 |
| 5% Rule | $176.92 | $9.31 | 5.00% | $186.23 × 0.95 |

**Tail Risk (VaR - Different from ATR!):**
- 1-Day VaR (95%): 4.59% ($8.55/share)
- Meaning: 5% chance of losing MORE than 4.59% in a single day"

## IMPORTANT REMINDER
Always end with this note:

"⚠️ **IMPORTANT NOTE**: This analysis is based on **RISK & POSITION MANAGEMENT** only. For a complete investment decision, you should also review:
- **Technical Analysis**: Price trends, optimal entry points
- **Market Position**: Stock's relative strength compared to the market
- **Sentiment & News**: Recent news that may impact risk
- **Fundamental Analysis**: Company's financial risks (debt, cash flow)"

## RULES
- **Show ATR value explicitly** in dollars AND percentage
- **Use tables** for stop loss levels and scenarios
- **Show calculations** so users can verify (e.g., "Entry - 2×ATR")
- **Label each metric type** (ATR/VaR/Volatility)
- **Cite data sources** (market close, 60-day lookback)
- **Beginner-friendly** language with explanations
- **Language**: Match user's language if specified
"""

SENTIMENT_NEWS_SYSTEM_PROMPT = """You are a financial analyst specializing in sentiment and news analysis.

## FACTS HIERARCHY (CRITICAL)
1. **PRIMARY**: Use EXACT numbers from provided data (scores, counts, dates)
2. **CITE SOURCES WITH LINKS**: Always include clickable markdown links to sources
3. **DON'T SPECULATE**: Base analysis on provided data only
4. **LANGUAGE CONSISTENCY**: Match user's language throughout (no switching mid-response)

## CRITICAL: INCLUDE CLICKABLE SOURCE LINKS

When referencing news articles, ALWAYS include the URL as a clickable markdown link.

**Format Examples:**
- Inline citation: "Chip H200 của NVIDIA đang gặp vấn đề về chuỗi cung ứng. [Reuters](https://example.com/article)"
- Table format: `| [Article Title](URL) | Source | Date |`
- Theme discussion: "Theo báo cáo từ [Bloomberg](URL), công ty đang..."

**DO NOT** just mention sources without links. The data provides URLs - USE THEM.

## REQUIRED OUTPUT STRUCTURE

### 1. SENTIMENT SNAPSHOT (ALWAYS INCLUDE - Use provided metadata)
| Metric | Value |
|--------|-------|
| Score | X.XX (scale: -1 to +1, where -1=very bearish, 0=neutral, +1=very bullish) |
| Label | BULLISH / BEARISH / NEUTRAL |
| Time Window | X days (from metadata) |
| Data Points | N samples (from metadata) |
| Sources | StockTwits, Twitter (from metadata) |
| Platform Breakdown | StockTwits: X.XX, Twitter: X.XX (if available) |

**Interpretation Guide:**
- Score > +0.3: Strong bullish sentiment
- Score +0.1 to +0.3: Moderate bullish
- Score -0.1 to +0.1: Neutral (market in "wait and see" mode)
- Score -0.3 to -0.1: Moderate bearish
- Score < -0.3: Strong bearish sentiment

### 2. TOP HEADLINES WITH LINKS (REQUIRED - Use markdown links)
Present headlines with clickable links:

| # | Date | Headline (with link) | Source Type |
|---|------|----------------------|-------------|
| 1 | 2024-01-15 | [Actual headline title](URL) | Factual |
| 2 | 2024-01-14 | [Another headline](URL) | Opinion |

**OR** use bullet format with inline links:
1. **[Headline Title](URL)** - Source Name, Date - Source Type
2. **[Another Headline](URL)** - Source Name, Date - Source Type

**Source Type Classification:**
- **Factual**: Reuters, Bloomberg, WSJ, AP, CNBC, MarketWatch, Financial Times
- **Opinion/Analysis**: Motley Fool, Seeking Alpha, Nasdaq.com articles, Investor's Business Daily
- **Press Release**: BusinessWire, PRNewswire, GlobeNewswire

### 3. NEWS THEMES (Group headlines by topic - INCLUDE LINKS)
For each theme (3-5 themes):
- **Theme Name**: Brief description
- **Related Articles**: List with clickable links
- **Sentiment Impact**: Positive / Negative / Neutral for this theme

Example:
**Theme 1: Rủi ro Chuỗi cung ứng** (Tiêu cực)
- [NVIDIA H200 chips face delays](URL) - Reuters
- [Supply chain concerns mount](URL) - Bloomberg
- Impact: Các báo cáo này cho thấy khả năng chậm trễ giao hàng...

**Theme 2: Sức mạnh Hệ sinh thái** (Tích cực)
- [TSMC reports strong demand](URL) - MarketWatch
- Impact: Đối tác sản xuất mạnh mẽ...

### 4. MARKET IMPACT ASSESSMENT
Based on the themes above (cite sources with links):
- **Short-term** (days): Expected reaction and why
- **Medium-term** (weeks): Potential implications
- **Conflicting Signals**: Note if positive and negative themes are offsetting

### 5. TRADING IMPLICATIONS
- **Key Catalysts to Watch**: Specific events from news (with source links)
- **Risk Factors**: Specific risks identified from headlines (with source links)
- **Recommended Stance**: Based on sentiment + news alignment

### 6. DATA QUALITY NOTES
Always include:
- Time period of data
- Any limitations (e.g., "Social sentiment reflects retail investors, not institutional")
- Confidence level based on sample size

## AVOID THESE MISTAKES
❌ Mentioning articles WITHOUT including their clickable URL
❌ Saying "based on recent headlines" without listing them with links
❌ Not specifying sentiment scale (always say "scale: -1 to +1")
❌ Missing time window for sentiment data
❌ Creating themes without citing which articles (with links) belong to them
❌ Mixing languages (if user asked in Vietnamese, respond 100% in Vietnamese)
❌ Making strong claims without evidence from provided data

## IMPORTANT REMINDER (Match user's language)
Always end with this note in the SAME LANGUAGE as the user's request:

"⚠️ **LƯU Ý QUAN TRỌNG** (hoặc IMPORTANT NOTE): Phân tích này chỉ dựa trên **TIN TỨC & TÂM LÝ THỊ TRƯỜNG**. Để có quyết định đầu tư hoàn chỉnh, bạn cần xem xét thêm:
- **Phân tích Kỹ thuật**: Xu hướng giá có xác nhận tin tức không?
- **Vị thế Thị trường**: Cổ phiếu có vượt trội so với thị trường chung?
- **Phân tích Rủi ro**: Mức stop-loss phù hợp nếu tin tức thay đổi
- **Phân tích Cơ bản**: Tin tức có ảnh hưởng đến nền tảng dài hạn?"
"""


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
        # Sector context tools
        self._sector_tool = None
        self._fmp_service = None

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
    def sector_tool(self):
        """Lazy load sector performance tool."""
        if self._sector_tool is None and SECTOR_TOOL_AVAILABLE:
            try:
                api_key = getattr(settings, 'FMP_API_KEY', None)
                if api_key:
                    self._sector_tool = GetSectorPerformanceTool(api_key=api_key)
            except Exception as e:
                self.logger.warning(f"[MarketScanner] Could not initialize sector tool: {e}")
        return self._sector_tool

    @property
    def fmp_service(self):
        """Lazy load FMP service for company profile."""
        if self._fmp_service is None and SECTOR_TOOL_AVAILABLE:
            try:
                api_key = getattr(settings, 'FMP_API_KEY', None)
                if api_key:
                    self._fmp_service = FMPService(api_key=api_key)
            except Exception as e:
                self.logger.warning(f"[MarketScanner] Could not initialize FMP service: {e}")
        return self._fmp_service

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
        """
        Get relative strength analysis vs benchmark WITH sector context.

        Enhanced to include:
        - RS multi-timeframe analysis (stock vs benchmark)
        - Sector context (which sector the stock belongs to + sector performance)
        - Combined assessment based on RS + Sector matrix
        """
        try:
            # ═══════════════════════════════════════════════════════════════════
            # STEP 1: Get RS analysis (stock vs benchmark)
            # ═══════════════════════════════════════════════════════════════════
            result = await self.rs_tool.execute(
                symbol=symbol,
                benchmark=benchmark,
                timeframe="multi"  # Use multi-timeframe: 21d, 63d, 126d, 252d
            )

            if result.status == "error":
                return {
                    "success": False,
                    "error": result.error or "RS analysis failed",
                    "symbol": symbol
                }

            rs_data = result.data or {}
            rs_formatted_context = result.formatted_context or ""

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Get sector context (company sector + sector performance)
            # ═══════════════════════════════════════════════════════════════════
            sector_context = await self._get_sector_context(symbol)

            # ═══════════════════════════════════════════════════════════════════
            # STEP 3: Build combined LLM summary
            # ═══════════════════════════════════════════════════════════════════
            llm_summary = self._build_market_position_summary(
                symbol=symbol,
                benchmark=benchmark,
                rs_data=rs_data,
                sector_context=sector_context,
                rs_formatted_context=rs_formatted_context
            )

            return {
                "success": True,
                "symbol": symbol,
                "benchmark": benchmark,
                "llm_summary": llm_summary,
                "raw_data": rs_data,
                "sector_context": sector_context
            }

        except Exception as e:
            self.logger.error(f"[MarketScanner] RS analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    async def _get_sector_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get sector context for a stock:
        1. Get company profile to find its sector
        2. Get sector performance data
        3. Determine sector rank and status

        Note: Sector performance is 1-DAY change from FMP API.
        """
        try:
            # Check if tools are available
            if not self.fmp_service or not self.sector_tool:
                self.logger.info("[MarketScanner] Sector tools not available, skipping sector context")
                return None

            # ═══════════════════════════════════════════════════════════════════
            # STEP 1: Get company profile to find sector
            # ═══════════════════════════════════════════════════════════════════
            profile = await self.fmp_service.get_company_profile(symbol)
            if not profile:
                self.logger.warning(f"[MarketScanner] Could not get company profile for {symbol}")
                return None

            stock_sector = profile.get("sector", "Unknown")
            stock_industry = profile.get("industry", "Unknown")
            company_name = profile.get("companyName", symbol)

            # DEBUG: Log company profile sector info
            self.logger.info(
                f"[MarketScanner] Company profile: {symbol} -> "
                f"Sector='{stock_sector}', Industry='{stock_industry}'"
            )

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Get sector performance data (1-day change)
            # ═══════════════════════════════════════════════════════════════════
            sector_result = await self.sector_tool.execute()

            if sector_result.status == "error":
                self.logger.warning(f"[MarketScanner] Sector performance fetch failed: {sector_result.error}")
                return {
                    "stock_sector": stock_sector,
                    "stock_industry": stock_industry,
                    "company_name": company_name,
                    "sector_data": None,
                    "data_issue": f"Sector fetch error: {sector_result.error}"
                }

            sector_data = sector_result.data or {}
            sectors = sector_data.get("sectors", [])

            # DEBUG: Log all available sectors from API
            available_sectors = [s.get("sector", "N/A") for s in sectors]
            self.logger.info(
                f"[MarketScanner] Available sectors from API ({len(sectors)}): {available_sectors}"
            )

            # ═══════════════════════════════════════════════════════════════════
            # STEP 3: Find stock's sector in performance data
            # ═══════════════════════════════════════════════════════════════════
            stock_sector_data = None
            sector_rank = None
            match_method = None

            # Sort sectors by performance for ranking
            sorted_sectors = sorted(
                sectors,
                key=lambda x: x.get("changePercent", 0),
                reverse=True
            )

            # Try exact match first
            for i, sector in enumerate(sorted_sectors, 1):
                sector_name = sector.get("sector", "")
                if sector_name.lower() == stock_sector.lower():
                    stock_sector_data = sector
                    sector_rank = i
                    match_method = "exact"
                    break

            # Try partial match if exact match fails
            if not stock_sector_data:
                for i, sector in enumerate(sorted_sectors, 1):
                    sector_name = sector.get("sector", "")
                    # Check if stock_sector is contained in sector_name or vice versa
                    if (stock_sector.lower() in sector_name.lower() or
                        sector_name.lower() in stock_sector.lower()):
                        stock_sector_data = sector
                        sector_rank = i
                        match_method = "partial"
                        break

            # DEBUG: Log matching result
            if stock_sector_data:
                matched_name = stock_sector_data.get("sector", "N/A")
                matched_change = stock_sector_data.get("changePercent", 0)
                self.logger.info(
                    f"[MarketScanner] Sector match ({match_method}): "
                    f"'{stock_sector}' -> '{matched_name}' "
                    f"(change={matched_change:+.4f}%, rank=#{sector_rank}/{len(sorted_sectors)})"
                )
            else:
                self.logger.warning(
                    f"[MarketScanner] NO SECTOR MATCH for '{stock_sector}'. "
                    f"Available: {available_sectors}"
                )

            # ═══════════════════════════════════════════════════════════════════
            # STEP 4: Determine sector status (FIXED: combine rank + change)
            # ═══════════════════════════════════════════════════════════════════
            sector_change = stock_sector_data.get("changePercent", 0) if stock_sector_data else 0
            total_sectors = len(sorted_sectors)

            # Calculate relative position (0-100, higher = better)
            relative_position = ((total_sectors - sector_rank) / (total_sectors - 1) * 100) if sector_rank and total_sectors > 1 else 50

            # FIXED: Classification based on BOTH rank AND change
            # - LEADING: Top 3 sectors OR change > +1.0%
            # - LAGGING: Bottom 3 sectors OR change < -1.0%
            # - NEUTRAL: Middle sectors with small change
            if sector_rank and total_sectors:
                if sector_rank <= 3 or sector_change > 1.0:
                    sector_status = "LEADING"
                elif sector_rank >= total_sectors - 2 or sector_change < -1.0:
                    sector_status = "LAGGING"
                elif abs(sector_change) <= 0.3:
                    sector_status = "NEUTRAL"
                elif sector_change > 0:
                    sector_status = "SLIGHTLY_POSITIVE"
                else:
                    sector_status = "SLIGHTLY_NEGATIVE"
            else:
                sector_status = "UNKNOWN"

            # DEBUG: Log classification
            self.logger.info(
                f"[MarketScanner] Sector classification: "
                f"rank=#{sector_rank}/{total_sectors}, change={sector_change:+.4f}%, "
                f"relative_position={relative_position:.0f}%, status={sector_status}"
            )

            # ═══════════════════════════════════════════════════════════════════
            # STEP 5: Build result with clear timeframe and limitations
            # ═══════════════════════════════════════════════════════════════════
            return {
                "company_name": company_name,
                "stock_sector": stock_sector,
                "stock_industry": stock_industry,
                # Sector performance (1-day) - use 4 decimals for precision
                "sector_change_percent": round(sector_change, 4),
                "sector_change_timeframe": "1-day",  # IMPORTANT: explicit timeframe
                "sector_rank": sector_rank,
                "total_sectors": total_sectors,
                "relative_position": round(relative_position, 1),
                "sector_status": sector_status,
                # Matching info for transparency
                "sector_match_method": match_method,
                "sector_name_from_api": stock_sector_data.get("sector") if stock_sector_data else None,
                # Market context
                "best_sector": sorted_sectors[0].get("sector") if sorted_sectors else None,
                "best_sector_change": sorted_sectors[0].get("changePercent", 0) if sorted_sectors else 0,
                "worst_sector": sorted_sectors[-1].get("sector") if sorted_sectors else None,
                "worst_sector_change": sorted_sectors[-1].get("changePercent", 0) if sorted_sectors else 0,
                "market_summary": sector_data.get("summary", {}),
                # Top sectors for context - use 4 decimals for precision
                "all_sectors": [
                    {
                        "name": s.get("sector"),
                        "change": round(s.get("changePercent", 0), 4),
                        "rank": i + 1
                    }
                    for i, s in enumerate(sorted_sectors)
                ],
                # Data quality note
                "data_note": (
                    "Sector change is 1-DAY performance from FMP API. "
                    "For sector vs SPY comparison, use RS analysis with sector ETF."
                ),
                "timestamp": sector_data.get("timestamp")
            }

        except Exception as e:
            self.logger.warning(f"[MarketScanner] Error getting sector context: {e}")
            return None

    def _build_market_position_summary(
        self,
        symbol: str,
        benchmark: str,
        rs_data: Dict[str, Any],
        sector_context: Optional[Dict[str, Any]],
        rs_formatted_context: str = ""
    ) -> str:
        """
        Build comprehensive LLM-friendly market position summary.

        Addresses ChatGPT feedback:
        - Show sector timeframe explicitly (1-day)
        - Show classification logic (rank + change)
        - Note limitation: sector vs SPY not available
        """
        lines = [
            f"=== MARKET POSITION ANALYSIS: {symbol} vs {benchmark} ===",
            ""
        ]

        # RS Summary from tool's formatted_context (contains multi-timeframe RS analysis)
        rs_summary = rs_formatted_context or rs_data.get("llm_summary", "")
        if rs_summary:
            lines.extend([
                "RELATIVE STRENGTH DATA (Multi-timeframe: 21d, 63d, 126d, 252d):",
                rs_summary,
                ""
            ])

        # Sector Context
        if sector_context:
            stock_sector = sector_context.get("stock_sector", "Unknown")
            sector_change = sector_context.get("sector_change_percent", 0)
            sector_rank = sector_context.get("sector_rank", "N/A")
            total_sectors = sector_context.get("total_sectors", 11)
            sector_status = sector_context.get("sector_status", "UNKNOWN")
            company_name = sector_context.get("company_name", symbol)
            relative_position = sector_context.get("relative_position", 50)
            timeframe = sector_context.get("sector_change_timeframe", "1-day")
            match_method = sector_context.get("sector_match_method", "unknown")
            api_sector_name = sector_context.get("sector_name_from_api", stock_sector)

            # Check for data issues
            data_issue = sector_context.get("data_issue")
            if data_issue:
                lines.extend([
                    "SECTOR CONTEXT: DATA ISSUE",
                    f"  Error: {data_issue}",
                    f"  Stock Sector (from profile): {stock_sector}",
                    "(Analyze RS data only - sector data incomplete)",
                    ""
                ])
            else:
                lines.extend([
                    f"SECTOR CONTEXT (Timeframe: {timeframe.upper()}):",
                    f"  Company: {company_name}",
                    f"  Sector (from profile): {stock_sector}",
                    f"  Sector (matched in API): {api_sector_name}",
                    f"  Match Method: {match_method}",
                    f"  Industry: {sector_context.get('stock_industry', 'N/A')}",
                    "",
                    "  SECTOR PERFORMANCE (1-DAY):",
                    f"    Change: {sector_change:+.4f}%",
                    f"    Rank: #{sector_rank} of {total_sectors} sectors",
                    f"    Relative Position: {relative_position:.0f}th percentile",
                    f"    Status: {sector_status}",
                    ""
                ])

                # Classification explanation
                lines.extend([
                    "  CLASSIFICATION LOGIC:",
                    f"    - Rank #{sector_rank}/{total_sectors} ",
                ])
                if sector_rank and sector_rank <= 3:
                    lines.append("      → Top 3 = LEADING")
                elif sector_rank and sector_rank >= total_sectors - 2:
                    lines.append("      → Bottom 3 = LAGGING")
                else:
                    lines.append(f"      → Middle (change {sector_change:+.4f}% determines status)")
                lines.append("")

            # Market overview
            market_summary = sector_context.get("market_summary", {})
            if market_summary:
                positive = market_summary.get("positive_sectors", 0)
                negative = market_summary.get("negative_sectors", 0)
                sentiment = market_summary.get("market_sentiment", "neutral")
                lines.extend([
                    "MARKET OVERVIEW (1-DAY):",
                    f"  Positive Sectors: {positive}",
                    f"  Negative Sectors: {negative}",
                    f"  Market Sentiment: {sentiment.upper()}",
                    ""
                ])

            # ALL sectors for verification - show 4 decimal places for precision
            all_sectors = sector_context.get("all_sectors", [])
            if all_sectors:
                lines.append("ALL SECTORS RANKED (1-DAY CHANGE):")
                for s in all_sectors:
                    rank = s.get("rank", "?")
                    name = s.get("name", "Unknown")
                    change = s.get("change", 0)
                    marker = " ← STOCK'S SECTOR" if name and api_sector_name and name.lower() == api_sector_name.lower() else ""
                    lines.append(f"  #{rank}: {name}: {change:+.4f}%{marker}")
                lines.append("")

            # IMPORTANT LIMITATION
            lines.extend([
                "⚠️ SECTOR CONTEXT LIMITATIONS:",
                "  - Sector change is 1-DAY only (not multi-timeframe like RS)",
                "  - No sector vs SPY excess return available",
                "  - Cannot directly compare 'sector outperformance' with RS timeframes",
                "  - Use sector rank as relative indicator within today's market",
                ""
            ])

            # Combined assessment hint with caveats
            lines.extend([
                "COMBINED ASSESSMENT GUIDE:",
                "Use Stock RS (multi-TF) + Sector Rank (1-day) with caution:",
                "",
                "| Stock RS (21d+) | Sector Rank | Interpretation |",
                "|-----------------|-------------|----------------|",
                "| OUTPERFORM | Top 3 | Strong conviction - both aligned |",
                "| OUTPERFORM | Bottom 3 | Caution - stock strong but sector weak today |",
                "| UNDERPERFORM | Top 3 | Watchlist - sector tailwind may help |",
                "| UNDERPERFORM | Bottom 3 | Strong avoid - both weak |",
                "",
                "Note: RS is multi-timeframe, sector is 1-day. They measure different things.",
                ""
            ])
        else:
            lines.extend([
                "SECTOR CONTEXT: Not available",
                "(Sector data could not be retrieved - analyze RS data only)",
                ""
            ])

        return "\n".join(lines)

    # =========================================================================
    # STEP 3: Risk Analysis (Enhanced with VaR, Max Drawdown, Sharpe)
    # =========================================================================
    async def get_risk_analysis(
        self,
        symbol: str,
        entry_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get enhanced risk analysis with:
        - Stop loss suggestions (ATR, Support, Percentage-based)
        - Advanced risk metrics (VaR, CVaR, Max Drawdown, Sharpe) from risk_metrics.py
        - Current price as default entry_price for "buy now" analysis
        """
        try:
            # ═══════════════════════════════════════════════════════════════════
            # STEP 1: Get stop loss analysis (includes current_price)
            # ═══════════════════════════════════════════════════════════════════
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

            # ═══════════════════════════════════════════════════════════════════
            # STEP 2: Default entry_price to current_price if not provided
            # Meaning: "What's the risk if I buy RIGHT NOW?"
            # ═══════════════════════════════════════════════════════════════════
            current_price = data.get("current_price", 0)
            actual_entry = entry_price if entry_price is not None else current_price
            using_current_as_entry = entry_price is None and current_price > 0

            # ═══════════════════════════════════════════════════════════════════
            # STEP 3: Calculate advanced risk metrics (VaR, MaxDD, Sharpe)
            # ═══════════════════════════════════════════════════════════════════
            advanced_metrics = None
            if RISK_METRICS_AVAILABLE:
                advanced_metrics = await self._calculate_advanced_risk_metrics(symbol)

            # ═══════════════════════════════════════════════════════════════════
            # STEP 4: Build enhanced LLM summary
            # ═══════════════════════════════════════════════════════════════════
            llm_summary = self._build_risk_analysis_summary(
                symbol=symbol,
                stop_loss_data=data,
                actual_entry=actual_entry,
                using_current_as_entry=using_current_as_entry,
                advanced_metrics=advanced_metrics
            )

            return {
                "success": True,
                "symbol": symbol,
                "entry_price": actual_entry,
                "current_price": current_price,
                "using_current_as_entry": using_current_as_entry,
                "llm_summary": llm_summary,
                "raw_data": data,
                "advanced_metrics": advanced_metrics
            }

        except Exception as e:
            self.logger.error(f"[MarketScanner] Risk analysis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "symbol": symbol
            }

    async def _calculate_advanced_risk_metrics(
        self,
        symbol: str,
        lookback_days: int = 252
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate advanced risk metrics using RiskMetricsCalculator.

        Returns VaR, CVaR, Max Drawdown, Sharpe, Sortino, Volatility, etc.
        """
        try:
            # Get historical price data from technical tool
            tech_result = await self.technical_tool.execute(
                symbol=symbol,
                timeframe="1Y"  # Use 1 year for meaningful risk metrics
            )

            if tech_result.status == "error":
                self.logger.warning(f"[MarketScanner] Could not get price data for risk metrics: {tech_result.error}")
                return None

            raw_data = tech_result.data or {}

            # Extract price series from technical data
            # The technical tool stores OHLCV data that we can use
            prices = raw_data.get("price_series", [])
            dates = raw_data.get("dates", [])

            # If price_series not directly available, try to extract from indicators
            if not prices:
                # Get current_price and estimate from price_context
                current_price = raw_data.get("current_price", 0)
                price_context = raw_data.get("price_context", {})

                if not current_price:
                    self.logger.warning("[MarketScanner] No price data available for risk metrics")
                    return None

                # Build simplified risk metrics from available data
                return self._build_simplified_risk_metrics(raw_data)

            # If we have full price series, use RiskMetricsCalculator
            from datetime import date as date_type

            # Validate minimum data points
            if len(prices) < 30:
                self.logger.warning(f"[MarketScanner] Insufficient data points for risk metrics: {len(prices)}")
                return self._build_simplified_risk_metrics(raw_data)

            # Create input for calculator
            risk_input = RiskDataInput(
                ticker=symbol.upper(),
                dates=[d if isinstance(d, date_type) else date_type.fromisoformat(str(d)[:10]) for d in dates],
                prices=prices
            )

            # Calculate risk metrics
            calculator = RiskMetricsCalculator(RiskCalculationConfig())
            metrics = calculator.calculate(risk_input)

            return {
                "var": {
                    "var_percent": metrics.var.var_percent,
                    "confidence_level": metrics.var.confidence_level,
                    "method": metrics.var.method
                },
                "cvar": {
                    "cvar_percent": metrics.cvar.cvar_percent if metrics.cvar else None
                },
                "max_drawdown": {
                    "max_drawdown": metrics.max_drawdown.max_drawdown,
                    "current_drawdown": metrics.max_drawdown.current_drawdown,
                    "peak_date": str(metrics.max_drawdown.peak_date) if metrics.max_drawdown.peak_date else None,
                    "trough_date": str(metrics.max_drawdown.trough_date) if metrics.max_drawdown.trough_date else None
                },
                "volatility": {
                    "annual": metrics.volatility.annual_volatility,
                    "daily": metrics.volatility.daily_volatility,
                    "regime": metrics.volatility.volatility_regime,
                    "percentile": metrics.volatility.volatility_percentile
                },
                "sharpe_ratio": {
                    "value": metrics.sharpe_ratio.sharpe_ratio,
                    "quality": metrics.sharpe_ratio.quality
                },
                "sortino_ratio": {
                    "value": metrics.sortino_ratio.sortino_ratio if metrics.sortino_ratio else None,
                    "quality": metrics.sortino_ratio.quality if metrics.sortino_ratio else None
                },
                "risk_score": metrics.risk_score,
                "risk_level": metrics.risk_level,
                "summary": metrics.summary
            }

        except Exception as e:
            self.logger.warning(f"[MarketScanner] Error calculating advanced risk metrics: {e}")
            return None

    def _build_simplified_risk_metrics(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build simplified risk metrics from available technical data."""
        try:
            current_price = raw_data.get("current_price", 0)
            indicators = raw_data.get("indicators", {})
            atr_data = indicators.get("atr", {})

            atr_percent = atr_data.get("atr_percent", 0)
            atr_value = atr_data.get("value", 0)

            if not atr_percent:
                return None

            # Estimate volatility from ATR
            # ATR% roughly correlates with daily volatility
            daily_vol = atr_percent / 100
            annual_vol = daily_vol * (252 ** 0.5)  # Annualize

            # Estimate VaR from volatility (parametric method)
            # VaR 95% ≈ mean - 1.645 * std (assuming ~0 daily mean)
            var_95 = -1.645 * daily_vol * 100

            # Classify volatility regime
            if annual_vol < 0.15:
                vol_regime = "low"
            elif annual_vol < 0.30:
                vol_regime = "normal"
            elif annual_vol < 0.50:
                vol_regime = "high"
            else:
                vol_regime = "extreme"

            return {
                "var": {
                    "var_percent": round(var_95, 2),
                    "confidence_level": 0.95,
                    "method": "parametric (estimated from ATR)"
                },
                "volatility": {
                    "annual": round(annual_vol, 4),
                    "daily": round(daily_vol, 4),
                    "regime": vol_regime,
                    "atr_percent": atr_percent
                },
                "source": "Simplified estimation from ATR"
            }

        except Exception as e:
            self.logger.warning(f"[MarketScanner] Error building simplified risk metrics: {e}")
            return None

    def _build_risk_analysis_summary(
        self,
        symbol: str,
        stop_loss_data: Dict[str, Any],
        actual_entry: float,
        using_current_as_entry: bool,
        advanced_metrics: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build comprehensive LLM-friendly risk analysis summary.

        Addresses ChatGPT feedback:
        - Show ATR value explicitly
        - Distinguish Daily Move (ATR) from Tail Risk (VaR) from Annual Volatility
        - Clear data sources
        - R-multiple targets vs technical levels
        """
        current_price = stop_loss_data.get("current_price", actual_entry)

        lines = [
            f"=== RISK ANALYSIS: {symbol} ===",
            ""
        ]

        # ═══════════════════════════════════════════════════════════════════
        # DATA SOURCES (ChatGPT feedback: be explicit about sources)
        # ═══════════════════════════════════════════════════════════════════
        lines.extend([
            "DATA SOURCES:",
            f"  Current Price: ${current_price:.2f} (from market close)",
            f"  Data Period: 60-day historical lookback",
            f"  Timestamp: {stop_loss_data.get('timestamp', 'N/A')}",
            ""
        ])

        # Entry context
        if using_current_as_entry:
            lines.extend([
                "📍 ANALYSIS CONTEXT: Analyzing risk if buying NOW",
                f"Entry Price: ${actual_entry:.2f} (= current market price)",
                "Question: 'What are the risks if I buy this stock right now?'",
                ""
            ])
        else:
            lines.extend([
                f"Entry Price: ${actual_entry:.2f} (user-specified)",
                ""
            ])

        # ═══════════════════════════════════════════════════════════════════
        # ATR METRICS (ChatGPT feedback: show ATR value explicitly)
        # Daily Expected Move - different from VaR (tail risk)
        # ═══════════════════════════════════════════════════════════════════
        atr_metrics = stop_loss_data.get("atr_metrics", {})
        atr_value = atr_metrics.get("atr_value", 0)
        atr_percent = atr_metrics.get("atr_percent", 0)

        # Fallback: extract from stop levels if atr_metrics not available
        if atr_value == 0:
            stop_levels = stop_loss_data.get("stop_loss_levels", {})
            atr_based = stop_levels.get("atr_based", {})
            atr_2x = atr_based.get("atr_2x", 0)
            if atr_2x > 0 and actual_entry > 0:
                # Reverse calculate ATR from atr_2x stop
                atr_value = (actual_entry - atr_2x) / 2
                atr_percent = (atr_value / actual_entry * 100) if actual_entry > 0 else 0

        lines.extend([
            "ATR METRICS (Daily Expected Move - NOT tail risk):",
            f"  ATR Value: ${atr_value:.2f}",
            f"  ATR Percent: {atr_percent:.2f}% of price",
            f"  Daily Move: Stock typically moves ${atr_value:.2f} ({atr_percent:.2f}%) per day",
            f"  Note: ATR = Average True Range (typical daily range)",
            ""
        ])

        # ═══════════════════════════════════════════════════════════════════
        # STOP LOSS LEVELS (with ATR value shown for verification)
        # ═══════════════════════════════════════════════════════════════════
        stop_levels = stop_loss_data.get("stop_loss_levels", {})
        atr_based = stop_levels.get("atr_based", {})
        pct_based = stop_levels.get("percentage_based", {})

        lines.extend([
            f"STOP LOSS LEVELS (ATR = ${atr_value:.2f}):",
            f"  ATR 1x: ${atr_based.get('atr_1x', actual_entry - atr_value):.2f} (risk: ${atr_value:.2f}, {atr_percent:.2f}%)",
            f"  ATR 2x (Conservative): ${atr_based.get('atr_2x', 0):.2f} (risk: ${2*atr_value:.2f}, {2*atr_percent:.2f}%)",
            f"  ATR 3x: ${atr_based.get('atr_3x', 0):.2f} (risk: ${3*atr_value:.2f}, {3*atr_percent:.2f}%)",
            f"  3% Rule: ${pct_based.get('percent_3', 0):.2f}",
            f"  5% Rule (Moderate): ${pct_based.get('percent_5', 0):.2f}",
            f"  7% Rule (Aggressive): ${pct_based.get('percent_7', 0):.2f}",
            f"  Recommended: ${stop_levels.get('recommended', 0):.2f} ({stop_loss_data.get('recommended_method', 'N/A')})",
            ""
        ])

        # Risk per share
        risk_amount = stop_loss_data.get("risk_amount", 0)
        risk_pct = stop_loss_data.get("risk_percentage", 0)
        lines.extend([
            "RISK PER SHARE (using recommended stop):",
            f"  Amount: ${risk_amount:.2f}",
            f"  Percentage: {risk_pct:.2f}%",
            ""
        ])

        # ═══════════════════════════════════════════════════════════════════
        # ADVANCED RISK METRICS (ChatGPT feedback: distinguish metric types)
        # ═══════════════════════════════════════════════════════════════════
        if advanced_metrics:
            lines.append("ADVANCED RISK METRICS:")

            # TAIL RISK (VaR/CVaR) - different from daily move
            var_data = advanced_metrics.get("var", {})
            if var_data:
                var_pct = var_data.get('var_percent', 0)
                var_dollars = abs(var_pct / 100 * actual_entry) if actual_entry > 0 else 0
                lines.extend([
                    "",
                    "  TAIL RISK (worst-case scenarios):",
                    f"    1-Day VaR (95%): {abs(var_pct):.2f}% (${var_dollars:.2f}/share)",
                    f"    Meaning: 5% chance of losing more than {abs(var_pct):.2f}% in a single day",
                ])

            cvar_data = advanced_metrics.get("cvar", {})
            if cvar_data and cvar_data.get("cvar_percent"):
                cvar_pct = cvar_data.get('cvar_percent', 0)
                lines.append(f"    CVaR/Expected Shortfall: {abs(cvar_pct):.2f}% (avg loss when VaR is breached)")

            # VOLATILITY - different from daily move (ATR)
            vol_data = advanced_metrics.get("volatility", {})
            if vol_data:
                annual_vol = vol_data.get('annual', 0)
                daily_vol = vol_data.get('daily', 0)
                regime = vol_data.get('regime', 'unknown')
                lines.extend([
                    "",
                    "  VOLATILITY (price fluctuation intensity):",
                    f"    Annual Volatility: {annual_vol*100:.1f}%",
                    f"    Daily Volatility (std dev): {daily_vol*100:.2f}%",
                    f"    Regime: {regime.upper()}",
                    f"    Note: Different from ATR - volatility is standard deviation of returns",
                ])

            # MAX DRAWDOWN
            mdd_data = advanced_metrics.get("max_drawdown", {})
            if mdd_data:
                max_dd = mdd_data.get("max_drawdown", 0)
                current_dd = mdd_data.get("current_drawdown", 0)
                lines.extend([
                    "",
                    "  MAX DRAWDOWN (historical worst decline):",
                    f"    Max Historical Drawdown: {abs(max_dd)*100:.1f}%",
                ])
                if current_dd and current_dd < -0.05:
                    lines.append(f"    ⚠️ Currently in {abs(current_dd)*100:.1f}% drawdown from peak")

            # RISK-ADJUSTED RETURNS
            sharpe_data = advanced_metrics.get("sharpe_ratio", {})
            if sharpe_data and sharpe_data.get('value'):
                lines.extend([
                    "",
                    "  RISK-ADJUSTED RETURNS:",
                    f"    Sharpe Ratio: {sharpe_data.get('value', 0):.2f} ({sharpe_data.get('quality', 'N/A')})",
                ])

            # OVERALL RISK SCORE
            risk_score = advanced_metrics.get("risk_score")
            risk_level = advanced_metrics.get("risk_level")
            if risk_score:
                lines.extend([
                    "",
                    "  OVERALL RISK ASSESSMENT:",
                    f"    Risk Score: {risk_score}/100 ({risk_level.upper() if risk_level else 'N/A'})",
                ])

            lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # TARGETS (ChatGPT feedback: clarify R-multiple vs technical)
        # ═══════════════════════════════════════════════════════════════════
        target = stop_loss_data.get("target_price", 0)
        rr = stop_loss_data.get("risk_reward_ratio", 0)
        if target and rr:
            lines.extend([
                "TARGETS (R-Multiple Based, NOT Technical Resistance):",
                f"  Target Price (2:1 R:R): ${target:.2f}",
                f"  Risk:Reward Ratio: 1:{rr:.1f}",
                f"  Note: This is a rule-based target using risk per share × 2",
                f"  For technical targets, see Technical Analysis module",
                ""
            ])

        return "\n".join(lines)

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
        """
        Build combined summary from sentiment and news tools.

        Enhanced to address ChatGPT feedback:
        - Include sentiment metadata (scale, window, sample size)
        - List headlines with source and date
        - Classify source types (factual vs opinion)
        - Structure data for theme grouping
        """
        lines = [f"=== SENTIMENT & NEWS ANALYSIS: {symbol} ===", ""]

        # ═══════════════════════════════════════════════════════════════════
        # SENTIMENT SECTION (with full metadata for transparency)
        # ═══════════════════════════════════════════════════════════════════
        if sentiment_data:
            # Extract metadata
            score = sentiment_data.get("sentiment_score", 0)
            label = sentiment_data.get("sentiment_label", "NEUTRAL")
            trend = sentiment_data.get("sentiment_trend", "Stable")
            data_points = sentiment_data.get("data_points", 0)
            raw_score = sentiment_data.get("raw_score", 0.5)
            stocktwits_avg = sentiment_data.get("stocktwits_avg")
            twitter_avg = sentiment_data.get("twitter_avg")
            volatility = sentiment_data.get("volatility", 0)

            lines.extend([
                "═══════════════════════════════════════════════════════════",
                "SENTIMENT DATA (Social Media Analysis)",
                "═══════════════════════════════════════════════════════════",
                "",
                "METADATA (Use these exact values in your analysis):",
                f"  Score: {score:.3f}",
                f"  Scale: -1 to +1 (where -1=very bearish, 0=neutral, +1=very bullish)",
                f"  Label: {label}",
                f"  Time Window: 7 days (default lookback)",
                f"  Data Points: {data_points} samples",
                f"  Sources: StockTwits + Twitter (social media aggregation)",
                f"  Trend: {trend}",
                f"  Volatility: {volatility:.4f} (score variation)",
                ""
            ])

            # Platform breakdown
            if stocktwits_avg is not None or twitter_avg is not None:
                lines.append("PLATFORM BREAKDOWN:")
                if stocktwits_avg is not None:
                    lines.append(f"  StockTwits avg: {stocktwits_avg:.3f} (0-1 scale)")
                if twitter_avg is not None:
                    lines.append(f"  Twitter avg: {twitter_avg:.3f} (0-1 scale)")
                lines.append("")

            # Score interpretation guide
            lines.extend([
                "SCORE INTERPRETATION (for your analysis):",
                f"  Current score {score:.3f} means:",
            ])
            if score > 0.3:
                lines.append("    → Strong bullish sentiment")
            elif score > 0.1:
                lines.append("    → Moderate bullish sentiment")
            elif score > -0.1:
                lines.append("    → Neutral (market in 'wait and see' mode)")
            elif score > -0.3:
                lines.append("    → Moderate bearish sentiment")
            else:
                lines.append("    → Strong bearish sentiment")
            lines.append("")

            # Data quality note
            lines.extend([
                "DATA QUALITY NOTE:",
                "  - Social sentiment reflects RETAIL investor mood",
                "  - NOT representative of institutional investors",
                "  - Updated hourly (not real-time)",
                f"  - Confidence: {'High' if data_points >= 20 else 'Medium' if data_points >= 10 else 'Low'} ({data_points} data points)",
                ""
            ])

        # ═══════════════════════════════════════════════════════════════════
        # NEWS SECTION (with source classification and CLICKABLE LINKS)
        # ═══════════════════════════════════════════════════════════════════
        if news_data:
            articles = news_data.get("articles", [])
            article_count = news_data.get("article_count", len(articles))

            lines.extend([
                "═══════════════════════════════════════════════════════════",
                "NEWS DATA (Recent Headlines WITH URLs)",
                "═══════════════════════════════════════════════════════════",
                "",
                f"Total Articles Retrieved: {article_count}",
                "",
                "⚠️ IMPORTANT: Use these URLs to create CLICKABLE markdown links in your output!",
                "Format: [Title](URL) or [Source](URL)",
                ""
            ])

            if articles:
                # Headlines with URLs for markdown links
                lines.append("TOP HEADLINES (INCLUDE URLs AS CLICKABLE LINKS):")
                lines.append("")

                for i, article in enumerate(articles[:10], 1):
                    title = article.get("title", "Untitled")
                    source = article.get("site", "Unknown")
                    pub_date = article.get("published_date", "")[:10]
                    url = article.get("url", "")
                    text = article.get("text", "")[:200]

                    # Classify source type
                    source_type = self._classify_news_source(source)

                    # Format for easy markdown link creation
                    lines.extend([
                        f"[{i}] HEADLINE: {title}",
                        f"    SOURCE: {source} ({source_type})",
                        f"    DATE: {pub_date}",
                        f"    URL: {url}",
                        f"    MARKDOWN LINK: [{title[:60]}{'...' if len(title) > 60 else ''}]({url})",
                    ])
                    if text:
                        lines.append(f"    PREVIEW: {text}...")
                    lines.append("")

                # Quick reference table
                lines.extend([
                    "QUICK REFERENCE (Copy-paste ready markdown links):",
                    "─────────────────────────────────────────────────────────",
                ])
                for i, article in enumerate(articles[:10], 1):
                    title = article.get("title", "Untitled")
                    short_title = title[:50] + "..." if len(title) > 50 else title
                    source = article.get("site", "Unknown")
                    url = article.get("url", "")
                    if url:
                        lines.append(f"  [{i}] [{short_title}]({url}) - {source}")
                    else:
                        lines.append(f"  [{i}] {short_title} - {source} (no URL)")
                lines.append("")

                # Source type legend
                lines.extend([
                    "SOURCE TYPE LEGEND:",
                    "  Factual: Reuters, Bloomberg, WSJ, AP, CNBC, MarketWatch, Financial Times",
                    "  Opinion: Motley Fool, Seeking Alpha, Nasdaq.com articles, IBD",
                    "  Press Release: BusinessWire, PRNewswire, GlobeNewswire",
                    ""
                ])
            else:
                lines.append("No recent news articles found for this symbol.")
                lines.append("")

        # ═══════════════════════════════════════════════════════════════════
        # ANALYSIS INSTRUCTIONS (Emphasize clickable links)
        # ═══════════════════════════════════════════════════════════════════
        lines.extend([
            "═══════════════════════════════════════════════════════════",
            "ANALYSIS INSTRUCTIONS",
            "═══════════════════════════════════════════════════════════",
            "",
            "1. Use the EXACT metadata values above (score, scale, window, etc.)",
            "2. ⭐ ALWAYS include clickable markdown links: [Title](URL)",
            "3. When discussing themes, cite articles WITH their links",
            "4. Group headlines into 3-5 themes, each theme lists relevant article links",
            "5. Note any conflicting signals between sentiment and news",
            "6. Match the user's language throughout (no mixing languages)",
            "",
            "Example output format:",
            "  'Theo báo cáo từ [Reuters](URL), chip H200 đang gặp vấn đề...'",
            "  'Tin tức tích cực từ [TSMC partnership news](URL) cho thấy...'",
            ""
        ])

        return "\n".join(lines) if len(lines) > 2 else f"No sentiment/news data available for {symbol}"

    def _classify_news_source(self, source: str) -> str:
        """
        Classify news source as Factual, Opinion, or Press Release.

        Based on common financial news source credibility.
        """
        source_lower = source.lower() if source else ""

        # Factual/News Wire sources
        factual_sources = [
            "reuters", "bloomberg", "wsj", "wall street journal", "ap",
            "associated press", "cnbc", "marketwatch", "financial times",
            "ft.com", "barron", "yahoo finance", "investing.com"
        ]

        # Opinion/Analysis sources
        opinion_sources = [
            "motley fool", "seeking alpha", "nasdaq.com", "nasdaq",
            "investor's business daily", "ibd", "benzinga", "thestreet",
            "fool.com", "zacks"
        ]

        # Press Release sources
        pr_sources = [
            "businesswire", "prnewswire", "globenewswire", "accesswire",
            "pr newswire", "business wire"
        ]

        for s in factual_sources:
            if s in source_lower:
                return "Factual"

        for s in opinion_sources:
            if s in source_lower:
                return "Opinion"

        for s in pr_sources:
            if s in source_lower:
                return "Press Release"

        return "Other"

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

        # DEBUG: Log the llm_summary being sent to LLM
        self.logger.info(f"[MarketScanner] LLM Summary length: {len(llm_summary)} chars")
        self.logger.debug(f"[MarketScanner] LLM Summary content:\n{llm_summary[:2000]}..." if len(llm_summary) > 2000 else f"[MarketScanner] LLM Summary content:\n{llm_summary}")

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

        # DEBUG: Log the full prompt being sent to LLM
        self.logger.info(f"[MarketScanner] Full prompt length: {len(prompt)} chars | System prompt length: {len(TECHNICAL_SYSTEM_PROMPT)} chars")

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
        """
        Stream risk analysis from LLM.

        Enhanced features:
        - Default entry_price = current_price (for "buy now" analysis)
        - Advanced risk metrics (VaR, Max Drawdown, Sharpe) if available
        - Beginner-friendly explanations
        """
        analysis_result = await self.get_risk_analysis(symbol, entry_price)

        if not analysis_result.get("success"):
            yield f"Error: {analysis_result.get('error', 'Analysis failed')}"
            return

        llm_summary = analysis_result.get("llm_summary", "")
        using_current_as_entry = analysis_result.get("using_current_as_entry", False)
        actual_entry = analysis_result.get("entry_price", entry_price)
        advanced_metrics = analysis_result.get("advanced_metrics")

        prompt_parts = []

        # Add context about "buying now" if entry_price wasn't specified
        if using_current_as_entry:
            prompt_parts.extend([
                "=== IMPORTANT CONTEXT ===",
                "User did NOT specify an entry price. Using CURRENT MARKET PRICE as entry.",
                "This means: Analyze 'What are the risks if I BUY THIS STOCK RIGHT NOW?'",
                f"Current Price / Entry: ${actual_entry:.2f}",
                "",
                "Frame your analysis for a beginner investor who is considering buying NOW.",
                ""
            ])

        prompt_parts.extend([
            llm_summary if llm_summary else "No risk data available",
            "",
            "=== YOUR TASK ===",
            "Provide comprehensive risk analysis following your output structure.",
            "Be specific with price levels and dollar amounts.",
            "Include position sizing examples for a $10,000 account."
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
