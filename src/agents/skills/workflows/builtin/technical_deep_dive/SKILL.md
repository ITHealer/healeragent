---
name: technical-deep-dive
description: Comprehensive multi-timeframe technical analysis with actionable trading signals. Triggers for deep technical analysis, multi-timeframe analysis, trading setup, entry/exit points.
triggers:
  - technical analysis
  - multi-timeframe
  - trading setup
  - entry point
  - exit point
  - support resistance
  - phân tích kỹ thuật
  - điểm vào
  - điểm ra
tools_hint:
  - getTechnicalIndicators
  - getRelativeStrength
  - getIchimokuCloud
  - getFibonacciLevels
  - getStockPrice
  - getStockNews
max_tokens: 2000
---

# Technical Deep Dive Workflow

Follow these steps for comprehensive technical analysis.

## Step 1: Gather Multi-Timeframe Data

Call these tools IN PARALLEL:
- `getTechnicalIndicators(symbol="[TICKER]")` → RSI, MACD, Bollinger, Moving Averages
- `getRelativeStrength(symbol="[TICKER]")` → RS rating vs market/sector
- `getStockPrice(symbol="[TICKER]")` → Current price, volume, price change
- `getIchimokuCloud(symbol="[TICKER]")` → Ichimoku analysis (if available)
- `getFibonacciLevels(symbol="[TICKER]")` → Key fib retracement levels (if available)

## Step 2: Trend Analysis

Determine the primary trend across timeframes:
- **Long-term (Weekly):** 200-day MA direction, long-term trend
- **Medium-term (Daily):** 50-day MA direction, intermediate trend
- **Short-term (Hourly/4H):** 20-day MA, recent momentum

**Confluence check:** All timeframes aligned = strong signal. Mixed = caution.

## Step 3: Key Technical Levels

Identify and rank:
1. **Support levels:** Where buyers are likely to step in
2. **Resistance levels:** Where selling pressure increases
3. **Fibonacci levels:** 38.2%, 50%, 61.8% retracements
4. **Ichimoku levels:** Cloud support/resistance, Tenkan/Kijun crosses

## Step 4: Momentum & Oscillator Analysis

Analyze:
- **RSI:** Overbought (>70) / Oversold (<30) / Divergences
- **MACD:** Signal line crossovers, histogram direction
- **Bollinger Bands:** Squeeze (low volatility) / Breakout signals
- **Volume:** Confirmation of price moves, unusual volume

## Step 5: Trading Setup

If applicable, identify:
- **Entry zone:** Price range for potential entry
- **Stop loss:** Based on key support or ATR-based
- **Target levels:** Based on resistance, Fibonacci extensions
- **Risk/Reward ratio:** Must be at least 2:1

## Step 6: Present Results

Structure your response:
1. **Trend Summary:** Bullish/Bearish/Neutral with confidence level
2. **Key Levels Table:** Support, Resistance, Fibonacci levels
3. **Indicator Signals:** RSI, MACD, BB readings with interpretation
4. **Relative Strength:** vs. market and sector peers
5. **Trading Setup (if applicable):** Entry, Stop, Target, R/R
6. **Risk Factors:** What could invalidate the analysis

**Language:** Match the user's language throughout.
