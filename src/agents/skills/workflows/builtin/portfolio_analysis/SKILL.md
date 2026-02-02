---
name: portfolio-analysis
description: Comprehensive portfolio analysis with optimization, diversification assessment, and rebalancing suggestions. Triggers for portfolio review, optimization, diversification, allocation.
triggers:
  - portfolio
  - diversification
  - allocation
  - rebalance
  - optimize portfolio
  - danh mục
  - phân bổ
  - đa dạng hóa
tools_hint:
  - optimizePortfolio
  - getCorrelationMatrix
  - analyzePortfolioDiversification
  - suggestRebalancing
  - getStockPrice
  - getFinancialRatios
max_tokens: 2000
---

# Portfolio Analysis Workflow

Follow these steps for comprehensive portfolio analysis.

## Step 1: Gather Portfolio Data

Call these tools IN PARALLEL for all symbols in the portfolio:
- `getStockPrice(symbol="[TICKER]")` for each holding → Current prices
- `getFinancialRatios(symbol="[TICKER]")` for each → Key metrics
- `getCorrelationMatrix(symbols=[...])` → Cross-correlation (if available)

## Step 2: Risk Assessment

Analyze portfolio-level risk:
- **Concentration risk:** Any single position >20% of portfolio?
- **Sector concentration:** Any sector >40%?
- **Correlation risk:** Highly correlated holdings (>0.8)?
- **Beta:** Portfolio-weighted beta vs. market

Use `analyzePortfolioDiversification` if available.

## Step 3: Performance Analysis

For each holding and portfolio overall:
- Return (1M, 3M, YTD, 1Y)
- Sharpe Ratio (risk-adjusted return)
- Maximum Drawdown
- Volatility

## Step 4: Optimization

Use `optimizePortfolio` if available to find:
- **Maximum Sharpe Ratio** portfolio (best risk-adjusted return)
- **Minimum Volatility** portfolio (lowest risk)
- **Current vs. Optimal** allocation comparison

## Step 5: Rebalancing Suggestions

Based on analysis, suggest:
- Which positions to increase/decrease
- New positions to consider for diversification
- Positions to potentially exit (high correlation, underperforming)

Use `suggestRebalancing` if available.

## Step 6: Present Results

Structure your response:
1. **Portfolio Summary:** Total value, number of holdings, overall allocation
2. **Risk Metrics Table:** Beta, Sharpe, Max Drawdown, Volatility
3. **Diversification Score:** With specific improvement suggestions
4. **Correlation Matrix (if available):** Highlight high correlations
5. **Optimization Results:** Current vs. suggested allocation
6. **Action Items:** Specific rebalancing recommendations
7. **Caveats:** Past performance, model limitations

**Language:** Match the user's language throughout.
