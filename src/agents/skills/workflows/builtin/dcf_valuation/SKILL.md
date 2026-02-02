---
name: dcf-valuation
description: Performs discounted cash flow (DCF) valuation analysis to estimate intrinsic value per share. Triggers when user asks for fair value, intrinsic value, DCF, valuation, price target, undervalued/overvalued analysis.
triggers:
  - fair value
  - intrinsic value
  - DCF
  - valuation
  - price target
  - undervalued
  - overvalued
  - what is it worth
  - định giá
  - giá trị nội tại
  - giá hợp lý
tools_hint:
  - getCashFlow
  - getFinancialRatios
  - getGrowthMetrics
  - getStockPrice
  - getBalanceSheet
  - calculateDCF
max_tokens: 2500
---

# DCF Valuation Workflow

Follow these steps IN ORDER. Use the tools available to gather data at each step.

## Step 1: Gather Financial Data

Call these tools IN PARALLEL in your first data-gathering turn:
- `getCashFlow(symbol="[TICKER]", period="annual", limit=5)` → Extract free_cash_flow history
- `getFinancialRatios(symbol="[TICKER]")` → Extract P/E, ROE, ROIC, debt_to_equity
- `getGrowthMetrics(symbol="[TICKER]")` → Extract revenue_growth, FCF growth rates
- `getStockPrice(symbol="[TICKER]")` → Current market price
- `getBalanceSheet(symbol="[TICKER]")` → Total debt, cash, shares outstanding

**If FCF is missing:** Calculate as: Operating Cash Flow - Capital Expenditure

## Step 2: Calculate FCF Growth Rate

From the 5-year cash flow history, calculate the Compound Annual Growth Rate (CAGR).

**Cross-validate with:**
- YoY FCF growth from getGrowthMetrics
- Revenue growth rate (FCF growth shouldn't sustainably exceed revenue growth)
- Analyst estimates if available

**Growth rate rules:**
- Stable FCF history → Use CAGR with 10-20% haircut
- Volatile FCF → Weight recent years more heavily
- **Cap at 15%** for projection (sustained higher growth is rare)

## Step 3: Estimate Discount Rate (WACC)

Use these defaults as starting points:

| Sector | WACC Range |
|--------|-----------|
| Technology | 8-12% |
| Consumer Staples | 7-8% |
| Financials | 8-10% |
| Healthcare | 8-10% |
| Utilities | 6-7% |
| Energy | 9-11% |

**Adjust for company-specific factors:**
- High debt (D/E > 1.5): +1-2%
- Small cap (< $2B): +1-2%
- Market leader with moat: -0.5-1%
- Recurring revenue: -0.5-1%

**Sanity check:** WACC should be 2-4% below ROIC for value-creating companies.

## Step 4: Project Future Cash Flows (5 Years + Terminal)

- **Years 1-5:** Apply growth rate with annual decay factor (×0.95 each year)
  - Year 1: FCF × (1 + growth_rate)
  - Year 2: FCF × (1 + growth_rate × 0.95)
  - Year 3: FCF × (1 + growth_rate × 0.90)
  - etc.
- **Terminal Value:** Gordon Growth Model with 2.5% terminal growth (GDP proxy)
  - TV = FCF_Year5 × (1 + 0.025) / (WACC - 0.025)

## Step 5: Calculate Present Value & Fair Value

1. Discount each year's FCF to present: PV = FCF / (1 + WACC)^n
2. Discount terminal value: PV_TV = TV / (1 + WACC)^5
3. Enterprise Value = Sum of PV(FCFs) + PV(Terminal Value)
4. Equity Value = Enterprise Value - Total Debt + Cash
5. **Fair Value per Share = Equity Value / Shares Outstanding**

If `calculateDCF` tool is available, use it to compute steps 4-5 automatically.

## Step 6: Sensitivity Analysis

Create a 3×3 matrix varying:
- **WACC:** base ±1% (e.g., 8%, 9%, 10%)
- **Terminal growth:** 2.0%, 2.5%, 3.0%

Present as a formatted table showing fair value per share at each combination.

## Step 7: Validation Checks

Before presenting results, verify:
1. **Terminal value ratio:** Should be 50-80% of total EV for mature companies
   - If >90%: growth rate may be too aggressive
   - If <40%: near-term projections may be too high
2. **Implied P/E:** Fair value ÷ EPS should be reasonable (10-30x for most companies)
3. **vs. Current Price:** If >100% upside/downside, double-check assumptions

## Step 8: Present Results

Structure your response with:
1. **Valuation Summary:** Current price vs. fair value, upside/downside %
2. **Key Assumptions Table:** Growth rate, WACC, terminal growth, with sources
3. **5-Year FCF Projection Table:** Each year's projected FCF and present value
4. **Sensitivity Matrix:** 3×3 table showing fair value range
5. **Investment Thesis:** Buy/Hold/Sell recommendation with reasoning
6. **Caveats:** DCF limitations + company-specific risks

**Language:** Match the user's language throughout.
