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
  - calculateComparables
max_tokens: 3500
---

# DCF Valuation Workflow

Follow these steps IN ORDER. Use the tools available to gather data at each step.

## Step 1: Gather Financial Data

Call these tools IN PARALLEL in your first data-gathering turn:
- `getCashFlow(symbol="[TICKER]", period="annual", limit=5)` → Extract free_cash_flow history
- `getFinancialRatios(symbol="[TICKER]")` → Extract P/E, P/B, ROE, ROIC, EV/EBITDA, debt_to_equity
- `getGrowthMetrics(symbol="[TICKER]")` → Extract revenue_growth, FCF growth rates
- `getStockPrice(symbol="[TICKER]")` → Current market price
- `getBalanceSheet(symbol="[TICKER]")` → Total debt, cash, shares outstanding

Select 3-4 industry peers for comparable analysis (e.g., for MSFT: AAPL, GOOGL, AMZN).
Peer data will be fetched automatically in Step 7 using `calculateComparables`.

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

Use the `calculateDCF` tool which automatically computes:
1. 5-year FCF projections with present values
2. Terminal value and its present value
3. Enterprise Value = Sum of PV(FCFs) + PV(Terminal Value)
4. Equity Value = Enterprise Value - Total Debt + Cash
5. **Fair Value per Share = Equity Value / Shares Outstanding**

**New parameters available:**
- `normalize_fcf=true`: Use 3-year average FCF for companies with volatile cash flows

**The tool automatically provides:**
- **TV as % of EV**: If >75%, the valuation is heavily dependent on long-term assumptions
- **Implied Growth Rate (Reverse DCF)**: What perpetual growth the market is pricing in
- **2D Sensitivity Matrix**: WACC × Terminal Growth matrix (5×5)
- **Validation Warnings**: Alerts for potential issues with assumptions

## Step 5.5: Interpret Reverse DCF (Market Expectations)

The `calculateDCF` tool returns `implied_growth_rate` — the terminal growth rate the market is implicitly pricing in.

**Interpretation:**
- **Implied growth > Your assumption**: Market is more optimistic than your model → Stock may be fairly valued or overvalued
- **Implied growth < Your assumption**: Market is more pessimistic → Potential undervaluation opportunity
- **Implied growth > 4%**: Market expects exceptional long-term growth (verify if realistic)
- **Implied growth < 0%**: Market prices in decline (investigate structural risks)

Use this to cross-validate your assumptions against market expectations.

## Step 6: Sensitivity Analysis

**The `calculateDCF` tool now provides a 5×5 sensitivity matrix automatically.**

Review the matrix in `sensitivity_2d`:
- **WACC (rows):** base ±2% in 1% increments
- **Terminal growth (columns):** 1.5% to 3.5%

**Key insights to highlight:**
1. **Range of fair values**: Min to Max from the matrix
2. **Current price position**: Where does current price fall in the matrix? (marked with *)
3. **Most likely scenario**: Your base case (center of matrix)
4. **Margin of safety**: What WACC/growth combination justifies current price?

## Step 7: Comparable Analysis (Peer Multiples)

**USE THE `calculateComparables` TOOL** to automatically fetch peer ratios and compute implied fair values:

```
calculateComparables(symbol="[TICKER]", peers=["PEER1", "PEER2", "PEER3"])
```

This tool will:
1. Fetch P/E (TTM), P/E (Forward), P/B, P/S, EV/EBITDA ratios for ALL peers automatically
2. Fetch the target company's EPS, Book Value/Share, Revenue/Share
3. Calculate implied fair values: Target metric × Peer average multiple
4. Return a complete comparison table with actual numerical values

**DO NOT manually fetch `getFinancialRatios` for each peer** — the `calculateComparables` tool handles this in parallel.

**P/E Type Clarification (NEW):**
The tool now distinguishes between:
- **P/E (TTM)**: Trailing Twelve Months — based on actual historical earnings
- **P/E (Forward)**: Based on analyst earnings estimates for next fiscal year

The output indicates which type is used. When comparing:
- Use **TTM P/E** for stable, mature companies with predictable earnings
- Use **Forward P/E** for growth companies where future earnings matter more
- If peers have mixed types, note this limitation in your analysis

The tool returns:
- Peer comparison table with **actual P/E, P/B, P/S, EV/EBITDA values** with type labels
- Implied fair values for each multiple (using peer averages and medians)
- Average and median intrinsic value estimates
- Upside/downside potential and verdict

**Present the results as a formatted table:**

| Company | Price | P/E (TTM) | P/E (Fwd) | P/B | P/S | EV/EBITDA |
|---------|-------|-----------|-----------|-----|-----|-----------|
| Target  | $xxx | actual | actual | actual | actual | actual |
| Peer 1  | - | actual | actual | actual | actual | actual |
| Peer 2  | - | actual | actual | actual | actual | actual |
| **Avg** | - | avg | - | avg | avg | avg |

**CRITICAL:** Every cell must show a real number from the API. No "-" or "N/A" for ratios that exist.

## Step 8: Validation Checks

**The `calculateDCF` tool now provides automatic validation warnings.** Review and address them.

**Automatic warnings include:**
- ⚠️ **TV > 75% of EV**: Valuation heavily depends on long-term assumptions
- ⚠️ **Terminal growth > 3%**: Exceeds typical GDP growth; verify if realistic
- ⚠️ **Extreme upside (>100%)** or **downside (<-50%)**: Double-check all inputs
- ⚠️ **Implied growth > 4%**: Market expects exceptional growth
- ⚠️ **Implied growth < 0%**: Market prices in structural decline
- ⚠️ **WACC outside 6-15%**: May be too aggressive or conservative

**Additional manual checks:**
1. **Implied P/E Check:** Fair value ÷ EPS should be reasonable
   - Value stocks: 8-15x
   - Average companies: 15-25x
   - Growth stocks: 25-40x
   - If >50x, growth assumptions may be too aggressive
2. **FCF Volatility:** If high volatility was flagged, consider using `normalize_fcf=true`
3. **Cross-validation:** Compare DCF fair value vs. Comparable fair value (see Step 8.5)

## Step 8.5: Valuation Reconciliation (DCF vs. Comparable)

**CRITICAL STEP:** Reconcile different valuation methods to arrive at a final fair value range.

**Reconciliation Framework:**

| Scenario | DCF vs. Comparable | Interpretation | Recommendation |
|----------|-------------------|----------------|----------------|
| **Converging** | Within 15% | High confidence | Use average as fair value |
| **DCF Higher** | >15% gap | DCF may be too optimistic OR peers are undervalued | Weight more toward Comparable |
| **Comparable Higher** | >15% gap | Peers may be overvalued OR DCF assumptions too conservative | Weight more toward DCF |
| **Large Divergence** | >50% gap | Significant uncertainty | Present as wide range; investigate differences |

**Root Cause Analysis when diverging:**

1. **If DCF >> Comparable:**
   - Is growth rate assumption too aggressive vs. peers?
   - Is WACC too low (less risk than peers)?
   - Does company have unique competitive advantage justifying premium?

2. **If Comparable >> DCF:**
   - Are peers in a bubble (especially check forward P/E)?
   - Is FCF depressed due to temporary factors?
   - Did you use TTM metrics during a weak quarter?

**Final Fair Value Calculation:**

```
Weighted Fair Value = (DCF × DCF_Weight) + (Comparable × Comp_Weight)

Weights based on confidence:
- High confidence in both: 50% / 50%
- Better data quality on one: 60% / 40%
- One method has major issues: 70% / 30%
```

**Triangulation with Market Expectations:**
Use Reverse DCF implied growth to check if market agrees with your thesis:
- Market implied growth ≈ Your growth assumption → Fair value likely accurate
- Market implied growth << Your assumption → Potential upside (or you're too optimistic)
- Market implied growth >> Your assumption → Market may know something you don't

## Step 9: Present Results

Structure your response with these sections:

### 9.1 Executive Summary
- Current price vs. Fair value range (DCF + Comparable weighted)
- Upside/downside % and verdict (Undervalued/Fairly Valued/Overvalued)
- Confidence level (High/Medium/Low based on DCF-Comparable convergence)

### 9.2 Key Assumptions Table
| Parameter | Value | Source/Rationale |
|-----------|-------|------------------|
| FCF Base Year | $X B | API (or Normalized 3-yr avg) |
| Growth Rate (Y1-5) | X% declining | CAGR ± haircut |
| WACC | X% | Sector default ± adjustments |
| Terminal Growth | 2.5% | GDP proxy |

### 9.3 DCF Valuation Output
- Include the full DCF output from the tool
- Highlight: TV as % of EV, Implied growth vs. your assumption
- Address any validation warnings

### 9.4 Sensitivity Matrix (5×5)
Present the auto-generated matrix. Highlight:
- Base case (center)
- Where current price falls
- "Margin of safety" scenarios

### 9.5 Comparable Analysis Table
Include P/E type labels (TTM/Forward). Show both peer table and implied values.

### 9.6 Reconciliation Summary
| Method | Fair Value | Weight | Contribution |
|--------|-----------|--------|--------------|
| DCF | $XXX | 50% | $XXX |
| Comparable | $XXX | 50% | $XXX |
| **Weighted** | **$XXX** | 100% | |

Explain any divergence and your weighting rationale.

### 9.7 Investment Thesis
- **Verdict:** Buy / Hold / Sell
- **Target Price Range:** $XX - $XX (from sensitivity + comparable range)
- **Key drivers:** Why this stock is mispriced
- **Catalysts:** What could unlock value
- **Risks:** What could go wrong

### 9.8 Caveats
- DCF limitations (relies on projections, sensitive to terminal value)
- Data recency (when was FCF/earnings data from?)
- Company-specific risks

**Language:** Match the user's language throughout.
