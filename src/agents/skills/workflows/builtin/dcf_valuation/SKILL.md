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

## Step 1.5: Document FCF Source (CRITICAL)

**CRITICAL:** Always cite the FCF source explicitly in your output. Users must be able to verify every number.

### If using API data directly:

```
📊 FCF BASE CALCULATION:

Source: FMP API, Cash Flow Statement FY2024
Operating Cash Flow: $XX,XXX M
Capital Expenditure: $(XX,XXX) M
Free Cash Flow = OCF - CapEx = $XX,XXX M

FCF Base for DCF: $XX,XXX M
```

### If using normalized FCF:

```
📊 FCF BASE CALCULATION (Normalized):

FCF History:
- FY2024: $XX,XXX M
- FY2023: $XX,XXX M
- FY2022: $XX,XXX M

3-Year Average: $XX,XXX M
Coefficient of Variation: XX.X%
Decision: Using normalized FCF due to [reason: AI capex cycle / one-time items / etc.]
```

**MANDATORY REQUIREMENTS:**
1. **NEVER** state growth rates (e.g., "FCF grows 14%") without the base amount ("14% of $59,000M")
2. **ALWAYS** show the source (FMP API, which fiscal year)
3. **ALWAYS** show OCF and CapEx components, not just the final FCF number

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

## Step 2.5: FCF Normalization Decision (CRITICAL for Tech/High-Growth Companies)

**For companies in heavy investment cycles (AI/Cloud: MSFT, GOOGL, AMZN, META, NVDA):**

### 2.5.1 Check FCF Volatility:
1. Get 3-year FCF history from getCashFlow
2. Calculate coefficient of variation (CV):
   ```
   CV = (Standard Deviation / Mean) × 100%
   ```
3. If CV > 30%, FCF is volatile → Consider normalization

### 2.5.2 Normalization Decision Table:

| Scenario | FCF Trend | Recommendation |
|----------|-----------|----------------|
| Heavy AI/Cloud capex (current cycle) | FCF declining | Use normalized (3yr avg) OR add capex-back analysis |
| Capex cycle ending | FCF recovering | Use current FCF |
| Stable operations | Consistent FCF | Use current FCF |
| One-time events affecting FCF | Distorted | Use normalized (3yr avg) |

### 2.5.3 Output Format (MANDATORY):

```
📊 FCF NORMALIZATION ANALYSIS:

FY2024 FCF: $XX,XXX M (YoY change: XX%)
FY2023 FCF: $XX,XXX M
FY2022 FCF: $XX,XXX M
3-Year Average: $XX,XXX M
Volatility (CV): XX.X%

Decision: [Using current FCF / Using normalized FCF]
Rationale: [Explain why - e.g., "Despite AI datacenter investment, FCF
remains stable with CV < 30%. Capex increase is growth investment,
not operational weakness."]
```

**ALWAYS** explain your FCF choice in the output. Never silently choose a method.

## Step 3: Estimate Discount Rate (WACC)

### 3.1 WACC Formula (MANDATORY to show derivation)

```
WACC = (E/V × Re) + (D/V × Rd × (1-T))

Where:
- Re = Cost of Equity = Rf + β × ERP
- Rd = Cost of Debt = Interest Expense / Total Debt
- E/V = Equity weight = Market Cap / (Market Cap + Total Debt)
- D/V = Debt weight = Total Debt / (Market Cap + Total Debt)
- T = Corporate tax rate (typically 21% for US companies)
```

### 3.2 Component Sources

| Component | Symbol | Source | How to Get |
|-----------|--------|--------|------------|
| Risk-free Rate | Rf | 10-year Treasury | getEconomicIndicator("TREASURY10Y") or default 4.0-4.5% |
| Beta | β | Company profile | getCompanyProfile() → beta field |
| Equity Risk Premium | ERP | Market consensus | Default 5.0-6.0% for US market |
| Total Debt | D | Balance sheet | getBalanceSheet() → totalDebt |
| Market Cap | E | Stock price | getStockPrice() × shares_outstanding |
| Interest Expense | Int | Income statement | From financial ratios or ~4-6% for investment grade |
| Tax Rate | T | Income statement | Effective tax rate or default 21% |

### 3.3 WACC Precision Rule (CRITICAL)

**NEVER round WACC arbitrarily.** This is a common error that destroys valuation credibility.

| ❌ BAD | ✅ GOOD |
|--------|---------|
| "WACC = 8.95%, rounded to 8.5%" | "WACC = 8.95%, using 9.0% (rounded to nearest 0.5% for sensitivity)" |
| "WACC ≈ 9%" (no derivation) | "WACC = 8.97% (see derivation below), using 9.0% in model" |
| Arbitrary rounding (8.95% → 8.5%) | Systematic rounding (always to nearest 0.25% or 0.5%) |

**Rounding Discipline:**
- If you must round, round to nearest **0.25%** (e.g., 8.97% → 9.00%)
- ALWAYS show the calculated WACC before rounding
- ALWAYS explain rounding rationale: "Rounded to 9.0% for sensitivity matrix alignment"
- Sensitivity matrix should include the EXACT calculated WACC as base case, not a rounded version

### 3.4 WACC Calculation Output Format (MANDATORY)

**CRITICAL: Always show this calculation in your output, not just the final WACC number.**

```
📊 WACC DERIVATION:

COMPONENT VALUES:
- Risk-free Rate (Rf): X.XX% (Source: 10Y Treasury / FMP API)
- Beta (β): X.XX (Source: FMP Company Profile)
- Equity Risk Premium (ERP): X.X% (Market consensus)
- Total Debt: $XX,XXX M (Source: FMP Balance Sheet FY2024)
- Market Cap: $XXX,XXX M (Source: FMP, current price × shares)
- Cost of Debt (Rd): X.XX% (Interest expense / Total debt)
- Tax Rate (T): XX% (Effective rate from financials)

CALCULATION:
Cost of Equity (Re) = Rf + β × ERP
                    = X.XX% + X.XX × X.X%
                    = X.XX%

Equity Weight (E/V) = Market Cap / (Market Cap + Debt)
                    = $XXX,XXX / ($XXX,XXX + $XX,XXX)
                    = XX.X%

Debt Weight (D/V)   = 100% - E/V = XX.X%

WACC = (E/V × Re) + (D/V × Rd × (1-T))
     = (XX.X% × X.XX%) + (XX.X% × X.XX% × 0.79)
     = X.XX%

WACC USED: X.X% (rounded)
```

### 3.5 Sector Defaults (Use when data unavailable)

| Sector | WACC Range | Typical Beta |
|--------|-----------|--------------|
| Technology | 8-12% | 1.0-1.3 |
| Consumer Staples | 7-8% | 0.6-0.8 |
| Financials | 8-10% | 1.0-1.2 |
| Healthcare | 8-10% | 0.8-1.0 |
| Utilities | 6-7% | 0.4-0.6 |
| Energy | 9-11% | 1.1-1.4 |

### 3.6 Company-Specific Adjustments

**Adjust from calculated/sector WACC for:**
- High debt (D/E > 1.5): +1-2%
- Small cap (< $2B market cap): +1-2%
- Market leader with moat: -0.5-1%
- Recurring revenue model: -0.5-1%
- Emerging market exposure: +1-2%

**Sanity check:** WACC should be 2-4% below ROIC for value-creating companies. If WACC > ROIC, company is destroying value.

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

## Step 5.6: Consistency Check (CRITICAL - MANDATORY)

**The DCF fair value MUST EXACTLY match the sensitivity matrix base case.** This is a non-negotiable requirement.

### Consistency Rule:
```
DCF_Fair_Value == Sensitivity_Matrix[Base_WACC][Base_TGR]
```

| ❌ INCONSISTENT (UNACCEPTABLE) | ✅ CONSISTENT (REQUIRED) |
|--------------------------------|--------------------------|
| DCF Fair Value: $418.50 | DCF Fair Value: $418.50 |
| Sensitivity Base Case: $459.00 | Sensitivity Base Case: $418.50 |
| (Different values = ERROR) | (Identical values = CORRECT) |

### Why Inconsistency Happens:
1. **WACC rounding**: DCF uses 8.95%, sensitivity uses 8.5% → Different results
2. **Different FCF base**: Tool vs. manual calculation mismatch
3. **Terminal growth mismatch**: Hardcoded 2.5% in sensitivity vs. different value in DCF

### How to Ensure Consistency:
1. **Use EXACT same inputs** for both DCF calculation and sensitivity matrix
2. **WACC in sensitivity base row** = WACC used in DCF (no rounding difference)
3. **TGR in sensitivity base column** = Terminal Growth used in DCF
4. **Verify before presenting**: Check that `intrinsic_value` from DCF equals `sensitivity_2d[base_wacc_row][base_tgr_col]`

### Mandatory Consistency Statement:
Include this in your output:
```
✅ CONSISTENCY CHECK:
- DCF Fair Value: $XXX.XX
- Sensitivity Base Case (WACC X.X%, TGR X.X%): $XXX.XX
- Status: ✅ MATCHED / ❌ MISMATCH (investigate if mismatch)
```

**If there is a mismatch, DO NOT present the analysis. Debug first.**

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

### 9.3 DCF Valuation Output (MUST INCLUDE ALL COMPONENTS)

**MANDATORY: Include the following from calculateDCF output:**

#### 9.3.1 FCF Projection Table:
```
📈 FCF PROJECTIONS:

| Year | FCF ($M) | Growth Rate | Discount Factor | Present Value ($M) |
|------|----------|-------------|-----------------|-------------------|
| Base | $XX,XXX | - | - | - |
| Y1 | $XX,XXX | XX.X% | 0.XXX | $XX,XXX |
| Y2 | $XX,XXX | XX.X% | 0.XXX | $XX,XXX |
| Y3 | $XX,XXX | XX.X% | 0.XXX | $XX,XXX |
| Y4 | $XX,XXX | XX.X% | 0.XXX | $XX,XXX |
| Y5 | $XX,XXX | XX.X% | 0.XXX | $XX,XXX |

Sum of PV (FCF Years 1-5): $XXX,XXX M
```

#### 9.3.2 Terminal Value Breakdown:
```
📊 TERMINAL VALUE:

Terminal FCF (Y5 × (1 + TGR)): $XX,XXX M
Terminal Growth Rate (TGR): X.X%
WACC: X.X%
Terminal Value = FCF × (1 + TGR) / (WACC - TGR) = $XXX,XXX M
PV of Terminal Value: $XXX,XXX M
TV as % of Enterprise Value: XX.X%  [Flag if >75%]
```

#### 9.3.3 Reverse DCF Analysis (MANDATORY):
```
🔄 REVERSE DCF (What Market is Pricing In):

Current Market Price: $XXX.XX
Implied Terminal Growth Rate: X.XX%
Your Model Assumption: X.XX%

INTERPRETATION:
- [If Implied < Assumed]: Market expects LOWER growth than your model
  → Potential UNDERVALUATION if your assumptions are correct

- [If Implied > Assumed]: Market expects HIGHER growth than your model
  → Stock may be FAIRLY VALUED or OVERVALUED

- [If Implied > 4%]: Market pricing in exceptional perpetual growth
  → Verify if this is realistic for the company

- [If Implied < 0%]: Market pricing in structural decline
  → Investigate fundamental risks

Market Expectation vs. Your View: [Aligned / Divergent - explain]
```

**This Reverse DCF section is MANDATORY. It helps users understand if market agrees with your assumptions.**

#### 9.3.4 Validation Warnings:
Address all warnings from the tool output (TV %, extreme upside/downside, WACC range, etc.)

### 9.4 Sensitivity Matrix (MANDATORY - Full 5×5 Display)

**YOU MUST include the complete 5×5 matrix from calculateDCF output.**

```
📊 SENSITIVITY ANALYSIS: Fair Value per Share

| WACC ↓ \ TGR → | 1.5% | 2.0% | 2.5% | 3.0% | 3.5% |
|----------------|------|------|------|------|------|
| X.0% (Base-2%) | $XXX | $XXX | $XXX | $XXX | $XXX |
| X.0% (Base-1%) | $XXX | $XXX | $XXX | $XXX | $XXX |
| X.0% (Base)    | $XXX | $XXX | [$XXX]* | $XXX | $XXX |
| X.0% (Base+1%) | $XXX | $XXX | $XXX | $XXX | $XXX |
| X.0% (Base+2%) | $XXX | $XXX | $XXX | $XXX | $XXX |

* = Base case (your assumptions)
Current Market Price: $XXX.XX
```

**MANDATORY Interpretation:**

1. **Value Range:**
   - Minimum fair value: $XXX (highest WACC, lowest TGR)
   - Maximum fair value: $XXX (lowest WACC, highest TGR)
   - Spread: $XXX to $XXX (XX% range)

2. **Current Price Position:**
   - At $XXX current price, market is pricing ~X.X% WACC and ~X.X% TGR
   - Compared to your base case: [above/below/at fair value]

3. **Margin of Safety Scenarios:**
   - Conservative (High WACC + Low TGR): $XXX → XX% [upside/downside]
   - Optimistic (Low WACC + High TGR): $XXX → XX% upside
   - Your base case: $XXX → XX% [upside/downside]

4. **Key Insight:**
   "Even under conservative assumptions (X% WACC, X% TGR), fair value of $XXX
   still implies XX% [upside/downside] from current price."

**DO NOT summarize this matrix. Present it in full so users can explore scenarios.**

### 9.5 Comparable Analysis Table (MANDATORY DETAIL)

**YOU MUST show implied price from EACH multiple, not just the average.**

#### 9.5.1 Peer Comparison Table:
```
📊 PEER MULTIPLES COMPARISON:

| Company | Price | P/E (TTM) | P/E (Fwd) | P/B | P/S | EV/EBITDA |
|---------|-------|-----------|-----------|-----|-----|-----------|
| [TARGET] | $XXX | XX.Xx | XX.Xx | X.Xx | X.Xx | XX.Xx |
| Peer 1 | $XXX | XX.Xx | XX.Xx | X.Xx | X.Xx | XX.Xx |
| Peer 2 | $XXX | XX.Xx | XX.Xx | X.Xx | X.Xx | XX.Xx |
| Peer 3 | $XXX | XX.Xx | XX.Xx | X.Xx | X.Xx | XX.Xx |
| **Median** | - | XX.Xx | - | X.Xx | X.Xx | XX.Xx |
| **Average** | - | XX.Xx | - | X.Xx | X.Xx | XX.Xx |

Source: FMP API (TTM basis as of [date])
```

#### 9.5.2 Implied Fair Value from Each Multiple (CRITICAL):
```
📈 IMPLIED FAIR VALUES BY MULTIPLE:

| Multiple | Target Value | Peer Median | Peer Avg | Implied Price (Median) | Implied Price (Avg) |
|----------|--------------|-------------|----------|------------------------|---------------------|
| P/E (TTM) | XX.Xx | XX.Xx | XX.Xx | $XXX (EPS $XX × 33.0) | $XXX |
| P/B | X.Xx | X.Xx | XX.Xx | $XXX (BV $XX × X.X) | $XXX* |
| P/S | X.Xx | X.Xx | X.Xx | $XXX (Rev/Sh $XX × X.X) | $XXX |
| EV/EBITDA | XX.Xx | XX.Xx | XX.Xx | $XXX | $XXX |

* Outlier detected: [Peer X] has extreme P/B of XXx → Using median instead of average
```

#### 9.5.3 Weighted Comparable Fair Value:
```
📊 WEIGHTED COMPARABLE VALUATION:

Method: Exclude outliers, weight by reliability

| Multiple | Implied Price | Weight | Contribution | Rationale |
|----------|---------------|--------|--------------|-----------|
| P/E (TTM) | $XXX | 40% | $XXX | Most reliable for profitable companies |
| P/S | $XXX | 30% | $XXX | Good for growth comparison |
| EV/EBITDA | $XXX | 30% | $XXX | Accounts for capital structure |
| P/B | EXCLUDED | 0% | - | Outlier bias from [Peer X] |
| **TOTAL** | - | **100%** | **$XXX** | |

Comparable Fair Value: $XXX
```

**CRITICAL: Calculation must be VISIBLE. Never just output a single average number without showing how each multiple contributes.**

#### 9.5.4 Outlier Handling:
If any peer has extreme multiples (>2x median or <0.5x median):
- Flag the outlier explicitly
- Explain the cause (e.g., "AAPL P/B of 45x due to capital return program")
- Use MEDIAN instead of AVERAGE for that multiple
- Or exclude from weighted calculation with explanation

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

### 9.8 Risk Framework with Scenario Analysis (MANDATORY)

**A professional valuation MUST include a risk framework with probability-weighted scenarios.**

#### 9.8.1 Bull/Base/Bear Scenario Table:

```
📊 SCENARIO ANALYSIS:

| Scenario | Probability | Fair Value | Upside/Downside | Key Assumptions |
|----------|-------------|------------|-----------------|-----------------|
| 🐂 Bull | 25% | $XXX | +XX% | [Specific quantitative triggers] |
| ⚖️ Base | 50% | $XXX | +XX% | [Your DCF base case assumptions] |
| 🐻 Bear | 25% | $XXX | -XX% | [Specific quantitative triggers] |

Probability-Weighted Fair Value: $XXX
(= 0.25 × Bull + 0.50 × Base + 0.25 × Bear)
```

#### 9.8.2 Quantitative Triggers (CRITICAL - NOT QUALITATIVE):

**NEVER use vague qualitative triggers. Every trigger MUST be measurable.**

| ❌ QUALITATIVE (UNACCEPTABLE) | ✅ QUANTITATIVE (REQUIRED) |
|-------------------------------|---------------------------|
| "AI momentum continues" | "Azure revenue growth >25% YoY for 2 consecutive quarters" |
| "Economy weakens" | "US GDP growth <1.5% or unemployment >5%" |
| "Competition intensifies" | "Market share decline >3% or pricing pressure reduces margins >200bps" |
| "Strong execution" | "FCF margin improves to >35% from current 32%" |

**Each scenario MUST have 2-3 quantitative triggers:**

```
🐂 BULL CASE TRIGGERS:
1. Azure revenue growth sustains >30% YoY through FY2026
2. AI Copilot reaches 50M+ paid subscribers by end of FY2025
3. Operating margin expands to >47% (from current 44%)

⚖️ BASE CASE TRIGGERS:
1. Azure growth normalizes to 20-25% YoY
2. FCF grows at 10-12% annually
3. WACC remains stable at ~9%

🐻 BEAR CASE TRIGGERS:
1. Cloud growth decelerates to <15% due to enterprise spending cuts
2. AI competition from Google/Amazon erodes margins by >300bps
3. Regulatory action impacts bundling strategy (EU/US)
```

#### 9.8.3 Position Sizing & Risk Management:

```
📊 RISK MANAGEMENT:

INVALIDATION LEVELS:
- Stop-loss trigger: $XXX (-XX% from current) - Price where thesis is invalidated
- Thesis invalidation: [Specific event, e.g., "Azure growth <10% for 2 quarters"]

POSITION SIZING (Based on conviction & risk):
- High conviction (DCF + Comparable converge): Up to 5-8% of portfolio
- Medium conviction (some uncertainty): 2-4% of portfolio
- Speculative (high divergence/uncertainty): 1-2% of portfolio

RISK-REWARD ANALYSIS:
- Upside to Bull case: +XX% ($XXX)
- Downside to Bear case: -XX% ($XXX)
- Risk/Reward Ratio: X.Xx (should be >2.0 for Buy recommendation)
```

**MANDATORY: A Buy recommendation requires Risk/Reward Ratio >2.0**

### 9.9 Caveats
- DCF limitations (relies on projections, sensitive to terminal value)
- Data recency (when was FCF/earnings data from?)
- Company-specific risks
- Sensitivity to assumptions (see matrix for range of outcomes)

### 9.10 Data Sources & Traceability Section (MANDATORY)

**At the end of every valuation, include a Data Sources section with FULL TRACEABILITY:**

```
📋 DATA SOURCES & TRACEABILITY:

ANALYSIS METADATA:
- Analysis Date: [today's date, e.g., February 4, 2026]
- Snapshot ID: [unique identifier, e.g., MSFT-DCF-2026-02-04-v1]
- Analyst: AI Assistant (HealerAgent)

DATA FRESHNESS:
| Data Type | Source | Period/Date | API Endpoint |
|-----------|--------|-------------|--------------|
| Financial Statements | FMP API | FY2024 (10-K filed Jan 2025) | /v3/cash-flow-statement/MSFT |
| Balance Sheet | FMP API | FY2024 Q4 | /v3/balance-sheet-statement/MSFT |
| Current Stock Price | FMP Real-time | Feb 4, 2026 09:35 UTC | /v3/quote/MSFT |
| Beta | FMP Company Profile | 5Y Monthly vs S&P500 | /v3/profile/MSFT |
| Peer Multiples | FMP Key Metrics | TTM as of Feb 2026 | /v3/ratios/[PEER] |
| Treasury Rate | FMP Economic | Feb 3, 2026 | /v3/treasury |

FISCAL YEAR MAPPING:
- Company fiscal year ends: June 30
- FY2024 = July 2023 - June 2024
- Most recent quarter: Q2 FY2025 (Oct-Dec 2024)

Notes:
- All figures in USD millions unless otherwise stated
- TTM = Trailing Twelve Months (last 4 quarters)
- Forward estimates based on analyst consensus where applicable
```

**TRACEABILITY REQUIREMENTS:**
1. **Snapshot ID**: Every analysis must have a unique identifier for reproducibility
2. **Fiscal Year Clarity**: Always clarify fiscal vs calendar year (especially for MSFT, AAPL, etc.)
3. **Data Timestamp**: Stock prices must include time (markets move intraday)
4. **API Endpoints**: List which FMP endpoints were used (aids debugging)

**Language:** Match the user's language throughout.

---

## Data Citation Requirements (APPLIES TO ENTIRE ANALYSIS)

**CRITICAL: Every number in your output MUST have a source citation.**

### Citation Format by Data Type:

| Data Type | Citation Example |
|-----------|------------------|
| Financial metrics | "P/E = 25.6x (Source: FMP API, TTM as of Feb 2026)" |
| FCF | "FCF = $59,000M (Source: FMP Cash Flow Statement, FY2024)" |
| WACC components | "Beta = 0.9 (Source: FMP Company Profile)" |
| Growth rates | "5Y Revenue CAGR = 12% (Source: FMP Growth Metrics)" |
| Peer multiples | "AAPL P/E = 33.7x (Source: FMP Key Metrics, TTM basis)" |
| Stock price | "Current price = $411 (Source: FMP Quote, Feb 4, 2026, 09:35 UTC)" |

### Enhanced Traceability (NEW):

**Every valuation MUST include:**
1. **Snapshot ID**: Unique identifier (e.g., `MSFT-DCF-2026-02-04-v1`)
2. **Fiscal Year**: Clearly state which fiscal year data comes from (FY2024 = Jul 2023 - Jun 2024 for MSFT)
3. **Price Timestamp**: Include time for stock prices (markets move intraday)
4. **API Endpoints**: Reference which FMP endpoints were called

### Inline Citation Examples:

**GOOD (with source):**
> FCF của MSFT năm 2024 là **$59,000M** (Source: FMP Cash Flow Statement, FY2024),
> giảm 8% so với năm trước do tăng đầu tư AI datacenter.

**BAD (no source):**
> FCF của MSFT là khoảng $59B, giảm nhẹ so với năm trước.

### Numbers That MUST Be Cited:
1. ✅ FCF (current and historical)
2. ✅ Growth rates (revenue, FCF, earnings)
3. ✅ WACC components (Rf, Beta, ERP)
4. ✅ All multiples (P/E, P/B, P/S, EV/EBITDA)
5. ✅ Debt and cash positions
6. ✅ Shares outstanding
7. ✅ Current stock price

**NEVER output a number without indicating where it came from. A professional valuation is reproducible.**
