# Market Scanner - Technical Documentation

## Tài liệu Kỹ thuật: Hệ thống Phân tích Thị trường 6 Bước

**Version:** 2.0
**Last Updated:** 2026-01-22
**Author:** Senior Financial Technical Analyst

---

## 📋 Mục lục

1. [Tổng quan Kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Step 1: Technical Analysis](#2-step-1-technical-analysis)
3. [Step 2: Market Position (Relative Strength)](#3-step-2-market-position-relative-strength)
4. [Step 3: Risk Analysis](#4-step-3-risk-analysis)
5. [Step 4: Sentiment & News Analysis](#5-step-4-sentiment--news-analysis)
6. [Step 5: Fundamental Analysis](#6-step-5-fundamental-analysis)
7. [Step 6: Synthesis Report](#7-step-6-synthesis-report)
8. [Hệ thống Scoring](#8-hệ-thống-scoring)
9. [Caching Strategy](#9-caching-strategy)
10. [Pipeline V3 Architecture](#10-pipeline-v3-architecture)

---

## 1. Tổng quan Kiến trúc

### 1.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MARKET SCANNER PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Request (symbol: "NVDA")                                             │
│          │                                                                  │
│          ▼                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐     │
│   │  PARALLEL EXECUTION (Steps 1-5)                                   │     │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────┐ │     │
│   │  │Technical│ │Position │ │  Risk   │ │Sentiment│ │ Fundamental │ │     │
│   │  │Analysis │ │Analysis │ │Analysis │ │Analysis │ │  Analysis   │ │     │
│   │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──────┬──────┘ │     │
│   │       │           │           │           │              │        │     │
│   │       └───────────┴───────────┴───────────┴──────────────┘        │     │
│   │                               │                                   │     │
│   │                       ┌───────▼───────┐                           │     │
│   │                       │  Redis Cache  │                           │     │
│   │                       │ (per step TTL)│                           │     │
│   │                       └───────┬───────┘                           │     │
│   └───────────────────────────────┼──────────────────────────────────┘     │
│                                   │                                         │
│                           ┌───────▼───────┐                                 │
│                           │   SYNTHESIS   │                                 │
│                           │    (Step 6)   │                                 │
│                           └───────┬───────┘                                 │
│                                   │                                         │
│   ┌───────────────────────────────┼──────────────────────────────────┐     │
│   │  Pipeline V3 Layers           │                                   │     │
│   │  ┌────────────────────────────▼────────────────────────────────┐ │     │
│   │  │ Layer 1: Canonical Data Builder                             │ │     │
│   │  │ (Single Source of Truth for all metrics)                    │ │     │
│   │  └────────────────────────────┬────────────────────────────────┘ │     │
│   │                               │                                   │     │
│   │  ┌────────────────────────────▼────────────────────────────────┐ │     │
│   │  │ Layer 2: LLM Generation                                     │ │     │
│   │  │ (Canonical data in prompt for consistency)                  │ │     │
│   │  └────────────────────────────┬────────────────────────────────┘ │     │
│   │                               │                                   │     │
│   │  ┌────────────────────────────▼────────────────────────────────┐ │     │
│   │  │ Layer 3: Report Linter                                      │ │     │
│   │  │ (Deterministic validation)                                  │ │     │
│   │  └────────────────────────────┬────────────────────────────────┘ │     │
│   │                               │                                   │     │
│   │  ┌────────────────────────────▼────────────────────────────────┐ │     │
│   │  │ Layer 4: Targeted Repair                                    │ │     │
│   │  │ (Auto-fix if Critical/High issues found)                    │ │     │
│   │  └────────────────────────────┬────────────────────────────────┘ │     │
│   └───────────────────────────────┼──────────────────────────────────┘     │
│                                   │                                         │
│                           ┌───────▼───────┐                                 │
│                           │  FINAL REPORT │                                 │
│                           │  (Streaming)  │                                 │
│                           └───────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Nguyên tắc Thiết kế

| Nguyên tắc | Mô tả |
|------------|-------|
| **LLM Summary First** | Sử dụng `llm_summary` từ tool làm nguồn chính (đã tối ưu cho LLM) |
| **Raw Data for Audit** | Giữ `raw_data` để kiểm tra/xác minh, không gửi trực tiếp đến LLM |
| **No Derived Logic** | Tránh logic suy diễn có thể xung đột với kết luận của tool |
| **Facts Hierarchy** | Prompt có thứ bậc rõ ràng: Primary → Secondary → Derived |
| **Binding Scoring** | LLM phải tuân theo điểm số đã tính toán (không được tự ý thay đổi) |

---

## 2. Step 1: Technical Analysis

### 2.1 Mục đích

Phân tích kỹ thuật toàn diện bao gồm xu hướng, momentum, volatility, volume và các mức hỗ trợ/kháng cự.

### 2.2 API Endpoint

```
POST /scanner/technical/stream
```

### 2.3 Các Indicators & Thuật toán

#### 2.3.1 Trend Indicators

| Indicator | Công thức | Ý nghĩa |
|-----------|-----------|---------|
| **SMA (Simple Moving Average)** | `SMA(n) = Σ(Close_i) / n` | Giá trung bình n phiên gần nhất |
| **EMA (Exponential MA)** | `EMA = Close × k + EMA_prev × (1-k)` <br> `k = 2/(n+1)` | MA có trọng số cao hơn cho giá gần nhất |
| **ADX (Average Directional Index)** | Đo lường **sức mạnh** xu hướng (không phải hướng) | 0-20: Không có xu hướng <br> 20-40: Xu hướng phát triển <br> >40: Xu hướng mạnh |

**Giải thích ADX:**
- ADX KHÔNG cho biết xu hướng TĂNG hay GIẢM
- ADX chỉ cho biết xu hướng MẠNH hay YẾU
- Để biết hướng: dùng +DI và -DI
  - +DI > -DI: Xu hướng tăng
  - -DI > +DI: Xu hướng giảm

#### 2.3.2 Momentum Indicators

| Indicator | Công thức | Giải thích |
|-----------|-----------|------------|
| **RSI (Relative Strength Index)** | `RSI = 100 - (100 / (1 + RS))` <br> `RS = Avg Gain / Avg Loss (14 periods)` | >70: Overbought (có thể điều chỉnh) <br> <30: Oversold (có thể phục hồi) <br> 40-60: Vùng trung lập |
| **MACD** | `MACD Line = EMA12 - EMA26` <br> `Signal = EMA9(MACD)` <br> `Histogram = MACD - Signal` | MACD > Signal: Momentum tăng <br> Histogram mở rộng: Momentum đang tăng tốc |
| **Stochastic** | `%K = (Close - Low14) / (High14 - Low14) × 100` <br> `%D = SMA3(%K)` | >80: Overbought <br> <20: Oversold |

**Cách đọc RSI:**
```
RSI = 45.2 → Neutral (không overbought, không oversold)
RSI đang tăng từ 35 → 45 → Momentum đang cải thiện
RSI giảm từ 65 → 55 → Momentum đang suy yếu
```

#### 2.3.3 Volatility Indicators

| Indicator | Công thức | Ứng dụng |
|-----------|-----------|----------|
| **ATR (Average True Range)** | `TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)` <br> `ATR = SMA14(TR)` | Đo độ biến động trung bình hàng ngày <br> Dùng để tính stop-loss: `Stop = Entry - 2×ATR` |
| **ATR%** | `ATR% = (ATR / Price) × 100` | <2%: Biến động thấp <br> 2-4%: Biến động trung bình <br> >4%: Biến động cao |
| **Bollinger Bands** | `Middle = SMA20` <br> `Upper = SMA20 + 2×StdDev` <br> `Lower = SMA20 - 2×StdDev` | Bandwidth < 5%: Squeeze (sắp breakout) <br> Bandwidth > 10%: Volatility cao |

**Ví dụ ATR-based Stop Loss:**
```
Entry Price: $186.00
ATR: $5.20 (2.8% của giá)
Stop Loss (2x ATR): $186 - (2 × $5.20) = $175.60
Risk per share: $10.40 (5.6%)
```

#### 2.3.4 Volume Indicators

| Indicator | Công thức | Ngưỡng quan trọng |
|-----------|-----------|-------------------|
| **RVOL (Relative Volume)** | `RVOL = Current Volume / SMA20(Volume)` | <0.8x: Volume thấp (breakout có thể fail) <br> **≥1.2x: Minimum cho breakout confirmation** <br> ≥1.5x: Ideal confirmation <br> ≥2x: High conviction |
| **OBV (On-Balance Volume)** | `OBV += Volume (if Close > Close_prev)` <br> `OBV -= Volume (if Close < Close_prev)` | OBV tăng + Price tăng: Accumulation <br> OBV giảm + Price tăng: Distribution (warning!) |
| **AVWAP (Anchored VWAP)** | `AVWAP = Σ(Price × Volume) / Σ(Volume)` từ anchor date | Price > AVWAP: Buyers có lãi → Bullish <br> Price < AVWAP: Buyers lỗ → Bearish |

### 2.4 Output Structure

```markdown
### 1. SNAPSHOT
- Symbol: NVDA - Daily chart, 1Y data
- Current Price: $186.23
- Data as of: 2026-01-21

### 2. TL;DR
BUY - Xu hướng tăng với momentum đang cải thiện

### 3. Trend Analysis
- Price vs SMA20: $186.23 vs $180.45 (+3.2%)
- Price vs SMA50: $186.23 vs $175.30 (+6.2%)
- Price vs SMA200: $186.23 vs $145.00 (+28.4%) ✓
- ADX: 32.5 → Xu hướng đang phát triển

### 4. Momentum
- RSI(14): 58.3 → Neutral, trending up
- MACD: Line=2.15, Signal=1.85, Histogram=+0.30 (expanding)

### 5. Volatility & Volume
- ATR: $5.20 (2.8%)
- RVOL: 1.35x → Volume confirmation ✓
- BB Width: 8.2% → Normal volatility

### 6. Key Levels
- Support Zone: $175 - $178
- Resistance Zone: $190 - $195

### 7. Trading Plan
Entry: Wait for pullback to $180 (SMA20)
Stop: $169 (2×ATR below SMA50)
Target: $200 (previous high)
```

---

## 3. Step 2: Market Position (Relative Strength)

### 3.1 Mục đích

Đánh giá sức mạnh tương đối của cổ phiếu so với benchmark (SPY) và trong ngữ cảnh sector.

### 3.2 API Endpoint

```
POST /scanner/position/stream
```

### 3.3 Thuật toán Relative Strength

#### 3.3.1 Công thức cơ bản

```
Excess Return = Stock Return - Benchmark Return (%)
RS Score = 50 + Excess Return (capped 1-99)
```

**Ví dụ:**
```
NVDA 21-day return: +8.94%
SPY 21-day return: +3.02%
Excess Return: +5.92 percentage points
RS Score: 50 + 5.92 = 55.92 → OUTPERFORM
```

#### 3.3.2 Multi-timeframe Analysis

| Timeframe | Trading Days | Ý nghĩa |
|-----------|--------------|---------|
| **21d** | ~1 tháng | Short-term momentum |
| **63d** | ~1 quý | Medium-term strength |
| **126d** | ~6 tháng | Intermediate trend |
| **252d** | ~1 năm | Long-term trend |

#### 3.3.3 Classification Rules

| Classification | Điều kiện | Action |
|----------------|-----------|--------|
| **LEADER** ✅ | Outperform (>1% excess) trong ≥3 timeframes <br> RS Score > 55 đa số timeframes | High conviction long |
| **EMERGING/ROTATION** | 21d outperforming nhưng 63d/126d chưa confirm | Watchlist only |
| **NEUTRAL** | Mixed signals hoặc excess ≈ 0% | No RS edge |
| **LAGGARD** ⚠️ | Underperform (<-1% excess) trong ≥3 timeframes | Avoid for longs |

#### 3.3.4 Sector Context

**Lưu ý quan trọng:**
- Sector data là **1-day change**
- RS data là **multi-timeframe** (21d/63d/126d)
- **KHÔNG thể so sánh trực tiếp** - chúng đo lường khác nhau!

| Sector Rank | Status |
|-------------|--------|
| #1-3 | LEADING |
| #4-8 | NEUTRAL |
| #9-11 | LAGGING |

**Combined Analysis Matrix:**

| Stock RS | Sector (1-Day) | Kết luận | Confidence |
|----------|----------------|----------|------------|
| OUTPERFORM | Top 3 | Aligned - HIGH CONVICTION | Higher |
| OUTPERFORM | Bottom 3 | Conflicting - stock strong but sector weak | Lower |
| UNDERPERFORM | Top 3 | Sector tailwind may help | Medium |
| UNDERPERFORM | Bottom 3 | Both weak - AVOID | Higher |

### 3.4 Tại sao so sánh với SPY?

**SPY (SPDR S&P 500 ETF):**
- Đại diện cho "overall US market"
- Benchmark chuẩn cho fund managers
- Loại bỏ market noise để thấy sức mạnh riêng của stock

**Alpha = Stock Return - SPY Return**
- Nếu stock +10% nhưng SPY +15% → bạn đang underperform!
- RS leaders tend to continue outperforming (momentum effect)

---

## 4. Step 3: Risk Analysis

### 4.1 Mục đích

Đánh giá rủi ro và cung cấp các mức stop-loss, position sizing guidance.

### 4.2 API Endpoint

```
POST /scanner/risk/stream
```

### 4.3 Các Metrics & Thuật toán

#### 4.3.1 Three Different Risk Metrics (CRITICAL DISTINCTION)

| Metric | Đo lường | Nguồn | Ứng dụng |
|--------|----------|-------|----------|
| **ATR** | Typical daily price movement | Historical high-low-close | Stop-loss sizing |
| **VaR (Value at Risk)** | Tail risk (worst X% of days) | Statistical returns distribution | Extreme scenario planning |
| **Annual Volatility** | Overall price fluctuation | Standard deviation of returns | Risk regime classification |

**KHÔNG ĐƯỢC NHẦM LẪN:**
```
ATR $5.00 (2.5%) = "Stock TYPICALLY moves $5/day"
VaR -4.5% = "5% chance of losing MORE than 4.5% in a single day"
Annual Vol 45% = "Overall HIGH volatility stock"
```

#### 4.3.2 Stop Loss Methods

| Method | Công thức | Khi nào dùng |
|--------|-----------|--------------|
| **ATR 1x** | `Entry - 1×ATR` | Aggressive (tight stop) |
| **ATR 2x** | `Entry - 2×ATR` | **Conservative (recommended)** |
| **ATR 3x** | `Entry - 3×ATR` | Wide stop (for volatile stocks) |
| **5% Rule** | `Entry × 0.95` | Simple percentage |
| **7% Rule** | `Entry × 0.93` | Aggressive percentage |
| **Structure-based** | Below key support level | Technical stop |

**Ví dụ ATR Stop Calculation:**
```
Entry Price: $186.23
ATR: $4.86 (2.61%)

ATR 1x Stop: $186.23 - $4.86 = $181.37 (2.61% risk)
ATR 2x Stop: $186.23 - $9.72 = $176.51 (5.22% risk) ← Recommended
ATR 3x Stop: $186.23 - $14.58 = $171.65 (7.83% risk)
```

#### 4.3.3 VaR Calculation

```python
# Parametric VaR (95% confidence)
VaR_95 = μ - 1.645 × σ

# Where:
# μ = mean daily return
# σ = standard deviation of daily returns
# 1.645 = z-score for 95% confidence
```

**Interpretation:**
```
VaR_95 = -4.59%
→ Có 5% khả năng mất HƠN 4.59% trong 1 ngày
→ Trong 20 ngày giao dịch (1 tháng), có thể xảy ra 1 ngày như vậy
```

#### 4.3.4 Position Sizing Formula

```
Position Size = (Account Risk $) / (Stop Distance $)

Example:
- Account: $100,000
- Risk per trade: 2% = $2,000
- Entry: $186.23
- Stop (2x ATR): $176.51
- Stop Distance: $9.72

Position Size = $2,000 / $9.72 = 205 shares
Position Value = 205 × $186.23 = $38,177 (38% of account)
```

#### 4.3.5 Max Drawdown

```python
# Rolling Max Drawdown
Peak = max(Price_history up to date)
Drawdown = (Current_Price - Peak) / Peak × 100
Max_Drawdown = min(all Drawdowns)
```

**Interpretation:**
```
Max Drawdown = -25%
→ Worst peak-to-trough decline historically
→ If you bought at peak, you would have seen -25% paper loss
```

### 4.4 Volatility Regime Classification

| Annual Volatility | Regime | Implication |
|-------------------|--------|-------------|
| <15% | LOW | Tight stops OK, larger position size |
| 15-30% | NORMAL | Standard ATR-based stops |
| 30-50% | HIGH | Wider stops, smaller position |
| >50% | EXTREME | Very wide stops, reduce position significantly |

---

## 5. Step 4: Sentiment & News Analysis

### 5.1 Mục đích

Phân tích sentiment từ social media và tin tức gần đây để đánh giá tâm lý thị trường.

### 5.2 API Endpoint

```
POST /scanner/sentiment/stream
```

### 5.3 Sentiment Scoring

#### 5.3.1 Sentiment Score Scale

| Score Range | Label | Interpretation |
|-------------|-------|----------------|
| > +0.3 | Strong Bullish | Tâm lý rất tích cực |
| +0.1 to +0.3 | Moderate Bullish | Tâm lý tích cực vừa phải |
| -0.1 to +0.1 | Neutral | "Wait and see" |
| -0.3 to -0.1 | Moderate Bearish | Tâm lý tiêu cực vừa phải |
| < -0.3 | Strong Bearish | Tâm lý rất tiêu cực |

#### 5.3.2 Sample Size Validation

```python
MIN_SENTIMENT_SAMPLE_SIZE = 5

if sample_size < MIN_SENTIMENT_SAMPLE_SIZE:
    sentiment = None  # Mark as invalid
    warning = "Insufficient data - do NOT display score"
```

**Quan trọng:** Nếu sample_size < 5, báo cáo **KHÔNG ĐƯỢC** hiển thị sentiment score!

#### 5.3.3 Sentiment Confidence Calculation

```python
def calculate_sentiment_confidence(sample_size, signal_strength):
    if sample_size >= 100:
        size_score = 70
    elif sample_size >= 50:
        size_score = 55
    elif sample_size >= 30:
        size_score = 40
    elif sample_size >= 5:
        size_score = 25
    else:
        return "INVALID"

    signal_bonus = min(30, signal_strength * 50) if sample_size >= 20 else 0
    total_score = size_score + signal_bonus

    if total_score >= 80: return "HIGH"
    elif total_score >= 50: return "MODERATE"
    elif total_score >= 30: return "LOW"
    else: return "VERY_LOW"
```

#### 5.3.4 News Classification

| Source Type | Examples | Weight |
|-------------|----------|--------|
| **Factual** | Reuters, Bloomberg, WSJ, AP, CNBC | Higher reliability |
| **Opinion/Analysis** | Seeking Alpha, Motley Fool | Lower reliability |
| **Press Release** | BusinessWire, PRNewswire | Company-biased |

### 5.4 News Theme Extraction

Tin tức được nhóm theo themes:
- **Earnings/Guidance** - Kết quả kinh doanh
- **Product/Technology** - Ra mắt sản phẩm mới
- **Competition** - Cạnh tranh
- **Regulatory** - Quy định pháp lý
- **M&A** - Mua bán sáp nhập
- **Management** - Thay đổi lãnh đạo

---

## 6. Step 5: Fundamental Analysis

### 6.1 Mục đích

Phân tích cơ bản toàn diện bao gồm định giá, tăng trưởng, lợi nhuận, và sức khỏe tài chính.

### 6.2 API Endpoint

```
POST /scanner/fundamental/stream
```

### 6.3 Các Metrics & Thuật toán

#### 6.3.1 Valuation Metrics

| Metric | Công thức | Interpretation |
|--------|-----------|----------------|
| **P/E TTM** | `Price / EPS_TTM` | <15: Value <br> 15-25: Fair <br> >25: Growth premium |
| **P/E Forward** | `Price / EPS_Forward` | Analyst estimates |
| **P/S** | `Market Cap / Revenue` | <1: Potentially cheap <br> >5: Expensive |
| **P/B** | `Price / Book Value per Share` | <1: Below asset value |
| **EV/EBITDA** | `Enterprise Value / EBITDA` | Industry-specific benchmarks |
| **PEG** | `P/E / EPS Growth Rate` | <1: Undervalued given growth |

**P/E FY vs P/E TTM Distinction:**
```
P/E FY: Uses EPS from latest fiscal year (e.g., FY ending Jan 2025)
        → May not include recent quarters

P/E TTM: Uses trailing 12-month EPS (rolling sum of last 4 quarters)
         → More current, matches Yahoo Finance
         → Preferred for comparison
```

#### 6.3.2 Profitability Metrics

| Metric | Công thức | Good Threshold |
|--------|-----------|----------------|
| **Gross Margin** | `(Revenue - COGS) / Revenue × 100` | >40% for tech |
| **Operating Margin** | `Operating Income / Revenue × 100` | >20% |
| **Net Margin** | `Net Income / Revenue × 100` | >15% |
| **ROE** | `Net Income / Shareholders Equity × 100` | >15% |
| **ROA** | `Net Income / Total Assets × 100` | >10% |
| **ROIC** | `NOPAT / Invested Capital × 100` | > WACC |

**Percentage Normalization:**
```python
# Different APIs return different formats:
# FMP: ROE = 0.31 means 31%
# Yahoo: ROE = 31 means 31%

def normalize_percentage(value):
    if abs(value) < 1:
        return value * 100  # Convert decimal to %
    else:
        return value  # Already in % form
```

#### 6.3.3 Growth Metrics

| Metric | Công thức |
|--------|-----------|
| **Revenue Growth YoY** | `(Revenue_current - Revenue_prev) / Revenue_prev × 100` |
| **EPS Growth YoY** | `(EPS_current - EPS_prev) / EPS_prev × 100` |
| **Revenue CAGR 5Y** | `(Revenue_latest / Revenue_5y_ago)^(1/5) - 1` |
| **EPS CAGR 5Y** | `(EPS_latest / EPS_5y_ago)^(1/5) - 1` |

#### 6.3.4 Intrinsic Value Calculations

##### Graham Formula

```python
# Benjamin Graham's Intrinsic Value Formula
V = EPS × (8.5 + 2g) × (4.4 / Y)

# Where:
# EPS = Earnings per share (TTM)
# g = Expected growth rate (5-year)
# Y = Current AAA corporate bond yield
# 8.5 = P/E for no-growth company
# 4.4 = Average AAA bond yield when Graham wrote

# Example:
EPS = $10.50
g = 12% (0.12)
Y = 4.4%

V = 10.50 × (8.5 + 2×12) × (4.4/4.4)
V = 10.50 × 32.5 × 1.0
V = $341.25
```

##### DCF (Discounted Cash Flow)

```python
# DCF Formula
Intrinsic Value = Σ(FCF_t / (1 + WACC)^t) + Terminal Value / (1 + WACC)^n

# Terminal Value
TV = FCF_n × (1 + g) / (WACC - g)

# Where:
# FCF_t = Free Cash Flow at year t
# WACC = Weighted Average Cost of Capital
# g = Terminal growth rate (typically 2-3%)
# n = Projection period (typically 5 years)

# WACC Calculation (simplified CAPM):
WACC = Rf + β × (Rm - Rf)

# Where:
# Rf = Risk-free rate (10-year Treasury ~4.5%)
# β = Stock beta
# Rm - Rf = Market risk premium (~5.5%)
```

**DCF Sensitivity Analysis:**
```
DCF value is HIGHLY sensitive to assumptions:
- WACC ±1% can change value by 20-30%
- Terminal growth ±0.5% can change value by 10-15%

Always show sensitivity grid:
         | TG 2.0% | TG 2.5% | TG 3.0% |
WACC 9%  | $280    | $310    | $345    |
WACC 10% | $250    | $275    | $305    |
WACC 11% | $225    | $245    | $270    |
```

#### 6.3.5 Valuation Verdict Logic

```python
def determine_verdict(current_price, graham_value, dcf_value):
    values = [v for v in [graham_value, dcf_value] if v and v > 0]
    if not values:
        return "insufficient_data"

    avg_intrinsic = sum(values) / len(values)
    upside_pct = ((avg_intrinsic - current_price) / current_price) * 100

    if upside_pct > 30:
        return "significantly_undervalued"
    elif upside_pct > 15:
        return "undervalued"
    elif upside_pct > -10:
        return "fairly_valued"
    elif upside_pct > -25:
        return "overvalued"
    else:
        return "significantly_overvalued"
```

---

## 7. Step 6: Synthesis Report

### 7.1 Mục đích

Tổng hợp 5 bước phân tích thành báo cáo đầu tư toàn diện với khuyến nghị rõ ràng.

### 7.2 API Endpoints

```
POST /scanner/synthesis/stream      # V1: Multiple LLM calls
POST /scanner/synthesis/v2/stream   # V2: Single LLM call (Recommended)
```

### 7.3 V2 Architecture (Single LLM Call)

**Key Improvements:**
1. **Single LLM Call** - All data in one context → 100% consistency
2. **Binding Scoring** - LLM MUST follow pre-calculated score
3. **Raw Data Metrics** - Uses structured data, not truncated LLM text
4. **Pipeline V3** - Canonical data + Linting + Auto-repair

### 7.4 Synthesis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYNTHESIS V2 PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Gather Data                                           │
│  ─────────────────────                                          │
│  • Check cache for all 5 steps                                  │
│  • Run missing steps if run_missing_steps=True                  │
│  • Minimum 2 steps required                                     │
│                                                                 │
│  Phase 2: Calculate Scoring (BINDING)                           │
│  ─────────────────────────────────────                          │
│  • ScoringService.calculate_composite_score()                   │
│  • LLM CANNOT override this score                               │
│                                                                 │
│  Phase 3: Fetch Enrichment Data (Parallel)                      │
│  ─────────────────────────────────────────                      │
│  • Earnings Calendar (FMP API)                                  │
│  • Peer Comparison (FMP stock_peers API)                        │
│  • Analyst Consensus (FMP API)                                  │
│  • Insider Trading (SEC Form 4)                                 │
│  • Seasonal Analysis                                            │
│  • Web Search (optional - for news enrichment)                  │
│                                                                 │
│  Phase 4: Build Canonical Data (Pipeline V3 Layer 1)            │
│  ─────────────────────────────────────────────────              │
│  • Single source of truth for all metrics                       │
│  • Cross-validate data from all sources                         │
│  • Resolve conflicts with priority rules                        │
│  • Calculate data quality score (0-100)                         │
│                                                                 │
│  Phase 5: Generate Report (Pipeline V3 Layer 2)                 │
│  ─────────────────────────────────────────────                  │
│  • Single consolidated prompt with canonical data               │
│  • Include trading plan (pre-calculated)                        │
│  • Include scenario analysis                                    │
│  • Stream response via SSE                                      │
│                                                                 │
│  Phase 6: Lint Report (Pipeline V3 Layer 3)                     │
│  ──────────────────────────────────────────                     │
│  • Check metric consistency with canonical data                 │
│  • Check sentiment validity (sample size)                       │
│  • Check price consistency                                      │
│  • Check required sections                                      │
│  • Check stop-loss has calculation                              │
│                                                                 │
│  Phase 7: Repair if Needed (Pipeline V3 Layer 4)                │
│  ───────────────────────────────────────────────                │
│  • If Critical/High issues found → auto-repair                  │
│  • Re-lint to verify repairs                                    │
│  • Output repaired content                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.5 Report Structure

```markdown
## PART A: RAW DATA ONLY (No Interpretation)
1. Technical Indicators - RSI, MACD, ADX, MAs
2. Market Position - RS scores, Sector context
3. Risk Metrics - ATR, VaR, Volatility
4. Sentiment Data - Score, Sample size
5. Fundamental Metrics - P/E, P/S, ROE, etc.

## PART B: INTERPRETATION & ANALYSIS
6. Growth Investor Perspective
7. Value/Dividend Investor Perspective
8. Fair Value Assessment (Graham/DCF with assumptions)
9. Scenario Analysis (Bull/Base/Bear)

## PART C: EXTERNAL DATA
10. Analyst Consensus
11. Insider Trading
12. News & Catalysts (with inline citations)

## PART D: ACTION PLAN
13. Executive Summary
14. Price Levels with Methodology
    - NEW INVESTORS: Entry conditions
    - EXISTING HOLDERS: Stop/reduce triggers
```

---

## 8. Hệ thống Scoring

### 8.1 Component Weights

| Component | Weight | Factors |
|-----------|--------|---------|
| **Fundamental** | 30% | P/E, Growth, Profitability, Debt |
| **Technical** | 20% | Trend, Momentum, Volume |
| **Risk** | 20% | Volatility, Max DD, Stop quality |
| **Position** | 15% | Relative Strength vs market |
| **Sentiment** | 15% | News tone, Social sentiment |

### 8.2 Scoring Scale

| Score | Recommendation | Action |
|-------|----------------|--------|
| 80-100 | 🟢🟢 STRONG BUY | High conviction long |
| 65-79 | 🟢 BUY | Long with normal position |
| 45-64 | 🟡 HOLD | No action / watchlist |
| 30-44 | 🔴 SELL | Reduce exposure |
| 0-29 | 🔴🔴 STRONG SELL | Exit position |

### 8.3 Scoring Algorithm

```python
def calculate_composite_score(step_data):
    component_scores = {
        "fundamental": score_fundamental(step_data.get("fundamental")),
        "technical": score_technical(step_data.get("technical")),
        "risk": score_risk(step_data.get("risk")),
        "position": score_position(step_data.get("position")),
        "sentiment": score_sentiment(step_data.get("sentiment")),
    }

    # Weighted average
    total_score = sum(
        cs["score"] * cs["weight"]
        for cs in component_scores.values()
    )

    # Calculate confidence
    confidence = calculate_confidence(component_scores)

    # Distribution (BUY/HOLD/SELL percentages)
    buy_pct, hold_pct, sell_pct = calculate_distribution(total_score, confidence)

    return {
        "composite_score": total_score,
        "recommendation": get_recommendation(total_score),
        "distribution": {"buy": buy_pct, "hold": hold_pct, "sell": sell_pct},
        "confidence": confidence,
        "components": component_scores
    }
```

### 8.4 Example Scoring

```
NVDA Composite Score: 72.5 → BUY

Component Breakdown:
┌─────────────┬───────┬────────┬─────────────────────────────────┐
│ Component   │ Score │ Weight │ Key Signals                     │
├─────────────┼───────┼────────┼─────────────────────────────────┤
│ Fundamental │ 75    │ 30%    │ +Growth mạnh, +Margin tốt       │
│ Technical   │ 68    │ 20%    │ +Uptrend, +MACD bullish         │
│ Risk        │ 65    │ 20%    │ ~Volatility trung bình          │
│ Position    │ 80    │ 15%    │ +Outperform SPY, +Sector leader │
│ Sentiment   │ 70    │ 15%    │ +Bullish sentiment              │
└─────────────┴───────┴────────┴─────────────────────────────────┘

Distribution: Buy 65% | Hold 25% | Sell 10%
Confidence: 78%
```

---

## 9. Caching Strategy

### 9.1 Cache Configuration

| Step | TTL | Lý do |
|------|-----|-------|
| **Technical** | 180s (3 min) | Price data changes frequently |
| **Position** | 300s (5 min) | RS vs benchmark |
| **Risk** | 300s (5 min) | Stop loss tied to price |
| **Sentiment** | 600s (10 min) | News less volatile |
| **Fundamental** | 900s (15 min) | Fundamentals rarely change intraday |

### 9.2 Cache Key Format

```
scanner:{SYMBOL}:{step_name}:{time_bucket}

# Time bucket = 3-minute intervals
time_bucket = int(timestamp) // 180

# Example:
scanner:NVDA:technical:9876543
```

### 9.3 Freshness Labels

```python
def get_freshness_label(age_seconds, step_name):
    ttl = STEP_TTL[step_name]

    if age_seconds < ttl * 0.5:
        return "fresh"      # Data is recent
    elif age_seconds < ttl:
        return "stale"      # Still valid but aging
    else:
        return "expired"    # Needs refresh
```

---

## 10. Pipeline V3 Architecture

### 10.1 Four Layers

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 1: CANONICAL DATA BUILDER                                 │
│ ─────────────────────────────────                               │
│ Purpose: Single source of truth for ALL metrics                 │
│                                                                 │
│ • Extracts metrics from all 5 steps                             │
│ • Cross-validates between sources                               │
│ • Resolves conflicts with priority rules                        │
│ • Calculates data quality score (0-100)                         │
│ • Generates warnings for inconsistencies                        │
│                                                                 │
│ Output: canonical_data dict with verified values                │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: LLM GENERATION                                         │
│ ──────────────────────                                          │
│ Purpose: Generate report with canonical data in context         │
│                                                                 │
│ • Canonical data included in prompt                             │
│ • LLM MUST use exact canonical values                           │
│ • Binding scoring (cannot override pre-calculated score)        │
│ • Single consolidated prompt                                    │
│                                                                 │
│ Output: Generated report text                                   │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: REPORT LINTER                                          │
│ ─────────────────────                                           │
│ Purpose: Deterministic validation of generated report           │
│                                                                 │
│ Checks:                                                         │
│ • Metric consistency (report values match canonical)            │
│ • Sentiment validity (not shown if sample < 5)                  │
│ • Price consistency (within 2% variance)                        │
│ • Required sections present                                     │
│ • Stop-loss has calculation formula                             │
│                                                                 │
│ Issue Severity:                                                 │
│ • CRITICAL: Must fix before output                              │
│ • HIGH: Should fix, affects quality                             │
│ • MEDIUM: Nice to fix                                           │
│ • LOW: Informational                                            │
│                                                                 │
│ Output: List of issues with fix instructions                    │
├─────────────────────────────────────────────────────────────────┤
│ Layer 4: TARGETED REPAIR                                        │
│ ─────────────────────────                                       │
│ Purpose: Auto-fix Critical/High issues                          │
│                                                                 │
│ Trigger: needs_repair = (CRITICAL > 0) OR (HIGH >= 2)           │
│                                                                 │
│ • Sends issues + canonical data to LLM for repair               │
│ • Re-lints repaired report to verify                            │
│ • Outputs repaired content with notice                          │
│                                                                 │
│ Output: Repaired report (or original if no issues)              │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Canonical Data Structure

```python
canonical_data = {
    "symbol": "NVDA",
    "timestamp": "2026-01-22T10:30:00",

    "price": {
        "current": 186.23,
        "source": "risk_step",  # Primary source for price
        "timestamp": "2026-01-21T16:00:00"
    },

    "valuation": {
        "pe_ttm": 35.2,
        "pe_forward": 28.5,
        "ps_ttm": 18.3,
        "pb_ttm": 12.1,
        "ev_ebitda": 28.7,
        "source": "fundamental_step_TTM"
    },

    "profitability": {
        "gross_margin": 74.5,    # Already normalized to %
        "operating_margin": 62.3,
        "net_margin": 55.2,
        "roe": 89.4,
        "roa": 45.2,
        "source": "fundamental_step_TTM"
    },

    "technical": {
        "rsi": 58.3,
        "macd_line": 2.15,
        "macd_signal": 1.85,
        "macd_histogram": 0.30,
        "adx": 32.5,
        "sma_20": 180.45,
        "sma_50": 175.30,
        "sma_200": 145.00,
        "source": "technical_step"
    },

    "risk": {
        "atr_value": 4.86,
        "atr_percent": 2.61,
        "var_95": 4.59,
        "volatility_annual": 42.3,
        "max_drawdown": -18.5,
        "source": "risk_step"
    },

    "sentiment": {  # NULL if sample_size < 5
        "score": 0.25,
        "sample_size": 47,
        "label": "Moderate Bullish",
        "confidence": "MODERATE",
        "source": "sentiment_step"
    },

    "relative_strength": {
        "rs_21d": 5.92,
        "rs_63d": -2.46,
        "rs_126d": -1.34,
        "rs_rating": "ROTATION_CANDIDATE",
        "source": "position_step"
    },

    "intrinsic_value": {
        "graham_value": 245.50,
        "dcf_value": 280.00,
        "dcf_assumptions": {
            "wacc": "10%",
            "terminal_growth": "2.5%",
            "fcf_base": "$28.5B"
        },
        "source": "fundamental_step"
    },

    "_warnings": [
        "Price variance from technical: $186.50 vs avg $186.23 (0.1% diff)"
    ],
    "_data_quality_score": 87,
    "_sources_used": ["risk", "technical", "position", "sentiment", "fundamental"]
}
```

### 10.3 Linting Rules

| Rule | Pattern | Severity |
|------|---------|----------|
| ROE mismatch | `ROE: X%` ≠ canonical | HIGH if >5%, MEDIUM if >1% |
| Invalid sentiment | Score shown when sample < 5 | CRITICAL |
| Price mismatch | Price differs by >2% | HIGH |
| Missing section | Required section not found | MEDIUM |
| Stop-loss no calculation | Stop mentioned without formula | MEDIUM |

---

## Appendix: API Reference

### A1. Scanner Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/scanner/technical/stream` | POST | Technical Analysis |
| `/scanner/position/stream` | POST | Market Position (RS) |
| `/scanner/risk/stream` | POST | Risk Analysis |
| `/scanner/sentiment/stream` | POST | Sentiment & News |
| `/scanner/fundamental/stream` | POST | Fundamental Analysis |
| `/scanner/synthesis/stream` | POST | Synthesis V1 |
| `/scanner/synthesis/v2/stream` | POST | **Synthesis V2 (Recommended)** |
| `/scanner/cache/status` | POST | Check cache status |

### A2. Request Schema

```json
{
    "session_id": "optional-session-id",
    "symbol": "NVDA",
    "question_input": "Optional specific question",
    "target_language": "vi",
    "model_name": "gpt-4.1-nano-2025-04-14",
    "provider_type": "openai"
}
```

### A3. Response Events (SSE)

```json
// Progress event
{"type": "progress", "step": "scoring", "message": "Calculating..."}

// Content event (streaming text)
{"type": "content", "section": "report_body", "content": "## Technical Analysis\n..."}

// Data event (structured data)
{"type": "data", "section": "scoring", "data": {"composite_score": 72.5, ...}}

// Done event
{"type": "done"}

// Error event
{"type": "error", "error": "Error message"}
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-01-22 | Pipeline V3, Canonical Data Builder, Report Linter |
| 1.5 | 2025-12-15 | Single LLM call architecture (V2) |
| 1.0 | 2025-10-01 | Initial 5-step + synthesis architecture |
