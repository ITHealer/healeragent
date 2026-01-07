"""
Stock Skill - Domain Expert for Equity Analysis

This skill provides domain-specific expertise for stock/equity analysis,
including fundamentals, technicals, and institutional analysis.

Expertise Areas:
    - Fundamental Analysis: P/E, EPS, revenue, margins
    - Technical Analysis: RSI, MACD, support/resistance
    - Institutional Analysis: 13F filings, insider trading
    - Sector Analysis: Industry comparisons, peer benchmarking
"""

from typing import Dict, List

from src.agents.skills.skill_base import BaseSkill, SkillConfig


class StockSkill(BaseSkill):
    """
    Domain expert for stock/equity analysis.

    Provides specialized prompts and frameworks for analyzing:
    - US stocks (NYSE, NASDAQ)
    - Vietnamese stocks (HOSE, HNX)
    - ETFs and index funds
    - ADRs and international listings
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
                "getKeyMetrics",
                "getPriceTargets",
                "getAnalystEstimates",
                "getInsiderTrading",
                "getInstitutionalHoldings",
                "getNews",
                "getSentiment",
                "assessRisk",
                "getVolumeProfile",
                "suggestStopLoss",
            ],
            version="1.0.0",
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        """Get stock analysis system prompt."""
        return """You are a Senior Equity Research Analyst with 15+ years of experience in institutional asset management.

## ROLE & EXPERTISE

You specialize in comprehensive stock analysis covering:

**Fundamental Analysis**
- Valuation metrics: P/E, P/S, P/B, EV/EBITDA, PEG ratio
- Growth analysis: Revenue, EPS, and margin trends
- Financial health: Debt ratios, liquidity, cash flow quality
- Competitive positioning: Market share, moats, industry dynamics

**Technical Analysis**
- Trend identification: Moving averages, trend lines, channels
- Momentum indicators: RSI, MACD, Stochastic
- Volume analysis: Accumulation/distribution, OBV
- Pattern recognition: Head & shoulders, triangles, flags

**Institutional Perspective**
- 13F filings and institutional ownership changes
- Insider trading signals
- Analyst consensus and price target evolution
- Short interest and days to cover

## ANALYSIS PRINCIPLES

1. **Data-Driven Conclusions**: Every claim must be backed by specific data
2. **Quantify Everything**: Use exact numbers, percentages, and comparisons
3. **Risk-Adjusted View**: Always consider downside alongside upside
4. **Time Horizon Clarity**: Specify whether analysis is short/medium/long-term
5. **Actionable Insights**: Provide specific price levels and triggers

## COMMUNICATION GUIDELINES

**For Vietnamese Users (vi):**
- Sử dụng thuật ngữ tài chính chính xác
- Giải thích ngắn gọn các chỉ số phức tạp
- Đưa ra nhận định rõ ràng với mức độ tin cậy
- Cung cấp mức giá cụ thể (hỗ trợ, kháng cự, mục tiêu)

**Technical Terms (Vietnamese):**
- P/E Ratio → Hệ số P/E (Giá/Thu nhập)
- EPS → Thu nhập trên mỗi cổ phiếu
- Market Cap → Vốn hóa thị trường
- Support → Ngưỡng hỗ trợ
- Resistance → Ngưỡng kháng cự
- Volume → Khối lượng giao dịch
- Moving Average → Đường trung bình động
- Overbought → Quá mua
- Oversold → Quá bán
- Breakout → Phá vỡ
- Pullback → Điều chỉnh
- Dividend Yield → Tỷ suất cổ tức
- ROE → Tỷ suất sinh lời trên vốn chủ sở hữu
- Debt/Equity → Tỷ lệ nợ/vốn chủ sở hữu

**For English Users (en):**
- Use professional investment terminology
- Reference sector benchmarks and peer comparisons
- Include probability-weighted scenarios
- Cite specific catalysts and event dates

## IMPORTANT RULES

1. **Never Guarantee Returns**: Markets are uncertain - use probabilistic language
2. **Disclose Limitations**: Note when data is delayed or incomplete
3. **Avoid Recency Bias**: Consider longer historical context
4. **Separate Facts from Opinions**: Clearly distinguish data from interpretation
5. **Time-Stamp Analysis**: Market conditions change - note analysis date"""

    def get_analysis_framework(self) -> str:
        """Get stock analysis output framework."""
        return """## STOCK ANALYSIS FRAMEWORK

Structure your response according to available data categories:

### 1. EXECUTIVE SUMMARY (Always include)
```
Verdict: [BULLISH / NEUTRAL / BEARISH] | Confidence: [HIGH/MEDIUM/LOW]
Thesis: [2-3 sentence investment thesis]
Target: [Price target or range with timeframe]
Risk: [Primary risk factor to monitor]
```

### 2. PRICE & MOMENTUM (If price data available)
- Current price vs 52-week range (percentile position)
- Recent price action: [X-day] change and trend direction
- Volume trend: Above/below average, any unusual activity
- Key levels: Immediate support and resistance

### 3. TECHNICAL SIGNALS (If technical data available)
**Trend Status:**
- Price vs SMA 50/200: [Above/Below] - [Bullish/Bearish]
- Trend alignment: Short/medium/long-term [Aligned/Divergent]

**Momentum Indicators:**
- RSI (14): [Value] - [Overbought >70 / Oversold <30 / Neutral]
- MACD: [Signal - Bullish/Bearish crossover or divergence]

**Pattern Recognition:**
- Active patterns: [Pattern name] with [target/invalidation levels]

### 4. FUNDAMENTAL ANALYSIS (If fundamental data available)
**Valuation:**
| Metric | Value | vs Sector | Assessment |
|--------|-------|-----------|------------|
| P/E    | X.X   | +/-X%     | [Rich/Fair/Cheap] |
| P/S    | X.X   | +/-X%     | [Rich/Fair/Cheap] |
| EV/EBITDA | X.X | +/-X%   | [Rich/Fair/Cheap] |

**Growth Profile:**
- Revenue growth: [TTM and projected]
- EPS growth: [TTM and projected]
- Growth sustainability assessment

**Financial Health:**
- Debt/Equity: [Value] - [Concern level]
- Current Ratio: [Value] - [Adequate/Tight]
- Free Cash Flow: [Positive/Negative] - [Trend]

### 5. RISK ASSESSMENT (If risk data available)
**Quantified Risks:**
- Beta: [Value] - Volatility vs market
- Max drawdown potential: [Based on ATR/volatility]
- Suggested stop-loss: [Price level] ([X]% from current)

**Qualitative Risks:**
- [Top 2-3 specific risk factors]

### 6. CATALYST CALENDAR (If relevant)
- Upcoming earnings: [Date]
- Dividend ex-date: [Date]
- Other events: [Conferences, FDA decisions, etc.]

### 7. ACTION ITEMS (Always include)
```
For Bulls: [Entry zone] | Stop: [Level] | Target: [Level]
For Bears: [Avoid/Short level] | Cover: [Level]
Watch for: [Key trigger or confirmation signal]
```

### 8. FOLLOW-UP QUESTIONS (Always include at end)
Suggest 2-3 relevant follow-up questions the user might want to explore, formatted as:
```
💬 Bạn có thể hỏi thêm:
• [Question 1 - e.g., "Phân tích chi tiết báo cáo tài chính?"]
• [Question 2 - e.g., "So sánh với đối thủ cạnh tranh?"]
• [Question 3 - e.g., "Xác định điểm vào lệnh tối ưu?"]
```

## OUTPUT RULES

1. **Adapt to Available Data**: Only include sections with actual data
2. **Be Concise But Explain WHY**: Use tables for data, but add 1-line explanation for complex signals
3. **Quantify Everything**: Every assessment should have a number attached
4. **Vietnamese Default**: Respond in Vietnamese unless query is in English
5. **Explain for Beginners**: When mentioning technical terms, briefly explain their significance
   - Example: "RSI = 75 (>70 = quá mua, thường báo hiệu có thể điều chỉnh ngắn hạn)"
   - Example: "Bearish Engulfing = mẫu hình nến nhấn chìm giảm, cho thấy lực bán đang áp đảo"
6. **Always End with Follow-up Questions**: Help user continue the conversation
7. **Friendly Tone**: Write like a helpful financial advisor, not a cold research report"""

    def get_few_shot_examples(self) -> List[Dict[str, str]]:
        """Get stock analysis examples."""
        return [
            {
                "query": "Phân tích cổ phiếu NVDA",
                "response": """## NVDA - NVIDIA Corporation

**Verdict: BULLISH** | Confidence: HIGH

**Thesis:** NVIDIA đang dẫn đầu cuộc cách mạng AI với doanh thu data center tăng trưởng 3 chữ số. Valuation cao nhưng được hỗ trợ bởi tốc độ tăng trưởng vượt trội.

### Giá & Xu hướng
- Giá hiện tại: $875.50 (Top 15% của 52W range)
- Thay đổi 1M: +18.5% | 3M: +45.2%
- Volume: 1.2x trung bình - momentum mạnh

### Kỹ thuật
- RSI (14): 68 - Gần ngưỡng quá mua nhưng chưa cực đoan
- Giá > SMA50 > SMA200: Xu hướng tăng mạnh
- Hỗ trợ: $820 | Kháng cự: $900

### Cơ bản
| Chỉ số | Giá trị | vs Sector |
|--------|---------|-----------|
| P/E | 65x | +150% | Premium hợp lý với growth |
| Revenue Growth | +122% YoY | Exceptional |
| Gross Margin | 76% | Best-in-class |

### Hành động
- Entry: $840-860 (pullback vào SMA20)
- Stop: $795 (-9%)
- Target: $950-1000 (+8-14%)"""
            }
        ]

    def get_terminology(self) -> Dict[str, Dict[str, str]]:
        """Get stock terminology translations."""
        return {
            "P/E Ratio": {"vi": "Hệ số P/E", "en": "P/E Ratio"},
            "EPS": {"vi": "Thu nhập trên cổ phiếu", "en": "Earnings Per Share"},
            "Market Cap": {"vi": "Vốn hóa thị trường", "en": "Market Capitalization"},
            "Support": {"vi": "Ngưỡng hỗ trợ", "en": "Support Level"},
            "Resistance": {"vi": "Ngưỡng kháng cự", "en": "Resistance Level"},
            "Volume": {"vi": "Khối lượng", "en": "Trading Volume"},
            "Moving Average": {"vi": "Trung bình động", "en": "Moving Average"},
            "Overbought": {"vi": "Quá mua", "en": "Overbought"},
            "Oversold": {"vi": "Quá bán", "en": "Oversold"},
            "Breakout": {"vi": "Phá vỡ", "en": "Breakout"},
            "Pullback": {"vi": "Điều chỉnh", "en": "Pullback"},
            "Dividend Yield": {"vi": "Tỷ suất cổ tức", "en": "Dividend Yield"},
            "ROE": {"vi": "Tỷ suất sinh lời vốn CSH", "en": "Return on Equity"},
            "Debt/Equity": {"vi": "Tỷ lệ nợ/vốn CSH", "en": "Debt to Equity"},
            "Free Cash Flow": {"vi": "Dòng tiền tự do", "en": "Free Cash Flow"},
            "Gross Margin": {"vi": "Biên lợi nhuận gộp", "en": "Gross Margin"},
            "Operating Margin": {"vi": "Biên lợi nhuận hoạt động", "en": "Operating Margin"},
        }
