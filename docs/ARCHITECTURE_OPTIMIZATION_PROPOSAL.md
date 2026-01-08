# Architecture Optimization Proposal

## 1. Vấn Đề Hiện Tại

### 1.1 Flow Hiện Tại (3 LLM Calls)

```
User Query: "Phân tích NVDA, GOOGLE, AMAZON..."
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│ LLM CALL #1: CLASSIFIER (gpt-4.1-nano, ~3500ms)              │
│                                                               │
│ Input:  Query text                                           │
│ Output: symbols=["GOOGLE", "AMAZON"], categories, query_type │
│                                                               │
│ ❌ LỖI: Symbols là raw text, không phải tickers!             │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│ LLM CALL #2: SYMBOL RESOLVER (gpt-4.1-nano, ~500ms each)     │
│                                                               │
│ Input:  ["GOOGLE", "AMAZON", "NETFLIX"]                      │
│ Output: resolved_symbols = [{symbol: "GOOGL"}, ...]          │
│                                                               │
│ ✅ FIX: Giờ update classification.symbols với normalized     │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│ LLM CALL #3: ROUTER (gpt-4.1-mini, ~3500ms)                  │
│                                                               │
│ Input:  Query + symbols + ALL tool summaries                 │
│ Output: selected_tools, complexity, strategy                 │
│                                                               │
│ ⚠️ TRÙNG LẶP: Router cũng extract intent từ query           │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│ UNIFIED AGENT → TOOL EXECUTION                               │
│                                                               │
│ Gọi FMP API với symbols đã normalize (GOOGL, AMZN, NFLX)    │
└──────────────────────────────────────────────────────────────┘

Tổng thời gian: ~7-10 giây cho classification + routing
```

### 1.2 Vấn Đề Chính

| Vấn đề | Mô tả | Impact |
|--------|-------|--------|
| **Symbol không normalize** | GOOGLE → vẫn là GOOGLE | FMP API fail |
| **3 LLM calls** | Classifier + Resolver + Router | ~7-10s latency |
| **Trùng lặp logic** | Classifier extract symbols, Router cũng parse query | Redundant |
| **Category không cần thiết** | Agent loop có access ALL tools | Có thể bỏ |

---

## 2. Giải Pháp Đã Implement

### 2.1 Fix Symbol Normalization (✅ Done)

```python
# unified_classifier.py - _resolve_symbols()

# CRITICAL: Update classification.symbols with normalized tickers
if result.resolved_symbols:
    normalized_symbols = []
    for rs in result.resolved_symbols:
        ticker = rs.symbol
        if ticker and ticker not in normalized_symbols:
            normalized_symbols.append(ticker)

    if normalized_symbols:
        old_symbols = classification.symbols
        classification.symbols = normalized_symbols  # ← KEY FIX
        self.logger.info(
            f"[CLASSIFIER] Symbols normalized: {old_symbols} → {normalized_symbols}"
        )
```

**Result:**
```
Before: symbols = ["GOOGLE", "AMAZON", "NETFLIX"]
After:  symbols = ["GOOGL", "AMZN", "NFLX"]
```

---

## 3. Đề Xuất Tối Ưu Hóa

### Option A: Merge Classifier + Router (Recommended)

**Ý tưởng:** Một LLM call làm cả classification VÀ routing.

```
┌──────────────────────────────────────────────────────────────┐
│ UNIFIED LLM CALL (1 call thay vì 3)                          │
│                                                               │
│ Input:                                                        │
│   - Query text                                                │
│   - ALL tool summaries (from catalog)                         │
│   - UI context (active tab)                                   │
│                                                               │
│ Output:                                                       │
│   - symbols: ["GOOGL", "AMZN"]  ← Already normalized!        │
│   - selected_tools: ["getStockPrice", "getTechnicalIndicators"]│
│   - complexity: "complex"                                     │
│   - strategy: "iterative"                                     │
│   - market_type: "stock"                                      │
│                                                               │
│ Model: gpt-4.1-mini (cần thông minh hơn vì làm nhiều việc)  │
└──────────────────────────────────────────────────────────────┘
```

**Prompt Structure:**
```
You are a financial assistant router. Given a user query:

1. EXTRACT symbols mentioned (convert company names to tickers):
   - "Google" → "GOOGL"
   - "Amazon" → "AMZN"
   - "Netflix" → "NFLX"

2. SELECT tools needed from this catalog:
   {tool_catalog}

3. DETERMINE complexity and strategy.

User Query: "{query}"
UI Context: {ui_context}

Output JSON:
{
  "symbols": ["GOOGL", "AMZN"],
  "selected_tools": ["getStockPrice", "getTechnicalIndicators"],
  "complexity": "complex",
  "strategy": "iterative",
  "reasoning": "..."
}
```

**Benefits:**
- ⏱️ **Latency**: 3-4s thay vì 7-10s (giảm 50-60%)
- 💰 **Cost**: 1 LLM call thay vì 3
- 🎯 **Accuracy**: LLM thấy cả query + tools → quyết định tốt hơn
- 🔄 **Consistency**: Một quyết định nhất quán

---

### Option B: Keep Separate but Pipeline Better

Nếu muốn giữ separation of concerns:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Classifier      │────▶│ Symbol Resolver │────▶│ Router          │
│ (gpt-4.1-nano)  │     │ (Cache/LLM)     │     │ (gpt-4.1-nano)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
   Extract raw            Normalize to            Select tools
   symbols               tickers                  & strategy

Optimization: Run Classifier + Router in PARALLEL, share symbols
```

---

### Option C: Let Agent Loop Handle Everything (ChatGPT-style)

**Ý tưởng:** Bỏ Classifier + Router, để Agent Loop tự quyết định.

```
User Query ──────▶ Agent Loop (với ALL tools available)
                        │
                        ▼
                   LLM decides which tools to call
                        │
                        ▼
                   Execute tools
                        │
                        ▼
                   Synthesize response
```

**Pros:**
- Đơn giản nhất
- Giống ChatGPT/Claude
- Linh hoạt

**Cons:**
- Có thể gọi nhiều tools không cần thiết
- Không có pre-selection optimization
- Khó control cost

---

## 4. Recommendation

### Short-term (Immediate): ✅ Done
- Fix symbol normalization bug
- Ensure `classification.symbols` has normalized tickers

### Medium-term: Option A
1. Create `UnifiedRouter` class that:
   - Receives query + tool catalog
   - Outputs: symbols (normalized) + selected_tools + complexity
   - Single LLM call

2. Remove separate Classifier and Router

### Long-term: Hybrid
- Simple queries → Direct to Agent Loop (no pre-routing)
- Complex queries → Use UnifiedRouter for optimization

---

## 5. Semantic Search Tool Analysis

### Current Role
```
┌─────────────────────────────────────────┐
│ Semantic Search Tool                    │
│                                         │
│ Used by Router to find relevant tools   │
│ when LLM is uncertain                   │
└─────────────────────────────────────────┘
```

### Evaluation

| Aspect | Status | Notes |
|--------|--------|-------|
| Tool Discovery | ✅ Good | Helps find tools by semantic meaning |
| Speed | ⚠️ Adds latency | Extra embedding call |
| Necessity | ❓ Questionable | LLM already sees ALL tool summaries |

### Recommendation
- **Keep** semantic search for large tool catalogs (50+ tools)
- **Consider removing** if catalog is small (<20 tools)
- Router LLM với full tool summaries đã đủ thông minh

---

## 6. Implementation Roadmap

```
Phase 1: Fix Bugs (✅ Complete)
├── Symbol normalization bug
└── Update classification.symbols

Phase 2: Optimize Current Flow
├── Parallel Classifier + Router calls
├── Better caching for symbol resolution
└── Reduce Symbol Resolver LLM calls (use cache first)

Phase 3: Unified Router (Optional)
├── Merge Classifier + Router
├── Single LLM call for all pre-processing
└── Benchmark and compare
```

---

## 7. Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Pre-processing latency | ~7-10s | ~3-4s |
| LLM calls before agent | 3 | 1-2 |
| Symbol resolution accuracy | ~70% | ~95% |
| Tool selection accuracy | Unknown | Track |
