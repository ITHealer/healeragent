# HealerAgent - AI Chatbot Architecture Flow

**Version:** 2.0 (với Multimodal Classification)
**Last Updated:** 2026-01-05
**Author:** Claude AI Architecture Review

---

## 1. Tổng Quan Kiến Trúc

HealerAgent là một AI chatbot đa phương thức (multimodal) cho phân tích tài chính, được xây dựng theo kiến trúc tương tự ChatGPT và Claude AI.

### 1.1 So Sánh với ChatGPT/Claude

| Feature | HealerAgent | ChatGPT | Claude | Đánh Giá |
|---------|-------------|---------|--------|----------|
| Unified Classification | 1 LLM call | Internal | Internal | Tương đương |
| Multimodal (Vision) | Supported | Native | Native | Tương đương |
| Tool/Function Calling | Inline loop | Native | Native | Tương đương |
| Memory System | 3-tier | Context only | Context only | Tốt hơn |
| Streaming | SSE | SSE | SSE | Tương đương |
| Caching | Redis + Local | Internal | Internal | Tương đương |

### 1.2 Điểm Mạnh

- **Memory Tiers**: Core Memory (user profile) + Working Memory (session) + Recursive Summary
- **Two-Level Tool Loading**: Token-efficient (summary for selection, full for execution)
- **Soft Context Inheritance**: UI context guides symbol resolution
- **Provider Abstraction**: OpenAI, OpenRouter, Gemini, Ollama

### 1.3 Điểm Cần Cải Thiện

- Classification Cache TTL quá ngắn (120s → nên 5 phút)
- Symbol Disambiguation early return (nên graceful response)
- Tool Registry cold start (nên pre-warm)

---

## 2. Full Architecture Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER REQUEST                                        │
│                                                                                  │
│   ┌────────────────────────────────────────────────────────────────────────┐   │
│   │  {                                                                      │   │
│   │    "query": "Phân tích chart này",                                     │   │
│   │    "images": [{"source": "url", "data": "https://..."}],              │   │
│   │    "session_id": "abc123",                                             │   │
│   │    "ui_context": {"active_tab": "stock"}                               │   │
│   │  }                                                                      │   │
│   └────────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: API ROUTER                                                               │
│ File: src/routers/v2/chat_assistant.py                                           │
│                                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ POST /api/v2/chat-assistant/chat                                            │ │
│ │                                                                             │ │
│ │ 1. API Key Auth (Depends)                                                   │ │
│ │ 2. Validate Request (query or images required)                              │ │
│ │ 3. Process Images → List[ProcessedImage]                                   │ │
│ │ 4. Create/Resume Session                                                    │ │
│ │ 5. Stream Response via SSE                                                  │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
│ Key Code (lines 525-546):                                                        │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ # Process images                                                            │ │
│ │ if data.images:                                                            │ │
│ │     processed_images = await _process_images(data.images)                  │ │
│ │                                                                             │ │
│ │ # Build classification context WITH images                                  │ │
│ │ ctx = ClassifierContext(                                                    │ │
│ │     query=query,                                                            │ │
│ │     conversation_history=[],                                                │ │
│ │     ui_context=data.ui_context.model_dump(),                               │ │
│ │     images=processed_images  # ← CRITICAL for vision classification       │ │
│ │ )                                                                           │ │
│ │ classification = await classifier.classify(ctx)                             │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: IMAGE PROCESSING                                                         │
│ File: src/utils/image/image_processor.py                                         │
│                                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ ImageProcessor.process()                                                    │ │
│ │                                                                             │ │
│ │ Input Types:                                                                │ │
│ │ - URL: "https://example.com/chart.png"                                     │ │
│ │ - Base64: "iVBORw0KGgoAAAANSUhEUgAA..."                                    │ │
│ │ - File: "/path/to/image.png"                                               │ │
│ │ - Data URL: "data:image/png;base64,..."                                    │ │
│ │                                                                             │ │
│ │ Output: ProcessedImage                                                      │ │
│ │ - base64_data: str                                                          │ │
│ │ - media_type: "image/png"                                                   │ │
│ │ - size_bytes: int                                                           │ │
│ │ - original_source: ImageSource                                              │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
│ Features:                                                                        │
│ - HTTP retry with exponential backoff (requests library)                        │
│ - Magic bytes format detection                                                   │
│ - Size validation (max 20MB)                                                     │
│ - Parallel batch processing                                                      │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: UNIFIED CLASSIFICATION (1 LLM Call)                                      │
│ File: src/agents/classification/unified_classifier.py                            │
│                                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ UnifiedClassifier.classify(context)                                         │ │
│ │                                                                             │ │
│ │           ┌──────────────────────────────────────────────┐                  │ │
│ │           │          Has Images?                         │                  │ │
│ │           │      context.has_images()                    │                  │ │
│ │           └──────────────────┬───────────────────────────┘                  │ │
│ │                              │                                              │ │
│ │              ┌───────────────┴───────────────┐                              │ │
│ │              ▼                               ▼                              │ │
│ │   ┌──────────────────────┐       ┌──────────────────────┐                  │ │
│ │   │ Text-Only Model      │       │ Vision Model         │                  │ │
│ │   │ gpt-4.1-nano        │       │ qwen-vl-72b          │                  │ │
│ │   │ _call_llm()          │       │ _call_llm_with_vision()│                │ │
│ │   └──────────────────────┘       └──────────────────────┘                  │ │
│ │                                                                             │ │
│ │   Image Analysis Instructions:                                              │ │
│ │   - STOCK CHART: Extract ticker, patterns, timeframe                       │ │
│ │   - CRYPTO CHART: Cryptocurrency, movements, patterns                      │ │
│ │   - FINANCIAL DATA: Metrics, symbols, data points                          │ │
│ │   - DOCUMENT: Company name, financial metrics                              │ │
│ │   - TRADING SCREENSHOT: Symbols, prices, positions                         │ │
│ │                                                                             │ │
│ │   Output: UnifiedClassificationResult                                       │ │
│ │   ┌────────────────────────────────────────────────────────────────────┐  │ │
│ │   │ query_type: "stock_specific" | "crypto_specific" | ...             │  │ │
│ │   │ symbols: ["NVDA", "AAPL"]                                          │  │ │
│ │   │ tool_categories: ["technical", "price"]                            │  │ │
│ │   │ requires_tools: true                                               │  │ │
│ │   │ confidence: 0.95                                                   │  │ │
│ │   │ reasoning: "User sent chart of NVDA with RSI..."                  │  │ │
│ │   │ classification_method: "llm_vision"                                │  │ │
│ │   └────────────────────────────────────────────────────────────────────┘  │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
│ Cache Strategy:                                                                  │
│ - Text queries: Redis cache (120s TTL)                                          │
│ - Multimodal queries: NO cache (images unique)                                  │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: MODE ROUTING                                                             │
│ File: src/handlers/v2/mode_router.py                                             │
│                                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                     Classification Result Analysis                          │ │
│ │                                                                             │ │
│ │                          requires_tools?                                    │ │
│ │                              │                                              │ │
│ │          ┌──────────────────┼──────────────────┐                            │ │
│ │          ▼                  ▼                  ▼                            │ │
│ │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │ │
│ │   │ No Tools    │    │ Normal Mode │    │Deep Research│                    │ │
│ │   │ ~10%        │    │ ~80%        │    │ ~10%        │                    │ │
│ │   │             │    │             │    │             │                    │ │
│ │   │ Direct LLM  │    │ 2-3 LLM     │    │ 7-Phase     │                    │ │
│ │   │ Response    │    │ Calls       │    │ Pipeline    │                    │ │
│ │   └─────────────┘    └─────────────┘    └─────────────┘                    │ │
│ │                                                                             │ │
│ │   Triggers for Deep Research:                                               │ │
│ │   - query_type == SCREENER                                                  │ │
│ │   - symbols.length > 3                                                      │ │
│ │   - Keywords: "so sánh", "compare", "portfolio"                            │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5A: NORMAL MODE HANDLER (80% traffic)                                       │
│ File: src/handlers/v2/normal_mode_chat_handler.py                                │
│                                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ NormalModeChatHandler.handle_chat()                                         │ │
│ │                                                                             │ │
│ │ STEP 1: Context Loading (~50ms)                                            │ │
│ │ ┌───────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ Core Memory    │ Working Memory │ Chat History │ Summary             │ │ │
│ │ │ (User Profile) │ (Session)      │ (10 msgs)    │ (Compressed)        │ │ │
│ │ └───────────────────────────────────────────────────────────────────────┘ │ │
│ │                                                                             │ │
│ │ STEP 2: Symbol Ambiguity Check                                             │ │
│ │ ┌───────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ IF symbol ambiguous (e.g., "BTC" = Bitcoin or stock ticker?)          │ │ │
│ │ │ → Return clarification message OR use UI context to assume            │ │ │
│ │ └───────────────────────────────────────────────────────────────────────┘ │ │
│ │                                                                             │ │
│ │ STEP 3: Run Agent                                                           │ │
│ │ ┌───────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ NormalModeAgent.run_stream()                                          │ │ │
│ │ │ - Build messages with tool summaries                                  │ │ │
│ │ │ - LLM decides which tools to call                                     │ │ │
│ │ │ - Execute tools in parallel                                           │ │ │
│ │ │ - Loop until no more tools or max turns (10)                          │ │ │
│ │ └───────────────────────────────────────────────────────────────────────┘ │ │
│ │                                                                             │ │
│ │ STEP 4: Post-Processing (Async/Background)                                  │ │
│ │ ┌───────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ - Save to database                                                    │ │ │
│ │ │ - Update Core Memory (if significant)                                 │ │ │
│ │ │ - Create summary (if needed)                                          │ │ │
│ │ └───────────────────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 6: AGENT EXECUTION                                                          │
│ File: src/agents/normal_mode/normal_mode_agent.py                                │
│                                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                     AGENT LOOP (Max 10 Turns)                               │ │
│ │                                                                             │ │
│ │   TURN 1:                                                                   │ │
│ │   ┌─────────────────────────────────────────────────────────────────────┐ │ │
│ │   │ _build_initial_messages()                                           │ │ │
│ │   │                                                                     │ │ │
│ │   │ Messages:                                                           │ │ │
│ │   │ ┌───────────────────────────────────────────────────────────────┐  │ │ │
│ │   │ │ System: Financial assistant + Context + Tool Summaries        │  │ │ │
│ │   │ │ History: Previous messages                                    │  │ │ │
│ │   │ │ User: Query text + [Image if multimodal]                      │  │ │ │
│ │   │ └───────────────────────────────────────────────────────────────┘  │ │ │
│ │   │                                                                     │ │ │
│ │   │ → LLM Provider (OpenAI/OpenRouter/Gemini/Ollama)                   │ │ │
│ │   │ → Parse response for tool_calls                                     │ │ │
│ │   └─────────────────────────────────────────────────────────────────────┘ │ │
│ │                              │                                              │ │
│ │              ┌───────────────┴───────────────┐                              │ │
│ │              ▼                               ▼                              │ │
│ │   ┌──────────────────────┐       ┌──────────────────────┐                  │ │
│ │   │ Tool Calls Found     │       │ No Tools = Final     │                  │ │
│ │   │                      │       │ Response             │                  │ │
│ │   │ _execute_tools_      │       │                      │                  │ │
│ │   │ in_parallel()        │       │ → Stream to User     │                  │ │
│ │   └──────────┬───────────┘       └──────────────────────┘                  │ │
│ │              │                                                              │ │
│ │              ▼                                                              │ │
│ │   ┌──────────────────────────────────────────────────────────────────────┐ │ │
│ │   │ TURN 2+: Add tool results → LLM → More tools or Final response      │ │ │
│ │   └──────────────────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 7: TOOL SYSTEM                                                              │
│ File: src/agents/tools/registry.py                                               │
│                                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ ToolRegistry (Singleton + Circuit Breaker)                                  │ │
│ │                                                                             │ │
│ │ Categories (31 Atomic Tools):                                               │ │
│ │ ┌─────────────┬─────────────┬─────────────┬─────────────┐                  │ │
│ │ │ price       │ technical   │ fundamentals│ news        │                  │ │
│ │ │ get_price   │ get_rsi     │ get_pe_ratio│ get_news    │                  │ │
│ │ │ get_quote   │ get_macd    │ get_eps     │ get_events  │                  │ │
│ │ └─────────────┴─────────────┴─────────────┴─────────────┘                  │ │
│ │ ┌─────────────┬─────────────┬─────────────┬─────────────┐                  │ │
│ │ │ market      │ crypto      │ discovery   │ web         │                  │ │
│ │ │ get_indices │ get_btc     │ screen_stock│ web_search  │                  │ │
│ │ │ get_sectors │ get_eth     │ find_stocks │             │                  │ │
│ │ └─────────────┴─────────────┴─────────────┴─────────────┘                  │ │
│ │                                                                             │ │
│ │ Two-Level Loading Strategy:                                                 │ │
│ │ ┌───────────────────────────────────────────────────────────────────────┐ │ │
│ │ │ LEVEL 1: Summary (~50-100 tokens/tool)                                │ │ │
│ │ │ - Name, short description, category                                   │ │ │
│ │ │ - Used in prompts for LLM to choose tools                            │ │ │
│ │ │                                                                       │ │ │
│ │ │ LEVEL 2: Full Schema (~200-400 tokens/tool)                          │ │ │
│ │ │ - All parameters with types, descriptions, examples                  │ │ │
│ │ │ - Capabilities and limitations                                        │ │ │
│ │ │ - Only loaded when tool actually selected                            │ │ │
│ │ └───────────────────────────────────────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 8: MEMORY SYSTEM (3-Tier)                                                   │
│                                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │                                                                             │ │
│ │   ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐     │ │
│ │   │ TIER 1:           │  │ TIER 2:           │  │ TIER 3:           │     │ │
│ │   │ Core Memory       │  │ Working Memory    │  │ Recursive Summary │     │ │
│ │   │                   │  │                   │  │                   │     │ │
│ │   │ File: YAML        │  │ Type: Session Dict│  │ Trigger: >N msgs  │     │ │
│ │   │ Size: ~2K tokens  │  │ Size: ~8K tokens  │  │ Size: Variable    │     │ │
│ │   │                   │  │                   │  │                   │     │ │
│ │   │ Contents:         │  │ Contents:         │  │ Contents:         │     │ │
│ │   │ - User profile    │  │ - Current symbols │  │ - Compressed      │     │ │
│ │   │ - Preferences     │  │ - Task plan       │  │   conversation    │     │ │
│ │   │ - Language        │  │ - Tool results    │  │   history         │     │ │
│ │   │                   │  │ - Reasoning       │  │                   │     │ │
│ │   │ Persistence:      │  │ Persistence:      │  │ Persistence:      │     │ │
│ │   │ Permanent (YAML)  │  │ Session only      │  │ Database          │     │ │
│ │   └───────────────────┘  └───────────────────┘  └───────────────────┘     │ │
│ │                                                                             │ │
│ │   Symbol Continuity: Working Memory preserves symbols across turns         │ │
│ │   (TTL: 5 turns before considered stale)                                   │ │
│ │                                                                             │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 9: STREAMING RESPONSE (SSE)                                                 │
│ File: src/services/streaming_event_service.py                                    │
│                                                                                   │
│ ┌─────────────────────────────────────────────────────────────────────────────┐ │
│ │ Event Types:                                                                │ │
│ │                                                                             │ │
│ │ {"type": "session_start", "data": {"mode": "auto"}}                        │ │
│ │ {"type": "classifying", "data": {"status": "in_progress"}}                 │ │
│ │ {"type": "classified", "data": {                                           │ │
│ │     "query_type": "stock_specific",                                        │ │
│ │     "symbols": ["NVDA"],                                                   │ │
│ │     "classification_method": "llm_vision",  ← Shows vision was used        │ │
│ │     "reasoning": "Chart shows NVDA with RSI..."                            │ │
│ │ }}                                                                          │ │
│ │ {"type": "thinking", "content": "Analyzing technical indicators..."}       │ │
│ │ {"type": "tool_calls", "tools": [{"name": "get_rsi", ...}]}               │ │
│ │ {"type": "tool_results", "results": [...]}                                 │ │
│ │ {"type": "content", "content": "Based on the RSI indicator...", ...}      │ │
│ │ {"type": "done", "total_turns": 2, "total_tool_calls": 3}                 │ │
│ │                                                                             │ │
│ └─────────────────────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Multimodal Classification Flow (New Feature)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL CLASSIFICATION FLOW                        │
│                                                                          │
│   User Input                                                             │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ Query: "Phân tích chart này"                                     │  │
│   │ Image: [Financial chart with NVDA stock, RSI indicator]         │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                        │                                 │
│                                        ▼                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ ImageProcessor.process()                                         │  │
│   │ → Download image from URL                                        │  │
│   │ → Convert to base64                                              │  │
│   │ → Validate format (PNG, JPEG, WebP, GIF)                        │  │
│   │ → Return ProcessedImage                                          │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                        │                                 │
│                                        ▼                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ ClassifierContext(images=processed_images)                       │  │
│   │                                                                  │  │
│   │ context.has_images() → True                                      │  │
│   │ context.get_image_count() → 1                                    │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                        │                                 │
│                                        ▼                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ UnifiedClassifier.classify()                                     │  │
│   │                                                                  │  │
│   │ if is_multimodal:                                                │  │
│   │     → _call_llm_with_vision(prompt, images)                      │  │
│   │     → Vision Model: qwen/qwen2.5-vl-72b-instruct:free           │  │
│   │ else:                                                            │  │
│   │     → _call_llm(prompt)                                          │  │
│   │     → Text Model: gpt-4.1-nano                                   │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                        │                                 │
│                                        ▼                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ Vision Model Prompt Includes:                                    │  │
│   │                                                                  │  │
│   │ "IMPORTANT - IMAGE ANALYSIS (Multimodal Input):                 │  │
│   │  The user has attached image(s). Carefully analyze:             │  │
│   │  - STOCK CHART: Extract ticker, patterns, timeframe            │  │
│   │  - CRYPTO CHART: Cryptocurrency, movements, patterns           │  │
│   │  - FINANCIAL DATA: Metrics, symbols, data points               │  │
│   │  - DOCUMENT: Company name, financial metrics                   │  │
│   │  - TRADING SCREENSHOT: Symbols, prices, positions"             │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                        │                                 │
│                                        ▼                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │ Classification Result:                                           │  │
│   │                                                                  │  │
│   │ {                                                                │  │
│   │   "query_type": "stock_specific",                               │  │
│   │   "symbols": ["NVDA"],  ← Extracted from image!                 │  │
│   │   "tool_categories": ["technical", "price"],                    │  │
│   │   "requires_tools": true,                                       │  │
│   │   "reasoning": "Image shows NVDA chart with RSI indicator...", │  │
│   │   "classification_method": "llm_vision"                         │  │
│   │ }                                                                │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. LLM Call Count Analysis

| Scenario | LLM Calls | Breakdown |
|----------|-----------|-----------|
| Simple query (no tools) | 1 | Classification only |
| Normal query (with tools) | 2-3 | 1 Classification + 1-2 Agent turns |
| Complex query (Deep Research) | 4-7 | 1 Classification + 1 Planning + 1-2 Agent + 1-2 Summary |
| Multimodal query | 2-4 | 1 Vision Classification + 1-3 Agent turns |

---

## 5. Key Files Reference

| Layer | File | Purpose |
|-------|------|---------|
| API Entry | `src/routers/v2/chat_assistant.py` | FastAPI routes, image processing |
| Image | `src/utils/image/image_processor.py` | URL/base64 processing, validation |
| Classification | `src/agents/classification/unified_classifier.py` | Query intent detection |
| Classification | `src/agents/classification/models.py` | ClassifierContext, result types |
| Handler | `src/handlers/v2/normal_mode_chat_handler.py` | Normal mode pipeline |
| Handler | `src/handlers/v2/chat_handler.py` | Deep research pipeline |
| Agent | `src/agents/normal_mode/normal_mode_agent.py` | Tool execution loop |
| Tools | `src/agents/tools/registry.py` | Tool registry + circuit breaker |
| Tools | `src/agents/tools/base.py` | ToolSchema, ToolOutput |
| Memory | `src/agents/memory/core_memory.py` | Persistent user memory |
| Memory | `src/agents/memory/working_memory.py` | Session memory |
| LLM | `src/providers/provider_factory.py` | OpenAI/OpenRouter/Gemini/Ollama |
| Streaming | `src/services/streaming_event_service.py` | SSE event emitter |

---

## 6. Known Issues & Recommendations

### Critical (Should Fix)

1. **Classification Cache TTL** (120s → 300s)
   - Location: `unified_classifier.py:CACHE_TTL_SECONDS`
   - Impact: Repeated queries require re-classification

2. **Symbol Disambiguation** (Early return → Graceful response)
   - Location: `normal_mode_chat_handler.py:_check_symbol_ambiguity`
   - Impact: User gets unclear error instead of helpful response

### Performance

1. **Tool Registry Cold Start**
   - Solution: Pre-warm in `app.py` startup event

2. **Context Building Overhead**
   - Solution: Incremental context, diff-based updates

### Future Enhancements

1. **Response Caching** - Cache final responses for repeated queries
2. **Model Tiering** - Cheap model for classification, expensive for response
3. **Agent Orchestration** - Multi-agent collaboration for complex queries

---

## 7. Testing Multimodal Classification

```python
# Test request with image
import httpx

response = httpx.post(
    "http://localhost:8000/api/v2/chat-assistant/chat",
    json={
        "query": "Phân tích chart này",
        "images": [{
            "source": "url",
            "data": "https://example.com/nvda-chart.png"
        }],
        "ui_context": {"active_tab": "stock"}
    },
    headers={"X-API-Key": "your-api-key"}
)

# Expected classification_method: "llm_vision"
# Expected symbols: ["NVDA"] (extracted from image)
```

---

**End of Architecture Documentation**
