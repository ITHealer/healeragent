# Chat Agent Architecture Redesign Proposal

**Date:** 2026-01-22
**Status:** Proposal
**Author:** AI Architecture Review

---

## Executive Summary

Đề xuất tái cấu trúc AI chatbot flow để:
1. Hỗ trợ **Response Modes** (Fast/Auto/Expert) như các AI chatbot hiện đại
2. **Loại bỏ Intent Classification** khi dùng large models (redundant)
3. **Đơn giản hóa architecture** để dễ scale và maintain
4. **Unified provider abstraction** để tránh bugs khi switch models

---

## 1. Current Architecture Analysis

### 1.1 Existing Flow (V2 Endpoint)

```
┌─────────────────────────────────────────────────────────────────┐
│                     CURRENT FLOW (7 PHASES)                      │
└─────────────────────────────────────────────────────────────────┘

User Request
    │
    ▼
[Phase 1] Context Building (~100ms)
    ├─ Load Core Memory
    ├─ Load Conversation Summary
    ├─ Load Recent History (10 messages)
    └─ Load Working Memory Symbols
    │
    ▼
[Phase 2] Intent Classification (~300-800ms) ⚠️ BOTTLENECK
    ├─ 1 LLM call (gpt-4.1-mini)
    ├─ Symbol extraction & normalization
    ├─ Complexity determination
    ├─ Market type detection
    └─ Analysis type classification
    │
    ▼
[Phase 3] Agent Execution (~3-30s)
    ├─ 1-6 LLM calls with tools
    ├─ Tool execution (parallel)
    └─ Streaming response
    │
    ▼
[Phase 4-7] Post-processing (~500ms)
    ├─ Save conversation
    ├─ Update working memory
    ├─ Create summaries
    └─ Chart resolution
```

### 1.2 Problems Identified

| Problem | Impact | Severity |
|---------|--------|----------|
| Intent Classification always runs | +300-800ms latency, +cost | HIGH |
| No user control over response speed | Poor UX | HIGH |
| Complex 7-phase pipeline | Hard to maintain | MEDIUM |
| GPT/Gemini separation caused bugs | Reliability issues | HIGH |
| Memory update blocks response | Slower perceived speed | MEDIUM |

### 1.3 Intent Classification - Redundancy Analysis

#### When Classification IS Needed:
```
Small Models (gpt-4o-mini, gemini-flash):
├─ Cannot reliably select correct tools from 30+ options
├─ May misunderstand complex queries
├─ Benefit from pre-filtered tool set
└─ Need symbol normalization hints
```

#### When Classification is NOT Needed:
```
Large Models (gpt-4o, gemini-2.5-pro, claude-sonnet):
├─ Excellent tool selection from full catalog
├─ Understand context and intent natively
├─ Can normalize "Google" → "GOOGL" themselves
└─ Self-determine complexity and steps needed
```

**Conclusion:** Intent Classification is **REDUNDANT** for large models.

---

## 2. Proposed Architecture

### 2.1 Response Modes Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     RESPONSE MODE SELECTION                      │
│                                                                  │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                │
│   │  AUTO   │      │  FAST   │      │ EXPERT  │                │
│   │   🔄    │      │   ⚡    │      │   🧠    │                │
│   │ Default │      │ Speed   │      │ Quality │                │
│   └─────────┘      └─────────┘      └─────────┘                │
│                                                                  │
│   Adapts to        Respond          Think further               │
│   each query       quicker          to explore                  │
│                    to act sooner    deeper                      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Mode Specifications

#### ⚡ FAST Mode
```yaml
Purpose: "Respond quicker to act sooner"
Target Latency: 3-8 seconds

Configuration:
  model: gpt-4o-mini / gemini-2.0-flash
  use_classifier: true  # Guide small model
  max_turns: 2
  tool_set: filtered_top_5
  web_search: false
  system_prompt: condensed

Use Cases:
  - "Giá AAPL?"
  - "PE ratio MSFT?"
  - "RSI là gì?"
  - Simple lookups
  - Quick definitions
```

#### 🧠 EXPERT Mode
```yaml
Purpose: "Think further to explore deeper"
Target Latency: 15-60 seconds

Configuration:
  model: gpt-4o / gemini-2.5-pro / claude-sonnet
  use_classifier: false  # Model decides everything
  max_turns: 6
  tool_set: all_tools
  web_search: true
  system_prompt: full_with_examples

Use Cases:
  - "So sánh toàn diện NVDA và AMD"
  - "Phân tích kỹ thuật + định giá GOOGL"
  - "Chiến lược đầu tư AI stocks 2026"
  - Multi-step analysis
  - Research tasks
```

#### 🔄 AUTO Mode (Default)
```yaml
Purpose: "Adapts models to each query"

Selection Logic (NO LLM CALL - Pure Heuristics):

  def select_mode(query, context):
      # Rule 1: Short simple queries → FAST
      if len(query) < 50 and no_complex_indicators(query):
          return FAST

      # Rule 2: Complex keywords → EXPERT
      complex_keywords = [
          "so sánh", "phân tích chi tiết", "toàn diện",
          "nghiên cứu", "chiến lược", "đánh giá sâu",
          "compare", "analyze", "comprehensive", "research"
      ]
      if any(kw in query.lower() for kw in complex_keywords):
          return EXPERT

      # Rule 3: Multi-symbol queries → EXPERT
      if count_symbols(query) >= 3:
          return EXPERT

      # Rule 4: Context continuity
      if context.previous_mode == EXPERT:
          return EXPERT  # Maintain context

      # Default: FAST for speed
      return FAST
```

### 2.3 New Simplified Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEW FLOW (3 PHASES)                           │
└─────────────────────────────────────────────────────────────────┘

                        User Request
                    + response_mode param
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ [PHASE 1] MODE ROUTER (No LLM, ~10ms)                           │
│                                                                  │
│   Input: query, response_mode, context                          │
│   Output: effective_mode, model_config                          │
│                                                                  │
│   if response_mode == "auto":                                   │
│       effective_mode = heuristic_select(query, context)         │
│   else:                                                         │
│       effective_mode = response_mode                            │
│                                                                  │
│   model_config = MODE_CONFIGS[effective_mode]                   │
└─────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌──────────────────────┐      ┌──────────────────────┐
│     FAST PATH        │      │    EXPERT PATH       │
│──────────────────────│      │──────────────────────│
│                      │      │                      │
│ [Optional Classifier]│      │ [Skip Classifier]    │
│ Quick intent check   │      │ Model self-decides   │
│ ~300ms if needed     │      │                      │
└──────────────────────┘      └──────────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ [PHASE 2] UNIFIED AGENT EXECUTION                                │
│                                                                  │
│   Same flow for all modes, different configs:                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Config from MODE_CONFIGS:                                │   │
│   │   - model_name: str                                      │   │
│   │   - provider_type: str                                   │   │
│   │   - max_turns: int                                       │   │
│   │   - tool_set: list[str] | "all"                         │   │
│   │   - enable_web_search: bool                              │   │
│   │   - system_prompt_version: str                           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Agent Loop:                                                    │
│   for turn in range(max_turns):                                 │
│       response = await llm.chat(messages, tools)                │
│       if no_tool_calls(response):                               │
│           break  # Generate final response                      │
│       results = await execute_tools_parallel(tool_calls)        │
│       messages.append(results)                                  │
│                                                                  │
│   yield streaming_response                                       │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ [PHASE 3] ASYNC POST-PROCESSING (Non-blocking)                   │
│                                                                  │
│   # Response already streaming to user                          │
│   # These run in background:                                    │
│                                                                  │
│   asyncio.create_task(save_conversation(...))                   │
│   asyncio.create_task(update_memory(...))                       │
│   asyncio.create_task(create_summary_if_needed(...))            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Provider Abstraction (Unified Multi-Model)

### 3.1 Problem: Why GPT/Gemini Separation Failed

```
Previous Approach (FAILED):
├─ Separate code paths for GPT vs Gemini
├─ Different message formats not properly converted
├─ Gemini thought_signature not preserved across turns
├─ Tool call format differences caused silent failures
└─ Debugging nightmare - which path had the bug?
```

### 3.2 Solution: Single Flow + Provider Interface

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROVIDER ABSTRACTION LAYER                    │
└─────────────────────────────────────────────────────────────────┘

                    ┌───────────────────┐
                    │   UNIFIED AGENT   │
                    │                   │
                    │ Uses abstract     │
                    │ LLMProvider       │
                    │ interface only    │
                    └─────────┬─────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │           LLMProvider Interface          │
        │─────────────────────────────────────────│
        │ + format_messages(msgs) → provider_fmt  │
        │ + format_tools(tools) → provider_fmt    │
        │ + call(messages, tools) → response      │
        │ + parse_response(resp) → unified_fmt    │
        │ + stream(messages, tools) → events      │
        │ + handle_special_features(...)          │
        └─────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ OpenAIProvider  │  │ GeminiProvider  │  │ OpenRouterProv  │
│─────────────────│  │─────────────────│  │─────────────────│
│ Native format   │  │ Convert msgs    │  │ Multi-model     │
│ Native tools    │  │ Convert tools   │  │ gateway         │
│                 │  │ Handle thought_ │  │                 │
│                 │  │ signature       │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 3.3 Unified Response Format

```python
@dataclass
class UnifiedLLMResponse:
    """All providers return this format."""

    content: Optional[str]  # Text response
    tool_calls: List[ToolCall]  # Normalized tool calls
    finish_reason: str  # "stop" | "tool_calls" | "length"

    # Provider-specific (preserved for multi-turn)
    raw_response: Any  # Original response object
    provider_metadata: Dict  # thought_signature, etc.

    # Usage tracking
    input_tokens: int
    output_tokens: int


@dataclass
class ToolCall:
    """Normalized tool call format."""

    id: str
    name: str
    arguments: Dict[str, Any]

    # Provider-specific (for response)
    raw_call: Any  # Original format for sending back
```

---

## 4. Configuration-Based Mode System

### 4.1 Mode Configurations

```python
MODE_CONFIGS = {
    "fast": ModeConfig(
        display_name="Fast",
        description="Respond quicker to act sooner",
        icon="⚡",

        # Model settings
        model_name="gpt-4o-mini",
        provider_type="openai",
        fallback_model="gemini-2.0-flash",
        fallback_provider="gemini",

        # Behavior settings
        use_classifier=True,
        max_turns=2,
        tool_selection="filtered",  # Top 5 relevant
        enable_web_search=False,

        # Prompt settings
        system_prompt_version="condensed",
        include_examples=False,

        # Timeouts
        classifier_timeout_ms=500,
        agent_turn_timeout_ms=10000,
        total_timeout_ms=15000,
    ),

    "expert": ModeConfig(
        display_name="Expert",
        description="Think further to explore deeper",
        icon="🧠",

        # Model settings
        model_name="gpt-4o",
        provider_type="openai",
        fallback_model="gemini-2.5-pro",
        fallback_provider="gemini",

        # Behavior settings
        use_classifier=False,  # Model decides everything
        max_turns=6,
        tool_selection="all",
        enable_web_search=True,

        # Prompt settings
        system_prompt_version="full",
        include_examples=True,

        # Timeouts
        classifier_timeout_ms=0,  # Not used
        agent_turn_timeout_ms=30000,
        total_timeout_ms=120000,
    ),

    "auto": ModeConfig(
        display_name="Auto",
        description="Adapts models to each query",
        icon="🔄",

        # Determined at runtime by heuristics
        # Falls back to "fast" or "expert" config
    ),
}
```

### 4.2 API Request Changes

```python
class ChatRequest(BaseModel):
    """Updated request model."""

    # Existing fields
    query: str
    session_id: Optional[str]
    user_id: str

    # NEW: Response mode selection
    response_mode: Literal["auto", "fast", "expert"] = "auto"

    # Optional: Override specific settings
    model_override: Optional[str] = None  # Force specific model
    provider_override: Optional[str] = None  # Force provider
```

### 4.3 SSE Events (Updated)

```javascript
// NEW: Mode selection event
{
    type: "mode_selected",
    mode: "fast",  // or "expert"
    reason: "auto_heuristic: short_query",  // Why this mode
    config: { model: "gpt-4o-mini", max_turns: 2 }
}

// Existing events continue...
{ type: "turn_start", turn: 1, max_turns: 2 }
{ type: "tool_calls", tools: [...] }
{ type: "content", content: "...", is_final: false }
{ type: "done", mode_used: "fast", total_time_ms: 4500 }
```

---

## 5. Implementation Plan

### Phase 1: Foundation (Week 1)
```
□ Add response_mode to ChatRequest
□ Create MODE_CONFIGS dictionary
□ Implement heuristic mode selector for AUTO
□ Add mode_selected SSE event
□ Update frontend to show mode indicator
```

### Phase 2: Fast Mode (Week 2)
```
□ Create condensed system prompt
□ Implement tool filtering logic
□ Set up 2-turn limit
□ Test latency targets (3-8s)
□ A/B test against current flow
```

### Phase 3: Expert Mode (Week 3)
```
□ Remove classifier for expert mode
□ Enable full tool set
□ Configure 6-turn limit
□ Enable web search by default
□ Test quality vs current flow
```

### Phase 4: Provider Abstraction (Week 4)
```
□ Define LLMProvider interface
□ Refactor OpenAI provider
□ Refactor Gemini provider (thought_signature)
□ Add fallback mechanism
□ Integration testing
```

### Phase 5: Cleanup (Week 5)
```
□ Remove legacy /chat endpoint (if ready)
□ Move memory updates to background
□ Performance optimization
□ Documentation
□ Monitoring dashboards
```

---

## 6. Metrics & Success Criteria

### 6.1 Latency Targets

| Mode | P50 | P90 | P99 |
|------|-----|-----|-----|
| Fast | 4s | 6s | 10s |
| Expert | 20s | 40s | 60s |
| Current | 8s | 15s | 30s |

### 6.2 Quality Metrics

```
Fast Mode:
  - Answer accuracy: ≥ 90% (simple queries)
  - Tool selection accuracy: ≥ 85%

Expert Mode:
  - Answer completeness: ≥ 95%
  - Multi-step success rate: ≥ 90%
  - User satisfaction: ≥ 4.5/5
```

### 6.3 Cost Efficiency

```
Expected savings from skipping classifier in Expert mode:
  - ~$0.002 per request (gpt-4.1-mini call)
  - At 10K requests/day = ~$600/month saved

Expected increase from larger models:
  - Expert mode costs ~3x Fast mode
  - But only used for ~30% of queries (AUTO selection)
```

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Fast mode quality drops | Medium | High | A/B test, fallback to classifier |
| Expert mode too slow | Low | Medium | Timeout limits, streaming |
| AUTO heuristics wrong | Medium | Medium | Monitor, tune thresholds |
| Provider abstraction bugs | Medium | High | Comprehensive tests |
| User confusion with modes | Low | Low | Good UX, tooltips |

---

## 8. Decision Points

### 8.1 Cần quyết định trước khi implement:

1. **Default mode cho user mới?**
   - Option A: AUTO (adaptive)
   - Option B: FAST (speed first)
   - Recommendation: **AUTO**

2. **Cho phép user override model không?**
   - Option A: Yes, advanced settings
   - Option B: No, hide complexity
   - Recommendation: **Yes, but hidden in settings**

3. **Giữ legacy /chat endpoint?**
   - Option A: Deprecate immediately
   - Option B: Parallel run 1 month
   - Recommendation: **Parallel run, then deprecate**

4. **Memory update strategy?**
   - Option A: Background async (non-blocking)
   - Option B: Sync after response (blocking)
   - Recommendation: **Background async**

---

## Appendix A: Heuristic Mode Selection Logic

```python
class AutoModeSelector:
    """Pure heuristic mode selection - NO LLM CALLS."""

    COMPLEX_KEYWORDS = {
        "vi": [
            "so sánh", "phân tích chi tiết", "toàn diện",
            "nghiên cứu", "chiến lược", "đánh giá sâu",
            "giải thích", "tại sao", "như thế nào"
        ],
        "en": [
            "compare", "analyze", "comprehensive",
            "research", "strategy", "deep dive",
            "explain", "why", "how does"
        ]
    }

    SIMPLE_PATTERNS = [
        r"^giá\s+\w+\??$",  # "giá AAPL?"
        r"^price\s+\w+\??$",  # "price AAPL?"
        r"^\w+\s+là gì\??$",  # "RSI là gì?"
        r"^what is\s+\w+\??$",  # "what is RSI?"
    ]

    def select(
        self,
        query: str,
        context: ChatContext
    ) -> Literal["fast", "expert"]:
        """Select mode based on query characteristics."""

        query_lower = query.lower().strip()

        # Rule 1: Very short simple queries → FAST
        if len(query) < 30:
            for pattern in self.SIMPLE_PATTERNS:
                if re.match(pattern, query_lower):
                    return "fast"

        # Rule 2: Complex keywords → EXPERT
        for lang, keywords in self.COMPLEX_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                return "expert"

        # Rule 3: Multiple symbols (≥3) → EXPERT
        symbols = extract_symbols(query)
        if len(symbols) >= 3:
            return "expert"

        # Rule 4: Long query (>150 chars) → EXPERT
        if len(query) > 150:
            return "expert"

        # Rule 5: Previous turn was expert → Continue EXPERT
        if context.previous_mode == "expert":
            return "expert"

        # Rule 6: Contains question requiring reasoning → EXPERT
        reasoning_indicators = [
            "tại sao", "why", "nên", "should",
            "có nên", "liệu", "whether"
        ]
        if any(ind in query_lower for ind in reasoning_indicators):
            return "expert"

        # Default: FAST for speed
        return "fast"
```

---

## Appendix B: Migration Checklist

```
Pre-migration:
□ Backup current config
□ Set up feature flags
□ Create rollback plan
□ Notify stakeholders

During migration:
□ Deploy with feature flag OFF
□ Enable for internal testing
□ Enable for 10% traffic
□ Monitor metrics
□ Enable for 50% traffic
□ Monitor 24h
□ Enable for 100%

Post-migration:
□ Remove feature flags
□ Update documentation
□ Archive legacy code
□ Performance report
```

---

**Document Version:** 1.0
**Last Updated:** 2026-01-22
**Next Review:** After Phase 1 implementation
