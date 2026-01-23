# Chat Response Modes - Implementation Plan

**Version:** 2.0
**Date:** 2026-01-22
**Status:** Planning
**Author:** Architecture Team

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Industry Research Validation](#2-industry-research-validation)
3. [Current Codebase Inventory](#3-current-codebase-inventory)
4. [Proposed Architecture](#4-proposed-architecture)
5. [Detailed Mode Specifications](#5-detailed-mode-specifications)
6. [Implementation Tasks](#6-implementation-tasks)
7. [Migration Strategy](#7-migration-strategy)
8. [Risk Assessment](#8-risk-assessment)

---

## 1. Executive Summary

### 1.1 Goals
1. Implement **Response Modes** (Fast/Auto/Expert) like industry leaders
2. **Remove redundant Intent Classification** for large models
3. **Simplify architecture** from 7 phases to 3 phases
4. **Unified provider abstraction** to prevent GPT/Gemini bugs
5. **No hardcoded keywords** - use LLM semantic understanding for multilingual support

### 1.2 Key Decisions (Verified by Research)

| Question | Answer | Evidence |
|----------|--------|----------|
| Intent Classification redundant with large models? | ✅ YES | Claude/GPT-5 self-decide; GPT-5 router 94% accuracy |
| Use hardcoded keywords for routing? | ❌ NO | Industry uses LLM semantic understanding |
| Need separate models for each mode? | ❌ NO | Claude Hybrid: "same model, two modes" |
| User control over response mode? | ✅ YES | GPT-5: Auto/Fast/Thinking modes |

### 1.3 Expected Outcomes

| Metric | Current | Fast Mode | Expert Mode |
|--------|---------|-----------|-------------|
| P50 Latency | 8-15s | **3-6s** | 15-40s |
| LLM Calls | 2-7 | **1-2** | 1-6 |
| User Control | None | ✅ Full | ✅ Full |
| Multilingual | Partial | ✅ Full | ✅ Full |

---

## 2. Industry Research Validation

### 2.1 Claude AI - Hybrid Reasoning (Verified ✅)

> "Claude can operate in two modes: a fast, nearly-instant response mode or a slower, more deliberative thinking mode for complex tasks. **Crucially, this isn't a separate model** – it's the same Claude model being allowed to think longer."
> — [Claude AI Extended Thinking](https://claude-ai.chat/blog/extended-thinking-mode/)

**Key Takeaways:**
- Single model, configurable "thinking budget"
- Model self-decides when to use extended thinking
- Developers control via `thinking_budget` parameter (100 → 128K tokens)
- No separate classifier needed

**Reference:** [Anthropic Extended Thinking Docs](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)

### 2.2 GPT-5 - Auto Routing Architecture (Verified ✅)

> "GPT-5 is a unified system with a smart, efficient model that answers most questions, a deeper reasoning model (GPT-5 thinking), and a **real-time router** that quickly decides which to use based on conversation type, complexity, tool needs."
> — [OpenAI GPT-5](https://openai.com/index/introducing-gpt-5/)

**Key Takeaways:**
- **94% accuracy** in complexity detection
- Router trained on real signals (user switches, preference rates)
- Three modes: **Auto, Fast, Thinking**
- `reasoning_effort` parameter: minimal/low/medium/high

**Reference:** [GPT-5 Router Explained](https://rankstudio.net/articles/en/gpt-5-router-explained)

### 2.3 Intent Classification Best Practices (Verified ✅)

> "With many tools/intents, relying solely on the LLM can be brittle. A **hybrid system** that combines semantic search with LLM strikes a practical balance."
> — [AI Agent Routing Best Practices](https://www.patronus.ai/ai-agent-development/ai-agent-routing)

**Industry Recommendation:**

| Scenario | Approach | Rationale |
|----------|----------|-----------|
| Few tools (<10) | Direct LLM | Simple, model handles well |
| Many tools (30+) | Hybrid | Filter first, then LLM decides |
| Small model | Classifier + Filtered Tools | Guide model selection |
| Large model | Direct with All Tools | Model self-decides |

**Reference:** [Intent Classification 2025](https://labelyourdata.com/articles/machine-learning/intent-classification)

---

## 3. Current Codebase Inventory

### 3.1 Existing Components (Can Reuse ✅)

| Component | Location | Status | Reuse Plan |
|-----------|----------|--------|------------|
| **UnifiedAgent** | `src/agents/unified/unified_agent.py` | ✅ Production | Core agent loop - keep |
| **IntentClassifier** | `src/agents/classification/intent_classifier.py` | ✅ Production | Use for FAST mode only |
| **ToolCatalog** | `src/agents/tools/tool_catalog.py` | ✅ Production | Keep - 31 tools |
| **Provider Factory** | `src/providers/provider_factory.py` | ✅ Production | Enhance abstraction |
| **Memory System** | `src/agents/memory/` | ✅ Production | Keep all tiers |
| **Streaming SSE** | `src/routers/v2/chat.py` | ✅ Production | Add new events |
| **Working Memory** | `src/agents/memory/working_memory.py` | ✅ Production | Keep for cross-turn |

### 3.2 Components to Modify

| Component | Location | Changes Needed |
|-----------|----------|----------------|
| **ChatRequest** | `src/routers/v2/chat.py` | Add `response_mode` param |
| **ChatHandler** | `src/handlers/v2/chat_handler.py` | Simplify to 3 phases |
| **chat_assistant.py** | `src/routers/v2/chat_assistant.py` | Add mode routing |
| **Provider Interface** | `src/providers/base_provider.py` | Stronger abstraction |

### 3.3 New Components to Create

| Component | Purpose | Priority |
|-----------|---------|----------|
| **ModeRouter** | LLM-based complexity classification | P0 |
| **ModeConfig** | Configuration for each mode | P0 |
| **FastModeExecutor** | Optimized path for simple queries | P1 |
| **ExpertModeExecutor** | Full capability path | P1 |
| **UnifiedProviderInterface** | Provider abstraction layer | P2 |

### 3.4 Current Flow Analysis

```
CURRENT V2/V3 FLOW (7 Phases):
┌────────────────────────────────────────────────────────────────────┐
│ Phase 1: Context Loading (100ms)                                   │
│   ├─ Load Core Memory                                              │
│   ├─ Load Recursive Summary                                        │
│   ├─ Load Recent History (10 messages)                             │
│   └─ Load Working Memory                                           │
├────────────────────────────────────────────────────────────────────┤
│ Phase 2: Intent Classification (300-800ms) ⚠️ BOTTLENECK           │
│   ├─ 1 LLM call (gpt-4.1-mini)                                     │
│   ├─ Symbol extraction & normalization                              │
│   ├─ Complexity determination                                       │
│   └─ Market type detection                                          │
├────────────────────────────────────────────────────────────────────┤
│ Phase 3-5: Planning + Memory Search + Tool Execution               │
│   ├─ Multiple sub-phases                                            │
│   └─ Sequential processing                                          │
├────────────────────────────────────────────────────────────────────┤
│ Phase 6: LLM Response Generation (2-30s)                           │
│   └─ Agent loop with tools                                          │
├────────────────────────────────────────────────────────────────────┤
│ Phase 7: Post-processing (500ms) ⚠️ BLOCKS RESPONSE                │
│   ├─ Save conversation                                              │
│   ├─ Update memory                                                  │
│   └─ Create summary                                                 │
└────────────────────────────────────────────────────────────────────┘

TOTAL: 7 phases, 2-7 LLM calls, complex branching
```

---

## 4. Proposed Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PROPOSED FLOW (3 Phases)                         │
└─────────────────────────────────────────────────────────────────────┘

                        User Request
                    + response_mode: "auto" | "fast" | "expert"
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [PHASE 1] MODE ROUTER (~100-400ms)                                  │
│                                                                      │
│   if response_mode == "fast":                                       │
│       → Use IntentClassifier + Filtered Tools                       │
│   elif response_mode == "expert":                                   │
│       → Skip Classifier, Use All Tools                              │
│   elif response_mode == "auto":                                     │
│       → LLM Semantic Classification (NO hardcode keywords)          │
│       → Route to FAST or EXPERT based on complexity                 │
│                                                                      │
│   Output: effective_mode, model_config, tool_set                    │
└─────────────────────────────────────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                              ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│      FAST PATH           │    │     EXPERT PATH          │
│──────────────────────────│    │──────────────────────────│
│ Model: gpt-4o-mini       │    │ Model: gpt-4o / gemini   │
│ +IntentClassifier        │    │ NO Classifier            │
│ Tools: Filtered Top 5-8  │    │ Tools: ALL 31            │
│ Max turns: 2             │    │ Max turns: 6             │
│ Web search: OFF          │    │ Web search: ON           │
│ System prompt: Condensed │    │ System prompt: Full      │
└──────────────────────────┘    └──────────────────────────┘
              │                              │
              └──────────────┬───────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [PHASE 2] UNIFIED AGENT EXECUTION                                   │
│                                                                      │
│   Same agent loop for both paths (different configs):               │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │ for turn in range(config.max_turns):                        │   │
│   │     response = await provider.chat(messages, tools)         │   │
│   │     if no_tool_calls(response): break                       │   │
│   │     results = await execute_tools_parallel(tool_calls)      │   │
│   │     messages.append(tool_results)                           │   │
│   │     yield streaming_events                                  │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   yield final_response (streaming)                                  │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│ [PHASE 3] ASYNC POST-PROCESSING (Non-blocking)                      │
│                                                                      │
│   # Response already streaming to user!                             │
│   asyncio.create_task(save_conversation(...))                       │
│   asyncio.create_task(update_memory(...))                           │
│   asyncio.create_task(create_summary_if_needed(...))                │
│                                                                      │
│   # User sees response immediately, background tasks run async      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Mode Router - LLM Semantic Classification (NO Hardcode)

```python
# ❌ OLD APPROACH (Hardcoded - breaks multilingual)
COMPLEX_KEYWORDS = ["so sánh", "phân tích", "compare"]  # Limited languages!

# ✅ NEW APPROACH (LLM Semantic - works ALL languages)
async def classify_complexity(query: str, context: dict) -> Literal["fast", "expert"]:
    """
    Use small LLM to understand query semantics.
    Works in Vietnamese, English, Chinese, Japanese, etc.

    Cost: ~$0.0003 per classification (gpt-4o-mini)
    Latency: ~150-300ms
    """
    prompt = f"""Classify query complexity for a financial AI assistant.

Query: "{query}"
Recent symbols: {context.get('recent_symbols', [])}
Previous complexity: {context.get('previous_mode', 'unknown')}

FAST = Simple lookups, single data points, definitions, greetings
EXPERT = Multi-step analysis, comparisons, research, strategy

Return JSON: {{"complexity": "fast"|"expert", "reason": "brief"}}"""

    response = await llm_call(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=50
    )
    return json.loads(response.content)["complexity"]
```

**Why LLM Semantic Classification?**

| Query (Any Language) | Hardcode Detection | LLM Understanding |
|---------------------|-------------------|-------------------|
| "Giá AAPL?" (Vietnamese) | ❌ Miss | ✅ FAST |
| "详细比较苹果和微软" (Chinese) | ❌ Miss | ✅ EXPERT |
| "NVDA 分析して" (Japanese) | ❌ Miss | ✅ Depends on depth |
| "Quick MSFT check" | ✅ Partial | ✅ FAST |
| "Deep dive on GOOGL" | ✅ Partial | ✅ EXPERT |

---

## 5. Detailed Mode Specifications

### 5.1 ⚡ FAST Mode

```yaml
Mode: FAST
Icon: ⚡
Description: "Respond quicker to act sooner"

Target Metrics:
  latency_p50: 3-4 seconds
  latency_p90: 6 seconds
  llm_calls: 1-2

Configuration:
  # Model Selection
  primary_model: gpt-4o-mini
  fallback_model: gemini-2.0-flash
  provider: openai (primary), gemini (fallback)

  # Classifier
  use_classifier: true
  classifier_model: gpt-4o-mini
  classifier_timeout_ms: 500

  # Tools
  tool_selection: filtered
  max_tools: 8  # Top relevant tools only
  tool_categories:
    - price (always)
    - technical (if technical query)
    - fundamentals (if fundamental query)
    - memory (if recall query)

  # Agent Loop
  max_turns: 2
  turn_timeout_ms: 10000
  total_timeout_ms: 15000

  # Features
  enable_web_search: false
  enable_thinking_display: false  # Instant response feel
  enable_tool_search: false  # Pre-filtered tools

  # System Prompt
  system_prompt_version: condensed
  include_examples: false
  max_system_tokens: 1500

Use Cases:
  - "Giá AAPL hiện tại?"
  - "PE ratio của MSFT?"
  - "RSI là gì?"
  - "NVDA đang bullish hay bearish?"
  - Price lookups
  - Single indicator queries
  - Definitions
  - Quick sentiment checks
```

### 5.2 🧠 EXPERT Mode

```yaml
Mode: EXPERT
Icon: 🧠
Description: "Think further to explore deeper"

Target Metrics:
  latency_p50: 15-25 seconds
  latency_p90: 45 seconds
  llm_calls: 1-6 (no wasted classifier call!)

Configuration:
  # Model Selection
  primary_model: gpt-4o
  fallback_model: gemini-2.5-pro
  alternative_model: claude-sonnet  # If user prefers
  provider: openai (primary), gemini (fallback)

  # Classifier
  use_classifier: false  # Model self-decides!

  # Tools
  tool_selection: all
  max_tools: 31  # Full catalog
  tool_categories: ALL

  # Agent Loop
  max_turns: 6
  turn_timeout_ms: 30000
  total_timeout_ms: 120000

  # Features
  enable_web_search: true
  enable_thinking_display: true  # Show reasoning process
  enable_tool_search: true  # Dynamic discovery
  enable_finance_guru: true  # Advanced calculations

  # System Prompt
  system_prompt_version: full
  include_examples: true
  max_system_tokens: 4000

Use Cases:
  - "So sánh toàn diện NVDA và AMD"
  - "Phân tích kỹ thuật + định giá GOOGL"
  - "Chiến lược đầu tư AI stocks 2026"
  - "Tìm cổ phiếu growth với P/E < 30 và revenue growth > 20%"
  - Multi-step research
  - Comparative analysis
  - Investment strategy
  - Deep fundamental analysis
  - Screening with complex criteria
```

### 5.3 🔄 AUTO Mode (Default)

```yaml
Mode: AUTO
Icon: 🔄
Description: "Adapts models to each query"

Behavior:
  # LLM-based classification (NO hardcode!)
  classifier_type: llm_semantic
  classifier_model: gpt-4o-mini
  classifier_latency: 150-300ms

  # Decision Logic
  routing_to_fast:
    - Simple lookups (price, single indicator)
    - Short queries (< 30 chars, simple structure)
    - Definitions and explanations
    - Greetings and small talk

  routing_to_expert:
    - Multi-symbol queries (≥2 symbols)
    - Comparison requests
    - Strategy and research queries
    - Complex analysis requests
    - Multi-step reasoning needed

  # Context Continuity
  inherit_previous_mode: true  # If previous was EXPERT, stay EXPERT

  # Fallback
  default_on_uncertainty: fast  # Prefer speed when unsure

Cost Analysis:
  # Additional cost for AUTO mode classification
  classifier_cost_per_query: ~$0.0003 (gpt-4o-mini)
  classifier_cost_per_1000: ~$0.30

  # Break-even analysis
  # If classifier saves 50% of queries from EXPERT (expensive)
  # Then overall cost is LOWER with AUTO mode
```

### 5.4 Mode Comparison Matrix

| Feature | ⚡ FAST | 🧠 EXPERT | 🔄 AUTO |
|---------|--------|----------|---------|
| **Primary Model** | gpt-4o-mini | gpt-4o | Depends |
| **Classifier** | ✅ Yes | ❌ No | LLM Semantic |
| **Tools Available** | 8 filtered | 31 all | Depends |
| **Max Turns** | 2 | 6 | Depends |
| **Web Search** | ❌ | ✅ | Depends |
| **Thinking Display** | ❌ | ✅ | Depends |
| **Latency Target** | 3-6s | 15-45s | Variable |
| **Cost/Query** | Low | High | Medium |
| **Best For** | Lookups | Research | General |

---

## 6. Implementation Tasks

### 6.1 Phase 1: Foundation (Week 1) - Priority P0

| Task | File(s) | Effort | Description |
|------|---------|--------|-------------|
| 1.1 | Add `response_mode` to ChatRequest | `src/routers/v2/chat.py` | 2h | Add enum param: "auto"\|"fast"\|"expert" |
| 1.2 | Create ModeConfig dataclass | `src/config/mode_config.py` (new) | 3h | Configuration for each mode |
| 1.3 | Create ModeRouter class | `src/agents/routing/mode_router.py` (new) | 4h | LLM-based complexity classification |
| 1.4 | Add `mode_selected` SSE event | `src/agents/streaming/stream_events.py` | 1h | Notify frontend of mode selection |
| 1.5 | Update API docs | `src/routers/v2/chat.py` | 1h | Document new parameter |
| 1.6 | Unit tests for ModeRouter | `tests/agents/test_mode_router.py` | 3h | Test classification accuracy |

**Deliverable:** `POST /chat/stream?response_mode=auto|fast|expert` endpoint working

### 6.2 Phase 2: Fast Mode Implementation (Week 2) - Priority P0

| Task | File(s) | Effort | Description |
|------|---------|--------|-------------|
| 2.1 | Create condensed system prompt | `src/helpers/system_prompts.py` | 3h | Shorter, focused prompt for FAST |
| 2.2 | Implement tool filtering | `src/agents/tools/tool_filter.py` (new) | 4h | Select top 5-8 relevant tools |
| 2.3 | Create FastModeExecutor | `src/agents/executors/fast_executor.py` (new) | 4h | Optimized execution path |
| 2.4 | Configure 2-turn limit | `src/config/mode_config.py` | 1h | Hard limit enforcement |
| 2.5 | Disable thinking display | `src/agents/streaming/stream_events.py` | 1h | Option to skip reasoning events |
| 2.6 | Performance testing | Test suite | 4h | Verify 3-6s latency target |

**Deliverable:** FAST mode achieving 3-6s latency for simple queries

### 6.3 Phase 3: Expert Mode Implementation (Week 3) - Priority P0

| Task | File(s) | Effort | Description |
|------|---------|--------|-------------|
| 3.1 | Skip classifier logic | `src/handlers/v2/chat_handler.py` | 2h | Bypass classifier for EXPERT |
| 3.2 | Create full system prompt | `src/helpers/system_prompts.py` | 3h | Comprehensive prompt with examples |
| 3.3 | Enable all tools (31) | `src/agents/tools/tool_catalog.py` | 1h | Full catalog loading |
| 3.4 | Configure 6-turn limit | `src/config/mode_config.py` | 1h | Extended reasoning support |
| 3.5 | Enable web search default | `src/config/mode_config.py` | 1h | Auto-enable for EXPERT |
| 3.6 | Quality testing | Test suite | 4h | Compare with current flow |

**Deliverable:** EXPERT mode with model self-deciding (no classifier overhead)

### 6.4 Phase 4: AUTO Mode & LLM Router (Week 4) - Priority P1

| Task | File(s) | Effort | Description |
|------|---------|--------|-------------|
| 4.1 | LLM semantic classifier | `src/agents/routing/mode_router.py` | 4h | Multilingual complexity detection |
| 4.2 | Context continuity logic | `src/agents/routing/mode_router.py` | 2h | Inherit previous mode |
| 4.3 | Caching strategy | `src/agents/routing/mode_router.py` | 2h | Cache similar queries |
| 4.4 | Fallback handling | `src/agents/routing/mode_router.py` | 2h | Default to FAST on uncertainty |
| 4.5 | Multilingual testing | Test suite | 4h | Test Vietnamese, Chinese, Japanese |
| 4.6 | A/B testing setup | Infrastructure | 4h | Compare AUTO vs manual selection |

**Deliverable:** AUTO mode with LLM-based routing (no hardcode keywords)

### 6.5 Phase 5: Provider Abstraction (Week 5) - Priority P2

| Task | File(s) | Effort | Description |
|------|---------|--------|-------------|
| 5.1 | Define UnifiedProvider interface | `src/providers/base_provider.py` | 3h | Abstract interface |
| 5.2 | Refactor OpenAI provider | `src/providers/openai_provider.py` | 4h | Implement interface |
| 5.3 | Refactor Gemini provider | `src/providers/gemini_provider.py` | 4h | Handle thought_signature |
| 5.4 | Implement fallback mechanism | `src/providers/provider_factory.py` | 3h | Auto-switch on failure |
| 5.5 | Provider health checks | `src/providers/health_check.py` (new) | 2h | Monitor provider status |
| 5.6 | Integration testing | Test suite | 4h | Test both providers |

**Deliverable:** Single flow working with GPT/Gemini without bugs

### 6.6 Phase 6: Optimization & Polish (Week 6) - Priority P2

| Task | File(s) | Effort | Description |
|------|---------|--------|-------------|
| 6.1 | Background memory updates | `src/handlers/v2/chat_handler.py` | 3h | Non-blocking post-processing |
| 6.2 | Performance profiling | New tooling | 4h | Identify bottlenecks |
| 6.3 | Monitoring dashboards | Infrastructure | 4h | Track mode usage, latency |
| 6.4 | Documentation update | `/docs/` | 3h | API docs, architecture docs |
| 6.5 | Legacy cleanup | Various | 4h | Remove unused code paths |
| 6.6 | Production readiness | All | 4h | Final testing, staging deploy |

**Deliverable:** Production-ready system with monitoring

---

## 7. Migration Strategy

### 7.1 Rollout Plan

```
Week 1-2: Development
├─ Feature branch development
├─ Unit tests
└─ Local testing

Week 3: Internal Testing
├─ Deploy to staging
├─ Team testing (internal users)
└─ Bug fixes

Week 4: Gradual Rollout
├─ Enable for 10% traffic (feature flag)
├─ Monitor metrics
├─ Enable for 50% traffic
└─ Monitor 24h

Week 5: Full Rollout
├─ Enable for 100% traffic
├─ Monitor 48h
└─ Disable feature flags

Week 6: Cleanup
├─ Remove legacy code
├─ Archive old endpoints
└─ Documentation final
```

### 7.2 Feature Flags

```python
FEATURE_FLAGS = {
    "response_modes_enabled": True,  # Master switch
    "fast_mode_enabled": True,       # Enable FAST mode
    "expert_mode_enabled": True,     # Enable EXPERT mode
    "auto_mode_enabled": True,       # Enable AUTO mode
    "llm_router_enabled": True,      # LLM-based routing
    "background_memory": True,       # Async memory updates
}
```

### 7.3 Backward Compatibility

```python
# Old API (still works)
POST /chat/stream
{
    "query": "Giá AAPL?",
    "model_name": "gpt-4o"
}
# → Defaults to AUTO mode

# New API
POST /chat/stream
{
    "query": "Giá AAPL?",
    "response_mode": "fast",  # NEW
    "model_name": "gpt-4o-mini"
}
```

---

## 8. Risk Assessment

### 8.1 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| FAST mode quality drops | Medium | High | A/B test, fallback to classifier |
| EXPERT mode too slow | Low | Medium | Timeout limits, streaming |
| LLM router inconsistent | Medium | Medium | Cache results, tune prompts |
| Provider abstraction bugs | Medium | High | Comprehensive tests, gradual rollout |
| User confusion with modes | Low | Low | Good UX, tooltips, defaults |
| Cost increase (more LLM calls for AUTO) | Medium | Low | Monitor, optimize caching |

### 8.2 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| FAST mode P50 latency | ≤ 4s | Monitoring dashboard |
| FAST mode P90 latency | ≤ 6s | Monitoring dashboard |
| EXPERT mode quality | ≥ 95% satisfaction | User feedback |
| AUTO mode routing accuracy | ≥ 90% | Manual review sample |
| System availability | ≥ 99.5% | Uptime monitoring |
| Error rate | ≤ 1% | Error tracking |

### 8.3 Rollback Plan

```
IF issues detected:
1. Set feature_flags["response_modes_enabled"] = False
2. All requests fall back to current V2 flow
3. Investigate and fix issues
4. Re-enable with gradual rollout
```

---

## Appendix A: File Structure After Implementation

```
src/
├── routers/v2/
│   ├── chat.py                      # Updated: +response_mode param
│   └── chat_assistant.py            # Updated: mode routing logic
│
├── agents/
│   ├── routing/                     # NEW FOLDER
│   │   ├── __init__.py
│   │   ├── mode_router.py           # LLM-based complexity classification
│   │   └── mode_config.py           # Mode configurations
│   │
│   ├── executors/                   # NEW FOLDER
│   │   ├── __init__.py
│   │   ├── base_executor.py         # Abstract executor
│   │   ├── fast_executor.py         # FAST mode executor
│   │   └── expert_executor.py       # EXPERT mode executor
│   │
│   ├── tools/
│   │   ├── tool_filter.py           # NEW: Tool filtering for FAST mode
│   │   └── ... (existing)
│   │
│   └── unified/
│       └── unified_agent.py         # Updated: configurable by mode
│
├── providers/
│   ├── base_provider.py             # Updated: Stronger interface
│   ├── openai_provider.py           # Updated: Implement interface
│   ├── gemini_provider.py           # Updated: Implement interface
│   └── health_check.py              # NEW: Provider health monitoring
│
├── config/
│   └── mode_config.py               # NEW: Mode configurations
│
└── helpers/
    └── system_prompts.py            # Updated: FAST + EXPERT prompts
```

---

## Appendix B: SSE Event Flow

```javascript
// === MODE SELECTION ===
{"type": "mode_selecting", "query": "Giá AAPL?"}
{"type": "mode_selected", "mode": "fast", "reason": "simple_lookup", "model": "gpt-4o-mini"}

// === FAST MODE FLOW ===
{"type": "classifying"}  // Only in FAST mode
{"type": "classified", "symbols": ["AAPL"], "complexity": "direct"}
{"type": "turn_start", "turn": 1, "max_turns": 2}
{"type": "tool_calls", "tools": [{"name": "GetStockPrice", "arguments": {"symbol": "AAPL"}}]}
{"type": "tool_results", "results": [{"tool": "GetStockPrice", "data": {"price": 185.50}}]}
{"type": "content", "content": "Apple (AAPL) đang giao dịch ở mức $185.50...", "is_final": false}
{"type": "content", "content": "", "is_final": true}
{"type": "done", "mode": "fast", "total_turns": 1, "total_time_ms": 3500}

// === EXPERT MODE FLOW ===
{"type": "mode_selected", "mode": "expert", "reason": "comparison_query", "model": "gpt-4o"}
{"type": "turn_start", "turn": 1, "max_turns": 6}
{"type": "thinking", "content": "I need to compare NVDA and AMD across multiple dimensions..."}
{"type": "tool_calls", "tools": [{"name": "GetFinancialRatios"}, {"name": "GetTechnicalIndicators"}]}
{"type": "tool_results", "results": [...]}
{"type": "turn_start", "turn": 2, "max_turns": 6}
// ... continues with more turns
{"type": "content", "content": "## So sánh NVDA vs AMD...", "is_final": false}
{"type": "done", "mode": "expert", "total_turns": 4, "total_time_ms": 25000}
```

---

## Appendix C: Decision Log

| Date | Decision | Rationale | Owner |
|------|----------|-----------|-------|
| 2026-01-22 | Use LLM for AUTO mode routing | Industry standard, multilingual support | Team |
| 2026-01-22 | Skip classifier for EXPERT mode | Large models self-decide, reduces latency | Team |
| 2026-01-22 | Default to AUTO mode | Best UX, adapts to user needs | Team |
| 2026-01-22 | Background memory updates | Reduce perceived latency | Team |
| 2026-01-22 | 3-phase architecture | Simpler than 7-phase, easier to maintain | Team |

---

**Document Version:** 2.0
**Last Updated:** 2026-01-22
**Next Review:** After Phase 1 implementation

---

## References

1. [Claude AI Extended Thinking](https://claude-ai.chat/blog/extended-thinking-mode/)
2. [Anthropic Extended Thinking Docs](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)
3. [OpenAI GPT-5 Introduction](https://openai.com/index/introducing-gpt-5/)
4. [GPT-5 Router Explained](https://rankstudio.net/articles/en/gpt-5-router-explained)
5. [GPT-5.1 Two Models, Automatic Routing](https://www.datacamp.com/blog/gpt-5-1)
6. [AI Agent Routing Best Practices](https://www.patronus.ai/ai-agent-development/ai-agent-routing)
7. [Intent Classification 2025 Techniques](https://labelyourdata.com/articles/machine-learning/intent-classification)
8. [Intent Detection in the Age of LLMs](https://arxiv.org/html/2410.01627v1)
