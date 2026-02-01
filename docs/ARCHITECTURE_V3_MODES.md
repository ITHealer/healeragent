# HealerAgent AI Chatbot Architecture - V3 Mode System

## Table of Contents
1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Gap Analysis](#2-gap-analysis)
3. [Target Architecture: 3-Mode System](#3-target-architecture)
4. [Detailed Implementation Plan](#4-implementation-plan)
5. [File System Structure](#5-file-system-structure)
6. [SSE Event Protocol](#6-sse-event-protocol)
7. [Migration Strategy](#7-migration-strategy)

---

## 1. Current Architecture Analysis

### 1.1 Two Parallel Pipelines (Problem)

HealerAgent currently has **TWO separate chat pipelines** that evolved independently:

#### Pipeline A: `src/routers/v2/chat.py` + `ChatHandler`
```
POST /chat/complete  (non-streaming)
POST /chat/stream    (streaming)

Flow: 7-Phase Upfront Planning Pipeline
  Phase 1: Context Loading (Core Memory + Summary + History + Working Memory)
  Phase 2: Planning (Classify → Load Tools → Create TaskPlan) [2 LLM calls]
  Phase 3: Memory Search
  Phase 4: Tool Execution (TaskExecutor with retry)
  Phase 5: Context Assembly (NO LLM)
  Phase 6: LLM Response Generation (streaming)
  Phase 7: Post-Processing (background memory updates)

Model: Always uses client-provided model_name
Mode: Has `response_mode` param in ChatRequest but NEVER uses it
Router: ModeRouter exists but is DISCONNECTED from this pipeline
```

#### Pipeline B: `src/routers/v2/chat_assistant.py` + `UnifiedAgent`
```
POST /chat-assistant/chat/v2  (streaming SSE)
POST /chat-assistant/chat/v3  (delegates to v2, enables Finance Guru)

Flow: ChatGPT-style Agentic Loop
  Phase 0: Process images (multimodal)
  Phase 1: Session Start + Working Memory Setup
  Phase 1.5: Context Loading via ContextBuilder
  Phase 1.6: Context Compaction Check
  Phase 2: Intent Classification [1 LLM call]
  Phase 3: Agent Loop (UnifiedAgent sees ALL tools, iterative)
  Phase 4: Done + Charts
  Post: LEARN Hook + SAVE + SUMMARY

Model: Client-provided, agent decides tools iteratively
Mode: `mode` param exists but routes between "normal" vs "deep_research" (legacy)
Router: Uses `ModeRouter` but for normal/deep_research, NOT for Instant/Thinking/Auto
```

### 1.2 Key Components Map

```
src/
├── routers/v2/
│   ├── chat.py                          # Pipeline A endpoints
│   └── chat_assistant.py               # Pipeline B endpoints (v2, v3)
│
├── handlers/v2/
│   ├── chat_handler.py                  # Pipeline A: 7-phase ChatHandler
│   ├── mode_router.py                   # Legacy mode router (normal/deep_research)
│   └── normal_mode_chat_handler.py      # Legacy normal mode handler
│
├── agents/
│   ├── unified/                         # Pipeline B: ChatGPT-style agent
│   │   └── unified_agent.py             # UnifiedAgent (agentic loop)
│   ├── classification/
│   │   ├── unified_classifier.py        # Pipeline A classifier
│   │   └── intent_classifier.py         # Pipeline B classifier (better)
│   ├── planning/
│   │   └── planning_agent.py            # 3-stage planning (Pipeline A only)
│   ├── action/
│   │   └── task_executor.py             # Task execution (Pipeline A only)
│   ├── routing/
│   │   └── mode_router.py              # NEW ModeRouter (Fast/Auto/Expert) - DISCONNECTED
│   ├── streaming/
│   │   ├── streaming_chat_handler.py    # StreamingChatHandler (Pipeline A streaming)
│   │   ├── stream_events.py             # Stream event types
│   │   └── agent_tree.py               # Agent tree tracking
│   ├── memory/
│   │   ├── core_memory.py               # Core Memory (Persona + Human)
│   │   ├── recursive_summary.py         # Recursive Summary Manager
│   │   ├── working_memory_integration.py # Working Memory (per-request)
│   │   └── memory_update_agent.py       # Background memory updates
│   └── tools/
│       ├── registry.py                  # Tool Registry (31 tools)
│       ├── tool_loader.py               # Tool loading
│       ├── base.py                      # Base tool class
│       └── {category}/*.py              # 31 atomic tools
│
├── config/
│   └── mode_config.py                   # ModeConfig (Fast/Auto/Expert) - EXISTS but unused
│
├── services/
│   ├── streaming_event_service.py       # Pipeline B: SSE event emitter
│   ├── context_builder.py               # Pipeline B: ContextBuilder
│   ├── context_management_service.py    # Context compaction
│   ├── conversation_compactor.py        # Conversation compression
│   ├── think_tool_service.py            # Think Tool
│   ├── tool_search_service.py           # Tool search/discovery
│   └── memory_search_service.py         # Memory vector search
│
└── providers/
    └── provider_factory.py              # Multi-LLM provider routing
```

### 1.3 What Currently Works Well

1. **UnifiedAgent (Pipeline B)** - ChatGPT-style agentic loop with iterative tool calling
2. **IntentClassifier** - Single LLM call classification with symbol validation
3. **Memory System** - 3-tier (Core + Summary + Working Memory) with cross-turn continuity
4. **Tool System** - 31 atomic tools with registry, schemas, circuit breakers
5. **Streaming Events** - Rich SSE protocol with ThinkingTimeline, tool progress
6. **Context Compaction** - Auto-compress when approaching token limits
7. **LEARN Hook** - Background memory updates post-execution

---

## 2. Gap Analysis

### 2.1 Critical Gaps

| # | Gap | Impact | Severity |
|---|-----|--------|----------|
| 1 | **ModeRouter disconnected** - `mode_router.py` and `mode_config.py` exist but are never integrated into any pipeline | No mode-based routing works | CRITICAL |
| 2 | **No Instant Mode path** - No fast path that skips planning and uses nano model | All queries go through full pipeline (~5-15s) | CRITICAL |
| 3 | **No auto-escalation** - When Instant mode gets insufficient data, no mechanism to escalate to Thinking mode | User gets poor answers for complex queries in Instant mode | HIGH |
| 4 | **Model selection ignores mode** - Client always sends model_name, ModeConfig's per-mode model selection never applies | Wrong model for query complexity | HIGH |
| 5 | **Two pipelines = maintenance burden** - Pipeline A (ChatHandler) and Pipeline B (UnifiedAgent) duplicate context loading, memory, saving | Bug fixes needed in 2 places | MEDIUM |
| 6 | **ThinkingTimeline not mode-aware** - Always shows thinking UI regardless of mode | Instant mode shows unnecessary thinking | MEDIUM |
| 7 | **No mode SSE events** - `mode_selecting` and `mode_selected` events defined in docs but never emitted | Frontend can't show mode selection | MEDIUM |

### 2.2 Architecture Comparison with Industry Leaders

| Feature | ChatGPT 5.2 | Claude AI | HealerAgent Current | HealerAgent Target |
|---------|-------------|-----------|---------------------|-------------------|
| Mode selection | Instant/Thinking/Auto | Extended Thinking toggle | N/A (mode exists but disconnected) | Instant/Thinking/Auto |
| Auto-routing | LLM classifies complexity | Model self-decides budget | ModeRouter exists but unused | LLM semantic classification |
| Model per mode | Different models | Single model, variable thinking | Single model always | gpt-4.1-nano / gpt-4.1 / gpt-4o |
| Thinking display | "Thought for Xs" timeline | Thinking block before response | ThinkingTimeline exists | Mode-aware ThinkingTimeline |
| Escalation | Auto when needed | N/A | None | Instant -> Thinking auto-escalate |
| Tool selection | Agent decides | Agent decides | Agent decides (Pipeline B) or planned (Pipeline A) | Mode-aware: filtered (Instant) or all (Thinking) |
| Streaming | Progressive | Progressive | SSE with events | Enhanced SSE with mode events |

---

## 3. Target Architecture

### 3.1 Unified V3 Pipeline with 3-Mode System

```
┌─────────────────────────────────────────────────────────────────────┐
│                    POST /chat-assistant/chat/v3                       │
│                                                                       │
│  ChatRequest { query, session_id, response_mode, model_name, ... }   │
│                                                                       │
│  response_mode: "instant" | "thinking" | "auto" (default)            │
└─────────────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Phase 0: Setup & Context Loading                   │
│                                                                       │
│  1. Session Setup + Working Memory                                    │
│  2. ContextBuilder: Core Memory + Summary + History + WM Symbols      │
│  3. Context Compaction (if needed)                                    │
│  4. Process Images (if multimodal)                                    │
└─────────────────────────────────────┬─────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Phase 1: Mode Resolution                           │
│                                                                       │
│  ┌──────────────────┐    ┌──────────────────┐                        │
│  │ User chose       │    │ User chose       │                        │
│  │ "instant"?       │    │ "thinking"?      │                        │
│  │ → Use INSTANT    │    │ → Use THINKING   │                        │
│  └──────────────────┘    └──────────────────┘                        │
│                                                                       │
│  ┌──────────────────────────────────────────┐                        │
│  │ User chose "auto" (default)?             │                        │
│  │                                           │                        │
│  │  1. Quick heuristics (< 1ms):             │                        │
│  │     - Short query + 0-1 symbols → INSTANT │                        │
│  │     - Multiple symbols → THINKING         │                        │
│  │     - Context continuity → inherit prev   │                        │
│  │                                           │                        │
│  │  2. If undecided → LLM classify (200ms): │                        │
│  │     - gpt-4.1-nano with JSON output       │                        │
│  │     - Returns: simple/complex + confidence│                        │
│  │     - Cache result (5min TTL)             │                        │
│  │                                           │                        │
│  │  3. Map: simple → INSTANT, complex → THINKING │                   │
│  └──────────────────────────────────────────┘                        │
│                                                                       │
│  SSE: emit mode_selecting → mode_selected                            │
│  Result: ModeConfig { model, max_turns, features, ... }              │
└─────────────────────────────────────┬─────────────────────────────────┘
                                      │
                          ┌───────────┴───────────┐
                          │                       │
                          ▼                       ▼
┌────────────────────────────────┐ ┌────────────────────────────────────┐
│      INSTANT MODE PATH         │ │       THINKING MODE PATH           │
│                                │ │                                    │
│  Model: gpt-4.1-nano          │ │  Model: gpt-4.1 or gpt-4o         │
│  Target: < 3 seconds          │ │  Target: 10-30 seconds             │
│  Max turns: 2                  │ │  Max turns: 6                      │
│  Thinking display: OFF         │ │  Thinking display: ON              │
│  Web search: OFF               │ │  Web search: ON (if needed)        │
│  Tool search mode: OFF         │ │  Tool search mode: ON              │
│  Finance Guru: OFF             │ │  Finance Guru: ON                  │
│                                │ │                                    │
│  Phase 2: IntentClassifier     │ │  Phase 2: IntentClassifier         │
│    (1 LLM call, nano model)    │ │    (1 LLM call, standard model)    │
│                                │ │                                    │
│  Phase 3: Agent Loop           │ │  Phase 3: Planning Agent           │
│    (1-2 tool calls max)        │ │    (Create detailed TaskPlan)      │
│    (No evaluation loop)        │ │                                    │
│    (Filtered tools: 8-10)      │ │  Phase 4: Evaluation Loop          │
│                                │ │    (Execute → Evaluate → Retry)    │
│  Phase 4: Stream Response      │ │    (Max 3 iterations)              │
│    (Direct, no thinking UI)    │ │    (All 31+ tools available)       │
│                                │ │                                    │
│  ┌──────────────────────────┐  │ │  Phase 5: Stream Response          │
│  │ ESCALATION CHECK         │  │ │    (With ThinkingTimeline UI)      │
│  │                          │  │ │                                    │
│  │ After Phase 3, check:    │  │ │  Phase 6: Post-Processing          │
│  │ - Tool errors > 50%?     │  │ │    (LEARN + SAVE + SUMMARY)        │
│  │ - No data returned?      │  │ │                                    │
│  │ - Confidence < 0.5?      │  │ └────────────────────────────────────┘
│  │                          │  │
│  │ If YES → escalate to     │  │
│  │ THINKING mode silently   │  │
│  │ (user sees better result)│  │
│  └──────────────────────────┘  │
│                                │
│  Phase 5: Post-Processing      │
│    (LEARN + SAVE + SUMMARY)    │
└────────────────────────────────┘
```

### 3.2 Mode Configurations (Target)

```
┌──────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Feature          │ INSTANT MODE ⚡       │ THINKING MODE 🧠     │ AUTO MODE 🔄         │
├──────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ Model (primary)  │ gpt-4.1-nano         │ gpt-4.1              │ Depends on classify  │
│ Model (fallback) │ gpt-4o-mini          │ gpt-4o               │ gpt-4.1-nano         │
│ Target latency   │ < 3 seconds          │ 10-30 seconds        │ 3-30 seconds         │
│ Max agent turns  │ 2                    │ 6                    │ 2-6                  │
│ Tool count       │ 8-10 (filtered)      │ 31+ (all)            │ Depends on classify  │
│ Web search       │ OFF                  │ ON                   │ ON if complex        │
│ Thinking display │ OFF                  │ ON (timeline)        │ Depends on classify  │
│ Think Tool       │ OFF                  │ ON                   │ ON if complex        │
│ Finance Guru     │ OFF                  │ ON                   │ ON if complex        │
│ Tool search mode │ OFF (direct tools)   │ ON (dynamic discover)│ Depends on classify  │
│ Evaluation loop  │ NO                   │ YES (max 3)          │ Depends on classify  │
│ Escalation       │ → THINKING if needed │ N/A                  │ Auto                 │
│ System prompt    │ Condensed (1.5K tok) │ Full (4K tok)        │ Depends              │
│ Classifier model │ Same nano model      │ Not needed           │ gpt-4.1-nano (200ms) │
└──────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘
```

### 3.3 Instant Mode - Detailed Flow

```
User: "Giá AAPL?" + mode=instant

1. [0ms] Setup + Context Loading (parallel)
   ├── Load Working Memory symbols
   ├── Load Core Memory
   └── Load last 5 messages (not 10)

2. [50ms] Mode Resolution: INSTANT (explicit or auto-classified)
   └── SSE: mode_selected { mode: "instant", model: "gpt-4.1-nano" }

3. [100ms] IntentClassifier (nano model, 1 LLM call)
   ├── symbols: ["AAPL"]
   ├── requires_tools: true
   ├── complexity: direct
   └── SSE: classified { symbols: ["AAPL"], complexity: "direct" }

4. [200ms] Agent Loop (max 2 turns)
   ├── Turn 1: getStockPrice(symbol="AAPL")
   ├── SSE: tool_calls → tool_results
   └── Turn 2: Generate response (nano model, streaming)

5. [2000ms] Stream Response
   └── SSE: content chunks (NO thinking timeline)

6. [2500ms] Post-Processing (background)
   └── SAVE conversation + LEARN memory update

Total: ~2.5 seconds
LLM calls: 2 (classify + response)
Tool calls: 1
```

### 3.4 Thinking Mode - Detailed Flow

```
User: "Phân tích toàn diện AAPL - technical, fundamental, risk, so sánh thị trường" + mode=thinking

1. [0ms] Setup + Context Loading
   ├── Load full context (10 messages + summary + core memory)
   └── Context compaction if needed

2. [50ms] Mode Resolution: THINKING
   └── SSE: mode_selected { mode: "thinking", model: "gpt-4.1" }

3. [200ms] IntentClassifier (standard model)
   ├── symbols: ["AAPL"]
   ├── requires_tools: true
   ├── complexity: agent_loop
   └── SSE: classified + thinking_timeline starts

4. [400ms] Planning Agent creates detailed TaskPlan
   ├── Task 1: getStockPrice + getTechnicalIndicators
   ├── Task 2: getFinancialRatios + getGrowthMetrics
   ├── Task 3: assessRisk + getVolumeProfile
   ├── Task 4: getMarketIndices + getSectorPerformance
   └── SSE: planning_progress → planning_complete

5. [2000ms] Execution Loop (with evaluation)
   ├── Execute all tasks (parallel where possible)
   ├── SSE: tool_start → tool_complete (for each tool)
   ├── Evaluation: "Data sufficient?" (1 LLM call)
   ├── If insufficient: execute additional tools (up to 3 iterations)
   └── SSE: thinking_timeline steps

6. [5000ms] Stream Response (gpt-4.1, full system prompt)
   ├── SSE: thinking_summary { duration: "Thought for 8s", steps: [...] }
   └── SSE: content chunks

7. [15000ms] Post-Processing
   └── SAVE + LEARN + SUMMARY

Total: ~15 seconds
LLM calls: 3-5 (classify + plan + evaluate + response)
Tool calls: 6-12
```

### 3.5 Auto Mode - Detailed Flow

```
User: "NVDA" + mode=auto (default)

1. [0ms] Setup + Context Loading

2. [50ms] Mode Resolution: AUTO
   ├── SSE: mode_selecting { method: "auto" }
   ├── Quick heuristics: short query (4 chars) + 1 symbol → INSTANT
   ├── Result: INSTANT (confidence: 0.85)
   └── SSE: mode_selected { mode: "instant", reason: "very_short_query" }

3. [100ms] → Follow INSTANT path

---

User: "So sánh NVDA và AMD về mặt fundamental" + mode=auto

1. [0ms] Setup + Context Loading

2. [50ms] Mode Resolution: AUTO
   ├── SSE: mode_selecting { method: "auto" }
   ├── Quick heuristics: 2 symbols detected → THINKING
   ├── Result: THINKING (confidence: 0.90)
   └── SSE: mode_selected { mode: "thinking", reason: "multi_symbol_detected" }

3. [200ms] → Follow THINKING path
```

### 3.6 Escalation Mechanism

```
Instant Mode executes...
Agent Loop completes with:
  - 2/3 tool calls failed
  - No meaningful data returned
  - OR response confidence < 0.5

Escalation triggered:
  1. Log: "[ESCALATION] Instant → Thinking | reason: insufficient_data"
  2. SSE: mode_escalated { from: "instant", to: "thinking", reason: "..." }
  3. Re-execute with THINKING mode config:
     - Switch model: gpt-4.1-nano → gpt-4.1
     - Expand tools: 8 → 31
     - Increase turns: 2 → 6
     - Enable ThinkingTimeline
  4. User sees: seamless better response (slightly longer wait)
```

---

## 4. Implementation Plan

### Phase 1: Mode Infrastructure (Foundation)

#### Task 1.1: Rename Response Modes to Match User Terms
**File:** `src/config/mode_config.py`
**Changes:**
- Rename `ResponseMode.FAST` → `ResponseMode.INSTANT`
- Rename `ResponseMode.EXPERT` → `ResponseMode.THINKING`
- Keep `ResponseMode.AUTO`
- Update `INSTANT_MODE_CONFIG`, `THINKING_MODE_CONFIG`, `AUTO_MODE_CONFIG`
- Update models: gpt-4o-mini → gpt-4.1-nano (instant), gpt-4o → gpt-4.1 (thinking)
- Add `evaluation_loop` field to ModeConfig
- Add `escalation_enabled` field to ModeConfig
- Add `max_history_messages` field (5 for instant, 10 for thinking)

#### Task 1.2: Integrate ModeRouter into V3 Pipeline
**File:** `src/agents/routing/mode_router.py`
**Changes:**
- Update `QueryComplexity` enum: SIMPLE/COMPLEX
- Update classification prompt to support Instant/Thinking terminology
- Update heuristics for better accuracy
- Add escalation detection method: `should_escalate(tool_results, error_rate, confidence)`
- Add method: `get_mode_config(mode_result) → ModeConfig`

#### Task 1.3: Create Mode-Aware Configuration Resolver
**New File:** `src/services/mode_resolver_service.py`
**Purpose:** Single entry point for resolving mode → config, including:
- Explicit mode selection (user chose instant/thinking)
- Auto classification (LLM + heuristics)
- Escalation decisions
- Override model/features per mode

```python
class ModeResolverService:
    async def resolve_mode(
        query: str,
        user_mode: str,  # "instant" | "thinking" | "auto"
        context: ModeContext
    ) -> ResolvedMode:
        """Returns: mode, config, model_name, features"""

    def should_escalate(
        mode: str,
        tool_results: List,
        error_count: int,
        confidence: float
    ) -> EscalationDecision:
        """Decide if instant should escalate to thinking"""
```

### Phase 2: V3 Pipeline Refactor

#### Task 2.1: Create Unified V3 Chat Handler
**New File:** `src/handlers/v3/chat_handler.py`
**Purpose:** Single handler that unifies both pipelines with mode awareness

```python
class V3ChatHandler:
    """
    Unified V3 Chat Handler with 3-Mode System.

    Replaces: ChatHandler (Pipeline A) + stream_chat_v2 (Pipeline B)

    Flow:
    1. Setup + Context Loading
    2. Mode Resolution
    3. Intent Classification (mode-aware model)
    4. Agent Execution (mode-aware config)
    5. Escalation Check (instant only)
    6. Response Streaming (mode-aware display)
    7. Post-Processing
    """
```

**Key Design Decisions:**
- Base on Pipeline B (UnifiedAgent) as it's more mature
- Add mode-aware wrapping layer on top
- Use IntentClassifier (not PlanningAgent) for both modes
- For Thinking mode: add evaluation loop after agent execution
- For Instant mode: limit agent turns and disable features

#### Task 2.2: Create V3 Route
**New File:** `src/routers/v3/chat.py`
**Changes:**
- New router with prefix `/v3`
- Endpoint: `POST /v3/chat/stream`
- Updated `ChatRequest` with `response_mode` field properly typed
- Delegate to V3ChatHandler

#### Task 2.3: Update ChatRequest Schema
**File:** `src/routers/v3/chat.py` (new)
**Changes:**
```python
class V3ChatRequest(BaseModel):
    query: str
    session_id: Optional[str]
    response_mode: Literal["instant", "thinking", "auto"] = "auto"
    model_name: Optional[str] = None  # None = mode decides
    provider_type: str = "openai"
    # ... other fields
```

**Key:** `model_name` is now Optional. If None, the mode config decides the model. If provided, it overrides (for advanced users).

### Phase 3: Instant Mode Implementation

#### Task 3.1: Instant Mode Agent Configuration
**File:** `src/handlers/v3/chat_handler.py`
**Changes:**
- When mode=INSTANT:
  - Use gpt-4.1-nano for classification AND response
  - Limit UnifiedAgent to max_turns=2
  - Disable tool_search_mode (use direct tool names)
  - Disable web_search, think_tool, finance_guru
  - Reduce conversation history to last 5 messages
  - Use condensed system prompt
  - Skip ThinkingTimeline emission

#### Task 3.2: Tool Filtering for Instant Mode
**New File:** `src/services/tool_filter_service.py`
**Purpose:** Filter tool catalog based on mode config
```python
class ToolFilterService:
    def filter_for_instant(
        all_tools: List[ToolSchema],
        categories: List[str] = ["price", "technical", "fundamentals", "news"]
    ) -> List[ToolSchema]:
        """Return only essential tools for instant mode"""
        # Max 10 tools
        # Prioritize: GetStockPrice, GetTechnicalIndicators, GetStockNews, etc.
```

#### Task 3.3: Escalation Logic
**File:** `src/handlers/v3/chat_handler.py`
**Changes:**
- After instant mode agent completes, evaluate:
  - `error_rate = failed_tools / total_tools`
  - `has_meaningful_data = any(result.success for result in results)`
  - `response_confidence` from agent output
- If escalation triggered:
  - Re-run with thinking mode config
  - Emit `mode_escalated` SSE event
  - Use accumulated tool results (don't re-fetch successful ones)

### Phase 4: Thinking Mode Implementation

#### Task 4.1: Evaluation Loop Service
**New File:** `src/services/evaluation_service.py`
**Purpose:** "Data sufficient?" check after tool execution

```python
class EvaluationService:
    async def evaluate_completeness(
        query: str,
        intent: IntentResult,
        tool_results: List[Dict],
        iteration: int,
        max_iterations: int = 3
    ) -> EvaluationResult:
        """
        Evaluate if gathered data is sufficient.
        Returns: sufficient (bool), missing_data (list), suggested_tools (list)
        Uses 1 LLM call with nano model for speed.
        """
```

#### Task 4.2: Thinking Mode Extended Pipeline
**File:** `src/handlers/v3/chat_handler.py`
**Changes:**
- When mode=THINKING:
  - Use gpt-4.1 for classification and response
  - Enable ALL features (web search, think tool, finance guru, tool search)
  - After agent loop completes, run EvaluationService
  - If evaluation says insufficient: execute suggested tools, re-evaluate
  - Max 3 evaluation iterations
  - Enable full ThinkingTimeline with detailed steps
  - Use full system prompt with examples

#### Task 4.3: Enhanced ThinkingTimeline for Thinking Mode
**File:** `src/agents/streaming/stream_events.py`
**Changes:**
- Add `mode_selecting` event type
- Add `mode_selected` event type
- Add `mode_escalated` event type
- Add `evaluation_start` / `evaluation_complete` event types
- Make ThinkingTimeline mode-aware: skip emissions for instant mode

### Phase 5: SSE Protocol Enhancement

#### Task 5.1: New SSE Events for Mode System
**File:** `src/services/streaming_event_service.py`
**Changes:**
```python
# New event emitters
def emit_mode_selecting(method: str) -> str
def emit_mode_selected(mode: str, reason: str, model: str, confidence: float) -> str
def emit_mode_escalated(from_mode: str, to_mode: str, reason: str) -> str
def emit_evaluation_start(iteration: int) -> str
def emit_evaluation_complete(sufficient: bool, missing: list) -> str
```

#### Task 5.2: Mode-Aware Event Filtering
**File:** `src/handlers/v3/chat_handler.py`
**Changes:**
- In instant mode: suppress thinking_timeline events, evaluation events
- In thinking mode: emit all events
- In auto mode: emit mode_selecting before classification, then follow resolved mode

### Phase 6: Integration & Testing

#### Task 6.1: Register V3 Router
**File:** `src/main.py`
**Changes:**
- Import and include v3 router
- Keep v2 routes for backward compatibility

#### Task 6.2: Update Provider Factory for Mode Models
**File:** `src/providers/provider_factory.py`
**Changes:**
- Add method: `get_model_for_mode(mode_config: ModeConfig) -> str`
- Support gpt-4.1-nano, gpt-4.1, gpt-4o model routing

#### Task 6.3: Integration Tests
**New Files:** `tests/v3/`
- test_mode_resolution.py
- test_instant_mode.py
- test_thinking_mode.py
- test_auto_mode.py
- test_escalation.py

---

## 5. File System Structure

### New Files to Create

```
src/
├── routers/
│   └── v3/
│       ├── __init__.py
│       └── chat.py                     # V3 chat endpoint with response_mode
│
├── handlers/
│   └── v3/
│       ├── __init__.py
│       └── chat_handler.py             # Unified V3 ChatHandler (3-mode)
│
├── services/
│   ├── mode_resolver_service.py        # Mode resolution + escalation decisions
│   ├── tool_filter_service.py          # Filter tools based on mode
│   └── evaluation_service.py           # "Data sufficient?" evaluation loop
│
└── config/
    └── mode_config.py                  # Updated: INSTANT/THINKING/AUTO configs
```

### Files to Modify

```
src/
├── config/
│   └── mode_config.py                  # Rename FAST→INSTANT, EXPERT→THINKING
│
├── agents/
│   └── routing/
│       └── mode_router.py              # Update for instant/thinking terminology
│
├── services/
│   └── streaming_event_service.py      # Add mode_selecting/selected/escalated events
│
├── agents/
│   └── streaming/
│       └── stream_events.py            # Add new event types
│
├── providers/
│   └── provider_factory.py             # Add get_model_for_mode()
│
└── main.py                             # Register v3 router
```

### Files NOT Modified (Backward Compatible)

```
src/routers/v2/chat.py                  # Keep as-is (Pipeline A)
src/routers/v2/chat_assistant.py        # Keep as-is (Pipeline B: v2, v3 legacy)
src/handlers/v2/chat_handler.py         # Keep as-is (Pipeline A handler)
src/agents/unified/unified_agent.py     # Keep as-is (reused by V3)
src/agents/classification/intent_classifier.py  # Keep as-is (reused by V3)
```

---

## 6. SSE Event Protocol (V3)

### Complete Event Flow - Auto Mode

```
1. SSE: session_start    { version: "v3", session_id: "..." }
2. SSE: mode_selecting   { method: "auto", query_length: 45 }
3. SSE: mode_selected    { mode: "instant", reason: "single_symbol_simple", model: "gpt-4.1-nano", confidence: 0.85 }
4. SSE: classifying      { }
5. SSE: classified       { query_type: "stock_specific", symbols: ["AAPL"], ... }
6. SSE: turn_start       { turn: 1, max_turns: 2 }
7. SSE: tool_calls       { tools: [{ name: "getStockPrice", arguments: { symbol: "AAPL" } }] }
8. SSE: tool_results     { results: [{ name: "getStockPrice", success: true, ... }] }
9. SSE: content          { content: "AAPL is currently..." }
10. SSE: content         { content: "trading at $198.50..." }
... more content chunks
11. SSE: thinking_summary { total_duration_ms: 2100, steps: [...] }
12. SSE: done            { total_turns: 1, total_tool_calls: 1, total_time_ms: 2500 }
13. [DONE]
```

### Complete Event Flow - Thinking Mode

```
1. SSE: session_start    { version: "v3" }
2. SSE: mode_selected    { mode: "thinking", reason: "explicit_user_selection", model: "gpt-4.1" }
3. SSE: classifying      { }
4. SSE: thinking_step    { phase: "classification", action: "Analyzing query..." }
5. SSE: classified       { ... }
6. SSE: thinking_step    { phase: "tool_selection", action: "Agent starting with 31 tools" }
7. SSE: turn_start       { turn: 1 }
8. SSE: tool_calls       { tools: [{ name: "getStockPrice" }, { name: "getTechnicalIndicators" }] }
9. SSE: thinking_step    { phase: "tool_execution", action: "Calling 2 tools", details: "..." }
10. SSE: tool_results    { results: [...] }
11. SSE: turn_start      { turn: 2 }
12. SSE: tool_calls      { tools: [{ name: "getFinancialRatios" }, { name: "assessRisk" }] }
13. SSE: tool_results    { results: [...] }
14. SSE: evaluation      { iteration: 1, sufficient: true }
15. SSE: content         { content: "..." }
... more content
16. SSE: thinking_summary { total_duration_ms: 12000, steps: [...] }
17. SSE: sources         { citations: [...] }  // if web search used
18. SSE: done            { total_turns: 3, total_tool_calls: 5, mode: "thinking" }
19. [DONE]
```

### Escalation Event Flow

```
... (instant mode events)
7. SSE: tool_results     { results: [{ success: false }, { success: false }] }
8. SSE: mode_escalated   { from: "instant", to: "thinking", reason: "tool_error_rate_high" }
9. SSE: thinking_step    { phase: "escalation", action: "Upgrading to deeper analysis..." }
... (continues with thinking mode events)
```

---

## 7. Migration Strategy

### Phase 1: Non-Breaking (Add V3 alongside V2)
- Create `/v3/chat/stream` endpoint
- Keep all V2 endpoints unchanged
- V3 reuses V2 components (UnifiedAgent, IntentClassifier, etc.)
- Frontend can opt-in to V3 by changing endpoint

### Phase 2: Feature Parity
- V3 supports all V2 features (images, reply_to, ui_context, etc.)
- V3 adds mode system on top
- V2 remains as fallback

### Phase 3: Deprecation (Future)
- Mark V2 endpoints as deprecated
- Frontend migrates to V3
- Remove V2 after full migration

### Backward Compatibility Rules
1. `response_mode="auto"` is default - behaves like V2 for simple queries
2. If `model_name` is explicitly provided, it overrides mode-based model selection
3. All existing SSE events remain unchanged - new events are additions
4. UnifiedAgent, IntentClassifier, tools - all reused, not rewritten
5. Memory system (Core + Summary + Working) - unchanged

---

## Summary: Implementation Priority Order

```
Priority 1 (Foundation):
  1.1 Update mode_config.py (rename + new fields)
  1.2 Update mode_router.py (integrate)
  1.3 Create mode_resolver_service.py

Priority 2 (Core Pipeline):
  2.1 Create V3 ChatHandler
  2.2 Create V3 Route
  2.3 Update ChatRequest schema

Priority 3 (Instant Mode):
  3.1 Instant mode agent config
  3.2 Tool filtering
  3.3 Escalation logic

Priority 4 (Thinking Mode):
  4.1 Evaluation service
  4.2 Extended pipeline
  4.3 Enhanced ThinkingTimeline

Priority 5 (SSE Protocol):
  5.1 New SSE events
  5.2 Mode-aware filtering

Priority 6 (Integration):
  6.1 Register router
  6.2 Provider factory update
  6.3 Tests
```

Each task is independent enough to be implemented and tested individually. The dependency chain is:
`Phase 1 → Phase 2 → Phase 3 & 4 (parallel) → Phase 5 → Phase 6`
