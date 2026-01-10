# HEALERAGENT GAP ANALYSIS - Architecture Document V2 vs Current Implementation

## Generated: 2026-01-10
## Purpose: Compare ARCHITECTURE_DESIGN_V2.md with current codebase to identify gaps and create prioritized task list

---

## EXECUTIVE SUMMARY

Based on detailed analysis of the architecture document and current codebase, the implementation is **~75% complete**. Key gaps are:

1. **ContextBuilder Service** - NOT implemented as centralized service
2. **Thinking Display Timeline** - SSE events exist but need UI timeline enhancement
3. **Circuit Breaker Pattern** - NOT implemented
4. **Graceful Degradation** - Partial implementation

---

## SECTION 1: IMPLEMENTED FEATURES (What's Working)

### 1.1 Core Agent Architecture
| Feature | Status | Location |
|---------|--------|----------|
| UnifiedAgent with complexity-based execution | IMPLEMENTED | `src/agents/unified/unified_agent.py` |
| SIMPLE/MEDIUM/COMPLEX strategies | IMPLEMENTED | UnifiedAgent:219-259 |
| Agent loop with max_turns | IMPLEMENTED | run_stream_with_all_tools() |
| Tool selection strategies | IMPLEMENTED | _build_agent_messages() |
| Skill-based domain prompts | IMPLEMENTED | SkillRegistry |

### 1.2 Memory System (4-Tier)
| Tier | Status | Location |
|------|--------|----------|
| Core Memory (YAML) | IMPLEMENTED | `src/agents/memory/core_memory.py` |
| Recursive Summary (PostgreSQL) | IMPLEMENTED | `src/agents/memory/recursive_summary.py` |
| Working Memory (Redis) | IMPLEMENTED | `src/agents/memory/working_memory_integration.py` |
| Recent Conversations (PostgreSQL) | IMPLEMENTED | chat_assistant.py Phase 1.5 |

### 1.3 SSE Streaming Events
| Event Type | Status | Location |
|------------|--------|----------|
| THINKING_START/DELTA/END | IMPLEMENTED | `stream_events.py:26-28` |
| TOOL_START/PROGRESS/COMPLETE | IMPLEMENTED | `stream_events.py:31-33` |
| PLANNING_START/PROGRESS/COMPLETE | IMPLEMENTED | `stream_events.py:36-38` |
| LLM_THOUGHT/DECISION/ACTION | IMPLEMENTED | `stream_events.py:52-54` |
| AGENT_NODE (tree tracking) | IMPLEMENTED | `stream_events.py:57` |
| TEXT_DELTA/COMPLETE | IMPLEMENTED | `stream_events.py:44-46` |

### 1.4 Intent Classification
| Feature | Status | Location |
|---------|--------|----------|
| IntentClassifier with IntentResult | IMPLEMENTED | `src/agents/classification/intent_classifier.py` |
| Symbol validation | IMPLEMENTED | IntentResult.validated_symbols |
| Complexity routing | IMPLEMENTED | IntentComplexity enum |
| conversation_summary parameter | IMPLEMENTED (fixed in previous session) | classify() method |

### 1.5 Agent Tree Tracking
| Feature | Status | Location |
|---------|--------|----------|
| TreeNode with NodeType/NodeStatus | IMPLEMENTED | `src/agents/streaming/agent_tree.py` |
| AgentTree class | IMPLEMENTED | AgentTree:93-368 |
| Tree visualization | IMPLEMENTED | visualize() method |
| TreeNodeContext (context manager) | IMPLEMENTED | agent_tree.py:441-498 |

---

## SECTION 2: GAPS AND MISSING FEATURES

### 2.1 ContextBuilder Service (CRITICAL GAP)

**Document Specification:**
```python
class ContextBuilder:
    """Centralized context assembly service"""
    async def build_context(
        session_id: str,
        user_id: int
    ) -> AgentContext:
        return AgentContext(
            core_memory=await load_core_memory(user_id),
            conversation_summary=await load_summary(session_id),
            recent_messages=await load_recent(session_id, K=10),
            working_memory=await load_working_memory(session_id)
        )
```

**Current Implementation:**
- Context assembly is scattered across `chat_assistant.py` (Phases 1-1.7)
- No centralized ContextBuilder class
- Each phase loads context independently

**Gap Analysis:**
- Missing: Single service that assembles all context
- Missing: AgentContext dataclass
- Missing: Reusable context building logic

**Priority:** HIGH

---

### 2.2 Thinking Display Timeline for UI (CRITICAL GAP - User Emphasized)

**User Request:**
> "Thiết kế và triển khai thinking display dạng SSE với timeline view để thể hiện đầy đủ thought process cùng dấu hiệu các cuộc gọi LLM để tôi thể hiện thought process lên UI."

**Document Specification:**
```
Timeline View Format:
┌─────────────────────────────────────────────────────────────────┐
│ 🧠 Thinking...                                                   │
│ ├── [0.0s] Analyzing query: "Phân tích NVDA"                   │
│ ├── [0.1s] 🔍 LLM Call: Intent Classification                  │
│ ├── [0.3s] Detected symbols: NVDA (stock)                      │
│ ├── [0.4s] 🔍 LLM Call: Tool Selection                         │
│ ├── [0.6s] Selected 4 tools: price, technicals, news...        │
│ ├── [0.8s] 🔧 Tool: getStockPrice(NVDA) → $142.50              │
│ ├── [1.2s] 🔧 Tool: getTechnicalIndicators(NVDA) → RSI=68      │
│ ├── [1.5s] 🔍 LLM Call: Synthesis                               │
│ └── [2.0s] ✅ Response ready                                    │
└─────────────────────────────────────────────────────────────────┘
```

**Current Implementation:**
- SSE events exist (THINKING_START, TOOL_START, etc.)
- No structured timeline format
- No elapsed time tracking
- No LLM call indicators (🔍)

**Missing Components:**
1. `ThinkingTimelineEvent` - Structured event for timeline
2. `elapsed_time` field in events
3. `llm_call_indicator` boolean field
4. Timeline aggregation on frontend
5. "Thought for Xs" display (ChatGPT style)

**Priority:** CRITICAL (User emphasized)

---

### 2.3 Circuit Breaker Pattern (HIGH GAP)

**Document Specification:**
```python
class CircuitBreaker:
    states = [CLOSED, OPEN, HALF_OPEN]
    failure_threshold = 5
    recovery_timeout = 30s

    async def call(self, func):
        if self.state == OPEN:
            if time_since_failure > recovery_timeout:
                self.state = HALF_OPEN
            else:
                raise CircuitOpenError()

        try:
            result = await func()
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            if failures >= threshold:
                self.state = OPEN
            raise
```

**Current Implementation:**
- No circuit breaker
- Tool failures just return error status
- No automatic recovery mechanism

**Priority:** HIGH (Production stability)

---

### 2.4 Graceful Degradation (MEDIUM GAP)

**Document Specification:**
```python
# If some tools fail, continue with partial data
async def execute_with_graceful_degradation(tools):
    results = await gather(*tools, return_exceptions=True)
    successful = [r for r in results if not isinstance(r, Exception)]

    if len(successful) >= min_required:
        return synthesize_partial_response(successful)
    else:
        return fallback_response()
```

**Current Implementation:**
- Partial support in `_execute_tools_parallel()`
- Return exceptions=True is used
- Missing: Explicit fallback response strategy
- Missing: min_required threshold check

**Priority:** MEDIUM

---

### 2.5 Adaptive Max Turns (PARTIAL)

**Document Specification:**
```python
def calculate_max_turns(intent_result):
    base = 4
    if intent_result.complexity == COMPLEX:
        base = 6
    if len(intent_result.symbols) > 3:
        base += 2
    return min(base, 10)
```

**Current Implementation:**
- `router_decision.suggested_max_turns` exists
- Hardcoded max_turns=6 in some places
- Not fully adaptive based on symbols count

**Priority:** LOW (Partial implementation works)

---

### 2.6 SSE Cancellation Handling (MEDIUM GAP)

**Document Specification:**
```python
async def stream_with_cancellation():
    cancellation_token = CancellationToken()

    try:
        async for event in agent.run_stream():
            if cancellation_token.is_cancelled:
                yield {"type": "cancelled"}
                break
            yield event
    finally:
        cleanup_resources()
```

**Current Implementation:**
- No explicit cancellation token
- No cleanup on client disconnect
- FastAPI background tasks not used for cleanup

**Priority:** MEDIUM

---

## SECTION 3: PRIORITIZED TASK LIST

### Sprint 1: Critical Foundation (1-2 weeks)

#### Task 1.1: Implement Thinking Display Timeline SSE
**Priority:** P0 (User emphasized)
**Files to modify:**
- `src/agents/streaming/stream_events.py` - Add timeline events
- `src/agents/unified/unified_agent.py` - Emit timeline events
- `src/routers/v2/chat_assistant.py` - Handle timeline in SSE

**New Components:**
```python
@dataclass
class ThinkingTimelineEvent:
    elapsed_ms: int
    phase: str  # "classification", "tool_selection", "tool_execution", "synthesis"
    action: str
    is_llm_call: bool
    details: Optional[str] = None

    # Example: [0.3s] 🔍 LLM Call: Intent Classification
```

**Deliverables:**
1. ThinkingTimelineEvent dataclass
2. Timeline tracking in UnifiedAgent
3. Frontend-ready SSE format
4. "Thought for Xs" calculation

---

#### Task 1.2: Implement ContextBuilder Service
**Priority:** P1
**Files to create:**
- `src/services/context_builder.py` - New service

**New Components:**
```python
@dataclass
class AgentContext:
    core_memory: Optional[str]
    conversation_summary: Optional[str]
    recent_messages: List[Dict]
    working_memory: Optional[WorkingMemoryState]

class ContextBuilder:
    async def build_context(session_id, user_id) -> AgentContext
    async def build_prompt_context() -> str  # Formatted for LLM
```

**Benefits:**
- Centralized context logic
- Easier testing
- Reusable across endpoints

---

### Sprint 2: Production Stability (1 week)

#### Task 2.1: Implement Circuit Breaker
**Priority:** P1
**Files to create:**
- `src/utils/circuit_breaker.py`

**Integration points:**
- LLM provider calls
- External API calls (FMP, Binance)
- Tool execution

---

#### Task 2.2: Enhance Graceful Degradation
**Priority:** P2
**Files to modify:**
- `src/agents/unified/unified_agent.py`

**Changes:**
- Add min_required threshold
- Implement fallback_response()
- Better partial success messaging

---

### Sprint 3: UX Enhancements (1 week)

#### Task 3.1: SSE Cancellation Handling
**Priority:** P2
**Files to modify:**
- `src/routers/v2/chat_assistant.py`

**Changes:**
- Implement CancellationToken
- Add cleanup on disconnect
- Background task for resource cleanup

---

#### Task 3.2: Adaptive Max Turns Enhancement
**Priority:** P3
**Files to modify:**
- `src/agents/unified/unified_agent.py`

**Changes:**
- Dynamic max_turns based on symbols count
- Configurable thresholds

---

## SECTION 4: IMPLEMENTATION CHECKLIST

### Phase 1: Thinking Timeline (CRITICAL)
- [ ] Create ThinkingTimelineEvent dataclass
- [ ] Add elapsed_ms tracking to StreamState
- [ ] Add is_llm_call indicator to events
- [ ] Emit timeline events in UnifiedAgent
- [ ] Test with frontend mock
- [ ] Add "Thought for Xs" calculation

### Phase 2: ContextBuilder
- [ ] Create AgentContext dataclass
- [ ] Create ContextBuilder service
- [ ] Migrate context loading from chat_assistant.py
- [ ] Add caching layer (optional)
- [ ] Update IntentClassifier to use ContextBuilder
- [ ] Update UnifiedAgent to use ContextBuilder

### Phase 3: Stability
- [ ] Implement CircuitBreaker class
- [ ] Integrate with LLM provider
- [ ] Integrate with external APIs
- [ ] Add graceful degradation thresholds
- [ ] Implement fallback responses

### Phase 4: Polish
- [ ] SSE cancellation handling
- [ ] Adaptive max_turns
- [ ] Documentation updates
- [ ] Integration tests

---

## SECTION 5: QUICK WINS (Can implement immediately)

1. **Add elapsed_ms to SSE events** - Simple field addition
2. **Add is_llm_call boolean** - Tag events appropriately
3. **Rename "reasoning" events to "thinking_step"** - Better frontend mapping

---

## APPENDIX: File Reference

| Component | File Path | Lines |
|-----------|-----------|-------|
| UnifiedAgent | `src/agents/unified/unified_agent.py` | 2462 |
| StreamEvents | `src/agents/streaming/stream_events.py` | ~1071 |
| AgentTree | `src/agents/streaming/agent_tree.py` | 540 |
| ThinkTool | `src/agents/tools/reasoning/think_tool.py` | 324 |
| ChatAssistant | `src/routers/v2/chat_assistant.py` | ~2500 |
| RecursiveSummary | `src/agents/memory/recursive_summary.py` | 740 |
| IntentClassifier | `src/agents/classification/intent_classifier.py` | ~500 |

---

*Generated by Claude Code analysis*
