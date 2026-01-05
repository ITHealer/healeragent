# Implementation Plan: LLM-as-Router Architecture

> **Goal**: Triển khai 2-phase tool selection như ChatGPT, tối ưu token và loại bỏ "category blindness"
> **Date**: 2026-01-05

---

## 1. Phân Tích Vấn Đề Hiện Tại

### Current Flow (Category-Based)
```
Query → Classifier (1 LLM) → Categories → Filter Tools by Category → Agent Loop
                                              ↓
                            [Only tools in selected categories visible to LLM]
```

### Problems
| Issue | Impact | Root Cause |
|-------|--------|------------|
| Category Blindness | LLM không thấy tools ngoài categories đã chọn | Classifier quyết định categories TRƯỚC khi LLM thấy tools |
| Mode Duplication | Normal Mode + Deep Research giống ~70% | Chưa merge, maintain 2 codebase |
| Token Waste | Full schema cho TẤT CẢ tools trong category | Không có 2-level loading thực sự |
| Miss-classification | Classifier sai category → mất tools | Single point of failure |

---

## 2. Proposed Architecture (ChatGPT-Style)

### New Flow (LLM-as-Router)
```
Query → LLM Router (sees ALL tool summaries) → Selected Tools + Complexity
                                                      ↓
           Load Full Schemas (selected only) → Unified Agent → Response
```

### Key Benefits
- **No Category Blindness**: Router LLM thấy TẤT CẢ 38 tools (summaries)
- **Token Efficient**: ~2000 tokens cho routing vs ~8000 tokens full schemas
- **Complexity-Aware**: Router output complexity → Agent adapts strategy
- **Single Agent**: Merge Normal + Deep Research → easier maintenance

---

## 3. Implementation Tasks

### Phase 1: Tool Catalog (Priority: HIGH)

#### File: `src/agents/tools/tool_catalog.py`

**Purpose**: Generate 2-level tool descriptions for Router

```python
@dataclass
class ToolSummary:
    """Lightweight tool description for Router (~50 tokens each)"""
    name: str
    category: str
    one_liner: str  # 1 sentence description
    capabilities: List[str]  # 3-5 bullet points
    typical_use: str  # When to use this tool

@dataclass
class ToolFullSchema:
    """Full schema for Agent execution (~200-400 tokens each)"""
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: Dict[str, str]
    examples: List[str]
```

**Tasks**:
- [ ] Create `ToolCatalog` class with `get_all_summaries()` and `get_full_schemas(tool_names)`
- [ ] Auto-generate summaries from existing `ToolSchema`
- [ ] Cache summaries (they rarely change)
- [ ] Add `to_router_format()` for Router LLM prompt

**Estimated Tokens**:
```
Current: 38 tools × 200 tokens = 7,600 tokens (all tools)
Proposed:
  - Router: 38 tools × 50 tokens = 1,900 tokens
  - Agent: 5 tools × 200 tokens = 1,000 tokens
  - Total: ~3,000 tokens (60% reduction)
```

---

### Phase 2: LLM Tool Router (Priority: HIGH)

#### File: `src/agents/router/llm_tool_router.py`

**Purpose**: Single LLM call to select tools + determine complexity

**Input**: Query + All Tool Summaries
**Output**:
```python
@dataclass
class RouterDecision:
    selected_tools: List[str]  # Tool names to use
    complexity: Complexity  # simple/medium/complex
    execution_strategy: str  # direct/iterative/parallel
    reasoning: str  # Why these tools were selected
    confidence: float
```

**Complexity Rules**:
```python
class Complexity(Enum):
    SIMPLE = "simple"    # 1-2 tools, direct answer
    MEDIUM = "medium"    # 3-5 tools, may need iteration
    COMPLEX = "complex"  # 6+ tools, needs planning
```

**LLM Prompt Structure**:
```xml
<tool_catalog>
[All tool summaries, ~50 tokens each]
</tool_catalog>

<query>{user_query}</query>

<instructions>
Select the tools needed to answer this query.
Output:
- selected_tools: ["tool1", "tool2", ...]
- complexity: simple|medium|complex
- execution_strategy: direct|iterative|parallel
- reasoning: Why these tools
</instructions>
```

**Tasks**:
- [ ] Create `LLMToolRouter` class
- [ ] Build router prompt with tool summaries
- [ ] Parse structured output (JSON in XML tags)
- [ ] Add fallback for parsing failures
- [ ] Integrate with ToolCatalog

---

### Phase 3: Unified Agent (Priority: HIGH)

#### File: `src/agents/unified/unified_agent.py`

**Purpose**: Single agent that adapts execution based on complexity

**Merge From**:
- `NormalModeAgent` (simple queries, agent loop)
- `StreamingChatHandler` (complex queries, 7-phase pipeline)

**Execution Strategies**:

| Complexity | Strategy | Max Turns | Planning |
|------------|----------|-----------|----------|
| SIMPLE | Direct | 2 | No |
| MEDIUM | Iterative | 4 | Optional |
| COMPLEX | Parallel + Planning | 6 | Yes |

```python
class UnifiedAgent:
    async def run(
        self,
        query: str,
        selected_tools: List[str],
        complexity: Complexity,
        strategy: str,
        ...
    ) -> AgentResult:

        # Load full schemas for selected tools only
        tool_schemas = self.catalog.get_full_schemas(selected_tools)

        if complexity == Complexity.SIMPLE:
            return await self._execute_simple(query, tool_schemas)
        elif complexity == Complexity.MEDIUM:
            return await self._execute_iterative(query, tool_schemas)
        else:
            return await self._execute_complex(query, tool_schemas)
```

**Tasks**:
- [ ] Create `UnifiedAgent` base structure
- [ ] Implement `_execute_simple()` (from NormalModeAgent)
- [ ] Implement `_execute_iterative()` (enhanced NormalModeAgent)
- [ ] Implement `_execute_complex()` (from StreamingChatHandler)
- [ ] Add streaming support for all strategies
- [ ] Implement tool loading from ToolCatalog

---

### Phase 4: Integration (Priority: MEDIUM)

#### File: `src/routers/v2/chat_assistant.py`

**New Flow**:
```python
async def chat_stream(...):
    # 1. Classify (keep for language, symbols extraction)
    classification = await classifier.classify(context)

    # 2. Route (NEW - LLM selects tools)
    router_decision = await tool_router.route(
        query=query,
        symbols=classification.symbols,
        context=context,
    )

    # 3. Execute with Unified Agent
    async for event in unified_agent.run_stream(
        query=query,
        selected_tools=router_decision.selected_tools,
        complexity=router_decision.complexity,
        strategy=router_decision.execution_strategy,
        classification=classification,
        ...
    ):
        yield emitter.emit(event)
```

**Tasks**:
- [ ] Update `chat_assistant.py` to use new flow
- [ ] Keep Classifier for language/symbol extraction (fast, cached)
- [ ] Add Router between Classifier and Agent
- [ ] Remove Mode Router (replaced by complexity-based strategy)
- [ ] Update SSE events for new flow

---

### Phase 5: SSE Events Update (Priority: LOW)

**New Events**:
```python
# Router decision event
{
    "type": "reasoning",
    "data": {
        "phase": "tool_routing",
        "action": "decision",
        "content": "Selected 3 tools: getStockPrice, getTechnicalIndicators, getNews",
        "metadata": {
            "selected_tools": ["getStockPrice", "getTechnicalIndicators", "getNews"],
            "complexity": "medium",
            "strategy": "iterative"
        }
    }
}
```

---

## 4. Migration Strategy

### Step 1: Build New Components (Non-Breaking)
```
src/agents/
├── tools/
│   └── tool_catalog.py      # NEW
├── router/
│   └── llm_tool_router.py   # NEW
└── unified/
    └── unified_agent.py     # NEW
```

### Step 2: Feature Flag Integration
```python
# settings.py
USE_LLM_ROUTER = env_bool("USE_LLM_ROUTER", default=False)

# chat_assistant.py
if settings.USE_LLM_ROUTER:
    # New flow
else:
    # Current flow
```

### Step 3: Gradual Rollout
1. Enable for internal testing
2. Enable for 10% traffic
3. Monitor latency, accuracy
4. Full rollout

---

## 5. Performance Considerations

### Token Usage Comparison

| Component | Current | Proposed |
|-----------|---------|----------|
| Classifier | 500 | 500 |
| Tool Loading | 7,600 (all) | 3,000 (summary + selected) |
| Agent Prompt | 2,000 | 2,000 |
| **Total** | **10,100** | **5,500** |

**Savings: ~45% token reduction**

### Latency Impact

| Operation | Current | Proposed |
|-----------|---------|----------|
| Classification | 200ms | 200ms |
| Tool Router | N/A | +150ms |
| Tool Loading | 50ms | 30ms (fewer tools) |
| Agent Loop | 800ms | 700ms |
| **Total** | **1,050ms** | **1,080ms** |

**Net Impact**: +30ms latency, 45% fewer tokens

---

## 6. File Structure

```
src/agents/
├── tools/
│   ├── base.py              # Existing
│   ├── registry.py          # Existing
│   └── tool_catalog.py      # NEW - Tool summaries/schemas
├── router/
│   └── llm_tool_router.py   # NEW - LLM-based router
├── unified/
│   └── unified_agent.py     # NEW - Merged agent
├── normal_mode/
│   └── normal_mode_agent.py # DEPRECATED (merged into unified)
└── streaming/
    └── streaming_chat_handler.py  # DEPRECATED (merged into unified)
```

---

## 7. Acceptance Criteria

### Functional
- [ ] Router selects correct tools for 95%+ queries
- [ ] No category blindness - all relevant tools considered
- [ ] Complexity detection accurate for query types
- [ ] Streaming works for all complexity levels

### Performance
- [ ] Token usage reduced by 40%+
- [ ] Latency increase < 100ms
- [ ] Memory usage stable

### Quality
- [ ] All existing tests pass
- [ ] New unit tests for Router, Catalog, UnifiedAgent
- [ ] E2E tests for new flow

---

## 8. Timeline

| Task | Effort | Priority |
|------|--------|----------|
| Tool Catalog | 2h | P0 |
| LLM Tool Router | 3h | P0 |
| Unified Agent (Simple) | 2h | P0 |
| Unified Agent (Medium/Complex) | 4h | P1 |
| Integration | 2h | P1 |
| Testing | 2h | P1 |
| **Total** | **15h** | |

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Router selects wrong tools | Bad responses | Fallback to category-based if confidence < 0.7 |
| Latency increase | UX degradation | Parallelize router + classifier |
| Complex queries fail | User frustration | Keep complex strategy as fallback |

---

## Summary

**Key Changes**:
1. **Tool Catalog**: 2-level tool descriptions (summary + full)
2. **LLM Router**: Single LLM sees ALL tools, selects relevant ones
3. **Unified Agent**: Merge Normal + Deep Research, adapt by complexity
4. **No Category Blindness**: Every tool visible to Router

**Benefits**:
- 45% token reduction
- Better tool selection accuracy
- Simpler codebase (1 agent vs 2)
- Production-ready architecture like ChatGPT
