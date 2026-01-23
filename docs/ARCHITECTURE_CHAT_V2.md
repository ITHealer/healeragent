# HealerAgent Chat Architecture Documentation

> **Author**: Auto-generated architecture documentation
> **Last Updated**: 2026-01-18
> **Version**: 2.0

## Overview

HealerAgent has 2 chat API endpoints:

| Endpoint | Status | Description |
|----------|--------|-------------|
| `/chat/v2` | **CURRENT** | Intent Classifier + All Tools (ChatGPT-style) |
| `/chat` | **LEGACY** | Mode Router + UnifiedClassifier + LLMToolRouter |

**IMPORTANT**: New features should be developed for `/chat/v2`. The `/chat` endpoint is maintained for backward compatibility only.

---

## `/chat/v2` Architecture (RECOMMENDED)

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         /chat/v2 FLOW (2 LLM Calls)                         │
└─────────────────────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 0: Image Processing (optional)                                         │
│   - ProcessedImage[] if images provided                                      │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Session + Working Memory Setup                                      │
│   - WorkingMemoryIntegration (session-scoped scratchpad)                     │
│   - flow_id generation                                                       │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1.5: Context Loading (ContextBuilder)                                  │
│   - Core Memory (user profile/preferences)                                   │
│   - Conversation Summary (older messages compressed)                         │
│   - Working Memory Symbols (cross-turn symbol continuity)                    │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 1.6: Context Compaction Check                                          │
│   - Auto-compress if token count exceeds threshold                           │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Intent Classification (1 LLM Call)                    ◄── LLM #1   │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │ IntentClassifier.classify()                             │               │
│   │   Input:  query, ui_context, history, wm_symbols        │               │
│   │   Output: IntentResult                                  │               │
│   │     - validated_symbols: ["GOOGL", "AAPL"]  (normalized)│               │
│   │     - market_type: stock | crypto | both | none         │               │
│   │     - complexity: direct | agent_loop                   │               │
│   │     - requires_tools: bool                              │               │
│   │     - response_language: "en" | "vi"                    │               │
│   └─────────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ├── if requires_tools == false ──────────────────────────────────────────┐
    │                                                                         │
    │                                                            Direct Response
    │                                                            (No tools needed)
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 3: Agent Loop with ALL Tools (Iterative)                 ◄── LLM #2+ │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │ UnifiedAgent.run_stream_with_all_tools()                │               │
│   │                                                         │               │
│   │   Tool Loading Strategy:                                │               │
│   │   ┌─────────────────────────────────────────────────┐   │               │
│   │   │ if enable_tool_search_mode == true:             │   │               │
│   │   │   • Start with: tool_search + think (~600 tok)  │   │               │
│   │   │   • Discover tools dynamically (85% savings)    │   │               │
│   │   │                                                 │   │               │
│   │   │ else:                                           │   │               │
│   │   │   • Load ALL tools from ToolCatalog (~15K tok)  │   │               │
│   │   └─────────────────────────────────────────────────┘   │               │
│   │                                                         │               │
│   │   Agent Loop (max_turns=6):                             │               │
│   │   ┌────────────────────────────────────────────┐        │               │
│   │   │  THINK → Select Tools → Execute → OBSERVE  │◄───┐   │               │
│   │   │     │                                      │    │   │               │
│   │   │     └──────────────────────────────────────┼────┘   │               │
│   │   │                (repeat until done)         │        │               │
│   │   └────────────────────────────────────────────┘        │               │
│   │                                                         │               │
│   │   Events: turn_start, tool_calls, tool_results,         │               │
│   │           content, thinking, done                       │               │
│   └─────────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 4: Post-Processing                                                     │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │ 4.1 LEARN Phase: LearnHook.on_execution_complete()                   │  │
│   │     - Update memory based on tool results                            │  │
│   │                                                                      │  │
│   │ 4.2 SAVE Phase: _save_conversation_turn()                            │  │
│   │     - Persist to database (CRITICAL for memory)                      │  │
│   │                                                                      │  │
│   │ 4.3 SUMMARY Phase: RecursiveSummaryManager.check_and_create_summary()│  │
│   │     - Create/update recursive summary if threshold reached           │  │
│   │                                                                      │  │
│   │ 4.4 Chart Resolution: ChartResolver.resolve_from_tool_results()      │  │
│   │     - Map tool results to frontend charts                            │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
  Response (SSE Stream)
```

### File Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/routers/v2/chat_assistant.py` | API endpoint | `stream_chat_v2()` @line 1178 |
| `src/agents/classification/intent_classifier.py` | Classification | `IntentClassifier.classify()` |
| `src/agents/unified/unified_agent.py` | Agent execution | `run_stream_with_all_tools()` @line 2330 |
| `src/agents/unified/unified_agent.py` | Build prompts | `_build_agent_messages_all_tools()` @line 2900 |
| `src/agents/tools/tool_catalog.py` | Tool management | `ToolCatalog` |
| `src/services/tool_search_service.py` | Dynamic discovery | `ToolSearchService` |
| `src/agents/skills/stock_skill.py` | Stock prompts | `StockSkill` |
| `src/agents/skills/crypto_skill.py` | Crypto prompts | `CryptoSkill` |
| `src/services/context_builder.py` | Context assembly | `ContextBuilder.build_context()` |
| `src/agents/hooks/learn_hook.py` | Memory update | `LearnHook.on_execution_complete()` |

### Key Parameters

```python
# ChatRequest parameters for /chat/v2
enable_tool_search_mode: bool = False  # Token savings (~85%)
enable_think_tool: bool = False        # Explicit reasoning tool
enable_web_search: bool = False        # Force inject web search
enable_thinking: bool = True           # Show thinking process
max_turns: int = 6                     # Agent loop limit
```

---

## `/chat` Architecture (LEGACY)

> **WARNING**: This endpoint is for backward compatibility only. Do not add new features here.

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      /chat FLOW (LEGACY - 3+ LLM Calls)                     │
└─────────────────────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ UnifiedClassifier.classify()                                   ◄── LLM #1  │
│   - Determines: query_type, tool_categories, requires_tools                 │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ModeRouter.determine_mode()                                                  │
│   - Output: DEEP_RESEARCH | NORMAL                                          │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ├── if DEEP_RESEARCH ──► DeepResearchHandler (7-phase pipeline)
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ NormalModeChatHandler._stream_normal_mode()                                  │
│   │                                                                          │
│   ├─► LLMToolRouter.route()                                    ◄── LLM #2  │
│   │     - Sees ALL tool summaries                                           │
│   │     - Selects relevant tools                                            │
│   │     - Returns: RouterDecision                                           │
│   │                                                                          │
│   └─► UnifiedAgent.run_stream()                                ◄── LLM #3+ │
│         - Uses pre-selected tools (NOT all tools)                           │
│         - Uses RouterDecision (NOT IntentResult)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Legacy File Reference

| File | Purpose | Status |
|------|---------|--------|
| `src/routers/v2/chat_assistant.py` | Legacy endpoint | `stream_chat_legacy()` @line 943 |
| `src/agents/classification/unified_classifier.py` | Legacy classifier | `UnifiedClassifier` |
| `src/handlers/v2/mode_router.py` | Mode routing | `ModeRouter` |
| `src/handlers/v2/normal_mode_chat_handler.py` | Normal mode | Uses `run_stream()` |
| `src/agents/router/llm_tool_router.py` | **NOT USED FOR /chat/v2** | `LLMToolRouter` |

---

## Key Differences: `/chat/v2` vs `/chat`

| Aspect | `/chat/v2` | `/chat` (Legacy) |
|--------|------------|------------------|
| Classification | `IntentClassifier` | `UnifiedClassifier` |
| Tool Selection | Agent sees ALL tools | `LLMToolRouter` pre-selects |
| Agent Method | `run_stream_with_all_tools()` | `run_stream()` |
| Input Data | `IntentResult` | `RouterDecision` |
| LLM Calls | 2 minimum | 3+ minimum |
| Tool Search | Optional (85% savings) | Not available |

---

## Component Details

### 1. IntentClassifier (NEW - /chat/v2)

Location: `src/agents/classification/intent_classifier.py`

```python
@dataclass
class IntentResult:
    intent_summary: str
    reasoning: str
    validated_symbols: List[str]  # Already normalized! (GOOGLE → GOOGL)
    market_type: IntentMarketType  # stock | crypto | both | none
    complexity: IntentComplexity   # direct | agent_loop
    requires_tools: bool
    response_language: str
```

### 2. UnifiedAgent (NEW Architecture)

Location: `src/agents/unified/unified_agent.py`

**Method for /chat/v2**:
```python
async def run_stream_with_all_tools(
    self,
    query: str,
    intent_result: Any,  # IntentResult from IntentClassifier
    enable_tool_search_mode: bool = False,  # Token savings
    max_turns: int = 6,
    ...
) -> AsyncGenerator[Dict[str, Any], None]:
    """Agent sees ALL tools and decides which to call (ChatGPT-style)."""
```

**Method for /chat (Legacy)**:
```python
async def run_stream(
    self,
    query: str,
    router_decision: RouterDecision,  # From LLMToolRouter
    classification: Optional[UnifiedClassificationResult] = None,
    ...
) -> AsyncGenerator[Dict[str, Any], None]:
    """Agent uses pre-selected tools from Router."""
```

### 3. Skill System

Location: `src/agents/skills/`

Skills provide domain-specific system prompts:

- `StockSkill`: Equity analysis expertise
- `CryptoSkill`: Cryptocurrency analysis
- `MixedSkill`: Multi-asset analysis

Used by `_build_agent_messages_all_tools()` to select appropriate prompt based on `market_type`.

### 4. Tool Catalog

Location: `src/agents/tools/tool_catalog.py`

2-level tool description system:
- **ToolSummary** (~50 tokens): For Router LLM
- **ToolFullSchema** (~200-400 tokens): For Agent execution

Token savings: ~60% compared to always loading full schemas.

### 5. Tool Search Service

Location: `src/services/tool_search_service.py`

When `enable_tool_search_mode=True`:
- Agent starts with ONLY `tool_search` + `think` tools (~600 tokens)
- Discovers relevant tools via semantic search
- 85% token savings vs loading all tools (~15K tokens)

---

## Default Configurations

### Recommended Defaults for /chat/v2

```python
# For production use
enable_tool_search_mode = True   # Save ~85% tokens
enable_think_tool = False        # Enable only when deep reasoning needed
enable_web_search = False        # Enable only for news queries
max_turns = 6                    # Sufficient for most queries
```

### Event Types (SSE)

| Event | Description |
|-------|-------------|
| `session_start` | Session initialized |
| `classifying` | Classification starting |
| `classified` | Classification complete with IntentResult |
| `turn_start` | Agent turn starting |
| `tool_calls` | Tools being called |
| `tool_results` | Tool execution results |
| `thinking` | Think tool output |
| `content` | Response content (streaming) |
| `done` | Execution complete |
| `error` | Error occurred |

---

## Migration Guide

### From `/chat` to `/chat/v2`

1. Change endpoint: `/chat` → `/chat/v2`
2. Update response handling:
   - `classification` event now contains `IntentResult` fields
   - No separate `router_decision` event
3. Consider enabling `enable_tool_search_mode=True` for token savings
4. Update any code depending on `RouterDecision` to use `IntentResult`

---

## Files NOT Used by /chat/v2

These files are **LEGACY** and only used by `/chat`:

| File | Purpose | Used By |
|------|---------|---------|
| `src/agents/router/llm_tool_router.py` | 2-phase tool selection | `/chat` only |
| `src/handlers/v2/normal_mode_chat_handler.py` | Normal mode handling | `/chat` only |
| `src/handlers/v2/mode_router.py` | DEEP_RESEARCH routing | `/chat` only |
| `src/agents/classification/unified_classifier.py` | Legacy classifier | `/chat` only |

---

## Appendix: Tool Categories

### Stock-Specific Tools (require symbol)
- `getStockPrice`, `getStockPerformance`, `getPriceTargets`
- `getTechnicalIndicators`, `getSupportResistance`, `detectChartPatterns`
- `getIncomeStatement`, `getBalanceSheet`, `getCashFlow`, `getFinancialRatios`
- `getGrowthMetrics`, `getAnalystRatings`, `getStockNews`, `getSentiment`
- `assessRisk`, `getVolumeProfile`, `suggestStopLoss`

### Market-Wide Tools (no symbol required)
- `getMarketIndices`, `getTopGainers`, `getTopLosers`, `getMostActives`
- `getSectorPerformance`, `getMarketBreadth`, `getMarketMovers`
- `getMarketNews`, `getEconomicData`, `getStockHeatmap`
- `stockScreener`

### Utility Tools
- `webSearch`, `serpSearch` - Web search
- `think` - Explicit reasoning
- `tool_search` - Dynamic tool discovery

### Memory Tools
- `getRecentConversations`, `searchConversationHistory`
- `searchRecallMemory`, `searchArchivalMemory`
- `memoryUserEdits`
