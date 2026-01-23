# Invest Agent Implementation Plan

**Version:** 1.0
**Date:** 2026-01-23
**Status:** Draft
**Author:** Architecture Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Design Principles](#3-design-principles)
4. [Lessons Learned from Existing Agents](#4-lessons-learned-from-existing-agents)
5. [Proposed Directory Structure](#5-proposed-directory-structure)
6. [Module Specifications](#6-module-specifications)
7. [Integration Strategy](#7-integration-strategy)
8. [Implementation Phases](#8-implementation-phases)
9. [Risk Mitigation](#9-risk-mitigation)
10. [Success Metrics](#10-success-metrics)

---

## 1. Executive Summary

### 1.1 Objective

Triển khai logic mới cho **Response Modes (FAST/AUTO/EXPERT)** trong một folder độc lập `src/agents/invest_agent/` mà:

1. **KHÔNG ảnh hưởng** đến logic xử lý và API trong `src/routers/v2/chat_assistant.py` (đang chạy production)
2. **Học hỏi** từ kiến trúc tốt của `src/agents/deep_research/` và `src/agents/normal_mode/`
3. **Tích hợp** web search skill như các AI Chatbot lớn (ChatGPT, Gemini, Claude)
4. **Dễ scale và maintenance** với naming conventions rõ ràng

### 1.2 Key Benefits

| Benefit | Description |
|---------|-------------|
| **Production Safety** | Code mới hoàn toàn tách biệt, không touch production |
| **Clean Architecture** | Mỗi mode có executor riêng, dễ hiểu và maintain |
| **Feature Flag Control** | Rollout từng phase, rollback nhanh nếu có issue |
| **Skill Integration** | Web search, computation skills như ChatGPT/Claude |
| **Provider Flexibility** | Dễ dàng switch giữa GPT/Gemini/OpenRouter |

### 1.3 Non-Goals (Scope Exclusions)

- ❌ Không modify `chat_assistant.py` trong phase 1
- ❌ Không thay thế `normal_mode_agent.py` ngay
- ❌ Không implement full deep research capability
- ❌ Không add new tools (chỉ sử dụng tools hiện có)

---

## 2. Problem Statement

### 2.1 Current State

```
src/routers/v2/chat_assistant.py
├── /chat         - Legacy production endpoint (UnifiedClassifier + ModeRouter)
├── /chat/v2      - New IntentClassifier + UnifiedAgent với ALL tools
└── /chat/v3      - V2 + Finance Guru tools

Issues:
- Logic FAST/AUTO/EXPERT chưa được implement rõ ràng
- Không có tool filtering cho FAST mode
- Không có mode-specific prompts
- Web search chưa integrated như major AI chatbots
```

### 2.2 Why New Folder?

| Approach | Pros | Cons |
|----------|------|------|
| **Modify existing** | Ít code mới | Risk production, complex merge |
| **New folder (Chosen)** | Safe, clean, testable | More initial work |

**Decision:** Tạo folder mới `invest_agent/` để:
- Develop và test isolated
- Feature flag để enable gradually
- Rollback = disable feature flag (không cần code revert)

---

## 3. Design Principles

### 3.1 From Deep Research (`src/agents/deep_research/`)

**What to Learn:**

| Pattern | Description | Apply to InvestAgent |
|---------|-------------|---------------------|
| **Orchestrator Pattern** | Central coordinator for phases | `InvestAgentOrchestrator` điều phối modes |
| **Worker Abstraction** | `BaseWorker` + specialized workers | Mode executors kế thừa base |
| **Streaming Events** | `ArtifactEmitter` for SSE | Reuse event format |
| **Prompts Folder** | Separate prompt templates | `prompts/fast_prompts.py`, etc. |
| **Config Dataclass** | `DeepResearchConfig` | `InvestAgentConfig` |
| **State Management** | `AgentState` enum | Mode state tracking |

**Code Reference:**
```python
# From src/agents/deep_research/orchestrator.py:51-76
class DeepResearchOrchestrator(BaseDeepResearchAgent):
    """
    Lead Agent for Deep Research.
    Coordinates the entire research process from clarification to final report.
    """
    def __init__(self, config: DeepResearchConfig = None):
        self.research_id = f"dr_{uuid.uuid4().hex[:12]}"
        self.state = AgentState.IDLE
        # ...
```

### 3.2 From Normal Mode (`src/agents/normal_mode/`)

**What to Learn:**

| Pattern | Description | Apply to InvestAgent |
|---------|-------------|---------------------|
| **Simple Agent Loop** | Think → Act → Observe | FAST mode uses this |
| **Tool Registry** | `ToolRegistry.execute_tool()` | Reuse for tool execution |
| **Streaming Support** | `run_stream()` async generator | All modes support streaming |
| **Message Building** | Structured system prompts | Mode-specific prompts |
| **Result Dataclass** | `AgentResult` | `InvestAgentResult` |

**Code Reference:**
```python
# From src/agents/normal_mode/normal_mode_agent.py:149-176
async def run(
    self,
    query: str,
    classification: Optional[UnifiedClassificationResult] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    ...
) -> AgentResult:
    """Run the agent loop."""
    # MAX_TURNS = 10, tools executed in parallel
```

### 3.3 Design Principles Summary

```
1. SEPARATION OF CONCERNS
   - Each mode has its own executor
   - Shared utilities in /shared/
   - No God classes

2. COMPOSITION OVER INHERITANCE
   - Executors compose ToolFilter, PromptBuilder, etc.
   - Easy to swap components

3. FAIL SAFE
   - Feature flags control rollout
   - Graceful degradation if new code fails
   - Fallback to existing UnifiedAgent

4. OBSERVABLE
   - SSE events for all phases
   - Thinking timeline for UI
   - Metrics and logging

5. TESTABLE
   - Unit tests per module
   - Integration tests for flows
   - Mock-friendly interfaces
```

---

## 4. Lessons Learned from Existing Agents

### 4.1 Good Patterns to Adopt

| Pattern | Source | Description |
|---------|--------|-------------|
| **Prompt Separation** | `deep_research/prompts/` | Each use case has dedicated prompt file |
| **Event Emitter** | `deep_research/streaming/` | Centralized SSE event creation |
| **Config Dataclass** | `deep_research/models.py` | Type-safe configuration |
| **Factory Function** | `normal_mode_agent.py:1146` | `get_normal_mode_agent()` singleton |
| **Flow ID Logging** | `normal_mode_agent.py:172` | `flow_id = f"NM-{uuid.uuid4().hex[:8]}"` |
| **Parallel Tool Execution** | `normal_mode_agent.py:868` | `asyncio.gather()` for speed |

### 4.2 Anti-Patterns to Avoid

| Anti-Pattern | Where Found | Better Approach |
|--------------|-------------|-----------------|
| **Mixed Concerns** | `chat_assistant.py` có quá nhiều logic | Tách thành modules riêng |
| **Hard-coded Values** | Scattered timeouts/limits | Centralize in config |
| **Implicit Dependencies** | Global singletons | Explicit dependency injection |
| **No Feature Flags** | Direct deployment | Feature flag for gradual rollout |

### 4.3 File Naming Conventions

**From existing codebase:**
```
src/agents/
├── deep_research/
│   ├── __init__.py              # Export public API
│   ├── base_agent.py            # Abstract base
│   ├── orchestrator.py          # Main coordinator
│   ├── models.py                # Data classes
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── clarification.py     # Purpose-specific
│   │   ├── planning.py
│   │   └── synthesis.py
│   ├── streaming/
│   │   └── artifact_emitter.py
│   ├── synthesis/
│   │   └── report_generator.py
│   └── workers/
│       └── base_worker.py
```

**Naming Rules:**
1. **Modules:** `snake_case` (e.g., `mode_router.py`)
2. **Classes:** `PascalCase` (e.g., `InvestAgentOrchestrator`)
3. **Functions:** `snake_case` (e.g., `get_invest_agent()`)
4. **Constants:** `UPPER_SNAKE_CASE` (e.g., `MAX_TURNS`)
5. **Private:** Prefix `_` (e.g., `_current_phase`)

---

## 5. Proposed Directory Structure

### 5.1 Complete Structure

```
src/agents/invest_agent/
├── __init__.py                          # Public exports
├── README.md                            # Module documentation
│
├── config/
│   ├── __init__.py
│   ├── mode_config.py                   # FAST/AUTO/EXPERT configurations
│   ├── feature_flags.py                 # Feature toggles for rollout
│   └── provider_config.py               # Provider-specific settings
│
├── core/
│   ├── __init__.py
│   ├── orchestrator.py                  # Main InvestAgentOrchestrator
│   ├── mode_selector.py                 # FAST/AUTO/EXPERT selection logic
│   ├── models.py                        # Data classes (InvestAgentResult, etc.)
│   └── base_executor.py                 # Abstract executor interface
│
├── executors/
│   ├── __init__.py
│   ├── fast_executor.py                 # FAST mode: 2 turns, filtered tools
│   ├── auto_executor.py                 # AUTO mode: intelligent switching
│   └── expert_executor.py               # EXPERT mode: 6 turns, all tools
│
├── prompts/
│   ├── __init__.py
│   ├── system_prompts.py                # Base system prompts (VI/EN)
│   ├── fast_prompts.py                  # Condensed prompts for FAST
│   ├── expert_prompts.py                # Full prompts with examples
│   └── provider_prompts.py              # GPT/Gemini-specific variations
│
├── tools/
│   ├── __init__.py
│   ├── tool_filter.py                   # Filter tools for FAST mode
│   └── tool_categories.py               # Category definitions
│
├── skills/
│   ├── __init__.py
│   ├── web_search_skill.py              # Web search integration (ChatGPT-style)
│   ├── computation_skill.py             # Finance Guru calculations
│   └── base_skill.py                    # Skill interface
│
├── streaming/
│   ├── __init__.py
│   ├── event_types.py                   # SSE event definitions
│   └── thinking_timeline.py             # ChatGPT-style "Thought for Xs"
│
└── tests/
    ├── __init__.py
    ├── test_orchestrator.py
    ├── test_mode_selector.py
    ├── test_fast_executor.py
    ├── test_expert_executor.py
    └── conftest.py                      # Pytest fixtures
```

### 5.2 File Responsibilities

| File | Responsibility | Lines Est. |
|------|----------------|------------|
| `orchestrator.py` | Coordinate modes, handle streaming | 300-400 |
| `mode_selector.py` | Decide FAST/AUTO/EXPERT | 150-200 |
| `fast_executor.py` | Execute FAST mode logic | 200-250 |
| `expert_executor.py` | Execute EXPERT mode logic | 250-300 |
| `tool_filter.py` | Filter tools for FAST | 150-200 |
| `web_search_skill.py` | Web search integration | 200-250 |

---

## 6. Module Specifications

### 6.1 Core Orchestrator

```python
# src/agents/invest_agent/core/orchestrator.py
"""
InvestAgentOrchestrator - Main entry point for invest agent.

Coordinates mode selection and execution, similar to how
DeepResearchOrchestrator coordinates research phases.
"""

from dataclasses import dataclass
from typing import AsyncGenerator, Dict, Any, Optional, List
from enum import Enum

from src.agents.invest_agent.config.mode_config import InvestAgentConfig
from src.agents.invest_agent.config.feature_flags import FeatureFlags
from src.agents.invest_agent.core.mode_selector import ModeSelector
from src.agents.invest_agent.executors.fast_executor import FastExecutor
from src.agents.invest_agent.executors.expert_executor import ExpertExecutor


class ResponseMode(str, Enum):
    """Response mode enum"""
    FAST = "fast"
    AUTO = "auto"
    EXPERT = "expert"


@dataclass
class ExecutionContext:
    """Context passed to executors"""
    session_id: str
    user_id: str
    query: str
    mode: ResponseMode
    language: str = "en"
    provider_type: str = "openai"
    model_name: str = "gpt-4o-mini"
    conversation_history: Optional[List[Dict]] = None
    core_memory: Optional[str] = None
    images: Optional[List] = None


class InvestAgentOrchestrator:
    """
    Main orchestrator for InvestAgent.

    Responsibilities:
    1. Select appropriate mode (FAST/AUTO/EXPERT)
    2. Delegate to mode-specific executor
    3. Handle streaming events
    4. Manage feature flags

    Usage:
        orchestrator = InvestAgentOrchestrator()
        async for event in orchestrator.run(
            query="AAPL stock price?",
            session_id="sess_123",
            user_id="user_456",
        ):
            yield event
    """

    def __init__(
        self,
        config: Optional[InvestAgentConfig] = None,
        feature_flags: Optional[FeatureFlags] = None,
    ):
        self.config = config or InvestAgentConfig()
        self.flags = feature_flags or FeatureFlags()

        self.mode_selector = ModeSelector(config=self.config)
        self.fast_executor = FastExecutor(config=self.config)
        self.expert_executor = ExpertExecutor(config=self.config)

        self._flow_id: Optional[str] = None

    async def run(
        self,
        query: str,
        session_id: str,
        user_id: str,
        explicit_mode: Optional[str] = None,
        intent_result: Optional[Any] = None,
        **kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main entry point for InvestAgent.

        Flow:
        1. Generate flow_id for tracing
        2. Check feature flags
        3. Select mode (or use explicit)
        4. Delegate to executor
        5. Stream events back

        Args:
            query: User query
            session_id: Session ID
            user_id: User ID
            explicit_mode: User-requested mode (fast/expert)
            intent_result: Pre-computed intent classification
            **kwargs: Additional context (images, history, etc.)

        Yields:
            SSE events: mode_selected, turn_start, tool_calls, content, done
        """
        import uuid
        self._flow_id = f"IA-{uuid.uuid4().hex[:8]}"

        # Feature flag check
        if not self.flags.is_enabled("invest_agent_v1"):
            # Fallback to existing behavior
            yield {"type": "fallback", "reason": "feature_disabled"}
            return

        # Build execution context
        context = ExecutionContext(
            session_id=session_id,
            user_id=user_id,
            query=query,
            mode=ResponseMode.AUTO,
            language=kwargs.get("language", "en"),
            provider_type=kwargs.get("provider_type", "openai"),
            model_name=kwargs.get("model_name", "gpt-4o-mini"),
            conversation_history=kwargs.get("conversation_history"),
            core_memory=kwargs.get("core_memory"),
            images=kwargs.get("images"),
        )

        # Mode selection
        if explicit_mode:
            context.mode = ResponseMode(explicit_mode.lower())
            yield {"type": "mode_selected", "mode": context.mode.value, "source": "explicit"}
        else:
            selected_mode = await self.mode_selector.select(
                query=query,
                intent_result=intent_result,
            )
            context.mode = selected_mode
            yield {"type": "mode_selected", "mode": context.mode.value, "source": "auto"}

        # Execute based on mode
        executor = self._get_executor(context.mode)

        async for event in executor.execute(context):
            yield event

    def _get_executor(self, mode: ResponseMode):
        """Get executor for mode"""
        if mode == ResponseMode.FAST:
            return self.fast_executor
        elif mode == ResponseMode.EXPERT:
            return self.expert_executor
        else:  # AUTO
            return self.fast_executor  # AUTO starts with FAST, may escalate
```

### 6.2 Mode Selector

```python
# src/agents/invest_agent/core/mode_selector.py
"""
ModeSelector - Intelligent mode selection.

Determines whether query should use FAST (simple) or EXPERT (complex) mode.
"""

from typing import Optional, Any
from src.agents.invest_agent.core.orchestrator import ResponseMode


class ModeSelector:
    """
    Selects FAST/EXPERT mode based on query characteristics.

    FAST Mode triggers:
    - Single symbol queries ("AAPL price?")
    - Simple lookups (price, basic metrics)
    - Low complexity detected

    EXPERT Mode triggers:
    - Multi-symbol comparison ("Compare AAPL vs MSFT")
    - Deep analysis requests ("Full analysis of NVDA")
    - Complex questions with multiple parts
    - Research queries
    """

    # Keywords that suggest EXPERT mode
    EXPERT_KEYWORDS = [
        "compare", "analysis", "analyze", "research",
        "vs", "versus", "comprehensive", "detailed",
        "deep dive", "report", "strategy", "portfolio",
        "dcf", "valuation", "so sánh", "phân tích",
    ]

    # Query patterns for FAST mode
    FAST_PATTERNS = [
        r"^\w+\s+price\??$",           # "AAPL price?"
        r"^price\s+of\s+\w+\??$",       # "price of AAPL?"
        r"^what\s+is\s+\w+\??$",        # "what is AAPL?"
        r"^giá\s+\w+\??$",              # "giá AAPL?" (Vietnamese)
    ]

    def __init__(self, config=None):
        self.config = config
        self._complexity_threshold = 0.6

    async def select(
        self,
        query: str,
        intent_result: Optional[Any] = None,
    ) -> ResponseMode:
        """
        Select mode based on query and intent.

        Priority:
        1. Use intent_result.complexity if available
        2. Keyword matching
        3. Pattern matching
        4. Default to FAST
        """
        query_lower = query.lower()

        # Check intent result complexity
        if intent_result and hasattr(intent_result, 'complexity'):
            complexity = intent_result.complexity
            if complexity.value in ['complex', 'research']:
                return ResponseMode.EXPERT

        # Keyword check
        for keyword in self.EXPERT_KEYWORDS:
            if keyword in query_lower:
                return ResponseMode.EXPERT

        # Multi-symbol check
        if intent_result and hasattr(intent_result, 'validated_symbols'):
            if len(intent_result.validated_symbols) > 2:
                return ResponseMode.EXPERT

        # Default to FAST for most queries
        return ResponseMode.FAST
```

### 6.3 Fast Executor

```python
# src/agents/invest_agent/executors/fast_executor.py
"""
FastExecutor - Optimized for speed.

Target: 3-6 second response time
Max turns: 2
Tools: Filtered set (8 max)
"""

from typing import AsyncGenerator, Dict, Any, List
from src.agents.invest_agent.core.base_executor import BaseExecutor
from src.agents.invest_agent.tools.tool_filter import ToolFilter
from src.agents.invest_agent.prompts.fast_prompts import FastPromptBuilder


class FastExecutor(BaseExecutor):
    """
    Executor for FAST mode.

    Characteristics:
    - Uses smaller model (gpt-4o-mini / gemini-2.0-flash)
    - Maximum 2 agent turns
    - Filtered tool set (8 max based on query categories)
    - Condensed system prompt (~1500 tokens)
    - No web search (unless explicitly needed)
    - No thinking display

    Similar to NormalModeAgent but with stricter constraints.
    """

    MAX_TURNS = 2
    MAX_TOOLS = 8

    def __init__(self, config=None):
        super().__init__(config)
        self.tool_filter = ToolFilter(max_tools=self.MAX_TOOLS)
        self.prompt_builder = FastPromptBuilder()

    async def execute(
        self,
        context: "ExecutionContext",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute query in FAST mode.

        Flow:
        1. Filter relevant tools (max 8)
        2. Build condensed prompt
        3. Run agent loop (max 2 turns)
        4. Stream response
        """
        import time
        start_time = time.time()

        # Emit mode info
        yield {
            "type": "mode_info",
            "mode": "fast",
            "max_turns": self.MAX_TURNS,
            "max_tools": self.MAX_TOOLS,
        }

        # Filter tools
        filtered_tools = self.tool_filter.filter(
            query=context.query,
            categories=self._detect_categories(context.query),
        )

        yield {
            "type": "tools_filtered",
            "count": len(filtered_tools),
            "tools": [t.get("function", {}).get("name") for t in filtered_tools[:5]],
        }

        # Build messages
        messages = self.prompt_builder.build(
            query=context.query,
            language=context.language,
            history=context.conversation_history,
            core_memory=context.core_memory,
        )

        # Agent loop
        total_tool_calls = 0

        for turn in range(1, self.MAX_TURNS + 1):
            yield {"type": "turn_start", "turn": turn, "max_turns": self.MAX_TURNS}

            # LLM call with tools
            response = await self._call_llm(
                messages=messages,
                tools=filtered_tools,
                model=self._get_fast_model(context.provider_type),
            )

            tool_calls = self._parse_tool_calls(response)
            content = response.get("content", "")

            # No tool calls = done
            if not tool_calls:
                if content:
                    for chunk in self._chunk_content(content):
                        yield {"type": "content", "content": chunk}
                break

            # Execute tools
            total_tool_calls += len(tool_calls)
            yield {
                "type": "tool_calls",
                "tools": [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls],
            }

            tool_results = await self._execute_tools(tool_calls)
            yield {
                "type": "tool_results",
                "results": [{"tool": tc.name, "success": r.get("status") == "success"}
                           for tc, r in zip(tool_calls, tool_results)],
            }

            # Update messages
            messages = self._update_messages(messages, tool_calls, tool_results)

        # Done
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield {
            "type": "done",
            "mode": "fast",
            "total_turns": turn,
            "total_tool_calls": total_tool_calls,
            "total_time_ms": elapsed_ms,
        }

    def _get_fast_model(self, provider_type: str) -> str:
        """Get fast model for provider"""
        models = {
            "openai": "gpt-4o-mini",
            "gemini": "gemini-2.0-flash",
            "openrouter": "openai/gpt-4o-mini",
        }
        return models.get(provider_type, "gpt-4o-mini")

    def _detect_categories(self, query: str) -> List[str]:
        """Detect tool categories from query"""
        categories = ["price"]  # Always include price

        query_lower = query.lower()

        if any(kw in query_lower for kw in ["technical", "indicator", "rsi", "macd"]):
            categories.append("technical")
        if any(kw in query_lower for kw in ["news", "tin", "headline"]):
            categories.append("news")
        if any(kw in query_lower for kw in ["financial", "ratio", "pe", "revenue"]):
            categories.append("fundamentals")
        if any(kw in query_lower for kw in ["risk", "volatility"]):
            categories.append("risk")

        return categories
```

### 6.4 Expert Executor

```python
# src/agents/invest_agent/executors/expert_executor.py
"""
ExpertExecutor - Full capability mode.

Target: Comprehensive analysis (15-45 seconds acceptable)
Max turns: 6
Tools: ALL tools (31+)
Web search: Enabled
"""

from typing import AsyncGenerator, Dict, Any
from src.agents.invest_agent.core.base_executor import BaseExecutor
from src.agents.invest_agent.prompts.expert_prompts import ExpertPromptBuilder
from src.agents.invest_agent.skills.web_search_skill import WebSearchSkill


class ExpertExecutor(BaseExecutor):
    """
    Executor for EXPERT mode.

    Characteristics:
    - Uses larger model (gpt-4o / gemini-2.5-pro)
    - Maximum 6 agent turns
    - ALL tools available (31+)
    - Full system prompt with examples (~4000 tokens)
    - Web search enabled
    - Thinking display enabled
    - Parallel tool execution

    Similar to DeepResearchOrchestrator but single-phase.
    """

    MAX_TURNS = 6

    def __init__(self, config=None):
        super().__init__(config)
        self.prompt_builder = ExpertPromptBuilder()
        self.web_search = WebSearchSkill()

    async def execute(
        self,
        context: "ExecutionContext",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute query in EXPERT mode.

        Flow:
        1. Load ALL tools + web search
        2. Build full prompt with examples
        3. Run agent loop (max 6 turns)
        4. Enable thinking display
        5. Stream comprehensive response
        """
        import time
        start_time = time.time()

        # Emit mode info
        yield {
            "type": "mode_info",
            "mode": "expert",
            "max_turns": self.MAX_TURNS,
            "web_search_enabled": True,
        }

        # Get ALL tools + web search
        all_tools = self._get_all_tools()
        all_tools.extend(self.web_search.get_tool_definitions())

        yield {
            "type": "tools_loaded",
            "count": len(all_tools),
            "categories": self._get_tool_categories(all_tools),
        }

        # Build messages with full prompt
        messages = self.prompt_builder.build(
            query=context.query,
            language=context.language,
            history=context.conversation_history,
            core_memory=context.core_memory,
            include_examples=True,  # EXPERT includes few-shot examples
        )

        # Agent loop
        total_tool_calls = 0
        turn = 0

        for turn in range(1, self.MAX_TURNS + 1):
            yield {"type": "turn_start", "turn": turn, "max_turns": self.MAX_TURNS}

            # LLM call with all tools
            response = await self._call_llm(
                messages=messages,
                tools=all_tools,
                model=self._get_expert_model(context.provider_type),
                enable_thinking=True,
            )

            # Emit thinking if present
            if response.get("thinking"):
                yield {
                    "type": "thinking",
                    "content": response["thinking"],
                    "phase": f"turn_{turn}",
                }

            tool_calls = self._parse_tool_calls(response)
            content = response.get("content", "")

            # No tool calls = done
            if not tool_calls:
                if content:
                    for chunk in self._chunk_content(content):
                        yield {"type": "content", "content": chunk}
                break

            # Execute tools in parallel
            total_tool_calls += len(tool_calls)
            yield {
                "type": "tool_calls",
                "tools": [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls],
            }

            tool_results = await self._execute_tools_parallel(tool_calls)

            # Handle web search results specially
            for tc, result in zip(tool_calls, tool_results):
                if tc.name == "webSearch" and result.get("citations"):
                    yield {
                        "type": "sources",
                        "citations": result["citations"],
                        "count": len(result["citations"]),
                    }

            yield {
                "type": "tool_results",
                "results": [{"tool": tc.name, "success": r.get("status") == "success"}
                           for tc, r in zip(tool_calls, tool_results)],
            }

            # Update messages
            messages = self._update_messages(messages, tool_calls, tool_results)

        # Done
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield {
            "type": "done",
            "mode": "expert",
            "total_turns": turn,
            "total_tool_calls": total_tool_calls,
            "total_time_ms": elapsed_ms,
        }

    def _get_expert_model(self, provider_type: str) -> str:
        """Get expert model for provider"""
        models = {
            "openai": "gpt-4o",
            "gemini": "gemini-2.5-pro",
            "openrouter": "anthropic/claude-3.5-sonnet",
        }
        return models.get(provider_type, "gpt-4o")
```

### 6.5 Web Search Skill

```python
# src/agents/invest_agent/skills/web_search_skill.py
"""
WebSearchSkill - ChatGPT-style web search integration.

Provides real-time information beyond financial data tools:
- Current events
- Recent news
- Company announcements
- Market sentiment from web sources
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Web search result"""
    title: str
    url: str
    snippet: str
    source: str


class WebSearchSkill:
    """
    Web search skill for real-time information.

    Features:
    - Search for current events
    - Get recent news beyond FMP news
    - Verify facts with web sources
    - Return citations for transparency

    Integration:
    - Uses Tavily/Brave/SerpAPI as backend
    - Returns structured citations for FE rendering
    - Limits results to avoid context overflow
    """

    MAX_RESULTS = 5

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._get_api_key()

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling format"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "webSearch",
                    "description": (
                        "Search the web for current information, recent news, "
                        "or facts that may not be available in financial data tools. "
                        "Use for: current leaders, recent events, company announcements, "
                        "market-moving news. Do NOT use for stock prices or financials."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Max results (1-5)",
                                "default": 3,
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    async def execute(
        self,
        query: str,
        max_results: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute web search.

        Returns:
            {
                "status": "success",
                "results": [...],
                "citations": [...],  # For FE rendering
            }
        """
        # Implementation depends on backend (Tavily, Brave, etc.)
        results = await self._search(query, min(max_results, self.MAX_RESULTS))

        return {
            "status": "success",
            "query": query,
            "results": [r.__dict__ for r in results],
            "citations": [
                {"title": r.title, "url": r.url, "source": r.source}
                for r in results
            ],
        }

    async def _search(self, query: str, max_results: int) -> List[SearchResult]:
        """Actual search implementation"""
        # TODO: Implement with Tavily/Brave API
        # For now, return placeholder
        return []

    def _get_api_key(self) -> Optional[str]:
        """Get API key from settings"""
        from src.utils.config import settings
        return getattr(settings, 'WEB_SEARCH_API_KEY', None)
```

### 6.6 Tool Filter

```python
# src/agents/invest_agent/tools/tool_filter.py
"""
ToolFilter - Filter tools for FAST mode.

Reduces tool set from 31+ to 8 max based on:
- Query categories (price, technical, fundamentals, etc.)
- Always-include essentials
- Priority ranking
"""

from typing import List, Dict, Any, Optional


class ToolFilter:
    """
    Filters tools for FAST mode to reduce prompt tokens.

    Strategy:
    1. Always include essential tools (GetStockPrice)
    2. Add category-specific tools based on query
    3. Fill remaining slots from priority list
    4. Cap at max_tools (default 8)
    """

    # Tools always included
    ESSENTIAL_TOOLS = [
        "GetStockPrice",
    ]

    # Category to tools mapping
    CATEGORY_TOOLS = {
        "price": [
            "GetStockPrice",
            "GetStockPerformance",
            "GetPriceTargets",
        ],
        "technical": [
            "GetTechnicalIndicators",
            "DetectChartPatterns",
            "GetSupportResistance",
        ],
        "fundamentals": [
            "GetFinancialRatios",
            "GetIncomeStatement",
            "GetGrowthMetrics",
        ],
        "news": [
            "GetStockNews",
            "GetMarketNews",
            "GetCompanyEvents",
        ],
        "risk": [
            "AssessRisk",
            "GetSentiment",
            "GetVolumeProfile",
        ],
        "market": [
            "GetMarketMovers",
            "GetSectorPerformance",
            "GetTopGainers",
        ],
    }

    # Default priority when no category detected
    DEFAULT_PRIORITY = [
        "price", "technical", "fundamentals", "news"
    ]

    def __init__(self, max_tools: int = 8):
        self.max_tools = max_tools

    def filter(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        all_tools: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter tools based on query and categories.

        Args:
            query: User query (for fallback detection)
            categories: Pre-detected categories
            all_tools: Full tool catalog

        Returns:
            Filtered list of tools (max N)
        """
        if all_tools is None:
            all_tools = self._get_all_tools()

        selected = []
        selected_names = set()

        # 1. Essential tools first
        for name in self.ESSENTIAL_TOOLS:
            tool = self._find_tool(all_tools, name)
            if tool:
                selected.append(tool)
                selected_names.add(name)

        # 2. Category-specific tools
        categories = categories or self._detect_categories(query)
        for category in categories:
            if len(selected) >= self.max_tools:
                break
            for tool_name in self.CATEGORY_TOOLS.get(category, []):
                if tool_name not in selected_names:
                    tool = self._find_tool(all_tools, tool_name)
                    if tool and len(selected) < self.max_tools:
                        selected.append(tool)
                        selected_names.add(tool_name)

        # 3. Fill from default priority
        if len(selected) < self.max_tools:
            for category in self.DEFAULT_PRIORITY:
                if len(selected) >= self.max_tools:
                    break
                for tool_name in self.CATEGORY_TOOLS.get(category, []):
                    if tool_name not in selected_names:
                        tool = self._find_tool(all_tools, tool_name)
                        if tool and len(selected) < self.max_tools:
                            selected.append(tool)
                            selected_names.add(tool_name)

        return selected

    def _find_tool(
        self,
        tools: List[Dict],
        name: str,
    ) -> Optional[Dict[str, Any]]:
        """Find tool by name"""
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name == name:
                return tool
        return None

    def _detect_categories(self, query: str) -> List[str]:
        """Fallback category detection"""
        query_lower = query.lower()
        categories = ["price"]  # Default

        keywords = {
            "technical": ["technical", "indicator", "rsi", "macd", "chart"],
            "fundamentals": ["financial", "ratio", "pe", "revenue", "earnings"],
            "news": ["news", "tin", "headline", "announcement"],
            "risk": ["risk", "volatility", "sentiment"],
        }

        for category, kws in keywords.items():
            if any(kw in query_lower for kw in kws):
                categories.append(category)

        return categories

    def _get_all_tools(self) -> List[Dict[str, Any]]:
        """Get full tool catalog"""
        from src.agents.tools.registry import get_registry
        registry = get_registry()
        return [
            schema.to_openai_function()
            for schema in registry.get_all_schemas().values()
        ]
```

---

## 7. Integration Strategy

### 7.1 Phase 1: Standalone Testing (No Production Impact)

```python
# New endpoint in src/routers/v2/chat_invest.py
# SEPARATE from chat_assistant.py

@router.post("/chat/invest")
async def chat_invest(request: Request, data: ChatRequest):
    """
    New InvestAgent endpoint for testing.

    Feature flag: invest_agent_v1
    If disabled, returns 501 Not Implemented.
    """
    if not feature_flags.is_enabled("invest_agent_v1"):
        raise HTTPException(501, "InvestAgent not enabled")

    orchestrator = InvestAgentOrchestrator()

    async def generate():
        async for event in orchestrator.run(
            query=data.query,
            session_id=data.session_id,
            user_id=str(request.state.user_id),
            explicit_mode=data.mode,
        ):
            yield format_sse_event(event)

    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 7.2 Phase 2: Shadow Mode (Compare Results)

```python
# In chat_assistant.py (minimal change)

async def stream_chat_v2(...):
    # ... existing code ...

    # Shadow mode: run InvestAgent in parallel, log results
    if feature_flags.is_enabled("invest_agent_shadow"):
        asyncio.create_task(_shadow_invest_agent(
            query=query,
            session_id=session_id,
            user_id=user_id,
        ))

    # ... continue with existing logic ...
```

### 7.3 Phase 3: Gradual Rollout

```python
# Feature flag with percentage rollout
feature_flags = {
    "invest_agent_v1": {
        "enabled": True,
        "percentage": 10,  # 10% of requests
        "whitelist_users": ["user_123", "user_456"],
    }
}
```

### 7.4 Integration Points

| Touchpoint | Description | Risk |
|------------|-------------|------|
| `/chat/invest` | New endpoint, no existing code | None |
| `chat_assistant.py` | Shadow mode only | Very Low |
| `UnifiedAgent` | Fallback when flag disabled | None |
| `ToolRegistry` | Reuse existing | None |
| `StreamEventEmitter` | Reuse SSE format | None |

---

## 8. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

| Task | Description | Owner |
|------|-------------|-------|
| Create folder structure | `src/agents/invest_agent/` | Dev |
| Config module | `mode_config.py`, `feature_flags.py` | Dev |
| Core orchestrator | Basic orchestrator skeleton | Dev |
| Mode selector | FAST/EXPERT selection logic | Dev |

**Deliverables:**
- [ ] Folder structure created
- [ ] Config classes implemented
- [ ] Orchestrator skeleton works
- [ ] Unit tests pass

### Phase 2: Executors (Week 2)

| Task | Description | Owner |
|------|-------------|-------|
| Base executor | Abstract interface | Dev |
| Fast executor | 2 turns, filtered tools | Dev |
| Expert executor | 6 turns, all tools | Dev |
| Tool filter | Category-based filtering | Dev |

**Deliverables:**
- [ ] FastExecutor runs simple queries
- [ ] ExpertExecutor runs complex queries
- [ ] Tool filtering reduces to 8 tools

### Phase 3: Prompts & Skills (Week 3)

| Task | Description | Owner |
|------|-------------|-------|
| Fast prompts | Condensed VI/EN prompts | Dev |
| Expert prompts | Full prompts with examples | Dev |
| Web search skill | ChatGPT-style search | Dev |
| Provider prompts | GPT/Gemini variations | Dev |

**Deliverables:**
- [ ] Mode-specific prompts work
- [ ] Web search integrated
- [ ] Provider-specific optimizations

### Phase 4: Integration & Testing (Week 4)

| Task | Description | Owner |
|------|-------------|-------|
| New endpoint | `/chat/invest` | Dev |
| Shadow mode | Compare with existing | Dev |
| Integration tests | E2E test suite | QA |
| Performance tests | Latency benchmarks | Dev |

**Deliverables:**
- [ ] `/chat/invest` endpoint works
- [ ] Shadow mode logs comparisons
- [ ] Tests pass
- [ ] FAST P50 < 4s

---

## 9. Risk Mitigation

### 9.1 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Production disruption | Low | High | Separate endpoint, feature flags |
| Performance regression | Medium | Medium | Benchmark against existing |
| Tool filtering breaks queries | Medium | Medium | Fallback to all tools |
| Web search quota exceeded | Low | Low | Rate limiting, caching |
| Mode selector wrong choice | Medium | Low | User can override mode |

### 9.2 Fallback Strategy

```python
# In orchestrator
async def run(self, ...):
    try:
        # ... InvestAgent logic ...
    except Exception as e:
        logger.error(f"InvestAgent failed: {e}")

        # Fallback to existing UnifiedAgent
        yield {"type": "fallback", "reason": str(e)}

        unified_agent = get_unified_agent()
        async for event in unified_agent.run_stream_with_all_tools(...):
            yield event
```

### 9.3 Rollback Procedure

1. Disable feature flag: `invest_agent_v1 = False`
2. No code revert needed
3. All traffic returns to existing flow
4. Investigate and fix issues
5. Re-enable gradually

---

## 10. Success Metrics

### 10.1 Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| FAST mode P50 latency | ≤ 4 seconds | APM monitoring |
| FAST mode P90 latency | ≤ 6 seconds | APM monitoring |
| EXPERT mode P50 latency | ≤ 20 seconds | APM monitoring |
| Tool filter accuracy | > 90% relevant | Manual review |
| Mode selection accuracy | > 85% correct | User feedback |

### 10.2 Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Response quality | No degradation | A/B testing |
| Error rate | < 1% | Error tracking |
| User satisfaction | > 4/5 rating | Feedback |
| Code coverage | > 80% | Test suite |

### 10.3 Business Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Token usage reduction | 30% in FAST | Cost tracking |
| Response time improvement | 40% faster for simple queries | APM |
| Feature adoption | 50% of queries use FAST | Analytics |

---

## Appendix A: File Template Examples

### A.1 `__init__.py` Template

```python
# src/agents/invest_agent/__init__.py
"""
InvestAgent - Response Mode Agent for FAST/AUTO/EXPERT modes.

This module provides intelligent response mode selection and execution
for financial analysis queries.

Usage:
    from src.agents.invest_agent import InvestAgentOrchestrator

    orchestrator = InvestAgentOrchestrator()
    async for event in orchestrator.run(query="AAPL price?", ...):
        print(event)

Public API:
    - InvestAgentOrchestrator: Main entry point
    - ResponseMode: FAST/AUTO/EXPERT enum
    - InvestAgentConfig: Configuration class
    - get_invest_agent: Factory function (singleton)
"""

from src.agents.invest_agent.core.orchestrator import (
    InvestAgentOrchestrator,
    ResponseMode,
    ExecutionContext,
)
from src.agents.invest_agent.config.mode_config import InvestAgentConfig
from src.agents.invest_agent.config.feature_flags import FeatureFlags

__all__ = [
    "InvestAgentOrchestrator",
    "ResponseMode",
    "ExecutionContext",
    "InvestAgentConfig",
    "FeatureFlags",
    "get_invest_agent",
]

# Singleton
_orchestrator_instance = None


def get_invest_agent(
    config: InvestAgentConfig = None,
) -> InvestAgentOrchestrator:
    """Get singleton InvestAgent orchestrator."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = InvestAgentOrchestrator(config=config)
    return _orchestrator_instance
```

### A.2 Config Template

```python
# src/agents/invest_agent/config/mode_config.py
"""
Mode configuration for InvestAgent.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class InvestAgentConfig:
    """Configuration for InvestAgent."""

    # FAST mode settings
    fast_max_turns: int = 2
    fast_max_tools: int = 8
    fast_timeout_ms: int = 10000  # 10s
    fast_model_openai: str = "gpt-4o-mini"
    fast_model_gemini: str = "gemini-2.0-flash"

    # EXPERT mode settings
    expert_max_turns: int = 6
    expert_timeout_ms: int = 60000  # 60s
    expert_model_openai: str = "gpt-4o"
    expert_model_gemini: str = "gemini-2.5-pro"
    expert_enable_web_search: bool = True
    expert_enable_thinking: bool = True

    # AUTO mode settings
    auto_complexity_threshold: float = 0.6
    auto_escalation_enabled: bool = True  # Allow FAST → EXPERT

    # Prompt settings
    fast_max_prompt_tokens: int = 1500
    expert_max_prompt_tokens: int = 4000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "fast_max_turns": self.fast_max_turns,
            "fast_max_tools": self.fast_max_tools,
            "expert_max_turns": self.expert_max_turns,
            "expert_enable_web_search": self.expert_enable_web_search,
        }
```

---

## Appendix B: Related Documents

- [ARCHITECTURE_REVIEW_AND_SCALING_PLAN.md](./ARCHITECTURE_REVIEW_AND_SCALING_PLAN.md) - Original scaling plan
- [CHAT_RESPONSE_MODES_IMPLEMENTATION_PLAN.md](./CHAT_RESPONSE_MODES_IMPLEMENTATION_PLAN.md) - Mode specifications
- [CHAT_AGENT_REDESIGN_PROPOSAL.md](./CHAT_AGENT_REDESIGN_PROPOSAL.md) - Architecture redesign

---

**Document Version:** 1.0
**Last Updated:** 2026-01-23
**Next Review:** After Phase 1 completion

---

**Summary:**

Document này mô tả chi tiết kế hoạch triển khai `invest_agent` folder mới với:
- **Separation từ production code** - Không modify `chat_assistant.py`
- **Clean architecture** - Học từ `deep_research/` và `normal_mode/`
- **Feature flag control** - Gradual rollout, easy rollback
- **Skill integration** - Web search như ChatGPT
- **Clear naming** - File/folder theo conventions đã có

Total estimated effort: **4 weeks** cho full implementation.
