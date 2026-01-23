# Architecture Review & Scaling Plan for Response Modes

**Version:** 1.0
**Date:** 2026-01-23
**Status:** Review Complete
**Author:** Architecture Review

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Architecture Assessment](#2-current-architecture-assessment)
3. [Strengths Analysis](#3-strengths-analysis)
4. [Areas for Improvement](#4-areas-for-improvement)
5. [Proposed Directory Structure](#5-proposed-directory-structure)
6. [Detailed Implementation Tasks](#6-detailed-implementation-tasks)
7. [Provider-Specific Considerations](#7-provider-specific-considerations)
8. [Implementation Priority](#8-implementation-priority)

---

## 1. Executive Summary

### 1.1 Review Objective

Evaluate the current codebase architecture for scalability with:
- **Response Modes**: FAST / AUTO / EXPERT
- **Multiple Providers**: OpenAI (GPT), Google (Gemini), OpenRouter, Ollama
- **Provider-Specific Prompts**: Different prompt optimizations per provider

### 1.2 Overall Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Mode Configuration** | ⭐⭐⭐⭐⭐ Excellent | Well-structured `ModeConfig` dataclass |
| **Provider Abstraction** | ⭐⭐⭐⭐ Good | Solid base, needs enhancement |
| **Prompt Management** | ⭐⭐⭐ Adequate | Needs provider-specific templates |
| **Directory Organization** | ⭐⭐⭐⭐ Good | Some restructuring needed |
| **Documentation** | ⭐⭐⭐⭐⭐ Excellent | Comprehensive implementation plans exist |
| **Scalability Readiness** | ⭐⭐⭐⭐ Good | Foundation is solid, minor gaps to fill |

### 1.3 Key Verdict

**The architecture is fundamentally sound and well-designed for scaling.** The existing mode configurations, provider factory pattern, and documentation are excellent. Minor restructuring is needed for:
1. Provider-specific prompt templates
2. Mode-specific executors
3. Centralized feature flags
4. Model capability catalog

---

## 2. Current Architecture Assessment

### 2.1 Directory Structure Overview

```
src/
├── agents/
│   ├── classification/          # Intent classifier
│   ├── routing/                 # ✅ Mode router (NEW)
│   │   └── mode_router.py
│   ├── memory/                  # Memory tiers (core, recursive, working)
│   ├── planning/                # Planning agents
│   ├── streaming/               # SSE events
│   ├── tools/                   # 31+ tools by category
│   │   ├── price/
│   │   ├── technical/
│   │   ├── fundamentals/
│   │   ├── news/
│   │   ├── market/
│   │   ├── risk/
│   │   └── ...
│   └── unified/                 # Main agent executor
│
├── config/
│   └── mode_config.py           # ✅ Response mode configurations
│
├── providers/
│   ├── base_provider.py         # ✅ Abstract interface
│   ├── openai_provider.py       # ✅ OpenAI implementation
│   ├── gemini_provider.py       # ✅ Gemini implementation
│   ├── ollama_provider.py       # ✅ Ollama implementation
│   ├── openrouter_provider.py   # ✅ OpenRouter implementation
│   └── provider_factory.py      # ✅ Factory with typo correction
│
├── prompts/
│   ├── prompt_manager.py        # Basic template system
│   └── planning_prompts.py
│
├── handlers/v2/
│   └── chat_handler.py          # Main request handler
│
└── routers/v2/
    ├── chat.py                  # API endpoints
    └── chat_assistant.py        # Chat logic
```

### 2.2 Existing Components Analysis

#### Mode Configuration (`src/config/mode_config.py`)

**Status: EXCELLENT** ✅

```python
# Already implemented:
- ResponseMode enum (FAST, AUTO, EXPERT)
- ModeConfig dataclass with 20+ configuration options
- FAST_MODE_CONFIG, EXPERT_MODE_CONFIG, AUTO_MODE_CONFIG
- MODE_CONFIGS lookup dictionary
- get_mode_config() and get_effective_config() helpers
- ModeClassificationResult dataclass
- RESPONSE_MODES_FEATURE_FLAGS
```

**Assessment**: This is well-designed and production-ready. No changes needed.

#### Provider System (`src/providers/`)

**Status: GOOD** ✅

```python
# Already implemented:
- ModelProvider abstract base class
- 4 provider implementations (OpenAI, Gemini, Ollama, OpenRouter)
- ModelProviderFactory with:
  - Typo correction ("geminni" -> "gemini")
  - API key management
  - Provider availability checking
```

**Assessment**: Solid foundation. Needs enhancement for:
- Provider health checks
- Fallback mechanism
- Provider-specific feature flags

#### Prompt Management (`src/prompts/`)

**Status: NEEDS IMPROVEMENT** ⚠️

```python
# Current state:
- Basic PromptTemplate class
- Generic templates in prompt_manager.py
- No provider-specific variations
- No mode-specific system prompts
```

**Assessment**: Major gap for scaling. Needs:
- Provider-specific prompt variations (GPT vs Gemini have different optimal formats)
- Mode-specific system prompts (FAST condensed vs EXPERT full)
- Template versioning

#### Mode Router (`src/agents/routing/mode_router.py`)

**Status: IMPLEMENTED** ✅

Based on the implementation plan, the ModeRouter class exists with:
- LLM-based complexity classification
- Context continuity (inherits previous mode)
- Classification caching
- Quick heuristics bypass

---

## 3. Strengths Analysis

### 3.1 What's Already Excellent

| Component | Why It's Good |
|-----------|---------------|
| **ModeConfig dataclass** | Comprehensive configuration covering model, tools, timeouts, prompts |
| **Feature flags** | Easy rollout control with `RESPONSE_MODES_FEATURE_FLAGS` |
| **Provider Factory** | Clean abstraction, typo correction, centralized creation |
| **Tool Organization** | Well-categorized by domain (price, technical, fundamentals, etc.) |
| **Memory System** | 3-tier hierarchy (core, recursive, working) |
| **Documentation** | Industry research-backed implementation plans |
| **Streaming Events** | Comprehensive SSE event types for UI feedback |

### 3.2 Reusable Components

These components can be used as-is for scaling:

```
✅ src/config/mode_config.py          - Mode configurations
✅ src/providers/provider_factory.py  - Provider creation
✅ src/providers/*_provider.py        - All 4 provider implementations
✅ src/agents/routing/mode_router.py  - Mode routing logic
✅ src/agents/tools/tool_catalog.py   - Tool registry
✅ src/agents/memory/                  - Memory system
✅ src/agents/unified/unified_agent.py - Agent executor
```

---

## 4. Areas for Improvement

### 4.1 Gap Analysis

| Gap | Impact | Priority | Effort |
|-----|--------|----------|--------|
| No provider-specific prompts | GPT/Gemini have different optimal formats | HIGH | Medium |
| No mode-specific executors | Code duplication risk | MEDIUM | Medium |
| Feature flags in mode_config | Should be centralized | LOW | Low |
| No model capability catalog | Manual capability tracking | MEDIUM | Low |
| No provider health checks | Silent failures possible | MEDIUM | Low |
| No tool filter for FAST mode | Tool filtering not implemented | HIGH | Medium |

### 4.2 Provider-Specific Prompt Gap

**Current Problem:**
```python
# prompt_manager.py - ONE template for all providers
"chat_system": PromptTemplate(
    "You are a helpful assistant..."  # Same for GPT and Gemini
)
```

**Why It Matters:**
- **GPT models**: Prefer concise system prompts, support `response_format`
- **Gemini models**: Need `thought_signature` preservation, different tool call format
- **Claude models** (via OpenRouter): Support extended thinking budget

**Proposed Solution:**
```
src/prompts/templates/
├── modes/
│   ├── fast_system.py       # Condensed prompt for FAST mode
│   └── expert_system.py     # Full prompt for EXPERT mode
└── providers/
    ├── openai_prompts.py    # GPT-optimized variations
    ├── gemini_prompts.py    # Gemini-optimized variations
    └── openrouter_prompts.py # Claude/other model variations
```

### 4.3 Mode Executor Gap

**Current Problem:**
- Mode-specific logic will be scattered in chat_handler.py
- No dedicated execution path for FAST vs EXPERT

**Proposed Solution:**
```
src/agents/executors/
├── __init__.py
├── base_executor.py         # Abstract executor interface
├── fast_executor.py         # Optimized for speed (2 turns, filtered tools)
└── expert_executor.py       # Full capability (6 turns, all tools)
```

---

## 5. Proposed Directory Structure

### 5.1 Enhanced Structure for Scaling

```
src/
├── config/
│   ├── __init__.py
│   ├── mode_config.py              # ✅ KEEP - Response mode configs
│   ├── feature_flags.py            # 🆕 NEW - Centralized feature flags
│   │
│   ├── providers/                  # 🆕 NEW - Provider-specific configs
│   │   ├── __init__.py
│   │   ├── base_provider_config.py # Base configuration class
│   │   ├── openai_config.py        # GPT models, limits, defaults
│   │   ├── gemini_config.py        # Gemini models, limits, defaults
│   │   ├── ollama_config.py        # Ollama endpoint configs
│   │   └── openrouter_config.py    # OpenRouter model mappings
│   │
│   └── models/                     # 🆕 NEW - Model capability catalog
│       ├── __init__.py
│       └── model_catalog.py        # Supported models & capabilities
│
├── providers/                      # ✅ KEEP - Enhance existing
│   ├── __init__.py
│   ├── base_provider.py            # ✅ KEEP - Abstract interface
│   ├── openai_provider.py          # ✅ KEEP - Add health check
│   ├── gemini_provider.py          # ✅ KEEP - Add health check
│   ├── ollama_provider.py          # ✅ KEEP - Add health check
│   ├── openrouter_provider.py      # ✅ KEEP - Add health check
│   ├── provider_factory.py         # ✅ KEEP - Add fallback logic
│   └── health_check.py             # 🆕 NEW - Provider health monitoring
│
├── prompts/
│   ├── __init__.py
│   ├── prompt_manager.py           # ✅ KEEP - Enhance for modes/providers
│   ├── planning_prompts.py         # ✅ KEEP
│   │
│   └── templates/                  # 🆕 NEW - Organized templates
│       ├── __init__.py
│       │
│       ├── modes/                  # Mode-specific system prompts
│       │   ├── __init__.py
│       │   ├── fast_mode_prompts.py    # Condensed prompts
│       │   └── expert_mode_prompts.py  # Full prompts with examples
│       │
│       └── providers/              # Provider-specific variations
│           ├── __init__.py
│           ├── openai_prompts.py       # GPT-optimized
│           ├── gemini_prompts.py       # Gemini-optimized
│           └── openrouter_prompts.py   # Multi-model gateway
│
├── agents/
│   ├── routing/                    # ✅ KEEP
│   │   ├── __init__.py
│   │   └── mode_router.py
│   │
│   ├── executors/                  # 🆕 NEW - Mode executors
│   │   ├── __init__.py
│   │   ├── base_executor.py        # Abstract executor
│   │   ├── fast_executor.py        # FAST mode execution
│   │   └── expert_executor.py      # EXPERT mode execution
│   │
│   ├── tools/
│   │   ├── ... (existing)
│   │   └── tool_filter.py          # 🆕 NEW - Tool filtering for FAST
│   │
│   └── ... (existing directories)
│
└── handlers/v2/
    └── chat_handler.py             # ✅ MODIFY - Integrate mode routing
```

### 5.2 New Files Summary

| File | Purpose | Priority |
|------|---------|----------|
| `src/config/feature_flags.py` | Centralized feature flag management | P2 |
| `src/config/providers/*.py` | Provider-specific configurations | P1 |
| `src/config/models/model_catalog.py` | Model capabilities & limits | P2 |
| `src/providers/health_check.py` | Provider availability monitoring | P2 |
| `src/prompts/templates/modes/*.py` | Mode-specific prompts | P0 |
| `src/prompts/templates/providers/*.py` | Provider-specific prompts | P1 |
| `src/agents/executors/*.py` | Mode execution logic | P1 |
| `src/agents/tools/tool_filter.py` | Tool filtering for FAST | P0 |

---

## 6. Detailed Implementation Tasks

### 6.1 Phase 1: Core Infrastructure (Priority P0)

#### Task 1.1: Mode-Specific System Prompts

**File:** `src/prompts/templates/modes/fast_mode_prompts.py`

```python
"""
FAST Mode System Prompts - Condensed for speed
Target: 1500 tokens max system prompt
"""

FAST_SYSTEM_PROMPT_VI = """
Bạn là trợ lý tài chính. Trả lời ngắn gọn, chính xác.

Quy tắc:
- Trả lời trực tiếp, không giải thích dài
- Ưu tiên dữ liệu số liệu cụ thể
- Nếu cần nhiều bước, chỉ thực hiện tối đa 2 bước
"""

FAST_SYSTEM_PROMPT_EN = """
You are a financial assistant. Respond concisely and accurately.

Rules:
- Answer directly without lengthy explanations
- Prioritize specific numerical data
- Maximum 2 steps if multi-step needed
"""
```

**File:** `src/prompts/templates/modes/expert_mode_prompts.py`

```python
"""
EXPERT Mode System Prompts - Full capability
Target: 4000 tokens max system prompt
"""

EXPERT_SYSTEM_PROMPT_VI = """
Bạn là chuyên gia phân tích tài chính cao cấp với khả năng:

## Phân tích kỹ thuật
- Đọc và phân tích các chỉ báo (RSI, MACD, Moving Averages)
- Nhận diện pattern chart
- Xác định support/resistance

## Phân tích cơ bản
- Đọc báo cáo tài chính (Income Statement, Balance Sheet, Cash Flow)
- Tính toán và so sánh ratios
- Đánh giá định giá (P/E, P/B, DCF)

## Nghiên cứu thị trường
- Theo dõi tin tức và sự kiện
- Phân tích sentiment
- So sánh với peers

Quy trình:
1. Hiểu rõ câu hỏi và context
2. Lên kế hoạch các bước cần thiết
3. Thu thập dữ liệu qua tools
4. Phân tích và tổng hợp
5. Trả lời đầy đủ với reasoning

Bạn có quyền sử dụng web search và tất cả tools có sẵn.
"""
```

#### Task 1.2: Tool Filter for FAST Mode

**File:** `src/agents/tools/tool_filter.py`

```python
"""
Tool filtering for FAST mode
Selects top N relevant tools based on query classification
"""

from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ToolFilterConfig:
    """Configuration for tool filtering"""
    max_tools: int = 8
    always_include: List[str] = None  # Tools always available
    category_priority: List[str] = None  # Category order

# Category to tool mapping
TOOL_CATEGORIES = {
    "price": [
        "GetStockPrice",
        "GetStockPerformance",
        "GetPriceTargets"
    ],
    "technical": [
        "GetTechnicalIndicators",
        "GetChartPatterns",
        "GetSupportResistance",
        "GetRelativeStrength"
    ],
    "fundamentals": [
        "GetFinancialRatios",
        "GetIncomeStatement",
        "GetBalanceSheet",
        "GetGrowthMetrics"
    ],
    "news": [
        "GetStockNews",
        "GetMarketNews"
    ],
    "memory": [
        "RetrieveUserPreferences"
    ]
}

class ToolFilter:
    """Filters tools based on mode and query classification"""

    def __init__(self, config: ToolFilterConfig = None):
        self.config = config or ToolFilterConfig(
            max_tools=8,
            always_include=["GetStockPrice", "RetrieveUserPreferences"],
            category_priority=["price", "technical", "fundamentals", "memory"]
        )

    def filter_for_fast_mode(
        self,
        all_tools: List[Dict[str, Any]],
        query_categories: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter tools for FAST mode based on query classification.

        Args:
            all_tools: Full tool catalog
            query_categories: Detected categories from classifier

        Returns:
            Filtered list of tools (max N)
        """
        selected_tools = []
        selected_names = set()

        # 1. Always include essential tools
        for tool in all_tools:
            if tool["name"] in self.config.always_include:
                selected_tools.append(tool)
                selected_names.add(tool["name"])

        # 2. Add tools from detected categories
        if query_categories:
            for category in query_categories:
                if category in TOOL_CATEGORIES:
                    for tool_name in TOOL_CATEGORIES[category]:
                        if tool_name not in selected_names:
                            tool = self._find_tool(all_tools, tool_name)
                            if tool and len(selected_tools) < self.config.max_tools:
                                selected_tools.append(tool)
                                selected_names.add(tool_name)

        # 3. Fill remaining slots from priority categories
        for category in self.config.category_priority:
            if len(selected_tools) >= self.config.max_tools:
                break
            for tool_name in TOOL_CATEGORIES.get(category, []):
                if tool_name not in selected_names:
                    tool = self._find_tool(all_tools, tool_name)
                    if tool and len(selected_tools) < self.config.max_tools:
                        selected_tools.append(tool)
                        selected_names.add(tool_name)

        return selected_tools

    def _find_tool(self, tools: List[Dict], name: str) -> Dict[str, Any]:
        """Find tool by name in tool list"""
        for tool in tools:
            if tool.get("name") == name:
                return tool
        return None
```

### 6.2 Phase 2: Provider-Specific Enhancements (Priority P1)

#### Task 2.1: Provider Configuration

**File:** `src/config/providers/openai_config.py`

```python
"""
OpenAI Provider Configuration
Model-specific settings for GPT models
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class OpenAIModelConfig:
    """Configuration for an OpenAI model"""
    model_id: str
    display_name: str
    max_tokens: int
    supports_function_calling: bool = True
    supports_json_mode: bool = True
    supports_vision: bool = False
    supports_streaming: bool = True
    context_window: int = 128000
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

# Model configurations
OPENAI_MODELS = {
    "gpt-4o": OpenAIModelConfig(
        model_id="gpt-4o",
        display_name="GPT-4o",
        max_tokens=16384,
        supports_vision=True,
        context_window=128000,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
    ),
    "gpt-4o-mini": OpenAIModelConfig(
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        max_tokens=16384,
        supports_vision=True,
        context_window=128000,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
    ),
    "gpt-4-turbo": OpenAIModelConfig(
        model_id="gpt-4-turbo",
        display_name="GPT-4 Turbo",
        max_tokens=4096,
        supports_vision=True,
        context_window=128000,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
    ),
}

# Recommended models per mode
OPENAI_MODE_MODELS = {
    "fast": "gpt-4o-mini",
    "expert": "gpt-4o",
    "auto": "gpt-4o-mini",  # Default, may upgrade
}

# OpenAI-specific prompt optimizations
OPENAI_PROMPT_GUIDELINES = """
- Use clear section headers with ##
- Prefer numbered lists for steps
- Keep system prompt under 4000 tokens for best performance
- Use JSON mode for structured outputs when possible
"""
```

**File:** `src/config/providers/gemini_config.py`

```python
"""
Gemini Provider Configuration
Model-specific settings for Google Gemini models
"""

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class GeminiModelConfig:
    """Configuration for a Gemini model"""
    model_id: str
    display_name: str
    max_tokens: int
    supports_function_calling: bool = True
    supports_vision: bool = True
    supports_streaming: bool = True
    context_window: int = 1000000  # Gemini has huge context
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
    preserve_thought_signature: bool = True  # Important for multi-turn

GEMINI_MODELS = {
    "gemini-2.0-flash": GeminiModelConfig(
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        max_tokens=8192,
        context_window=1000000,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
    ),
    "gemini-2.5-pro": GeminiModelConfig(
        model_id="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        max_tokens=8192,
        context_window=2000000,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
    ),
    "gemini-1.5-pro": GeminiModelConfig(
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        max_tokens=8192,
        context_window=1000000,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
    ),
}

GEMINI_MODE_MODELS = {
    "fast": "gemini-2.0-flash",
    "expert": "gemini-2.5-pro",
    "auto": "gemini-2.0-flash",
}

# Gemini-specific considerations
GEMINI_SPECIAL_HANDLING = """
- IMPORTANT: Preserve thought_signature across multi-turn conversations
- Tool call format differs from OpenAI - handled by provider
- Supports very long context - can include more history
- May need grounding for factual queries
"""
```

#### Task 2.2: Provider-Specific Prompt Templates

**File:** `src/prompts/templates/providers/openai_prompts.py`

```python
"""
OpenAI/GPT-optimized prompt templates
Designed for optimal performance with GPT-4 family models
"""

from typing import Dict, Any

class OpenAIPromptBuilder:
    """Builds prompts optimized for OpenAI models"""

    @staticmethod
    def build_system_prompt(
        mode: str,
        language: str = "en",
        include_examples: bool = True
    ) -> str:
        """
        Build system prompt optimized for GPT models.

        GPT optimization notes:
        - Prefers structured markdown with headers
        - Works well with numbered instructions
        - Response quality improves with few-shot examples
        """
        base_prompt = OPENAI_BASE_PROMPTS.get(mode, {}).get(language, "")

        if include_examples and mode == "expert":
            base_prompt += "\n\n" + OPENAI_FEW_SHOT_EXAMPLES.get(language, "")

        return base_prompt

    @staticmethod
    def adjust_for_model(prompt: str, model_name: str) -> str:
        """Adjust prompt based on specific model capabilities"""
        if "mini" in model_name:
            # Shorter context for mini models
            return prompt[:6000] if len(prompt) > 6000 else prompt
        return prompt

OPENAI_BASE_PROMPTS = {
    "fast": {
        "vi": """Bạn là trợ lý tài chính nhanh gọn.

## Quy tắc
1. Trả lời trực tiếp trong 1-2 câu
2. Ưu tiên số liệu cụ thể
3. Không giải thích dài dòng""",

        "en": """You are a quick financial assistant.

## Rules
1. Answer directly in 1-2 sentences
2. Prioritize specific numbers
3. No lengthy explanations"""
    },

    "expert": {
        "vi": """Bạn là chuyên gia phân tích tài chính cao cấp.

## Khả năng
- Phân tích kỹ thuật chuyên sâu
- Phân tích cơ bản toàn diện
- Nghiên cứu và so sánh đa chiều

## Quy trình phân tích
1. Hiểu câu hỏi và xác định scope
2. Lập kế hoạch thu thập dữ liệu
3. Gọi tools cần thiết
4. Phân tích và tổng hợp
5. Đưa ra kết luận với reasoning""",

        "en": """You are a senior financial analyst expert.

## Capabilities
- Deep technical analysis
- Comprehensive fundamental analysis
- Multi-dimensional research and comparison

## Analysis Process
1. Understand question and scope
2. Plan data collection
3. Call necessary tools
4. Analyze and synthesize
5. Provide conclusion with reasoning"""
    }
}

OPENAI_FEW_SHOT_EXAMPLES = {
    "vi": """
## Ví dụ phân tích

User: So sánh AAPL và MSFT
Assistant: Tôi sẽ so sánh hai công ty theo các tiêu chí:

**1. Định giá**
[Gọi GetFinancialRatios cho cả hai]

**2. Tăng trưởng**
[Gọi GetGrowthMetrics]

**3. Kỹ thuật**
[Gọi GetTechnicalIndicators]

Kết luận: [Tổng hợp dựa trên data]""",

    "en": """
## Analysis Example

User: Compare AAPL and MSFT
Assistant: I will compare both companies on these criteria:

**1. Valuation**
[Call GetFinancialRatios for both]

**2. Growth**
[Call GetGrowthMetrics]

**3. Technical**
[Call GetTechnicalIndicators]

Conclusion: [Synthesis based on data]"""
}
```

**File:** `src/prompts/templates/providers/gemini_prompts.py`

```python
"""
Gemini-optimized prompt templates
Designed for optimal performance with Google Gemini models

IMPORTANT: Gemini has different behavior:
- Uses thought_signature for reasoning continuity
- Has larger context window (1M+ tokens)
- Different tool call format (handled by provider)
"""

from typing import Dict, Any

class GeminiPromptBuilder:
    """Builds prompts optimized for Gemini models"""

    @staticmethod
    def build_system_prompt(
        mode: str,
        language: str = "en",
        include_grounding: bool = False
    ) -> str:
        """
        Build system prompt optimized for Gemini models.

        Gemini optimization notes:
        - Can handle very long contexts
        - Works well with natural language instructions
        - May benefit from grounding for factual queries
        """
        base_prompt = GEMINI_BASE_PROMPTS.get(mode, {}).get(language, "")

        if include_grounding:
            base_prompt += "\n\n" + GEMINI_GROUNDING_INSTRUCTION

        return base_prompt

    @staticmethod
    def preserve_thought_signature(messages: list, thought_signature: str) -> list:
        """
        Preserve Gemini's thought_signature across turns.
        This is critical for reasoning continuity in multi-turn.
        """
        if thought_signature and messages:
            # Add thought signature to maintain reasoning context
            for msg in messages:
                if msg.get("role") == "assistant":
                    if "thought_signature" not in msg:
                        msg["thought_signature"] = thought_signature
        return messages

GEMINI_BASE_PROMPTS = {
    "fast": {
        "vi": """Bạn là trợ lý tài chính nhanh của Google.
Trả lời ngắn gọn và chính xác. Tối đa 2 bước xử lý.""",

        "en": """You are Google's quick financial assistant.
Answer concisely and accurately. Maximum 2 processing steps."""
    },

    "expert": {
        "vi": """Bạn là chuyên gia phân tích tài chính của Google với khả năng:

Phân tích kỹ thuật: Chỉ báo, patterns, trends
Phân tích cơ bản: Báo cáo tài chính, ratios, định giá
Nghiên cứu thị trường: Tin tức, sentiment, so sánh

Bạn có context window rất lớn và có thể xử lý nhiều data cùng lúc.
Hãy phân tích toàn diện và đưa ra insight sâu sắc.""",

        "en": """You are Google's expert financial analyst with capabilities:

Technical Analysis: Indicators, patterns, trends
Fundamental Analysis: Financial reports, ratios, valuation
Market Research: News, sentiment, comparisons

You have a very large context window and can process lots of data simultaneously.
Provide comprehensive analysis and deep insights."""
    }
}

GEMINI_GROUNDING_INSTRUCTION = """
When providing factual information, ensure accuracy by:
- Using tool results for real-time data
- Clearly stating when information might be outdated
- Distinguishing between analysis and facts
"""
```

### 6.3 Phase 3: Mode Executors (Priority P1)

#### Task 3.1: Base Executor Interface

**File:** `src/agents/executors/base_executor.py`

```python
"""
Base executor interface for response modes
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional
from dataclasses import dataclass

from src.config.mode_config import ModeConfig

@dataclass
class ExecutionContext:
    """Context for execution"""
    session_id: str
    user_id: str
    query: str
    mode_config: ModeConfig
    provider_type: str
    model_name: str
    language: str = "en"
    working_memory: Optional[Dict] = None
    history_messages: Optional[list] = None

@dataclass
class ExecutionResult:
    """Result from execution"""
    content: str
    tool_calls_made: int
    turns_used: int
    total_time_ms: int
    tokens_used: Dict[str, int]
    mode_used: str

class BaseExecutor(ABC):
    """
    Abstract base class for mode executors.

    Subclasses implement specific execution paths:
    - FastExecutor: Optimized for speed (filtered tools, 2 turns)
    - ExpertExecutor: Full capability (all tools, 6 turns)
    """

    def __init__(self, mode_config: ModeConfig):
        self.mode_config = mode_config

    @abstractmethod
    async def execute(
        self,
        context: ExecutionContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the query and yield streaming events.

        Args:
            context: Execution context with all necessary info

        Yields:
            Streaming events (mode_selected, tool_calls, content, done, etc.)
        """
        pass

    @abstractmethod
    def get_tools(self, query_categories: list = None) -> list:
        """Get available tools for this executor"""
        pass

    @abstractmethod
    def get_system_prompt(self, language: str) -> str:
        """Get system prompt for this executor"""
        pass
```

#### Task 3.2: Fast Executor

**File:** `src/agents/executors/fast_executor.py`

```python
"""
Fast mode executor - Optimized for speed
Target: 3-6 second response time
"""

import time
from typing import AsyncGenerator, Dict, Any

from src.agents.executors.base_executor import BaseExecutor, ExecutionContext
from src.agents.tools.tool_filter import ToolFilter
from src.config.mode_config import FAST_MODE_CONFIG
from src.prompts.templates.modes.fast_mode_prompts import (
    FAST_SYSTEM_PROMPT_VI,
    FAST_SYSTEM_PROMPT_EN
)

class FastExecutor(BaseExecutor):
    """
    Executor for FAST mode.

    Characteristics:
    - Uses smaller model (gpt-4o-mini / gemini-2.0-flash)
    - Maximum 2 agent turns
    - Filtered tool set (8 max)
    - Condensed system prompt
    - No web search
    - No thinking display
    """

    def __init__(self):
        super().__init__(FAST_MODE_CONFIG)
        self.tool_filter = ToolFilter()

    async def execute(
        self,
        context: ExecutionContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute in FAST mode with speed optimizations"""
        start_time = time.time()

        # 1. Emit mode selection event
        yield {
            "type": "mode_selected",
            "mode": "fast",
            "model": context.model_name,
            "max_turns": self.mode_config.max_turns
        }

        # 2. Get filtered tools
        tools = self.get_tools(context.working_memory.get("query_categories"))

        # 3. Build messages with condensed prompt
        system_prompt = self.get_system_prompt(context.language)
        messages = [
            {"role": "system", "content": system_prompt},
            *context.history_messages[-4:],  # Limited history
            {"role": "user", "content": context.query}
        ]

        # 4. Execute agent loop (max 2 turns)
        turns_used = 0
        for turn in range(self.mode_config.max_turns):
            turns_used += 1
            yield {"type": "turn_start", "turn": turn + 1}

            # LLM call with timeout
            response = await self._call_llm(
                messages,
                tools,
                timeout_ms=self.mode_config.turn_timeout_ms
            )

            if not response.tool_calls:
                # Final response
                yield {"type": "content", "content": response.content}
                break

            # Execute tools
            yield {"type": "tool_calls", "tools": response.tool_calls}
            tool_results = await self._execute_tools(response.tool_calls)
            yield {"type": "tool_results", "results": tool_results}

            messages.append({"role": "assistant", "tool_calls": response.tool_calls})
            messages.append({"role": "tool", "results": tool_results})

        # 5. Done
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield {
            "type": "done",
            "mode": "fast",
            "turns_used": turns_used,
            "total_time_ms": elapsed_ms
        }

    def get_tools(self, query_categories: list = None) -> list:
        """Get filtered tools for FAST mode"""
        from src.agents.tools.tool_loader import get_all_tools
        all_tools = get_all_tools()
        return self.tool_filter.filter_for_fast_mode(all_tools, query_categories)

    def get_system_prompt(self, language: str) -> str:
        """Get condensed system prompt"""
        if language.startswith("vi"):
            return FAST_SYSTEM_PROMPT_VI
        return FAST_SYSTEM_PROMPT_EN
```

#### Task 3.3: Expert Executor

**File:** `src/agents/executors/expert_executor.py`

```python
"""
Expert mode executor - Full capability
Target: Comprehensive analysis (15-45 seconds acceptable)
"""

import time
from typing import AsyncGenerator, Dict, Any

from src.agents.executors.base_executor import BaseExecutor, ExecutionContext
from src.config.mode_config import EXPERT_MODE_CONFIG
from src.prompts.templates.modes.expert_mode_prompts import (
    EXPERT_SYSTEM_PROMPT_VI,
    EXPERT_SYSTEM_PROMPT_EN
)

class ExpertExecutor(BaseExecutor):
    """
    Executor for EXPERT mode.

    Characteristics:
    - Uses larger model (gpt-4o / gemini-2.5-pro)
    - Maximum 6 agent turns
    - ALL tools available (31+)
    - Full system prompt with examples
    - Web search enabled
    - Thinking display enabled
    - NO classifier (model self-decides)
    """

    def __init__(self):
        super().__init__(EXPERT_MODE_CONFIG)

    async def execute(
        self,
        context: ExecutionContext
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute in EXPERT mode with full capability"""
        start_time = time.time()

        # 1. Emit mode selection
        yield {
            "type": "mode_selected",
            "mode": "expert",
            "model": context.model_name,
            "max_turns": self.mode_config.max_turns
        }

        # 2. Get ALL tools
        tools = self.get_tools()

        # 3. Build messages with full prompt
        system_prompt = self.get_system_prompt(context.language)
        messages = [
            {"role": "system", "content": system_prompt},
            *context.history_messages[-10:],  # More history for context
            {"role": "user", "content": context.query}
        ]

        # 4. Execute agent loop (up to 6 turns)
        turns_used = 0
        for turn in range(self.mode_config.max_turns):
            turns_used += 1
            yield {"type": "turn_start", "turn": turn + 1}

            # LLM call with extended timeout
            response = await self._call_llm(
                messages,
                tools,
                timeout_ms=self.mode_config.turn_timeout_ms,
                enable_thinking=True
            )

            # Emit thinking if present
            if response.thinking:
                yield {"type": "thinking", "content": response.thinking}

            if not response.tool_calls:
                yield {"type": "content", "content": response.content}
                break

            # Execute tools in parallel
            yield {"type": "tool_calls", "tools": response.tool_calls}
            tool_results = await self._execute_tools_parallel(response.tool_calls)
            yield {"type": "tool_results", "results": tool_results}

            messages.append({"role": "assistant", "tool_calls": response.tool_calls})
            messages.append({"role": "tool", "results": tool_results})

        # 5. Done
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield {
            "type": "done",
            "mode": "expert",
            "turns_used": turns_used,
            "total_time_ms": elapsed_ms
        }

    def get_tools(self, query_categories: list = None) -> list:
        """Get ALL tools for EXPERT mode"""
        from src.agents.tools.tool_loader import get_all_tools
        return get_all_tools()  # No filtering

    def get_system_prompt(self, language: str) -> str:
        """Get full system prompt with examples"""
        if language.startswith("vi"):
            return EXPERT_SYSTEM_PROMPT_VI
        return EXPERT_SYSTEM_PROMPT_EN
```

---

## 7. Provider-Specific Considerations

### 7.1 OpenAI (GPT) Considerations

| Aspect | Consideration |
|--------|---------------|
| **Token Limits** | gpt-4o-mini: 128K context, 16K output |
| **Tool Calls** | Native function calling, parallel execution |
| **Best Practice** | Use `response_format: json_object` for structured outputs |
| **Cost** | Mini models are 30x cheaper than full models |
| **Streaming** | Full SSE support |

### 7.2 Gemini Considerations

| Aspect | Consideration |
|--------|---------------|
| **thought_signature** | MUST preserve across multi-turn conversations |
| **Context Window** | Huge (1M+ tokens) - can include more history |
| **Tool Format** | Different from OpenAI - provider handles conversion |
| **Grounding** | May need for factual accuracy |
| **Cost** | Flash models are very cheap |

### 7.3 OpenRouter Considerations

| Aspect | Consideration |
|--------|---------------|
| **Multi-Model** | Gateway to 100+ models (Claude, Llama, etc.) |
| **Fallback** | Can switch models if one fails |
| **Site Metadata** | Pass site_url and site_name for attribution |
| **Rate Limits** | Varies by underlying provider |

### 7.4 Prompt Adjustment Matrix

| Provider | FAST Prompt | EXPERT Prompt | Special |
|----------|-------------|---------------|---------|
| OpenAI | Markdown headers, numbered lists | Full with examples | JSON mode |
| Gemini | Natural language | Extended context OK | thought_signature |
| OpenRouter | Depends on model | Depends on model | Model-specific |
| Ollama | Simpler, shorter | Moderate length | Local latency |

---

## 8. Implementation Priority

### 8.1 Priority Matrix

| Priority | Task | Files | Effort | Impact |
|----------|------|-------|--------|--------|
| **P0** | Mode-specific prompts | `src/prompts/templates/modes/` | 2 days | High |
| **P0** | Tool filter for FAST | `src/agents/tools/tool_filter.py` | 1 day | High |
| **P1** | Provider configs | `src/config/providers/` | 2 days | Medium |
| **P1** | Provider prompts | `src/prompts/templates/providers/` | 2 days | Medium |
| **P1** | Mode executors | `src/agents/executors/` | 3 days | High |
| **P2** | Feature flags | `src/config/feature_flags.py` | 0.5 day | Low |
| **P2** | Model catalog | `src/config/models/` | 1 day | Medium |
| **P2** | Health checks | `src/providers/health_check.py` | 1 day | Medium |

### 8.2 Implementation Sequence

```
Week 1: Foundation (P0)
├── Day 1-2: Mode-specific prompts (fast + expert)
├── Day 3: Tool filter implementation
├── Day 4: Integration with chat_handler
└── Day 5: Testing FAST mode latency

Week 2: Provider Support (P1)
├── Day 1-2: Provider configurations
├── Day 3-4: Provider-specific prompts
└── Day 5: Integration testing with GPT + Gemini

Week 3: Executors (P1)
├── Day 1: Base executor interface
├── Day 2-3: Fast executor implementation
├── Day 4-5: Expert executor implementation

Week 4: Polish (P2)
├── Day 1: Feature flags extraction
├── Day 2: Model capability catalog
├── Day 3: Provider health checks
├── Day 4-5: End-to-end testing
```

### 8.3 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| FAST mode P50 | ≤ 4 seconds | Latency monitoring |
| FAST mode P90 | ≤ 6 seconds | Latency monitoring |
| Provider switching | < 500ms | Fallback latency |
| Code duplication | < 10% | Code review |
| Test coverage | > 80% | Test suite |

---

## Appendix A: Existing Files Reference

### Files to KEEP (No changes needed)

```
✅ src/config/mode_config.py
✅ src/providers/base_provider.py
✅ src/providers/openai_provider.py
✅ src/providers/gemini_provider.py
✅ src/providers/ollama_provider.py
✅ src/providers/openrouter_provider.py
✅ src/providers/provider_factory.py
✅ src/agents/routing/mode_router.py
✅ src/agents/tools/tool_catalog.py
✅ src/agents/unified/unified_agent.py
```

### Files to MODIFY

```
📝 src/handlers/v2/chat_handler.py - Add mode routing
📝 src/prompts/prompt_manager.py - Add mode/provider selection
📝 src/routers/v2/chat.py - Already has response_mode param
```

### Files to CREATE

```
🆕 src/config/feature_flags.py
🆕 src/config/providers/__init__.py
🆕 src/config/providers/openai_config.py
🆕 src/config/providers/gemini_config.py
🆕 src/config/providers/ollama_config.py
🆕 src/config/providers/openrouter_config.py
🆕 src/config/models/model_catalog.py
🆕 src/providers/health_check.py
🆕 src/prompts/templates/__init__.py
🆕 src/prompts/templates/modes/__init__.py
🆕 src/prompts/templates/modes/fast_mode_prompts.py
🆕 src/prompts/templates/modes/expert_mode_prompts.py
🆕 src/prompts/templates/providers/__init__.py
🆕 src/prompts/templates/providers/openai_prompts.py
🆕 src/prompts/templates/providers/gemini_prompts.py
🆕 src/prompts/templates/providers/openrouter_prompts.py
🆕 src/agents/executors/__init__.py
🆕 src/agents/executors/base_executor.py
🆕 src/agents/executors/fast_executor.py
🆕 src/agents/executors/expert_executor.py
🆕 src/agents/tools/tool_filter.py
```

---

## Appendix B: Related Documentation

- [CHAT_RESPONSE_MODES_IMPLEMENTATION_PLAN.md](./CHAT_RESPONSE_MODES_IMPLEMENTATION_PLAN.md) - Original implementation plan
- [CHAT_AGENT_REDESIGN_PROPOSAL.md](./CHAT_AGENT_REDESIGN_PROPOSAL.md) - Architecture redesign proposal
- [ARCHITECTURE_CHAT_V2.md](./ARCHITECTURE_CHAT_V2.md) - Current V2 implementation details

---

**Document Version:** 1.0
**Last Updated:** 2026-01-23
**Next Review:** After Phase 1 (Week 1) completion

---

**Summary:**
The current architecture is **well-designed and ready for scaling**. The foundation (mode configurations, provider factory, routing) is solid. Minor enhancements needed for provider-specific prompts, mode executors, and tool filtering. Total estimated effort: **4 weeks** for full implementation.
