"""
LLM Tool Router - ChatGPT-Style 2-Phase Tool Selection

The Router receives a query and ALL tool summaries, then outputs:
- selected_tools: Which tools are needed
- complexity: simple/medium/complex
- execution_strategy: How to execute tools

This eliminates "category blindness" by letting the LLM see all tools
before making selection decisions.

Usage:
    router = LLMToolRouter()
    decision = await router.route(
        query="What is NVDA's RSI?",
        symbols=["NVDA"],
        context=classifier_context,
    )

    # decision.selected_tools = ["getStockPrice", "getTechnicalIndicators"]
    # decision.complexity = Complexity.SIMPLE
    # decision.execution_strategy = ExecutionStrategy.DIRECT
"""

import json
import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.agents.tools.tool_catalog import ToolCatalog, get_tool_catalog


class Complexity(str, Enum):
    """Query complexity levels."""
    SIMPLE = "simple"      # 1-2 tools, direct answer, max 2 turns
    MEDIUM = "medium"      # 3-5 tools, may need iteration, max 4 turns
    COMPLEX = "complex"    # 6+ tools, needs planning, max 6 turns


class ExecutionStrategy(str, Enum):
    """Execution strategies for different complexity levels."""
    DIRECT = "direct"          # Execute tools once, synthesize
    ITERATIVE = "iterative"    # Agent loop with tool calls
    PARALLEL = "parallel"      # Planning + parallel execution


@dataclass
class RouterDecision:
    """
    Router output containing tool selection and execution strategy.

    Attributes:
        selected_tools: List of tool names to use
        complexity: Query complexity (simple/medium/complex)
        execution_strategy: How to execute (direct/iterative/parallel)
        reasoning: Why these tools were selected
        confidence: Confidence score 0-1
        suggested_max_turns: Suggested max turns for agent loop
        metadata: Additional metadata
    """
    selected_tools: List[str]
    complexity: Complexity
    execution_strategy: ExecutionStrategy
    reasoning: str
    confidence: float = 0.9
    suggested_max_turns: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterDecision":
        """Create from dict."""
        # Parse complexity
        complexity_str = data.get("complexity", "medium").lower()
        try:
            complexity = Complexity(complexity_str)
        except ValueError:
            complexity = Complexity.MEDIUM

        # Parse strategy
        strategy_str = data.get("execution_strategy", "iterative").lower()
        try:
            strategy = ExecutionStrategy(strategy_str)
        except ValueError:
            strategy = ExecutionStrategy.ITERATIVE

        # Determine max turns based on complexity
        max_turns_map = {
            Complexity.SIMPLE: 2,
            Complexity.MEDIUM: 4,
            Complexity.COMPLEX: 6,
        }

        return cls(
            selected_tools=data.get("selected_tools", []),
            complexity=complexity,
            execution_strategy=strategy,
            reasoning=data.get("reasoning", ""),
            confidence=data.get("confidence", 0.9),
            suggested_max_turns=max_turns_map.get(complexity, 4),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_tools": self.selected_tools,
            "complexity": self.complexity.value,
            "execution_strategy": self.execution_strategy.value,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "suggested_max_turns": self.suggested_max_turns,
            "metadata": self.metadata,
        }

    @classmethod
    def fallback(cls, reason: str = "Router failed") -> "RouterDecision":
        """Create fallback decision when router fails."""
        return cls(
            selected_tools=[],
            complexity=Complexity.MEDIUM,
            execution_strategy=ExecutionStrategy.ITERATIVE,
            reasoning=f"Fallback: {reason}",
            confidence=0.5,
            suggested_max_turns=4,
            metadata={"fallback": True, "reason": reason},
        )


class LLMToolRouter(LoggerMixin):
    """
    LLM-based tool router that selects tools from full catalog.

    Key Features:
    - Sees ALL tools (via summaries) - no category blindness
    - Single LLM call for routing decision
    - Determines complexity and execution strategy
    - Fallback to heuristics if LLM fails

    Architecture:
    ```
    Query + Symbols → Router LLM (sees all tool summaries)
                            ↓
    RouterDecision(selected_tools, complexity, strategy)
    ```
    """

    # Lightweight model for routing (fast, cheap)
    DEFAULT_MODEL = "gpt-4.1-nano"
    DEFAULT_PROVIDER = ProviderType.OPENAI

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
        catalog: Optional[ToolCatalog] = None,
        max_retries: int = 2,
    ):
        """
        Initialize router.

        Args:
            model_name: LLM model for routing (lightweight model recommended)
            provider_type: Provider type (openai, etc.)
            catalog: ToolCatalog instance (uses singleton if not provided)
            max_retries: Max retries on failure
        """
        super().__init__()

        self.model_name = model_name or settings.ROUTER_MODEL or self.DEFAULT_MODEL
        self.provider_type = provider_type or settings.ROUTER_PROVIDER or self.DEFAULT_PROVIDER
        self.max_retries = max_retries

        # Get tool catalog
        self.catalog = catalog or get_tool_catalog()

        # LLM provider
        self.llm_provider = LLMGeneratorProvider()
        self.api_key = ModelProviderFactory._get_api_key(self.provider_type)

        self.logger.info(
            f"[ROUTER] Initialized: model={self.model_name}, "
            f"tools={len(self.catalog.get_tool_names())}"
        )

    async def route(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        context: Optional[Any] = None,
        use_heuristics_fallback: bool = True,
    ) -> RouterDecision:
        """
        Route query to appropriate tools.

        Args:
            query: User query
            symbols: Extracted symbols (from classifier)
            context: Optional ClassifierContext for additional info
            use_heuristics_fallback: Fall back to heuristics if LLM fails

        Returns:
            RouterDecision with selected tools and strategy
        """
        start_time = datetime.now()

        try:
            # Build prompt with all tool summaries
            prompt = self._build_prompt(query, symbols, context)

            # Call LLM
            result = await self._call_llm(prompt)

            # Parse result
            decision = RouterDecision.from_dict(result)

            # Validate tools exist
            decision.selected_tools = self._validate_tools(decision.selected_tools)

            # Log decision
            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self.logger.info("─" * 50)
            self.logger.info("🔀 ROUTER DECISION")
            self.logger.info("─" * 50)
            self.logger.info(f"  ├─ Tools: {decision.selected_tools}")
            self.logger.info(f"  ├─ Complexity: {decision.complexity.value}")
            self.logger.info(f"  ├─ Strategy: {decision.execution_strategy.value}")
            self.logger.info(f"  ├─ Max Turns: {decision.suggested_max_turns}")
            self.logger.info(f"  └─ ⏱️ Time: {elapsed_ms}ms")

            return decision

        except Exception as e:
            self.logger.error(f"[ROUTER] LLM routing failed: {e}", exc_info=True)

            if use_heuristics_fallback:
                self.logger.info("[ROUTER] Falling back to heuristics")
                return self._heuristic_route(query, symbols)

            return RouterDecision.fallback(str(e))

    def _build_prompt(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        context: Optional[Any] = None,
    ) -> str:
        """Build routing prompt with tool catalog."""

        # Get formatted tool catalog
        tool_catalog = self.catalog.format_for_router()

        # Context info
        symbols_hint = f"Symbols: {', '.join(symbols)}" if symbols else "No specific symbols"
        context_hint = ""
        if context and hasattr(context, 'working_memory_summary'):
            if context.working_memory_summary:
                context_hint = f"Context: {context.working_memory_summary[:200]}"

        return f"""You are a tool routing system. Given a user query and a catalog of available tools,
select the most appropriate tools to answer the query.

{tool_catalog}

<query>{query}</query>

<context>
{symbols_hint}
{context_hint}
</context>

<instructions>
1. Analyze the query and determine what information is needed
2. Select the minimum set of tools required to answer
3. Determine query complexity:
   - SIMPLE: 1-2 tools, straightforward answer (e.g., "What is AAPL price?")
   - MEDIUM: 3-5 tools, needs some analysis (e.g., "Analyze NVDA technicals")
   - COMPLEX: 6+ tools, comprehensive analysis (e.g., "Compare AAPL vs GOOGL")
4. Choose execution strategy:
   - direct: Execute tools once, synthesize answer
   - iterative: May need multiple rounds based on results
   - parallel: Execute all tools in parallel, then synthesize

IMPORTANT:
- Only select tools that are actually needed
- Don't select tools for information already in context
- For conversational queries (greetings), select NO tools
- For general knowledge, select NO tools
- For real-time info (current events, leaders), select webSearch
</instructions>

<output_format>
Provide your response inside <routing> tags as valid JSON:

<routing>
{{
  "selected_tools": ["tool1", "tool2"],
  "complexity": "simple|medium|complex",
  "execution_strategy": "direct|iterative|parallel",
  "reasoning": "Brief explanation of tool selection",
  "confidence": 0.0-1.0
}}
</routing>
</output_format>

Now analyze the query and provide your routing decision:
"""

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM and parse response."""

        for attempt in range(self.max_retries + 1):
            try:
                params = {
                    "model_name": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a tool routing system. "
                                "Output your decision inside <routing> tags as valid JSON."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "provider_type": self.provider_type,
                    "api_key": self.api_key,
                    "max_tokens": 500,
                    "temperature": 0.1,
                }

                response = await self.llm_provider.generate_response(**params)
                content = (
                    response.get("content", "")
                    if isinstance(response, dict)
                    else str(response)
                )
                content = content.strip()

                # Extract routing JSON
                json_str = self._extract_tag_content(content, "routing")
                if not json_str:
                    json_str = self._extract_json(content)

                if not json_str:
                    raise ValueError("No routing found in response")

                return json.loads(json_str)

            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"[ROUTER] JSON parse error (attempt {attempt + 1}): {e}"
                )
                if attempt == self.max_retries:
                    raise
            except Exception as e:
                self.logger.warning(
                    f"[ROUTER] Error (attempt {attempt + 1}): {e}"
                )
                if attempt == self.max_retries:
                    raise

        raise RuntimeError("Failed to route after retries")

    def _extract_tag_content(self, content: str, tag: str) -> Optional[str]:
        """Extract content from XML-style tags."""
        pattern = rf"<{tag}[^>]*>(.*?)</{tag}>"
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    def _extract_json(self, content: str) -> Optional[str]:
        """Extract JSON from content (fallback)."""
        start = content.find("{")
        if start == -1:
            return None

        depth = 0
        for i, char in enumerate(content[start:], start):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return content[start:i + 1]

        return None

    def _validate_tools(self, tool_names: List[str]) -> List[str]:
        """Validate and filter tool names."""
        valid_tools = self.catalog.get_tool_names()
        validated = []

        for name in tool_names:
            if name in valid_tools:
                validated.append(name)
            else:
                self.logger.warning(f"[ROUTER] Unknown tool: {name}")

        return validated

    def _heuristic_route(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
    ) -> RouterDecision:
        """
        Fallback heuristic routing when LLM fails.

        Uses simple rules based on query keywords and symbols.
        """
        query_lower = query.lower()
        tools = []
        complexity = Complexity.SIMPLE
        strategy = ExecutionStrategy.DIRECT

        # Check for conversational/greetings
        greetings = ["hello", "hi", "xin chào", "thank", "cảm ơn", "bye", "goodbye"]
        if any(g in query_lower for g in greetings):
            return RouterDecision(
                selected_tools=[],
                complexity=Complexity.SIMPLE,
                execution_strategy=ExecutionStrategy.DIRECT,
                reasoning="Heuristic: Conversational query",
                confidence=0.8,
            )

        # Check for real-time info
        realtime_keywords = [
            "current", "who is", "latest", "today",
            "hiện tại", "ai là", "mới nhất", "hôm nay"
        ]
        if any(kw in query_lower for kw in realtime_keywords):
            tools.append("webSearch")

        # If symbols present, add price tool
        if symbols:
            tools.append("getStockPrice")

            # Check for technical analysis
            tech_keywords = ["rsi", "macd", "technical", "kỹ thuật", "indicator"]
            if any(kw in query_lower for kw in tech_keywords):
                tools.append("getTechnicalIndicators")

            # Check for fundamental analysis
            fund_keywords = ["p/e", "earnings", "revenue", "fundamental", "cơ bản", "báo cáo"]
            if any(kw in query_lower for kw in fund_keywords):
                tools.extend(["getIncomeStatement", "getFinancialRatios"])

            # Check for news
            news_keywords = ["news", "tin", "tin tức", "announcement"]
            if any(kw in query_lower for kw in news_keywords):
                tools.append("getNews")

        # Determine complexity
        if len(tools) <= 2:
            complexity = Complexity.SIMPLE
            strategy = ExecutionStrategy.DIRECT
        elif len(tools) <= 5:
            complexity = Complexity.MEDIUM
            strategy = ExecutionStrategy.ITERATIVE
        else:
            complexity = Complexity.COMPLEX
            strategy = ExecutionStrategy.PARALLEL

        # Deduplicate
        tools = list(dict.fromkeys(tools))

        return RouterDecision(
            selected_tools=tools,
            complexity=complexity,
            execution_strategy=strategy,
            reasoning="Heuristic fallback based on keywords and symbols",
            confidence=0.6,
            metadata={"fallback": True, "method": "heuristic"},
        )


# Singleton instance
_router_instance: Optional[LLMToolRouter] = None


def get_tool_router(
    model_name: Optional[str] = None,
    provider_type: Optional[str] = None,
) -> LLMToolRouter:
    """Get singleton LLMToolRouter instance."""
    global _router_instance

    if _router_instance is None:
        _router_instance = LLMToolRouter(
            model_name=model_name,
            provider_type=provider_type,
        )

    return _router_instance


def reset_router():
    """Reset singleton (for testing)."""
    global _router_instance
    _router_instance = None


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_router():
        print("=" * 60)
        print("Testing LLM Tool Router")
        print("=" * 60)

        router = get_tool_router()

        # Test queries
        test_cases = [
            ("What is NVDA's current price?", ["NVDA"]),
            ("Analyze AAPL technicals and fundamentals", ["AAPL"]),
            ("Compare GOOGL vs MSFT performance", ["GOOGL", "MSFT"]),
            ("Hello, how are you?", []),
            ("Who is the CEO of Apple?", ["AAPL"]),
        ]

        for query, symbols in test_cases:
            print(f"\n--- Query: {query} ---")
            print(f"Symbols: {symbols}")

            decision = await router.route(query=query, symbols=symbols)

            print(f"Selected Tools: {decision.selected_tools}")
            print(f"Complexity: {decision.complexity.value}")
            print(f"Strategy: {decision.execution_strategy.value}")
            print(f"Reasoning: {decision.reasoning}")
            print(f"Confidence: {decision.confidence}")

    asyncio.run(test_router())
