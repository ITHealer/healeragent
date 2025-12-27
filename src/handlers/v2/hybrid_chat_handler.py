"""
Hybrid Chat Handler - Intelligent Mode Selection

Wraps ChatHandler with QueryRouter for optimal execution path selection:
- SIMPLE: Direct response via SimpleModeHandler (1 LLM call)
- PARALLEL: Standard ChatHandler pipeline (2-3 LLM calls)
- AGENTIC: Enhanced execution with re-planning (3-15+ LLM calls)

Performance Characteristics:
┌──────────┬─────────────┬─────────────┬──────────────┐
│ Mode     │ LLM Calls   │ Tools       │ Response Time│
├──────────┼─────────────┼─────────────┼──────────────┤
│ SIMPLE   │ 0-1         │ 0           │ < 2s         │
│ PARALLEL │ 2-3         │ 2-15        │ 3-10s        │
│ AGENTIC  │ 3-15+       │ Variable    │ 10-60s       │
└──────────┴─────────────┴─────────────┴──────────────┘

Usage:
    handler = HybridChatHandler()

    async for chunk in handler.handle(
        query="giá AAPL",
        session_id="...",
        user_id="...",
        model_name="gpt-4.1-nano",
        provider_type="openai"
    ):
        yield chunk
"""

import time
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Any

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings

# Routing components
from src.agents.routing import (
    QueryRouter,
    ExecutionMode,
    RoutingResult,
    get_query_router,
    SimpleModeHandler,
    get_simple_mode_handler
)

# Standard handler
from src.handlers.v2.chat_handler import ChatHandler

# Memory for symbol context
from src.agents.memory.working_memory import WorkingMemory

# Provider
from src.providers.provider_factory import ProviderType


class HybridChatHandler(LoggerMixin):
    """
    Hybrid Chat Handler with Intelligent Mode Selection

    Automatically routes queries to optimal execution path:
    - Greetings/thanks → SIMPLE (instant response)
    - Known analysis → PARALLEL (efficient parallel execution)
    - Discovery/research → AGENTIC (adaptive with re-planning)

    Example:
        handler = HybridChatHandler()

        async for chunk in handler.handle(
            query="Xin chào",
            session_id="sess_123",
            user_id="user_456"
        ):
            print(chunk)  # Instant greeting response
    """

    def __init__(
        self,
        router_model: str = "gpt-4.1-nano",
        simple_model: str = "gpt-4.1-nano",
        enable_llm_routing: bool = True
    ):
        """
        Initialize Hybrid Chat Handler

        Args:
            router_model: Model for query routing (small/fast preferred)
            simple_model: Model for simple mode responses
            enable_llm_routing: Use LLM for ambiguous query classification
        """
        super().__init__()

        # Initialize components
        self.router = get_query_router(
            model_name=router_model,
            enable_llm_routing=enable_llm_routing
        )

        self.simple_handler = get_simple_mode_handler(
            model_name=simple_model
        )

        self.standard_handler = ChatHandler()

        # Working memory for symbol context
        self._working_memory_cache: Dict[str, WorkingMemory] = {}

        # Statistics
        self._stats = {
            "total_requests": 0,
            "simple_requests": 0,
            "parallel_requests": 0,
            "agentic_requests": 0,
            "avg_routing_time_ms": 0,
            "avg_response_time_ms": 0
        }

        self.logger.info(
            f"[HYBRID:INIT] HybridChatHandler initialized with "
            f"router_model={router_model}, simple_model={simple_model}"
        )

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================

    async def handle(
        self,
        query: str,
        session_id: str,
        user_id: str,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
        chart_displayed: Optional[bool] = None,
        organization_id: Optional[str] = None,
        enable_thinking: bool = True,
        stream: bool = True,
        enable_think_tool: bool = False,
        enable_compaction: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Handle chat request with intelligent mode selection

        Args:
            query: User query
            session_id: Session ID
            user_id: User ID
            model_name: LLM model (default from settings)
            provider_type: LLM provider (default from settings)
            chart_displayed: Whether chart is displayed
            organization_id: Organization ID
            enable_thinking: Enable extended thinking
            stream: Enable streaming
            enable_think_tool: Enable Think Tool
            enable_compaction: Enable context compaction
            **kwargs: Additional parameters

        Yields:
            Response chunks (SSE format for streaming)
        """
        start_time = time.time()
        flow_id = f"HYB-{datetime.now().strftime('%H%M%S')}-{session_id[:8]}"

        model_name = model_name or settings.MODEL_DEFAULT
        provider_type = provider_type or settings.PROVIDER_DEFAULT

        self._stats["total_requests"] += 1

        self.logger.info(f"[HYBRID:{flow_id}] ════════════════════════════════════════")
        self.logger.info(f"[HYBRID:{flow_id}] Query: '{query[:80]}...'")

        try:
            # ================================================================
            # PHASE 1: Route Query
            # ================================================================
            recent_symbols = self._get_recent_symbols(session_id)

            routing_result = await self.router.route(
                query=query,
                context={"user_id": user_id, "session_id": session_id},
                recent_symbols=recent_symbols
            )

            self.logger.info(
                f"[HYBRID:{flow_id}] Routed to {routing_result.mode.value.upper()} "
                f"(confidence={routing_result.confidence:.2f}, method={routing_result.route_method})"
            )

            # ================================================================
            # PHASE 2: Execute Based on Mode
            # ================================================================
            if routing_result.mode == ExecutionMode.SIMPLE:
                # Simple mode - fast path
                self._stats["simple_requests"] += 1

                async for chunk in self._handle_simple(
                    query=query,
                    routing_result=routing_result,
                    flow_id=flow_id,
                    stream=stream
                ):
                    yield chunk

            elif routing_result.mode == ExecutionMode.PARALLEL:
                # Parallel mode - standard handler
                self._stats["parallel_requests"] += 1

                async for chunk in self._handle_parallel(
                    query=query,
                    routing_result=routing_result,
                    session_id=session_id,
                    user_id=user_id,
                    model_name=model_name,
                    provider_type=provider_type,
                    chart_displayed=chart_displayed,
                    organization_id=organization_id,
                    enable_thinking=enable_thinking,
                    stream=stream,
                    enable_think_tool=enable_think_tool,
                    enable_compaction=enable_compaction,
                    flow_id=flow_id
                ):
                    yield chunk

            else:
                # Agentic mode - enhanced execution
                self._stats["agentic_requests"] += 1

                async for chunk in self._handle_agentic(
                    query=query,
                    routing_result=routing_result,
                    session_id=session_id,
                    user_id=user_id,
                    model_name=model_name,
                    provider_type=provider_type,
                    chart_displayed=chart_displayed,
                    organization_id=organization_id,
                    enable_thinking=enable_thinking,
                    stream=stream,
                    enable_think_tool=True,  # Always enable for agentic
                    enable_compaction=enable_compaction,
                    flow_id=flow_id
                ):
                    yield chunk

            # ================================================================
            # PHASE 3: Update Statistics
            # ================================================================
            elapsed_ms = int((time.time() - start_time) * 1000)
            self._update_avg_response_time(elapsed_ms)

            self.logger.info(
                f"[HYBRID:{flow_id}] Completed in {elapsed_ms}ms "
                f"(mode={routing_result.mode.value})"
            )

        except Exception as e:
            self.logger.error(f"[HYBRID:{flow_id}] Error: {e}", exc_info=True)
            yield self._format_error_response(str(e), routing_result.detected_language)

    # ========================================================================
    # MODE HANDLERS
    # ========================================================================

    async def _handle_simple(
        self,
        query: str,
        routing_result: RoutingResult,
        flow_id: str,
        stream: bool
    ) -> AsyncGenerator[str, None]:
        """
        Handle SIMPLE mode - fast path without planning

        For: greetings, thanks, goodbye, definitions
        """
        self.logger.info(f"[HYBRID:{flow_id}] → SIMPLE mode (fast path)")

        if stream:
            async for event in self.simple_handler.handle_stream(
                query=query,
                hint=routing_result.simple_response_hint,
                language=routing_result.detected_language
            ):
                if event["type"] == "text_delta":
                    yield event["content"]
                elif event["type"] == "done":
                    self.logger.info(
                        f"[HYBRID:{flow_id}] Simple response completed "
                        f"(tokens={event.get('tokens_used', 0)})"
                    )
        else:
            response = await self.simple_handler.handle(
                query=query,
                hint=routing_result.simple_response_hint,
                language=routing_result.detected_language
            )
            yield response.content

    async def _handle_parallel(
        self,
        query: str,
        routing_result: RoutingResult,
        session_id: str,
        user_id: str,
        model_name: str,
        provider_type: str,
        flow_id: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Handle PARALLEL mode - standard planning + parallel execution

        For: known symbols, multi-tool analysis
        """
        self.logger.info(
            f"[HYBRID:{flow_id}] → PARALLEL mode "
            f"(symbols={routing_result.symbols})"
        )

        # Delegate to standard handler
        async for chunk in self.standard_handler.handle_chat_with_reasoning(
            query=query,
            session_id=session_id,
            user_id=user_id,
            model_name=model_name,
            provider_type=provider_type,
            **kwargs
        ):
            yield chunk

        # Update symbol cache
        if routing_result.symbols:
            self._update_recent_symbols(session_id, routing_result.symbols)

    async def _handle_agentic(
        self,
        query: str,
        routing_result: RoutingResult,
        session_id: str,
        user_id: str,
        model_name: str,
        provider_type: str,
        flow_id: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Handle AGENTIC mode - adaptive execution with potential re-planning

        For: discovery, conditional logic, research tasks
        """
        self.logger.info(
            f"[HYBRID:{flow_id}] → AGENTIC mode "
            f"(categories={routing_result.categories})"
        )

        # For now, delegate to standard handler with think_tool enabled
        # TODO: Implement full Manus-style loop with re-planning
        async for chunk in self.standard_handler.handle_chat_with_reasoning(
            query=query,
            session_id=session_id,
            user_id=user_id,
            model_name=model_name,
            provider_type=provider_type,
            enable_think_tool=True,  # Always enable for agentic
            **kwargs
        ):
            yield chunk

        # Update symbol cache from any discovered symbols
        if routing_result.symbols:
            self._update_recent_symbols(session_id, routing_result.symbols)

    # ========================================================================
    # SYMBOL CONTEXT MANAGEMENT
    # ========================================================================

    def _get_recent_symbols(self, session_id: str) -> List[str]:
        """Get recent symbols from session context"""
        wm = self._working_memory_cache.get(session_id)
        if wm:
            symbols_entry = wm.get_entries_by_type("SYMBOLS")
            if symbols_entry:
                return symbols_entry[0].content.get("symbols", [])
        return []

    def _update_recent_symbols(self, session_id: str, symbols: List[str]):
        """Update recent symbols in session context"""
        if session_id not in self._working_memory_cache:
            self._working_memory_cache[session_id] = WorkingMemory(
                session_id=session_id,
                max_tokens=2000  # Small cache
            )

        wm = self._working_memory_cache[session_id]
        wm.add_symbols(symbols)

        # Cleanup old sessions (keep last 100)
        if len(self._working_memory_cache) > 100:
            oldest = list(self._working_memory_cache.keys())[0]
            del self._working_memory_cache[oldest]

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _format_error_response(self, error: str, language: str) -> str:
        """Format error response based on language"""
        messages = {
            "vi": f"Xin lỗi, đã xảy ra lỗi: {error}. Vui lòng thử lại.",
            "en": f"Sorry, an error occurred: {error}. Please try again.",
            "zh": f"抱歉，发生错误：{error}。请重试。"
        }
        return messages.get(language, messages["en"])

    def _update_avg_response_time(self, elapsed_ms: int):
        """Update average response time statistic"""
        total = self._stats["total_requests"]
        current_avg = self._stats["avg_response_time_ms"]

        # Running average
        self._stats["avg_response_time_ms"] = (
            (current_avg * (total - 1) + elapsed_ms) / total
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return {
            **self._stats,
            "router_stats": self.router.get_stats()
        }

    def reset_stats(self):
        """Reset all statistics"""
        self._stats = {
            "total_requests": 0,
            "simple_requests": 0,
            "parallel_requests": 0,
            "agentic_requests": 0,
            "avg_routing_time_ms": 0,
            "avg_response_time_ms": 0
        }
        self.router.reset_stats()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_hybrid_handler: Optional[HybridChatHandler] = None


def get_hybrid_handler(
    router_model: str = "gpt-4.1-nano",
    simple_model: str = "gpt-4.1-nano"
) -> HybridChatHandler:
    """
    Get singleton HybridChatHandler instance

    Args:
        router_model: Model for routing
        simple_model: Model for simple responses

    Returns:
        HybridChatHandler singleton
    """
    global _hybrid_handler

    if _hybrid_handler is None:
        _hybrid_handler = HybridChatHandler(
            router_model=router_model,
            simple_model=simple_model
        )

    return _hybrid_handler
