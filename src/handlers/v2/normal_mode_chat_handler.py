"""
Normal Mode Chat Handler

Optimized chat handler for Normal Mode queries (90% of traffic).
Routes queries through a simplified 2-3 LLM call pipeline:

Flow:
1. Unified Classification (1 LLM call)
   - Determines query type, tool categories, and if tools are needed
   - Merges what was previously 2 LLM calls into 1

2. Agent Loop (1-3 LLM calls)
   - LLM decides tools inline (no upfront planning)
   - Executes tools in parallel
   - Continues until response ready

Total: 2-3 LLM calls (vs 4+ in the old pipeline)

For complex queries (Deep Research Mode), falls back to the existing
7-phase ChatHandler pipeline.

Usage:
    handler = NormalModeChatHandler()

    async for chunk in handler.handle_chat(
        query="What is AAPL's price?",
        session_id="...",
        user_id="...",
    ):
        print(chunk, end="")
"""

import time
import uuid
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Any

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings

# Classification
from src.agents.classification import (
    UnifiedClassifier,
    ClassifierContext,
    UnifiedClassificationResult,
    QueryType,
    get_unified_classifier,
)

# Asset Resolution (Symbol disambiguation)
from src.services.asset import (
    get_symbol_cache,
    get_asset_resolver,
    AssetClass,
)

# Normal Mode Agent
from src.agents.normal_mode import (
    NormalModeAgent,
    AgentResult,
    get_normal_mode_agent,
)

# Memory components
from src.agents.memory.core_memory import get_core_memory
from src.agents.memory.recursive_summary import get_recursive_summary_manager
from src.agents.memory.memory_update_agent import get_memory_update_agent
from src.agents.memory.working_memory_integration import (
    WorkingMemoryIntegration,
    setup_working_memory_for_request,
)

# Context Management (compaction)
from src.services.context_management_service import (
    ContextManagementService,
    PreparedContext,
    get_context_manager,
)

# Database
from src.database.repository.sessions import SessionRepository
from src.database.repository.chat import ChatRepository

# LLM Provider
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType

# Chart Resolution
from src.agents.charts import (
    resolve_charts_from_classification,
    charts_to_dict_list,
    ChartInfo,
)


class NormalModeChatHandler(LoggerMixin):
    """
    Optimized Chat Handler for Normal Mode

    Key differences from ChatHandler:
    1. Single classification call (UnifiedClassifier)
    2. No separate planning phase
    3. LLM decides tools inline
    4. Parallel tool execution
    5. 2-3 total LLM calls (vs 4+)

    For complex queries that require deep research,
    use the fallback to the existing ChatHandler.
    """

    # Configuration
    DEEP_RESEARCH_THRESHOLD = 0.7  # Confidence threshold for complex queries
    MAX_AGENT_TURNS = 10

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
        fallback_handler=None,  # Optional ChatHandler for Deep Research
        enable_compaction: bool = True,
    ):
        """
        Initialize NormalModeChatHandler.

        Args:
            model_name: Override default model
            provider_type: Override default provider
            fallback_handler: ChatHandler for Deep Research Mode (optional)
            enable_compaction: Enable automatic context compaction
        """
        super().__init__()

        self.model_name = model_name or settings.MODEL_DEFAULT
        self.provider_type = provider_type or settings.PROVIDER_DEFAULT
        self.enable_compaction = enable_compaction

        # Classification
        self.classifier = get_unified_classifier(
            model_name=self.model_name,
            provider_type=self.provider_type,
        )

        # Normal Mode Agent
        self.agent = get_normal_mode_agent(
            model_name=self.model_name,
            provider_type=self.provider_type,
        )

        # Memory components
        self.core_memory = get_core_memory()
        self.summary_manager = get_recursive_summary_manager()
        self.memory_update = get_memory_update_agent()
        self.session_repo = SessionRepository()

        # Chat repository for message persistence
        self.chat_repo = ChatRepository()

        # Context Management Service (handles compaction)
        try:
            self.context_manager = get_context_manager(enable_compaction=enable_compaction)
            self.logger.info(f"[NORMAL_MODE_HANDLER] Context compaction: {'ON' if enable_compaction else 'OFF'}")
        except Exception as e:
            self.logger.warning(f"[NORMAL_MODE_HANDLER] ContextManagementService not available: {e}")
            self.context_manager = None

        # LLM for direct responses
        self.llm_provider = LLMGeneratorProvider()

        # Fallback for Deep Research Mode
        self.fallback_handler = fallback_handler

        self.logger.info(
            f"[NORMAL_MODE_HANDLER] Initialized: model={self.model_name}, "
            f"provider={self.provider_type}, compaction={enable_compaction}"
        )

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================

    async def handle_chat(
        self,
        query: str,
        session_id: str,
        user_id: str,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
        chart_displayed: Optional[bool] = None,
        organization_id: Optional[str] = None,
        enable_thinking: bool = True,
        enable_llm_events: bool = True,
        stream: bool = True,
        enable_web_search: bool = False,
        classification: Optional[UnifiedClassificationResult] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle chat request with optimized Normal Mode flow.

        Flow:
        1. Load context
        2. Unified classification
        3. Route to Normal Mode Agent or fallback

        Args:
            query: User query
            session_id: Session identifier
            user_id: User identifier
            model_name: Override model
            provider_type: Override provider
            chart_displayed: Chart state
            organization_id: Organization ID
            enable_thinking: Enable thinking events (Manus AI style)
            enable_llm_events: Enable LLM decision events
            stream: Enable streaming

        Yields:
            Dict events with types: thinking, llm_thought, turn_start,
            tool_calls, tool_results, content, done, error
        """
        flow_id = f"NM-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
        flow_start = time.time()

        # Use provided or default settings
        model = model_name or self.model_name
        provider = provider_type or self.provider_type

        self.logger.info(f"[{flow_id}] ========================================")
        self.logger.info(f"[{flow_id}] NORMAL MODE HANDLER START")
        self.logger.info(f"[{flow_id}] Query: {query[:100]}...")
        self.logger.info(f"[{flow_id}] ========================================")

        # Setup working memory
        wm_integration = setup_working_memory_for_request(
            session_id=session_id,
            user_id=user_id,
            flow_id=flow_id,
        )

        # Track response for saving
        response_chunks = []
        question_id = None

        try:
            # ================================================================
            # STEP 0: SAVE USER QUESTION
            # ================================================================
            try:
                question_id = self.chat_repo.save_user_question(
                    session_id=session_id,
                    created_at=datetime.now(),
                    created_by=user_id,
                    content=query,
                )
                self.logger.debug(f"[{flow_id}] Saved user question: {question_id}")
            except Exception as e:
                self.logger.warning(f"[{flow_id}] Failed to save user question: {e}")

            # ================================================================
            # STEP 1: LOAD CONTEXT
            # ================================================================
            context_data = await self._load_context(
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id,
                wm_integration=wm_integration,
            )

            # ================================================================
            # STEP 2: UNIFIED CLASSIFICATION (skip if already provided)
            # ================================================================
            if classification is None:
                classification = await self._classify_query(
                    flow_id=flow_id,
                    query=query,
                    context_data=context_data,
                )
            else:
                self.logger.info(f"[{flow_id}] Using pre-computed classification (skipped LLM call)")

            # Force web search category when enabled by user
            if enable_web_search:
                if "web" not in classification.tool_categories:
                    classification.tool_categories.append("web")
                if not classification.requires_tools:
                    classification.requires_tools = True
                self.logger.info(f"[{flow_id}] Web search enabled - forced web category")

            # Log classification result
            self.logger.info(
                f"[{flow_id}] Classification: type={classification.query_type.value}, "
                f"requires_tools={classification.requires_tools}, "
                f"symbols={classification.symbols}, "
                f"categories={classification.tool_categories}"
            )

            # ================================================================
            # STEP 2.3: CHECK SYMBOL AMBIGUITY
            # ================================================================
            if classification.symbols:
                ambiguous_symbols = await self._check_symbol_ambiguity(
                    flow_id=flow_id,
                    symbols=classification.symbols,
                    language=classification.response_language,
                )

                if ambiguous_symbols:
                    # Found ambiguous symbol - ask user to clarify
                    disambiguation_msg = ambiguous_symbols[0].get("message", "")
                    self.logger.info(
                        f"[{flow_id}] Ambiguous symbol found - asking user to clarify"
                    )

                    yield {"type": "disambiguation", "data": ambiguous_symbols}
                    yield {"type": "content", "content": disambiguation_msg}
                    yield {"type": "done", "requires_clarification": True}

                    # Save to DB as assistant response
                    if question_id:
                        try:
                            self.chat_repo.save_assistant_response(
                                session_id=session_id,
                                created_at=datetime.now(),
                                question_id=question_id,
                                content=disambiguation_msg,
                                response_time=time.time() - flow_start,
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to save disambiguation response: {e}")

                    return  # Early return - wait for user clarification

            # ================================================================
            # STEP 2.5: RESOLVE CHARTS FROM CLASSIFICATION
            # ================================================================
            charts = resolve_charts_from_classification(
                classification=classification,
                query=query,
                max_charts=3,
            )
            charts_data = charts_to_dict_list(charts)

            if charts_data:
                self.logger.info(
                    f"[{flow_id}] Charts resolved: {[c['type'] for c in charts_data]}"
                )

            # ================================================================
            # STEP 3: ROUTE BASED ON CLASSIFICATION
            # ================================================================

            # Check if Deep Research Mode is needed
            needs_deep_research = self._needs_deep_research(classification, query)

            if needs_deep_research and self.fallback_handler:
                # Deep Research Mode - use full ChatHandler pipeline
                self.logger.info(f"[{flow_id}] Routing to Deep Research Mode (fallback handler)")

                async for chunk in self.fallback_handler.handle_chat_with_reasoning(
                    query=query,
                    chart_displayed=chart_displayed,
                    session_id=session_id,
                    user_id=user_id,
                    model_name=model,
                    provider_type=provider,
                    organization_id=organization_id,
                    enable_thinking=enable_thinking,
                    stream=stream,
                ):
                    # Deep Research returns strings - wrap in dict for consistency
                    if isinstance(chunk, str):
                        response_chunks.append(chunk)
                        yield {"type": "content", "content": chunk}
                    else:
                        if chunk.get("type") == "content":
                            response_chunks.append(chunk.get("content", ""))
                        # Intercept done event to add charts
                        if chunk.get("type") == "done":
                            if charts_data:
                                chunk["charts"] = charts_data
                        yield chunk
            else:
                # Normal Mode - optimized 2-3 LLM call path
                if needs_deep_research:
                    self.logger.warning(
                        f"[{flow_id}] Query suggests Deep Research but no fallback handler - "
                        f"using Normal Mode"
                    )
                else:
                    self.logger.info(f"[{flow_id}] Running Normal Mode")

                async for event in self._run_normal_mode(
                    flow_id=flow_id,
                    query=query,
                    classification=classification,
                    context_data=context_data,
                    wm_integration=wm_integration,
                    system_language=classification.response_language,
                    user_id=user_id,
                    session_id=session_id,
                    enable_thinking=enable_thinking,
                    enable_llm_events=enable_llm_events,
                    charts=charts_data,
                ):
                    # Collect content for saving to database
                    if event.get("type") == "content":
                        response_chunks.append(event.get("content", ""))
                    yield event

            # ================================================================
            # STEP 4: POST-PROCESSING
            # ================================================================
            flow_time = time.time() - flow_start
            full_response = "".join(response_chunks)

            # Update working memory with classification results
            wm_integration.save_classification(
                query_type=classification.query_type.value,
                categories=classification.tool_categories,
                symbols=classification.symbols,
                language=classification.response_language,
                reasoning=classification.intent_summary,
            )

            # Save assistant response to database
            if question_id and response_chunks:
                try:
                    self.chat_repo.save_assistant_response(
                        session_id=session_id,
                        created_at=datetime.now(),
                        question_id=question_id,
                        content=full_response,
                        response_time=flow_time,
                    )
                    self.logger.debug(
                        f"[{flow_id}] Saved assistant response: {len(full_response)} chars"
                    )
                except Exception as e:
                    self.logger.warning(f"[{flow_id}] Failed to save assistant response: {e}")

            # ================================================================
            # MEMORY UPDATES (Background tasks)
            # ================================================================
            # 1. Check if conversation needs summarization
            try:
                summary_result = await self.summary_manager.check_and_create_summary(
                    session_id=session_id,
                    user_id=int(user_id) if user_id else 0,
                )
                if summary_result and summary_result.get("created"):
                    self.logger.info(
                        f"[{flow_id}] Created conversation summary: "
                        f"version={summary_result.get('version')}"
                    )
            except Exception as e:
                self.logger.warning(f"[{flow_id}] Failed to create summary: {e}")

            # 2. Analyze conversation for memory updates (user preferences, portfolio, etc.)
            try:
                memory_result = await self.memory_update.analyze_for_updates(
                    user_id=str(user_id) if user_id else "0",
                    user_message=query,
                    assistant_message=full_response,  # Fixed: was assistant_response
                )
                if memory_result and memory_result.get("updated"):
                    self.logger.info(
                        f"[{flow_id}] Memory updated: action={memory_result.get('action')}, "
                        f"categories={memory_result.get('categories')}"
                    )
            except Exception as e:
                self.logger.warning(f"[{flow_id}] Failed to update memory: {e}")

            self.logger.info(
                f"[{flow_id}] Flow completed in {flow_time:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"[{flow_id}] Error: {e}", exc_info=True)

            wm_integration.save_error(
                error_type="flow_error",
                message=str(e),
                recoverable=True,
            )

            yield {"type": "error", "error": str(e)}

        finally:
            wm_integration.complete_request(clear_task_data=True)

    # ========================================================================
    # STREAMING SUPPORT
    # ========================================================================

    async def handle_chat_stream(
        self,
        query: str,
        session_id: str,
        user_id: str,
        enable_thinking: bool = True,
        enable_llm_events: bool = True,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle chat with structured streaming events including thinking/reasoning.

        Yields events with type and data for frontend consumption:
        - context_loaded: Memory context loaded
        - classifying: Classification in progress
        - classified: Classification result
        - thinking: AI reasoning process
        - llm_thought: LLM thought events
        - turn_start: New agent turn
        - tool_calls: Tools being called
        - tool_results: Tool execution results
        - content: Response content
        - done: Processing complete
        - error: Error occurred

        Args:
            query: User query
            session_id: Session identifier
            user_id: User identifier
            enable_thinking: Enable thinking events
            enable_llm_events: Enable LLM decision events
            **kwargs: Additional arguments
        """
        flow_id = f"NM-{uuid.uuid4().hex[:8]}"

        try:
            # Load context
            wm_integration = setup_working_memory_for_request(
                session_id=session_id,
                user_id=user_id,
                flow_id=flow_id,
            )

            context_data = await self._load_context(
                flow_id=flow_id,
                user_id=user_id,
                session_id=session_id,
                wm_integration=wm_integration,
            )

            yield {"type": "context_loaded", "data": {"symbols": context_data.get("current_symbols", [])}}

            # Classification
            yield {"type": "classifying", "data": {}}

            classification = await self._classify_query(
                flow_id=flow_id,
                query=query,
                context_data=context_data,
            )

            yield {
                "type": "classified",
                "data": {
                    "query_type": classification.query_type.value,
                    "requires_tools": classification.requires_tools,
                    "symbols": classification.symbols,
                    "categories": classification.tool_categories,
                    "intent_summary": classification.intent_summary,
                }
            }

            # Run agent with streaming and thinking
            async for event in self.agent.run_stream(
                query=query,
                classification=classification,
                conversation_history=context_data.get("chat_history", []),
                system_language=classification.response_language,
                user_id=int(user_id) if user_id else None,
                session_id=session_id,
                core_memory=context_data.get("core_memory"),
                conversation_summary=context_data.get("conversation_summary"),
                enable_thinking=enable_thinking,
                enable_llm_events=enable_llm_events,
            ):
                yield event

        except Exception as e:
            self.logger.error(f"[{flow_id}] Stream error: {e}", exc_info=True)
            yield {"type": "error", "data": {"error": str(e)}}

    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================

    async def _load_context(
        self,
        flow_id: str,
        user_id: str,
        session_id: str,
        wm_integration: WorkingMemoryIntegration,
        enable_compaction: bool = True,
    ) -> Dict[str, Any]:
        """
        Load all context needed for classification and agent.

        Uses ContextManagementService for automatic compaction when:
        - Context exceeds token threshold (100K tokens)
        - enable_compaction is True
        - ContextManagementService is available
        """
        start = time.time()

        self.logger.info(f"[{flow_id}] Loading context...")

        # Working memory (request-level context)
        working_context = wm_integration.get_context_for_planning(max_tokens=500)
        current_symbols = wm_integration.get_current_symbols()

        # Try using ContextManagementService for automatic compaction
        if self.context_manager and enable_compaction:
            try:
                # Get raw chat history for compaction check
                recent_chat = await self.session_repo.get_session_messages(
                    session_id=session_id,
                    limit=50,  # Get more for potential compaction
                )

                # CRITICAL: Reverse to chronological order (oldest first)
                # Repository returns newest first (desc), but LLM needs oldest first
                recent_chat.reverse()

                # Format messages for context manager
                messages = []
                for msg in recent_chat:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if content:
                        messages.append({"role": role, "content": content})

                # Prepare context with automatic compaction
                prepared_context: PreparedContext = await self.context_manager.prepare_context(
                    user_id=int(user_id) if user_id else 0,
                    session_id=session_id,
                    messages=messages,
                    system_prompt="",  # Will be added by agent
                    force_compaction=False,
                )

                elapsed = time.time() - start

                # Log compaction result
                if prepared_context.was_compacted:
                    self.logger.info(
                        f"[{flow_id}] ✅ Context COMPACTED: "
                        f"{prepared_context.compaction_result.original_tokens if prepared_context.compaction_result else '?'} → "
                        f"{prepared_context.compaction_result.final_tokens if prepared_context.compaction_result else '?'} tokens"
                    )

                self.logger.info(
                    f"[{flow_id}] Context loaded in {elapsed:.2f}s: "
                    f"history={len(prepared_context.messages)}, symbols={current_symbols}, "
                    f"compacted={prepared_context.was_compacted}, tokens={prepared_context.total_tokens}"
                )

                return {
                    "core_memory": prepared_context.core_memory,
                    "working_context": working_context,
                    "current_symbols": current_symbols,
                    "chat_history": prepared_context.messages,
                    "conversation_summary": prepared_context.summary,
                    "was_compacted": prepared_context.was_compacted,
                    "total_tokens": prepared_context.total_tokens,
                }

            except Exception as e:
                self.logger.warning(f"[{flow_id}] ContextManagementService failed, falling back: {e}")
                # Fall through to manual loading

        # Fallback: Manual loading without compaction
        # Core memory (user profile, preferences)
        core_memory = await self.core_memory.load_core_memory(user_id)

        # Conversation summary (for long conversations)
        conversation_summary = None
        try:
            summary = await self.summary_manager.get_active_summary(session_id)
            if summary:
                # get_active_summary returns string, not dict
                conversation_summary = self.summary_manager.format_summary_for_context(summary)
                self.logger.debug(f"[{flow_id}] Loaded conversation summary")
        except Exception as e:
            self.logger.warning(f"[{flow_id}] Failed to load conversation summary: {e}")

        # Chat history (recent messages)
        recent_chat = await self.session_repo.get_session_messages(
            session_id=session_id,
            limit=10,
        )

        # CRITICAL: Reverse to chronological order (oldest first)
        # Repository returns newest first (desc), but LLM needs oldest first
        recent_chat.reverse()

        # Format for agent
        chat_history = []
        for msg in recent_chat:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                chat_history.append({"role": role, "content": content})

        elapsed = time.time() - start
        self.logger.info(
            f"[{flow_id}] Context loaded in {elapsed:.2f}s (no compaction): "
            f"history={len(chat_history)}, symbols={current_symbols}, "
            f"has_summary={conversation_summary is not None}"
        )

        return {
            "core_memory": core_memory,
            "working_context": working_context,
            "current_symbols": current_symbols,
            "chat_history": chat_history,
            "conversation_summary": conversation_summary,
            "was_compacted": False,
            "total_tokens": 0,
        }

    async def _classify_query(
        self,
        flow_id: str,
        query: str,
        context_data: Dict[str, Any],
    ) -> UnifiedClassificationResult:
        """Run unified classification."""
        start = time.time()

        self.logger.info(f"[{flow_id}] Classifying query...")

        # Build classifier context
        context = ClassifierContext(
            query=query,
            conversation_history=context_data.get("chat_history", []),
            core_memory_summary=str(context_data.get("core_memory", "")),
            working_memory_summary=context_data.get("working_context", ""),
        )

        # Classify
        result = await self.classifier.classify(context)

        elapsed = time.time() - start
        self.logger.info(f"[{flow_id}] Classification completed in {elapsed:.2f}s")

        return result

    def _needs_deep_research(
        self,
        classification: UnifiedClassificationResult,
        query: str,
    ) -> bool:
        """
        Determine if query needs Deep Research Mode.

        Deep Research Mode is needed for:
        - Screener queries (find stocks by criteria)
        - Multi-step analysis requests
        - Comparison queries
        - Complex research questions
        """
        # Screener always needs Deep Research
        if classification.query_type == QueryType.SCREENER:
            return True

        # Check for complex query indicators
        complex_indicators = [
            "compare",
            "so sánh",
            "find all",
            "tìm tất cả",
            "research",
            "nghiên cứu",
            "deep analysis",
            "phân tích sâu",
            "portfolio",
            "danh mục",
        ]

        query_lower = query.lower()
        for indicator in complex_indicators:
            if indicator in query_lower:
                return True

        # Many symbols might need Deep Research
        if len(classification.symbols) > 3:
            return True

        return False

    async def _check_symbol_ambiguity(
        self,
        flow_id: str,
        symbols: List[str],
        language: str = "vi",
    ) -> List[Dict[str, Any]]:
        """
        Check if any symbols are ambiguous (exist in both crypto and stock).

        Simple O(1) check - no LLM, no context resolution.
        Just check cache and return options.

        Args:
            flow_id: Flow identifier for logging
            symbols: List of symbols from classification
            language: Response language (vi/en)

        Returns:
            List of disambiguation data for ambiguous symbols (empty if none)
        """
        try:
            resolver = get_asset_resolver()
            return resolver.check_symbols(symbols, language)
        except Exception as e:
            self.logger.warning(f"[{flow_id}] Symbol ambiguity check failed: {e}")
            return []  # Don't block on errors

    async def _run_normal_mode(
        self,
        flow_id: str,
        query: str,
        classification: UnifiedClassificationResult,
        context_data: Dict[str, Any],
        wm_integration: WorkingMemoryIntegration,
        system_language: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_thinking: bool = True,
        enable_llm_events: bool = True,
        charts: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the Normal Mode agent with TRUE STREAMING from LLM.

        Now yields dict events (not just strings) to support:
        - thinking: AI reasoning process
        - llm_thought: LLM thought events
        - turn_start: New turn started
        - tool_calls: Tools being called
        - tool_results: Tool execution results
        - content: Streaming content chunks
        - done: Agent completed (includes charts)
        - error: Error occurred
        """
        start = time.time()

        self.logger.info(f"[{flow_id}] Running agent loop with streaming...")

        total_turns = 0
        total_tool_calls = 0

        try:
            # Use run_stream for TRUE streaming from LLM
            async for event in self.agent.run_stream(
                query=query,
                classification=classification,
                conversation_history=context_data.get("chat_history", []),
                system_language=system_language,
                user_id=int(user_id) if user_id else None,
                session_id=session_id,
                core_memory=context_data.get("core_memory"),
                conversation_summary=context_data.get("conversation_summary"),
                enable_thinking=enable_thinking,
                enable_llm_events=enable_llm_events,
            ):
                event_type = event.get("type", "")

                if event_type == "content":
                    # Stream content chunks as dict events
                    content = event.get("content", "")
                    if content:
                        yield {"type": "content", "content": content}

                elif event_type == "thinking":
                    # Forward thinking events (Manus AI style)
                    if enable_thinking:
                        yield {
                            "type": "thinking",
                            "content": event.get("content", ""),
                            "phase": event.get("phase", "reasoning"),
                        }

                elif event_type == "llm_thought":
                    # Forward LLM thought events
                    if enable_llm_events:
                        yield {
                            "type": "llm_thought",
                            "thought": event.get("thought", ""),
                            "context": event.get("context", ""),
                        }

                elif event_type == "turn_start":
                    total_turns = event.get("turn", total_turns)
                    self.logger.debug(f"[{flow_id}] Turn {total_turns} started")
                    yield {"type": "turn_start", "turn": total_turns}

                elif event_type == "tool_calls":
                    tools = event.get("tools", [])
                    total_tool_calls += len(tools)
                    self.logger.debug(f"[{flow_id}] Tool calls: {[t['name'] for t in tools]}")
                    yield {"type": "tool_calls", "tools": tools}

                elif event_type == "tool_results":
                    results = event.get("results", [])
                    self.logger.debug(f"[{flow_id}] Tool results: {len(results)} items")
                    yield {"type": "tool_results", "results": results}

                elif event_type == "done":
                    total_turns = event.get("total_turns", total_turns)
                    total_tool_calls = event.get("total_tool_calls", total_tool_calls)
                    done_event = {
                        "type": "done",
                        "total_turns": total_turns,
                        "total_tool_calls": total_tool_calls,
                    }
                    # Include charts if available
                    if charts:
                        done_event["charts"] = charts
                    yield done_event

                elif event_type == "error":
                    error = event.get("error", "Unknown error")
                    self.logger.error(f"[{flow_id}] Agent error: {error}")
                    yield {"type": "error", "error": error}

                elif event_type == "max_turns_reached":
                    # Forward max turns event
                    yield {"type": "thinking", "content": f"Reached maximum turns ({event.get('turns', 10)})", "phase": "max_turns"}

            elapsed = time.time() - start
            self.logger.info(
                f"[{flow_id}] Agent completed: {total_turns} turns, "
                f"{total_tool_calls} tool calls, {elapsed:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"[{flow_id}] Agent stream error: {e}", exc_info=True)
            yield {"type": "error", "error": str(e)}


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

_handler_instance: Optional[NormalModeChatHandler] = None


def get_normal_mode_chat_handler(
    fallback_handler=None,
    enable_compaction: bool = True,
) -> NormalModeChatHandler:
    """
    Get singleton NormalModeChatHandler instance.

    Args:
        fallback_handler: ChatHandler for Deep Research Mode
        enable_compaction: Enable automatic context compaction

    Returns:
        NormalModeChatHandler instance
    """
    global _handler_instance

    if _handler_instance is None:
        _handler_instance = NormalModeChatHandler(
            fallback_handler=fallback_handler,
            enable_compaction=enable_compaction,
        )

    return _handler_instance


def reset_handler():
    """Reset singleton instance (for testing)."""
    global _handler_instance
    _handler_instance = None