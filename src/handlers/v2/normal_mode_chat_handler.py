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
import asyncio
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
        images: Optional[List[Any]] = None,
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
            images: Optional list of ProcessedImage for multimodal analysis

        Yields:
            Dict events with types: thinking, llm_thought, turn_start,
            tool_calls, tool_results, content, done, error
        """
        flow_id = f"NM-{datetime.now().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
        flow_start = time.time()

        # Use provided or default settings
        model = model_name or self.model_name
        provider = provider_type or self.provider_type

        self.logger.info("")
        self.logger.info("═" * 60)
        self.logger.info(f"🚀 REQUEST START | Session: {session_id[:8]}... | User: {user_id}")
        self.logger.info("═" * 60)
        display_query = query[:50] + "..." if len(query) > 50 else query
        self.logger.info(f"📥 Query: \"{display_query}\"")

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
            # STEP 1.5: PRE-RESOLVE SYMBOLS (NEW - before classification)
            # Enriches context so LLM classifier has better symbol info
            # Flow: Query → Symbol Pre-check → Enrich Context → LLM Classify
            # ================================================================
            pre_resolved = await self._pre_resolve_symbols(
                flow_id=flow_id,
                query=query,
                ui_context=context_data.get("ui_context"),
            )

            # Add pre-resolved info to context for classification
            if pre_resolved.get("context_hint"):
                context_data["pre_resolved_hint"] = pre_resolved["context_hint"]
                context_data["pre_resolved_symbols"] = pre_resolved["pre_resolved"]

            # ================================================================
            # STEP 2: UNIFIED CLASSIFICATION (skip if already provided)
            # Supports multimodal: images are analyzed for better intent detection
            # Now with pre-resolved symbol context for better accuracy!
            # ================================================================
            if classification is None:
                classification = await self._classify_query(
                    flow_id=flow_id,
                    query=query,
                    context_data=context_data,
                    images=images,  # Pass images for multimodal classification
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

            # Classification result logged by UnifiedClassifier with visual structure

            # ================================================================
            # STEP 2.3: SMART SYMBOL DISAMBIGUATION (Improved UX)
            # Instead of always asking user, use context to make smart decisions:
            # 1. UI context (active_tab) as strong signal
            # 2. Classification confidence
            # 3. Query language patterns (e.g., "coin", "crypto" → crypto)
            # ================================================================
            disambiguation_note = None  # Note to include in response if processed with guess

            if classification.symbols:
                ambiguous_symbols = await self._check_symbol_ambiguity(
                    flow_id=flow_id,
                    symbols=classification.symbols,
                    query=query,
                    context_data=context_data,
                )

                if ambiguous_symbols:
                    # Get UI context for smart resolution
                    ui_context = context_data.get("ui_context", {})
                    active_tab = ui_context.get("active_tab") if ui_context else None

                    # SMART DISAMBIGUATION: Use UI context or confidence to proceed
                    should_proceed = False
                    best_guess_type = None

                    # Strategy 1: Use active_tab as strong signal
                    if active_tab in ["stock", "crypto"]:
                        best_guess_type = active_tab
                        should_proceed = True
                        self.logger.info(
                            f"[{flow_id}] Using UI context '{active_tab}' for disambiguation"
                        )

                    # Strategy 2: Check query for strong indicators
                    elif self._has_strong_asset_indicator(query):
                        best_guess_type = self._detect_asset_type_from_query(query)
                        if best_guess_type:
                            should_proceed = True
                            self.logger.info(
                                f"[{flow_id}] Query indicates '{best_guess_type}'"
                            )

                    # Strategy 3: High confidence classification
                    elif classification.confidence >= 0.8 and classification.market_type:
                        best_guess_type = classification.market_type.value if hasattr(classification.market_type, 'value') else str(classification.market_type)
                        if best_guess_type != "both":
                            should_proceed = True
                            self.logger.info(
                                f"[{flow_id}] High confidence ({classification.confidence}) → {best_guess_type}"
                            )

                    if should_proceed and best_guess_type:
                        # Proceed with best guess, add note about alternatives
                        symbol = ambiguous_symbols[0].get("symbol", "")
                        options = ambiguous_symbols[0].get("options", [])

                        # Filter classification to use best guess asset type
                        classification = self._apply_asset_type_override(
                            classification, best_guess_type
                        )

                        # Build note for response
                        other_options = [
                            opt for opt in options
                            if opt.get("asset_class") != best_guess_type
                        ]
                        if other_options:
                            other_names = ", ".join(
                                f"{opt.get('name', symbol)} ({opt.get('asset_class', 'unknown')})"
                                for opt in other_options[:2]
                            )
                            disambiguation_note = (
                                f"💡 *Lưu ý: '{symbol}' có thể là {other_names}. "
                                f"Nếu bạn muốn phân tích loại khác, hãy nói rõ nhé.*"
                            )

                        yield {"type": "disambiguation_resolved", "data": {
                            "symbol": symbol,
                            "resolved_as": best_guess_type,
                            "alternatives_noted": bool(other_options),
                        }}
                    else:
                        # Need clarification - but make it user-friendly
                        disambiguation_msg = ambiguous_symbols[0].get("message", "")
                        self.logger.info(
                            f"[{flow_id}] Ambiguous symbol - asking user to clarify"
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
                self.logger.info("")
                self.logger.info("─" * 50)
                self.logger.info(f"🛤️ ROUTING: Deep Research Mode")
                self.logger.info("─" * 50)

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
                self.logger.info("")
                self.logger.info("─" * 50)
                if needs_deep_research:
                    self.logger.info(f"🛤️ ROUTING: Normal Mode (Deep Research unavailable)")
                else:
                    self.logger.info(f"🛤️ ROUTING: Normal Mode")
                self.logger.info("─" * 50)

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
                    images=images,
                    model_name=model,  # Pass user-provided model
                    provider_type=provider,  # Pass user-provided provider
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
            # MEMORY UPDATES (Fire-and-forget background tasks)
            # These don't block user response - run in background
            # ================================================================

            async def _background_summary():
                """Background task: Check if conversation needs summarization"""
                try:
                    result = await self.summary_manager.check_and_create_summary(
                        session_id=session_id,
                        user_id=int(user_id) if user_id else 0,
                    )
                    if result and result.get("created"):
                        self.logger.info(
                            f"[{flow_id}] Created conversation summary: "
                            f"version={result.get('version')}"
                        )
                except Exception as e:
                    self.logger.warning(f"[{flow_id}] Background summary failed: {e}")

            async def _background_memory_update():
                """Background task: Analyze conversation for memory updates"""
                try:
                    result = await self.memory_update.analyze_for_updates(
                        user_id=str(user_id) if user_id else "0",
                        user_message=query,
                        assistant_message=full_response,
                    )
                    if result and result.get("updated"):
                        self.logger.info(
                            f"[{flow_id}] Memory updated: action={result.get('action')}, "
                            f"categories={result.get('categories')}"
                        )
                except Exception as e:
                    self.logger.warning(f"[{flow_id}] Background memory update failed: {e}")

            # Schedule background tasks (fire-and-forget)
            asyncio.create_task(_background_summary())
            asyncio.create_task(_background_memory_update())

            self.logger.info("")
            self.logger.info("═" * 60)
            self.logger.info(f"✨ REQUEST COMPLETE | Total: {flow_time:.2f}s")
            self.logger.info("═" * 60)
            self.logger.info("")

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
        images: Optional[List[Any]] = None,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle chat with structured streaming events including thinking/reasoning.

        Supports multimodal input (images) for vision-based classification.

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
            images: Optional list of ProcessedImage for multimodal
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

            # Classification (with multimodal support)
            yield {"type": "classifying", "data": {"has_images": images is not None and len(images) > 0}}

            classification = await self._classify_query(
                flow_id=flow_id,
                query=query,
                context_data=context_data,
                images=images,  # Pass images for multimodal classification
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

    async def _pre_resolve_symbols(
        self,
        flow_id: str,
        query: str,
        ui_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        PRE-RESOLVE symbols BEFORE classification.

        This enriches the context so the LLM classifier has better information
        about which symbols exist and their types.

        New Flow:
            Query → Symbol Pre-check → Enrich Context → LLM Classify
        Old Flow:
            Query → LLM Classify → Symbol Resolve → Override

        Args:
            flow_id: Flow identifier for logging
            query: User query
            ui_context: Optional UI context (active_tab, etc.)

        Returns:
            Dict with pre_resolved_symbols info for classification enrichment
        """
        import re

        result = {
            "pre_resolved": [],       # List of resolved symbol info
            "potential_symbols": [],  # Raw extracted symbols
            "has_stock_suffix": False,  # .HK, .SS, etc.
            "context_hint": "",       # Text hint for classifier
        }

        try:
            # 1. Extract potential symbols from query using patterns
            # Stock patterns: AAPL, NVDA, VNM, 0700.HK, 600519.SS
            stock_pattern = r'\b([A-Z]{1,5})\b|\b(\d{4,6}\.[A-Z]{2})\b'
            # Crypto patterns: BTC, ETH, BTCUSDT
            crypto_pattern = r'\b([A-Z]{2,10}(?:USDT|USD|BUSD)?)\b'

            matches = re.findall(stock_pattern, query.upper())
            potential = set()
            for m in matches:
                symbol = m[0] or m[1]
                if symbol and len(symbol) >= 2:
                    potential.add(symbol)

            # Check for exchange suffixes (strong signal for stock)
            stock_suffixes = ['.HK', '.SS', '.SZ', '.VN', '.SI', '.KS', '.T', '.L']
            for suffix in stock_suffixes:
                if suffix in query.upper():
                    result["has_stock_suffix"] = True
                    break

            result["potential_symbols"] = list(potential)

            if not potential:
                return result

            # 2. Lookup in symbol cache
            symbol_cache = get_symbol_cache()

            for symbol in list(potential)[:5]:  # Limit to 5 to avoid overhead
                # Clean symbol (remove USDT suffix for lookup)
                clean_symbol = symbol.replace("USDT", "").replace("USD", "")
                if len(clean_symbol) < 2:
                    continue

                # Use correct API: exists() with AssetClass and lookup()
                from src.services.asset.symbol_cache import AssetClass

                is_crypto = symbol_cache.exists(clean_symbol, AssetClass.CRYPTO)
                is_stock = symbol_cache.exists(clean_symbol, AssetClass.STOCK)

                # Check ambiguity via lookup()
                symbol_info, is_ambiguous = symbol_cache.lookup(clean_symbol)

                info = {
                    "symbol": clean_symbol,
                    "original": symbol,
                    "is_crypto": is_crypto,
                    "is_stock": is_stock,
                    "is_ambiguous": is_ambiguous,
                }

                # Get detailed info if available
                if symbol_info:
                    info["name"] = symbol_info.name
                    info["asset_class"] = symbol_info.asset_class.value
                    if symbol_info.exchange:
                        info["exchange"] = symbol_info.exchange

                result["pre_resolved"].append(info)

            # 3. Build context hint for classifier
            if result["pre_resolved"]:
                hints = []
                for info in result["pre_resolved"]:
                    if info.get("is_ambiguous"):
                        hints.append(f"{info['symbol']}: AMBIGUOUS (crypto AND stock)")
                    elif info.get("is_crypto") and not info.get("is_stock"):
                        hints.append(f"{info['symbol']}: cryptocurrency")
                    elif info.get("is_stock") and not info.get("is_crypto"):
                        exchange = info.get("exchange", "")
                        hints.append(f"{info['symbol']}: stock{' (' + exchange + ')' if exchange else ''}")

                if hints:
                    result["context_hint"] = "Pre-resolved symbols: " + "; ".join(hints)

            # 4. Apply UI context for disambiguation hints
            if ui_context:
                active_tab = ui_context.get("active_tab")
                if active_tab in ["stock", "crypto"]:
                    result["context_hint"] += f" | UI context suggests: {active_tab}"

            if result["context_hint"]:
                self.logger.info(f"[{flow_id}] Pre-resolve: {result['context_hint']}")

        except Exception as e:
            self.logger.warning(f"[{flow_id}] Pre-resolve symbols failed: {e}")

        return result

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

        self.logger.info("")
        self.logger.info("─" * 50)
        self.logger.info(f"📋 CONTEXT LOADING")
        self.logger.info("─" * 50)

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
                    original = prepared_context.compaction_result.original_tokens if prepared_context.compaction_result else '?'
                    final = prepared_context.compaction_result.final_tokens if prepared_context.compaction_result else '?'
                    self.logger.info(f"  ├─ 🗜️ Context COMPACTED: {original} → {final} tokens")

                self.logger.info(f"  ├─ History: {len(prepared_context.messages)} messages")
                self.logger.info(f"  ├─ Symbols: {current_symbols}")
                self.logger.info(f"  ├─ Tokens: {prepared_context.total_tokens}")
                self.logger.info(f"  └─ ⏱️ Time: {elapsed:.2f}s")

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
        # OPTIMIZED: Load all context in parallel using asyncio.gather()
        # This reduces ~50ms → ~20ms by parallelizing independent operations

        async def _load_core_memory():
            """Load core memory (user profile, preferences)"""
            return await self.core_memory.load_core_memory(user_id)

        async def _load_summary():
            """Load conversation summary"""
            try:
                summary = await self.summary_manager.get_active_summary(session_id)
                if summary:
                    return self.summary_manager.format_summary_for_context(summary)
            except Exception as e:
                self.logger.warning(f"[{flow_id}] Failed to load summary: {e}")
            return None

        async def _load_chat_history():
            """Load recent chat history"""
            recent_chat = await self.session_repo.get_session_messages(
                session_id=session_id,
                limit=10,
            )
            # CRITICAL: Reverse to chronological order (oldest first)
            recent_chat.reverse()
            # Format for agent
            return [
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in recent_chat
                if msg.get("content")
            ]

        # Execute all loads in parallel
        core_memory, conversation_summary, chat_history = await asyncio.gather(
            _load_core_memory(),
            _load_summary(),
            _load_chat_history(),
        )

        elapsed = time.time() - start
        self.logger.info(f"  ├─ History: {len(chat_history)} messages")
        self.logger.info(f"  ├─ Symbols: {current_symbols}")
        self.logger.info(f"  ├─ Summary: {'Yes' if conversation_summary else 'No'}")
        self.logger.info(f"  └─ ⏱️ Time: {elapsed:.2f}s (no compaction)")

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
        images: Optional[List[Any]] = None,
    ) -> UnifiedClassificationResult:
        """
        Run unified classification with multimodal support.

        When images are provided, uses vision-capable model to analyze
        both text and images for better intent classification.

        Args:
            flow_id: Flow identifier for logging
            query: User's text query
            context_data: Context including history, memory, ui_context
            images: Optional list of ProcessedImage for multimodal classification

        Returns:
            UnifiedClassificationResult with classification result
        """
        start = time.time()
        # Classification logging is handled by UnifiedClassifier with visual structure

        # Build classifier context with multimodal support
        context = ClassifierContext(
            query=query,
            conversation_history=context_data.get("chat_history", []),
            core_memory_summary=str(context_data.get("core_memory", "")),
            working_memory_summary=context_data.get("working_context", ""),
            # Soft Context Inheritance: pass UI context for symbol resolution
            ui_context=context_data.get("ui_context"),
            # Multimodal: pass images for vision-based classification
            images=images,
            # Pre-resolved symbols (NEW - enriches context BEFORE LLM)
            pre_resolved_hint=context_data.get("pre_resolved_hint"),
            pre_resolved_symbols=context_data.get("pre_resolved_symbols"),
        )

        # Classify (detailed logging handled by UnifiedClassifier)
        result = await self.classifier.classify(context)
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
        query: str,
        context_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Check if any symbols are ambiguous (exist in both crypto and stock).

        Args:
            flow_id: Flow identifier for logging
            symbols: List of symbols from classification
            query: User query for context-based resolution
            context_data: Context data including chat history

        Returns:
            List of disambiguation data for ambiguous symbols (empty if none)
        """
        try:
            symbol_cache = get_symbol_cache()
            asset_resolver = get_asset_resolver()

            ambiguous_results = []

            for symbol in symbols:
                if symbol_cache.is_ambiguous(symbol):
                    # Try to resolve using context
                    chat_history = context_data.get("chat_history", [])

                    resolved, _ = await asset_resolver.resolve(
                        query=query,
                        conversation_context=chat_history,
                        skip_confirmation=False,
                        use_llm=False,  # Fast check without LLM
                    )

                    # Check if this symbol needs confirmation
                    for asset in resolved:
                        if asset.symbol == symbol and asset.needs_confirmation:
                            # Get disambiguation options
                            options = asset_resolver.get_disambiguation_options(symbol)

                            ambiguous_results.append({
                                "symbol": symbol,
                                "options": options,
                                "message": asset.disambiguation_message or self._build_disambiguation_message(symbol, options),
                            })

                            self.logger.info(
                                f"[{flow_id}] Symbol '{symbol}' is ambiguous: "
                                f"{len(options)} options"
                            )
                            break

            return ambiguous_results

        except Exception as e:
            self.logger.warning(f"[{flow_id}] Symbol ambiguity check failed: {e}")
            return []  # Don't block on errors

    def _build_disambiguation_message(
        self,
        symbol: str,
        options: List[Dict[str, Any]],
    ) -> str:
        """Build user-friendly disambiguation message (multi-language)"""
        lines = [f"Tôi thấy '{symbol}' có thể là:"]

        for i, opt in enumerate(options, 1):
            asset_type = opt.get("asset_class", "unknown")
            name = opt.get("name", symbol)
            exchange = opt.get("exchange", "")

            if asset_type == "crypto":
                desc = opt.get("description", "Cryptocurrency")
                lines.append(f"  {i}. {name} ({desc})")
            elif asset_type == "stock":
                exchange_info = f" - {exchange}" if exchange else ""
                lines.append(f"  {i}. {name} (Stock{exchange_info})")
            else:
                lines.append(f"  {i}. {name} ({asset_type})")

        lines.append("\nBạn muốn phân tích loại nào? Vui lòng trả lời với số hoặc tên loại tài sản.")

        return "\n".join(lines)

    def _has_strong_asset_indicator(self, query: str) -> bool:
        """
        Check if query has strong indicators for asset type.

        Supports multiple languages: Vietnamese, English, Chinese.
        """
        query_lower = query.lower()

        # Crypto indicators (multi-language)
        crypto_keywords = [
            # English
            "crypto", "cryptocurrency", "coin", "token", "blockchain", "defi",
            "bitcoin", "ethereum", "altcoin", "stablecoin",
            # Vietnamese
            "tiền mã hóa", "tiền điện tử", "tiền ảo", "đồng coin", "token",
            # Chinese
            "加密", "加密货币", "虚拟货币", "代币",
        ]

        # Stock indicators (multi-language)
        stock_keywords = [
            # English
            "stock", "share", "equity", "dividend", "earnings", "eps", "p/e",
            "market cap", "ipo", "nasdaq", "nyse", "exchange",
            # Vietnamese
            "cổ phiếu", "chứng khoán", "cổ tức", "lợi nhuận", "vốn hóa",
            "sàn chứng khoán", "hose", "hnx", "upcom",
            # Chinese
            "股票", "股份", "股市", "证券", "股息",
        ]

        for keyword in crypto_keywords + stock_keywords:
            if keyword in query_lower:
                return True

        return False

    def _detect_asset_type_from_query(self, query: str) -> Optional[str]:
        """
        Detect asset type from query keywords.

        Returns: "stock", "crypto", or None if ambiguous
        """
        query_lower = query.lower()

        crypto_score = 0
        stock_score = 0

        # Crypto indicators
        crypto_keywords = [
            "crypto", "cryptocurrency", "coin", "token", "blockchain",
            "tiền mã hóa", "tiền điện tử", "tiền ảo", "đồng coin",
            "加密", "加密货币", "虚拟货币", "代币",
        ]

        # Stock indicators
        stock_keywords = [
            "stock", "share", "equity", "dividend", "earnings",
            "cổ phiếu", "chứng khoán", "cổ tức",
            "股票", "股份", "股市", "证券",
        ]

        for keyword in crypto_keywords:
            if keyword in query_lower:
                crypto_score += 1

        for keyword in stock_keywords:
            if keyword in query_lower:
                stock_score += 1

        if crypto_score > stock_score:
            return "crypto"
        elif stock_score > crypto_score:
            return "stock"

        return None

    def _apply_asset_type_override(
        self,
        classification: UnifiedClassificationResult,
        asset_type: str,
    ) -> UnifiedClassificationResult:
        """
        Apply asset type override to classification result.

        Updates market_type and adjusts tool_categories accordingly.
        """
        from src.agents.classification.models import MarketType

        # Update market type
        if asset_type == "crypto":
            classification.market_type = MarketType.CRYPTO
            # Ensure crypto category is included
            if "crypto" not in classification.tool_categories:
                classification.tool_categories.append("crypto")
            # Remove stock-only categories if present
            classification.tool_categories = [
                c for c in classification.tool_categories
                if c not in ["fundamentals"]  # Crypto doesn't have fundamentals
            ]
        elif asset_type == "stock":
            classification.market_type = MarketType.STOCK
            # Remove crypto category
            classification.tool_categories = [
                c for c in classification.tool_categories
                if c != "crypto"
            ]

        return classification

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
        images: Optional[List[Any]] = None,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
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

        self.logger.info("")
        self.logger.info("─" * 50)
        self.logger.info(f"🤖 AGENT LOOP")
        self.logger.info("─" * 50)

        total_turns = 0
        total_tool_calls = 0

        try:
            # Use run_stream for TRUE streaming from LLM
            # Pass user-provided model_name/provider_type for final response
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
                images=images,
                model_name=model_name,  # User-provided model for final response
                provider_type=provider_type,  # User-provided provider
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
                    self.logger.info(f"  ├─ [TURN {total_turns}] Processing...")
                    yield {"type": "turn_start", "turn": total_turns}

                elif event_type == "tool_calls":
                    tools = event.get("tools", [])
                    total_tool_calls += len(tools)
                    tool_names = [t['name'] for t in tools]
                    self.logger.info(f"  │  🔧 Tools: {tool_names}")
                    yield {"type": "tool_calls", "tools": tools}

                elif event_type == "tool_results":
                    results = event.get("results", [])
                    self.logger.info(f"  │  ✅ Results: {len(results)} items")
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
            self.logger.info(f"  └─ ✅ Agent complete: {total_turns} turns, {total_tool_calls} tools, {elapsed:.2f}s")

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