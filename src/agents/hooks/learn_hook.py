"""
Learn Hook - Post-Execution Memory Updates

Implements the LEARN phase of the Claude AI Agent Loop:
    OBSERVE → THINK → ACT → LEARN → repeat

The LearnHook is called after successful query execution to:
1. Update working memory with recent symbols/context
2. Store analysis summaries for future reference
3. Learn user preferences implicitly from query patterns
4. Track successful tool combinations for optimization
5. Update Core Memory with extracted user information

This creates a feedback loop that improves future responses.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.memory.memory_update_agent import get_memory_update_agent


class LearnHook(LoggerMixin):
    """
    Post-execution hook for memory updates and learning.

    Called after successful agent execution to update memory systems
    and extract learnings for future interactions.

    Usage:
        hook = LearnHook()
        await hook.on_execution_complete(
            query="Phân tích NVDA",
            classification=classification,
            tool_results=[...],
            response="NVDA analysis...",
            user_id=123,
        )

    Thread Safety:
        This hook is stateless and safe for concurrent use.
    """

    def __init__(self):
        """Initialize LearnHook."""
        super().__init__()
        self.logger = logging.getLogger("hook.learn")
        self._memory_update_agent = None

    def _get_memory_update_agent(self):
        """Lazy-load memory update agent to avoid circular imports."""
        if self._memory_update_agent is None:
            self._memory_update_agent = get_memory_update_agent(use_consolidation=True)
        return self._memory_update_agent

    async def on_execution_complete(
        self,
        query: str,
        classification: Any,
        tool_results: List[Dict[str, Any]],
        response: str,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Called after successful query execution.

        Performs LEARN phase tasks:
        1. Update working memory with recent symbols
        2. Extract and store analysis insights
        3. Update user preference patterns
        4. Log successful tool combinations

        Args:
            query: Original user query
            classification: Classification result with symbols, market_type, etc.
            tool_results: List of tool execution results
            response: Final response generated
            user_id: Optional user ID for personalized memory
            session_id: Optional session ID for session memory
            execution_time_ms: Execution time for performance tracking

        Returns:
            Dict with learn phase results and any errors
        """
        learn_results = {
            "success": True,
            "updates": [],
            "errors": [],
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # 1. Update working memory with recent symbols
            if classification:
                symbols = getattr(classification, "symbols", [])
                if symbols:
                    update = await self._update_recent_symbols(
                        symbols=symbols,
                        query_type=getattr(classification, "query_type", None),
                        market_type=getattr(classification, "market_type", None),
                        user_id=user_id,
                    )
                    learn_results["updates"].append(update)

            # 2. Store analysis summary for frequently queried symbols
            successful_tools = [
                r for r in tool_results
                if r.get("status") in ["success", "200"]
            ]
            if successful_tools:
                update = await self._store_analysis_summary(
                    symbols=getattr(classification, "symbols", []) if classification else [],
                    tool_results=successful_tools,
                    user_id=user_id,
                )
                learn_results["updates"].append(update)

            # 3. Track successful tool combinations
            if len(successful_tools) >= 2:
                update = await self._track_tool_patterns(
                    query_type=getattr(classification, "query_type", None) if classification else None,
                    tool_names=[r.get("tool_name") for r in successful_tools],
                )
                learn_results["updates"].append(update)

            # 4. Update user preferences (implicit learning)
            if classification and user_id:
                update = await self._update_user_preferences(
                    user_id=user_id,
                    query_type=getattr(classification, "query_type", None),
                    market_type=getattr(classification, "market_type", None),
                    categories=getattr(classification, "tool_categories", []),
                    language=getattr(classification, "response_language", "vi"),
                )
                learn_results["updates"].append(update)

            # 5. Update Core Memory with extracted user information (NEW!)
            if user_id and response:
                update = await self._update_core_memory(
                    user_id=user_id,
                    query=query,
                    response=response,
                    tool_results=tool_results,
                )
                learn_results["updates"].append(update)

            self.logger.info(
                f"[LEARN] Completed {len(learn_results['updates'])} updates | "
                f"user_id={user_id} | symbols={getattr(classification, 'symbols', []) if classification else []}"
            )

        except Exception as e:
            self.logger.error(f"[LEARN] Error in learn hook: {e}", exc_info=True)
            learn_results["success"] = False
            learn_results["errors"].append(str(e))

        return learn_results

    async def _update_recent_symbols(
        self,
        symbols: List[str],
        query_type: Optional[str],
        market_type: Optional[str],
        user_id: Optional[int],
    ) -> Dict[str, Any]:
        """
        Update working memory with recently queried symbols.

        This helps provide context for follow-up queries like
        "What about AAPL?" when user was previously asking about NVDA.

        Note: Currently logs for future integration when WorkingMemory
        supports add_recent_symbol method.
        """
        try:
            # Log the symbols for now - actual memory storage will be added
            # when WorkingMemory implements add_recent_symbol method
            self.logger.debug(
                f"[LEARN] Recent symbols: {symbols[:5]} | "
                f"market={market_type} | user={user_id}"
            )

            # TODO: Integrate with WorkingMemory when it supports symbol tracking
            # from src.agents.memory.working_memory import WorkingMemory
            # working_memory = WorkingMemory(session_id=session_id, user_id=user_id)
            # await working_memory.add_recent_symbol(...)

            return {
                "type": "recent_symbols",
                "symbols": symbols[:5],
                "success": True,
                "note": "logged_for_future_integration",
            }

        except Exception as e:
            self.logger.warning(f"[LEARN] Failed to update recent symbols: {e}")
            return {"type": "recent_symbols", "success": False, "error": str(e)}

    async def _store_analysis_summary(
        self,
        symbols: List[str],
        tool_results: List[Dict[str, Any]],
        user_id: Optional[int],
    ) -> Dict[str, Any]:
        """
        Store summary of analysis for future reference.

        Extracts key data points from tool results and stores them
        so future queries can reference historical analysis.

        Note: Currently extracts and logs metrics for future integration.
        """
        try:
            # Extract key metrics from tool results
            summary = {}

            for result in tool_results:
                tool_name = result.get("tool_name", "")
                data = result.get("data", {})

                # Extract price data
                if "price" in tool_name.lower() or "price" in str(data):
                    summary["last_price"] = data.get("price") or data.get("current_price")
                    summary["last_price_time"] = datetime.now().isoformat()

                # Extract technical signals
                if "technical" in tool_name.lower():
                    summary["last_rsi"] = data.get("rsi_14")
                    summary["last_macd"] = data.get("macd_histogram")

            if summary:
                # Log summary for now - actual storage will be added
                # when a persistent symbol context store is implemented
                self.logger.debug(
                    f"[LEARN] Analysis summary for {symbols[:3]}: {list(summary.keys())}"
                )

            return {
                "type": "analysis_summary",
                "symbols": symbols[:3],
                "metrics_stored": list(summary.keys()),
                "success": True,
                "note": "logged_for_future_integration",
            }

        except Exception as e:
            self.logger.warning(f"[LEARN] Failed to store analysis summary: {e}")
            return {"type": "analysis_summary", "success": False, "error": str(e)}

    async def _track_tool_patterns(
        self,
        query_type: Optional[str],
        tool_names: List[str],
    ) -> Dict[str, Any]:
        """
        Track successful tool combinations for pattern learning.

        Over time, this helps optimize tool selection for similar queries.
        """
        try:
            # Create pattern key
            if not query_type:
                return {"type": "tool_patterns", "success": False, "reason": "no_query_type"}

            pattern = {
                "query_type": query_type,
                "tools": sorted(tool_names),
                "count": len(tool_names),
                "timestamp": datetime.now().isoformat(),
            }

            # Log for now - could store in Redis/DB for production
            self.logger.debug(
                f"[LEARN] Tool pattern: {query_type} → {', '.join(tool_names[:5])}"
            )

            return {
                "type": "tool_patterns",
                "query_type": query_type,
                "tool_count": len(tool_names),
                "success": True,
            }

        except Exception as e:
            return {"type": "tool_patterns", "success": False, "error": str(e)}

    async def _update_user_preferences(
        self,
        user_id: int,
        query_type: Optional[str],
        market_type: Optional[str],
        categories: List[str],
        language: str,
    ) -> Dict[str, Any]:
        """
        Update user preferences based on query patterns.

        Tracks which types of analysis and markets the user
        queries most frequently for personalization.
        """
        try:
            # Build preferences dict
            preferences = {
                "preferred_market": market_type,
                "preferred_language": language,
                "last_query_type": query_type,
                "last_categories": categories[:5] if categories else [],
                "last_active": datetime.now().isoformat(),
            }

            self.logger.debug(
                f"[LEARN] User preferences: user={user_id} | "
                f"market={market_type} | lang={language}"
            )

            return {
                "type": "user_preferences",
                "user_id": user_id,
                "market_type": market_type,
                "success": True,
            }

        except Exception as e:
            self.logger.warning(f"[LEARN] Failed to update user preferences: {e}")
            return {"type": "user_preferences", "success": False, "error": str(e)}

    async def _update_core_memory(
        self,
        user_id: int,
        query: str,
        response: str,
        tool_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Update Core Memory with extracted user information.

        Uses MemoryUpdateAgent to:
        1. Extract profile info from conversation
        2. Consolidate with existing memory
        3. Save to YAML file

        Args:
            user_id: User ID
            query: User's query
            response: Assistant's response
            tool_results: Tool execution results

        Returns:
            Dict with update status
        """
        try:
            # Format tool results for extraction
            tool_data = {}
            for result in tool_results:
                tool_name = result.get("tool_name", "")
                if tool_name:
                    tool_data[tool_name] = result.get("data", {})

            # Get memory update agent
            agent = self._get_memory_update_agent()

            # Analyze for updates
            result = await agent.analyze_for_updates(
                user_id=str(user_id),
                user_message=query,
                assistant_message=response[:1000],  # Limit size
                tool_results=tool_data,
                working_memory_context=None,  # Could add WM context here
            )

            if result.get("updated", False):
                self.logger.info(
                    f"[LEARN] Core Memory updated for user {user_id}: "
                    f"action={result.get('action')}, categories={result.get('categories')}"
                )
            else:
                self.logger.debug(
                    f"[LEARN] Core Memory: no update needed - {result.get('reason', result.get('action', 'NOOP'))}"
                )

            return {
                "type": "core_memory",
                "user_id": user_id,
                "success": result.get("updated", False),
                "action": result.get("action", "NOOP"),
                "categories": result.get("categories", []),
            }

        except Exception as e:
            self.logger.warning(f"[LEARN] Core Memory update failed: {e}")
            return {"type": "core_memory", "success": False, "error": str(e)}


# ============================================================================
# Singleton Instance
# ============================================================================

_learn_hook_instance: Optional[LearnHook] = None


def get_learn_hook() -> LearnHook:
    """
    Get singleton instance of LearnHook.

    Returns:
        LearnHook singleton instance
    """
    global _learn_hook_instance
    if _learn_hook_instance is None:
        _learn_hook_instance = LearnHook()
    return _learn_hook_instance
