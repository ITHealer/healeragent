"""
Wrapper around the existing IntentClassifier from src/agents/classification/.

Why: The invest_agent module needs intent classification but must NOT duplicate
the existing IntentClassifier logic. This wrapper provides a clean interface
that adapts the existing classifier's output into the invest_agent's own types,
adding default handling and error recovery.

How: Imports and delegates to the real IntentClassifier. Catches failures and
returns safe defaults so the orchestrator always gets a usable result.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.invest_agent.core.exceptions import InvestAgentError

logger = logging.getLogger(__name__)


class ClassificationResult(BaseModel):
    """Simplified classification result for the invest_agent orchestrator.

    Maps from the full IntentResult but only exposes what the orchestrator needs.
    """
    intent_summary: str = ""
    symbols: List[str] = Field(default_factory=list)
    complexity: str = "direct"  # "direct" or "agent_loop"
    requires_tools: bool = False
    analysis_type: str = "general"
    response_language: str = "en"
    confidence: float = 0.5
    query_type: str = ""


class IntentWrapper:
    """Wraps the existing IntentClassifier for use by the invest_agent module.

    Why a wrapper instead of direct use: The orchestrator shouldn't depend on
    the internal structure of IntentResult or know about the classifier's
    retry/fallback logic. This wrapper provides:
    1. A stable interface that won't break if IntentClassifier changes
    2. Graceful fallback to heuristics if the classifier fails entirely
    3. Adaptation of IntentResult -> ClassificationResult
    """

    def __init__(self, model_name: Optional[str] = None, provider_type: Optional[str] = None):
        self._classifier = None
        self._model_name = model_name
        self._provider_type = provider_type

    def _ensure_classifier(self):
        """Lazy-init the real IntentClassifier to avoid import-time side effects."""
        if self._classifier is None:
            try:
                from src.agents.classification.intent_classifier import IntentClassifier
                self._classifier = IntentClassifier(
                    model_name=self._model_name,
                    provider_type=self._provider_type,
                )
            except Exception as e:
                logger.error(f"[IntentWrapper] Failed to initialize IntentClassifier: {e}")
                raise InvestAgentError(f"IntentClassifier init failed: {e}")

    async def classify(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        ui_context: Optional[Dict[str, Any]] = None,
        working_memory_symbols: Optional[List[str]] = None,
    ) -> ClassificationResult:
        """Classify a user query using the existing IntentClassifier.

        Always returns a ClassificationResult, even on failure (falls back to
        heuristic defaults). The orchestrator can safely use the result without
        checking for None.
        """
        self._ensure_classifier()

        try:
            result = await self._classifier.classify(
                query=query,
                conversation_history=conversation_history,
                ui_context=ui_context,
                working_memory_symbols=working_memory_symbols,
            )

            return ClassificationResult(
                intent_summary=result.intent_summary,
                symbols=result.validated_symbols,
                complexity=result.complexity.value if result.complexity else "direct",
                requires_tools=result.requires_tools,
                analysis_type=result.analysis_type.value if result.analysis_type else "general",
                response_language=result.response_language or "en",
                confidence=result.confidence,
                query_type=result.query_type or "",
            )

        except Exception as e:
            logger.warning(
                f"[IntentWrapper] Classification failed: {e}. Using heuristic fallback."
            )
            return self._heuristic_fallback(query)

    @staticmethod
    def _heuristic_fallback(query: str) -> ClassificationResult:
        """Basic heuristic classification when the LLM classifier fails.

        Extracts symbols from query using regex and estimates complexity
        from query length and keyword presence.
        """
        import re

        # Extract potential stock symbols (1-5 uppercase letters)
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query)
        # Filter common English words
        common_words = {"I", "A", "THE", "AND", "OR", "NOT", "FOR", "IS", "IT", "IN", "TO", "OF", "AT", "ON", "BY"}
        symbols = [s for s in symbols if s not in common_words]

        # Estimate complexity
        word_count = len(query.split())
        complex_keywords = {"compare", "analysis", "fundamental", "technical", "risk", "portfolio", "backtest", "valuation"}
        has_complex_keyword = any(kw in query.lower() for kw in complex_keywords)

        is_complex = word_count > 15 or len(symbols) > 1 or has_complex_keyword

        return ClassificationResult(
            intent_summary=f"Heuristic: {'complex' if is_complex else 'simple'} query about {symbols or 'general topic'}",
            symbols=symbols[:5],  # Limit to 5 symbols
            complexity="agent_loop" if is_complex else "direct",
            requires_tools=bool(symbols) or has_complex_keyword,
            analysis_type="general",
            response_language="en",
            confidence=0.3,  # Low confidence for heuristic
            query_type="heuristic_fallback",
        )
