"""
Classification Module

Unified classification system that merges query classification and
tool necessity validation into a single LLM call.

Usage:
    from src.agents.classification import (
        UnifiedClassifier,
        UnifiedClassificationResult,
        ClassifierContext,
        get_unified_classifier,
    )

    # Create context
    context = ClassifierContext(
        query="Phân tích kỹ thuật NVDA",
        conversation_history=[...],
        working_memory_summary="current_symbols: NVDA"
    )

    # Classify
    classifier = get_unified_classifier()
    result = await classifier.classify(context)

    # Use result
    if result.requires_tools:
        tools = load_tools(result.tool_categories)
        # ... execute tools
    else:
        # Direct LLM response
        pass
"""

from .models import (
    UnifiedClassificationResult,
    ClassifierContext,
    QueryType,
    MarketType,
    VALID_CATEGORIES,
)

from .unified_classifier import (
    UnifiedClassifier,
    get_unified_classifier,
    reset_classifier,
    clear_classification_cache,
    clear_classification_cache_sync,
)

# New: IntentClassifier for simplified architecture
from .intent_classifier import (
    IntentClassifier,
    IntentResult,
    IntentComplexity,
    IntentMarketType,
    get_intent_classifier,
    reset_intent_classifier,
    clear_intent_cache,
)

__all__ = [
    # Models
    "UnifiedClassificationResult",
    "ClassifierContext",
    "QueryType",
    "MarketType",
    "VALID_CATEGORIES",
    # Classifier (Legacy)
    "UnifiedClassifier",
    "get_unified_classifier",
    "reset_classifier",
    "clear_classification_cache",  # async - clears Redis + local
    "clear_classification_cache_sync",  # sync - clears local only
    # Intent Classifier (New - simplified architecture)
    "IntentClassifier",
    "IntentResult",
    "IntentComplexity",
    "IntentMarketType",
    "get_intent_classifier",
    "reset_intent_classifier",
    "clear_intent_cache",
]