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

__all__ = [
    # Models
    "UnifiedClassificationResult",
    "ClassifierContext",
    "QueryType",
    "MarketType",
    "VALID_CATEGORIES",
    # Classifier
    "UnifiedClassifier",
    "get_unified_classifier",
    "reset_classifier",
    "clear_classification_cache",  # async - clears Redis + local
    "clear_classification_cache_sync",  # sync - clears local only
]