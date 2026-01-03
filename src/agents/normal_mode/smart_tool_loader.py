"""
Smart Tool Loader - Adaptive Tool Loading with Semantic Search

Implements Progressive Disclosure pattern for tool management:
- Level 1: Category-based filtering (fast, no embeddings)
- Level 2: Semantic search with embeddings (when > SEMANTIC_THRESHOLD tools)

Loading Strategies:
1. If total_tools <= LOAD_ALL_THRESHOLD (40): Load ALL tools
2. If total_tools > 40: Filter by categories first
3. If filtered > MAX_TOOLS_AFTER_FILTER (20): Use semantic ranking

Pattern inspired by:
- OpenAI's function calling (load all when small)
- AnyTool's semantic tool selection (embeddings for large registries)

Usage:
    loader = SmartToolLoader()

    # Get tools based on classification
    result = await loader.load_tools(
        classification=classification,
        query="Phân tích kỹ thuật NVDA",
        max_tools=20
    )

    print(f"Loaded {result.loaded_count} tools via {result.loading_method}")
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.tools.registry import ToolRegistry
from src.agents.tools.tool_loader import get_registry
from src.agents.classification.models import UnifiedClassificationResult


# ============================================================================
# CONFIGURATION
# ============================================================================

# Loading thresholds
LOAD_ALL_THRESHOLD = 40          # Load all tools if registry <= this size
MAX_TOOLS_AFTER_FILTER = 20      # Max tools after category filtering
SEMANTIC_TOP_K = 15              # Top K tools from semantic search

# Embedding cache TTL
EMBEDDING_CACHE_TTL_SECONDS = 3600  # 1 hour


class LoadingMethod(str, Enum):
    """How tools were loaded"""
    ALL = "all"                   # Loaded all tools (small registry)
    CATEGORY = "category"         # Filtered by categories
    SEMANTIC = "semantic"         # Semantic search ranking
    HYBRID = "hybrid"             # Category + semantic


@dataclass
class ToolLoadingResult:
    """Result of tool loading operation"""
    tools: List[Dict[str, Any]]        # OpenAI function format
    loading_method: LoadingMethod
    total_available: int
    loaded_count: int
    categories_used: List[str] = field(default_factory=list)
    semantic_scores: Optional[Dict[str, float]] = None
    load_time_ms: float = 0.0


# ============================================================================
# TOOL EMBEDDING CACHE
# ============================================================================

class ToolEmbeddingCache:
    """
    Cache embeddings for tool descriptions.

    Embeddings are computed once and cached for fast semantic search.
    Cache is invalidated when tools are added/removed.
    """

    def __init__(self):
        self._embeddings: Dict[str, np.ndarray] = {}
        self._descriptions: Dict[str, str] = {}
        self._last_updated: Optional[datetime] = None
        self._model = None

    def _get_embedding_model(self):
        """Lazy load embedding model"""
        if self._model is None:
            try:
                from src.helpers.text_embedding_helper import text_embedding_model
                self._model = text_embedding_model
            except Exception:
                # Fallback: use sentence transformers directly
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._model

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text"""
        model = self._get_embedding_model()

        # Check if it's fastembed TextEmbedding
        if hasattr(model, 'embed'):
            # fastembed returns generator
            embeddings = list(model.embed([text]))
            return np.array(embeddings[0])
        else:
            # sentence-transformers
            return model.encode(text, convert_to_numpy=True)

    def build_cache(self, tools: Dict[str, Any]) -> None:
        """Build embedding cache for all tools"""
        for tool_name, tool_info in tools.items():
            if tool_name not in self._embeddings:
                # Build description text
                desc = self._build_tool_description(tool_name, tool_info)
                self._descriptions[tool_name] = desc
                self._embeddings[tool_name] = self._compute_embedding(desc)

        self._last_updated = datetime.now()

    def _build_tool_description(self, name: str, info: Any) -> str:
        """Build searchable description for tool"""
        parts = [name]

        if hasattr(info, 'description'):
            parts.append(info.description)
        elif isinstance(info, dict) and 'description' in info:
            parts.append(info['description'])

        if hasattr(info, 'category'):
            parts.append(f"category: {info.category}")
        elif isinstance(info, dict) and 'category' in info:
            parts.append(f"category: {info['category']}")

        return " | ".join(parts)

    def get_embedding(self, tool_name: str) -> Optional[np.ndarray]:
        """Get cached embedding for tool"""
        return self._embeddings.get(tool_name)

    def compute_query_embedding(self, query: str) -> np.ndarray:
        """Compute embedding for query"""
        return self._compute_embedding(query)

    def is_valid(self) -> bool:
        """Check if cache is valid"""
        if self._last_updated is None:
            return False
        age = (datetime.now() - self._last_updated).total_seconds()
        return age < EMBEDDING_CACHE_TTL_SECONDS

    def clear(self) -> None:
        """Clear cache"""
        self._embeddings.clear()
        self._descriptions.clear()
        self._last_updated = None


# Global embedding cache
_embedding_cache = ToolEmbeddingCache()


# ============================================================================
# SMART TOOL LOADER
# ============================================================================

class SmartToolLoader(LoggerMixin):
    """
    Adaptive tool loading based on classification and query semantics.

    Strategies:
    1. LOAD ALL: When registry size <= 40 tools (OpenAI/Claude style)
    2. CATEGORY: Filter by tool_categories from classification
    3. SEMANTIC: Rank by embedding similarity to query
    4. HYBRID: Category filter + semantic ranking

    Thread-safe singleton pattern.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, registry: Optional[ToolRegistry] = None):
        if self._initialized:
            return

        super().__init__()

        self.registry = registry or get_registry()
        self._embedding_cache = _embedding_cache

        # Pre-build tool metadata
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._category_index: Dict[str, List[str]] = {}
        self._build_metadata()

        self._initialized = True

        self.logger.info(
            f"[SMART_LOADER] Initialized: {len(self._tool_metadata)} tools, "
            f"{len(self._category_index)} categories"
        )

    def _build_metadata(self) -> None:
        """Build tool metadata and category index"""
        all_schemas = self.registry.get_all_schemas()

        for tool_name, schema in all_schemas.items():
            self._tool_metadata[tool_name] = {
                "name": tool_name,
                "category": schema.category,
                "description": schema.description,
                "requires_symbol": schema.requires_symbol,
                "schema": schema,
            }

            # Build category index
            category = schema.category
            if category not in self._category_index:
                self._category_index[category] = []
            self._category_index[category].append(tool_name)

    async def load_tools(
        self,
        classification: UnifiedClassificationResult,
        query: str,
        max_tools: int = MAX_TOOLS_AFTER_FILTER,
    ) -> ToolLoadingResult:
        """
        Load tools based on classification and query.

        Args:
            classification: Classification result with tool_categories
            query: Original user query (for semantic search)
            max_tools: Maximum tools to return

        Returns:
            ToolLoadingResult with loaded tools and metadata
        """
        start_time = datetime.now()
        total_tools = len(self._tool_metadata)

        self.logger.info(
            f"[SMART_LOADER] Loading tools: total={total_tools}, "
            f"categories={classification.tool_categories}, "
            f"threshold={LOAD_ALL_THRESHOLD}"
        )

        # ================================================================
        # STRATEGY 1: Load ALL if small registry
        # ================================================================
        if total_tools <= LOAD_ALL_THRESHOLD:
            tools = self._get_all_tools_openai_format()
            load_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(
                f"[SMART_LOADER] Strategy=ALL: loaded {len(tools)} tools "
                f"in {load_time:.1f}ms"
            )

            return ToolLoadingResult(
                tools=tools,
                loading_method=LoadingMethod.ALL,
                total_available=total_tools,
                loaded_count=len(tools),
                load_time_ms=load_time,
            )

        # ================================================================
        # STRATEGY 2: Filter by categories
        # ================================================================
        categories = classification.tool_categories or []

        if not categories:
            # No categories specified - use semantic search on all
            return await self._semantic_search(
                query=query,
                tool_names=list(self._tool_metadata.keys()),
                max_tools=max_tools,
                start_time=start_time,
            )

        # Get tools by categories
        filtered_tools = self._get_tools_by_categories(categories)

        self.logger.info(
            f"[SMART_LOADER] Category filter: {len(filtered_tools)} tools "
            f"from categories {categories}"
        )

        # If within limit, return category-filtered
        if len(filtered_tools) <= max_tools:
            tools = self._convert_to_openai_format(filtered_tools)
            load_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(
                f"[SMART_LOADER] Strategy=CATEGORY: loaded {len(tools)} tools "
                f"in {load_time:.1f}ms"
            )

            return ToolLoadingResult(
                tools=tools,
                loading_method=LoadingMethod.CATEGORY,
                total_available=total_tools,
                loaded_count=len(tools),
                categories_used=categories,
                load_time_ms=load_time,
            )

        # ================================================================
        # STRATEGY 3: HYBRID - Category + Semantic ranking
        # ================================================================
        return await self._semantic_search(
            query=query,
            tool_names=filtered_tools,
            max_tools=max_tools,
            start_time=start_time,
            categories_used=categories,
        )

    async def _semantic_search(
        self,
        query: str,
        tool_names: List[str],
        max_tools: int,
        start_time: datetime,
        categories_used: Optional[List[str]] = None,
    ) -> ToolLoadingResult:
        """
        Rank tools by semantic similarity to query.

        Uses cosine similarity between query embedding and tool embeddings.
        """
        self.logger.info(
            f"[SMART_LOADER] Semantic search: ranking {len(tool_names)} tools"
        )

        try:
            # Ensure embeddings are cached
            if not self._embedding_cache.is_valid():
                self.logger.info("[SMART_LOADER] Building embedding cache...")
                self._embedding_cache.build_cache(self._tool_metadata)

            # Compute query embedding
            query_embedding = self._embedding_cache.compute_query_embedding(query)

            # Compute similarities
            scores: List[Tuple[str, float]] = []

            for tool_name in tool_names:
                tool_embedding = self._embedding_cache.get_embedding(tool_name)
                if tool_embedding is not None:
                    # Cosine similarity
                    similarity = self._cosine_similarity(query_embedding, tool_embedding)
                    scores.append((tool_name, similarity))

            # Sort by score descending
            scores.sort(key=lambda x: x[1], reverse=True)

            # Take top K
            top_tools = [name for name, _ in scores[:max_tools]]
            score_dict = {name: score for name, score in scores[:max_tools]}

            # Convert to OpenAI format
            tools = self._convert_to_openai_format(top_tools)
            load_time = (datetime.now() - start_time).total_seconds() * 1000

            # Log top tools
            top_3 = scores[:3]
            self.logger.info(
                f"[SMART_LOADER] Strategy=SEMANTIC: loaded {len(tools)} tools "
                f"in {load_time:.1f}ms | "
                f"Top 3: {[(n, f'{s:.3f}') for n, s in top_3]}"
            )

            return ToolLoadingResult(
                tools=tools,
                loading_method=LoadingMethod.HYBRID if categories_used else LoadingMethod.SEMANTIC,
                total_available=len(self._tool_metadata),
                loaded_count=len(tools),
                categories_used=categories_used or [],
                semantic_scores=score_dict,
                load_time_ms=load_time,
            )

        except Exception as e:
            self.logger.warning(
                f"[SMART_LOADER] Semantic search failed: {e}, "
                f"falling back to category filter"
            )

            # Fallback: just take first max_tools
            tools = self._convert_to_openai_format(tool_names[:max_tools])
            load_time = (datetime.now() - start_time).total_seconds() * 1000

            return ToolLoadingResult(
                tools=tools,
                loading_method=LoadingMethod.CATEGORY,
                total_available=len(self._tool_metadata),
                loaded_count=len(tools),
                categories_used=categories_used or [],
                load_time_ms=load_time,
            )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _get_tools_by_categories(self, categories: List[str]) -> List[str]:
        """Get tool names by categories (union)"""
        tool_names = []
        seen = set()

        for category in categories:
            for tool_name in self._category_index.get(category, []):
                if tool_name not in seen:
                    tool_names.append(tool_name)
                    seen.add(tool_name)

        return tool_names

    def _get_all_tools_openai_format(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function format"""
        return [
            self._tool_metadata[name]["schema"].to_openai_function()
            for name in self._tool_metadata
        ]

    def _convert_to_openai_format(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """Convert tool names to OpenAI function format"""
        return [
            self._tool_metadata[name]["schema"].to_openai_function()
            for name in tool_names
            if name in self._tool_metadata
        ]

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def get_available_categories(self) -> List[str]:
        """Get list of all available categories"""
        return list(self._category_index.keys())

    def get_tool_count(self) -> int:
        """Get total number of tools"""
        return len(self._tool_metadata)

    def get_category_counts(self) -> Dict[str, int]:
        """Get tool count per category"""
        return {cat: len(tools) for cat, tools in self._category_index.items()}

    def refresh_metadata(self) -> None:
        """Refresh tool metadata (call after tools added/removed)"""
        self._tool_metadata.clear()
        self._category_index.clear()
        self._embedding_cache.clear()
        self._build_metadata()

        self.logger.info(
            f"[SMART_LOADER] Refreshed: {len(self._tool_metadata)} tools"
        )


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_loader_instance: Optional[SmartToolLoader] = None


def get_smart_tool_loader() -> SmartToolLoader:
    """Get singleton SmartToolLoader instance"""
    global _loader_instance

    if _loader_instance is None:
        _loader_instance = SmartToolLoader()

    return _loader_instance


def reset_smart_tool_loader() -> None:
    """Reset singleton instance (for testing)"""
    global _loader_instance
    if _loader_instance is not None:
        _loader_instance._embedding_cache.clear()
    _loader_instance = None