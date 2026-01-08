"""
Tool Search Service - Semantic Tool Discovery for GPT Agent

Provides dynamic tool discovery using semantic embeddings, enabling the agent
to find relevant tools on-demand instead of loading all tools upfront.

Key Features:
- Semantic search using all-MiniLM-L6-v2 embeddings (384 dimensions)
- Pre-computed embeddings with TTL-based caching
- `tool_search` meta-tool for GPT to call during conversation
- Tool reference pattern for efficient token usage

Token Savings:
- Before: ~15,000 tokens (loading 31+ tools × ~500 tokens each)
- After: ~500 tokens (just tool_search) + ~1,500 tokens (top-5 results)
- Savings: ~85%

Usage:
    # Initialize service
    service = get_tool_search_service()

    # Search for tools semantically
    results = await service.search("analyze stock price trends")

    # Get tool definitions for injection
    tool_defs = service.get_tool_definitions(results.tool_names)

Architecture:
    Query → Embedding → Cosine Similarity → Top-K Tools → Schema Injection
"""

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# CONFIGURATION
# ============================================================================

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast, local
EMBEDDING_DIMENSIONS = 384

# Search parameters
DEFAULT_TOP_K = 5
MAX_TOP_K = 15
MIN_SIMILARITY_THRESHOLD = 0.15  # Minimum cosine similarity to include

# Cache settings
EMBEDDING_CACHE_TTL_SECONDS = 3600  # 1 hour

# Tool search meta-tool definition (for GPT)
TOOL_SEARCH_DEFINITION = {
    "type": "function",
    "function": {
        "name": "tool_search",
        "description": """Search for available tools that can help with a financial analysis task.
Use this tool when you need a capability but don't know which specific tool to use.

Returns: List of relevant tool names with descriptions that can be used for the task.

Example queries:
- "analyze stock price and technical indicators"
- "get financial news and sentiment"
- "cryptocurrency price and market data"
- "risk metrics and portfolio analysis"
- "compare multiple stocks"
""",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of what tools you need (e.g., 'analyze NVDA technical indicators', 'get cryptocurrency prices', 'compare stock performance')",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of tools to return (default: 5, max: 15)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ToolSearchResult:
    """Result of a single tool match."""

    name: str
    category: str
    description: str
    similarity_score: float
    requires_symbol: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "similarity_score": round(self.similarity_score, 3),
            "requires_symbol": self.requires_symbol,
        }


@dataclass
class ToolSearchResponse:
    """Complete response from tool search."""

    query: str
    results: List[ToolSearchResult]
    total_tools_searched: int
    search_time_ms: float

    @property
    def tool_names(self) -> List[str]:
        """Get list of tool names."""
        return [r.name for r in self.results]

    @property
    def top_tool(self) -> Optional[str]:
        """Get top matching tool name."""
        return self.results[0].name if self.results else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "tool_names": self.tool_names,
            "total_tools_searched": self.total_tools_searched,
            "search_time_ms": round(self.search_time_ms, 2),
        }

    def format_for_llm(self) -> str:
        """Format results for LLM context injection."""
        if not self.results:
            return "No matching tools found for your query."

        lines = [f"Found {len(self.results)} relevant tools:\n"]

        for i, result in enumerate(self.results, 1):
            symbol_note = " (requires stock symbol)" if result.requires_symbol else ""
            lines.append(
                f"{i}. **{result.name}** [{result.category}]{symbol_note}\n"
                f"   {result.description}\n"
                f"   Relevance: {result.similarity_score:.0%}\n"
            )

        lines.append(
            "\nYou can now use these tools directly. "
            "Specify the tool name and required parameters."
        )

        return "\n".join(lines)


# ============================================================================
# EMBEDDING CACHE
# ============================================================================

class ToolEmbeddingManager(LoggerMixin):
    """
    Manages tool embeddings for semantic search.

    Features:
    - Lazy loading of embedding model
    - Pre-computed tool embeddings with TTL cache
    - Efficient cosine similarity computation
    """

    def __init__(self):
        super().__init__()

        self._model = None
        self._embeddings: Dict[str, np.ndarray] = {}
        self._tool_texts: Dict[str, str] = {}
        self._last_updated: Optional[datetime] = None

    def _get_model(self):
        """Lazy load embedding model."""
        if self._model is None:
            try:
                # Try using existing text_embedding_model from helpers
                from src.helpers.text_embedding_helper import text_embedding_model
                self._model = text_embedding_model
                self.logger.info(
                    f"[TOOL_SEARCH] Using FastEmbed text embedding model"
                )
            except Exception as e:
                # Fallback to SentenceTransformer
                self.logger.warning(
                    f"[TOOL_SEARCH] FastEmbed not available ({e}), "
                    f"falling back to SentenceTransformer"
                )
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(EMBEDDING_MODEL)
                self.logger.info(
                    f"[TOOL_SEARCH] Loaded SentenceTransformer: {EMBEDDING_MODEL}"
                )
        return self._model

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        model = self._get_model()

        if hasattr(model, 'embed'):
            # FastEmbed returns generator
            embeddings = list(model.embed([text]))
            return np.array(embeddings[0])
        else:
            # SentenceTransformer
            return model.encode(text, convert_to_numpy=True)

    def build_tool_embeddings(self, tools: Dict[str, Any]) -> None:
        """
        Build embeddings for all tools.

        Args:
            tools: Dict of tool_name -> tool_info (ToolSchema or dict)
        """
        start_time = datetime.now()

        for tool_name, tool_info in tools.items():
            if tool_name in self._embeddings:
                continue  # Already cached

            # Build searchable text
            text = self._build_tool_text(tool_name, tool_info)
            self._tool_texts[tool_name] = text

            # Compute embedding
            self._embeddings[tool_name] = self._compute_embedding(text)

        self._last_updated = datetime.now()
        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        self.logger.info(
            f"[TOOL_SEARCH] Built embeddings for {len(self._embeddings)} tools "
            f"in {elapsed:.1f}ms"
        )

    def _build_tool_text(self, name: str, info: Any) -> str:
        """Build searchable text representation of tool."""
        parts = [f"Tool: {name}"]

        # Description
        desc = ""
        if hasattr(info, 'description'):
            desc = info.description
        elif isinstance(info, dict):
            desc = info.get('description', '')
        if desc:
            parts.append(f"Description: {desc}")

        # Category
        category = ""
        if hasattr(info, 'category'):
            category = info.category
        elif isinstance(info, dict):
            category = info.get('category', '')
        if category:
            parts.append(f"Category: {category}")

        # Capabilities (if available)
        capabilities = []
        if hasattr(info, 'capabilities'):
            capabilities = info.capabilities or []
        elif isinstance(info, dict):
            capabilities = info.get('capabilities', [])
        if capabilities:
            caps_text = ", ".join(capabilities[:5])
            parts.append(f"Capabilities: {caps_text}")

        # Usage hints (if available)
        hints = []
        if hasattr(info, 'usage_hints'):
            hints = info.usage_hints or []
        elif isinstance(info, dict):
            hints = info.get('usage_hints', [])
        if hints:
            hints_text = "; ".join(hints[:3])
            parts.append(f"Use when: {hints_text}")

        return " | ".join(parts)

    def compute_query_embedding(self, query: str) -> np.ndarray:
        """Compute embedding for search query."""
        return self._compute_embedding(query)

    def get_tool_embedding(self, tool_name: str) -> Optional[np.ndarray]:
        """Get cached embedding for tool."""
        return self._embeddings.get(tool_name)

    def is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._last_updated is None:
            return False
        age = (datetime.now() - self._last_updated).total_seconds()
        return age < EMBEDDING_CACHE_TTL_SECONDS

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embeddings.clear()
        self._tool_texts.clear()
        self._last_updated = None
        self.logger.info("[TOOL_SEARCH] Cleared embedding cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_tools": len(self._embeddings),
            "cache_valid": self.is_cache_valid(),
            "last_updated": self._last_updated.isoformat() if self._last_updated else None,
            "embedding_dimensions": EMBEDDING_DIMENSIONS,
        }


# ============================================================================
# TOOL SEARCH SERVICE
# ============================================================================

class ToolSearchService(LoggerMixin):
    """
    Semantic Tool Search Service for GPT Agent.

    Enables dynamic tool discovery using embeddings, reducing initial context
    from ~15,000 tokens to ~500 tokens while maintaining tool discoverability.

    Usage:
        service = get_tool_search_service()

        # Search for tools
        response = await service.search("analyze stock technical indicators")

        # Get tool definitions for found tools
        tool_defs = service.get_tool_definitions(response.tool_names)

        # Inject into agent's tools list
        agent_tools = [TOOL_SEARCH_DEFINITION] + tool_defs
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        super().__init__()

        self._embedding_manager = ToolEmbeddingManager()
        self._tool_registry = None
        self._tool_catalog = None
        self._tool_metadata: Dict[str, Dict[str, Any]] = {}
        self._category_index: Dict[str, List[str]] = {}

        self._initialized = True
        self.logger.info("[TOOL_SEARCH] ToolSearchService initialized")

    def _ensure_loaded(self) -> None:
        """Ensure tools and embeddings are loaded."""
        if self._tool_metadata:
            return

        # Load from registry
        try:
            from src.agents.tools.tool_loader import get_registry
            self._tool_registry = get_registry()
            all_schemas = self._tool_registry.get_all_schemas()

            for tool_name, schema in all_schemas.items():
                self._tool_metadata[tool_name] = {
                    "name": tool_name,
                    "category": schema.category,
                    "description": schema.description,
                    "requires_symbol": getattr(schema, 'requires_symbol', False),
                    "schema": schema,
                }

                # Build category index
                category = schema.category
                if category not in self._category_index:
                    self._category_index[category] = []
                self._category_index[category].append(tool_name)

            self.logger.info(
                f"[TOOL_SEARCH] Loaded {len(self._tool_metadata)} tools "
                f"from {len(self._category_index)} categories"
            )
        except Exception as e:
            self.logger.error(f"[TOOL_SEARCH] Failed to load registry: {e}")

    def _ensure_embeddings(self) -> None:
        """Ensure embeddings are computed."""
        self._ensure_loaded()

        if not self._embedding_manager.is_cache_valid():
            self.logger.info("[TOOL_SEARCH] Building embedding cache...")
            self._embedding_manager.build_tool_embeddings(self._tool_metadata)

    # ========================================================================
    # PUBLIC API - Search
    # ========================================================================

    async def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        category_filter: Optional[List[str]] = None,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    ) -> ToolSearchResponse:
        """
        Search for tools using semantic similarity.

        Args:
            query: Natural language description of needed capability
            top_k: Number of tools to return (default: 5, max: 15)
            category_filter: Optional list of categories to search within
            min_similarity: Minimum cosine similarity threshold

        Returns:
            ToolSearchResponse with ranked results
        """
        start_time = datetime.now()

        # Validate inputs
        top_k = min(max(1, top_k), MAX_TOP_K)

        # Ensure embeddings are ready
        self._ensure_embeddings()

        # Get query embedding
        query_embedding = self._embedding_manager.compute_query_embedding(query)

        # Determine which tools to search
        if category_filter:
            tool_names = []
            for cat in category_filter:
                tool_names.extend(self._category_index.get(cat, []))
            tool_names = list(set(tool_names))  # Remove duplicates
        else:
            tool_names = list(self._tool_metadata.keys())

        # Compute similarities
        scores: List[Tuple[str, float]] = []

        for tool_name in tool_names:
            tool_embedding = self._embedding_manager.get_tool_embedding(tool_name)
            if tool_embedding is not None:
                similarity = self._cosine_similarity(query_embedding, tool_embedding)
                if similarity >= min_similarity:
                    scores.append((tool_name, similarity))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Build results
        results = []
        for tool_name, score in scores[:top_k]:
            tool_info = self._tool_metadata.get(tool_name, {})
            results.append(ToolSearchResult(
                name=tool_name,
                category=tool_info.get("category", "unknown"),
                description=tool_info.get("description", "")[:200],  # Truncate
                similarity_score=score,
                requires_symbol=tool_info.get("requires_symbol", False),
            ))

        elapsed = (datetime.now() - start_time).total_seconds() * 1000

        # Log results
        if results:
            top_3 = [(r.name, f"{r.similarity_score:.3f}") for r in results[:3]]
            self.logger.info(
                f"[TOOL_SEARCH] Query: '{query[:50]}...' | "
                f"Top 3: {top_3} | {elapsed:.1f}ms"
            )
        else:
            self.logger.warning(
                f"[TOOL_SEARCH] No results for query: '{query[:50]}...'"
            )

        return ToolSearchResponse(
            query=query,
            results=results,
            total_tools_searched=len(tool_names),
            search_time_ms=elapsed,
        )

    def search_sync(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        category_filter: Optional[List[str]] = None,
    ) -> ToolSearchResponse:
        """Synchronous version of search for non-async contexts."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.search(query, top_k, category_filter)
        )

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    # ========================================================================
    # PUBLIC API - Tool Definitions
    # ========================================================================

    def get_tool_definition(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get OpenAI function definition for a single tool.

        Args:
            tool_name: Name of the tool

        Returns:
            OpenAI function definition or None if not found
        """
        self._ensure_loaded()

        tool_info = self._tool_metadata.get(tool_name)
        if not tool_info:
            return None

        schema = tool_info.get("schema")
        if schema and hasattr(schema, 'to_openai_function'):
            return schema.to_openai_function()

        return None

    def get_tool_definitions(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get OpenAI function definitions for multiple tools.

        Args:
            tool_names: List of tool names

        Returns:
            List of OpenAI function definitions
        """
        definitions = []
        for name in tool_names:
            definition = self.get_tool_definition(name)
            if definition:
                definitions.append(definition)
        return definitions

    def get_tool_search_tool(self) -> Dict[str, Any]:
        """Get the tool_search meta-tool definition."""
        return TOOL_SEARCH_DEFINITION.copy()

    # ========================================================================
    # PUBLIC API - Handler for tool_search calls
    # ========================================================================

    async def handle_tool_search_call(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Handle a tool_search function call from GPT.

        This is called when GPT invokes the tool_search tool.
        Returns both a text response and tool definitions for injection.

        Args:
            query: The search query from GPT
            top_k: Number of tools to return

        Returns:
            Tuple of (response_text, tool_definitions)
            - response_text: Human-readable result for GPT
            - tool_definitions: OpenAI function schemas to inject
        """
        # Perform search
        response = await self.search(query, top_k)

        # Get tool definitions for found tools
        tool_defs = self.get_tool_definitions(response.tool_names)

        # Format response for GPT
        text_response = response.format_for_llm()

        self.logger.info(
            f"[TOOL_SEARCH] Handled search: '{query[:40]}...' → "
            f"{len(tool_defs)} tools found"
        )

        return text_response, tool_defs

    # ========================================================================
    # PUBLIC API - Utilities
    # ========================================================================

    def get_all_tool_names(self) -> List[str]:
        """Get all available tool names."""
        self._ensure_loaded()
        return list(self._tool_metadata.keys())

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        self._ensure_loaded()
        return list(self._category_index.keys())

    def get_tools_by_category(self, category: str) -> List[str]:
        """Get tool names in a category."""
        self._ensure_loaded()
        return self._category_index.get(category, []).copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        self._ensure_loaded()

        return {
            "total_tools": len(self._tool_metadata),
            "categories": len(self._category_index),
            "category_counts": {
                cat: len(tools)
                for cat, tools in self._category_index.items()
            },
            "embedding_stats": self._embedding_manager.get_stats(),
            "estimated_token_savings": {
                "before": len(self._tool_metadata) * 400,  # ~400 tokens per tool
                "after": 500 + (DEFAULT_TOP_K * 300),  # tool_search + top-k results
                "savings_percent": round(
                    (1 - (500 + DEFAULT_TOP_K * 300) / (len(self._tool_metadata) * 400)) * 100
                ) if self._tool_metadata else 0,
            },
        }

    def refresh(self) -> None:
        """Refresh tool metadata and embeddings."""
        self._tool_metadata.clear()
        self._category_index.clear()
        self._embedding_manager.clear_cache()
        self._ensure_loaded()
        self._ensure_embeddings()
        self.logger.info("[TOOL_SEARCH] Refreshed tool metadata and embeddings")


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_service_instance: Optional[ToolSearchService] = None


def get_tool_search_service() -> ToolSearchService:
    """Get singleton ToolSearchService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ToolSearchService()
    return _service_instance


def reset_tool_search_service() -> None:
    """Reset singleton instance (for testing)."""
    global _service_instance
    if _service_instance is not None:
        _service_instance._embedding_manager.clear_cache()
    _service_instance = None


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_service():
        print("=" * 60)
        print("Testing Tool Search Service")
        print("=" * 60)

        service = get_tool_search_service()

        # Get stats
        stats = service.get_stats()
        print(f"\nStats: {stats}")

        # Test searches
        test_queries = [
            "analyze stock price and technical indicators",
            "get financial news and sentiment",
            "cryptocurrency price data",
            "risk analysis and portfolio metrics",
        ]

        for query in test_queries:
            print(f"\n--- Query: {query} ---")
            response = await service.search(query, top_k=3)
            print(f"Results: {response.tool_names}")
            for r in response.results:
                print(f"  - {r.name}: {r.similarity_score:.3f}")

        # Test handler
        print("\n--- Testing handler ---")
        text, tool_defs = await service.handle_tool_search_call(
            "analyze NVDA technical indicators"
        )
        print(f"Response:\n{text}")
        print(f"Tool definitions: {len(tool_defs)}")

        # Get tool_search definition
        tool_search = service.get_tool_search_tool()
        print(f"\ntool_search definition: {tool_search['function']['name']}")

    asyncio.run(test_service())
