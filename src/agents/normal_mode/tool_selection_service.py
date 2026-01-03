from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.tools.registry import ToolRegistry
from src.agents.tools.tool_loader import get_registry


class DetailLevel(str, Enum):
    """Token cost levels for tool information."""
    NAME_ONLY = "name_only"  # ~10 tokens
    SUMMARY = "summary"      # ~50-100 tokens
    FULL = "full"            # ~200-400 tokens


@dataclass
class ToolSummary:
    """Lightweight tool info for selection phase."""
    name: str
    category: str
    description: str
    requires_symbol: bool
    required_params: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "requires_symbol": self.requires_symbol,
            "params": self.required_params,
        }

    def to_text(self) -> str:
        """Format for prompt inclusion."""
        params = ", ".join(self.required_params) if self.required_params else "none"
        return f"- {self.name} ({self.category}): {self.description} [params: {params}]"


class ToolSelectionService(LoggerMixin):
    """
    Two-level tool loading: summaries for selection, full schemas for execution.
    Uses singleton pattern with lazy initialization.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Ensure single instance (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, registry: Optional[ToolRegistry] = None):
        if self._initialized:
            return
        super().__init__()

        self.registry = registry or get_registry()

        # Caches: tool_name -> ToolSummary, category -> [tool_names]
        self._summary_cache: Dict[str, ToolSummary] = {}
        self._category_cache: Dict[str, List[str]] = {}

        self._build_caches()
        self._initialized = True

        self.logger.info(f"[TOOLS:INIT] Loaded {len(self._summary_cache)} tools")

    def _build_caches(self):
        """Build summary and category index caches from registry."""
        all_schemas = self.registry.get_all_schemas()

        for tool_name, schema in all_schemas.items():
            # Truncate description to first sentence or 100 chars
            short_desc = schema.description
            if ". " in short_desc:
                short_desc = short_desc.split(". ")[0] + "."
            if len(short_desc) > 100:
                short_desc = short_desc[:97] + "..."

            # Create lightweight summary
            self._summary_cache[tool_name] = ToolSummary(
                name=tool_name,
                category=schema.category,
                description=short_desc,
                requires_symbol=schema.requires_symbol,
                required_params=[p.name for p in schema.parameters if p.required],
            )

            # Index by category for fast filtering
            if schema.category not in self._category_cache:
                self._category_cache[schema.category] = []
            self._category_cache[schema.category].append(tool_name)

    # -------------------------------------------------------------------------
    # LEVEL 1: Summaries for tool selection (~50-100 tokens/tool)
    # -------------------------------------------------------------------------

    def get_tool_summaries(
        self,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[ToolSummary]:
        """Get lightweight summaries, optionally filtered by categories."""
        if categories:
            # Collect tools from specified categories, deduplicate
            tool_names = []
            for cat in categories:
                tool_names.extend(self._category_cache.get(cat, []))
            tool_names = list(dict.fromkeys(tool_names))
        else:
            tool_names = list(self._summary_cache.keys())

        summaries = [self._summary_cache[name] for name in tool_names]
        return summaries[:limit] if limit else summaries

    def get_summaries_as_text(
        self,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> str:
        """Format summaries as text grouped by category (for prompt injection)."""
        summaries = self.get_tool_summaries(categories=categories, limit=limit)

        lines = []
        current_category = None

        # Group by category for readability
        for summary in sorted(summaries, key=lambda s: (s.category, s.name)):
            if summary.category != current_category:
                if current_category is not None:
                    lines.append("")
                lines.append(f"[{summary.category.upper()}]")
                current_category = summary.category
            lines.append(summary.to_text())

        return "\n".join(lines)

    def get_summaries_as_dict(
        self,
        categories: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get summaries as dictionaries for JSON serialization."""
        summaries = self.get_tool_summaries(categories=categories, limit=limit)
        return [s.to_dict() for s in summaries]

    # -------------------------------------------------------------------------
    # LEVEL 2: Full schemas for execution (~200-400 tokens/tool)
    # -------------------------------------------------------------------------

    def get_tools_for_execution(
        self,
        tool_names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get full OpenAI function schemas for LLM tool execution.
        Priority: tool_names > categories > all tools.
        """
        # Priority 1: Specific tools by name
        if tool_names:
            result = []
            for name in tool_names:
                schema = self.registry.get_schema(name)
                if schema:
                    result.append(schema.to_openai_function())
            return result

        # Priority 2: Tools by category (deduplicated)
        if categories:
            result = []
            seen = set()
            for cat in categories:
                tools = self.registry.get_tools_by_category(cat)
                for tool_name, tool in tools.items():
                    if tool_name not in seen:
                        result.append(tool.get_schema().to_openai_function())
                        seen.add(tool_name)
            return result

        # Default: All tools
        all_schemas = self.registry.get_all_schemas()
        return [schema.to_openai_function() for schema in all_schemas.values()]

    def get_single_tool_for_execution(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get single tool's OpenAI function schema."""
        schema = self.registry.get_schema(tool_name)
        return schema.to_openai_function() if schema else None

    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------

    def get_available_categories(self) -> List[str]:
        """List all tool categories."""
        return list(self._category_cache.keys())

    def get_tools_in_category(self, category: str) -> List[str]:
        """List tool names in a category."""
        return self._category_cache.get(category, [])

    def get_category_tool_count(self) -> Dict[str, int]:
        """Get tool count per category."""
        return {cat: len(tools) for cat, tools in self._category_cache.items()}

    def tool_exists(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._summary_cache

    def get_tool_category(self, tool_name: str) -> Optional[str]:
        """Get category of a tool."""
        summary = self._summary_cache.get(tool_name)
        return summary.category if summary else None

    def estimate_token_cost(
        self,
        categories: Optional[List[str]] = None,
        detail_level: DetailLevel = DetailLevel.SUMMARY,
    ) -> int:
        """
        Estimate token cost for tool definitions.
        NAME_ONLY: ~10, SUMMARY: ~75, FULL: ~300 tokens per tool.
        """
        count = len(self.get_tool_summaries(categories=categories))

        token_map = {
            DetailLevel.NAME_ONLY: 10,
            DetailLevel.SUMMARY: 75,
            DetailLevel.FULL: 300,
        }
        return count * token_map.get(detail_level, 75)

    def refresh_cache(self):
        """Rebuild caches after registry changes."""
        self._summary_cache.clear()
        self._category_cache.clear()
        self._build_caches()
        self.logger.info(f"[TOOLS:REFRESH] {len(self._summary_cache)} tools cached")


# Singleton instance
_service_instance: Optional[ToolSelectionService] = None


def get_tool_selection_service() -> ToolSelectionService:
    """Get or create singleton instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ToolSelectionService()
    return _service_instance


def reset_tool_selection_service():
    """Reset singleton for testing."""
    global _service_instance
    _service_instance = None