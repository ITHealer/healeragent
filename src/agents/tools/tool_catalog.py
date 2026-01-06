"""
Tool Catalog - 2-Level Tool Description System

Provides lightweight tool summaries for Router LLM and full schemas for Agent execution.
This enables the ChatGPT-style 2-phase tool selection:
1. Router sees ALL tool summaries (~50 tokens each) → selects relevant tools
2. Agent gets full schemas (~200-400 tokens) for selected tools only

Token Savings:
- Current: 38 tools × 200 tokens = 7,600 tokens
- Proposed: 38 × 50 (router) + 5 × 200 (agent) = 2,900 tokens
- Savings: ~60%
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from functools import lru_cache

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.tools.base import ToolSchema, ToolParameter


@dataclass
class ToolSummary:
    """
    Lightweight tool description for Router LLM (~50 tokens).

    Contains just enough information for the Router to decide
    whether this tool is relevant for the query.
    """
    name: str
    category: str
    one_liner: str  # Single sentence description
    capabilities: List[str]  # 3-5 short bullet points
    typical_use: str  # When to use this tool
    requires_symbol: bool = False

    def to_router_format(self) -> str:
        """Format for Router LLM prompt."""
        caps = " | ".join(self.capabilities[:3])
        return (
            f"• {self.name} [{self.category}]: {self.one_liner}\n"
            f"  Use when: {self.typical_use}\n"
            f"  Capabilities: {caps}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "one_liner": self.one_liner,
            "capabilities": self.capabilities,
            "typical_use": self.typical_use,
            "requires_symbol": self.requires_symbol,
        }


@dataclass
class ToolFullSchema:
    """
    Full tool schema for Agent execution (~200-400 tokens).

    Contains complete information needed to call the tool.
    """
    name: str
    category: str
    description: str
    parameters: List[Dict[str, Any]]
    returns: Dict[str, str]
    examples: List[str] = field(default_factory=list)

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            param_name = param.get("name", "")
            param_type = param.get("type", "string")

            prop = {
                "type": param_type,
                "description": param.get("description", ""),
            }

            # Handle array types - OpenAI requires 'items' property
            if param_type == "array":
                items_type = param.get("items_type", "string")  # Default to string
                if param.get("allowed_values"):
                    prop["items"] = {
                        "type": items_type,
                        "enum": param["allowed_values"]
                    }
                else:
                    prop["items"] = {"type": items_type}

            if param.get("enum"):
                prop["enum"] = param["enum"]
            if param.get("default") is not None:
                prop["default"] = param["default"]

            properties[param_name] = prop

            if param.get("required", False):
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "examples": self.examples,
        }


class ToolCatalog(LoggerMixin):
    """
    Central catalog for 2-level tool descriptions.

    Features:
    - Auto-generates summaries from ToolSchema
    - Caches both levels (summaries rarely change)
    - Provides formatted output for Router and Agent

    Usage:
        catalog = ToolCatalog(registry)

        # For Router - get all summaries
        summaries = catalog.get_all_summaries()
        router_prompt = catalog.format_for_router()

        # For Agent - get full schemas for selected tools
        schemas = catalog.get_full_schemas(["getStockPrice", "getTechnicalIndicators"])
    """

    def __init__(self, registry=None):
        """
        Initialize catalog.

        Args:
            registry: ToolRegistry instance (lazy loaded if not provided)
        """
        super().__init__()
        self._registry = registry
        self._summaries_cache: Dict[str, ToolSummary] = {}
        self._schemas_cache: Dict[str, ToolFullSchema] = {}
        self._initialized = False

    def _get_registry(self):
        """Lazy load registry."""
        if self._registry is None:
            from src.agents.tools.tool_loader import get_registry
            self._registry = get_registry()
        return self._registry

    def _ensure_initialized(self):
        """Initialize caches from registry."""
        if self._initialized:
            return

        registry = self._get_registry()
        all_schemas = registry.get_all_schemas()

        for tool_name, schema in all_schemas.items():
            # Generate summary
            summary = self._schema_to_summary(schema)
            self._summaries_cache[tool_name] = summary

            # Generate full schema
            full_schema = self._schema_to_full(schema)
            self._schemas_cache[tool_name] = full_schema

        self._initialized = True
        self.logger.info(
            f"[CATALOG] Initialized with {len(self._summaries_cache)} tools"
        )

    def _schema_to_summary(self, schema: ToolSchema) -> ToolSummary:
        """Convert ToolSchema to lightweight summary."""
        # Extract one-liner from description (first sentence)
        desc = schema.description or ""
        one_liner = desc.split(".")[0] + "." if "." in desc else desc
        one_liner = one_liner[:100]  # Truncate if too long

        # Extract capabilities (first 3)
        capabilities = []
        if schema.capabilities:
            for cap in schema.capabilities[:3]:
                # Remove emoji prefixes like "✅ "
                clean = cap.lstrip("✅❌⚠️ ")
                capabilities.append(clean[:50])  # Truncate

        # Generate typical use from usage_hints
        typical_use = ""
        if schema.usage_hints:
            # Find a "USE THIS" hint
            for hint in schema.usage_hints:
                if "USE THIS" in hint:
                    # Extract the condition
                    parts = hint.split("→")
                    if len(parts) > 0:
                        typical_use = parts[0].replace("User asks:", "").strip()
                        typical_use = typical_use[:80]
                        break

        if not typical_use:
            typical_use = f"Query about {schema.category}"

        return ToolSummary(
            name=schema.name,
            category=schema.category,
            one_liner=one_liner,
            capabilities=capabilities,
            typical_use=typical_use,
            requires_symbol=getattr(schema, 'requires_symbol', False),
        )

    def _schema_to_full(self, schema: ToolSchema) -> ToolFullSchema:
        """Convert ToolSchema to full schema."""
        # Convert parameters
        params = []
        if schema.parameters:
            for param in schema.parameters:
                param_dict = {
                    "name": param.name,
                    "type": param.type,
                    "description": param.description,
                    "required": param.required,
                    "enum": param.enum if hasattr(param, 'enum') else None,
                }
                # Include array-specific properties for OpenAI schema generation
                if param.type == "array":
                    param_dict["items_type"] = getattr(param, 'items_type', 'string')
                    if hasattr(param, 'allowed_values') and param.allowed_values:
                        param_dict["allowed_values"] = param.allowed_values
                # Include default value if present
                if hasattr(param, 'default') and param.default is not None:
                    param_dict["default"] = param.default
                params.append(param_dict)

        # Extract examples from usage_hints
        examples = []
        if schema.usage_hints:
            for hint in schema.usage_hints[:2]:
                if "→" in hint:
                    examples.append(hint[:100])

        return ToolFullSchema(
            name=schema.name,
            category=schema.category,
            description=schema.description,
            parameters=params,
            returns=schema.returns or {},
            examples=examples,
        )

    # =========================================================================
    # PUBLIC API - For Router
    # =========================================================================

    def get_all_summaries(self) -> Dict[str, ToolSummary]:
        """Get all tool summaries for Router."""
        self._ensure_initialized()
        return self._summaries_cache.copy()

    def get_summaries_by_category(self, category: str) -> Dict[str, ToolSummary]:
        """Get summaries for a specific category."""
        self._ensure_initialized()
        return {
            name: summary
            for name, summary in self._summaries_cache.items()
            if summary.category == category
        }

    def format_for_router(self, max_tools: int = None) -> str:
        """
        Format all tool summaries for Router LLM prompt.

        Args:
            max_tools: Optional limit on number of tools

        Returns:
            Formatted string for Router prompt
        """
        self._ensure_initialized()

        lines = ["<tool_catalog>"]

        # Group by category for better organization
        by_category: Dict[str, List[ToolSummary]] = {}
        for summary in self._summaries_cache.values():
            cat = summary.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(summary)

        tool_count = 0
        for category, summaries in sorted(by_category.items()):
            lines.append(f"\n## {category.upper()} TOOLS")
            for summary in summaries:
                if max_tools and tool_count >= max_tools:
                    break
                lines.append(summary.to_router_format())
                tool_count += 1

            if max_tools and tool_count >= max_tools:
                break

        lines.append("\n</tool_catalog>")

        return "\n".join(lines)

    def get_tool_names(self) -> List[str]:
        """Get all available tool names."""
        self._ensure_initialized()
        return list(self._summaries_cache.keys())

    def get_categories(self) -> List[str]:
        """Get all available categories."""
        self._ensure_initialized()
        categories = set()
        for summary in self._summaries_cache.values():
            categories.add(summary.category)
        return sorted(categories)

    # =========================================================================
    # PUBLIC API - For Agent
    # =========================================================================

    def get_full_schema(self, tool_name: str) -> Optional[ToolFullSchema]:
        """Get full schema for a single tool."""
        self._ensure_initialized()
        return self._schemas_cache.get(tool_name)

    def get_full_schemas(self, tool_names: List[str]) -> Dict[str, ToolFullSchema]:
        """
        Get full schemas for selected tools only.

        This is the key optimization - only load full schemas for tools
        that the Router has selected.

        Args:
            tool_names: List of tool names selected by Router

        Returns:
            Dict of tool_name -> ToolFullSchema
        """
        self._ensure_initialized()

        result = {}
        for name in tool_names:
            if name in self._schemas_cache:
                result[name] = self._schemas_cache[name]
            else:
                self.logger.warning(f"[CATALOG] Tool not found: {name}")

        return result

    def get_openai_functions(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        """
        Get OpenAI function definitions for selected tools.

        Args:
            tool_names: List of tool names selected by Router

        Returns:
            List of OpenAI function definitions
        """
        schemas = self.get_full_schemas(tool_names)
        return [schema.to_openai_function() for schema in schemas.values()]

    # =========================================================================
    # STATS & DEBUG
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        self._ensure_initialized()

        categories = {}
        for summary in self._summaries_cache.values():
            cat = summary.category
            categories[cat] = categories.get(cat, 0) + 1

        # Estimate token counts
        router_text = self.format_for_router()
        router_tokens = len(router_text.split()) * 1.3  # Rough estimate

        return {
            "total_tools": len(self._summaries_cache),
            "categories": categories,
            "estimated_router_tokens": int(router_tokens),
            "avg_summary_tokens": int(router_tokens / max(len(self._summaries_cache), 1)),
        }

    def refresh(self):
        """Refresh caches from registry."""
        self._initialized = False
        self._summaries_cache.clear()
        self._schemas_cache.clear()
        self._ensure_initialized()


# Singleton instance
_catalog_instance: Optional[ToolCatalog] = None


def get_tool_catalog() -> ToolCatalog:
    """Get singleton ToolCatalog instance."""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = ToolCatalog()
    return _catalog_instance


def reset_catalog():
    """Reset singleton (for testing)."""
    global _catalog_instance
    _catalog_instance = None


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_catalog():
        print("=" * 60)
        print("Testing Tool Catalog")
        print("=" * 60)

        catalog = get_tool_catalog()

        # Get stats
        stats = catalog.get_stats()
        print(f"\nStats: {json.dumps(stats, indent=2)}")

        # Get all summaries
        summaries = catalog.get_all_summaries()
        print(f"\nTotal summaries: {len(summaries)}")

        # Print first 3 summaries
        print("\nSample summaries:")
        for i, (name, summary) in enumerate(summaries.items()):
            if i >= 3:
                break
            print(f"\n{summary.to_router_format()}")

        # Get full schemas for selected tools
        selected = ["getStockPrice", "getTechnicalIndicators", "getNews"]
        schemas = catalog.get_full_schemas(selected)
        print(f"\nFull schemas for {selected}: {len(schemas)}")

        # Get OpenAI functions
        functions = catalog.get_openai_functions(selected)
        print(f"\nOpenAI functions: {len(functions)}")

        # Format for router
        router_text = catalog.format_for_router(max_tools=5)
        print(f"\nRouter format (first 5 tools):\n{router_text[:500]}...")

    asyncio.run(test_catalog())
