"""
Tool call validator with fuzzy name matching and schema validation.

Why: LLMs frequently hallucinate tool names (e.g., "get_stock_price" instead of
"getStockPrice") or pass malformed arguments. This validator sits between the
LLM output and the executor, auto-correcting names and validating arguments
*before* any tool is invoked. This prevents wasted API calls and cryptic errors.

How: Uses difflib for fuzzy string matching against the registry's known tool
names. Validates arguments against each tool's Pydantic-based ToolSchema.
"""

import json
import logging
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.agents.tools import ToolRegistry, get_registry, ToolSchema
from src.invest_agent.core.exceptions import ToolValidationError

logger = logging.getLogger(__name__)

# Similarity threshold for fuzzy matching. 0.6 is lenient enough to catch
# common LLM errors (snake_case vs camelCase) while avoiding false positives.
FUZZY_MATCH_CUTOFF = 0.55


class ValidatedToolCall(BaseModel):
    """A tool call that has passed name resolution and argument validation."""
    id: str = ""
    original_name: str = Field(description="The raw name from LLM output")
    resolved_name: str = Field(description="The matched registry name")
    arguments: Dict[str, Any] = Field(default_factory=dict)
    was_fuzzy_matched: bool = False


class ToolCallValidator:
    """Validates and auto-corrects LLM-generated tool calls.

    Why this class: The LLM is unreliable in tool name formatting. Without
    validation, we'd either crash on unknown tools or silently skip them.
    This validator provides a reliable bridge: it resolves names, validates
    args, and returns clean tool calls ready for the executor.

    How it integrates: The Orchestrator calls `validate_tool_calls()` on every
    raw tool call batch before passing them to ToolExecutor.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self._registry = registry or get_registry()
        self._tool_names: List[str] = []
        self._name_lookup: Dict[str, str] = {}  # lowercase -> actual name
        self._refresh_tool_index()

    def _refresh_tool_index(self) -> None:
        """Build a lookup index from the registry for fast fuzzy matching.

        Why: We build both a lowercase map (for exact case-insensitive matching)
        and keep the original names list (for difflib fuzzy matching).
        """
        all_schemas = self._registry.get_all_schemas()
        self._tool_names = list(all_schemas.keys())
        self._name_lookup = {name.lower(): name for name in self._tool_names}

        # Also index common snake_case variants: getStockPrice -> get_stock_price
        for name in self._tool_names:
            snake = self._camel_to_snake(name)
            self._name_lookup[snake] = name

    @staticmethod
    def _camel_to_snake(name: str) -> str:
        """Convert camelCase to snake_case for reverse matching."""
        import re
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    def resolve_tool_name(self, raw_name: str) -> Optional[str]:
        """Resolve a potentially misspelled tool name to a registry name.

        Resolution order:
        1. Exact match
        2. Case-insensitive exact match
        3. Snake-case to camelCase conversion
        4. Fuzzy match (difflib)

        Returns None if no match found above the threshold.
        """
        # 1. Exact match
        if raw_name in self._name_lookup.values():
            return raw_name

        # 2. Case-insensitive
        lower = raw_name.lower()
        if lower in self._name_lookup:
            return self._name_lookup[lower]

        # 3. Strip common prefixes (e.g., "functions.getStockPrice")
        if "." in raw_name:
            stripped = raw_name.rsplit(".", 1)[-1]
            return self.resolve_tool_name(stripped)

        # 4. Fuzzy match
        matches = get_close_matches(
            raw_name,
            self._tool_names,
            n=1,
            cutoff=FUZZY_MATCH_CUTOFF,
        )
        if matches:
            logger.info(
                f"[ToolValidator] Fuzzy matched '{raw_name}' -> '{matches[0]}'"
            )
            return matches[0]

        # Also try fuzzy against lowercase keys
        matches_lower = get_close_matches(
            lower,
            list(self._name_lookup.keys()),
            n=1,
            cutoff=FUZZY_MATCH_CUTOFF,
        )
        if matches_lower:
            resolved = self._name_lookup[matches_lower[0]]
            logger.info(
                f"[ToolValidator] Fuzzy matched (lowercase) '{raw_name}' -> '{resolved}'"
            )
            return resolved

        return None

    def validate_arguments(
        self,
        tool_name: str,
        raw_args: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Validate and clean tool arguments against the tool's schema.

        Returns:
            (validated_args, warnings) where warnings list non-fatal issues
            like missing optional params or type coercions.

        Why not just let the tool crash: Early validation gives us a chance
        to fix common issues (string-to-int coercion, strip whitespace from
        symbols) before wasting an API call.
        """
        schema: Optional[ToolSchema] = self._registry.get_schema(tool_name)
        if not schema:
            return raw_args, [f"No schema found for tool '{tool_name}', passing args as-is"]

        validated = {}
        warnings = []

        # Build a lookup of expected parameters
        param_lookup = {p.name: p for p in schema.parameters}

        for param_name, param_spec in param_lookup.items():
            if param_name in raw_args:
                value = raw_args[param_name]
                # Type coercion for common mismatches
                value = self._coerce_type(value, param_spec.type, param_name)
                validated[param_name] = value
            elif param_spec.required:
                if param_spec.default is not None:
                    validated[param_name] = param_spec.default
                    warnings.append(
                        f"Missing required param '{param_name}', using default: {param_spec.default}"
                    )
                else:
                    warnings.append(f"Missing required param '{param_name}' with no default")
            # Optional params without values are simply omitted

        # Warn about unexpected params (LLM hallucinated extra args)
        for key in raw_args:
            if key not in param_lookup:
                warnings.append(f"Unexpected param '{key}' not in schema, ignoring")

        return validated, warnings

    @staticmethod
    def _coerce_type(value: Any, expected_type: str, param_name: str) -> Any:
        """Best-effort type coercion for common LLM mistakes."""
        try:
            if expected_type == "string" and not isinstance(value, str):
                return str(value)
            if expected_type in ("number", "integer") and isinstance(value, str):
                return float(value) if expected_type == "number" else int(value)
            if expected_type == "boolean" and isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            if expected_type == "array" and isinstance(value, str):
                # LLM sometimes passes "[\"AAPL\"]" as string
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                except (json.JSONDecodeError, TypeError):
                    pass
        except (ValueError, TypeError):
            pass  # Return original value if coercion fails
        return value

    def validate_tool_calls(
        self,
        raw_calls: List[Dict[str, Any]],
    ) -> Tuple[List[ValidatedToolCall], List[Dict[str, str]]]:
        """Validate a batch of raw tool calls from LLM output.

        Args:
            raw_calls: List of dicts with keys: id, name (or function.name),
                       arguments (dict or JSON string).

        Returns:
            (valid_calls, skipped) where skipped contains reasons for rejection.

        How: For each call, resolve the name, parse arguments, validate schema.
        Invalid calls are skipped with a warning rather than raising exceptions.
        """
        valid = []
        skipped = []

        for call in raw_calls:
            call_id = call.get("id", "")

            # Extract name (handles both flat and nested formats)
            raw_name = call.get("name") or call.get("function", {}).get("name", "")
            if not raw_name:
                skipped.append({"id": call_id, "reason": "No tool name in call"})
                continue

            # Resolve name
            resolved = self.resolve_tool_name(raw_name)
            if not resolved:
                skipped.append({
                    "id": call_id,
                    "name": raw_name,
                    "reason": f"Unknown tool '{raw_name}', no fuzzy match found",
                })
                logger.warning(f"[ToolValidator] Skipping unknown tool: '{raw_name}'")
                continue

            # Parse arguments
            raw_args = call.get("arguments") or call.get("function", {}).get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    skipped.append({
                        "id": call_id,
                        "name": raw_name,
                        "reason": f"Invalid JSON arguments: {raw_args[:100]}",
                    })
                    continue

            # Validate arguments
            validated_args, warnings = self.validate_arguments(resolved, raw_args)
            for w in warnings:
                logger.debug(f"[ToolValidator] {resolved}: {w}")

            valid.append(ValidatedToolCall(
                id=call_id,
                original_name=raw_name,
                resolved_name=resolved,
                arguments=validated_args,
                was_fuzzy_matched=(raw_name != resolved),
            ))

        return valid, skipped
