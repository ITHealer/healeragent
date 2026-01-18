"""
CrossModelToolAdapter - Standardize Tool Definitions for Cross-Model Compatibility

This module provides adapter functions to normalize tool definitions across different
LLM providers (OpenAI, Gemini, Anthropic) to ensure consistent tool calling behavior.

Key Principles (from research):
1. Pass schema constraints INTO tool description (not just JSON schema)
2. Keep descriptions clear, concise, and action-oriented
3. Include inline examples in description
4. Use narrow, focused responsibilities per tool

References:
- Mastra: Reduced error rates from 15% to 3% by passing constraints to instructions
- Google: "Write clear descriptions that explicitly state when to use each tool"
- Gemini silently ignores some JSON schema constraints - must be in description

Usage:
    from src.agents.tools.cross_model_adapter import (
        adapt_tool_for_provider,
        get_cross_model_tools,
        ProviderType
    )

    # Get adapted tools for Gemini
    tools = get_cross_model_tools(registry, provider="gemini")
"""

import re
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from src.utils.logger.custom_logging import LoggerMixin


class ProviderType(str, Enum):
    """Supported LLM providers for tool adaptation."""
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    UNIVERSAL = "universal"  # Cross-model compatible format


@dataclass
class AdaptedToolDefinition:
    """Cross-model compatible tool definition."""
    name: str
    description: str  # Enhanced with constraints
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    def to_gemini_format(self) -> Dict[str, Any]:
        """Convert to Gemini function declaration format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class CrossModelToolAdapter(LoggerMixin):
    """
    Adapts tool definitions for cross-model compatibility.

    Key optimizations:
    1. Embeds parameter constraints into description (Gemini ignores JSON schema constraints)
    2. Adds explicit "WHEN TO USE" section for better tool selection
    3. Standardizes description format across all tools
    4. Keeps descriptions concise (<500 chars for routing, full for execution)
    """

    # Maximum description length for different contexts
    MAX_DESCRIPTION_ROUTING = 300   # For tool selection/routing
    MAX_DESCRIPTION_EXECUTION = 800  # For actual tool execution

    def __init__(self):
        super().__init__()

    def adapt_tool(
        self,
        tool_schema: Dict[str, Any],
        provider: ProviderType = ProviderType.UNIVERSAL,
        context: str = "execution"  # "routing" or "execution"
    ) -> AdaptedToolDefinition:
        """
        Adapt a tool schema for cross-model compatibility.

        Args:
            tool_schema: Original tool schema (from to_json_schema() or to_openai_function())
            provider: Target provider
            context: "routing" for selection, "execution" for calling

        Returns:
            AdaptedToolDefinition with enhanced description
        """
        name = tool_schema.get("name") or tool_schema.get("function", {}).get("name", "unknown")
        original_desc = (
            tool_schema.get("description") or
            tool_schema.get("function", {}).get("description", "")
        )
        parameters = (
            tool_schema.get("parameters") or
            tool_schema.get("function", {}).get("parameters", {})
        )
        metadata = tool_schema.get("metadata", {})

        # Build enhanced description with embedded constraints
        enhanced_desc = self._build_enhanced_description(
            name=name,
            original_desc=original_desc,
            parameters=parameters,
            metadata=metadata,
            context=context,
            provider=provider,
        )

        # Simplify parameters for Gemini (it ignores some constraints anyway)
        adapted_params = self._simplify_parameters(parameters, provider)

        return AdaptedToolDefinition(
            name=name,
            description=enhanced_desc,
            parameters=adapted_params,
            metadata=metadata,
        )

    def _build_enhanced_description(
        self,
        name: str,
        original_desc: str,
        parameters: Dict[str, Any],
        metadata: Dict[str, Any],
        context: str,
        provider: ProviderType,
    ) -> str:
        """
        Build enhanced description with embedded constraints.

        Format (cross-model optimized):
        ```
        [Core function description - 1 sentence]

        WHEN TO USE: [Specific triggers]
        PARAMETERS: [Required params with constraints]
        RETURNS: [Key output fields]
        EXAMPLE: [One-liner usage example]
        ```
        """
        # Extract core description (first sentence)
        core_desc = self._extract_core_description(original_desc)

        # Extract usage hints from metadata
        usage_hints = metadata.get("usage_hints", [])
        capabilities = metadata.get("capabilities", [])

        # Build WHEN TO USE section (critical for tool selection)
        when_to_use = self._build_when_to_use(name, usage_hints, capabilities)

        # Build parameter constraints section
        param_section = self._build_parameter_constraints(parameters, provider)

        # Build example (helps all models understand intent)
        example = self._build_example(name, parameters)

        # Assemble enhanced description
        if context == "routing":
            # Shorter version for routing/selection
            max_len = self.MAX_DESCRIPTION_ROUTING
            enhanced = f"{core_desc}\n\nWHEN TO USE: {when_to_use}"
            if len(enhanced) < max_len - 50:
                enhanced += f"\n\nEXAMPLE: {example}"
        else:
            # Full version for execution
            max_len = self.MAX_DESCRIPTION_EXECUTION
            enhanced = f"{core_desc}\n\nWHEN TO USE: {when_to_use}"
            if param_section:
                enhanced += f"\n\nPARAMETERS:\n{param_section}"
            enhanced += f"\n\nEXAMPLE: {example}"

        # Truncate if needed
        if len(enhanced) > max_len:
            enhanced = enhanced[:max_len - 3] + "..."

        return enhanced

    def _extract_core_description(self, description: str) -> str:
        """Extract first sentence as core description."""
        # Remove markdown and special chars
        clean = re.sub(r'\*\*|__|`|#', '', description)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Get first sentence (up to period, question mark, or 150 chars)
        match = re.match(r'^([^.!?\n]+[.!?]?)', clean)
        if match:
            first_sentence = match.group(1).strip()
            if len(first_sentence) > 150:
                first_sentence = first_sentence[:147] + "..."
            return first_sentence

        return clean[:150] + "..." if len(clean) > 150 else clean

    def _build_when_to_use(
        self,
        name: str,
        usage_hints: List[str],
        capabilities: List[str]
    ) -> str:
        """Build WHEN TO USE section from usage hints."""
        triggers = []

        # Extract key triggers from usage hints
        for hint in usage_hints[:3]:  # Max 3 hints
            # Extract the key phrase after "→ USE THIS"
            if "USE THIS" in hint.upper():
                trigger = hint.split("→")[0].strip() if "→" in hint else hint
                trigger = re.sub(r'User asks:?\s*', '', trigger, flags=re.IGNORECASE)
                trigger = trigger.strip("'\"")
                if trigger:
                    triggers.append(trigger)

        # If no triggers from hints, infer from capabilities
        if not triggers and capabilities:
            for cap in capabilities[:2]:
                cap_clean = re.sub(r'^[✅❌]\s*', '', cap)
                triggers.append(cap_clean)

        # Default based on tool name
        if not triggers:
            # Convert camelCase to readable
            readable_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name).lower()
            triggers = [f"User asks about {readable_name}"]

        return " | ".join(triggers[:3])

    def _build_parameter_constraints(
        self,
        parameters: Dict[str, Any],
        provider: ProviderType,
    ) -> str:
        """
        Build parameter constraints section.

        Critical for Gemini which ignores JSON schema constraints.
        """
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        if not properties:
            return ""

        lines = []
        for param_name, param_def in properties.items():
            is_required = param_name in required
            param_type = param_def.get("type", "string")
            description = param_def.get("description", "")

            # Build constraint string
            constraints = []
            if is_required:
                constraints.append("REQUIRED")
            if param_def.get("enum"):
                enum_vals = param_def["enum"][:4]  # Max 4 values
                constraints.append(f"one of: {enum_vals}")
            if param_def.get("pattern"):
                constraints.append(f"format: {param_def['pattern']}")
            if param_def.get("minimum") is not None:
                constraints.append(f"min: {param_def['minimum']}")
            if param_def.get("maximum") is not None:
                constraints.append(f"max: {param_def['maximum']}")
            if param_def.get("default") is not None:
                constraints.append(f"default: {param_def['default']}")

            constraint_str = f" ({', '.join(constraints)})" if constraints else ""

            # Truncate description if too long
            short_desc = description[:80] + "..." if len(description) > 80 else description

            lines.append(f"- {param_name} ({param_type}){constraint_str}: {short_desc}")

        return "\n".join(lines)

    def _build_example(self, name: str, parameters: Dict[str, Any]) -> str:
        """Build a simple usage example."""
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        # Build example args from required params
        example_args = {}
        for param_name in required[:3]:  # Max 3 params in example
            param_def = properties.get(param_name, {})

            # Use example value, enum value, or placeholder
            if param_def.get("examples"):
                example_args[param_name] = param_def["examples"][0]
            elif param_def.get("enum"):
                example_args[param_name] = param_def["enum"][0]
            elif param_def.get("default") is not None:
                example_args[param_name] = param_def["default"]
            else:
                # Generate placeholder based on type
                ptype = param_def.get("type", "string")
                if ptype == "string":
                    if "symbol" in param_name.lower():
                        example_args[param_name] = "AAPL"
                    else:
                        example_args[param_name] = f"<{param_name}>"
                elif ptype in ("number", "integer"):
                    example_args[param_name] = 10
                elif ptype == "boolean":
                    example_args[param_name] = True

        # Format as function call
        args_str = ", ".join(f'{k}="{v}"' if isinstance(v, str) else f'{k}={v}'
                            for k, v in example_args.items())
        return f'{name}({args_str})'

    def _simplify_parameters(
        self,
        parameters: Dict[str, Any],
        provider: ProviderType,
    ) -> Dict[str, Any]:
        """
        Simplify parameters for providers that ignore complex constraints.

        Gemini ignores: minLength, maxLength, minItems, maxItems, pattern (sometimes)
        Keep these in description instead.
        """
        if provider not in (ProviderType.GEMINI, ProviderType.UNIVERSAL):
            return parameters

        simplified = {
            "type": "object",
            "properties": {},
            "required": parameters.get("required", [])
        }

        for name, prop in parameters.get("properties", {}).items():
            # Keep only universally supported fields
            simple_prop = {
                "type": prop.get("type", "string"),
                "description": prop.get("description", ""),
            }

            # Keep enum (well supported)
            if prop.get("enum"):
                simple_prop["enum"] = prop["enum"]

            # Keep items for arrays (required)
            if prop.get("type") == "array" and prop.get("items"):
                simple_prop["items"] = {"type": prop["items"].get("type", "string")}
                if prop["items"].get("enum"):
                    simple_prop["items"]["enum"] = prop["items"]["enum"]

            # Keep default
            if prop.get("default") is not None:
                simple_prop["default"] = prop["default"]

            simplified["properties"][name] = simple_prop

        return simplified

    def adapt_tools_batch(
        self,
        tools: List[Dict[str, Any]],
        provider: ProviderType = ProviderType.UNIVERSAL,
        context: str = "execution",
    ) -> List[Dict[str, Any]]:
        """
        Adapt multiple tools and return in provider-specific format.

        Args:
            tools: List of tool schemas
            provider: Target provider
            context: "routing" or "execution"

        Returns:
            List of adapted tool definitions in provider format
        """
        adapted = []
        for tool in tools:
            try:
                adapted_tool = self.adapt_tool(tool, provider, context)

                if provider == ProviderType.OPENAI:
                    adapted.append(adapted_tool.to_openai_format())
                elif provider == ProviderType.GEMINI:
                    adapted.append(adapted_tool.to_gemini_format())
                else:
                    adapted.append(adapted_tool.to_openai_format())  # Default to OpenAI format

            except Exception as e:
                self.logger.warning(f"Failed to adapt tool: {e}")
                adapted.append(tool)  # Fallback to original

        return adapted


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_cross_model_tools(
    registry,  # ToolRegistry
    provider: str = "universal",
    context: str = "execution",
) -> List[Dict[str, Any]]:
    """
    Get all tools adapted for cross-model compatibility.

    Args:
        registry: ToolRegistry instance
        provider: "openai", "gemini", "anthropic", "universal"
        context: "routing" or "execution"

    Returns:
        List of adapted tool definitions
    """
    adapter = CrossModelToolAdapter()
    provider_type = ProviderType(provider.lower())

    # Get all tool schemas
    tools = registry.get_all_tool_schemas()

    return adapter.adapt_tools_batch(tools, provider_type, context)


def adapt_single_tool(
    tool_schema: Dict[str, Any],
    provider: str = "universal",
) -> Dict[str, Any]:
    """
    Adapt a single tool for cross-model compatibility.

    Args:
        tool_schema: Tool schema dict
        provider: Target provider

    Returns:
        Adapted tool definition
    """
    adapter = CrossModelToolAdapter()
    provider_type = ProviderType(provider.lower())
    adapted = adapter.adapt_tool(tool_schema, provider_type)
    return adapted.to_openai_format()


# ============================================================================
# CROSS-MODEL PROMPT TEMPLATES
# ============================================================================

TOOL_CALLING_SYSTEM_PROMPT = """## Tool Usage Instructions

You have access to specialized tools to gather real-time data. Follow these principles:

### Tool Calling Protocol
1. **ANALYZE FIRST**: Understand what data you need before calling tools
2. **CALL TOOLS**: Use tools to fetch actual data - never fabricate numbers
3. **SYNTHESIZE**: Combine tool results into a coherent response

### Tool Selection Rules
- Read each tool's "WHEN TO USE" section to decide if it's relevant
- Call multiple tools in parallel when they're independent
- If a tool fails, note the limitation and continue with available data

### Parameter Guidelines
- Always provide REQUIRED parameters
- Use valid values from enum lists when specified
- Follow the format patterns in parameter descriptions

### Response Quality
- Cite specific data from tool results
- Acknowledge when data is missing or unavailable
- Match the user's language in your response
"""

TOOL_CALLING_INSTRUCTION_GEMINI = """
IMPORTANT FOR FUNCTION CALLING:
- You MUST call tools to get real data before responding
- Provide all required parameters for each tool
- Use exact parameter values from the allowed options
- Call multiple tools in a single turn when possible
"""

TOOL_CALLING_INSTRUCTION_OPENAI = """
You have tools available. Use them to gather data before responding.
Always call tools first, then synthesize the results.
"""
