"""
Response Mode Configuration

Defines the configuration for different response modes:
- FAST: Quick responses with minimal LLM calls
- AUTO: LLM-based complexity classification (default)
- EXPERT: Full capability with extended reasoning

Based on industry best practices from Claude AI Extended Thinking
and GPT-5 Auto-Routing architectures.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


class ResponseMode(str, Enum):
    """
    Response modes for the chat API

    FAST: Quick responses for simple queries
    AUTO: LLM decides based on query complexity (default)
    EXPERT: Thorough analysis for complex queries
    """
    FAST = "fast"
    AUTO = "auto"
    EXPERT = "expert"


@dataclass
class ModeConfig:
    """
    Configuration for a response mode

    Defines all parameters that differ between FAST, AUTO, and EXPERT modes.
    """
    # Mode identification
    mode: ResponseMode
    display_name: str
    icon: str
    description: str

    # Model configuration
    primary_model: str
    fallback_model: str
    provider_type: str

    # Classifier settings
    use_classifier: bool
    classifier_model: Optional[str] = None
    classifier_timeout_ms: int = 500

    # Tool configuration
    tool_selection: Literal["filtered", "all"] = "all"
    max_tools: int = 31
    tool_categories: List[str] = field(default_factory=list)

    # Agent loop limits
    max_turns: int = 6
    turn_timeout_ms: int = 30000
    total_timeout_ms: int = 120000

    # Feature flags
    enable_web_search: bool = True
    enable_thinking_display: bool = True
    enable_tool_search: bool = True
    enable_finance_guru: bool = True

    # System prompt configuration
    system_prompt_version: Literal["condensed", "full"] = "full"
    include_examples: bool = True
    max_system_tokens: int = 4000

    # Target metrics (for monitoring)
    target_latency_p50_ms: int = 15000
    target_latency_p90_ms: int = 45000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "mode": self.mode.value,
            "display_name": self.display_name,
            "icon": self.icon,
            "description": self.description,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "provider_type": self.provider_type,
            "use_classifier": self.use_classifier,
            "classifier_model": self.classifier_model,
            "tool_selection": self.tool_selection,
            "max_tools": self.max_tools,
            "max_turns": self.max_turns,
            "enable_web_search": self.enable_web_search,
            "enable_thinking_display": self.enable_thinking_display,
            "system_prompt_version": self.system_prompt_version,
            "target_latency_p50_ms": self.target_latency_p50_ms,
            "target_latency_p90_ms": self.target_latency_p90_ms,
        }


# ============================================================================
# PREDEFINED MODE CONFIGURATIONS
# ============================================================================

FAST_MODE_CONFIG = ModeConfig(
    mode=ResponseMode.FAST,
    display_name="Fast",
    icon="⚡",
    description="Respond quicker to act sooner",

    # Model - use smaller, faster model
    primary_model="gpt-4o-mini",
    fallback_model="gemini-2.0-flash",
    provider_type="openai",

    # Classifier - YES for FAST mode to filter tools
    use_classifier=True,
    classifier_model="gpt-4o-mini",
    classifier_timeout_ms=500,

    # Tools - filtered to relevant subset
    tool_selection="filtered",
    max_tools=8,
    tool_categories=["price", "technical", "fundamentals", "memory"],

    # Agent loop - strict limits
    max_turns=2,
    turn_timeout_ms=10000,
    total_timeout_ms=15000,

    # Features - minimal for speed
    enable_web_search=False,
    enable_thinking_display=False,
    enable_tool_search=False,
    enable_finance_guru=False,

    # System prompt - condensed
    system_prompt_version="condensed",
    include_examples=False,
    max_system_tokens=1500,

    # Target metrics
    target_latency_p50_ms=3500,
    target_latency_p90_ms=6000,
)

EXPERT_MODE_CONFIG = ModeConfig(
    mode=ResponseMode.EXPERT,
    display_name="Expert",
    icon="🧠",
    description="Think further to explore deeper",

    # Model - use most capable model
    primary_model="gpt-4o",
    fallback_model="gemini-2.5-pro",
    provider_type="openai",

    # Classifier - NO! Large models self-decide
    use_classifier=False,
    classifier_model=None,

    # Tools - ALL available
    tool_selection="all",
    max_tools=31,
    tool_categories=[],  # Empty = all categories

    # Agent loop - extended limits
    max_turns=6,
    turn_timeout_ms=30000,
    total_timeout_ms=120000,

    # Features - all enabled
    enable_web_search=True,
    enable_thinking_display=True,
    enable_tool_search=True,
    enable_finance_guru=True,

    # System prompt - full with examples
    system_prompt_version="full",
    include_examples=True,
    max_system_tokens=4000,

    # Target metrics
    target_latency_p50_ms=20000,
    target_latency_p90_ms=45000,
)

AUTO_MODE_CONFIG = ModeConfig(
    mode=ResponseMode.AUTO,
    display_name="Auto",
    icon="🔄",
    description="Adapts models to each query",

    # Model - will be overridden based on classification
    primary_model="gpt-4o-mini",  # Default to fast, upgrade if needed
    fallback_model="gemini-2.0-flash",
    provider_type="openai",

    # Classifier - LLM semantic classification
    use_classifier=True,
    classifier_model="gpt-4o-mini",
    classifier_timeout_ms=300,

    # Tools - start filtered, expand if needed
    tool_selection="filtered",
    max_tools=8,
    tool_categories=["price", "technical", "fundamentals", "memory"],

    # Agent loop - moderate limits
    max_turns=4,
    turn_timeout_ms=20000,
    total_timeout_ms=60000,

    # Features - selective
    enable_web_search=False,  # Enable only if routed to EXPERT
    enable_thinking_display=False,
    enable_tool_search=False,
    enable_finance_guru=False,

    # System prompt - start condensed
    system_prompt_version="condensed",
    include_examples=False,
    max_system_tokens=2000,

    # Target metrics (variable)
    target_latency_p50_ms=8000,
    target_latency_p90_ms=25000,
)


# Mode lookup dictionary
MODE_CONFIGS: Dict[ResponseMode, ModeConfig] = {
    ResponseMode.FAST: FAST_MODE_CONFIG,
    ResponseMode.EXPERT: EXPERT_MODE_CONFIG,
    ResponseMode.AUTO: AUTO_MODE_CONFIG,
}


def get_mode_config(mode: str) -> ModeConfig:
    """
    Get configuration for a response mode

    Args:
        mode: Response mode string ("fast", "auto", "expert")

    Returns:
        ModeConfig for the specified mode

    Raises:
        ValueError: If mode is not recognized
    """
    try:
        response_mode = ResponseMode(mode.lower())
        return MODE_CONFIGS[response_mode]
    except (ValueError, KeyError):
        # Default to AUTO if invalid mode
        return MODE_CONFIGS[ResponseMode.AUTO]


def get_effective_config(
    base_mode: str,
    classified_complexity: Optional[str] = None
) -> ModeConfig:
    """
    Get effective configuration based on mode and optional classification

    For AUTO mode, this returns FAST or EXPERT config based on classification.
    For explicit modes (FAST/EXPERT), returns that mode's config directly.

    Args:
        base_mode: User-selected response mode
        classified_complexity: Optional complexity from LLM router ("fast" or "expert")

    Returns:
        Effective ModeConfig to use for processing
    """
    mode = ResponseMode(base_mode.lower())

    # Explicit mode selection - use directly
    if mode == ResponseMode.FAST:
        return FAST_MODE_CONFIG
    elif mode == ResponseMode.EXPERT:
        return EXPERT_MODE_CONFIG

    # AUTO mode - route based on classification
    if mode == ResponseMode.AUTO:
        if classified_complexity == "expert":
            return EXPERT_MODE_CONFIG
        else:
            # Default to FAST for AUTO when unsure
            return FAST_MODE_CONFIG

    return FAST_MODE_CONFIG


# ============================================================================
# CLASSIFICATION RESULT
# ============================================================================

@dataclass
class ModeClassificationResult:
    """
    Result from the LLM-based mode router

    Used by ModeRouter to return classification decision.
    """
    effective_mode: ResponseMode
    reason: str
    confidence: float
    detection_method: str  # "llm_semantic", "explicit_user", "context_inheritance"
    query_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effective_mode": self.effective_mode.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "detection_method": self.detection_method,
            "query_features": self.query_features,
        }


# ============================================================================
# FEATURE FLAGS
# ============================================================================

RESPONSE_MODES_FEATURE_FLAGS = {
    "response_modes_enabled": True,      # Master switch
    "fast_mode_enabled": True,           # Enable FAST mode
    "expert_mode_enabled": True,         # Enable EXPERT mode
    "auto_mode_enabled": True,           # Enable AUTO mode
    "llm_router_enabled": True,          # LLM-based routing for AUTO
    "background_memory": True,           # Async memory updates
    "mode_inheritance": True,            # Inherit mode from previous turn
}


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature flag is enabled"""
    return RESPONSE_MODES_FEATURE_FLAGS.get(feature_name, False)
