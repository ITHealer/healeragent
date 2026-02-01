"""
Mode configuration and request schemas for the invest_agent module.

Why: Centralizes all mode-related settings so that the orchestrator and executor
can adapt behavior (model, max_turns, features) based on the resolved mode.
Pydantic models ensure request validation at the API boundary.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class AgentMode(str, Enum):
    """Three operating modes for the invest agent.

    - INSTANT: Single-pass, lightweight model, no evaluation loop.
    - THINKING: Multi-turn state machine with evaluation and planning.
    - AUTO: Dynamically routes to INSTANT or THINKING based on query analysis.
    """
    INSTANT = "instant"
    THINKING = "thinking"
    AUTO = "auto"


class ModeConfig(BaseModel):
    """Runtime configuration resolved per-mode.

    Why: Different modes require different models, turn limits, and feature flags.
    The orchestrator reads this config to adapt its behavior without branching logic
    scattered across multiple files.
    """
    mode: AgentMode
    model_name: str = Field(description="Primary LLM model for this mode")
    fallback_model: str = Field(description="Fallback model if primary fails")
    max_turns: int = Field(ge=1, le=10, description="Maximum agent loop iterations")
    use_tools: bool = Field(default=True, description="Whether tool calling is enabled")
    enable_evaluation: bool = Field(
        default=False,
        description="Whether to run data-sufficiency evaluation after tool execution"
    )
    enable_thinking_display: bool = Field(
        default=False,
        description="Whether to emit thinking timeline SSE events"
    )
    enable_web_search: bool = Field(default=False)
    enable_tool_search: bool = Field(
        default=False,
        description="Dynamic tool discovery vs direct tool names"
    )
    max_history_messages: int = Field(
        default=10,
        description="How many conversation history messages to load"
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, ge=100)
    context_offload_threshold: int = Field(
        default=2000,
        description="Character count above which tool results are offloaded to filesystem"
    )


# Pre-defined mode configurations
INSTANT_MODE_CONFIG = ModeConfig(
    mode=AgentMode.INSTANT,
    model_name="gpt-4.1-nano",
    fallback_model="gpt-4o-mini",
    max_turns=2,
    use_tools=True,
    enable_evaluation=False,
    enable_thinking_display=False,
    enable_web_search=False,
    enable_tool_search=False,
    max_history_messages=5,
    temperature=0.1,
    max_tokens=4000,
    context_offload_threshold=2000,
)

THINKING_MODE_CONFIG = ModeConfig(
    mode=AgentMode.THINKING,
    model_name="gpt-4.1",
    fallback_model="gpt-4o",
    max_turns=6,
    use_tools=True,
    enable_evaluation=True,
    enable_thinking_display=True,
    enable_web_search=True,
    enable_tool_search=True,
    max_history_messages=10,
    temperature=0.2,
    max_tokens=16000,
    context_offload_threshold=2000,
)


class UIContextInput(BaseModel):
    """UI context forwarded from the frontend."""
    active_tab: str = "stock"
    recent_symbols: List[str] = Field(default_factory=list)
    preferred_quote_currency: str = "USD"


class InvestChatRequest(BaseModel):
    """Incoming chat request schema for POST /invest/chat/stream.

    Why Pydantic: Validates at the API boundary so downstream code never
    deals with malformed input. Optional fields have sensible defaults
    matching the 'auto' mode behavior.
    """
    query: str = Field(min_length=1, max_length=5000, description="User question")
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for conversation continuity"
    )
    response_mode: Literal["instant", "thinking", "auto"] = Field(
        default="auto",
        description="Explicit mode selection or auto-routing"
    )
    enable_thinking: bool = Field(
        default=True,
        description="Legacy flag: if False, forces instant mode"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Override model selection. If None, mode config decides."
    )
    provider_type: str = Field(default="openai")
    ui_context: Optional[UIContextInput] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    user_id: Optional[int] = None
