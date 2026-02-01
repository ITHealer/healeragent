"""
Custom exceptions for the invest_agent module.

Why: Typed exceptions allow the orchestrator to catch specific failure classes
and decide on recovery strategy (retry, fallback, or graceful degradation)
instead of blanket try/except that hides root causes.
"""


class InvestAgentError(Exception):
    """Base exception for all invest_agent errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ToolExecutionError(InvestAgentError):
    """Raised when a tool fails after all retry attempts.

    The orchestrator catches this and injects a formatted error string
    into the LLM context instead of crashing the agent loop.
    """

    def __init__(self, tool_name: str, message: str, is_transient: bool = False):
        super().__init__(message, {"tool_name": tool_name, "is_transient": is_transient})
        self.tool_name = tool_name
        self.is_transient = is_transient


class ModeResolutionError(InvestAgentError):
    """Raised when mode resolution fails (e.g., LLM classification timeout).

    Recovery: Fall back to INSTANT mode as the safest default.
    """
    pass


class ArtifactStorageError(InvestAgentError):
    """Raised when artifact save/load operations fail.

    Recovery: Keep the full result in-context instead of offloading.
    Non-fatal - the pipeline continues with degraded context efficiency.
    """
    pass


class EvaluationError(InvestAgentError):
    """Raised when the data-sufficiency evaluation step fails.

    Recovery: Skip evaluation and proceed to synthesis with available data.
    """
    pass


class ToolValidationError(InvestAgentError):
    """Raised when tool call validation fails (bad name, invalid args).

    Recovery: Skip the invalid tool call and log a warning.
    """

    def __init__(self, tool_name: str, message: str):
        super().__init__(message, {"tool_name": tool_name})
        self.tool_name = tool_name
