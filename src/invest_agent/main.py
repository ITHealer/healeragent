"""
Entry point and dependency wiring for the invest_agent module.

Why: All component initialization and dependency injection happens here.
This keeps individual modules decoupled (they accept dependencies via
constructor) while providing a single place to configure the full pipeline.

How: Lazy singleton pattern - the orchestrator is created on first access
and reused for all subsequent requests. Dependencies are wired bottom-up:
SessionStore -> ArtifactManager -> ToolExecutor -> Orchestrator.
"""

import logging
from typing import Optional

from src.invest_agent.orchestration.orchestrator import Orchestrator
from src.invest_agent.orchestration.intent_wrapper import IntentWrapper
from src.invest_agent.orchestration.mode_resolver import ModeResolver
from src.invest_agent.execution.validator import ToolCallValidator
from src.invest_agent.execution.tool_executor import ToolExecutor
from src.invest_agent.execution.evaluator import DataEvaluator
from src.invest_agent.storage.session_store import SessionStore
from src.invest_agent.storage.artifact_manager import ArtifactManager

logger = logging.getLogger(__name__)

# Module-level singleton
_orchestrator: Optional[Orchestrator] = None

# Default system prompt for the invest_agent
DEFAULT_SYSTEM_PROMPT = """You are HealerAgent, an expert AI investment analyst assistant.

Your capabilities:
- Real-time stock prices, technical indicators, and chart patterns
- Fundamental analysis: financial statements, ratios, growth metrics
- Risk assessment, sentiment analysis, and volume profiling
- Market overview: indices, sectors, movers, breadth
- Cryptocurrency prices and technicals
- Stock screening and discovery
- Portfolio analysis, valuation models (DCF, Graham, DDM)
- Backtesting and strategy comparison

Instructions:
- Always use tools to get real data. Never fabricate financial numbers.
- When you have the data, provide clear, structured analysis with specific numbers.
- If a tool fails, acknowledge the limitation and work with available data.
- Respond in the same language as the user's query.
- Be concise for simple queries, detailed for complex analysis requests.
"""


def get_orchestrator() -> Orchestrator:
    """Get or create the module-level Orchestrator singleton.

    Why lazy init: The ToolRegistry may not be initialized at import time
    (tools are loaded during app startup). Lazy initialization ensures
    we create the orchestrator only when the first request arrives.
    """
    global _orchestrator

    if _orchestrator is not None:
        return _orchestrator

    logger.info("[InvestAgent] Initializing orchestrator...")

    # Bottom-up dependency construction
    session_store = SessionStore()

    artifact_manager = ArtifactManager(
        session_store=session_store,
        offload_threshold=2000,
    )

    tool_validator = ToolCallValidator()  # Uses get_registry() internally

    tool_executor = ToolExecutor(
        artifact_manager=artifact_manager,
        max_retries=2,
    )

    intent_wrapper = IntentWrapper()
    mode_resolver = ModeResolver()
    evaluator = DataEvaluator(max_eval_iterations=3)

    _orchestrator = Orchestrator(
        intent_wrapper=intent_wrapper,
        mode_resolver=mode_resolver,
        tool_validator=tool_validator,
        tool_executor=tool_executor,
        evaluator=evaluator,
        artifact_manager=artifact_manager,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    logger.info("[InvestAgent] Orchestrator initialized successfully")
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset the orchestrator singleton (useful for testing)."""
    global _orchestrator
    _orchestrator = None
