"""
Deep Research Workers Module

Contains specialized worker agents for different research aspects:
- BaseWorker: Base class for all workers
- MarketAnalystWorker: Market position and trends
- FinancialAnalystWorker: Financial analysis
- TechnicalAnalystWorker: Technical analysis
- WebResearcherWorker: Web search and news
"""

from src.agents.deep_research.workers.base_worker import (
    ResearchWorker,
    WORKER_PROMPTS,
)

__all__ = [
    "ResearchWorker",
    "WORKER_PROMPTS",
]
