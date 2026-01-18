"""
Finance Guru - Agent Tools

Layer 3 of the 3-layer architecture: Agent-callable interfaces.

These tools wrap Finance Guru calculators into the standard
HealerAgent tool format (BaseTool), making them callable by
the AI agent during the THINK → ACT → OBSERVE loop.

TOOL CATEGORIES:
- Valuation: calculateDCF, calculateGraham, calculateDDM
- Portfolio: analyzePortfolio, calculateCorrelation, suggestRebalancing
- Risk: calculateSortino, calculateCalmar (enhanced metrics beyond assessRisk)
- Backtest: runBacktest, compareStrategies

USAGE:
    # Tools are registered automatically via tool_loader.py
    # Agent calls them like any other tool:

    Turn 2: ACT
    → calculateDCF(
        ticker="AAPL",
        fcf_history=[99B, 104B, 111B, 99B, 108B],
        growth_rate=0.08,
        discount_rate=0.10
      )

Author: HealerAgent Development Team
"""

# Will be populated as tools are implemented
__all__ = []
