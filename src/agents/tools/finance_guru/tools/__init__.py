"""
Finance Guru - Agent Tools

Layer 3 of the 3-layer architecture: Agent-callable interfaces.

These tools wrap Finance Guru calculators into the standard
HealerAgent tool format (BaseTool), making them callable by
the AI agent during the THINK → ACT → OBSERVE loop.

TOOL CATEGORIES:
- Enhanced Technical (Phase 2): Ichimoku, Fibonacci, Williams %R, CCI, Parabolic SAR
- Valuation: calculateDCF, calculateGraham, calculateDDM [Coming]
- Portfolio: analyzePortfolio, calculateCorrelation, suggestRebalancing [Coming]
- Risk: calculateSortino, calculateCalmar (enhanced metrics beyond assessRisk) [Coming]
- Backtest: runBacktest, compareStrategies [Coming]

USAGE:
    # Tools are registered automatically via tool_loader.py
    # Agent calls them like any other tool:

    Turn 1: ACT
    → getIchimokuCloud(symbol="AAPL")

    Turn 2: ACT
    → getFibonacciLevels(symbol="AAPL", lookback=50)

Author: HealerAgent Development Team
"""

from src.agents.tools.finance_guru.tools.technical_enhanced import (
    GetIchimokuCloudTool,
    GetFibonacciLevelsTool,
    GetWilliamsRTool,
    GetCCITool,
    GetParabolicSARTool,
    GetEnhancedTechnicalsTool,
)

__all__ = [
    # Enhanced Technical Tools (Phase 2)
    "GetIchimokuCloudTool",
    "GetFibonacciLevelsTool",
    "GetWilliamsRTool",
    "GetCCITool",
    "GetParabolicSARTool",
    "GetEnhancedTechnicalsTool",
]
