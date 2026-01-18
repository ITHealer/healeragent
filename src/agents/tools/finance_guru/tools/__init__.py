"""
Finance Guru - Agent Tools

Layer 3 of the 3-layer architecture: Agent-callable interfaces.

These tools wrap Finance Guru calculators into the standard
HealerAgent tool format (BaseTool), making them callable by
the AI agent during the THINK → ACT → OBSERVE loop.

TOOL CATEGORIES:
- Enhanced Technical (Phase 2): Ichimoku, Fibonacci, Williams %R, CCI, Parabolic SAR
- Enhanced Risk (Phase 3): VaR, Sharpe, Sortino, Calmar, Max Drawdown, Beta/Alpha
- Valuation: calculateDCF, calculateGraham, calculateDDM [Coming]
- Portfolio: analyzePortfolio, calculateCorrelation, suggestRebalancing [Coming]
- Backtest: runBacktest, compareStrategies [Coming]

USAGE:
    # Tools are registered automatically via tool_loader.py
    # Agent calls them like any other tool:

    Turn 1: ACT
    → getIchimokuCloud(symbol="AAPL")

    Turn 2: ACT
    → getRiskMetrics(symbol="AAPL", benchmark_symbol="SPY")

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

from src.agents.tools.finance_guru.tools.risk_metrics import (
    GetRiskMetricsTool,
    GetVaRTool,
    GetSharpeRatioTool,
    GetMaxDrawdownTool,
    GetBetaAlphaTool,
)

from src.agents.tools.finance_guru.tools.portfolio import (
    OptimizePortfolioTool,
    GetCorrelationMatrixTool,
    GetEfficientFrontierTool,
    AnalyzePortfolioDiversificationTool,
    SuggestRebalancingTool,
)

from src.agents.tools.finance_guru.tools.valuation import (
    CalculateDCFTool,
    CalculateGrahamTool,
    CalculateDDMTool,
    GetValuationSummaryTool,
)

from src.agents.tools.finance_guru.tools.backtest import (
    RunBacktestTool,
    CompareStrategiesTool,
)

__all__ = [
    # Enhanced Technical Tools (Phase 2)
    "GetIchimokuCloudTool",
    "GetFibonacciLevelsTool",
    "GetWilliamsRTool",
    "GetCCITool",
    "GetParabolicSARTool",
    "GetEnhancedTechnicalsTool",
    # Enhanced Risk Tools (Phase 3)
    "GetRiskMetricsTool",
    "GetVaRTool",
    "GetSharpeRatioTool",
    "GetMaxDrawdownTool",
    "GetBetaAlphaTool",
    # Portfolio Analysis Tools (Phase 4)
    "OptimizePortfolioTool",
    "GetCorrelationMatrixTool",
    "GetEfficientFrontierTool",
    "AnalyzePortfolioDiversificationTool",
    "SuggestRebalancingTool",
    # Valuation Tools (Phase 1)
    "CalculateDCFTool",
    "CalculateGrahamTool",
    "CalculateDDMTool",
    "GetValuationSummaryTool",
    # Backtest Tools (Phase 5)
    "RunBacktestTool",
    "CompareStrategiesTool",
]
