"""
Finance Guru - Calculators

Layer 2 of the 3-layer architecture: Pure computation functions.

Calculators implement financial calculations with these principles:
1. PURE FUNCTIONS: No side effects, same input always produces same output
2. TYPE SAFETY: All inputs/outputs are validated Pydantic models
3. EDUCATIONAL: Each method includes docstrings explaining the finance concepts
4. TESTABLE: Easy to unit test with known inputs/outputs

MODULES:
- base.py: Base calculator interface and utilities
- technical_enhanced.py: Enhanced technical indicators (Phase 2)
  - Ichimoku Cloud, Fibonacci, Williams %R, CCI, Parabolic SAR
- risk_metrics.py: Enhanced risk metrics (Phase 3)
  - VaR, CVaR, Sharpe, Sortino, Treynor, Calmar, Max Drawdown, Beta/Alpha
- valuation_calculator.py: Valuation methods (DCF, Graham, DDM) [Coming]
- portfolio_calculator.py: Portfolio analysis (correlation, rebalancing) [Coming]

Author: HealerAgent Development Team
"""

from src.agents.tools.finance_guru.calculators.base import (
    BaseCalculator,
    CalculationContext,
    CalculationError,
    InsufficientDataError,
)

from src.agents.tools.finance_guru.calculators.technical_enhanced import (
    # Calculators
    IchimokuCalculator,
    FibonacciCalculator,
    WilliamsRCalculator,
    CCICalculator,
    ParabolicSARCalculator,
    EnhancedTechnicalCalculator,
    # Convenience functions
    calculate_ichimoku,
    calculate_fibonacci,
    calculate_williams_r,
    calculate_cci,
    calculate_parabolic_sar,
    calculate_enhanced_technical,
)

from src.agents.tools.finance_guru.calculators.risk_metrics import (
    RiskMetricsCalculator,
    calculate_risk_metrics,
    calculate_var,
    calculate_sharpe,
)

__all__ = [
    # Base
    "BaseCalculator",
    "CalculationContext",
    "CalculationError",
    "InsufficientDataError",
    # Enhanced Technical Calculators (Phase 2)
    "IchimokuCalculator",
    "FibonacciCalculator",
    "WilliamsRCalculator",
    "CCICalculator",
    "ParabolicSARCalculator",
    "EnhancedTechnicalCalculator",
    # Convenience functions
    "calculate_ichimoku",
    "calculate_fibonacci",
    "calculate_williams_r",
    "calculate_cci",
    "calculate_parabolic_sar",
    "calculate_enhanced_technical",
    # Risk Metrics Calculator (Phase 3)
    "RiskMetricsCalculator",
    "calculate_risk_metrics",
    "calculate_var",
    "calculate_sharpe",
]
