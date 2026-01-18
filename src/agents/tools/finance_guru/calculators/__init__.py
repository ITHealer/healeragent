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
- risk_calculator.py: Risk metrics (VaR, Sharpe, Sortino, etc.)
- valuation_calculator.py: Valuation methods (DCF, Graham, DDM)
- portfolio_calculator.py: Portfolio analysis (correlation, rebalancing)
- technical_calculator.py: Technical indicators

Author: HealerAgent Development Team
"""

from src.agents.tools.finance_guru.calculators.base import (
    BaseCalculator,
    CalculationContext,
    CalculationError,
)

__all__ = [
    "BaseCalculator",
    "CalculationContext",
    "CalculationError",
]
