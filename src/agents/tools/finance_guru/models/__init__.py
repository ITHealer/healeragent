"""
Finance Guru - Pydantic Models

Layer 1 of the 3-layer architecture: Data validation and type safety.

This module provides validated data structures for all Finance Guru calculations.
Pydantic ensures:
- Type safety at runtime
- Automatic validation on instantiation
- Clear error messages for invalid data
- JSON serialization/deserialization

MODULES:
- base.py: Common base models and types
- price_data.py: Price series and market data models
- valuation.py: Valuation calculation models (DCF, Graham, DDM)
- portfolio.py: Portfolio analysis models
- risk.py: Risk metrics models

Author: HealerAgent Development Team
"""

from src.agents.tools.finance_guru.models.base import (
    # Base classes
    BaseFinanceModel,
    BaseCalculationResult,
    # Common types
    TimeSeries,
    PricePoint,
    FinancialPeriod,
    # Enums
    MarketType,
    Currency,
    SignalType,
    TrendDirection,
    # Validation helpers
    FinanceValidationError,
)

from src.agents.tools.finance_guru.models.price_data import (
    PriceDataInput,
    OHLCVData,
    ReturnsData,
)

__all__ = [
    # Base
    "BaseFinanceModel",
    "BaseCalculationResult",
    # Types
    "TimeSeries",
    "PricePoint",
    "FinancialPeriod",
    # Enums
    "MarketType",
    "Currency",
    "SignalType",
    "TrendDirection",
    # Errors
    "FinanceValidationError",
    # Price Data
    "PriceDataInput",
    "OHLCVData",
    "ReturnsData",
]
