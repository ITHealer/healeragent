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
- technical_enhanced.py: Enhanced technical indicator models (Phase 2)
- valuation.py: Valuation calculation models (DCF, Graham, DDM) [Coming]
- portfolio.py: Portfolio analysis models [Coming]
- risk.py: Risk metrics models [Coming]

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

from src.agents.tools.finance_guru.models.technical_enhanced import (
    # Input Model
    OHLCDataInput,
    # Ichimoku Cloud
    IchimokuConfig,
    IchimokuLineOutput,
    IchimokuCloudOutput,
    IchimokuSignals,
    IchimokuOutput,
    # Fibonacci
    FibonacciConfig,
    FibonacciLevel,
    FibonacciOutput,
    # Williams %R
    WilliamsRConfig,
    WilliamsROutput,
    # CCI
    CCIConfig,
    CCIOutput,
    # Parabolic SAR
    ParabolicSARConfig,
    ParabolicSAROutput,
    # Combined
    EnhancedTechnicalOutput,
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
    # Enhanced Technical (Phase 2)
    "OHLCDataInput",
    "IchimokuConfig",
    "IchimokuLineOutput",
    "IchimokuCloudOutput",
    "IchimokuSignals",
    "IchimokuOutput",
    "FibonacciConfig",
    "FibonacciLevel",
    "FibonacciOutput",
    "WilliamsRConfig",
    "WilliamsROutput",
    "CCIConfig",
    "CCIOutput",
    "ParabolicSARConfig",
    "ParabolicSAROutput",
    "EnhancedTechnicalOutput",
]
