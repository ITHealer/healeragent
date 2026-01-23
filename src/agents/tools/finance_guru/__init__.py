"""
Finance Guru - Quantitative Analysis Module for HealerAgent

This module provides advanced financial computation capabilities for the AI chatbot,
following a 3-layer architecture:

    Layer 1: Models (Pydantic) - Data validation and type safety
    Layer 2: Calculators - Pure computation functions
    Layer 3: Tools - Agent-callable interfaces

ARCHITECTURE NOTE:
================================================================================
Finance Guru tools are COMPUTATION tools that work with data from existing
DATA RETRIEVAL tools. The Agent Loop (THINK → ACT → OBSERVE) remains unchanged.

Example Flow:
    User: "Tính DCF cho AAPL"

    Turn 1 (THINK): "Need FCF data first"
    Turn 1 (ACT): getCashFlow(symbol="AAPL")  [EXISTING TOOL]
    Turn 1 (OBSERVE): FCF history received

    Turn 2 (THINK): "Now run DCF calculation"
    Turn 2 (ACT): calculateDCF(fcf=[...], growth=0.08)  [FINANCE GURU TOOL]
    Turn 2 (OBSERVE): Intrinsic value calculated

    Turn 3: Generate response

MODULES:
- models/: Pydantic models for input validation and output structure
- calculators/: Pure computation functions (no side effects)
- validators/: Data quality validation
- tools/: Agent-callable tool interfaces

USAGE:
    from src.agents.tools.finance_guru import (
        # Models
        DCFInput, DCFResult, PortfolioPosition,
        # Calculators
        ValuationCalculator, RiskCalculator,
        # Tools
        CalculateDCFTool, AnalyzePortfolioTool,
    )

Author: HealerAgent Development Team
Created: 2025-01-18
Version: 1.0.0
"""

# Version info
__version__ = "1.0.0"
__author__ = "HealerAgent Development Team"

# Lazy imports to avoid circular dependencies
# Models, Calculators, and Tools will be imported when needed

__all__ = [
    # Version
    "__version__",
    "__author__",
]
