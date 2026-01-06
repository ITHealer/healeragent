"""
Crypto Technical Calculators Module

Pure Python implementations of technical indicators and SMC analysis.
These functions operate on OHLCV data arrays - no API calls.

Usage:
    from src.agents.tools.crypto.calculators import technicals, smc

    # Calculate RSI
    rsi = technicals.calculate_rsi(closes)

    # Detect market structure
    structure = smc.detect_market_structure(ohlcv)
"""

from src.agents.tools.crypto.calculators import technicals
from src.agents.tools.crypto.calculators import smc

__all__ = ["technicals", "smc"]
