"""
Invest Agent Module - V3 Mode System

A modular, self-contained AI chatbot architecture for investment analysis.
Operates independently from src/agents/ while reusing its atomic tools.

Three modes of operation:
- Instant: Fast responses using lightweight models (< 3s)
- Thinking: Deep analysis with evaluation loops (10-30s)
- Auto: Intelligent routing between Instant and Thinking
"""

__version__ = "0.1.0"
