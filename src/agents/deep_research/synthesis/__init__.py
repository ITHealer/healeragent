"""
Deep Research Synthesis Module

Contains components for combining worker results into final reports:
- ReportGenerator: Main synthesis engine
- Templates: Report templates
"""

from src.agents.deep_research.synthesis.report_generator import (
    ReportGenerator,
)

__all__ = [
    "ReportGenerator",
]
