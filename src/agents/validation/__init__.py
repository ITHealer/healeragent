# File: src/agents/validation/__init__.py
"""
Validation Module - Ground Truth Validation

Components:
- ValidationAgent: Deterministic validation without LLM
- Business Rules: Domain-specific validation for financial data
- Error Classification: Determine if errors are retryable

NO LLM calls - fast, deterministic validation (< 20ms)
"""

from .validation_agent import (
    ValidationAgent,
    ValidationResult,
    is_retryable_error,
    calculate_backoff,
    create_quick_validation,
    ErrorType,
    classify_error,
)

__all__ = [
    'ValidationAgent',
    'ValidationResult',
    'is_retryable_error',
    'calculate_backoff',
    'create_quick_validation',
    'ErrorType',
    'classify_error',
]