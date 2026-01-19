"""
Finance Guru - Base Calculator Interface

Provides the foundation for all Finance Guru calculators.
All calculators should inherit from BaseCalculator and follow
the established patterns for validation, logging, and error handling.

DESIGN PRINCIPLES:
1. PURE FUNCTIONS: Calculators should not have side effects
2. VALIDATED I/O: All inputs/outputs use Pydantic models
3. ERROR HANDLING: Clear errors with actionable suggestions
4. LOGGING: Consistent logging for debugging and monitoring
5. EDUCATIONAL: Methods explain the financial concepts

Author: HealerAgent Development Team
Created: 2025-01-18
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel

from src.agents.tools.finance_guru.models.base import (
    BaseCalculationResult,
    BaseFinanceModel,
    FinanceValidationError,
)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class CalculationError(Exception):
    """
    Exception raised when a calculation fails.

    Provides context about what went wrong and potential fixes.
    """

    def __init__(
        self,
        message: str,
        calculator: Optional[str] = None,
        method: Optional[str] = None,
        input_summary: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ):
        self.calculator = calculator
        self.method = method
        self.input_summary = input_summary
        self.suggestion = suggestion

        # Build detailed message
        parts = [message]
        if calculator:
            parts.append(f"Calculator: {calculator}")
        if method:
            parts.append(f"Method: {method}")
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")

        super().__init__(" | ".join(parts))


class InsufficientDataError(CalculationError):
    """
    Raised when there's not enough data for a calculation.

    Common in finance where statistical calculations require
    minimum sample sizes for validity.
    """

    def __init__(
        self,
        message: str,
        required: int,
        provided: int,
        **kwargs,
    ):
        self.required = required
        self.provided = provided

        suggestion = f"Provide at least {required} data points (got {provided})"
        super().__init__(message, suggestion=suggestion, **kwargs)


# =============================================================================
# CALCULATION CONTEXT
# =============================================================================

@dataclass
class CalculationContext:
    """
    Context information for a calculation.

    Tracks metadata about the calculation for logging, debugging,
    and auditing purposes.
    """

    # Identification
    calculator_name: str
    method_name: str = "calculate"  # Default method name

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Input summary (for logging, not full data)
    input_summary: Dict[str, Any] = field(default_factory=dict)

    # Warnings generated during calculation
    warnings: List[str] = field(default_factory=list)

    # Debug info
    debug_info: Dict[str, Any] = field(default_factory=dict)

    # Error tracking
    error_message: Optional[str] = None

    def start(self) -> None:
        """Mark calculation as started."""
        self.started_at = datetime.utcnow()
        self.completed_at = None
        self.error_message = None

    def complete(self) -> None:
        """Mark calculation as complete."""
        self.completed_at = datetime.utcnow()

    def fail(self, error: str) -> None:
        """Mark calculation as failed."""
        self.completed_at = datetime.utcnow()
        self.error_message = error

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate duration in milliseconds."""
        if self.completed_at is None or self.started_at is None:
            return None
        delta = self.completed_at - self.started_at
        return delta.total_seconds() * 1000

    @property
    def elapsed_ms(self) -> int:
        """Get elapsed time in milliseconds (alias for compatibility)."""
        duration = self.duration_ms
        return int(duration) if duration is not None else 0

    def add_warning(self, warning: str) -> None:
        """Add a warning to the context."""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "calculator": self.calculator_name,
            "method": self.method_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "warnings_count": len(self.warnings),
            "input_summary": self.input_summary,
            "error": self.error_message,
        }


# =============================================================================
# BASE CALCULATOR
# =============================================================================

# Type variables for generic calculator
InputT = TypeVar("InputT", bound=BaseFinanceModel)
OutputT = TypeVar("OutputT", bound=BaseCalculationResult)
ConfigT = TypeVar("ConfigT", bound=BaseModel)


class BaseCalculator(ABC, Generic[InputT, OutputT, ConfigT]):
    """
    Base class for all Finance Guru calculators.

    Provides common functionality:
    - Input validation
    - Output validation
    - Error handling
    - Logging
    - Calculation context tracking

    USAGE EXAMPLE:
        class RiskCalculator(BaseCalculator[PriceDataInput, RiskMetricsOutput, RiskConfig]):

            def __init__(self, config: RiskConfig):
                super().__init__(config)

            def calculate(self, data: PriceDataInput) -> RiskMetricsOutput:
                # Implementation here
                pass

    All subclasses must implement:
    - calculate(): The main calculation method
    - _validate_input(): Input-specific validation
    - _get_minimum_data_points(): Minimum data requirement
    """

    def __init__(self, config: Optional[ConfigT] = None):
        """
        Initialize calculator with configuration.

        Args:
            config: Optional configuration model. If None, uses defaults.
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._context: Optional[CalculationContext] = None

    @property
    def name(self) -> str:
        """Calculator name for logging and errors."""
        return self.__class__.__name__

    # =========================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def calculate(self, data: InputT, **kwargs) -> OutputT:
        """
        Perform the calculation.

        This is the main method that subclasses must implement.
        It should:
        1. Validate inputs (beyond Pydantic validation)
        2. Perform the calculation
        3. Return a validated output model

        Args:
            data: Input data (validated Pydantic model)
            **kwargs: Additional parameters

        Returns:
            Calculation result (validated Pydantic model)

        Raises:
            CalculationError: If calculation fails
            InsufficientDataError: If not enough data
        """
        pass

    def _validate_input(self, data: InputT) -> None:
        """
        Perform calculator-specific input validation.

        Override this to add validation beyond what Pydantic provides.
        For example, checking that data has enough points for the calculation.

        Args:
            data: Input data to validate

        Raises:
            FinanceValidationError: If validation fails
        """
        pass

    def _get_minimum_data_points(self) -> int:
        """
        Return minimum data points required for calculation.

        Override this in subclasses to specify minimum requirements.

        Returns:
            Minimum number of data points (default: 2)
        """
        return 2

    # =========================================================================
    # COMMON METHODS
    # =========================================================================

    def safe_calculate(self, data: InputT, **kwargs) -> OutputT:
        """
        Execute calculation with full error handling and logging.

        This is the recommended entry point for calculations.
        It wraps calculate() with:
        - Input validation
        - Context tracking
        - Error handling
        - Logging

        Args:
            data: Input data
            **kwargs: Additional parameters

        Returns:
            Calculation result

        Raises:
            CalculationError: With detailed error information
        """
        # Create context
        self._context = CalculationContext(
            calculator_name=self.name,
            method_name="calculate",
            input_summary=self._summarize_input(data),
        )
        self._context.start()  # Start timing

        try:
            # Log start
            self.logger.info(
                f"[{self.name}] Starting calculation | "
                f"Input: {self._context.input_summary}"
            )

            # Validate input
            self._validate_input(data)

            # Perform calculation
            result = self.calculate(data, **kwargs)

            # Add warnings to result
            if self._context.warnings:
                result.warnings.extend(self._context.warnings)

            # Complete context
            self._context.complete()

            # Log success (use elapsed_ms which has None-safe fallback to 0)
            self.logger.info(
                f"[{self.name}] ✅ Calculation complete | "
                f"Duration: {self._context.elapsed_ms}ms | "
                f"Warnings: {len(self._context.warnings)}"
            )

            return result

        except FinanceValidationError as e:
            self.logger.error(f"[{self.name}] ❌ Validation error: {e}")
            raise CalculationError(
                str(e),
                calculator=self.name,
                method="calculate",
                input_summary=self._context.input_summary,
            ) from e

        except Exception as e:
            self.logger.error(
                f"[{self.name}] ❌ Calculation error: {e}",
                exc_info=True,
            )
            raise CalculationError(
                str(e),
                calculator=self.name,
                method="calculate",
                input_summary=self._context.input_summary,
                suggestion="Check input data and configuration",
            ) from e

    def _summarize_input(self, data: InputT) -> Dict[str, Any]:
        """
        Create a summary of input data for logging.

        Override this in subclasses for better summaries.

        Args:
            data: Input data

        Returns:
            Dictionary summary (should not contain full data)
        """
        summary = {"type": type(data).__name__}

        # Common fields
        if hasattr(data, "ticker"):
            summary["ticker"] = data.ticker
        if hasattr(data, "dates") and data.dates:
            summary["data_points"] = len(data.dates)
            summary["date_range"] = f"{data.dates[0]} to {data.dates[-1]}"

        return summary

    def add_warning(self, warning: str) -> None:
        """
        Add a warning to the current calculation context.

        Use this during calculation to flag potential issues
        without stopping the calculation.

        Args:
            warning: Warning message
        """
        if self._context:
            self._context.add_warning(warning)
        self.logger.warning(f"[{self.name}] ⚠️ {warning}")

    def check_minimum_data(self, data_points: int, required: Optional[int] = None) -> None:
        """
        Check if we have enough data points.

        Args:
            data_points: Number of data points available
            required: Required minimum (uses _get_minimum_data_points if None)

        Raises:
            InsufficientDataError: If not enough data
        """
        min_required = required or self._get_minimum_data_points()

        if data_points < min_required:
            raise InsufficientDataError(
                f"Need at least {min_required} data points for {self.name}",
                required=min_required,
                provided=data_points,
                calculator=self.name,
            )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_returns(returns: List[float], min_points: int = 30) -> None:
    """
    Validate return series for statistical calculations.

    Args:
        returns: List of returns
        min_points: Minimum required points

    Raises:
        InsufficientDataError: If not enough data
        FinanceValidationError: If returns are invalid
    """
    if len(returns) < min_points:
        raise InsufficientDataError(
            f"Need at least {min_points} returns for reliable statistics",
            required=min_points,
            provided=len(returns),
        )

    # Check for NaN or Inf
    import math
    for i, ret in enumerate(returns):
        if math.isnan(ret) or math.isinf(ret):
            raise FinanceValidationError(
                f"Invalid return at index {i}: {ret}",
                field="returns",
                suggestion="Check for division by zero or missing data",
            )


def annualize_return(daily_return: float, trading_days: int = 252) -> float:
    """
    Annualize a daily return.

    Args:
        daily_return: Average daily return
        trading_days: Number of trading days per year (default: 252)

    Returns:
        Annualized return
    """
    return daily_return * trading_days


def annualize_volatility(daily_volatility: float, trading_days: int = 252) -> float:
    """
    Annualize daily volatility.

    Volatility scales with square root of time.

    Args:
        daily_volatility: Daily standard deviation
        trading_days: Number of trading days per year

    Returns:
        Annualized volatility
    """
    import math
    return daily_volatility * math.sqrt(trading_days)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Base classes
    "BaseCalculator",
    "CalculationContext",
    # Exceptions
    "CalculationError",
    "InsufficientDataError",
    # Utilities
    "validate_returns",
    "annualize_return",
    "annualize_volatility",
]
