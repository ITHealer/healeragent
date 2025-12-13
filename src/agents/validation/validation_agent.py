

# """
# Validation Agent - Ground Truth Validation (Anthropic Architecture)

# Philosophy:
# 1. NO LLM Validation - use deterministic rules
# 2. Schema + Business Rules validation
# 3. Retry only for transient errors (network, rate limit)
# 4. Graceful degradation - use available data even if some tools fail

# Architecture:
# ┌─────────────────────────────────────────────────────────────────┐
# │                    VALIDATION PIPELINE                           │
# ├─────────────────────────────────────────────────────────────────┤
# │  Stage 1: Status Check (< 1ms)                                  │
# │    → Is status success/error?                                   │
# │    → If error, is it retryable?                                 │
# │                                                                  │
# │  Stage 2: Schema Validation (< 5ms)                             │
# │    → Required fields present?                                   │
# │    → Correct data types?                                        │
# │                                                                  │
# │  Stage 3: Business Rules (< 10ms)                               │
# │    → Domain-specific constraints                                │
# │    → Financial data sanity checks                               │
# │                                                                  │
# │  Total: < 20ms (vs 500ms+ for LLM validation)                   │
# └─────────────────────────────────────────────────────────────────┘
# """

# import logging
# from typing import Dict, Any, List, Optional, Callable
# from dataclasses import dataclass, field
# from enum import Enum
# from pydantic import BaseModel, Field


# # ============================================================================
# # Error Classification
# # ============================================================================

# class ErrorType(Enum):
#     """Classification of errors for retry decision"""
#     TRANSIENT = "transient"     # Network, rate limit - should retry
#     PERMANENT = "permanent"     # Invalid input, not found - don't retry
#     UNKNOWN = "unknown"         # Unknown - don't retry


# # Patterns for transient (retryable) errors
# TRANSIENT_ERROR_PATTERNS = [
#     # Network errors
#     "connection", "timeout", "network", "unreachable",
#     "connect error", "connection refused", "connection reset",
#     "ssl", "certificate",
#     # Rate limits
#     "rate limit", "too many requests", "429", "quota exceeded",
#     "throttle", "throttling",
#     # Temporary server errors
#     "502", "503", "504", "service unavailable", "bad gateway",
#     "gateway timeout", "internal server error", "500",
#     # API temporary issues
#     "temporary", "try again", "retry", "overloaded",
# ]

# # Patterns for permanent (non-retryable) errors
# PERMANENT_ERROR_PATTERNS = [
#     # Invalid input
#     "invalid symbol", "symbol not found", "not found", "404",
#     "invalid parameter", "bad request", "400",
#     # Authorization
#     "unauthorized", "forbidden", "401", "403", "api key",
#     # Data issues
#     "no data", "data not available", "empty result",
#     # Permanent failures
#     "not supported", "deprecated", "disabled",
# ]


# def classify_error(error_msg: str) -> ErrorType:
#     """
#     Classify error type for retry decision
    
#     Args:
#         error_msg: Error message from tool
        
#     Returns:
#         ErrorType indicating if error is retryable
#     """
#     if not error_msg:
#         return ErrorType.UNKNOWN
    
#     error_lower = error_msg.lower()
    
#     # Check transient first (higher priority for retry)
#     for pattern in TRANSIENT_ERROR_PATTERNS:
#         if pattern in error_lower:
#             return ErrorType.TRANSIENT
    
#     # Check permanent
#     for pattern in PERMANENT_ERROR_PATTERNS:
#         if pattern in error_lower:
#             return ErrorType.PERMANENT
    
#     return ErrorType.UNKNOWN


# def is_retryable_error(error_msg: str) -> bool:
#     """Quick check if error is retryable"""
#     return classify_error(error_msg) == ErrorType.TRANSIENT


# # ============================================================================
# # Business Rules for Financial Data
# # ============================================================================

# @dataclass
# class BusinessRuleResult:
#     """Result of business rule validation"""
#     valid: bool
#     errors: List[str] = field(default_factory=list)
#     warnings: List[str] = field(default_factory=list)


# class FinancialBusinessRules:
#     """
#     Domain-specific validation rules for financial data
    
#     These are sanity checks to catch obviously wrong data
#     """
    
#     @staticmethod
#     def validate_stock_price(data: Dict[str, Any]) -> BusinessRuleResult:
#         """
#         Validate stock price data
        
#         Rules:
#         - Price must be positive (> 0)
#         - Price should be reasonable (< $1,000,000 per share)
#         - Volume must be non-negative
#         - Change percent should be reasonable (-100% to +1000%)
#         """
#         errors = []
#         warnings = []
        
#         # Extract price from various possible keys
#         price = (
#             data.get('price') or 
#             data.get('current_price') or
#             data.get('currentPrice') or
#             data.get('regularMarketPrice')
#         )
        
#         if price is not None:
#             if not isinstance(price, (int, float)):
#                 errors.append(f"Price must be numeric, got {type(price).__name__}")
#             elif price <= 0:
#                 errors.append(f"Price must be positive, got {price}")
#             elif price > 1_000_000:
#                 warnings.append(f"Price unusually high: ${price}")
        
#         # Extract volume
#         volume = data.get('volume') or data.get('avgVolume')
#         if volume is not None:
#             if isinstance(volume, (int, float)) and volume < 0:
#                 errors.append(f"Volume cannot be negative, got {volume}")
        
#         # Extract change percent
#         change = (
#             data.get('change_percent') or 
#             data.get('changesPercentage') or
#             data.get('changePercent')
#         )
#         if change is not None:
#             if isinstance(change, (int, float)):
#                 if change < -100:
#                     errors.append(f"Change cannot be < -100%, got {change}%")
#                 elif change > 1000:
#                     warnings.append(f"Change unusually high: {change}%")
        
#         return BusinessRuleResult(
#             valid=len(errors) == 0,
#             errors=errors,
#             warnings=warnings
#         )
    
#     @staticmethod
#     def validate_technical_indicators(data: Dict[str, Any]) -> BusinessRuleResult:
#         """
#         Validate technical indicator data
        
#         Rules:
#         - RSI must be between 0 and 100
#         - Moving averages must be positive
#         - ATR must be non-negative
#         """
#         errors = []
#         warnings = []
        
#         # RSI validation
#         rsi = data.get('rsi') or data.get('rsi_14') or data.get('RSI')
#         if rsi is not None:
#             if isinstance(rsi, (int, float)):
#                 if rsi < 0 or rsi > 100:
#                     errors.append(f"RSI must be 0-100, got {rsi}")
        
#         # Moving averages (check common keys)
#         for ma_key in ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26', 'SMA', 'EMA']:
#             ma_value = data.get(ma_key)
#             if ma_value is not None:
#                 if isinstance(ma_value, (int, float)) and ma_value <= 0:
#                     errors.append(f"{ma_key} must be positive, got {ma_value}")
        
#         # ATR validation
#         atr = data.get('atr') or data.get('atr_14') or data.get('ATR')
#         if atr is not None:
#             if isinstance(atr, (int, float)) and atr < 0:
#                 errors.append(f"ATR cannot be negative, got {atr}")
        
#         return BusinessRuleResult(
#             valid=len(errors) == 0,
#             errors=errors,
#             warnings=warnings
#         )
    
#     @staticmethod
#     def validate_risk_assessment(data: Dict[str, Any]) -> BusinessRuleResult:
#         """
#         Validate risk assessment data
        
#         Rules:
#         - Beta typically between -2 and 5 (warning if outside)
#         - Volatility must be non-negative
#         - Risk score should be 0-100 if present
#         """
#         errors = []
#         warnings = []
        
#         # Beta validation
#         beta = data.get('beta')
#         if beta is not None:
#             if isinstance(beta, (int, float)):
#                 if beta < -5 or beta > 10:
#                     warnings.append(f"Beta unusually extreme: {beta}")
        
#         # Volatility validation
#         volatility = (
#             data.get('volatility') or 
#             data.get('annual_volatility') or
#             data.get('annualVolatility')
#         )
#         if volatility is not None:
#             if isinstance(volatility, (int, float)) and volatility < 0:
#                 errors.append(f"Volatility cannot be negative, got {volatility}")
        
#         # Risk score validation
#         risk_score = data.get('risk_score') or data.get('riskScore')
#         if risk_score is not None:
#             if isinstance(risk_score, (int, float)):
#                 if risk_score < 0 or risk_score > 100:
#                     warnings.append(f"Risk score outside 0-100 range: {risk_score}")
        
#         return BusinessRuleResult(
#             valid=len(errors) == 0,
#             errors=errors,
#             warnings=warnings
#         )
    
#     @staticmethod
#     def validate_financial_statements(data: Dict[str, Any]) -> BusinessRuleResult:
#         """
#         Validate financial statement data
        
#         Basic sanity checks for financial data
#         """
#         errors = []
#         warnings = []
        
#         # Revenue check (can be negative in rare cases, so just warn)
#         revenue = data.get('revenue') or data.get('totalRevenue')
#         if revenue is not None:
#             if isinstance(revenue, (int, float)) and revenue < 0:
#                 warnings.append(f"Negative revenue: {revenue}")
        
#         # Total assets should be positive
#         assets = data.get('totalAssets')
#         if assets is not None:
#             if isinstance(assets, (int, float)) and assets <= 0:
#                 warnings.append(f"Total assets not positive: {assets}")
        
#         return BusinessRuleResult(
#             valid=len(errors) == 0,
#             errors=errors,
#             warnings=warnings
#         )
    
#     @staticmethod
#     def validate_generic(data: Dict[str, Any]) -> BusinessRuleResult:
#         """Generic validation - just check data exists"""
#         return BusinessRuleResult(valid=True)


# # Tool name to business rule mapping
# BUSINESS_RULES_MAP: Dict[str, Callable] = {
#     # Price tools
#     'getStockPrice': FinancialBusinessRules.validate_stock_price,
#     'getStockPerformance': FinancialBusinessRules.validate_stock_price,
#     'getPriceTargets': FinancialBusinessRules.validate_stock_price,
    
#     # Technical tools
#     'getTechnicalIndicators': FinancialBusinessRules.validate_technical_indicators,
#     'detectChartPatterns': FinancialBusinessRules.validate_generic,
#     'getSupportResistance': FinancialBusinessRules.validate_generic,
    
#     # Risk tools
#     'assessRisk': FinancialBusinessRules.validate_risk_assessment,
#     'getVolumeProfile': FinancialBusinessRules.validate_generic,
#     'getSentiment': FinancialBusinessRules.validate_generic,
    
#     # Fundamental tools
#     'getIncomeStatement': FinancialBusinessRules.validate_financial_statements,
#     'getBalanceSheet': FinancialBusinessRules.validate_financial_statements,
#     'getCashFlow': FinancialBusinessRules.validate_financial_statements,
#     'getFinancialRatios': FinancialBusinessRules.validate_financial_statements,
#     'getGrowthMetrics': FinancialBusinessRules.validate_financial_statements,

#     # News & Events tools
#     'getStockNews': FinancialBusinessRules.validate_generic,
#     'getEarningsCalendar': FinancialBusinessRules.validate_generic,
#     'getCompanyEvents': FinancialBusinessRules.validate_generic,

#     # Market tools
#     'getMarketIndices': FinancialBusinessRules.validate_generic,
#     'getSectorPerformance': FinancialBusinessRules.validate_generic,
#     'getMarketMovers': FinancialBusinessRules.validate_generic,
#     'getMarketBreadth': FinancialBusinessRules.validate_generic,
#     'getStockHeatmap': FinancialBusinessRules.validate_generic,
#     'getMarketNews': FinancialBusinessRules.validate_generic,

#     # Discovery tools
#     'stockScreener': FinancialBusinessRules.validate_generic,

#     # Crypto tools
#     'getCryptoPrice': FinancialBusinessRules.validate_generic,
#     'getCryptoTechnicals': FinancialBusinessRules.validate_generic
# }


# # ============================================================================
# # Validation Result Model
# # ============================================================================

# class ValidationResult(BaseModel):
#     """Result of ground truth validation"""
    
#     # Core result
#     is_sufficient: bool = Field(
#         description="Whether tool results are sufficient"
#     )
#     confidence: float = Field(
#         ge=0.0, le=1.0,
#         default=1.0,
#         description="Confidence score 0-1"
#     )
    
#     # Error details
#     missing_data: List[str] = Field(
#         default_factory=list,
#         description="List of missing data fields"
#     )
#     validation_errors: List[str] = Field(
#         default_factory=list,
#         description="Business rule violations"
#     )
#     validation_warnings: List[str] = Field(
#         default_factory=list,
#         description="Non-critical warnings"
#     )
    
#     # Recovery guidance
#     next_action: str = Field(
#         default="complete",
#         description="complete | retry | skip | use_partial"
#     )
#     is_retryable: bool = Field(
#         default=False,
#         description="Whether error is transient and retryable"
#     )
#     reasoning: str = Field(
#         default="",
#         description="Explanation of validation decision"
#     )
    
#     # Execution details
#     error_type: Optional[str] = Field(
#         default=None,
#         description="transient | permanent | unknown"
#     )


# # ============================================================================
# # Validation Agent
# # ============================================================================

# class ValidationAgent:
#     """
#     Ground Truth Validation Agent (Anthropic Architecture)
    
#     Validation Pipeline:
#     1. Status Check - is tool execution successful?
#     2. Schema Validation - are required fields present?
#     3. Business Rules - does data pass sanity checks?
    
#     Key Principles:
#     - NO LLM calls (deterministic, fast)
#     - Retry only for transient errors
#     - Use partial data when possible (graceful degradation)
#     """
    
#     def __init__(
#         self,
#         provider_type: str = None,  # Deprecated, kept for compatibility
#         model_name: str = None,     # Deprecated, kept for compatibility
#         logger: Optional[logging.Logger] = None
#     ):
#         """
#         Initialize validation agent
        
#         Args:
#             provider_type: Deprecated
#             model_name: Deprecated
#             logger: Logger instance
#         """
#         self.logger = logger or logging.getLogger(__name__)
#         self.logger.info("✅ ValidationAgent initialized (Ground Truth mode)")
    
#     async def validate_tool_results(
#         self,
#         original_query: str,
#         tool_name: str,
#         tool_params: Dict[str, Any],
#         tool_results: Dict[str, Any],
#         query_intent: str = "",
#         symbols: List[str] = None
#     ) -> ValidationResult:
#         """
#         Main validation method - Ground Truth approach
        
#         Pipeline:
#         1. Status Check → Is execution successful?
#         2. Schema Check → Are required fields present?
#         3. Business Rules → Does data pass sanity checks?
        
#         Args:
#             original_query: User's original question
#             tool_name: Tool that was executed
#             tool_params: Parameters used for tool
#             tool_results: Results returned by tool
#             query_intent: Not used (kept for compatibility)
#             symbols: Not used (kept for compatibility)
            
#         Returns:
#             ValidationResult with decision and guidance
#         """
        
#         self.logger.debug(f"[VALIDATION] Tool: {tool_name}")
        
#         # =====================================================================
#         # STAGE 1: Status Check (< 1ms)
#         # =====================================================================
        
#         status = tool_results.get('status', 'unknown')
        
#         # Normalize status
#         is_success = status in [200, '200', 'success', 'ok', True]
#         is_error = status in ['error', 'failed', 'failure', False]
        
#         self.logger.debug(f"[VALIDATION] Stage 1 - Status: {status} (success={is_success})")
        
#         # Handle error status
#         if is_error:
#             error_msg = tool_results.get('error', '') or tool_results.get('message', '')
#             error_type = classify_error(error_msg)
#             is_retry = error_type == ErrorType.TRANSIENT
            
#             self.logger.warning(
#                 f"[VALIDATION] ❌ Tool error: {error_msg[:100]} "
#                 f"(type={error_type.value}, retryable={is_retry})"
#             )
            
#             return ValidationResult(
#                 is_sufficient=False,
#                 confidence=1.0,
#                 next_action="retry" if is_retry else "skip",
#                 is_retryable=is_retry,
#                 error_type=error_type.value,
#                 reasoning=f"Tool returned error: {error_msg[:200]}"
#             )
        
#         # =====================================================================
#         # STAGE 2: Schema Check - Data Presence (< 5ms)
#         # =====================================================================
        
#         # Extract data from various locations
#         data = (
#             tool_results.get('data') or 
#             tool_results.get('raw_data') or 
#             tool_results
#         )
        
#         has_data = self._has_meaningful_data(data)
        
#         self.logger.debug(f"[VALIDATION] Stage 2 - Has data: {has_data}")
        
#         if not has_data:
#             # Check content field as fallback
#             content = tool_results.get('content', '')
#             has_content = content and len(str(content).strip()) > 20
            
#             if has_content:
#                 self.logger.info(f"[VALIDATION] ✅ Has content ({len(content)} chars)")
#                 return ValidationResult(
#                     is_sufficient=True,
#                     confidence=0.9,
#                     next_action="complete",
#                     reasoning="Tool returned content response"
#                 )
            
#             # Truly empty - but not necessarily an error
#             # (e.g., "no news available" is valid)
#             self.logger.warning(f"[VALIDATION] ⚠️ No data found")
#             return ValidationResult(
#                 is_sufficient=True,  # Not an error, just no data
#                 confidence=0.7,
#                 missing_data=["data"],
#                 next_action="use_partial",
#                 reasoning="No data available (valid state - may have no results)"
#             )
        
#         # =====================================================================
#         # STAGE 3: Business Rules Validation (< 10ms)
#         # =====================================================================
        
#         # Extract actual tool name (remove task_ prefix if present)
#         actual_tool_name = tool_name
#         if tool_name.startswith("task_"):
#             # For task validation, try to find tool name from results
#             tools_executed = tool_results.get('tools_executed', [])
#             if tools_executed:
#                 actual_tool_name = tools_executed[0]
        
#         business_result = self._validate_business_rules(actual_tool_name, data)
        
#         if business_result.errors:
#             self.logger.warning(
#                 f"[VALIDATION] ⚠️ Business rule violations: {business_result.errors}"
#             )
#             # Business rule failures are NOT retryable - data is wrong, not network
#             return ValidationResult(
#                 is_sufficient=False,
#                 confidence=0.9,
#                 validation_errors=business_result.errors,
#                 validation_warnings=business_result.warnings,
#                 next_action="skip",  # Don't retry - data itself is problematic
#                 is_retryable=False,
#                 reasoning=f"Business rule validation failed: {', '.join(business_result.errors)}"
#             )
        
#         if business_result.warnings:
#             self.logger.info(f"[VALIDATION] Warnings: {business_result.warnings}")
        
#         # =====================================================================
#         # SUCCESS: All validations passed
#         # =====================================================================
        
#         self.logger.info(f"[VALIDATION] ✅ Tool {tool_name} PASSED all validations")
        
#         return ValidationResult(
#             is_sufficient=True,
#             confidence=1.0,
#             validation_warnings=business_result.warnings,
#             next_action="complete",
#             is_retryable=False,
#             reasoning="All validations passed"
#         )
    
#     def _has_meaningful_data(self, data: Any) -> bool:
#         """
#         Check if data contains meaningful values
        
#         Handles: dict, list, string, numbers
#         """
#         if data is None:
#             return False
        
#         if isinstance(data, dict):
#             if len(data) == 0:
#                 return False
            
#             # Check if at least one meaningful value exists
#             for key, value in data.items():
#                 # Skip metadata keys
#                 if key in ['status', 'error', 'message', 'timestamp', 'tool_name']:
#                     continue
                    
#                 if value is not None:
#                     if isinstance(value, (str, list, dict)):
#                         if len(value) > 0:
#                             return True
#                     else:
#                         # Numbers, booleans
#                         return True
#             return False
        
#         if isinstance(data, list):
#             return len(data) > 0
        
#         if isinstance(data, str):
#             return len(data.strip()) > 0
        
#         # Numbers, booleans
#         return True
    
#     def _validate_business_rules(
#         self,
#         tool_name: str,
#         data: Dict[str, Any]
#     ) -> BusinessRuleResult:
#         """
#         Apply business rules validation for specific tool
        
#         Args:
#             tool_name: Name of the tool
#             data: Data to validate
            
#         Returns:
#             BusinessRuleResult with errors and warnings
#         """
#         # Get validator for this tool
#         validator = BUSINESS_RULES_MAP.get(tool_name)
        
#         if validator:
#             try:
#                 return validator(data)
#             except Exception as e:
#                 self.logger.warning(f"Business rule validation error: {e}")
#                 # Don't fail on validation errors - let data through
#                 return BusinessRuleResult(valid=True, warnings=[str(e)])
        
#         # No specific rules - pass by default
#         return BusinessRuleResult(valid=True)
    
#     def validate_for_retry(self, tool_results: Dict[str, Any]) -> bool:
#         """
#         Quick check if tool result warrants a retry
        
#         Used by TaskExecutor for fast retry decision
        
#         Args:
#             tool_results: Results from tool execution
            
#         Returns:
#             True if should retry, False otherwise
#         """
#         status = tool_results.get('status', 'unknown')
        
#         # Only consider retrying errors
#         if status not in ['error', 'failed', 'failure']:
#             return False
        
#         error_msg = tool_results.get('error', '') or tool_results.get('message', '')
#         return is_retryable_error(error_msg)
    
#     def quick_check_empty_results(self, tool_results: Dict[str, Any]) -> bool:
#         """
#         Quick check if tool results are obviously empty/failed
        
#         Used by chat_handler to skip LLM validation for obvious failures.
#         This is a fast pre-check before full validation.
        
#         Args:
#             tool_results: Results from tool execution
            
#         Returns:
#             True if results are obviously empty/failed (skip validation)
#             False if results might have data (proceed to validation)
#         """
#         if not tool_results:
#             return True
        
#         # Check for error status
#         status = tool_results.get('status', 'unknown')
#         if status in ['error', 'failed', 'failure']:
#             return True
        
#         # Check for empty data
#         data = (
#             tool_results.get('data') or 
#             tool_results.get('raw_data') or
#             tool_results.get('result')
#         )
        
#         if data is None:
#             # Check if there's content
#             content = tool_results.get('content', '')
#             if not content or len(str(content).strip()) < 10:
#                 return True
        
#         if isinstance(data, dict) and len(data) == 0:
#             return True
        
#         if isinstance(data, list) and len(data) == 0:
#             return True
        
#         if isinstance(data, str) and len(data.strip()) == 0:
#             return True
        
#         # Has some data - not obviously empty
#         return False


# # ============================================================================
# # Helper Functions
# # ============================================================================

# def should_retry_with_validation(
#     validation_result: ValidationResult,
#     retry_count: int,
#     max_retries: int = 2
# ) -> bool:
#     """
#     Decide if should retry based on validation result
    
#     IMPORTANT: Only retry for TRANSIENT errors
    
#     Args:
#         validation_result: Result from validation
#         retry_count: Current retry count
#         max_retries: Maximum retries allowed
        
#     Returns:
#         True if should retry, False otherwise
#     """
#     # Already at max retries
#     if retry_count >= max_retries:
#         return False
    
#     # Successful validation - no retry needed
#     if validation_result.is_sufficient:
#         return False
    
#     # Only retry if it's a transient error
#     return validation_result.is_retryable


# def create_quick_validation(
#     is_success: bool,
#     has_data: bool = True,
#     error_msg: str = ""
# ) -> ValidationResult:
#     """
#     Create quick validation result for fast-path
    
#     Used by TaskExecutor to skip full validation for atomic tools
#     """
#     if is_success and has_data:
#         return ValidationResult(
#             is_sufficient=True,
#             confidence=1.0,
#             next_action="complete",
#             reasoning="Quick validation: success with data"
#         )
    
#     if is_success and not has_data:
#         return ValidationResult(
#             is_sufficient=True,
#             confidence=0.8,
#             missing_data=["data"],
#             next_action="use_partial",
#             reasoning="Quick validation: success but no data"
#         )
    
#     # Error case
#     error_type = classify_error(error_msg)
#     is_retry = error_type == ErrorType.TRANSIENT
    
#     return ValidationResult(
#         is_sufficient=False,
#         confidence=1.0,
#         next_action="retry" if is_retry else "skip",
#         is_retryable=is_retry,
#         error_type=error_type.value,
#         reasoning=f"Quick validation: error - {error_msg[:100]}"
#     )


# # ============================================================================
# # Exponential Backoff Helper
# # ============================================================================

# def calculate_backoff(attempt: int, base: float = 1.0, max_backoff: float = 8.0) -> float:
#     """
#     Calculate exponential backoff delay
    
#     Args:
#         attempt: Current attempt number (0-indexed)
#         base: Base delay in seconds
#         max_backoff: Maximum delay in seconds
        
#     Returns:
#         Delay in seconds
#     """
#     delay = base * (2 ** attempt)
#     return min(delay, max_backoff)


# File: src/agents/validation/validation_agent.py
"""
Validation Agent - Ground Truth Validation (No LLM)

ARCHITECTURE:
- Stage 1: Status Check (< 1ms) - Is execution successful?
- Stage 2: Schema Validation (< 5ms) - Are required fields present?
- Stage 3: Business Rules (< 10ms) - Domain-specific sanity checks

Total: < 20ms (vs 500ms+ for LLM validation)

Key Principles:
- NO LLM calls (deterministic, fast)
- Retry only for transient errors
- Graceful degradation (use partial data when possible)
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field


# ============================================================================
# ERROR CLASSIFICATION
# ============================================================================

class ErrorType(Enum):
    """Classification of errors for retry decision"""
    TRANSIENT = "transient"   # Network, rate limit - should retry
    PERMANENT = "permanent"   # Invalid input, not found - don't retry
    UNKNOWN = "unknown"       # Unknown - don't retry


# Transient errors (retryable)
TRANSIENT_PATTERNS = [
    "connection", "timeout", "network", "unreachable",
    "rate limit", "too many requests", "429", "quota",
    "throttle", "502", "503", "504", "service unavailable",
    "gateway timeout", "internal server error", "500",
    "temporary", "try again", "retry", "overloaded",
]

# Permanent errors (non-retryable)
PERMANENT_PATTERNS = [
    "invalid symbol", "symbol not found", "not found", "404",
    "invalid parameter", "bad request", "400",
    "unauthorized", "forbidden", "401", "403", "api key",
    "no data", "data not available", "empty result",
    "not supported", "deprecated", "disabled",
]


def classify_error(error_msg: str) -> ErrorType:
    """Classify error type for retry decision"""
    if not error_msg:
        return ErrorType.UNKNOWN
    
    error_lower = error_msg.lower()
    
    for pattern in TRANSIENT_PATTERNS:
        if pattern in error_lower:
            return ErrorType.TRANSIENT
    
    for pattern in PERMANENT_PATTERNS:
        if pattern in error_lower:
            return ErrorType.PERMANENT
    
    return ErrorType.UNKNOWN


def is_retryable_error(error_msg: str) -> bool:
    """Quick check if error is retryable"""
    return classify_error(error_msg) == ErrorType.TRANSIENT


def calculate_backoff(attempt: int, base: float = 1.0, max_backoff: float = 8.0) -> float:
    """Calculate exponential backoff delay"""
    delay = base * (2 ** attempt)
    return min(delay, max_backoff)


# ============================================================================
# BUSINESS RULES
# ============================================================================

@dataclass
class BusinessRuleResult:
    """Result of business rule validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class FinancialBusinessRules:
    """Domain-specific validation rules for financial data"""
    
    @staticmethod
    def validate_stock_price(data: Dict[str, Any]) -> BusinessRuleResult:
        """Validate stock price data"""
        errors = []
        warnings = []
        
        # Extract price
        price = (
            data.get('price') or 
            data.get('current_price') or
            data.get('currentPrice') or
            data.get('regularMarketPrice')
        )
        
        if price is not None:
            if not isinstance(price, (int, float)):
                errors.append(f"Price must be numeric, got {type(price).__name__}")
            elif price <= 0:
                errors.append(f"Price must be positive, got {price}")
            elif price > 1_000_000:
                warnings.append(f"Price unusually high: ${price}")
        
        # Extract volume
        volume = data.get('volume') or data.get('avgVolume')
        if volume is not None and isinstance(volume, (int, float)) and volume < 0:
            errors.append(f"Volume cannot be negative, got {volume}")
        
        # Extract change percent
        change = (
            data.get('change_percent') or 
            data.get('changesPercentage') or
            data.get('changePercent')
        )
        if change is not None and isinstance(change, (int, float)):
            if change < -100:
                errors.append(f"Change cannot be < -100%, got {change}%")
            elif change > 1000:
                warnings.append(f"Change unusually high: {change}%")
        
        return BusinessRuleResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    @staticmethod
    def validate_technical_indicators(data: Dict[str, Any]) -> BusinessRuleResult:
        """Validate technical indicator data"""
        errors = []
        warnings = []
        
        # RSI validation (must be 0-100)
        rsi = data.get('rsi') or data.get('rsi_14') or data.get('RSI')
        if rsi is not None and isinstance(rsi, (int, float)):
            if rsi < 0 or rsi > 100:
                errors.append(f"RSI must be 0-100, got {rsi}")
        
        # Moving averages (must be positive)
        for ma_key in ['sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26']:
            ma_value = data.get(ma_key)
            if ma_value is not None and isinstance(ma_value, (int, float)) and ma_value <= 0:
                errors.append(f"{ma_key} must be positive, got {ma_value}")
        
        # ATR validation (must be non-negative)
        atr = data.get('atr') or data.get('atr_14') or data.get('ATR')
        if atr is not None and isinstance(atr, (int, float)) and atr < 0:
            errors.append(f"ATR cannot be negative, got {atr}")
        
        return BusinessRuleResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    @staticmethod
    def validate_risk_assessment(data: Dict[str, Any]) -> BusinessRuleResult:
        """Validate risk assessment data"""
        errors = []
        warnings = []
        
        # Beta validation
        beta = data.get('beta')
        if beta is not None and isinstance(beta, (int, float)):
            if beta < -5 or beta > 10:
                warnings.append(f"Beta unusually extreme: {beta}")
        
        # Volatility validation
        volatility = data.get('volatility') or data.get('annual_volatility')
        if volatility is not None and isinstance(volatility, (int, float)) and volatility < 0:
            errors.append(f"Volatility cannot be negative, got {volatility}")
        
        return BusinessRuleResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    @staticmethod
    def validate_financial_statements(data: Dict[str, Any]) -> BusinessRuleResult:
        """Validate financial statement data"""
        errors = []
        warnings = []
        
        revenue = data.get('revenue') or data.get('totalRevenue')
        if revenue is not None and isinstance(revenue, (int, float)) and revenue < 0:
            warnings.append(f"Negative revenue: {revenue}")
        
        assets = data.get('totalAssets')
        if assets is not None and isinstance(assets, (int, float)) and assets <= 0:
            warnings.append(f"Total assets not positive: {assets}")
        
        return BusinessRuleResult(valid=len(errors) == 0, errors=errors, warnings=warnings)
    
    @staticmethod
    def validate_generic(data: Dict[str, Any]) -> BusinessRuleResult:
        """Generic validation - just check data exists"""
        return BusinessRuleResult(valid=True)


# Tool to validator mapping
BUSINESS_RULES_MAP: Dict[str, Callable] = {
    # Prices tools
    'getStockPrice': FinancialBusinessRules.validate_stock_price,
    'getStockPerformance': FinancialBusinessRules.validate_stock_price,
    'getPriceTargets': FinancialBusinessRules.validate_stock_price,

    # Technical indicators tools
    'getTechnicalIndicators': FinancialBusinessRules.validate_technical_indicators,
    'detectChartPatterns': FinancialBusinessRules.validate_generic,
    'getSupportResistance': FinancialBusinessRules.validate_generic,

    # Risk tools
    'assessRisk': FinancialBusinessRules.validate_risk_assessment,
    'getVolumeProfile': FinancialBusinessRules.validate_generic,
    'getSentiment': FinancialBusinessRules.validate_generic,
    'suggestStopLoss': FinancialBusinessRules.validate_generic,

    # Fundamentals tools
    'getIncomeStatement': FinancialBusinessRules.validate_financial_statements,
    'getBalanceSheet': FinancialBusinessRules.validate_financial_statements,
    'getCashFlow': FinancialBusinessRules.validate_financial_statements,
    'getFinancialRatios': FinancialBusinessRules.validate_financial_statements,
    'getGrowthMetrics': FinancialBusinessRules.validate_financial_statements,

    # News tools
    'getStockNews': FinancialBusinessRules.validate_generic,
    'getEarningsCalendar': FinancialBusinessRules.validate_generic,
    'getCompanyEvents': FinancialBusinessRules.validate_generic,

    # Market tools
    'getMarketIndices': FinancialBusinessRules.validate_generic,
    'getSectorPerformance': FinancialBusinessRules.validate_generic,
    'getMarketMovers': FinancialBusinessRules.validate_generic,
    'getMarketBreadth': FinancialBusinessRules.validate_generic,
    'getStockHeatmap': FinancialBusinessRules.validate_generic,
    'getMarketNews': FinancialBusinessRules.validate_generic,

    # Discovery tools
    'stockScreener': FinancialBusinessRules.validate_generic,

    # Crypto tools
    'getCryptoPrice': FinancialBusinessRules.validate_generic,
    'getCryptoTechnicals': FinancialBusinessRules.validate_generic,
}


# ============================================================================
# VALIDATION RESULT
# ============================================================================

class ValidationResult(BaseModel):
    """Result of ground truth validation"""
    
    is_sufficient: bool = Field(description="Whether tool results are sufficient")
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    
    missing_data: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    
    next_action: str = Field(default="complete", description="complete|retry|skip|use_partial")
    is_retryable: bool = Field(default=False)
    reasoning: str = Field(default="")
    error_type: Optional[str] = Field(default=None)


# ============================================================================
# VALIDATION AGENT
# ============================================================================

class ValidationAgent:
    """
    Ground Truth Validation Agent (No LLM)
    
    Pipeline:
    1. Status Check - Is tool execution successful?
    2. Schema Validation - Are required fields present?
    3. Business Rules - Does data pass sanity checks?
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize validation agent"""
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info("✅ ValidationAgent initialized (Ground Truth mode)")
    
    async def validate_tool_results(
        self,
        original_query: str,
        tool_name: str,
        tool_params: Dict[str, Any],
        tool_results: Dict[str, Any],
        **kwargs
    ) -> ValidationResult:
        """
        Main validation method
        
        Args:
            original_query: User's original question
            tool_name: Tool that was executed
            tool_params: Parameters used
            tool_results: Results returned
            
        Returns:
            ValidationResult with decision
        """
        
        self.logger.debug(f"[VALIDATION] Tool: {tool_name}")
        
        # =====================================================================
        # STAGE 1: STATUS CHECK (< 1ms)
        # =====================================================================
        
        status = tool_results.get('status', 'unknown')
        is_success = status in [200, '200', 'success', 'ok', True]
        is_error = status in ['error', 'failed', 'failure', False]
        
        if is_error:
            error_msg = tool_results.get('error', '') or tool_results.get('message', '')
            error_type = classify_error(error_msg)
            is_retry = error_type == ErrorType.TRANSIENT
            
            self.logger.warning(
                f"[VALIDATION] ❌ Error: {error_msg[:100]} (retryable={is_retry})"
            )
            
            return ValidationResult(
                is_sufficient=False,
                confidence=1.0,
                next_action="retry" if is_retry else "skip",
                is_retryable=is_retry,
                error_type=error_type.value,
                reasoning=f"Tool error: {error_msg[:200]}"
            )
        
        # =====================================================================
        # STAGE 2: DATA PRESENCE CHECK (< 5ms)
        # =====================================================================
        
        data = tool_results.get('data') or tool_results.get('raw_data') or tool_results
        has_data = self._has_meaningful_data(data)
        
        if not has_data:
            # Check content field as fallback
            content = tool_results.get('content', '')
            if content and len(str(content).strip()) > 20:
                self.logger.info(f"[VALIDATION] ✅ Has content ({len(content)} chars)")
                return ValidationResult(
                    is_sufficient=True,
                    confidence=0.9,
                    next_action="complete",
                    reasoning="Tool returned content response"
                )
            
            self.logger.warning("[VALIDATION] ⚠️ No data found")
            return ValidationResult(
                is_sufficient=True,  # Not error, just no data
                confidence=0.7,
                missing_data=["data"],
                next_action="use_partial",
                reasoning="No data available (valid state)"
            )
        
        # =====================================================================
        # STAGE 3: BUSINESS RULES (< 10ms)
        # =====================================================================
        
        business_result = self._validate_business_rules(tool_name, data)
        
        if business_result.errors:
            self.logger.warning(f"[VALIDATION] ⚠️ Business violations: {business_result.errors}")
            return ValidationResult(
                is_sufficient=False,
                confidence=0.9,
                validation_errors=business_result.errors,
                validation_warnings=business_result.warnings,
                next_action="skip",
                is_retryable=False,
                reasoning=f"Business rule failed: {', '.join(business_result.errors)}"
            )
        
        # =====================================================================
        # SUCCESS
        # =====================================================================
        
        self.logger.info(f"[VALIDATION] ✅ {tool_name} PASSED")
        
        return ValidationResult(
            is_sufficient=True,
            confidence=1.0,
            validation_warnings=business_result.warnings,
            next_action="complete",
            reasoning="All validations passed"
        )
    
    def _has_meaningful_data(self, data: Any) -> bool:
        """Check if data contains meaningful values"""
        if data is None:
            return False
        
        if isinstance(data, dict):
            if len(data) == 0:
                return False
            for key, value in data.items():
                if key in ['status', 'error', 'message', 'timestamp', 'tool_name']:
                    continue
                if value is not None:
                    if isinstance(value, (str, list, dict)) and len(value) > 0:
                        return True
                    elif not isinstance(value, (str, list, dict)):
                        return True
            return False
        
        if isinstance(data, list):
            return len(data) > 0
        
        if isinstance(data, str):
            return len(data.strip()) > 0
        
        return True
    
    def _validate_business_rules(self, tool_name: str, data: Dict[str, Any]) -> BusinessRuleResult:
        """Apply business rules validation"""
        validator = BUSINESS_RULES_MAP.get(tool_name)
        
        if validator:
            try:
                return validator(data)
            except Exception as e:
                self.logger.warning(f"Business validation error: {e}")
                return BusinessRuleResult(valid=True, warnings=[str(e)])
        
        return BusinessRuleResult(valid=True)
    
    def validate_for_retry(self, tool_results: Dict[str, Any]) -> bool:
        """Quick check if should retry"""
        status = tool_results.get('status', 'unknown')
        if status not in ['error', 'failed', 'failure']:
            return False
        
        error_msg = tool_results.get('error', '') or tool_results.get('message', '')
        return is_retryable_error(error_msg)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_quick_validation(
    is_success: bool,
    has_data: bool = True,
    error_msg: str = ""
) -> ValidationResult:
    """Create quick validation result for fast-path"""
    
    if is_success and has_data:
        return ValidationResult(
            is_sufficient=True,
            confidence=1.0,
            next_action="complete",
            reasoning="Quick validation: success with data"
        )
    
    if is_success and not has_data:
        return ValidationResult(
            is_sufficient=True,
            confidence=0.8,
            missing_data=["data"],
            next_action="use_partial",
            reasoning="Quick validation: success but no data"
        )
    
    error_type = classify_error(error_msg)
    is_retry = error_type == ErrorType.TRANSIENT
    
    return ValidationResult(
        is_sufficient=False,
        confidence=1.0,
        next_action="retry" if is_retry else "skip",
        is_retryable=is_retry,
        error_type=error_type.value,
        reasoning=f"Quick validation: error - {error_msg[:100]}"
    )