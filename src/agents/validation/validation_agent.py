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
        self.logger.info("ValidationAgent initialized")
    
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