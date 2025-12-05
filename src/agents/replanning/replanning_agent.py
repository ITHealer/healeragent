"""
Replanning Agent - 3-Level Recovery System

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                         REPLANNING DECISION TREE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Tool Failed                                                                 │
│      ↓                                                                       │
│  ┌───────────────────────────────────────┐                                  │
│  │ Level 1: Can fix params?              │                                  │
│  │ (missing field, wrong format, etc.)   │                                  │
│  └───────────────┬───────────────────────┘                                  │
│                  ├─ YES → Fix params & Retry                                │
│                  │         ↓                                                 │
│                  │    Still fails?                                           │
│                  │         ↓                                                 │
│                  └─ NO ──→ Continue                                          │
│                            ↓                                                 │
│  ┌───────────────────────────────────────┐                                  │
│  │ Level 2: Has fallback tool?           │                                  │
│  │ (check fallback registry)             │                                  │
│  └───────────────┬───────────────────────┘                                  │
│                  ├─ YES → Use fallback tool                                 │
│                  │         ↓                                                 │
│                  │    Fallback fails?                                        │
│                  │         ↓                                                 │
│                  └─ NO ──→ Continue                                          │
│                            ↓                                                 │
│  ┌───────────────────────────────────────┐                                  │
│  │ Level 3: Can create new plan?         │                                  │
│  │ (exclude failed tools/symbols)        │                                  │
│  └───────────────┬───────────────────────┘                                  │
│                  ├─ YES → Full Replan                                       │
│                  │         ↓                                                 │
│                  │    Execute new plan                                       │
│                  │                                                           │
│                  └─ NO ──→ Graceful Degradation                             │
│                            ↓                                                 │
│                    Return partial results + explain limitation               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    replanning_agent = ReplanningAgent(planning_agent)
    
    # When tool fails
    decision = await replanning_agent.decide_recovery_strategy(
        failed_tool="getStockPrice",
        error_info=error_info,
        original_plan=plan,
        execution_history=history
    )
    
    if decision.action == ReplanAction.LOCAL_REPAIR:
        # Retry with fixed params
        fixed_params = decision.fixed_params
        
    elif decision.action == ReplanAction.TOOL_FALLBACK:
        # Use alternative tool
        fallback_tool = decision.fallback_tool
        
    elif decision.action == ReplanAction.FULL_REPLAN:
        # Execute new plan
        new_plan = decision.new_plan
        
    else:  # GRACEFUL_DEGRADATION
        # Use available data, inform user of limitations
        user_message = decision.user_message
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from pydantic import BaseModel, Field


# ============================================================================
# Enums & Constants
# ============================================================================

class ReplanAction(str, Enum):
    """Possible replanning actions"""
    LOCAL_REPAIR = "local_repair"           # Level 1: Fix params
    TOOL_FALLBACK = "tool_fallback"         # Level 2: Use alternative tool
    FULL_REPLAN = "full_replan"             # Level 3: Create new plan
    GRACEFUL_DEGRADATION = "graceful_degradation"  # Cannot recover
    NO_ACTION = "no_action"                 # No replanning needed


class ErrorCategory(str, Enum):
    """Error categories for classification"""
    PARAM_ERROR = "param_error"           # Fixable parameter issues
    SYMBOL_ERROR = "symbol_error"         # Invalid/unknown symbol
    API_ERROR = "api_error"               # Transient API issues
    DATA_ERROR = "data_error"             # No data available
    AUTH_ERROR = "auth_error"             # Authorization issues
    RATE_LIMIT = "rate_limit"             # Rate limiting
    UNKNOWN = "unknown"


# ============================================================================
# Error Classification Patterns
# ============================================================================

ERROR_PATTERNS = {
    ErrorCategory.PARAM_ERROR: [
        "missing required", "missing parameter", "invalid parameter",
        "required field", "invalid format", "parameter out of range",
        "invalid date", "invalid value", "wrong type"
    ],
    ErrorCategory.SYMBOL_ERROR: [
        "symbol not found", "invalid symbol", "ticker not found",
        "unknown symbol", "no such ticker", "symbol does not exist",
        "không tìm thấy mã", "mã không tồn tại"
    ],
    ErrorCategory.API_ERROR: [
        "timeout", "connection", "network", "unreachable",
        "500", "502", "503", "504", "internal server error",
        "service unavailable", "gateway timeout"
    ],
    ErrorCategory.DATA_ERROR: [
        "no data", "data not available", "empty result",
        "no results", "not found", "404"
    ],
    ErrorCategory.AUTH_ERROR: [
        "unauthorized", "forbidden", "401", "403",
        "api key", "authentication", "permission denied"
    ],
    ErrorCategory.RATE_LIMIT: [
        "rate limit", "too many requests", "429",
        "quota exceeded", "throttle"
    ]
}


# Patterns for fixable parameter errors
FIXABLE_PARAM_PATTERNS = {
    # Missing field patterns -> suggested fix
    "missing_exchange": {
        "patterns": ["missing.*exchange", "exchange.*required"],
        "fix_key": "exchange",
        "fix_logic": "infer_from_symbol"
    },
    "missing_period": {
        "patterns": ["missing.*period", "period.*required"],
        "fix_key": "period",
        "fix_value": "1y"  # Default to 1 year
    },
    "invalid_date_format": {
        "patterns": ["invalid date", "date format", "wrong date"],
        "fix_key": "date",
        "fix_logic": "reformat_date"
    },
    "missing_limit": {
        "patterns": ["missing.*limit", "limit.*required"],
        "fix_key": "limit",
        "fix_value": 10  # Default limit
    }
}


# ============================================================================
# Fallback Tool Registry
# ============================================================================

# Define fallback chains for each tool
# Format: tool_name -> list of fallback tools in priority order
TOOL_FALLBACK_REGISTRY: Dict[str, List[str]] = {
    # Price & Performance Tools
    "getStockPrice": [
        # "getStockPriceFromYahoo",  # TODO: Implement Yahoo fallback
        # "getQuoteData",             # TODO: Implement quote fallback
    ],
    "getStockPerformance": [
        # "getBasicPerformance",
    ],
    
    # Technical Analysis Tools
    "getTechnicalIndicators": [
        # "getBasicTechnicals",
        # "getPriceHistory",  # Can calculate indicators from price history
    ],
    "detectChartPatterns": [
        # "getBasicPatterns",
    ],
    
    # Risk Management Tools
    "assessRisk": [
        # "getBasicRiskMetrics",
    ],
    
    # Fundamental Tools
    "getIncomeStatement": [
        # "getBasicFinancials",
    ],
    "getBalanceSheet": [
        # "getBasicFinancials",
    ],
    "getCashFlow": [
        # "getBasicFinancials",
    ],
    
    # News Tools
    "getStockNews": [
        # "getMarketNews",
    ],
}


# ============================================================================
# Similar Symbol Database (for typo correction)
# ============================================================================

# Common symbol typos and their corrections
SYMBOL_CORRECTIONS: Dict[str, str] = {
    # NVIDIA variations
    "NVIDDA": "NVDA",
    "NVIDA": "NVDA",
    "NVIDEA": "NVDA",
    "NIVIDIA": "NVDA",
    
    # Apple variations
    "APPL": "AAPL",
    "APLE": "AAPL",
    
    # Tesla variations
    "TLSA": "TSLA",
    "TELSA": "TSLA",
    
    # Microsoft variations
    "MSFT": "MSFT",
    "MIRCOSOFT": "MSFT",
    "MICROSFT": "MSFT",
    
    # Google/Alphabet variations
    "GOOG": "GOOGL",
    "GOOGLE": "GOOGL",
    "GOGLE": "GOOGL",
    
    # Amazon variations
    "AMZN": "AMZN",
    "AMAZN": "AMZN",
    "AMAZON": "AMZN",
    
    # Meta/Facebook variations
    "FB": "META",
    "FACEBOOK": "META",
    "FACBOOK": "META",
}


# Company name to symbol mapping
COMPANY_TO_SYMBOL: Dict[str, str] = {
    "nvidia": "NVDA",
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "qualcomm": "QCOM",
    "broadcom": "AVGO",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "paypal": "PYPL",
    "disney": "DIS",
    "boeing": "BA",
    "coca-cola": "KO",
    "pepsi": "PEP",
    "walmart": "WMT",
    "costco": "COST",
    "starbucks": "SBUX",
    "nike": "NKE",
    "visa": "V",
    "mastercard": "MA",
    "jpmorgan": "JPM",
    "goldman": "GS",
    "berkshire": "BRK.B",
    "johnson": "JNJ",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "merck": "MRK",
    "exxon": "XOM",
    "chevron": "CVX",
}


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ErrorInfo:
    """Information about a tool execution error"""
    tool_name: str
    error_message: str
    error_code: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    category: ErrorCategory = ErrorCategory.UNKNOWN
    is_retryable: bool = False
    
    def __post_init__(self):
        """Auto-classify error category"""
        if self.category == ErrorCategory.UNKNOWN:
            self.category = self._classify_error()
            self.is_retryable = self.category in [
                ErrorCategory.API_ERROR,
                ErrorCategory.RATE_LIMIT
            ]
    
    def _classify_error(self) -> ErrorCategory:
        """Classify error based on message patterns"""
        error_lower = self.error_message.lower()
        
        for category, patterns in ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_lower:
                    return category
        
        return ErrorCategory.UNKNOWN


@dataclass
class ExecutionHistory:
    """History of tool executions for learning"""
    successful_tools: List[str] = field(default_factory=list)
    failed_tools: List[Tuple[str, str]] = field(default_factory=list)  # (tool_name, error)
    retry_counts: Dict[str, int] = field(default_factory=dict)
    total_execution_time: float = 0.0


class ReplanDecision(BaseModel):
    """Decision from replanning agent"""
    
    # Action to take
    action: ReplanAction = Field(description="Replanning action to take")
    level: int = Field(default=0, description="Replanning level (1, 2, or 3)")
    
    # Level 1: Local Repair
    fixed_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fixed parameters for retry"
    )
    
    # Level 2: Tool Fallback
    fallback_tool: Optional[str] = Field(
        default=None,
        description="Alternative tool to use"
    )
    fallback_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for fallback tool"
    )
    
    # Level 3: Full Replan
    new_plan: Optional[Any] = Field(
        default=None,
        description="New TaskPlan if full replanning"
    )
    corrected_symbols: List[str] = Field(
        default_factory=list,
        description="Corrected symbol list"
    )
    
    # Graceful Degradation
    should_continue: bool = Field(
        default=True,
        description="Whether to continue with remaining tasks"
    )
    
    # User Communication (multilingual)
    user_message: str = Field(
        default="",
        description="Message to inform user about the situation"
    )
    user_message_vi: str = Field(
        default="",
        description="Vietnamese version of user message"
    )
    
    # Metadata
    reasoning: str = Field(default="", description="Why this decision was made")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ============================================================================
# Replanning Agent
# ============================================================================

class ReplanningAgent:
    """
    3-Level Replanning Agent
    
    Handles error recovery through:
    1. Level 1: Local Repair - Fix parameters
    2. Level 2: Tool Fallback - Use alternative tools
    3. Level 3: Full Replanning - Create new plan
    
    Falls back to graceful degradation if all levels fail.
    """
    
    def __init__(
        self,
        planning_agent = None,  # PlanningAgent instance for Level 3
        max_replan_attempts: int = 3,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize replanning agent
        
        Args:
            planning_agent: PlanningAgent instance for full replanning
            max_replan_attempts: Maximum number of replan attempts
            logger: Logger instance
        """
        self.planning_agent = planning_agent
        self.max_replan_attempts = max_replan_attempts
        self.logger = logger or logging.getLogger(__name__)
        
        # Track replanning statistics
        self.stats = {
            "total_replans": 0,
            "level_1_success": 0,
            "level_2_success": 0,
            "level_3_success": 0,
            "graceful_degradations": 0
        }
        
        self.logger.info("✅ ReplanningAgent initialized (3-Level Recovery)")
    
    async def decide_recovery_strategy(
        self,
        failed_tool: str,
        error_info: ErrorInfo,
        original_plan: Any,  # TaskPlan
        execution_history: ExecutionHistory,
        user_query: str = "",
        detected_language: str = "en"
    ) -> ReplanDecision:
        """
        Main entry point - decide recovery strategy based on error
        
        Args:
            failed_tool: Name of the failed tool
            error_info: Information about the error
            original_plan: Original TaskPlan
            execution_history: History of executions
            user_query: Original user query
            detected_language: Detected language (en/vi/auto)
            
        Returns:
            ReplanDecision with action and details
        """
        self.stats["total_replans"] += 1
        
        self.logger.info(f"[REPLAN] Analyzing failure: {failed_tool}")
        self.logger.info(f"[REPLAN] Error: {error_info.error_message[:100]}")
        self.logger.info(f"[REPLAN] Category: {error_info.category.value}")
        
        # Check retry count
        retry_count = execution_history.retry_counts.get(failed_tool, 0)
        if retry_count >= self.max_replan_attempts:
            self.logger.warning(f"[REPLAN] Max retries reached for {failed_tool}")
            return self._create_graceful_degradation(
                error_info, detected_language,
                reason="Max retry attempts reached"
            )
        
        # =====================================================================
        # LEVEL 1: Local Repair (Fix Parameters)
        # =====================================================================
        if error_info.category == ErrorCategory.PARAM_ERROR:
            self.logger.info("[REPLAN] Attempting Level 1: Local Repair")
            
            level_1_result = self._try_local_repair(error_info)
            
            if level_1_result.action == ReplanAction.LOCAL_REPAIR:
                self.stats["level_1_success"] += 1
                return level_1_result
        
        # =====================================================================
        # LEVEL 2: Tool Fallback
        # =====================================================================
        if error_info.category in [ErrorCategory.API_ERROR, ErrorCategory.RATE_LIMIT]:
            self.logger.info("[REPLAN] Attempting Level 2: Tool Fallback")
            
            level_2_result = self._try_tool_fallback(
                failed_tool, error_info, execution_history
            )
            
            if level_2_result.action == ReplanAction.TOOL_FALLBACK:
                self.stats["level_2_success"] += 1
                return level_2_result
        
        # =====================================================================
        # LEVEL 3: Full Replanning
        # =====================================================================
        if error_info.category in [ErrorCategory.SYMBOL_ERROR, ErrorCategory.DATA_ERROR]:
            self.logger.info("[REPLAN] Attempting Level 3: Full Replanning")
            
            level_3_result = await self._try_full_replan(
                error_info=error_info,
                original_plan=original_plan,
                execution_history=execution_history,
                user_query=user_query,
                detected_language=detected_language
            )
            
            if level_3_result.action == ReplanAction.FULL_REPLAN:
                self.stats["level_3_success"] += 1
                return level_3_result
        
        # =====================================================================
        # GRACEFUL DEGRADATION
        # =====================================================================
        self.logger.warning("[REPLAN] All recovery strategies failed, graceful degradation")
        self.stats["graceful_degradations"] += 1
        
        return self._create_graceful_degradation(
            error_info, detected_language,
            reason="No recovery strategy available"
        )
    
    # ========================================================================
    # Level 1: Local Repair
    # ========================================================================
    
    def _try_local_repair(self, error_info: ErrorInfo) -> ReplanDecision:
        """
        Level 1: Try to fix parameters based on error message
        
        Examples:
        - Missing exchange -> Infer from symbol
        - Wrong date format -> Reformat
        - Missing limit -> Add default
        """
        error_lower = error_info.error_message.lower()
        fixed_params = dict(error_info.params)  # Copy original params
        fixes_applied = []
        
        for fix_name, fix_config in FIXABLE_PARAM_PATTERNS.items():
            # Check if error matches this pattern
            matched = False
            for pattern in fix_config["patterns"]:
                if re.search(pattern, error_lower):
                    matched = True
                    break
            
            if not matched:
                continue
            
            # Apply fix
            fix_key = fix_config["fix_key"]
            
            if "fix_value" in fix_config:
                # Static value fix
                fixed_params[fix_key] = fix_config["fix_value"]
                fixes_applied.append(f"{fix_key}={fix_config['fix_value']}")
                
            elif fix_config.get("fix_logic") == "infer_from_symbol":
                # Infer exchange from symbol
                symbol = fixed_params.get("symbol", "")
                exchange = self._infer_exchange_from_symbol(symbol)
                if exchange:
                    fixed_params[fix_key] = exchange
                    fixes_applied.append(f"{fix_key}={exchange}")
                    
            elif fix_config.get("fix_logic") == "reformat_date":
                # Try to reformat date
                date_val = fixed_params.get("date", "")
                reformatted = self._reformat_date(date_val)
                if reformatted:
                    fixed_params[fix_key] = reformatted
                    fixes_applied.append(f"{fix_key}={reformatted}")
        
        if fixes_applied:
            self.logger.info(f"[REPLAN:L1] Applied fixes: {', '.join(fixes_applied)}")
            
            return ReplanDecision(
                action=ReplanAction.LOCAL_REPAIR,
                level=1,
                fixed_params=fixed_params,
                reasoning=f"Fixed parameters: {', '.join(fixes_applied)}",
                confidence=0.8
            )
        
        # Cannot fix
        return ReplanDecision(
            action=ReplanAction.NO_ACTION,
            level=1,
            reasoning="Cannot identify fixable parameter issue"
        )
    
    def _infer_exchange_from_symbol(self, symbol: str) -> Optional[str]:
        """Infer exchange from symbol pattern"""
        if not symbol:
            return None
        
        # US stocks (most common)
        if re.match(r'^[A-Z]{1,5}$', symbol):
            return "NASDAQ"  # Default to NASDAQ for US tickers
        
        # Add more exchange inference logic as needed
        return None
    
    def _reformat_date(self, date_str: str) -> Optional[str]:
        """Try to reformat date to YYYY-MM-DD"""
        if not date_str:
            return None
        
        # Try common formats
        formats = [
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%m-%d-%Y",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        return None
    
    # ========================================================================
    # Level 2: Tool Fallback
    # ========================================================================
    
    def _try_tool_fallback(
        self,
        failed_tool: str,
        error_info: ErrorInfo,
        execution_history: ExecutionHistory
    ) -> ReplanDecision:
        """
        Level 2: Try to use fallback tool
        
        Note: Currently fallback registry is mostly empty.
        This provides the structure for future implementation.
        """
        # Get fallback chain for this tool
        fallback_chain = TOOL_FALLBACK_REGISTRY.get(failed_tool, [])
        
        if not fallback_chain:
            self.logger.info(f"[REPLAN:L2] No fallback defined for {failed_tool}")
            return ReplanDecision(
                action=ReplanAction.NO_ACTION,
                level=2,
                reasoning=f"No fallback tool available for {failed_tool}"
            )
        
        # Find first fallback that hasn't failed
        failed_tools_set = {t for t, _ in execution_history.failed_tools}
        
        for fallback_tool in fallback_chain:
            if fallback_tool and fallback_tool not in failed_tools_set:
                self.logger.info(f"[REPLAN:L2] Using fallback: {fallback_tool}")
                
                return ReplanDecision(
                    action=ReplanAction.TOOL_FALLBACK,
                    level=2,
                    fallback_tool=fallback_tool,
                    fallback_params=error_info.params,  # Same params
                    reasoning=f"Using fallback tool {fallback_tool} instead of {failed_tool}",
                    confidence=0.7
                )
        
        # All fallbacks exhausted or failed
        self.logger.warning(f"[REPLAN:L2] All fallbacks exhausted for {failed_tool}")
        return ReplanDecision(
            action=ReplanAction.NO_ACTION,
            level=2,
            reasoning="All fallback tools exhausted or previously failed"
        )
    
    # ========================================================================
    # Level 3: Full Replanning
    # ========================================================================
    
    async def _try_full_replan(
        self,
        error_info: ErrorInfo,
        original_plan: Any,
        execution_history: ExecutionHistory,
        user_query: str,
        detected_language: str
    ) -> ReplanDecision:
        """
        Level 3: Full replanning - create new plan
        
        Handles:
        - Symbol corrections (typos)
        - Data availability issues
        - Scope adjustments
        """
        
        # =====================================================================
        # CASE 1: Symbol Error - Try to correct symbol
        # =====================================================================
        if error_info.category == ErrorCategory.SYMBOL_ERROR:
            invalid_symbol = error_info.params.get("symbol", "")
            
            # Try to find correct symbol
            corrected_symbol = self._try_correct_symbol(invalid_symbol, user_query)
            
            if corrected_symbol and corrected_symbol != invalid_symbol:
                self.logger.info(
                    f"[REPLAN:L3] Symbol correction: {invalid_symbol} → {corrected_symbol}"
                )
                
                # Create new plan with corrected symbol
                if self.planning_agent:
                    try:
                        # Modify query with correct symbol
                        corrected_query = self._replace_symbol_in_query(
                            user_query, invalid_symbol, corrected_symbol
                        )
                        
                        # Create new plan
                        new_plan = await self.planning_agent.think_and_plan(
                            query=corrected_query,
                            recent_chat=[],
                            core_memory={},
                            summary=None
                        )
                        
                        # Create user message
                        user_msg_en = (
                            f"I understood you meant '{corrected_symbol}' "
                            f"instead of '{invalid_symbol}'. Analyzing now..."
                        )
                        user_msg_vi = (
                            f"Tôi hiểu bạn muốn nói '{corrected_symbol}' "
                            f"thay vì '{invalid_symbol}'. Đang phân tích..."
                        )
                        
                        return ReplanDecision(
                            action=ReplanAction.FULL_REPLAN,
                            level=3,
                            new_plan=new_plan,
                            corrected_symbols=[corrected_symbol],
                            user_message=user_msg_en if detected_language != "vi" else user_msg_vi,
                            user_message_vi=user_msg_vi,
                            reasoning=f"Corrected symbol {invalid_symbol} → {corrected_symbol}",
                            confidence=0.85
                        )
                        
                    except Exception as e:
                        self.logger.error(f"[REPLAN:L3] Failed to create new plan: {e}")
            
            # Cannot correct - suggest alternatives
            suggestions = self._get_similar_symbols(invalid_symbol)
            
            if suggestions:
                user_msg_en = (
                    f"Symbol '{invalid_symbol}' not found. "
                    f"Did you mean: {', '.join(suggestions)}?"
                )
                user_msg_vi = (
                    f"Không tìm thấy mã '{invalid_symbol}'. "
                    f"Bạn có ý nói: {', '.join(suggestions)}?"
                )
            else:
                user_msg_en = f"Symbol '{invalid_symbol}' not found. Please check the ticker symbol."
                user_msg_vi = f"Không tìm thấy mã '{invalid_symbol}'. Vui lòng kiểm tra lại mã cổ phiếu."
            
            return ReplanDecision(
                action=ReplanAction.GRACEFUL_DEGRADATION,
                level=3,
                should_continue=False,
                user_message=user_msg_en if detected_language != "vi" else user_msg_vi,
                user_message_vi=user_msg_vi,
                reasoning=f"Cannot correct symbol {invalid_symbol}"
            )
        
        # =====================================================================
        # CASE 2: Data Error - Adjust scope or inform user
        # =====================================================================
        if error_info.category == ErrorCategory.DATA_ERROR:
            # Check if it's a future data request
            if self._is_future_data_request(user_query):
                user_msg_en = (
                    "The requested data is not yet available (future period). "
                    "I can show you the latest available data instead."
                )
                user_msg_vi = (
                    "Dữ liệu yêu cầu chưa có (thời điểm trong tương lai). "
                    "Tôi có thể hiển thị dữ liệu mới nhất hiện có."
                )
                
                # Could create new plan with adjusted timeframe
                # For now, graceful degradation
                return ReplanDecision(
                    action=ReplanAction.GRACEFUL_DEGRADATION,
                    level=3,
                    should_continue=True,  # Continue with other tasks
                    user_message=user_msg_en if detected_language != "vi" else user_msg_vi,
                    user_message_vi=user_msg_vi,
                    reasoning="Future data requested - using latest available"
                )
            
            # Generic data unavailable
            user_msg_en = "Some requested data is not available. Showing available information."
            user_msg_vi = "Một số dữ liệu yêu cầu không có sẵn. Hiển thị thông tin có thể cung cấp."
            
            return ReplanDecision(
                action=ReplanAction.GRACEFUL_DEGRADATION,
                level=3,
                should_continue=True,
                user_message=user_msg_en if detected_language != "vi" else user_msg_vi,
                user_message_vi=user_msg_vi,
                reasoning="Data not available"
            )
        
        # No specific handling - graceful degradation
        return ReplanDecision(
            action=ReplanAction.NO_ACTION,
            level=3,
            reasoning="No specific Level 3 handling for this error"
        )
    
    def _try_correct_symbol(self, invalid_symbol: str, query: str) -> Optional[str]:
        """
        Try to correct an invalid symbol
        
        Strategies:
        1. Direct typo correction (NVIDDA -> NVDA)
        2. Company name extraction (nvidia -> NVDA)
        3. Fuzzy matching
        """
        upper_symbol = invalid_symbol.upper()
        
        # Strategy 1: Direct correction from known typos
        if upper_symbol in SYMBOL_CORRECTIONS:
            return SYMBOL_CORRECTIONS[upper_symbol]
        
        # Strategy 2: Check if query contains company name
        query_lower = query.lower()
        for company_name, symbol in COMPANY_TO_SYMBOL.items():
            if company_name in query_lower:
                return symbol
        
        # Strategy 3: Fuzzy matching (simple edit distance)
        # Find symbols with small edit distance
        similar = self._find_similar_symbols(upper_symbol)
        if similar:
            return similar[0]  # Return most similar
        
        return None
    
    def _find_similar_symbols(self, symbol: str, max_distance: int = 2) -> List[str]:
        """Find similar symbols using edit distance"""
        # Common US stock symbols
        common_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
            "AMD", "INTC", "QCOM", "AVGO", "NFLX", "ADBE", "CRM",
            "PYPL", "DIS", "BA", "KO", "PEP", "WMT", "COST", "NKE",
            "V", "MA", "JPM", "GS", "BRK.B", "JNJ", "PFE", "MRK"
        ]
        
        similar = []
        for known_symbol in common_symbols:
            distance = self._edit_distance(symbol, known_symbol)
            if distance <= max_distance:
                similar.append((known_symbol, distance))
        
        # Sort by distance
        similar.sort(key=lambda x: x[1])
        return [s[0] for s in similar]
    
    def _get_similar_symbols(self, symbol: str) -> List[str]:
        """Get list of similar symbols for suggestions"""
        return self._find_similar_symbols(symbol, max_distance=2)[:3]
    
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _replace_symbol_in_query(
        self,
        query: str,
        old_symbol: str,
        new_symbol: str
    ) -> str:
        """Replace symbol in query while preserving context"""
        # Case-insensitive replacement
        pattern = re.compile(re.escape(old_symbol), re.IGNORECASE)
        return pattern.sub(new_symbol, query)
    
    def _is_future_data_request(self, query: str) -> bool:
        """Check if query is asking for future data"""
        future_indicators = [
            "2025", "2026", "2027",  # Future years
            "next quarter", "next year",
            "quý tới", "năm tới", "tương lai",
            "forecast", "prediction", "dự báo"
        ]
        
        query_lower = query.lower()
        return any(ind in query_lower for ind in future_indicators)
    
    # ========================================================================
    # Graceful Degradation
    # ========================================================================
    
    def _create_graceful_degradation(
        self,
        error_info: ErrorInfo,
        detected_language: str,
        reason: str
    ) -> ReplanDecision:
        """
        Create graceful degradation response
        
        Philosophy: Use available data, inform user of limitations
        """
        # Generic messages by error category
        messages = {
            ErrorCategory.API_ERROR: {
                "en": "Some data sources are temporarily unavailable. Showing available information.",
                "vi": "Một số nguồn dữ liệu tạm thời không khả dụng. Hiển thị thông tin có sẵn."
            },
            ErrorCategory.RATE_LIMIT: {
                "en": "Rate limit reached. Please try again in a moment.",
                "vi": "Đã đạt giới hạn truy vấn. Vui lòng thử lại sau giây lát."
            },
            ErrorCategory.AUTH_ERROR: {
                "en": "Authentication issue. Some features may be limited.",
                "vi": "Lỗi xác thực. Một số tính năng có thể bị giới hạn."
            },
            ErrorCategory.SYMBOL_ERROR: {
                "en": f"Could not find symbol '{error_info.params.get('symbol', 'unknown')}'.",
                "vi": f"Không tìm thấy mã '{error_info.params.get('symbol', 'unknown')}'."
            },
            ErrorCategory.DATA_ERROR: {
                "en": "Requested data is not available. Showing what we have.",
                "vi": "Dữ liệu yêu cầu không có sẵn. Hiển thị những gì có thể."
            },
            ErrorCategory.UNKNOWN: {
                "en": "An unexpected error occurred. Continuing with available data.",
                "vi": "Đã xảy ra lỗi không mong muốn. Tiếp tục với dữ liệu có sẵn."
            }
        }
        
        msg = messages.get(error_info.category, messages[ErrorCategory.UNKNOWN])
        
        return ReplanDecision(
            action=ReplanAction.GRACEFUL_DEGRADATION,
            level=0,
            should_continue=error_info.category not in [
                ErrorCategory.SYMBOL_ERROR,
                ErrorCategory.AUTH_ERROR
            ],
            user_message=msg["en"] if detected_language != "vi" else msg["vi"],
            user_message_vi=msg["vi"],
            reasoning=reason,
            confidence=0.5
        )
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get replanning statistics"""
        total = self.stats["total_replans"]
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            "level_1_success_rate": self.stats["level_1_success"] / total,
            "level_2_success_rate": self.stats["level_2_success"] / total,
            "level_3_success_rate": self.stats["level_3_success"] / total,
            "recovery_rate": (
                self.stats["level_1_success"] + 
                self.stats["level_2_success"] + 
                self.stats["level_3_success"]
            ) / total
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            "total_replans": 0,
            "level_1_success": 0,
            "level_2_success": 0,
            "level_3_success": 0,
            "graceful_degradations": 0
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_error_info(
    tool_name: str,
    error_message: str,
    params: Dict[str, Any] = None,
    error_code: str = None
) -> ErrorInfo:
    """Helper to create ErrorInfo instance"""
    return ErrorInfo(
        tool_name=tool_name,
        error_message=error_message,
        error_code=error_code,
        params=params or {}
    )


def create_execution_history() -> ExecutionHistory:
    """Helper to create empty ExecutionHistory"""
    return ExecutionHistory()