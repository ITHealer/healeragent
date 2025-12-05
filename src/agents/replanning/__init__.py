"""
Replanning Module - 3-Level Error Recovery System

This module provides error recovery strategies for the chatbot agent:

Level 1: Local Repair
    - Fix parameter errors (missing fields, wrong format)
    - Retry with corrected parameters

Level 2: Tool Fallback  
    - Use alternative tools when primary tool fails
    - Fallback chain support

Level 3: Full Replanning
    - Symbol correction (typos)
    - Scope adjustment
    - Create entirely new plan

Graceful Degradation:
    - Use available data when recovery fails
    - Inform user of limitations
    - Continue with remaining tasks

Usage:
    from src.agents.replanning import (
        ReplanningAgent,
        ReplanAction,
        ReplanDecision,
        ErrorInfo,
        ExecutionHistory,
        create_error_info,
        create_execution_history,
    )
    
    # Initialize
    replanning_agent = ReplanningAgent(planning_agent=planning_agent)
    
    # When tool fails
    decision = await replanning_agent.decide_recovery_strategy(
        failed_tool="getStockPrice",
        error_info=error_info,
        original_plan=plan,
        execution_history=history,
        user_query=query,
        detected_language="vi"
    )
    
    # Handle decision
    if decision.action == ReplanAction.LOCAL_REPAIR:
        # Retry with fixed params
        ...
    elif decision.action == ReplanAction.TOOL_FALLBACK:
        # Use fallback tool
        ...
    elif decision.action == ReplanAction.FULL_REPLAN:
        # Execute new plan
        ...
    else:
        # Graceful degradation
        ...
"""

from src.agents.replanning.replanning_agent import (
    # Main Agent
    ReplanningAgent,
    
    # Enums
    ReplanAction,
    ErrorCategory,
    
    # Data Models
    ReplanDecision,
    ErrorInfo,
    ExecutionHistory,
    
    # Helper Functions
    create_error_info,
    create_execution_history,
    
    # Constants (for extension)
    TOOL_FALLBACK_REGISTRY,
    SYMBOL_CORRECTIONS,
    COMPANY_TO_SYMBOL,
    ERROR_PATTERNS,
    FIXABLE_PARAM_PATTERNS,
)

__all__ = [
    # Main Agent
    "ReplanningAgent",
    
    # Enums
    "ReplanAction",
    "ErrorCategory",
    
    # Data Models
    "ReplanDecision",
    "ErrorInfo",
    "ExecutionHistory",
    
    # Helper Functions
    "create_error_info",
    "create_execution_history",
    
    # Constants
    "TOOL_FALLBACK_REGISTRY",
    "SYMBOL_CORRECTIONS",
    "COMPANY_TO_SYMBOL",
    "ERROR_PATTERNS",
    "FIXABLE_PARAM_PATTERNS",
]