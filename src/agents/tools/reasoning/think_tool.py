# File: src/agents/tools/reasoning/think_tool.py
"""
Think Tool - Anthropic τ-Bench Pattern Implementation

PURPOSE:
The think tool allows the agent to pause and reason about:
- Tool outputs before taking next action
- Policy compliance verification
- Symbol resolution from conversation context
- Sequential decision validation

WHEN TO USE:
- Analyzing tool outputs in long chains
- Navigating policy-heavy environments
- Making sequential decisions where mistakes are costly

WHEN NOT TO USE:
- Non-sequential tool calls (single tool execution)
- Simple instruction following
- Conversational queries

Based on:
- Anthropic Research: "The think tool" (March 2025)
- τ-Bench evaluation methodology
- SWE-Bench implementation patterns

Reference: https://www.anthropic.com/research/the-think-tool
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)


class ThinkTool(BaseTool):
    """
    Think Tool - Structured Reasoning During Tool Execution
    
    This tool provides a dedicated space for the agent to:
    - Analyze tool outputs before next action
    - Verify policy compliance
    - Plan sequential steps
    - Cache intermediate reasoning
    
    CRITICAL: This tool does NOT:
    - Fetch new data
    - Modify state
    - Execute external actions
    
    It simply logs the thought and returns it for context.
    
    Usage:
        result = await tool.execute(
            thought="Analyzing AAPL data: RSI=28 (oversold)..."
        )
    """
    
    def __init__(self):
        """Initialize ThinkTool"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.schema = ToolSchema(
            name="think",
            category="reasoning",
            description=(
                "Use this tool to pause and think about the current situation. "
                "It will NOT obtain new information or change any state, "
                "but allows structured reasoning before taking action. "
                "Use when complex reasoning, policy verification, or "
                "analysis of tool outputs is needed."
            ),
            parameters=[
                ToolParameter(
                    name="thought",
                    type="string",
                    required=True,
                    description=(
                        "Your reasoning process. Include:\n"
                        "- Analysis of current tool outputs\n"
                        "- Policy/rules to verify\n"
                        "- Next steps to consider\n"
                        "- Potential issues to watch for"
                    )
                ),
                ToolParameter(
                    name="reasoning_type",
                    type="string",
                    required=False,
                    description="Type of reasoning being performed",
                    enum=[
                        "tool_analysis",      # Analyzing tool outputs
                        "policy_check",       # Verifying compliance
                        "symbol_resolution",  # Resolving stock symbols
                        "planning",           # Planning next steps
                        "error_recovery",     # Recovering from errors
                        "validation",         # Validating assumptions
                        "general"             # General reasoning
                    ],
                    default="general"
                )
            ],
            returns={
                "thought": "The reasoning that was logged",
                "reasoning_type": "Type of reasoning performed",
                "timestamp": "When the thought was recorded"
            },
            capabilities=[
                "Pause and analyze tool outputs",
                "Verify policy compliance before action",
                "Plan sequential decision steps",
                "Cache reasoning for context continuity",
                "Error analysis and recovery planning"
            ],
            limitations=[
                "Does NOT fetch new data",
                "Does NOT modify any state",
                "Does NOT execute external actions",
                "Only logs thought for context"
            ],
            usage_hints=[
                "Use after receiving tool results that need analysis",
                "Use before taking actions with significant consequences",
                "Use when navigating complex financial policies",
                "Use to verify symbol resolution in ambiguous contexts"
            ],
            requires_symbol=False,
            typical_execution_time_ms=10  # Nearly instant
        )
    
    async def execute(
        self,
        thought: str,
        reasoning_type: str = "general"
    ) -> ToolOutput:
        """
        Execute think tool - log reasoning and return
        
        Args:
            thought: The reasoning/thinking content
            reasoning_type: Category of reasoning
            
        Returns:
            ToolOutput with the logged thought
        """
        tool_name = self.schema.name
        
        # Validate input
        if not thought or len(thought.strip()) < 10:
            return create_error_output(
                tool_name=tool_name,
                error_message="Thought must be at least 10 characters for meaningful reasoning",
                error_type="validation_error"
            )
        
        # Validate reasoning type
        valid_types = [
            "tool_analysis", "policy_check", "symbol_resolution",
            "planning", "error_recovery", "validation", "general"
        ]
        if reasoning_type not in valid_types:
            reasoning_type = "general"
        
        try:
            timestamp = datetime.now().isoformat()
            
            # Log the thought
            self.logger.info(
                f"[THINK][{reasoning_type.upper()}] {thought[:200]}..."
            )
            
            # Format thought for context
            formatted_context = self._format_thought_for_context(
                thought=thought,
                reasoning_type=reasoning_type,
                timestamp=timestamp
            )
            
            return create_success_output(
                tool_name=tool_name,
                data={
                    "thought": thought,
                    "reasoning_type": reasoning_type,
                    "timestamp": timestamp,
                    "logged": True
                },
                formatted_context=formatted_context,
                symbols=[]  # Think tool doesn't return symbols
            )
            
        except Exception as e:
            self.logger.error(f"[{tool_name}] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name=tool_name,
                error_message=str(e),
                error_type="execution_error"
            )
    
    def _format_thought_for_context(
        self,
        thought: str,
        reasoning_type: str,
        timestamp: str
    ) -> str:
        """Format thought for LLM context window"""
        
        type_emoji = {
            "tool_analysis": "🔍",
            "policy_check": "📋",
            "symbol_resolution": "🎯",
            "planning": "📝",
            "error_recovery": "🔧",
            "validation": "✅",
            "general": "💭"
        }
        
        emoji = type_emoji.get(reasoning_type, "💭")
        
        return f"""
{emoji} REASONING [{reasoning_type.upper()}]:
{thought}
"""


# ============================================================================
# Think Tool Prompts for Different Domains
# ============================================================================

class ThinkToolPrompts:
    """
    Pre-built prompts for Think Tool usage in different scenarios
    
    Based on τ-Bench optimized prompts that showed 54% improvement
    in policy-heavy domains like finance.
    """
    
    @staticmethod
    def get_financial_analysis_prompt() -> str:
        """
        Optimized prompt for financial analysis scenarios
        
        Place this in system prompt when using Think Tool
        """
        return """
## Using the Think Tool for Financial Analysis

Before taking any action or responding to the user after receiving tool results,
use the think tool as a scratchpad to:

1. LIST the specific data points received from tools
2. CHECK if data quality is sufficient for analysis
3. VERIFY that the analysis complies with financial best practices
4. ITERATE over results for correctness

Here are examples of when and how to use think:

<think_example_1>
User wants to analyze AAPL stock
Tool returned: RSI=28.5, MACD=-2.1, Price=$185
- RSI 28.5 indicates oversold condition (< 30 threshold)
- Negative MACD suggests bearish momentum
- Need to check: news sentiment, fundamental ratios
- Recommendation type: Wait for confirmation before entry
- Confidence level: Medium (technicals oversold but momentum negative)
</think_example_1>

<think_example_2>
User asks for stock screening with P/E < 15
Tool returned: 5 stocks [INTC, VZ, GM, T, F]
- Verify: All P/E ratios are indeed < 15
- Check: Are these in sectors user typically invests?
- Risk assessment: Low P/E could indicate value trap
- Additional data needed: Debt/Equity, Revenue growth
- Plan: Present top 3 with key metrics
</think_example_2>

<think_example_3>
User references "that stock from earlier"
- Need to resolve: Which symbol from conversation history?
- Search history: Found NVDA mentioned in last 3 messages
- Confidence: High (NVDA is the most recent symbol)
- Verification: User context supports NVDA interpretation
</think_example_3>
"""

    @staticmethod
    def get_symbol_resolution_prompt() -> str:
        """
        Prompt for symbol resolution in multilingual contexts
        """
        return """
## Using Think Tool for Symbol Resolution

When user queries contain ambiguous stock references, use think to:

1. IDENTIFY reference words: "it", "that stock", "nó", "cổ phiếu đó"
2. SEARCH conversation history for recent symbols
3. VERIFY symbol matches user's intent
4. DOCUMENT reasoning for transparency

<think_example>
Query: "Giá của nó là bao nhiêu?" (What is its price?)
History scan:
- Turn 2: User asked about NVDA
- Turn 1: User mentioned Tesla
Reference "nó" likely refers to NVDA (most recent)
Confidence: High
Action: Query price for NVDA
</think_example>
"""

    @staticmethod
    def get_error_recovery_prompt() -> str:
        """
        Prompt for error recovery scenarios
        """
        return """
## Using Think Tool for Error Recovery

When tool execution fails or returns unexpected results:

1. ANALYZE the error type (API error, invalid symbol, rate limit)
2. DETERMINE if retry is appropriate
3. PLAN alternative approaches
4. COMMUNICATE transparently with user

<think_example>
Tool getStockPrice failed for symbol "APPL"
- Error type: Invalid symbol (likely typo)
- Correction: User probably meant "AAPL"
- Verification: AAPL exists in symbol registry
- Plan: Retry with corrected symbol
- User message: Inform about typo correction
</think_example>
"""


# ============================================================================
# Factory Function
# ============================================================================

def create_think_tool() -> ThinkTool:
    """Factory function to create ThinkTool instance"""
    return ThinkTool()