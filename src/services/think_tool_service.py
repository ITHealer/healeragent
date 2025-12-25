import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.reasoning.think_tool import ThinkTool, ThinkToolPrompts
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType


class ThinkToolService:
    """
    Service for integrating Think Tool into agent workflows
    
    Provides methods to inject structured thinking at critical points:
    - Before planning (understand query intent)
    - After tool outputs (analyze results)
    - Before actions (verify compliance)
    - On errors (recovery planning)
    """
    
    def __init__(
        self,
        enabled: bool = False,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        auto_inject_prompts: bool = True
    ):
        """
        Initialize ThinkToolService
        
        Args:
            enabled: Whether think tool is enabled
            model_name: Model for generating thoughts (if LLM-assisted)
            provider_type: LLM provider
            auto_inject_prompts: Auto-inject thinking prompts in system
        """
        self.enabled = enabled
        self.model_name = model_name
        self.provider_type = provider_type
        self.auto_inject_prompts = auto_inject_prompts
        
        self.logger = logging.getLogger(__name__)
        self.think_tool = ThinkTool()
        self.llm_provider = LLMGeneratorProvider()
        
        # Track thinking calls
        self.think_calls: List[Dict[str, Any]] = []
        
        if enabled:
            self.logger.info("[THINK SERVICE] Think Tool enabled")
    
    # ========================================================================
    # SYSTEM PROMPT ENHANCEMENT
    # ========================================================================
    
    def get_enhanced_system_prompt(
        self,
        base_prompt: str,
        domain: str = "financial"
    ) -> str:
        """
        Enhance system prompt with think tool instructions
        
        Args:
            base_prompt: Original system prompt
            domain: Domain for specialized prompts
            
        Returns:
            Enhanced system prompt with think instructions
        """
        if not self.enabled or not self.auto_inject_prompts:
            return base_prompt
        
        # Get domain-specific think instructions
        if domain == "financial":
            think_instructions = ThinkToolPrompts.get_financial_analysis_prompt()
        else:
            think_instructions = self._get_generic_think_instructions()
        
        # Append to base prompt
        enhanced = f"""{base_prompt}

{think_instructions}
"""
        
        self.logger.debug("[THINK SERVICE] Enhanced system prompt with think instructions")
        
        return enhanced
    
    def _get_generic_think_instructions(self) -> str:
        """Generic think tool instructions"""
        return """
## Using the Think Tool

You have access to a "think" tool. Use it to pause and reason about complex situations.

When to use think:
- After receiving tool results that need analysis
- Before taking actions with significant consequences
- When navigating complex policies or rules
- When resolving ambiguous references

Example:
<think_example>
Received tool output: {data}
- Key data points: ...
- Quality assessment: ...
- Next steps: ...
- Potential issues: ...
</think_example>
"""
    
    # ========================================================================
    # THINKING INTEGRATION POINTS
    # ========================================================================
    
    async def think_before_planning(
        self,
        query: str,
        history_context: str = "",
        symbols_in_context: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate thought before planning phase
        
        Use for:
        - Understanding query intent
        - Resolving symbol references
        - Identifying required data
        
        Returns:
            Thought result or None if disabled
        """
        if not self.enabled:
            return None
        
        thought_content = f"""
Analyzing query before planning:

QUERY: "{query}"

CONTEXT SYMBOLS: {symbols_in_context or []}

ANALYSIS:
1. Query Type: [What type of query is this?]
2. Symbols Needed: [Which symbols need to be resolved?]
3. Reference Resolution: [Any "it/that/nó" references?]
4. Required Data: [What data is needed to answer?]
5. Potential Challenges: [Any complexities?]
"""
        
        result = await self.think_tool.execute(
            thought=thought_content,
            reasoning_type="planning"
        )
        
        self._record_think_call("planning", thought_content, result)
        
        return result
    
    async def think_about_tool_output(
        self,
        tool_name: str,
        tool_output: Dict[str, Any],
        expected_data: List[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate thought after receiving tool output
        
        Use for:
        - Validating output quality
        - Planning next steps
        - Identifying issues
        
        Returns:
            Thought result or None if disabled
        """
        if not self.enabled:
            return None
        
        status = tool_output.get("status", "unknown")
        data = tool_output.get("data", {})
        
        thought_content = f"""
Analyzing tool output:

TOOL: {tool_name}
STATUS: {status}
EXPECTED DATA: {expected_data or []}

OUTPUT ANALYSIS:
1. Status Check: [Is execution successful?]
2. Data Quality: [Is data complete and valid?]
3. Key Metrics: [Important data points]
4. Missing Data: [What's missing if any?]
5. Next Action: [What should happen next?]

DATA SUMMARY:
{self._summarize_data(data)}
"""
        
        result = await self.think_tool.execute(
            thought=thought_content,
            reasoning_type="tool_analysis"
        )
        
        self._record_think_call("tool_analysis", thought_content, result)
        
        return result
    
    async def think_for_symbol_resolution(
        self,
        query: str,
        history_symbols: List[str],
        reference_words: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate thought for symbol resolution
        
        Use when query contains reference words like:
        - "it", "that", "this stock"
        - "nó", "cổ phiếu đó", "mã này"
        
        Returns:
            Thought result with resolved symbol
        """
        if not self.enabled:
            return None
        
        thought_content = f"""
Resolving symbol reference:

QUERY: "{query}"
REFERENCE WORDS FOUND: {reference_words}
SYMBOLS IN HISTORY: {history_symbols}

RESOLUTION PROCESS:
1. Reference Words: {reference_words}
2. Recent Symbols: {history_symbols[-3:] if history_symbols else []}
3. Most Likely Symbol: [Based on recency and context]
4. Confidence Level: [High/Medium/Low]
5. Verification: [Does this make sense in context?]
"""
        
        result = await self.think_tool.execute(
            thought=thought_content,
            reasoning_type="symbol_resolution"
        )
        
        self._record_think_call("symbol_resolution", thought_content, result)
        
        return result
    
    async def think_for_error_recovery(
        self,
        error_type: str,
        error_message: str,
        failed_tool: str,
        attempted_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate thought for error recovery
        
        Use when:
        - Tool execution fails
        - Validation fails
        - Unexpected results
        
        Returns:
            Thought result with recovery plan
        """
        if not self.enabled:
            return None
        
        thought_content = f"""
Planning error recovery:

ERROR TYPE: {error_type}
ERROR MESSAGE: {error_message}
FAILED TOOL: {failed_tool}
ATTEMPTED PARAMS: {attempted_params}

RECOVERY ANALYSIS:
1. Error Category: [Transient/Permanent/Validation]
2. Root Cause: [Why did this fail?]
3. Retry Viable: [Should we retry?]
4. Alternative Approach: [What else can we try?]
5. User Communication: [What to tell user?]
"""
        
        result = await self.think_tool.execute(
            thought=thought_content,
            reasoning_type="error_recovery"
        )
        
        self._record_think_call("error_recovery", thought_content, result)
        
        return result
    
    async def think_before_synthesis(
        self,
        query: str,
        all_tool_results: Dict[str, Any],
        symbols_analyzed: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate thought before final synthesis
        
        Use to:
        - Review all gathered data
        - Identify key insights
        - Plan response structure
        
        Returns:
            Thought result for synthesis guidance
        """
        if not self.enabled:
            return None
        
        successful_tools = [
            name for name, result in all_tool_results.items()
            if result.get("status") == "200"
        ]
        
        thought_content = f"""
Preparing synthesis:

ORIGINAL QUERY: "{query}"
SYMBOLS ANALYZED: {symbols_analyzed}
SUCCESSFUL TOOLS: {successful_tools}

SYNTHESIS PLANNING:
1. Query Coverage: [Did we answer the question?]
2. Key Findings: [Top 3-5 insights]
3. Data Conflicts: [Any contradictory data?]
4. Response Structure: [How to organize answer?]
5. Recommendations: [What to suggest?]
"""
        
        result = await self.think_tool.execute(
            thought=thought_content,
            reasoning_type="planning"
        )
        
        self._record_think_call("synthesis_planning", thought_content, result)
        
        return result
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """Summarize data for thought context"""
        if not data:
            return "No data available"
        
        lines = []
        for key, value in list(data.items())[:10]:  # Limit to 10 items
            if isinstance(value, (dict, list)):
                lines.append(f"- {key}: [complex data]")
            else:
                lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)
    
    def _record_think_call(
        self,
        reasoning_type: str,
        thought: str,
        result: Any
    ):
        """Record think call for analytics"""
        self.think_calls.append({
            "timestamp": datetime.now().isoformat(),
            "reasoning_type": reasoning_type,
            "thought_preview": thought[:200],
            "success": result.status == "200" if hasattr(result, "status") else True
        })
        
        self.logger.info(
            f"[THINK SERVICE] Recorded {reasoning_type} thought "
            f"(total: {len(self.think_calls)})"
        )
    
    def get_think_stats(self) -> Dict[str, Any]:
        """Get statistics about think calls in current session"""
        if not self.think_calls:
            return {
                "total_calls": 0,
                "enabled": self.enabled
            }
        
        by_type = {}
        for call in self.think_calls:
            t = call["reasoning_type"]
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "total_calls": len(self.think_calls),
            "by_type": by_type,
            "enabled": self.enabled
        }
    
    def reset_session(self):
        """Reset think calls for new session"""
        self.think_calls = []
    
    def is_enabled(self) -> bool:
        """Check if think tool is enabled"""
        return self.enabled
    
    def set_enabled(self, enabled: bool):
        """Enable or disable think tool"""
        self.enabled = enabled
        self.logger.info(f"[THINK SERVICE] Think tool {'enabled' if enabled else 'disabled'}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_think_tool_service(
    enabled: bool = False,
    model_name: str = "gpt-4.1-nano"
) -> ThinkToolService:
    """Factory function to create ThinkToolService"""
    return ThinkToolService(
        enabled=enabled,
        model_name=model_name
    )