import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from functools import lru_cache

from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# TEMPLATE LOADER
# ============================================================================

TEMPLATES_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=20)
def load_template(template_name: str) -> str:
    """
    Load template from file with caching.

    Args:
        template_name: Name of template file (without .txt extension)

    Returns:
        Template string content
    """
    template_path = TEMPLATES_DIR / f"{template_name}.txt"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def clear_template_cache():
    """Clear the template cache (useful for development/testing)"""
    load_template.cache_clear()


# ============================================================================
# PROMPT BUILDER CLASS
# ============================================================================

class PlanningPromptBuilder(LoggerMixin):
    """
    Builder class for Planning Agent prompts.

    Usage:
        builder = PlanningPromptBuilder()
        prompt = builder.build_classify_prompt(
            query="giá AAPL",
            history_text="...",
            capability="basic"
        )
    """

    def __init__(self):
        super().__init__()
        self._validate_templates()

    def _validate_templates(self):
        """Validate that all required templates exist"""
        required = [
            "planning_classify_basic",
            "planning_classify_intermediate",
            "planning_classify_advanced",
            "planning_thinking_validate",
            "planning_stage3_create_plan"
        ]

        for template_name in required:
            try:
                load_template(template_name)
            except FileNotFoundError as e:
                self.logger.warning(f"Template not found: {template_name}")

    # ========================================================================
    # DATE CONTEXT
    # ========================================================================

    def get_date_context(self) -> str:
        """Get current date context string"""
        from datetime import datetime

        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M UTC")

        return f"""<current_context>
Date: {current_date} {current_time}
Data Status: Real-time market data available through tools
Note: Tools fetch LIVE data from financial APIs
</current_context>"""

    # ========================================================================
    # CLASSIFICATION PROMPTS (STAGE 1)
    # ========================================================================

    def build_classify_prompt(
        self,
        query: str,
        history_text: str,
        capability: str = "basic",
        working_memory_context: Optional[str] = None,
        core_memory: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build classification prompt for Stage 1.

        Args:
            query: User query
            history_text: Formatted chat history
            capability: Model capability level (basic, intermediate, advanced)
            working_memory_context: Working memory context string
            core_memory: Core memory dict

        Returns:
            Formatted prompt string
        """
        # Select template based on capability
        template_name = f"planning_classify_{capability}"

        try:
            template = load_template(template_name)
        except FileNotFoundError:
            self.logger.warning(f"Template {template_name} not found, using basic")
            template = load_template("planning_classify_basic")

        # Build sections
        wm_section = self._build_working_memory_section(
            working_memory_context, capability
        )
        core_memory_section = self._build_core_memory_section(
            core_memory, capability
        )

        # Format template
        return template.format(
            date_context=self.get_date_context(),
            query=query,
            history_text=history_text,
            wm_section=wm_section,
            core_memory_section=core_memory_section
        )

    def _build_working_memory_section(
        self,
        working_memory_context: Optional[str],
        capability: str
    ) -> str:
        """Build working memory section based on capability"""
        if not working_memory_context:
            return ""

        if capability == "basic":
            return f"""
WORKING MEMORY (current task state, symbols from previous turns):
{working_memory_context}

Use this context to understand ongoing tasks and symbol references.
"""
        else:
            return f"""
<working_memory>
{working_memory_context}
</working_memory>
"""

    def _build_core_memory_section(
        self,
        core_memory: Optional[Dict[str, Any]],
        capability: str
    ) -> str:
        """Build core memory section based on capability"""
        if not core_memory:
            return ""

        human_block = core_memory.get('human', '')
        if not human_block:
            return ""

        # Limit length
        max_length = 1200 if capability == "advanced" else 1000
        human_truncated = human_block[:max_length] if len(human_block) > max_length else human_block

        if capability == "basic":
            return f"""
USER PROFILE (from Core Memory):
{human_truncated}

SYMBOL RESOLUTION FROM USER PROFILE:
When user says:
- "my stocks", "các cổ phiếu của tôi", "cổ phiếu yêu thích" → extract symbols from watchlist/portfolio above
- "tài khoản", "portfolio" → use portfolio symbols
- References to investments without specific tickers → check user profile

If symbols are found in user profile, include them in the "symbols" output.
"""
        elif capability == "intermediate":
            return f"""
<user_profile>
{human_truncated}
</user_profile>

<user_profile_instructions>
Use this to understand user's watchlist, portfolio, and preferences.
When user references "my stocks", "cổ phiếu của tôi", "favorites" → extract symbols from profile.
</user_profile_instructions>
"""
        else:  # advanced
            return f"""
<user_profile source="core_memory">
{human_truncated}
</user_profile>
"""

    # ========================================================================
    # THINKING VALIDATION PROMPT
    # ========================================================================

    def build_thinking_prompt(
        self,
        query: str,
        query_type: str,
        categories: List[str],
        symbols: List[str],
        reasoning: str
    ) -> str:
        """
        Build thinking validation prompt.

        Args:
            query: Original query
            query_type: Classified query type
            categories: Selected categories
            symbols: Detected symbols
            reasoning: Classification reasoning

        Returns:
            Formatted prompt string
        """
        template = load_template("planning_thinking_validate")

        return template.format(
            date_context=self.get_date_context(),
            query=query,
            query_type=query_type,
            categories=categories,
            symbols=symbols,
            reasoning=reasoning
        )

    # ========================================================================
    # TASK PLANNING PROMPT (STAGE 3)
    # ========================================================================

    def build_plan_prompt(
        self,
        query: str,
        query_type: str,
        symbols: List[str],
        response_language: str,
        validated_intent: str,
        tools_text: str
    ) -> str:
        """
        Build task planning prompt for Stage 3.

        Args:
            query: User query
            query_type: Classified query type
            symbols: Detected symbols
            response_language: Target response language
            validated_intent: Validated user intent
            tools_text: Formatted tools description

        Returns:
            Formatted prompt string
        """
        template = load_template("planning_stage3_create_plan")

        symbols_display = symbols if symbols else "None - may need screening"
        symbols_json = json.dumps(symbols) if symbols else "[]"

        return template.format(
            date_context=self.get_date_context(),
            query=query,
            query_type=query_type,
            symbols_display=symbols_display,
            symbols_json=symbols_json,
            response_language=response_language,
            validated_intent=validated_intent,
            tools_text=tools_text
        )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_prompt_builder: Optional[PlanningPromptBuilder] = None


def get_planning_prompt_builder() -> PlanningPromptBuilder:
    """Get singleton PlanningPromptBuilder instance"""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PlanningPromptBuilder()
    return _prompt_builder


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_classify_prompt(**kwargs) -> str:
    """Convenience function for building classification prompt"""
    return get_planning_prompt_builder().build_classify_prompt(**kwargs)


def build_thinking_prompt(**kwargs) -> str:
    """Convenience function for building thinking prompt"""
    return get_planning_prompt_builder().build_thinking_prompt(**kwargs)


def build_plan_prompt(**kwargs) -> str:
    """Convenience function for building plan prompt"""
    return get_planning_prompt_builder().build_plan_prompt(**kwargs)