"""
Base Prompt Template - Golden Principles

This module defines the core principles shared across all LLM providers,
derived from analysis of Google, OpenAI, and Claude system prompts.

7 Golden Principles:
1. Identity & Context - Model name, date, environment
2. Trustworthiness - Never fabricate, partial > clarify
3. Citation & Sources - Proper attribution, copyright respect
4. Tool Usage - Scale to complexity, don't offer without tools
5. Response Style - Natural tone, avoid over-formatting
6. Safety - Refuse harmful requests clearly
7. Web Search Decision - Framework for when to search
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum


class ResponseStyle(Enum):
    """Response style options"""
    CONCISE = "concise"       # Short, direct answers
    DETAILED = "detailed"     # Comprehensive analysis
    NARRATIVE = "narrative"   # Story-driven, flowing prose
    TECHNICAL = "technical"   # Technical accuracy focus


class AnalysisType(Enum):
    """Analysis type for finance domain"""
    BASIC = "basic"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    VALUATION = "valuation"
    PORTFOLIO = "portfolio"
    COMPARISON = "comparison"
    GENERAL = "general"


@dataclass
class PromptContext:
    """Context information for prompt generation"""
    # Core context
    current_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    language: str = "vi"  # vi, en, zh
    timezone: str = "Asia/Ho_Chi_Minh"

    # Domain context
    symbols: List[str] = field(default_factory=list)
    market_type: str = "stock"  # stock, crypto, mixed
    analysis_type: str = "general"
    categories: List[str] = field(default_factory=list)

    # User context
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    user_profile: Optional[str] = None
    conversation_summary: Optional[str] = None

    # Tool context
    available_tools: List[str] = field(default_factory=list)
    enable_web_search: bool = False
    enable_tool_search: bool = False
    enable_think_tool: bool = False

    # Style context
    response_style: ResponseStyle = ResponseStyle.DETAILED
    min_words: int = 500
    max_words: int = 1500


class BasePromptTemplate(ABC):
    """
    Base class for all prompt templates.

    Implements the 7 Golden Principles shared across all LLM providers.
    Subclasses implement provider-specific optimizations.
    """

    PROVIDER_NAME: str = "base"
    SUPPORTED_MODELS: List[str] = []

    # =========================================================================
    # GOLDEN PRINCIPLE 1: Identity & Context
    # =========================================================================

    def get_identity_block(self, context: PromptContext) -> str:
        """
        Returns model identity and context information.
        Override in subclasses for provider-specific identity.
        """
        return f"""Current date: {context.current_date}
Timezone: {context.timezone}
Language: {context.language}"""

    # =========================================================================
    # GOLDEN PRINCIPLE 2: Trustworthiness
    # =========================================================================

    def get_trustworthiness_block(self) -> str:
        """
        Returns trustworthiness guidelines - shared across all providers.
        """
        return """## Trustworthiness Guidelines

CRITICAL REQUIREMENTS:
- NEVER fabricate information, data, or statistics
- If unsure, acknowledge uncertainty rather than guess
- Partial completion is MUCH better than asking clarifying questions
- Use tools to verify information before making claims
- Be honest about limitations and knowledge boundaries

DATA INTEGRITY:
- USE ALL DATA from tool results - do not cherry-pick
- CITE EXACT NUMBERS as they appear in data
- NO FABRICATION - only use numbers that exist in tool results
- SOURCE ATTRIBUTION - reference which tool provided each data point"""

    # =========================================================================
    # GOLDEN PRINCIPLE 3: Citation & Sources
    # =========================================================================

    def get_citation_block(self, enable_web_search: bool = False) -> str:
        """
        Returns citation guidelines.
        More detailed when web search is enabled.
        """
        base_citation = """## Citation Guidelines

When providing information from tools or searches:
- Reference the source of each data point
- Use specific numbers and dates from the data
- Distinguish between factual data and your analysis/interpretation"""

        if enable_web_search:
            web_citation = """

WEB SEARCH SOURCES (MANDATORY when web search is used):
At the end of your response, include a "📚 Sources" section listing ALL web sources:

## 📚 Sources
- [Title 1](URL1)
- [Title 2](URL2)
- [Title 3](URL3)

REQUIREMENTS:
- Every web search citation MUST be included
- Use markdown link format: [Title](URL)
- Include the actual page title, not generic descriptions
- List sources in order of relevance"""
        else:
            web_citation = ""

        return base_citation + web_citation

    # =========================================================================
    # GOLDEN PRINCIPLE 4: Tool Usage
    # =========================================================================

    def get_tool_usage_block(self, context: PromptContext) -> str:
        """
        Returns tool usage guidelines based on context.
        """
        base_tools = """## Tool Usage Guidelines

GENERAL PRINCIPLES:
- Call tools FIRST to get actual data before making claims
- Scale tool calls to query complexity (1 for simple, 5+ for complex)
- Do not offer tasks that require tools you don't have access to
- Combine tools effectively for comprehensive analysis"""

        additions = []

        if context.enable_tool_search:
            additions.append("""
TOOL SEARCH MODE (ENABLED):
1. FIRST: Call tool_search to discover available tools for the query
2. THEN: Call ALL relevant discovered tools in ONE TURN (parallel execution)
3. FINALLY: Synthesize results and respond

❌ WRONG (Sequential - SLOW):
   Turn 1: tool_search → discover tools
   Turn 2: getTechnicalIndicators(AAPL)
   Turn 3: getFinancialRatios(AAPL)

✅ CORRECT (Parallel - FAST):
   Turn 1: tool_search → discover tools
   Turn 2: [getTechnicalIndicators(AAPL), getFinancialRatios(AAPL), getStockNews(AAPL)]""")

        if context.enable_think_tool:
            additions.append("""
THINK TOOL (ENABLED - MANDATORY):
Use the think tool to plan and analyze:

1. think(reasoning_type="planning") → outline your approach BEFORE calling tools
2. Call relevant tools → gather data
3. think(reasoning_type="analyzing") → analyze results and formulate response
4. Provide comprehensive response""")

        if context.enable_web_search:
            additions.append("""
WEB SEARCH (ENABLED):
- Use webSearch for latest news, current prices, recent events
- Search when information may have changed since knowledge cutoff
- Always include sources in your response""")

        return base_tools + "\n".join(additions)

    # =========================================================================
    # GOLDEN PRINCIPLE 5: Response Style
    # =========================================================================

    def get_response_style_block(self, context: PromptContext) -> str:
        """
        Returns response style guidelines based on context.
        """
        style_map = {
            ResponseStyle.CONCISE: """## Response Style: CONCISE
- Keep responses short and direct
- Focus on key points only
- Target: 200-400 words
- Use bullet points sparingly
- Lead with the most important information""",

            ResponseStyle.DETAILED: f"""## Response Style: DETAILED
- Provide comprehensive analysis
- Cover multiple aspects of the topic
- Target: {context.min_words}-{context.max_words} words
- Use structured sections when helpful
- Include supporting data and context""",

            ResponseStyle.NARRATIVE: """## Response Style: NARRATIVE
- Write in flowing, story-driven prose
- Connect ideas naturally without excessive headers
- Target: 500-800 words
- Tell a story, don't dump data
- Highlight KEY metrics only, weave them into narrative""",

            ResponseStyle.TECHNICAL: """## Response Style: TECHNICAL
- Focus on technical accuracy
- Use proper terminology
- Include specific numbers and calculations
- Reference methodologies and frameworks
- Target: 600-1000 words"""
        }

        base_style = style_map.get(context.response_style, style_map[ResponseStyle.DETAILED])

        # Add universal style guidelines
        universal = """

UNIVERSAL STYLE RULES:
- Match the user's language (vi/en/zh)
- Avoid excessive formatting (headers, bullets) for simple questions
- Show don't tell - never explain compliance to instructions
- Do NOT praise user questions ("Great question!", "Love this!")
- Go straight into your answer
- Be warm but professional"""

        return base_style + universal

    # =========================================================================
    # GOLDEN PRINCIPLE 6: Safety
    # =========================================================================

    def get_safety_block(self) -> str:
        """
        Returns safety guidelines - shared across all providers.
        """
        return """## Safety Guidelines

CONTENT BOUNDARIES:
- Refuse requests for harmful financial advice that could cause significant loss
- Do not provide specific investment recommendations without proper caveats
- Never fabricate financial data or predictions
- Always include risk disclaimers for investment-related advice

HARMFUL CONTENT:
- Do not search for or reference sources promoting fraud or scams
- Refuse to help with market manipulation strategies
- Do not provide advice designed to circumvent regulations

WHEN REFUSING:
- Explain clearly why you cannot help
- Suggest safer alternatives when possible
- Maintain a helpful, professional tone"""

    # =========================================================================
    # GOLDEN PRINCIPLE 7: Web Search Decision Framework
    # =========================================================================

    def get_search_decision_block(self) -> str:
        """
        Returns web search decision framework.
        """
        return """## Web Search Decision Framework

ALWAYS SEARCH FOR:
- Current prices, rates, or values
- Recent news or events (within past month)
- Current holders of positions (CEO, President, etc.)
- Information that may have changed since knowledge cutoff
- Real-time market data

NEVER SEARCH FOR:
- Historical facts that don't change
- Fundamental concepts or definitions
- Well-established technical knowledge
- Information already provided by specialized tools

SEARCH QUALITY:
- Use concise queries (1-6 words work best)
- Start broad, then narrow if needed
- Prefer authoritative sources
- Cross-reference when information conflicts"""

    # =========================================================================
    # Abstract Methods - Provider-Specific Implementation
    # =========================================================================

    @abstractmethod
    def get_system_prompt(self, context: PromptContext) -> str:
        """
        Returns the complete system prompt for this provider.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_model_identity(self, model_name: str) -> str:
        """
        Returns model-specific identity string.
        Must be implemented by subclasses.
        """
        pass

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def compose_prompt(
        self,
        context: PromptContext,
        domain_prompt: str = "",
        analysis_framework: str = "",
        additional_instructions: str = ""
    ) -> str:
        """
        Composes a complete system prompt from components.
        """
        sections = []

        # 1. Identity & Context
        sections.append(self.get_identity_block(context))

        # 2. Domain-specific prompt (if provided)
        if domain_prompt:
            sections.append(domain_prompt)

        # 3. Trustworthiness
        sections.append(self.get_trustworthiness_block())

        # 4. Tool Usage
        if context.available_tools or context.enable_tool_search:
            sections.append(self.get_tool_usage_block(context))

        # 5. Citation
        sections.append(self.get_citation_block(context.enable_web_search))

        # 6. Response Style
        sections.append(self.get_response_style_block(context))

        # 7. Analysis Framework (if provided)
        if analysis_framework:
            sections.append(f"## Analysis Framework\n{analysis_framework}")

        # 8. Additional Instructions (if provided)
        if additional_instructions:
            sections.append(additional_instructions)

        # 9. Safety (always last before user context)
        sections.append(self.get_safety_block())

        # 10. User context (if available)
        if context.user_profile or context.conversation_summary:
            user_section = self._format_user_context(context)
            if user_section:
                sections.append(user_section)

        return "\n\n---\n\n".join(sections)

    def _format_user_context(self, context: PromptContext) -> str:
        """Formats user context section"""
        parts = []

        if context.user_profile:
            parts.append(f"""<USER_PROFILE>
{context.user_profile}
</USER_PROFILE>""")

        if context.conversation_summary:
            parts.append(f"""<CONVERSATION_SUMMARY>
{context.conversation_summary}
</CONVERSATION_SUMMARY>""")

        return "\n\n".join(parts) if parts else ""

    def get_synthesis_instructions(self, context: PromptContext, tool_results: str) -> str:
        """
        Returns instructions for synthesizing tool results into response.
        """
        return f"""## Synthesis Task

You have received data from various tools. Your task is to synthesize this information into a comprehensive response.

### Tool Results Data
{tool_results}

### Requirements

1. **USE ALL DATA**: Every relevant data point from tools MUST be included
2. **CITE SPECIFIC NUMBERS**: Quote exact values, don't round or approximate
3. **EXPLAIN SIGNIFICANCE**: What does each metric mean for the user?
4. **PROVIDE CONTEXT**: Compare to benchmarks, historical ranges
5. **IDENTIFY PATTERNS**: Connect data points to reveal trends

### Response Length
Target: {context.min_words}-{context.max_words} words

### Actionable Output
- Provide specific recommendations with levels/targets
- Include risk assessment
- Offer both short-term and long-term perspectives
- State what to monitor going forward"""
