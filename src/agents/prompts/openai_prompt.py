"""
OpenAI-Optimized Prompt Template

Optimized for GPT-5, GPT-5-mini, GPT-5.1-mini models.

Key optimizations based on ChatGPT system prompt analysis:
1. Direct identity statement ("You are...")
2. Strong trustworthiness emphasis
3. No sycophantic flattery
4. Clear tool usage patterns
5. Citation format with references
6. Show-don't-tell style guidance
"""

from typing import List
from .base_prompt import BasePromptTemplate, PromptContext, ResponseStyle


class OpenAIPromptTemplate(BasePromptTemplate):
    """
    OpenAI-optimized prompt template for GPT-5 family models.

    Characteristics:
    - Direct, imperative style
    - Strong emphasis on trustworthiness and accuracy
    - Clear tool call patterns
    - No filler phrases or sycophancy
    """

    PROVIDER_NAME = "openai"
    SUPPORTED_MODELS = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
    ]

    # Model-specific configurations
    MODEL_CONFIGS = {
        "gpt-5": {
            "identity": "GPT-5",
            "capabilities": "advanced reasoning and analysis",
            "style": "comprehensive",
        },
        "gpt-5-mini": {
            "identity": "GPT-5 Mini",
            "capabilities": "efficient analysis",
            "style": "concise",
        },
        "gpt-5.1-mini": {
            "identity": "GPT-5.1 Mini",
            "capabilities": "fast and accurate analysis",
            "style": "balanced",
        },
        "gpt-4o": {
            "identity": "GPT-4o",
            "capabilities": "multimodal reasoning",
            "style": "comprehensive",
        },
        "gpt-4o-mini": {
            "identity": "GPT-4o Mini",
            "capabilities": "efficient multimodal analysis",
            "style": "concise",
        },
    }

    def get_model_identity(self, model_name: str) -> str:
        """Returns OpenAI model-specific identity"""
        config = self.MODEL_CONFIGS.get(
            model_name,
            {"identity": "GPT", "capabilities": "analysis", "style": "balanced"}
        )
        return config["identity"]

    def get_identity_block(self, context: PromptContext, model_name: str = "gpt-5") -> str:
        """
        OpenAI-style identity block.
        Uses direct "You are" statement like ChatGPT system prompt.
        """
        config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS["gpt-5"])

        return f"""You are a Senior Financial Analyst powered by {config['identity']}, specialized in {config['capabilities']}.

Current date: {context.current_date}
Language: {self._get_language_name(context.language)}

Knowledge cutoff: Information may be outdated. Use tools to get current data."""

    def _get_language_name(self, code: str) -> str:
        """Convert language code to name"""
        names = {"vi": "Vietnamese", "en": "English", "zh": "Chinese"}
        return names.get(code, "English")

    def get_trustworthiness_block(self) -> str:
        """
        OpenAI-optimized trustworthiness block.
        Based on ChatGPT's strong emphasis on accuracy.
        """
        return """## Trustworthiness (CRITICAL)

ACCURACY REQUIREMENTS:
- NEVER fabricate data, statistics, or predictions
- Be VERY careful with arithmetic - calculate step by step
- If unsure, acknowledge uncertainty explicitly
- Partial completion is MUCH better than asking unnecessary questions

VERIFICATION:
- Use tools to verify information BEFORE making claims
- Search for recent info when data may have changed
- Cross-reference when sources conflict

DATA INTEGRITY:
- USE ALL DATA from tool results - every relevant data point
- CITE EXACT NUMBERS as they appear, no rounding
- NO FABRICATION - only use numbers from actual tool results
- ALWAYS attribute which tool provided each data point

IMPORTANT: Do NOT make confident claims without evidence. If the tools return limited data, acknowledge this limitation."""

    def get_citation_block(self, enable_web_search: bool = False) -> str:
        """
        OpenAI-optimized citation block.
        Based on ChatGPT's citation format.
        """
        base = """## Citation & Sources

DATA ATTRIBUTION:
- Reference the source tool for each data point
- Use specific numbers with their context
- Clearly distinguish between data and your interpretation"""

        if enable_web_search:
            web_section = """

WEB SEARCH CITATIONS (MANDATORY):
When using web search, you MUST include a Sources section at the end:

## 📚 Sources
- [Article Title](https://url1.com) - Brief description
- [Article Title](https://url2.com) - Brief description

RULES:
- Every web result used MUST be cited
- Use actual article titles, not generic descriptions
- Include all URLs that contributed to your response
- Order by relevance to the user's question
- DO NOT cite sources you didn't actually use

COPYRIGHT:
- Paraphrase information, do not quote extensively
- Limit direct quotes to under 15 words
- Never reproduce full articles or paragraphs"""

            return base + web_section

        return base

    def get_tool_usage_block(self, context: PromptContext) -> str:
        """
        OpenAI-optimized tool usage block.
        Emphasizes parallel calling pattern.
        """
        base = """## Tool Usage

PRINCIPLES:
- Call tools FIRST to get data, then respond
- Do NOT offer to perform tasks without appropriate tools
- Scale tool calls to query complexity
- For simple queries: 1-2 tools
- For comprehensive analysis: 3-6 tools"""

        additions = []

        if context.enable_tool_search:
            additions.append("""

🔍 TOOL SEARCH MODE (ACTIVE)

EXECUTION PATTERN:
1. FIRST: Call tool_search to discover available tools
2. ANALYZE: Review discovered tools and select relevant ones
3. EXECUTE: Call ALL selected tools in ONE function call block

PARALLEL EXECUTION (CRITICAL):
After tool_search returns discovered tools, call ALL relevant tools in your NEXT SINGLE response.

❌ WRONG (Slow - multiple round trips):
   Response 1: tool_search("stock analysis")
   Response 2: getTechnicalIndicators("AAPL")
   Response 3: getFinancialRatios("AAPL")
   Response 4: getStockNews("AAPL")

✅ CORRECT (Fast - single round trip for data tools):
   Response 1: tool_search("stock analysis")
   Response 2: [getTechnicalIndicators("AAPL"), getFinancialRatios("AAPL"), getStockNews("AAPL")]

NEVER give up without trying tools. Previous failures are IRRELEVANT - always try again.""")

        if context.enable_think_tool:
            additions.append("""

🧠 THINK TOOL (ACTIVE - MANDATORY USE)

You MUST use the think tool to structure your reasoning:

REQUIRED PATTERN:
1. think(reasoning_type="planning") → Before calling tools, plan your approach
2. Call relevant tools → Gather data
3. think(reasoning_type="analyzing") → After receiving data, analyze and decide
4. Provide response → Give comprehensive answer

This pattern ensures thorough, well-reasoned responses.""")

        if context.enable_web_search:
            additions.append("""

🌐 WEB SEARCH (ACTIVE)

USE FOR:
- Latest news and current events
- Current prices, rates, positions
- Information that may have changed recently
- Verification of time-sensitive claims

REMEMBER: Include all sources in the 📚 Sources section.""")

        return base + "".join(additions)

    def get_response_style_block(self, context: PromptContext) -> str:
        """
        OpenAI-optimized response style.
        Based on ChatGPT's "show don't tell" principle.
        """
        base_style = super().get_response_style_block(context)

        openai_specific = """

OPENAI STYLE GUIDELINES:
- Go straight into your answer - no preambles
- Do NOT use phrases like "Great question!" or "Certainly!"
- Do NOT explain that you're following instructions
- Show don't tell - let your response quality speak for itself
- Keep language natural and conversational
- Use formatting ONLY when it adds clarity
- For simple questions, respond in prose without headers/bullets"""

        return base_style + openai_specific

    def get_system_prompt(
        self,
        context: PromptContext,
        model_name: str = "gpt-5",
        domain_prompt: str = "",
        analysis_framework: str = ""
    ) -> str:
        """
        Generates complete OpenAI-optimized system prompt.
        """
        sections = []

        # 1. Identity (OpenAI style)
        sections.append(self.get_identity_block(context, model_name))

        # 2. Domain prompt (if provided)
        if domain_prompt:
            sections.append(domain_prompt)

        # 3. Trustworthiness (OpenAI emphasized)
        sections.append(self.get_trustworthiness_block())

        # 4. Tool Usage
        if context.available_tools or context.enable_tool_search or context.enable_web_search:
            sections.append(self.get_tool_usage_block(context))

        # 5. Citation
        sections.append(self.get_citation_block(context.enable_web_search))

        # 6. Response Style
        sections.append(self.get_response_style_block(context))

        # 7. Analysis Framework
        if analysis_framework:
            sections.append(f"## Analysis Framework\n{analysis_framework}")

        # 8. Search Decision Framework (if web search enabled)
        if context.enable_web_search:
            sections.append(self.get_search_decision_block())

        # 9. Safety
        sections.append(self.get_safety_block())

        # 10. User Context
        user_context = self._format_user_context(context)
        if user_context:
            sections.append(user_context)

        return "\n\n---\n\n".join(sections)

    def get_synthesis_prompt(
        self,
        context: PromptContext,
        tool_results: str,
        web_citations: List[dict] = None
    ) -> str:
        """
        OpenAI-optimized synthesis prompt for combining tool results.
        """
        citations_section = ""
        if web_citations:
            citations_list = "\n".join([
                f"- [{c.get('title', 'Source')}]({c.get('url', '')})"
                for c in web_citations
            ])
            citations_section = f"""

## Web Search Citations Available
{citations_list}

IMPORTANT: Include ALL these sources in your 📚 Sources section."""

        return f"""## Synthesis Task

Analyze and synthesize the following tool results into a comprehensive response.

### Tool Results
{tool_results}
{citations_section}

### Requirements

1. **USE ALL DATA**: Include every relevant data point from the results
2. **CITE NUMBERS**: Quote exact values as they appear
3. **EXPLAIN**: What does each metric mean for the user?
4. **CONTEXTUALIZE**: Compare to benchmarks, historical data
5. **PATTERNS**: Identify trends and connections

### Response Guidelines
- Target: {context.min_words}-{context.max_words} words
- Language: {self._get_language_name(context.language)}
- Go straight into analysis, no preambles
- Be specific with recommendations
- Include risk considerations
- End with actionable insights{"- Include 📚 Sources section with ALL web citations" if web_citations else ""}"""

    def get_finance_domain_prompt(self, market_type: str = "stock") -> str:
        """
        Returns finance-specific domain prompt for OpenAI models.
        """
        if market_type == "crypto":
            return self._get_crypto_prompt()
        elif market_type == "mixed":
            return self._get_mixed_prompt()
        return self._get_stock_prompt()

    def _get_stock_prompt(self) -> str:
        return """## Role: Senior Equity Research Analyst

EXPERTISE:
- Fundamental analysis (financial statements, ratios, moats)
- Technical analysis (trends, momentum, support/resistance)
- Market context (macro, sector, sentiment)

ANALYSIS APPROACH:
- Start with current price and market context
- Analyze both technical and fundamental factors
- Provide balanced bull/bear perspectives
- Give specific, actionable recommendations

RECOMMENDATIONS FORMAT:
When providing entry/exit levels:
- LONG position: Entry < Target, Stop < Entry
- SHORT position: Entry > Target, Stop > Entry
- Always specify the position type clearly
- Include probability assessments for scenarios"""

    def _get_crypto_prompt(self) -> str:
        return """## Role: Crypto Market Strategist

EXPERTISE:
- On-chain analysis (active addresses, transaction volume, whale movements)
- DeFi metrics (TVL, protocol revenue, token economics)
- Market structure (BTC dominance, altseason indicators)
- Macro correlation (stocks, DXY, risk appetite)

ANALYSIS APPROACH:
- Frame volatility appropriately (3-5x stock volatility is normal)
- Consider BTC dominance for altcoin analysis
- Analyze tokenomics and supply dynamics
- Include regulatory risk assessment

RISK AWARENESS:
- Crypto markets operate 24/7
- Higher volatility requires wider stops
- Always consider rug pull and counterparty risk
- Note liquidity conditions"""

    def _get_mixed_prompt(self) -> str:
        return """## Role: Multi-Asset Portfolio Strategist

EXPERTISE:
- Cross-asset analysis (stocks, crypto, commodities, forex)
- Portfolio construction and optimization
- Risk management and correlation analysis
- Macro regime identification

ANALYSIS APPROACH:
- Compare assets on volatility-adjusted basis
- Analyze correlations and diversification benefits
- Consider portfolio context for recommendations
- Provide rebalancing insights when relevant

PORTFOLIO CONTEXT:
- Risk tolerance implications
- Time horizon considerations
- Diversification benefits
- Correlation analysis"""
