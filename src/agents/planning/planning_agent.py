import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.agents.planning.task_models import (
    Task,
    TaskPlan,
    ToolCall,
    TaskPriority,
    TaskStatus,
)

try:
    from src.agents.tools import get_registry
    ATOMIC_TOOLS_AVAILABLE = True
except ImportError:
    ATOMIC_TOOLS_AVAILABLE = False


# ============================================================================
# CONSTANTS
# ============================================================================

# Valid categories
VALID_CATEGORIES = [
    "price",        # Stock prices, quotes, performance
    "technical",    # Technical indicators, chart patterns
    "fundamentals", # P/E, ROE, financial ratios
    "news",         # News, events, announcements
    "market",       # Market overview, indices, trending
    "risk",         # Risk assessment, volatility
    "crypto",       # Cryptocurrency
    "discovery",    # Stock screening
    "memory",       # Cross-session memory search
]

# Categories that indicate STRONG financial intent (not just contextual)
STRONG_FINANCIAL_CATEGORIES = ["discovery", "risk", "technical", "fundamentals"]


class ModelCapability(Enum):
    """Model capability levels for adaptive prompting"""
    BASIC = "basic"           # GPT-4.1-nano
    INTERMEDIATE = "intermediate"  # GPT-4o-mini
    ADVANCED = "advanced"     # GPT-5-nano


class PlanningAgent(LoggerMixin):
    """
    Production Planning Agent with 3-Stage Flow and Working Memory Support
    
    Stage 1: Classify query semantically (LLM decides categories)
    Stage 2: Load tools from registry by categories (no LLM)
    Stage 3: Create task plan with complete tool schemas (LLM plans)
    
    Key Principles:
    - NO hardcoded keyword matching
    - Semantic understanding via LLM
    - Multilingual support
    - Working Memory for task continuity
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: str = None,
    ):
        super().__init__()
        
        self.model_name = model_name or "gpt-4.1-nano"
        self.provider_type = provider_type or ProviderType.OPENAI
        
        # Detect model capability
        self.capability = self._detect_model_capability(self.model_name)
        
        self.logger.info(f"[PLANNING:INIT] Model: {self.model_name}")
        self.logger.info(f"[PLANNING:INIT] Capability: {self.capability.value}")
        
        # Initialize providers
        self.llm_provider = LLMGeneratorProvider()
        self.api_key = ModelProviderFactory._get_api_key(self.provider_type)
        
        # Initialize tool registry
        self.tool_registry = None
        if ATOMIC_TOOLS_AVAILABLE:
            try:
                self.tool_registry = get_registry()
                summary = self.tool_registry.get_summary()
                # self.logger.info(
                #     f"[PLANNING:INIT] Registry: {summary['total_tools']} tools "
                #     f"in {len(summary['categories'])} categories"
                # )
            except Exception as e:
                self.logger.error(f"[PLANNING:INIT] Registry failed: {e}")
    
    def _detect_model_capability(self, model_name: str) -> ModelCapability:
        """Detect model capability from name"""
        model_lower = model_name.lower()
        
        if "gpt-5" in model_lower or "gpt5" in model_lower:
            return ModelCapability.ADVANCED
        elif "gpt-4o" in model_lower or "gpt4o" in model_lower:
            return ModelCapability.INTERMEDIATE
        else:
            return ModelCapability.BASIC
    
    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================
    
    async def think_and_plan(
        self,
        query: str,
        recent_chat: List[Dict[str, str]] = None,
        core_memory: Dict[str, Any] = None,
        summary: str = None,
        working_memory_context: str = None,
    ) -> TaskPlan:
        """
        Main planning method - 3 Stage Flow with Working Memory Support
        
        Args:
            query: User query (any language)
            recent_chat: Recent chat history
            core_memory: User's core memory (long-term)
            summary: Conversation summary
            working_memory_context: Current task state from Working Memory (NEW)
            
        Returns:
            TaskPlan with tasks to execute
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"[PLANNING] ══════════════════════════════════════")
            self.logger.info(f"[PLANNING] Query: '{query[:100]}...'")
            
            if working_memory_context:
                self.logger.info(f"[PLANNING] Working Memory: {len(working_memory_context)} chars")
            
            # ================================================================
            # PHASE 0: VALIDATE & FORMAT HISTORY
            # ================================================================
            formatted_history = self._validate_and_format_history(recent_chat)
            self.logger.info(f"[PLANNING:P0] History: {len(formatted_history)} messages")
            
            # ================================================================
            # STAGE 1: CLASSIFY QUERY & SELECT CATEGORIES
            # ================================================================
            self.logger.info(f"[PLANNING:S1] ── Stage 1: Classification ──")
            
            classification = await self._stage1_classify(
                query=query,
                history=formatted_history,
                working_memory_context=working_memory_context,
                core_memory=core_memory
            )
            
            query_type = classification.get("query_type", "stock_specific")
            categories = classification.get("categories", [])
            symbols = classification.get("symbols", [])
            response_language = classification.get("response_language", "auto")
            reasoning = classification.get("reasoning", "")
            
            self._log_stage1_result(query_type, categories, symbols, reasoning, response_language)
            
            # ================================================================
            # IMPROVED SAFEGUARD: Only override with STRONG signals
            # ================================================================
            # Fix: Only override if there are SYMBOLS (explicit financial request)
            # OR if categories contain strong signals like "discovery", "risk"
            # Do NOT override just because "market" category is present
            
            should_override = False
            override_reason = ""
            
            if query_type == "conversational":
                if symbols:
                    # Has explicit symbols → definitely financial
                    should_override = True
                    override_reason = f"explicit symbols detected: {symbols}"
                    query_type = "stock_specific"
                elif any(cat in STRONG_FINANCIAL_CATEGORIES for cat in categories):
                    # Has strong financial category
                    should_override = True
                    strong_cats = [c for c in categories if c in STRONG_FINANCIAL_CATEGORIES]
                    override_reason = f"strong financial categories: {strong_cats}"
                    
                    if "discovery" in categories:
                        query_type = "screener"
                    else:
                        query_type = "stock_specific"
                # else: Keep as conversational - weak signals like "market" alone
                # are likely just from context bleeding
            
            if should_override:
                self.logger.warning(
                    f"[PLANNING:S1] ⚠️ OVERRIDE: conversational → {query_type} "
                    f"({override_reason})"
                )
            
            # ================================================================
            # Early exit: TRULY conversational (no symbols AND no strong categories)
            # ================================================================
            if query_type == "conversational":
                elapsed = self._get_elapsed_ms(start_time)
                self.logger.info(f"[PLANNING:S1] → No tools needed ({elapsed}ms)")
                
                return TaskPlan(
                    tasks=[],
                    query_intent=query_type,
                    strategy="direct_answer",
                    response_language=response_language,
                    reasoning="Conversational query - no tools needed"
                )
            
            # No categories = nothing to do
            if not categories:
                elapsed = self._get_elapsed_ms(start_time)
                self.logger.info(f"[PLANNING:S1] → No categories selected ({elapsed}ms)")
                
                return TaskPlan(
                    tasks=[],
                    query_intent=query_type,
                    strategy="direct_answer",
                    response_language=response_language,
                    reasoning="No relevant categories for this query"
                )
            
            # ================================================================
            # THINKING: VALIDATE NECESSITY
            # ================================================================
            self.logger.info(f"[PLANNING:THINK] ── Thinking: Validate Intent ──")
            
            thinking = await self._thinking_validate_necessity(
                query, query_type, categories, symbols, reasoning
            )
            
            self.logger.info(f"[PLANNING:THINK] Need tools: {thinking['need_tools']}")
            
            # ================================================================
            # IMPROVED: Only override think result for GENUINE financial queries
            # ================================================================
            if not thinking["need_tools"]:
                # Check if this is a GENUINE financial query that needs tools
                has_symbols = bool(symbols)
                has_strong_financial_intent = query_type in ["stock_specific", "screener"]
                has_strong_categories = any(cat in STRONG_FINANCIAL_CATEGORIES for cat in categories)
                
                # Only override if there's strong evidence of financial intent
                if has_symbols or (has_strong_financial_intent and has_strong_categories):
                    self.logger.warning(
                        f"[PLANNING:THINK] ⚠️ Financial query detected - overriding to use tools"
                    )
                else:
                    # Trust the thinking result - no tools needed
                    elapsed = self._get_elapsed_ms(start_time)
                    self.logger.info(f"[PLANNING:THINK] → No tools needed ({elapsed}ms)")
                    return TaskPlan(
                        tasks=[],
                        query_intent=thinking["final_intent"],
                        strategy="direct_answer",
                        response_language=response_language,
                        reasoning=thinking["reason"]
                    )
            
            # ================================================================
            # STAGE 2: LOAD TOOLS FROM REGISTRY
            # ================================================================
            self.logger.info(f"[PLANNING:S2] ── Stage 2: Load Tools ──")
            
            available_tools = self._stage2_load_tools(categories)
            
            if not available_tools:
                self.logger.warning(f"[PLANNING:S2] ❌ No tools found")
                elapsed = self._get_elapsed_ms(start_time)
                
                return TaskPlan(
                    tasks=[],
                    query_intent=thinking["final_intent"],
                    strategy="direct_answer",
                    response_language=response_language,
                    reasoning="No tools available for selected categories"
                )
            
            # ================================================================
            # STAGE 3: CREATE TASK PLAN
            # ================================================================
            self.logger.info(f"[PLANNING:S3] ── Stage 3: Create Plan ──")
            
            context = self._build_context_string(
                formatted_history, core_memory, summary, working_memory_context
            )
            
            plan_dict = await self._stage3_create_plan(
                query=query,
                query_type=query_type,
                symbols=symbols,
                response_language=response_language,
                available_tools=available_tools,
                context=context,
                validated_intent=thinking["final_intent"]
            )
            
            # Parse into TaskPlan
            task_plan = self._parse_task_plan_dict(
                plan_dict, symbols, response_language
            )
            task_plan.query_intent = thinking["final_intent"]
            
            # ================================================================
            # FINAL LOGGING
            # ================================================================
            elapsed = self._get_elapsed_ms(start_time)
            
            self.logger.info(f"[PLANNING] ══════════════════════════════════════")
            self.logger.info(f"[PLANNING] COMPLETE ({elapsed}ms)")
            self.logger.info(f"[PLANNING] Query Intent: {task_plan.query_intent}")
            self.logger.info(f"[PLANNING] Reasoning: {task_plan.reasoning}")
            self.logger.info(f"[PLANNING] Tasks: {len(task_plan.tasks)}")
            self.logger.info(f"[PLANNING] Strategy: {task_plan.strategy}")
            self.logger.info(f"[PLANNING] Symbols: {task_plan.symbols}")
            self.logger.info(f"[PLANNING] Language: {task_plan.response_language}")
            self.logger.info(f"[PLANNING] ──────────────────────────────────────")
                        
            for idx, task in enumerate(task_plan.tasks, 1):
                tools = [t.tool_name for t in task.tools_needed]
                self.logger.info(f"[PLANNING]   Task {idx}: {task.description}")
                self.logger.info(f"[PLANNING]     Tools: {tools}")

                for tool_call in task.tools_needed:
                    if tool_call.params:
                        params_str = ', '.join(f"{k}={v}" for k, v in tool_call.params.items())
                        self.logger.info(f"[PLANNING]     → {tool_call.tool_name}({params_str})")
            
            return task_plan
            
        except Exception as e:
            self.logger.error(f"[PLANNING] ❌ Error: {e}", exc_info=True)
            
            return TaskPlan(
                tasks=[],
                query_intent="error",
                strategy="direct_answer",
                response_language="auto",
                reasoning=f"Planning error: {str(e)}"
            )
    
    # ========================================================================
    # STAGE 1: CLASSIFICATION (Semantic - with Working Memory)
    # ========================================================================
    
    async def _stage1_classify(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None,
        core_memory: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Stage 1: Classify query semantically with Working Memory context
        
        Working Memory provides:
        - Current symbols in context
        - Previous query intent
        - Task continuity information
        
        NO hardcoded patterns - LLM decides everything
        """
        # Select classifier based on model capability
        if self.capability == "advanced":  # ModelCapability.ADVANCED
            result = await self._classify_advanced(
                query, history, working_memory_context, core_memory
            )
        elif self.capability == "intermediate":  # ModelCapability.INTERMEDIATE  
            result = await self._classify_intermediate(
                query, history, working_memory_context, core_memory
            )
        else:
            result = await self._classify_basic(
                query, history, working_memory_context, core_memory
            )
        
        # Validate categories (filter invalid ones)
        result = self._validate_categories_output(result)
        
        # Auto-add price if symbols present but no price category
        if result.get("symbols") and "price" not in result.get("categories", []):
            result["categories"].append("price")
            self.logger.info(f"[PLANNING:S1] Auto-added 'price' category")
        if result.get("symbols") and "news" not in result.get("categories", []):
            result["categories"].append("news")
        
        return result
    
    async def _classify_basic(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None,
        core_memory: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Classification for basic models with Working Memory"""
        
        history_text = self._format_history_for_prompt(history, max_messages=4)
        date_context = self._get_current_date_context()
        
        # Build working memory section
        wm_section = ""
        if working_memory_context:
            wm_section = f"""
WORKING MEMORY (current task state, symbols from previous turns):
{working_memory_context}

Use this context to understand ongoing tasks and symbol references.
"""
        
        core_memory_section = ""
        if core_memory:
            human_block = core_memory.get('human', '')
            if human_block:
                # Limit to prevent token overflow
                human_truncated = human_block[:1000] if len(human_block) > 1000 else human_block
                core_memory_section = f"""
USER PROFILE (from Core Memory):
{human_truncated}

SYMBOL RESOLUTION FROM USER PROFILE:
When user says:
- "my stocks", "các cổ phiếu của tôi", "cổ phiếu yêu thích" → extract symbols from watchlist/portfolio above
- "tài khoản", "portfolio" → use portfolio symbols
- References to investments without specific tickers → check user profile

If symbols are found in user profile, include them in the "symbols" output.
"""
            
        prompt = f"""{date_context}

You are a financial query analyzer. Analyze the query semantically.

CURRENT QUERY: {query}

RECENT CONVERSATION:
{history_text}
{wm_section}
{core_memory_section}

YOUR TASK:
1. Understand what the user is asking about (semantic analysis)
2. Extract stock symbols if mentioned (explicit or referenced)
3. Determine which tool categories are needed
4. Detect the response language

CRITICAL: CONVERSATIONAL DETECTION
TRUE conversational messages are ONLY:
- Greetings: hello, hi, xin chào, chào bạn
- Thanks: thanks, cảm ơn, thank you, ok cảm ơn
- Bye: goodbye, tạm biệt
- Simple acknowledgments without any request

IMPORTANT: These are NOT conversational:
- Asking about past conversations → memory_recall
- "What did we discuss?" → memory_recall  
- "What stock did I analyze?" → memory_recall
- "Tôi vừa hỏi gì?" → memory_recall
- "Chúng ta đã nói gì về X?" → memory_recall

3. SCREENER: Finding stocks by criteria
    - If user is looking for stocks that meet certain conditions
    - Examples:
        * "Find me stocks haveing market cap > $10B"
        * "Find me tech stocks with P/E < 20"
        * "Find me stocks with high dividend yield"
        * "Find me stocks with high beta"
    Load "discovery" and other relevant categories such as "fundamentals", "technical", "risk", "price", "news", etc.
    
SYMBOL EXTRACTION RULES:
- Look for uppercase stock tickers: AAPL, MSFT, NVDA, CRM, BTC, ETH
- Check for reference words that point to previous context
- Check Working Memory for current symbols in context (or symbols from previous turns)
- If user refers to "it", "that stock", etc., use symbols from context

CATEGORIES (select based on user intent):
- price: Stock prices, quotes, performance data
- technical: Technical indicators, chart patterns, RSI, MACD
- fundamentals: P/E ratio, financials, earnings, revenue
- news: News, events, announcements, calendar
- market: Market overview, indices, trending stocks, top gainers/losers
- risk: Risk assessment, volatility, stop loss suggestions
- crypto: Cryptocurrency data (BTC, ETH, etc.)
- discovery: Stock screening, finding stocks by criteria
- memory: Recall past conversations, what was discussed, what symbols were mentioned before

QUERY TYPES:
- stock_specific: Query about specific stock(s)
- screener: Finding stocks by criteria
- market_level: Market overview, indices
- conversational: ONLY greetings, thanks, bye (NO other categories)
- memory_recall: Asking about past conversations, previous analysis, what was discussed → MUST select "memory" category
- task_continuation: Continuing a previous task

CRITICAL RULES:
1. If query asks "what did we discuss", "what stock", "vừa hỏi gì", "đã nói gì" → query_type: "memory_recall", categories: ["memory"]
2. memory_recall ALWAYS needs categories: ["memory"]
3. conversational ONLY for greetings/thanks/bye with categories: []

OUTPUT JSON ONLY:
{{
  "query_type": "...",
  "categories": [...],
  "symbols": [...],
  "reasoning": "Brief explanation",
  "response_language": "vi|en"
}}
"""
        
        return await self._call_llm_json(prompt, "Stage1")
    
    async def _classify_intermediate(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None,
        core_memory: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Classification for intermediate models with Working Memory"""
        
        history_text = self._format_history_for_prompt(history, max_messages=6)
        date_context = self._get_current_date_context()
        
        wm_section = ""
        if working_memory_context:
            wm_section = f"""
<working_memory>
{working_memory_context}
</working_memory>
"""
        core_memory_section = ""
        if core_memory:
            human_block = core_memory.get('human', '')
            if human_block:
                human_truncated = human_block[:1000] if len(human_block) > 1000 else human_block
                core_memory_section = f"""
<user_profile>
{human_truncated}
</user_profile>

<user_profile_instructions>
Use this to understand user's watchlist, portfolio, and preferences.
When user references "my stocks", "cổ phiếu của tôi", "favorites" → extract symbols from profile.
</user_profile_instructions>
"""
            
        prompt = f"""{date_context}

<task>Semantic financial query analysis with context awareness</task>

<query>{query}</query>

<history>
{history_text}
</history>
{wm_section}
{core_memory_section}

<instructions>
Analyze the query semantically to understand user intent.

CRITICAL: CONVERSATIONAL vs MEMORY_RECALL
- Conversational: ONLY greetings (hello, hi, xin chào), thanks (cảm ơn), bye (tạm biệt)
- Memory Recall: Asking about past conversations, previous analysis, what was discussed

MEMORY RECALL INDICATORS (select categories: ["memory"]):
- "What did we discuss about X?"
- "What stock did I ask about?"
- "Tôi vừa hỏi gì?", "Chúng ta đã nói gì?"
- "Do you remember our conversation about..."
- "What symbols did I mention?"
- Any question about past conversation content

If user asks about previous conversations → query_type: "memory_recall", categories: ["memory"]

STEP 1: Context Check
- Check Working Memory for current task state
- Check if this is a continuation of previous query
- Identify symbols from all sources (query, history, working memory)

STEP 2: Symbol Detection
- Find explicit stock tickers (uppercase: AAPL, MSFT, NVDA)
- Detect reference words pointing to context
- Use symbols from Working Memory if user references "it", "that", etc.

STEP 3: Intent Classification
- stock_specific: About specific stocks
- screener: Finding stocks by criteria
- market_level: Market overview
- conversational: ONLY greeting/thanks/bye (categories: [])
- memory_recall: Questions about past conversations (categories: ["memory"])
- task_continuation: Continuing previous task

STEP 4: Category Selection
Select based on user intent:
- price, technical, fundamentals, news, market, risk, crypto, discovery
- memory: For recalling past conversations, what was discussed
</instructions>

<output>
Return JSON:
{{
  "query_type": "...",
  "categories": [...],
  "symbols": [...],
  "reasoning": "Semantic analysis explanation",
  "response_language": "vi|en"
}}
</output>
"""
        
        return await self._call_llm_json(prompt, "Stage1")
    
    async def _classify_advanced(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None,
        core_memory: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Classification for advanced models with CoT and Working Memory"""
        
        history_text = self._format_history_for_prompt(history, max_messages=6)
        date_context = self._get_current_date_context()
        
        wm_section = ""
        if working_memory_context:
            wm_section = f"""
<working_memory>
{working_memory_context}
</working_memory>
"""
        core_memory_section = ""
        if core_memory:
            human_block = core_memory.get('human', '')
            if human_block:
                human_truncated = human_block[:1200] if len(human_block) > 1200 else human_block
                core_memory_section = f"""
<user_profile source="core_memory">
{human_truncated}
</user_profile>
"""
        prompt = f"""{date_context}

<role>Advanced financial query analyzer with context awareness</role>

<input>
<query>{query}</query>
<history>
{history_text}
</history>
{wm_section}
{core_memory_section}
</input>

<instructions>
Think step-by-step to analyze this query semantically.

CRITICAL FIRST CHECK - DISTINGUISH CAREFULLY:
1. CONVERSATIONAL (categories: []):
Is this a simple conversational message?
- Greetings: hello, hi, xin chào
- Thanks: thanks, cảm ơn, ok cảm ơn  
- Bye: goodbye, tạm biệt
- General chat without financial request

2. MEMORY_RECALL (categories: ["memory"]):
   - Any question about past conversations
   - Examples:
     * "What did we discuss?"
     * "What stock did I ask about?"
     * "Tôi vừa hỏi gì?", "Tôi vừa phân tích cổ phiếu nào?"
     * "Chúng ta đã nói gì về X?"
     * "Do you remember when we talked about..."
     * "Kiểm tra lại cuộc trò chuyện trước"

3. SCREENER: Finding stocks by criteria
    - If user is looking for stocks that meet certain conditions
    - Examples:
        * "Find me stocks haveing market cap > $10B"
        * "Find me tech stocks with P/E < 20"
        * "Find me stocks with high dividend yield"
        * "Find me stocks with high beta"
    Load "discovery" and other relevant categories such as "fundamentals", "technical", "risk", "price", "news", etc.

If user asks about PREVIOUS CONVERSATIONS → memory_recall, NOT conversational!

REASONING PROCESS:
1. Is this asking about past conversations? → memory_recall with categories: ["memory"]
2. Is this a continuation of a previous task? Check Working Memory.
3. What is the user semantically asking about?
4. Are there explicit stock symbols?
5. Are there reference words pointing to history or working memory?
6. What is the underlying intent?
7. Which tool categories are ACTUALLY needed?

CATEGORIES:
- price: Stock prices, quotes, performance
- technical: Technical indicators, chart patterns
- fundamentals: Financial ratios, earnings
- news: News, events, calendar
- market: Market overview, indices, trending stocks, top gainers, top losers, top active
- risk: Risk assessment, volatility
- crypto: Cryptocurrency
- discovery: Stock screening
- memory: Recall past conversations, what was discussed before
</instructions>

<output_format>
First think in [THOUGHT] block, then output JSON:

[THOUGHT]
... your reasoning ...
[/THOUGHT]
```json
{{
  "query_type": "stock_specific|screener|market_level|conversational|memory_recall|task_continuation",
  "categories": ["price", ...],
  "symbols": ["SYMBOL", ...],
  "reasoning": "Brief explanation",
  "response_language": "vi|en"
}}
```
</output_format>
"""
        
        return await self._call_llm_json(prompt, "Stage1")
    
    # ========================================================================
    # STAGE 2: LOAD TOOLS FROM REGISTRY
    # ========================================================================
    
    def _stage2_load_tools(
        self,
        categories: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Stage 2: Load tools from registry by categories
        
        Progressive Disclosure:
        - Only load tools from selected categories
        - Typical: 7-15 tools vs 31 total
        - NO LLM call needed
        """
        if not self.tool_registry:
            self.logger.error(f"[PLANNING:S2] Tool registry not available")
            return []
        
        available_tools = []
        tools_per_category = {}
        
        for category in categories:
            tools_dict = self.tool_registry.get_tools_by_category(category)
            
            if not tools_dict:
                self.logger.warning(f"[PLANNING:S2] Category '{category}' has no tools")
                continue
            
            category_tools = []
            
            for tool_name in tools_dict:
                schema = self.tool_registry.get_schema(tool_name)
                
                if schema:
                    tool_schema = schema.to_json_schema()
                    available_tools.append(tool_schema)
                    category_tools.append(tool_name)
            
            tools_per_category[category] = category_tools
        
        self.logger.info(f"[PLANNING:S2] Categories: {categories}")
        self.logger.info(f"[PLANNING:S2] Tools loaded: {len(available_tools)}")
        
        for cat, tools in tools_per_category.items():
            self.logger.info(f"[PLANNING:S2]   • {cat}: {tools}")
        
        if self.tool_registry:
            total = len(self.tool_registry.get_all_tools())
            self.logger.info(
                f"[PLANNING:S2] Progressive Disclosure: {len(available_tools)}/{total} tools"
            )
        
        return available_tools
    
    # ========================================================================
    # THINKING LAYER
    # ========================================================================
    
    async def _thinking_validate_necessity(
        self,
        query: str,
        query_type: str,
        categories: List[str],
        symbols: List[str],
        reasoning: str
    ) -> Dict[str, Any]:
        """
        Thinking layer: Validate if tools are actually needed
        
        Prevents false negatives where LLM says "no tools" for financial queries
        """
        date_context = self._get_current_date_context()
        
        prompt = f"""{date_context}

Validate if external tools are needed for this query.

QUERY: "{query}"
CLASSIFICATION:
- Type: {query_type}
- Categories: {categories}
- Symbols: {symbols}
- Reasoning: {reasoning}

CRITICAL KNOWLEDGE:
You are an AI with a knowledge cutoff. You CANNOT provide:
- Real-time stock prices
- Current financial data
- Recent news
- Live technical indicators
- Current market state
- Past conversation history (need memory tools)

DECISION RULES:
1. Greeting/Thanks/General Chat → NO tools (just respond naturally)
2. memory_recall (asking about past conversations) → YES tools (need searchConversationHistory)
3. Price/quote request with SYMBOL → YES tools (MUST fetch real-time)
4. Financial analysis with SYMBOL → YES tools (MUST fetch data)
5. Stock screening → YES tools ((MUST fetch data)
6. Market overview request → YES tools (MUST fetch current state)

IMPORTANT: 
- Simple greetings like "thanks", "ok cảm ơn" do NOT need tools!
- BUT questions like "what did we discuss?" NEED memory tools!

OUTPUT JSON:
{{
  "need_tools": true/false,
  "final_intent": "Clear description of user intent",
  "reason": "Why tools are or aren't needed"
}}
"""
        
        try:
            result = await self._call_llm_json(prompt, "Thinking")
            
            if "need_tools" not in result:
                result["need_tools"] = query_type not in ["conversational", "memory_recall"]
            if "final_intent" not in result:
                result["final_intent"] = query
            if "reason" not in result:
                result["reason"] = "Validated"
            
            return result
            
        except Exception as e:
            self.logger.error(f"[PLANNING:THINK] Error: {e}")
            return {
                "need_tools": query_type not in ["conversational", "memory_recall"],
                "final_intent": query,
                "reason": "Fallback validation"
            }
    
    # ========================================================================
    # STAGE 3: CREATE TASK PLAN
    # ========================================================================
    
    async def _stage3_create_plan(
        self,
        query: str,
        query_type: str,
        symbols: List[str],
        response_language: str,
        available_tools: List[Dict[str, Any]],
        context: str,
        validated_intent: str
    ) -> Dict[str, Any]:
        """
        Stage 3: Create detailed task plan
        
        Shows COMPLETE tool schemas to LLM for accurate planning
        """
        date_context = self._get_current_date_context()
        
        tools_text = self._format_tools_for_prompt(available_tools)
        
        self.logger.info(f"[PLANNING:S3] Tools for LLM: {len(available_tools)}")
        
        prompt = f"""{date_context}

<role>Task planner for financial AI assistant. Create execution plans using the available tools.</role>

<input>
Current Query: {query}
Type: {query_type}
Symbols: {symbols if symbols else "None - may need screening"}
Language: {response_language}
Intent: {validated_intent}
</input>

<available_tools>
{tools_text}
</available_tools>

<planning_rules>
1. Use EXACT tool names from available_tools
2. Match params to tool schema
3. Strategy: "parallel" for independent tasks, "sequential" for dependent
4. For tools with "requires_symbol": include {{"symbol": "TICKER"}} in params
5. For tools without symbol requirement: use appropriate params or {{}}
6. Each task should be focused and simple
7. Include all necessary tools for complete analysis
8. CRYPTO SYMBOL RULE: DO NOT strip the "USD" suffix - it's required for crypto API calls (e.g., "BTCUSD", "ETHUSD")

CRITICAL - SEQUENTIAL DEPENDENCIES:
When query_type is "screener" or other category and needs follow-up analysis:
1. Task 1: Run stockScreener (returns symbols) - NO dependencies
2. Task 2+: 
   - Use placeholder: "symbol": "<FROM_TASK_1>"
   - MUST include: "dependencies": [1]
   
The placeholder <FROM_TASK_N> will be replaced with actual symbols at execution time.
If you use a placeholder, you MUST include the corresponding dependency!

WRONG (placeholder without dependency):
{{"tool_name": "getFinancialRatios", "params": {{"symbol": "<FROM_TASK_1>"}}}}
"dependencies": []  ← WRONG!

CORRECT (placeholder WITH dependency):
{{"tool_name": "getFinancialRatios", "params": {{"symbol": "<FROM_TASK_1>"}}}}
"dependencies": [1]  ← CORRECT!
</planning_rules>

<tool_category_guide>
| Category | Tool Pattern | Params Example |
|----------|--------------|----------------|
| price | getStockPrice, getStockQuote | {{"symbol": "AAPL"}} |
| technical | getTechnicalIndicators | {{"symbol": "AAPL", "indicators": ["RSI","MACD"]}} |
| fundamentals | getFinancialRatios, getGrowthMetrics | {{"symbol": "AAPL"}} |
| news | getStockNews | {{"symbol": "AAPL", "limit": 5}} |
| market | getMarketMovers, getSectorPerformance | {{"mover_type": "gainers"}} or {{}} |
| discovery | stockScreener | {{"sector": "Technology", "country": "US", "limit": 15}} |
| crypto | getCryptoPrice, getCryptoTechnicals | {{"symbol": "BTC"}} |
| risk | getRiskMetrics | {{"symbol": "AAPL"}} |
| memory | searchConversationHistory | {{"query": "...", "limit": 5}} |
</tool_category_guide>

<screener_mapping>
When using stockScreener, map user criteria:
- "công nghệ" / "technology" / "tech" → sector: "Technology"
- "tài chính" / "financial" → sector: "Financial Services"
- "y tế" / "healthcare" → sector: "Healthcare"
- "năng lượng" / "energy" → sector: "Energy"
- "Mỹ" / "US" / "America" → country: "US"
- "Việt Nam" / "VN" → country: "VN"
</screener_mapping>

<examples>
Example 1 - Screener (discovery):
Query: "tìm cổ phiếu công nghệ Mỹ"
Type: screener
Tools: stockScreener
```json
{{"query_intent":"Screen US tech stocks","strategy":"sequential","estimated_complexity":"moderate","symbols":[],"response_language":"vi","reasoning":"Stock screening by sector and country","tasks":[{{"id":1,"description":"Screen US Technology stocks","tools_needed":[{{"tool_name":"stockScreener","params":{{"sector":"Technology","country":"US","is_actively_trading":true,"limit":15}}}}],"expected_data":["symbols","stocks"],"priority":"high","dependencies":[]}}]}}
```

Example 2 - Price (single symbol):
Query: "giá AAPL"
Type: stock_specific
Tools: getStockPrice
```json
{{"query_intent":"Get AAPL price","strategy":"parallel","estimated_complexity":"simple","symbols":["AAPL"],"response_language":"vi","reasoning":"Single price query","tasks":[{{"id":1,"description":"Get AAPL current price","tools_needed":[{{"tool_name":"getStockPrice","params":{{"symbol":"AAPL"}}}}],"expected_data":["price","change","volume"],"priority":"high","dependencies":[]}}]}}
```

Example 3 - Crypto Analysis (Keep FULL symbol):
Query: "Phân tích kỹ thuật BTCUSD"
Type: stock_specific
Tools: getCryptoPrice, getCryptoTechnicals
```json
{{"query_intent":"Technical analysis for BTCUSD","strategy":"parallel","estimated_complexity":"moderate","symbols":["BTCUSD"],"response_language":"vi","reasoning":"Crypto analysis needs price and technicals - use FULL symbol BTCUSD","tasks":[{{"id":1,"description":"Get BTC current price","tools_needed":[{{"tool_name":"getCryptoPrice","params":{{"symbol":"BTCUSD"}}}}],"expected_data":["price","change"],"priority":"high","dependencies":[]}},{{"id":2,"description":"Get BTC technical indicators","tools_needed":[{{"tool_name":"getCryptoTechnicals","params":{{"symbol":"BTCUSD","timeframe":"1hour"}}}}],"expected_data":["RSI","MACD","trend"],"priority":"high","dependencies":[]}}]}}
```

Example 4 - Technical Analysis:
Query: "phân tích kỹ thuật NVDA"
Type: stock_specific
Tools: getStockPrice, getTechnicalIndicators
```json
{{"query_intent":"NVDA technical analysis","strategy":"parallel","estimated_complexity":"moderate","symbols":["NVDA"],"response_language":"vi","reasoning":"Need price and technical indicators","tasks":[{{"id":1,"description":"Get NVDA price","tools_needed":[{{"tool_name":"getStockPrice","params":{{"symbol":"NVDA"}}}}],"expected_data":["price"],"priority":"high","dependencies":[]}},{{"id":2,"description":"Get NVDA technicals","tools_needed":[{{"tool_name":"getTechnicalIndicators","params":{{"symbol":"NVDA"}}}}],"expected_data":["RSI","MACD","SMA"],"priority":"high","dependencies":[]}}]}}
```

Example 5 - Full Analysis (multiple categories):
Query: "phân tích toàn diện MSFT"
Type: stock_specific
Tools: getStockPrice, getTechnicalIndicators, getFinancialRatios, getStockNews
```json
{{"query_intent":"Comprehensive MSFT analysis","strategy":"parallel","estimated_complexity":"complex","symbols":["MSFT"],"response_language":"vi","reasoning":"Full analysis needs price, technicals, fundamentals, news","tasks":[{{"id":1,"description":"Get MSFT price","tools_needed":[{{"tool_name":"getStockPrice","params":{{"symbol":"MSFT"}}}}],"expected_data":["price"],"priority":"high","dependencies":[]}},{{"id":2,"description":"Get MSFT technicals","tools_needed":[{{"tool_name":"getTechnicalIndicators","params":{{"symbol":"MSFT"}}}}],"expected_data":["RSI","MACD"],"priority":"high","dependencies":[]}},{{"id":3,"description":"Get MSFT financials","tools_needed":[{{"tool_name":"getFinancialRatios","params":{{"symbol":"MSFT"}}}}],"expected_data":["PE","ROE"],"priority":"medium","dependencies":[]}},{{"id":4,"description":"Get MSFT news","tools_needed":[{{"tool_name":"getStockNews","params":{{"symbol":"MSFT","limit":5}}}}],"expected_data":["news"],"priority":"medium","dependencies":[]}}]}}
```

Example 6 - Market Overview:
Query: "thị trường hôm nay thế nào"
Type: market_level
Tools: getSectorPerformance, getTopGainers, getTopLosers, getTopActive
```json
{{"query_intent":"Today's market overview","strategy":"parallel","estimated_complexity":"moderate","symbols":[],"response_language":"vi","reasoning":"Market overview needs sector performance and top movers","tasks":[{{"id":1,"description":"Get sector performance","tools_needed":[{{"tool_name":"getSectorPerformance","params":{{}}}}],"expected_data":["sectors"],"priority":"high","dependencies":[]}},{{"id":2,"description":"Get top gainers","tools_needed":[{{"tool_name":"getTopGainers","params":{{"limit":5}}}}],"expected_data":["gainers"],"priority":"high","dependencies":[]}},{{"id":3,"description":"Get top losers","tools_needed":[{{"tool_name":"getTopLosers","params":{{"limit":5}}}}],"expected_data":["losers"],"priority":"high","dependencies":[]}},{{"id":4,"description":"Get top active stocks","tools_needed":[{{"tool_name":"getTopActive","params":{{"limit":5}}}}],"expected_data":["active_stocks"],"priority":"medium","dependencies":[]}}]}}
```

Example 7 - Crypto:
Query: "giá BTC và ETH"
Type: stock_specific
Tools: getCryptoPrice
```json
{{"query_intent":"Get BTC and ETH prices","strategy":"parallel","estimated_complexity":"simple","symbols":["BTC","ETH"],"response_language":"vi","reasoning":"Crypto price query","tasks":[{{"id":1,"description":"Get BTC price","tools_needed":[{{"tool_name":"getCryptoPrice","params":{{"symbol":"BTC"}}}}],"expected_data":["price"],"priority":"high","dependencies":[]}},{{"id":2,"description":"Get ETH price","tools_needed":[{{"tool_name":"getCryptoPrice","params":{{"symbol":"ETH"}}}}],"expected_data":["price"],"priority":"high","dependencies":[]}}]}}
```

Example 8 - Compare stocks:
Query: "so sánh AAPL và MSFT"
Type: stock_specific
Tools: getStockPrice, getFinancialRatios
```json
{{"query_intent":"Compare AAPL vs MSFT","strategy":"parallel","estimated_complexity":"moderate","symbols":["AAPL","MSFT"],"response_language":"vi","reasoning":"Compare needs price and fundamentals for both","tasks":[{{"id":1,"description":"Get AAPL price","tools_needed":[{{"tool_name":"getStockPrice","params":{{"symbol":"AAPL"}}}}],"expected_data":["price"],"priority":"high","dependencies":[]}},{{"id":2,"description":"Get MSFT price","tools_needed":[{{"tool_name":"getStockPrice","params":{{"symbol":"MSFT"}}}}],"expected_data":["price"],"priority":"high","dependencies":[]}},{{"id":3,"description":"Get AAPL financials","tools_needed":[{{"tool_name":"getFinancialRatios","params":{{"symbol":"AAPL"}}}}],"expected_data":["PE","ROE"],"priority":"medium","dependencies":[]}},{{"id":4,"description":"Get MSFT financials","tools_needed":[{{"tool_name":"getFinancialRatios","params":{{"symbol":"MSFT"}}}}],"expected_data":["PE","ROE"],"priority":"medium","dependencies":[]}}]}}
```

Example 9 - Risk Analysis:
Query: "đánh giá rủi ro TSLA"
Type: stock_specific  
Tools: getStockPrice, getRiskMetrics, getTechnicalIndicators
```json
{{"query_intent":"TSLA risk assessment","strategy":"parallel","estimated_complexity":"moderate","symbols":["TSLA"],"response_language":"vi","reasoning":"Risk analysis needs price, risk metrics, and volatility indicators","tasks":[{{"id":1,"description":"Get TSLA price","tools_needed":[{{"tool_name":"getStockPrice","params":{{"symbol":"TSLA"}}}}],"expected_data":["price"],"priority":"high","dependencies":[]}},{{"id":2,"description":"Get TSLA risk metrics","tools_needed":[{{"tool_name":"getRiskMetrics","params":{{"symbol":"TSLA"}}}}],"expected_data":["beta","volatility"],"priority":"high","dependencies":[]}},{{"id":3,"description":"Get TSLA technicals","tools_needed":[{{"tool_name":"getTechnicalIndicators","params":{{"symbol":"TSLA"}}}}],"expected_data":["RSI","ATR"],"priority":"medium","dependencies":[]}}]}}
```
</examples>

<dependency_rules>
WHEN TO USE DEPENDENCIES:
- query_type="screener" AND user wants analysis → SEQUENTIAL with <FROM_TASK_N>
- query_type="stock_specific" with known symbols → PARALLEL, no dependencies
- query_type="market_level" → PARALLEL, no dependencies

PLACEHOLDER SYNTAX:
- Use "<FROM_TASK_1>" when Task N depends on Task 1's output
- The number matches the dependency task's id
- TaskExecutor will expand this to actual symbols from screener results
</dependency_rules>

<output_format>
OUTPUT JSON ONLY:
{{
  "query_intent": "{validated_intent}",
  "strategy": "parallel|sequential",
  "estimated_complexity": "simple|moderate|complex",
  "symbols": {json.dumps(symbols) if symbols else "[]"},
  "response_language": "{response_language}",
  "reasoning": "Brief explanation of the plan",
  "tasks": [...]
}}
</output_format>
"""
        
        return await self._call_llm_json(prompt, "Stage3")
    
    def _format_tools_for_prompt(
        self,
        tools: List[Dict[str, Any]],
        max_tools: int = 20
    ) -> str:
        """Format tools with COMPLETE information for LLM"""
        lines = []
        
        for tool in tools[:max_tools]:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")[:100]
            
            metadata = tool.get("metadata", {})
            requires_symbol = metadata.get("requires_symbol", True)
            
            params = tool.get("parameters", {}).get("properties", {})
            required_params = tool.get("parameters", {}).get("required", [])
            
            param_info = []
            for param_name, param_def in params.items():
                is_required = param_name in required_params
                param_type = param_def.get("type", "string")
                req_marker = "*" if is_required else ""
                param_info.append(f"{param_name}{req_marker}:{param_type}")
            
            params_str = ", ".join(param_info) if param_info else "none"
            symbol_str = "✓" if requires_symbol else "✗"
            
            lines.append(
                f"• {name}\n"
                f"  Description: {description}\n"
                f"  Symbol required: {symbol_str}\n"
                f"  Parameters: {params_str}"
            )
        
        return "\n\n".join(lines)
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _get_current_date_context(self) -> str:
        """Get current date context for prompts"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M UTC")
        
        return f"""<current_context>
Date: {current_date} {current_time}
Data Status: Real-time market data available through tools
Note: Tools fetch LIVE data from financial APIs
</current_context>"""
    
    def _validate_and_format_history(
        self,
        recent_chat: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Validate and format chat history"""
        if not recent_chat:
            return []
        
        validated = []
        for msg in recent_chat:
            if not isinstance(msg, dict):
                continue
            
            role = msg.get("role", "").strip().lower()
            content = msg.get("content", "").strip()
            
            if not role or not content:
                continue
            
            if role not in ["user", "assistant"]:
                continue
            
            validated.append({
                "role": role,
                "content": content,
                "timestamp": msg.get("created_at", "")
            })
        
        if all(msg.get("timestamp") for msg in validated):
            validated = sorted(validated, key=lambda x: x["timestamp"])
        
        return validated
    
    def _format_history_for_prompt(
        self,
        history: List[Dict[str, str]],
        max_messages: int = 6
    ) -> str:
        """Format history for LLM prompt"""
        if not history:
            return "No previous conversation"
        
        recent = history[-max_messages:]
        
        lines = []
        for i, msg in enumerate(recent):
            role = msg["role"].upper()
            content = msg["content"][:200]
            lines.append(f"Turn {i+1} [{role}]: {content}")
        
        return "\n".join(lines)
    
    def _validate_categories_output(
        self,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate categories - filter invalid ones"""
        original = result.get("categories", [])
        
        if not original:
            return result
        
        valid = [c for c in original if c in VALID_CATEGORIES]
        invalid = [c for c in original if c not in VALID_CATEGORIES]
        
        if invalid:
            self.logger.warning(f"[PLANNING:S1] Filtered invalid categories: {invalid}")
        
        result["categories"] = valid
        return result
    
    def _build_context_string(
        self,
        history: List[Dict[str, str]],
        core_memory: Dict[str, Any],
        summary: str,
        working_memory_context: str = None
    ) -> str:
        """Build context string from all sources including Working Memory"""
        parts = []
        
        if core_memory:
            lines = [f"  • {k}: {v}" for k, v in core_memory.items()]
            parts.append(f"User Info:\n" + "\n".join(lines))
        
        if summary:
            parts.append(f"Summary: {summary}")
        
        if working_memory_context:
            parts.append(f"Current Task State:\n{working_memory_context}")
        
        return "\n\n".join(parts) if parts else ""
    
    async def _call_llm_json(
        self,
        prompt: str,
        stage: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response with retry logic
        
        Args:
            prompt: The prompt to send
            stage: Stage name for logging
            max_retries: Number of retries on JSON parse failure
        """
        content = ""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Set appropriate max_tokens based on stage
                max_tokens = 2000 if stage == "Stage3" else 1000
                
                params = {
                    "model_name": self.model_name,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a financial assistant. Return ONLY valid JSON, no markdown, no explanation."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "provider_type": self.provider_type,
                    "api_key": self.api_key,
                    "enable_thinking": False,
                    "max_tokens": max_tokens
                }
                
                if self.capability != ModelCapability.ADVANCED:
                    params["temperature"] = 0.1
                
                response = await self.llm_provider.generate_response(**params)
                
                content = response.get("content", "") if isinstance(response, dict) else str(response)
                content = content.strip()
                
                # Remove thought blocks
                if "[/THOUGHT]" in content:
                    _, content = content.split("[/THOUGHT]", 1)
                
                # Remove markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    parts = content.split("```")
                    if len(parts) >= 2:
                        content = parts[1]
                
                content = content.strip()
                
                # Fix JSON issues
                content = self._fix_json_braces(content)
                content = self._fix_truncated_json(content, stage)
                
                return json.loads(content)
                
            except json.JSONDecodeError as e:
                last_error = e
                self.logger.warning(
                    f"[{stage}] JSON parse error (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )
                
                if attempt < max_retries:
                    # Try to recover or retry
                    recovered = self._try_recover_json(content, stage)
                    if recovered:
                        self.logger.info(f"[{stage}] ✓ Recovered JSON from truncated response")
                        return recovered
                    
                    self.logger.info(f"[{stage}] Retrying with simplified prompt...")
                    prompt = self._simplify_prompt_for_retry(prompt, stage)
                    continue
                else:
                    self.logger.error(f"[{stage}] JSON parse failed after {max_retries + 1} attempts")
                    self.logger.error(f"[{stage}] Content: {content[:500]}")
                    
                    # Return fallback based on stage
                    return self._get_fallback_response(stage)
                    
            except Exception as e:
                self.logger.error(f"[{stage}] LLM error: {e}", exc_info=True)
                raise
        
        # Should not reach here, but return fallback
        return self._get_fallback_response(stage)


    def _fix_truncated_json(self, content: str, stage: str) -> str:
        """
        Fix truncated JSON responses
        
        Common issues:
        - Response cut off mid-string
        - Missing closing brackets for arrays/objects
        """
        if not content:
            return '{}'
        
        # Check if JSON seems complete
        if content.endswith('}'):
            return content
        
        # For Stage3, we need tasks array
        if stage == "Stage3":
            # Find last complete task object
            last_task_end = content.rfind('}]')
            if last_task_end > 0:
                # Try to close the JSON properly
                content = content[:last_task_end + 2] + '}'
                return content
            
            # Find if we have tasks array started
            if '"tasks"' in content and '[' in content:
                tasks_start = content.find('"tasks"')
                bracket_start = content.find('[', tasks_start)
                
                if bracket_start > 0:
                    # Check if we're inside a task object
                    last_open_brace = content.rfind('{')
                    if last_open_brace > bracket_start:
                        # We're inside a task object, close it and the array
                        content = content + '}]}'
                    else:
                        # Just close array and main object
                        content = content + ']}'
                    return content
        
        # Generic fix - close any open structures
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Add missing closures
        if open_brackets > 0:
            content = content + (']' * open_brackets)
        if open_braces > 0:
            content = content + ('}' * open_braces)
        
        return content


    def _try_recover_json(self, content: str, stage: str) -> Optional[Dict[str, Any]]:
        """Try to recover partial JSON"""
        import re
        
        if not content:
            return None
        
        # Try parsing with fixes first
        try:
            fixed = self._fix_truncated_json(content, stage)
            return json.loads(fixed)
        except:
            pass
        
        # For Stage3, try to extract at least the metadata
        if stage == "Stage3":
            try:
                intent_match = re.search(r'"query_intent"\s*:\s*"([^"]*)"', content)
                strategy_match = re.search(r'"strategy"\s*:\s*"([^"]*)"', content)
                symbols_match = re.search(r'"symbols"\s*:\s*\[([^\]]*)\]', content)
                language_match = re.search(r'"response_language"\s*:\s*"([^"]*)"', content)
                
                if intent_match:
                    symbols = []
                    if symbols_match:
                        symbols_str = symbols_match.group(1)
                        symbols = [s.strip().strip('"') for s in symbols_str.split(',') if s.strip()]
                    
                    # Try to extract tasks if any complete
                    tasks = []
                    tasks_match = re.search(r'"tasks"\s*:\s*\[(.*)', content, re.DOTALL)
                    if tasks_match:
                        tasks_content = tasks_match.group(1)
                        # Find complete task objects
                        task_pattern = r'\{[^{}]*"tool_name"[^{}]*\}'
                        task_matches = re.findall(task_pattern, tasks_content)
                        
                        for i, task_str in enumerate(task_matches[:3]):  # Max 3 tasks
                            try:
                                # Parse individual task tools
                                tool_name_match = re.search(r'"tool_name"\s*:\s*"([^"]*)"', task_str)
                                if tool_name_match:
                                    tasks.append({
                                        "id": i + 1,
                                        "description": f"Execute {tool_name_match.group(1)}",
                                        "tools_needed": [{
                                            "tool_name": tool_name_match.group(1),
                                            "params": {"symbol": symbols[0] if symbols else ""}
                                        }],
                                        "expected_data": [],
                                        "priority": "medium",
                                        "dependencies": []
                                    })
                            except:
                                continue
                    
                    return {
                        "query_intent": intent_match.group(1),
                        "strategy": strategy_match.group(1) if strategy_match else "parallel",
                        "estimated_complexity": "moderate",
                        "symbols": symbols,
                        "response_language": language_match.group(1) if language_match else "vi",
                        "reasoning": "Recovered from truncated response",
                        "tasks": tasks
                    }
            except Exception as e:
                pass
        
        # For Stage1, try to extract classification
        if stage == "Stage1":
            try:
                query_type_match = re.search(r'"query_type"\s*:\s*"([^"]*)"', content)
                categories_match = re.search(r'"categories"\s*:\s*\[([^\]]*)\]', content)
                symbols_match = re.search(r'"symbols"\s*:\s*\[([^\]]*)\]', content)
                
                if query_type_match:
                    categories = []
                    if categories_match:
                        cat_str = categories_match.group(1)
                        categories = [c.strip().strip('"') for c in cat_str.split(',') if c.strip()]
                    
                    symbols = []
                    if symbols_match:
                        sym_str = symbols_match.group(1)
                        symbols = [s.strip().strip('"') for s in sym_str.split(',') if s.strip()]
                    
                    return {
                        "query_type": query_type_match.group(1),
                        "categories": categories,
                        "symbols": symbols,
                        "reasoning": "Recovered from truncated response",
                        "response_language": "vi"
                    }
            except:
                pass
        
        return None


    def _simplify_prompt_for_retry(self, prompt: str, stage: str) -> str:
        """Simplify prompt for retry attempt"""
        if stage == "Stage3":
            return prompt + """

    IMPORTANT FOR RETRY: Keep response SHORT.
    - Maximum 2-3 tasks only
    - Brief descriptions
    - Essential tools only
    - No lengthy explanations
    - Return valid JSON immediately
    """
        return prompt


    def _get_fallback_response(self, stage: str) -> Dict[str, Any]:
        """Get fallback response for failed JSON parse"""
        if stage == "Stage1":
            return {
                "query_type": "stock_specific",
                "categories": ["price"],
                "symbols": [],
                "reasoning": "Fallback: classification failed",
                "response_language": "vi"
            }
        elif stage == "Stage3":
            return {
                "query_intent": "Analysis request",
                "strategy": "parallel",
                "estimated_complexity": "simple",
                "symbols": [],
                "response_language": "vi",
                "reasoning": "Fallback: plan creation failed",
                "tasks": []
            }
        elif stage == "Thinking":
            return {
                "need_tools": True,
                "final_intent": "Financial analysis",
                "reason": "Fallback: validation failed"
            }
        else:
            return {}


    # ============================================================================
    # ALSO UPDATE _fix_json_braces to be more robust:
    # ============================================================================

    def _fix_json_braces(self, content: str) -> str:
        """
        Fix mismatched braces in JSON response
        
        Common LLM errors:
        - Extra trailing braces: {}}} → {}
        - Missing closing braces
        - Unclosed strings
        """
        if not content:
            return '{}'
        
        # Remove any trailing incomplete strings
        # If we have an unclosed quote, find and close it
        quote_count = content.count('"')
        if quote_count % 2 == 1:
            # Odd number of quotes - unclosed string
            last_quote = content.rfind('"')
            if last_quote > 0:
                # Check if this is the start of an unclosed string
                before_quote = content[:last_quote]
                if before_quote.endswith(':') or before_quote.endswith(': '):
                    # It's a value that got cut off, add closing quote
                    content = content + '"'
        
        # Count braces
        open_count = content.count('{')
        close_count = content.count('}')
        
        if open_count == close_count:
            return content
        
        if close_count > open_count:
            # Too many closing braces - remove from end
            excess = close_count - open_count
            while excess > 0 and content.rstrip().endswith('}'):
                content = content.rstrip()[:-1]
                excess -= 1
            return content
        
        if open_count > close_count:
            # Missing closing braces - add at end
            missing = open_count - close_count
            content = content.rstrip() + ('}' * missing)
            return content
        
        return content
    
    def _parse_task_plan_dict(
        self,
        plan_dict: Dict[str, Any],
        symbols: List[str],
        response_language: str
    ) -> TaskPlan:
        """Parse JSON dict into TaskPlan object"""
        tasks = []
        
        for task_data in plan_dict.get('tasks', []):
            tools_needed = []
            
            for tool_data in task_data.get('tools_needed', []):
                if "symbol" in tool_data and "params" not in tool_data:
                    tool_data["params"] = {"symbol": tool_data.pop("symbol")}
                
                tools_needed.append(ToolCall(**tool_data))
            
            task = Task(
                id=task_data.get('id', len(tasks) + 1),
                description=task_data.get('description', ''),
                tools_needed=tools_needed,
                expected_data=task_data.get('expected_data', []),
                priority=TaskPriority(task_data.get('priority', 'medium')),
                dependencies=task_data.get('dependencies', []),
                status=TaskStatus.PENDING,
                done=False
            )
            tasks.append(task)
        
        return TaskPlan(
            tasks=tasks,
            query_intent=plan_dict.get('query_intent', ''),
            strategy=plan_dict.get('strategy', 'parallel'),
            estimated_complexity=plan_dict.get('estimated_complexity', 'simple'),
            symbols=symbols or plan_dict.get('symbols', []),
            response_language=response_language,
            reasoning=plan_dict.get('reasoning', '')
        )
    
    def _log_stage1_result(
        self,
        query_type: str,
        categories: List[str],
        symbols: List[str],
        reasoning: str,
        language: str
    ):
        """Log Stage 1 results"""
        self.logger.info(f"[PLANNING:S1] Classification Complete")
        self.logger.info(f"[PLANNING:S1]   Type: {query_type}")
        self.logger.info(f"[PLANNING:S1]   Categories: {categories}")
        self.logger.info(f"[PLANNING:S1]   Symbols: {symbols}")
        self.logger.info(f"[PLANNING:S1]   Language: {language}")
        self.logger.info(f"[PLANNING:S1]   Reasoning: {reasoning}")
    
    def _get_elapsed_ms(self, start_time: datetime) -> int:
        """Get elapsed time in milliseconds"""
        return int((datetime.now() - start_time).total_seconds() * 1000)