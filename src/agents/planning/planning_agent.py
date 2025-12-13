# File: src/agents/planning/planning_agent.py
"""
Production-Ready Planning Agent with 3-Stage Flow and Working Memory Support

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PLANNING AGENT 3-STAGE FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Stage 1: CLASSIFY & SELECT CATEGORIES (1 LLM call)                         │
│  - Semantic analysis of query                                               │
│  - Context-aware symbol extraction from history + working memory            │
│  - Category selection based on intent                                       │
│                                                                             │
│  Stage 2: LOAD TOOLS FROM REGISTRY (0 LLM calls)                            │
│  - Load ONLY tools from selected categories                                 │
│  - Get complete ToolSchema for each tool                                    │
│  - Progressive disclosure: 7-15 tools typical vs 31 total                   │
│                                                                             │
│  Stage 3: CREATE TASK PLAN (1 LLM call)                                     │
│  - Show COMPLETE tool schemas to LLM                                        │
│  - LLM creates detailed execution plan                                      │
│  - Output: TaskPlan with exact tool calls                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Working Memory Integration:
- Receives working_memory_context from chat_handler
- Uses context for symbol continuity across turns
- Supports task continuation (if existing task in progress)

Features:
- Multi-model support (GPT-4.1-nano, GPT-4o-mini, GPT-5-nano)
- Multilingual support (no hardcoded keywords)
- Semantic understanding via LLM reasoning
"""

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

# Valid categories - used for validation only, NOT for classification
VALID_CATEGORIES = [
    "price",        # Stock prices, quotes, performance
    "technical",    # Technical indicators, chart patterns
    "fundamentals", # P/E, ROE, financial ratios
    "news",         # News, events, announcements
    "market",       # Market overview, indices, trending
    "risk",         # Risk assessment, volatility
    "crypto",       # Cryptocurrency
    "discovery",    # Stock screening
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
                self.logger.info(
                    f"[PLANNING:INIT] ✅ Registry: {summary['total_tools']} tools "
                    f"in {len(summary['categories'])} categories"
                )
            except Exception as e:
                self.logger.error(f"[PLANNING:INIT] ❌ Registry failed: {e}")
    
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
                working_memory_context=working_memory_context
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
            self.logger.info(f"[PLANNING] ✅ COMPLETE ({elapsed}ms)")
            self.logger.info(f"[PLANNING] Tasks: {len(task_plan.tasks)}")
            self.logger.info(f"[PLANNING] Strategy: {task_plan.strategy}")
            self.logger.info(f"[PLANNING] Symbols: {task_plan.symbols}")
            
            for idx, task in enumerate(task_plan.tasks, 1):
                tools = [t.tool_name for t in task.tools_needed]
                self.logger.info(f"[PLANNING]   Task {idx}: {tools}")
            
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
        working_memory_context: str = None
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
        if self.capability == ModelCapability.ADVANCED:
            result = await self._classify_advanced(query, history, working_memory_context)
        elif self.capability == ModelCapability.INTERMEDIATE:
            result = await self._classify_intermediate(query, history, working_memory_context)
        else:
            result = await self._classify_basic(query, history, working_memory_context)
        
        # Validate categories (filter invalid ones)
        result = self._validate_categories_output(result)
        
        # Auto-add price if symbols present but no price category
        if result.get("symbols") and "price" not in result.get("categories", []):
            result["categories"].append("price")
            self.logger.info(f"[PLANNING:S1] Auto-added 'price' category")
        
        return result
    
    async def _classify_basic(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None
    ) -> Dict[str, Any]:
        """Classification for basic models with Working Memory"""
        
        history_text = self._format_history_for_prompt(history, max_messages=4)
        date_context = self._get_current_date_context()
        
        # Build working memory section
        wm_section = ""
        if working_memory_context:
            wm_section = f"""
WORKING MEMORY (current task state):
{working_memory_context}

Use this context to understand ongoing tasks and symbol references.
"""
        
        prompt = f"""{date_context}

You are a financial query analyzer. Analyze the query semantically.

CURRENT QUERY: {query}

RECENT CONVERSATION:
{history_text}
{wm_section}
YOUR TASK:
1. Understand what the user is asking about (semantic analysis)
2. Extract stock symbols if mentioned (explicit or referenced)
3. Determine which tool categories are needed
4. Detect the response language

CRITICAL: CONVERSATIONAL DETECTION
If the user is:
- Greeting (hello, hi, xin chào)
- Thanking (thanks, cảm ơn, thank you)
- Saying bye (goodbye, tạm biệt)
- General chat without financial request
→ Return query_type: "conversational" with EMPTY categories []

ONLY add categories if user EXPLICITLY asks for financial data.

SYMBOL EXTRACTION RULES:
- Look for uppercase stock tickers: AAPL, MSFT, NVDA, CRM, BTC, ETH
- Check for reference words that point to previous context
- Check Working Memory for current symbols in context
- If user refers to "it", "that stock", etc., use symbols from context

CATEGORIES (select ONLY if explicitly needed):
- price: Stock prices, quotes, performance data
- technical: Technical indicators, chart patterns, RSI, MACD
- fundamentals: P/E ratio, financials, earnings, revenue
- news: News, events, announcements, calendar
- market: Market overview, indices, trending stocks, top gainers/losers
- risk: Risk assessment, volatility, stop loss suggestions
- crypto: Cryptocurrency data (BTC, ETH, etc.)
- discovery: Stock screening, finding stocks by criteria

QUERY TYPES:
- stock_specific: Query about specific stock(s)
- screener: Finding stocks by criteria
- market_level: Market overview, indices
- conversational: Greeting, thanks, general chat (NO CATEGORIES NEEDED)
- memory_recall: Asking about past conversations
- task_continuation: Continuing a previous task

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
        working_memory_context: str = None
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
        
        prompt = f"""{date_context}

<task>Semantic financial query analysis with context awareness</task>

<query>{query}</query>

<history>
{history_text}
</history>
{wm_section}
<instructions>
Analyze the query semantically to understand user intent.

CRITICAL: CONVERSATIONAL DETECTION FIRST
Before anything else, check if this is a conversational message:
- Greetings: hello, hi, xin chào, chào
- Thanks: thanks, thank you, cảm ơn, ok cảm ơn
- Bye: goodbye, tạm biệt
- General chat: how are you, oke

If YES → query_type: "conversational", categories: [], symbols: []

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
- conversational: Greeting/chat (NO CATEGORIES!)
- task_continuation: Continuing previous task

STEP 4: Category Selection
Select ONLY if user explicitly requests:
- price, technical, fundamentals, news, market, risk, crypto, discovery
</instructions>

<o>
Return JSON:
{{
  "query_type": "...",
  "categories": [...],
  "symbols": [...],
  "reasoning": "Semantic analysis explanation",
  "response_language": "vi|en"
}}
</o>
"""
        
        return await self._call_llm_json(prompt, "Stage1")
    
    async def _classify_advanced(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None
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
        
        prompt = f"""{date_context}

<s>Advanced financial query analyzer with context awareness</s>

<input>
<query>{query}</query>
<history>
{history_text}
</history>
{wm_section}
</input>

<instructions>
Think step-by-step to analyze this query semantically.

CRITICAL FIRST CHECK:
Is this a simple conversational message?
- Greetings: hello, hi, xin chào
- Thanks: thanks, cảm ơn, ok cảm ơn  
- Bye: goodbye, tạm biệt
- General chat without financial request

If YES → Return conversational with EMPTY categories

REASONING PROCESS:
1. Is this a continuation of a previous task? Check Working Memory.
2. What is the user semantically asking about?
3. Are there explicit stock symbols?
4. Are there reference words pointing to history or working memory?
5. What is the underlying intent?
6. Which tool categories are ACTUALLY needed?

CATEGORIES:
- price: Stock prices, quotes, performance
- technical: Technical indicators, chart patterns
- fundamentals: Financial ratios, earnings
- news: News, events, calendar
- market: Market overview, indices, trending stocks
- risk: Risk assessment, volatility
- crypto: Cryptocurrency
- discovery: Stock screening
</instructions>

<output_format>
First think in [THOUGHT] block, then output JSON:

[THOUGHT]
... your reasoning ...
[/THOUGHT]
```json
{{
  "query_type": "stock_specific|screener|market_level|conversational|task_continuation",
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
            self.logger.error(f"[PLANNING:S2] ❌ Tool registry not available")
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

DECISION RULES:
1. Greeting/Thanks/General Chat → NO tools (just respond naturally)
2. Past conversation recall → NO tools (use history)
3. Price/quote request with SYMBOL → YES tools (MUST fetch real-time)
4. Financial analysis with SYMBOL → YES tools (MUST fetch data)
5. Stock screening → YES tools (MUST query database)
6. Market overview request → YES tools (MUST fetch current state)

IMPORTANT: Simple greetings like "thanks", "ok cảm ơn" do NOT need tools!

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

Create an execution plan using the available tools.

INTENT: {validated_intent}
QUERY: {query}
SYMBOLS: {symbols if symbols else "None - may need screening"}
LANGUAGE: {response_language}

AVAILABLE TOOLS (use EXACT names):
{tools_text}

PLANNING RULES:
1. Use EXACT tool names from the list above
2. If tool requires symbol → params: {{"symbol": "AAPL"}}
3. Strategy: "parallel" if tasks are independent, "sequential" if dependent
4. Each task should be focused and simple
5. Include all necessary tools for complete analysis

EXAMPLES:

Example 1 - Single Stock Price:
Query: "giá CRM"
Symbols: ["CRM"]
Plan:
{{
  "query_intent": "Get CRM current price",
  "strategy": "parallel",
  "symbols": ["CRM"],
  "response_language": "vi",
  "reasoning": "Single price query for CRM",
  "tasks": [{{
    "id": 1,
    "description": "Get CRM current price",
    "tools_needed": [{{"tool_name": "getStockPrice", "params": {{"symbol": "CRM"}}}}],
    "expected_data": ["price", "change", "volume"],
    "priority": "high",
    "dependencies": []
  }}]
}}

Example 2 - Full Analysis:
Query: "Phân tích AAPL"
Symbols: ["AAPL"]
Plan:
{{
  "query_intent": "Comprehensive analysis of AAPL",
  "strategy": "parallel",
  "symbols": ["AAPL"],
  "response_language": "vi",
  "reasoning": "Full analysis requires price, technicals, fundamentals",
  "tasks": [
    {{
      "id": 1,
      "description": "Get AAPL price",
      "tools_needed": [{{"tool_name": "getStockPrice", "params": {{"symbol": "AAPL"}}}}],
      "expected_data": ["price"],
      "priority": "high",
      "dependencies": []
    }},
    {{
      "id": 2,
      "description": "Get AAPL technicals",
      "tools_needed": [{{"tool_name": "getTechnicalIndicators", "params": {{"symbol": "AAPL"}}}}],
      "expected_data": ["RSI", "MACD"],
      "priority": "medium",
      "dependencies": []
    }},
    {{
      "id": 3,
      "description": "Get AAPL financials",
      "tools_needed": [{{"tool_name": "getFinancialRatios", "params": {{"symbol": "AAPL"}}}}],
      "expected_data": ["PE", "ROE"],
      "priority": "medium",
      "dependencies": []
    }}
  ]
}}

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
            self.logger.warning(f"[PLANNING:S1] ⚠️ Filtered invalid categories: {invalid}")
        
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
        stage: str
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON response"""
        try:
            params = {
                "model_name": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a financial assistant. Return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                "provider_type": self.provider_type,
                "api_key": self.api_key,
                "enable_thinking": False
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
                content = content.split("```")[1].split("```")[0]
            
            content = content.strip()
            
            # Fix JSON braces (common LLM error)
            content = self._fix_json_braces(content)
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"[{stage}] JSON parse error: {e}")
            self.logger.error(f"[{stage}] Content: {content[:300]}")
            raise
        except Exception as e:
            self.logger.error(f"[{stage}] LLM error: {e}", exc_info=True)
            raise
    
    def _fix_json_braces(self, content: str) -> str:
        """
        Fix mismatched braces in JSON response
        
        Common LLM errors:
        - Extra trailing braces: {}}} → {}
        - Missing closing braces: {"a": {"b": 1} → {"a": {"b": 1}}
        """
        if not content:
            return '{}'
        
        # Count opening and closing braces
        open_count = content.count('{')
        close_count = content.count('}')
        
        if open_count == close_count:
            return content
        
        if close_count > open_count:
            # Too many closing braces - remove extras from end
            excess = close_count - open_count
            while excess > 0 and content.endswith('}'):
                content = content[:-1]
                excess -= 1
            return content
        
        if open_count > close_count:
            # Missing closing braces - add at end
            missing = open_count - close_count
            content = content + ('}' * missing)
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
        self.logger.info(f"[PLANNING:S1] ✅ Classification Complete")
        self.logger.info(f"[PLANNING:S1]   Type: {query_type}")
        self.logger.info(f"[PLANNING:S1]   Categories: {categories}")
        self.logger.info(f"[PLANNING:S1]   Symbols: {symbols}")
        self.logger.info(f"[PLANNING:S1]   Language: {language}")
        self.logger.info(f"[PLANNING:S1]   Reasoning: {reasoning[:100]}...")
    
    def _get_elapsed_ms(self, start_time: datetime) -> int:
        """Get elapsed time in milliseconds"""
        return int((datetime.now() - start_time).total_seconds() * 1000)