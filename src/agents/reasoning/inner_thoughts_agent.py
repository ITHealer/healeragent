# Version 2:
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict

from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.config import settings


# ============================================================================
# Decision Models 
# ============================================================================

class RecallParams(BaseModel):
    """Parameters for recall memory search"""
    model_config = ConfigDict(extra='forbid')
    topic: Optional[str] = Field(default=None, description="Topic to search for")
    symbols: Optional[List[str]] = Field(default_factory=list, description="Symbols to search")
    date_filter: Optional[str] = Field(default=None, description="Date filter")
    period: Optional[str] = Field(default=None, description="Time period")
    limit: int = Field(default=10, description="Max results")


class MemorySearchDecision(BaseModel):
    """Decision about memory searches"""
    model_config = ConfigDict(extra='forbid')
    need_recall_search: bool = Field(default=False, description="Search conversation history?")
    need_archival_search: bool = Field(default=False, description="Search knowledge base?") 
    recall_strategy: Optional[str] = Field(default=None, description="temporal/symbol/topic/hybrid")
    recall_params: Optional[RecallParams] = Field(default=None, description="Recall search parameters")
    archival_query: Optional[str] = Field(default=None, description="Query for knowledge base")
    reasoning: str = Field(default="", description="Why this search strategy")


class ToolParams(BaseModel):
    """Parameters for tool execution"""
    model_config = ConfigDict(extra='forbid')
    symbols: Optional[List[str]] = Field(default_factory=list, description="Symbols for tool")
    timeframe: Optional[str] = Field(default=None, description="Time frame")
    additional_params: Optional[Dict[str, str]] = Field(default_factory=dict, description="Other params")


class ToolExecutionDecision(BaseModel):
    """Decision about tool execution"""
    model_config = ConfigDict(extra='forbid')
    need_tool: bool = Field(default=False, description="Execute financial tools?")
    tool_sequence: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Sequence of tools to execute. Each dict: {tool_name, params, purpose}"
    )
    # Keep for backward compatibility and fallback
    tool_name: Optional[str] = Field(default=None, description="Primary tool (deprecated)")
    tool_params: Optional[ToolParams] = Field(default=None, description="Primary tool params (deprecated)")
    
    symbols: List[str] = Field(default_factory=list, description="All extracted symbols")
    temporal_context: Optional[Dict[str, str]] = Field(default_factory=dict, description="Temporal info")
    reasoning: str = Field(default="", description="Why these tools")


class ResponseStrategy(BaseModel):
    """Overall response strategy"""
    model_config = ConfigDict(extra='forbid')
    strategy: str = Field(description="direct_answer/search_first/tool_first/hybrid")
    confidence: float = Field(default=0.5, description="Confidence score 0-1")
    language: str = Field(default="auto", description="Detected language")
    tone: str = Field(default="professional", description="Response tone")


class QueryAnalysis(BaseModel):
    """Query understanding results"""
    model_config = ConfigDict(extra='forbid')
    intent: str = Field(description="What user wants")
    type: str = Field(description="Query type")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    
    
class InnerThoughtsDecision(BaseModel):
    """Complete decision from Inner Thoughts reasoning"""
    model_config = ConfigDict(extra='forbid')
    query_analysis: QueryAnalysis = Field(description="Query intent and type")
    memory_decision: MemorySearchDecision = Field(description="Memory search decisions")
    tool_decision: ToolExecutionDecision = Field(description="Tool execution decisions")
    response_strategy: ResponseStrategy = Field(description="Response approach")
    inner_thoughts: str = Field(description="Brief reasoning summary")


# ============================================================================
# Inner Thoughts Agent
# ============================================================================

class InnerThoughtsAgent(LoggerMixin):
    """
    Reasoning Agent with Deep Agents pattern
    - Analyzes query semantically (multi-language)
    - Decides memory searches
    - Determines tool execution
    """

    def __init__(
        self,
        model_name: Optional[str] = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI
    ):
        super().__init__()
        
        self.model_name = model_name
        self.provider_type = provider_type
    
    
    async def think_and_decide(
        self,
        query: str,
        recent_chat: List[Dict[str, str]],
        core_memory: Dict[str, str],
        summary: Optional[str] = None,
        available_tools: Optional[List[str]] = None
    ) -> InnerThoughtsDecision:
        """
        Main reasoning method - Deep thinking before action
        
        Args:
            query: User query (ANY language)
            recent_chat: Last 5-10 messages
            core_memory: User's core memory
            summary: Recursive summary if exists
            available_tools: List of available tools
            
        Returns:
            Complete decision with memory + tool plans
        """
        try:
            start_time = datetime.now()
            
            # Build reasoning prompt
            reasoning_prompt = self._build_deep_reasoning_prompt(
                query=query,
                recent_chat=recent_chat,
                core_memory=core_memory,
                summary=summary,
                available_tools=available_tools or []
            )
            
            # Get structured decision from LLM
            decision = await self._llm_reasoning(reasoning_prompt)
            
            elapsed = (datetime.now() - start_time).total_seconds() * 1000
            self.logger.info(
                f"[INNER THOUGHTS] {decision.inner_thoughts} ({elapsed:.0f}ms)"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Error in think_and_decide: {e}")
            return self._create_fallback_decision(query)
    
    
#     def _build_deep_reasoning_prompt(
#         self,
#         query: str,
#         recent_chat: List[Dict],
#         core_memory: Dict,
#         summary: Optional[str],
#         available_tools: List[str]
#     ) -> str:
#         """
#         Build reasoning prompt with multi-tool support  
        
#         Args:
#             query: User query (ANY language)
#             recent_chat: Last 5-10 messages
#             core_memory: User's core memory
#             summary: Recursive summary if exists
#             available_tools: List of available tools
            
#         Returns:
#             Reasoning prompt
#         """
        
#         # Format recent chat
#         chat_context = ""
#         if recent_chat:
#             chat_context = "\n".join([
#                 f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}..."
#                 for msg in recent_chat[-5:]
#             ])
        
#         # Core memory summary
#         persona = (core_memory.get('persona', '')[:200] + '...') if core_memory.get('persona') else "Not set"
#         human = (core_memory.get('human', '')[:300] + '...') if core_memory.get('human') else "Not set"
        
#         # Summary status
#         has_summary = bool(summary and len(summary.strip()) > 20)
#         summary_preview = (summary[:300] + '...') if has_summary else "No summary"
        
#         # Available tools
#         tools_list = "\n".join([f"- {tool}" for tool in available_tools]) if available_tools else "No tools"
        

#         # Get current date/time
#         current_date = datetime.now().strftime("%Y-%m-%d")
#         current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         day_of_week = datetime.now().strftime("%A")

#         prompt = f"""You are an intelligent reasoning agent for a financial chatbot.
# Analyze the query and determine the optimal strategy using ALL available context.

# **Current Date**: 
# {current_date} ({day_of_week})

# **Current DateTime**: 
# {current_datetime}

# ==================================================
# CONTEXT
# ==================================================

# **User Query (in ANY language):**
# {query}

# **Recent Conversation:**
# {chat_context or "No recent conversation"}

# **Core Memory:**
# - Persona: {persona}
# - Human: {human}

# **Conversation Summary:**
# {summary_preview}

# **Available Tools:**
# {tools_list}

# ==================================================
# CRITICAL INSTRUCTION: MULTI-TOOL ANALYSIS
# ==================================================

# **Many queries require MULTIPLE tools to answer completely.**

# Carefully analyze if the query has:
# 1. Multiple parts (connected by "AND", "và", commas, etc.)
# 2. Comparison requests (compare X vs Y)
# 3. Multiple data needs (price + news, analysis + market overview)

# **Examples requiring MULTIPLE tools:**

# **Example 1 - Stock Analysis (3 tools)**:
# Query: "Phân tích cổ phiếu AAPL" or "Analyze AAPL"
# → 3 tools needed for complete analysis:
#   1. showStockPrice (current price, valuation)
#   2. showStockFinancials (fundamentals, earnings)
#   3. showStockNews (recent developments)
# Reasoning: "Phân tích"/"analyze" = comprehensive view = price + fundamentals + news

# **Example 2 - Stock + Market Context (2 tools)**:
# Query: "Analyze NVDA and market overview" / "Phân tích NVDA và tổng quan thị trường"
# → 2 tools:
#   1. showStockFinancials (NVDA)
#   2. showMarketOverview (market context)

# **Example 3 - Price + News (2 tools)**:
# Query: "NVDA price and latest news"
# → 2 tools:
#   1. showStockPrice (current price)
#   2. showStockNews (news)

# **Example 4 - Stock Comparison (2-3 tools)**:
# Query: "Compare AAPL and MSFT"
# → Multiple calls to showStockFinancials or showStockPrice for each symbol

# **Example 5 - Full Analysis Query (3 tools)**:
# Query: "Phân tích chỉ số AAPL và tổng quan thị trường hôm nay"
# → 3 tools needed:
#   1. showStockPrice (AAPL current price today)
#   2. showStockFinancials (AAPL fundamentals)
#   3. showMarketOverview (market conditions today)
# Reasoning: "Phân tích chỉ số" = need price + fundamentals + market context

# **Example 6 - Technical Analysis (2 tools)**:
# Query: "Phân tích kỹ thuật VNM ngày 1/11 và tổng quan thị trường"
# → 2 tools:
#   1. showStockChart (VNM technical)
#   2. showMarketOverview (VN market)

# **CRITICAL DETECTION RULES**:
# - "phân tích" / "analyze" + stock symbol → Need AT LEAST 2-3 tools (price + financials, optionally news)
# - "phân tích chỉ số" / "analyze stock" → MUST include showStockPrice (for current price)
# - "và tổng quan" / "and overview" → Add showMarketOverview
# - "và tin tức" / "and news" → Add showStockNews  
# - "kỹ thuật" / "technical" → Use showStockChart
# - Simple "giá" / "price" query → Only showStockPrice

# **If query requires only ONE tool, return sequence with 1 tool**
# **If query requires MULTIPLE tools, return sequence with ALL necessary tools**

# ==================================================
# TOOL DESCRIPTIONS (READ CAREFULLY)
# ==================================================

# **PRICE & VALUATION TOOLS**:
# - showStockPrice: Current price, price performance, valuation metrics, price targets. 
#   Returns: Current price, day change, 52-week range, market cap, P/E ratio, price trends.
#   Use for: "AAPL price?", "How is X performing?", "X valuation", "giá cổ phiếu"

# - cryptoChart: Cryptocurrency prices and performance data.
#   Returns: Current crypto price, 24h change, volume, market cap.
#   Use for: "Bitcoin price", "BTC today", "giá ETH"

# **FUNDAMENTAL ANALYSIS TOOLS**:
# - showStockFinancials: Financial statements, earnings, revenue, margins, cash flow.
#   Returns: Quarterly/annual financials, earnings reports, balance sheet, P/E, revenue growth.
#   Use for: "AAPL financials", "earnings report", "revenue", "báo cáo tài chính"

# **TECHNICAL & CHART TOOLS**:
# - showStockChart: Visual charts with technical indicators and patterns.
#   Returns: Price charts with indicators (RSI, MACD, MA), volume, trends.
#   Use for: "AAPL chart", "technical analysis", "biểu đồ", "kỹ thuật"

# **NEWS & EVENTS TOOLS**:
# - showStockNews: Latest news, events, announcements for stocks/crypto.
#   Returns: Recent news articles, events, market-moving information.
#   Use for: "AAPL news", "why is X moving", "tin tức", "sự kiện"

# **MARKET OVERVIEW TOOLS**:
# - showMarketOverview: Overall market conditions, indices (S&P 500, NASDAQ, DOW).
#   Returns: Market indices, sector performance, market sentiment, trends.
#   Use for: "market today", "how's the market", "thị trường", "tổng quan"

# - showTrendingStocks: Currently trending stocks and top movers.
#   Returns: Trending stocks, gainers, losers, most active.
#   Use for: "trending stocks", "top movers", "cổ phiếu nổi bật"

# - showStockHeatmap: Visual heatmap of stock performance.
#   Returns: Visual representation of market/sector performance.
#   Use for: "market heatmap", "sector performance visualization"

# ==================================================
# YOUR TASK
# ==================================================

# Analyze the query and decide:

# 1. **Query Analysis**: Understand intent, type, entities
# 2. **Memory Search**: Do we need recall/archival search?
# 3. **Tool Execution**: 
#    - Identify ALL tools needed (could be 1, 2, 3, or more)
#    - For each tool, specify:
#      * tool_name
#      * params (symbols, timeframe, etc.)
#      * purpose (why this tool)
#    - Order tools logically
# 4. **Response Strategy**: Direct answer, tool-first, hybrid, etc.

# ==================================================
# OUTPUT FORMAT (MUST BE VALID JSON)
# ==================================================

# Return ONLY valid JSON in this exact structure:

# {{
#   "query_analysis": {{
#     "intent": "string describing what user wants",
#     "type": "analysis/comparison/lookup/news/general",
#     "entities": ["list", "of", "extracted", "entities"]
#   }},
#   "memory_decision": {{
#     "need_recall_search": false,
#     "need_archival_search": false,
#     "recall_strategy": null,
#     "recall_params": null,
#     "archival_query": null,
#     "reasoning": "why search or not search memory"
#   }},
#   "tool_decision": {{
#     "need_tool": true,
#     "tool_sequence": [
#       {{
#         "tool_name": "showStockPrice",
#         "params": {{
#           "symbols": ["AAPL"]
#         }},
#         "purpose": "Get current AAPL price and valuation metrics"
#       }},
#       {{
#         "tool_name": "showStockFinancials",
#         "params": {{
#           "symbols": ["AAPL"]
#         }},
#         "purpose": "Get AAPL fundamental analysis and financial data"
#       }},
#       {{
#         "tool_name": "showMarketOverview",
#         "params": {{}},
#         "purpose": "Get overall market conditions today"
#       }}
#     ],
#     "symbols": ["AAPL"],
#     "temporal_context": {{}},
#     "reasoning": "Query 'phân tích chỉ số AAPL và tổng quan thị trường' requires comprehensive analysis: current price + fundamentals + market context = 3 tools"
#   }},
#   "response_strategy": {{
#     "strategy": "tool_first",
#     "confidence": 0.95,
#     "language": "vi",
#     "tone": "professional"
#   }},
#   "inner_thoughts": "User wants complete AAPL analysis with market context. Need 3 tools: price (current value), financials (fundamentals), market overview (context)"
# }}

# ==================================================
# IMPORTANT RULES
# ==================================================

# 1. **tool_sequence** is an ARRAY - can have 1, 2, 3 or more tools
# 2. Each tool in sequence has: tool_name (string), params (object), purpose (string)
# 3. Extract ALL symbols mentioned in query
# 4. If query has temporal reference (date, yesterday, today, hôm nay, etc.), include in params
# 5. Order tools logically (usually: price → financials → news → market)
# 6. If only 1 tool needed, tool_sequence has 1 element
# 7. If 2+ tools needed, tool_sequence has 2+ elements
# 8. Language detection: "vi" for Vietnamese, "en" for English, "auto" if mixed

# **KEYWORD-BASED TOOL SELECTION**:
# - **"phân tích" / "analyze" + symbol** → MINIMUM 2 tools (showStockPrice + showStockFinancials)
# - **"phân tích chỉ số" / "analyze stock"** → MUST include showStockPrice for current price
# - **"kỹ thuật" / "technical"** → Use showStockChart
# - **"tin tức" / "news"** → Add showStockNews
# - **"tổng quan" / "overview"** → Add showMarketOverview
# - **"giá" / "price" alone** → Only showStockPrice
# - **"biểu đồ" / "chart"** → Only showStockChart

# **RESPOND WITH VALID JSON ONLY - NO OTHER TEXT**
# """

#         return prompt
    

    def _build_deep_reasoning_prompt(
        self,
        query: str,
        recent_chat: List[Dict],
        core_memory: Dict,
        summary: Optional[str],
        available_tools: List[str]
    ) -> str:
        """
        Build reasoning prompt with multi-tool support
        CRITICAL: Current query is PRIMARY, past context is SECONDARY
        """
        
        # Format recent chat history
        chat_context = ""
        if recent_chat:
            chat_context = "\nRecent conversation (for reference only):\n"
            for msg in recent_chat[-5:]:  # Last 5 messages
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')[:100]  # Truncate
                chat_context += f"- {role}: {content}\n"
        
        # Format core memory
        memory_context = ""
        if core_memory and (core_memory.get('human') or core_memory.get('persona')):
            memory_context = "\nCore Memory (background info):\n"
            if core_memory.get('human'):
                memory_context += f"About user: {core_memory['human'][:200]}\n"
            if core_memory.get('persona'):
                memory_context += f"About assistant: {core_memory['persona'][:200]}\n"
        
        # Format summary
        summary_context = ""
        if summary:
            summary_context = f"\nConversation Summary (past topics): {summary[:300]}\n"
        
        prompt = f"""You are an intelligent reasoning agent for a financial chatbot. Your task is to analyze the CURRENT user query and decide what actions to take.

    ═══════════════════════════════════════════════════════════════
    🎯 CRITICAL INSTRUCTION - READ CAREFULLY
    ═══════════════════════════════════════════════════════════════

    **PRIMARY FOCUS: CURRENT QUERY**
    The user's CURRENT query below is THE MOST IMPORTANT factor in your decision.
    Past context (memory, summary, chat history) is ONLY for REFERENCE and should NEVER override the current query's intent.

    **CURRENT USER QUERY** (THIS IS WHAT MATTERS MOST):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    >>> "{query}" <
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ═══════════════════════════════════════════════════════════════
    📋 DECISION FRAMEWORK (FOLLOW THIS ORDER)
    ═══════════════════════════════════════════════════════════════

    **STEP 1: ANALYZE CURRENT QUERY FIRST**
    Before looking at any past context, determine the CURRENT query type:

    **GREETING/CASUAL CHAT** (NO TOOLS NEEDED):
    - Greetings: "hi", "hello", "hey", "chào", "xin chào", "hi there"
    - Casual chat: "how are you", "what's up", "thanks", "bye", "ok", "cool"
    - Acknowledgments: "got it", "understood", "I see", "alright"
    - Simple questions: "what can you do?", "help", "who are you?"

    **If current query is greeting/casual → STOP HERE:**
    {{
    "need_tool": false,
    "strategy": "direct_answer",
    "reasoning": "Greeting or casual conversation, no tools needed"
    }}

    **FINANCIAL QUERY** (TOOLS MIGHT BE NEEDED):
    - Explicit stock mention: "analyze AAPL", "AAPL price", "how is Tesla doing?"
    - Crypto mention: "Bitcoin price", "ETH analysis"
    - Market query: "market overview", "trending stocks", "what's happening in the market?"
    - Financial data request: "show me financials", "revenue growth", "P/E ratio"

    **If current query is financial BUT vague/incomplete → Ask for clarification FIRST:**
    Examples:
    - "analyze it" → "Which stock would you like me to analyze?"
    - "show me the chart" → "Which symbol's chart would you like to see?"
    - "what about the price?" → "Which stock's price are you asking about?"

    **STEP 2: CHECK IF QUERY IS COMPLETE**
    Does the current query contain enough information to proceed?
    - ✅ Complete: "Analyze AAPL stock", "Bitcoin price today", "Market overview"
    - ❌ Incomplete: "what about it?", "show me", "how is it doing?"

    **For incomplete queries:** Check recent chat (last 1-2 messages ONLY) for context.
    - If recent chat has symbol → Use that symbol
    - If recent chat is unrelated → Ask user for clarification

    **STEP 3: DETERMINE IF TOOLS ARE NEEDED**
    Only use tools if:
    1. Current query explicitly requests financial data/analysis
    2. Query is complete (has symbol or clear intent)
    3. Query is NOT just greeting/acknowledgment

    ═══════════════════════════════════════════════════════════════
    📚 BACKGROUND CONTEXT (FOR REFERENCE ONLY - DO NOT LET THIS OVERRIDE CURRENT QUERY)
    ═══════════════════════════════════════════════════════════════
    {memory_context}
    {summary_context}
    {chat_context}

    **⚠️ WARNING:** 
    - Do NOT assume user wants to continue previous topic unless current query explicitly refers to it
    - Do NOT use tools just because past context mentions a symbol
    - Do NOT analyze stocks unless current query asks for it
    - IGNORE past context if current query is greeting/casual

    ═══════════════════════════════════════════════════════════════
    🛠️ AVAILABLE TOOLS
    ═══════════════════════════════════════════════════════════════

    Tools you can use (ONLY if current query needs them):
    {', '.join(available_tools)}

    **Tool Guidelines:**
    - showStockPrice: Current price, performance, targets
    - showStockFinancials: Income/balance/cashflow statements
    - showStockChart: Technical analysis, indicators
    - showStockNews: Latest news and sentiment
    - showMarketOverview: Overall market conditions (NO symbol needed)
    - showTrendingStocks: Top movers, gainers, losers

    ═══════════════════════════════════════════════════════════════
    📝 RESPONSE FORMAT
    ═══════════════════════════════════════════════════════════════

    You MUST respond with valid JSON in this EXACT format:

    {{
    "query_analysis": {{
        "intent": "greeting/casual/stock_analysis/crypto_analysis/market_overview/clarification_needed",
        "type": "greeting/question/command/request",
        "entities": ["list", "of", "extracted", "entities"]
    }},
    "memory_decision": {{
        "need_recall_search": false,
        "need_archival_search": false,
        "recall_strategy": null,
        "recall_params": null,
        "archival_query": null,
        "reasoning": ""
    }},
    "tool_decision": {{
        "need_tool": false or true,
        "tool_sequence": [
        {{
            "tool_name": "toolName",
            "params": {{"symbols": ["SYMBOL"], "timeframe": "optional"}},
            "purpose": "Why this tool"
        }}
        ],
        "symbols": ["extracted", "symbols"],
        "temporal_context": {{}},
        "reasoning": "Why these tools or why no tools"
    }},
    "response_strategy": {{
        "strategy": "direct_answer/search_first/tool_first/hybrid",
        "confidence": 0.0 to 1.0,
        "language": "vi/en/auto",
        "tone": "professional/casual"
    }},
    "inner_thoughts": "Brief reasoning about decision"
    }}

    ═══════════════════════════════════════════════════════════════
    ✅ CORRECT EXAMPLES
    ═══════════════════════════════════════════════════════════════

    **Example 1: Greeting (NO TOOLS)**
    Current Query: "hi"
    Past Context: User previously asked about AAPL

    CORRECT Response:
    {{
    "query_analysis": {{"intent": "greeting", "type": "greeting", "entities": []}},
    "memory_decision": {{"need_recall_search": false, "need_archival_search": false, "reasoning": "Simple greeting"}},
    "tool_decision": {{"need_tool": false, "tool_sequence": [], "symbols": [], "reasoning": "Greeting requires no tools"}},
    "response_strategy": {{"strategy": "direct_answer", "confidence": 1.0, "language": "en", "tone": "casual"}},
    "inner_thoughts": "User is greeting me, respond warmly without tools"
    }}

    **Example 2: Casual chat (NO TOOLS)**
    Current Query: "thanks, that's helpful"
    Past Context: User previously asked about Bitcoin

    CORRECT Response:
    {{
    "query_analysis": {{"intent": "casual", "type": "acknowledgment", "entities": []}},
    "tool_decision": {{"need_tool": false, "tool_sequence": [], "symbols": [], "reasoning": "Acknowledgment, no action needed"}},
    "response_strategy": {{"strategy": "direct_answer", "confidence": 1.0, "language": "en", "tone": "casual"}},
    "inner_thoughts": "User is acknowledging previous response, no tools needed"
    }}

    **Example 3: Explicit stock request (USE TOOLS)**
    Current Query: "Analyze AAPL stock"

    CORRECT Response:
    {{
    "query_analysis": {{"intent": "stock_analysis", "type": "request", "entities": ["AAPL"]}},
    "tool_decision": {{
        "need_tool": true,
        "tool_sequence": [
        {{"tool_name": "showStockPrice", "params": {{"symbols": ["AAPL"]}}, "purpose": "Get current price"}},
        {{"tool_name": "showStockFinancials", "params": {{"symbols": ["AAPL"]}}, "purpose": "Get financials"}}
        ],
        "symbols": ["AAPL"],
        "reasoning": "User explicitly requests AAPL analysis"
    }},
    "response_strategy": {{"strategy": "tool_first", "confidence": 0.95, "language": "en"}},
    "inner_thoughts": "Clear stock analysis request, need price and financial data"
    }}

    **Example 4: Vague query with context (USE CONTEXT CAREFULLY)**
    Current Query: "what about the price?"
    Recent Chat (last message): "User: analyze AAPL"

    CORRECT Response:
    {{
    "query_analysis": {{"intent": "stock_analysis", "type": "question", "entities": ["AAPL"]}},
    "tool_decision": {{
        "need_tool": true,
        "tool_sequence": [{{"tool_name": "showStockPrice", "params": {{"symbols": ["AAPL"]}}, "purpose": "Get AAPL price"}}],
        "symbols": ["AAPL"],
        "reasoning": "Query is vague but recent context shows user discussing AAPL"
    }},
    "inner_thoughts": "Vague query but clear from immediate context user means AAPL price"
    }}

    **Example 5: Vague query with UNRELATED context (ASK FOR CLARIFICATION)**
    Current Query: "what about the price?"
    Recent Chat (last message): "User: thanks" (unrelated)

    CORRECT Response:
    {{
    "query_analysis": {{"intent": "clarification_needed", "type": "question", "entities": []}},
    "tool_decision": {{"need_tool": false, "tool_sequence": [], "symbols": [], "reasoning": "Query is vague and no recent context"}},
    "response_strategy": {{"strategy": "direct_answer", "confidence": 0.3}},
    "inner_thoughts": "Query is vague, need to ask which stock user means"
    }}

    ═══════════════════════════════════════════════════════════════
    ❌ WRONG EXAMPLES (WHAT NOT TO DO)
    ═══════════════════════════════════════════════════════════════

    **WRONG Example 1:**
    Current Query: "hi"
    Past Context: User previously discussed AAPL

    ❌ WRONG Response:
    {{
    "tool_decision": {{"need_tool": true, "tool_sequence": [{{"tool_name": "showStockPrice", "params": {{"symbols": ["AAPL"]}}}}]}}
    }}
    ❌ Why wrong: Greeting should NOT trigger tools just because of past context!

    **WRONG Example 2:**
    Current Query: "ok got it"
    Past Context: User asked about Bitcoin

    ❌ WRONG Response:
    {{
    "tool_decision": {{"need_tool": true, "tool_sequence": [{{"tool_name": "cryptoChart"}}]}}
    }}
    ❌ Why wrong: Acknowledgment should NOT trigger tools!

    **WRONG Example 3:**
    Current Query: "Thanks for the analysis"
    Past Context: User asked about TSLA

    ❌ WRONG Response:
    {{
    "tool_decision": {{"need_tool": true, "tool_sequence": [{{"tool_name": "showStockPrice"}}]}}
    }}
    ❌ Why wrong: User is thanking, not requesting new analysis!

    ═══════════════════════════════════════════════════════════════
    🎯 CRITICAL REMINDERS BEFORE YOU RESPOND
    ═══════════════════════════════════════════════════════════════

    1. ✅ READ CURRENT QUERY FIRST - This is what matters most!
    2. ✅ If greeting/casual → need_tool: false, DONE
    3. ✅ If financial query but vague → Ask for clarification
    4. ✅ If financial query and complete → Use tools
    5. ✅ Past context is REFERENCE ONLY - Do NOT let it override current intent
    6. ✅ When in doubt, choose NO TOOLS rather than unnecessary tools

    **Golden Rule:** If current query doesn't explicitly ask for financial data, DON'T use tools!

    ═══════════════════════════════════════════════════════════════

    Now analyze the CURRENT QUERY at the top and provide your decision in valid JSON format.

    REMEMBER: Current query "{query}" is PRIMARY. Past context is SECONDARY reference only!
    """
        
        return prompt
    
    async def _llm_reasoning(self, prompt: str) -> InnerThoughtsDecision:
        """
        Get structured decision from LLM
        """
        try:
            return await self._llm_reasoning_json_mode(prompt)
                
        except Exception as e:
            self.logger.error(f"LLM reasoning error: {e}")
            raise
    
    
    async def _llm_reasoning_json_mode(self, prompt: str) -> InnerThoughtsDecision:
        if self.provider_type == ProviderType.OPENAI:
            api_key = ModelProviderFactory._get_api_key(ProviderType.OPENAI)
            client = OpenAI(api_key=api_key)
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an intelligent reasoning agent. Always respond with valid JSON only. Support multiple tool execution."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},  # JSON mode
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
        else:  # Ollama
            base_url = settings.OLLAMA_ENDPOINT
            if not base_url.endswith('/v1'):
                base_url = f"{base_url}/v1"
                
            client = OpenAI(base_url=base_url, api_key="ollama")
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a reasoning agent. Always respond with valid JSON only. Support multiple tool execution."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.1,
                format="json"  # Ollama-specific parameter
            )
            
            content = response.choices[0].message.content
        
        # Parse and validate JSON
        try:
            decision_dict = json.loads(content)
            return self._parse_decision_dict(decision_dict)
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.debug(f"Raw response: {content[:500]}...")
            raise ValueError(f"Invalid JSON from LLM: {e}")
    
    
    def _parse_decision_dict(self, data: Dict) -> InnerThoughtsDecision:
        """Parse decision dictionary into Pydantic models"""
        try:
            # Parse query_analysis
            query_data = data.get('query_analysis', {})
            query_analysis = QueryAnalysis(
                intent=query_data.get('intent', 'unknown'),
                type=query_data.get('type', 'general'),
                entities=query_data.get('entities', [])
            )
            
            # Parse memory_decision with nested recall_params
            memory_data = data.get('memory_decision', {})
            recall_params = None
            if memory_data.get('recall_params'):
                recall_params = RecallParams(**memory_data['recall_params'])
            
            memory_decision = MemorySearchDecision(
                need_recall_search=memory_data.get('need_recall_search', False),
                need_archival_search=memory_data.get('need_archival_search', False),
                recall_strategy=memory_data.get('recall_strategy'),
                recall_params=recall_params,
                archival_query=memory_data.get('archival_query'),
                reasoning=memory_data.get('reasoning', '')
            )
            
            # Parse tool_decision with tool_sequence support
            tool_data = data.get('tool_decision', {})
            
            # Parse tool_sequence - already as List[Dict]
            tool_sequence = tool_data.get('tool_sequence', [])
            
            # Validate tool_sequence structure
            validated_sequence = []
            for step_data in tool_sequence:
                if isinstance(step_data, dict):
                    # Ensure required fields exist
                    validated_step = {
                        'tool_name': step_data.get('tool_name', ''),
                        'params': step_data.get('params', {}),
                        'purpose': step_data.get('purpose', '')
                    }
                    validated_sequence.append(validated_step)
            
            tool_sequence = validated_sequence
            
            # Fallback: If tool_sequence empty but tool_name exists (backward compat)
            if not tool_sequence and tool_data.get('tool_name'):
                params_dict = {}
                if tool_data.get('tool_params'):
                    # Convert ToolParams to dict
                    tool_params_obj = tool_data['tool_params']
                    if isinstance(tool_params_obj, dict):
                        params_dict = tool_params_obj
                    else:
                        params_dict = {
                            'symbols': tool_params_obj.get('symbols', []),
                            'timeframe': tool_params_obj.get('timeframe'),
                            'additional_params': tool_params_obj.get('additional_params', {})
                        }
                
                tool_sequence = [{
                    'tool_name': tool_data['tool_name'],
                    'params': params_dict,
                    'purpose': "Primary tool execution"
                }]
            
            # Parse legacy tool_params for backward compatibility
            tool_params = None
            if tool_data.get('tool_params'):
                tool_params = ToolParams(**tool_data['tool_params'])
            
            tool_decision = ToolExecutionDecision(
                need_tool=tool_data.get('need_tool', False),
                tool_sequence=tool_sequence,
                tool_name=tool_data.get('tool_name'),  # Keep for compat
                tool_params=tool_params,  # Keep for compat
                symbols=tool_data.get('symbols', []),
                temporal_context=tool_data.get('temporal_context', {}),
                reasoning=tool_data.get('reasoning', '')
            )
            
            # Parse response_strategy
            strategy_data = data.get('response_strategy', {})
            response_strategy = ResponseStrategy(
                strategy=strategy_data.get('strategy', 'direct_answer'),
                confidence=strategy_data.get('confidence', 0.5),
                language=strategy_data.get('language', 'auto'),
                tone=strategy_data.get('tone', 'professional')
            )
            
            # Create final decision
            return InnerThoughtsDecision(
                query_analysis=query_analysis,
                memory_decision=memory_decision,
                tool_decision=tool_decision,
                response_strategy=response_strategy,
                inner_thoughts=data.get('inner_thoughts', 'Reasoning completed')
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing decision dict: {e}")
            self.logger.debug(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)[:500]}")
            raise ValueError(f"Failed to parse decision: {e}")
    
    
    def _create_fallback_decision(self, query: str) -> InnerThoughtsDecision:
        """Fallback if reasoning fails"""
        return InnerThoughtsDecision(
            query_analysis=QueryAnalysis(
                intent="unknown",
                type="general",
                entities=[]
            ),
            memory_decision=MemorySearchDecision(
                need_recall_search=False,
                need_archival_search=False,
                reasoning="Fallback - no search"
            ),
            tool_decision=ToolExecutionDecision(
                need_tool=False,
                tool_sequence=[],
                reasoning="Fallback - no tools"
            ),
            response_strategy=ResponseStrategy(
                strategy="direct_answer",
                confidence=0.3,
                language="auto",
                tone="professional"
            ),
            inner_thoughts="Using fallback due to reasoning error"
        )