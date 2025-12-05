import json
from datetime import datetime
from typing import Dict, List, Optional, Any

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


class PlanningAgent(LoggerMixin):
    """
    PATTERN:
    - Stage 1: SEMANTIC category selection (NOT keyword matching)
    - Stage 2: Load ALL tools from categories  
    - Stage 3: Show COMPLETE tool schemas to LLM
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: str = None,
    ):
        super().__init__()
        
        self.model_name = model_name or "gpt-4.1-nano"
        self.provider_type = provider_type or ProviderType.OPENAI
        
        # Initialize LLM Provider
        self.llm_provider = LLMGeneratorProvider()
        
        # Get API key
        self.api_key = ModelProviderFactory._get_api_key(self.provider_type)
        
        # Initialize Tool Registry
        self.tool_registry = None
        if ATOMIC_TOOLS_AVAILABLE:
            try:
                self.tool_registry = get_registry()
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
    

    async def think_and_plan(
        self,
        query: str,
        recent_chat: List[Dict[str, str]] = None,
        core_memory: Dict[str, Any] = None,
        summary: str = None,
    ) -> TaskPlan:
        """
        Main planning method: 3-stage progressive disclosure
        
        Args:
            query: User query
            recent_chat: Recent chat history
            core_memory: User memory
            summary: Conversation summary
            
        Returns:
            TaskPlan with structured tasks
        """
        try:
            start_time = datetime.now()
            
            # Build context (for Stage 3 only)
            context = self._build_context_string(recent_chat, core_memory, summary)
            
            # ================================================================
            # STAGE 1: SEMANTIC CATEGORY SELECTION
            # ================================================================
            classification = await self._classify_query_and_select_categories(query)
            
            query_type = classification.get("query_type", "stock_specific")
            categories = classification.get("categories", [])
            symbols = classification.get("symbols", [])
            response_language = classification.get("response_language", "auto")
            
            if symbols:
              if "price" not in categories:
                  self.logger.info(f"[STAGE 1] Auto-adding 'price' category because symbols {symbols} were detected")
                  categories.append("price")
              
              # Auto-add 'news' for better context if analyzing stock
              if query_type == "stock_specific" and "news" not in categories:
                  categories.append("news")
                  
            self.logger.info(f"[STAGE 1] ✅ Categories: {categories}")
            self.logger.info(f"[STAGE 1] Query type: {query_type}, Symbols: {symbols}")
            
            # If conversational, return empty plan
            if query_type == "conversational" or not categories:
                return TaskPlan(
                    tasks=[],
                    query_intent=query_type,
                    strategy="direct_answer",
                    response_language=response_language,
                    reasoning="Conversational query, no tools needed"
                )
            
            # ================================================================
            # STAGE 2: Load ALL Tools from Selected Categories
            # ================================================================
            self.logger.info(f"[STAGE 2] Loading tools from {len(categories)} categories...")
            
            available_tools = []
            for category in categories:
                # Get tools in category
                tools_dict = self.tool_registry.get_tools_by_category(category)
                
                for tool_name, tool_instance in tools_dict.items():
                    # Get schema and convert to JSON
                    schema = self.tool_registry.get_schema(tool_name)
                    if schema:
                        tool_json = schema.to_json_schema()
                        available_tools.append(tool_json)
            
            self.logger.info(f"[STAGE 2] ✅ Loaded {len(available_tools)} tools from {len(categories)} categories")
            
            if not available_tools:
                self.logger.warning("[STAGE 2] No tools found, loading defaults")
                # Fallback: get tools from common categories
                for cat in ['price', 'news']:
                    tools_dict = self.tool_registry.get_tools_by_category(cat)
                    for tool_name in list(tools_dict.keys())[:5]:
                        schema = self.tool_registry.get_schema(tool_name)
                        if schema:
                            tool_json = schema.to_json_schema()
                            available_tools.append(tool_json)
            
            # ================================================================
            # STAGE 3: Show COMPLETE Tool Schemas
            # ================================================================
            self.logger.info("[STAGE 3] Planning with complete tool schemas...")
            
            task_plan_dict = await self._create_detailed_task_plan(
                query=query,
                query_type=query_type,
                symbols=symbols,
                response_language=response_language,
                available_tools=available_tools,
                context=context
            )
            
            # Parse into TaskPlan
            task_plan = self._parse_task_plan_dict(
                task_plan_dict, 
                symbols, 
                response_language
            )
            
            elapsed = int((datetime.now() - start_time).total_seconds() * 1000)
            self.logger.info(
                f"[PLANNING] ✅ {len(task_plan.tasks)} tasks planned "
                f"({task_plan.estimated_complexity}) ({elapsed}ms)"
            )
            
            return task_plan
            
        except Exception as e:
            self.logger.error(f"Error in think_and_plan: {e}", exc_info=True)
            return TaskPlan(
                tasks=[],
                query_intent="error",
                strategy="direct_answer",
                response_language="auto",
                reasoning=f"Planning error: {str(e)}"
            )
    
    # ========================================================================
    # STAGE 1: SEMANTIC CATEGORY SELECTION
    # ========================================================================
    
    async def _classify_query_and_select_categories(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        STAGE 1: SEMANTIC category selection
        
        CRITICAL: 
        - Understand MEANING, not keywords
        - Select ALL relevant categories
        - NO context (context only in Stage 3)
        """
        
        prompt = f"""You are a financial query analyzer with SEMANTIC understanding.

{'═' * 80}
QUERY (Analyze semantic meaning)
{'═' * 80}

>>> {query} <<<

{'═' * 80}
AVAILABLE CATEGORIES (Select ALL relevant)
{'═' * 80}

discovery: Find/screen stocks by criteria
  → Use when: User wants to DISCOVER stocks (NO specific symbol)
  → Examples: "tìm cổ phiếu", "find stocks", "screen for"
  
price: Stock prices, quotes, performance
  → Use when: User asks about PRICE/PERFORMANCE
  → Examples: "giá", "price", "performance", "how much"
  
technical: RSI, MACD, Bollinger, chart patterns  
  → Use when: User mentions TECHNICAL indicators
  → Examples: "RSI", "MACD", "overbought", "oversold", "kỹ thuật"
  
fundamentals: P/E, ROE, financial ratios, statements
  → Use when: User asks about FINANCIAL metrics
  → Examples: "P/E", "ROE", "tài chính", "fundamentals", "earnings"
  
risk: Volatility, risk assessment, stop-loss
  → Use when: User mentions RISK analysis
  → Examples: "risk", "volatility", "rủi ro", "an toàn"
  
news: News articles, events, earnings calendar
  → Use when: User asks about NEWS/EVENTS
  → Examples: "tin tức", "news", "latest", "announcement"
  
market: Market indices, sector performance, heatmaps
  → Use when: User asks about OVERALL market
  → Examples: "market", "thị trường", "sector", "overall"
  
crypto: Cryptocurrency prices and technicals
  → Use when: User mentions CRYPTO (BTC, ETH, etc.)
  → Examples: "bitcoin", "crypto", "BTC", "ETH"

{'═' * 80}
SEMANTIC ANALYSIS RULES (NOT keyword matching!)
{'═' * 80}

Think step-by-step about MEANING:

1. SPECIFIC COMPANY mentioned? (Apple, AAPL, Tesla, TSLA)
   → Extract symbol
   → What does user want to know? (price? tech? fundamentals?)
   → Select categories for those aspects

2. DISCOVERY query? (find, search, screen, tìm)
   → SELECT "discovery"
   → What criteria? (P/E? RSI? News?)
   → Select additional categories for criteria

3. MARKET overview? (how's market, thị trường)
   → SELECT "market"
   
4. GREETING? (hello, hi, xin chào)
   → query_type: "conversational"
   → categories: []

EXAMPLE REASONING:

Query: "Tìm cổ phiếu công nghệ có P/E thấp, RSI oversold, tin tức tốt"
Meaning: Find stocks (discovery) + P/E metric (fundamentals) + RSI (technical) + news (news)
Output: {{"categories": ["discovery", "fundamentals", "technical", "news"]}}

Query: "Phân tích AAPL về giá và kỹ thuật"  
Meaning: Analyze AAPL (symbol given) + price (price) + technicals (technical)
Output: {{"categories": ["price", "technical"], "symbols": ["AAPL"]}}

Query: "How's the market today?"
Meaning: Market overview (market)
Output: {{"categories": ["market"]}}

{'═' * 80}
CRITICAL RULES
{'═' * 80}

✅ SELECT ALL RELEVANT CATEGORIES (not just one!)
✅ Understand MEANING, not keywords
✅ Specific symbol → NEVER "discovery"
✅ User wants to FIND → ALWAYS include "discovery"

{'═' * 80}
OUTPUT (JSON ONLY)
{'═' * 80}

{{
  "query_type": "stock_specific" | "screener" | "market_level" | "conversational",
  "categories": ["cat1", "cat2", "cat3"],
  "symbols": ["AAPL"],
  "reasoning": "Brief explanation of why these categories",
  "response_language": "vi" | "en"
}}

Now analyze semantically:
"""
        
        return await self._call_llm_and_parse_json(prompt, "Stage 1")
    
    # ========================================================================
    # STAGE 3: DETAILED TASK PLANNING
    # ========================================================================
    
    async def _create_detailed_task_plan(
        self,
        query: str,
        query_type: str,
        symbols: List[str],
        response_language: str,
        available_tools: List[Dict[str, Any]],
        context: str
    ) -> Dict[str, Any]:
        """
        STAGE 3: Create detailed task plan with COMPLETE tool schemas
        
        CRITICAL: Show EXACT tool names from registry
        """
        
        # Build prompt with COMPLETE tool information
        prompt = self._build_detailed_task_prompt(
            query=query,
            query_type=query_type,
            symbols=symbols,
            response_language=response_language,
            available_tools=available_tools,
            context=context
        )
        
        return await self._call_llm_and_parse_json(prompt, "Stage 3")
    
    def _build_detailed_task_prompt(
        self,
        query: str,
        query_type: str,
        symbols: List[str],
        response_language: str,
        available_tools: List[Dict[str, Any]],
        context: str
    ) -> str:
        """Build Stage 3 prompt showing COMPLETE tool schemas"""
        
        # Format tool schemas with ALL details
        tools_text = []
        
        for tool in available_tools:
            metadata = tool.get("metadata", {})
            params_props = tool.get("parameters", {}).get("properties", {})
            required_params = tool.get("parameters", {}).get("required", [])
            
            # Build parameter list
            param_list = []
            for param_name, param_info in params_props.items():
                param_desc = param_info.get("description", "N/A")
                is_required = "REQUIRED" if param_name in required_params else "optional"
                param_list.append(f"  • {param_name} ({is_required}): {param_desc}")
            
            tool_text = f"""
{'─' * 70}
TOOL: {tool['name']}
{'─' * 70}
Category: {metadata.get('category', 'unknown')}
Requires Symbol: {"✅ YES" if metadata.get('requires_symbol', True) else "❌ NO"}

Description:
  {tool['description']}

✅ Capabilities:
{chr(10).join([f"  • {cap}" for cap in metadata.get('capabilities', ['N/A'])])}

❌ Limitations:
{chr(10).join([f"  • {lim}" for lim in metadata.get('limitations', ['N/A'])])}

🎯 When to Use:
{chr(10).join([f"  → {hint}" for hint in metadata.get('usage_hints', ['N/A'])])}

Parameters:
{chr(10).join(param_list) if param_list else "  • No parameters"}
"""
            tools_text.append(tool_text)
        
        prompt = f"""You are a financial task planner. Create execution plan using EXACT tool names.

{'═' * 80}
CONTEXT (From previous conversation)
{'═' * 80}

{context if context else "No previous context"}

{'═' * 80}
CURRENT QUERY
{'═' * 80}

Query: {query}
Type: {query_type}
Symbols Detected: {symbols if symbols else "None"}
Language: {response_language}

{'═' * 80}
AVAILABLE TOOLS (USE EXACT NAMES)
{'═' * 80}

READ EACH TOOL CAREFULLY. Use EXACT tool names below.

{''.join(tools_text)}

{'═' * 80}
PLANNING INSTRUCTIONS
{'═' * 80}

Your Mission:
1. Read EACH tool's complete information above
2. Use ONLY tools listed above (EXACT names)
3. Match user need to tool capabilities
4. Create executable task plan

CRITICAL: Use EXACT tool names like "getTechnicalIndicators", "getSentiment"
NEVER invent tool names like "technicalAnalysis", "marketSentimentAnalysis"

EXECUTION STRATEGY:

**PARALLEL**: Independent tasks run together (Price + News + Analysis)
  Example: Get AAPL price + Get AAPL news (both independent)
  
**SEQUENTIAL**: Task B needs Task A output
  Example: 
    Task 1: stockScreener → finds symbols
    Task 2: getTechnicalIndicators symbols=<FROM_TASK_1> → analyzes symbols from Task 1
  Use dependencies: [1]

PARAMETER RULES:

1. Tool says "Requires Symbol: YES" → MUST include symbol in params
   Example: {{"tool_name": "getTechnicalIndicators", "params": {{"symbol": "AAPL"}}}}
   
2. Tool says "Requires Symbol: NO" → Do NOT add symbol
   Example: {{"tool_name": "stockScreener", "params": {{"sector": "Technology", "limit": 10}}}}
   
3. For screener queries:
   - Use stockScreener with limit=5-10 (NOT 50)
   - Sequential execution: Task 2 depends on Task 1
   - Use "<FROM_TASK_1>" for injection

4. For stock-specific queries:
   - Parallel execution (independent tools)
   - Include symbol in ALL tool params
   - NO screener needed (symbol already known)

{'═' * 80}
EXAMPLES (Use as reference for structure)
{'═' * 80}

Example 1: Stock-Specific (Parallel)
Query: "Analyze AAPL price and technicals"
Tools available: getStockPrice, getTechnicalIndicators
Output:
{{
  "strategy": "parallel",
  "tasks": [
    {{
      "id": 1,
      "description": "Get AAPL current price",
      "tools_needed": [{{"tool_name": "getStockPrice", "params": {{"symbol": "AAPL"}}}}],
      "dependencies": [],
      "priority": "high"
    }},
    {{
      "id": 2,
      "description": "Get AAPL technical indicators",
      "tools_needed": [{{"tool_name": "getTechnicalIndicators", "params": {{"symbol": "AAPL"}}}}],
      "dependencies": [],
      "priority": "high"
    }}
  ]
}}

Example 2: Screener (Sequential)
Query: "Find tech stocks with low P/E"
Tools available: stockScreener, getFinancialRatios
Output:
{{
  "strategy": "sequential",
  "tasks": [
    {{
      "id": 1,
      "description": "Screen tech stocks",
      "tools_needed": [{{"tool_name": "stockScreener", "params": {{"sector": "Technology", "limit": 10}}}}],
      "dependencies": [],
      "priority": "high"
    }},
    {{
      "id": 2,
      "description": "Get financial ratios for screened stocks",
      "tools_needed": [{{"tool_name": "getFinancialRatios", "params": {{"symbols": "<FROM_TASK_1>"}}}}],
      "dependencies": [1],
      "priority": "medium"
    }}
  ]
}}

Example 3: Stock Analysis (Implicit Price Requirement)
Query: "Analyze GOOGL market performance"
Symbols: ["GOOGL"]
Output:
{{
  "strategy": "parallel",
  "reasoning": "Need price context first, then specific market data.",
  "tasks": [
    {{
      "id": 1,
      "description": "Get GOOGL current price for context",
      "tools_needed": [{{"tool_name": "getStockPrice", "params": {{"symbol": "GOOGL"}}}}],
      "priority": "high",
      "dependencies": []
    }},
    {{
      "id": 2,
      "description": "Get sector performance",
      "tools_needed": [{{"tool_name": "getSectorPerformance", "params": {{"symbol": "GOOGL"}}}}],
      "priority": "high",
      "dependencies": []
    }},
    {{
      "id": 3,
      "description": "Get market indices",
      "tools_needed": [{{"tool_name": "getMarketIndices", "params": {{}}}}],
      "priority": "medium",
      "dependencies": []
    }}
  ]
}}

{'═' * 80}
OUTPUT (JSON ONLY)
{'═' * 80}

{{
  "query_intent": "What user wants to achieve",
  "strategy": "parallel" | "sequential",
  "estimated_complexity": "simple" | "moderate" | "complex",
  "symbols": {symbols if symbols else []},
  "response_language": "{response_language}",
  "reasoning": "Why these specific tools were selected",
  "tasks": [
    {{
      "id": 1,
      "description": "Clear task description",
      "tools_needed": [
        {{
          "tool_name": "EXACT_tool_name_from_list_above",
          "params": {{"key": "value"}}
        }}
      ],
      "expected_data": ["field1", "field2"],
      "priority": "high" | "medium" | "low",
      "dependencies": []
    }}
  ]
}}

CRITICAL REMINDERS:
- Use EXACT tool names from the list
- For screener: limit=5-10, sequential with dependencies
- For stock-specific: parallel, include symbol
- Use "<FROM_TASK_X>" for dependency injection

Now create your plan:
"""
        
        return prompt
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _build_context_string(
        self,
        recent_chat: List[Dict[str, str]],
        core_memory: Dict[str, Any],
        summary: str
    ) -> str:
        """Build context string for Stage 3"""
        
        context_parts = []
        
        # Core memory
        if core_memory:
            memory_lines = [f"  • {k}: {v}" for k, v in core_memory.items()]
            context_parts.append(f"User Preferences:\n" + "\n".join(memory_lines))
        
        # Summary
        if summary:
            context_parts.append(f"Conversation Summary:\n  {summary}")
        
        # Recent chat (last 6 messages = 3 turns)
        if recent_chat:
            recent = recent_chat[-6:]
            chat_lines = []
            for msg in recent:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                chat_lines.append(f"  {role}: {content}")
            
            context_parts.append(f"Recent Messages:\n" + "\n".join(chat_lines))
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    async def _call_llm_and_parse_json(
        self,
        prompt: str,
        stage_name: str
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON response"""
        
        try:
            # Call LLM
            response = await self.llm_provider.generate_response(
                model_name=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial planning assistant. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                provider_type=self.provider_type,
                api_key=self.api_key,
                temperature=0.1,
                enable_thinking=False
            )
            
            # Extract content
            if isinstance(response, dict):
                content = response.get("content", "")
            else:
                content = str(response)
            
            # Clean JSON
            content = content.strip()
            
            # Remove markdown blocks
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            content = content.strip()
            
            # Parse JSON
            result = json.loads(content)
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"[{stage_name}] JSON parse error: {e}")
            self.logger.error(f"[{stage_name}] Content: {content[:500]}")
            raise
        except Exception as e:
            self.logger.error(f"[{stage_name}] Error: {e}", exc_info=True)
            raise
    
    def _parse_task_plan_dict(
        self,
        plan_dict: Dict[str, Any],
        symbols: List[str],
        response_language: str
    ) -> TaskPlan:
        """Parse JSON dict into TaskPlan object"""
        
        tasks = []
        for task_data in plan_dict.get('tasks', []):
            # Parse tools_needed
            tools_needed = []
            for tool_data in task_data.get('tools_needed', []):
                # Handle symbol outside params
                if "symbol" in tool_data:
                    if "params" not in tool_data:
                        tool_data["params"] = {}
                    if "symbol" not in tool_data["params"]:
                        tool_data["params"]["symbol"] = tool_data.pop("symbol")
                    else:
                        tool_data.pop("symbol")
                
                tools_needed.append(ToolCall(**tool_data))
            
            # Create task
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
        
        # Create plan
        plan = TaskPlan(
            tasks=tasks,
            query_intent=plan_dict.get('query_intent', ''),
            strategy=plan_dict.get('strategy', 'sequential'),
            estimated_complexity=plan_dict.get('estimated_complexity', 'simple'),
            symbols=symbols,
            response_language=response_language,
            reasoning=plan_dict.get('reasoning', '')
        )
        
        return plan