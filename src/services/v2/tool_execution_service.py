import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.tools import get_registry, ToolOutput
from src.helpers.data_formatter import FinancialDataFormatter

MEMORY_TOOLS = [
    'searchConversationHistory',
    'getRecentConversations', 
    'searchRecallMemory',
    'searchArchivalMemory'
]

class ToolExecutionService(LoggerMixin):
    """
    Tool Execution Service
    """

    # Tools that DON'T require symbol
    NO_SYMBOL_TOOLS = {
        'getSectorPerformance',
        'getMarketMovers',
        'getMarketIndices',
        'getMarketBreadth',
        'getStockHeatmap',
        'getEarningsCalendar',
        'stockScreener',
        'getTopGainers',
        'getTopLosers',
        'getMostActives',
    }
    
    # Tools with optional symbol
    OPTIONAL_SYMBOL_TOOLS = {
        'getEarningsCalendar',
        'getStockNews',
    }
    
    def __init__(self):
        super().__init__()
        
        # ====================================================================
        # Initialize Tool Registry
        # ====================================================================
        self.tool_registry = None
        
        try:
            self.tool_registry = get_registry()
            summary = self.tool_registry.get_summary()
            
            # List all tool names by category
            # for category, tools in summary.get('tools_by_category', {}).items():
            #     self.logger.info(f"  {category}: {tools}")
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Failed to initialize tool registry: {e}")
            raise RuntimeError(f"Tool registry initialization failed: {e}")

    async def execute_single_tool(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        query: str,
        chat_history: str,
        system_language: str,
        provider_type: str,
        model_name: str,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None 
    ) -> Dict[str, Any]:
        """
        Execute atomic tool WITH CONTEXT
        
        Args:
            tool_name: Tool to execute
            tool_params: Base parameters from planning
            query: User's original query (for context)
            chat_history: Conversation history (for smart defaults)
            system_language: Response language (for formatting)
            provider_type: LLM provider (for intelligent tools)
            model_name: LLM model (for capabilities)
            user_id: User ID for memory tools (NEW)
            session_id: Session ID for memory tools (NEW)
            
        Returns:
            Dict with status, data, formatted_context, etc.
        """
        
        self.logger.info(f"[TOOL EXEC] Executing: {tool_name}")
        self.logger.debug(f"[TOOL EXEC] Query context: {query[:100]}...")
        self.logger.debug(f"[TOOL EXEC] Language: {system_language}")
        
        # Check registry
        if not self.tool_registry:
            return {
                'tool_name': tool_name,
                'status': 'error',
                'error': 'Tool registry not initialized'
            }
        
        tool_instance = self.tool_registry.get_tool(tool_name)
        if not tool_instance:
            available = self.tool_registry.list_tools()
            return {
                'tool_name': tool_name,
                'status': 'error',
                'error': f"Tool '{tool_name}' not found. Available: {available}"
            }
        
        self.logger.info(f"[ATOMIC] Found tool: {tool_name}")
        
        execution_context = {
            'query': query,
            'system_language': system_language,
            'chat_history': chat_history,
            'provider_type': provider_type,
            'session_id': session_id,
            'user_id': user_id
        }
        
        # Execute with context
        return await self._execute_atomic_tool_with_context(
            tool_name=tool_name,
            tool_params=tool_params,
            tool_instance=tool_instance,
            execution_context=execution_context
        )
    

    async def _execute_atomic_tool_with_context(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        tool_instance: Any,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute tool with context awareness
        """
        
        # Extract context
        query = execution_context.get('query', '')
        language = execution_context.get('system_language', 'en')
        
        # Determine if symbol required
        requires_symbol = self._check_requires_symbol(tool_name, tool_instance)
        
        symbol = None
        if requires_symbol:
            symbol = self._extract_symbol(tool_params)
            if not symbol:
                return {
                    'tool_name': tool_name,
                    'status': 'error',
                    'error': f"Tool '{tool_name}' requires symbol but none provided"
                }
            self.logger.info(f"[ATOMIC] Symbol: {symbol}")
        else:
            self.logger.info(f"[ATOMIC] Tool '{tool_name}' does NOT require symbol")
        
        # ====================================================================
        # Memory tools: Inject user_id and session_id from context
        # ====================================================================
        if tool_name in MEMORY_TOOLS:
            # Inject user_id from context if not provided or placeholder
            if 'user_id' not in tool_params or tool_params.get('user_id') in [None, 'user', '', 'current_user']:
                context_user_id = execution_context.get('user_id')
                if context_user_id:
                    tool_params['user_id'] = str(context_user_id)
                    self.logger.info(f"[ATOMIC] Injected user_id: {context_user_id}")
                else:
                    self.logger.warning(f"[ATOMIC] No user_id available for memory tool")
            
            # Inject session_id if needed and not provided
            if 'session_id' not in tool_params or not tool_params.get('session_id'):
                context_session_id = execution_context.get('session_id')
                if context_session_id:
                    tool_params['session_id'] = context_session_id
                    self.logger.info(f"[ATOMIC] Injected session_id: {context_session_id}")
        
        # Build parameters
        atomic_params = self._build_tool_params(
            tool_name=tool_name,
            tool_params=tool_params,
            symbol=symbol,
            requires_symbol=requires_symbol
        )
        
        self.logger.info(f"[ATOMIC] Executing: {tool_name}({atomic_params})")
        
        try:
            # Execute via registry
            result: ToolOutput = await self.tool_registry.execute_tool(
                tool_name=tool_name,
                params=atomic_params
            )
            
            # Format result with context
            return self._format_result_with_context(
                tool_name=tool_name,
                result=result,
                symbol=symbol,
                language=language
            )
            
        except Exception as e:
            self.logger.error(f"[ATOMIC] Exception: {e}", exc_info=True)
            return {
                'tool_name': tool_name,
                'status': 'error',
                'error': str(e),
                'symbols': [symbol] if symbol else []
            }
    
    def _format_result_with_context(
        self,
        tool_name: str,
        result: ToolOutput,
        symbol: Optional[str],
        language: str = 'en'
    ) -> Dict[str, Any]:
        """
        Format result with context awareness
        
        FIX 2: Handle memory tools formatting properly
        """
        
        if result.status in ["success", "partial"]:
            base_response = {
                'tool_name': tool_name,
                'status': '200',
                'symbols': [symbol] if symbol else [],
                'data': result.data,
                'execution_time_ms': result.execution_time_ms,
                'metadata': {
                    'source': 'atomic_tools',
                    'tool_name': tool_name,
                    'language': language,
                    **result.metadata
                },
                'raw_data': result.data
            }
            
            if result.status == "partial":
                base_response['metadata']['partial'] = True
                missing = result.metadata.get('missing_fields', [])
                base_response['warning'] = f"Partial data: missing {missing}"
            
            # ================================================================
            # FIX 2: Handle memory tools formatting separately
            # ================================================================
            try:
                if tool_name in MEMORY_TOOLS:
                    # Memory tools have different output format
                    formatted_context = self._format_memory_tool_result(
                        tool_name=tool_name,
                        data=result.data
                    )
                else:
                    # Financial tools use FinancialDataFormatter
                    formatted_context = FinancialDataFormatter.format_by_tool_name(
                        tool_name=tool_name,
                        data=result.data
                    )
                
                base_response['formatted_context'] = formatted_context
                
                self.logger.info(
                    f"[FORMAT] {tool_name} → {len(formatted_context)} chars"
                )
                
            except Exception as e:
                self.logger.warning(f"[FORMAT] Formatting failed: {e}")
                # Fallback: use JSON dump
                base_response['formatted_context'] = json.dumps(
                    result.data, indent=2, ensure_ascii=False, default=str
                )
            
            # Log success
            if result.status == "success":
                self.logger.info(
                    f"[ATOMIC] {tool_name} succeeded "
                    f"(time: {result.execution_time_ms}ms, "
                    f"data_keys: {list(result.data.keys()) if result.data else []})"
                )
            
            return base_response
        
        else:
            # Error
            self.logger.error(f"[ATOMIC] {tool_name} failed: {result.error}")
            return {
                'tool_name': tool_name,
                'status': 'error',
                'error': result.error or 'Unknown error',
                'symbols': [symbol] if symbol else [],
                'metadata': result.metadata
            }
    
    # ========================================================================
    # Memory Tool Formatting
    # ========================================================================
    
    def _format_memory_tool_result(
        self,
        tool_name: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Format memory tool results for LLM context
        
        Memory tools return different structure than financial tools:
        - searchConversationHistory: {query, results, total_found, has_more}
        - getRecentConversations: {conversations, count}
        - searchRecallMemory: {query, results, session_id}
        - searchArchivalMemory: {query, results, user_id}
        """
        if not data:
            return "No memory data available."
        
        lines = []
        
        if tool_name == 'searchConversationHistory':
            query = data.get('query', '')
            results = data.get('results', [])
            total = data.get('total_found', 0)
            
            if not results:
                return f"No past conversations found about '{query}'."
            
            lines.append(f"📝 PAST CONVERSATIONS about '{query}' ({total} found):")
            
            for i, result in enumerate(results[:5], 1):
                title = result.get('conversation_title', 'Untitled')
                snippet = result.get('snippet', '')[:200]
                timestamp = result.get('timestamp', '')
                url = result.get('url', '')
                
                lines.append(f"\n{i}. [{title}]")
                if timestamp:
                    lines.append(f"   📅 {timestamp}")
                lines.append(f"   {snippet}...")
                if url:
                    lines.append(f"   → {url}")
        
        elif tool_name == 'getRecentConversations':
            conversations = data.get('conversations', [])
            count = data.get('count', 0)
            
            if not conversations:
                return "No recent conversations found."
            
            lines.append(f"📋 RECENT CONVERSATIONS ({count} found):")
            
            for i, conv in enumerate(conversations[:10], 1):
                title = conv.get('title', 'Untitled')
                updated = conv.get('updated_at', '')
                url = conv.get('url', '')
                
                lines.append(f"\n{i}. {title}")
                if updated:
                    lines.append(f"   📅 Updated: {updated}")
                if url:
                    lines.append(f"   → {url}")
        
        elif tool_name == 'searchRecallMemory':
            query = data.get('query', '')
            results = data.get('results', [])
            
            if not results:
                return f"No recall memory found for '{query}'."
            
            lines.append(f"🧠 RECALL MEMORY for '{query}':")
            
            for i, result in enumerate(results[:5], 1):
                content = result.get('content', '')[:200]
                score = result.get('score', 0)
                
                lines.append(f"\n{i}. (score: {score:.2f})")
                lines.append(f"   {content}...")
        
        elif tool_name == 'searchArchivalMemory':
            query = data.get('query', '')
            results = data.get('results', [])
            
            if not results:
                return f"No archival memory found for '{query}'."
            
            lines.append(f"📚 ARCHIVAL MEMORY for '{query}':")
            
            for i, result in enumerate(results[:5], 1):
                content = result.get('content', '')[:200]
                timestamp = result.get('timestamp', '')
                
                lines.append(f"\n{i}. {content}...")
                if timestamp:
                    lines.append(f"   📅 {timestamp}")
        
        else:
            # Fallback for unknown memory tools
            return json.dumps(data, indent=2, ensure_ascii=False, default=str)
        
        return '\n'.join(lines)
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _check_requires_symbol(self, tool_name: str, tool_instance: Any) -> bool:
        """Check if tool requires symbol"""
        # Check from schema first
        if tool_instance and hasattr(tool_instance, 'schema') and tool_instance.schema:
            return tool_instance.schema.requires_symbol
        
        # Fallback to predefined list
        if tool_name in self.NO_SYMBOL_TOOLS:
            return False
        
        # Memory tools don't require symbol
        if tool_name in MEMORY_TOOLS:
            return False
        
        # Default: assume symbol is required
        return True
    
    def _extract_symbol(self, tool_params: Dict[str, Any]) -> Optional[str]:
        """Extract symbol from various possible locations"""
        symbol = None
        
        # Try 1: Direct symbol parameter
        if 'symbol' in tool_params and tool_params['symbol']:
            symbol = tool_params['symbol']
        
        # Try 2: symbols array (take first)
        elif 'symbols' in tool_params and tool_params['symbols']:
            symbols = tool_params['symbols']
            if isinstance(symbols, list) and len(symbols) > 0:
                symbol = symbols[0]
            elif isinstance(symbols, str):
                symbol = symbols
        
        # Try 3: Nested params
        elif 'params' in tool_params and isinstance(tool_params['params'], dict):
            nested = tool_params['params']
            if 'symbol' in nested and nested['symbol']:
                symbol = nested['symbol']
            elif 'symbols' in nested and nested['symbols']:
                syms = nested['symbols']
                symbol = syms[0] if isinstance(syms, list) and len(syms) > 0 else syms
        
        # Normalize symbol
        if symbol and isinstance(symbol, str):
            symbol = symbol.upper().strip()
        
        return symbol
    
    def _build_tool_params(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        symbol: Optional[str],
        requires_symbol: bool
    ) -> Dict[str, Any]:
        """Build parameters for specific tool"""
        atomic_params = {}
        handled = False

        # Add symbol if required
        if requires_symbol and symbol:
            atomic_params['symbol'] = symbol
            handled = True

        # ================================================================
        # Memory Tools - Pass through all relevant params
        # ================================================================
        if tool_name == 'searchConversationHistory':
            handled = True
            for key in ['query', 'user_id', 'max_results', 'date_range']:
                if key in tool_params and tool_params[key] is not None:
                    atomic_params[key] = tool_params[key]
        
        elif tool_name == 'getRecentConversations':
            handled = True
            for key in ['user_id', 'n', 'sort_order', 'before', 'after']:
                if key in tool_params and tool_params[key] is not None:
                    atomic_params[key] = tool_params[key]
        
        elif tool_name == 'searchRecallMemory':
            handled = True
            for key in ['query', 'session_id', 'user_id', 'timeframe', 'limit']:
                if key in tool_params and tool_params[key] is not None:
                    atomic_params[key] = tool_params[key]
        
        elif tool_name == 'searchArchivalMemory':
            handled = True
            for key in ['query', 'user_id', 'limit', 'include_global']:
                if key in tool_params and tool_params[key] is not None:
                    atomic_params[key] = tool_params[key]

        # ================================================================
        # Technical Analysis Tools
        # ================================================================
        elif tool_name == 'getTechnicalIndicators':
            handled = True
            indicators = (
                tool_params.get('indicators') or 
                tool_params.get('params', {}).get('indicators')
            )
            if indicators:
                atomic_params['indicators'] = indicators
            
            timeframe = (
                tool_params.get('timeframe') or 
                tool_params.get('params', {}).get('timeframe')
            )
            if timeframe:
                atomic_params['timeframe'] = timeframe
        
        elif tool_name == 'detectChartPatterns':
            handled = True
            lookback = (
                tool_params.get('lookback_days') or 
                tool_params.get('params', {}).get('lookback_days')
            )
            if lookback:
                atomic_params['lookback_days'] = lookback
        
        elif tool_name == 'getRelativeStrength':
            handled = True
            benchmark = (
                tool_params.get('benchmark') or 
                tool_params.get('params', {}).get('benchmark', 'SPY')
            )
            atomic_params['benchmark'] = benchmark
        
        # ================================================================
        # Risk Tools
        # ================================================================
        elif tool_name == 'assessRisk':
            handled = True
            lookback = (
                tool_params.get('lookback_days') or 
                tool_params.get('params', {}).get('lookback_days')
            )
            if lookback:
                atomic_params['lookback_days'] = lookback
        
        elif tool_name == 'suggestStopLoss':
            handled = True
            risk_percent = (
                tool_params.get('risk_percent') or 
                tool_params.get('params', {}).get('risk_percent')
            )
            if risk_percent:
                atomic_params['risk_percent'] = risk_percent
        
        # ================================================================
        # Fundamentals Tools
        # ================================================================
        elif tool_name in ['getIncomeStatement', 'getBalanceSheet', 'getCashFlow', 
                          'getFinancialRatios', 'getGrowthMetrics']:
            handled = True
            period = (
                tool_params.get('period') or 
                tool_params.get('params', {}).get('period', 'annual')
            )
            atomic_params['period'] = period
            
            limit = (
                tool_params.get('limit') or 
                tool_params.get('params', {}).get('limit')
            )
            if limit:
                atomic_params['limit'] = limit
        
        # ================================================================
        # News Tools
        # ================================================================
        elif tool_name == 'getStockNews':
            handled = True
            limit = (
                tool_params.get('limit') or 
                tool_params.get('params', {}).get('limit', 10)
            )
            atomic_params['limit'] = limit
        
        elif tool_name == 'getEarningsCalendar':
            handled = True
            from_date = (
                tool_params.get('from_date') or 
                tool_params.get('params', {}).get('from_date')
            )
            if from_date:
                atomic_params['from_date'] = from_date
            
            to_date = (
                tool_params.get('to_date') or 
                tool_params.get('params', {}).get('to_date')
            )
            if to_date:
                atomic_params['to_date'] = to_date
        
        # ================================================================
        # Market Tools
        # ================================================================
        elif tool_name == 'getMarketMovers':
            handled = True
            mover_type = (
                tool_params.get('mover_type') or 
                tool_params.get('params', {}).get('mover_type', 'gainers')
            )
            atomic_params['mover_type'] = mover_type
        
        elif tool_name in ('getTopGainers', 'getTopLosers', 'getMostActives'):
            handled = True
            self.logger.debug(f"[BUILD_PARAMS] {tool_name}: no params needed")

        elif tool_name == 'getSectorPerformance':
            handled = True
            date = (
                tool_params.get('date') or 
                tool_params.get('params', {}).get('date')
            )
            if date:
                atomic_params['date'] = date
        
        elif tool_name == 'getStockHeatmap':
            handled = True
            group_by = (
                tool_params.get('group_by') or 
                tool_params.get('params', {}).get('group_by', 'sector')
            )
            atomic_params['group_by'] = group_by
        
        # ================================================================
        # Crypto Tools
        # ================================================================
        elif tool_name == 'getCryptoPrice':
            handled = True
            # Symbol already added above
        
        elif tool_name == 'getCryptoTechnicals':
            handled = True
            indicators = (
                tool_params.get('indicators') or 
                tool_params.get('params', {}).get('indicators')
            )
            if indicators:
                atomic_params['indicators'] = indicators
        
        # ================================================================
        # Discovery Tools
        # ================================================================
        elif tool_name == 'stockScreener':
            handled = True
            screener_params = [
                'sector', 'industry', 'country', 'exchange',
                'market_cap_more_than', 'market_cap_lower_than',
                'price_more_than', 'price_lower_than',
                'volume_more_than', 'beta_more_than', 'beta_lower_than',
                'dividend_more_than', 'is_etf', 'is_actively_trading',
                'limit'
            ]
            for param in screener_params:
                value = (
                    tool_params.get(param) or
                    tool_params.get('params', {}).get(param)
                )
                if value is not None:
                    atomic_params[param] = value
        
        elif tool_name == 'think':
            handled = True
            thought = tool_params.get('thought') or tool_params.get('params', {}).get('thought')
            if thought:
                atomic_params['thought'] = thought

                
        # ================================================================
        # Fallback: Pass-through for unknown tools
        # ================================================================
        if not handled:
            self.logger.warning(
                f"[BUILD_PARAMS] Unknown tool '{tool_name}', passing through all params"
            )
            for key, value in tool_params.items():
                if key != 'params' and value is not None:
                    atomic_params[key] = value
            
            if 'params' in tool_params and isinstance(tool_params['params'], dict):
                for key, value in tool_params['params'].items():
                    if value is not None and key not in atomic_params:
                        atomic_params[key] = value
                        
        return atomic_params
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def list_available_tools(self) -> Dict[str, List[str]]:
        """Get list of all available tools by category"""
        if not self.tool_registry:
            return {}
        
        summary = self.tool_registry.get_summary()
        return summary.get('tools_by_category', {})
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get schema for a specific tool"""
        if not self.tool_registry:
            return None
        
        schema = self.tool_registry.get_schema(tool_name)
        if schema:
            return schema.to_json_schema()
        return None
    
    def is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available"""
        if not self.tool_registry:
            return False
        return self.tool_registry.get_tool(tool_name) is not None