# File: src/services/v2/tool_execution_service.py
"""
Tool Execution Service - Atomic Tools with Formatting Integration

IMPROVEMENTS:
✅ Adds formatted_context to all ToolOutput responses
✅ Uses FinancialDataFormatter for domain knowledge
✅ Cleans up unused code
✅ Compatible with existing TaskExecutor flow
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from src.utils.logger.custom_logging import LoggerMixin

# ============================================================================
# Import Atomic Tools Registry & Formatter
# ============================================================================
from src.agents.tools import get_registry, ToolOutput
from src.helpers.data_formatter import FinancialDataFormatter


class ToolExecutionService(LoggerMixin):
    """
    Tool Execution Service - Atomic Tools Only
    
    NEW Features:
    - ✅ Automatic formatting via FinancialDataFormatter
    - ✅ Both raw_data and formatted_context in response
    - ✅ Compatible with existing TaskExecutor
    
    Original Features:
    - Direct registry lookup (no mapping needed)
    - Proper handling for tools that don't require symbol
    - Handle partial status as success
    - Clear error messages
    """
    
    # ========================================================================
    # Tools Configuration
    # ========================================================================
    
    # Tools that DON'T require symbol
    NO_SYMBOL_TOOLS = {
        'getSectorPerformance',
        'getMarketMovers',
        'getMarketIndices',
        'getMarketBreadth',
        'getStockHeatmap',
        'getEarningsCalendar',
        'stockScreener',
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
            self.logger.info("=" * 60)
            self.logger.info("✅ Tool Execution Service - Atomic Tools + Formatting")
            self.logger.info(f"Total tools: {summary['total_tools']}")
            self.logger.info(f"Categories: {list(summary['categories'].keys())}")
            
            # List all tool names by category
            for category, tools in summary.get('tools_by_category', {}).items():
                self.logger.info(f"  {category}: {tools}")
            
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.error(f"❌ CRITICAL: Failed to initialize tool registry: {e}")
            raise RuntimeError(f"Tool registry initialization failed: {e}")
    
    # ========================================================================
    # Main Execute Method (Called by TaskExecutor)
    # ========================================================================
    # async def execute_single_tool(
    #     self,
    #     tool_name: str,
    #     tool_params: Dict[str, Any],
    #     query: str,
    #     chat_history: str,
    #     system_language: str,
    #     provider_type: str,
    #     model_name: str
    # ) -> Dict[str, Any]:
    #     """
    #     Execute a single atomic tool WITH FORMATTING
        
    #     ✅ NEW: This method now automatically adds formatted_context
        
    #     Args:
    #         tool_name: Tool to execute (e.g., "getStockPrice", "getSectorPerformance")
    #         tool_params: Parameters for tool
    #         query: User's original query (for context)
    #         chat_history: Conversation history (for context)
    #         system_language: Response language
    #         provider_type: LLM provider
    #         model_name: LLM model
            
    #     Returns:
    #         Dict with standardized results:
    #         {
    #             'tool_name': str,
    #             'status': '200' | 'error',
    #             'data': dict (if success),
    #             'formatted_context': str (NEW - formatted for LLM),
    #             'error': str (if error),
    #             'symbols': list,
    #             'execution_time_ms': int,
    #             'metadata': dict
    #         }
    #     """
        
    #     self.logger.info(f"[TOOL EXEC] Executing tool: {tool_name}")
    #     self.logger.debug(f"[TOOL EXEC] Full params: {tool_params}")
        
    #     # ====================================================================
    #     # Check registry availability
    #     # ====================================================================
    #     if not self.tool_registry:
    #         error_msg = "Tool registry not initialized"
    #         self.logger.error(f"[TOOL EXEC] ❌ {error_msg}")
    #         return {
    #             'tool_name': tool_name,
    #             'status': 'error',
    #             'error': error_msg
    #         }
        
    #     # ====================================================================
    #     # Check if tool exists
    #     # ====================================================================
    #     tool_instance = self.tool_registry.get_tool(tool_name)
        
    #     if not tool_instance:
    #         available_tools = self.tool_registry.list_tools()
    #         error_msg = (
    #             f"Tool '{tool_name}' not found in registry. "
    #             f"Available atomic tools: {available_tools}"
    #         )
    #         self.logger.error(f"[TOOL EXEC] ❌ {error_msg}")
            
    #         return {
    #             'tool_name': tool_name,
    #             'status': 'error',
    #             'error': error_msg,
    #             'available_tools': available_tools
    #         }
        
    #     # ====================================================================
    #     # Execute atomic tool
    #     # ====================================================================
    #     self.logger.info(f"[ATOMIC] ✅ Found atomic tool: {tool_name}")
        
    #     return await self._execute_atomic_tool(
    #         tool_name=tool_name,
    #         tool_params=tool_params,
    #         tool_instance=tool_instance
    #     )

    async def execute_single_tool(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        query: str,              # ✅ KEEP: For intelligent parameter inference
        chat_history: str,       # ✅ KEEP: For conversation-aware defaults
        system_language: str,    # ✅ KEEP: For locale-specific formatting
        provider_type: str,      # ✅ KEEP: For potential LLM-assisted tools
        model_name: str          # ✅ KEEP: For model-specific capabilities
    ) -> Dict[str, Any]:
        """
        Execute atomic tool WITH CONTEXT
        
        WHY KEEP ALL PARAMETERS?
        - query: Enables smart parameter interpretation
        - chat_history: Provides conversation context for defaults
        - system_language: Controls output formatting/localization
        - provider_type/model_name: Future LLM-assisted tool execution
        
        Args:
            tool_name: Tool to execute
            tool_params: Base parameters from planning
            query: User's original query (for context)
            chat_history: Conversation history (for smart defaults)
            system_language: Response language (for formatting)
            provider_type: LLM provider (for intelligent tools)
            model_name: LLM model (for capabilities)
            
        Returns:
            Dict with:
            {
                'status': '200' | 'error',
                'data': dict,
                'formatted_context': str (formatted for LLM),
                'symbols': list,
                'execution_context': {
                    'query': str,
                    'language': str,
                    'applied_defaults': dict
                }
            }
        """
        
        self.logger.info(f"[TOOL EXEC] Executing: {tool_name}")
        self.logger.debug(f"[TOOL EXEC] Query context: {query[:100]}...")
        self.logger.debug(f"[TOOL EXEC] Language: {system_language}")
        
        # Check registry
        if not self.tool_registry:
            return {'tool_name': tool_name, 'status': 'error', 'error': 'Registry not initialized'}
        
        tool_instance = self.tool_registry.get_tool(tool_name)
        if not tool_instance:
            available = self.tool_registry.list_tools()
            return {
                'tool_name': tool_name,
                'status': 'error',
                'error': f"Tool '{tool_name}' not found. Available: {available}"
            }
        
        self.logger.info(f"[ATOMIC] ✅ Found tool: {tool_name}")
        
        # Execute with context
        return await self._execute_atomic_tool_with_context(
            tool_name=tool_name,
            tool_params=tool_params,
            tool_instance=tool_instance,
            # Pass context for potential use
            execution_context={
                'query': query,
                'chat_history': chat_history,
                'system_language': system_language,
                'provider_type': provider_type,
                'model_name': model_name
            }
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
        
        FUTURE ENHANCEMENTS (Anthropic Pattern):
        1. Apply conversation-aware defaults
        2. Interpret ambiguous parameters using query
        3. Format output based on language
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
        
        # Build parameters
        atomic_params = self._build_tool_params(
            tool_name=tool_name,
            tool_params=tool_params,
            symbol=symbol,
            requires_symbol=requires_symbol
        )
        
        # TODO: Future - Apply smart defaults from context
        # applied_defaults = self._apply_conversation_defaults(
        #     tool_name=tool_name,
        #     params=atomic_params,
        #     chat_history=execution_context.get('chat_history', '')
        # )
        
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
                language=language  # Use for locale-specific formatting
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
        
        Args:
            language: For locale-specific number/date formatting
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
                    'language': language,  # Track language for formatting
                    **result.metadata
                },
                'raw_data': result.data
            }
            
            if result.status == "partial":
                base_response['metadata']['partial'] = True
                missing = result.metadata.get('missing_fields', [])
                base_response['warning'] = f"Partial data: missing {missing}"
            
            # ✅ Add formatted context using FinancialDataFormatter
            try:
                formatted_context = FinancialDataFormatter.format_by_tool_name(
                    tool_name=tool_name,
                    data=result.data
                )
                
                # TODO: Future - Apply language-specific formatting
                # if language == 'vi':
                #     formatted_context = self._apply_vietnamese_formatting(formatted_context)
                
                base_response['formatted_context'] = formatted_context
                
                self.logger.info(
                    f"[FORMAT] ✅ {tool_name} → {len(formatted_context)} chars"
                )
                
            except Exception as e:
                self.logger.warning(f"[FORMAT] ⚠️ Formatting failed: {e}")
                base_response['formatted_context'] = json.dumps(
                    result.data, indent=2, ensure_ascii=False, default=str
                )
            
            return base_response
        
        else:
            # Error
            self.logger.error(f"[ATOMIC] ❌ {tool_name} failed: {result.error}")
            return {
                'tool_name': tool_name,
                'status': 'error',
                'error': result.error or 'Unknown error',
                'symbols': [symbol] if symbol else [],
                'metadata': result.metadata
            }
    # ========================================================================
    # Atomic Tool Execution with Formatting
    # ========================================================================
    async def _execute_atomic_tool(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        tool_instance: Any
    ) -> Dict[str, Any]:
        """
        Execute atomic tool via registry + Add formatting
        
        FLOW:
        1. Execute tool → get raw ToolOutput
        2. Format using FinancialDataFormatter
        3. Return both raw_data and formatted_context
        
        Args:
            tool_name: Atomic tool name
            tool_params: Tool parameters from planning
            tool_instance: Tool instance from registry
            
        Returns:
            Dict with standardized results INCLUDING formatted_context
        """
        
        # ====================================================================
        # Determine if symbol is required
        # ====================================================================
        requires_symbol = self._check_requires_symbol(tool_name, tool_instance)
        
        symbol = None
        
        # ====================================================================
        # Extract symbol only if required
        # ====================================================================
        if requires_symbol:
            symbol = self._extract_symbol(tool_params)
            
            if not symbol:
                error_msg = (
                    f"Tool '{tool_name}' requires symbol but none provided. "
                    f"Params received: {tool_params}"
                )
                self.logger.error(f"[ATOMIC] ❌ {error_msg}")
                return {
                    'tool_name': tool_name,
                    'status': 'error',
                    'error': error_msg
                }
            
            self.logger.info(f"[ATOMIC] Extracted symbol: {symbol}")
        else:
            self.logger.info(f"[ATOMIC] Tool '{tool_name}' does NOT require symbol")
        
        # ====================================================================
        # Build parameters for tool
        # ====================================================================
        atomic_params = self._build_tool_params(
            tool_name=tool_name,
            tool_params=tool_params,
            symbol=symbol,
            requires_symbol=requires_symbol
        )
        
        self.logger.info(f"[ATOMIC] Calling registry.execute_tool({tool_name}, {atomic_params})")
        
        # ====================================================================
        # Execute via registry
        # ====================================================================
        try:
            result: ToolOutput = await self.tool_registry.execute_tool(
                tool_name=tool_name,
                params=atomic_params
            )
            
            # ✅ NEW: Format result with FinancialDataFormatter
            return self._format_result_with_context(
                tool_name=tool_name,
                result=result,
                symbol=symbol
            )
            
        except Exception as e:
            self.logger.error(
                f"[ATOMIC] Exception executing {tool_name}: {e}", 
                exc_info=True
            )
            return {
                'tool_name': tool_name,
                'status': 'error',
                'error': str(e),
                'symbols': [symbol] if symbol else []
            }
    
    # ========================================================================
    # NEW: Format Result with FinancialDataFormatter
    # ========================================================================
    def _format_result_with_context(
        self,
        tool_name: str,
        result: ToolOutput,
        symbol: Optional[str],
        language: str = 'en' 
    ) -> Dict[str, Any]:
        """
        Format ToolOutput to standardized dict format + Add formatted_context
        
        ✅ NEW: Adds formatted_context using FinancialDataFormatter
        
        Handles:
        - success: status='200'
        - partial: status='200' (still has data)
        - error: status='error'
        """
        
        if result.status in ["success", "partial"]:
            # ================================================================
            # Step 1: Create base response (existing logic)
            # ================================================================
            base_response = {
                'tool_name': tool_name,
                'status': '200',
                'symbols': [symbol] if symbol else [],
                'data': result.data,
                'execution_time_ms': result.execution_time_ms,
                'metadata': {
                    'source': 'atomic_tools',
                    'tool_name': tool_name,
                    **result.metadata
                },
                'raw_data': result.data  # For backward compatibility
            }
            
            if result.status == "partial":
                base_response['metadata']['partial'] = True
                missing_fields = result.metadata.get('missing_fields', [])
                base_response['warning'] = f"Partial data: missing {missing_fields}"
            
            # ================================================================
            # Step 2: ✅ NEW - Add formatted_context using FinancialDataFormatter
            # ================================================================
            try:
                formatted_context = FinancialDataFormatter.format_by_tool_name(
                    tool_name=tool_name,
                    data=result.data
                )
                
                # TODO: Future - use language parameter for localization
                # if language == 'vi':
                #     formatted_context = self._apply_vietnamese_formatting(formatted_context)
                
                base_response['formatted_context'] = formatted_context
                
                self.logger.info(
                    f"[FORMAT] ✅ {tool_name} → {len(formatted_context)} chars formatted"
                )
                
            except Exception as e:
                self.logger.warning(
                    f"[FORMAT] ⚠️ Formatting failed for {tool_name}: {e}"
                )
                # Fallback: use JSON dump
                base_response['formatted_context'] = json.dumps(
                    result.data, 
                    indent=2, 
                    ensure_ascii=False,
                    default=str
                )
            
            # Log success
            if result.status == "success":
                self.logger.info(
                    f"[ATOMIC] ✅ {tool_name} succeeded "
                    f"(time: {result.execution_time_ms}ms, "
                    f"data_keys: {list(result.data.keys()) if result.data else []})"
                )
            else:
                self.logger.warning(
                    f"[ATOMIC] ⚠️ {tool_name} partial success "
                    f"(time: {result.execution_time_ms}ms)"
                )
            
            return base_response
        
        else:
            # Error status
            self.logger.error(f"[ATOMIC] ❌ {tool_name} failed: {result.error}")
            return {
                'tool_name': tool_name,
                'status': 'error',
                'error': result.error or 'Unknown error',
                'symbols': [symbol] if symbol else [],
                'metadata': result.metadata
            }
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _check_requires_symbol(self, tool_name: str, tool_instance: Any) -> bool:
        """
        Check if tool requires symbol
        
        Priority:
        1. Check from schema (most accurate)
        2. Check from NO_SYMBOL_TOOLS list (fallback)
        """
        # Check from schema first
        if tool_instance and hasattr(tool_instance, 'schema') and tool_instance.schema:
            return tool_instance.schema.requires_symbol
        
        # Fallback to predefined list
        if tool_name in self.NO_SYMBOL_TOOLS:
            return False
        
        # Default: assume symbol is required
        return True
    
    def _extract_symbol(self, tool_params: Dict[str, Any]) -> Optional[str]:
        """
        Extract symbol from various possible locations
        
        Tries:
        1. Direct 'symbol' parameter
        2. 'symbols' array (take first)
        3. Nested 'params.symbol'
        """
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
        """
        Build parameters for specific tool
        
        Each tool may have different parameter requirements.
        """
        atomic_params = {}
        
        # Add symbol if required
        if requires_symbol and symbol:
            atomic_params['symbol'] = symbol
        
        # ================================================================
        # Tool-specific parameter extraction
        # ================================================================
        
        # --- Technical Analysis Tools ---
        if tool_name == 'getTechnicalIndicators':
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
            lookback = (
                tool_params.get('lookback_days') or 
                tool_params.get('params', {}).get('lookback_days')
            )
            if lookback:
                atomic_params['lookback_days'] = lookback
        
        elif tool_name == 'getRelativeStrength':
            benchmark = (
                tool_params.get('benchmark') or 
                tool_params.get('params', {}).get('benchmark', 'SPY')
            )
            atomic_params['benchmark'] = benchmark
        
        # --- Risk Tools ---
        elif tool_name == 'assessRisk':
            lookback = (
                tool_params.get('lookback_days') or 
                tool_params.get('params', {}).get('lookback_days')
            )
            if lookback:
                atomic_params['lookback_days'] = lookback
        
        elif tool_name == 'suggestStopLoss':
            risk_percent = (
                tool_params.get('risk_percent') or 
                tool_params.get('params', {}).get('risk_percent')
            )
            if risk_percent:
                atomic_params['risk_percent'] = risk_percent
        
        # --- Fundamentals Tools ---
        elif tool_name in ['getIncomeStatement', 'getBalanceSheet', 'getCashFlow', 
                          'getFinancialRatios', 'getGrowthMetrics']:
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
        
        # --- News Tools ---
        elif tool_name == 'getStockNews':
            limit = (
                tool_params.get('limit') or 
                tool_params.get('params', {}).get('limit', 10)
            )
            atomic_params['limit'] = limit
        
        elif tool_name == 'getEarningsCalendar':
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
        
        # --- Market Tools ---
        elif tool_name == 'getMarketMovers':
            mover_type = (
                tool_params.get('mover_type') or 
                tool_params.get('params', {}).get('mover_type', 'gainers')
            )
            atomic_params['mover_type'] = mover_type
        
        elif tool_name == 'getSectorPerformance':
            date = (
                tool_params.get('date') or 
                tool_params.get('params', {}).get('date')
            )
            if date:
                atomic_params['date'] = date
        
        elif tool_name == 'getStockHeatmap':
            group_by = (
                tool_params.get('group_by') or 
                tool_params.get('params', {}).get('group_by', 'sector')
            )
            atomic_params['group_by'] = group_by
        
        # --- Crypto Tools ---
        elif tool_name == 'getCryptoPrice':
            # Symbol should be like "BTC" or "BTCUSD"
            pass  # Symbol already added above
        
        elif tool_name == 'getCryptoTechnicals':
            indicators = (
                tool_params.get('indicators') or 
                tool_params.get('params', {}).get('indicators')
            )
            if indicators:
                atomic_params['indicators'] = indicators
        
        # --- Discovery Tools ---
        elif tool_name == 'stockScreener':
            # Map all screener parameters
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