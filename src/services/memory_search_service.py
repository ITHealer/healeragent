# # Version 2:
# import re
# import json
# from datetime import datetime, timedelta
# from typing import List, Dict, Optional, Any

# from src.utils.logger.custom_logging import LoggerMixin
# from src.agents.memory.memory_manager import MemoryManager
# from src.helpers.chat_management_helper import ChatService
# from src.database.repository.sessions import SessionRepository
# from src.providers.provider_factory import ProviderType

# from src.helpers.llm_chat_helper import (
#     analyze_stock_stream,
#     analyze_market_overview_stream,
#     analyze_stock_trending_stream,
#     analyze_stock_heatmap_stream,
#     general_chat_bot_stream
# )


# class MemorySearchService(LoggerMixin):


#     def __init__(self):
#         super().__init__()
#         self.memory_manager = MemoryManager()
#         self.session_repo = SessionRepository()
#         self.chat_service = ChatService()
    
    
#     async def search_recall_memory(
#         self,
#         session_id: str,
#         strategy: str,
#         params: Dict[str, Any],
#         user_id: Optional[str] = None
#     ) -> List[Dict[str, Any]]:
#         """Search conversation history with different strategies"""
#         try:
#             results = []
            
#             if strategy == "temporal":
#                 results = await self._temporal_search(
#                     session_id=session_id,
#                     params=params,
#                     user_id=user_id
#                 )
            
#             elif strategy == "symbol":
#                 results = await self._symbol_search(
#                     session_id=session_id,
#                     symbols=params.get('symbols', []),
#                     limit=params.get('limit', 10)
#                 )
            
#             elif strategy == "topic":
#                 results = await self._topic_search(
#                     session_id=session_id,
#                     topic=params.get('topic', ''),
#                     limit=params.get('limit', 10)
#                 )
            
#             elif strategy == "hybrid":
#                 temporal_results = await self._temporal_search(
#                     session_id=session_id,
#                     params=params,
#                     user_id=user_id
#                 )
                
#                 symbol_results = await self._symbol_search(
#                     session_id=session_id,
#                     symbols=params.get('symbols', []),
#                     limit=5
#                 )
                
#                 # Merge and deduplicate
#                 seen_ids = set()
#                 for msg in temporal_results + symbol_results:
#                     msg_id = msg.get('id')
#                     if msg_id and msg_id not in seen_ids:
#                         results.append(msg)
#                         seen_ids.add(msg_id)
            
#             self.logger.info(
#                 f"[RECALL SEARCH] Strategy: {strategy}, Found: {len(results)} messages"
#             )
            
#             return results
            
#         except Exception as e:
#             self.logger.error(f"Error in recall search: {e}")
#             return []
    
    
#     async def search_archival_memory(
#         self,
#         query: str,
#         user_id: Optional[str] = None,
#         limit: int = 5
#     ) -> List[Dict[str, Any]]:
#         """Search knowledge base (vector DB)"""
#         try:
#             collection_name = f"archival_{user_id}" if user_id else "global_knowledge"
            
#             memories = await self.memory_manager.search_relevant_memory(
#                 query=query,
#                 collection_name=collection_name,
#                 top_k=limit
#             )
            
#             results = []
#             for mem in memories:
#                 results.append({
#                     'content': mem.get('content', ''),
#                     'metadata': mem.get('metadata', {}),
#                     'source': mem.get('metadata', {}).get('source', 'knowledge_base'),
#                     'score': mem.get('metadata', {}).get('score', 0.0)
#                 })
            
#             self.logger.info(
#                 f"[ARCHIVAL SEARCH] Query: '{query[:50]}...', Found: {len(results)} docs"
#             )
            
#             return results
            
#         except Exception as e:
#             self.logger.error(f"Error in archival search: {e}")
#             return []
    
    
#     async def execute_tool_sequence(
#         self,
#         tool_sequence: List[Dict[str, Any]],
#         query: str,
#         chat_history: str,
#         system_language: str,
#         provider_type: str = ProviderType.OPENAI,
#         model_name: str = "gpt-4.1-nano"
#     ) -> List[Dict[str, Any]]:
#         """
#         Execute multiple tools in sequence
        
#         Args:
#             tool_sequence: List of tools to execute [{'tool_name': '...', 'params': {...}, 'purpose': '...'}]
#             query: Original user query
#             chat_history: Formatted conversation history
#             system_language: User's language
#             provider_type: LLM provider
#             model_name: Model to use
            
#         Returns:
#             List of tool execution results
#         """
#         all_results = []
        
#         try:
#             self.logger.info(f"[TOOL SEQUENCE] Executing {len(tool_sequence)} tool(s)")
            
#             for idx, tool_step in enumerate(tool_sequence, 1):
#                 tool_name = tool_step.get('tool_name')
#                 tool_params = tool_step.get('params', {})
#                 tool_purpose = tool_step.get('purpose', '')
                
#                 self.logger.info(f"[TOOL {idx}/{len(tool_sequence)}] â–¶ Executing: {tool_name}")
#                 self.logger.info(f"[TOOL {idx}] Purpose: {tool_purpose}")
#                 self.logger.info(f"[TOOL {idx}] Params: {json.dumps(tool_params, ensure_ascii=False)}")
                
#                 try:
#                     result = await self._execute_single_tool(
#                         tool_name=tool_name,
#                         tool_params=tool_params,
#                         query=query,
#                         chat_history=chat_history,
#                         system_language=system_language,
#                         provider_type=provider_type,
#                         model_name=model_name
#                     )
                    
#                     result['tool_index'] = idx
#                     result['tool_purpose'] = tool_purpose
#                     all_results.append(result)
                    
#                     self.logger.info(f"[TOOL {idx}]  Completed - Status: {result.get('status', 'unknown')}")
                    
#                 except Exception as e:
#                     self.logger.error(f"[TOOL {idx}]  Error: {e}")
#                     all_results.append({
#                         'tool_name': tool_name,
#                         'tool_index': idx,
#                         'status': 'error',
#                         'error': str(e),
#                         'tool_purpose': tool_purpose
#                     })
            
#             self.logger.info(f"[TOOL SEQUENCE] Complete - {len(all_results)}/{len(tool_sequence)} executed")
#             return all_results
            
#         except Exception as e:
#             self.logger.error(f"[TOOL SEQUENCE] Fatal error: {e}")
#             return [{
#                 'status': 'error',
#                 'error': f"Tool sequence execution failed: {str(e)}"
#             }]
    
    
#     async def _execute_single_tool(
#         self,
#         tool_name: str,
#         tool_params: Dict[str, Any],
#         query: str,
#         chat_history: str,
#         system_language: str,
#         provider_type: str,
#         model_name: str
#     ) -> Dict[str, Any]:
#         """
#         Execute a single tool
#         """
#         try:
#             # Extract symbols from params
#             symbols = tool_params.get('symbols', [])
            
#             self.logger.info(f"[TOOL EXEC] Tool: {tool_name}")
#             self.logger.info(f"[TOOL EXEC] Symbols extracted: {symbols}")
            
#             # =================================================================
#             # ROUTE TO CORRECT EXECUTION FUNCTION
#             # =================================================================
            
#             if tool_name == 'showStockPrice':
#                 # Fetch price data from FMP
                
#                 return {
#                     'tool_name': tool_name,
#                     'status': 'not_implemented',
#                     'message': 'User needs to implement FMP data fetching',
#                     'symbols': symbols
#                 }
            
#             elif tool_name == 'showStockFinancials':
#                 # Fetch financials from FMP

#                 return {
#                     'tool_name': tool_name,
#                     'status': 'not_implemented',
#                     'message': 'User needs to implement FMP data fetching',
#                     'symbols': symbols
#                 }
            
#             elif tool_name == 'showStockChart':
                
#                 data = {
#                     'symbol': symbols[0] if symbols else 'UNKNOWN',
#                 }
                
#                 # Collect streaming response
#                 response_text = ""
#                 async for chunk in analyze_stock_stream(
#                     data=data,
#                     user_query=query,
#                     system_language=system_language,
#                     model_name=model_name,
#                     chat_history=chat_history,
#                     provider_type=provider_type
#                 ):
#                     response_text += chunk
                
#                 return {
#                     'tool_name': tool_name,
#                     'status': 'success',
#                     'response': response_text,
#                     'symbols': symbols
#                 }
            
#             elif tool_name == 'showStockNews':
#                 # Fetch news from FMP 
                

#                 return {
#                     'tool_name': tool_name,
#                     'status': 'not_implemented',
#                     'message': 'User needs to implement FMP news fetching',
#                     'symbols': symbols
#                 }
            
#             elif tool_name == 'showMarketOverview':
#                 # Fetch market data 

#                 return {
#                     'tool_name': tool_name,
#                     'status': 'not_implemented',
#                     'message': 'User needs to implement market data fetching'
#                 }
            
#             elif tool_name == 'showTrendingStocks':
#                 # Fetch trending data 
#                 self.logger.info(f"[TOOL EXEC] âš ï¸ showTrendingStocks needs trending data - COMMENTED OUT")
#                 return {
#                     'tool_name': tool_name,
#                     'status': 'not_implemented',
#                     'message': 'User needs to implement trending data fetching'
#                 }
            
#             elif tool_name == 'showStockHeatmap':
#                 # Fetch heatmap data 
                

#                 return {
#                     'tool_name': tool_name,
#                     'status': 'not_implemented',
#                     'message': 'User needs to implement heatmap data fetching'
#                 }
            
#             elif tool_name == 'cryptoChart':
#                 # Fetch crypto data first
                

#                 return {
#                     'tool_name': tool_name,
#                     'status': 'not_implemented',
#                     'message': 'User needs to implement crypto data fetching',
#                     'symbols': symbols
#                 }
            
#             else:
#                 self.logger.warning(f"[TOOL EXEC] Unknown tool: {tool_name}")
#                 return {
#                     'tool_name': tool_name,
#                     'status': 'unknown_tool',
#                     'error': f'Tool {tool_name} not recognized'
#                 }
                
#         except Exception as e:
#             self.logger.error(f"[TOOL EXEC] Error executing {tool_name}: {e}")
#             return {
#                 'tool_name': tool_name,
#                 'status': 'error',
#                 'error': str(e)
#             }
    
    
#     async def execute_tool_with_context(
#         self,
#         tool_name: str,
#         tool_params: Dict[str, Any],
#         query: str,
#         chat_history: str,
#         system_language: str,
#         provider_type: str = ProviderType.OPENAI,
#         model_name: str = "gpt-4.1-nano"
#     ) -> Dict[str, Any]:
        
#         return await self._execute_single_tool(
#             tool_name=tool_name,
#             tool_params=tool_params,
#             query=query,
#             chat_history=chat_history,
#             system_language=system_language,
#             provider_type=provider_type,
#             model_name=model_name
#         )
    
    
#     # =========================================================================
#     # Private helper methods
#     # =========================================================================
    
#     async def _temporal_search(
#         self,
#         session_id: str,
#         params: Dict,
#         user_id: str
#     ) -> List[Dict]:
#         """Search by time period"""
#         try:
#             period = params.get('period', 'today')
#             now = datetime.now()
            
#             if period == 'today':
#                 start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
#             elif period == 'yesterday':
#                 start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
#             elif period == 'last_week':
#                 start_date = now - timedelta(days=7)
#             elif period == 'last_month':
#                 start_date = now - timedelta(days=30)
#             else:
#                 start_date = now - timedelta(days=7)
            
#             messages = await self.session_repo.get_session_messages_by_date_range(
#                 session_id=session_id,
#                 start_date=start_date,
#                 end_date=now,
#                 limit=params.get('limit', 20)
#             )
            
#             return messages
            
#         except Exception as e:
#             self.logger.error(f"Temporal search error: {e}")
#             return []
    
    
#     async def _symbol_search(
#         self,
#         session_id: str,
#         symbols: List[str],
#         limit: int
#     ) -> List[Dict]:
#         """Search by stock/crypto symbols"""
#         try:
#             if not symbols:
#                 return []
            
#             all_messages = await self.session_repo.get_session_messages(
#                 session_id=session_id,
#                 limit=100
#             )
            
#             filtered = []
#             for msg in all_messages:
#                 content = msg.get('content', '').upper()
#                 if any(symbol.upper() in content for symbol in symbols):
#                     filtered.append(msg)
#                     if len(filtered) >= limit:
#                         break
            
#             return filtered
            
#         except Exception as e:
#             self.logger.error(f"Symbol search error: {e}")
#             return []
    
    
#     async def _topic_search(
#         self,
#         session_id: str,
#         topic: str,
#         limit: int
#     ) -> List[Dict]:
#         """Search by topic/keywords using semantic search"""
#         try:
#             collection_name = f"memory_short_{session_id[:8]}"
            
#             memories = await self.memory_manager.search_relevant_memory(
#                 query=topic,
#                 collection_name=collection_name,
#                 top_k=limit
#             )
            
#             messages = []
#             for mem in memories:
#                 messages.append({
#                     'content': mem.get('content', ''),
#                     'metadata': mem.get('metadata', {}),
#                     'role': mem.get('metadata', {}).get('role', 'assistant'),
#                     'created_at': mem.get('metadata', {}).get('timestamp', '')
#                 })
            
#             return messages
            
#         except Exception as e:
#             self.logger.error(f"Topic search error: {e}")
#             return []
    
    
#     async def format_search_results_for_context(
#         self,
#         recall_results: List[Dict],
#         archival_results: List[Dict]
#     ) -> str:
#         """Format search results for inclusion in LLM context"""
#         context_parts = []
        
#         if recall_results:
#             context_parts.append("📝 RELEVANT CONVERSATION HISTORY:")
#             for i, msg in enumerate(recall_results[:5], 1):
#                 role = msg.get('role', 'user')
#                 content = msg.get('content', '')[:200]
#                 timestamp = msg.get('created_at', '')
#                 context_parts.append(
#                     f"{i}. [{role.upper()}] {timestamp}: {content}..."
#                 )
#             context_parts.append("")
        
#         if archival_results:
#             context_parts.append("📚 RELEVANT KNOWLEDGE:")
#             for i, doc in enumerate(archival_results[:3], 1):
#                 content = doc.get('content', '')[:300]
#                 source = doc.get('source', 'knowledge_base')
#                 context_parts.append(
#                     f"{i}. [{source}]: {content}..."
#                 )
#             context_parts.append("")
        
#         return "\n".join(context_parts) if context_parts else ""
    
    
#     async def format_tool_results_for_context(
#         self,
#         tool_results: List[Dict[str, Any]]
#     ) -> str:
#         """
#         Format tool execution results for LLM context
        
#         Args:
#             tool_results: List of tool execution results
            
#         Returns:
#             Formatted string for context
#         """
#         if not tool_results:
#             return ""
        
#         context_parts = ["🔧 TOOL EXECUTION RESULTS:\n"]
        
#         for result in tool_results:
#             tool_name = result.get('tool_name', 'unknown')
#             tool_index = result.get('tool_index', 0)
#             status = result.get('status', 'unknown')
            
#             context_parts.append(f"\n**Tool {tool_index}: {tool_name}**")
#             context_parts.append(f"Status: {status}")
            
#             if status == 'success':
#                 response = result.get('response', '')
#                 if response:
#                     # Truncate long responses
#                     preview = response[:500] + "..." if len(response) > 500 else response
#                     context_parts.append(f"Result: {preview}")
                
#                 symbols = result.get('symbols', [])
#                 if symbols:
#                     context_parts.append(f"Symbols: {', '.join(symbols)}")
            
#             elif status == 'error':
#                 error = result.get('error', 'Unknown error')
#                 context_parts.append(f"Error: {error}")
            
#             elif status == 'not_implemented':
#                 message = result.get('message', '')
#                 context_parts.append(f"Note: {message}")
            
#             context_parts.append("")
        
#         return "\n".join(context_parts)



import re
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.memory.memory_manager import MemoryManager
from src.helpers.chat_management_helper import ChatService
from src.database.repository.sessions import SessionRepository
from src.providers.provider_factory import ProviderType

from src.services.v2.tool_execution_service import ToolExecutionService

from src.helpers.v2.chat_helper import (
    analyze_stock_stream,
    analyze_market_overview_stream,
    analyze_stock_trending_stream,
    analyze_stock_heatmap_stream,
    general_chat_bot_stream
)


class MemorySearchService(LoggerMixin):

    def __init__(self):
        super().__init__()
        self.memory_manager = MemoryManager()
        self.session_repo = SessionRepository()
        self.chat_service = ChatService()
        
        # Initialize Enhanced Tool Execution Service
        self.tool_execution_service = ToolExecutionService()
    
    
    async def search_recall_memory(
        self,
        session_id: str,
        strategy: str,
        params: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search conversation history with different strategies"""
        try:
            results = []
            
            if strategy == "temporal":
                results = await self._temporal_search(
                    session_id=session_id,
                    params=params,
                    user_id=user_id
                )
            
            elif strategy == "symbol":
                results = await self._symbol_search(
                    session_id=session_id,
                    symbols=params.get('symbols', []),
                    limit=params.get('limit', 10)
                )
            
            elif strategy == "topic":
                results = await self._topic_search(
                    session_id=session_id,
                    topic=params.get('topic', ''),
                    limit=params.get('limit', 10)
                )
            
            elif strategy == "hybrid":
                temporal_results = await self._temporal_search(
                    session_id=session_id,
                    params=params,
                    user_id=user_id
                )
                
                symbol_results = await self._symbol_search(
                    session_id=session_id,
                    symbols=params.get('symbols', []),
                    limit=5
                )
                
                # Merge and deduplicate
                seen_ids = set()
                for msg in temporal_results + symbol_results:
                    msg_id = msg.get('id')
                    if msg_id and msg_id not in seen_ids:
                        results.append(msg)
                        seen_ids.add(msg_id)
            
            self.logger.info(
                f"[RECALL SEARCH] Strategy: {strategy}, Found: {len(results)} messages"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in recall search: {e}")
            return []
    
    
    async def search_archival_memory(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search knowledge base (Qdrant)"""
        try:
            results = await self.memory_manager.search_relevant_memory(
                query=query,
                user_id=user_id,
                limit=limit
            )
            
            self.logger.info(f"[ARCHIVAL SEARCH] Found: {len(results)} documents")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in archival search: {e}")
            return []
    
    
    async def execute_memory_searches(
        self,
        query: str,
        recall_params: Optional[Dict],
        archival_query: Optional[str],
        need_recall: bool,
        need_archival: bool,
        session_id: str,
        user_id: str
    ) -> Dict[str, List]:
        """Execute memory searches based on Inner Thoughts decision"""
        
        results = {
            'recall_results': [],
            'archival_results': []
        }
        
        try:
            # Recall search
            if need_recall and recall_params:
                strategy = recall_params.get('strategy', 'topic')
                
                # Build search params
                search_params = {
                    'topic': recall_params.get('topic'),
                    'symbols': recall_params.get('symbols', []),
                    'date_filter': recall_params.get('date_filter'),
                    'period': recall_params.get('period'),
                    'limit': recall_params.get('limit', 10)
                }
                
                results['recall_results'] = await self.search_recall_memory(
                    session_id=session_id,
                    strategy=strategy,
                    params=search_params,
                    user_id=user_id
                )
            
            # Archival search
            if need_archival and archival_query:
                results['archival_results'] = await self.search_archival_memory(
                    query=archival_query,
                    user_id=user_id,
                    limit=5
                )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in memory searches: {e}")
            return results
    
    
    async def execute_tool_sequence(
        self,
        tool_sequence: List[Dict],
        query: str,
        chat_history: List,
        system_language: str,
        provider_type: str,
        model_name: str
    ) -> List[Dict[str, Any]]:
        """
        Execute tool sequence
        
        Args:
            tool_sequence: List of tools to execute
            query: Original user query
            chat_history: Formatted conversation history
            system_language: User's language
            provider_type: LLM provider
            model_name: Model to use
            
        Returns:
            List of tool execution results
        """
        all_results = []
        
        try:
            self.logger.info(f"[TOOL SEQUENCE] Executing {len(tool_sequence)} tool(s)")
            
            for idx, tool_step in enumerate(tool_sequence, 1):
                tool_name = tool_step.get('tool_name')
                tool_params = tool_step.get('params', {})
                tool_purpose = tool_step.get('purpose', '')
                
                self.logger.info(f"[TOOL {idx}/{len(tool_sequence)}] ▶ Executing: {tool_name}")
                self.logger.info(f"[TOOL {idx}] Purpose: {tool_purpose}")
                self.logger.info(f"[TOOL {idx}] Params: {json.dumps(tool_params, ensure_ascii=False)}")
                
                try:
                    result = await self._execute_single_tool(
                        tool_name=tool_name,
                        tool_params=tool_params,
                        query=query,
                        chat_history=chat_history,
                        system_language=system_language,
                        provider_type=provider_type,
                        model_name=model_name
                    )
                    
                    result['tool_index'] = idx
                    result['tool_purpose'] = tool_purpose
                    all_results.append(result)
                    
                    self.logger.info(f"[TOOL {idx}] ✓ Completed - Status: {result.get('status', 'unknown')}")
                    
                except Exception as e:
                    self.logger.error(f"[TOOL {idx}] ✗ Error: {e}")
                    all_results.append({
                        'tool_name': tool_name,
                        'tool_index': idx,
                        'status': 'error',
                        'error': str(e),
                        'tool_purpose': tool_purpose
                    })
            
            self.logger.info(f"[TOOL SEQUENCE] Complete - {len(all_results)}/{len(tool_sequence)} executed")
            return all_results
            
        except Exception as e:
            self.logger.error(f"[TOOL SEQUENCE] Fatal error: {e}")
            return [{
                'status': 'error',
                'error': f"Tool sequence execution failed: {str(e)}"
            }]
    
    
    async def _execute_single_tool(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        query: str,
        chat_history: str,
        system_language: str,
        provider_type: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Execute a single tool - DELEGATES TO ToolExecutionService
        
        This replaces the old "not_implemented" stubs with full implementation
        """
        try:
            self.logger.info(f"[TOOL EXEC] Delegating to ToolExecutionService")
            self.logger.info(f"[TOOL EXEC] Tool: {tool_name}")
            self.logger.info(f"[TOOL EXEC] Params: {json.dumps(tool_params, ensure_ascii=False)}")
            
            result = await self.tool_execution_service.execute_single_tool(
                tool_name=tool_name,
                tool_params=tool_params,
                query=query,
                chat_history=chat_history,
                system_language=system_language,
                provider_type=provider_type,
                model_name=model_name
            )
            self.logger.info(f"[TOOL EXEC] Result: {result}")
            self.logger.info(f"[TOOL EXEC] Result status: {result.get('status', 'unknown')}")
            
            # Old code expected '200'/'error'
            if result.get('status') == 'success':
                result['status'] = '200'  # For backward compatibility
            
            return result
            
        except Exception as e:
            self.logger.error(f"[TOOL EXEC] Error executing {tool_name}: {e}", exc_info=True)
            return {
                'tool_name': tool_name,
                'status': 'error',
                'error': str(e),
                'symbols': tool_params.get('symbols', [])
            }
    
    
    async def execute_tool_with_context(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        query: str,
        conversation_history: List,
        provider_type: str = ProviderType.OPENAI,
        model_name: str = "gpt-4.1-nano"
    ) -> Dict[str, Any]:
        """
        Execute tool with conversation context
        
        Args:
            tool_name: Tool to execute
            tool_params: Tool parameters
            query: User query
            conversation_history: List of [content, role] tuples
            provider_type: LLM provider
            model_name: Model name
            
        Returns:
            Tool execution result
        """
        
        # Format conversation history as string
        chat_history_str = ""
        for msg in conversation_history:
            content = msg[0] if len(msg) > 0 else ""
            role = msg[1] if len(msg) > 1 else "user"
            chat_history_str += f"{role}: {content}\n"
        
        # Detect language from query
        system_language = self._detect_language(query)
        
        return await self._execute_single_tool(
            tool_name=tool_name,
            tool_params=tool_params,
            query=query,
            chat_history=chat_history_str,
            system_language=system_language,
            provider_type=provider_type,
            model_name=model_name
        )
    
    
    async def format_search_results_for_context(
        self,
        recall_results: List[Dict],
        archival_results: List[Dict]
    ) -> str:
        """Format search results for context assembly"""
        
        context_parts = []
        
        # Format recall results
        if recall_results:
            context_parts.append("[CONVERSATION HISTORY]")
            for idx, msg in enumerate(recall_results[:5], 1):
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:200]
                created_at = msg.get('created_at', '')
                
                context_parts.append(
                    f"{idx}. [{role}] ({created_at}): {content}..."
                )
        
        # Format archival results
        if archival_results:
            context_parts.append("\n[KNOWLEDGE BASE]")
            for idx, doc in enumerate(archival_results[:3], 1):
                content = doc.get('content', '')[:300]
                context_parts.append(f"{idx}. {content}...")
        
        return "\n".join(context_parts)
    
    
    # =========================================================================
    # Private helper methods
    # =========================================================================
    
    async def _temporal_search(
        self,
        session_id: str,
        params: Dict,
        user_id: str
    ) -> List[Dict]:
        """Search by time period"""
        try:
            period = params.get('period', 'today')
            now = datetime.now()
            
            if period == 'today':
                start_date = now.replace(hour=0, minute=0, second=0)
            elif period == 'yesterday':
                start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
            elif period == 'last_week':
                start_date = now - timedelta(days=7)
            elif period == 'last_month':
                start_date = now - timedelta(days=30)
            else:
                start_date = now - timedelta(days=7)
            
            # Search messages
            messages = await self.session_repo.get_session_messages(
                session_id=session_id,
                limit=params.get('limit', 10),
                after_date=start_date
            )
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error in temporal search: {e}")
            return []
    
    
    async def _symbol_search(
        self,
        session_id: str,
        symbols: List[str],
        limit: int = 10
    ) -> List[Dict]:
        """Search by stock/crypto symbols"""
        try:
            if not symbols:
                return []
            
            # Build pattern to match symbols
            symbol_pattern = '|'.join(symbols)
            
            messages = await self.session_repo.search_messages_by_content(
                session_id=session_id,
                search_pattern=symbol_pattern,
                limit=limit
            )
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error in symbol search: {e}")
            return []
    
    
    async def _topic_search(
        self,
        session_id: str,
        topic: str,
        limit: int = 10
    ) -> List[Dict]:
        """Search by topic/keyword"""
        try:
            if not topic:
                return []
            
            messages = await self.session_repo.search_messages_by_content(
                session_id=session_id,
                search_pattern=topic,
                limit=limit
            )
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error in topic search: {e}")
            return []
    
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Check for Vietnamese characters
        vietnamese_pattern = r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]'
        
        if re.search(vietnamese_pattern, text.lower()):
            return 'vi'
        
        return 'en'