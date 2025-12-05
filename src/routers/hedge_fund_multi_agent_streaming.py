# """
# Streaming API endpoint for Hedge Fund Multi-Agent Analysis
# ==========================================================

# This module adds streaming support to the hedge fund multi-agent system.
# Clients can receive real-time progress updates as each agent completes their analysis.

# Stream Event Types:
# - agent_start: Agent begins analysis
# - agent_complete: Agent finishes with signals
# - risk_complete: Risk management completed
# - portfolio_complete: Portfolio decisions ready
# - final: Complete response with all data
# - error: Error occurred

# Usage:
#     POST /api/hedge-fund/analyze/stream
#     {
#         "tickers": ["AAPL", "MSFT"],
#         "agents": ["warren_buffett", "technical_analyst"],
#         "model_name": "gpt-4.1-nano",
#         "provider": "openai",
#         "session_id": "session-123"
#     }
# """

# from fastapi import APIRouter, HTTPException, Query, Request, Depends
# from fastapi.responses import StreamingResponse
# from typing import List, Optional, Dict, Any, AsyncGenerator
# from pydantic import BaseModel
# from enum import Enum
# from datetime import datetime
# from dateutil.relativedelta import relativedelta
# import json
# import logging
# import asyncio

# # Import LangGraph components
# from langchain_core.messages import HumanMessage
# from langgraph.graph import END, StateGraph

# # Import hedge fund components
# from src.hedge_fund.agents.portfolio_manager import portfolio_management_agent
# from src.hedge_fund.agents.risk_manager import risk_management_agent
# from src.hedge_fund.graph.state import AgentState
# from src.hedge_fund.utils.analysts import ANALYST_CONFIG, get_analyst_nodes
# from src.hedge_fund.utils.progress import progress
# from src.hedge_fund.llm.models import normalize_provider_name

# # Import services
# from src.utils.config import settings
# from src.models.equity import APIResponse, APIResponseData
# from src.services.background_tasks import trigger_summary_update_nowait

# from src.agents.memory.memory_manager import MemoryManager
# from src.providers.provider_factory import ModelProviderFactory, ProviderType
# from src.helpers.chat_management_helper import ChatService
# from src.handlers.llm_chat_handler import ChatMessageHistory
# from src.routers.llm_chat import analyze_conversation_importance
# from src.handlers.api_key_authenticator_handler import APIKeyAuth
# from src.helpers.llm_helper import LLMGeneratorProvider

# # Setup logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# # Stream event types
# class StreamEventType(str, Enum):
#     AGENT_START = "agent_start"
#     AGENT_COMPLETE = "agent_complete"
#     RISK_COMPLETE = "risk_complete"
#     PORTFOLIO_COMPLETE = "portfolio_complete"
#     MEMORY_UPDATE = "memory_update"
#     FINAL = "final"
#     ERROR = "error"


# class StreamingMultiAgentService:
#     """Service for streaming multi-agent analysis with memory integration"""
    
#     def __init__(self):
#         self.memory_manager = MemoryManager()
#         self.chat_service = ChatService()
#         self.llm_provider = LLMGeneratorProvider()
    
#     def create_workflow(self, selected_analysts: List[str]) -> StateGraph:
#         """Create workflow with selected analysts - identical to non-streaming version"""
#         workflow = StateGraph(AgentState)
        
#         def start(state: AgentState):
#             """Initialize the workflow with the input message."""
#             return state
        
#         workflow.add_node("start_node", start)
        
#         # Get analyst nodes from configuration
#         analyst_nodes = get_analyst_nodes()
        
#         # Add selected analyst nodes
#         for analyst_key in selected_analysts:
#             if analyst_key in analyst_nodes:
#                 node_name, node_func = analyst_nodes[analyst_key]
#                 workflow.add_node(node_name, node_func)
#                 workflow.add_edge("start_node", node_name)
#             else:
#                 logger.warning(f"Agent {analyst_key} not found in configuration")
        
#         # Always add risk and portfolio management
#         workflow.add_node("risk_management_agent", risk_management_agent)
#         workflow.add_node("portfolio_manager", portfolio_management_agent)
        
#         # Connect selected analysts to risk management
#         for analyst_key in selected_analysts:
#             if analyst_key in analyst_nodes:
#                 node_name = analyst_nodes[analyst_key][0]
#                 workflow.add_edge(node_name, "risk_management_agent")
        
#         workflow.add_edge("risk_management_agent", "portfolio_manager")
#         workflow.add_edge("portfolio_manager", END)
        
#         workflow.set_entry_point("start_node")
#         return workflow
    
#     async def stream_analysis(
#         self,
#         tickers: List[str],
#         agents: List[str],
#         model_name: str,
#         provider: str,
#         session_id: Optional[str] = None,
#         collection_name: Optional[str] = None,
#         initial_cash: float = 100000.0,
#         margin_requirement: float = 0.0,
#         show_reasoning: bool = True,
#         user_id: Optional[str] = None
#     ) -> AsyncGenerator[str, None]:
#         """
#         Stream multi-agent analysis with real-time progress updates
        
#         Yields SSE-formatted events as each agent completes
#         """
        
#         try:
#             # Validate agents
#             valid_agents = []
#             for agent in agents:
#                 if agent in ANALYST_CONFIG or agent in get_analyst_nodes():
#                     valid_agents.append(agent)
#                 else:
#                     logger.warning(f"Invalid agent: {agent}")
            
#             if not valid_agents:
#                 yield self._format_sse_event(StreamEventType.ERROR, {
#                     "error": "No valid agents selected"
#                 })
#                 return
            
#             # Set dates
#             end_date = datetime.now().strftime("%Y-%m-%d")
#             start_date = (datetime.now() - relativedelta(months=3)).strftime("%Y-%m-%d")
            
#             # Initialize progress
#             progress.start()
#             progress.agent_status.clear()
            
#             # === PHASE 1: Get Memory Context ===
#             context = ""
#             memory_stats = {}
#             document_references = []
            
#             if session_id and user_id:
#                 try:
#                     yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
#                         "status": "loading_memory",
#                         "message": "Loading conversation memory..."
#                     })
                    
#                     context, memory_stats, document_references = await self.memory_manager.get_relevant_context(
#                         session_id=session_id,
#                         user_id=user_id,
#                         current_query=f"Analyze {', '.join(tickers)} using {', '.join(valid_agents)}",
#                         llm_provider=self.llm_provider,
#                         max_short_term=5,
#                         max_long_term=3,
#                         base_collection=collection_name
#                     )
                    
#                     yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
#                         "status": "memory_loaded",
#                         "stats": memory_stats,
#                         "message": f"Loaded {memory_stats.get('total_memories', 0)} memories"
#                     })
                    
#                     logger.info(f"Retrieved memory context: {memory_stats}")
#                 except Exception as e:
#                     logger.error(f"Error getting memory context: {e}")
#                     yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
#                         "status": "memory_error",
#                         "error": str(e)
#                     })
            
#             # === PHASE 2: Get Chat History ===
#             chat_history = ""
#             if session_id:
#                 try:
#                     chat_history = ChatMessageHistory.string_message_chat_history(session_id)
#                 except Exception as e:
#                     logger.error(f"Error fetching history: {e}")
            
#             # === PHASE 3: Build Enhanced Initial Message ===
#             initial_content = "Make trading decisions based on the provided data."
            
#             if context:
#                 initial_content = f"{context}\n\n{initial_content}"
            
#             if chat_history:
#                 initial_content = f"[Previous Conversation]\n{chat_history}\n\n{initial_content}"
            
#             # === PHASE 4: Create Portfolio Structure ===
#             portfolio = {
#                 "cash": initial_cash,
#                 "margin_requirement": margin_requirement,
#                 "margin_used": 0.0,
#                 "positions": {
#                     ticker: {
#                         "long": 0,
#                         "short": 0,
#                         "long_cost_basis": 0.0,
#                         "short_cost_basis": 0.0,
#                         "short_margin_used": 0.0
#                     }
#                     for ticker in tickers
#                 },
#                 "realized_gains": {
#                     ticker: {"long": 0.0, "short": 0.0}
#                     for ticker in tickers
#                 }
#             }
            
#             # === PHASE 5: Create Initial State ===
#             initial_state = {
#                 "messages": [HumanMessage(content=initial_content)],
#                 "data": {
#                     "tickers": tickers,
#                     "portfolio": portfolio,
#                     "start_date": start_date,
#                     "end_date": end_date,
#                     "analyst_signals": {}
#                 },
#                 "metadata": {
#                     "show_reasoning": show_reasoning,
#                     "model_name": model_name,
#                     "model_provider": normalize_provider_name(provider),
#                     "session_id": session_id,
#                     "collection_name": collection_name,
#                     "user_id": user_id,
#                     "context": context,
#                     "document_references": document_references
#                 }
#             }
            
#             # === PHASE 6: Save User Question ===
#             question_id = None
#             if session_id and user_id:
#                 try:
#                     question_content = f"Analyze {', '.join(tickers)} using agents: {', '.join(valid_agents)}"
#                     question_id = self.chat_service.save_user_question(
#                         session_id=session_id,
#                         created_at=datetime.now(),
#                         created_by=user_id,
#                         content=question_content
#                     )
#                 except Exception as e:
#                     logger.error(f"Error saving user question: {e}")
            
#             # === PHASE 7: Create and Stream Workflow ===
#             logger.info(f"Starting streaming analysis with agents: {valid_agents} for tickers: {tickers}")
            
#             workflow = self.create_workflow(valid_agents)
#             app = workflow.compile()
            
#             # Track completed agents and accumulated results
#             completed_agents = []
#             agent_analysis = {}
#             risk_analysis = None
#             trading_decisions = {}
#             portfolio_summary = []
            
#             # Get analyst nodes mapping for node name -> agent key
#             analyst_nodes = get_analyst_nodes()
#             node_to_agent = {node_name: agent_key for agent_key, (node_name, _) in analyst_nodes.items()}
            
#             # Stream workflow execution
#             async for chunk in app.astream(initial_state, stream_mode="updates"):
#                 # chunk is a dict with node_name as key and state update as value
#                 for node_name, state_update in chunk.items():
                    
#                     logger.info(f"Stream update from node: {node_name}")
                    
#                     # Handle analyst nodes
#                     if node_name in node_to_agent:
#                         agent_key = node_to_agent[node_name]
#                         agent_display = ANALYST_CONFIG.get(agent_key, {}).get("display_name", agent_key)
                        
#                         # Emit start event
#                         yield self._format_sse_event(StreamEventType.AGENT_START, {
#                             "agent": agent_key,
#                             "display_name": agent_display,
#                             "message": f"{agent_display} analyzing..."
#                         })
                        
#                         # Extract signals from state update
#                         if "data" in state_update and "analyst_signals" in state_update["data"]:
#                             agent_node_name = f"{agent_key}_agent"
#                             if agent_node_name in state_update["data"]["analyst_signals"]:
#                                 agent_signals = state_update["data"]["analyst_signals"][agent_node_name]
                                
#                                 # Process signals
#                                 processed_signals = {}
#                                 for ticker, signal_data in agent_signals.items():
#                                     if isinstance(signal_data, dict):
#                                         processed_signals[ticker] = {
#                                             "signal": signal_data.get("signal", "neutral"),
#                                             "confidence": signal_data.get("confidence", 0.0),
#                                             "reasoning": signal_data.get("reasoning", "")[:200] if isinstance(signal_data.get("reasoning"), str) else str(signal_data.get("reasoning", ""))[:200]
#                                         }
                                
#                                 agent_analysis[agent_key] = processed_signals
#                                 completed_agents.append(agent_key)
                                
#                                 # Emit completion event with signals
#                                 yield self._format_sse_event(StreamEventType.AGENT_COMPLETE, {
#                                     "agent": agent_key,
#                                     "display_name": agent_display,
#                                     "signals": processed_signals,
#                                     "message": f"{agent_display} completed analysis"
#                                 })
                    
#                     # Handle risk management node
#                     elif node_name == "risk_management_agent":
#                         yield self._format_sse_event(StreamEventType.AGENT_START, {
#                             "agent": "risk_manager",
#                             "display_name": "Risk Manager",
#                             "message": "Evaluating portfolio risk..."
#                         })
                        
#                         if "data" in state_update and "analyst_signals" in state_update["data"]:
#                             risk_analysis = state_update["data"]["analyst_signals"].get("risk_management_agent")
                            
#                             yield self._format_sse_event(StreamEventType.RISK_COMPLETE, {
#                                 "risk_analysis": risk_analysis,
#                                 "message": "Risk assessment completed"
#                             })
                    
#                     # Handle portfolio manager node
#                     elif node_name == "portfolio_manager":
#                         yield self._format_sse_event(StreamEventType.AGENT_START, {
#                             "agent": "portfolio_manager",
#                             "display_name": "Portfolio Manager",
#                             "message": "Making trading decisions..."
#                         })
                        
#                         # Extract trading decisions from messages
#                         if "messages" in state_update:
#                             for msg in state_update["messages"]:
#                                 if hasattr(msg, 'content') and hasattr(msg, 'name'):
#                                     if msg.name == "portfolio_manager":
#                                         try:
#                                             decisions_data = json.loads(msg.content)
#                                             for ticker, decision in decisions_data.items():
#                                                 trading_decisions[ticker] = {
#                                                     "action": decision.get("action", "hold"),
#                                                     "quantity": decision.get("quantity", 0),
#                                                     "confidence": decision.get("confidence", 0.0),
#                                                     "reasoning": decision.get("reasoning", "")
#                                                 }
                                                
#                                                 portfolio_summary.append({
#                                                     "ticker": ticker,
#                                                     "action": decision.get("action", "hold"),
#                                                     "quantity": decision.get("quantity", 0),
#                                                     "confidence": decision.get("confidence", 0.0)
#                                                 })
#                                         except json.JSONDecodeError:
#                                             logger.warning("Could not parse portfolio decisions")
                        
#                         yield self._format_sse_event(StreamEventType.PORTFOLIO_COMPLETE, {
#                             "trading_decisions": trading_decisions,
#                             "portfolio_summary": portfolio_summary,
#                             "message": "Trading decisions completed"
#                         })
            
#             # === PHASE 8: Generate Strategy ===
#             portfolio_strategy = self._generate_strategy(trading_decisions, margin_requirement, valid_agents)
            
#             # === PHASE 9: Build Response Text for Memory ===
#             response_text = self._build_response_text(
#                 agent_analysis, risk_analysis, trading_decisions, portfolio_strategy
#             )
            
#             # === PHASE 10: Analyze Importance ===
#             importance_score = 0.5
#             if session_id and user_id:
#                 try:
#                     analysis_model = "gpt-4.1-nano" if provider.lower() == "openai" else model_name
                    
#                     importance_score = await analyze_conversation_importance(
#                         query=f"Analyze {', '.join(tickers)} using agents: {', '.join(valid_agents)}",
#                         response=response_text,
#                         llm_provider=self.llm_provider,
#                         model_name=analysis_model,
#                         provider_type=provider
#                     )
                    
#                     logger.info(f"LLM analysis - Importance: {importance_score}")
#                 except Exception as e:
#                     logger.error(f"Error analyzing conversation: {e}")
            
#             # === PHASE 11: Store in Memory ===
#             if session_id and user_id:
#                 try:
#                     await self.memory_manager.store_conversation_turn(
#                         session_id=session_id,
#                         user_id=user_id,
#                         query=f"Analyze {', '.join(tickers)} using agents: {', '.join(valid_agents)}",
#                         response=response_text,
#                         metadata={
#                             "tickers": tickers,
#                             "agents": valid_agents,
#                             "model": model_name,
#                             "provider": provider
#                         },
#                         importance_score=importance_score,
#                         base_collection=collection_name
#                     )
                    
#                     # Save assistant response
#                     if question_id:
#                         self.chat_service.save_assistant_response(
#                             session_id=session_id,
#                             created_at=datetime.now(),
#                             question_id=question_id,
#                             content=response_text,
#                             response_time=0.1
#                         )
                    
#                     # Trigger summary update
#                     trigger_summary_update_nowait(session_id=session_id, user_id=user_id)
                    
#                     yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
#                         "status": "saved",
#                         "importance_score": importance_score,
#                         "message": "Analysis saved to memory"
#                     })
                    
#                 except Exception as save_error:
#                     logger.error(f"Error saving to memory: {str(save_error)}")
#                     yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
#                         "status": "save_error",
#                         "error": str(save_error)
#                     })
            
#             # === PHASE 12: Send Final Complete Response ===
#             final_response = {
#                 "session_id": session_id,
#                 "timestamp": datetime.now().isoformat(),
#                 "agent_analysis": agent_analysis,
#                 "risk_analysis": risk_analysis,
#                 "trading_decisions": trading_decisions,
#                 "portfolio_summary": portfolio_summary,
#                 "portfolio_strategy": portfolio_strategy,
#                 "model_used": model_name,
#                 "provider_used": provider,
#                 "tickers_analyzed": tickers,
#                 "agents_used": valid_agents,
#                 "memory_stats": memory_stats
#             }
            
#             yield self._format_sse_event(StreamEventType.FINAL, final_response)
            
#             # Send completion marker
#             yield "data: [DONE]\n\n"
            
#         except Exception as e:
#             logger.error(f"Error in streaming analysis: {str(e)}", exc_info=True)
#             yield self._format_sse_event(StreamEventType.ERROR, {
#                 "error": str(e),
#                 "error_type": type(e).__name__
#             })
#             yield "data: [DONE]\n\n"
        
#         finally:
#             progress.stop()
    
#     def _format_sse_event(self, event_type: StreamEventType, data: Dict[str, Any]) -> str:
#         """Format data as Server-Sent Event"""
#         event_data = {
#             "type": event_type,
#             "data": data,
#             "timestamp": datetime.now().isoformat()
#         }
#         return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
    
#     def _build_response_text(
#         self,
#         agent_analysis: Dict,
#         risk_analysis: Optional[Dict],
#         trading_decisions: Dict,
#         portfolio_strategy: str
#     ) -> str:
#         """Build comprehensive response text for memory storage"""
        
#         parts = []
        
#         # Add agent analysis summary
#         if agent_analysis:
#             parts.append("## Agent Analysis Summary\n")
#             for agent, signals in agent_analysis.items():
#                 agent_display = ANALYST_CONFIG.get(agent, {}).get("display_name", agent)
#                 parts.append(f"\n**{agent_display}:**")
#                 for ticker, signal in signals.items():
#                     parts.append(f"- {ticker}: {signal.get('signal')} (Confidence: {signal.get('confidence', 0):.1f}%)")
#                     if signal.get('reasoning'):
#                         parts.append(f"  Reasoning: {signal.get('reasoning')[:200]}...")
        
#         # Add risk analysis if present
#         if risk_analysis:
#             parts.append("\n## Risk Assessment\n")
#             parts.append(str(risk_analysis)[:500] + "...")
        
#         # Add trading decisions
#         if trading_decisions:
#             parts.append("\n## Trading Decisions\n")
#             for ticker, decision in trading_decisions.items():
#                 parts.append(f"- {ticker}: {decision['action'].upper()} {decision['quantity']} shares (Confidence: {decision['confidence']:.1f}%)")
        
#         # Add portfolio strategy
#         parts.append(f"\n## Portfolio Strategy\n{portfolio_strategy}")
        
#         return "\n".join(parts)
    
#     def _generate_strategy(
#         self, 
#         decisions: Dict[str, Dict], 
#         margin_req: float,
#         agents_used: List[str]
#     ) -> str:
#         """Generate strategy text based on decisions and agents used"""
        
#         if not decisions:
#             return "No trading decisions generated."
        
#         # Count actions
#         actions = {}
#         for decision in decisions.values():
#             action = decision.get("action", "hold")
#             actions[action] = actions.get(action, 0) + 1
        
#         # Build strategy
#         parts = []
        
#         # Mention agents used
#         agent_names = [ANALYST_CONFIG.get(a, {}).get("display_name", a) for a in agents_used]
#         parts.append(f"Analysis performed by: {', '.join(agent_names)}.")
        
#         # Action summary
#         if actions:
#             action_summary = ", ".join([f"{count} {action}" for action, count in actions.items()])
#             parts.append(f"Actions: {action_summary}.")
        
#         # Margin info
#         if margin_req > 0:
#             parts.append(f"Margin available for short positions (requirement: {margin_req:.2f}).")
        
#         parts.append("Monitor positions and adjust based on market conditions.")
        
#         return " ".join(parts)


# # Initialize streaming service
# streaming_service = StreamingMultiAgentService()


# # ==================== EXAMPLE USAGE ====================

# def create_streaming_router(base_router: APIRouter, api_key_auth: APIKeyAuth):
    
#     @base_router.post("/analyze/stream")
#     async def analyze_multi_agent_stream(
#         request: Request,
#         tickers: List[str],
#         agents: List[str],
#         model_name: str = "gpt-4.1-nano",
#         provider: str = "openai",
#         session_id: Optional[str] = None,
#         collection_name: Optional[str] = None,
#         initial_cash: float = 100000.0,
#         margin_requirement: float = 0.0,
#         show_reasoning: bool = True,
#         api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
#     ):
#         """
#         Stream multi-agent trading analysis with real-time progress updates.
        
#         This endpoint streams events as each agent completes their analysis:
        
#         Event Types:
#         - agent_start: Agent begins analysis
#         - agent_complete: Agent finishes with signals  
#         - risk_complete: Risk management completed
#         - portfolio_complete: Portfolio decisions ready
#         - memory_update: Memory operations progress
#         - final: Complete response with all data
#         - error: Error occurred
#         """
#         try:
#             # Get user_id from request state
#             user_id = getattr(request.state, "user_id", None)
            
#             # Create async generator
#             async def event_generator():
#                 async for event in streaming_service.stream_analysis(
#                     tickers=tickers,
#                     agents=agents,
#                     model_name=model_name,
#                     provider=provider,
#                     session_id=session_id,
#                     collection_name=collection_name,
#                     initial_cash=initial_cash,
#                     margin_requirement=margin_requirement,
#                     show_reasoning=show_reasoning,
#                     user_id=user_id
#                 ):
#                     yield event
            
#             return StreamingResponse(
#                 event_generator(),
#                 media_type="text/event-stream",
#                 headers={
#                     "Cache-Control": "no-cache",
#                     "Connection": "keep-alive",
#                     "X-Accel-Buffering": "no"
#                 }
#             )
            
#         except Exception as e:
#             logger.error(f"Error setting up stream: {str(e)}", exc_info=True)
#             raise HTTPException(status_code=500, detail=str(e))
    
#     @base_router.get("/analyze/{tickers}/stream")
#     async def analyze_quick_stream(
#         request: Request,
#         tickers: str,
#         agents: str = Query(..., description="Comma-separated agent names"),
#         model_name: str = Query("gpt-4.1-nano", description="LLM model name"),
#         provider: str = Query("openai", description="LLM provider"),
#         session_id: Optional[str] = Query(None, description="Session ID"),
#         collection_name: Optional[str] = Query(None, description="Collection name"),
#         initial_cash: float = Query(100000.0, description="Initial cash"),
#         margin_requirement: float = Query(0.0, description="Margin requirement"),
#         api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
#     ):
#         """
#         Quick GET endpoint for streaming multi-agent analysis.
        
#         Examples:
#         - Single agent: /analyze/AAPL/stream?agents=warren_buffett
#         - Multiple agents: /analyze/AAPL,MSFT/stream?agents=warren_buffett,technical_analyst
#         - All analysts: /analyze/NVDA/stream?agents=ALL
#         """
#         try:
#             # Get user_id from request state
#             user_id = getattr(request.state, "user_id", None)
            
#             # Parse tickers
#             ticker_list = [t.strip().upper() for t in tickers.split(",")]
            
#             # Parse agents
#             if agents.upper() == "ALL":
#                 agent_list = list(ANALYST_CONFIG.keys())
#             else:
#                 agent_list = [a.strip().lower() for a in agents.split(",")]
            
#             # Create async generator
#             async def event_generator():
#                 async for event in streaming_service.stream_analysis(
#                     tickers=ticker_list,
#                     agents=agent_list,
#                     model_name=model_name,
#                     provider=provider,
#                     session_id=session_id,
#                     collection_name=collection_name,
#                     initial_cash=initial_cash,
#                     margin_requirement=margin_requirement,
#                     show_reasoning=True,
#                     user_id=user_id
#                 ):
#                     yield event
            
#             return StreamingResponse(
#                 event_generator(),
#                 media_type="text/event-stream",
#                 headers={
#                     "Cache-Control": "no-cache",
#                     "Connection": "keep-alive",
#                     "X-Accel-Buffering": "no"
#                 }
#             )
            
#         except Exception as e:
#             logger.error(f"Error: {str(e)}", exc_info=True)
#             raise HTTPException(status_code=500, detail=str(e))
    
#     return base_router


from fastapi import APIRouter, HTTPException, Query, Request, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any, AsyncGenerator
from pydantic import BaseModel
from enum import Enum
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import logging
import asyncio

# Import LangGraph components
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

# Import hedge fund components
from src.hedge_fund.agents.portfolio_manager import portfolio_management_agent
from src.hedge_fund.agents.risk_manager import risk_management_agent
from src.hedge_fund.graph.state import AgentState
from src.hedge_fund.utils.analysts import ANALYST_CONFIG, get_analyst_nodes
from src.hedge_fund.utils.progress import progress
from src.hedge_fund.llm.models import normalize_provider_name

# Import services
from src.utils.config import settings
from src.models.equity import APIResponse, APIResponseData
from src.services.background_tasks import trigger_summary_update_nowait

from src.agents.memory.memory_manager import MemoryManager
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.chat_management_helper import ChatService
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.routers.llm_chat import analyze_conversation_importance
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.helpers.llm_helper import LLMGeneratorProvider

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Stream event types
class StreamEventType(str, Enum):
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    RISK_COMPLETE = "risk_complete"
    PORTFOLIO_COMPLETE = "portfolio_complete"
    # MEMORY_UPDATE = "memory_update"  # Commented out - not in non-streaming response
    FINAL = "final"
    ERROR = "error"


class StreamingMultiAgentService:
    """Service for streaming multi-agent analysis with memory integration"""
    
    def __init__(self):
        self.memory_manager = MemoryManager()
        self.chat_service = ChatService()
        self.llm_provider = LLMGeneratorProvider()
    
    def create_workflow(self, selected_analysts: List[str]) -> StateGraph:
        """Create workflow with selected analysts - identical to non-streaming version"""
        workflow = StateGraph(AgentState)
        
        def start(state: AgentState):
            """Initialize the workflow with the input message."""
            return state
        
        workflow.add_node("start_node", start)
        
        # Get analyst nodes from configuration
        analyst_nodes = get_analyst_nodes()
        
        # Add selected analyst nodes
        for analyst_key in selected_analysts:
            if analyst_key in analyst_nodes:
                node_name, node_func = analyst_nodes[analyst_key]
                workflow.add_node(node_name, node_func)
                workflow.add_edge("start_node", node_name)
            else:
                logger.warning(f"Agent {analyst_key} not found in configuration")
        
        # Always add risk and portfolio management
        workflow.add_node("risk_management_agent", risk_management_agent)
        workflow.add_node("portfolio_manager", portfolio_management_agent)
        
        # Connect selected analysts to risk management
        for analyst_key in selected_analysts:
            if analyst_key in analyst_nodes:
                node_name = analyst_nodes[analyst_key][0]
                workflow.add_edge(node_name, "risk_management_agent")
        
        workflow.add_edge("risk_management_agent", "portfolio_manager")
        workflow.add_edge("portfolio_manager", END)
        
        workflow.set_entry_point("start_node")
        return workflow
    
    async def stream_analysis(
        self,
        tickers: List[str],
        agents: List[str],
        model_name: str,
        provider: str,
        session_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        initial_cash: float = 100000.0,
        margin_requirement: float = 0.0,
        show_reasoning: bool = True,
        user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream multi-agent analysis with real-time progress updates
        
        Yields SSE-formatted events as each agent completes
        """
        
        try:
            # Validate agents
            valid_agents = []
            for agent in agents:
                if agent in ANALYST_CONFIG or agent in get_analyst_nodes():
                    valid_agents.append(agent)
                else:
                    logger.warning(f"Invalid agent: {agent}")
            
            if not valid_agents:
                yield self._format_sse_event(StreamEventType.ERROR, {
                    "error": "No valid agents selected"
                })
                return
            
            # Set dates
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - relativedelta(months=3)).strftime("%Y-%m-%d")
            
            # Initialize progress
            progress.start()
            progress.agent_status.clear()
            
            # === PHASE 1: Get Memory Context ===
            context = ""
            memory_stats = {}
            document_references = []
            
            if session_id and user_id:
                try:
                    # Silently load memory without emitting events
                    # yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
                    #     "status": "loading_memory",
                    #     "message": "Loading conversation memory..."
                    # })
                    
                    context, memory_stats, document_references = await self.memory_manager.get_relevant_context(
                        session_id=session_id,
                        user_id=user_id,
                        current_query=f"Analyze {', '.join(tickers)} using {', '.join(valid_agents)}",
                        llm_provider=self.llm_provider,
                        max_short_term=5,
                        max_long_term=3,
                        base_collection=collection_name
                    )
                    
                    # yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
                    #     "status": "memory_loaded",
                    #     "stats": memory_stats,
                    #     "message": f"Loaded {memory_stats.get('total_memories', 0)} memories"
                    # })
                    
                    logger.info(f"Retrieved memory context: {memory_stats}")
                except Exception as e:
                    logger.error(f"Error getting memory context: {e}")
                    # yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
                    #     "status": "memory_error",
                    #     "error": str(e)
                    # })
            
            # === PHASE 2: Get Chat History ===
            chat_history = ""
            if session_id:
                try:
                    chat_history = ChatMessageHistory.string_message_chat_history(session_id)
                except Exception as e:
                    logger.error(f"Error fetching history: {e}")
            
            # === PHASE 3: Build Enhanced Initial Message ===
            initial_content = "Make trading decisions based on the provided data."
            
            if context:
                initial_content = f"{context}\n\n{initial_content}"
            
            if chat_history:
                initial_content = f"[Previous Conversation]\n{chat_history}\n\n{initial_content}"
            
            # === PHASE 4: Create Portfolio Structure ===
            portfolio = {
                "cash": initial_cash,
                "margin_requirement": margin_requirement,
                "margin_used": 0.0,
                "positions": {
                    ticker: {
                        "long": 0,
                        "short": 0,
                        "long_cost_basis": 0.0,
                        "short_cost_basis": 0.0,
                        "short_margin_used": 0.0
                    }
                    for ticker in tickers
                },
                "realized_gains": {
                    ticker: {"long": 0.0, "short": 0.0}
                    for ticker in tickers
                }
            }
            
            # === PHASE 5: Create Initial State ===
            initial_state = {
                "messages": [HumanMessage(content=initial_content)],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {}
                },
                "metadata": {
                    "show_reasoning": show_reasoning,
                    "model_name": model_name,
                    "model_provider": normalize_provider_name(provider),
                    "session_id": session_id,
                    "collection_name": collection_name,
                    "user_id": user_id,
                    "context": context,
                    "document_references": document_references
                }
            }
            
            # === PHASE 6: Save User Question ===
            question_id = None
            if session_id and user_id:
                try:
                    question_content = f"Analyze {', '.join(tickers)} using agents: {', '.join(valid_agents)}"
                    question_id = self.chat_service.save_user_question(
                        session_id=session_id,
                        created_at=datetime.now(),
                        created_by=user_id,
                        content=question_content
                    )
                except Exception as e:
                    logger.error(f"Error saving user question: {e}")
            
            # === PHASE 7: Create and Stream Workflow ===
            logger.info(f"Starting streaming analysis with agents: {valid_agents} for tickers: {tickers}")
            
            workflow = self.create_workflow(valid_agents)
            app = workflow.compile()
            
            # Track completed agents and accumulated results
            completed_agents = []
            agent_analysis = {}
            risk_analysis = None
            trading_decisions = {}
            portfolio_summary = []
            
            # Get analyst nodes mapping for node name -> agent key
            analyst_nodes = get_analyst_nodes()
            node_to_agent = {node_name: agent_key for agent_key, (node_name, _) in analyst_nodes.items()}
            
            # Stream workflow execution
            async for chunk in app.astream(initial_state, stream_mode="updates"):
                # chunk is a dict with node_name as key and state update as value
                for node_name, state_update in chunk.items():
                    
                    logger.info(f"Stream update from node: {node_name}")
                    
                    # Handle analyst nodes
                    if node_name in node_to_agent:
                        agent_key = node_to_agent[node_name]
                        agent_display = ANALYST_CONFIG.get(agent_key, {}).get("display_name", agent_key)
                        
                        # Emit start event
                        yield self._format_sse_event(StreamEventType.AGENT_START, {
                            "agent": agent_key,
                            "display_name": agent_display,
                            "message": f"{agent_display} analyzing..."
                        })
                        
                        # Extract signals from state update
                        if "data" in state_update and "analyst_signals" in state_update["data"]:
                            agent_node_name = f"{agent_key}_agent"
                            if agent_node_name in state_update["data"]["analyst_signals"]:
                                agent_signals = state_update["data"]["analyst_signals"][agent_node_name]
                                
                                # Process signals - match non-streaming format
                                processed_signals = {}
                                for ticker, signal_data in agent_signals.items():
                                    if isinstance(signal_data, dict):
                                        processed_signals[ticker] = {
                                            "signal": signal_data.get("signal", "neutral"),
                                            "confidence": int(signal_data.get("confidence", 0.0)),  # Convert to int
                                            "reasoning": signal_data.get("reasoning", ""),
                                            "metrics": signal_data.get("metrics", None),
                                            "strategy_signals": signal_data.get("strategy_signals", None)
                                        }
                                
                                agent_analysis[agent_key] = processed_signals
                                completed_agents.append(agent_key)
                                
                                # Emit completion event with signals
                                yield self._format_sse_event(StreamEventType.AGENT_COMPLETE, {
                                    "agent": agent_key,
                                    "display_name": agent_display,
                                    "signals": processed_signals,
                                    "message": f"{agent_display} completed analysis"
                                })
                    
                    # Handle risk management node
                    elif node_name == "risk_management_agent":
                        yield self._format_sse_event(StreamEventType.AGENT_START, {
                            "agent": "risk_manager",
                            "display_name": "Risk Manager",
                            "message": "Evaluating portfolio risk..."
                        })
                        
                        if "data" in state_update and "analyst_signals" in state_update["data"]:
                            risk_analysis = state_update["data"]["analyst_signals"].get("risk_management_agent")
                            
                            yield self._format_sse_event(StreamEventType.RISK_COMPLETE, {
                                "risk_analysis": risk_analysis,
                                "message": "Risk assessment completed"
                            })
                    
                    # Handle portfolio manager node
                    elif node_name == "portfolio_manager":
                        yield self._format_sse_event(StreamEventType.AGENT_START, {
                            "agent": "portfolio_manager",
                            "display_name": "Portfolio Manager",
                            "message": "Making trading decisions..."
                        })
                        
                        # Extract trading decisions from messages
                        if "messages" in state_update:
                            for msg in state_update["messages"]:
                                if hasattr(msg, 'content') and hasattr(msg, 'name'):
                                    if msg.name == "portfolio_manager":
                                        try:
                                            decisions_data = json.loads(msg.content)
                                            for ticker, decision in decisions_data.items():
                                                # Match non-streaming format with confidence as int
                                                trading_decisions[ticker] = {
                                                    "action": decision.get("action", "hold"),
                                                    "quantity": decision.get("quantity", 0),
                                                    "confidence": int(decision.get("confidence", 0.0)),
                                                    "reasoning": decision.get("reasoning", "")
                                                }
                                                
                                                portfolio_summary.append({
                                                    "ticker": ticker,
                                                    "action": decision.get("action", "hold"),
                                                    "quantity": decision.get("quantity", 0),
                                                    "confidence": int(decision.get("confidence", 0.0))
                                                })
                                        except json.JSONDecodeError:
                                            logger.warning("Could not parse portfolio decisions")
                        
                        yield self._format_sse_event(StreamEventType.PORTFOLIO_COMPLETE, {
                            "trading_decisions": trading_decisions,
                            "portfolio_summary": portfolio_summary,
                            "message": "Trading decisions completed"
                        })
            
            # === PHASE 8: Generate Strategy ===
            portfolio_strategy = self._generate_strategy(trading_decisions, margin_requirement, valid_agents)
            
            # === PHASE 9: Build Response Text for Memory ===
            response_text = self._build_response_text(
                agent_analysis, risk_analysis, trading_decisions, portfolio_strategy
            )
            
            # === PHASE 10: Analyze Importance ===
            importance_score = 0.5
            if session_id and user_id:
                try:
                    analysis_model = "gpt-4.1-nano" if provider.lower() == "openai" else model_name
                    
                    importance_score = await analyze_conversation_importance(
                        query=f"Analyze {', '.join(tickers)} using agents: {', '.join(valid_agents)}",
                        response=response_text,
                        llm_provider=self.llm_provider,
                        model_name=analysis_model,
                        provider_type=provider
                    )
                    
                    logger.info(f"LLM analysis - Importance: {importance_score}")
                except Exception as e:
                    logger.error(f"Error analyzing conversation: {e}")
            
            # === PHASE 11: Store in Memory ===
            if session_id and user_id:
                try:
                    await self.memory_manager.store_conversation_turn(
                        session_id=session_id,
                        user_id=user_id,
                        query=f"Analyze {', '.join(tickers)} using agents: {', '.join(valid_agents)}",
                        response=response_text,
                        metadata={
                            "tickers": tickers,
                            "agents": valid_agents,
                            "model": model_name,
                            "provider": provider
                        },
                        importance_score=importance_score,
                        base_collection=collection_name
                    )
                    
                    # Save assistant response
                    if question_id:
                        self.chat_service.save_assistant_response(
                            session_id=session_id,
                            created_at=datetime.now(),
                            question_id=question_id,
                            content=response_text,
                            response_time=0.1
                        )
                    
                    # Trigger summary update
                    trigger_summary_update_nowait(session_id=session_id, user_id=user_id)
                    
                    # Silently save to memory without emitting event
                    # yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
                    #     "status": "saved",
                    #     "importance_score": importance_score,
                    #     "message": "Analysis saved to memory"
                    # })
                    
                except Exception as save_error:
                    logger.error(f"Error saving to memory: {str(save_error)}")
                    # yield self._format_sse_event(StreamEventType.MEMORY_UPDATE, {
                    #     "status": "save_error",
                    #     "error": str(save_error)
                    # })
            
            # === PHASE 12: Send Final Complete Response ===
            # Match the non-streaming response structure exactly
            # final_response = {
            #     "analysis": {
            #         "tickers": tickers,
            #         "agents": valid_agents,
            #         "model": f"{provider} - {model_name}",
            #         "timestamp": datetime.now().isoformat()
            #     },
            #     "agent_analysis": agent_analysis,
            #     "risk_analysis": risk_analysis,
            #     "trading_decisions": trading_decisions,
            #     "portfolio_summary": portfolio_summary,
            #     "portfolio_strategy": portfolio_strategy,
            #     "metadata": {
            #         "session_id": session_id,
            #         "initial_cash": initial_cash,
            #         "margin_requirement": margin_requirement,
            #         "memory_stats": memory_stats
            #     }
            # }
            
            # yield self._format_sse_event(StreamEventType.FINAL, final_response)
            
            # Send completion marker
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming analysis: {str(e)}", exc_info=True)
            yield self._format_sse_event(StreamEventType.ERROR, {
                "error": str(e),
                "error_type": type(e).__name__
            })
            yield "data: [DONE]\n\n"
        
        finally:
            progress.stop()
    
    def _format_sse_event(self, event_type: StreamEventType, data: Dict[str, Any]) -> str:
        """Format data as Server-Sent Event"""
        event_data = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        return f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
        # return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    def _build_response_text(
        self,
        agent_analysis: Dict,
        risk_analysis: Optional[Dict],
        trading_decisions: Dict,
        portfolio_strategy: str
    ) -> str:
        """Build comprehensive response text for memory storage"""
        
        parts = []
        
        # Add agent analysis summary
        if agent_analysis:
            parts.append("## Agent Analysis Summary\n")
            for agent, signals in agent_analysis.items():
                agent_display = ANALYST_CONFIG.get(agent, {}).get("display_name", agent)
                parts.append(f"\n**{agent_display}:**")
                for ticker, signal in signals.items():
                    parts.append(f"- {ticker}: {signal.get('signal')} (Confidence: {signal.get('confidence', 0)}%)")
                    if signal.get('reasoning'):
                        parts.append(f"  Reasoning: {signal.get('reasoning')[:200]}...")
        
        # Add risk analysis if present
        if risk_analysis:
            parts.append("\n## Risk Assessment\n")
            parts.append(str(risk_analysis)[:500] + "...")
        
        # Add trading decisions
        if trading_decisions:
            parts.append("\n## Trading Decisions\n")
            for ticker, decision in trading_decisions.items():
                parts.append(f"- {ticker}: {decision['action'].upper()} {decision['quantity']} shares (Confidence: {decision['confidence']}%)")
        
        # Add portfolio strategy
        parts.append(f"\n## Portfolio Strategy\n{portfolio_strategy}")
        
        return "\n".join(parts)
    
    def _generate_strategy(
        self, 
        decisions: Dict[str, Dict], 
        margin_req: float,
        agents_used: List[str]
    ) -> str:
        """Generate strategy text based on decisions and agents used"""
        
        if not decisions:
            return "No trading decisions generated."
        
        # Count actions
        actions = {}
        for decision in decisions.values():
            action = decision.get("action", "hold")
            actions[action] = actions.get(action, 0) + 1
        
        # Build strategy
        parts = []
        
        # Mention agents used
        agent_names = [ANALYST_CONFIG.get(a, {}).get("display_name", a) for a in agents_used]
        parts.append(f"Analysis performed by: {', '.join(agent_names)}.")
        
        # Action summary
        if actions:
            action_summary = ", ".join([f"{count} {action}" for action, count in actions.items()])
            parts.append(f"Actions: {action_summary}.")
        
        # Margin info
        if margin_req > 0:
            parts.append(f"Margin available for short positions (requirement: {margin_req:.2f}).")
        
        parts.append("Monitor positions and adjust based on market conditions.")
        
        return " ".join(parts)


# Initialize streaming service
streaming_service = StreamingMultiAgentService()


# ==================== EXAMPLE USAGE ====================

def create_streaming_router(base_router: APIRouter, api_key_auth: APIKeyAuth):
    
    @base_router.post("/analyze/stream")
    async def analyze_multi_agent_stream(
        request: Request,
        tickers: List[str],
        agents: List[str],
        model_name: str = "gpt-4.1-nano",
        provider: str = "openai",
        session_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        initial_cash: float = 100000.0,
        margin_requirement: float = 0.0,
        show_reasoning: bool = True,
        api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
    ):
        """
        Stream multi-agent trading analysis with real-time progress updates.
        
        This endpoint streams events as each agent completes their analysis:
        
        Event Types:
        - agent_start: Agent begins analysis
        - agent_complete: Agent finishes with signals  
        - risk_complete: Risk management completed
        - portfolio_complete: Portfolio decisions ready
        - final: Complete response with all data
        - error: Error occurred
        """
        try:
            # Get user_id from request state
            user_id = getattr(request.state, "user_id", None)
            
            # Create async generator
            async def event_generator():
                async for event in streaming_service.stream_analysis(
                    tickers=tickers,
                    agents=agents,
                    model_name=model_name,
                    provider=provider,
                    session_id=session_id,
                    collection_name=collection_name,
                    initial_cash=initial_cash,
                    margin_requirement=margin_requirement,
                    show_reasoning=show_reasoning,
                    user_id=user_id
                ):
                    yield event
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
            
        except Exception as e:
            logger.error(f"Error setting up stream: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    @base_router.get("/analyze/{tickers}/stream")
    async def analyze_quick_stream(
        request: Request,
        tickers: str,
        agents: str = Query(..., description="Comma-separated agent names"),
        model_name: str = Query("gpt-4.1-nano", description="LLM model name"),
        provider: str = Query("openai", description="LLM provider"),
        session_id: Optional[str] = Query(None, description="Session ID"),
        collection_name: Optional[str] = Query(None, description="Collection name"),
        initial_cash: float = Query(100000.0, description="Initial cash"),
        margin_requirement: float = Query(0.0, description="Margin requirement"),
        api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
    ):
        """
        Quick GET endpoint for streaming multi-agent analysis.
        
        Examples:
        - Single agent: /analyze/AAPL/stream?agents=warren_buffett
        - Multiple agents: /analyze/AAPL,MSFT/stream?agents=warren_buffett,technical_analyst
        - All analysts: /analyze/NVDA/stream?agents=ALL
        """
        try:
            # Get user_id from request state
            user_id = getattr(request.state, "user_id", None)
            
            # Parse tickers
            ticker_list = [t.strip().upper() for t in tickers.split(",")]
            
            # Parse agents
            if agents.upper() == "ALL":
                agent_list = list(ANALYST_CONFIG.keys())
            else:
                agent_list = [a.strip().lower() for a in agents.split(",")]
            
            # Create async generator
            async def event_generator():
                async for event in streaming_service.stream_analysis(
                    tickers=ticker_list,
                    agents=agent_list,
                    model_name=model_name,
                    provider=provider,
                    session_id=session_id,
                    collection_name=collection_name,
                    initial_cash=initial_cash,
                    margin_requirement=margin_requirement,
                    show_reasoning=True,
                    user_id=user_id
                ):
                    yield event
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
            
        except Exception as e:
            logger.error(f"Error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    return base_router