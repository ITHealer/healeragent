from fastapi import APIRouter, HTTPException, Query, Request, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import logging

# Import LangGraph components
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

# Import hedge fund components - use EXACT same imports as terminal
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
from src.routers.hedge_fund_multi_agent_streaming import create_streaming_router

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api/hedge-fund")

api_key_auth = APIKeyAuth()

# Enums
class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"

class AgentType(str, Enum):
    """All available agents from ANALYST_CONFIG"""
    ASWATH_DAMODARAN = "aswath_damodaran"
    BEN_GRAHAM = "ben_graham"
    BILL_ACKMAN = "bill_ackman"
    CATHIE_WOOD = "cathie_wood"
    CHARLIE_MUNGER = "charlie_munger"
    MICHAEL_BURRY = "michael_burry"
    PETER_LYNCH = "peter_lynch"
    PHIL_FISHER = "phil_fisher"
    STANLEY_DRUCKENMILLER = "stanley_druckenmiller"
    WARREN_BUFFETT = "warren_buffett"
    TECHNICAL_ANALYST = "technical_analyst"
    FUNDAMENTALS_ANALYST = "fundamentals_analyst"
    SENTIMENT_ANALYST = "sentiment_analyst"
    VALUATION_ANALYST = "valuation_analyst"

# Response Models
class AgentSignal(BaseModel):
    signal: str
    confidence: float
    reasoning: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None  # For technical/fundamental agents
    strategy_signals: Optional[Dict[str, Any]] = None  # For technical analyst

class TradingDecision(BaseModel):
    action: str
    quantity: int
    confidence: float
    reasoning: str

class PortfolioSummaryItem(BaseModel):
    ticker: str
    action: str
    quantity: int
    confidence: float

class MultiAgentResponse(BaseModel):
    # Session info
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Analysis results by agent and ticker
    agent_analysis: Dict[str, Dict[str, AgentSignal]]  # {"agent_name": {"ticker": signal}}
    
    # Risk analysis (if included)
    risk_analysis: Optional[Dict[str, Any]] = None
    
    # Trading decisions
    trading_decisions: Dict[str, TradingDecision]  # {"ticker": decision}
    
    # Portfolio summary
    portfolio_summary: List[PortfolioSummaryItem]
    portfolio_strategy: str
    
    # Metadata
    model_used: str
    provider_used: str
    tickers_analyzed: List[str]
    agents_used: List[str]

    memory_stats: Optional[Dict[str, Any]] = None

# Service class
class MultiAgentService:
    
    def __init__(self):
        """Initialize Multi-Agent Service with memory management"""
        # Initialize existing components
        self.chat_service = ChatService()
        
        # Initialize memory manager with default model
        self.memory_manager = MemoryManager()
        
        # Initialize LLM provider for analysis
        self.llm_provider = LLMGeneratorProvider()
        

    def create_workflow(self, selected_analysts: List[str]) -> StateGraph:
        """Create workflow with selected analysts - EXACTLY like terminal"""
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

    async def analyze_with_agents(
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
        user_id: Optional[str] = None  # Add user_id parameter
    ) -> MultiAgentResponse:
        """Run analysis with selected agents with memory integration"""
        
        # Validate agents
        valid_agents = []
        for agent in agents:
            if agent in ANALYST_CONFIG or agent in get_analyst_nodes():
                valid_agents.append(agent)
            else:
                logger.warning(f"Invalid agent: {agent}")
        
        if not valid_agents:
            raise HTTPException(status_code=400, detail="No valid agents selected")
        
        # Set dates (3 months back like terminal)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - relativedelta(months=3)).strftime("%Y-%m-%d")
        
        # Initialize progress
        progress.start()
        progress.agent_status.clear()

        try:
            # 1. Get memory context if session exists
            context = ""
            memory_stats = {}
            document_references = []
            
            if session_id and user_id:
                try:
                    context, memory_stats, document_references = await self.memory_manager.get_relevant_context(
                        session_id=session_id,
                        user_id=user_id,
                        current_query=f"Analyze {', '.join(tickers)} using {', '.join(valid_agents)}",
                        llm_provider=self.llm_provider,
                        max_short_term=5,
                        max_long_term=3,
                        base_collection=collection_name
                    )
                    logger.info(f"Retrieved memory context: {memory_stats}")
                except Exception as e:
                    logger.error(f"Error getting memory context: {e}")
            
            # 2. Get chat history if exists
            chat_history = ""
            if session_id:
                try:
                    chat_history = ChatMessageHistory.string_message_chat_history(session_id)
                except Exception as e:
                    logger.error(f"Error fetching history: {e}")
            
            # 3. Build enhanced initial message with context
            initial_content = "Make trading decisions based on the provided data."
            
            if context:
                initial_content = f"{context}\n\n{initial_content}"
            
            if chat_history:
                initial_content = f"[Previous Conversation]\n{chat_history}\n\n{initial_content}"
            
            # 4. Create portfolio structure
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
            
            # 5. Create initial state with context
            initial_state = {
                "messages": [
                    HumanMessage(content=initial_content)
                ],
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
                    "user_id": user_id,  # Add user_id to metadata
                    "context": context,  # Store context in metadata
                    "document_references": document_references  # Store document references
                }
            }
            
            # 6. Save user question if session exists
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
            
            # 7. Create and run workflow
            logger.info(f"Running analysis with agents: {valid_agents} for tickers: {tickers}")
            workflow = self.create_workflow(valid_agents)
            app = workflow.compile()
            
            # Run the workflow
            final_state = app.invoke(initial_state)
            
            # 8. Extract results
            agent_analysis = {}
            
            # Process each agent's signals
            for agent_key in valid_agents:
                agent_name = f"{agent_key}_agent"
                if agent_name in final_state["data"]["analyst_signals"]:
                    agent_signals = final_state["data"]["analyst_signals"][agent_name]
                    agent_analysis[agent_key] = {}
                    
                    for ticker, signal_data in agent_signals.items():
                        # Handle different agent output formats
                        if isinstance(signal_data, dict):
                            # Special handling for agents with complex output
                            if agent_key in ["technical_analyst", "fundamentals_analyst"]:
                                reasoning_text = None
                                if agent_key == "technical_analyst" and "strategy_signals" in signal_data:
                                    strategies = signal_data.get("strategy_signals", {})
                                    summary_parts = []
                                    for strat_name, strat_data in strategies.items():
                                        if isinstance(strat_data, dict):
                                            signal = strat_data.get("signal", "neutral")
                                            conf = strat_data.get("confidence", 0)
                                            summary_parts.append(f"{strat_name}: {signal} ({conf}%)")
                                    reasoning_text = "; ".join(summary_parts) if summary_parts else "Technical analysis completed"
                                
                                elif agent_key == "fundamentals_analyst":
                                    reasoning_text = str(signal_data.get("reasoning", "Fundamental analysis completed"))
                                    if isinstance(signal_data.get("reasoning"), dict):
                                        reasoning_dict = signal_data.get("reasoning", {})
                                        parts = []
                                        for key, value in reasoning_dict.items():
                                            if isinstance(value, dict):
                                                parts.append(f"{key}: {value.get('signal', 'N/A')}")
                                            else:
                                                parts.append(f"{key}: {value}")
                                        reasoning_text = "; ".join(parts)
                                
                                agent_signal = AgentSignal(
                                    signal=signal_data.get("signal", "neutral"),
                                    confidence=signal_data.get("confidence", 0.0),
                                    reasoning=reasoning_text,
                                    metrics=signal_data.get("metrics"),
                                    strategy_signals=signal_data.get("strategy_signals")
                                )
                            else:
                                # Standard agent format
                                agent_signal = AgentSignal(
                                    signal=signal_data.get("signal", "neutral"),
                                    confidence=signal_data.get("confidence", 0.0),
                                    reasoning=signal_data.get("reasoning") if isinstance(signal_data.get("reasoning"), str) else str(signal_data.get("reasoning", "")),
                                    metrics=signal_data.get("metrics"),
                                    strategy_signals=signal_data.get("strategy_signals")
                                )
                            agent_analysis[agent_key][ticker] = agent_signal
            
            # Extract risk analysis
            risk_analysis = final_state["data"]["analyst_signals"].get("risk_management_agent")
            
            # Extract trading decisions
            trading_decisions = {}
            portfolio_summary = []
            
            # Parse portfolio manager output
            if final_state["messages"]:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content') and hasattr(last_message, 'name'):
                    if last_message.name == "portfolio_manager":
                        try:
                            decisions_data = json.loads(last_message.content)
                            for ticker, decision in decisions_data.items():
                                trading_decisions[ticker] = TradingDecision(
                                    action=decision.get("action", "hold"),
                                    quantity=decision.get("quantity", 0),
                                    confidence=decision.get("confidence", 0.0),
                                    reasoning=decision.get("reasoning", "")
                                )
                                
                                portfolio_summary.append(PortfolioSummaryItem(
                                    ticker=ticker,
                                    action=decision.get("action", "hold"),
                                    quantity=decision.get("quantity", 0),
                                    confidence=decision.get("confidence", 0.0)
                                ))
                        except json.JSONDecodeError:
                            logger.warning("Could not parse portfolio decisions")
            
            # Generate portfolio strategy
            portfolio_strategy = self._generate_strategy(trading_decisions, margin_requirement, valid_agents)
            
            # 9. Build assistant response text
            response_text = self._build_response_text(
                agent_analysis, risk_analysis, trading_decisions, portfolio_strategy
            )
            
            # 10. Analyze conversation importance (if session exists)
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
            
            # 11. Store conversation in memory
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
                    
                except Exception as save_error:
                    logger.error(f"Error saving to memory: {str(save_error)}")
            
            # 12. Build response with memory stats
            response = MultiAgentResponse(
                session_id=session_id,
                timestamp=datetime.now(),
                agent_analysis=agent_analysis,
                risk_analysis=risk_analysis,
                trading_decisions=trading_decisions,
                portfolio_summary=portfolio_summary,
                portfolio_strategy=portfolio_strategy,
                model_used=model_name,
                provider_used=provider,
                tickers_analyzed=tickers,
                agents_used=valid_agents,
                memory_stats=memory_stats  # Include memory stats
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in multi-agent analysis: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            progress.stop()
            
    def _build_response_text(
        self,
        agent_analysis: Dict,
        risk_analysis: Optional[Dict],
        trading_decisions: Dict,
        portfolio_strategy: str
    ) -> str:
        """Build a comprehensive response text from analysis results"""
        
        parts = []
        
        # Add agent analysis summary
        if agent_analysis:
            parts.append("## Agent Analysis Summary\n")
            for agent, signals in agent_analysis.items():
                agent_display = ANALYST_CONFIG.get(agent, {}).get("display_name", agent)
                parts.append(f"\n**{agent_display}:**")
                for ticker, signal in signals.items():
                    parts.append(f"- {ticker}: {signal.signal} (Confidence: {signal.confidence:.1f}%)")
                    if signal.reasoning:
                        parts.append(f"  Reasoning: {signal.reasoning[:200]}...")
        
        # Add risk analysis if present
        if risk_analysis:
            parts.append("\n## Risk Assessment\n")
            parts.append(str(risk_analysis)[:500] + "...")
        
        # Add trading decisions
        if trading_decisions:
            parts.append("\n## Trading Decisions\n")
            for ticker, decision in trading_decisions.items():
                parts.append(f"- {ticker}: {decision.action.upper()} {decision.quantity} shares (Confidence: {decision.confidence:.1f}%)")
        
        # Add portfolio strategy
        parts.append(f"\n## Portfolio Strategy\n{portfolio_strategy}")
        
        return "\n".join(parts)
    
    def _generate_strategy(
        self, 
        decisions: Dict[str, TradingDecision], 
        margin_req: float,
        agents_used: List[str]
    ) -> str:
        """Generate strategy text based on decisions and agents used"""
        
        if not decisions:
            return "No trading decisions generated."
        
        # Count actions
        actions = {}
        for decision in decisions.values():
            action = decision.action
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
    
    # def _generate_strategy(
    #     self, 
    #     decisions: Dict[str, TradingDecision], 
    #     margin_req: float,
    #     agents_used: List[str]
    # ) -> str:
    #     """Generate strategy text based on decisions and agents used"""
        
    #     if not decisions:
    #         return "No trading decisions generated."
        
    #     # Count actions
    #     actions = {}
    #     for decision in decisions.values():
    #         action = decision.action
    #         actions[action] = actions.get(action, 0) + 1
        
    #     # Build strategy
    #     parts = []
        
    #     # Mention agents used
    #     agent_names = [ANALYST_CONFIG.get(a, {}).get("display_name", a) for a in agents_used]
    #     parts.append(f"Analysis performed by: {', '.join(agent_names)}.")
        
    #     # Action summary
    #     if actions:
    #         action_summary = ", ".join([f"{count} {action}" for action, count in actions.items()])
    #         parts.append(f"Actions: {action_summary}.")
        
    #     # Margin info
    #     if margin_req > 0:
    #         parts.append(f"Margin available for short positions (requirement: {margin_req:.2f}).")
        
    #     parts.append("Monitor positions and adjust based on market conditions.")
        
    #     return " ".join(parts)

# Initialize service
multi_agent_service = MultiAgentService()


# API Endpoints
@router.post("/analyze")
async def analyze_multi_agent(
    request: Request,
    tickers: List[str],
    agents: List[AgentType],
    model_name: str = None,
    provider: ProviderType = ProviderType.OPENAI,
    session_id: Optional[str] = None,
    collection_name: Optional[str] = None,
    initial_cash: float = 100000.0,
    margin_requirement: float = 0.0,
    show_reasoning: bool = True,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Run trading analysis with multiple selected agents.
    
    This endpoint allows you to:
    - Select specific agents to run (e.g., Warren Buffett + Technical Analyst)
    - Analyze multiple tickers simultaneously
    - Get combined signals from all selected agents
    - Receive final trading decisions from portfolio manager
    
    Example request:
    ```json
    {
        "tickers": ["AAPL", "MSFT", "NVDA"],
        "agents": ["warren_buffett", "technical_analyst", "fundamentals_analyst"],
        "model_name": "gpt-4o",
        "provider": "openai"
    }
    ```
    """
    try:
        # Get user_id from request state
        user_id = getattr(request.state, "user_id", None)
        
        # Convert enum values to strings
        agent_keys = [agent.value for agent in agents]
        
        # Run analysis with memory integration
        result = await multi_agent_service.analyze_with_agents(
            tickers=tickers,
            agents=agent_keys,
            model_name=model_name,
            provider=provider.value,
            session_id=session_id,
            collection_name=collection_name,
            initial_cash=initial_cash,
            margin_requirement=margin_requirement,
            show_reasoning=show_reasoning,
            user_id=user_id  # Pass user_id
        )
        
        # Format response
        formatted_response = {
            "analysis": {
                "tickers": result.tickers_analyzed,
                "agents": result.agents_used,
                "model": f"{result.provider_used} - {result.model_used}",
                "timestamp": result.timestamp.isoformat()
            },
            "agent_analysis": {
                agent: {
                    ticker: signal.model_dump()
                    for ticker, signal in signals.items()
                }
                for agent, signals in result.agent_analysis.items()
            },
            "risk_analysis": result.risk_analysis,
            "trading_decisions": {
                ticker: decision.model_dump()
                for ticker, decision in result.trading_decisions.items()
            },
            "portfolio_summary": [item.model_dump() for item in result.portfolio_summary],
            "portfolio_strategy": result.portfolio_strategy,
            "metadata": {
                "session_id": result.session_id,
                "initial_cash": initial_cash,
                "margin_requirement": margin_requirement,
                "memory_stats": result.memory_stats  # Include memory stats
            }
        }

        # Return in API response wrapper
        response_data = APIResponseData(data=[formatted_response])
        return APIResponse(
            data=response_data,
            message=f"Multi-agent analysis completed for {', '.join(tickers)}",
            provider_used=f"multi_agent_{provider.value}",
            status="200"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{tickers}")
async def analyze_quick(
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
    Quick GET endpoint for multi-agent analysis.
    
    Examples:
    - Single agent: /analyze/AAPL?agents=warren_buffett
    - Multiple agents: /analyze/AAPL,MSFT?agents=warren_buffett,technical_analyst
    - All analysts: /analyze/NVDA?agents=ALL
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
        
        # Run analysis with memory
        result = await multi_agent_service.analyze_with_agents(
            tickers=ticker_list,
            agents=agent_list,
            model_name=model_name,
            provider=provider,
            session_id=session_id,
            collection_name=collection_name,
            initial_cash=initial_cash,
            margin_requirement=margin_requirement,
            show_reasoning=True,
            user_id=user_id  # Pass user_id
        )
        
        # Format compact response for GET endpoint
        formatted_response = {
            "tickers": result.tickers_analyzed,
            "agents": result.agents_used,
            "signals": {
                agent: {
                    ticker: {
                        "signal": signal.signal,
                        "confidence": signal.confidence,
                        "reasoning": signal.reasoning
                    }
                    for ticker, signal in signals.items()
                }
                for agent, signals in result.agent_analysis.items()
            },
            "decisions": {
                ticker: {
                    "action": decision.action,
                    "quantity": decision.quantity,
                    "confidence": decision.confidence,
                    "reasoning": decision.reasoning
                }
                for ticker, decision in result.trading_decisions.items()
            },
            "strategy": result.portfolio_strategy,
            "memory_stats": result.memory_stats  # Include memory stats
        }
        
        response_data = APIResponseData(data=[formatted_response])
        return APIResponse(
            data=response_data,
            message=f"Analysis completed with {len(agent_list)} agents",
            provider_used=f"multi_agent_{provider}",
            status="200"
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def list_available_agents():
    """
    List all available agents with their display names and descriptions.
    """
    agents = []
    for key, config in ANALYST_CONFIG.items():
        agents.append({
            "key": key,
            "display_name": config["display_name"],
            "order": config["order"]
        })
    
    # Sort by order
    agents.sort(key=lambda x: x["order"])
    
    return APIResponse(
        data=APIResponseData(data=agents),
        message="Available agents listed",
        provider_used="system",
        status="200"
    )


router = create_streaming_router(router, api_key_auth)