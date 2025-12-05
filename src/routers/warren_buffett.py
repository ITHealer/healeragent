from fastapi import APIRouter, HTTPException, Query
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

# Import hedge fund components  
from src.hedge_fund.agents.warren_buffett import warren_buffett_agent
from src.hedge_fund.agents.portfolio_manager import portfolio_management_agent
from src.hedge_fund.agents.risk_manager import risk_management_agent
from src.hedge_fund.graph.state import AgentState
from src.hedge_fund.utils.progress import progress
from src.hedge_fund.llm.models import normalize_provider_name

# Import services
from src.utils.config import settings
from src.models.equity import APIResponse, APIResponseData

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/api/warren-buffett")

# Enums
class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"

# Response Models
class AgentAnalysis(BaseModel):
    signal: str
    confidence: float
    reasoning: str

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

class WarrenBuffettResponse(BaseModel):
    # Session info
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Analysis results (matching terminal output structure)
    agent_analysis: Dict[str, Dict[str, AgentAnalysis]]  # {"warren_buffett": {"AAPL": {...}}}
    
    # Trading decisions
    trading_decision: Dict[str, TradingDecision]  # {"AAPL": {...}}
    
    # Portfolio summary
    portfolio_summary: List[PortfolioSummaryItem]
    portfolio_strategy: str
    
    # Metadata
    model_used: str
    provider_used: str
    tickers_analyzed: List[str]

# Service class
class WarrenBuffettService:
    
    def create_full_workflow(self) -> StateGraph:
        """Create workflow with Warren Buffett + Risk + Portfolio Manager"""
        workflow = StateGraph(AgentState)
        
        def start(state: AgentState):
            return state
        
        workflow.add_node("start_node", start)
        workflow.add_node("warren_buffett_agent", warren_buffett_agent)
        workflow.add_node("risk_management_agent", risk_management_agent)
        workflow.add_node("portfolio_manager", portfolio_management_agent)
        
        # Connect nodes
        workflow.add_edge("start_node", "warren_buffett_agent")
        workflow.add_edge("warren_buffett_agent", "risk_management_agent")
        workflow.add_edge("risk_management_agent", "portfolio_manager")
        workflow.add_edge("portfolio_manager", END)
        
        workflow.set_entry_point("start_node")
        return workflow
    
    async def analyze_tickers(
        self,
        tickers: List[str],
        model_name: str = "gpt-5-nano-2025-08-07",
        provider: str = "openai",
        session_id: Optional[str] = None,
        collection_name: Optional[str] = None,
        initial_cash: float = 100000.0,
        margin_requirement: float = 0.0
    ) -> WarrenBuffettResponse:
        """Run full analysis matching terminal output"""
        
        # Set dates (3 months back like terminal)
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - relativedelta(months=3)).strftime("%Y-%m-%d")
        
        # Initialize progress
        progress.start()
        
        try:
            # Create portfolio structure (matching terminal)
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
            
            # Create initial state (matching terminal)
            initial_state = {
                "messages": [
                    HumanMessage(
                        content="Make trading decisions based on the provided data."
                    )
                ],
                "data": {
                    "tickers": tickers,
                    "portfolio": portfolio,
                    "start_date": start_date,
                    "end_date": end_date,
                    "analyst_signals": {}
                },
                "metadata": {
                    "show_reasoning": True,  # Always show reasoning like terminal
                    "model_name": model_name,
                    "model_provider": normalize_provider_name(provider),
                    "session_id": session_id,
                    "collection_name": collection_name
                }
            }
            
            # Create and run workflow
            logger.info(f"Running Warren Buffett analysis for {tickers}")
            workflow = self.create_full_workflow()
            app = workflow.compile()
            
            # Run the workflow
            final_state = app.invoke(initial_state)
            
            # Extract results (matching terminal format)
            
            # 1. Agent Analysis (Warren Buffett signals)
            agent_analysis = {}
            warren_signals = final_state["data"]["analyst_signals"].get("warren_buffett_agent", {})
            if warren_signals:
                agent_analysis["warren_buffett"] = {}
                for ticker, signal_data in warren_signals.items():
                    agent_analysis["warren_buffett"][ticker] = AgentAnalysis(
                        signal=signal_data.get("signal", "neutral"),
                        confidence=signal_data.get("confidence", 0.0),
                        reasoning=signal_data.get("reasoning", "")
                    )
            
            # 2. Trading Decisions (from portfolio manager)
            trading_decisions = {}
            portfolio_summary = []
            
            # Parse last message from portfolio manager
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
                                
                                # Add to portfolio summary
                                portfolio_summary.append(PortfolioSummaryItem(
                                    ticker=ticker,
                                    action=decision.get("action", "hold"),
                                    quantity=decision.get("quantity", 0),
                                    confidence=decision.get("confidence", 0.0)
                                ))
                        except json.JSONDecodeError:
                            logger.warning("Could not parse portfolio decisions")
            
            # 3. Generate portfolio strategy
            portfolio_strategy = self._generate_strategy_text(trading_decisions, margin_requirement)
            
            # Build response
            response = WarrenBuffettResponse(
                session_id=session_id,
                timestamp=datetime.now(),
                agent_analysis=agent_analysis,
                trading_decision=trading_decisions,
                portfolio_summary=portfolio_summary,
                portfolio_strategy=portfolio_strategy,
                model_used=model_name,
                provider_used=provider,
                tickers_analyzed=tickers
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            progress.stop()
    
    def _generate_strategy_text(self, decisions: Dict[str, TradingDecision], margin_req: float) -> str:
        """Generate strategy text matching terminal output"""
        
        # Count actions
        bullish = sum(1 for d in decisions.values() if d.action in ["buy", "long"])
        bearish = sum(1 for d in decisions.values() if d.action in ["short", "sell"])
        neutral = sum(1 for d in decisions.values() if d.action == "hold")
        
        # Calculate average confidence
        avg_confidence = 0
        if decisions:
            confidences = [d.confidence for d in decisions.values()]
            avg_confidence = sum(confidences) / len(confidences)
        
        # Build strategy text (matching terminal format)
        strategy_parts = []
        
        # Overall signal
        if avg_confidence > 90:
            strategy_parts.append(f"High-confidence signal ({avg_confidence:.0f}%).")
        else:
            strategy_parts.append(f"Moderate-confidence signal ({avg_confidence:.0f}%).")
        
        # Position summary
        if bearish > 0:
            strategy_parts.append(f"Bearish on {bearish} position(s).")
        if bullish > 0:
            strategy_parts.append(f"Bullish on {bullish} position(s).")
        if neutral > 0:
            strategy_parts.append(f"Neutral on {neutral} position(s).")
        
        # Margin info
        if margin_req > 0:
            strategy_parts.append(f"Given the input margin_requirement = {margin_req:.2f}, shorting is allowed.")
        else:
            strategy_parts.append("No margin available for short positions.")
        
        # Risk management
        strategy_parts.append("Monitor closely and manage risk (stops/size reductions) if price moves against the positions.")
        
        return " ".join(strategy_parts)

# Initialize service
warren_buffett_service = WarrenBuffettService()

# API Endpoint - Single endpoint matching terminal behavior
@router.get("/analyze/{tickers}")
async def analyze_tickers(
    tickers: str,  # Can be single ticker "AAPL" or comma-separated "AAPL,MSFT,NVDA"
    model_name: str = Query("gpt-5-nano-2025-08-07", description="LLM model name"),
    provider: str = Query("openai", description="LLM provider (openai, anthropic, groq, ollama)"),
    session_id: Optional[str] = Query(None, description="Session ID for tracking"),
    collection_name: Optional[str] = Query(None, description="Collection name for documents"),
    initial_cash: float = Query(100000.0, description="Initial portfolio cash"),
    margin_requirement: float = Query(0.0, description="Margin requirement (0.0 = no shorting)"),
    temperature: float = Query(0.7, description="LLM temperature for consistency (0.0-1.0)"),
    seed: Optional[int] = Query(None, description="Random seed for reproducible results")
):
    """
    Analyze stocks using Warren Buffett principles with full workflow.
    
    This endpoint mimics the terminal behavior:
    - Runs Warren Buffett agent
    - Runs Risk Management agent  
    - Runs Portfolio Manager agent
    - Returns complete analysis and trading decisions
    
    Examples:
    - Single ticker: /analyze/AAPL
    - Multiple tickers: /analyze/AAPL,MSFT,NVDA
    """
    try:
        # Parse tickers (support both single and comma-separated)
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        
        # Validate tickers
        if not ticker_list:
            raise HTTPException(status_code=400, detail="Please provide at least one ticker")
        
        # Log the analysis request
        logger.info(f"Analyzing tickers: {ticker_list} with model: {model_name}/{provider}")
        
        # Run analysis
        result = await warren_buffett_service.analyze_tickers(
            tickers=ticker_list,
            model_name=model_name,
            provider=provider,
            session_id=session_id,
            collection_name=collection_name,
            initial_cash=initial_cash,
            margin_requirement=margin_requirement
        )
        
        # Format response to match terminal output structure
        formatted_response = {
            "analysis": {
                "tickers": result.tickers_analyzed,
                "model": f"{result.provider_used} - {result.model_used}",
                "timestamp": result.timestamp.isoformat()
            },
            "agent_analysis": result.agent_analysis,
            "trading_decision": result.trading_decision,
            "portfolio_summary": [item.model_dump() for item in result.portfolio_summary],
            "portfolio_strategy": result.portfolio_strategy,
            "metadata": {
                "session_id": result.session_id,
                "initial_cash": initial_cash,
                "margin_requirement": margin_requirement
            }
        }
        
        # Return in API response wrapper
        response_data = APIResponseData(data=[formatted_response])
        return APIResponse(
            data=response_data,
            message=f"Warren Buffett analysis completed for {', '.join(ticker_list)}",
            provider_used=f"warren_buffett_{provider}",
            status="200"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Additional convenience endpoint for POST with body
@router.post("/analyze")
async def analyze_tickers_post(
    tickers: List[str],
    model_name: str = "gpt-5-nano-2025-08-07",
    provider: str = "openai",
    session_id: Optional[str] = None,
    collection_name: Optional[str] = None,
    initial_cash: float = 100000.0,
    margin_requirement: float = 0.0
):
    """
    Alternative POST endpoint for analyzing multiple tickers.
    Same functionality as GET but accepts JSON body.
    """
    # Convert to comma-separated string and call the main endpoint
    tickers_str = ",".join(tickers)
    return await analyze_tickers(
        tickers=tickers_str,
        model_name=model_name,
        provider=provider,
        session_id=session_id,
        collection_name=collection_name,
        initial_cash=initial_cash,
        margin_requirement=margin_requirement
    )