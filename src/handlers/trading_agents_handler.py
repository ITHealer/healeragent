import os
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from src.tradingagents.graph.trading_graph import TradingAgentsGraph
from src.tradingagents.default_config import DEFAULT_CONFIG

class TradingAgentsHandler(LoggerMixin):
    """Handler for TradingAgents framework integration"""
    
    def __init__(self):
        super().__init__()

        self.executor = ThreadPoolExecutor(max_workers=5)
        self._graph_cache = {}
        

    def _get_custom_config(
        self, 
        llm_provider: Optional[str] = None,
        deep_think_llm: Optional[str] = None,
        quick_think_llm: Optional[str] = None,
        max_debate_rounds: Optional[int] = None,
        online_tools: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Create custom configuration based on user inputs"""
        config = DEFAULT_CONFIG.copy()
        
        # Update config with environment variables
        config["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
        config["fmp_api_key"] = os.getenv("FMP_API_KEY", "")
        
        # Update with user preferences
        if llm_provider:
            config["llm_provider"] = llm_provider
        if deep_think_llm:
            config["deep_think_llm"] = deep_think_llm
        if quick_think_llm:
            config["quick_think_llm"] = quick_think_llm
        if max_debate_rounds is not None:
            config["max_debate_rounds"] = max_debate_rounds
        if online_tools is not None:
            config["online_tools"] = online_tools
            
        # Set project directory
        config["project_dir"] = os.path.join(os.getcwd(), "src/tradingagents")
        
        return config
    
    def _get_graph_key(self, config: Dict[str, Any], selected_analysts: List[str]) -> str:
        """Generate unique key for graph caching"""
        analyst_str = "-".join(sorted(selected_analysts))
        return f"{config['llm_provider']}_{config['deep_think_llm']}_{analyst_str}"
    
    async def analyze_ticker(
        self,
        ticker: str,
        trade_date: str,
        selected_analysts: List[str] = None,
        llm_provider: str = "openai",
        deep_think_llm: str = "gpt-4.1-nano",
        quick_think_llm: str = "gpt-4.1-nano",
        max_debate_rounds: int = 1,
        online_tools: bool = False,
        debug: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze a ticker using TradingAgents framework
        
        Args:
            ticker: Stock ticker symbol (e.g., "NVDA")
            trade_date: Trading date in YYYY-MM-DD format
            selected_analysts: List of analysts to use
            llm_provider: LLM provider (openai, anthropic, google)
            deep_think_llm: Model for deep thinking
            quick_think_llm: Model for quick thinking
            max_debate_rounds: Number of debate rounds
            online_tools: Whether to use online tools
            debug: Whether to run in debug mode
            
        Returns:
            Dict containing analysis results and trading decision
        """
        try:
            self.logger.info(f"Starting analysis for {ticker} on {trade_date}")
            
            # Default analysts
            if selected_analysts is None:
                selected_analysts = ["market", "social", "news", "fundamentals"]
            
            # Create custom config
            config = self._get_custom_config(
                llm_provider=llm_provider,
                deep_think_llm=deep_think_llm,
                quick_think_llm=quick_think_llm,
                max_debate_rounds=max_debate_rounds,
                online_tools=online_tools
            )
            
            # Check cache or create new graph
            graph_key = self._get_graph_key(config, selected_analysts)
            if graph_key not in self._graph_cache:
                self.logger.info(f"Creating new TradingAgentsGraph with key: {graph_key}")
                self._graph_cache[graph_key] = TradingAgentsGraph(
                    selected_analysts=selected_analysts,
                    debug=debug,
                    config=config
                )
            
            graph = self._graph_cache[graph_key]
            
            # Run analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            state, decision = await loop.run_in_executor(
                self.executor,
                graph.propagate,
                ticker,
                trade_date
            )
            
            # Extract key information from state
            result = {
                "ticker": ticker,
                "trade_date": trade_date,
                "decision": decision,
                "reports": {},
                "debate_history": {},
                "timestamp": datetime.now().isoformat()
            }
            
            # Extract analyst reports
            if state.get("market_report"):
                result["reports"]["market"] = state["market_report"]
            if state.get("sentiment_report"):
                result["reports"]["sentiment"] = state["sentiment_report"]
            if state.get("news_report"):
                result["reports"]["news"] = state["news_report"]
            if state.get("fundamentals_report"):
                result["reports"]["fundamentals"] = state["fundamentals_report"]
                
            # Extract debate history
            if state.get("investment_debate_state"):
                debate_state = state["investment_debate_state"]
                result["debate_history"]["bull"] = debate_state.get("bull_history", "")
                result["debate_history"]["bear"] = debate_state.get("bear_history", "")
                result["debate_history"]["judge_decision"] = debate_state.get("judge_decision", "")
                
            # Extract risk analysis
            if state.get("risk_debate_state"):
                risk_state = state["risk_debate_state"]
                result["risk_analysis"] = {
                    "risky": risk_state.get("risky_history", ""),
                    "safe": risk_state.get("safe_history", ""),
                    "neutral": risk_state.get("neutral_history", ""),
                    "final_decision": state.get("final_trade_decision", "")
                }
            
            self.logger.info(f"Analysis completed for {ticker} on {trade_date}: {decision}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {ticker}: {str(e)}", exc_info=True)
            raise
    
    async def reflect_on_decision(
        self,
        ticker: str,
        trade_date: str,
        returns_losses: float,
        selected_analysts: List[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Reflect on a trading decision with actual returns/losses
        
        Args:
            ticker: Stock ticker symbol
            trade_date: Trading date
            returns_losses: Actual returns/losses percentage
            selected_analysts: List of analysts used
            config: Custom configuration
            
        Returns:
            Dict containing reflection results
        """
        try:
            if selected_analysts is None:
                selected_analysts = ["market", "social", "news", "fundamentals"]
                
            if config is None:
                config = self._get_custom_config()
                
            graph_key = self._get_graph_key(config, selected_analysts)
            if graph_key not in self._graph_cache:
                self._graph_cache[graph_key] = TradingAgentsGraph(
                    selected_analysts=selected_analysts,
                    debug=False,
                    config=config
                )
                
            graph = self._graph_cache[graph_key]
            
            # First propagate to get state
            loop = asyncio.get_event_loop()
            state, _ = await loop.run_in_executor(
                self.executor,
                graph.propagate,
                ticker,
                trade_date
            )
            
            # Then reflect
            reflections = await loop.run_in_executor(
                self.executor,
                graph.reflect_and_remember,
                returns_losses
            )
            
            return {
                "ticker": ticker,
                "trade_date": trade_date,
                "returns_losses": returns_losses,
                "reflections": reflections,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error reflecting on {ticker}: {str(e)}", exc_info=True)
            raise
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self._graph_cache.clear()




    async def analyze_ticker_stream(
        self,
        ticker: str,
        trade_date: str,
        selected_analysts: List[str],
        llm_provider: str = "openai",
        deep_think_llm: str = "gpt-4.1-nano",
        quick_think_llm: str = "gpt-4.1-nano",
        max_debate_rounds: int = 1,
        online_tools: bool = True,
        stream_mode: str = "detailed"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream analysis updates progressively
        
        Yields updates at each stage of the analysis process
        """
        try:
            # Default analysts
            if selected_analysts is None:
                selected_analysts = ["market", "social", "news", "fundamentals"]
            
            # Create custom config
            config = self._get_custom_config(
                llm_provider=llm_provider,
                deep_think_llm=deep_think_llm,
                quick_think_llm=quick_think_llm,
                max_debate_rounds=max_debate_rounds,
                online_tools=online_tools
            )
            
            # Check cache or create new graph
            graph_key = self._get_graph_key(config, selected_analysts)
            if graph_key not in self._graph_cache:
                self.logger.info(f"Creating new TradingAgentsGraph with key: {graph_key}")
                
                # Create graph with streaming enabled
                self._graph_cache[graph_key] = TradingAgentsGraph(
                    selected_analysts=selected_analysts,
                    debug=True,  # Enable debug for streaming
                    config=config
                )
            
            graph = self._graph_cache[graph_key]
            
            # Stream through graph execution
            stage_count = 0
            async for chunk in self._stream_graph_execution(
                graph, ticker, trade_date, selected_analysts, stream_mode
            ):
                stage_count += 1
                
                if chunk:
                    yield chunk
                
                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.05)
            
        except Exception as e:
            self.logger.error(f"Error in streaming analysis: {str(e)}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "content": str(e)
            }
    
    async def _stream_graph_execution(
        self, 
        graph: TradingAgentsGraph,
        ticker: str,
        trade_date: str,
        selected_analysts: List[str],
        mode: str = "detailed"
    ) -> AsyncGenerator[Dict, None]:
        """Execute graph with streaming and yield chunks"""
        
        try:
            # Since TradingAgentsGraph.propagate is synchronous, 
            # we need to create a custom streaming approach
            
            # For now, we'll simulate streaming by breaking down the process
            # This is a simplified version - you may need to modify TradingAgentsGraph
            # to support true streaming through its internal graph.stream() method
            
            # Stage 1: Initialize
            yield {
                "type": "progress",
                "stage": "initialization",
                "content": f"Starting analysis for {ticker} on {trade_date}",
                "important": True
            }
            
            # Stage 2: Analyst reports (simulate progressive updates)
            for analyst in selected_analysts:
                yield {
                    "type": "progress",
                    "stage": f"{analyst}_analysis",
                    "content": f"Running {analyst} analysis...",
                    "important": True
                }
                await asyncio.sleep(0.5)  # Simulate processing time
            
            # Stage 3: Run actual analysis in executor
            loop = asyncio.get_event_loop()
            
            # This runs the full analysis - ideally we'd modify TradingAgentsGraph
            # to support yielding intermediate results
            state, decision = await loop.run_in_executor(
                None,  # Use default executor
                graph.propagate,
                ticker,
                trade_date
            )
            
            # Stage 4: Extract and yield results progressively
            if state.get("market_report"):
                yield {
                    "type": "result",
                    "stage": "market_report",
                    "content": state["market_report"], 
                    "important": True
                }
            
            if state.get("sentiment_report"):
                yield {
                    "type": "result",
                    "stage": "sentiment_report",
                    "content": state["sentiment_report"],
                    "important": True
                }
            
            if state.get("news_report"):
                yield {
                    "type": "result",
                    "stage": "news_report",
                    "content": state["news_report"],
                    "important": True
                }
            
            if state.get("fundamentals_report"):
                yield {
                    "type": "result",
                    "stage": "fundamentals_report",
                    "content": state["fundamentals_report"],
                    "important": True
                }
            
            # Stage 5: Debate results
            if state.get("investment_debate_state"):
                debate = state["investment_debate_state"]
                if debate.get("bull_history"):
                    yield {
                        "type": "result",
                        "stage": "bull_case",
                        "content": debate["bull_history"],
                        "important": True
                    }
                if debate.get("bear_history"):
                    yield {
                        "type": "result",
                        "stage": "bear_case",
                        "content": debate["bear_history"],
                        "important": True
                    }
            
            # Stage 6: Final decision
            yield {
                "type": "final",
                "stage": "decision",
                "content": decision,
                "data": {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "decision": decision,
                    "final_trade_decision": state.get("final_trade_decision", "")
                },
                "important": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in graph execution: {str(e)}")
            yield {
                "type": "error",
                "stage": "execution_error",
                "content": str(e)
            }
    
    def _extract_stage_info(self, chunk: Dict) -> Dict:
        """Extract meaningful information from graph chunk"""
        
        # Determine current node/stage
        stage = "processing"
        content = ""
        important = False
        
        if "market_report" in chunk:
            stage = "market_analysis"
            content = "Analyzing market indicators and technical patterns..."
            important = True
            
        elif "sentiment_report" in chunk:
            stage = "social_sentiment"
            content = "Evaluating social media sentiment and retail investor mood..."
            important = True
            
        elif "news_report" in chunk:
            stage = "news_analysis"
            content = "Processing recent news and global events..."
            important = True
            
        elif "fundamentals_report" in chunk:
            stage = "fundamentals"
            content = "Examining financial statements and insider activity..."
            important = True
            
        elif "investment_debate_state" in chunk:
            debate_state = chunk.get("investment_debate_state", {})
            if debate_state.get("bull_history"):
                stage = "bull_analysis"
                content = "Bull case: " + str(debate_state.get("current_response", ""))
                important = True
            elif debate_state.get("bear_history"):
                stage = "bear_analysis"
                content = "Bear case: " + str(debate_state.get("current_response", ""))
                important = True
                
        elif "risk_debate_state" in chunk:
            stage = "risk_assessment"
            content = "Evaluating risk levels and position sizing..."
            important = True
            
        elif "final_trade_decision" in chunk:
            stage = "final_decision"
            content = chunk.get("final_trade_decision", "")
            important = True
        
        return {
            "stage": stage,
            "content": content,
            "important": important,
            "raw_data": chunk if stage == "final_decision" else None
        }
    
    def _parse_chunk(self, chunk: Dict, stage_num: int, selected_analysts: List[str]) -> Optional[Dict]:
        """Parse chunk based on selected analysts and stage"""
        
        stage = chunk.get("stage", "")
        
        # Filter based on selected analysts
        if stage == "market_analysis" and "market" not in selected_analysts:
            return None
        elif stage == "social_sentiment" and "social" not in selected_analysts:
            return None
        elif stage == "news_analysis" and "news" not in selected_analysts:
            return None
        elif stage == "fundamentals" and "fundamentals" not in selected_analysts:
            return None
        
        # Return formatted chunk
        return {
            "type": "progress" if stage != "final_decision" else "result",
            "stage": stage,
            "stage_number": stage_num,
            "content": chunk.get("content", ""),
            "data": chunk.get("raw_data") if stage == "final_decision" else None
        }

# Singleton instance
trading_agents_handler = TradingAgentsHandler()