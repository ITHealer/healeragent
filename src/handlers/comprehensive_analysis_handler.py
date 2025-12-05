from typing import Dict, Any, List

from src.stock.crawlers.market_data_provider import MarketData
from src.handlers.technical_analysis_handler import get_technical_analysis
from src.handlers.risk_analysis_handler import RiskAnalysisHandler
from src.handlers.volume_profile_handler import VolumeProfileHandler
from src.handlers.pattern_recognition_handler import PatternRecognitionHandler
from src.handlers.relative_strength_handler import RelativeStrengthHandler
from src.utils.logger.custom_logging import LoggerMixin


class ComprehensiveAnalysisHandler(LoggerMixin):
    
    def __init__(self):
        super().__init__()
        self.market_data = MarketData()
        self.risk_handler = RiskAnalysisHandler(self.market_data)
        self.volume_handler = VolumeProfileHandler()
        self.pattern_handler = PatternRecognitionHandler(self.market_data)
        self.rs_handler = RelativeStrengthHandler(self.market_data)
        
    async def perform_comprehensive_analysis(
        self,
        symbol: str,
        lookback_days: int = 252,
        analyses: List[str] = None
    ) -> Dict[str, Any]:
        """
        Summary analysis
        
        Args:
            symbol: Stock code
            lookback_days: Number of historical days to retrieve
            analyses: List of analyses to perform

        Returns:
            Dict[str, Any]: Summary analysis results
        """
        try:
            # Default perform all analysis if not specified
            if analyses is None:
                analyses = ["technical", "risk", "volume", "pattern", "relative_strength"]
            
            # Fetch data only once
            self.logger.info(f"Fetching data for {symbol} with {lookback_days} dates")
            df = await self.market_data.get_historical_data_lookback_ver2(
                ticker=symbol,
                lookback_days=lookback_days
            )
            
            # Data format for analysis
            stock_data = df.reset_index().to_dict(orient="records")
            
            results = {
                "symbol": symbol,
                "lookback_days": lookback_days
            }
            
            # Perform the analysis
            if "technical" in analyses:
                self.logger.info(f"Technical analysis for {symbol}")
                technical_results = get_technical_analysis(symbol, stock_data)
                results["technical_analysis"] = {
                    "message": f"Technical analysis completed for {symbol}",
                    "data": technical_results
                }
            
            if "risk" in analyses:
                self.logger.info(f"Analyzing risks for {symbol}")
                risk_results = await self.risk_handler.suggest_stop_loss_levels(
                    symbol=symbol,
                    lookback_days=min(60, lookback_days),
                    df=df 
                )
                results["risk_analysis"] = {
                    "message": f"Risk analysis completed for {symbol}",
                    "data": risk_results
                }
            
            if "volume" in analyses:
                self.logger.info(f"Analyzing volume for {symbol}")
                volume_results = await self.volume_handler.get_volume_profile(
                    symbol=symbol,
                    lookback_days=min(60, lookback_days),
                    num_bins=10,
                    df=df
                )
                results["volume_profile"] = {
                    "message": f"Volume profile analysis completed for {symbol}",
                    "data": volume_results
                }
            
            if "pattern" in analyses:
                self.logger.info(f"Recognizing pattern for {symbol}")
                pattern_results = await self.pattern_handler.analyze_patterns(
                    symbol=symbol,
                    lookback_days=min(90, lookback_days),
                    df=df
                )
                
                results["pattern_recognition"] = {
                    "message": f"Pattern analysis completed for {symbol}",
                    "data": pattern_results
                }
                
                if "summary" not in pattern_results:
                    pattern_results["summary"] = self.pattern_handler.format_pattern_results(pattern_results)
            
            if "relative_strength" in analyses:
                self.logger.info(f"Calculating relative strength for {symbol}")
                rs_results = await self.rs_handler.get_relative_strength(
                    symbol=symbol,
                    benchmark="SPY" 
                )
                results["relative_strength"] = {
                    "message": f"Relative strength analysis completed for {symbol}",
                    "data": rs_results
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Meta-analysis error for {symbol}: {str(e)}")
            raise Exception(f"Analysis error {symbol}: {str(e)}")

    
    def extract_summaries(self, analysis_data):
        summaries = {
            "symbol": analysis_data.get("symbol", "Unknown"),
            "technical_analysis": (
                analysis_data.get("technical_analysis", {})
                .get("data", {})
                .get("analysis_summary", "No technical analysis available")
            ),
            "risk_analysis": (
                analysis_data.get("risk_analysis", {})
                .get("data", {})
                .get("suggested_stop_levels", "No risk analysis available")
            ),
            "volume_profile": (
                analysis_data.get("volume_profile", {})
                .get("data", {})
                .get("summary", "No volume profile available")
            ),
            "pattern_recognition": (
                analysis_data.get("pattern_recognition", {})
                .get("data", {})
                .get("summary", "No pattern recognition available")
            ),
            "relative_strength": (
                analysis_data.get("relative_strength", {})
                .get("data", {})
                .get("relative_strength_summary", "No relative strength analysis available")
            )
        }
        
        key_metrics = {}
        try:
            tech_data = analysis_data.get("technical_analysis", {}).get("data", {})
            risk_data = analysis_data.get("risk_analysis", {}).get("data", {})
            
            key_metrics = {
                "price": tech_data.get("latest_price", 0),
                "rsi": tech_data.get("momentum", {}).get("rsi", 0),
                "macd_bullish": tech_data.get("momentum", {}).get("macd_bullish", False),
                "moving_averages": {
                    "sma_20": risk_data.get("stop_levels", {}).get("sma_20", 0),
                    "sma_50": risk_data.get("stop_levels", {}).get("sma_50", 0)
                },
                "stop_levels": {
                    "atr_2x": risk_data.get("stop_levels", {}).get("atr_2x", 0),
                    "percent_5": risk_data.get("stop_levels", {}).get("percent_5", 0),
                    "recent_swing": risk_data.get("stop_levels", {}).get("recent_swing", 0)
                }
            }
        except Exception as e:
            print(f"Error extracting metrics: {str(e)}")
            key_metrics = {
                "price": 0,
                "rsi": 0,
                "macd_bullish": False,
                "moving_averages": {"sma_20": 0, "sma_50": 0},
                "stop_levels": {"atr_2x": 0, "percent_5": 0, "recent_swing": 0}
            }
        
        return {"summaries": summaries, "key_metrics": key_metrics}
    

    def create_prompt_from_extracted_data(extracted_data):        
        summaries = extracted_data["summaries"]
        key_metrics = extracted_data["key_metrics"]
        
        prompt = f"""
        You are a technical stock analyst expert. Analyze the provided stock data and give a clear investment recommendation.
        
        The data contains:
        1. Technical analysis summary: {summaries["technical_analysis"]}
        2. Risk analysis summary: {summaries["risk_analysis"]}
        3. Volume profile summary: {summaries["volume_profile"]}
        4. Pattern recognition summary: {summaries["pattern_recognition"]}
        5. Relative strength summary: {summaries["relative_strength"]}
        
        Key metrics:
        - Current price: ${key_metrics["price"]}
        - RSI: {key_metrics["rsi"]}
        - MACD bullish: {key_metrics["macd_bullish"]}
        - SMA 20: ${key_metrics["moving_averages"]["sma_20"]}
        - SMA 50: ${key_metrics["moving_averages"]["sma_50"]}
        
        Provide a comprehensive analysis with the following format:
        **1. Potential Strengths:**
        * [List with SPECIFIC VALUES from the data]
        
        **2. Risks and Weaknesses:**
        * [List with SPECIFIC VALUES from the data]
        
        **3. Recommendation:**
        * [Clear decision: BUY/SELL/WAIT]
        * [Specific price levels for action]
        
        **Summary:**
        [Brief conclusion with definitive stance]
        
        IMPORTANT: Give a STRONG, CLEAR recommendation and use SPECIFIC VALUES from the data.
        """
        
        return prompt
    