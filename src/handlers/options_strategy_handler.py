# import asyncio
# import pandas as pd
# import numpy as np
# from datetime import datetime, timedelta
# from typing import Dict, Any, Optional, List, AsyncGenerator

# from src.utils.logger.custom_logging import LoggerMixin
# from src.providers.provider_factory import ModelProviderFactory, ProviderType
# from src.stock.crawlers.market_data_provider import MarketData
# from src.helpers.options_strategy_llm_helper import OptionsStrategyLLMHelper


# class OptionsStrategyHandler(LoggerMixin):
#     """Handler for options strategy analysis and market sentiment"""
    
#     def __init__(self):
#         super().__init__()
#         self.market_data_fetcher = MarketData()
#         self.llm_helper = OptionsStrategyLLMHelper()
        
#     async def analyze_strategy(
#         self,
#         symbol: str,
#         model_name: str,
#         provider_type: str,
#         api_key: Optional[str] = None
#     ) -> Dict[str, Any]:
#         """
#         Analyze a stock and recommend options strategies.
        
#         Args:
#             symbol: Stock symbol to analyze
#             model_name: LLM model name
#             provider_type: Provider type (ollama, openai, gemini)
#             api_key: API key for the provider
            
#         Returns:
#             Dictionary containing analysis results and strategy recommendations
#         """
#         try:
#             self.logger.info(f"Starting options strategy analysis for {symbol}")
            
#             # Fetch market data
#             market_data = await self._fetch_and_prepare_data(symbol, lookback_days=730)
            
#             # Calculate technical indicators
#             technical_metrics = self._calculate_technical_indicators(market_data)
            
#             # Determine potential strategies
#             initial_strategies = self._determine_strategies(technical_metrics)
            
#             # Get LLM-enhanced recommendations
#             llm_recommendations = await self.llm_helper.generate_strategy_recommendations(
#                 symbol=symbol,
#                 market_data=market_data,
#                 technical_metrics=technical_metrics,
#                 initial_strategies=initial_strategies,
#                 model_name=model_name,
#                 provider_type=provider_type,
#                 api_key=api_key
#             )
#             # return {
#             #     # "symbol": symbol,
#             #     # "analysis_timestamp": datetime.now().isoformat(),
#             #     # "market_metrics": technical_metrics,
#             #     "interpretation": llm_recommendations,
#             #     # "data_period": {
#             #     #     "start": market_data.index[0].strftime("%Y-%m-%d") if not market_data.empty else None,
#             #     #     "end": market_data.index[-1].strftime("%Y-%m-%d") if not market_data.empty else None
#             #     # }
#             # }
#             return llm_recommendations
#         except Exception as e:
#             self.logger.error(f"Error in options strategy analysis: {str(e)}")
#             raise


#     async def stream_strategy_analysis(
#         self,
#         symbol: str,
#         model_name: str,
#         provider_type: str,
#         api_key: Optional[str] = None,
#         language_instruction: Optional[str] = None
#     ) -> AsyncGenerator[Dict[str, Any], None]:
#         """
#         Stream options strategy analysis results.
        
#         Yields:
#             Dictionary chunks containing analysis data and streaming recommendations
#         """
#         try:
#             self.logger.info(f"Starting streaming options strategy analysis for {symbol}")
            
#             # Fetch market data
#             market_data = await self._fetch_and_prepare_data(symbol, lookback_days=730)
            
#             # Calculate technical indicators
#             technical_metrics = self._calculate_technical_indicators(market_data)
            
#             # Log technical metrics for debugging
#             self.logger.info(f"Technical metrics keys: {list(technical_metrics.keys())}")
            
#             # Determine potential strategies
#             initial_strategies = self._determine_strategies(technical_metrics)
            
#             # Build technical indicators with safe access
#             technical_indicators_data = {}
            
#             # Map the correct keys from technical_metrics
#             # Adjust these mappings based on what your _calculate_technical_indicators actually returns
#             key_mappings = {
#                 "rsi": "rsi_14",  # or whatever key your function uses
#                 "macd_signal": "macd_signal",
#                 "volatility": "volatility_20",  # or "annualized_volatility"
#                 "trend": "trend",
#                 "support": "support_level",
#                 "resistance": "resistance_level"
#             }
            
#             for display_key, metric_key in key_mappings.items():
#                 # Try multiple possible keys
#                 value = technical_metrics.get(metric_key)
#                 if value is None and display_key in technical_metrics:
#                     value = technical_metrics.get(display_key)
                
#                 technical_indicators_data[display_key] = value if value is not None else "N/A"
            
#             # Stream LLM recommendations
#             full_response = []
#             async for chunk in self.llm_helper.stream_strategy_recommendations(
#                 symbol=symbol,
#                 market_data=market_data,
#                 technical_metrics=technical_metrics,
#                 initial_strategies=initial_strategies,
#                 model_name=model_name,
#                 provider_type=provider_type,
#                 api_key=api_key,
#                 language_instruction=language_instruction
#             ):
#                 if chunk:
#                     full_response.append(chunk)
#                     yield {
#                         "type": "recommendation_chunk",
#                         "content": chunk
#                     }
            
#         except Exception as e:
#             self.logger.error(f"Error in streaming strategy analysis: {str(e)}", exc_info=True)
#             yield {
#                 "type": "error",
#                 "error": str(e)
#             }

    
#     async def analyze_sentiment(
#         self,
#         symbol: str,
#         use_llm: bool = False,
#         model_name: str = None,
#         provider_type: str = None,
#         api_key: Optional[str] = None,
#         # lookback_days: int = 20
#     ) -> Dict[str, Any]:
#         """
#         Analyze market sentiment (bullish/bearish) for a symbol.
        
#         Args:
#             symbol: Stock symbol to analyze
#             model_name: LLM model name
#             provider_type: Provider type
#             api_key: API key for the provider
#             lookback_days: Number of days to analyze
            
#         Returns:
#             Dictionary containing sentiment analysis
#         """
#         try:
#             self.logger.info(f"Starting sentiment analysis for {symbol}")
            
#             # Config
#             lookback_days = 20

#             # Fetch recent market data
#             market_data = await self._fetch_and_prepare_data(symbol, lookback_days=lookback_days)
            
#             # Calculate sentiment indicators
#             sentiment_metrics = self._calculate_sentiment_indicators(market_data)

#             # Perform technical sentiment analysis
#             technical_sentiment = self._analyze_technical_sentiment(sentiment_metrics)
            
#             # If LLM is not requested, return technical analysis only
#             if not use_llm:
#                 return {
#                     "symbol": symbol,
#                     # "analysis_timestamp": datetime.now().isoformat(),
#                     "sentiment": technical_sentiment["sentiment"],
#                     # "confidence": technical_sentiment["confidence"],
#                     # "key_factors": technical_sentiment["key_factors"],
#                     # "technical_signals": sentiment_metrics,
#                     "recommendation": technical_sentiment["recommendation"],
#                     # "analysis_method": "technical_only",
#                     "analysis": None
#                 }
            
#             # Get LLM sentiment analysis
#             sentiment_analysis = await self.llm_helper.analyze_market_sentiment(
#                 symbol=symbol,
#                 market_data=market_data,
#                 sentiment_metrics=sentiment_metrics,
#                 model_name=model_name,
#                 provider_type=provider_type,
#                 api_key=api_key
#             )
            
#             return {
#                 "symbol": symbol,
#                 # "analysis_timestamp": datetime.now().isoformat(),
#                 "sentiment": sentiment_analysis["sentiment"],
#                 # "confidence": sentiment_analysis["confidence"],
#                 # "key_factors": sentiment_analysis["key_factors"],
#                 # "technical_signals": sentiment_metrics,
#                 "recommendation": sentiment_analysis["recommendation"],
#                 "analysis": sentiment_analysis["detailed_analysis"]
#             }
            
#         except Exception as e:
#             self.logger.error(f"Error in sentiment analysis: {str(e)}")
#             raise
    
#     async def _fetch_and_prepare_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
#         """Fetch and prepare market data from FMP"""
#         try:
#             # Use the existing market data fetcher
#             df = await self.market_data_fetcher.get_historical_data_lookback_ver2(symbol, lookback_days)
            
#             if df.empty:
#                 raise ValueError(f"No data available for symbol {symbol}")
            
#             # Ensure we have the required columns
#             required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
#             missing_columns = [col for col in required_columns if col not in df.columns]
#             if missing_columns:
#                 raise ValueError(f"Missing required columns: {missing_columns}")
            
#             return df
            
#         except Exception as e:
#             self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
#             raise
    
#     def _calculate_technical_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
#         """Calculate comprehensive technical indicators"""
#         try:
#             close = df['Close']
#             volume = df['Volume']
            
#             # Price metrics
#             current_price = close.iloc[-1]
            
#             # Moving averages
#             sma_20 = close.rolling(window=20).mean().iloc[-1]
#             sma_50 = close.rolling(window=50).mean().iloc[-1]
#             sma_200 = close.rolling(window=200).mean().iloc[-1]
            
#             # RSI
#             rsi_14 = self._calculate_rsi(close, period=14)
            
#             # Historical Volatility
#             historical_volatility = self._calculate_historical_volatility(close, lookback=20)
            
#             # Support and Resistance
#             support, resistance = self._calculate_support_resistance(df, window=20)
            
#             # Volume metrics
#             volume_avg_20 = volume.rolling(window=20).mean().iloc[-1]
#             volume_ratio = volume.iloc[-1] / volume_avg_20 if volume_avg_20 > 0 else 1
            
#             # Price change metrics
#             price_change_20d = (current_price - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0
            
#             return {
#                 "current_price": round(current_price, 2),
#                 "sma_20": round(sma_20, 2),
#                 "sma_50": round(sma_50, 2),
#                 "sma_200": round(sma_200, 2),
#                 "rsi_14": round(rsi_14, 2),
#                 "historical_volatility": round(historical_volatility, 4),
#                 "support_level": round(support, 2),
#                 "resistance_level": round(resistance, 2),
#                 "volume_avg_20": round(volume_avg_20, 0),
#                 "volume_ratio": round(volume_ratio, 2),
#                 "price_change_20d_pct": round(price_change_20d * 100, 2)
#             }
            
#         except Exception as e:
#             self.logger.error(f"Error calculating technical indicators: {str(e)}")
#             raise
    
#     def _calculate_sentiment_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
#         """Calculate indicators specifically for sentiment analysis"""
#         try:
#             close = df['Close']
#             volume = df['Volume']
            
#             # Price trend
#             price_change_pct = ((close.iloc[-1] - close.iloc[0]) / close.iloc[0]) * 100
            
#             # Moving average alignment
#             sma_5 = close.rolling(window=5).mean().iloc[-1]
#             sma_10 = close.rolling(window=10).mean().iloc[-1]
#             sma_20 = close.rolling(window=20).mean().iloc[-1]
            
#             ma_alignment_bullish = sma_5 > sma_10 > sma_20
#             ma_alignment_bearish = sma_5 < sma_10 < sma_20
            
#             # Momentum
#             rsi = self._calculate_rsi(close, period=14)
            
#             # Volume trend
#             volume_trend = (volume.rolling(window=5).mean().iloc[-1] / 
#                           volume.rolling(window=20).mean().iloc[-1]) if len(volume) >= 20 else 1
            
#             # Volatility
#             volatility = close.pct_change().std() * np.sqrt(252)
            
#             # Higher highs/lows analysis
#             highs = df['High'].rolling(window=5).max()
#             lows = df['Low'].rolling(window=5).min()
            
#             higher_highs = (highs.iloc[-1] > highs.iloc[-5]) if len(highs) >= 5 else False
#             higher_lows = (lows.iloc[-1] > lows.iloc[-5]) if len(lows) >= 5 else False
            
#             return {
#                 "price_change_pct": round(price_change_pct, 2),
#                 "ma_alignment_bullish": bool(ma_alignment_bullish),
#                 "ma_alignment_bearish": bool(ma_alignment_bearish),
#                 "rsi": round(rsi, 2),
#                 "volume_trend_ratio": round(volume_trend, 2),
#                 "volatility": round(volatility, 4),
#                 "higher_highs": bool(higher_highs),
#                 "higher_lows": bool(higher_lows),
#                 "current_price": round(close.iloc[-1], 2),
#                 "period_high": round(df['High'].max(), 2),
#                 "period_low": round(df['Low'].min(), 2)
#             }
            
#         except Exception as e:
#             self.logger.error(f"Error calculating sentiment indicators: {str(e)}")
#             raise
    
#     def _calculate_rsi(self, series: pd.Series, period: int = 14) -> float:
#         """Calculate Relative Strength Index"""
#         delta = series.diff()
#         gain = delta.where(delta > 0, 0)
#         loss = -delta.where(delta < 0, 0)
        
#         avg_gain = gain.rolling(window=period).mean()
#         avg_loss = loss.rolling(window=period).mean()
        
#         rs = avg_gain / avg_loss
#         rsi = 100 - (100 / (1 + rs))
        
#         return rsi.iloc[-1] if not rsi.empty else 50
    
#     def _calculate_historical_volatility(self, series: pd.Series, lookback: int = 20) -> float:
#         """Calculate annualized historical volatility"""
#         log_returns = np.log(series / series.shift(1))
#         volatility = log_returns.rolling(window=lookback).std() * np.sqrt(252)
#         return volatility.iloc[-1] if not volatility.empty else 0
    
#     def _calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> tuple:
#         """Calculate support and resistance levels"""
#         recent_data = df.tail(window)
#         support = recent_data['Low'].min()
#         resistance = recent_data['High'].max()
#         return support, resistance
    
#     # def _determine_strategies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
#     #     """Determine potential options strategies based on metrics"""
#     #     strategies = []
#     #     hv = metrics['historical_volatility']
#     #     rsi = metrics['rsi_14']
#     #     current_price = metrics['current_price']
        
#     #     # High volatility strategies
#     #     if hv > 0.25:
#     #         if 45 <= rsi <= 55:
#     #             strategies.append({
#     #                 "name": "Long Straddle",
#     #                 "condition": "High volatility, neutral RSI",
#     #                 "confidence": 0.85
#     #             })
#     #         else:
#     #             strategies.append({
#     #                 "name": "Long Strangle", 
#     #                 "condition": "High volatility, directional bias",
#     #                 "confidence": 0.80
#     #             })
        
#     #     # Low volatility strategies
#     #     elif hv < 0.15:
#     #         if 40 <= rsi <= 60:
#     #             strategies.append({
#     #                 "name": "Iron Condor",
#     #                 "condition": "Low volatility, range-bound",
#     #                 "confidence": 0.85
#     #             })
#     #         if 45 <= rsi <= 55:
#     #             strategies.append({
#     #                 "name": "Butterfly Spread",
#     #                 "condition": "Very low volatility, neutral",
#     #                 "confidence": 0.80
#     #             })
        
#     #     # Directional strategies
#     #     if rsi > 60 and current_price > metrics['sma_50']:
#     #         strategies.append({
#     #             "name": "Bull Call Spread",
#     #             "condition": "Bullish trend",
#     #             "confidence": 0.75
#     #         })
#     #     elif rsi < 40 and current_price < metrics['sma_50']:
#     #         strategies.append({
#     #             "name": "Bear Put Spread",
#     #             "condition": "Bearish trend",
#     #             "confidence": 0.75
#     #         })
        
#     #     # Sort by confidence
#     #     strategies.sort(key=lambda x: x['confidence'], reverse=True)
        
#     #     return strategies[:3]  # Return top 3 strategies

#     def _determine_strategies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
#         """Determine potential options strategies based on metrics"""
#         strategies = []
#         hv = metrics['historical_volatility']
#         rsi = metrics['rsi_14']
#         current_price = metrics['current_price']
#         sma_50 = metrics.get('sma_50', current_price)
        
#         # Calculate volatility-based strike ranges for recommendations
#         daily_vol = hv / np.sqrt(252)
#         expected_move_30d = current_price * daily_vol * np.sqrt(30)
        
#         # High volatility strategies
#         if hv > 0.25:
#             if 45 <= rsi <= 55:
#                 strategies.append({
#                     "name": "Long Straddle",
#                     "condition": "High volatility, neutral RSI",
#                     "confidence": 0.85,
#                     "key_parameters": {
#                         "atm_strike": round(current_price),
#                         "expected_move": round(expected_move_30d, 2),
#                         "volatility": round(hv * 100, 1)
#                     }
#                 })
#             else:
#                 strategies.append({
#                     "name": "Long Strangle", 
#                     "condition": "High volatility, directional bias",
#                     "confidence": 0.80,
#                     "key_parameters": {
#                         "call_strike": round(current_price + expected_move_30d * 0.5),
#                         "put_strike": round(current_price - expected_move_30d * 0.5),
#                         "volatility": round(hv * 100, 1)
#                     }
#                 })
        
#         # Low volatility strategies
#         elif hv < 0.15:
#             if 40 <= rsi <= 60:
#                 strategies.append({
#                     "name": "Iron Condor",
#                     "condition": "Low volatility, range-bound",
#                     "confidence": 0.85,
#                     "key_parameters": {
#                         "short_call": round(current_price + expected_move_30d * 0.5),
#                         "long_call": round(current_price + expected_move_30d),
#                         "short_put": round(current_price - expected_move_30d * 0.5),
#                         "long_put": round(current_price - expected_move_30d),
#                         "range_width": round(expected_move_30d, 2)
#                     }
#                 })
#             if 45 <= rsi <= 55:
#                 strategies.append({
#                     "name": "Butterfly Spread",
#                     "condition": "Very low volatility, neutral",
#                     "confidence": 0.80,
#                     "key_parameters": {
#                         "lower_strike": round(current_price - expected_move_30d * 0.3),
#                         "middle_strike": round(current_price),
#                         "upper_strike": round(current_price + expected_move_30d * 0.3),
#                         "max_profit_zone": f"${round(current_price - 2, 2)} - ${round(current_price + 2, 2)}"
#                     }
#                 })
        
#         # Directional strategies
#         if rsi > 60 and current_price > sma_50:
#             strategies.append({
#                 "name": "Bull Call Spread",
#                 "condition": "Bullish trend, momentum positive",
#                 "confidence": 0.75,
#                 "key_parameters": {
#                     "long_strike": round(current_price),
#                     "short_strike": round(current_price + expected_move_30d * 0.5),
#                     "trend_strength": "Strong" if rsi > 70 else "Moderate"
#                 }
#             })
#         elif rsi < 40 and current_price < sma_50:
#             strategies.append({
#                 "name": "Bear Put Spread",
#                 "condition": "Bearish trend, momentum negative",
#                 "confidence": 0.75,
#                 "key_parameters": {
#                     "long_strike": round(current_price),
#                     "short_strike": round(current_price - expected_move_30d * 0.5),
#                     "trend_strength": "Strong" if rsi < 30 else "Moderate"
#                 }
#             })
        
#         # Sort by confidence
#         strategies.sort(key=lambda x: x['confidence'], reverse=True)
        
#         return strategies[:3]  # Return top 3 strategies
    

#     def _analyze_technical_sentiment(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Analyze sentiment based purely on technical indicators.
        
#         This method uses a scoring system based on multiple technical factors
#         to determine if the market sentiment is bullish, bearish, or neutral.
#         """
#         # Initialize scoring
#         bullish_score = 0
#         bearish_score = 0
#         total_weight = 0
#         key_factors = []
        
#         # 1. Price Trend Analysis (Weight: 3)
#         weight = 3
#         total_weight += weight
#         if metrics['price_change_pct'] > 5:
#             bullish_score += weight
#             key_factors.append(f"Strong price increase: {metrics['price_change_pct']:.1f}%")
#         elif metrics['price_change_pct'] > 2:
#             bullish_score += weight * 0.5
#             key_factors.append(f"Moderate price increase: {metrics['price_change_pct']:.1f}%")
#         elif metrics['price_change_pct'] < -5:
#             bearish_score += weight
#             key_factors.append(f"Strong price decline: {metrics['price_change_pct']:.1f}%")
#         elif metrics['price_change_pct'] < -2:
#             bearish_score += weight * 0.5
#             key_factors.append(f"Moderate price decline: {metrics['price_change_pct']:.1f}%")
        
#         # 2. Moving Average Alignment (Weight: 4)
#         weight = 4
#         total_weight += weight
#         if metrics['ma_alignment_bullish']:
#             bullish_score += weight
#             key_factors.append("Bullish MA alignment (5>10>20)")
#         elif metrics['ma_alignment_bearish']:
#             bearish_score += weight
#             key_factors.append("Bearish MA alignment (5<10<20)")
        
#         # 3. RSI Analysis (Weight: 3)
#         weight = 3
#         total_weight += weight
#         rsi = metrics['rsi']
#         if rsi > 70:
#             bearish_score += weight * 0.5  # Overbought can lead to reversal
#             key_factors.append(f"RSI overbought: {rsi:.1f}")
#         elif rsi > 60:
#             bullish_score += weight
#             key_factors.append(f"RSI bullish: {rsi:.1f}")
#         elif rsi < 30:
#             bullish_score += weight * 0.5  # Oversold can lead to bounce
#             key_factors.append(f"RSI oversold: {rsi:.1f}")
#         elif rsi < 40:
#             bearish_score += weight
#             key_factors.append(f"RSI bearish: {rsi:.1f}")
        
#         # 4. Higher Highs/Lows Pattern (Weight: 3)
#         weight = 3
#         total_weight += weight
#         if metrics['higher_highs'] and metrics['higher_lows']:
#             bullish_score += weight
#             key_factors.append("Uptrend pattern: Higher highs and higher lows")
#         elif not metrics['higher_highs'] and not metrics['higher_lows']:
#             bearish_score += weight
#             key_factors.append("Downtrend pattern: Lower highs and lower lows")
        
#         # 5. Volume Analysis (Weight: 2)
#         weight = 2
#         total_weight += weight
#         volume_ratio = metrics['volume_trend_ratio']
#         if volume_ratio > 1.2:
#             if metrics['price_change_pct'] > 0:
#                 bullish_score += weight
#                 key_factors.append(f"High volume on price increase: {volume_ratio:.2f}x")
#             else:
#                 bearish_score += weight
#                 key_factors.append(f"High volume on price decrease: {volume_ratio:.2f}x")
        
#         # 6. Price vs Range Position (Weight: 2)
#         weight = 2
#         total_weight += weight
#         price_range_position = ((metrics['current_price'] - metrics['period_low']) / 
#                                (metrics['period_high'] - metrics['period_low'])) if metrics['period_high'] != metrics['period_low'] else 0.5
        
#         if price_range_position > 0.8:
#             bullish_score += weight
#             key_factors.append("Price near period high")
#         elif price_range_position < 0.2:
#             bearish_score += weight
#             key_factors.append("Price near period low")
        
#         # Calculate final scores
#         bullish_percentage = (bullish_score / total_weight) * 100
#         bearish_percentage = (bearish_score / total_weight) * 100
        
#         # Determine sentiment
#         sentiment_threshold = 15  # Minimum difference needed for directional bias
        
#         if bullish_percentage - bearish_percentage > sentiment_threshold:
#             sentiment = "BULLISH"
#             confidence = min(0.9, 0.5 + (bullish_percentage - bearish_percentage) / 100)
#         elif bearish_percentage - bullish_percentage > sentiment_threshold:
#             sentiment = "BEARISH"
#             confidence = min(0.9, 0.5 + (bearish_percentage - bullish_percentage) / 100)
#         else:
#             sentiment = "NEUTRAL"
#             confidence = 0.6
        
#         # Generate recommendation based on sentiment
#         if sentiment == "BULLISH":
#             recommendation = f"Technical indicators suggest bullish momentum. Consider long positions with stop loss at ${metrics['period_low']:.2f}"
#         elif sentiment == "BEARISH":
#             recommendation = f"Technical indicators suggest bearish pressure. Consider defensive positions or wait for better entry"
#         else:
#             recommendation = f"Mixed signals. Consider range-bound strategies between ${metrics['period_low']:.2f} - ${metrics['period_high']:.2f}"
        
#         return {
#             "sentiment": sentiment,
#             "confidence": round(confidence, 2),
#             "key_factors": key_factors[:5],  # Top 5 factors
#             "recommendation": recommendation,
#             "technical_scores": {
#                 "bullish_score": round(bullish_percentage, 1),
#                 "bearish_score": round(bearish_percentage, 1)
#             }
#         }