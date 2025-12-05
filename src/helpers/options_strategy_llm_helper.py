import json
from typing import Dict, Any, List, Optional, AsyncGenerator
import pandas as pd

from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.llm_chat_helper import analyze_stock
from src.helpers.llm_helper import LLMGeneratorProvider

class OptionsStrategyLLMHelper(LoggerMixin):
    """Helper class for LLM-based options strategy analysis"""
    
    def __init__(self):
        super().__init__()
        
    async def generate_strategy_recommendations(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        technical_metrics: Dict[str, Any],
        initial_strategies: List[Dict[str, Any]],
        model_name: str,
        provider_type: str,
        api_key: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate detailed options strategy recommendations using LLM.
        
        Args:
            symbol: Stock symbol
            market_data: Historical price data
            technical_metrics: Calculated technical indicators
            initial_strategies: Pre-determined strategies based on rules
            model_name: LLM model to use
            provider_type: Provider type
            api_key: API key for the provider
            
        Returns:
            List of enhanced strategy recommendations
        """
        try:
            # Prepare the prompt
            prompt = self._create_strategy_prompt(
                symbol, 
                technical_metrics, 
                initial_strategies,
                market_data
            )
            
            # Get LLM response
            if provider_type == ProviderType.OLLAMA:
                response = await analyze_stock(prompt, model_name)
            else:
                # Use provider factory for other providers
                provider = ModelProviderFactory.create_provider(
                    provider_type=provider_type,
                    model_name=model_name,
                    api_key=api_key
                )
                
                messages = [
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
                
                response_data = await provider.generate(
                    messages=messages,
                    temperature=0.7
                )
                response = response_data.get("content", "")
            
            # Parse and enhance strategies
            enhanced_strategies = self._parse_llm_response(response, initial_strategies)
            
            return enhanced_strategies
            
        except Exception as e:
            self.logger.error(f"Error generating strategy recommendations: {str(e)}")
            # Return initial strategies as fallback
            return initial_strategies
    

    async def stream_strategy_recommendations(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        technical_metrics: Dict[str, Any],
        initial_strategies: List[Dict[str, Any]],
        model_name: str,
        provider_type: str,
        api_key: Optional[str] = None,
        language_instruction: Optional[str] = "English"
    ) -> AsyncGenerator[str, None]:
        """
        Stream detailed options strategy recommendations using LLM.
        """
        try:
            # Prepare the prompt
            prompt = self._create_strategy_prompt(
                symbol, 
                technical_metrics, 
                initial_strategies,
                market_data
            )
            
            messages = [
                {"role": "system", "content": self._get_system_prompt(language_instruction)},
                {"role": "user", "content": prompt}
            ]
            
            # Stream LLM response
            if provider_type == ProviderType.OLLAMA:
                # Use LLMGeneratorProvider for Ollama streaming
                llm_provider = LLMGeneratorProvider()
                async for chunk in llm_provider.stream_response(
                    model_name=model_name,
                    messages=messages,
                    provider_type=provider_type,
                    api_key=api_key,
                    temperature=0.7,
                    clean_thinking=True
                ):
                    yield chunk
            else:
                # Use provider factory for other providers
                provider = ModelProviderFactory.create_provider(
                    provider_type=provider_type,
                    model_name=model_name,
                    api_key=api_key
                )
                
                await provider.initialize()
                
                async for chunk in provider.stream(messages, temperature=0.7):
                    yield chunk
                    
        except Exception as e:
            self.logger.error(f"Error streaming strategy recommendations: {str(e)}")
            yield f"Error generating recommendations: {str(e)}"


    async def analyze_market_sentiment(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        sentiment_metrics: Dict[str, Any],
        model_name: str,
        provider_type: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze market sentiment using LLM.
        
        Args:
            symbol: Stock symbol
            market_data: Recent price data
            sentiment_metrics: Calculated sentiment indicators
            model_name: LLM model to use
            provider_type: Provider type
            api_key: API key for the provider
            
        Returns:
            Dictionary containing sentiment analysis
        """
        try:
            # Create sentiment analysis prompt
            prompt = self._create_sentiment_prompt(symbol, sentiment_metrics)
            
            # Get LLM response
            if provider_type == ProviderType.OLLAMA:
                response = await analyze_stock(prompt, model_name)
            else:
                provider = ModelProviderFactory.create_provider(
                    provider_type=provider_type,
                    model_name=model_name,
                    api_key=api_key
                )
                
                messages = [
                    {"role": "system", "content": self._get_sentiment_system_prompt()},
                    {"role": "user", "content": prompt}
                ]
                
                response_data = await provider.generate(
                    messages=messages,
                    temperature=0.7
                )
                response = response_data.get("content", "")
            
            # Parse sentiment analysis
            sentiment_analysis = self._parse_sentiment_response(response, sentiment_metrics)
            
            return sentiment_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            # Return basic sentiment based on metrics
            return self._fallback_sentiment_analysis(sentiment_metrics)
    
    def _get_system_prompt(self, language_instruction) -> str:
        """Get system prompt for options strategy analysis"""

        return """You are an expert options trader and financial analyst. Your role is to provide detailed, actionable options trading strategies based on technical analysis and market conditions.

{language_instruction}

When analyzing stocks and recommending strategies, you should:
1. Consider the current market conditions and technical indicators
2. Provide specific entry points, strike prices, and expiration recommendations
3. Include risk management guidelines and position sizing
4. Explain the rationale behind each strategy clearly
5. Consider the trader's risk tolerance and market experience

Always provide practical, implementable advice with specific numbers and clear reasoning."""
    
    def _get_sentiment_system_prompt(self) -> str:
        """Get system prompt for sentiment analysis"""
        return """You are a market sentiment analyst specializing in technical analysis. Your role is to determine whether a stock is currently bullish, bearish, or neutral based on technical indicators and price action.
ALWAYS begin with: "I'm your DeepInvest assistant. As your financial market analyst, I've reviewed the comprehensive fundamental data and here are my insights:"

When analyzing sentiment, you should:
1. Consider multiple technical indicators holistically
2. Identify key support and resistance levels
3. Assess trend strength and momentum
4. Provide confidence levels for your assessment
5. Give actionable recommendations based on the sentiment

Be objective and data-driven in your analysis."""
    
    def _create_strategy_prompt(
        self, 
        symbol: str, 
        metrics: Dict[str, Any],
        strategies: List[Dict[str, Any]],
        market_data: pd.DataFrame
    ) -> str:
        """Create prompt for strategy recommendation"""
        
        # Get recent price action summary
        recent_prices = market_data.tail(5)
        price_summary = f"Recent 5-day prices: " + ", ".join([
            f"${row['Close']:.2f}" for _, row in recent_prices.iterrows()
        ])
        
        prompt = f"""Analyze the following options trading opportunity for {symbol}:

CURRENT MARKET DATA:
- Current Price: ${metrics['current_price']}
- RSI (14): {metrics['rsi_14']}
- Historical Volatility: {metrics['historical_volatility']:.2%}
- SMA20: ${metrics['sma_20']}
- SMA50: ${metrics['sma_50']}
- SMA200: ${metrics['sma_200']}
- Support Level: ${metrics['support_level']}
- Resistance Level: ${metrics['resistance_level']}
- Volume Ratio (current/20d avg): {metrics['volume_ratio']}
- 20-day Price Change: {metrics['price_change_20d_pct']}%

{price_summary}

POTENTIAL STRATEGIES IDENTIFIED:
"""
        
        for i, strategy in enumerate(strategies, 1):
            prompt += f"\n{i}. {strategy['name']}"
            prompt += f"\n   Condition: {strategy['condition']}"
            prompt += f"\n   Initial Confidence: {strategy['confidence']:.0%}"
        
        prompt += """

For each strategy above, please provide:

1. **Detailed Entry Recommendations**:
   - Specific strike prices (use actual dollar amounts based on current price)
   - Recommended expiration dates (be specific, e.g., "30-45 days" or specific monthly expirations)
   - Optimal entry timing considerations

2. **Risk Management**:
   - Maximum position size as % of portfolio
   - Stop loss levels or exit criteria
   - Break-even points
   - Maximum loss and profit scenarios

3. **Market Conditions Assessment**:
   - Why this strategy fits current conditions
   - Key risks to watch for
   - Adjustment triggers if market moves against position

4. **Execution Tips**:
   - Best practices for entering the position
   - Common mistakes to avoid
   - When to consider rolling or closing

Please be specific with numbers and provide actionable advice that a trader can immediately implement."""
        
        return prompt
    
    def _create_sentiment_prompt(self, symbol: str, metrics: Dict[str, Any]) -> str:
        """Create prompt for sentiment analysis"""
        
        prompt = f"""Analyze the market sentiment for {symbol} based on the following technical data:

PRICE ACTION:
- Current Price: ${metrics['current_price']}
- Period High: ${metrics['period_high']}
- Period Low: ${metrics['period_low']}
- Price Change: {metrics['price_change_pct']}%

TECHNICAL INDICATORS:
- RSI (14): {metrics['rsi']}
- Moving Average Alignment:
  * Bullish Alignment (5>10>20): {'Yes' if metrics['ma_alignment_bullish'] else 'No'}
  * Bearish Alignment (5<10<20): {'Yes' if metrics['ma_alignment_bearish'] else 'No'}
- Volume Trend (5d/20d): {metrics['volume_trend_ratio']}
- Volatility: {metrics['volatility']:.2%}

TREND ANALYSIS:
- Making Higher Highs: {'Yes' if metrics['higher_highs'] else 'No'}
- Making Higher Lows: {'Yes' if metrics['higher_lows'] else 'No'}

Based on this data, please provide:

1. **Overall Sentiment**: Is the stock BULLISH, BEARISH, or NEUTRAL?

2. **Confidence Level**: How confident are you in this assessment? (0-100%)

3. **Key Supporting Factors**: List the top 3-5 technical factors supporting your sentiment call

4. **Contrarian Indicators**: What factors, if any, contradict the primary sentiment?

5. **Actionable Recommendation**: What specific action should a trader take?
   - If bullish: Entry points, targets, stop loss
   - If bearish: Short entry or avoid levels
   - If neutral: Range to trade within

6. **Time Horizon**: Over what timeframe is this sentiment likely to play out?

Please provide a clear, data-driven analysis with specific price levels and reasoning."""
        
        return prompt
    
    def _parse_llm_response(
        self, 
        response: str, 
        initial_strategies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse LLM response and enhance initial strategies"""
        
        enhanced_strategies = []
        
        for strategy in initial_strategies:
            # Create enhanced strategy with LLM details
            enhanced = {
                "strategy_name": strategy["name"],
                "confidence_score": strategy["confidence"],
                "market_condition": strategy["condition"],
                # "strategy_recommendation": response,  # Full response for now
                # "entry_points": self._extract_entry_points(response, strategy["name"]),
                # "risk_metrics": self._extract_risk_metrics(response, strategy["name"])
            }
            enhanced_strategies.append(enhanced)
        
        result = {
            "strategies": enhanced_strategies,
            "interpretation": response
        }

        return result
    
    def _parse_sentiment_response(
        self, 
        response: str, 
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse sentiment analysis response"""
        
        # Try to extract sentiment from response
        response_lower = response.lower()
        
        if "bullish" in response_lower:
            sentiment = "BULLISH"
        elif "bearish" in response_lower:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
        
        # Extract confidence (look for percentage)
        confidence = 0.7  # default
        import re
        conf_match = re.search(r'(\d+)%', response)
        if conf_match:
            confidence = int(conf_match.group(1)) / 100
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            # "key_factors": self._extract_key_factors(response),
            "recommendation": self._extract_recommendation(response),
            "detailed_analysis": response
        }
    
    def _extract_entry_points(self, response: str, strategy_name: str) -> Dict[str, Any]:
        """Extract entry points from LLM response"""
        # This is a simplified extraction - in production, use more sophisticated parsing
        import re
        
        entry_points = {
            "suggested_strikes": [],
            "expiration": "30-45 days",
            "entry_criteria": "See detailed recommendation"
        }
        
        # Look for strike prices (simple regex for dollar amounts)
        strikes = re.findall(r'\$(\d+(?:\.\d{2})?)', response)
        if strikes:
            entry_points["suggested_strikes"] = [float(s) for s in strikes[:4]]  # Max 4 strikes
        
        return entry_points
    
    def _extract_risk_metrics(self, response: str, strategy_name: str) -> Dict[str, Any]:
        """Extract risk metrics from LLM response"""
        return {
            "max_loss": "See detailed recommendation",
            "max_profit": "See detailed recommendation",
            "breakeven": "See detailed recommendation",
            "position_sizing": "1-2% of portfolio recommended"
        }
    
    def _extract_key_factors(self, response: str) -> List[str]:
        """Extract key factors from sentiment analysis"""
        # Simple extraction - look for bullet points or numbered items
        factors = []
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                factors.append(line.lstrip('0123456789.-* '))
        
        return factors[:5]  # Return top 5 factors
    
    def _extract_recommendation(self, response: str) -> str:
        """Extract actionable recommendation"""
        # Look for recommendation section
        lines = response.split('\n')
        for i, line in enumerate(lines):
            if 'recommendation' in line.lower() or 'action' in line.lower():
                # Return next few lines as recommendation
                return ' '.join(lines[i:i+3])
        
        return "See detailed analysis for recommendations"
    
    def _fallback_sentiment_analysis(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Provide basic sentiment analysis without LLM"""
        
        # Simple rule-based sentiment
        bullish_signals = 0
        bearish_signals = 0
        
        if metrics['ma_alignment_bullish']:
            bullish_signals += 2
        if metrics['ma_alignment_bearish']:
            bearish_signals += 2
            
        if metrics['rsi'] > 60:
            bullish_signals += 1
        elif metrics['rsi'] < 40:
            bearish_signals += 1
            
        if metrics['price_change_pct'] > 5:
            bullish_signals += 1
        elif metrics['price_change_pct'] < -5:
            bearish_signals += 1
            
        if metrics['higher_highs'] and metrics['higher_lows']:
            bullish_signals += 2
        
        # Determine sentiment
        if bullish_signals > bearish_signals + 1:
            sentiment = "BULLISH"
            confidence = min(0.8, 0.5 + bullish_signals * 0.1)
        elif bearish_signals > bullish_signals + 1:
            sentiment = "BEARISH"
            confidence = min(0.8, 0.5 + bearish_signals * 0.1)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.6
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            # "key_factors": [
            #     f"RSI at {metrics['rsi']}",
            #     f"Price change: {metrics['price_change_pct']}%",
            #     f"MA alignment: {'Bullish' if metrics['ma_alignment_bullish'] else 'Bearish' if metrics['ma_alignment_bearish'] else 'Mixed'}"
            # ],
            "recommendation": f"{sentiment} bias detected. Trade with caution.",
            "detailed_analysis": "LLM analysis not available. Using technical indicators only."
        }