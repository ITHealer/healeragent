from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime
import statistics
import logging

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod


class SentimentAnalysisHandler(LoggerMixin):
    """Handler for analyzing social sentiment data and its market impact"""
    
    def __init__(self):
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()
        
    async def analyze_sentiment_impact(
        self,
        symbol: str,
        query: Optional[str],
        target_language: Optional[str],
        sentiment_data: List[Dict[str, Any]],
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment data and its potential market impact
        """
        try:
            # Log raw data structure for debugging
            if sentiment_data and len(sentiment_data) > 0:
                self.logger.info(f"Sample sentiment data structure: {sentiment_data[0]}")
            
            # Process and aggregate sentiment data
            processed_data = self._process_sentiment_data(sentiment_data)
            
            # Validate processed data has all required keys
            required_keys = ['sentiment_level', 'average_sentiment', 'sentiment_trend', 
                           'volatility', 'recent_shift', 'data_points']
            
            for key in required_keys:
                if key not in processed_data:
                    self.logger.error(f"Missing key in processed data: {key}")
                    processed_data[key] = "N/A"  # Set default value
            
            # Create analysis prompt
            detection_method = ""
            if len(query.split()) < 2:
                detection_method = DetectionMethod.LLM
            else:
                detection_method = DetectionMethod.LIBRARY

            # Language detection
            language_info = await language_detector.detect(
                text=query,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )

            detected_language = language_info["detected_language"]

            if detected_language:
                lang_name = {
                    "en": "English",
                    "vi": "Vietnamese", 
                    "zh": "Chinese",
                    "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
                    "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
                }.get(detected_language, "the detected language")
                
            language_instruction = f"""
            CRITICAL LANGUAGE REQUIREMENT:
            You MUST respond ENTIRELY in {lang_name} language.
            - ALL text, explanations, and analysis must be in {lang_name}
            - Use appropriate financial terminology for {lang_name}
            - Format numbers and dates according to {lang_name} conventions
            """

            prompt = self._create_sentiment_analysis_prompt(symbol, processed_data, language_instruction)
            
            # Get LLM analysis
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.3
            )
            
            analysis_text = response["content"]
            
            return {
                "symbol": symbol,
                "sentiment_summary": processed_data,
                "market_impact_analysis": analysis_text,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}", exc_info=True)
            raise Exception(f"Sentiment analysis failed: {str(e)}")
    

    async def stream_sentiment_impact(
        self,
        symbol: str,
        sentiment_data: List[Dict[str, Any]],
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        memory_context: str = "",
        user_question: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream sentiment analysis and its potential market impact
        """
        try:
            # Log raw data structure for debugging
            if sentiment_data and len(sentiment_data) > 0:
                self.logger.info(f"Sample sentiment data structure: {sentiment_data[0]}")
            
            # Process and aggregate sentiment data
            processed_data = self._process_sentiment_data(sentiment_data)
            
            # Validate processed data has all required keys
            required_keys = ['sentiment_level', 'average_sentiment', 'sentiment_trend', 
                        'volatility', 'recent_shift', 'data_points']
            
            for key in required_keys:
                if key not in processed_data:
                    self.logger.error(f"Missing key in processed data: {key}")
                    processed_data[key] = "N/A"  # Set default value
            

            detection_method = ""
            query = user_question if user_question else symbol
            
            if len(query.split()) < 2:
                detection_method = DetectionMethod.LLM
            else:
                detection_method = DetectionMethod.LIBRARY

            language_info = await language_detector.detect(
                text=query,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )

            detected_language = language_info["detected_language"]

            if detected_language:
                lang_name = {
                    "en": "English",
                    "vi": "Vietnamese", 
                    "zh": "Chinese",
                    "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
                    "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
                }.get(detected_language, "the detected language")
                
            language_instruction = f"""
            CRITICAL LANGUAGE REQUIREMENT:
            You MUST respond ENTIRELY in {lang_name} language.
            - ALL text, explanations, and analysis must be in {lang_name}
            - Use appropriate financial terminology for {lang_name}
            - Format numbers and dates according to {lang_name} conventions
            """
            
            # Create analysis prompt
            prompt = self._create_sentiment_analysis_prompt(
                symbol=symbol,
                processed_data=processed_data,
                memory_context=memory_context,  
                user_question=query if query else "",  
                language_instruction=language_instruction
            )
            
            # Get LLM analysis
            messages = [
                {"role": "system", "content": self._get_system_prompt(language_instruction)},
                {"role": "user", "content": prompt}
            ]
            
            # Stream response
            async for chunk in self.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.3,
                clean_thinking=True
            ):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Error streaming sentiment analysis for {symbol}: {str(e)}", exc_info=True)
            yield f"Error in sentiment analysis: {str(e)}"
            
    def _process_sentiment_data(self, sentiment_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process raw sentiment data into aggregated metrics"""
        
        try:
            if not sentiment_data:
                return self._get_empty_sentiment_summary()
            
            # Extract sentiment scores
            sentiment_scores = []
            dates = []
            stocktwits_scores = []
            twitter_scores = []
            
            for item in sentiment_data:
                # Handle both Pydantic model and dict
                if hasattr(item, 'model_dump'):
                    item = item.model_dump()
                elif hasattr(item, 'dict'):
                    item = item.dict()
                
                # Extract date - handle datetime objects
                date = None
                if 'date' in item:
                    if isinstance(item['date'], datetime):
                        date = item['date'].strftime('%Y-%m-%d %H:%M')
                    else:
                        date = str(item['date'])
                
                # Extract sentiment scores from FMP structure
                stocktwits_sentiment = None
                twitter_sentiment = None
                
                # Get StockTwits sentiment
                if 'stocktwitsSentiment' in item and item['stocktwitsSentiment'] is not None:
                    try:
                        stocktwits_sentiment = float(item['stocktwitsSentiment'])
                        stocktwits_scores.append(stocktwits_sentiment)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid stocktwitsSentiment value: {item['stocktwitsSentiment']}")
                
                # Get Twitter sentiment
                if 'twitterSentiment' in item and item['twitterSentiment'] is not None:
                    try:
                        twitter_sentiment = float(item['twitterSentiment'])
                        twitter_scores.append(twitter_sentiment)
                    except (ValueError, TypeError):
                        self.logger.warning(f"Invalid twitterSentiment value: {item['twitterSentiment']}")
                
                # Calculate combined sentiment (average of both if available)
                combined_score = None
                if stocktwits_sentiment is not None and twitter_sentiment is not None:
                    combined_score = (stocktwits_sentiment + twitter_sentiment) / 2
                elif stocktwits_sentiment is not None:
                    combined_score = stocktwits_sentiment
                elif twitter_sentiment is not None:
                    combined_score = twitter_sentiment
                
                if combined_score is not None:
                    sentiment_scores.append(combined_score)
                    dates.append(date or "Unknown")
                    
                    # Log for debugging
                    self.logger.debug(f"Processed sentiment - Date: {date}, "
                                    f"StockTwits: {stocktwits_sentiment}, "
                                    f"Twitter: {twitter_sentiment}, "
                                    f"Combined: {combined_score}")
            
            if not sentiment_scores:
                self.logger.warning("No valid sentiment scores found in data after processing")
                return self._get_empty_sentiment_summary()
            
            # Calculate metrics
            avg_sentiment = statistics.mean(sentiment_scores)
            avg_stocktwits = statistics.mean(stocktwits_scores) if stocktwits_scores else None
            avg_twitter = statistics.mean(twitter_scores) if twitter_scores else None
            
            # Calculate volatility
            volatility = statistics.stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
            
            # Determine trend
            trend = self._calculate_trend(sentiment_scores)
            recent_shift = self._calculate_recent_shift(sentiment_scores)
            
            # Determine sentiment level - FMP sentiment is already 0-1 scale
            sentiment_level = self._get_sentiment_level_normalized(avg_sentiment)
            
            # Calculate social media metrics
            total_posts = sum(item.get('stocktwitsPosts', 0) + item.get('twitterPosts', 0) 
                             for item in sentiment_data[:10])  # Last 10 entries
            total_likes = sum(item.get('stocktwitsLikes', 0) + item.get('twitterLikes', 0) 
                             for item in sentiment_data[:10])
            
            summary = {
                "data_points": len(sentiment_scores),
                "average_sentiment": round(avg_sentiment, 3),
                "sentiment_level": sentiment_level,
                "sentiment_trend": trend,
                "volatility": round(volatility, 3),
                "recent_shift": recent_shift,
                "latest_sentiment": round(sentiment_scores[0], 3) if sentiment_scores else 0,
                "date_range": f"{dates[-1] if dates else 'N/A'} to {dates[0] if dates else 'N/A'}",
                "min_sentiment": round(min(sentiment_scores), 3) if sentiment_scores else 0,
                "max_sentiment": round(max(sentiment_scores), 3) if sentiment_scores else 0,
                # Additional platform-specific metrics
                "stocktwits_avg_sentiment": round(avg_stocktwits, 3) if avg_stocktwits else "N/A",
                "twitter_avg_sentiment": round(avg_twitter, 3) if avg_twitter else "N/A",
                "recent_social_activity": {
                    "total_posts": total_posts,
                    "total_likes": total_likes,
                    "engagement_rate": round(total_likes / total_posts, 2) if total_posts > 0 else 0
                }
            }
            
            self.logger.info(f"Successfully processed {len(sentiment_scores)} sentiment data points")
            return summary
            
        except Exception as e:
            self.logger.error(f"Error processing sentiment data: {str(e)}", exc_info=True)
            return self._get_empty_sentiment_summary()
    
    def _get_empty_sentiment_summary(self) -> Dict[str, Any]:
        """Return empty sentiment summary with all required keys"""
        return {
            "data_points": 0,
            "average_sentiment": 0,
            "sentiment_level": "No Data",
            "sentiment_trend": "No Data",
            "volatility": 0,
            "recent_shift": "No Data",
            "latest_sentiment": 0,
            "date_range": "N/A",
            "min_sentiment": 0,
            "max_sentiment": 0,
            "stocktwits_avg_sentiment": "N/A",
            "twitter_avg_sentiment": "N/A",
            "recent_social_activity": {
                "total_posts": 0,
                "total_likes": 0,
                "engagement_rate": 0
            }
        }
    
    def _calculate_trend(self, sentiment_scores: List[float]) -> str:
        """Calculate sentiment trend"""
        if len(sentiment_scores) < 5:
            return "Insufficient Data"
        
        recent_avg = statistics.mean(sentiment_scores[:5])
        older_avg = statistics.mean(sentiment_scores[5:10] if len(sentiment_scores) >= 10 else sentiment_scores[5:])
        
        if recent_avg > older_avg + 0.05:  # Adjusted threshold for 0-1 scale
            return "Improving"
        elif recent_avg < older_avg - 0.05:
            return "Declining"
        else:
            return "Stable"
    
    def _calculate_recent_shift(self, sentiment_scores: List[float]) -> str:
        """Calculate recent shift in sentiment"""
        if len(sentiment_scores) < 5:
            return "N/A"
        
        recent_avg = statistics.mean(sentiment_scores[:5])
        older_avg = statistics.mean(sentiment_scores[5:10] if len(sentiment_scores) >= 10 else sentiment_scores[5:])
        
        if older_avg != 0:
            shift_pct = ((recent_avg - older_avg) / abs(older_avg)) * 100
            return f"{shift_pct:+.1f}%"
        return "N/A"
    
    def _get_sentiment_level_normalized(self, avg_sentiment: float) -> str:
        """Determine sentiment level for 0-1 scale (FMP format)"""
        if avg_sentiment >= 0.8:
            return "Very Positive"
        elif avg_sentiment >= 0.65:
            return "Positive"
        elif avg_sentiment >= 0.55:
            return "Slightly Positive"
        elif avg_sentiment >= 0.45:
            return "Neutral"
        elif avg_sentiment >= 0.35:
            return "Slightly Negative"
        elif avg_sentiment >= 0.2:
            return "Negative"
        else:
            return "Very Negative"
    
    def _get_system_prompt(self, language_instruction) -> str:
        return f"""You are a financial sentiment analyst specializing in social media sentiment analysis and its impact on stock markets.

    {language_instruction} 
    
    Your role is to:
    1. Interpret social sentiment data and explain what it means
    2. Analyze potential market impact based on sentiment trends
    3. Identify sentiment-driven trading opportunities or risks
    4. Provide actionable insights for investors
    5. Build upon previous analyses when relevant context is available
    6. Answer specific user questions while incorporating sentiment analysis

    Guidelines:
    - Be specific about the sentiment levels and their implications
    - Consider both short-term and long-term market impacts
    - Highlight any concerning patterns or positive signals
    - Relate sentiment to potential price movements
    - Reference previous analyses when they add value to current insights
    - Address user's specific questions directly while maintaining comprehensive analysis
    - Keep analysis concise but comprehensive
    - Use data to support your conclusions
    - Maintain continuity in your analytical approach when building on previous work"""

    
    def _create_sentiment_analysis_prompt(self, symbol: str, processed_data: Dict[str, Any], memory_context: str, user_question: str, language_instruction: str) -> str:
        """Create enhanced prompt with platform-specific data"""
        
        # Build social activity section
        social_activity = processed_data.get('recent_social_activity', {})
        social_section = ""
        if social_activity:
            social_section = f"""
**Social Media Activity:**
- Total Posts (Recent): {social_activity.get('total_posts', 0)}
- Total Likes: {social_activity.get('total_likes', 0)}
- Engagement Rate: {social_activity.get('engagement_rate', 0)}
- StockTwits Sentiment: {processed_data.get('stocktwits_avg_sentiment', 'N/A')}
- Twitter Sentiment: {processed_data.get('twitter_avg_sentiment', 'N/A')}"""
  

        context_section = ""
        if memory_context and memory_context.strip():
            context_section = f"""
    **Previous Analysis Context:**
    {memory_context}

    **Analytical Continuity:**
    When relevant, reference previous findings to show progression, changes, or confirmations in sentiment patterns. Highlight how current data compares to historical analysis."""

        question_section = ""
        if user_question and user_question.strip():
            question_section = f"""
    **Specific User Question:**
    "{user_question}"

    **Response Priority:**
    Address the above question directly while integrating it with the comprehensive sentiment analysis. Ensure the specific question receives focused attention within the broader analytical framework."""


            return f"""Analyze the social sentiment data for {symbol} and provide insights based on the following information:
    {context_section}
    {question_section}

    ## Current Sentiment Data:
    **Sentiment Overview:**
    - Data Points: {processed_data.get('data_points', 0)}
    - Average Sentiment Score: {processed_data.get('average_sentiment', 0)} ({processed_data.get('sentiment_level', 'Unknown')})
    - Latest Sentiment: {processed_data.get('latest_sentiment', 0)}
    - Trend: {processed_data.get('sentiment_trend', 'Unknown')}
    - Recent Shift: {processed_data.get('recent_shift', 'N/A')}
    - Volatility: {processed_data.get('volatility', 0)}
    - Date Range: {processed_data.get('date_range', 'N/A')}
    - Min/Max Range: {processed_data.get('min_sentiment', 0)} to {processed_data.get('max_sentiment', 0)}
    {social_section}

    Please provide a comprehensive analysis covering:

    1. **Sentiment Interpretation**: What does the current sentiment level indicate about market perception?

    2. **Platform Analysis**: Compare StockTwits vs Twitter sentiment - what do differences suggest?

    3. **Social Engagement**: What does the engagement rate tell us about investor interest?

    4. **Trend Analysis**: How is sentiment changing over time and what does this suggest?

    5. **Market Impact Assessment**: 
    - Short-term impact (1-5 days)
    - Medium-term impact (1-4 weeks)
    - Key risk factors

    6. **Trading Implications**: Based on sentiment data, what should investors consider?

    7. **Red Flags or Positive Signals**: Any notable patterns that deserve attention?

    Focus on actionable insights and be specific about potential price impacts."""


# Create singleton instance
sentiment_analysis_handler = SentimentAnalysisHandler()