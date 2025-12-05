from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod


logger_mixin = LoggerMixin()

class NewsAnalysisHandler(LoggerMixin):
    """
    Handler for news analysis operations.
    Analyzes company news and provides market impact insights using AI.
    """
    
    def __init__(self):
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()
        self.logger = LoggerMixin().logger
        
    async def analyze_company_news(
        self,
        symbol: str,
        news_data: List[Dict[str, Any]],
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        include_trading_signals: bool = True,
        question_input: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze company news using LLM for market impact assessment.
        
        Args:
            symbol: Stock symbol
            news_data: List of news items from FMP
            model_name: LLM model to use
            provider_type: Provider type (openai/ollama/gemini)
            api_key: API key for provider
            include_trading_signals: Whether to include trading recommendations
            
        Returns:
            Dict containing news analysis and market impact assessment
        """
        try:
            if not news_data:
                return {
                    "symbol": symbol,
                    "news_count": 0,
                    "analysis": "No recent news available for analysis.",
                    "market_impact": "neutral",
                    "sentiment_score": 0,
                    "key_insights": {}
                }
            
            # Sort news by date (most recent first)
            sorted_news = sorted(
                news_data, 
                key=lambda x: self._parse_date(x.get("publishedDate", "")), 
                reverse=True
            )
            
            detection_method = ""
            if len(question_input.split()) < 2:
                detection_method = DetectionMethod.LLM
            else:
                detection_method = DetectionMethod.LIBRARY

            # Language detection
            language_info = await language_detector.detect(
                text=question_input,
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
            prompt = self._create_news_analysis_prompt(
                symbol=symbol,
                news_items=sorted_news[:10],  # Analyze up to 10 most recent news
                include_trading_signals=include_trading_signals
            )
            
            # Generate AI analysis
            messages = [
                {"role": "system", "content": self._get_system_prompt(language_instruction)},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.3
            )
            
            analysis_text = response.get("content", "Analysis generation failed.")
            
            # Extract structured insights
            market_impact = self._extract_market_impact(analysis_text)
            sentiment_score = self._calculate_sentiment_score(sorted_news)
            key_insights = self._extract_key_insights(sorted_news, analysis_text)
            
            # Identify critical news
            critical_news = self._identify_critical_news(sorted_news)
            
            return {
                "symbol": symbol,
                "news_count": len(news_data),
                "latest_news_date": sorted_news[0].get("publishedDate") if sorted_news else None,
                "analysis": analysis_text,
                "market_impact": market_impact,
                "sentiment_score": sentiment_score,
                "key_insights": key_insights,
                "critical_news": critical_news,
                "news_summary": self._create_news_summary(sorted_news[:5])
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news for {symbol}: {str(e)}")
            raise Exception(f"News analysis error: {str(e)}")


    async def stream_company_news_analysis(
        self,
        symbol: str,
        news_data: List[Dict[str, Any]],
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        include_trading_signals: bool = True,
        memory_context: str = "",  # Add memory context
        question_input: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream company news analysis using LLM for market impact assessment.
        """
        try:
            if not news_data:
                yield "No recent news available for analysis."
                return
            
            # Sort news by date (most recent first)
            sorted_news = sorted(
                news_data, 
                key=lambda x: self._parse_date(x.get("publishedDate", "")), 
                reverse=True
            )
            
            detection_method = ""
            if len(question_input.split()) < 2:
                detection_method = DetectionMethod.LLM
            else:
                detection_method = DetectionMethod.LIBRARY

            # Language detection
            language_info = await language_detector.detect(
                text=question_input,
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
            prompt = self._create_news_analysis_prompt(
                symbol=symbol,
                news_items=sorted_news[:10],  # Analyze up to 10 most recent news
                include_trading_signals=include_trading_signals,

            )

            if memory_context:
                enhanced_prompt = f"""Previous news analyses and market insights:
    {memory_context}

    Current news analysis request:
    {prompt}

    Please consider the previous analyses and patterns when providing your current assessment.
    Note any significant changes in sentiment, new developments, or evolving trends compared to previous analyses."""
            else:
                enhanced_prompt = prompt
            
            # Generate AI analysis
            messages = [
                {"role": "system", "content": self._get_system_prompt(language_instruction)},
                {"role": "user", "content": enhanced_prompt}
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
            self.logger.error(f"Error streaming news analysis for {symbol}: {str(e)}")
            yield f"Error analyzing news: {str(e)}"

    
    def _get_system_prompt(self, language_instruction: Optional[str]) -> str:
        """Get the system prompt for news analysis."""
        
        return """You are an expert financial analyst specializing in market sentiment and news impact analysis. 

{language_instruction}

Your role is to analyze company news and assess their potential impact on stock price and market sentiment.

Your analysis should cover:
1. **Sentiment Assessment**: Overall positive, negative, or neutral tone
2. **Market Impact**: How the news might affect stock price (short-term and long-term)
3. **Key Themes**: Main topics and trends from the news
4. **Risk Factors**: Any concerns or red flags
5. **Opportunities**: Positive catalysts or growth indicators
6. **Trading Implications**: What this means for investors

Provide analysis that is:
- Objective and balanced
- Specific about potential price impacts
- Clear about timeframes (immediate vs long-term effects)
- Actionable for traders and investors

Use clear formatting with sections and bullet points."""

    def _create_news_analysis_prompt(
        self,
        symbol: str,
        news_items: List[Dict[str, Any]],
        include_trading_signals: bool
    ) -> str:
        """Create a detailed prompt for news analysis."""
        
        # Format news items for analysis
        news_text = f"Analyzing {len(news_items)} recent news items for {symbol}:\n\n"
        
        for i, news in enumerate(news_items[:10], 1):
            news_text += f"**News {i}:**\n"
            news_text += f"📅 Date: {news.get('publishedDate', 'Unknown')}\n"
            news_text += f"📰 Title: {news.get('title', 'No title')}\n"
            news_text += f"🔗 Source: {news.get('site', 'Unknown source')}\n"
            news_text += f"📝 Content: {news.get('text', 'No content available')}\n\n"
        
        prompt = f"""{news_text}

Please provide a comprehensive news analysis covering:

1. **Overall Sentiment Analysis**
   - What is the overall tone of recent news? (Bullish/Bearish/Neutral)
   - Are there consistent themes across multiple news items?

2. **Market Impact Assessment**
   - Short-term impact (1-5 days): How might these news affect immediate price action?
   - Medium-term impact (1-4 weeks): What trends might develop?
   - Long-term implications (1+ months): Any fundamental changes suggested?

3. **Key Themes & Catalysts**
   - What are the main topics being discussed?
   - Any product launches, earnings, partnerships, or strategic changes?
   - Industry trends affecting the company?

4. **Risk Analysis**
   - What concerns or negative factors are mentioned?
   - Any regulatory, competitive, or operational risks?
   - How significant are these risks?

5. **Opportunity Assessment**
   - What positive developments are highlighted?
   - Growth catalysts or competitive advantages mentioned?
   - Market expansion or innovation opportunities?
"""

        if include_trading_signals:
            prompt += """
6. **Trading Implications**
   - Based on this news flow, what's the likely price direction?
   - Key support/resistance levels to watch?
   - Recommended action: Buy/Hold/Sell/Wait?
   - Risk/Reward assessment
"""

        prompt += f"""
Please structure your analysis clearly and provide specific, actionable insights for {symbol} investors."""

        return prompt
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            # Handle format: "2024-02-28 05:55:00"
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        except:
            try:
                # Fallback to date only
                return datetime.strptime(date_str[:10], "%Y-%m-%d")
            except:
                return datetime.now()
    
    def _extract_market_impact(self, analysis_text: str) -> str:
        """Extract market impact from analysis text."""
        analysis_lower = analysis_text.lower()
        
        # Look for explicit impact mentions
        if any(word in analysis_lower for word in ["very bullish", "strong buy", "significant positive"]):
            return "strong_positive"
        elif any(word in analysis_lower for word in ["bullish", "positive impact", "buy signal"]):
            return "positive"
        elif any(word in analysis_lower for word in ["very bearish", "strong sell", "significant negative"]):
            return "strong_negative"
        elif any(word in analysis_lower for word in ["bearish", "negative impact", "sell signal"]):
            return "negative"
        else:
            return "neutral"
    
    def _calculate_sentiment_score(self, news_items: List[Dict[str, Any]]) -> float:
        """
        Calculate sentiment score from news items (-100 to +100).
        This is a simplified version - in production, use proper NLP sentiment analysis.
        """
        if not news_items:
            return 0
        
        positive_words = [
            "growth", "profit", "revenue", "beat", "upgrade", "innovation", "partnership",
            "expansion", "record", "strong", "surge", "gain", "rise", "breakthrough"
        ]
        
        negative_words = [
            "loss", "decline", "fall", "miss", "downgrade", "lawsuit", "investigation",
            "recall", "delay", "cut", "weak", "concern", "risk", "threat", "problem"
        ]
        
        total_score = 0
        
        for news in news_items:
            text = f"{news.get('title', '')} {news.get('text', '')}".lower()
            
            # Count positive and negative words
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            # Calculate item score
            item_score = (pos_count - neg_count) * 10
            total_score += max(min(item_score, 100), -100)  # Bound between -100 and 100
        
        # Average score
        avg_score = total_score / len(news_items)
        return round(avg_score, 2)
    
    def _extract_key_insights(self, news_items: List[Dict[str, Any]], analysis_text: str) -> Dict[str, Any]:
        """Extract key insights from news and analysis."""
        # Time-based categorization
        now = datetime.now()
        last_24h = sum(1 for news in news_items 
                      if (now - self._parse_date(news.get("publishedDate", ""))).days < 1)
        last_week = sum(1 for news in news_items 
                       if (now - self._parse_date(news.get("publishedDate", ""))).days < 7)
        
        # Source diversity
        unique_sources = len(set(news.get("site", "") for news in news_items))
        
        return {
            "news_velocity": {
                "last_24h": last_24h,
                "last_week": last_week,
                "trend": "high" if last_24h > 3 else "normal" if last_24h > 0 else "low"
            },
            "source_diversity": unique_sources,
            "coverage_type": self._classify_coverage_type(news_items),
            "main_topics": self._extract_main_topics(news_items),
            "urgency_level": self._assess_urgency(news_items, analysis_text)
        }
    
    def _classify_coverage_type(self, news_items: List[Dict[str, Any]]) -> str:
        """Classify the type of news coverage."""
        titles = " ".join(news.get("title", "").lower() for news in news_items)
        
        if any(word in titles for word in ["earnings", "revenue", "profit", "quarterly"]):
            return "earnings_related"
        elif any(word in titles for word in ["product", "launch", "announce", "release"]):
            return "product_announcement"
        elif any(word in titles for word in ["partnership", "acquisition", "merger", "deal"]):
            return "corporate_action"
        elif any(word in titles for word in ["analyst", "upgrade", "downgrade", "rating"]):
            return "analyst_coverage"
        else:
            return "general_news"
    
    def _extract_main_topics(self, news_items: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from news items."""
        # This is simplified - in production, use NLP topic modeling
        topics = []
        
        all_text = " ".join(f"{news.get('title', '')} {news.get('text', '')}" 
                           for news in news_items).lower()
        
        topic_keywords = {
            "earnings": ["earnings", "revenue", "profit", "quarterly", "beat", "miss"],
            "product": ["product", "launch", "release", "innovation", "technology"],
            "market": ["market", "share", "competition", "industry", "sector"],
            "management": ["ceo", "executive", "leadership", "board", "appointment"],
            "regulatory": ["regulation", "compliance", "investigation", "lawsuit", "legal"],
            "financial": ["debt", "cash", "dividend", "buyback", "capital"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                topics.append(topic)
        
        return topics[:3]  # Return top 3 topics
    
    def _assess_urgency(self, news_items: List[Dict[str, Any]], analysis_text: str) -> str:
        """Assess urgency level of news."""
        # Check for time-sensitive keywords
        urgent_keywords = ["breaking", "urgent", "immediately", "now", "today", "alert"]
        all_text = f"{analysis_text} " + " ".join(news.get("title", "") for news in news_items)
        
        if any(keyword in all_text.lower() for keyword in urgent_keywords):
            return "high"
        elif len(news_items) > 5 and self._calculate_sentiment_score(news_items) > 50:
            return "medium"
        else:
            return "low"
    
    # def _identify_critical_news(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    #     """Identify the most critical news items."""
    #     critical_keywords = [
    #         "earnings", "merger", "acquisition", "lawsuit", "investigation",
    #         "fda", "approval", "recall", "bankruptcy", "layoff", "restructuring"
    #     ]
        
    #     critical_news = []
        
    #     for news in news_items[:5]:  # Check top 5 news
    #         title = news.get("title", "").lower()
    #         text = news.get("text", "").lower()
            
    #         if any(keyword in title or keyword in text for keyword in critical_keywords):
    #             critical_news.append({
    #                 "title": news.get("title"),
    #                 "date": news.get("publishedDate"),
    #                 "summary": text[:200] + "..." if len(text) > 200 else text
    #             })
        
    #     return critical_news[:3]  # Return top 3 critical news
    
    def _convert_to_dict(self, item: Union[Dict, Any]) -> Dict[str, Any]:
        """Convert item to dictionary, handling both dict and Pydantic models."""
        if isinstance(item, dict):
            return item
        elif hasattr(item, 'model_dump'):
            return item.model_dump()
        elif hasattr(item, 'dict'):
            return item.dict()
        else:
            # Try to access as attributes
            return {
                "publishedDate": getattr(item, "publishedDate", None),
                "title": getattr(item, "title", None),
                "site": getattr(item, "site", None),
                "text": getattr(item, "text", None),
                "url": getattr(item, "url", None),
                "symbol": getattr(item, "symbol", None),
                "image": getattr(item, "image", None)
            }

    def _create_news_summary(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Create a summary of news items."""
        summary = []
        
        for news in news_items:
            news_dict = self._convert_to_dict(news)
            
            summary.append({
                "date": news_dict.get("publishedDate", "Unknown"),
                "title": news_dict.get("title", "No title"),
                "source": news_dict.get("site", "Unknown"),
                "brief": (news_dict.get("text", "")[:100] + "...") if news_dict.get("text") else "No content"
            })
        
        return summary

    def _identify_critical_news(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Identify the most critical news items."""
        critical_keywords = [
            "earnings", "merger", "acquisition", "lawsuit", "investigation",
            "fda", "approval", "recall", "bankruptcy", "layoff", "restructuring"
        ]
        
        critical_news = []
        
        for news in news_items[:5]:  # Check top 5 news
            news_dict = self._convert_to_dict(news)
            
            title = (news_dict.get("title", "") or "").lower()
            text = (news_dict.get("text", "") or "").lower()
            
            if any(keyword in title or keyword in text for keyword in critical_keywords):
                critical_news.append({
                    "title": news_dict.get("title"),
                    "date": news_dict.get("publishedDate"),
                    "summary": text[:200] + "..." if len(text) > 200 else text
                })
        
        return critical_news[:3]