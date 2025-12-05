from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import json
import random
from src.schemas.question_suggestion import (
    UserLevel, 
    SuggestedQuestion,
    SuggestedQuestion_v2,
    QuestionSuggestionRequest
)
from src.services.news_service import NewsService
from src.helpers.chat_management_helper import ChatService  
from src.services.llm_service import LLMService
from src.providers.provider_factory import ProviderType, ModelProviderFactory
from src.helpers.llm_helper import LLMGeneratorProvider
from src.services.session_summary_service import SessionSummaryService

logger = logging.getLogger(__name__)

class QuestionType:
    TECHNICAL = "technical_expert"
    CRYPTO = "crypto_analyst"
    FUNDAMENTAL = "fundamental_guru"
    SENTIMENT = "sentiment_analyzer"

llm_service_v1 = LLMService()

class QuestionSuggestionService:
    def __init__(self, news_service: NewsService, chat_service: ChatService, llm_service: LLMGeneratorProvider):
        self.news_service = news_service
        self.chat_service = chat_service
        self.llm_service = llm_service
        self.llm_provider = LLMGeneratorProvider()
        self.summary_service = SessionSummaryService(
            chat_service=chat_service,
            llm_service=llm_service
        )
        
        # Define supported tools and their capabilities
        self.supported_tools = {
            "price_analysis": {
                "name": "Stock Price & Valuation",
                "examples": ["How is AAPL performing?", "TSLA price target", "NVDA valuation metrics"],
                "requires_symbol": True
            },
            "financials": {
                "name": "Financial Analysis",
                "examples": ["AAPL earnings report", "MSFT revenue growth", "GOOGL profit margins"],
                "requires_symbol": True
            },
            "technical_analysis": {
                "name": "Technical Analysis & Charts",
                "examples": ["Show AAPL chart", "TSLA technical analysis", "BTC-USD price pattern"],
                "requires_symbol": True
            },
            "news": {
                "name": "Stock News",
                "examples": ["Latest news on AAPL", "Why is TSLA moving?", "NVDA recent events"],
                "requires_symbol": True
            },
            "market_overview": {
                "name": "Market Overview",
                "examples": ["How is the market today?", "S&P 500 performance", "Market sentiment"],
                "requires_symbol": False
            },
            "trending": {
                "name": "Trending Stocks",
                "examples": ["What stocks are trending?", "Top gainers today", "Biggest losers"],
                "requires_symbol": False
            },
            "heatmap": {
                "name": "Stock Heatmap",
                "examples": ["Show stock heatmap", "Tech sector heatmap", "Market cap heatmap"],
                "requires_symbol": False
            }
        }
        
        # Popular stocks for examples
        self.popular_stocks = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "SPY", "QQQ"]
        self.popular_crypto = ["BTC-USD", "ETH-USD", "DOGE-USD", "ADA-USD", "SOL-USD"]
    
    async def generate_question_suggestions(
        self,
        request: QuestionSuggestionRequest
    ) -> List[SuggestedQuestion]:
        """Generate question suggestions based on context"""
        
        # 1. Build context
        context = await self._build_context(request)
        
        # 2. Generate questions using LLM
        questions = await self._generate_questions_with_llm(
            context=context,
            k=request.k,
            user_level=request.user_level,
            model_name=request.model_name,
            provider_type=request.provider_type
        )
        
        return questions
    
    async def _build_context(
        self, 
        request: QuestionSuggestionRequest
    ) -> Dict[str, Any]:
        """Build context from news and chat history"""
        
        context = {
            "user_level": request.user_level.value,
            "timestamp": datetime.now().isoformat(),
            "market_news": [],
            "chat_history": [],
            "mentioned_symbols": set()  # Track symbols mentioned in chat
        }
        
        # 1. Get latest news (5 most recent)
        try:
            logger.info("Fetching latest news for context...")
            news_list = await self.news_service.get_general_news(page=0)
            
            if news_list:
                latest_news = news_list[:5]
                context["market_news"] = [
                    {
                        "title": news.title,
                        "description": news.description,
                        "source": news.source_site,
                        "date": news.date,
                        "category": news.category
                    }
                    for news in latest_news
                ]
                
                # Extract symbols from news titles/descriptions
                for news in latest_news:
                    symbols = self._extract_symbols(f"{news.title} {news.description or ''}")
                    context["mentioned_symbols"].update(symbols)
                    
                logger.info(f"Added {len(latest_news)} news items to context")
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
        
        # 2. Get chat history if session_id provided
        if request.session_id:
            try:
                logger.info(f"Fetching chat history for session: {request.session_id}")
                chat_history = await self.chat_service.get_chat_history(
                    session_id=request.session_id,
                    limit=3
                )
                
                if chat_history:
                    context["chat_history"] = [
                        {
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat() if hasattr(msg.timestamp, 'isoformat') else str(msg.timestamp)
                        }
                        for msg in chat_history
                    ]
                    
                    # Extract symbols from chat history
                    for msg in chat_history:
                        symbols = self._extract_symbols(msg.content)
                        context["mentioned_symbols"].update(symbols)
                        
                    logger.info(f"Added {len(chat_history)} chat messages to context")
            except Exception as e:
                logger.error(f"Error fetching chat history: {e}")
        
        # Convert set to list for JSON serialization
        context["mentioned_symbols"] = list(context["mentioned_symbols"])
        
        return context
    
    def _extract_symbols(self, text: str) -> set:
        """Extract stock symbols from text"""
        import re
        symbols = set()
        
        # Pattern for stock symbols (2-5 uppercase letters, optionally followed by -USD for crypto)
        pattern = r'\b[A-Z]{2,5}(?:-USD)?\b'
        matches = re.findall(pattern, text)
        
        # Filter to keep only likely stock symbols
        for match in matches:
            # Skip common words that match pattern
            if match not in ['CEO', 'USA', 'NYSE', 'IPO', 'GDP', 'API', 'USD', 'EUR']:
                symbols.add(match)
        
        return symbols
    
    async def _generate_questions_with_llm(
        self,
        context: Dict[str, Any],
        k: int,
        user_level: UserLevel,
        model_name: str,
        provider_type: str
    ) -> List[SuggestedQuestion]:
        """Generate questions using LLM based on context"""
        
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)

        # Prepare prompt
        prompt = self._create_prompt(context, k, user_level)
        
        # Call LLM with configuration from settings
        try:
            response = await llm_service_v1.generate(
                prompt=prompt,
                temperature=0.7,
                max_tokens=1500,
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            print(f"abc {response}")
            # Parse response
            questions = self._parse_llm_response(response)
            
            # Ensure we have enough questions
            if len(questions) < k:
                # Add fallback questions
                fallback = self._get_fallback_questions(k - len(questions), user_level, context)
                questions.extend(fallback)
            
            return questions[:k]
            
        except Exception as e:
            logger.error(f"Error generating questions with LLM: {e}")
            # Return fallback questions
            return self._get_fallback_questions(k, user_level, context)
    
    def _create_prompt(
        self,
        context: Dict[str, Any],
        k: int,
        user_level: UserLevel
    ) -> str:
        """Create prompt for LLM"""
        
        # Get random symbols for examples
        example_stocks = random.sample(self.popular_stocks, min(3, len(self.popular_stocks)))
        example_crypto = random.sample(self.popular_crypto, min(2, len(self.popular_crypto)))
        
        prompt = f"""You are a stock market assistant helping generate relevant questions for a {user_level.value} investor.

APPLICATION CAPABILITIES:
Our application supports these specific tools:

1. Stock Price & Valuation Analysis
   - Current price, performance metrics
   - Price targets and valuation metrics
   - Examples: "How is AAPL performing?", "What's TSLA's price target?"

2. Financial Analysis
   - Earnings reports, revenue, profit margins
   - P/E ratios, balance sheets, cash flow
   - Examples: "Show me MSFT's earnings", "GOOGL revenue growth"

3. Technical Analysis & Charts
   - Visual charts and technical indicators
   - Price patterns and trends
   - Examples: "Show NVDA chart", "BTC-USD technical analysis"

4. Stock News
   - Latest news and events
   - Earnings announcements
   - Examples: "Latest news on META", "Why is AMD moving?"

5. Market Overview
   - Market indices (S&P 500, NASDAQ, DOW)
   - Sector performance and sentiment
   - Examples: "How's the market today?", "Market sentiment"

6. Trending Stocks
   - Top gainers and losers
   - Most active stocks
   - Examples: "What's trending?", "Top gainers today"

7. Stock Heatmap
   - Visual market heatmap
   - Sector and industry performance
   - Examples: "Show heatmap", "Tech sector heatmap"

8. Cryptocurrency Analysis
   - Crypto prices and charts
   - Supports: BTC-USD, ETH-USD, etc.
   - Examples: "BTC-USD price", "Ethereum chart"

CURRENT CONTEXT:
"""
        
        # Add recent news to prompt
        if context["market_news"]:
            prompt += "\nRecent Market News:\n"
            for i, news in enumerate(context["market_news"][:3], 1):
                prompt += f"{i}. {news['title']} ({news['date']})\n"
        
        # Add mentioned symbols
        if context.get("mentioned_symbols"):
            prompt += f"\nSymbols mentioned: {', '.join(context['mentioned_symbols'][:5])}\n"
        
        # Add chat history context if available
        if context["chat_history"]:
            prompt += "\nRecent conversation topics:\n"
            for msg in context["chat_history"][-3:]:
                if msg["role"] == "user":
                    prompt += f"- {msg['content'][:100]}...\n"
        
        prompt += f"""

REQUIREMENTS:
- Generate exactly {k} questions suitable for a {user_level.value} investor
- Questions should be specific and actionable
- Include stock symbols when relevant (e.g., AAPL, TSLA, BTC-USD)
- Mix different types of analysis (price, technical, fundamental, news)
- If news mentions specific stocks, include questions about those stocks
- For beginner level, focus on educational and basic analysis questions
- For intermediate/advanced, include more technical and strategic questions

EXAMPLES OF GOOD QUESTIONS:
- "What's the current price and performance of {example_stocks[0]}?"
- "Show me the technical chart for {example_stocks[1]}"
- "What's the latest news on {example_stocks[2]}?"
- "How are tech stocks performing today?"
- "What are the top gainers in the market?"
- "Show me {example_crypto[0]} price analysis"

OUTPUT FORMAT (JSON):
[
    {{
        "question": "Specific question with symbol if applicable",
        "category": "price|technical|fundamental|news|market|trending|crypto",
        "relevance_reason": "Why this question is relevant"
    }}
]

Generate {k} diverse, relevant questions:"""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> List[SuggestedQuestion]:
        """Parse LLM response to list of questions"""
        try:
            # Try to extract JSON from response
            # Handle case where LLM might add extra text
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                questions_data = json.loads(json_str)
            else:
                # Try parsing entire response
                questions_data = json.loads(response)
            
            questions = []
            for q_data in questions_data:
                if isinstance(q_data, dict) and q_data.get("question"):
                    question = SuggestedQuestion(
                        question=q_data.get("question", ""),
                        category=q_data.get("category", "general"),
                        relevance_reason=q_data.get("relevance_reason", "Relevant to current market context")
                    )
                    questions.append(question)
            
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            return []
    
    def _get_fallback_questions(
        self, 
        k: int, 
        user_level: UserLevel,
        context: Optional[Dict[str, Any]] = None
    ) -> List[SuggestedQuestion]:
        """Generate fallback questions when LLM fails"""
        
        # Use mentioned symbols or popular ones
        symbols = []
        if context and context.get("mentioned_symbols"):
            symbols = context["mentioned_symbols"][:3]
        if len(symbols) < 3:
            symbols.extend(random.sample(self.popular_stocks, 3 - len(symbols)))
        
        beginner_questions = [
            SuggestedQuestion(
                question=f"What's the current price and performance of {symbols[0]}?",
                category="price",
                relevance_reason="Track price movements of popular stocks"
            ),
            SuggestedQuestion(
                question="What are the top gainers in the market today?",
                category="trending",
                relevance_reason="Discover high-performing stocks"
            ),
            SuggestedQuestion(
                question=f"Show me the latest news on {symbols[1]}",
                category="news",
                relevance_reason="Stay updated with market events"
            ),
            SuggestedQuestion(
                question="How is the overall market performing today?",
                category="market",
                relevance_reason="Understand market conditions"
            ),
            SuggestedQuestion(
                question=f"Can you show me a chart for {symbols[2]}?",
                category="technical",
                relevance_reason="Visualize price trends"
            ),
            SuggestedQuestion(
                question="What are the biggest losers today?",
                category="trending",
                relevance_reason="Identify underperforming stocks"
            ),
            SuggestedQuestion(
                question=f"What's the financial health of {symbols[0]}?",
                category="fundamental",
                relevance_reason="Understand company fundamentals"
            ),
            SuggestedQuestion(
                question="Show me the market heatmap",
                category="market",
                relevance_reason="Visualize sector performance"
            )
        ]
        
        intermediate_questions = [
            SuggestedQuestion(
                question=f"What's the technical analysis for {symbols[0]}?",
                category="technical",
                relevance_reason="Advanced chart analysis"
            ),
            SuggestedQuestion(
                question=f"Compare the P/E ratios of {symbols[0]} vs {symbols[1]}",
                category="fundamental",
                relevance_reason="Comparative valuation analysis"
            ),
            SuggestedQuestion(
                question="Which sectors are outperforming today?",
                category="market",
                relevance_reason="Sector rotation insights"
            ),
            SuggestedQuestion(
                question=f"What's the options flow for {symbols[2]}?",
                category="technical",
                relevance_reason="Advanced market sentiment"
            )
        ]
        
        # Select questions based on level
        if user_level == UserLevel.BEGINNER:
            questions = beginner_questions
        elif user_level == UserLevel.INTERMEDIATE:
            questions = beginner_questions[:4] + intermediate_questions[:4]
        else:  # Advanced
            questions = intermediate_questions + beginner_questions
        
        # Shuffle and return requested number
        random.shuffle(questions)
        return questions[:k]
    



    # NEW
    async def generate_question_suggestions_v2(
        self,
        request: QuestionSuggestionRequest,
        k: int = 8
    ) -> List[SuggestedQuestion_v2]:
        """
        Generate question suggestions based on question type, user level, and asset type
        
        Args:
            request: Request object containing question_type, user_level, asset_type, model_name, provider_type
            k: Number of questions to generate
            
        Returns:
            List of suggested questions filtered by asset type
        """
        try:
            # 1. Build context from news and events
            context = await self._build_context_v2(request)
            
            # 2. Generate questions using LLM
            questions = await self._generate_questions_with_llm_v2(
                context=context,
                k=k,
                question_type=request.question_type,
                user_level=request.user_level,
                model_name=request.model_name,
                provider_type=request.provider_type
            )
            
            return questions
            
        except Exception as e:
            logger.error(f"Error generating question suggestions: {e}")
            # Return fallback questions with asset_type
            return self._get_fallback_questions_v2(k, request.question_type, request.user_level, request.asset_type)
    
    # async def _build_context_v2(self, request: QuestionSuggestionRequest) -> Dict[str, Any]:
    #     """Build context from news and events based on question type"""
    #     context = {
    #         "question_type": request.question_type,
    #         "user_level": request.user_level,
    #         "news": [],
    #         "events": [],
    #         "symbols": set(),
    #         "current_date": datetime.now().strftime("%Y-%m-%d")
    #     }
        
    #     try:
    #         # 1. Fetch news based on question type
    #         if request.question_type == QuestionType.TECHNICAL:
    #             # Get stock news for technical analysis
    #             # Using the API endpoint logic from router
    #             from src.routers.equity import get_stock_news
                
    #             # Call the stock news endpoint
    #             result = await get_stock_news(page=0, limit=10, redis_client=None)
                
    #             if result and hasattr(result, 'data') and result.data and hasattr(result.data, 'data'):
    #                 news_items = result.data.data
    #                 context["news"] = [
    #                     {
    #                         "title": item.title,
    #                         "text": item.text[:200] if hasattr(item, 'text') else "",
    #                         "symbol": item.symbol if hasattr(item, 'symbol') else None,
    #                         "publishedDate": item.publishedDate if hasattr(item, 'publishedDate') else ""
    #                     }
    #                     for item in news_items[:5]  # Top 5 news
    #                 ]
    #                 # Extract symbols
    #                 context["symbols"].update([item.symbol for item in news_items if hasattr(item, 'symbol') and item.symbol])
                    
    #         elif request.question_type == QuestionType.CRYPTO:
    #             # Get crypto news
    #             from src.routers.equity import get_crypto_news
                
    #             result = await get_crypto_news(page=0, limit=10, redis_client=None)
                
    #             if result and hasattr(result, 'data') and result.data and hasattr(result.data, 'data'):
    #                 news_items = result.data.data
    #                 context["news"] = [
    #                     {
    #                         "title": item.title,
    #                         "text": item.text[:200] if hasattr(item, 'text') else "",
    #                         "symbol": item.symbol if hasattr(item, 'symbol') else None,
    #                         "publishedDate": item.publishedDate if hasattr(item, 'publishedDate') else ""
    #                     }
    #                     for item in news_items[:5]
    #                 ]
    #                 context["symbols"].update([item.symbol for item in news_items if hasattr(item, 'symbol') and item.symbol])
                    
    #         else:
    #             # Get general news for Fundamental and Sentiment
    #             news_list = await self.news_service.get_general_news(page=0)
    #             if news_list:
    #                 context["news"] = [
    #                     {
    #                         "title": item.title,
    #                         "text": item.description[:200] if item.description else "",
    #                         "symbol": getattr(item, 'symbol', None),
    #                         "publishedDate": item.date
    #                     }
    #                     for item in news_list[:5]
    #                 ]
    #                 # Extract symbols from news
    #                 for item in news_list[:5]:
    #                     if hasattr(item, 'symbol') and item.symbol:
    #                         context["symbols"].add(item.symbol)
    #                     # Also try to extract from title/description
    #                     text = f"{item.title} {item.description or ''}"
    #                     extracted = self._extract_symbols_v2(text)
    #                     context["symbols"].update(extracted)
            
    #         # 2. Fetch events (press releases) for Technical Expert  
    #         if request.question_type == QuestionType.TECHNICAL:
    #             events_data = await self.news_service.get_latest_press_releases(page=0, limit=5)
    #             if events_data:
    #                 context["events"] = [
    #                     {
    #                         "title": event.title,
    #                         "date": event.published_date,
    #                         "symbol": event.symbol if hasattr(event, 'symbol') else None
    #                     }
    #                     for event in events_data[:3]  # Top 3 events
    #                 ]
    #                 # Extract symbols from events
    #                 context["symbols"].update([event.symbol for event in events_data if hasattr(event, 'symbol') and event.symbol])
            
    #         # Convert symbols set to list for easier use
    #         context["symbols"] = list(context["symbols"])[:10]  # Limit to 10 symbols
            
    #     except Exception as e:
    #         logger.error(f"Error building context: {e}")
    #         # Return basic context on error
    #         context["symbols"] = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]  # Default symbols
            
    #     return context

    async def _build_context_v2(self, request: QuestionSuggestionRequest) -> Dict[str, Any]:
        """Build context from news and events based on question type and asset type"""
        context = {
            "question_type": request.question_type,
            "user_level": request.user_level,
            "asset_type": request.asset_type,
            "news": [],
            "events": [],
            "symbols": set(),
            "current_date": datetime.now().strftime("%Y-%m-%d")
        }
        
        try:
            # Determine which news source to use based on asset_type
            if request.asset_type == "crypto":
                # Always fetch crypto news for crypto asset type
                from src.routers.equity import get_crypto_news
                
                result = await get_crypto_news(page=0, limit=10, redis_client=None)
                
                if result and hasattr(result, 'data') and result.data and hasattr(result.data, 'data'):
                    news_items = result.data.data
                    context["news"] = [
                        {
                            "title": item.title,
                            "text": item.text[:200] if hasattr(item, 'text') else "",
                            "symbol": item.symbol if hasattr(item, 'symbol') else None,
                            "publishedDate": item.publishedDate if hasattr(item, 'publishedDate') else ""
                        }
                        for item in news_items[:5]
                    ]
                    context["symbols"].update([item.symbol for item in news_items if hasattr(item, 'symbol') and item.symbol])
                    
            elif request.asset_type == "stock":
                # Fetch stock news for stock asset type
                if request.question_type == QuestionType.TECHNICAL or request.question_type == QuestionType.FUNDAMENTAL:
                    # Get stock news for technical/fundamental analysis
                    from src.routers.equity import get_stock_news
                    
                    result = await get_stock_news(page=0, limit=10, redis_client=None)
                    
                    if result and hasattr(result, 'data') and result.data and hasattr(result.data, 'data'):
                        news_items = result.data.data
                        context["news"] = [
                            {
                                "title": item.title,
                                "text": item.text[:200] if hasattr(item, 'text') else "",
                                "symbol": item.symbol if hasattr(item, 'symbol') else None,
                                "publishedDate": item.publishedDate if hasattr(item, 'publishedDate') else ""
                            }
                            for item in news_items[:5]
                        ]
                        context["symbols"].update([item.symbol for item in news_items if hasattr(item, 'symbol') and item.symbol])
                        
                else:
                    # Get general news for Sentiment
                    news_list = await self.news_service.get_general_news(page=0)
                    if news_list:
                        context["news"] = [
                            {
                                "title": item.title,
                                "text": item.description[:200] if item.description else "",
                                "symbol": getattr(item, 'symbol', None),
                                "publishedDate": item.date
                            }
                            for item in news_list[:5]
                        ]
                        # Extract symbols from news
                        for item in news_list[:5]:
                            if hasattr(item, 'symbol') and item.symbol:
                                context["symbols"].add(item.symbol)
                            # Also try to extract from title/description
                            text = f"{item.title} {item.description or ''}"
                            extracted = self._extract_symbols_v2(text)
                            # Filter crypto symbols if asset_type is stock
                            extracted_filtered = {s for s in extracted if not s.endswith('-USD')}
                            context["symbols"].update(extracted_filtered)
            
            # Fetch events (press releases) for stock technical analysis
            if request.asset_type == "stock" and request.question_type == QuestionType.TECHNICAL:
                events_data = await self.news_service.get_latest_press_releases(page=0, limit=5)
                if events_data:
                    context["events"] = [
                        {
                            "title": event.title,
                            "date": event.published_date,
                            "symbol": event.symbol if hasattr(event, 'symbol') else None
                        }
                        for event in events_data[:3]
                    ]
                    context["symbols"].update([event.symbol for event in events_data if hasattr(event, 'symbol') and event.symbol])
            
            # Convert symbols set to list for easier use
            context["symbols"] = list(context["symbols"])[:10]  # Limit to 10 symbols
            
        except Exception as e:
            logger.error(f"Error building context: {e}")
            # Return basic context on error based on asset_type
            if request.asset_type == "crypto":
                context["symbols"] = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOGE-USD"]
            else:  # stock
                context["symbols"] = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
            
        return context

    
    def _extract_symbols_v2(self, text: str) -> set:
        """Extract stock symbols from text"""
        import re
        symbols = set()
        
        # Pattern for stock symbols (2-5 uppercase letters, optionally followed by -USD for crypto)
        pattern = r'\b[A-Z]{2,5}(?:-USD)?\b'
        matches = re.findall(pattern, text)
        
        # Filter to keep only likely stock symbols
        for match in matches:
            # Skip common words that match pattern
            if match not in ['CEO', 'USA', 'NYSE', 'IPO', 'GDP', 'API', 'USD', 'EUR', 'THE', 'AND', 'FOR']:
                symbols.add(match)
        
        return symbols
    
    async def _generate_questions_with_llm_v2(
        self,
        context: Dict[str, Any],
        k: int,
        question_type: str,
        user_level: str,
        model_name: str,
        provider_type: str
    ) -> List[SuggestedQuestion_v2]:
        """Generate questions using LLM based on context"""
        
        try:
            # Get API key
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            # Create prompt
            prompt = self._create_prompt_v2(context, k, question_type, user_level)
            
            # Format messages
            messages = [
                {"role": "system", "content": self._get_system_prompt_v2(question_type)},
                {"role": "user", "content": prompt}
            ]
            
            # Call LLM
            response = await self.llm_service.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.7,
                max_tokens=1500
            )
            
            # Parse response
            content = response.get("content", "")
            questions = self._parse_llm_response_v2(content, context)
            
            # Ensure we have enough questions
            if len(questions) < k:
                fallback = self._get_fallback_questions_v2(k - len(questions), question_type, user_level)
                questions.extend(fallback)
            
            return questions[:k]
            
        except Exception as e:
            logger.error(f"Error generating questions with LLM: {e}")
            return self._get_fallback_questions_v2(k, question_type, user_level)
    
    def _get_system_prompt_v2(self, question_type: str) -> str:
        """Get system prompt based on question type"""
        
        base_prompt = """You are a financial market assistant helping users ask relevant questions about stocks, crypto, and market analysis.
Your task is to generate insightful questions that users can ask based on recent market news and events."""
        
        type_specific = {
            QuestionType.TECHNICAL: """
Focus on technical analysis questions including:
- Price movements and chart patterns
- Technical indicators (RSI, MACD, Moving Averages)
- Support and resistance levels
- Trading volume and momentum""",
            
            QuestionType.CRYPTO: """
Focus on cryptocurrency analysis questions including:
- Crypto price movements and market cap
- DeFi and blockchain technology trends
- Bitcoin dominance and altcoin performance
- Crypto regulatory news and adoption""",
            
            QuestionType.FUNDAMENTAL: """
Focus on fundamental analysis questions including:
- Financial statements and earnings
- Company valuations (P/E, P/B ratios)
- Revenue growth and profitability
- Industry comparisons and market position""",
            
            QuestionType.SENTIMENT: """
Focus on market sentiment questions including:
- News impact on stock prices
- Social media trends and investor sentiment
- Market fear and greed indicators
- Analyst ratings and institutional moves"""
        }
        
        return base_prompt + type_specific.get(question_type, "")
    
#     def _create_prompt_v2(
#         self,
#         context: Dict[str, Any],
#         k: int,
#         question_type: str,
#         user_level: str
#     ) -> str:
#         """Create prompt for LLM"""
        
#         # Format news items
#         news_text = ""
#         if context["news"]:
#             news_items = []
#             for news in context["news"][:3]:
#                 symbol = news.get("symbol", "N/A")
#                 title = news.get("title", "")
#                 news_items.append(f"- [{symbol}] {title}")
#             news_text = "\n".join(news_items)
        
#         # Format events
#         events_text = ""
#         if context["events"]:
#             event_items = []
#             for event in context["events"][:2]:
#                 symbol = event.get("symbol", "N/A")
#                 title = event.get("title", "")
#                 event_items.append(f"- [{symbol}] {title}")
#             events_text = "\n".join(event_items)
        
#         # Get available symbols
#         symbols = context["symbols"][:5] if context["symbols"] else ["AAPL", "MSFT", "GOOGL"]
        
#         prompt = f"""Generate {k} relevant questions for a {user_level} investor interested in {question_type.replace('_', ' ')}.

# Current Date: {context["current_date"]}

# Recent Market News:
# {news_text if news_text else "No recent news available"}

# Recent Events:
# {events_text if events_text else "No recent events available"}

# Active Symbols in the market: {', '.join(symbols)}

# Requirements:
# 1. Questions should be specific and actionable
# 2. Include relevant stock/crypto symbols when appropriate
# 3. Match the complexity to the {user_level} level
# 4. Focus on {question_type.replace('_', ' ')} perspective
# 5. Reference recent news/events when relevant

# Generate exactly {k} questions. Format each question on a new line starting with a number and period (e.g., "1. What is...").
# """
        
#         return prompt

    def _create_prompt_v2(
        self,
        context: Dict[str, Any],
        k: int,
        question_type: str,
        user_level: str
    ) -> str:
        """Create prompt for LLM based on asset type"""
        
        asset_type = context.get("asset_type", "stock")
        
        # Format news items
        news_text = ""
        if context["news"]:
            news_items = []
            for news in context["news"][:3]:
                symbol = news.get("symbol", "N/A")
                title = news.get("title", "")
                news_items.append(f"- [{symbol}] {title}")
            news_text = "\n".join(news_items)
        
        # Format events
        events_text = ""
        if context["events"]:
            event_items = []
            for event in context["events"][:2]:
                symbol = event.get("symbol", "N/A")
                title = event.get("title", "")
                event_items.append(f"- [{symbol}] {title}")
            events_text = "\n".join(event_items)
        
        # Get available symbols
        symbols = context["symbols"][:5] if context["symbols"] else []
        if not symbols:
            symbols = ["BTC-USD", "ETH-USD"] if asset_type == "crypto" else ["AAPL", "MSFT"]
        
        # Asset-specific instructions
        if asset_type == "crypto":
            asset_instruction = """
    IMPORTANT - CRYPTO FOCUS:
    - ALL questions MUST be about cryptocurrencies only
    - Use crypto symbols like BTC-USD, ETH-USD, SOL-USD, ADA-USD
    - Focus on crypto-specific topics: DeFi, blockchain, crypto adoption, regulatory news
    - DO NOT suggest questions about traditional stocks like AAPL, TSLA, MSFT
    """
            example_symbols = "BTC-USD, ETH-USD, SOL-USD"
        else:  # stock
            asset_instruction = """
    IMPORTANT - STOCK FOCUS:
    - ALL questions MUST be about stocks/equities only
    - Use stock symbols like AAPL, MSFT, GOOGL, TSLA, NVDA
    - Focus on traditional stock market topics: earnings, fundamentals, stock performance
    - DO NOT suggest questions about cryptocurrencies (BTC-USD, ETH-USD, etc.)
    """
            example_symbols = "AAPL, MSFT, GOOGL"
        
        prompt = f"""Generate {k} relevant questions for a {user_level} investor interested in {question_type.replace('_', ' ')}.

    {asset_instruction}

    Current Date: {context["current_date"]}

    Recent {asset_type.upper()} Market News:
    {news_text if news_text else f"No recent {asset_type} news available"}

    Recent Events:
    {events_text if events_text else "No recent events available"}

    Active {asset_type.upper()} Symbols: {', '.join(symbols)}

    Requirements:
    1. Questions should be specific and actionable
    2. Include relevant {asset_type} symbols when appropriate (e.g., {example_symbols})
    3. Match the complexity to the {user_level} level
    4. Focus on {question_type.replace('_', ' ')} perspective
    5. Reference recent {asset_type} news/events when relevant
    6. STRICTLY focus on {asset_type} market only

    Generate exactly {k} questions. Format each question on a new line starting with a number and period (e.g., "1. What is...").
    """
        
        return prompt

    
    def _parse_llm_response_v2(self, response: str, context: Dict[str, Any]) -> List[SuggestedQuestion_v2]:
        """Parse LLM response into SuggestedQuestion objects"""
        questions = []
        
        import re

        try:
            # Split response into lines and find numbered questions
            lines = response.strip().split('\n')
            
            for line in lines:
                # Match numbered questions (1. Question text)
                match = re.match(r'^\d+\.\s*(.+)$', line.strip())
                if match:
                    question_text = match.group(1).strip()
                    
                    # Determine category based on keywords
                    category = self._determine_category(question_text)
                    
                    # Extract symbol if present
                    symbol_match = re.search(r'\b([A-Z]{2,5})\b', question_text)
                    symbol = symbol_match.group(1) if symbol_match else None
                    
                    # Create relevance reason
                    relevance = self._generate_relevance_reason(question_text, context)
                    
                    questions.append(SuggestedQuestion_v2(
                        question=question_text,
                        category=category,
                        relevance_reason=relevance,
                        symbol=symbol
                    ))
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
        
        return questions
    
    def _determine_category(self, question: str) -> str:
        """Determine question category based on keywords"""
        
        technical_keywords = ["chart", "pattern", "resistance", "support", "indicator", "RSI", "MACD", "volume", "technical"]
        fundamental_keywords = ["earnings", "revenue", "profit", "P/E", "valuation", "financial", "growth", "margin"]
        sentiment_keywords = ["news", "sentiment", "analyst", "rating", "trend", "social", "fear", "greed"]
        market_keywords = ["market", "sector", "industry", "index", "S&P", "Dow", "Nasdaq"]
        
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in technical_keywords):
            return "technical"
        elif any(keyword in question_lower for keyword in fundamental_keywords):
            return "fundamental"
        elif any(keyword in question_lower for keyword in sentiment_keywords):
            return "sentiment"
        elif any(keyword in question_lower for keyword in market_keywords):
            return "market"
        else:
            return "general"
    
    def _generate_relevance_reason(self, question: str, context: Dict[str, Any]) -> str:
        """Generate relevance reason for the question"""
        
        # Check if question relates to recent news
        for news in context.get("news", []):
            if news.get("symbol") and news["symbol"] in question:
                return f"Related to recent news about {news['symbol']}"
        
        # Check if question relates to events
        for event in context.get("events", []):
            if event.get("symbol") and event["symbol"] in question:
                return f"Related to recent event for {event['symbol']}"
        
        # Default reasons based on category
        if "earnings" in question.lower():
            return "Earnings season analysis"
        elif "technical" in question.lower() or "chart" in question.lower():
            return "Technical analysis opportunity"
        elif "news" in question.lower():
            return "Recent market developments"
        else:
            return "Market insight question"
    
    # def _get_fallback_questions_v2(self, k: int, question_type: str, user_level: str) -> List[SuggestedQuestion_v2]:
    #     """Get fallback questions when LLM fails"""
        
    #     fallback_pools = {
    #         QuestionType.TECHNICAL: [
    #             {"q": "What's the technical analysis for AAPL?", "c": "technical", "r": "Popular stock analysis"},
    #             {"q": "Show me the TSLA price chart with indicators", "c": "technical", "r": "Chart analysis request"},
    #             {"q": "What are the support levels for NVDA?", "c": "technical", "r": "Support/resistance analysis"},
    #             {"q": "Is MSFT showing bullish patterns?", "c": "technical", "r": "Pattern recognition"},
    #             {"q": "What's the RSI reading for SPY?", "c": "technical", "r": "Technical indicator check"},
    #         ],
    #         QuestionType.CRYPTO: [
    #             {"q": "What's the current Bitcoin price trend?", "c": "crypto", "r": "BTC market analysis"},
    #             {"q": "How is Ethereum performing today?", "c": "crypto", "r": "ETH performance check"},
    #             {"q": "What are the top gaining cryptocurrencies?", "c": "crypto", "r": "Crypto market movers"},
    #             {"q": "Is it a good time to buy Bitcoin?", "c": "crypto", "r": "BTC investment timing"},
    #             {"q": "What's the crypto market sentiment?", "c": "sentiment", "r": "Market sentiment analysis"},
    #         ],
    #         QuestionType.FUNDAMENTAL: [
    #             {"q": "What are AAPL's latest earnings?", "c": "fundamental", "r": "Earnings analysis"},
    #             {"q": "How is MSFT's revenue growing?", "c": "fundamental", "r": "Revenue growth check"},
    #             {"q": "What's the P/E ratio for GOOGL?", "c": "fundamental", "r": "Valuation metrics"},
    #             {"q": "Which tech stocks have the best fundamentals?", "c": "fundamental", "r": "Sector comparison"},
    #             {"q": "What's TSLA's profit margin trend?", "c": "fundamental", "r": "Profitability analysis"},
    #         ],
    #         QuestionType.SENTIMENT: [
    #             {"q": "What's the market sentiment today?", "c": "sentiment", "r": "Overall market mood"},
    #             {"q": "Why is NVDA stock moving?", "c": "sentiment", "r": "Stock movement analysis"},
    #             {"q": "What are analysts saying about AAPL?", "c": "sentiment", "r": "Analyst opinions"},
    #             {"q": "Which stocks are trending on social media?", "c": "sentiment", "r": "Social trends"},
    #             {"q": "Is there fear or greed in the market?", "c": "sentiment", "r": "Market psychology"},
    #         ]
    #     }
        
    #     pool = fallback_pools.get(question_type, fallback_pools[QuestionType.TECHNICAL])
        
    #     # Adjust complexity based on user level
    #     if user_level == UserLevel.BEGINNER:
    #         # Filter for simpler questions
    #         pool = [q for q in pool if "ratio" not in q["q"].lower() and "RSI" not in q["q"]]
        
    #     # Randomly select k questions
    #     selected = random.sample(pool, min(k, len(pool)))
        
    #     return [
    #         SuggestedQuestion_v2(
    #             question=item["q"],
    #             category=item["c"],
    #             relevance_reason=item["r"]
    #         )
    #         for item in selected
    #     ]

    def _get_fallback_questions_v2(self, k: int, question_type: str, user_level: str, asset_type: str = "stock") -> List[SuggestedQuestion_v2]:
        """Get fallback questions when LLM fails, filtered by asset type"""
        
        if asset_type == "crypto":
            # Crypto-focused fallback questions
            fallback_pools = {
                QuestionType.TECHNICAL: [
                    {"q": "What's the technical analysis for BTC-USD?", "c": "technical", "r": "Bitcoin analysis"},
                    {"q": "Show me the ETH-USD price chart with indicators", "c": "technical", "r": "Ethereum chart analysis"},
                    {"q": "What are the support levels for SOL-USD?", "c": "technical", "r": "Solana support analysis"},
                    {"q": "Is Bitcoin showing bullish patterns?", "c": "technical", "r": "BTC pattern recognition"},
                    {"q": "What's the RSI reading for Ethereum?", "c": "technical", "r": "ETH technical indicator"},
                ],
                QuestionType.CRYPTO: [
                    {"q": "What's the current Bitcoin price trend?", "c": "crypto", "r": "BTC market analysis"},
                    {"q": "How is Ethereum performing today?", "c": "crypto", "r": "ETH performance check"},
                    {"q": "What are the top gaining cryptocurrencies?", "c": "crypto", "r": "Crypto market movers"},
                    {"q": "Is it a good time to buy Bitcoin?", "c": "crypto", "r": "BTC investment timing"},
                    {"q": "What's the crypto market sentiment?", "c": "sentiment", "r": "Crypto sentiment analysis"},
                ],
                QuestionType.FUNDAMENTAL: [
                    {"q": "What's the market cap of Bitcoin?", "c": "fundamental", "r": "BTC valuation"},
                    {"q": "How is crypto adoption growing?", "c": "fundamental", "r": "Adoption metrics"},
                    {"q": "What's the trading volume for Ethereum?", "c": "fundamental", "r": "ETH volume analysis"},
                    {"q": "Which cryptocurrencies have strong fundamentals?", "c": "fundamental", "r": "Crypto comparison"},
                    {"q": "What's driving Solana's growth?", "c": "fundamental", "r": "SOL growth analysis"},
                ],
                QuestionType.SENTIMENT: [
                    {"q": "What's the crypto market sentiment today?", "c": "sentiment", "r": "Overall crypto mood"},
                    {"q": "Why is Bitcoin price moving?", "c": "sentiment", "r": "BTC movement analysis"},
                    {"q": "What are analysts saying about Ethereum?", "c": "sentiment", "r": "ETH analyst opinions"},
                    {"q": "Which cryptocurrencies are trending on social media?", "c": "sentiment", "r": "Crypto social trends"},
                    {"q": "Is there fear or greed in the crypto market?", "c": "sentiment", "r": "Crypto market psychology"},
                ]
            }
        else:  # stock
            # Stock-focused fallback questions
            fallback_pools = {
                QuestionType.TECHNICAL: [
                    {"q": "What's the technical analysis for AAPL?", "c": "technical", "r": "Apple analysis"},
                    {"q": "Show me the TSLA price chart with indicators", "c": "technical", "r": "Tesla chart analysis"},
                    {"q": "What are the support levels for NVDA?", "c": "technical", "r": "NVIDIA support analysis"},
                    {"q": "Is MSFT showing bullish patterns?", "c": "technical", "r": "Microsoft pattern recognition"},
                    {"q": "What's the RSI reading for SPY?", "c": "technical", "r": "S&P 500 technical indicator"},
                ],
                QuestionType.CRYPTO: [
                    {"q": "How are tech stocks performing?", "c": "market", "r": "Tech sector performance"},
                    {"q": "What's driving semiconductor stocks?", "c": "fundamental", "r": "Chip sector analysis"},
                    {"q": "Are growth stocks outperforming?", "c": "market", "r": "Growth vs value"},
                    {"q": "What's the outlook for AI stocks?", "c": "sentiment", "r": "AI sector sentiment"},
                    {"q": "Which sectors are leading the market?", "c": "market", "r": "Sector rotation"},
                ],
                QuestionType.FUNDAMENTAL: [
                    {"q": "What are AAPL's latest earnings?", "c": "fundamental", "r": "Apple earnings"},
                    {"q": "How is MSFT's revenue growing?", "c": "fundamental", "r": "Microsoft revenue growth"},
                    {"q": "What's the P/E ratio for GOOGL?", "c": "fundamental", "r": "Google valuation"},
                    {"q": "Which tech stocks have the best fundamentals?", "c": "fundamental", "r": "Tech sector comparison"},
                    {"q": "What's TSLA's profit margin trend?", "c": "fundamental", "r": "Tesla profitability"},
                ],
                QuestionType.SENTIMENT: [
                    {"q": "What's the market sentiment today?", "c": "sentiment", "r": "Overall market mood"},
                    {"q": "Why is NVDA stock moving?", "c": "sentiment", "r": "NVIDIA movement analysis"},
                    {"q": "What are analysts saying about AAPL?", "c": "sentiment", "r": "Apple analyst opinions"},
                    {"q": "Which stocks are trending on social media?", "c": "sentiment", "r": "Stock social trends"},
                    {"q": "Is there fear or greed in the stock market?", "c": "sentiment", "r": "Market psychology"},
                ]
            }
        
        pool = fallback_pools.get(question_type, fallback_pools[QuestionType.TECHNICAL])
        
        # Adjust complexity based on user level
        if user_level == UserLevel.BEGINNER:
            # Filter for simpler questions
            pool = [q for q in pool if "ratio" not in q["q"].lower() and "RSI" not in q["q"]]
        
        # Randomly select k questions
        selected = random.sample(pool, min(k, len(pool)))
        
        return [
            SuggestedQuestion_v2(
                question=item["q"],
                category=item["c"],
                relevance_reason=item["r"]
            )
            for item in selected
        ]
        

    # TODO: 
    async def generate_question_suggestions_with_context(
        self,
        data: QuestionSuggestionRequest
    ) -> List[SuggestedQuestion]:
        """
        Enhanced question generation with conversation summary context
        """
        try:
            # Get or create summary from cache
            summary_data = None
            if data.session_id:
                summary_data = await self.summary_service.get_or_create_summary(
                    session_id=data.session_id
                )
                
                # Trigger background update for next time
                await self.summary_service.trigger_background_update(data.session_id)
            
            # Build context with summary
            context = await self._build_context_with_summary(
                request=data,
                summary_data=summary_data
            )
            
            # Generate questions
            questions = await self._generate_smart_questions(
                context=context,
                k=data.k,
                user_level=data.user_level,
                model_name=data.model_name,
                provider_type=data.provider_type
            )
            
            logger.info(f"Generated {len(questions)} questions with summary context")
            return questions
            
        except Exception as e:
            logger.error(f"Error in question generation with summary: {e}")
            return await self.generate_question_suggestions(data)

    async def _build_context_with_summary(
        self,
        request: QuestionSuggestionRequest,
        summary_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build context combining latest message and summary"""
        
        context = {
            "user_level": request.user_level.value,
            "timestamp": datetime.now().isoformat(),
            "latest_message": None,
            "summary": summary_data,
            "mentioned_symbols": set(),
            "combined_topics": [],
            "unresolved_points": []
        }
        
        # Get latest message
        if request.session_id:
            try:
                latest_history = self.chat_service.get_chat_history(
                    session_id=request.session_id,
                    limit=1
                )
                
                if latest_history:
                    latest_content, latest_role = latest_history[0]
                    context["latest_message"] = {
                        "content": latest_content,
                        "role": latest_role
                    }
                    
                    # Extract from latest message
                    context["mentioned_symbols"].update(self._extract_symbols(latest_content))
                    
            except Exception as e:
                logger.error(f"Error getting latest message: {e}")
        
        # Merge summary data if available
        if summary_data:
            # Add summary text
            context["conversation_summary"] = summary_data.get("summary", "")
            
            # Merge key points
            context["key_points"] = summary_data.get("key_points", [])
            
            # Combine topics
            context["combined_topics"] = summary_data.get("topics", [])
            
            # Add unresolved points from summary
            context["unresolved_points"] = summary_data.get("unresolved", [])
            
            # Add symbols from summary
            if summary_data.get("mentioned_symbols"):
                context["mentioned_symbols"].update(summary_data["mentioned_symbols"])
            
            # Add metrics mentioned
            context["mentioned_metrics"] = summary_data.get("mentioned_metrics", [])
        
        # Convert set to list
        context["mentioned_symbols"] = list(context["mentioned_symbols"])
        
        return context


    async def _generate_smart_questions(
        self,
        context: Dict[str, Any],
        k: int,
        user_level: UserLevel,
        model_name: str,
        provider_type: str
    ) -> List[SuggestedQuestion]:
        """Generate questions using summary + latest message context"""
        
        try:
            api_key = ModelProviderFactory._get_api_key(provider_type)
            llm = await self.llm_service.get_llm(
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )
            
            # Build enhanced prompt
            messages = self._build_summary_based_prompt(context, k, user_level)
            
            # Generate response
            response = await llm.generate(messages)
            
            # Parse response
            response_content = self._extract_response_content(response)
            questions = self._parse_smart_response(response_content)
            
            # Add fallback if needed
            if len(questions) < k:
                fallback = self._get_context_aware_fallback(
                    context, k - len(questions), user_level
                )
                questions.extend(fallback)
            
            return questions[:k]
            
        except Exception as e:
            logger.error(f"Error generating smart questions: {e}")
            return self._get_context_aware_fallback(context, k, user_level)
        
    
    def _build_summary_based_prompt(
        self,
        context: Dict[str, Any],
        k: int,
        user_level: UserLevel
    ) -> List[Dict[str, str]]:
        """Build prompt using summary and latest message"""
        
        latest_msg = context.get("latest_message", {})
        summary = context.get("conversation_summary", "")
        
        system_prompt = """You are an intelligent financial assistant that generates highly relevant follow-up questions.
    You have access to both a conversation summary and the latest message.
    Your goal is to generate questions that:
    1. Address unresolved points from the conversation
    2. Naturally follow from the latest response
    3. Help users explore topics they've shown interest in
    4. Anticipate their next information needs"""
        
        user_prompt = f"""Generate {k} follow-up questions based on this context:

    USER LEVEL: {user_level.value}

    CONVERSATION SUMMARY (from previous 5 messages):
    {summary if summary else "No previous conversation"}

    KEY POINTS DISCUSSED:
    {', '.join(context.get('key_points', [])) if context.get('key_points') else 'None'}

    UNRESOLVED QUESTIONS:
    {', '.join(context.get('unresolved_points', [])) if context.get('unresolved_points') else 'None'}

    LATEST MESSAGE:
    Role: {latest_msg.get('role', 'unknown')}
    Content: {latest_msg.get('content', '')[:500]}

    CONTEXT INSIGHTS:
    - Symbols discussed: {', '.join(context.get('mentioned_symbols', [])[:5])}
    - Topics covered: {', '.join(context.get('combined_topics', []))}
    - Metrics mentioned: {', '.join(context.get('mentioned_metrics', []))}

    Generate {k} questions that:
    1. First priority: Address any unresolved points
    2. Second: Build on the latest message content
    3. Third: Explore topics from the summary that need clarification
    4. Match {user_level.value} expertise level
    5. Are specific and immediately actionable

    Output JSON array only:
    [
    {{
        "question": "specific question text",
        "category": "technical|fundamental|sentiment|market|strategy",
        "relevance_reason": "why this follows from the conversation"
    }}
    ]"""
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]


    def _parse_smart_response(self, response_content: str) -> List[SuggestedQuestion]:
        """Parse response from smart question generation"""
        
        # Clean thinking tags
        response_content = self.llm_service.clean_thinking(response_content)
        
        try:
            # Extract JSON
            json_start = response_content.find('[')
            json_end = response_content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_content[json_start:json_end]
                questions_data = json.loads(json_str)
            else:
                questions_data = json.loads(response_content)
            
            questions = []
            for q_data in questions_data:
                questions.append(
                    SuggestedQuestion(
                        question=q_data["question"],
                        category=q_data.get("category", "general"),
                        relevance_reason=q_data.get("relevance_reason", "Based on conversation context")
                    )
                )
            
            return questions
            
        except Exception as e:
            logger.error(f"Error parsing smart response: {e}")
            return []

    def _extract_response_content(self, response: Any) -> str:
        """Extract content from various response formats"""
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, dict):
            return response.get('content', '') or response.get('message', '') or str(response)
        else:
            return str(response)
        
    def _get_context_aware_fallback(
        self,
        context: Dict[str, Any],
        k: int,
        user_level: UserLevel
    ) -> List[SuggestedQuestion]:
        """Generate fallback questions based on context and summary"""
        
        questions = []
        
        # Priority 1: Address unresolved points
        if context.get("unresolved_points"):
            for point in context["unresolved_points"][:2]:
                questions.append(
                    SuggestedQuestion(
                        question=f"Can you clarify: {point}?",
                        category="general",
                        relevance_reason="Addressing unresolved point from conversation"
                    )
                )
        
        # Priority 2: Follow up on mentioned symbols
        if context.get("mentioned_symbols"):
            symbol = context["mentioned_symbols"][0]
            questions.append(
                SuggestedQuestion(
                    question=f"What's the outlook for {symbol} based on our discussion?",
                    category="strategy",
                    relevance_reason=f"Synthesizing {symbol} analysis"
                )
            )
        
        # Priority 3: Explore mentioned metrics
        if context.get("mentioned_metrics"):
            metric = context["mentioned_metrics"][0]
            questions.append(
                SuggestedQuestion(
                    question=f"How should I interpret the {metric} in this context?",
                    category="fundamental",
                    relevance_reason=f"Understanding {metric} implications"
                )
            )
        
        # Add level-specific questions
        if user_level == UserLevel.BEGINNER:
            questions.extend([
                SuggestedQuestion(
                    question="What are the key takeaways I should remember?",
                    category="general",
                    relevance_reason="Summarizing for clarity"
                ),
                SuggestedQuestion(
                    question="What should I research next to understand this better?",
                    category="strategy",
                    relevance_reason="Learning path guidance"
                )
            ])
        else:
            questions.extend([
                SuggestedQuestion(
                    question="What are the risks I haven't considered?",
                    category="strategy",
                    relevance_reason="Risk assessment"
                ),
                SuggestedQuestion(
                    question="How does this analysis compare to market consensus?",
                    category="market",
                    relevance_reason="Market perspective"
                )
            ])
        
        import random
        random.shuffle(questions)
        return questions[:k]