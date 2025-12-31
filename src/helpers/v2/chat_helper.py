import os
import json
import httpx
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, AsyncGenerator, List, Optional

from src.helpers.llm_helper import LLMGeneratorProvider
from src.helpers.redis_cache import get_cache, set_cache, get_redis_client_llm
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.services.news_service import NewsService
from src.models.equity import NewsItemOutput, APIResponse, APIResponseData
from src.utils.config import settings
from src.handlers.comprehensive_analysis_handler import ComprehensiveAnalysisHandler
from src.schemas.response import StockHeatmapPayload
from src.helpers.language_detector import language_detector, DetectionMethod


news_service = NewsService()
llm_provider = LLMGeneratorProvider()
analysis_instance = ComprehensiveAnalysisHandler()

# =============================================================================
# region General Chat APIs
# =============================================================================
def get_system_message_general_chat(enable_thinking: bool = True, model_name: str = "gpt-5-nano-2025-08-07", target_language: str = None) -> str:
    language_instruction = ""
    if target_language:
        lang_name = {
            "en": "English",
            "vi": "Vietnamese", 
            "zh": "Chinese",
            "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
            "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
        }.get(target_language, "the detected language")
        
        language_instruction = f"""
CRITICAL LANGUAGE REQUIREMENT:
You MUST respond ENTIRELY in {lang_name} language.
- ALL text, explanations, and analysis must be in {lang_name}
- Use appropriate financial terminology for {lang_name}
- Format numbers and dates according to {lang_name} conventions
"""
        
    return f"""You are ToponeLogic, a professional financial assistant specializing in stocks and cryptocurrency analysis.

{language_instruction}

LANGUAGE ADAPTATION:
- Detect and respond in the user's language automatically
- Use culturally appropriate financial terms and formats
- Maintain professional tone across all languages

CRITICAL INSTRUCTIONS:
1. ALWAYS carefully read and use the conversation history provided below
2. When answering questions, FIRST check the conversation history for relevant information
3. If the user asks about something mentioned in history, you MUST recall and use that information
4. Use specific numbers, dates, and metrics from context
5. Never invent financial data or make unsupported predictions
6. Adapt currency symbols, date formats, and number formats to user's region

RESPONSE STRUCTURE:
1. Answer the question general helpfully
2. For financial topics use context data to support your analysis if relevant and provide clear, actionable financial insights:
   - Analyze price trends, volume, fundamentals & technical indicators (stocks).  
   - Evaluate market cap, trading volume, price action & volatility (crypto).  
   - Highlight risks, opportunities and broader market context. 

TONE & STYLE:
- Professional yet accessible in user's native language
- Data-driven and objective across all languages
- Include relevant disclaimers using appropriate legal language
- Structure information clearly with local formatting preferences
- Use familiar financial terminology for the user's market/region

Remember: Think step-by-step detect language → analyze context → identify key insights → provide localized structured response."""

async def general_chat_bot(
    content: str, 
    system_language: str,
    chat_history: str, 
    model_name: str, 
    enable_thinking: bool = True,
    provider_type: str = ProviderType.OPENAI
) -> str:
    """Non-streaming chat with multi-provider support"""
    try:
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)
        
        if len(content.split()) < 2:
            detection_method = DetectionMethod.LLM
        else:
            detection_method = DetectionMethod.LIBRARY

        # Language detection
        language_info = await language_detector.detect(
            text=content,
            method=detection_method,
            system_language=system_language,
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key
        )

        detected_language = language_info["detected_language"]
        
        # System prompt
        system_message = get_system_message_general_chat(
            enable_thinking, 
            model_name, 
            detected_language
        )
        
        # QUESTION + History
        if chat_history and len(chat_history.strip()) > 0:
            user_content = f"""CURRENT QUESTION:
{content}

CONVERSATION HISTORY (For context only)
{chat_history}

Instructions: Answer the CURRENT QUESTION above in the SAME LANGUAGE it was asked, using relevant information from the history if needed."""
        else:
            user_content = content
            
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        # Ollama mdoel
        if provider_type == ProviderType.OLLAMA:
            async with httpx.AsyncClient(timeout=600.0) as client:
                base_url = os.getenv('OLLAMA_HOST')
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "No response.")
        else:
            response = await llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                enable_thinking=enable_thinking
            )
            return response.get("content", "No response.")
            
    except Exception as e:
        return f"Error contacting LLM: {str(e)}"


async def general_chat_bot_stream(
    content: str,
    system_language: str,
    chat_history: str, 
    model_name: str, 
    enable_thinking: bool = True,
    provider_type: str = ProviderType.OPENAI
) -> AsyncGenerator[str, None]:
    """Streaming version with multi-provider support"""
    try:
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)

        if len(content.split()) < 2:
            detection_method = DetectionMethod.LLM
        else:
            detection_method = DetectionMethod.LIBRARY

        # Language detection
        language_info = await language_detector.detect(
            text=content,
            method=detection_method,
            system_language=system_language,
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key
        )

        detected_language = language_info["detected_language"]

        # System prompt
        system_message = get_system_message_general_chat(enable_thinking, model_name, detected_language)

        if chat_history and len(chat_history.strip()) > 0:
            user_content = f"""=== CURRENT QUESTION (Please respond in this language) ===
{content}

=== CONVERSATION HISTORY (For context only) ===
{chat_history}

Instructions: Answer the CURRENT QUESTION above in the SAME LANGUAGE it was asked, using relevant information from the history if needed."""
        else:
            user_content = content
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        # Ollama model
        if provider_type == ProviderType.OLLAMA:
            base_url = os.getenv('OLLAMA_HOST')
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": True
                    }
                )
                
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith('data: '):
                        line = line[6:]
                    if line == '[DONE]':
                        break
                    try:
                        chunk = json.loads(line)
                        if 'message' in chunk and 'content' in chunk['message']:
                            content = chunk['message']['content']
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
        else:
            async for chunk in llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                clean_thinking=True,
                enable_thinking=enable_thinking
            ):
                yield chunk
                
    except Exception as e:
        yield f"Error contacting LLM: {str(e)}"


# =============================================================================
# region Market Overview APIs
# =============================================================================
async def analyze_market_overview(
    data: Any, 
    chat_history: str, 
    model_name: str,
    provider_type: str = ProviderType.OLLAMA
) -> str:
    """Market overview analysis with multi-provider support"""
    try:
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)

        # Get data
        market_data = data.data
        symbol_analyses = []
        for item in market_data:
            symbol = item.get("symbol", "Unknown")
            last_price = item.get("last_price", None)
            change = item.get("change",None)
            change_percent = item.get("change_percent", None)
            volume = item.get("volume", None)
            ma_50d = item.get("ma_50d", None)
            ma_200d = item.get("ma_200d", None)
            open_price = item.get("open", None)
            high = item.get("high", None)
            low = item.get("low", None)
            prev_close = item.get("prev_close", None)
            year_high = item.get("year_high", None)
            year_low = item.get("year_low", None)
            volume_avg = item.get("volume_average", None)
            volume_avg_10d = item.get("volume_average_10d", None)
            
            symbol_analysis = (
                f"**{symbol} ({item.get('name', 'Unknown')})**:\n"
                f"- Last Price: {last_price} {item.get('currency', 'USD')}\n"
                f"- Change: {change} ({change_percent}%)\n"
                f"- Volume: {volume}\n"
                f"- 50-Day Moving Average: {ma_50d}\n"
                f"- 200-Day Moving Average: {ma_200d}\n"
                f"- Open: {open_price}\n"
                f"- High: {high}\n"
                f"- Low: {low}\n"
                f"- Previous Close: {prev_close}\n"
                f"- 52-Week High: {year_high}\n"
                f"- 52-Week Low: {year_low}\n"
                f"- Average Volume: {volume_avg}\n"
                f"- 10-Day Average Volume: {volume_avg_10d}\n"
            )
            symbol_analyses.append(symbol_analysis)
        symbols_str = "\n".join(symbol_analyses) if symbol_analyses else "No symbol data available."

        detection_method = ""
        if len(data.question_input.split()) < 2:
            detection_method = DetectionMethod.LLM
        else:
            detection_method = DetectionMethod.LIBRARY

        # Language detection
        language_info = await language_detector.detect(
            text=data.question_input,
            method=detection_method,
            system_language=data.target_language,
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

        system_message = (f"""You are a professional financial advisor specializing in market analysis and investment guidance.\n
            
            {language_instruction}

            CONTEXT AWARENESS: When provided with previous tool responses or conversation history, you should:
            1. Reference and build upon previous analyses when relevant
            2. Avoid repeating information already provided
            3. Focus on new insights or changes since the last analysis
            4. Explicitly mention if you're updating or contradicting previous recommendations

            MARKET ANALYSIS FRAMEWORK:
            When analyzing market data (indices, prices, volumes, changes):

            1. COMPARATIVE ANALYSIS
            - Compare global markets performance (US, EU, Asia)
            - Identify relative strength/weakness between regions
            - Highlight sector rotation patterns

            2. PATTERN RECOGNITION  
            - Spot common trends across markets
            - Note significant divergences or correlations
            - Identify unusual percentage movements (>2%)

            3. VOLATILITY ASSESSMENT
            - Analyze intraday ranges (high-low spreads)
            - Compare current volatility to historical norms
            - Assess market sentiment (risk-on vs risk-off)

            4. ACTIONABLE INSIGHTS
            - Recommend specific sectors/regions for opportunities
            - Suggest portfolio adjustments based on trends
            - Provide risk management guidance

            RESPONSE STRUCTURE:
            - Lead with key market takeaway
            - Support with specific data points
            - End with 2-3 targeted follow-up questions about:
            • Specific indices/regions
            • Sector impact analysis  
            • Investment strategy implications
            • Historical context

            TONE: Data-driven, concise, investor-focused. Use specific numbers and percentages to support analysis."""
        )
        
        user_question = data.question_input if data.question_input else ""

        if chat_history and len(chat_history.strip()) > 0:
            user_content = f"""CURRENT ANALYSIS REQUEST 

{user_question}
Please analyze this market overview data and provide insights:

{symbols_str}

PREVIOUS CONTEXT (Reference if relevant)
{chat_history}

Provide comprehensive market analysis in the SAME LANGUAGE as this request."""
        else:
            user_content = f"""Please analyze this market overview data:
            
{symbols_str}

Provide comprehensive market analysis."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        # Ollama model
        if provider_type == ProviderType.OLLAMA:
            base_url = os.getenv('OLLAMA_HOST')
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "No response.")
        else:
            response = await llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            )
            return response.get("content", "No response.")
            
    except Exception as e:
        return f"Error contacting LLM: {str(e)}"
    
async def analyze_market_overview_stream(
    data: Any, 
    chat_history: str, 
    model_name: str,
    provider_type: str = ProviderType.OLLAMA
) -> AsyncGenerator[str, None]:
    """Market overview analysis with multi-provider support - STREAMING VERSION"""
    try:
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)

        market_data = data.data
        symbol_analyses = []
        for item in market_data:
            symbol = item.get("symbol", "Unknown")
            last_price = item.get("last_price", None)
            change = item.get("change",None)
            change_percent = item.get("change_percent", None)
            volume = item.get("volume", None)
            ma_50d = item.get("ma_50d", None)
            ma_200d = item.get("ma_200d", None)
            open_price = item.get("open", None)
            high = item.get("high", None)
            low = item.get("low", None)
            prev_close = item.get("prev_close", None)
            year_high = item.get("year_high", None)
            year_low = item.get("year_low", None)
            volume_avg = item.get("volume_average", None)
            volume_avg_10d = item.get("volume_average_10d", None)
            
            symbol_analysis = (
                f"**{symbol} ({item.get('name', 'Unknown')})**:\n"
                f"- Last Price: {last_price} {item.get('currency', 'USD')}\n"
                f"- Change: {change} ({change_percent}%)\n"
                f"- Volume: {volume}\n"
                f"- 50-Day Moving Average: {ma_50d}\n"
                f"- 200-Day Moving Average: {ma_200d}\n"
                f"- Open: {open_price}\n"
                f"- High: {high}\n"
                f"- Low: {low}\n"
                f"- Previous Close: {prev_close}\n"
                f"- 52-Week High: {year_high}\n"
                f"- 52-Week Low: {year_low}\n"
                f"- Average Volume: {volume_avg}\n"
                f"- 10-Day Average Volume: {volume_avg_10d}\n"
            )
            symbol_analyses.append(symbol_analysis)
        symbols_str = "\n".join(symbol_analyses) if symbol_analyses else "No symbol data available."

        detection_method = ""
        if len(data.question_input.split()) < 2:
            detection_method = DetectionMethod.LLM
        else:
            detection_method = DetectionMethod.LIBRARY

        # Language detection
        language_info = await language_detector.detect(
            text=data.question_input,
            method=detection_method,
            system_language=data.target_language,
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

        system_message = (f"""You are a professional financial advisor specializing in market analysis and investment guidance.\n
            
            {language_instruction}

            CONTEXT MANAGEMENT RULES
            **CRITICAL**: Before answering, analyze if the user's question relates to previous conversation:
            1. **USE PREVIOUS CONTEXT when user:**
            - Uses pronouns referring to previous topics ("it", "this stock", "that company")
            - Asks follow-up questions ("what about...", "and how about...", "additionally...")
            - References previous analysis ("as you mentioned", "you said earlier")
            - Compares with previous discussion ("compared to the previous stock")
            2. **IGNORE PREVIOUS CONTEXT when user:**
            - Asks about a completely NEW stock symbol
            - Starts a new topic unrelated to previous discussion
            - Asks a standalone question with complete information
            - Explicitly requests fresh analysis ("analyze AAPL" when previous was about TSLA)
            3. **CONTEXT DECISION PROCESS:**
            - First: Identify if current question is self-contained
            - Second: Check if it references previous conversation
            - Third: Only use relevant parts of history, not everything
            - Fourth: If unsure, prioritize the current question's direct needs
                          
            CONTEXT AWARENESS: When provided with previous tool responses or conversation history, you should:
            1. Reference and build upon previous analyses when relevant
            2. Avoid repeating information already provided
            3. Focus on new insights or changes since the last analysis
            4. Explicitly mention if you're updating or contradicting previous recommendations

            MARKET ANALYSIS FRAMEWORK:
            When analyzing market data (indices, prices, volumes, changes):

            1. COMPARATIVE ANALYSIS
            - Compare global markets performance (US, EU, Asia)
            - Identify relative strength/weakness between regions
            - Highlight sector rotation patterns

            2. PATTERN RECOGNITION  
            - Spot common trends across markets
            - Note significant divergences or correlations
            - Identify unusual percentage movements (>2%)

            3. VOLATILITY ASSESSMENT
            - Analyze intraday ranges (high-low spreads)
            - Compare current volatility to historical norms
            - Assess market sentiment (risk-on vs risk-off)

            4. ACTIONABLE INSIGHTS
            - Recommend specific sectors/regions for opportunities
            - Suggest portfolio adjustments based on trends
            - Provide risk management guidance

            RESPONSE STRUCTURE:
            - Lead with key market takeaway
            - Support with specific data points
            - End with 2-3 targeted follow-up questions about:
            • Specific indices/regions
            • Sector impact analysis  
            • Investment strategy implications
            • Historical context

            TONE: Data-driven, concise, investor-focused. Use specific numbers and percentages to support analysis."""
        )
        
        user_question = data.question_input if hasattr(data, 'question_input') else "Analyze this market data"
        
        if chat_history and len(chat_history.strip()) > 0:
            user_content = f"""=== USER'S QUESTION (Respond in this language) ===
{user_question}

=== MARKET DATA TO ANALYZE ===
{symbols_str}

=== CONVERSATION HISTORY (For context) ===
{chat_history}

Please provide comprehensive analysis addressing the user's question."""
        else:
            user_content = f"""=== USER'S QUESTION ===
{user_question}

=== MARKET DATA ===
{symbols_str}

Please provide comprehensive analysis."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        async for chunk in llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            clean_thinking=True
        ):
            yield chunk
            
    except Exception as e:
        error_msg = f"Error contacting LLM: {str(e)}"
        print(error_msg)
        yield error_msg


# =============================================================================
# region Trending APIs
# =============================================================================
async def analyze_stock_trending(
    data: Any, 
    chat_history: str, 
    model_name: str,
    provider_type: str = ProviderType.OPENAI
) -> str:
    """Stock trending analysis with multi-provider support"""
    try:
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)

        gainers = data.gainers 
        losers = data.losers    
        actives = data.actives

        gainers_str = "\n".join([
            f"**{item['symbol']}** - {item['name']}:\n"
            f"Price: {item['price']}, Change: {item['change']} ({item['percent_change']}%), Volume: {item['volume']}"
            for item in gainers
        ]) if gainers else "No gainers data available."

        losers_str = "\n".join([
            f"**{item['symbol']}** - {item['name']}:\n"
            f"Price: {item['price']}, Change: {item['change']} ({item['percent_change']}%), Volume: {item['volume']}"
            for item in losers
        ]) if losers else "No losers data available."

        active_str = "\n".join([
            f"**{item['symbol']}** - {item['name']}:\n"
            f"Price: {item['price']}, Change: {item['change']} ({item['percent_change']}%), Volume: {item['volume']}"
            for item in actives
        ]) if actives else "No active data available."

        detection_method = ""
        if len(data.question_input.split()) < 2:
            detection_method = DetectionMethod.LLM
        else:
            detection_method = DetectionMethod.LIBRARY

        # Language detection
        language_info = await language_detector.detect(
            text=data.question_input,
            method=detection_method,
            system_language=data.target_language,
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

        # System prompt
        system_message = f"""You are ToponeLogic, a professional financial market analyst with expertise in stock market analysis.

{language_instruction}

CRITICAL INSTRUCTIONS:

1. RESPONSE REQUIREMENTS:
   - Use clear, simple language suitable for investors of all levels
   - Format numbers appropriately (e.g., $1,234.56 for English, 1.234,56 for Vietnamese)
   - Include emojis sparingly for better readability (📈 for gains, 📉 for losses, 💰 for opportunities)

2. ANALYSIS FOCUS:
   - Key metrics: price changes, volume patterns, volatility
   - Market sentiment and momentum indicators
   - Risk assessment and opportunity identification
   - Actionable insights for investment decisions
"""
        user_question = data.question_input if hasattr(data, 'question_input') else "Analyze trending stocks"
        
        if chat_history:
            user_content = f"""=== USER'S QUESTION (Respond in this language) ===
{user_question}

=== TRENDING STOCKS DATA ===
Please analyze the following stock market data. Provide comprehensive insights about market trends, risks, and opportunities.

**📈 GAINERS:**
{gainers_str}

**📉 LOSERS:**
{losers_str}

**🔥 MOST ACTIVE:**
{active_str}

Structure your response with these sections:
1. **Market Overview** - Key trends and market sentiment
2. **Top Gainers Analysis** - Why these stocks are rising, opportunities
3. **Top Losers Analysis** - Risk factors and warning signs
4. **Most Active Stocks** - Volume analysis and implications
5. **Investment Recommendations** - Actionable advice with risk levels

=== PREVIOUS CONTEXT ===
{chat_history}

Analyze the trending stocks and provide insights."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        
        # Ollama model
        if provider_type == ProviderType.OLLAMA:
            base_url = os.getenv('OLLAMA_HOST')
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": False,
                        # "temperature": 0.7
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "No response.")
        else:
            response = await llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                # temperature=0.7
            )
            return response.get("content", "No response.")
            
    except Exception as e:
        print(f"Error in trending analysis: {str(e)}")
        traceback.print_exc()
        return f"Error analyzing market trends: {str(e)}"


async def analyze_stock_trending_stream(
    data: Any, 
    chat_history: str, 
    model_name: str,
    provider_type: str = ProviderType.OPENAI
) -> AsyncGenerator[str, None]:
    """Stock trending analysis with multi-provider support - STREAMING VERSION"""
    try:
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)

        gainers = data.gainers 
        losers = data.losers    
        actives = data.actives

        gainers_str = "\n".join([
            f"**{item['symbol']}** - {item['name']}:\n"
            f"Price: {item['price']}, Change: {item['change']} ({item['percent_change']}%), Volume: {item['volume']}"
            for item in gainers
        ]) if gainers else "No gainers data available."

        losers_str = "\n".join([
            f"**{item['symbol']}** - {item['name']}:\n"
            f"Price: {item['price']}, Change: {item['change']} ({item['percent_change']}%), Volume: {item['volume']}"
            for item in losers
        ]) if losers else "No losers data available."

        active_str = "\n".join([
            f"**{item['symbol']}** - {item['name']}:\n"
            f"Price: {item['price']}, Change: {item['change']} ({item['percent_change']}%), Volume: {item['volume']}"
            for item in actives
        ]) if actives else "No active data available."

        detection_method = ""
        if len(data.question_input.split()) < 2:
            detection_method = DetectionMethod.LLM
        else:
            detection_method = DetectionMethod.LIBRARY

        # Language detection
        language_info = await language_detector.detect(
            text=data.question_input,
            method=detection_method,
            system_language=data.target_language,
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

        # System prompt
        system_message = f"""You are ToponeLogic, a professional financial market analyst with expertise in stock market analysis.

{language_instruction}

CRITICAL INSTRUCTIONS:
1. RESPONSE REQUIREMENTS:
   - Use clear, simple language suitable for investors of all levels
   - Format numbers appropriately (e.g., $1,234.56 for English, 1.234,56 for Vietnamese)
   - Include emojis sparingly for better readability (📈 for gains, 📉 for losses, 💰 for opportunities)

2. ANALYSIS FOCUS:
   - Key metrics: price changes, volume patterns, volatility
   - Market sentiment and momentum indicators
   - Risk assessment and opportunity identification
   - Actionable insights for investment decisions

# CONTEXT MANAGEMENT RULES
**CRITICAL**: Before answering, analyze if the user's question relates to previous conversation:
1. **USE PREVIOUS CONTEXT when user:**
   - Uses pronouns referring to previous topics ("it", "this stock", "that company")
   - Asks follow-up questions ("what about...", "and how about...", "additionally...")
   - References previous analysis ("as you mentioned", "you said earlier")
   - Compares with previous discussion ("compared to the previous stock")
2. **IGNORE PREVIOUS CONTEXT when user:**
   - Asks about a completely NEW stock symbol
   - Starts a new topic unrelated to previous discussion
   - Asks a standalone question with complete information
   - Explicitly requests fresh analysis ("analyze AAPL" when previous was about TSLA)
3. **CONTEXT DECISION PROCESS:**
   - First: Identify if current question is self-contained
   - Second: Check if it references previous conversation
   - Third: Only use relevant parts of history, not everything
   - Fourth: If unsure, prioritize the current question's direct needs
"""
        user_question = data.question_input if hasattr(data, 'question_input') else "Analyze trending stocks"
        
        if chat_history:
            user_content = f"""=== USER'S QUESTION (Respond in this language) ===
{user_question}

=== TRENDING STOCKS DATA ===
Please analyze the following stock market data. Provide comprehensive insights about market trends, risks, and opportunities.

**📈 GAINERS:**
{gainers_str}

**📉 LOSERS:**
{losers_str}

**🔥 MOST ACTIVE:**
{active_str}

Structure your response with these sections:
1. **Market Overview** - Key trends and market sentiment
2. **Top Gainers Analysis** - Why these stocks are rising, opportunities
3. **Top Losers Analysis** - Risk factors and warning signs
4. **Most Active Stocks** - Volume analysis and implications
5. **Investment Recommendations** - Actionable advice with risk levels

=== PREVIOUS CONTEXT ===
{chat_history}

Analyze the trending stocks and provide insights."""
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
                
        async for chunk in llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            clean_thinking=True
        ):
            yield chunk
            
    except Exception as e:
        print(f"Error contacting LLM: {str(e)}")
        traceback.print_exc()
        yield f"Error contacting LLM: {str(e)}"


# =============================================================================
# region Stock Analysis APIs
# =============================================================================    
async def analyze_stock(
    data: dict, 
    user_query: str,
    system_language: str,
    model_name: str, 
    chat_history: str,
    provider_type: str = ProviderType.OPENAI
) -> str:
    """Stock analysis with multi-provider support"""
    
    # Get API Key
    api_key = ModelProviderFactory._get_api_key(provider_type)

    # Extract data
    extracted_data = analysis_instance.extract_summaries(data)
    summaries = extracted_data["summaries"]
    key_metrics = extracted_data["key_metrics"]
    
    # Get realtime news
    news_context = ""
    if data.get("symbol"):
        try:
            symbol = data["symbol"]
            limit = 5
            
            to_date = datetime.now().strftime("%Y-%m-%d")  # Today
            from_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

            # Initialize Redis client
            redis_client = await get_redis_client_llm()
            
            # Create cache key
            cache_key = f"company_news_{symbol.upper()}_limit_{limit}"
            
            # Check cache first
            cached_response = await get_cache(redis_client, cache_key, APIResponse[NewsItemOutput])
            
            if cached_response and cached_response.data and cached_response.data.data:
                # print(f"Get news from cache for {symbol}")
                news_items = cached_response.data.data
            else:
                # print(f"Get new news from API for {symbol}")
                news_items = await news_service.get_company_news(symbol.upper(), limit, from_date=from_date, to_date=to_date)
                
                if news_items:
                    response_data_payload = APIResponseData[NewsItemOutput](data=news_items)
                    api_response = APIResponse[NewsItemOutput](
                        message="OK",
                        status="200",
                        provider_used="fmp",
                        data=response_data_payload
                    )
                    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_CHART)
            
            # Format news context from descriptions
            if news_items:
                news_context = "RELEVANT NEWS:\n\n"
                for i, news_item in enumerate(news_items[:5], 1):
                    if news_item.description:
                        news_context += f"{i}. {news_item.title}\n"
                        news_context += f"   Date: {news_item.date}\n"
                        news_context += f"   Source: {news_item.source_site or 'Unknown'}\n"
                        news_context += f"   Description: {news_item.description}\n\n"
                
                # print(f"Fetched {len(news_items)} news for{symbol}")
            else:
                print(f"No news found for {symbol}")

            if redis_client:
                await redis_client.close()
                # print("Redis connection closed.")   
        except Exception as e:
            print(f"Error while retrieving news {data.get('symbol', 'N/A')}: {str(e)}")
    
    # print(f"News content: {news_context}")
    
    detection_method = ""
    if len(user_query.split()) < 2:
        detection_method = DetectionMethod.LLM
    else:
        detection_method = DetectionMethod.LIBRARY

    # Language detection
    language_info = await language_detector.detect(
        text=user_query,
        method=detection_method,
        system_language=system_language,
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
    - IMPORTANT: When dates/timestamps are provided in the data, ALWAYS mention them alongside the values
    - Example: "Price: $150 (as of 2024-03-15)" instead of just "Price: $150"
    """
    
    # System prompt
    system_message = f"""You are an expert financial analyst using ReAct (Reasoning and Acting) with Chain of Thought methodology.

{language_instruction}

# CONTEXT MANAGEMENT RULES
**CRITICAL**: Before answering, analyze if the user's question relates to previous conversation:
1. **USE PREVIOUS CONTEXT when user:**
   - Uses pronouns referring to previous topics ("it", "this stock", "that company")
   - Asks follow-up questions ("what about...", "and how about...", "additionally...")
   - References previous analysis ("as you mentioned", "you said earlier")
   - Compares with previous discussion ("compared to the previous stock")
2. **IGNORE PREVIOUS CONTEXT when user:**
   - Asks about a completely NEW stock symbol
   - Starts a new topic unrelated to previous discussion
   - Asks a standalone question with complete information
   - Explicitly requests fresh analysis ("analyze AAPL" when previous was about TSLA)
3. **CONTEXT DECISION PROCESS:**
   - First: Identify if current question is self-contained
   - Second: Check if it references previous conversation
   - Third: Only use relevant parts of history, not everything
   - Fourth: If unsure, prioritize the current question's direct needs

# DATA TRANSPARENCY RULES
**MANDATORY**: When presenting data, you MUST:
1. If a date/timestamp is provided with any metric, ALWAYS include it
2. Format: "[Metric]: [Value] (as of [Date])" or "[Metric] ([Date]): [Value]"
3. For news items: Always show the publication date
4. If NO date is provided in the data, state "current data" or "latest available"
5. Never invent or assume dates that aren't in the source data

Examples:
- "Current price: $150.25 (as of 2024-03-15)" ✓
- "RSI: 65 (calculated on 2024-03-14)" ✓  
- "Current price: $150.25" ✗ (missing date if available)

# ANALYSIS FRAMEWORK
## Step 1: Reasoning Process - INTERNAL ANALYSIS (do NOT reveal to user)
Analyze systematically:
1. TREND ASSESSMENT:
   - Price vs Moving Averages (20/50/200 SMA)
   - Trend strength and direction
   
2. MOMENTUM EVALUATION:
   - RSI position (oversold <30, neutral 30-70, overbought >70)
   - MACD signal (bullish/bearish crossover)
   
3. RISK ANALYSIS:
   - Support/Resistance levels
   - Stop loss positioning (ATR vs percentage)
   - Risk/Reward ratio calculation
   
4. VOLUME & PATTERNS:
   - Volume profile (accumulation/distribution)
   - Chart patterns (reliability assessment)
   
5. MARKET STRENGTH:
   - Relative strength vs benchmark
   - Sector performance context
   
6. NEWS IMPACT:
   - Fundamental catalysts
   - Sentiment shifts
   - Time sensitivity of news

## Step 2: Decision Matrix
Based on analysis, calculate signal strength:
- Count bullish vs bearish indicators
- Weight by reliability (Price action > Indicators > Patterns)
- Factor in news sentiment

## Step 3: Investment Recommendation
### For STRONG BUY (>70% bullish signals):
**🟢 STRONG BUY RECOMMENDATION**
- Entry: [specific price or condition]
- Target 1: [based on resistance]
- Target 2: [based on pattern]
- Stop Loss: [based on ATR or support]
- Position Size: [% of portfolio based on risk]
- Reasoning: [3-4 key factors]

### For MODERATE BUY (50-70% bullish):
**🟡 CONDITIONAL BUY**
- Wait for: [specific trigger]
- Entry zones: [price ranges]
- Risk factors to monitor: [list]
- Alternative strategy: [DCA approach]

### For NEUTRAL/WAIT (40-60% mixed):
**⚪ HOLD/WAIT**
- Current situation: [analysis]
- What to watch: [key levels/indicators]
- Decision triggers: [specific conditions]

### For SELL/AVOID (<40% bullish):
**🔴 AVOID/SELL RECOMMENDATION**
- Key concerns: [major risks]
- If holding: [exit strategy]
- Better alternatives: [suggestions]

# DECISION CRITERIA
Your recommendation MUST be based on:
1. **Technical Weight (60%)**:
   - Price action relative to MAs
   - Momentum indicators alignment
   - Volume confirmation
   
2. **Risk Assessment (25%)**:
   - Clear stop loss levels
   - Risk/Reward ratio > 1.5:1
   - Position sizing rules
   
3. **Market Context (15%)**:
   - Relative strength
   - News/Fundamental factors
   - Overall market conditions

## Summary
[1-2 sentences with clear action plan]

# OUTPUT REQUIREMENTS
1. Base recommendations on actual data values
2. Provide ONE clear recommendation (BUY/CONDITIONAL BUY/WAIT/AVOID)
3. Include SPECIFIC entry, stop, and target prices
4. Explain your reasoning with exact indicator values
5. Address the user's specific question if provided
6. Use simple language but be technically accurate
7. Be decisive - no wishy-washy recommendations
8. **DATE TRANSPARENCY**: Always include dates/timestamps when they exist in the source data

Remember: Investors need actionable advice with clear reasoning. Your analysis should give them confidence in their decision."""

    current_date = datetime.now().strftime("%Y-%m-%d")

    data_analysis = f"""
Analyze the stock with symbol {data["symbol"]}. Use the provided STRUCTURED data below instead of trying to parse the entire JSON.

TECHNICAL ANALYSIS SUMMARY:
{summaries["technical_analysis"]}

RISK ANALYSIS SUMMARY:
{summaries["risk_analysis"]}

VOLUME PROFILE SUMMARY:
{summaries["volume_profile"]}

PATTERN RECOGNITION SUMMARY:
{summaries["pattern_recognition"]}

RELATIVE STRENGTH SUMMARY:
{summaries["relative_strength"]}

KEY METRICS:
- Current price: ${key_metrics["price"]}
- RSI: {key_metrics["rsi"]}
- MACD bullish: {key_metrics["macd_bullish"]}
- SMA 20: ${key_metrics["moving_averages"]["sma_20"]}
- SMA 50: ${key_metrics["moving_averages"]["sma_50"]}
- Stop levels (ATR 2x): ${key_metrics["stop_levels"]["atr_2x"]}
- Stop levels (5%): ${key_metrics["stop_levels"]["percent_5"]}
- Recent swing low: ${key_metrics["stop_levels"]["recent_swing"]}

REMINDER: When presenting these metrics, include the date ({current_date}) to show data freshness.
"""
    
    if user_query:
        if chat_history:
            final_context = f"""=== USER'S QUESTION (Respond in this language) ===
{user_query}

=== COMPREHENSIVE ANALYSIS DATA ===
{data_analysis}

=== CONVERSATION HISTORY (For reference) ===
{chat_history}

=== CRITICAL DATE TRANSPARENCY REMINDER ===
You MUST include dates/timestamps whenever they appear in the data above.
- For prices and indicators: State the date they were calculated/retrieved
- For news: Always mention publication dates
- This ensures users understand data freshness and can make informed decisions

Please provide investment analysis addressing the user's question."""

        else:
            final_context = f"""=== USER'S QUESTION (Respond in this language) ===
{user_query}

=== COMPREHENSIVE ANALYSIS DATA ===
{data_analysis}

=== CRITICAL DATE TRANSPARENCY REMINDER ===
You MUST include dates/timestamps whenever they appear in the data above.
- For prices and indicators: State the date they were calculated/retrieved  
- For news: Always mention publication dates
- This ensures users understand data freshness and can make informed decisions

Please provide investment analysis."""
    else:
        # No specific query, just analyze
        final_context = f"""=== ANALYSIS REQUEST ===
Please analyze this stock data comprehensively.

    === DATA ===
    {data_analysis}

=== CRITICAL DATE TRANSPARENCY REMINDER ===
You MUST include dates/timestamps whenever they appear in the data above.
- For prices and indicators: State the date they were calculated/retrieved
- For news: Always mention publication dates  
- This ensures users understand data freshness and can make informed decisions
"""
        
    if news_context:
        final_context += f"\n\n{news_context}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": final_context}
    ]
    
    try:
        # Ollama model
        if provider_type == ProviderType.OLLAMA:
            base_url = os.getenv('OLLAMA_HOST')
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": False
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "No response.")
        else:
            response = await llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            )
            return response.get("content", "No response.")
            
    except Exception as e:
        print(f"Error contacting LLM: {str(e)}")
        traceback.print_exc()
        return f"Error contacting LLM: {str(e)}"


async def analyze_stock_stream(
    data: dict, 
    user_query: str,
    system_language: str,
    model_name: str, 
    chat_history: str,
    provider_type: str = ProviderType.OPENAI,
    lookback_days: Optional[int] = 7 
) -> AsyncGenerator[str, None]:
    """Stock analysis with multi-provider support - STREAMING VERSION"""
    
    # Get API Key
    api_key = ModelProviderFactory._get_api_key(provider_type)

    # Extract data
    extracted_data = analysis_instance.extract_summaries(data)
    summaries = extracted_data["summaries"]
    key_metrics = extracted_data["key_metrics"]
    
    # Get realtime news from API provider
    news_context = ""
    if data.get("symbol"):
        try:
            symbol = data["symbol"]
            limit = 10

            to_date = datetime.now().strftime("%Y-%m-%d")  # Today
            from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            # Initialize Redis client
            redis_client = await get_redis_client_llm()
            
            # Create cache key
            cache_key = f"company_news_{symbol.upper()}_limit_{limit}"
            
            # Check cache first
            cached_response = await get_cache(redis_client, cache_key, APIResponse[NewsItemOutput])
            
            if cached_response and cached_response.data and cached_response.data.data:
                # print(f"Get news from cache for {symbol}")
                news_items = cached_response.data.data
            else:
                # print(f"Get new news from API for {symbol}")
                news_items = await news_service.get_company_news(symbol.upper(), limit, from_date=from_date, to_date=to_date)
                
                if news_items:
                    response_data_payload = APIResponseData[NewsItemOutput](data=news_items)
                    api_response = APIResponse[NewsItemOutput](
                        message="OK",
                        status="200",
                        provider_used="fmp",
                        data=response_data_payload
                    )
                    await set_cache(redis_client, cache_key, api_response, expiry=settings.CACHE_TTL_CHART)
            
            # Format news context from descriptions
            if news_items:
                news_context = "RELEVANT NEWS:\n\n"
                for i, news_item in enumerate(news_items[:5], 1):
                    if news_item.description:
                        news_context += f"{i}. {news_item.title}\n"
                        news_context += f"   Date: {news_item.date}\n"
                        news_context += f"   Source: {news_item.source_site or 'Unknown'}\n"
                        news_context += f"   Description: {news_item.description}\n\n"
                
                # print(f"Fetched {len(news_items)} news for {symbol}")
            else:
                print(f"No news found for {symbol}")

            if redis_client:
                await redis_client.close()
                print("Redis connection closed.")   
        except Exception as e:
            print(f"Error while retrieving news {data.get('symbol', 'N/A')}: {str(e)}")
    
    # print(f"News content: {news_context}")
    
    detection_method = ""
    if len(user_query.split()) < 2:
        detection_method = DetectionMethod.LLM
    else:
        detection_method = DetectionMethod.LIBRARY

    # Language detection
    language_info = await language_detector.detect(
        text=user_query,
        method=detection_method,
        system_language=system_language,
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
    - IMPORTANT: When dates/timestamps are provided in the data, ALWAYS mention them alongside the values
    - Example: "Price: $150 (as of 2024-03-15)" instead of just "Price: $150"
    """
    
    # System prompt
    system_message = f"""You are an expert financial analyst using ReAct (Reasoning and Acting) with Chain of Thought methodology.

{language_instruction}

# CONTEXT MANAGEMENT RULES
**CRITICAL**: Before answering, analyze if the user's question relates to previous conversation:
1. **USE PREVIOUS CONTEXT when user:**
   - Uses pronouns referring to previous topics ("it", "this stock", "that company")
   - Asks follow-up questions ("what about...", "and how about...", "additionally...")
   - References previous analysis ("as you mentioned", "you said earlier")
   - Compares with previous discussion ("compared to the previous stock")
2. **IGNORE PREVIOUS CONTEXT when user:**
   - Asks about a completely NEW stock symbol
   - Starts a new topic unrelated to previous discussion
   - Asks a standalone question with complete information
   - Explicitly requests fresh analysis ("analyze AAPL" when previous was about TSLA)
3. **CONTEXT DECISION PROCESS:**
   - First: Identify if current question is self-contained
   - Second: Check if it references previous conversation
   - Third: Only use relevant parts of history, not everything
   - Fourth: If unsure, prioritize the current question's direct needs

# DATA TRANSPARENCY RULES
**MANDATORY**: When presenting data, you MUST:
1. If a date/timestamp is provided with any metric, ALWAYS include it
2. Format: "[Metric]: [Value] (as of [Date])" or "[Metric] ([Date]): [Value]"
3. For news items: Always show the publication date
4. If NO date is provided in the data, state "current data" or "latest available"
5. Never invent or assume dates that aren't in the source data

Examples:
- "Current price: $150.25 (as of 2024-03-15)" ✓
- "RSI: 65 (calculated on 2024-03-14)" ✓  
- "Current price: $150.25" ✗ (missing date if available)

# ANALYSIS FRAMEWORK
## Step 1: Reasoning Process - INTERNAL ANALYSIS (do NOT reveal to user)
Analyze systematically:
1. TREND ASSESSMENT:
   - Price vs Moving Averages (20/50/200 SMA)
   - Trend strength and direction
   
2. MOMENTUM EVALUATION:
   - RSI position (oversold <30, neutral 30-70, overbought >70)
   - MACD signal (bullish/bearish crossover)
   
3. RISK ANALYSIS:
   - Support/Resistance levels
   - Stop loss positioning (ATR vs percentage)
   - Risk/Reward ratio calculation
   
4. VOLUME & PATTERNS:
   - Volume profile (accumulation/distribution)
   - Chart patterns (reliability assessment)
   
5. MARKET STRENGTH:
   - Relative strength vs benchmark
   - Sector performance context
   
6. NEWS IMPACT:
   - Fundamental catalysts
   - Sentiment shifts
   - Time sensitivity of news

## Step 2: Decision Matrix
Based on analysis, calculate signal strength:
- Count bullish vs bearish indicators
- Weight by reliability (Price action > Indicators > Patterns)
- Factor in news sentiment

## Step 3: Investment Recommendation
### For STRONG BUY (>70% bullish signals):
**🟢 STRONG BUY RECOMMENDATION**
- Entry: [specific price or condition]
- Target 1: [based on resistance]
- Target 2: [based on pattern]
- Stop Loss: [based on ATR or support]
- Position Size: [% of portfolio based on risk]
- Reasoning: [3-4 key factors]

### For MODERATE BUY (50-70% bullish):
**🟡 CONDITIONAL BUY**
- Wait for: [specific trigger]
- Entry zones: [price ranges]
- Risk factors to monitor: [list]
- Alternative strategy: [DCA approach]

### For NEUTRAL/WAIT (40-60% mixed):
**⚪ HOLD/WAIT**
- Current situation: [analysis]
- What to watch: [key levels/indicators]
- Decision triggers: [specific conditions]

### For SELL/AVOID (<40% bullish):
**🔴 AVOID/SELL RECOMMENDATION**
- Key concerns: [major risks]
- If holding: [exit strategy]
- Better alternatives: [suggestions]

# DECISION CRITERIA
Your recommendation MUST be based on:
1. **Technical Weight (60%)**:
   - Price action relative to MAs
   - Momentum indicators alignment
   - Volume confirmation
   
2. **Risk Assessment (25%)**:
   - Clear stop loss levels
   - Risk/Reward ratio > 1.5:1
   - Position sizing rules
   
3. **Market Context (15%)**:
   - Relative strength
   - News/Fundamental factors
   - Overall market conditions

## Summary
[1-2 sentences with clear action plan]

# OUTPUT REQUIREMENTS
1. Base recommendations on actual data values
2. Provide ONE clear recommendation (BUY/CONDITIONAL BUY/WAIT/AVOID)
3. Include SPECIFIC entry, stop, and target prices
4. Explain your reasoning with exact indicator values
5. Address the user's specific question if provided
6. Use simple language but be technically accurate
7. Be decisive - no wishy-washy recommendations
8. **DATE TRANSPARENCY**: Always include dates/timestamps when they exist in the source data

Remember: Investors need actionable advice with clear reasoning. Your analysis should give them confidence in their decision."""

    current_date = datetime.now().strftime("%Y-%m-%d")

    data_analysis = f"""
Analyze the stock with symbol {data["symbol"]}. Use the provided STRUCTURED data below instead of trying to parse the entire JSON.

TECHNICAL ANALYSIS SUMMARY:
{summaries["technical_analysis"]}

RISK ANALYSIS SUMMARY:
{summaries["risk_analysis"]}

VOLUME PROFILE SUMMARY:
{summaries["volume_profile"]}

PATTERN RECOGNITION SUMMARY:
{summaries["pattern_recognition"]}

RELATIVE STRENGTH SUMMARY:
{summaries["relative_strength"]}

KEY METRICS:
- Current price: ${key_metrics["price"]}
- RSI: {key_metrics["rsi"]}
- MACD bullish: {key_metrics["macd_bullish"]}
- SMA 20: ${key_metrics["moving_averages"]["sma_20"]}
- SMA 50: ${key_metrics["moving_averages"]["sma_50"]}
- Stop levels (ATR 2x): ${key_metrics["stop_levels"]["atr_2x"]}
- Stop levels (5%): ${key_metrics["stop_levels"]["percent_5"]}
- Recent swing low: ${key_metrics["stop_levels"]["recent_swing"]}

REMINDER: When presenting these metrics, include the date ({current_date}) to show data freshness.
"""
    
    if user_query:
        if chat_history:
            final_context = f"""=== USER'S QUESTION (Respond in this language) ===
{user_query}

=== COMPREHENSIVE ANALYSIS DATA ===
{data_analysis}

=== CONVERSATION HISTORY (For reference) ===
{chat_history}

=== CRITICAL DATE TRANSPARENCY REMINDER ===
You MUST include dates/timestamps whenever they appear in the data above.
- For prices and indicators: State the date they were calculated/retrieved
- For news: Always mention publication dates
- This ensures users understand data freshness and can make informed decisions

Please provide investment analysis addressing the user's question."""

        else:
            final_context = f"""=== USER'S QUESTION (Respond in this language) ===
{user_query}

=== COMPREHENSIVE ANALYSIS DATA ===
{data_analysis}

=== CRITICAL DATE TRANSPARENCY REMINDER ===
You MUST include dates/timestamps whenever they appear in the data above.
- For prices and indicators: State the date they were calculated/retrieved  
- For news: Always mention publication dates
- This ensures users understand data freshness and can make informed decisions

Please provide investment analysis."""
    else:
        # No specific query, just analyze
        final_context = f"""=== ANALYSIS REQUEST ===
Please analyze this stock data comprehensively.

    === DATA ===
    {data_analysis}

=== CRITICAL DATE TRANSPARENCY REMINDER ===
You MUST include dates/timestamps whenever they appear in the data above.
- For prices and indicators: State the date they were calculated/retrieved
- For news: Always mention publication dates  
- This ensures users understand data freshness and can make informed decisions
"""
        
    if news_context:
        final_context += f"\n\n{news_context}"
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": final_context}
    ]

    try:        
        async for chunk in llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            clean_thinking=True
        ):
            yield chunk
            
    except Exception as e:
        print(f"Error contacting LLM: {str(e)}")
        traceback.print_exc()
        yield f"Error contacting LLM: {str(e)}"


# =============================================================================
# region HEATMAP Funcs
# =============================================================================
async def analyze_stock_heatmap(
    data: StockHeatmapPayload,
    heatmap_data: List[Dict[str, Any]], 
    chat_history: str, 
    model_name: str,
    provider_type: str = ProviderType.OPENAI,
    user_query: str = None
) -> str:
    """
    Analyze stock market heatmap data using LLM
    """
    # Get API Key
    api_key = ModelProviderFactory._get_api_key(provider_type)

    # Format heatmap data for LLM analysis
    formatted_data = format_heatmap_data(heatmap_data)

    # Create heatmap analysis prompt
    system_prompt = """LANGUAGE: Auto-detect and respond in the user's language.

You are ToponeLogic an expert in the stock market analyst specializing in heatmap visualization and market trends analysis multi-language.

=== CRITICAL INSTRUCTIONS ===:
When provided with previous tool responses or conversation history, you should:
1. Reference and build upon previous heatmap analyses when relevant
2. Highlight changes in sector performance if previously analyzed
3. Update any previous market assessments based on new data
    
Your task is to analyze S&P 500 heatmap data and provide key insights:
1. Identify sectors/industries performing well or poorly
2. Highlight notable stocks (top gainers/losers)
3. Comment on overall market trends and sentiment
4. Provide insights on market cap distribution, P/E ratios, and volume patterns
5. Alert on potential risks or investment opportunities
6. Explain what the heatmap reveals about market dynamics

Use clear, concise language and focus on actionable insights."""

    if user_query or (hasattr(data, 'question_input') and data.question_input):
        query = user_query or data.question_input
        
        if chat_history:
            user_content = f"""=== USER'S QUESTION (Respond in this language) ===
{query}

=== S&P 500 HEATMAP DATA ===
{formatted_data}

=== CONVERSATION HISTORY ===
{chat_history}

Analyze the heatmap and address the user's question."""
        else:
            user_content = f"""=== USER'S QUESTION (Respond in this language) ===
{query}

=== S&P 500 HEATMAP DATA ===
{formatted_data}

Analyze and provide insights."""
    else:
        user_content = f"""=== HEATMAP ANALYSIS REQUEST ===

{formatted_data}

Please analyze this S&P 500 heatmap data."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # Ollama model
    if provider_type == ProviderType.OLLAMA:
        base_url = os.getenv('OLLAMA_HOST')
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{base_url}/api/chat",
                json={
                    "model": model_name,
                    "messages": messages,
                    "stream": False
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("message", {}).get("content", "No response.")
    else:
        response = await llm_provider.generate_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key
        )
        return response.get("content", "No response.")
    
    return response["content"]


async def analyze_stock_heatmap_stream(
    data: Any, 
    heatmap_data: List[Dict[str, Any]], 
    chat_history: str, 
    model_name: str,
    provider_type: str = ProviderType.OPENAI,
    user_query: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """
    Stream analyze stock market heatmap data using LLM with enhanced context
    """
    try:
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)

        # Format heatmap data for LLM analysis
        formatted_data = format_heatmap_data(heatmap_data)
        
        detection_method = ""
        if len(user_query.split()) < 2:
            detection_method = DetectionMethod.LLM
        else:
            detection_method = DetectionMethod.LIBRARY

        # Language detection
        language_info = await language_detector.detect(
            text=user_query,
            method=detection_method,
            system_language=data.target_language,
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

        # Create heatmap analysis prompt
        system_prompt = f"""You are ToponeLogic an expert in the stock market analyst specializing in heatmap visualization and market trends analysis multi-language.

    {language_instruction}

    === CONTEXT MANAGEMENT RULES === 
    **CRITICAL**: Before answering, analyze if the user's question relates to previous conversation:
    1. **USE PREVIOUS CONTEXT when user:**
    - Uses pronouns referring to previous topics ("it", "this stock", "that company")
    - Asks follow-up questions ("what about...", "and how about...", "additionally...")
    - References previous analysis ("as you mentioned", "you said earlier")
    - Compares with previous discussion ("compared to the previous stock")
    2. **IGNORE PREVIOUS CONTEXT when user:**
    - Asks about a completely NEW stock symbol
    - Starts a new topic unrelated to previous discussion
    - Asks a standalone question with complete information
    - Explicitly requests fresh analysis ("analyze AAPL" when previous was about TSLA)
    3. **CONTEXT DECISION PROCESS:**
    - First: Identify if current question is self-contained
    - Second: Check if it references previous conversation
    - Third: Only use relevant parts of history, not everything
    - Fourth: If unsure, prioritize the current question's direct needs

    === CRITICAL INSTRUCTIONS ===:
    When provided with previous tool responses or conversation history, you should:
    1. Reference and build upon previous heatmap analyses when relevant
    2. Highlight changes in sector performance if previously analyzed
    3. Update any previous market assessments based on new data
        
    Your task is to analyze S&P 500 heatmap data and provide key insights:
    1. Identify sectors/industries performing well or poorly
    2. Highlight notable stocks (top gainers/losers)
    3. Comment on overall market trends and sentiment
    4. Provide insights on market cap distribution, P/E ratios, and volume patterns
    5. Alert on potential risks or investment opportunities
    6. Explain what the heatmap reveals about market dynamics

    Use clear, concise language and focus on actionable insights."""

        user_query = data.question_input if hasattr(data, 'question_input') else "Analyze heatmap stocks"
        
        if user_query:
            if chat_history:
                user_content = f"""=== USER'S QUESTION (Respond in this language) ===
    {user_query}

    === S&P 500 HEATMAP DATA ===
    {formatted_data}

    === CONVERSATION HISTORY ===
    {chat_history}

    Analyze the heatmap and address the user's question."""
            else:
                user_content = f"""=== USER'S QUESTION (Respond in this language) ===
    {user_query}

    === S&P 500 HEATMAP DATA ===
    {formatted_data}

    Analyze and provide insights."""
        else:
            user_content = f"""=== HEATMAP ANALYSIS REQUEST ===

    {formatted_data}

    Please analyze this S&P 500 heatmap data."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Ollama model
        if provider_type == ProviderType.OLLAMA:
            base_url = os.getenv('OLLAMA_HOST')
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "stream": True
                    }
                )
                
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith('data: '):
                        line = line[6:]
                    if line == '[DONE]':
                        break
                    try:
                        chunk = json.loads(line)
                        if 'message' in chunk and 'content' in chunk['message']:
                            content = chunk['message']['content']
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
        else:
            async for chunk in llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                clean_thinking=True,
                enable_thinking=False
            ):
                yield chunk
                
    except Exception as e:
        yield f"Error streaming heatmap analysis: {str(e)}"


def format_heatmap_data(heatmap_data: List[Dict[str, Any]]) -> str:
    """
    Format heatmap data into a structured format for LLM analysis
    """
    if not heatmap_data:
        return "No heatmap data available"
    
    # Group by sector
    sectors_data = {}
    for item in heatmap_data:
        sector = item.get('sector', 'Unknown')
        if sector not in sectors_data:
            sectors_data[sector] = []
        sectors_data[sector].append(item)
    
    # Format output
    formatted = "📊 S&P 500 HEATMAP ANALYSIS DATA\n\n"
    
    # Overall statistics
    total_stocks = len(heatmap_data)
    gainers = [s for s in heatmap_data if s.get('changesPercentage', 0) > 0]
    strong_gainers = [s for s in heatmap_data if s.get('changesPercentage', 0) > 3]
    losers = [s for s in heatmap_data if s.get('changesPercentage', 0) < 0]
    strong_losers = [s for s in heatmap_data if s.get('changesPercentage', 0) < -3]
    
    # Market breadth analysis
    advancement_ratio = len(gainers) / total_stocks if total_stocks > 0 else 0
    market_sentiment = "🟢 Risk-On" if advancement_ratio > 0.6 else "🔴 Risk-Off" if advancement_ratio < 0.4 else "🟡 Mixed"
    
    formatted += f"📈 **MARKET OVERVIEW** ({market_sentiment}): {total_stocks} stocks analyzed\n"
    formatted += f"✅ Advancing: {len(gainers)} ({advancement_ratio*100:.1f}%) | Strong: {len(strong_gainers)}\n"
    formatted += f"❌ Declining: {len(losers)} ({(1-advancement_ratio)*100:.1f}%) | Weak: {len(strong_losers)}\n"
    formatted += f"📊 Market Breadth Ratio: {len(gainers)}/{len(losers)} = {'Bullish' if advancement_ratio > 0.55 else 'Bearish' if advancement_ratio < 0.45 else 'Neutral'}\n\n"
    
    # Enhanced top movers with additional context
    top_gainers = sorted(heatmap_data, key=lambda x: x.get('changesPercentage', 0), reverse=True)[:8]
    top_losers = sorted(heatmap_data, key=lambda x: x.get('changesPercentage', 0))[:8]
    
    formatted += "🚀 **TOP GAINERS** (Momentum Leaders):\n"
    for i, stock in enumerate(top_gainers, 1):
        volume_indicator = "🔥" if stock.get('volume', 0) > stock.get('avgVolume', 1) * 2 else "📈"
        formatted += f"  {i}. {stock['symbol']} ({stock.get('name', 'N/A')}): {stock['changesPercentage']:.2f}% | ${stock.get('price', 0):.2f} {volume_indicator}\n"
    
    formatted += "\n📉 **TOP LOSERS** (Pressure Points):\n"
    for i, stock in enumerate(top_losers, 1):
        volume_indicator = "⚠️" if stock.get('volume', 0) > stock.get('avgVolume', 1) * 2 else "📉"
        formatted += f"  {i}. {stock['symbol']} ({stock.get('name', 'N/A')}): {stock['changesPercentage']:.2f}% | ${stock.get('price', 0):.2f} {volume_indicator}\n"
    
    # Enhanced sector analysis with performance categorization
    formatted += "\n📊 **SECTOR PERFORMANCE MATRIX**:\n"
    sector_performance = []
    
    for sector, stocks in sectors_data.items():
        avg_change = sum(s.get('changesPercentage', 0) for s in stocks) / len(stocks)
        total_market_cap = sum(s.get('marketCap', 0) for s in stocks)
        total_volume = sum(s.get('volume', 0) for s in stocks)
        sector_gainers = len([s for s in stocks if s.get('changesPercentage', 0) > 0])
        sector_breadth = sector_gainers / len(stocks) if stocks else 0
        
        # Performance categorization
        if avg_change > 1.5:
            performance_status = "🟢 STRONG"
        elif avg_change > 0:
            performance_status = "🟡 POSITIVE"
        elif avg_change > -1.5:
            performance_status = "🟠 WEAK"
        else:
            performance_status = "🔴 DECLINING"
            
        sector_performance.append({
            'sector': sector,
            'avg_change': avg_change,
            'status': performance_status,
            'breadth': sector_breadth,
            'stocks': stocks,
            'market_cap': total_market_cap,
            'volume': total_volume
        })
    
    # Sort by performance
    sector_performance.sort(key=lambda x: x['avg_change'], reverse=True)
    
    for sector_data in sector_performance:
        breadth_indicator = "💪" if sector_data['breadth'] > 0.7 else "⚖️" if sector_data['breadth'] > 0.4 else "🔻"
        
        formatted += f"\n{sector_data['status']} **{sector_data['sector']}** {breadth_indicator}\n"
        formatted += f"  - Performance: {sector_data['avg_change']:+.2f}% | Breadth: {sector_data['breadth']*100:.0f}% advancing\n"
        formatted += f"  - Stocks: {len(sector_data['stocks'])} | Market Cap: ${sector_data['market_cap']/1e9:.1f}B\n"
        formatted += f"  - Key Players: {', '.join([s['symbol'] for s in sorted(sector_data['stocks'], key=lambda x: abs(x.get('changesPercentage', 0)), reverse=True)[:4]])}\n"
    
    # Enhanced volume and liquidity analysis
    high_volume_stocks = [s for s in heatmap_data if s.get('volume', 0) > s.get('avgVolume', 1) * 2]
    institutional_activity = [s for s in high_volume_stocks if s.get('marketCap', 0) > 10e9]
    
    formatted += "\n💼 **INSTITUTIONAL ACTIVITY SIGNALS**:\n"
    formatted += f"  - High Volume Stocks: {len(high_volume_stocks)} (>200% avg volume)\n"
    formatted += f"  - Large Cap Activity: {len(institutional_activity)} stocks showing institutional interest\n"
    
    # Market cap distribution analysis
    mega_caps = [s for s in heatmap_data if s.get('marketCap', 0) > 200e9]
    large_caps = [s for s in heatmap_data if 50e9 <= s.get('marketCap', 0) <= 200e9]
    mid_caps = [s for s in heatmap_data if 10e9 <= s.get('marketCap', 0) < 50e9]
    
    mega_performance = sum(s.get('changesPercentage', 0) for s in mega_caps) / len(mega_caps) if mega_caps else 0
    large_performance = sum(s.get('changesPercentage', 0) for s in large_caps) / len(large_caps) if large_caps else 0
    
    formatted += "\n💰 **MARKET CAP PERFORMANCE ANALYSIS**:\n"
    formatted += f"  - Mega Cap (>$200B): {len(mega_caps)} stocks | Avg: {mega_performance:+.2f}%\n"
    formatted += f"  - Large Cap ($50B-$200B): {len(large_caps)} stocks | Avg: {large_performance:+.2f}%\n"
    formatted += f"  - Mid Cap ($10B-$50B): {len(mid_caps)} stocks\n"
    formatted += f"  - Size Factor: {'Large Cap Leading' if mega_performance > large_performance else 'Small Cap Outperforming'}\n"
    
    return formatted