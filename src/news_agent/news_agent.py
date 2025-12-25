from typing import Any, Dict, List, Literal, Optional
import json
from datetime import datetime
import os
import asyncio
import logging
from pathlib import Path
import re  # Added re explicitly for regex operations

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import existing tools
# Import các công cụ tìm kiếm và xử lý nội dung từ thư viện có sẵn
from src.news_agent.tools.tavily_tools import tavily_extract_content
from src.news_agent.tools.tavily_tools import (
    tavily_search, 
    tavily_extract_content,     
    tavily_crawl, 
    tavily_map_site
)

# Import content processor for article/video processing
# Import bộ xử lý nội dung để tách văn bản từ bài viết hoặc video
from src.handlers.content_processor import ContentProcessor, ContentTypeDetector

# ========================
# LOGGING SETUP
# ========================
LOG_DIR = Path("news_agent_logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Setup a logger with file and console handlers"""
    # Cấu hình logger để ghi log ra file và console
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    logger.handlers = []
    
    fh = logging.FileHandler(LOG_DIR / log_file, encoding='utf-8')
    fh.setLevel(level)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Setup component loggers
# Khởi tạo các logger cho từng thành phần
tavily_logger = setup_logger('tavily', 'tavily_searches.log')
processor_logger = setup_logger('content_processor', 'content_processing.log')
agent_logger = setup_logger('news_agent', 'agent_workflow.log')
error_logger = setup_logger('errors', 'errors.log', logging.ERROR)


# Create output directory for summaries
# Tạo thư mục đầu ra cho các file tóm tắt
OUTPUT_DIR = "news_crawl_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# DEFAULT FINANCIAL TOPICS
# Danh sách các chủ đề tài chính mặc định
default_news_topics = """
MARKET OVERVIEW:
  - S&P 500, Nasdaq, Dow Jones daily performance
  - Pre-market and after-hours movers
  - Market sentiment indicators (VIX, Put/Call ratio)
  - Federal Reserve decisions and FOMC meetings
  - CPI, PPI inflation data releases
  - Jobs reports and unemployment data
  - Key sources: Bloomberg, Reuters, CNBC, MarketWatch, Wall Street Journal

STOCKS & EQUITIES:
  - Magnificent 7 stocks (Apple, Microsoft, Google, Amazon, NVIDIA, Meta, Tesla)
  - Earnings reports and guidance updates
  - Analyst upgrades/downgrades
  - IPOs and SPACs news
  - Sector rotation analysis
  - Key sources: Seeking Alpha, Yahoo Finance, Benzinga, Barron's

CRYPTOCURRENCY:
  - Bitcoin price action and on-chain metrics
  - Ethereum developments and DeFi ecosystem
  - Regulatory news (SEC, CFTC, global regulations)
  - Major altcoins (SOL, BNB, XRP, ADA)
  - Stablecoin developments
  - Exchange news (Binance, Coinbase, etc.)
  - Key sources: CoinDesk, CoinTelegraph, The Block, Decrypt, CryptoSlate

AI & TECHNOLOGY SECTOR:
  - AI companies (OpenAI, Anthropic, Google AI, Meta AI)
  - Semiconductor stocks (NVDA, AMD, TSM, INTC, MU)
  - Cloud computing (AWS, Azure, Google Cloud)
  - Cybersecurity developments
  - Key sources: TechCrunch, The Verge, Ars Technica, Wired

COMMODITIES & FOREX:
  - Gold, Silver, Oil prices
  - USD index movements
  - Major currency pairs (EUR/USD, GBP/USD, USD/JPY)
  - Energy sector news
  - Key sources: Reuters Commodities, OilPrice.com, Kitco

TRUSTED FINANCIAL SOURCES:
  Primary: Bloomberg, Reuters, Wall Street Journal, Financial Times
  Market Data: Yahoo Finance, MarketWatch, CNBC, Bloomberg Terminal
  Crypto: CoinDesk, CoinTelegraph, The Block, Glassnode
  Analysis: Seeking Alpha, Benzinga, ZeroHedge, MacroVoices
"""

class NewsIntelligenceState(MessagesState):
    """State with content processor tracking"""
    # Định nghĩa trạng thái của quy trình LangGraph
    search_count: int = 0
    max_searches: int = 5
    final_summaries: List[Dict[str, Any]] = []
    processed_urls: List[str] = []
    content_processor_results: List[Dict[str, Any]] = []
    target_language: str = "en"
    tavily_search_results: List[Dict[str, Any]] = []
    processing_method: str = "tavily"
    symbols_to_track: List[str] = []
    sectors_to_analyze: List[str] = []
    additional_topics: str = ""
    max_results_per_search: int = 10
    model: str = "gpt-4.1-nano"
    provider_type: str = "openai"

class NewsContentProcessor:
    def __init__(self, api_key: str, model_name: str = "gpt-4.1-nano", provider_type: str = "openai"):
        """Initialize unified processor with provider support"""
        # Khởi tạo bộ xử lý nội dung với hỗ trợ đa nhà cung cấp LLM
        self.api_key = api_key
        self.model_name = model_name
        self.provider_type = provider_type
        self.logger = processor_logger
        
        # Initialize content processor for article/video extraction
        self.content_processor = ContentProcessor(
            api_key=api_key,
            model_name=model_name,
            provider_type=provider_type
        )
        
        # Initialize LLM for summaries
        if provider_type == "openai":
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=1.0,
                openai_api_key=api_key
            )
        elif provider_type == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=1.0
            )
        else:
            from langchain.chat_models import init_chat_model
            self.llm = init_chat_model(
                model=model_name,
                model_provider=provider_type,
                temperature=1.0
            )
    
    async def process_url_unified(
        self, 
        url: str, 
        target_language: str = "en",
        processing_method: str = "tavily"
    ) -> Optional[Dict[str, Any]]:
        """
        Process URL and return unified format output
        """
        # Xử lý URL và trả về định dạng thống nhất
        
        if not url or not url.startswith("http"):
            self.logger.warning(f"Invalid URL: {url}")
            return None
        
        try:
            if processing_method == "tavily":
                result = await self._process_with_tavily(url, target_language)
                if not result:
                    self.logger.info(f"Tavily failed, fallback to content_processor for {url}")
                    result = await self._process_with_content_processor(url, target_language)
            else:
                result = await self._process_with_content_processor(url, target_language)
                if not result:
                    self.logger.info(f"Content processor failed, fallback to Tavily for {url}")
                    result = await self._process_with_tavily(url, target_language)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process {url}: {str(e)}")
            return None
    
    
    # Methods 1: Content Processor (articles/videos) using crawl4AI and selenium
    async def _process_with_content_processor(
        self, 
        url: str, 
        target_language: str
    ) -> Optional[Dict[str, Any]]:
        """Process using content processor and format output"""
        # Xử lý bằng ContentProcessor (sâu hơn)
        
        try:
            self.logger.info(f"Processing with content_processor: {url}")
            
            # Process URL
            result = await self.content_processor.process_url(
                url=url,
                target_language=target_language,
                print_progress=False
            )
            
            if result.get("status") != "success":
                return None
            
            # Extract key information
            title = result.get("title", "Article")
            content_type = result.get("type", "article")
            
            # Create brief snippet from summary (first 200-300 chars)
            summary = result.get("summary", "")
            snippet = summary[:250] + "..." if len(summary) > 250 else summary
            
            # Get full content
            full_content = result.get("original_text") or result.get("transcript", "")
            
            # Build unified format
            unified_result = {
                "title": title,
                "url": url,
                "content": snippet,  # Brief snippet like Tavily
                "full_content": full_content,  # Full extracted content
                "full_summary": summary,  # Full AI-generated summary
                "source": self._extract_domain(url),
                "published_date": datetime.now().strftime("%Y-%m-%d"),  # Default to today
                "score": 0.95,  # High score for successfully processed
                "processing_method": "content_processor",
                "content_type": content_type,
                "source_language": result.get("source_language", "en"),
                "target_language": target_language
            }
            
            # Add video info if applicable
            if content_type == "video" and result.get("video_info"):
                unified_result["video_info"] = result["video_info"]
            
            # Removed icon checkmark
            self.logger.info(f"Successfully processed with content_processor: {url}")
            return unified_result
            
        except Exception as e:
            self.logger.error(f"Content processor failed for {url}: {str(e)}")
            return None
        
    
    # Methods 2: Tavily Extract
    async def _process_with_tavily(
        self, 
        url: str, 
        target_language: str
    ) -> Optional[Dict[str, Any]]:
        """Process using Tavily extract and format output"""
        # Xử lý bằng Tavily extract (nhanh hơn)
        
        try:
            self.logger.info(f"Processing with tavily_extract: {url}")
            
            # Extract content with Tavily
            result = tavily_extract_content.invoke({"url": url})
            
            if isinstance(result, str):
                data = json.loads(result)
            else:
                data = result
            
            if not data or not data.get('content'):
                return None
            
            # Generate summary snippet
            content = data.get('content', '')
            snippet = content[:250] + "..." if len(content) > 250 else content
            
            # Build unified format
            unified_result = {
                "title": data.get('title', 'Article'),
                "url": url,
                "content": snippet,  # Brief snippet
                "full_content": content,  # Full content from Tavily
                "full_summary": "",  # Will be generated in synthesis
                "source": self._extract_domain(url),
                "published_date": data.get('published_date', datetime.now().strftime("%Y-%m-%d")),
                "score": data.get('score', 0.9),
                "processing_method": "tavily",
                "content_type": "article",
                "source_language": "en",  # Tavily usually returns English
                "target_language": target_language
            }
            
            # Removed icon checkmark
            self.logger.info(f"Successfully processed with tavily: {url}")
            return unified_result
            
        except Exception as e:
            self.logger.error(f"Tavily extract failed for {url}: {str(e)}")
            return None
    

    async def process_batch_unified(
        self, 
        urls: List[str], 
        target_language: str = "en",
        processing_method: str = "tavily"  
    ) -> List[Dict[str, Any]]:
        """Process multiple URLs and return unified format"""
        # Xử lý hàng loạt URL
        
        results = []
        
        for url in urls:
            result = await self.process_url_unified(url, target_language, processing_method)
            if result:
                results.append(result)
            else:
                # Add placeholder for failed URLs
                results.append({
                    "title": "Failed to process",
                    "url": url,
                    "content": "",
                    "full_content": "",
                    "source": self._extract_domain(url),
                    "score": 0,
                    "processing_method": "failed",
                    "error": "Processing failed"
                })
        
        return results
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        # Trích xuất tên miền từ URL
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '')
        except:
            return 'unknown'


async def news_intelligence_agent(state: NewsIntelligenceState) -> dict:
    """Agent with model/provider support"""
    
    agent_logger.info(f"News Agent - Step {state['search_count'] + 1}/{state['max_searches']}")
    # Removed robot icon
    print(f"\n{'='*50}\nNews Agent - Step {state['search_count'] + 1}\n{'='*50}")
    
    model = state.get('model', 'gpt-4.1-nano')
    provider_type = state.get('provider_type', 'openai')

    if provider_type == "openai":
        llm = ChatOpenAI(model=model, temperature=1.0)
    elif provider_type == "ollama":
        llm = ChatOllama(model=model, temperature=1.0)
    else:
        from langchain.chat_models import init_chat_model
        llm = init_chat_model(
            model=model,
            model_provider=provider_type,
            temperature=1.0
        )

    all_tools = [tavily_search, tavily_extract_content, tavily_crawl, tavily_map_site]
    llm_with_tools = llm.bind_tools(all_tools)
    
    # Dynamic query building based on context
    search_queries = []
    
    # Symbol-specific queries
    if state.get('symbols_to_track'):
        symbols = state['symbols_to_track']
        for symbol in symbols[:3]:  # Top 3 symbols per iteration
            search_queries.append(f"{symbol} stock price news today analyst rating")
            search_queries.append(f"{symbol} earnings report guidance forecast")
    
    # Sector-specific queries
    elif state.get('sectors_to_analyze'):
        sectors = state['sectors_to_analyze']
        for sector in sectors:
            search_queries.append(f"{sector} sector analysis trends market outlook")
            search_queries.append(f"{sector} industry news leading companies developments")
    
    # Default financial queries
    else:
        search_queries = [
            "stock market today S&P 500 Nasdaq Dow Jones",
            "Bitcoin Ethereum cryptocurrency news today",
            "Federal Reserve FOMC interest rates inflation",
            "NVIDIA AMD semiconductor stocks AI chips",
            "earnings reports tech stocks AAPL MSFT GOOGL",
            "oil gold commodities forex USD",
            "IPO SPAC merger acquisition news",
            "SEC regulation crypto enforcement"
        ]
    
    # Select query based on iteration
    query_index = state['search_count'] % len(search_queries)
    suggested_query = search_queries[query_index] if search_queries else "financial market news today"
    
    # Add additional topics if provided
    if state.get('additional_topics'):
        suggested_query = f"{suggested_query} {state['additional_topics']}"
    
    # Get max results per search from state
    max_results = state.get('max_results_per_search', 10)
    
    system_prompt = f"""
You are a financial news analyst gathering market intelligence.

**Current Status:**
- Iteration: {state['search_count'] + 1}/{state['max_searches']}
- Max results per search: {max_results}

**Your Task:**
Search for RECENT news (last 24-48 hours) using specific queries.

**Suggested query for this iteration:**
"{suggested_query}"

**Important:**
- Use tavily_search with max_results={max_results}
- Focus on reputable sources
- Look for market-moving news with data and analysis

Call tavily_search now with the suggested query.
"""
    
    messages = state["messages"]
    invocation_messages = [HumanMessage(content=system_prompt)] + messages
    
    # Removed search icon
    print(f"Searching: {suggested_query[:80]}...")
    
    response = llm_with_tools.invoke(invocation_messages)
    new_search_count = state['search_count'] + 1
    
    # Removed checkmark icon
    print(f"Search complete. Iteration: {new_search_count}")
    
    return {
        "messages": messages + [response],
        "search_count": new_search_count
    }


async def process_content_node(state: NewsIntelligenceState) -> dict:
    """Process content - reuse Tavily search data instead of re-fetching"""
    
    # Removed graph icon
    print("\nProcessing News Content...")
    agent_logger.info("Starting unified content processing")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        error_logger.error("OPENAI_API_KEY not found")
        return state
    
    processing_method = state.get('processing_method', 'tavily')
    model = state.get('model', 'gpt-4.1-nano')
    provider_type = state.get('provider_type', 'openai')
    
    # Removed pin icon
    print(f"Processing: method={processing_method}, model={model}, provider={provider_type}")
    
    url_to_tavily_data = {}  # Store mapping
    urls_to_process = []
    
    for msg in state["messages"]:
        if msg.type == "tool" and msg.content:
            try:
                data = json.loads(msg.content)
                
                if isinstance(data, list):
                    tavily_logger.info(f"Tavily returned {len(data)} results")
                    
                    for idx, item in enumerate(data):
                        if isinstance(item, dict) and 'url' in item:
                            url = item['url']
                            title = item.get('title', 'No title')
                            
                            tavily_logger.info(f"  [{idx+1}] {title[:60]}... - {url}")
                            
                            if url and url.startswith("http"):
                                urls_to_process.append(url)
                                # Store original Tavily data
                                url_to_tavily_data[url] = item
                                
            except json.JSONDecodeError as e:
                error_logger.error(f"Failed to parse tool message: {str(e)}")
                continue
    
    # Filter new URLs
    new_urls = [url for url in urls_to_process if url not in state.get('processed_urls', [])]
    
    if new_urls:
        # Removed refresh icon
        print(f"Processing {len(new_urls)} URLs")

        if processing_method == "tavily":
            # Removed pin icon
            print("  Using Tavily search data (no re-fetch)")
            
            for idx, url in enumerate(new_urls):
                tavily_data = url_to_tavily_data.get(url)
                
                if not tavily_data:
                    print(f"[{idx+1}] No Tavily data for {url}")
                    continue
                
                # Convert Tavily search result to unified format
                unified_result = {
                    "title": tavily_data.get('title', 'Article'),
                    "url": url,
                    "content": tavily_data.get('content', '')[:250] + "...",  # Snippet
                    "full_content": tavily_data.get('content', ''),  # Full from search
                    "full_summary": "",  # Will be generated in synthesis
                    "source": NewsContentProcessor(api_key)._extract_domain(url),
                    "published_date": tavily_data.get('published_date', datetime.now().strftime("%Y-%m-%d")),
                    "score": float(tavily_data.get('score', 0.9)),
                    "processing_method": "tavily_search",  # Clear naming
                    "content_type": "article",
                    "source_language": "en",
                    "target_language": state.get('target_language', 'en')
                }
                
                state['content_processor_results'].append(unified_result)
                state['processed_urls'].append(url)
                
                # Removed checkmark icon
                print(f"  [{idx+1}] tavily_search: {unified_result['title'][:50]}...")
        
        elif processing_method == "content_processor":
            # Removed pin icon
            print("  Using content_processor for deep extraction")
            
            unified_processor = NewsContentProcessor(
                api_key=api_key,
                model_name=model,
                provider_type=provider_type
            )
            
            processed_results = await unified_processor.process_batch_unified(
                new_urls,
                target_language=state.get('target_language', 'en'),
                processing_method='content_processor'
            )
            
            for idx, (url, result) in enumerate(zip(new_urls, processed_results)):
                if result and not result.get('error'):
                    state['content_processor_results'].append(result)
                    state['processed_urls'].append(url)
                    # Removed checkmark icon
                    print(f"  [{idx+1}] content_processor: {result['title'][:50]}...")
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'Processing failed'
                    print(f"[{idx+1}] ERROR: {error_msg}")
                    
                    if url in url_to_tavily_data:
                        tavily_data = url_to_tavily_data[url]
                        fallback_result = {
                            "title": tavily_data.get('title', 'Article'),
                            "url": url,
                            "content": tavily_data.get('content', '')[:250] + "...",
                            "full_content": tavily_data.get('content', ''),
                            "full_summary": "",
                            "source": unified_processor._extract_domain(url),
                            "published_date": tavily_data.get('published_date', datetime.now().strftime("%Y-%m-%d")),
                            "score": float(tavily_data.get('score', 0.8)),
                            "processing_method": "tavily_search_fallback",
                            "content_type": "article",
                            "target_language": state.get('target_language', 'en')
                        }
                        state['content_processor_results'].append(fallback_result)
                        # Removed fallback/reload icon
                        print(f"  Fallback to Tavily: {fallback_result['title'][:50]}...")
        
        else:
            print(f"Unknown processing method: {processing_method}")
    
    else:
        # Removed pin icon
        print("No new URLs to process")
    
    return state


async def synthesis_agent(state: NewsIntelligenceState) -> dict:
    """synthesis with professional investor-focused format"""
    
    # Removed microscope icon
    print(f"\n{'='*50}\nProfessional News Synthesis\n{'='*50}")
    agent_logger.info("Starting enhanced news synthesis")

    model = state.get('model', 'gpt-4.1-nano')
    provider_type = state.get('provider_type', 'openai')

    if provider_type == "openai":
        llm = ChatOpenAI(model=model, temperature=0.7)
    elif provider_type == "ollama":
        llm = ChatOllama(model=model, temperature=0.7)
    else:
        from langchain.chat_models import init_chat_model
        llm = init_chat_model(
            model=model,
            model_provider=provider_type,
            temperature=0.7
        )
        
    # Get all processed articles
    all_articles = state.get('content_processor_results', [])
    
    if not all_articles:
        # Removed warning icon
        print("No articles found")
        return {"final_summaries": []}
    
    # Deduplicate
    unique_articles = deduplicate_news_items(all_articles)
    
    # Sort by relevance score
    unique_articles.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    final_summaries = []
    target_language = state.get('target_language', 'en')
    
    language_names = {
        "en": "English",
        "vi": "Vietnamese", 
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish"
    }
    
    target_lang_name = language_names.get(target_language, "English")
    
    # Get context for summaries
    symbols = state.get('symbols_to_track', [])
    sectors = state.get('sectors_to_analyze', [])
    
    for i, article in enumerate(unique_articles, 1):
        print(f"\n[{i}/{len(unique_articles)}] Processing: {article['title'][:60]}...")
        
        # Get content for analysis
        content = article.get('full_content', article.get('content', ''))
        
        if not content:
            continue
        
        # Generate professional formatted summary (NO ICONS IN PROMPT)
        summary_prompt = f"""
You are a senior financial analyst creating a comprehensive news digest for sophisticated investors.
Analyze this article and create a PROFESSIONAL INVESTMENT-FOCUSED summary in {target_lang_name}.

CRITICAL: Write EVERYTHING in {target_lang_name}, including:
- ALL section headers/titles must be translated to {target_lang_name}
- ALL content must be in {target_lang_name}

ARTICLE DATA:
Title: {article['title']}
Date: {article.get('published_date', 'Today')}
Content: {content[:4000]}

{"Tracking Symbols: " + ', '.join(symbols) if symbols else ""}
{"Analyzing Sectors: " + ', '.join(sectors) if sectors else ""}

Create a structured summary in {target_lang_name} following this EXACT format:

## [Translate "HEADLINE" to {target_lang_name}]
[Create a clear, factual headline - NO clickbait, 10-15 words max]

## [Translate "SOURCE INFO" to {target_lang_name}]
[Source Name] • [YYYY-MM-DD] • [Author if available]

## [Translate "TL;DR" to {target_lang_name}] 
[Core value/impact in 1-2 concise sentences - what investors MUST know]

## [Translate "KEY POINTS" to {target_lang_name}]
• **Point 1**: [Key fact with **bold numbers** like **$2.3B** or **+15.4%**]
• **Point 2**: [Another crucial point with **bold metrics**]
• **Point 3**: [Third point if significant]
• **Point 4**: [Fourth point if needed]
[Maximum 6 bullet points, each under 20 words]

## [Translate "MARKET IMPACT" to {target_lang_name}]

### [Translate "Equities" to {target_lang_name}]
• **Immediate**: [How stocks react today/tomorrow]
• **Sectors**: [Which sectors/stocks most affected]
• **Leaders**: [Specific tickers likely to move]

### [Translate "Crypto" to {target_lang_name}]
• **BTC/ETH**: [Impact on major cryptos]
• **Altcoins**: [Which alts affected and why]
• **DeFi/NFT**: [If relevant]

### [Translate "Macro" to {target_lang_name}]
• **Rates**: [Interest rate implications]
• **Dollar**: [USD/DXY impact]
• **Commodities**: [Gold/Oil if relevant]

## [Translate "TICKERS & SECTORS" to {target_lang_name}]
**Mentioned**: [AAPL, NVDA, BTC, ETH, SPY, etc.]
**Sectors**: [Technology, Finance, Energy, etc.]

## [Translate "SENTIMENT & RISK" to {target_lang_name}]
**Market Sentiment**: [Bullish / Neutral / Bearish]
**Confidence Level**: [High/Medium/Low]
**Key Risks**: [Main uncertainty or risk factor]

## [Translate "WATCHLIST" to {target_lang_name}] (Not investment advice)
1. **Watch**: [Specific level/event to monitor]
2. **Track**: [Data release or catalyst]
3. **Alert**: [Price level or indicator]
[Maximum 5 items]

## [Translate "NOTABLE QUOTE" to {target_lang_name}]
"{"{"}[Most impactful quote under 25 words]{"}"}" - [Source Name]

## [Translate "SOURCES" to {target_lang_name}]
• Original: {article['url']}

IMPORTANT RULES:
1. Use **bold** for ALL numbers, percentages, dollar amounts
2. Keep each bullet point under 20 words
3. Be specific with tickers and price levels
4. NO generic statements - only actionable insights
5. Format dates as YYYY-MM-DD
6. If article mentions video content, add "Video Insights" section
7. Write everything in {target_lang_name}
8. Focus on what matters for trading/investing decisions
9. DO NOT use emojis or icons in the output.

Create the summary now in {target_lang_name}:"""
        
        try:
            response = await llm.ainvoke(summary_prompt)
            formatted_summary = response.content
            # Removed checkmark icon
            # print(f"  Generated professional summary")
            
            # Parse the formatted summary to extract structured data
            structured_data = parse_formatted_summary(formatted_summary, article)
            
        except Exception as e:
            error_logger.error(f"Summary generation failed: {str(e)}")
            structured_data = create_fallback_summary(article, target_language)
        
        # Build final summary with BOTH old and new format for compatibility
        final_summary = {
            "title": article['title'],
            "url": article['url'],
            "source": article['source'],
            "published_date": article.get('published_date', datetime.now().strftime("%Y-%m-%d")),
            # IMPORTANT: Keep 'summary' field for backward compatibility
            "summary": formatted_summary if 'formatted_summary' in locals() else structured_data.get('formatted_summary', ''),
            # Also include new formatted fields
            "formatted_summary": formatted_summary if 'formatted_summary' in locals() else structured_data.get('formatted_summary', ''),
            "structured_data": structured_data,
            "content_snippet": article.get('content', '')[:200] + "...",
            "score": float(article.get('score', 0)) if article.get('score') is not None else 0.0,
            "content_type": article.get('content_type', 'article'),
            "processing_method": article['processing_method'],
            "source_language": article.get('source_language', 'en'),
            "target_language": target_language
        }
        
        # Add context tags
        if symbols:
            final_summary['related_symbols'] = symbols
        if sectors:
            final_summary['related_sectors'] = sectors
        
        # Add video info if exists
        if article.get('video_info'):
            final_summary['video_info'] = article['video_info']
            final_summary['has_video'] = True
        
        final_summaries.append(final_summary)
    
    # Generate market overview if multiple articles
    if len(final_summaries) > 3:
        market_overview = await generate_market_overview(final_summaries, target_language, llm)
        # Add overview as first item
        final_summaries.insert(0, market_overview)
    
    # Save results with enhanced format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    context_type = "symbols" if symbols else "sectors" if sectors else "market"
    summary_path = os.path.join(OUTPUT_DIR, f"professional_{context_type}_{timestamp}.json")
    
    # Also save HTML version for better readability
    html_path = os.path.join(OUTPUT_DIR, f"professional_{context_type}_{timestamp}.html")
    generate_html_report(final_summaries, html_path, target_language)
    
    # with open(summary_path, 'w', encoding='utf-8') as f:
    #     json.dump(final_summaries, f, indent=2, ensure_ascii=False)
    
    # Removed disk icon
    # print(f"\nSaved {len(final_summaries)} professional summaries")
    # # Removed globe icon
    # print(f"HTML: {html_path}")
    
    # Print summary statistics
    # Removed chart icon
    # print("\nProcessing Statistics:")
    # print(f"  - Total articles: {len(final_summaries)}")
    # print(f"  - With structured data: {sum(1 for s in final_summaries if s.get('structured_data'))}")
    # print(f"  - Videos included: {sum(1 for s in final_summaries if s.get('has_video'))}")
    
    return {"final_summaries": final_summaries}


def route_after_agent(state: NewsIntelligenceState) -> Literal["tools", "process_content", "synthesis"]:
    """Router for workflow"""
    # Điều hướng sau khi Agent thực thi
    print("Router: Deciding next step...")
    
    if state['search_count'] >= state['max_searches']:
        print("  -> Max searches reached. Routing to synthesis.")
        return "synthesis"
    
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        tool_name = last_message.tool_calls[0]['name']
        print(f"  -> Tool call '{tool_name}' found. Routing to tools.")
        return "tools"
    else:
        print("  -> No tool call. Routing to synthesis.")
        return "synthesis"
    

def deduplicate_news_items(items: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Deduplicate news items using semantic similarity"""
    # Loại bỏ tin trùng lặp dựa trên độ tương đồng ngữ nghĩa
    if not items or len(items) <= 1: 
        return items
    
    agent_logger.info(f"Deduplicating {len(items)} articles")
    
    try:
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        
        texts = []
        for item in items:
            # Use title and content snippet for deduplication
            text = f"{item.get('title', '')} {item.get('content', '')[:200]}"
            texts.append(text)
        
        embeddings = embeddings_model.embed_documents(texts)
        
        unique_indices, seen_indices = [], set()
        
        for i in range(len(items)):
            if i in seen_indices: 
                continue
                
            unique_indices.append(i)
            seen_indices.add(i)
            
            for j in range(i + 1, len(items)):
                if j in seen_indices: 
                    continue
                    
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                
                if similarity >= threshold:
                    agent_logger.info(f"Duplicate found: {similarity:.2%}")
                    seen_indices.add(j)
        
        unique_items = [items[i] for i in unique_indices]
        agent_logger.info(f"Kept {len(unique_items)} unique articles")
        return unique_items
        
    except Exception as e:
        error_logger.error(f"Deduplication error: {str(e)}")
        return items


def parse_formatted_summary(formatted_text: str, article: Dict) -> Dict[str, Any]:
    """Parse the formatted summary into structured data"""
    # Phân tích chuỗi tóm tắt thành dữ liệu có cấu trúc (REGEX UPDATED FOR NO ICONS)
    
    structured = {
        "formatted_summary": formatted_text,
        "sections": {}
    }
    
    # Extract sections using regex
    # Regex đã được cập nhật để không tìm các icon như ⚡, 🎯, etc.
    
    # Extract TL;DR
    tldr_match = re.search(r'##\s*TL;DR.*?\n(.*?)(?=##|\Z)', formatted_text, re.DOTALL | re.IGNORECASE)
    if tldr_match:
        structured["tldr"] = tldr_match.group(1).strip()
    
    # Extract key points
    key_points_match = re.search(r'##\s*KEY POINTS.*?\n(.*?)(?=##|\Z)', formatted_text, re.DOTALL | re.IGNORECASE)
    if key_points_match:
        points = re.findall(r'•\s*(.*?)(?=\n|$)', key_points_match.group(1))
        structured["key_points"] = [p.strip() for p in points]
    
    # Extract sentiment (Matches words only, no icons)
    sentiment_match = re.search(r'\*\*Market Sentiment\*\*:\s*\[(.*?)\]', formatted_text)
    if sentiment_match:
        sentiment_text = sentiment_match.group(1)
        if "Bullish" in sentiment_text:
            structured["sentiment"] = "bullish"
        elif "Bearish" in sentiment_text:
            structured["sentiment"] = "bearish"
        else:
            structured["sentiment"] = "neutral"
    
    # Extract mentioned tickers
    tickers_match = re.search(r'\*\*Mentioned\*\*:\s*\[(.*?)\]', formatted_text)
    if tickers_match:
        tickers_text = tickers_match.group(1)
        structured["tickers"] = [t.strip() for t in tickers_text.split(',')]
    
    # Extract watchlist items
    watchlist_match = re.search(r'##\s*WATCHLIST.*?\n(.*?)(?=##|\Z)', formatted_text, re.DOTALL | re.IGNORECASE)
    if watchlist_match:
        watchlist_items = re.findall(r'\d+\.\s*\*\*(.*?)\*\*:\s*(.*?)(?=\n|$)', watchlist_match.group(1))
        structured["watchlist"] = [{"action": item[0], "detail": item[1].strip()} for item in watchlist_items]
    
    # Extract notable quote
    quote_match = re.search(r'"\s*(.*?)\s*"\s*-\s*(.*?)(?=\n|$)', formatted_text)
    if quote_match:
        structured["quote"] = {
            "text": quote_match.group(1),
            "source": quote_match.group(2).strip()
        }
    
    return structured


def create_fallback_summary(article: Dict, target_language: str) -> Dict[str, Any]:
    """Create a basic structured summary as fallback"""
    # Tạo tóm tắt dự phòng (loại bỏ icon)
    
    summary_text = f"""
## {article['title']}

## SOURCE INFO
{article['source']} • {article.get('published_date', 'Today')}

## TL;DR
{article.get('content', '')[:300]}...

## KEY POINTS
• Article from {article['source']}
• Processing method: {article['processing_method']}
• Content type: {article['content_type']}

## MARKET IMPACT
Market impact analysis not available in fallback mode.

## SENTIMENT & RISK
**Market Sentiment**: [Neutral]
**Confidence Level**: Low (fallback mode)

## SOURCES
• Original: {article['url']}
"""
    
    return {
        "formatted_summary": summary_text,
        "tldr": article.get('content', '')[:150],
        "sentiment": "neutral",
        "is_fallback": True
    }


async def generate_market_overview(summaries: List[Dict], target_language: str, llm) -> Dict[str, Any]:
    """Generate an overall market overview from multiple summaries"""
    # Tạo tổng quan thị trường từ nhiều bài tóm tắt
    
    # Collect all sentiments and tickers
    all_sentiments = []
    all_tickers = set()
    all_sectors = set()
    
    for summary in summaries:
        if 'structured_data' in summary:
            data = summary['structured_data']
            if 'sentiment' in data:
                all_sentiments.append(data['sentiment'])
            if 'tickers' in data:
                all_tickers.update(data['tickers'])
        if 'related_sectors' in summary:
            all_sectors.update(summary['related_sectors'])
    
    # Calculate overall sentiment
    sentiment_counts = {
        'bullish': all_sentiments.count('bullish'),
        'neutral': all_sentiments.count('neutral'),
        'bearish': all_sentiments.count('bearish')
    }
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    
    # Language mapping for titles
    language_names = {
        "en": "English",
        "vi": "Vietnamese", 
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish"
    }
    
    target_lang_name = language_names.get(target_language, "English")
    
    # Create overview prompt - FULLY IN TARGET LANGUAGE - NO ICONS
    overview_prompt = f"""
Create a COMPREHENSIVE MARKET OVERVIEW summarizing {len(summaries)} news articles.
Write EVERYTHING in {target_lang_name}, including all headers, titles, and content.

Overall Sentiment: {overall_sentiment.upper()} (Bullish: {sentiment_counts['bullish']}, Neutral: {sentiment_counts['neutral']}, Bearish: {sentiment_counts['bearish']})
Top Tickers: {', '.join(list(all_tickers)[:10])}
Sectors Covered: {', '.join(all_sectors)}

IMPORTANT: Write ALL text in {target_lang_name}, including:
- The title "MARKET OVERVIEW" should be translated to {target_lang_name}
- All section headers must be in {target_lang_name}
- All content must be in {target_lang_name}
- DO NOT use emojis or icons.

Create the overview following this format (translate each section header to {target_lang_name}):

# [Translate "MARKET OVERVIEW" to {target_lang_name}]

## [Translate "Market Pulse" to {target_lang_name}]
[2-3 sentences on overall market direction in {target_lang_name}]

## [Translate "Top Movers & Themes" to {target_lang_name}]
- [Major trend with tickers in {target_lang_name}]
- [Second trend in {target_lang_name}]
- [Third trend in {target_lang_name}]

## [Translate "Key Risks" to {target_lang_name}]
- [Main risk factor in {target_lang_name}]
- [Secondary risk in {target_lang_name}]

## [Translate "What to Watch" to {target_lang_name}]
- [Key event/data in {target_lang_name}]
- [Important level in {target_lang_name}]

Write everything in {target_lang_name}:"""
    
    try:
        response = await llm.ainvoke(overview_prompt)
        overview_text = response.content
    except:
        # Fallback with proper translation
        overview_titles = {
            "en": "Market Overview",
            "vi": "Tổng quan thị trường",
            "zh": "市场概览",
            "ja": "市場概要",
            "ko": "시장 개요",
            "fr": "Aperçu du marché",
            "de": "Marktübersicht",
            "es": "Resumen del mercado"
        }
        
        title = overview_titles.get(target_language, "Market Overview")
        overview_text = f"# {title}\n\nProcessed {len(summaries)} articles. Overall sentiment: {overall_sentiment}"
    
    # Get translated title
    overview_titles_map = {
        "en": "MARKET OVERVIEW",
        "vi": "TỔNG QUAN THỊ TRƯỜNG",
        "zh": "市场概览",
        "ja": "市場概要",
        "ko": "시장 개요",
        "fr": "APERÇU DU MARCHÉ",
        "de": "MARKTÜBERSICHT",
        "es": "RESUMEN DEL MERCADO"
    }
    
    translated_title = overview_titles_map.get(target_language, "MARKET OVERVIEW")
    
    return {
        "title": translated_title,  # Use translated title
        "url": "",
        "source": "AI Analysis", 
        "published_date": datetime.now().strftime("%Y-%m-%d"),
        "summary": overview_text,
        "formatted_summary": overview_text,
        "structured_data": {
            "is_overview": True,
            "article_count": len(summaries),
            "overall_sentiment": overall_sentiment,
            "sentiment_distribution": sentiment_counts,
            "top_tickers": list(all_tickers)[:20],
            "sectors": list(all_sectors)
        },
        "content_snippet": f"Market overview of {len(summaries)} articles. Sentiment: {overall_sentiment}",
        "content_type": "overview",
        "processing_method": "synthesis",
        "target_language": target_language
    }


def generate_html_report(summaries: List[Dict], output_path: str, target_language: str):
    """Generate an HTML report for better readability"""
    # Tạo báo cáo HTML (đã loại bỏ icon trong giao diện)
    
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Professional Market Intelligence Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .article {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .overview {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .overview h1, .overview h2, .overview h3 {
            color: white;
        }
        h1 { font-size: 24px; margin-bottom: 8px; }
        h2 { 
            font-size: 18px; 
            margin-top: 20px; 
            margin-bottom: 12px;
            color: #2563eb;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 4px;
        }
        h3 { font-size: 16px; margin-top: 16px; margin-bottom: 8px; }
        .source-info {
            color: #6b7280;
            font-size: 14px;
            margin-bottom: 16px;
        }
        .tldr {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 12px;
            margin: 16px 0;
            border-radius: 4px;
        }
        .key-points {
            background: #f0f9ff;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
        }
        .key-points li {
            margin-bottom: 8px;
        }
        .sentiment {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin: 8px 0;
        }
        .bullish { background: #10b981; color: white; }
        .neutral { background: #f59e0b; color: white; }
        .bearish { background: #ef4444; color: white; }
        .tickers {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 12px 0;
        }
        .ticker {
            background: #e0e7ff;
            color: #4338ca;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 13px;
            font-weight: 600;
        }
        .watchlist {
            background: #fef3c7;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
        }
        .watchlist ol {
            margin: 8px 0 0 20px;
            padding: 0;
        }
        .watchlist li {
            margin-bottom: 8px;
        }
        strong {
            color: #1e40af;
            font-weight: 600;
        }
        .quote {
            font-style: italic;
            border-left: 3px solid #6b7280;
            padding-left: 16px;
            margin: 16px 0;
            color: #4b5563;
        }
        .market-impact {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }
        .impact-section {
            background: #f9fafb;
            padding: 12px;
            border-radius: 8px;
        }
        .impact-section h4 {
            margin: 0 0 8px 0;
            color: #374151;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .timestamp {
            text-align: center;
            color: #6b7280;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
        }
    </style>
</head>
<body>
"""
    
    for idx, summary in enumerate(summaries):
        is_overview = summary.get('structured_data', {}).get('is_overview', False)
        
        article_class = "article overview" if is_overview else "article"
        
        html_content += f'<div class="{article_class}">'
        
        # Add formatted summary as markdown-style HTML
        formatted_summary = summary.get('formatted_summary', '')
        
        # Convert markdown to HTML (basic conversion)
        formatted_html = formatted_summary.replace('##', '</h2><h2>').replace('#', '<h1>')
        formatted_html = formatted_html.replace('**', '<strong>').replace('**', '</strong>')
        formatted_html = formatted_html.replace('• ', '<li>').replace('\n\n', '</p><p>')
        formatted_html = f"<div>{formatted_html}</div>"
        
        html_content += formatted_html
        
        # Add source link if not overview (Icon removed)
        if not is_overview and summary.get('url'):
            html_content += f'<p><a href="{summary["url"]}" target="_blank">Read Original Article</a></p>'
        
        html_content += '</div>'
    
    # Add timestamp
    html_content += f"""
    <div class="timestamp">
        Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    # Removed memo icon
    print(f"HTML report saved to: {output_path}")    

# Build workflow
def create_news_workflow():
    """Create workflow"""
    # Tạo luồng xử lý (workflow)
    
    tools = [tavily_search, tavily_extract_content, tavily_crawl, tavily_map_site]
    tool_node = ToolNode(tools)
    
    workflow = StateGraph(NewsIntelligenceState)
    
    workflow.add_node("agent", news_intelligence_agent)
    workflow.add_node("tools", tool_node)
    workflow.add_node("process_content", process_content_node)
    workflow.add_node("synthesis", synthesis_agent)
    
    workflow.add_edge(START, "agent")
    workflow.add_edge("tools", "process_content")
    workflow.add_edge("process_content", "agent")
    
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "process_content": "process_content",
            "synthesis": "synthesis"
        }
    )
    
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()


news_intelligence_agent_compiled = create_news_workflow()