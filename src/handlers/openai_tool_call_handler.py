import os
from openai import OpenAI 
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.utils.constants import APIModelName
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.config import settings

TOOLS_REQUIRING_SYMBOLS = [
    "showStockChart", "showStockFinancials", "showStockPrice", 
    "showStockNews", "cryptoChart"
]

TOOLS_NO_PARAMS = [
    "showStockScreener", "showMarketOverview", "showTrendingStocks", "showStockHeatmap"
]

TOOLS_WITH_INTERVAL = ["showStockChart", "showStockFinancials", "cryptoChart"]


# Structured Output Models
class ToolArguments(BaseModel):
    symbols: Optional[List[str]] = Field(default_factory=list, description="List of stock/crypto symbols")

class FunctionCall(BaseModel):
    name: str = Field(description="Name of the tool to call")
    arguments: ToolArguments = Field(description="Arguments for the tool")

class ToolCallResponse(BaseModel):
    tool_calls: List[FunctionCall] = Field(default_factory=list, description="List of tool calls to execute")


def normalize_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not tool_calls:
        return []
    
    normalized_calls = []
    
    for call in tool_calls:
        try:
            if not isinstance(call, dict) or "function" not in call:
                print(f"Invalid tool call, missing field 'function': {call}")
                continue
            
            function_data = call["function"]
            if not isinstance(function_data, dict) or "name" not in function_data:
                print(f"Invalid tool function call, missing 'name' field: {function_data}")
                continue
            
            tool_name = function_data["name"]
            if tool_name in TOOLS_REQUIRING_SYMBOLS:
                if "arguments" not in function_data:
                    function_data["arguments"] = {}
                
                arguments = function_data["arguments"]
                if not isinstance(arguments, dict):
                    try:
                        if isinstance(arguments, str):
                            import json
                            arguments = json.loads(arguments)
                        else:
                            arguments = {}
                    except:
                        arguments = {}
                
                if "symbols" not in arguments:
                    arguments["symbols"] = None
                    print(f"Missing parameter 'symbols' for {tool_name}, set null")
                    continue 
                
                if not isinstance(arguments["symbols"], list):
                    if isinstance(arguments["symbols"], str):
                        arguments["symbols"] = [arguments["symbols"]]
                    else:
                        arguments["symbols"] = None
                        continue
                
                function_data["arguments"] = {"symbols": arguments["symbols"]}                
            else:
                function_data["arguments"] = {"symbols": []}
            
            normalized_calls.append(call)
            
        except Exception as e:
            print(f"Error processing tool call: {str(e)}")
            continue
    
    return normalized_calls

def format_response(normalized_calls: List[Dict[str, Any]]) -> Dict[str, Any]:    
    """
    Format response from normalized tool calls.
    """
    if not normalized_calls:
        return {
            "message": "failed",
            "status": "400",
            "data": {
                "symbol": [],
                "tool_name": ""
            }
        }
    
    all_symbols = []
    tool_name = ""
    
    for call in normalized_calls:
        function_data = call["function"]
        
        if not tool_name:
            tool_name = function_data["name"]
        
        arguments = function_data.get("arguments", {})
        
        if "symbols" in arguments and arguments["symbols"]:
            if isinstance(arguments["symbols"], list) and len(arguments["symbols"]) > 0:
                all_symbols.extend(arguments["symbols"])
    
    all_symbols = list(set(all_symbols))
    
    response = {
        "message": "success",
        "status": "200",
        "data": {
            "symbol": all_symbols,
            "tool_name": tool_name
        }
    }
    
    return response

# region Define Tool
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            'name': 'showStockPrice',
            'description': 'Shows current price, price performance, valuation metrics, price targets, and price-related analysis for specific stocks. Use for questions about: "How is X performing?", "X price target", "X valuation", "X stock movement", current price queries. NOT for quarterly/annual business performance.',
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'properties': {
                    'symbols': {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stock ticker symbols in standard format. Examples: AAPL, TSLA, MSFT, NVDA, GOOGL"
                    }
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'showStockFinancials',
            'description': 'Shows detailed financial data including earnings reports, revenue, profit margins, P/E ratios, balance sheets, cash flow statements, and quarterly/annual financial metrics for specific stocks.',
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'properties': {
                    'symbols': {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stock ticker symbols. Examples: AAPL, TSLA, MSFT"
                    }
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'showStockChart',
            'description': 'Shows visual charts and technical analysis for specific stocks. Use when users explicitly mention "chart", "graph", "visual", "technical analysis", or want to see price patterns and trends visually. NOT for quarterly/annual business performance.',
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'properties': {
                    'symbols': {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stock ticker symbols. Examples: AAPL, TSLA, MSFT"
                    }
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'showStockNews',
            'description': 'Shows latest news, recent events, earnings announcements, and market-moving information for specific stocks OR cryptocurrencies. Use for questions about "news", "why is X moving", "latest events", "Show me the latest news on symbol?", "recent developments" for both stocks and crypto.',
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'properties': {
                    'symbols': {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Stock ticker symbols. Examples: AAPL, TSLA, MSFT"
                    }
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'showMarketOverview',
            'description': 'Shows general market conditions, index performance (S&P 500, NASDAQ, DOW), sector performance, and overall market sentiment. Use for broad market questions without specific stocks mentioned.',
            'parameters': {
                'type': 'object',
                'properties': {}, 
                'required': []  
            },
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'showTrendingStocks',
            'description': 'Shows trending stocks, top gainers/losers, most active stocks, and market movers. Use for discovery queries like "trending stocks", "top performers", "biggest movers", "what stocks are hot".',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            },
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'showStockHeatmap',
            'description': 'Generates and displays a stock heatmap. Stocks can be filtered based on criteria like price range, market capitalization, dividend yield, P/E ratio, sector, industry, or country. Useful for discovering stocks matching specific financial metrics and visualizing their performance or other attributes in a heatmap format.',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': []
            },
        }
    },
    {
        "type": "function",
        "function": {
            'name': 'cryptoChart',
            'description': 'Shows cryptocurrency PRICE, PERFORMANCE, and CHARTS ONLY. Use ONLY for price queries, performance analysis, and visual charts such as for Bitcoin, Ethereum, Dogecoin, and other cryptocurrencies. DO NOT use for crypto news - use showStockNews instead.',
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'properties': {
                    'symbols': {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Crypto symbols with -USD suffix. Examples: BTC-USD, ETH-USD, DOGE-USD, ADA-USD"
                    }
                },
            },
        }
    },
]

async def agentic_openai_tool_call(
    prompt: str,
    model_name: str = APIModelName.GPT41Nano,
    provider_type: str = ProviderType.OPENAI
):
    """
    Agentic tool calling with support for OpenAI and Ollama providers.
    Uses Structured Outputs for consistent results.
    """
    
    system_context = """You are a multilingual stock and cryptocurrency market assistant expert. Understand the user's intent regardless of language and Match user requests to the right tool.

CORE PRINCIPLE: Only use tools when the user's question is DIRECTLY about stocks, cryptocurrencies, or financial markets.

DO NOT USE TOOLS when user asks about:
- Translation requests
- General conversation (greetings, how are you, etc.)
- Non-financial topics (weather, recipes, history, etc.)
- Programming/coding questions
- General knowledge questions
For these cases, respond naturally WITHOUT calling any tools.

TOOL SELECTION RULES (only for financial market queries):

1. showStockNews: News/updates/latest events for stocks OR cryptocurrencies
   ✅ USE FOR: "Bitcoin news", "BTC latest", "Tesla news", "AAPL updates", "ETH recent events", "crypto news"
   🎯 Keywords: news, latest, updates, recent, events, announcements, why moving
   📌 Symbols: Stock tickers (AAPL, TSLA) OR Crypto with -USD (BTC-USD, ETH-USD)

2. showStockPrice: Price/value/performance for STOCKS ONLY (NOT crypto)
   ✅ USE FOR: "Apple stock price", "NVDA performance", "Tesla valuation"
   ❌ DO NOT USE FOR: Crypto queries (use cryptoChart instead)
   🎯 Keywords: price, value, performance, valuation, target
   📌 Symbols: Stock tickers only (AAPL, TSLA, MSFT)

3. showStockChart: Visual chart/graph for STOCKS ONLY (NOT crypto)
   ✅ USE FOR: "Tesla chart", "MSFT graph", "technical analysis AAPL"
   ❌ DO NOT USE FOR: Crypto charts (use cryptoChart instead)
   🎯 Keywords: chart, graph, visual, technical analysis
   📌 Symbols: Stock tickers only (AAPL, TSLA, MSFT)

4. showStockFinancials: Earnings/revenue/financial reports for STOCKS
   ✅ USE FOR: "Apple Q4 earnings", "MSFT financial statements", "revenue"
   🎯 Keywords: earnings, revenue, financials, balance sheet, P/E ratio, profit
   📌 Symbols: Stock tickers only (AAPL, TSLA, MSFT)

5. cryptoChart: Cryptocurrency PRICE, PERFORMANCE, and CHARTS ONLY
   ✅ USE FOR: "Bitcoin price", "ETH chart", "BTC performance", "Dogecoin value"
   ❌ DO NOT USE FOR: Crypto news (use showStockNews instead)
   🎯 Keywords: price, chart, performance, value, analysis (for crypto)
   📌 Symbols: Crypto with -USD suffix (BTC-USD, ETH-USD, DOGE-USD)

6. showMarketOverview: General market conditions WITHOUT specific stocks/crypto
   ✅ USE FOR: "How is the market?", "S&P 500 today", "Market sentiment"
   🎯 Keywords: market, S&P, NASDAQ, DOW, sectors, overall

7. showTrendingStocks: Discovery queries for trending/top movers
   ✅ USE FOR: "What's trending?", "Top gainers", "Biggest losers", "Hot stocks"
   🎯 Keywords: trending, hot, movers, gainers, losers, active

8. showStockHeatmap: Visual heatmap or filtering stocks by criteria
   ✅ USE FOR: "Stock heatmap", "Top 10 tech stocks", "Filter by P/E"
   🎯 Keywords: heatmap, filter, top N, sector, industry

CRITICAL DECISION TREE FOR CRYPTO:
- IF query about crypto + "news/latest/updates/events" → showStockNews with BTC-USD
- IF query about crypto + "price/chart/performance/value" → cryptoChart with BTC-USD

EXAMPLES:
✅ "Show news btcusd" → showStockNews with ["BTC-USD"]
✅ "Bitcoin latest news" → showStockNews with ["BTC-USD"]
✅ "BTC price" → cryptoChart with ["BTC-USD"]
✅ "Ethereum chart" → cryptoChart with ["ETH-USD"]
✅ "Apple news" → showStockNews with ["AAPL"]
✅ "Tesla stock price" → showStockPrice with ["TSLA"]
✅ "Compare Apple and Tesla news" → showStockNews with ["AAPL", "TSLA"]

CONVERSION RULES:
- Company names to symbols: Apple→AAPL, Tesla→TSLA, Microsoft→MSFT, NVIDIA→NVDA
- Crypto to symbols: Bitcoin→BTC-USD, Ethereum→ETH-USD, Dogecoin→DOGE-USD

IMPORTANT:
- If question is NOT about financial markets → Do NOT use any tools
- Only call tools when user explicitly asks about stocks, crypto, or market data
- Do not invent symbols unless user explicitly mentions them"""
    
    try:
        if provider_type == ProviderType.OPENAI:
            # OpenAI with Structured Outputs
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")

            client = OpenAI(api_key=openai_api_key)
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_context},
                    {"role": "user", "content": f"User request: {prompt}\nSelect appropriate tool if needed."}
                ],
                tools=OPENAI_TOOLS,
                tool_choice="auto",
                temperature=0
            )

            raw_tool_calls = []
            if response.choices and response.choices[0].message.tool_calls:
                raw_tool_calls = [tc.model_dump() for tc in response.choices[0].message.tool_calls]

        else:
            ollama_base_url = settings.OLLAMA_ENDPOINT
            if not ollama_base_url.endswith('/v1'):
                ollama_base_url = f"{ollama_base_url}/v1"
            
            client = OpenAI(
                base_url=ollama_base_url,
                api_key="ollama"
            )
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_context},
                    {"role": "user", "content": f"User request: {prompt}\nSelect appropriate tool if needed."}
                ],
                tools=OPENAI_TOOLS,
                tool_choice="auto",
                temperature=0
            )
            
            raw_tool_calls = []
            if response.choices and response.choices[0].message.tool_calls:
                raw_tool_calls = [tc.model_dump() for tc in response.choices[0].message.tool_calls]


        if not raw_tool_calls:
            return {
                "message": "success",
                "status": "200",
                "data": {"symbol": [], "tool_name": ""}
            }

        normalized_tool_calls = normalize_tool_calls(raw_tool_calls)
        return format_response(normalized_tool_calls)

    except Exception as e:
        print(f"Error in agentic_openai_tool_call: {e}")
        raise e