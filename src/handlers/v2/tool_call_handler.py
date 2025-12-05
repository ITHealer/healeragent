import os
import json
from openai import OpenAI 
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from src.utils.constants import APIModelName
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.config import settings


# ============================================================================
# Pydantic Models
# ============================================================================

class ClarificationOption(BaseModel):
    type: str = Field(description="Type: 'stock' or 'crypto'")
    symbol: str = Field(description="Normalized symbol (e.g., BTC or BTC-USD)")
    description: str = Field(description="Human-readable description")

class TemporalContext(BaseModel):
    detected: bool = Field(default=False, description="Whether temporal context was found")
    period: Optional[str] = Field(default=None, description="Relative period: 'today', 'yesterday', 'last_week', 'last_month'")
    timeframe: Optional[str] = Field(default=None, description="Chart timeframe: '1d', '1w', '1m', '3m', '1y'")

class LLM_Decision(BaseModel):
    requires_tools: bool = Field(description="Whether tools are needed")
    tool_name: str = Field(default="", description="Selected tool name")
    symbols: List[str] = Field(default_factory=list, description="Extracted symbols")
    requires_clarification: bool = Field(default=False, description="Whether user clarification is needed")
    clarification_options: List[ClarificationOption] = Field(default_factory=list, description="Options for ambiguous symbols")
    temporal_context: TemporalContext = Field(default_factory=TemporalContext, description="Temporal information")
    reasoning: str = Field(default="", description="Brief explanation of decision")



# ============================================================================
# Define Tools
# ============================================================================

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            'name': 'showStockPrice',
            'description': 'Shows current price, price performance, valuation metrics, price targets, and price-related analysis for specific stocks. Use for questions about: "How is X performing?", "X price target", "X valuation", "X stock movement", current price queries. NOT for quarterly/annual business performance.',
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'additionalProperties': False,
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
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'additionalProperties': False,
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
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'additionalProperties': False,
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
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'additionalProperties': False,
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
            'strict': True,
            'parameters': {
                'type': 'object',
                'additionalProperties': False,
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
            'strict': True,
            'parameters': {
                'type': 'object',
                'additionalProperties': False,
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
            'strict': True,
            'parameters': {
                'type': 'object',
                'additionalProperties': False,
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
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': ['symbols'],
                'additionalProperties': False,
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

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are an intelligent, multilingual financial query analyzer with deep market knowledge.

Your task: Analyze user queries and make smart decisions about tool usage.

CORE PRINCIPLES:
1. Only use tools for financial market queries (stocks, crypto, market data)
2. Detect ambiguous cases and flag for clarification
3. Extract temporal context when present
4. Let your training knowledge handle symbol mapping (don't hard-code)

DECISION MAKING:

**General Chat Detection:**
- Greetings, general questions, non-financial topics → requires_tools = false
- Examples: "hello", "what is AI", "translate this", "how are you"

**Financial Query Processing:**
When query is about stocks/crypto/markets:

1. **Symbol Extraction & Normalization:**
   - Company names → Ticker symbols (Apple → AAPL, Microsoft → MSFT)
   - Crypto full names → Symbols with -USD (Bitcoin → BTC-USD, Ethereum → ETH-USD)
   - Use your training data knowledge for mappings

2. **Ambiguity Detection (CRITICAL):**
   When a symbol could mean BOTH stock AND crypto, flag for clarification:
   
   Ambiguous Cases:
   - "BTC" alone → could be Bitcoin crypto (BTC-USD) OR BTC stock
   - "COIN" → could be Coinbase stock OR general crypto reference
   - "RIOT" → could be Riot Platforms stock OR riot reference
   
   When ambiguous:
   - Set requires_clarification = true
   - Provide clarification_options with both interpretations
   - Do NOT make assumptions
   
   Clear Cases (no clarification needed):
   - "Bitcoin" / "bitcoin cryptocurrency" → clearly crypto → BTC-USD
   - "BTC-USD" → already normalized → BTC-USD
   - "Apple" / "AAPL stock" → clearly stock → AAPL
   - Context clearly indicates one type

3. **Temporal Context Extraction:**
   Parse time references in ANY language:
   - "yesterday" / "hôm qua" / "昨天" → period: "yesterday", timeframe: "1d"
   - "last week" / "tuần trước" → period: "last_week", timeframe: "1w"
   - "last month" / "tháng trước" → period: "last_month", timeframe: "1m"
   - "last year" / "năm ngoái" → period: "last_year", timeframe: "1y"
   - No time reference → detected: false

4. **Tool Selection:**
   Based on query intent:
   - Price/performance/valuation → showStockPrice (stocks) or cryptoChart (crypto) -> symbol required
   - Charts/graphs/technical → showStockChart (stocks) or cryptoChart (crypto) -> symbol required
   - News/updates/events → showStockNews (both stocks and crypto) -> symbol required
   - Financial statements/earnings → showStockFinancials (stocks only) -> symbol required
   - Market overview → showMarketOverview (no symbols needed)
   - Trending/top movers → showTrendingStocks (no symbols needed)
   - Heatmap/screening → showStockHeatmap (no symbols needed)

EXAMPLES:

Example 1 - General Chat:
Query: "Hello, how are you?"
Output: {
  "requires_tools": false,
  "tool_name": "",
  "symbols": [],
  "requires_clarification": false,
  "clarification_options": [],
  "temporal_context": {"detected": false},
  "reasoning": "General greeting - no financial data needed"
}

Example 2 - Clear Stock Query:
Query: "What's Apple stock price?"
Output: {
  "requires_tools": true,
  "tool_name": "showStockPrice",
  "symbols": ["AAPL"],
  "requires_clarification": false,
  "clarification_options": [],
  "temporal_context": {"detected": false},
  "reasoning": "Clear stock price query for AAPL"
}

Example 3 - Clear Crypto Query:
Query: "Bitcoin price today"
Output: {
  "requires_tools": true,
  "tool_name": "cryptoChart",
  "symbols": ["BTC-USD"],
  "requires_clarification": false,
  "clarification_options": [],
  "temporal_context": {"detected": true, "period": "today", "timeframe": "1d"},
  "reasoning": "Clear crypto price query with temporal context"
}

Example 4 - AMBIGUOUS Case (needs clarification):
Query: "giá BTC"
Output: {
  "requires_tools": true,
  "tool_name": "",
  "symbols": ["BTC"],
  "requires_clarification": true,
  "clarification_options": [
    {"type": "crypto", "symbol": "BTC-USD", "description": "Bitcoin cryptocurrency"},
    {"type": "stock", "symbol": "BTC", "description": "BTC stock ticker"}
  ],
  "temporal_context": {"detected": false},
  "reasoning": "BTC is ambiguous - could be crypto or stock ticker"
}

Example 5 - Temporal Context:
Query: "TSLA stock price last week"
Output: {
  "requires_tools": true,
  "tool_name": "showStockPrice",
  "symbols": ["TSLA"],
  "requires_clarification": false,
  "clarification_options": [],
  "temporal_context": {"detected": true, "period": "last_week", "timeframe": "1w"},
  "reasoning": "Stock price query with temporal context"
}

Example 6 - Multiple Symbols:
Query: "Compare Microsoft and Google revenue"
Output: {
  "requires_tools": true,
  "tool_name": "showStockFinancials",
  "symbols": ["MSFT", "GOOGL"],
  "requires_clarification": false,
  "clarification_options": [],
  "temporal_context": {"detected": false},
  "reasoning": "Financial comparison query for multiple stocks"
}

Example 7 - Vietnamese with Temporal:
Query: "giá ethereum hôm qua"
Output: {
  "requires_tools": true,
  "tool_name": "cryptoChart",
  "symbols": ["ETH-USD"],
  "requires_clarification": false,
  "clarification_options": [],
  "temporal_context": {"detected": true, "period": "yesterday", "timeframe": "1d"},
  "reasoning": "Crypto price query in Vietnamese with temporal context"
}

CRITICAL RULES:
1. When in doubt about stock vs crypto → FLAG for clarification
2. Use training data for company → ticker mappings
3. Parse temporal context in ANY language
4. Never invent symbols not mentioned by user
5. General chat → requires_tools = false

Now analyze the user query and provide decision in JSON format.
"""

# ============================================================================
# Helper Functions
# ============================================================================

def format_decision_response(decision: LLM_Decision) -> Dict[str, Any]:
    # Determine message based on decision
    if not decision.requires_tools:
        message = "no_tools_needed"
    elif decision.requires_clarification:
        message = "clarification_needed"
    else:
        message = "success"
    
    response = {
        "message": message,
        "status": "200",
        "data": {
            "symbol": decision.symbols,
            "tool_name": decision.tool_name,
            "requires_clarification": decision.requires_clarification,
            "clarification_options": [opt.dict() for opt in decision.clarification_options],
            "temporal_context": decision.temporal_context.dict(),
            "reasoning": decision.reasoning
        }
    }
    
    return response


async def llm_decision(
    prompt: str,
    model_name: str,
    provider_type: str,
    conversation_history: Optional[List[Dict]] = None
) -> LLM_Decision:
    """
    Get decision from LLM with structured output
    
    Args:
        prompt: User query
        model_name: LLM model name
        provider_type: Provider type (openai, ollama)
        conversation_history: Optional conversation context
    
    Returns:
        IntelligentDecision with all metadata
    """
    
    # Build context-aware prompt
    context = ""
    if conversation_history:
        context = "\n\nConversation Context:\n"
        for msg in conversation_history:  
            content = msg[0]
            role = msg[1]
            context += f"{role}: {content}\n"
    
    full_prompt = f"{context}\n\nCurrent Query: {prompt}\n\nAnalyze and provide decision:"
    
    try:
        if provider_type == ProviderType.OPENAI:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set.")
            
            client = OpenAI(api_key=openai_api_key)
            
            response = client.chat.completions.create( 
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
                tools=OPENAI_TOOLS,
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            decision_dict = json.loads(content)
            decision = LLM_Decision(**decision_dict)
            return decision
            
        else:
            # Ollama - use JSON mode
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            content = response.choices[0].message.content
            decision_dict = json.loads(content)
            decision = LLM_Decision(**decision_dict)
            return decision
            
    except Exception as e:
        print(f"Error in intelligent_llm_decision: {e}")
        # Fallback to safe default
        return LLM_Decision(
            requires_tools=False,
            reasoning=f"Error occurred: {str(e)}"
        )


# ============================================================================
# Main API Functions
# ============================================================================

async def tool_call(
    prompt: str,
    model_name: str = APIModelName.GPT41Nano,
    provider_type: str = ProviderType.OPENAI,
    conversation_history: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Key features:
    1. Detects when query is general chat → no tool calls
    2. Flags ambiguous symbols for user clarification
    3. Extracts temporal context from queries
    4. Uses LLM's training data for symbol mapping (no hard-coding)
    
    Args:
        prompt: User query in any language
        model_name: LLM model name
        provider_type: Provider type (openai, ollama)
        conversation_history: Optional list of previous messages for context
    
    Returns:
        Dict with metadata:
    """
    
    try:
        # Get decision from LLM
        decision = await llm_decision(
            prompt=prompt,
            model_name=model_name,
            provider_type=provider_type,
            conversation_history=conversation_history
        )
        
        # Format response
        response = format_decision_response(decision)
        
        return response
        
    except Exception as e:
        print(f"Error in tool_call: {e}")
        return {
            "message": "failed",
            "status": "500",
            "data": {
                "symbol": [],
                "tool_name": "",
                "requires_clarification": False,
                "clarification_options": [],
                "temporal_context": {"detected": False},
                "reasoning": f"Error: {str(e)}"
            }
        }
    
