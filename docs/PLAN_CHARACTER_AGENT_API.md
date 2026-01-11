# PLAN: Character Agent API Integration with Agent Loop

## Mục tiêu
Tích hợp các Character Agents (Warren Buffett, Ben Graham, Cathie Wood, etc.) vào Agent Loop hiện có, duy trì memory và conversation continuity.

---

## 1. KIẾN TRÚC TỔNG QUAN

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CHARACTER AGENT SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     API LAYER (New)                                  │   │
│  │  /api/v1/character-agents/{agent_id}/chat                           │   │
│  │  /api/v1/character-agents/{agent_id}/stream                         │   │
│  │  /api/v1/character-agents/list                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  CHARACTER REGISTRY (New)                            │   │
│  │  src/agents/characters/registry.py                                   │   │
│  │  ├─ CharacterRegistry (singleton)                                   │   │
│  │  ├─ CharacterConfig dataclass                                       │   │
│  │  └─ Analysis criteria per character                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  CHARACTER PERSONAS (New)                            │   │
│  │  src/agents/characters/personas/                                     │   │
│  │  ├─ base_persona.py          (Abstract base)                        │   │
│  │  ├─ warren_buffett.py        (Value investing)                      │   │
│  │  ├─ ben_graham.py            (Deep value)                           │   │
│  │  ├─ charlie_munger.py        (Mental models)                        │   │
│  │  ├─ cathie_wood.py           (Disruptive innovation)                │   │
│  │  ├─ michael_burry.py         (Contrarian)                           │   │
│  │  └─ ...                                                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              CHARACTER AGENT HANDLER (New)                           │   │
│  │  src/handlers/character_agent_handler.py                            │   │
│  │  ├─ Builds character-specific system prompt                         │   │
│  │  ├─ Injects analysis criteria into context                          │   │
│  │  ├─ Calls Agent Loop with character tools                           │   │
│  │  └─ Stores conversation with character metadata                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              AGENT LOOP (Existing - Enhanced)                        │   │
│  │  src/agents/unified/unified_agent.py                                │   │
│  │  ├─ Tool calling (with character-specific tools)                    │   │
│  │  ├─ Multi-turn execution                                            │   │
│  │  └─ Response generation IN CHARACTER                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                        │
│                                     ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              MEMORY SYSTEM (Existing - Enhanced)                     │   │
│  │  src/agents/memory/memory_manager.py                                │   │
│  │  ├─ Store with character_id metadata                                │   │
│  │  ├─ Retrieve character-specific context                             │   │
│  │  └─ Track character's analysis history                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CẤU TRÚC THƯ MỤC

```
src/
├── agents/
│   ├── characters/                     # NEW: Character system
│   │   ├── __init__.py
│   │   ├── registry.py                 # Character registry (singleton)
│   │   ├── models.py                   # Data models for characters
│   │   ├── analysis_criteria.py        # Analysis criteria per character
│   │   └── personas/                   # Character persona definitions
│   │       ├── __init__.py
│   │       ├── base_persona.py         # Abstract base class
│   │       ├── warren_buffett.py
│   │       ├── ben_graham.py
│   │       ├── charlie_munger.py
│   │       ├── peter_lynch.py
│   │       ├── cathie_wood.py
│   │       ├── michael_burry.py
│   │       ├── bill_ackman.py
│   │       ├── phil_fisher.py
│   │       ├── stanley_druckenmiller.py
│   │       └── aswath_damodaran.py
│   │
│   ├── tools/                          # ENHANCED: Add character tools
│   │   ├── financial_tools.py          # NEW: FMP data fetching tools
│   │   └── analysis_tools.py           # NEW: Analysis calculation tools
│   │
│   └── memory/                         # EXISTING: Enhanced
│       └── memory_manager.py           # Add character_id support
│
├── handlers/
│   └── character_agent_handler.py      # NEW: Character-specific handler
│
├── routers/
│   └── character_agent_chat.py         # NEW: API endpoints
│
└── hedge_fund/                         # EXISTING: Reference only
    └── agents/                         # Keep for backward compatibility
```

---

## 3. DATA MODELS

### 3.1 Character Configuration

```python
# src/agents/characters/models.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from enum import Enum

class InvestmentStyle(Enum):
    VALUE = "value"
    GROWTH = "growth"
    DEEP_VALUE = "deep_value"
    CONTRARIAN = "contrarian"
    MOMENTUM = "momentum"
    MACRO = "macro"
    ACTIVIST = "activist"
    QUANTITATIVE = "quantitative"

class RiskTolerance(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class AnalysisCriteria:
    """Metrics and thresholds specific to each character"""

    # Required metrics to fetch
    required_metrics: List[str] = field(default_factory=list)

    # Thresholds for bullish/bearish signals
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Weighted scoring formula
    scoring_weights: Dict[str, float] = field(default_factory=dict)

    # Red flags that trigger caution
    red_flags: List[str] = field(default_factory=list)

    # Green flags that trigger interest
    green_flags: List[str] = field(default_factory=list)

@dataclass
class CharacterConfig:
    """Complete configuration for a character agent"""

    # Identity
    agent_id: str
    name: str
    title: str
    description: str
    avatar_url: str

    # Investment profile
    investment_style: InvestmentStyle
    risk_tolerance: RiskTolerance
    time_horizon: str  # "short", "medium", "long", "forever"

    # Analysis criteria
    analysis_criteria: AnalysisCriteria

    # Personality & communication
    personality_traits: List[str]
    speaking_style: str
    famous_quotes: List[str]
    reference_investments: List[str]  # e.g., ["Coca-Cola", "See's Candies"]

    # Prompt components
    system_prompt: str
    analysis_prompt_template: str

    # Tools this character uses
    available_tools: List[str]

    # Specialties for routing
    specialties: List[str]
```

### 3.2 Character Response

```python
@dataclass
class CharacterAnalysis:
    """Analysis result from a character"""
    ticker: str
    signal: str  # "strong_buy", "buy", "hold", "sell", "strong_sell"
    confidence: float  # 0-100
    reasoning: str
    key_metrics: Dict[str, Any]
    character_scores: Dict[str, float]  # Character-specific scoring
    intrinsic_value: Optional[float] = None
    margin_of_safety: Optional[float] = None

@dataclass
class CharacterResponse:
    """Complete response from character agent"""
    content: str
    character_id: str
    character_name: str
    analyses: List[CharacterAnalysis]
    tickers_discussed: List[str]
    sources: List[str]
    thinking_process: Optional[str] = None
```

---

## 4. CHARACTER PERSONAS DESIGN

### 4.1 Base Persona (Abstract)

```python
# src/agents/characters/personas/base_persona.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from ..models import CharacterConfig, AnalysisCriteria, CharacterAnalysis

class BaseCharacterPersona(ABC):
    """Abstract base class for all character personas"""

    def __init__(self):
        self.config = self._build_config()

    @abstractmethod
    def _build_config(self) -> CharacterConfig:
        """Build character configuration"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the full system prompt for this character"""
        pass

    @abstractmethod
    def get_analysis_criteria(self) -> AnalysisCriteria:
        """Get analysis criteria specific to this character"""
        pass

    @abstractmethod
    def score_stock(self, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Score a stock based on character's criteria

        Returns dict with category scores like:
        {
            "fundamentals": 8.5,
            "moat": 7.0,
            "valuation": 6.5,
            "management": 8.0,
            "overall": 7.5
        }
        """
        pass

    @abstractmethod
    def format_analysis(self, analysis: CharacterAnalysis) -> str:
        """Format analysis in character's voice"""
        pass

    def get_tools(self) -> List[str]:
        """Get list of tools this character uses"""
        return self.config.available_tools

    def should_analyze(self, ticker: str, sector: str) -> bool:
        """Determine if this character should analyze this ticker

        Some characters have "circle of competence" restrictions
        """
        return True  # Override in subclasses if needed
```

### 4.2 Warren Buffett Persona

```python
# src/agents/characters/personas/warren_buffett.py

WARREN_BUFFETT_SYSTEM_PROMPT = '''
You are Warren Buffett, the legendary investor known as the "Oracle of Omaha."

## YOUR IDENTITY
You are Warren Buffett, Chairman and CEO of Berkshire Hathaway. You've built your
fortune through patient, long-term investing in wonderful businesses at fair prices.
You speak with folksy wisdom, using simple analogies that anyone can understand.

## INVESTMENT PHILOSOPHY - The Buffett Way
1. **Circle of Competence**: Only invest in businesses you understand deeply
   - STRONGLY PREFER: Consumer staples, banking, insurance, utilities, simple business models
   - APPROACH WITH CAUTION: Complex tech (except Apple), biotech, speculative industries

2. **Economic Moats**: Seek durable competitive advantages
   - Brand power (Coca-Cola)
   - Network effects
   - Switching costs
   - Cost advantages
   - Regulatory moats (insurance)

3. **Quality Management**: Look for honest, competent operators
   - Shareholder-friendly capital allocation
   - Skin in the game (insider ownership)
   - Clear communication
   - Rational compensation

4. **Financial Fortress**: Strong balance sheets
   - Low debt-to-equity (< 0.5 ideal)
   - Consistent profitability
   - High return on equity (> 15%)
   - Strong cash generation

5. **Intrinsic Value & Margin of Safety**
   - Always calculate what a business is worth
   - Only buy at significant discount (margin of safety > 25%)
   - "Price is what you pay, value is what you get"

6. **Long-term Perspective**
   - "Our favorite holding period is forever"
   - Think like a business owner, not a stock trader
   - Ignore short-term market fluctuations

## YOUR SPEAKING STYLE
- Use folksy, down-to-earth language
- Make complex concepts simple with everyday analogies
- Reference your past investments (See's Candies, Coca-Cola, GEICO, Apple)
- Quote your mentor Ben Graham and partner Charlie Munger
- Show genuine enthusiasm for great businesses
- Be humble about what you don't know
- Use humor and self-deprecation

## FAMOUS QUOTES YOU LIKE TO USE
- "Be fearful when others are greedy, and greedy when others are fearful"
- "It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price"
- "Our favorite holding period is forever"
- "Price is what you pay. Value is what you get"
- "Rule No. 1: Never lose money. Rule No. 2: Never forget rule No. 1"

## WHEN ANALYZING STOCKS
Always evaluate based on your criteria hierarchy:
1. Is it within your circle of competence?
2. Does it have a durable economic moat?
3. Is management honest and competent?
4. Is the balance sheet strong?
5. Is the price attractive relative to intrinsic value?

## CONFIDENCE LEVELS
- 90-100%: Exceptional business within circle, clear moat, strong management, significant margin of safety
- 70-89%: Good business with decent moat, some minor concerns
- 50-69%: Mixed signals, would need more research
- 30-49%: Outside expertise or meaningful concerns
- 10-29%: Poor business quality or significantly overvalued
'''

WARREN_BUFFETT_ANALYSIS_CRITERIA = AnalysisCriteria(
    required_metrics=[
        "roe", "roa", "roic",
        "debt_to_equity", "current_ratio",
        "gross_margin", "operating_margin", "net_margin",
        "revenue_growth_5y", "earnings_growth_5y",
        "free_cash_flow", "fcf_yield",
        "pe_ratio", "pb_ratio", "ev_ebitda",
        "dividend_yield", "payout_ratio",
        "insider_ownership"
    ],
    thresholds={
        "roe": {"excellent": 20, "good": 15, "acceptable": 10, "poor": 5},
        "debt_to_equity": {"excellent": 0.3, "good": 0.5, "acceptable": 1.0, "poor": 2.0},
        "gross_margin": {"excellent": 40, "good": 30, "acceptable": 20, "poor": 10},
        "net_margin": {"excellent": 20, "good": 15, "acceptable": 10, "poor": 5},
    },
    scoring_weights={
        "fundamentals": 0.25,
        "moat": 0.25,
        "management": 0.20,
        "consistency": 0.15,
        "valuation": 0.15
    },
    red_flags=[
        "high_debt_to_equity",
        "declining_roe",
        "negative_fcf",
        "frequent_equity_issuance",
        "aggressive_accounting"
    ],
    green_flags=[
        "consistent_roe_above_15",
        "growing_dividends",
        "share_buybacks",
        "low_capex_requirements",
        "pricing_power"
    ]
)
```

### 4.3 Ben Graham Persona (Deep Value)

```python
# Key differences from Buffett:
BEN_GRAHAM_ANALYSIS_CRITERIA = AnalysisCriteria(
    required_metrics=[
        "pb_ratio", "pe_ratio",
        "current_ratio", "quick_ratio",
        "debt_to_equity", "total_debt",
        "net_current_asset_value",  # NCAV calculation
        "book_value_per_share",
        "earnings_stability",  # 10-year history
        "dividend_history",  # Uninterrupted for 20 years
    ],
    thresholds={
        "pb_ratio": {"excellent": 0.67, "good": 1.0, "acceptable": 1.5, "poor": 2.0},
        "pe_ratio": {"excellent": 10, "good": 15, "acceptable": 20, "poor": 25},
        "current_ratio": {"excellent": 2.5, "good": 2.0, "acceptable": 1.5, "poor": 1.0},
        "margin_of_safety": {"required": 0.33},  # 33% minimum
    },
    # Focus: Quantitative, mechanical approach
    # Net-Net stocks (NCAV > Market Cap)
    # Defensive investor criteria
)
```

### 4.4 Cathie Wood Persona (Disruptive Innovation)

```python
# Completely different approach:
CATHIE_WOOD_ANALYSIS_CRITERIA = AnalysisCriteria(
    required_metrics=[
        "revenue_growth_yoy",
        "revenue_growth_3y_cagr",
        "rd_to_revenue",  # R&D spending
        "tam_estimate",  # Total Addressable Market
        "gross_margin_trend",
        "customer_acquisition_cost",
        "lifetime_value",
        "market_share_trend",
    ],
    thresholds={
        "revenue_growth_yoy": {"excellent": 50, "good": 30, "acceptable": 20, "poor": 10},
        "rd_to_revenue": {"excellent": 20, "good": 15, "acceptable": 10, "poor": 5},
        # No P/E threshold - expects losses in growth phase
    },
    scoring_weights={
        "innovation_potential": 0.30,
        "market_opportunity": 0.25,
        "execution": 0.20,
        "disruption_risk": 0.15,
        "management_vision": 0.10
    },
    # Focus: 5-year price targets, exponential growth
    # Comfortable with current losses if TAM is large
)
```

### 4.5 Michael Burry Persona (Contrarian)

```python
MICHAEL_BURRY_ANALYSIS_CRITERIA = AnalysisCriteria(
    required_metrics=[
        "short_interest",
        "short_ratio",
        "institutional_ownership_change",
        "insider_selling",
        "debt_to_assets",
        "interest_coverage",
        "sector_valuation_vs_history",
        "implied_volatility",
    ],
    thresholds={
        "short_interest": {"high": 20, "elevated": 10, "normal": 5},
        "overvaluation_signal": {"extreme": 2.0, "high": 1.5, "moderate": 1.2},
    },
    # Focus: Finding overvalued stocks, asymmetric bets
    # Looks for disconnects between fundamentals and price
)
```

---

## 5. TOOL INTEGRATION

### 5.1 Financial Data Tools

```python
# src/agents/tools/financial_tools.py

# Tools to be registered in tool_catalog.py

FINANCIAL_TOOLS = [
    {
        "name": "get_stock_fundamentals",
        "description": "Get fundamental metrics for a stock (ROE, margins, debt ratios, etc.)",
        "parameters": {
            "ticker": {"type": "string", "description": "Stock ticker symbol"},
            "metrics": {"type": "array", "items": {"type": "string"}, "description": "Specific metrics to fetch"}
        }
    },
    {
        "name": "get_financial_statements",
        "description": "Get income statement, balance sheet, or cash flow statement",
        "parameters": {
            "ticker": {"type": "string"},
            "statement_type": {"type": "string", "enum": ["income", "balance", "cashflow"]},
            "periods": {"type": "integer", "default": 5}
        }
    },
    {
        "name": "calculate_intrinsic_value",
        "description": "Calculate intrinsic value using DCF model",
        "parameters": {
            "ticker": {"type": "string"},
            "growth_rate": {"type": "number", "description": "Expected growth rate"},
            "discount_rate": {"type": "number", "description": "Required rate of return"},
            "terminal_multiple": {"type": "number", "default": 15}
        }
    },
    {
        "name": "get_stock_price_history",
        "description": "Get historical price data",
        "parameters": {
            "ticker": {"type": "string"},
            "period": {"type": "string", "enum": ["1M", "3M", "6M", "1Y", "5Y"]}
        }
    },
    {
        "name": "get_insider_trading",
        "description": "Get recent insider trading activity",
        "parameters": {
            "ticker": {"type": "string"},
            "days": {"type": "integer", "default": 90}
        }
    },
    {
        "name": "get_analyst_ratings",
        "description": "Get analyst ratings and price targets",
        "parameters": {
            "ticker": {"type": "string"}
        }
    }
]
```

### 5.2 Analysis Tools

```python
# src/agents/tools/analysis_tools.py

ANALYSIS_TOOLS = [
    {
        "name": "score_stock_buffett_style",
        "description": "Score a stock using Warren Buffett's value investing criteria",
        "parameters": {
            "ticker": {"type": "string"}
        }
    },
    {
        "name": "score_stock_graham_style",
        "description": "Score a stock using Ben Graham's deep value criteria (NCAV, P/B, etc.)",
        "parameters": {
            "ticker": {"type": "string"}
        }
    },
    {
        "name": "score_stock_growth_style",
        "description": "Score a stock using growth investing criteria (Cathie Wood style)",
        "parameters": {
            "ticker": {"type": "string"}
        }
    },
    {
        "name": "calculate_margin_of_safety",
        "description": "Calculate margin of safety between current price and intrinsic value",
        "parameters": {
            "ticker": {"type": "string"},
            "intrinsic_value": {"type": "number"}
        }
    }
]
```

---

## 6. API DESIGN

### 6.1 Endpoints

```python
# src/routers/character_agent_chat.py

# GET /api/v1/character-agents/list
# Response: List of available character agents with metadata

# POST /api/v1/character-agents/{agent_id}/chat
# Request:
{
    "message": "What do you think about AAPL?",
    "session_id": "optional-session-id",
    "model_name": "gpt-4.1-nano",
    "provider_type": "openai",
    "enable_thinking": false
}
# Response:
{
    "success": true,
    "session_id": "generated-or-provided",
    "character": {
        "id": "warren_buffett",
        "name": "Warren Buffett"
    },
    "content": "Let me share my thoughts on Apple...",
    "analyses": [
        {
            "ticker": "AAPL",
            "signal": "buy",
            "confidence": 78,
            "key_metrics": {...},
            "character_scores": {...}
        }
    ],
    "tickers_discussed": ["AAPL"],
    "sources": ["FMP API", "Company Filings"],
    "thinking_process": null,
    "metadata": {
        "turns": 3,
        "tools_used": ["get_stock_fundamentals", "calculate_intrinsic_value"],
        "response_time_ms": 2500
    }
}

# POST /api/v1/character-agents/{agent_id}/stream
# Same request, SSE response with chunks
```

### 6.2 Request/Response Models

```python
# src/routers/character_agent_chat.py

class CharacterChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    model_name: str = "gpt-4.1-nano"
    provider_type: ProviderType = ProviderType.OPENAI
    enable_thinking: bool = False

class CharacterAnalysisResult(BaseModel):
    ticker: str
    signal: str
    confidence: float
    reasoning: str
    key_metrics: Dict[str, Any]
    character_scores: Dict[str, float]
    intrinsic_value: Optional[float] = None
    margin_of_safety: Optional[float] = None

class CharacterChatResponse(BaseModel):
    success: bool
    session_id: str
    character: Dict[str, str]
    content: str
    analyses: List[CharacterAnalysisResult]
    tickers_discussed: List[str]
    sources: List[str]
    thinking_process: Optional[str] = None
    metadata: Dict[str, Any]
```

---

## 7. MEMORY INTEGRATION

### 7.1 Enhanced Memory Storage

```python
# When storing conversation turns, include character context:

await memory_manager.store_conversation_turn(
    session_id=session_id,
    user_id=user_id,
    query=user_message,
    response=character_response,
    metadata={
        "character_id": "warren_buffett",
        "character_name": "Warren Buffett",
        "tickers_analyzed": ["AAPL", "MSFT"],
        "signals": {
            "AAPL": {"signal": "buy", "confidence": 78},
            "MSFT": {"signal": "hold", "confidence": 65}
        },
        "conversation_type": "investment_analysis"
    },
    importance_score=importance
)
```

### 7.2 Character-Aware Context Retrieval

```python
# Modify memory retrieval to filter by character when needed:

async def get_character_context(
    session_id: str,
    user_id: str,
    character_id: str,
    current_query: str
) -> str:
    """Get conversation context relevant to this character"""

    # Get general context
    context, stats, refs = await memory_manager.get_relevant_context(...)

    # Filter for character-specific memories if needed
    # This allows the character to recall their previous analyses

    return context
```

---

## 8. IMPLEMENTATION PHASES

### Phase 1: Core Infrastructure (Day 1-2)
- [ ] Create directory structure
- [ ] Implement `CharacterConfig` and `AnalysisCriteria` models
- [ ] Create `BaseCharacterPersona` abstract class
- [ ] Implement `CharacterRegistry` singleton

### Phase 2: Character Personas (Day 2-3)
- [ ] Implement Warren Buffett persona (reference existing)
- [ ] Implement Ben Graham persona
- [ ] Implement Cathie Wood persona
- [ ] Implement Michael Burry persona
- [ ] Implement 2-3 additional personas

### Phase 3: Tools Integration (Day 3-4)
- [ ] Create financial data tools (wrap FMP API)
- [ ] Create analysis scoring tools
- [ ] Register tools in `tool_catalog.py`
- [ ] Test tool execution

### Phase 4: Handler & API (Day 4-5)
- [ ] Implement `CharacterAgentHandler`
- [ ] Create API router with endpoints
- [ ] Integrate with Agent Loop
- [ ] Add memory storage with character metadata

### Phase 5: Testing & Refinement (Day 5-6)
- [ ] Test multi-turn conversations
- [ ] Verify character voice consistency
- [ ] Test memory continuity
- [ ] Performance optimization

---

## 9. KEY DESIGN DECISIONS

### 9.1 Character Voice Consistency
- System prompt defines personality, philosophy, speaking style
- Analysis criteria determines WHAT metrics to focus on
- Scoring functions ensure consistent evaluation logic
- Response formatting maintains character voice

### 9.2 Separation of Concerns
- **Persona**: Defines WHO the character is
- **Criteria**: Defines WHAT they look for
- **Tools**: Defines HOW they get data
- **Handler**: Orchestrates the flow

### 9.3 Extensibility
- New characters: Just create new persona file
- New metrics: Add to analysis criteria
- New tools: Register in tool catalog
- All changes localized, minimal impact on other code

### 9.4 Memory Continuity
- Character ID stored with each conversation turn
- Previous analyses retrievable across sessions
- Character can recall their past recommendations

---

## 10. EXAMPLE CONVERSATION FLOW

```
Turn 1:
User: "What do you think about AAPL?"
Warren Buffett: [Analyzes AAPL with his criteria]
"Let me look at Apple through my lens of value investing...
Apple is one of those rare companies that sits firmly in my circle of competence.
[Detailed analysis with folksy language]
Signal: BUY with 78% confidence"

Memory Stored: {
    "character_id": "warren_buffett",
    "tickers": ["AAPL"],
    "signal": "buy",
    "confidence": 78
}

Turn 2:
User: "How does it compare to MSFT?"
Warren Buffett: [Recalls AAPL analysis from memory, analyzes MSFT]
"Now, comparing Microsoft to our earlier discussion about Apple...
Both are wonderful businesses, but let me tell you why I see them differently..."

Memory Stored: {
    "character_id": "warren_buffett",
    "tickers": ["MSFT"],
    "comparison": ["AAPL", "MSFT"],
    "previous_turn_referenced": true
}

Turn 3:
User: "Which one should I buy?"
Warren Buffett: [Uses both analyses from memory]
"Based on our analysis of both companies...
If I were deploying capital today, I'd lean towards..."
```

---

## 11. FILES TO CREATE

1. `src/agents/characters/__init__.py`
2. `src/agents/characters/models.py`
3. `src/agents/characters/registry.py`
4. `src/agents/characters/analysis_criteria.py`
5. `src/agents/characters/personas/__init__.py`
6. `src/agents/characters/personas/base_persona.py`
7. `src/agents/characters/personas/warren_buffett.py`
8. `src/agents/characters/personas/ben_graham.py`
9. `src/agents/characters/personas/cathie_wood.py`
10. `src/agents/characters/personas/michael_burry.py`
11. `src/agents/tools/financial_tools.py`
12. `src/agents/tools/analysis_tools.py`
13. `src/handlers/character_agent_handler.py`
14. `src/routers/character_agent_chat.py`

---

## 12. NEXT STEPS

1. **Review this plan** - Confirm architecture decisions
2. **Start Phase 1** - Core infrastructure
3. **Iterate** - Test each character, refine prompts
4. **Document** - API documentation, usage examples
