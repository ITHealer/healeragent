# HealerAgent - Agent Loop Architecture

## Overview

HealerAgent implements a **ChatGPT-style Agent Loop** architecture with complexity-based execution strategies. The system adapts its behavior based on query complexity, using an iterative tool-calling loop for complex queries.

## Architecture Diagram

```
                                    +------------------+
                                    |   API Request    |
                                    | POST /chat/v3    |
                                    +--------+---------+
                                             |
                                             v
+-----------------------------------------------------------------------------------+
|                              PHASE 1: CLASSIFICATION                               |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|    +------------------+      +-------------------+      +--------------------+     |
|    | chat_assistant.py| ---> | UnifiedClassifier | ---> | Symbol Resolution  |     |
|    +------------------+      +-------------------+      +--------------------+     |
|                                       |                                            |
|                                       v                                            |
|                              Classification Result:                                |
|                              - type: stock_specific/comparison/screener            |
|                              - symbols: [NVDA, AMD, ...]                           |
|                              - categories: [technical, price, ...]                 |
|                                                                                    |
+-----------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------+
|                              PHASE 2: ROUTING (LLM-based)                          |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|    +------------------+      +-------------------+      +--------------------+     |
|    |  LLMToolRouter   | ---> |   LLM Decision    | ---> |  RouterDecision    |     |
|    +------------------+      +-------------------+      +--------------------+     |
|                                                                                    |
|    Router receives ALL tool summaries and selects:                                |
|    - selected_tools: [getTechnicalIndicators, getSupportResistance, ...]          |
|    - complexity: SIMPLE | MEDIUM | COMPLEX                                        |
|    - execution_strategy: DIRECT | ITERATIVE | PARALLEL                            |
|    - suggested_max_turns: 2 | 4 | 6                                               |
|                                                                                    |
+-----------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------+
|                              PHASE 3: EXECUTION (UnifiedAgent)                     |
+-----------------------------------------------------------------------------------+
|                                                                                    |
|    Based on complexity, route to appropriate strategy:                            |
|                                                                                    |
|    SIMPLE (1-2 tools)          MEDIUM (3-5 tools)         COMPLEX (6+ tools)      |
|    +------------------+        +------------------+       +------------------+     |
|    | _stream_simple() |        | _stream_iterative()|     | _stream_complex()|     |
|    +------------------+        +------------------+       +------------------+     |
|           |                           |                          |                 |
|           v                           v                          v                 |
|    Execute ALL tools          AGENT LOOP (ChatGPT-style)   Same as ITERATIVE      |
|    in parallel once           See detailed flow below      but max_turns=6         |
|           |                           |                          |                 |
|           v                           v                          v                 |
|    +------------------+        +------------------+       +------------------+     |
|    | Synthesize       |        | Synthesize       |       | Synthesize       |     |
|    | Response         |        | Response         |       | Response         |     |
|    +------------------+        +------------------+       +------------------+     |
|                                                                                    |
+-----------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------+
|                              PHASE 4: RESPONSE                                     |
+-----------------------------------------------------------------------------------+
|    Stream response chunks to client via SSE                                       |
|    Final synthesis uses Skill-based prompts for domain expertise                  |
+-----------------------------------------------------------------------------------+
```

## Agent Loop Flow (MEDIUM/COMPLEX Strategy)

This is the **ChatGPT-style iterative agent loop** that executes multiple turns of tool calling:

```
+------------------------------------------+
|             AGENT LOOP START              |
|     (MEDIUM or COMPLEX complexity)        |
+------------------------------------------+
                    |
                    v
+------------------------------------------+
|         Build Initial Messages            |
|  - System prompt (from Skill)             |
|  - Pre-selected tools list                |
|  - User query                             |
+------------------------------------------+
                    |
                    v
        +------------------------+
        |      TURN N START      |
        |   (N = 1 to max_turns) |
        +------------------------+
                    |
                    v
+------------------------------------------+
|        Call LLM with Tools                |
|  - Send messages + available tools        |
|  - LLM decides which tools to call        |
+------------------------------------------+
                    |
                    v
          /------------------\
         /  Tool calls in     \
        <   LLM response?      >
         \                    /
          \------------------/
                |         |
               YES        NO
                |          |
                v          v
    +------------------+  +------------------+
    | Execute Tools     |  | Extract Final    |
    | (parallel)        |  | Response         |
    +------------------+  +------------------+
                |                   |
                v                   v
    +------------------+  +------------------+
    | Add results to   |  | AGENT LOOP END   |
    | message history  |  | Return response  |
    +------------------+  +------------------+
                |
                v
          /------------------\
         /  N < max_turns?    \
        <                      >
         \                    /
          \------------------/
                |         |
               YES        NO
                |          |
                v          v
        +------------+  +------------------+
        | Continue   |  | Force final      |
        | to Turn N+1|  | response         |
        +------------+  +------------------+
```

## Example: Discovery Query (4-Turn Agent Loop)

**Query:** "Find stocks with RSI < 30 on US exchanges. For found stocks, analyze patterns and fibonacci."

```
TURN 1:
  LLM Decision: Call stockScreener for NASDAQ and NYSE
  Tool Calls:
    - stockScreener(exchange=NASDAQ, limit=10)
    - stockScreener(exchange=NYSE, limit=10)
  Results: Found 20 stocks (NVDA, AAPL, GOOGL, MSFT, ...)

TURN 2:
  LLM Decision: Check RSI for all found stocks
  Tool Calls:
    - getTechnicalIndicators(symbol=NVDA, indicators=[RSI])
    - getTechnicalIndicators(symbol=AAPL, indicators=[RSI])
    - getTechnicalIndicators(symbol=GOOGL, indicators=[RSI])
    - ... (9 more)
  Results: RSI values for all stocks (none < 30, AAPL closest at 38)

TURN 3:
  LLM Decision: Analyze AAPL (closest to RSI < 30)
  Tool Calls:
    - detectChartPatterns(symbol=AAPL)
    - getSupportResistance(symbol=AAPL)
  Results: Patterns and S/R levels for AAPL

TURN 4:
  LLM Decision: No more tools needed
  Response: Synthesized analysis explaining no stocks have RSI < 30,
            but AAPL is closest with detailed pattern analysis
```

## Complexity Levels

| Complexity | Tools | Max Turns | Strategy | Use Case |
|------------|-------|-----------|----------|----------|
| SIMPLE | 1-2 | 2 | DIRECT | Price checks, simple lookups |
| MEDIUM | 3-5 | 4 | ITERATIVE | Analysis queries, comparisons |
| COMPLEX | 6+ | 6 | PARALLEL | Multi-step research, screening |

## Key Components

### 1. UnifiedClassifier (`src/agents/classification/`)
- Classifies query type (stock_specific, comparison, screener, etc.)
- Resolves symbols from user query
- Determines required analysis categories

### 2. LLMToolRouter (`src/agents/router/llm_tool_router.py`)
- Receives ALL available tool summaries
- LLM selects relevant tools for the query
- Determines complexity and execution strategy
- Eliminates "category blindness" issue

### 3. UnifiedAgent (`src/agents/unified/unified_agent.py`)
- Main agent that executes the loop
- Routes to appropriate strategy based on complexity
- Manages message history and tool results
- Streams response to client

### 4. Skill System (`src/agents/skills/`)
- Domain-specific prompts (StockSkill, CryptoSkill, MixedSkill)
- Analysis frameworks and output formatting
- Response quality guidelines

### 5. Tool Catalog (`src/agents/tools/`)
- Registry of all available tools
- Tool schemas with capabilities/limitations
- Execution wrappers with caching

## Verification: Agent Loop is Working

From the logs, we can confirm the agent loop is functioning correctly:

```
Query: "Find stocks with RSI < 30..."
Classification: type=screener, categories=[discovery, technical]
Router: tools=[stockScreener, getTechnicalIndicators, detectChartPatterns,
               getSupportResistance, webSearch]
       complexity=complex, strategy=iterative, max_turns=6

Turn 1: stockScreener x2 (NASDAQ + NYSE) -> 20 stocks
Turn 2: getTechnicalIndicators x9 -> RSI values
Turn 3: detectChartPatterns + getSupportResistance for AAPL
Turn 4: Final synthesis
```

The agent successfully:
1. Screened stocks first
2. Checked RSI for found stocks
3. Analyzed patterns for relevant stock
4. Generated comprehensive response

## Known Limitations

1. **Vietnamese stocks**: FMP API doesn't support VN stocks (VNM, HPG, etc.)
2. **Symbol format**: Now supports BRK-A, BRK-B style symbols (fixed)
3. **Tool errors**: Now properly logged with appropriate emoji (fixed)
