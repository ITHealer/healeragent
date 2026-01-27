# Chat V3 Streaming Architecture

## Overview

This document describes the real streaming architecture for `/api/v2/chat-assistant/chat/v3` endpoint, following ChatGPT/Claude standard patterns.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Client Request                                          │
│                     POST /api/v2/chat-assistant/chat/v3                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 1: INTENT CLASSIFICATION                            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  IntentClassifier.classify()                                             │   │
│  │  - Extract symbols, query type, complexity                               │   │
│  │  - Normalize symbols (GOOGLE → GOOGL)                                    │   │
│  │  - Determine if tools required                                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                             │
│                    SSE: emit_classifying(), emit_classified()                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: AGENT EXECUTION LOOP                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  UnifiedAgent.run_stream_with_all_tools()                                │   │
│  │                                                                          │   │
│  │  FOR each turn (max 6):                                                  │   │
│  │    ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │    │  NON-STREAMING LLM Call (_call_llm_with_tools)                  │  │   │
│  │    │  - Send messages + tools to LLM                                 │  │   │
│  │    │  - Parse tool_calls from response                               │  │   │
│  │    │  - Works with: OpenAI, Gemini, Claude                           │  │   │
│  │    └─────────────────────────────────────────────────────────────────┘  │   │
│  │                              │                                           │   │
│  │                              ▼                                           │   │
│  │    ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │    │  IF tool_calls:                                                 │  │   │
│  │    │    - Execute tools in parallel                                  │  │   │
│  │    │    - Append results to messages                                 │  │   │
│  │    │    - SSE: emit_tool_calls(), emit_tool_results()                │  │   │
│  │    │    → Continue to next turn                                      │  │   │
│  │    │                                                                 │  │   │
│  │    │  ELSE (no tool_calls):                                          │  │   │
│  │    │    → Go to PHASE 3: Final Synthesis                             │  │   │
│  │    └─────────────────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│              PHASE 3: REAL STREAMING SYNTHESIS (ChatGPT/Claude Style)            │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  llm_provider.stream_response()                                          │   │
│  │  - Native streaming API call (NOT simulated)                             │   │
│  │  - Real chunks from LLM provider                                         │   │
│  │  - Proper streaming UX                                                   │   │
│  │                                                                          │   │
│  │  Config:                                                                 │   │
│  │  - max_tokens: STREAMING_MAX_TOKENS (32000)                              │   │
│  │  - temperature: STREAMING_TEMPERATURE (0.3)                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                              ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  FOR each chunk from stream:                                             │   │
│  │    yield {"type": "content", "content": chunk, "is_final": false}        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                    SSE: emit_content() for each chunk                            │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 4: POST-PROCESSING                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  - Append web sources (if any)                                           │   │
│  │  - Save conversation to database                                         │   │
│  │  - Update memory (LEARN phase)                                           │   │
│  │  - Create/update summary                                                 │   │
│  │  - Resolve charts                                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                              │                                                   │
│                    SSE: emit_done(), [DONE]                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Changes from Previous Implementation

### Before (Fake Streaming)

```python
# OLD CODE - REMOVED
if assistant_content and len(assistant_content) > 100:
    # Simulate streaming by yielding content in chunks
    paragraphs = re.split(r'(\n{2,})', assistant_content)
    for chunk in chunks:
        yield {"type": "content", "content": chunk}
```

**Problems:**
- Not real streaming - chunks from cached content
- Poor UX - artificial delays
- Inconsistent with ChatGPT/Claude behavior

### After (Real Streaming)

```python
# NEW CODE - Approach A
async for chunk in self.llm_provider.stream_response(
    model_name=effective_model,
    messages=messages,
    provider_type=effective_provider,
    api_key=self.api_key,
    max_tokens=STREAMING_MAX_TOKENS,
    temperature=STREAMING_TEMPERATURE,
):
    yield {"type": "content", "content": chunk, "is_final": False}
```

**Benefits:**
- Real streaming from LLM API
- Consistent UX across providers (OpenAI, Gemini, Claude)
- Proper real-time response
- Clean, maintainable code

## Provider-Specific Behavior

### OpenAI
- Native streaming with `stream=True`
- Direct chunk yield
- Full tool calling support in both modes

### Gemini
- Native streaming with `stream=True` in `generate_content_async()`
- Retry with exponential backoff for rate limits
- Safety filter handling
- Tool results converted via `_convert_messages()`

### Ollama
- Native streaming support
- Local model execution

## Configuration

```python
# src/agents/unified/unified_agent.py

# Max tokens for streaming responses
STREAMING_MAX_TOKENS = 32000

# Temperature for final synthesis
STREAMING_TEMPERATURE = 0.3
```

## SSE Event Flow

```
1. session_start          → Client connected
2. classifying            → Starting classification
3. classified             → Classification complete
4. turn_start             → Agent turn N
5. tool_calls             → Tools being called
6. tool_results           → Tool execution results
7. [loop back to 4 if more tools needed]
8. content (multiple)     → Streaming response chunks
9. sources (optional)     → Web search sources
10. done                  → Processing complete
11. [DONE]                → Stream end marker
```

## Error Handling

- Network errors: Retry with exponential backoff (up to 3 attempts)
- Rate limits: Wait and retry
- Safety blocks: Immediate fail (no retry)
- Empty responses: Inject retry prompt

## Files Modified

1. `src/agents/unified/unified_agent.py`
   - Added `STREAMING_MAX_TOKENS`, `STREAMING_TEMPERATURE` constants
   - Replaced fake streaming with real streaming API calls
   - Updated all streaming calls to use constants

2. Flow affected:
   - `run_stream_with_all_tools()` - main agent loop
   - `_execute_iterative_stream()` - iterative strategy
   - `_execute_parallel_stream()` - parallel strategy
