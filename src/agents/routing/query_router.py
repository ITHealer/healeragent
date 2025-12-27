"""
Query Router - Intelligent Execution Mode Selection

Implements the Orchestration Agent pattern used by Claude, ChatGPT, and Manus AI.
Routes queries to optimal execution path based on complexity analysis.

Execution Modes:
- SIMPLE: 1 LLM call, 0-1 tools, ~1-3s (greetings, definitions, single lookups)
- PARALLEL: 2-3 LLM calls, parallel tools, ~3-10s (known symbols, multi-tool analysis)
- AGENTIC: 3-15+ LLM calls, adaptive, ~10-60s (discovery, conditional, research)

Design Principles:
- Deterministic fast paths for obvious cases (no LLM needed)
- LLM classification only for ambiguous queries
- Fail-safe to PARALLEL mode (most common case)
"""

import re
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType


# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class ExecutionMode(Enum):
    """Execution mode for query processing"""
    SIMPLE = "simple"      # Direct LLM, no/minimal tools
    PARALLEL = "parallel"  # Upfront planning + parallel execution
    AGENTIC = "agentic"    # Adaptive loop with re-planning


@dataclass
class RoutingResult:
    """Result of query routing decision"""

    mode: ExecutionMode
    confidence: float  # 0.0 - 1.0

    # Extracted information
    is_conversational: bool = False
    symbols: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    detected_language: str = "en"

    # Routing metadata
    reasoning: str = ""
    route_method: str = "deterministic"  # deterministic | llm | fallback
    routing_time_ms: int = 0

    # For SIMPLE mode
    simple_response_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.value,
            "confidence": self.confidence,
            "is_conversational": self.is_conversational,
            "symbols": self.symbols,
            "categories": self.categories,
            "detected_language": self.detected_language,
            "reasoning": self.reasoning,
            "route_method": self.route_method,
            "routing_time_ms": self.routing_time_ms
        }


# ============================================================================
# PATTERNS FOR DETERMINISTIC ROUTING
# ============================================================================

class RoutingPatterns:
    """Regex patterns for fast deterministic routing"""

    # Greeting patterns (multilingual)
    GREETINGS = [
        # English
        r"^(hi|hello|hey|good\s*(morning|afternoon|evening)|howdy)[\s\!\?\.\,]*$",
        r"^what'?s?\s+up[\s\!\?]*$",
        # Vietnamese
        r"^(xin\s*chào|chào|chào\s+bạn|hello|hi)[\s\!\?\.\,]*$",
        r"^(ê|ơi|hey)[\s\!\?\.\,]*$",
        # Chinese
        r"^(你好|您好|嗨|哈喽)[\s\!\?\.\,]*$",
    ]

    # Thank you patterns
    THANKS = [
        r"^(thanks?|thank\s*you|thx|ty)[\s\!\.\,]*$",
        r"^(cảm\s*ơn|cám\s*ơn|thanks?|ok\s+cảm\s*ơn)[\s\!\.\,]*$",
        r"^(谢谢|感谢)[\s\!\.\,]*$",
    ]

    # Goodbye patterns
    GOODBYE = [
        r"^(bye|goodbye|see\s*you|later|cya)[\s\!\.\,]*$",
        r"^(tạm\s*biệt|bye|bai)[\s\!\.\,]*$",
        r"^(再见|拜拜)[\s\!\.\,]*$",
    ]

    # Acknowledgment patterns
    ACKNOWLEDGMENT = [
        r"^(ok|okay|got\s*it|understood|sure|alright|yep|yes|no)[\s\!\.\,]*$",
        r"^(được|ok|ừ|ờ|rồi|hiểu\s*rồi)[\s\!\.\,]*$",
    ]

    # Stock symbol pattern (uppercase 1-5 letters, or with .XX suffix)
    STOCK_SYMBOL = r'\b([A-Z]{1,5})(?:\.[A-Z]{1,2})?\b'

    # Crypto symbol pattern (with USD suffix common)
    CRYPTO_SYMBOL = r'\b(BTC|ETH|XRP|SOL|ADA|DOGE|DOT|MATIC|LINK|UNI|AVAX|ATOM)(?:USD)?\b'

    # Discovery/screening keywords
    DISCOVERY_KEYWORDS = [
        r'\b(top|best|find|search|screen|filter|discover)\b',
        r'\b(tìm|lọc|sàng\s*lọc|top|tốt\s*nhất)\b',
        r'\b(stocks?\s+with|cổ\s+phiếu\s+có)\b',
    ]

    # Conditional logic keywords
    CONDITIONAL_KEYWORDS = [
        r'\b(if|when|unless|otherwise|then)\b',
        r'\b(nếu|khi|trừ\s*khi|thì)\b',
    ]

    # Research/complex task keywords
    RESEARCH_KEYWORDS = [
        r'\b(research|analyze\s+trend|deep\s+dive|comprehensive)\b',
        r'\b(nghiên\s*cứu|phân\s*tích\s+xu\s*hướng|toàn\s*diện)\b',
    ]

    # Definition questions (SIMPLE mode)
    DEFINITION_KEYWORDS = [
        r'\b(what\s+is|what\'s|define|explain|meaning\s+of)\b',
        r'\b(là\s+gì|nghĩa\s+là|giải\s+thích)\b',
    ]

    # Common words to exclude from symbol detection
    SYMBOL_EXCLUSIONS = {
        'I', 'A', 'THE', 'AND', 'OR', 'NOT', 'IS', 'ARE', 'WAS', 'BE',
        'TO', 'OF', 'IN', 'FOR', 'ON', 'WITH', 'AS', 'AT', 'BY', 'AN',
        'IT', 'IF', 'SO', 'UP', 'DO', 'NO', 'HE', 'WE', 'MY', 'OK',
        'API', 'LLM', 'AI', 'ML', 'USD', 'EUR', 'JPY', 'ETF', 'IPO',
        'CEO', 'CFO', 'CTO', 'COO', 'PE', 'EPS', 'ROE', 'ROI', 'RSI',
        'US', 'UK', 'EU', 'VN', 'CN', 'JP', 'KR', 'SG', 'HK',
        'TOP', 'GDP', 'YTD', 'QTD', 'MTD', 'ATH', 'ATL',
    }


# ============================================================================
# QUERY ROUTER
# ============================================================================

class QueryRouter(LoggerMixin):
    """
    Intelligent Query Router for Execution Mode Selection

    Routes queries to optimal execution path:
    - SIMPLE: Bypass planning, direct LLM response
    - PARALLEL: Standard planning + parallel tool execution
    - AGENTIC: Adaptive execution with potential re-planning

    Usage:
        router = QueryRouter()
        result = await router.route(
            query="giá AAPL hôm nay",
            context={"user_id": "...", "session_id": "..."}
        )

        if result.mode == ExecutionMode.SIMPLE:
            # Direct LLM response
        elif result.mode == ExecutionMode.PARALLEL:
            # Standard planning flow
        else:
            # Agentic execution
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        provider_type: ProviderType = ProviderType.OPENAI,
        enable_llm_routing: bool = True
    ):
        """
        Initialize Query Router

        Args:
            model_name: Model for LLM-based routing (when deterministic fails)
            provider_type: LLM provider
            enable_llm_routing: If False, only use deterministic routing
        """
        super().__init__()

        self.model_name = model_name
        self.provider_type = provider_type
        self.enable_llm_routing = enable_llm_routing

        self.llm_provider = LLMGeneratorProvider()
        self.api_key = ModelProviderFactory._get_api_key(provider_type)

        # Compile patterns for performance
        self._compile_patterns()

        # Routing statistics
        self._stats = {
            "total_routes": 0,
            "deterministic_routes": 0,
            "llm_routes": 0,
            "fallback_routes": 0,
            "mode_counts": {m.value: 0 for m in ExecutionMode}
        }

        self.logger.info(
            f"[ROUTER:INIT] QueryRouter initialized with model={model_name}, "
            f"llm_routing={'enabled' if enable_llm_routing else 'disabled'}"
        )

    def _compile_patterns(self):
        """Compile regex patterns for performance"""
        self._greeting_patterns = [
            re.compile(p, re.IGNORECASE) for p in RoutingPatterns.GREETINGS
        ]
        self._thanks_patterns = [
            re.compile(p, re.IGNORECASE) for p in RoutingPatterns.THANKS
        ]
        self._goodbye_patterns = [
            re.compile(p, re.IGNORECASE) for p in RoutingPatterns.GOODBYE
        ]
        self._ack_patterns = [
            re.compile(p, re.IGNORECASE) for p in RoutingPatterns.ACKNOWLEDGMENT
        ]
        self._discovery_patterns = [
            re.compile(p, re.IGNORECASE) for p in RoutingPatterns.DISCOVERY_KEYWORDS
        ]
        self._conditional_patterns = [
            re.compile(p, re.IGNORECASE) for p in RoutingPatterns.CONDITIONAL_KEYWORDS
        ]
        self._research_patterns = [
            re.compile(p, re.IGNORECASE) for p in RoutingPatterns.RESEARCH_KEYWORDS
        ]
        self._definition_patterns = [
            re.compile(p, re.IGNORECASE) for p in RoutingPatterns.DEFINITION_KEYWORDS
        ]
        self._symbol_pattern = re.compile(RoutingPatterns.STOCK_SYMBOL)
        self._crypto_pattern = re.compile(RoutingPatterns.CRYPTO_SYMBOL, re.IGNORECASE)

    # ========================================================================
    # MAIN ROUTING METHOD
    # ========================================================================

    async def route(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        recent_symbols: Optional[List[str]] = None
    ) -> RoutingResult:
        """
        Route query to optimal execution mode

        Args:
            query: User query text
            context: Optional context (user_id, session_id, etc.)
            recent_symbols: Symbols from recent conversation (for reference resolution)

        Returns:
            RoutingResult with mode and extracted information
        """
        start_time = datetime.now()
        context = context or {}
        recent_symbols = recent_symbols or []

        self.logger.info(f"[ROUTER] ════════════════════════════════════════")
        self.logger.info(f"[ROUTER] Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")

        try:
            # ================================================================
            # PHASE 1: Deterministic Fast Paths
            # ================================================================
            result = self._try_deterministic_routing(query, recent_symbols)

            if result is not None:
                result.routing_time_ms = self._elapsed_ms(start_time)
                self._update_stats(result, "deterministic")
                self._log_routing_result(result)
                return result

            # ================================================================
            # PHASE 2: LLM-based Routing (for ambiguous queries)
            # ================================================================
            if self.enable_llm_routing:
                result = await self._llm_routing(query, context, recent_symbols)

                if result is not None:
                    result.routing_time_ms = self._elapsed_ms(start_time)
                    self._update_stats(result, "llm")
                    self._log_routing_result(result)
                    return result

            # ================================================================
            # PHASE 3: Fallback to PARALLEL (safest default)
            # ================================================================
            symbols = self._extract_symbols(query)
            result = RoutingResult(
                mode=ExecutionMode.PARALLEL,
                confidence=0.6,
                symbols=symbols,
                reasoning="Fallback to PARALLEL mode for ambiguous query",
                route_method="fallback",
                detected_language=self._detect_language(query)
            )

            result.routing_time_ms = self._elapsed_ms(start_time)
            self._update_stats(result, "fallback")
            self._log_routing_result(result)
            return result

        except Exception as e:
            self.logger.error(f"[ROUTER] Error during routing: {e}", exc_info=True)

            # Emergency fallback
            return RoutingResult(
                mode=ExecutionMode.PARALLEL,
                confidence=0.5,
                reasoning=f"Error fallback: {str(e)}",
                route_method="error_fallback",
                routing_time_ms=self._elapsed_ms(start_time)
            )

    # ========================================================================
    # DETERMINISTIC ROUTING
    # ========================================================================

    def _try_deterministic_routing(
        self,
        query: str,
        recent_symbols: List[str]
    ) -> Optional[RoutingResult]:
        """
        Try deterministic routing based on patterns

        Returns None if no deterministic route found
        """
        query_stripped = query.strip()
        query_lower = query_stripped.lower()

        # --------------------------------------------------------------------
        # SIMPLE MODE: Greetings
        # --------------------------------------------------------------------
        if self._matches_any(query_stripped, self._greeting_patterns):
            return RoutingResult(
                mode=ExecutionMode.SIMPLE,
                confidence=0.99,
                is_conversational=True,
                reasoning="Greeting detected - direct response",
                route_method="deterministic",
                detected_language=self._detect_language(query),
                simple_response_hint="greeting"
            )

        # --------------------------------------------------------------------
        # SIMPLE MODE: Thanks
        # --------------------------------------------------------------------
        if self._matches_any(query_stripped, self._thanks_patterns):
            return RoutingResult(
                mode=ExecutionMode.SIMPLE,
                confidence=0.99,
                is_conversational=True,
                reasoning="Thank you message - direct response",
                route_method="deterministic",
                detected_language=self._detect_language(query),
                simple_response_hint="thanks"
            )

        # --------------------------------------------------------------------
        # SIMPLE MODE: Goodbye
        # --------------------------------------------------------------------
        if self._matches_any(query_stripped, self._goodbye_patterns):
            return RoutingResult(
                mode=ExecutionMode.SIMPLE,
                confidence=0.99,
                is_conversational=True,
                reasoning="Goodbye message - direct response",
                route_method="deterministic",
                detected_language=self._detect_language(query),
                simple_response_hint="goodbye"
            )

        # --------------------------------------------------------------------
        # SIMPLE MODE: Acknowledgment
        # --------------------------------------------------------------------
        if self._matches_any(query_stripped, self._ack_patterns):
            return RoutingResult(
                mode=ExecutionMode.SIMPLE,
                confidence=0.95,
                is_conversational=True,
                reasoning="Acknowledgment - direct response",
                route_method="deterministic",
                detected_language=self._detect_language(query),
                simple_response_hint="acknowledgment"
            )

        # --------------------------------------------------------------------
        # SIMPLE MODE: Definition/Explanation questions
        # --------------------------------------------------------------------
        if self._matches_any(query, self._definition_patterns):
            # Check if it's purely definitional (no symbols)
            symbols = self._extract_symbols(query)
            if not symbols:
                return RoutingResult(
                    mode=ExecutionMode.SIMPLE,
                    confidence=0.85,
                    is_conversational=False,
                    reasoning="Definition question - direct knowledge response",
                    route_method="deterministic",
                    detected_language=self._detect_language(query),
                    simple_response_hint="definition"
                )

        # --------------------------------------------------------------------
        # AGENTIC MODE: Discovery/Screening
        # --------------------------------------------------------------------
        if self._matches_any(query, self._discovery_patterns):
            symbols = self._extract_symbols(query)
            return RoutingResult(
                mode=ExecutionMode.AGENTIC,
                confidence=0.90,
                symbols=symbols,
                categories=["discovery"],
                reasoning="Discovery/screening task - requires adaptive execution",
                route_method="deterministic",
                detected_language=self._detect_language(query)
            )

        # --------------------------------------------------------------------
        # AGENTIC MODE: Conditional Logic
        # --------------------------------------------------------------------
        if self._matches_any(query, self._conditional_patterns):
            symbols = self._extract_symbols(query)
            return RoutingResult(
                mode=ExecutionMode.AGENTIC,
                confidence=0.88,
                symbols=symbols,
                reasoning="Conditional logic detected - requires adaptive execution",
                route_method="deterministic",
                detected_language=self._detect_language(query)
            )

        # --------------------------------------------------------------------
        # AGENTIC MODE: Research Tasks
        # --------------------------------------------------------------------
        if self._matches_any(query, self._research_patterns):
            return RoutingResult(
                mode=ExecutionMode.AGENTIC,
                confidence=0.85,
                reasoning="Research task detected - requires comprehensive analysis",
                route_method="deterministic",
                detected_language=self._detect_language(query)
            )

        # --------------------------------------------------------------------
        # PARALLEL MODE: Known Symbols Detected
        # --------------------------------------------------------------------
        symbols = self._extract_symbols(query)

        if symbols:
            # Multiple symbols = definitely parallel
            if len(symbols) >= 2:
                return RoutingResult(
                    mode=ExecutionMode.PARALLEL,
                    confidence=0.92,
                    symbols=symbols,
                    reasoning=f"Multiple symbols detected ({', '.join(symbols)}) - parallel execution",
                    route_method="deterministic",
                    detected_language=self._detect_language(query)
                )

            # Single symbol + short query = might be simple lookup
            if len(symbols) == 1 and len(query.split()) <= 5:
                # Check if it's a simple price query
                price_keywords = ['giá', 'price', 'quote', 'bao nhiêu', 'how much']
                if any(kw in query_lower for kw in price_keywords):
                    return RoutingResult(
                        mode=ExecutionMode.PARALLEL,
                        confidence=0.88,
                        symbols=symbols,
                        categories=["price"],
                        reasoning=f"Simple price lookup for {symbols[0]}",
                        route_method="deterministic",
                        detected_language=self._detect_language(query)
                    )

        # No deterministic route found
        return None

    # ========================================================================
    # LLM-BASED ROUTING
    # ========================================================================

    async def _llm_routing(
        self,
        query: str,
        context: Dict[str, Any],
        recent_symbols: List[str]
    ) -> Optional[RoutingResult]:
        """
        Use LLM to classify ambiguous queries
        """
        self.logger.info("[ROUTER] Using LLM for ambiguous query classification")

        prompt = self._build_routing_prompt(query, recent_symbols)

        try:
            response = await self.llm_provider.generate_response(
                model_name=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a query routing classifier. Return ONLY valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                provider_type=self.provider_type,
                api_key=self.api_key
            )

            content = response.get("content", "") if isinstance(response, dict) else str(response)

            # Parse JSON response
            result_dict = self._parse_llm_response(content)

            if result_dict:
                mode_str = result_dict.get("mode", "parallel").lower()
                mode = self._parse_mode(mode_str)

                return RoutingResult(
                    mode=mode,
                    confidence=result_dict.get("confidence", 0.75),
                    is_conversational=result_dict.get("is_conversational", False),
                    symbols=result_dict.get("symbols", []),
                    categories=result_dict.get("categories", []),
                    detected_language=result_dict.get("language", "en"),
                    reasoning=result_dict.get("reasoning", "LLM classification"),
                    route_method="llm"
                )

            return None

        except Exception as e:
            self.logger.warning(f"[ROUTER] LLM routing failed: {e}")
            return None

    def _build_routing_prompt(
        self,
        query: str,
        recent_symbols: List[str]
    ) -> str:
        """Build prompt for LLM-based routing"""

        recent_context = ""
        if recent_symbols:
            recent_context = f"""
<recent_context>
Symbols from recent conversation: {', '.join(recent_symbols)}
</recent_context>
"""

        return f"""<task>
Classify this query to determine the optimal execution mode.
</task>

<query>
{query}
</query>
{recent_context}
<modes>
1. SIMPLE: Conversational messages, greetings, thanks, definitions, general knowledge
   - No external tools needed
   - Direct LLM response
   - Examples: "hello", "what is P/E ratio?", "thanks"

2. PARALLEL: Financial queries with known scope
   - Specific symbols mentioned or implied
   - Multiple tools can run in parallel
   - Examples: "analyze AAPL", "compare MSFT and GOOGL", "price of NVDA"

3. AGENTIC: Complex tasks requiring adaptive execution
   - Discovery/screening (unknown symbols)
   - Conditional logic (if/then)
   - Research tasks
   - Examples: "top 10 tech stocks", "if RSI < 30 then buy signal", "research AI trends"
</modes>

<rules>
1. If user references "it", "that stock" → check recent_context for symbols
2. Greeting/thanks/bye → SIMPLE
3. Known symbols + analysis → PARALLEL
4. "find", "top", "best", "screen" without specific symbols → AGENTIC
5. "if", "when", conditional logic → AGENTIC
6. Definition questions without symbols → SIMPLE
7. When unsure, prefer PARALLEL
</rules>

<output_format>
Return JSON only:
{{
  "mode": "simple|parallel|agentic",
  "confidence": 0.0-1.0,
  "is_conversational": true/false,
  "symbols": ["AAPL", ...],
  "categories": ["price", "technical", ...],
  "language": "en|vi|zh",
  "reasoning": "Brief explanation"
}}
</output_format>"""

    def _parse_llm_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response to dict"""
        try:
            # Remove markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1]

            content = content.strip()
            return json.loads(content)

        except json.JSONDecodeError as e:
            self.logger.warning(f"[ROUTER] JSON parse error: {e}")
            return None

    def _parse_mode(self, mode_str: str) -> ExecutionMode:
        """Parse mode string to enum"""
        mode_map = {
            "simple": ExecutionMode.SIMPLE,
            "parallel": ExecutionMode.PARALLEL,
            "agentic": ExecutionMode.AGENTIC
        }
        return mode_map.get(mode_str.lower(), ExecutionMode.PARALLEL)

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _matches_any(self, text: str, patterns: List[re.Pattern]) -> bool:
        """Check if text matches any of the patterns"""
        return any(p.search(text) for p in patterns)

    def _extract_symbols(self, query: str) -> List[str]:
        """Extract stock/crypto symbols from query"""
        symbols = set()

        # Extract stock symbols
        stock_matches = self._symbol_pattern.findall(query)
        for match in stock_matches:
            if match not in RoutingPatterns.SYMBOL_EXCLUSIONS:
                symbols.add(match)

        # Extract crypto symbols
        crypto_matches = self._crypto_pattern.findall(query.upper())
        for match in crypto_matches:
            # Normalize crypto symbols
            if not match.endswith("USD"):
                symbols.add(match + "USD")
            else:
                symbols.add(match)

        return list(symbols)

    def _detect_language(self, query: str) -> str:
        """Simple language detection"""
        # Vietnamese indicators
        vn_chars = set('àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
        if any(c in vn_chars for c in query.lower()):
            return "vi"

        # Chinese indicators
        if re.search(r'[\u4e00-\u9fff]', query):
            return "zh"

        return "en"

    def _elapsed_ms(self, start_time: datetime) -> int:
        """Calculate elapsed time in milliseconds"""
        return int((datetime.now() - start_time).total_seconds() * 1000)

    def _update_stats(self, result: RoutingResult, method: str):
        """Update routing statistics"""
        self._stats["total_routes"] += 1
        self._stats[f"{method}_routes"] += 1
        self._stats["mode_counts"][result.mode.value] += 1

    def _log_routing_result(self, result: RoutingResult):
        """Log routing result"""
        self.logger.info(f"[ROUTER] ────────────────────────────────────────")
        self.logger.info(f"[ROUTER] Mode: {result.mode.value.upper()}")
        self.logger.info(f"[ROUTER] Confidence: {result.confidence:.2f}")
        self.logger.info(f"[ROUTER] Method: {result.route_method}")
        self.logger.info(f"[ROUTER] Symbols: {result.symbols}")
        self.logger.info(f"[ROUTER] Language: {result.detected_language}")
        self.logger.info(f"[ROUTER] Time: {result.routing_time_ms}ms")
        self.logger.info(f"[ROUTER] Reasoning: {result.reasoning}")
        self.logger.info(f"[ROUTER] ════════════════════════════════════════")

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return self._stats.copy()

    def reset_stats(self):
        """Reset routing statistics"""
        self._stats = {
            "total_routes": 0,
            "deterministic_routes": 0,
            "llm_routes": 0,
            "fallback_routes": 0,
            "mode_counts": {m.value: 0 for m in ExecutionMode}
        }


# ============================================================================
# SINGLETON & FACTORY
# ============================================================================

_router_instance: Optional[QueryRouter] = None


def get_query_router(
    model_name: str = "gpt-4.1-nano",
    provider_type: ProviderType = ProviderType.OPENAI,
    enable_llm_routing: bool = True
) -> QueryRouter:
    """
    Get singleton QueryRouter instance

    Args:
        model_name: Model for LLM routing
        provider_type: LLM provider
        enable_llm_routing: Enable LLM for ambiguous queries

    Returns:
        QueryRouter singleton instance
    """
    global _router_instance

    if _router_instance is None:
        _router_instance = QueryRouter(
            model_name=model_name,
            provider_type=provider_type,
            enable_llm_routing=enable_llm_routing
        )

    return _router_instance


def reset_router():
    """Reset the singleton router instance"""
    global _router_instance
    _router_instance = None
