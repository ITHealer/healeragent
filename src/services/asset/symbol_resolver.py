"""
Symbol Resolver Service

Multi-layer symbol resolution with Soft Context Inheritance.

Resolution Priority:
1. Explicit Pattern Match (.HK, .SS, BTCUSDT) → High confidence
2. Symbol Cache Lookup (known symbols) → High confidence
3. UI Context (active tab) + Pattern → Medium confidence
4. LLM Semantic Understanding (multilingual) → Medium confidence
5. Default Rules (Stock tab → US exchange) → Low confidence

Principle: "Assume smartly, Confirm explicitly, Correct gracefully"
"""

import re
import json
import hashlib
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.redis_cache import redis_manager

from .symbol_cache import (
    SymbolCacheService,
    SymbolInfo,
    AssetClass,
    get_symbol_cache,
)
from .symbol_resolution_models import (
    UIContext,
    ActiveTab,
    Exchange,
    ResolutionMethod,
    ConfidenceLevel,
    ResolutionInfo,
    AlternativeSymbol,
    ResolvedSymbol,
    SymbolResolutionResult,
    EXCHANGE_SUFFIX_PATTERNS,
    CRYPTO_PAIR_PATTERNS,
    DEFAULT_EXCHANGES,
)


# Redis cache config
RESOLUTION_CACHE_PREFIX = "symbol_resolution:"
RESOLUTION_CACHE_TTL = 300  # 5 minutes


class SymbolResolver(LoggerMixin):
    """
    Multi-layer symbol resolution service.

    Features:
    - Pattern-based detection (exchange suffixes, crypto pairs)
    - O(1) symbol cache lookup
    - UI context inheritance (active tab)
    - LLM semantic understanding (multilingual)
    - Redis caching for fast repeated lookups

    Usage:
        resolver = get_symbol_resolver()
        result = await resolver.resolve(
            symbols=["AAPL", "0700.HK", "Bitcoin"],
            query="分析一下 Bitcoin 和 腾讯",
            ui_context=UIContext(active_tab=ActiveTab.STOCK),
        )
    """

    _instance: Optional['SymbolResolver'] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        symbol_cache: Optional[SymbolCacheService] = None,
        llm_provider=None,
    ):
        if self._initialized:
            return

        super().__init__()
        self._cache = symbol_cache or get_symbol_cache()
        self._llm = llm_provider
        self._initialized = True
        self.logger.info("[SYMBOL_RESOLVER] Initialized")

    async def resolve(
        self,
        symbols: List[str],
        query: str = "",
        ui_context: Optional[UIContext] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_cache: bool = True,
    ) -> SymbolResolutionResult:
        """
        Resolve symbols with full context.

        Args:
            symbols: List of raw symbols/entities extracted from classification
            query: Original user query (for LLM semantic resolution)
            ui_context: UI context (active tab, recent symbols)
            conversation_history: For context clues
            use_cache: Whether to use Redis cache

        Returns:
            SymbolResolutionResult with resolved symbols and any ambiguities
        """
        self.logger.info(
            f"[SYMBOL_RESOLVER] Resolving {len(symbols)} symbols, "
            f"ui_context={ui_context.active_tab.value if ui_context else 'none'}"
        )

        if not symbols:
            return SymbolResolutionResult()

        ui_context = ui_context or UIContext()
        resolved_symbols = []
        needs_clarification = []
        unresolved = []

        for raw_symbol in symbols:
            # Check cache first
            if use_cache:
                cached = await self._get_from_cache(raw_symbol, ui_context)
                if cached:
                    resolved_symbols.append(cached)
                    continue

            # Multi-layer resolution
            result = await self._resolve_single(
                raw_symbol=raw_symbol,
                query=query,
                ui_context=ui_context,
                conversation_history=conversation_history or [],
            )

            if result is None:
                unresolved.append(raw_symbol)
            elif result.clarification_needed:
                needs_clarification.append(result)
            else:
                resolved_symbols.append(result)
                # Cache successful resolution
                if use_cache:
                    asyncio.create_task(
                        self._set_to_cache(raw_symbol, ui_context, result)
                    )

        # Build result
        all_resolved = resolved_symbols + [r for r in needs_clarification if not r.clarification_needed]
        overall_confidence = self._calculate_overall_confidence(all_resolved)

        clarification_messages = [
            r.clarification_message
            for r in needs_clarification
            if r.clarification_message
        ]

        result = SymbolResolutionResult(
            resolved_symbols=resolved_symbols,
            needs_clarification=needs_clarification,
            unresolved=unresolved,
            overall_confidence=overall_confidence,
            clarification_messages=clarification_messages,
        )

        self.logger.info(
            f"[SYMBOL_RESOLVER] Result: "
            f"resolved={len(resolved_symbols)}, "
            f"needs_clarification={len(needs_clarification)}, "
            f"unresolved={len(unresolved)}, "
            f"confidence={overall_confidence:.2f}"
        )

        return result

    async def _resolve_single(
        self,
        raw_symbol: str,
        query: str,
        ui_context: UIContext,
        conversation_history: List[Dict[str, str]],
    ) -> Optional[ResolvedSymbol]:
        """
        Resolve a single symbol through multiple layers.

        Resolution order:
        1. Pattern detection (exchange suffix, crypto pair)
        2. Symbol cache lookup
        3. UI context + ambiguity resolution
        4. LLM semantic (for unknown symbols or company names)
        5. Default rules (based on UI context)
        """
        symbol = raw_symbol.upper().strip()

        # Layer 1: Pattern Detection
        pattern_result = self._resolve_by_pattern(symbol)
        if pattern_result:
            self.logger.debug(f"[SYMBOL_RESOLVER] Pattern match: {symbol}")
            return pattern_result

        # Layer 2: Symbol Cache Lookup
        info, is_ambiguous = self._cache.lookup(symbol)

        if info and not is_ambiguous:
            # Clear match in cache
            return self._create_resolved_symbol(
                info=info,
                method=ResolutionMethod.CACHE,
                confidence=0.95,
                reasoning=f"Found in symbol cache: {info.name}",
                original_text=raw_symbol,
            )

        # Layer 3: Ambiguous Symbol - Try UI Context Resolution
        if is_ambiguous:
            resolved = self._resolve_ambiguity_with_context(
                symbol=symbol,
                ui_context=ui_context,
                query=query,
                conversation_history=conversation_history,
            )
            if resolved:
                return resolved

            # Still ambiguous - return with alternatives
            alternatives = self._get_alternatives(symbol)
            return ResolvedSymbol.ambiguous(
                symbol=symbol,
                alternatives=alternatives,
                original_text=raw_symbol,
            )

        # Layer 4: Not in cache - use LLM to resolve
        # This handles both company names (Tesla, Google) and misspelled tickers
        if info is None:
            llm_result = await self._resolve_by_llm(
                raw_symbol, query, ui_context
            )
            if llm_result:
                self.logger.debug(
                    f"[SYMBOL_RESOLVER] LLM resolved: {raw_symbol} → {llm_result.symbol}"
                )
                return llm_result

        # Layer 5: Apply default rules based on UI context
        if info is None and ui_context.active_tab != ActiveTab.NONE:
            return self._apply_default_rules(symbol, ui_context, raw_symbol)

        # Could not resolve
        return None

    def _resolve_by_pattern(self, symbol: str) -> Optional[ResolvedSymbol]:
        """
        Layer 1: Pattern-based detection.

        Detects:
        - Exchange suffixes: 0700.HK → HKEX, 600519.SS → Shanghai
        - Crypto pairs: BTCUSDT, BTC-USD → Crypto
        """
        # Check exchange suffix patterns
        for pattern, exchange in EXCHANGE_SUFFIX_PATTERNS.items():
            if re.search(pattern, symbol, re.IGNORECASE):
                # Extract base symbol
                base_symbol = re.sub(pattern, "", symbol, flags=re.IGNORECASE)

                return ResolvedSymbol(
                    symbol=symbol,  # Keep full symbol with suffix
                    name=f"{base_symbol} ({exchange.value})",
                    asset_type="stock",
                    exchange=exchange,
                    original_text=symbol,
                    resolution_info=ResolutionInfo.high_confidence(
                        method=ResolutionMethod.PATTERN,
                        pattern_matched=pattern,
                        reasoning=f"Exchange suffix detected: {pattern}",
                    ),
                )

        # Check crypto pair patterns
        for pattern in CRYPTO_PAIR_PATTERNS:
            match = re.match(pattern, symbol, re.IGNORECASE)
            if match:
                base = match.group(1).upper()
                quote = match.group(2).upper() if len(match.groups()) > 1 else "USD"

                return ResolvedSymbol(
                    symbol=base,
                    name=f"{base} Cryptocurrency",
                    asset_type="crypto",
                    trading_pair=f"{base}{quote}",
                    quote_currency=quote,
                    original_text=symbol,
                    resolution_info=ResolutionInfo.high_confidence(
                        method=ResolutionMethod.PATTERN,
                        pattern_matched=f"Crypto pair: {base}-{quote}",
                        reasoning=f"Crypto trading pair pattern matched",
                    ),
                )

        return None

    def _resolve_ambiguity_with_context(
        self,
        symbol: str,
        ui_context: UIContext,
        query: str,
        conversation_history: List[Dict[str, str]],
    ) -> Optional[ResolvedSymbol]:
        """
        Resolve ambiguous symbol using UI context.

        Priority:
        1. UI active tab (Stock tab → prefer stock interpretation)
        2. Recent symbols in UI (if same type as recent)
        """
        options = self._cache.get_ambiguous_options(symbol)
        if not options:
            return None

        # Strategy 1: Use active tab
        if ui_context.active_tab == ActiveTab.STOCK:
            stock_option = next(
                (o for o in options if o.asset_class == AssetClass.STOCK),
                None
            )
            if stock_option:
                return self._create_resolved_symbol(
                    info=stock_option,
                    method=ResolutionMethod.UI_CONTEXT,
                    confidence=0.85,
                    reasoning=f"Stock tab active, chose stock interpretation: {stock_option.name}",
                    original_text=symbol,
                    context_used="Stock tab active",
                )

        elif ui_context.active_tab == ActiveTab.CRYPTO:
            crypto_option = next(
                (o for o in options if o.asset_class == AssetClass.CRYPTO),
                None
            )
            if crypto_option:
                return self._create_resolved_symbol(
                    info=crypto_option,
                    method=ResolutionMethod.UI_CONTEXT,
                    confidence=0.85,
                    reasoning=f"Crypto tab active, chose crypto interpretation: {crypto_option.name}",
                    original_text=symbol,
                    context_used="Crypto tab active",
                )

        # Strategy 2: Check recent symbols for context clues
        if ui_context.recent_symbols:
            # If user recently viewed crypto, lean towards crypto
            recent_is_crypto = any(
                self._cache.exists(s, AssetClass.CRYPTO)
                for s in ui_context.recent_symbols[:3]
            )
            if recent_is_crypto:
                crypto_option = next(
                    (o for o in options if o.asset_class == AssetClass.CRYPTO),
                    None
                )
                if crypto_option:
                    return self._create_resolved_symbol(
                        info=crypto_option,
                        method=ResolutionMethod.UI_CONTEXT,
                        confidence=0.75,
                        reasoning=f"Recent symbols were crypto, chose: {crypto_option.name}",
                        original_text=symbol,
                        context_used="Recent crypto context",
                    )

        # Could not resolve with context
        return None

    def _apply_default_rules(
        self,
        symbol: str,
        ui_context: UIContext,
        original_text: str,
    ) -> Optional[ResolvedSymbol]:
        """
        Apply default rules when symbol not in cache.

        Rules:
        - Stock tab + unknown symbol → Assume US stock
        - Crypto tab + unknown symbol → Assume crypto token
        """
        if ui_context.active_tab == ActiveTab.STOCK:
            return ResolvedSymbol(
                symbol=symbol,
                name=f"{symbol} (assumed US stock)",
                asset_type="stock",
                exchange=Exchange.NYSE,
                original_text=original_text,
                resolution_info=ResolutionInfo.low_confidence(
                    method=ResolutionMethod.DEFAULT,
                    reasoning=f"Stock tab active, assuming US stock for unknown symbol",
                    context_used="Stock tab default rule",
                ),
            )

        elif ui_context.active_tab == ActiveTab.CRYPTO:
            return ResolvedSymbol(
                symbol=symbol,
                name=f"{symbol} (assumed cryptocurrency)",
                asset_type="crypto",
                trading_pair=f"{symbol}USDT",
                quote_currency="USDT",
                original_text=original_text,
                resolution_info=ResolutionInfo.low_confidence(
                    method=ResolutionMethod.DEFAULT,
                    reasoning=f"Crypto tab active, assuming cryptocurrency for unknown symbol",
                    context_used="Crypto tab default rule",
                ),
            )

        return None

    async def _resolve_by_llm(
        self,
        raw_text: str,
        query: str,
        ui_context: UIContext,
    ) -> Optional[ResolvedSymbol]:
        """
        Use LLM for semantic understanding of company/asset names.

        Handles:
        - Company names: GOOGLE → GOOGL, AMAZON → AMZN
        - Multilingual: "腾讯" → 0700.HK, "特斯拉" → TSLA
        - Crypto: "Bitcoin" → BTC
        """
        try:
            from src.helpers.llm_helper import LLMGeneratorProvider
            from src.providers.provider_factory import ProviderType

            tab_context = ""
            if ui_context.active_tab == ActiveTab.STOCK:
                tab_context = "User is viewing stocks. Prefer stock interpretation."
            elif ui_context.active_tab == ActiveTab.CRYPTO:
                tab_context = "User is viewing crypto. Prefer cryptocurrency interpretation."

            prompt = f"""Given the following text, identify the correct stock ticker or crypto symbol.

Text: "{raw_text}"
Full query: "{query}"
Context: {tab_context}

IMPORTANT RULES:
1. If text is a company NAME (not ticker), convert to official stock ticker
2. If text is a cryptocurrency name, return the crypto symbol
3. Company names may be uppercase (GOOGLE, AMAZON) - still convert to ticker
4. Return null only if you cannot identify the asset

Respond ONLY with JSON:
{{"symbol": "TICKER", "name": "FULL_NAME", "asset_type": "stock|crypto", "confidence": 0.0-1.0}}

Examples:
- "GOOGLE" → {{"symbol": "GOOGL", "name": "Alphabet Inc", "asset_type": "stock", "confidence": 0.95}}
- "AMAZON" → {{"symbol": "AMZN", "name": "Amazon.com Inc", "asset_type": "stock", "confidence": 0.95}}
- "NETFLIX" → {{"symbol": "NFLX", "name": "Netflix Inc", "asset_type": "stock", "confidence": 0.95}}
- "FACEBOOK" → {{"symbol": "META", "name": "Meta Platforms Inc", "asset_type": "stock", "confidence": 0.95}}
- "Bitcoin" → {{"symbol": "BTC", "name": "Bitcoin", "asset_type": "crypto", "confidence": 0.95}}
- "腾讯" → {{"symbol": "0700.HK", "name": "Tencent Holdings", "asset_type": "stock", "confidence": 0.9}}
- "TSLA" → {{"symbol": "TSLA", "name": "Tesla Inc", "asset_type": "stock", "confidence": 0.99}}
- "random123xyz" → {{"symbol": null, "name": null, "asset_type": null, "confidence": 0.0}}
"""

            llm = LLMGeneratorProvider()
            response = await llm.generate_response(
                model_name="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                provider_type=ProviderType.OPENAI,
                max_tokens=200,
                temperature=0.1,
            )

            content = response.get("content", "") if isinstance(response, dict) else str(response)

            # Parse JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                if data.get("symbol") and data.get("confidence", 0) > 0.7:
                    exchange = None
                    if data["asset_type"] == "stock" and "." in data["symbol"]:
                        # Extract exchange from suffix
                        for pattern, exch in EXCHANGE_SUFFIX_PATTERNS.items():
                            if re.search(pattern, data["symbol"], re.IGNORECASE):
                                exchange = exch
                                break

                    return ResolvedSymbol(
                        symbol=data["symbol"],
                        name=data.get("name", data["symbol"]),
                        asset_type=data["asset_type"],
                        exchange=exchange,
                        trading_pair=f"{data['symbol']}USDT" if data["asset_type"] == "crypto" else None,
                        quote_currency="USD" if data["asset_type"] == "stock" else "USDT",
                        original_text=raw_text,
                        resolution_info=ResolutionInfo.medium_confidence(
                            method=ResolutionMethod.LLM_SEMANTIC,
                            semantic_match=f"{raw_text} → {data['symbol']}",
                            reasoning=f"LLM identified: {raw_text} as {data.get('name', data['symbol'])}",
                        ),
                    )

        except Exception as e:
            self.logger.warning(f"[SYMBOL_RESOLVER] LLM resolution failed: {e}")

        return None

    def _looks_like_company_name(self, text: str) -> bool:
        """Check if text looks like a company name (not a symbol)"""
        # Symbols are typically uppercase letters only
        if re.match(r'^[A-Z]{1,6}$', text):
            return False

        # Contains lowercase, spaces, or non-ASCII → likely company name
        if any(c.islower() for c in text):
            return True
        if ' ' in text:
            return True
        if any(ord(c) > 127 for c in text):  # Non-ASCII (Chinese, Vietnamese, etc.)
            return True

        return False

    def _get_alternatives(self, symbol: str) -> List[AlternativeSymbol]:
        """Get alternative interpretations for ambiguous symbol"""
        options = self._cache.get_ambiguous_options(symbol)
        alternatives = []

        for opt in options:
            desc = ""
            if opt.asset_class == AssetClass.CRYPTO:
                desc = "Cryptocurrency"
            elif opt.asset_class == AssetClass.STOCK:
                desc = f"Stock on {opt.exchange}" if opt.exchange else "Stock"

            alternatives.append(AlternativeSymbol(
                symbol=opt.symbol,
                name=opt.name,
                asset_type=opt.asset_class.value,
                exchange=opt.exchange,
                description=desc,
            ))

        return alternatives

    def _create_resolved_symbol(
        self,
        info: SymbolInfo,
        method: ResolutionMethod,
        confidence: float,
        reasoning: str,
        original_text: str = "",
        context_used: Optional[str] = None,
    ) -> ResolvedSymbol:
        """Create ResolvedSymbol from SymbolInfo"""
        # Determine trading pair for crypto
        trading_pair = None
        quote_currency = "USD"

        if info.asset_class == AssetClass.CRYPTO:
            quote = info.default_quote if hasattr(info, 'default_quote') else "USDT"
            trading_pair = self._cache.get_trading_pair(info.symbol, quote=quote)
            quote_currency = quote

        # Determine exchange
        exchange = None
        if hasattr(info, 'exchange') and info.exchange:
            try:
                exchange = Exchange(info.exchange)
            except ValueError:
                pass

        # Determine confidence level
        if confidence >= 0.9:
            conf_level = ConfidenceLevel.HIGH
        elif confidence >= 0.7:
            conf_level = ConfidenceLevel.MEDIUM
        else:
            conf_level = ConfidenceLevel.LOW

        return ResolvedSymbol(
            symbol=info.symbol,
            name=info.name,
            asset_type=info.asset_class.value,
            exchange=exchange,
            trading_pair=trading_pair,
            quote_currency=quote_currency,
            original_text=original_text or info.symbol,
            resolution_info=ResolutionInfo(
                method=method,
                confidence=confidence,
                confidence_level=conf_level,
                context_used=context_used,
                reasoning=reasoning,
            ),
        )

    def _calculate_overall_confidence(
        self,
        resolved: List[ResolvedSymbol]
    ) -> float:
        """Calculate overall confidence from resolved symbols"""
        if not resolved:
            return 1.0

        confidences = [
            r.resolution_info.confidence
            for r in resolved
            if r.resolution_info
        ]

        if not confidences:
            return 1.0

        # Return minimum confidence (bottleneck)
        return min(confidences)

    # =========================================================================
    # Redis Cache Methods
    # =========================================================================

    def _make_cache_key(self, symbol: str, ui_context: UIContext) -> str:
        """Create cache key from symbol and context"""
        context_str = f"{symbol}|{ui_context.active_tab.value}"
        key_hash = hashlib.md5(context_str.encode()).hexdigest()
        return f"{RESOLUTION_CACHE_PREFIX}{key_hash}"

    async def _get_from_cache(
        self,
        symbol: str,
        ui_context: UIContext,
    ) -> Optional[ResolvedSymbol]:
        """Get cached resolution result"""
        cache_key = self._make_cache_key(symbol, ui_context)

        try:
            redis_client = await redis_manager.get_client()
            if redis_client:
                cached_json = await asyncio.wait_for(
                    redis_client.get(cache_key),
                    timeout=2.0
                )
                if cached_json:
                    if isinstance(cached_json, bytes):
                        cached_json = cached_json.decode('utf-8')
                    data = json.loads(cached_json)
                    return self._deserialize_resolved_symbol(data)
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            self.logger.debug(f"[SYMBOL_RESOLVER] Cache get error: {e}")

        return None

    async def _set_to_cache(
        self,
        symbol: str,
        ui_context: UIContext,
        result: ResolvedSymbol,
    ) -> None:
        """Cache resolution result"""
        cache_key = self._make_cache_key(symbol, ui_context)

        try:
            redis_client = await redis_manager.get_client()
            if redis_client:
                json_str = json.dumps(result.to_dict(), ensure_ascii=False)
                await asyncio.wait_for(
                    redis_client.set(cache_key, json_str, ex=RESOLUTION_CACHE_TTL),
                    timeout=2.0
                )
        except Exception as e:
            self.logger.debug(f"[SYMBOL_RESOLVER] Cache set error: {e}")

    def _deserialize_resolved_symbol(self, data: Dict[str, Any]) -> ResolvedSymbol:
        """Deserialize ResolvedSymbol from dict"""
        # Reconstruct resolution_info
        resolution_info = None
        if data.get("resolution_info"):
            ri = data["resolution_info"]
            try:
                resolution_info = ResolutionInfo(
                    method=ResolutionMethod(ri.get("method", "default")),
                    confidence=ri.get("confidence", 0.8),
                    confidence_level=ConfidenceLevel(ri.get("confidence_level", "medium")),
                    pattern_matched=ri.get("pattern_matched"),
                    context_used=ri.get("context_used"),
                    semantic_match=ri.get("semantic_match"),
                    reasoning=ri.get("reasoning", ""),
                )
            except (ValueError, KeyError):
                pass

        # Reconstruct exchange
        exchange = None
        if data.get("exchange"):
            try:
                exchange = Exchange(data["exchange"])
            except ValueError:
                pass

        # Reconstruct alternatives
        alternatives = []
        for alt_data in data.get("alternatives", []):
            alternatives.append(AlternativeSymbol(
                symbol=alt_data.get("symbol", ""),
                name=alt_data.get("name", ""),
                asset_type=alt_data.get("asset_type", ""),
                exchange=alt_data.get("exchange"),
                description=alt_data.get("description", ""),
            ))

        return ResolvedSymbol(
            symbol=data.get("symbol", ""),
            name=data.get("name", ""),
            asset_type=data.get("asset_type", ""),
            exchange=exchange,
            trading_pair=data.get("trading_pair"),
            quote_currency=data.get("quote_currency", "USD"),
            resolution_info=resolution_info,
            clarification_needed=data.get("clarification_needed", False),
            alternatives=alternatives,
            clarification_message=data.get("clarification_message"),
            original_text=data.get("original_text", ""),
        )

    async def resolve_with_confirmation(
        self,
        symbol: str,
        chosen_asset_type: str,
        ui_context: Optional[UIContext] = None,
    ) -> Optional[ResolvedSymbol]:
        """
        Resolve previously ambiguous symbol with user's choice.

        Args:
            symbol: The ambiguous symbol
            chosen_asset_type: "crypto" or "stock"
            ui_context: Optional UI context

        Returns:
            Resolved symbol or None
        """
        options = self._cache.get_ambiguous_options(symbol)

        # Find the chosen option
        target_class = (
            AssetClass.CRYPTO if chosen_asset_type == "crypto"
            else AssetClass.STOCK
        )

        chosen = next(
            (o for o in options if o.asset_class == target_class),
            None
        )

        if chosen is None:
            return None

        return self._create_resolved_symbol(
            info=chosen,
            method=ResolutionMethod.USER_CONFIRMED,
            confidence=1.0,
            reasoning=f"User confirmed: {symbol} is {chosen_asset_type}",
            original_text=symbol,
        )


# ============================================================================
# Singleton Accessor
# ============================================================================

_symbol_resolver: Optional[SymbolResolver] = None


def get_symbol_resolver(
    symbol_cache: Optional[SymbolCacheService] = None,
) -> SymbolResolver:
    """Get singleton SymbolResolver instance"""
    global _symbol_resolver

    if _symbol_resolver is None:
        _symbol_resolver = SymbolResolver(symbol_cache=symbol_cache)

    return _symbol_resolver


def reset_symbol_resolver():
    """Reset singleton instance (for testing)"""
    global _symbol_resolver
    _symbol_resolver = None
