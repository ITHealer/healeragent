from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
import json
import re

from src.utils.logger.custom_logging import LoggerMixin
from src.services.asset.symbol_cache import (
    SymbolCacheService,
    SymbolInfo,
    AssetClass,
    AssetSubClass,
    get_symbol_cache
)
from src.services.asset.crypto_validator import (
    CryptoSymbolValidator,
    get_crypto_validator,
)


@dataclass
class ExtractedEntity:
    """Entity extracted from user query by LLM"""
    raw_text: str           # Original text: "Bitcoin", "特斯拉"
    normalized: str         # Normalized symbol: "BTC", "TSLA"
    possible_types: List[str] = field(default_factory=list)  # ["crypto"], ["stock"], ["crypto", "stock"]
    confidence: float = 0.8
    context_words: List[str] = field(default_factory=list)


@dataclass
class ResolvedAsset:
    """Fully resolved asset with all info needed for analysis"""
    symbol: str
    name: str
    asset_class: AssetClass
    sub_class: str

    # For crypto
    trading_pair: Optional[str] = None      # BTCUSDT
    quote_currency: str = "USD"             # Display currency

    # For stock
    exchange: Optional[str] = None

    # Disambiguation
    needs_confirmation: bool = False
    alternatives: List[SymbolInfo] = field(default_factory=list)
    disambiguation_message: Optional[str] = None


class AssetResolver(LoggerMixin):
    """
    Resolves user query to specific assets with proper configuration.

    Flow:
    1. LLM extracts entities from query (multilingual)
    2. Symbol cache lookup (O(1) - no DB)
    3. Handle ambiguity (SOL = Solana or Renesola?)
    4. Auto-select quote currency (BTC → BTCUSDT)
    5. Load asset config (forbidden_methods, data_sources)
    """

    # Context keywords for disambiguation
    CRYPTO_CONTEXT_HINTS = {
        # English crypto terms
        "staking", "mining", "wallet", "blockchain", "defi", "nft",
        "hodl", "moon", "pump", "dump", "whale", "gas", "gwei",
        "layer", "chain", "token", "coin", "airdrop", "stake",
        "liquidity", "yield", "farm", "swap", "dex", "cex",
        # Vietnamese crypto terms
        "đào", "ví", "khối", "đồng", "xu", "sàn",
        # Chinese crypto terms
        "挖矿", "钱包", "区块链", "代币", "币", "链",
    }

    STOCK_CONTEXT_HINTS = {
        # English stock terms
        "earnings", "dividend", "eps", "revenue", "quarterly",
        "annual", "report", "sec", "filing", "shares", "stock",
        "equity", "market cap", "pe ratio", "valuation", "growth",
        "profit", "loss", "balance sheet", "income statement",
        # Vietnamese stock terms
        "cổ phiếu", "cổ tức", "doanh thu", "lợi nhuận", "báo cáo",
        "quý", "năm", "vốn hóa", "định giá",
        # Chinese stock terms
        "股票", "股息", "收入", "利润", "报告", "季度", "年度",
    }

    # Well-known symbol mappings (name -> symbol)
    NAME_TO_SYMBOL = {
        # Crypto (multilingual)
        "bitcoin": "BTC",
        "btc": "BTC",
        "ethereum": "ETH",
        "eth": "ETH",
        "solana": "SOL",
        "cardano": "ADA",
        "dogecoin": "DOGE",
        "ripple": "XRP",
        "比特币": "BTC",
        "以太坊": "ETH",
        "以太币": "ETH",

        # Stocks (multilingual)
        "apple": "AAPL",
        "苹果": "AAPL",
        "táo": "AAPL",
        "tesla": "TSLA",
        "特斯拉": "TSLA",
        "nvidia": "NVDA",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "amazon": "AMZN",
        "meta": "META",
        "facebook": "META",
    }

    _instance: Optional['AssetResolver'] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        symbol_cache: Optional[SymbolCacheService] = None,
        crypto_validator: Optional[CryptoSymbolValidator] = None,
        llm_provider=None,
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return

        super().__init__()
        self.cache = symbol_cache or get_symbol_cache()
        self.crypto_validator = crypto_validator or get_crypto_validator()
        self.llm = llm_provider

        self._initialized = True
        self.logger.info("[ASSET_RESOLVER] Initialized")

    async def resolve(
        self,
        query: str,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        skip_confirmation: bool = False,
        use_llm: bool = True,
    ) -> Tuple[List[ResolvedAsset], List[ExtractedEntity]]:
        """
        Main entry point: resolve all assets mentioned in query.

        Args:
            query: User's natural language query
            conversation_context: Previous messages for context
            skip_confirmation: If True, auto-pick most likely for ambiguous
            use_llm: If True, use LLM for entity extraction (slower but accurate)

        Returns:
            Tuple of (resolved_assets, unresolved_entities)
        """
        self.logger.debug(f"[ASSET_RESOLVER] Resolving query: {query[:100]}...")

        # Step 1: Extract entities
        if use_llm and self.llm:
            entities = await self._extract_entities_llm(query)
        else:
            entities = self._extract_entities_simple(query)

        if not entities:
            self.logger.debug("[ASSET_RESOLVER] No entities extracted")
            return [], []

        self.logger.debug(f"[ASSET_RESOLVER] Extracted {len(entities)} entities")

        resolved = []
        unresolved = []

        for entity in entities:
            # Step 2: Lookup in cache (O(1))
            info, is_ambiguous = self.cache.lookup(entity.normalized)

            if info is None:
                # Symbol not found in cache - try crypto validator
                crypto_result = await self.crypto_validator.validate(entity.normalized)
                if crypto_result.is_valid and crypto_result.exact_match:
                    # Found in crypto API
                    match = crypto_result.exact_match
                    asset = ResolvedAsset(
                        symbol=match.symbol,
                        name=match.name,
                        asset_class=AssetClass.CRYPTO,
                        sub_class=AssetSubClass.CRYPTO_OTHER.value,
                        trading_pair=self.cache.get_trading_pair(match.symbol),
                        quote_currency="USD",
                    )
                    resolved.append(asset)
                    # Add to cache for future
                    self.cache.add_symbol(
                        match.symbol, match.name, AssetClass.CRYPTO
                    )
                else:
                    unresolved.append(entity)
                continue

            # Step 3: Handle ambiguity
            if is_ambiguous and not skip_confirmation:
                # Try context-based resolution
                resolved_class = self._resolve_by_context(
                    entity,
                    query,
                    conversation_context or []
                )

                if resolved_class is None:
                    # Still ambiguous - need user confirmation
                    alternatives = self.cache.get_ambiguous_options(entity.normalized)
                    message = self._generate_disambiguation_message(entity, alternatives)

                    resolved.append(ResolvedAsset(
                        symbol=entity.normalized,
                        name=info.name,
                        asset_class=info.asset_class,
                        sub_class=info.sub_class.value if hasattr(info.sub_class, 'value') else str(info.sub_class),
                        needs_confirmation=True,
                        alternatives=alternatives,
                        disambiguation_message=message,
                    ))
                    continue
                else:
                    # Context resolved it
                    options = self.cache.get_ambiguous_options(entity.normalized)
                    info = next(
                        (o for o in options if o.asset_class == resolved_class),
                        info
                    )

            # Step 4: Build resolved asset
            asset = self._build_resolved_asset(info, entity)
            resolved.append(asset)

        self.logger.info(
            f"[ASSET_RESOLVER] Resolved: {len(resolved)} assets, "
            f"{len(unresolved)} unresolved, "
            f"{sum(1 for a in resolved if a.needs_confirmation)} need confirmation"
        )

        return resolved, unresolved

    async def resolve_with_confirmation(
        self,
        symbol: str,
        chosen_asset_class: AssetClass,
    ) -> Optional[ResolvedAsset]:
        """
        Resolve a previously ambiguous symbol with user's choice.

        Args:
            symbol: The ambiguous symbol
            chosen_asset_class: User's choice of asset class

        Returns:
            Resolved asset or None if not found
        """
        options = self.cache.get_ambiguous_options(symbol)
        chosen = next(
            (o for o in options if o.asset_class == chosen_asset_class),
            None
        )

        if chosen is None:
            return None

        entity = ExtractedEntity(
            raw_text=symbol,
            normalized=symbol,
            possible_types=[chosen_asset_class.value],
            confidence=1.0,
        )

        return self._build_resolved_asset(chosen, entity)

    def _extract_entities_simple(self, query: str) -> List[ExtractedEntity]:
        """
        Simple extraction without LLM (fast but less accurate).
        Uses regex + known symbol mappings.
        """
        entities = []
        query_lower = query.lower()
        query_upper = query.upper()

        # Check for known name mappings first
        for name, symbol in self.NAME_TO_SYMBOL.items():
            if name in query_lower:
                # Determine possible types
                possible = []
                if self.cache.exists(symbol, AssetClass.CRYPTO):
                    possible.append("crypto")
                if self.cache.exists(symbol, AssetClass.STOCK):
                    possible.append("stock")

                if not possible:
                    possible = ["crypto", "stock"]  # Unknown

                entities.append(ExtractedEntity(
                    raw_text=name,
                    normalized=symbol,
                    possible_types=possible,
                    confidence=0.9,
                ))

        # Find potential symbols (2-6 uppercase letters)
        # More restrictive pattern to reduce false positives
        pattern = r'\b([A-Z]{2,6})\b'
        matches = re.findall(pattern, query_upper)

        # Common false positives to skip
        skip_words = {
            "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
            "CEO", "CFO", "CTO", "IPO", "ETF", "API", "USD", "EUR",
            "VND", "THB", "JPY", "GBP", "CNY", "HKD", "SGD",
        }

        for match in matches:
            if match in skip_words:
                continue

            # Check if already added via name mapping
            if any(e.normalized == match for e in entities):
                continue

            # Check if it exists in cache
            info, is_ambiguous = self.cache.lookup(match)
            if info or is_ambiguous:
                possible = []
                if self.cache.exists(match, AssetClass.CRYPTO):
                    possible.append("crypto")
                if self.cache.exists(match, AssetClass.STOCK):
                    possible.append("stock")

                entities.append(ExtractedEntity(
                    raw_text=match,
                    normalized=match,
                    possible_types=possible if possible else ["crypto", "stock"],
                    confidence=0.8 if info else 0.6,
                ))

        return entities[:10]  # Limit to 10 entities

    async def _extract_entities_llm(self, query: str) -> List[ExtractedEntity]:
        """
        Use LLM to extract financial entities from query.
        Supports: English, Vietnamese, Chinese.
        """
        if self.llm is None:
            return self._extract_entities_simple(query)

        prompt = f"""Extract financial asset symbols from this query.
Be thorough and extract ALL mentioned assets.

Query: {query}

For each asset mentioned, provide:
1. raw_text: The exact text used (e.g., "Bitcoin", "特斯拉", "cổ phiếu Apple")
2. normalized: Standard symbol (e.g., "BTC", "TSLA", "AAPL")
3. possible_types: Asset types it could be (["crypto"], ["stock"], or both if ambiguous)
4. confidence: 0.0 to 1.0

Output as JSON array. If no assets found, return empty array [].

Examples:
- "分析一下BTC和苹果股票" → [{{"raw_text": "BTC", "normalized": "BTC", "possible_types": ["crypto"], "confidence": 1.0}}, {{"raw_text": "苹果股票", "normalized": "AAPL", "possible_types": ["stock"], "confidence": 1.0}}]
- "SOL đang pump" → [{{"raw_text": "SOL", "normalized": "SOL", "possible_types": ["crypto", "stock"], "confidence": 0.8}}]
- "Phân tích Bitcoin" → [{{"raw_text": "Bitcoin", "normalized": "BTC", "possible_types": ["crypto"], "confidence": 1.0}}]
"""

        try:
            from src.helpers.llm_helper import LLMGeneratorProvider
            from src.providers.provider_factory import ProviderType

            llm = LLMGeneratorProvider()
            response = await llm.generate_response(
                model_name="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                provider_type=ProviderType.OPENAI,
                max_tokens=500,
                temperature=0.1,
            )

            content = response.get("content", "[]") if isinstance(response, dict) else str(response)

            # Extract JSON from response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                entities = []
                for item in data:
                    entities.append(ExtractedEntity(
                        raw_text=item.get("raw_text", ""),
                        normalized=item.get("normalized", "").upper(),
                        possible_types=item.get("possible_types", []),
                        confidence=item.get("confidence", 0.8),
                        context_words=item.get("context_words", []),
                    ))
                return entities

        except Exception as e:
            self.logger.warning(f"[ASSET_RESOLVER] LLM extraction failed: {e}")

        return self._extract_entities_simple(query)

    def _resolve_by_context(
        self,
        entity: ExtractedEntity,
        query: str,
        conversation_context: List[Dict[str, str]]
    ) -> Optional[AssetClass]:
        """
        Try to resolve ambiguous symbol using context clues.

        Returns:
            AssetClass if confidently resolved, None if still ambiguous
        """
        # Collect all context words from query
        query_lower = query.lower()
        all_words = set(query_lower.split())

        # Add words from conversation history
        for msg in conversation_context[-5:]:
            content = msg.get("content", "")
            all_words.update(content.lower().split())

        # Score crypto vs stock hints
        crypto_score = len(all_words & self.CRYPTO_CONTEXT_HINTS)
        stock_score = len(all_words & self.STOCK_CONTEXT_HINTS)

        # Check entity's possible types
        if entity.possible_types:
            if len(entity.possible_types) == 1:
                if entity.possible_types[0] == "crypto":
                    return AssetClass.CRYPTO
                elif entity.possible_types[0] == "stock":
                    return AssetClass.STOCK

        self.logger.debug(
            f"[ASSET_RESOLVER] Context resolution for {entity.normalized}: "
            f"crypto_score={crypto_score}, stock_score={stock_score}"
        )

        # Need clear winner (2+ point difference)
        if crypto_score > stock_score + 1:
            return AssetClass.CRYPTO
        elif stock_score > crypto_score + 1:
            return AssetClass.STOCK
        else:
            return None  # Still ambiguous

    def _build_resolved_asset(
        self,
        info: SymbolInfo,
        entity: ExtractedEntity
    ) -> ResolvedAsset:
        """Build fully resolved asset with config"""

        # Determine trading pair for crypto
        trading_pair = None
        quote_currency = "USD"

        if info.asset_class == AssetClass.CRYPTO:
            trading_pair = self.cache.get_trading_pair(
                info.symbol,
                exchange="binance",
                quote=info.default_quote if hasattr(info, 'default_quote') else "USDT"
            )
            quote_currency = "USD"  # Always display in USD

        return ResolvedAsset(
            symbol=info.symbol,
            name=info.name,
            asset_class=info.asset_class,
            sub_class=info.sub_class.value if hasattr(info.sub_class, 'value') else str(info.sub_class),
            trading_pair=trading_pair,
            quote_currency=quote_currency,
            exchange=info.exchange if hasattr(info, 'exchange') else None,
            needs_confirmation=False,
        )

    def _generate_disambiguation_message(
        self,
        entity: ExtractedEntity,
        alternatives: List[SymbolInfo]
    ) -> str:
        """Generate user-friendly disambiguation message"""

        lines = [f"Tôi thấy '{entity.raw_text}' có thể là:"]

        for i, alt in enumerate(alternatives, 1):
            if alt.asset_class == AssetClass.CRYPTO:
                lines.append(
                    f"  {i}. {alt.name} (Cryptocurrency)"
                )
            elif alt.asset_class == AssetClass.STOCK:
                exchange_info = f" - {alt.exchange}" if alt.exchange else ""
                lines.append(
                    f"  {i}. {alt.name} (Stock{exchange_info})"
                )
            else:
                lines.append(f"  {i}. {alt.name} ({alt.asset_class.value})")

        lines.append("\nBạn muốn phân tích loại nào?")

        return "\n".join(lines)

    def get_disambiguation_options(self, symbol: str) -> List[Dict[str, Any]]:
        """Get disambiguation options in a structured format for UI"""
        options = self.cache.get_ambiguous_options(symbol)

        return [
            {
                "symbol": opt.symbol,
                "name": opt.name,
                "asset_class": opt.asset_class.value,
                "sub_class": opt.sub_class.value if hasattr(opt.sub_class, 'value') else str(opt.sub_class),
                "exchange": opt.exchange,
                "description": self._get_asset_description(opt),
            }
            for opt in options
        ]

    def _get_asset_description(self, info: SymbolInfo) -> str:
        """Generate description for an asset"""
        if info.asset_class == AssetClass.CRYPTO:
            sub = info.sub_class
            if sub == AssetSubClass.CRYPTO_LAYER1:
                return "Layer 1 blockchain"
            elif sub == AssetSubClass.CRYPTO_LAYER2:
                return "Layer 2 scaling solution"
            elif sub == AssetSubClass.CRYPTO_DEFI:
                return "DeFi protocol"
            elif sub == AssetSubClass.CRYPTO_MEME:
                return "Meme coin"
            elif sub == AssetSubClass.CRYPTO_STABLECOIN:
                return "Stablecoin"
            else:
                return "Cryptocurrency"
        elif info.asset_class == AssetClass.STOCK:
            return f"Stock on {info.exchange}" if info.exchange else "Stock"
        else:
            return info.asset_class.value


# ========================================
# SINGLETON ACCESSOR
# ========================================

_asset_resolver: Optional[AssetResolver] = None


def get_asset_resolver(
    symbol_cache: Optional[SymbolCacheService] = None,
    crypto_validator: Optional[CryptoSymbolValidator] = None,
) -> AssetResolver:
    """Get singleton AssetResolver instance"""
    global _asset_resolver

    if _asset_resolver is None:
        _asset_resolver = AssetResolver(
            symbol_cache=symbol_cache,
            crypto_validator=crypto_validator,
        )

    return _asset_resolver