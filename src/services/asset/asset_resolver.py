# src/services/asset/asset_resolver.py
"""
Simplified Asset Resolver

Simple disambiguation for symbols that exist in both crypto and stock.
No LLM extraction, no context hints, no quote currency mapping.
BE team handles the correct symbol and currency.

Usage:
    resolver = get_asset_resolver()

    # Check if symbol is ambiguous
    if resolver.is_ambiguous("SOL"):
        options = resolver.get_disambiguation_options("SOL")
        # Show options to user and wait for choice
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.utils.logger.custom_logging import LoggerMixin
from src.services.asset.symbol_cache import (
    SymbolCacheService,
    AssetClass,
    get_symbol_cache,
)


@dataclass
class ResolvedAsset:
    """Resolved asset with disambiguation info"""
    symbol: str
    name: str
    asset_class: AssetClass
    exchange: Optional[str] = None
    needs_confirmation: bool = False
    disambiguation_message: Optional[str] = None


class AssetResolver(LoggerMixin):
    """
    Simple asset resolver for symbol disambiguation.

    Flow:
    1. Check if symbol is ambiguous (exists in both crypto and stock)
    2. If ambiguous, return options for user to choose
    3. If not ambiguous, return the resolved asset

    No LLM extraction - classification already provides symbols.
    No quote currency mapping - BE team handles that.
    """

    _instance: Optional['AssetResolver'] = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, symbol_cache: Optional[SymbolCacheService] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return

        super().__init__()
        self.cache = symbol_cache or get_symbol_cache()
        self._initialized = True
        self.logger.info("[ASSET_RESOLVER] Initialized (simplified)")

    def is_ambiguous(self, symbol: str) -> bool:
        """Check if symbol exists in both crypto and stock"""
        return self.cache.is_ambiguous(symbol)

    def get_disambiguation_options(self, symbol: str) -> List[Dict[str, Any]]:
        """Get options for an ambiguous symbol"""
        return self.cache.get_disambiguation_options(symbol)

    def get_disambiguation_message(
        self,
        symbol: str,
        language: str = "vi"
    ) -> str:
        """
        Build user-friendly disambiguation message.

        Args:
            symbol: The ambiguous symbol
            language: Response language (vi/en)

        Returns:
            Formatted message asking user to choose
        """
        options = self.get_disambiguation_options(symbol)

        if not options:
            return ""

        if language == "vi":
            lines = [f"Tôi thấy '{symbol}' có thể là:"]
            for i, opt in enumerate(options, 1):
                asset_type = opt.get("asset_class", "unknown")
                name = opt.get("name", symbol)
                exchange = opt.get("exchange", "")

                if asset_type == "crypto":
                    lines.append(f"  {i}. {name} (Cryptocurrency)")
                elif asset_type == "stock":
                    exchange_info = f" - {exchange}" if exchange else ""
                    lines.append(f"  {i}. {name} (Cổ phiếu{exchange_info})")
                else:
                    lines.append(f"  {i}. {name} ({asset_type})")

            lines.append("\nBạn muốn phân tích loại nào? Vui lòng trả lời crypto hoặc stock.")
        else:
            lines = [f"I found that '{symbol}' could be:"]
            for i, opt in enumerate(options, 1):
                asset_type = opt.get("asset_class", "unknown")
                name = opt.get("name", symbol)
                exchange = opt.get("exchange", "")

                if asset_type == "crypto":
                    lines.append(f"  {i}. {name} (Cryptocurrency)")
                elif asset_type == "stock":
                    exchange_info = f" - {exchange}" if exchange else ""
                    lines.append(f"  {i}. {name} (Stock{exchange_info})")
                else:
                    lines.append(f"  {i}. {name} ({asset_type})")

            lines.append("\nWhich one would you like to analyze? Please reply with crypto or stock.")

        return "\n".join(lines)

    def check_symbols(
        self,
        symbols: List[str],
        language: str = "vi"
    ) -> List[Dict[str, Any]]:
        """
        Check a list of symbols for ambiguity.

        Args:
            symbols: List of symbols from classification
            language: Response language

        Returns:
            List of ambiguous symbols with options and messages
        """
        ambiguous_results = []

        for symbol in symbols:
            if self.is_ambiguous(symbol):
                options = self.get_disambiguation_options(symbol)
                message = self.get_disambiguation_message(symbol, language)

                ambiguous_results.append({
                    "symbol": symbol,
                    "options": options,
                    "message": message,
                })

                self.logger.info(
                    f"[ASSET_RESOLVER] Symbol '{symbol}' is ambiguous: "
                    f"{len(options)} options"
                )

        return ambiguous_results


# ========================================
# SINGLETON ACCESSOR
# ========================================

_asset_resolver: Optional[AssetResolver] = None


def get_asset_resolver(
    symbol_cache: Optional[SymbolCacheService] = None
) -> AssetResolver:
    """Get singleton AssetResolver instance"""
    global _asset_resolver
    if _asset_resolver is None:
        _asset_resolver = AssetResolver(symbol_cache=symbol_cache)
    return _asset_resolver
