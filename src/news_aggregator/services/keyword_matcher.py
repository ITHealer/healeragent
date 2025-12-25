# src/news_aggregator/services/keyword_matcher.py
"""
Keyword Matcher Service
Matches news articles against user-provided keywords and symbols
"""

import re
import logging
from typing import List, Set, Tuple
from collections import defaultdict

from src.news_aggregator.schemas.unified_news import UnifiedNewsItem

logger = logging.getLogger(__name__)


class KeywordMatcher:
    """
    Service to match news articles against keywords and symbols.
    
    Features:
    - Case-insensitive matching
    - Whole word matching for symbols
    - Fuzzy matching for company names
    - Relevance scoring
    """
    
    # Common stock suffixes to handle
    SYMBOL_SUFFIXES = [".US", ".NYSE", ".NASDAQ", "-USD", "USD"]
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for comparison"""
        s = symbol.upper().strip()
        
        # Remove common suffixes
        for suffix in self.SYMBOL_SUFFIXES:
            if s.endswith(suffix):
                s = s[:-len(suffix)]
        
        return s
    
    def _extract_symbols_from_text(self, text: str) -> Set[str]:
        """
        Extract potential stock symbols from text.
        Looks for uppercase sequences that could be tickers.
        """
        if not text:
            return set()
        
        # Pattern for stock symbols (2-5 uppercase letters)
        pattern = r'\b[A-Z]{2,5}\b'
        
        matches = re.findall(pattern, text)
        
        # Filter out common words that look like tickers
        common_words = {
            "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
            "CAN", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS",
            "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE", "WAY",
            "WHO", "BOY", "DID", "GET", "HIM", "LET", "PUT", "SAY",
            "SHE", "TOO", "USE", "CEO", "CFO", "IPO", "SEC", "USD",
            "ETF", "GDP", "API", "USA", "NYSE", "NASDAQ"
        }
        
        return {m for m in matches if m not in common_words}
    
    def _calculate_relevance_score(
        self,
        item: UnifiedNewsItem,
        keywords: List[str],
        symbols: List[str],
        matched_in_title: int,
        matched_in_content: int,
        matched_symbols: int
    ) -> float:
        """
        Calculate relevance score (0-1) based on match quality.
        
        Scoring weights:
        - Symbol match in title: 0.4
        - Keyword match in title: 0.3
        - Symbol match in item.symbols: 0.2
        - Keyword match in content: 0.1
        """
        if not keywords and not symbols:
            return 1.0  # No filters = everything is relevant
        
        total_filters = len(keywords) + len(symbols)
        if total_filters == 0:
            return 1.0
        
        # Calculate weighted score
        score = 0.0
        
        # Title matches are most important
        if matched_in_title > 0:
            score += 0.4 * min(matched_in_title / total_filters, 1.0)
        
        # Symbol matches
        if matched_symbols > 0:
            score += 0.3 * min(matched_symbols / max(len(symbols), 1), 1.0)
        
        # Content matches
        if matched_in_content > 0:
            score += 0.2 * min(matched_in_content / total_filters, 1.0)
        
        # Existing item.symbols match
        if item.symbols and symbols:
            item_symbols_normalized = {self._normalize_symbol(s) for s in item.symbols}
            filter_symbols_normalized = {self._normalize_symbol(s) for s in symbols}
            symbol_overlap = len(item_symbols_normalized & filter_symbols_normalized)
            if symbol_overlap > 0:
                score += 0.1 * min(symbol_overlap / len(filter_symbols_normalized), 1.0)
        
        return min(score, 1.0)
    
    def match_item(
        self,
        item: UnifiedNewsItem,
        keywords: List[str],
        symbols: List[str]
    ) -> Tuple[bool, List[str], float]:
        """
        Check if a news item matches the given keywords and symbols.
        
        Args:
            item: News item to check
            keywords: Keywords to match (company names, topics, etc.)
            symbols: Stock/crypto symbols to match
            
        Returns:
            Tuple of (is_match, matched_keywords, relevance_score)
        """
        # If no filters provided, match everything
        if not keywords and not symbols:
            return True, [], 1.0
        
        matched_keywords: List[str] = []
        matched_in_title = 0
        matched_in_content = 0
        matched_symbols = 0
        
        # Prepare search text
        title_lower = item.title.lower() if item.title else ""
        content_lower = (item.content or "").lower()
        search_text = f"{title_lower} {content_lower}"
        
        # Check keywords
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            if keyword_lower in title_lower:
                matched_keywords.append(keyword)
                matched_in_title += 1
            elif keyword_lower in content_lower:
                matched_keywords.append(keyword)
                matched_in_content += 1
        
        # Check symbols (whole word match for precision)
        for symbol in symbols:
            symbol_normalized = self._normalize_symbol(symbol)
            
            # Check in title (case insensitive)
            title_upper = item.title.upper() if item.title else ""
            if re.search(rf'\b{re.escape(symbol_normalized)}\b', title_upper):
                if symbol not in matched_keywords:
                    matched_keywords.append(symbol)
                matched_in_title += 1
                matched_symbols += 1
                continue
            
            # Check in item.symbols
            item_symbols_normalized = {self._normalize_symbol(s) for s in item.symbols}
            if symbol_normalized in item_symbols_normalized:
                if symbol not in matched_keywords:
                    matched_keywords.append(symbol)
                matched_symbols += 1
                continue
            
            # Check in content
            content_upper = (item.content or "").upper()
            if re.search(rf'\b{re.escape(symbol_normalized)}\b', content_upper):
                if symbol not in matched_keywords:
                    matched_keywords.append(symbol)
                matched_in_content += 1
                matched_symbols += 1
        
        # Calculate if it's a match
        is_match = len(matched_keywords) > 0
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(
            item, keywords, symbols,
            matched_in_title, matched_in_content, matched_symbols
        )
        
        return is_match, matched_keywords, relevance_score
    
    def filter_and_score(
        self,
        items: List[UnifiedNewsItem],
        keywords: List[str],
        symbols: List[str]
    ) -> List[Tuple[UnifiedNewsItem, List[str]]]:
        """
        Filter items by keywords/symbols and add relevance scores.
        
        Args:
            items: List of news items
            keywords: Keywords to match
            symbols: Symbols to match
            
        Returns:
            List of (item, matched_keywords) tuples, sorted by relevance
        """
        if not items:
            return []
        
        # If no filters, return all with default score
        if not keywords and not symbols:
            for item in items:
                item.relevance_score = 1.0
            return [(item, []) for item in items]
        
        results: List[Tuple[UnifiedNewsItem, List[str]]] = []
        
        for item in items:
            is_match, matched, score = self.match_item(item, keywords, symbols)
            
            if is_match:
                item.relevance_score = score
                results.append((item, matched))
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x[0].relevance_score, reverse=True)
        
        logger.info(
            f"[Matcher] Input: {len(items)}, Matched: {len(results)}, "
            f"Keywords: {keywords}, Symbols: {symbols}"
        )
        
        return results
    
    def extract_and_tag_symbols(self, items: List[UnifiedNewsItem]) -> List[UnifiedNewsItem]:
        """
        Extract potential symbols from text and add to item.symbols.
        Useful for Tavily results that don't have structured symbol data.
        """
        for item in items:
            if item.provider.value == "tavily" and not item.symbols:
                # Try to extract symbols from title
                potential_symbols = self._extract_symbols_from_text(item.title)
                
                # Add to item.symbols (limit to 3 to avoid noise)
                item.symbols.extend(list(potential_symbols)[:3])
        
        return items