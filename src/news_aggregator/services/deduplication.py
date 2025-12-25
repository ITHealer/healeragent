# src/news_aggregator/services/deduplication.py
"""
Deduplication Service
Removes duplicate news articles based on URL hash and title similarity
"""

import logging
from typing import List, Set, Dict, Optional
from datetime import datetime, timedelta

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

from src.news_aggregator.schemas.unified_news import UnifiedNewsItem

logger = logging.getLogger(__name__)


class DeduplicationService:
    """
    Service to remove duplicate news articles.
    
    Deduplication strategies:
    1. URL Hash - Fast, exact match
    2. Title Similarity - Fuzzy match for same story from different sources
    """
    
    def __init__(
        self,
        title_similarity_threshold: float = 0.85,
        time_window_hours: int = 48,
    ):
        """
        Initialize deduplication service.
        
        Args:
            title_similarity_threshold: Similarity score (0-1) to consider titles as duplicates
            time_window_hours: Only compare articles within this time window
        """
        self.title_threshold = title_similarity_threshold
        self.time_window = timedelta(hours=time_window_hours)
        
        # Cache for seen URL hashes (in-memory, resets per request)
        self._seen_url_hashes: Set[str] = set()
        
        if not RAPIDFUZZ_AVAILABLE:
            logger.warning("rapidfuzz not installed, falling back to basic title matching")
    
    def reset(self):
        """Reset internal caches (call before each aggregation)"""
        self._seen_url_hashes.clear()
    
    def _is_url_duplicate(self, item: UnifiedNewsItem) -> bool:
        """Check if URL is already seen"""
        if not item.url_hash:
            return False
        
        if item.url_hash in self._seen_url_hashes:
            return True
        
        self._seen_url_hashes.add(item.url_hash)
        return False
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """
        Calculate similarity between two titles.
        
        Uses rapidfuzz if available, otherwise basic matching.
        """
        if not title1 or not title2:
            return 0.0
        
        # Normalize titles
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()
        
        # Exact match
        if t1 == t2:
            return 1.0
        
        if RAPIDFUZZ_AVAILABLE:
            # Use token_sort_ratio for better matching of reordered words
            score = fuzz.token_sort_ratio(t1, t2) / 100.0
            return score
        else:
            # Basic: check word overlap
            words1 = set(t1.split())
            words2 = set(t2.split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1 & words2
            union = words1 | words2
            
            return len(intersection) / len(union) if union else 0.0
    
    def _is_title_duplicate(
        self,
        item: UnifiedNewsItem,
        existing_items: List[UnifiedNewsItem]
    ) -> Optional[UnifiedNewsItem]:
        """
        Check if title is similar to any existing item.
        
        Returns the matching item if found, None otherwise.
        """
        if not item.title_normalized:
            return None
        
        # Only compare items within time window
        item_time = item.published_at
        
        for existing in existing_items:
            # Skip if outside time window
            time_diff = abs((existing.published_at - item_time).total_seconds())
            if time_diff > self.time_window.total_seconds():
                continue
            
            similarity = self._calculate_title_similarity(
                item.title_normalized,
                existing.title_normalized or ""
            )
            
            if similarity >= self.title_threshold:
                return existing
        
        return None
    
    def deduplicate(
        self,
        items: List[UnifiedNewsItem],
        preserve_higher_priority: bool = True
    ) -> List[UnifiedNewsItem]:
        """
        Remove duplicates from a list of news items.
        
        Args:
            items: List of UnifiedNewsItem to deduplicate
            preserve_higher_priority: When duplicates found, keep the one
                                     from higher priority provider (lower number)
            
        Returns:
            Deduplicated list of UnifiedNewsItem
        """
        if not items:
            return []
        
        self.reset()
        
        # Sort by priority (lower = higher priority)
        sorted_items = sorted(items, key=lambda x: (
            # Provider priority
            1 if x.provider.value == "fmp" else 2,
            # Then by published date (newer first)
            -x.published_at.timestamp() if x.published_at else 0
        ))
        
        unique_items: List[UnifiedNewsItem] = []
        url_duplicates = 0
        title_duplicates = 0
        
        for item in sorted_items:
            # Check URL duplicate first (fast)
            if self._is_url_duplicate(item):
                url_duplicates += 1
                continue
            
            # Check title similarity (slower)
            existing_match = self._is_title_duplicate(item, unique_items)
            if existing_match:
                title_duplicates += 1
                
                if preserve_higher_priority:
                    # Already sorted by priority, so existing is already the better one
                    continue
                else:
                    # Replace with newer one
                    unique_items.remove(existing_match)
                    unique_items.append(item)
                    continue
            
            # Not a duplicate
            unique_items.append(item)
        
        logger.info(
            f"[Dedup] Input: {len(items)}, Output: {len(unique_items)}, "
            f"URL dups: {url_duplicates}, Title dups: {title_duplicates}"
        )
        
        return unique_items
    
    def deduplicate_by_url_only(self, items: List[UnifiedNewsItem]) -> List[UnifiedNewsItem]:
        """
        Fast deduplication using only URL hash.
        Use when performance is critical.
        """
        if not items:
            return []
        
        seen_hashes: Set[str] = set()
        unique_items = []
        
        for item in items:
            if item.url_hash and item.url_hash not in seen_hashes:
                seen_hashes.add(item.url_hash)
                unique_items.append(item)
        
        logger.info(f"[Dedup-URL] Input: {len(items)}, Output: {len(unique_items)}")
        
        return unique_items