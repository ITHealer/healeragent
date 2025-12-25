# src/news_aggregator/services/digest_generator.py
"""
Digest Generator Service
Uses LLM to rank, summarize, and create news digests
"""

import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.news_aggregator.schemas.unified_news import UnifiedNewsItem
from src.news_aggregator.schemas.response import NewsDigest, TopStory

logger = logging.getLogger(__name__)


# Digest generation prompt template
DIGEST_SYSTEM_PROMPT = """You are a financial news analyst creating a concise news digest.

Your task is to:
1. Identify the most important and market-moving news
2. Rank articles by importance (1-10 scale)
3. Generate a brief overall summary
4. Identify key themes and market sentiment

Output format (JSON):
{
    "summary_text": "2-3 sentences summarizing today's key news",
    "market_sentiment": "bullish" | "bearish" | "neutral" | "mixed",
    "key_themes": ["theme1", "theme2", "theme3"],
    "top_stories": [
        {
            "rank": 1,
            "article_id": "id from input",
            "title": "article title",
            "summary": "1 sentence summary",
            "importance_reason": "why this is important"
        }
    ]
}

Guidelines:
- Be concise and actionable
- Focus on market-moving news
- Highlight earnings, M&A, regulatory changes, major price movements
- For crypto: focus on regulatory news, major protocol updates, whale movements
"""

DIGEST_USER_PROMPT_TEMPLATE = """Analyze these news articles and create a digest.
Target language for output: {target_language}

Articles to analyze:
{articles_json}

Generate the digest following the format specified. Output ONLY valid JSON, no markdown."""


class DigestGenerator:
    """
    Service to generate news digests using LLM.
    
    Features:
    - Importance ranking
    - Summary generation
    - Sentiment analysis
    - Theme extraction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._llm_provider = None
    
    def _get_llm_provider(self):
        """Lazy load LLM provider to avoid circular imports"""
        if self._llm_provider is None:
            try:
                from src.helpers.llm_helper import LLMGeneratorProvider
                self._llm_provider = LLMGeneratorProvider()
            except ImportError:
                self.logger.error("LLMGeneratorProvider not available")
                raise
        return self._llm_provider
    
    def _prepare_articles_for_llm(
        self,
        items: List[UnifiedNewsItem],
        max_articles: int = 20
    ) -> str:
        """
        Prepare articles for LLM input.
        
        Args:
            items: List of news items (already sorted by relevance)
            max_articles: Max articles to include
            
        Returns:
            JSON string of article summaries
        """
        # Take top N articles
        selected = items[:max_articles]
        
        articles_data = []
        for item in selected:
            articles_data.append({
                "id": item.id,
                "title": item.title,
                "content": (item.content or "")[:300],  # Truncate for token efficiency
                "source": item.source_site,
                "published_at": item.published_at.isoformat() if item.published_at else "",
                "symbols": item.symbols,
                "category": item.category.value,
            })
        
        return json.dumps(articles_data, indent=2, ensure_ascii=False)
    
    def _parse_llm_response(
        self,
        response: str,
        items: List[UnifiedNewsItem],
        target_language: str
    ) -> NewsDigest:
        """
        Parse LLM response into NewsDigest.
        
        Args:
            response: Raw LLM response (should be JSON)
            items: Original news items for reference
            target_language: Target language
            
        Returns:
            NewsDigest object
        """
        try:
            # Clean response (remove markdown code blocks if present)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            
            data = json.loads(cleaned)
            
            # Create item lookup for reference
            item_lookup = {item.id: item for item in items}
            
            # Parse top stories
            top_stories = []
            for story_data in data.get("top_stories", [])[:5]:
                article_id = story_data.get("article_id", "")
                original_item = item_lookup.get(article_id)
                
                top_stories.append(TopStory(
                    rank=story_data.get("rank", len(top_stories) + 1),
                    article_id=article_id,
                    title=story_data.get("title", original_item.title if original_item else ""),
                    summary=story_data.get("summary", ""),
                    url=original_item.url if original_item else "",
                    source=original_item.source_site if original_item else None,
                    importance_reason=story_data.get("importance_reason", ""),
                ))
            
            return NewsDigest(
                summary_text=data.get("summary_text", "No summary available."),
                top_stories=top_stories,
                market_sentiment=data.get("market_sentiment", "neutral"),
                key_themes=data.get("key_themes", []),
                articles_analyzed=len(items),
                target_language=target_language,
            )
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            # Return basic digest with error
            return NewsDigest(
                summary_text=response[:500] if response else "Failed to generate summary.",
                top_stories=[],
                market_sentiment="neutral",
                key_themes=[],
                articles_analyzed=len(items),
                target_language=target_language,
            )
        except Exception as e:
            self.logger.error(f"Error parsing digest response: {e}")
            return NewsDigest(
                summary_text="Error generating digest.",
                articles_analyzed=len(items),
                target_language=target_language,
            )
    
    async def generate_digest(
        self,
        items: List[UnifiedNewsItem],
        target_language: str = "en",
        model_name: str = "gpt-4.1-nano",
        provider_type: str = "openai",
        max_articles: int = 20,
    ) -> NewsDigest:
        """
        Generate a news digest from articles.
        
        Args:
            items: List of news items (should be pre-filtered and sorted)
            target_language: Output language
            model_name: LLM model to use
            provider_type: LLM provider (openai, ollama, gemini)
            max_articles: Max articles to analyze
            
        Returns:
            NewsDigest with summary, top stories, sentiment
        """
        if not items:
            return NewsDigest(
                summary_text="No articles available for analysis.",
                articles_analyzed=0,
                target_language=target_language,
            )
        
        self.logger.info(f"[Digest] Generating for {len(items)} articles, lang={target_language}")
        
        # Prepare articles JSON
        articles_json = self._prepare_articles_for_llm(items, max_articles)
        
        # Build prompt
        user_prompt = DIGEST_USER_PROMPT_TEMPLATE.format(
            target_language=target_language,
            articles_json=articles_json,
        )
        
        messages = [
            {"role": "system", "content": DIGEST_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            # Get LLM provider
            llm_provider = self._get_llm_provider()
            
            # Get API key
            try:
                from src.providers.provider_factory import ModelProviderFactory
                api_key = ModelProviderFactory._get_api_key(provider_type)
            except ImportError:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
            
            # Generate response
            response = await llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.3,  # Lower temperature for consistency
                enable_thinking=False,
            )
            
            # Extract content from response
            if isinstance(response, dict):
                content = response.get("content", "")
            else:
                content = str(response)
            
            # Parse and return digest
            digest = self._parse_llm_response(content, items, target_language)
            
            self.logger.info(f"[Digest] Generated: {len(digest.top_stories)} top stories")
            
            return digest
            
        except Exception as e:
            self.logger.error(f"[Digest] Error generating: {e}")
            return NewsDigest(
                summary_text=f"Error generating digest: {str(e)}",
                articles_analyzed=len(items),
                target_language=target_language,
            )
    
    async def rank_articles(
        self,
        items: List[UnifiedNewsItem],
        model_name: str = "gpt-4.1-nano",
        provider_type: str = "openai",
    ) -> List[UnifiedNewsItem]:
        """
        Use LLM to rank articles by importance.
        Updates item.importance_score for each article.
        
        This is a lighter-weight operation than full digest generation.
        """
        if not items or len(items) <= 5:
            # For small lists, just use relevance score
            for i, item in enumerate(sorted(items, key=lambda x: x.relevance_score, reverse=True)):
                item.importance_score = 10.0 - (i * 0.5)  # Simple ranking
            return items
        
        # For larger lists, use LLM
        # This could be implemented with a simpler prompt focused only on ranking
        # For now, generate full digest and extract rankings
        
        digest = await self.generate_digest(
            items,
            target_language="en",
            model_name=model_name,
            provider_type=provider_type,
        )
        
        # Apply rankings from digest to items
        for story in digest.top_stories:
            for item in items:
                if item.id == story.article_id:
                    item.importance_score = 10.0 - (story.rank - 1) * 1.0
                    break
        
        return items