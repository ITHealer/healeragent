"""
SerpAPI Web Search Tool

Provides web search capability using SerpAPI (Google Search API).
Alternative to Tavily for more diverse search results.

Features:
- Async execution (non-blocking)
- Google search with knowledge graph support
- Answer box extraction for quick answers
- Redis caching to reduce API calls
- Rate limiting protection

Usage:
    from src.agents.tools.web import SerpSearchTool

    tool = SerpSearchTool()
    result = await tool.execute(
        query="NVIDIA stock analysis 2025",
        max_results=5
    )
"""

import os
import json
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output,
)

# Optional Redis import
try:
    from src.helpers.redis_cache import get_redis_client_llm
    REDIS_AVAILABLE = True
except (ImportError, TypeError) as e:
    REDIS_AVAILABLE = False
    get_redis_client_llm = None


class SerpSearchTool(BaseTool):
    """
    Web Search Tool using SerpAPI (Google Search).

    Category: web
    Data Source: SerpAPI (Google Search)
    Cache: Redis with 5-minute TTL

    Complements Tavily search with Google's knowledge graph and answer boxes.
    Use for fact-checking and getting authoritative sources.
    """

    CACHE_TTL = 300  # 5 minutes
    MAX_QUERY_LENGTH = 500
    DEFAULT_MAX_RESULTS = 5
    API_URL = "https://serpapi.com/search"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()

        self.api_key = api_key or os.environ.get("SERPAPI_KEY")
        if not self.api_key:
            logging.warning(
                "[SerpSearchTool] SERPAPI_KEY not set. "
                "SerpAPI search will be disabled."
            )

        self.logger = logging.getLogger(__name__)
        self._session: Optional[aiohttp.ClientSession] = None

        # Default search parameters
        self.default_params = {
            "engine": "google",
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
        }

        # Define tool schema
        self.schema = ToolSchema(
            name="serpSearch",
            category="web",
            description=(
                "Search Google via SerpAPI for authoritative information. "
                "Returns knowledge graph, answer boxes, and organic results. "
                "Use for: fact verification, official sources, comprehensive research. "
                "Complements Tavily search for diverse results."
            ),
            capabilities=[
                "Google search with knowledge graph",
                "Answer box extraction for direct answers",
                "Find authoritative and official sources",
                "News and recent articles",
                "Fact verification from multiple sources",
            ],
            limitations=[
                "Rate limited - use judiciously",
                "Results may overlap with Tavily",
                "Cannot access paywalled content",
            ],
            usage_hints=[
                "Use for official/authoritative sources",
                "Good for fact-checking and verification",
                "Combine with Tavily for comprehensive coverage",
            ],
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "Search query. Be specific and include context. "
                        "Examples: 'NVIDIA earnings Q4 2024', 'Fed interest rate decision 2025'"
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results (default: 5, max: 10)",
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=10,
                ),
                ToolParameter(
                    name="search_type",
                    type="string",
                    description="Search type: 'general', 'news', 'finance'",
                    required=False,
                    default="general",
                    enum=["general", "news", "finance"],
                ),
            ],
            returns={
                "query": "string - The search query used",
                "result_count": "number - Number of results",
                "answer_box": "object - Direct answer if available",
                "knowledge_graph": "object - Knowledge graph data if available",
                "results": "array - List of search results",
            },
            requires_symbol=False,
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        search_type: str = "general",
        **kwargs,
    ) -> ToolOutput:
        """
        Execute SerpAPI search.

        Args:
            query: Search query string
            max_results: Maximum number of results (1-10)
            search_type: "general", "news", or "finance"

        Returns:
            ToolOutput with search results
        """
        start_time = datetime.now()

        # Validate API key
        if not self.api_key:
            return create_error_output(
                tool_name="serpSearch",
                error="SerpAPI search not configured. SERPAPI_KEY missing.",
                metadata={"query": query[:100]},
            )

        # Validate query
        query = query.strip()
        if not query:
            return create_error_output(
                tool_name="serpSearch",
                error="Search query cannot be empty",
                metadata={},
            )

        if len(query) > self.MAX_QUERY_LENGTH:
            query = query[:self.MAX_QUERY_LENGTH]

        # Validate parameters
        max_results = min(max(1, max_results), 10)
        if search_type not in ["general", "news", "finance"]:
            search_type = "general"

        self.logger.info(
            f"[serpSearch] Executing: query='{query[:50]}...' "
            f"max_results={max_results} type={search_type}"
        )

        try:
            # Build cache key
            cache_key = f"serpSearch:{hash(query)}:{max_results}:{search_type}"

            # Try cache first
            cached_data = await self._get_cached_result(cache_key)
            if cached_data:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.info(f"[serpSearch] CACHE HIT ({int(execution_time)}ms)")

                return create_success_output(
                    tool_name="serpSearch",
                    data=cached_data,
                    metadata={
                        "query": query[:100],
                        "execution_time_ms": int(execution_time),
                        "from_cache": True,
                    },
                )

            # Execute search
            search_result = await self._execute_search(
                query=query,
                max_results=max_results,
                search_type=search_type,
            )

            if search_result is None:
                return create_error_output(
                    tool_name="serpSearch",
                    error="Search failed. No results returned.",
                    metadata={"query": query[:100]},
                )

            # Format results
            result_data = self._format_results(search_result, query, max_results)

            # Cache result
            await self._cache_result(cache_key, result_data)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(
                f"[serpSearch] SUCCESS ({int(execution_time)}ms) - "
                f"{result_data['result_count']} results"
            )

            return create_success_output(
                tool_name="serpSearch",
                data=result_data,
                formatted_context=self._create_llm_context(result_data),
                metadata={
                    "query": query[:100],
                    "execution_time_ms": int(execution_time),
                    "result_count": result_data["result_count"],
                    "from_cache": False,
                },
            )

        except Exception as e:
            self.logger.error(f"[serpSearch] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name="serpSearch",
                error=f"Search error: {str(e)}",
                metadata={"query": query[:100]},
            )

    async def _execute_search(
        self,
        query: str,
        max_results: int,
        search_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Execute SerpAPI search."""
        try:
            session = await self._get_session()

            # Build parameters
            params = {
                **self.default_params,
                "api_key": self.api_key,
                "q": query,
                "num": max_results,
                "output": "json",
            }

            # Add search type specific params
            if search_type == "news":
                params["tbm"] = "nws"
            elif search_type == "finance":
                # Add finance-related terms
                if "stock" not in query.lower() and "price" not in query.lower():
                    params["q"] = f"{query} stock finance"

            async with session.get(self.API_URL, params=params) as response:
                response.raise_for_status()
                return await response.json()

        except asyncio.TimeoutError:
            self.logger.error("[serpSearch] Request timed out")
            return None
        except aiohttp.ClientError as e:
            self.logger.error(f"[serpSearch] HTTP error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"[serpSearch] Unexpected error: {e}")
            return None

    def _format_results(
        self,
        raw_response: Dict[str, Any],
        query: str,
        max_results: int,
    ) -> Dict[str, Any]:
        """Format SerpAPI response to structured output."""
        result = {
            "query": query,
            "result_count": 0,
            "answer_box": None,
            "knowledge_graph": None,
            "results": [],
            "timestamp": datetime.now().isoformat(),
        }

        # Extract answer box (direct answer)
        if "answer_box" in raw_response:
            ab = raw_response["answer_box"]
            result["answer_box"] = {
                "type": ab.get("type", "unknown"),
                "answer": ab.get("answer") or ab.get("snippet") or ab.get("result"),
                "title": ab.get("title"),
                "link": ab.get("link"),
            }

        # Extract knowledge graph
        if "knowledge_graph" in raw_response:
            kg = raw_response["knowledge_graph"]
            result["knowledge_graph"] = {
                "title": kg.get("title"),
                "type": kg.get("type"),
                "description": kg.get("description"),
                "source": kg.get("source", {}).get("name"),
                "facts": {
                    k: v for k, v in kg.items()
                    if k not in ["title", "type", "description", "source", "images"]
                    and isinstance(v, (str, int, float))
                },
            }

        # Extract organic results
        organic_results = raw_response.get("organic_results", [])
        for item in organic_results[:max_results]:
            result["results"].append({
                "title": item.get("title", "")[:200],
                "url": item.get("link", ""),
                "content": item.get("snippet", "")[:500],
                "position": item.get("position"),
                "date": item.get("date"),
            })

        # Extract news results if available
        news_results = raw_response.get("news_results", [])
        for item in news_results[:max_results]:
            if len(result["results"]) >= max_results:
                break
            result["results"].append({
                "title": item.get("title", "")[:200],
                "url": item.get("link", ""),
                "content": item.get("snippet", "")[:500],
                "source": item.get("source"),
                "date": item.get("date"),
                "is_news": True,
            })

        result["result_count"] = len(result["results"])
        return result

    def _create_llm_context(self, result_data: Dict[str, Any]) -> str:
        """Create formatted context string for LLM consumption."""
        lines = [
            f"Google search results for: {result_data['query']}",
            "",
        ]

        # Add answer box if available
        if result_data.get("answer_box") and result_data["answer_box"].get("answer"):
            ab = result_data["answer_box"]
            lines.append("📋 DIRECT ANSWER:")
            lines.append(f"   {ab['answer']}")
            if ab.get("link"):
                lines.append(f"   Source: {ab['link']}")
            lines.append("")

        # Add knowledge graph if available
        if result_data.get("knowledge_graph"):
            kg = result_data["knowledge_graph"]
            if kg.get("title") or kg.get("description"):
                lines.append("📊 KNOWLEDGE GRAPH:")
                if kg.get("title"):
                    lines.append(f"   {kg['title']} ({kg.get('type', 'Entity')})")
                if kg.get("description"):
                    lines.append(f"   {kg['description'][:300]}")
                if kg.get("facts"):
                    for k, v in list(kg["facts"].items())[:5]:
                        lines.append(f"   • {k}: {v}")
                lines.append("")

        # Add search results
        if result_data.get("results"):
            lines.append(f"🔍 SEARCH RESULTS ({result_data['result_count']}):")
            lines.append("")

            for i, r in enumerate(result_data["results"], 1):
                is_news = r.get("is_news", False)
                prefix = "📰" if is_news else f"{i}."
                lines.append(f"{prefix} {r['title']}")
                if r.get("content"):
                    content = r["content"][:250]
                    if len(r["content"]) > 250:
                        content += "..."
                    lines.append(f"   {content}")
                if r.get("date"):
                    lines.append(f"   📅 {r['date']}")
                if r.get("source"):
                    lines.append(f"   Source: {r['source']}")
                lines.append(f"   🔗 {r['url']}")
                lines.append("")

        return "\n".join(lines)

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get result from Redis cache."""
        if not REDIS_AVAILABLE or get_redis_client_llm is None:
            return None

        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                cached_bytes = await redis_client.get(cache_key)
                if cached_bytes:
                    return json.loads(cached_bytes.decode("utf-8"))
        except Exception as e:
            self.logger.warning(f"[serpSearch] Cache read error: {e}")
        return None

    async def _cache_result(self, cache_key: str, data: Dict[str, Any]) -> None:
        """Cache result to Redis."""
        if not REDIS_AVAILABLE or get_redis_client_llm is None:
            return

        try:
            redis_client = await get_redis_client_llm()
            if redis_client:
                json_string = json.dumps(data)
                await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
        except Exception as e:
            self.logger.warning(f"[serpSearch] Cache write error: {e}")

    async def close(self):
        """Close aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            print("SERPAPI_KEY not set")
            return

        tool = SerpSearchTool(api_key=api_key)

        print("\n" + "=" * 60)
        print("Testing SerpSearchTool")
        print("=" * 60)

        result = await tool.execute(
            query="NVIDIA stock price 2025",
            max_results=5,
        )

        if result.status == "success":
            data = result.data
            print(f"Success: {data['result_count']} results")
            if data.get("answer_box"):
                print(f"\nAnswer Box: {data['answer_box']}")
            if data.get("knowledge_graph"):
                print(f"\nKnowledge Graph: {data['knowledge_graph']}")
        else:
            print(f"Error: {result.error}")

        await tool.close()

    asyncio.run(test())
