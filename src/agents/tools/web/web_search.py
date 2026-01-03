"""
Web Search Tool

Provides web search capability using Tavily API.
LLM can use this tool when FMP data is insufficient to answer user queries.

Features:
- Async execution (non-blocking)
- Smart search optimization for finance queries
- Redis caching to reduce API calls
- Rate limiting protection
- Structured output for LLM consumption

Usage:
    from src.agents.tools.web import WebSearchTool

    tool = WebSearchTool()
    result = await tool.execute(
        query="What is the latest news about NVIDIA earnings?",
        max_results=5
    )
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output,
)

# Optional Redis import - gracefully handle if not available
try:
    from src.helpers.redis_cache import get_redis_client_llm
    REDIS_AVAILABLE = True
except (ImportError, TypeError) as e:
    REDIS_AVAILABLE = False
    get_redis_client_llm = None
    logging.warning(f"[WebSearchTool] Redis not available, caching disabled: {e}")


# Thread pool for running sync Tavily client
_web_search_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="web_search_")


class WebSearchTool(BaseTool):
    """
    Web Search Tool using Tavily API.

    Category: web
    Data Source: Tavily Search API
    Cache: Redis with 5-minute TTL

    Use this tool when:
    - FMP data is not available or outdated
    - User asks about recent events not covered by financial data
    - Need to find specific information not in financial databases
    - Cross-referencing financial data with news/events

    Do NOT use this tool when:
    - FMP tools can provide the data (price, financials, etc.)
    - Query is purely about historical stock data
    - User asks about calculation or technical analysis
    """

    CACHE_TTL = 300  # 5 minutes - shorter TTL for web search
    MAX_QUERY_LENGTH = 500
    DEFAULT_MAX_RESULTS = 5

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()

        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            logging.warning(
                "[WebSearchTool] TAVILY_API_KEY not set. "
                "Web search will be disabled."
            )

        self.logger = logging.getLogger(__name__)
        self._client = None  # Lazy initialization

        # Define tool schema
        self.schema = ToolSchema(
            name="webSearch",
            category="web",
            description=(
                "Search the web for current information. SECONDARY tool - use only when "
                "Financial tools cannot provide sufficient information. "
                "MUST use for: breaking news, post-cutoff events, regulatory changes, "
                "NEW trading concepts/strategies (SMC, ICT, Order Blocks, FVG, BOS/CHoCH), "
                "unfamiliar terms, verification needs. "
                "SKIP for: prices, financials, technicals, well-known indicators (use FMP). "
                "Anti-hallucination: When uncertain about terms or facts, search first."
            ),
            capabilities=[
                "Search real-time web content",
                "Find recent news and events",
                "Access post-cutoff knowledge (new policies, products, events)",
                "Research new trading strategies and methodologies (SMC, ICT, Price Action concepts)",
                "Verify uncertain or time-sensitive information",
                "Get context for regulatory changes and executive statements",
                "Explain unfamiliar trading terms and modern technical concepts",
            ],
            limitations=[
                "SECONDARY to FMP tools - do not use for prices, financials, technicals",
                "Results limited to 3-5 for optimal context",
                "Cannot access paywalled content",
                "Prefer official sources (company sites, regulators, educational platforms)",
                "Rate limited - use judiciously",
            ],
            usage_hints=[
                "MUST search: new trading terms (SMC, ICT, Order Blocks, FVG, BOS, CHoCH)",
                "MUST search: time-sensitive info, post-cutoff knowledge, verification needs",
                "SKIP search: FMP tools can answer, classic indicators (RSI, MACD, MA)",
                "Query tips: 1-6 words, include 'trading' or 'strategy' for better results",
                "When uncertain about any trading concept → search first, never guess",
            ],
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "Search query. Be specific and include relevant context. "
                        "Examples: "
                        "'SMC Smart Money Concepts trading', "
                        "'ICT Order Blocks strategy', "
                        "'Fair Value Gap FVG trading', "
                        "'Break of Structure BOS CHoCH', "
                        "'Fed rate decision', "
                        "'crypto regulation Vietnam nowadays'"
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of results to return (default: 5, max: 10)",
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=10,
                ),
                ToolParameter(
                    name="search_depth",
                    type="string",
                    description="Search depth: 'basic' for quick search, 'advanced' for thorough search",
                    required=False,
                    default="advanced",
                    enum=["basic", "advanced"],
                ),
                ToolParameter(
                    name="time_range",
                    type="string",
                    description="Time range for results: 'day', 'week', 'month', 'year'",
                    required=False,
                    default="week",
                    enum=["day", "week", "month", "year"],
                ),
            ],
            returns={
                "query": "string - The search query used",
                "result_count": "number - Number of results returned",
                "results": "array - List of {title, url, content, score, published_date}",
                "search_time_ms": "number - Search execution time",
            },
            requires_symbol=False,
        )

    def _get_client(self):
        """Get or create Tavily client (lazy initialization)"""
        if self._client is None and self.api_key:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=self.api_key)
            except ImportError:
                self.logger.error("[WebSearchTool] tavily package not installed")
                raise ImportError("tavily package is required for web search")
        return self._client

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "advanced",
        time_range: str = "week",
        **kwargs,
    ) -> ToolOutput:
        """
        Execute web search.

        Args:
            query: Search query string
            max_results: Maximum number of results (1-10)
            search_depth: "basic" or "advanced"
            time_range: "day", "week", "month", "year"

        Returns:
            ToolOutput with search results
        """
        start_time = datetime.now()

        # Validate API key
        if not self.api_key:
            return create_error_output(
                tool_name="webSearch",
                error="Web search is not configured. TAVILY_API_KEY is missing.",
                metadata={"query": query[:100]},
            )

        # Validate and sanitize query
        query = query.strip()
        if not query:
            return create_error_output(
                tool_name="webSearch",
                error="Search query cannot be empty",
                metadata={},
            )

        if len(query) > self.MAX_QUERY_LENGTH:
            query = query[:self.MAX_QUERY_LENGTH]

        # Validate parameters
        max_results = min(max(1, max_results), 10)
        if search_depth not in ["basic", "advanced"]:
            search_depth = "advanced"
        if time_range not in ["day", "week", "month", "year"]:
            time_range = "week"

        self.logger.info(
            f"[webSearch] Executing: query='{query[:50]}...' "
            f"max_results={max_results} depth={search_depth} time={time_range}"
        )

        try:
            # Build cache key
            cache_key = f"webSearch:{hash(query)}:{max_results}:{search_depth}:{time_range}"

            # Try cache first
            cached_data = await self._get_cached_result(cache_key)
            if cached_data:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.info(f"[webSearch] CACHE HIT ({int(execution_time)}ms)")

                return create_success_output(
                    tool_name="webSearch",
                    data=cached_data,
                    metadata={
                        "query": query[:100],
                        "execution_time_ms": int(execution_time),
                        "from_cache": True,
                    },
                )

            # Execute search in thread pool (Tavily client is synchronous)
            search_result = await self._execute_search_async(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                time_range=time_range,
            )

            if search_result is None:
                return create_error_output(
                    tool_name="webSearch",
                    error="Search failed. No results returned.",
                    metadata={"query": query[:100]},
                )

            # Format results
            result_data = self._format_results(
                search_result,
                query,
                max_results,
            )

            # Cache result
            await self._cache_result(cache_key, result_data)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(
                f"[webSearch] SUCCESS ({int(execution_time)}ms) - "
                f"{result_data['result_count']} results"
            )

            return create_success_output(
                tool_name="webSearch",
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
            self.logger.error(f"[webSearch] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name="webSearch",
                error=f"Search error: {str(e)}",
                metadata={"query": query[:100]},
            )

    async def _execute_search_async(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        time_range: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Execute Tavily search in thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _web_search_executor,
                    self._sync_search,
                    query,
                    max_results,
                    search_depth,
                    time_range,
                ),
                timeout=30.0,  # 30 second timeout
            )
            return result
        except asyncio.TimeoutError:
            self.logger.error("[webSearch] Search timed out after 30s")
            return None
        except Exception as e:
            self.logger.error(f"[webSearch] Async execution error: {e}")
            return None

    def _sync_search(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        time_range: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronous Tavily search (runs in thread pool).
        """
        try:
            client = self._get_client()
            if not client:
                return None

            # Build search parameters
            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "time_range": time_range,
                "topic": "finance",  # Optimize for finance queries
                "exclude_domains": [
                    "reddit.com",
                    "twitter.com",
                    "x.com",
                    "facebook.com",
                ],
            }

            response = client.search(**search_params)
            return response if isinstance(response, dict) else None

        except Exception as e:
            self.logger.error(f"[webSearch] Tavily API error: {e}")
            return None

    def _format_results(
        self,
        raw_response: Dict[str, Any],
        query: str,
        max_results: int,
    ) -> Dict[str, Any]:
        """
        Format Tavily response to structured output.
        """
        results = raw_response.get("results", [])

        formatted_results = []
        for item in results[:max_results]:
            formatted_results.append({
                "title": item.get("title", "")[:200],
                "url": item.get("url", ""),
                "content": item.get("content", "")[:500],
                "score": round(float(item.get("score", 0)), 3),
                "published_date": item.get("published_date"),
            })

        return {
            "query": query,
            "result_count": len(formatted_results),
            "results": formatted_results,
            "search_time_ms": raw_response.get("response_time", 0),
            "timestamp": datetime.now().isoformat(),
        }

    def _create_llm_context(self, result_data: Dict[str, Any]) -> str:
        """
        Create formatted context string for LLM consumption.
        """
        if not result_data.get("results"):
            return f"Web search for '{result_data['query']}' returned no results."

        lines = [
            f"Web search results for: {result_data['query']}",
            f"Found {result_data['result_count']} results:",
            "",
        ]

        for i, result in enumerate(result_data["results"], 1):
            lines.append(f"{i}. {result['title']}")
            if result.get("content"):
                # Truncate content for LLM context
                content = result["content"][:300]
                if len(result["content"]) > 300:
                    content += "..."
                lines.append(f"   {content}")
            lines.append(f"   Source: {result['url']}")
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
            self.logger.warning(f"[webSearch] Cache read error: {e}")
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
                self.logger.debug(f"[webSearch] Cached: {cache_key}")
        except Exception as e:
            self.logger.warning(f"[webSearch] Cache write error: {e}")


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("TAVILY_API_KEY not set")
            return

        tool = WebSearchTool(api_key=api_key)

        print("\n" + "=" * 60)
        print("Testing WebSearchTool")
        print("=" * 60)

        # Test: Basic search
        print("\nTest: NVIDIA earnings search")
        result = await tool.execute(
            query="NVIDIA Q4 2024 earnings results",
            max_results=3,
        )

        if result.status == "success":
            data = result.data
            print(f"Success: {data['result_count']} results")
            for i, r in enumerate(data["results"], 1):
                print(f"\n{i}. {r['title'][:80]}")
                print(f"   Score: {r['score']}")
        else:
            print(f"Error: {result.error}")

    asyncio.run(test())