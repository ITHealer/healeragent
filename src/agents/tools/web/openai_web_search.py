"""
OpenAI Native Web Search Tool

Uses OpenAI's Responses API with native web_search capability.
This is the PRIMARY web search tool - much more powerful than Tavily/SerpAPI.

Features:
- Native OpenAI web search with citations
- Knowledge graph and answer box support
- Domain filtering for authoritative sources
- User location support for localized results
- Async streaming support
- Source tracking and URL citations

Configuration (.env):
- WEB_SEARCH_MODEL: Model for web search (default: gpt-5-mini)
- WEB_SEARCH_ALLOWED_DOMAINS: Comma-separated list of allowed domains (optional)
- OPENAI_API_KEY: OpenAI API key (required)

Usage:
    from src.agents.tools.web import OpenAIWebSearchTool

    tool = OpenAIWebSearchTool()
    result = await tool.execute(
        query="Latest NVIDIA earnings news",
        max_results=5
    )
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, AsyncGenerator
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

# Optional Redis import
try:
    from src.helpers.redis_cache import get_redis_client_llm
    REDIS_AVAILABLE = True
except (ImportError, TypeError) as e:
    REDIS_AVAILABLE = False
    get_redis_client_llm = None


# Thread pool for running sync OpenAI client
_openai_web_search_executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="openai_web_search_"
)


class OpenAIWebSearchTool(BaseTool):
    """
    OpenAI Native Web Search Tool.

    Category: web
    Data Source: OpenAI Web Search API (via Responses API)
    Cache: Redis with 3-minute TTL (shorter for fresh results)

    This is the PRIMARY web search tool. Uses OpenAI's native web_search
    capability which is more powerful and accurate than third-party APIs.

    Use this tool when:
    - Need latest news and events
    - Need to verify or supplement financial data
    - User asks about recent developments
    - Cross-referencing with multiple sources needed

    Key advantage over Tavily/SerpAPI:
    - Native integration with OpenAI models
    - Better citation and source tracking
    - More accurate answer extraction
    - Built-in domain filtering
    """

    CACHE_TTL = 180  # 3 minutes - shorter for fresh news
    MAX_QUERY_LENGTH = 500
    DEFAULT_MAX_RESULTS = 5

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__()

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model or os.environ.get("WEB_SEARCH_MODEL", "gpt-5-mini")

        if not self.api_key:
            logging.warning(
                "[OpenAIWebSearchTool] OPENAI_API_KEY not set. "
                "Web search will be disabled."
            )

        self.logger = logging.getLogger(__name__)
        self._client = None  # Lazy initialization

        # Load allowed domains from env (optional)
        allowed_domains_env = os.environ.get("WEB_SEARCH_ALLOWED_DOMAINS", "")
        self.allowed_domains = [
            d.strip() for d in allowed_domains_env.split(",")
            if d.strip()
        ] if allowed_domains_env else None

        # Default finance-related domains
        self.default_finance_domains = [
            "bloomberg.com",
            "reuters.com",
            "cnbc.com",
            "marketwatch.com",
            "seekingalpha.com",
            "finance.yahoo.com",
            "wsj.com",
            "ft.com",
            "fool.com",
            "investopedia.com",
            "barrons.com",
            "thestreet.com",
        ]

        # Define tool schema
        self.schema = ToolSchema(
            name="openaiWebSearch",
            category="web",
            description=(
                "Search the web using OpenAI's native web search. "
                "PRIMARY web search tool with better accuracy and citations. "
                "Use for: latest news, recent events, analyst opinions, "
                "market sentiment, regulatory changes, earnings reports. "
                "Automatically includes source URLs and citations."
            ),
            capabilities=[
                "Real-time web search with OpenAI integration",
                "Automatic source citation with URLs",
                "Knowledge graph and direct answer extraction",
                "Domain filtering for authoritative sources",
                "Finance-optimized search results",
                "Multi-source verification",
            ],
            limitations=[
                "Rate limited by OpenAI API",
                "Cannot access paywalled content",
                "Results may be summarized",
            ],
            usage_hints=[
                "Use for latest news and events",
                "Combine with FMP tools for comprehensive analysis",
                "Include stock ticker in query for better results",
                "Sources are automatically cited",
            ],
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "Search query. Be specific and include context. "
                        "Examples: 'NVDA earnings Q4 2024 results', "
                        "'Fed interest rate decision January 2025', "
                        "'Apple stock analyst ratings'"
                    ),
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of sources to return (default: 5, max: 10)",
                    required=False,
                    default=5,
                    min_value=1,
                    max_value=10,
                ),
                ToolParameter(
                    name="use_finance_domains",
                    type="boolean",
                    description="Filter to finance-specific domains (default: true for finance queries)",
                    required=False,
                    default=True,
                ),
                ToolParameter(
                    name="user_location",
                    type="string",
                    description="User location for localized results (e.g., 'US', 'VN')",
                    required=False,
                ),
            ],
            returns={
                "query": "string - The search query used",
                "model": "string - Model used for search",
                "answer": "string - Synthesized answer from search results",
                "citations": "array - List of {url, title, start_index, end_index}",
                "sources": "array - List of all sources consulted",
                "search_actions": "array - Search actions performed",
                "execution_time_ms": "number - Search execution time",
            },
            requires_symbol=False,
        )

    def _get_client(self):
        """Get or create OpenAI client (lazy initialization)."""
        if self._client is None and self.api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError:
                self.logger.error("[OpenAIWebSearchTool] openai package not installed")
                raise ImportError("openai>=2.15.0 is required for web search")
        return self._client

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        use_finance_domains: bool = True,
        user_location: Optional[str] = None,
        **kwargs,
    ) -> ToolOutput:
        """
        Execute OpenAI web search.

        Args:
            query: Search query string
            max_results: Maximum number of sources (1-10)
            use_finance_domains: Filter to finance domains
            user_location: User country code for localized results

        Returns:
            ToolOutput with search results and citations
        """
        start_time = datetime.now()

        # Validate API key
        if not self.api_key:
            return create_error_output(
                tool_name="openaiWebSearch",
                error="Web search not configured. OPENAI_API_KEY missing.",
                metadata={"query": query[:100]},
            )

        # Validate and sanitize query
        query = query.strip()
        if not query:
            return create_error_output(
                tool_name="openaiWebSearch",
                error="Search query cannot be empty",
                metadata={},
            )

        if len(query) > self.MAX_QUERY_LENGTH:
            query = query[:self.MAX_QUERY_LENGTH]

        # Validate parameters
        max_results = min(max(1, max_results), 10)

        self.logger.info(
            f"[openaiWebSearch] Executing: query='{query[:50]}...' "
            f"model={self.model} finance_domains={use_finance_domains}"
        )

        try:
            # Build cache key
            cache_key = f"openaiWebSearch:{hash(query)}:{max_results}:{use_finance_domains}"

            # Try cache first
            cached_data = await self._get_cached_result(cache_key)
            if cached_data:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.info(f"[openaiWebSearch] CACHE HIT ({int(execution_time)}ms)")

                return create_success_output(
                    tool_name="openaiWebSearch",
                    data=cached_data,
                    metadata={
                        "query": query[:100],
                        "execution_time_ms": int(execution_time),
                        "from_cache": True,
                    },
                )

            # Execute search in thread pool (OpenAI client is synchronous)
            search_result = await self._execute_search_async(
                query=query,
                max_results=max_results,
                use_finance_domains=use_finance_domains,
                user_location=user_location,
            )

            if search_result is None:
                return create_error_output(
                    tool_name="openaiWebSearch",
                    error="Search failed. No results returned.",
                    metadata={"query": query[:100]},
                )

            # Format results
            result_data = self._format_results(search_result, query)

            # Cache result
            await self._cache_result(cache_key, result_data)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(
                f"[openaiWebSearch] SUCCESS ({int(execution_time)}ms) - "
                f"{len(result_data.get('citations', []))} citations"
            )

            return create_success_output(
                tool_name="openaiWebSearch",
                data=result_data,
                formatted_context=self._create_llm_context(result_data),
                metadata={
                    "query": query[:100],
                    "execution_time_ms": int(execution_time),
                    "model": self.model,
                    "citation_count": len(result_data.get("citations", [])),
                    "from_cache": False,
                },
            )

        except Exception as e:
            self.logger.error(f"[openaiWebSearch] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name="openaiWebSearch",
                error=f"Search error: {str(e)}",
                metadata={"query": query[:100]},
            )

    async def execute_streaming(
        self,
        query: str,
        max_results: int = 5,
        use_finance_domains: bool = True,
        user_location: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute OpenAI web search with streaming events.

        Yields progress events for UI display:
        - {"type": "search_start", "query": "..."}
        - {"type": "search_action", "action": "searching", "queries": [...]}
        - {"type": "search_progress", "message": "..."}
        - {"type": "search_complete", "data": {...}}

        Args:
            query: Search query string
            max_results: Maximum number of sources
            use_finance_domains: Filter to finance domains
            user_location: User country code

        Yields:
            Dict events for SSE streaming
        """
        start_time = datetime.now()

        # Emit start event
        yield {
            "type": "search_start",
            "query": query[:100],
            "model": self.model,
            "timestamp": datetime.now().isoformat(),
        }

        # Validate
        if not self.api_key:
            yield {
                "type": "search_error",
                "error": "OPENAI_API_KEY not configured",
            }
            return

        query = query.strip()
        if not query:
            yield {
                "type": "search_error",
                "error": "Empty query",
            }
            return

        try:
            # Emit searching event
            yield {
                "type": "search_action",
                "action": "searching",
                "message": f"Searching web for: {query[:50]}...",
            }

            # Execute search
            search_result = await self._execute_search_async(
                query=query,
                max_results=max_results,
                use_finance_domains=use_finance_domains,
                user_location=user_location,
                include_sources=True,  # Request source list
            )

            if search_result is None:
                yield {
                    "type": "search_error",
                    "error": "No results returned",
                }
                return

            # Format results
            result_data = self._format_results(search_result, query)

            # Emit search actions if available
            if result_data.get("search_actions"):
                for action in result_data["search_actions"]:
                    yield {
                        "type": "search_action",
                        "action": action.get("type", "search"),
                        "queries": action.get("queries", []),
                        "domains": action.get("domains", []),
                    }

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Emit complete event
            yield {
                "type": "search_complete",
                "data": result_data,
                "execution_time_ms": int(execution_time),
                "citation_count": len(result_data.get("citations", [])),
            }

        except Exception as e:
            self.logger.error(f"[openaiWebSearch] Streaming error: {e}")
            yield {
                "type": "search_error",
                "error": str(e),
            }

    async def _execute_search_async(
        self,
        query: str,
        max_results: int,
        use_finance_domains: bool,
        user_location: Optional[str] = None,
        include_sources: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Execute OpenAI web search in thread pool."""
        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _openai_web_search_executor,
                    self._sync_search,
                    query,
                    max_results,
                    use_finance_domains,
                    user_location,
                    include_sources,
                ),
                timeout=60.0,  # 60 second timeout for web search
            )
            return result
        except asyncio.TimeoutError:
            self.logger.error("[openaiWebSearch] Search timed out after 60s")
            return None
        except Exception as e:
            self.logger.error(f"[openaiWebSearch] Async execution error: {e}")
            return None

    def _sync_search(
        self,
        query: str,
        max_results: int,
        use_finance_domains: bool,
        user_location: Optional[str],
        include_sources: bool,
    ) -> Optional[Dict[str, Any]]:
        """Synchronous OpenAI web search (runs in thread pool)."""
        try:
            client = self._get_client()
            if not client:
                return None

            # Build web search tool config
            web_search_config: Dict[str, Any] = {
                "type": "web_search",
            }

            # Add domain filtering if specified
            if use_finance_domains:
                domains = self.allowed_domains or self.default_finance_domains
                if domains:
                    web_search_config["filters"] = {
                        "allowed_domains": domains[:100]  # Max 100 domains
                    }

            # Add user location if specified
            if user_location:
                country_map = {
                    "VN": {"country": "VN", "city": "Ho Chi Minh City"},
                    "US": {"country": "US", "city": "New York"},
                    "UK": {"country": "GB", "city": "London"},
                    "GB": {"country": "GB", "city": "London"},
                }
                location = country_map.get(user_location.upper(), {"country": user_location.upper()})
                web_search_config["user_location"] = {
                    "type": "approximate",
                    **location,
                }

            # Build include list for source tracking
            include_list = []
            if include_sources:
                include_list.append("web_search_call.action.sources")

            # Make API call using Responses API
            kwargs = {
                "model": self.model,
                "tools": [web_search_config],
                "input": query,
            }

            if include_list:
                kwargs["include"] = include_list

            response = client.responses.create(**kwargs)

            # Parse response
            return self._parse_response(response)

        except Exception as e:
            self.logger.error(f"[openaiWebSearch] OpenAI API error: {e}")
            return None

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse OpenAI Responses API response."""
        result = {
            "output_text": "",
            "citations": [],
            "sources": [],
            "search_actions": [],
        }

        # Handle response.output list
        if hasattr(response, "output") and response.output:
            for item in response.output:
                item_type = getattr(item, "type", None)

                # Handle web_search_call
                if item_type == "web_search_call":
                    action_data = {
                        "id": getattr(item, "id", ""),
                        "type": "web_search",
                        "status": getattr(item, "status", ""),
                    }

                    # Extract action details if available
                    if hasattr(item, "action"):
                        action = item.action
                        action_data["queries"] = getattr(action, "queries", [])
                        action_data["domains"] = getattr(action, "domains", [])

                        # Extract sources if available
                        if hasattr(action, "sources"):
                            result["sources"].extend(action.sources)

                    result["search_actions"].append(action_data)

                # Handle message with content
                elif item_type == "message":
                    if hasattr(item, "content") and item.content:
                        for content_item in item.content:
                            content_type = getattr(content_item, "type", None)

                            if content_type == "output_text":
                                result["output_text"] = getattr(content_item, "text", "")

                                # Extract annotations (citations)
                                if hasattr(content_item, "annotations"):
                                    for ann in content_item.annotations:
                                        if getattr(ann, "type", None) == "url_citation":
                                            result["citations"].append({
                                                "url": getattr(ann, "url", ""),
                                                "title": getattr(ann, "title", ""),
                                                "start_index": getattr(ann, "start_index", 0),
                                                "end_index": getattr(ann, "end_index", 0),
                                            })

        # Fallback to output_text attribute
        if not result["output_text"] and hasattr(response, "output_text"):
            result["output_text"] = response.output_text

        return result

    def _format_results(
        self,
        raw_response: Dict[str, Any],
        query: str,
    ) -> Dict[str, Any]:
        """Format OpenAI response to structured output."""
        return {
            "query": query,
            "model": self.model,
            "answer": raw_response.get("output_text", ""),
            "citations": raw_response.get("citations", []),
            "sources": raw_response.get("sources", []),
            "search_actions": raw_response.get("search_actions", []),
            "timestamp": datetime.now().isoformat(),
        }

    def _create_llm_context(self, result_data: Dict[str, Any]) -> str:
        """Create formatted context string for LLM consumption."""
        lines = [
            f"🌐 Web Search Results for: {result_data['query']}",
            f"Model: {result_data['model']}",
            "",
        ]

        # Add main answer
        if result_data.get("answer"):
            lines.append("📋 ANSWER:")
            lines.append(result_data["answer"])
            lines.append("")

        # Add citations
        if result_data.get("citations"):
            lines.append(f"📚 CITATIONS ({len(result_data['citations'])}):")
            for i, citation in enumerate(result_data["citations"], 1):
                title = citation.get("title", "Untitled")
                url = citation.get("url", "")
                lines.append(f"  [{i}] {title}")
                lines.append(f"      🔗 {url}")
            lines.append("")

        # Add sources if available
        if result_data.get("sources"):
            lines.append(f"📰 SOURCES ({len(result_data['sources'])}):")
            for source in result_data["sources"][:10]:  # Limit to 10
                if isinstance(source, dict):
                    lines.append(f"  • {source.get('name', source.get('url', 'Unknown'))}")
                else:
                    lines.append(f"  • {source}")
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
            self.logger.warning(f"[openaiWebSearch] Cache read error: {e}")
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
            self.logger.warning(f"[openaiWebSearch] Cache write error: {e}")


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set")
            return

        tool = OpenAIWebSearchTool(api_key=api_key)

        print("\n" + "=" * 60)
        print("Testing OpenAIWebSearchTool")
        print("=" * 60)

        # Test 1: Basic search
        print("\nTest 1: NVIDIA earnings search")
        result = await tool.execute(
            query="NVIDIA Q4 2024 earnings results",
            max_results=5,
            use_finance_domains=True,
        )

        if result.status == "success":
            data = result.data
            print(f"Success!")
            print(f"Answer: {data.get('answer', '')[:200]}...")
            print(f"Citations: {len(data.get('citations', []))}")
            for c in data.get("citations", [])[:3]:
                print(f"  - {c.get('title', '')[:50]} | {c.get('url', '')[:50]}")
        else:
            print(f"Error: {result.error}")

        # Test 2: Streaming search
        print("\n" + "-" * 40)
        print("Test 2: Streaming search")
        async for event in tool.execute_streaming(
            query="Apple stock news today",
            max_results=3,
        ):
            print(f"Event: {event.get('type')} - {event}")

    asyncio.run(test())
