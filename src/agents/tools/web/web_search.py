"""
Web Search Tool

Provides web search capability with dual-backend support:
- PRIMARY: OpenAI Responses API with native web_search (gpt-5-mini)
- FALLBACK: Tavily API (when OpenAI fails, e.g., out of credits)

Features:
- Async execution (non-blocking)
- Smart search optimization for finance queries
- Redis caching to reduce API calls
- Automatic fallback mechanism
- Streaming support for SSE events
- Structured output with citations

Configuration (.env):
- OPENAI_API_KEY: Required for primary OpenAI web search
- WEB_SEARCH_MODEL: Model for OpenAI search (default: gpt-5-mini)
- TAVILY_API_KEY: Required for fallback Tavily search

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

# Optional Redis import - gracefully handle if not available
try:
    from src.helpers.redis_cache import get_redis_client_llm
    REDIS_AVAILABLE = True
except (ImportError, TypeError) as e:
    REDIS_AVAILABLE = False
    get_redis_client_llm = None
    logging.warning(f"[WebSearchTool] Redis not available, caching disabled: {e}")


# Thread pool for running sync clients
_web_search_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="web_search_")


class WebSearchTool(BaseTool):
    """
    Web Search Tool with OpenAI PRIMARY and Tavily FALLBACK.

    Category: web
    Data Source:
        - PRIMARY: OpenAI Responses API with web_search (gpt-5-mini)
        - FALLBACK: Tavily Search API
    Cache: Redis with 3-minute TTL

    Use this tool when:
    - FMP data is not available or outdated
    - User asks about recent events not covered by financial data
    - Need to find specific information not in financial databases
    - Cross-referencing financial data with news/events
    - News tools are called (auto-triggered for additional context)
    """

    CACHE_TTL = 180  # 3 minutes - shorter for fresh news
    MAX_QUERY_LENGTH = 500
    DEFAULT_MAX_RESULTS = 5

    # Default finance-related domains for OpenAI search
    DEFAULT_FINANCE_DOMAINS = [
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

    def __init__(
        self,
        api_key: Optional[str] = None,  # Tavily API key (for backward compatibility)
        openai_api_key: Optional[str] = None,
        web_search_model: Optional[str] = None,
    ):
        super().__init__()

        # OpenAI config (PRIMARY)
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.web_search_model = web_search_model or os.environ.get("WEB_SEARCH_MODEL", "gpt-5-mini")

        # Tavily config (FALLBACK)
        self.tavily_api_key = api_key or os.environ.get("TAVILY_API_KEY")

        self.logger = logging.getLogger(__name__)
        self._openai_client = None
        self._tavily_client = None

        # Log configuration
        if self.openai_api_key:
            self.logger.info(f"[WebSearchTool] OpenAI web search enabled (model: {self.web_search_model})")
        else:
            self.logger.warning("[WebSearchTool] OPENAI_API_KEY not set - OpenAI search disabled")

        if self.tavily_api_key:
            self.logger.info("[WebSearchTool] Tavily fallback enabled")
        else:
            self.logger.warning("[WebSearchTool] TAVILY_API_KEY not set - Tavily fallback disabled")

        # Define tool schema
        self.schema = ToolSchema(
            name="webSearch",
            category="web",
            description=(
                "Search the web for current information using OpenAI's native search. "
                "Use for: breaking news, recent events, analyst opinions, market sentiment, "
                "regulatory changes, earnings reports, unfamiliar trading terms. "
                "Returns verified citations with source URLs. "
                "Auto-triggered when news tools are called for additional context."
            ),
            capabilities=[
                "Real-time web search with verified citations",
                "Find recent news and market events",
                "Access post-cutoff knowledge (new policies, products, events)",
                "Research trading strategies and methodologies",
                "Verify uncertain or time-sensitive information",
                "Get context for regulatory changes and executive statements",
            ],
            limitations=[
                "SECONDARY to FMP tools - do not use for prices, financials, technicals",
                "Results limited to 3-5 for optimal context",
                "Cannot access paywalled content",
                "Rate limited - use judiciously",
            ],
            usage_hints=[
                "MUST search: time-sensitive info, recent news, verification needs",
                "SKIP search: FMP tools can answer, classic indicators (RSI, MACD)",
                "Query tips: Include stock ticker and context for better results",
                "Citations are automatically included - display them to user",
            ],
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "Search query. Be specific and include relevant context. "
                        "Examples: 'NVDA earnings Q4 2024', 'Fed interest rate decision 2025', "
                        "'Apple stock analyst ratings'"
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
                    name="use_finance_domains",
                    type="boolean",
                    description="Filter to finance-specific domains. Default: false (model decides sources automatically based on query)",
                    required=False,
                    default=False,
                ),
            ],
            returns={
                "query": "string - The search query used",
                "result_count": "number - Number of results returned",
                "answer": "string - Synthesized answer (OpenAI only)",
                "citations": "array - List of {title, url} citations",
                "results": "array - List of search results",
                "source": "string - 'openai' or 'tavily'",
                "search_time_ms": "number - Search execution time",
            },
            requires_symbol=False,
        )

    def _get_openai_client(self):
        """Get or create OpenAI client (lazy initialization)."""
        if self._openai_client is None and self.openai_api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.openai_api_key)
            except ImportError:
                self.logger.error("[WebSearchTool] openai package not installed")
        return self._openai_client

    def _get_tavily_client(self):
        """Get or create Tavily client (lazy initialization)."""
        if self._tavily_client is None and self.tavily_api_key:
            try:
                from tavily import TavilyClient
                self._tavily_client = TavilyClient(api_key=self.tavily_api_key)
            except ImportError:
                self.logger.error("[WebSearchTool] tavily package not installed")
        return self._tavily_client

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        use_finance_domains: bool = False,  # Default: let model decide sources
        search_depth: str = "advanced",  # For Tavily compatibility
        time_range: str = "week",  # For Tavily compatibility
        **kwargs,
    ) -> ToolOutput:
        """
        Execute web search.

        Strategy:
        1. Try OpenAI Responses API with web_search (PRIMARY)
        2. If fails (error, out of credits), fallback to Tavily

        Args:
            query: Search query string
            max_results: Maximum number of results (1-10)
            use_finance_domains: Filter to finance domains (default: False - model decides)
            search_depth: "basic" or "advanced" (Tavily only)
            time_range: Time range for results (Tavily only)

        Returns:
            ToolOutput with search results
        """
        start_time = datetime.now()

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

        max_results = min(max(1, max_results), 10)

        self.logger.info(
            f"[webSearch] Executing: query='{query[:50]}...' max_results={max_results}"
        )

        try:
            # Build cache key
            cache_key = f"webSearch:v2:{hash(query)}:{max_results}:{use_finance_domains}"

            # Try cache first
            cached_data = await self._get_cached_result(cache_key)
            if cached_data:
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                self.logger.info(f"[webSearch] CACHE HIT ({int(execution_time)}ms)")

                return create_success_output(
                    tool_name="webSearch",
                    data=cached_data,
                    formatted_context=self._create_llm_context(cached_data),
                    metadata={
                        "query": query[:100],
                        "execution_time_ms": int(execution_time),
                        "from_cache": True,
                        "source": cached_data.get("source", "cache"),
                    },
                )

            # ================================================================
            # PRIMARY: Try OpenAI web search
            # ================================================================
            result_data = None
            openai_error = None

            if self.openai_api_key:
                try:
                    self.logger.info(f"[webSearch] Trying OpenAI ({self.web_search_model})...")
                    result_data = await self._execute_openai_search(
                        query=query,
                        max_results=max_results,
                        use_finance_domains=use_finance_domains,
                    )
                    if result_data:
                        self.logger.info(
                            f"[webSearch] OpenAI SUCCESS: {len(result_data.get('citations', []))} citations"
                        )
                except Exception as e:
                    openai_error = str(e)
                    self.logger.warning(f"[webSearch] OpenAI failed: {e}")

            # ================================================================
            # FALLBACK: Try Tavily if OpenAI failed
            # ================================================================
            if result_data is None and self.tavily_api_key:
                self.logger.info("[webSearch] Falling back to Tavily...")
                try:
                    result_data = await self._execute_tavily_search(
                        query=query,
                        max_results=max_results,
                        search_depth=search_depth,
                        time_range=time_range,
                    )
                    if result_data:
                        self.logger.info(
                            f"[webSearch] Tavily SUCCESS: {result_data.get('result_count', 0)} results"
                        )
                except Exception as e:
                    self.logger.error(f"[webSearch] Tavily failed: {e}")

            # ================================================================
            # Handle results
            # ================================================================
            if result_data is None:
                error_msg = "Web search failed."
                if openai_error:
                    error_msg += f" OpenAI: {openai_error}."
                if not self.openai_api_key and not self.tavily_api_key:
                    error_msg += " No API keys configured."

                return create_error_output(
                    tool_name="webSearch",
                    error=error_msg,
                    metadata={"query": query[:100]},
                )

            # Cache result
            await self._cache_result(cache_key, result_data)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            self.logger.info(
                f"[webSearch] SUCCESS ({int(execution_time)}ms) - "
                f"source={result_data.get('source', 'unknown')}"
            )

            # ============================================================
            # DEBUG: Log full web search result for debugging
            # ============================================================
            self.logger.info(f"[webSearch] ========== FULL RESULT ==========")
            self.logger.info(f"[webSearch] Query: {result_data.get('query', '')}")
            self.logger.info(f"[webSearch] Source: {result_data.get('source', '')}")
            self.logger.info(f"[webSearch] Search Time: {result_data.get('search_time_ms', 0)}ms")

            # Log answer
            answer = result_data.get("answer", "")
            if answer:
                self.logger.info(f"[webSearch] ANSWER ({len(answer)} chars):")
                # Log answer in chunks to avoid truncation
                for i in range(0, min(len(answer), 3000), 500):
                    self.logger.info(f"[webSearch]   {answer[i:i+500]}")

            # Log citations with URLs
            citations = result_data.get("citations", [])
            self.logger.info(f"[webSearch] CITATIONS ({len(citations)}):")
            for i, c in enumerate(citations, 1):
                self.logger.info(
                    f"[webSearch]   [{i}] {c.get('title', 'No title')[:80]}"
                )
                self.logger.info(f"[webSearch]       URL: {c.get('url', 'No URL')}")

            self.logger.info(f"[webSearch] ================================")

            # Create formatted context
            formatted_ctx = self._create_llm_context(result_data)
            self.logger.info(f"[webSearch] Formatted context length: {len(formatted_ctx)} chars")

            return create_success_output(
                tool_name="webSearch",
                data=result_data,
                formatted_context=formatted_ctx,
                metadata={
                    "query": query[:100],
                    "execution_time_ms": int(execution_time),
                    "source": result_data.get("source", "unknown"),
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

    async def execute_streaming(
        self,
        query: str,
        max_results: int = 5,
        use_finance_domains: bool = False,  # Default: let model decide sources
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute web search with streaming events for SSE.

        Yields progress events for UI display:
        - {"type": "web_search_start", ...}
        - {"type": "web_search_progress", ...}
        - {"type": "web_search_complete", ...}
        - {"type": "web_search_result", ...}  # Final result

        Args:
            query: Search query string
            max_results: Maximum number of sources
            use_finance_domains: Filter to finance domains (default: False - model decides)

        Yields:
            Dict events for SSE streaming
        """
        start_time = datetime.now()

        # Emit start event
        yield {
            "type": "web_search_start",
            "query": query[:100],
            "model": self.web_search_model if self.openai_api_key else "tavily",
            "timestamp": datetime.now().isoformat(),
        }

        # Validate query
        query = query.strip()
        if not query:
            yield {
                "type": "web_search_error",
                "error": "Empty query",
            }
            yield {
                "type": "web_search_result",
                "tool_name": "webSearch",
                "status": "error",
                "error": "Empty query",
            }
            return

        try:
            # Emit searching event
            yield {
                "type": "web_search_progress",
                "action": "searching",
                "message": f"Searching: {query[:50]}...",
            }

            # Execute search (same logic as execute())
            result_data = None

            # Try OpenAI first
            if self.openai_api_key:
                yield {
                    "type": "web_search_progress",
                    "action": "openai_search",
                    "message": f"Using OpenAI ({self.web_search_model})...",
                }

                try:
                    result_data = await self._execute_openai_search(
                        query=query,
                        max_results=max_results,
                        use_finance_domains=use_finance_domains,
                    )
                except Exception as e:
                    self.logger.warning(f"[webSearch] OpenAI streaming failed: {e}")
                    yield {
                        "type": "web_search_progress",
                        "action": "fallback",
                        "message": f"OpenAI failed, trying Tavily...",
                    }

            # Fallback to Tavily
            if result_data is None and self.tavily_api_key:
                yield {
                    "type": "web_search_progress",
                    "action": "tavily_search",
                    "message": "Using Tavily search...",
                }

                try:
                    result_data = await self._execute_tavily_search(
                        query=query,
                        max_results=max_results,
                    )
                except Exception as e:
                    self.logger.error(f"[webSearch] Tavily streaming failed: {e}")

            if result_data is None:
                yield {
                    "type": "web_search_error",
                    "error": "All search backends failed",
                }
                yield {
                    "type": "web_search_result",
                    "tool_name": "webSearch",
                    "status": "error",
                    "error": "Search failed",
                }
                return

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Emit complete event
            yield {
                "type": "web_search_complete",
                "data": {
                    "answer": result_data.get("answer", "")[:1000],
                    "citation_count": len(result_data.get("citations", [])),
                    "citations": [
                        {"title": c.get("title", "")[:100], "url": c.get("url", "")}
                        for c in result_data.get("citations", [])[:5]
                    ],
                    "source": result_data.get("source", "unknown"),
                },
                "execution_time_ms": int(execution_time),
            }

            # Emit final result
            yield {
                "type": "web_search_result",
                "tool_name": "webSearch",
                "status": "success",
                "data": result_data,
                "formatted_context": self._create_llm_context(result_data),
            }

        except Exception as e:
            self.logger.error(f"[webSearch] Streaming error: {e}")
            yield {
                "type": "web_search_error",
                "error": str(e),
            }
            yield {
                "type": "web_search_result",
                "tool_name": "webSearch",
                "status": "error",
                "error": str(e),
            }

    async def _execute_openai_search(
        self,
        query: str,
        max_results: int,
        use_finance_domains: bool,
    ) -> Optional[Dict[str, Any]]:
        """Execute OpenAI Responses API web search."""
        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _web_search_executor,
                    self._sync_openai_search,
                    query,
                    max_results,
                    use_finance_domains,
                ),
                timeout=60.0,  # 60 second timeout
            )
            return result
        except asyncio.TimeoutError:
            self.logger.error("[webSearch] OpenAI search timed out after 60s")
            return None
        except Exception as e:
            self.logger.error(f"[webSearch] OpenAI async error: {e}")
            raise  # Re-raise to trigger fallback

    def _sync_openai_search(
        self,
        query: str,
        max_results: int,
        use_finance_domains: bool,
    ) -> Optional[Dict[str, Any]]:
        """Synchronous OpenAI search (runs in thread pool)."""
        import time
        start_time = time.time()

        try:
            client = self._get_openai_client()
            if not client:
                return None

            # Check if Responses API is available (requires openai>=1.60.0 or newer)
            if not hasattr(client, "responses"):
                self.logger.warning(
                    "[webSearch] OpenAI Responses API not available. "
                    "Upgrade openai library: pip install openai>=1.68.0"
                )
                return None

            # Build web search tool config - let model decide sources by default
            web_search_config: Dict[str, Any] = {
                "type": "web_search",
            }

            # Only add domain filtering if explicitly requested
            # By default, let the model decide appropriate sources
            if use_finance_domains and self.DEFAULT_FINANCE_DOMAINS:
                web_search_config["filters"] = {
                    "allowed_domains": self.DEFAULT_FINANCE_DOMAINS
                }
                self.logger.info("[webSearch] Using finance domain filter")
            else:
                self.logger.info("[webSearch] No domain filter - model will decide sources")

            # Make API call using Responses API
            response = client.responses.create(
                model=self.web_search_model,
                tools=[web_search_config],
                input=query,
            )

            # Calculate elapsed time
            elapsed_ms = int((time.time() - start_time) * 1000)

            # Parse response with timing
            return self._parse_openai_response(response, query, elapsed_ms=elapsed_ms)

        except AttributeError as e:
            self.logger.error(f"[webSearch] OpenAI Responses API not available: {e}")
            self.logger.info("[webSearch] Please upgrade: pip install openai>=1.68.0")
            raise
        except Exception as e:
            self.logger.error(f"[webSearch] OpenAI API error: {e}")
            raise  # Re-raise to trigger fallback

    def _parse_openai_response(self, response, query: str, elapsed_ms: int = 0) -> Dict[str, Any]:
        """Parse OpenAI Responses API response."""
        result = {
            "query": query,
            "source": "openai",
            "answer": "",
            "citations": [],
            "results": [],
            "search_actions": [],
            "search_time_ms": elapsed_ms,  # Required field
            "timestamp": datetime.now().isoformat(),
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
                    if hasattr(item, "action"):
                        action = item.action
                        action_data["queries"] = getattr(action, "queries", [])
                    result["search_actions"].append(action_data)

                # Handle message with content
                elif item_type == "message":
                    if hasattr(item, "content") and item.content:
                        for content_item in item.content:
                            content_type = getattr(content_item, "type", None)

                            if content_type == "output_text":
                                result["answer"] = getattr(content_item, "text", "")

                                # Extract annotations (citations)
                                if hasattr(content_item, "annotations"):
                                    for ann in content_item.annotations:
                                        if getattr(ann, "type", None) == "url_citation":
                                            citation = {
                                                "url": getattr(ann, "url", ""),
                                                "title": getattr(ann, "title", ""),
                                                "start_index": getattr(ann, "start_index", 0),
                                                "end_index": getattr(ann, "end_index", 0),
                                            }
                                            result["citations"].append(citation)
                                            # Also add to results for compatibility
                                            result["results"].append({
                                                "title": citation["title"],
                                                "url": citation["url"],
                                                "content": "",
                                            })

        # Fallback to output_text attribute
        if not result["answer"] and hasattr(response, "output_text"):
            result["answer"] = response.output_text

        result["result_count"] = len(result["citations"])
        return result

    async def _execute_tavily_search(
        self,
        query: str,
        max_results: int,
        search_depth: str = "advanced",
        time_range: str = "week",
    ) -> Optional[Dict[str, Any]]:
        """Execute Tavily search as fallback."""
        loop = asyncio.get_event_loop()

        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _web_search_executor,
                    self._sync_tavily_search,
                    query,
                    max_results,
                    search_depth,
                    time_range,
                ),
                timeout=30.0,
            )
            return result
        except asyncio.TimeoutError:
            self.logger.error("[webSearch] Tavily search timed out after 30s")
            return None
        except Exception as e:
            self.logger.error(f"[webSearch] Tavily async error: {e}")
            return None

    def _sync_tavily_search(
        self,
        query: str,
        max_results: int,
        search_depth: str,
        time_range: str,
    ) -> Optional[Dict[str, Any]]:
        """Synchronous Tavily search (runs in thread pool)."""
        try:
            client = self._get_tavily_client()
            if not client:
                return None

            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "time_range": time_range,
                "topic": "finance",
                "exclude_domains": ["reddit.com", "twitter.com", "x.com", "facebook.com"],
            }

            response = client.search(**search_params)

            if not isinstance(response, dict):
                return None

            # Format Tavily response
            results = response.get("results", [])
            formatted_results = []
            citations = []

            for item in results[:max_results]:
                formatted_results.append({
                    "title": item.get("title", "")[:200],
                    "url": item.get("url", ""),
                    "content": item.get("content", "")[:500],
                    "score": round(float(item.get("score", 0)), 3),
                    "published_date": item.get("published_date"),
                })
                citations.append({
                    "title": item.get("title", "")[:200],
                    "url": item.get("url", ""),
                })

            return {
                "query": query,
                "source": "tavily",
                "answer": "",  # Tavily doesn't provide synthesized answer
                "citations": citations,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "search_time_ms": response.get("response_time", 0),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"[webSearch] Tavily API error: {e}")
            return None

    def _create_llm_context(self, result_data: Dict[str, Any]) -> str:
        """Create formatted context string for LLM consumption."""
        source = result_data.get("source", "unknown")
        query = result_data.get("query", "")

        lines = [
            f"🌐 Web Search Results for: {query}",
            f"Source: {source.upper()}",
            "",
        ]

        # Add synthesized answer (OpenAI only)
        answer = result_data.get("answer", "")
        if answer:
            lines.append("📋 ANSWER:")
            lines.append(answer[:2000])
            lines.append("")

        # Add citations
        citations = result_data.get("citations", [])
        if citations:
            lines.append(f"📚 CITATIONS ({len(citations)}):")
            for i, c in enumerate(citations[:10], 1):
                title = c.get("title", "Untitled")[:100]
                url = c.get("url", "")
                lines.append(f"  [{i}] {title}")
                lines.append(f"      🔗 {url}")
            lines.append("")

        # Add results (Tavily format)
        results = result_data.get("results", [])
        if results and not answer:  # Only show if no answer (Tavily)
            lines.append(f"🔍 SEARCH RESULTS ({len(results)}):")
            for i, r in enumerate(results[:5], 1):
                lines.append(f"{i}. {r.get('title', '')}")
                content = r.get("content", "")
                if content:
                    lines.append(f"   {content[:250]}...")
                lines.append(f"   Source: {r.get('url', '')}")
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
        tool = WebSearchTool()

        print("\n" + "=" * 60)
        print("Testing WebSearchTool (OpenAI PRIMARY, Tavily FALLBACK)")
        print("=" * 60)

        # Test: Basic search
        print("\nTest: NVIDIA earnings search")
        result = await tool.execute(
            query="NVIDIA Q4 2024 earnings results",
            max_results=5,
        )

        if result.status == "success":
            data = result.data
            print(f"Success! Source: {data.get('source', 'unknown')}")
            print(f"Citations: {len(data.get('citations', []))}")
            if data.get("answer"):
                print(f"Answer: {data['answer'][:200]}...")
            for c in data.get("citations", [])[:3]:
                print(f"  - {c.get('title', '')[:50]} | {c.get('url', '')[:50]}")
        else:
            print(f"Error: {result.error}")

        # Test: Streaming
        print("\n" + "-" * 40)
        print("Test: Streaming search")
        async for event in tool.execute_streaming(
            query="Apple stock news today",
            max_results=3,
        ):
            print(f"Event: {event.get('type')}")

    asyncio.run(test())
