"""
Enhanced Web Search Tool

Combines web search with direct URL fetching and freshness validation.
Solves the problem of stale data from search indices.

Architecture:
1. Search: Find relevant URLs via OpenAI/Tavily
2. Fetch: Directly fetch top sources to get current content
3. Validate: Check timestamps and reject stale data
4. Tier: Prioritize official sources over news

This ensures real-time data accuracy for time-sensitive queries like gold prices.
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output,
)
from src.agents.tools.web.web_search import WebSearchTool
from src.agents.tools.web.smart_fetch import SmartFetchTool

logger = logging.getLogger(__name__)

# Thread pool for parallel operations
_enhanced_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="enhanced_search_")


class EnhancedWebSearchTool(BaseTool):
    """
    Enhanced Web Search with direct fetch and freshness validation.

    Flow:
    1. Web search to find relevant URLs
    2. Parallel fetch of top sources
    3. Timestamp validation for each source
    4. Source tiering and quality scoring
    5. Synthesize verified results

    Use this for:
    - Real-time price data (gold, commodities, forex)
    - Time-sensitive information
    - Queries where freshness is critical
    """

    # Freshness requirements by query type
    FRESHNESS_CONFIGS = {
        "gold_price": {"max_age_hours": 4, "require_today": True},
        "stock_price": {"max_age_hours": 1, "require_today": True},
        "crypto_price": {"max_age_hours": 1, "require_today": True},
        "news": {"max_age_hours": 24, "require_today": False},
        "general": {"max_age_hours": 168, "require_today": False},  # 1 week
    }

    # Keywords to detect query type
    QUERY_TYPE_KEYWORDS = {
        "gold_price": ["vàng", "gold", "sjc", "mi hồng", "doji", "pnj", "giá vàng"],
        "stock_price": ["stock", "chứng khoán", "cổ phiếu", "vnindex"],
        "crypto_price": ["bitcoin", "ethereum", "crypto", "btc", "eth", "tiền ảo"],
        "news": ["tin", "news", "báo", "article"],
    }

    # Preferred domains for Vietnamese gold prices
    GOLD_PRICE_DOMAINS = [
        "mihong.vn",
        "sjc.com.vn",
        "doji.vn",
        "pnj.com.vn",
        "btmc.vn",
        "webgia.com",
        "giavang.org",
        "tygia.com",
    ]

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
    ):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Initialize underlying tools
        self.web_search = WebSearchTool(
            openai_api_key=openai_api_key,
            api_key=tavily_api_key,
        )
        self.smart_fetch = SmartFetchTool()

        # Define tool schema
        self.schema = ToolSchema(
            name="enhancedWebSearch",
            category="web",
            description=(
                "Enhanced web search with freshness validation for real-time data. "
                "Combines search, direct fetch, and timestamp verification. "
                "Use for time-sensitive queries like gold prices, stock prices, etc. "
                "Automatically validates data freshness and warns about stale sources."
            ),
            capabilities=[
                "Web search with freshness validation",
                "Direct URL fetching to verify current data",
                "Source quality tiering (official > aggregator > news)",
                "Automatic staleness detection and warning",
                "Parallel fetching of multiple sources",
            ],
            limitations=[
                "Slower than regular web search (fetches multiple pages)",
                "Limited to publicly accessible pages",
                "Rate limited",
            ],
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="expected_date",
                    type="string",
                    description=(
                        "Expected date for the data (YYYY-MM-DD or DD/MM/YYYY). "
                        "Default: today. Used to validate freshness."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="max_age_hours",
                    type="integer",
                    description=(
                        "Maximum acceptable age of data in hours. "
                        "Default: auto-detected based on query type."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="fetch_top_n",
                    type="integer",
                    description=(
                        "Number of top search results to fetch and verify. "
                        "Default: 3. Higher = more thorough but slower."
                    ),
                    required=False,
                    default=3,
                ),
                ToolParameter(
                    name="preferred_domains",
                    type="array",
                    description=(
                        "List of preferred domains to prioritize. "
                        "E.g., ['mihong.vn', 'sjc.com.vn'] for gold prices."
                    ),
                    required=False,
                ),
            ],
            returns={
                "query": "string - The search query",
                "query_type": "string - Detected query type",
                "search_results": "array - Raw search results",
                "verified_sources": "array - Sources with freshness verification",
                "freshest_source": "object - The source with most recent data",
                "has_fresh_data": "boolean - Whether fresh data was found",
                "freshness_summary": "string - Human-readable freshness status",
                "recommendations": "array - Suggestions if data is stale",
            },
            requires_symbol=False,
        )

    async def execute(
        self,
        query: str,
        expected_date: Optional[str] = None,
        max_age_hours: Optional[int] = None,
        fetch_top_n: int = 3,
        preferred_domains: Optional[List[str]] = None,
        **kwargs,
    ) -> ToolOutput:
        """
        Execute enhanced web search with freshness validation.
        """
        start_time = datetime.now()

        # Detect query type
        query_type = self._detect_query_type(query)
        freshness_config = self.FRESHNESS_CONFIGS.get(query_type, self.FRESHNESS_CONFIGS["general"])

        # Use provided max_age or default from config
        if max_age_hours is None:
            max_age_hours = freshness_config["max_age_hours"]

        # Parse expected date
        if expected_date:
            target_date = self._parse_date(expected_date)
        else:
            target_date = datetime.now().date()

        self.logger.info(f"[enhancedWebSearch] Query: {query}")
        self.logger.info(f"[enhancedWebSearch] Type: {query_type}, Max age: {max_age_hours}h")

        try:
            # Step 1: Perform web search
            self.logger.info("[enhancedWebSearch] Step 1: Web search...")
            search_result = await self.web_search.execute(
                query=query,
                max_results=10,
            )

            if search_result.status != "success":
                return create_error_output(
                    tool_name="enhancedWebSearch",
                    error=f"Search failed: {search_result.error}",
                    metadata={"query": query},
                )

            # Extract URLs from search results
            search_data = search_result.data
            urls_to_fetch = self._extract_urls(search_data, preferred_domains or [])

            # Add preferred domain URLs if not in search results
            if query_type == "gold_price" and not preferred_domains:
                preferred_domains = self.GOLD_PRICE_DOMAINS

            if preferred_domains:
                for domain in preferred_domains[:3]:
                    domain_url = self._construct_url_for_domain(domain, query)
                    if domain_url and domain_url not in urls_to_fetch:
                        urls_to_fetch.insert(0, domain_url)

            urls_to_fetch = urls_to_fetch[:fetch_top_n]

            self.logger.info(f"[enhancedWebSearch] Step 2: Fetching {len(urls_to_fetch)} URLs...")

            # Step 2: Parallel fetch of top sources
            fetch_tasks = []
            for url in urls_to_fetch:
                fetch_tasks.append(
                    self.smart_fetch.execute(
                        url=url,
                        expected_date=expected_date,
                        max_age_hours=max_age_hours,
                    )
                )

            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Step 3: Process and validate results
            verified_sources = []
            for i, result in enumerate(fetch_results):
                if isinstance(result, Exception):
                    self.logger.warning(f"[enhancedWebSearch] Fetch {i} failed: {result}")
                    continue

                if result.status == "success":
                    verified_sources.append({
                        "url": urls_to_fetch[i],
                        "title": result.data.get("title", ""),
                        "is_fresh": result.data.get("is_fresh", False),
                        "timestamp": result.data.get("timestamp"),
                        "freshness_message": result.data.get("freshness_message", ""),
                        "source_tier": result.data.get("source_tier", "D"),
                        "source_score": result.data.get("source_score", 0.4),
                        "prices": result.data.get("prices"),
                        "content_preview": result.data.get("content", "")[:500],
                    })

            # Step 4: Find freshest source
            freshest_source = None
            has_fresh_data = False

            # Sort by: fresh first, then by source score
            verified_sources.sort(
                key=lambda x: (
                    x.get("is_fresh", False),
                    x.get("source_score", 0),
                ),
                reverse=True,
            )

            if verified_sources:
                freshest_source = verified_sources[0]
                has_fresh_data = freshest_source.get("is_fresh", False)

            # Step 5: Generate recommendations if no fresh data
            recommendations = []
            if not has_fresh_data:
                recommendations = self._generate_recommendations(
                    query_type, verified_sources, preferred_domains
                )

            # Build freshness summary
            freshness_summary = self._build_freshness_summary(
                verified_sources, has_fresh_data, target_date, query_type
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            result_data = {
                "query": query,
                "query_type": query_type,
                "expected_date": str(target_date),
                "max_age_hours": max_age_hours,
                "search_results": {
                    "source": search_data.get("source", "unknown"),
                    "answer": search_data.get("answer", "")[:2000],
                    "citations": search_data.get("citations", [])[:5],
                },
                "verified_sources": verified_sources,
                "freshest_source": freshest_source,
                "has_fresh_data": has_fresh_data,
                "freshness_summary": freshness_summary,
                "recommendations": recommendations,
                "execution_time_ms": int(execution_time),
            }

            self.logger.info(f"[enhancedWebSearch] SUCCESS in {int(execution_time)}ms")
            self.logger.info(f"[enhancedWebSearch] Fresh data: {has_fresh_data}")
            self.logger.info(f"[enhancedWebSearch] Verified sources: {len(verified_sources)}")

            return create_success_output(
                tool_name="enhancedWebSearch",
                data=result_data,
                formatted_context=self._create_llm_context(result_data),
                metadata={
                    "query": query,
                    "query_type": query_type,
                    "has_fresh_data": has_fresh_data,
                    "verified_source_count": len(verified_sources),
                    "execution_time_ms": int(execution_time),
                },
            )

        except Exception as e:
            self.logger.error(f"[enhancedWebSearch] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name="enhancedWebSearch",
                error=str(e),
                metadata={"query": query},
            )

    def _detect_query_type(self, query: str) -> str:
        """Detect query type from keywords."""
        query_lower = query.lower()

        for query_type, keywords in self.QUERY_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return query_type

        return "general"

    def _extract_urls(self, search_data: Dict[str, Any], preferred_domains: List[str]) -> List[str]:
        """Extract URLs from search results, prioritizing preferred domains."""
        urls = []
        preferred_urls = []

        # From citations
        for citation in search_data.get("citations", []):
            url = citation.get("url", "")
            if url:
                if any(domain in url for domain in preferred_domains):
                    preferred_urls.append(url)
                else:
                    urls.append(url)

        # From results
        for result in search_data.get("results", []):
            url = result.get("url", "")
            if url and url not in urls and url not in preferred_urls:
                if any(domain in url for domain in preferred_domains):
                    preferred_urls.append(url)
                else:
                    urls.append(url)

        return preferred_urls + urls

    def _construct_url_for_domain(self, domain: str, query: str) -> Optional[str]:
        """Construct a URL for a known domain based on query."""
        # Vietnamese gold price pages
        if "mi hồng" in query.lower() or "mihong" in query.lower():
            if domain == "mihong.vn":
                return "https://mihong.vn/bang-gia-vang/"
            elif domain == "webgia.com":
                return "https://webgia.com/gia-vang/mi-hong/"
            elif domain == "giavang.org":
                return "https://giavang.org/mi-hong/"

        if "sjc" in query.lower():
            if domain == "sjc.com.vn":
                return "https://sjc.com.vn/giavang/textContent.php"
            elif domain == "webgia.com":
                return "https://webgia.com/gia-vang/sjc/"

        if "doji" in query.lower():
            if domain == "doji.vn":
                return "https://doji.vn/bang-gia-vang/"
            elif domain == "webgia.com":
                return "https://webgia.com/gia-vang/doji/"

        # Default: try to fetch main gold price page
        if domain == "webgia.com":
            return "https://webgia.com/gia-vang/"
        elif domain == "giavang.org":
            return "https://giavang.org/"

        return None

    def _parse_date(self, date_str: str):
        """Parse date string to date object."""
        formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except:
                continue
        return datetime.now().date()

    def _generate_recommendations(
        self,
        query_type: str,
        verified_sources: List[Dict],
        preferred_domains: Optional[List[str]],
    ) -> List[str]:
        """Generate recommendations when fresh data is unavailable."""
        recommendations = []

        if query_type == "gold_price":
            recommendations.extend([
                "Liên hệ trực tiếp tiệm vàng Mi Hồng qua hotline để có giá chính xác nhất",
                "Kiểm tra trang chính thức mihong.vn hoặc fanpage Facebook của Mi Hồng",
                "Giá vàng có thể chưa được cập nhật do ngoài giờ giao dịch hoặc ngày nghỉ",
            ])
        elif query_type == "stock_price":
            recommendations.extend([
                "Kiểm tra trực tiếp trên HOSE/HNX hoặc app chứng khoán",
                "Thị trường có thể đang ngoài giờ giao dịch",
            ])
        else:
            recommendations.append(
                "Vui lòng kiểm tra trực tiếp nguồn chính thức để có thông tin mới nhất"
            )

        # Add info about what sources showed
        if verified_sources:
            stale_dates = set()
            for src in verified_sources:
                if src.get("timestamp"):
                    try:
                        ts = datetime.fromisoformat(src["timestamp"])
                        stale_dates.add(ts.strftime("%d/%m/%Y"))
                    except:
                        pass
            if stale_dates:
                recommendations.append(
                    f"Các nguồn kiểm tra chỉ có dữ liệu từ: {', '.join(sorted(stale_dates))}"
                )

        return recommendations

    def _build_freshness_summary(
        self,
        verified_sources: List[Dict],
        has_fresh_data: bool,
        target_date,
        query_type: str,
    ) -> str:
        """Build human-readable freshness summary."""
        if not verified_sources:
            return "Không thể xác minh độ mới của dữ liệu - không fetch được nguồn nào."

        if has_fresh_data:
            fresh_count = sum(1 for s in verified_sources if s.get("is_fresh"))
            return (
                f"✅ TÌM THẤY DỮ LIỆU MỚI: {fresh_count}/{len(verified_sources)} nguồn có dữ liệu "
                f"cập nhật cho ngày {target_date.strftime('%d/%m/%Y')}."
            )

        # No fresh data
        timestamps = []
        for src in verified_sources:
            if src.get("timestamp"):
                try:
                    ts = datetime.fromisoformat(src["timestamp"])
                    timestamps.append(ts)
                except:
                    pass

        if timestamps:
            newest = max(timestamps)
            age_days = (datetime.now() - newest).days
            return (
                f"⚠️ CẢNH BÁO: KHÔNG TÌM THẤY DỮ LIỆU NGÀY {target_date.strftime('%d/%m/%Y')}. "
                f"Dữ liệu mới nhất từ {newest.strftime('%d/%m/%Y')} (cách đây {age_days} ngày). "
                f"Các nguồn có thể chưa cập nhật hoặc đang ngoài giờ giao dịch."
            )

        return (
            f"⚠️ CẢNH BÁO: Không thể xác định thời gian cập nhật từ các nguồn. "
            "Vui lòng kiểm tra trực tiếp nguồn chính thức."
        )

    def _create_llm_context(self, data: Dict[str, Any]) -> str:
        """Create formatted context for LLM."""
        lines = [
            "═══════════════════════════════════════════════════════════",
            "🔍 ENHANCED WEB SEARCH RESULT (with Freshness Validation)",
            "═══════════════════════════════════════════════════════════",
            "",
            f"Query: {data['query']}",
            f"Query Type: {data['query_type']}",
            f"Expected Date: {data['expected_date']}",
            f"Max Age: {data['max_age_hours']} hours",
            "",
        ]

        # Freshness Summary (CRITICAL)
        has_fresh = data.get("has_fresh_data", False)
        icon = "✅" if has_fresh else "⚠️"
        lines.extend([
            "─────────────────────────────────────────────────────────",
            f"{icon} FRESHNESS VALIDATION RESULT",
            "─────────────────────────────────────────────────────────",
            data.get("freshness_summary", "Unknown"),
            "",
        ])

        # Recommendations if not fresh
        recommendations = data.get("recommendations", [])
        if recommendations:
            lines.extend([
                "💡 KHUYẾN NGHỊ:",
            ])
            for rec in recommendations:
                lines.append(f"  • {rec}")
            lines.append("")

        # Freshest source details
        freshest = data.get("freshest_source")
        if freshest:
            lines.extend([
                "─────────────────────────────────────────────────────────",
                "📊 NGUỒN TỐT NHẤT TÌM ĐƯỢC",
                "─────────────────────────────────────────────────────────",
                f"URL: {freshest.get('url', 'N/A')}",
                f"Title: {freshest.get('title', 'N/A')}",
                f"Tier: {freshest.get('source_tier', 'N/A')}",
                f"Fresh: {'Yes' if freshest.get('is_fresh') else 'No'}",
                f"Timestamp: {freshest.get('timestamp', 'N/A')}",
                f"Message: {freshest.get('freshness_message', 'N/A')}",
            ])

            if freshest.get("prices"):
                prices = freshest["prices"]
                lines.append("\n💰 PRICES FOUND:")
                if prices.get("buy"):
                    lines.append(f"  Mua vào: {prices['buy']}")
                if prices.get("sell"):
                    lines.append(f"  Bán ra: {prices['sell']}")
                if prices.get("all_prices"):
                    lines.append(f"  All: {', '.join(prices['all_prices'][:5])}")

            lines.append("")

        # All verified sources
        verified = data.get("verified_sources", [])
        if verified:
            lines.extend([
                "─────────────────────────────────────────────────────────",
                f"📋 ALL VERIFIED SOURCES ({len(verified)})",
                "─────────────────────────────────────────────────────────",
            ])
            for i, src in enumerate(verified, 1):
                fresh_mark = "✅" if src.get("is_fresh") else "⚠️"
                lines.append(
                    f"[{i}] {fresh_mark} {src.get('source_tier', 'D')} | "
                    f"{src.get('timestamp', 'No timestamp')}"
                )
                lines.append(f"    URL: {src.get('url', 'N/A')}")
                lines.append(f"    {src.get('freshness_message', '')[:100]}")
                lines.append("")

        # Search answer (from OpenAI)
        search_results = data.get("search_results", {})
        answer = search_results.get("answer", "")
        if answer:
            lines.extend([
                "─────────────────────────────────────────────────────────",
                "🤖 WEB SEARCH SYNTHESIZED ANSWER",
                "─────────────────────────────────────────────────────────",
                "⚠️ NOTE: This is from search index (may be cached/stale).",
                "Use 'VERIFIED SOURCES' above for fresh data.",
                "",
                answer[:2000],
                "",
            ])

        lines.extend([
            "═══════════════════════════════════════════════════════════",
        ])

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        tool = EnhancedWebSearchTool()

        print("Testing EnhancedWebSearchTool...")

        result = await tool.execute(
            query="giá vàng Mi Hồng hôm nay 28/01/2026",
            expected_date="2026-01-28",
            fetch_top_n=3,
        )

        if result.status == "success":
            print(f"Success!")
            print(f"Has fresh data: {result.data.get('has_fresh_data')}")
            print(f"Summary: {result.data.get('freshness_summary')}")
            print(f"Verified sources: {len(result.data.get('verified_sources', []))}")

            freshest = result.data.get("freshest_source")
            if freshest:
                print(f"Freshest: {freshest.get('url')}")
                print(f"Timestamp: {freshest.get('timestamp')}")
        else:
            print(f"Error: {result.error}")

    asyncio.run(test())
