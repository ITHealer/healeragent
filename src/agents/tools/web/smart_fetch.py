"""
Smart Fetch Tool

Fetches and parses web pages with timestamp validation and source quality scoring.
Designed to solve the problem of stale data from web search indices.

Key Features:
- Direct URL fetching (bypasses search index delay)
- Timestamp extraction and validation
- Source quality tiering (official > aggregator > news)
- Freshness gate (rejects stale data with clear messaging)
- Structured data extraction (tables, prices, etc.)

Use Cases:
- Real-time price data (gold, stocks, crypto)
- Time-sensitive information that requires today's data
- Verification of search results against actual page content
"""

import os
import re
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output,
)

logger = logging.getLogger(__name__)

# Thread pool for sync operations
_fetch_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="smart_fetch_")


@dataclass
class FetchResult:
    """Result of a page fetch operation."""
    success: bool
    url: str
    content: str = ""
    title: str = ""
    timestamp: Optional[datetime] = None
    timestamp_source: str = ""  # Where timestamp was found
    tables: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    error: str = ""
    fetch_method: str = ""

    def __post_init__(self):
        if self.tables is None:
            self.tables = []
        if self.metadata is None:
            self.metadata = {}


class SmartFetchTool(BaseTool):
    """
    Smart Fetch Tool for real-time web content with freshness validation.

    Unlike web search which uses cached indices, this tool:
    1. Fetches the actual page content directly
    2. Extracts and validates timestamps
    3. Scores source quality
    4. Rejects stale data with clear messaging

    Use for:
    - Real-time price queries (gold, commodities)
    - Time-sensitive data that must be from today
    - Verification of search results
    """

    # Source quality tiers
    TIER_A_OFFICIAL = 1.0  # Official sources
    TIER_B_AGGREGATOR = 0.8  # Reliable aggregators
    TIER_C_NEWS = 0.6  # News sites
    TIER_D_OTHER = 0.4  # Other sources

    # Known trusted domains by tier
    TRUSTED_DOMAINS = {
        # Tier A: Official Vietnamese gold shops
        "mihong.vn": TIER_A_OFFICIAL,
        "sjc.com.vn": TIER_A_OFFICIAL,
        "doji.vn": TIER_A_OFFICIAL,
        "pnj.com.vn": TIER_A_OFFICIAL,
        "btmc.vn": TIER_A_OFFICIAL,

        # Tier B: Price aggregators
        "webgia.com": TIER_B_AGGREGATOR,
        "giavang.org": TIER_B_AGGREGATOR,
        "tygia.com": TIER_B_AGGREGATOR,
        "vietstock.vn": TIER_B_AGGREGATOR,

        # Tier C: News sites
        "vnexpress.net": TIER_C_NEWS,
        "baonghean.vn": TIER_C_NEWS,
        "dantri.com.vn": TIER_C_NEWS,
        "thanhnien.vn": TIER_C_NEWS,
    }

    # Vietnamese date patterns
    VN_DATE_PATTERNS = [
        # "28/01/2026", "28-01-2026"
        r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
        # "28 tháng 01 năm 2026"
        r'(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})',
        # "Ngày 28/01/2026"
        r'[Nn]gày\s*(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
        # "Cập nhật: 28/01/2026 10:30"
        r'[Cc]ập\s*nhật[:\s]*(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
        # ISO format "2026-01-28"
        r'(\d{4})-(\d{2})-(\d{2})',
    ]

    # Price extraction patterns for Vietnamese gold
    VN_GOLD_PRICE_PATTERNS = [
        # "Mua vào: 179,400,000" or "Mua: 179.400.000"
        r'[Mm]ua(?:\s*vào)?[:\s]*([0-9.,]+)',
        # "Bán ra: 181,400,000" or "Bán: 181.400.000"
        r'[Bb]án(?:\s*ra)?[:\s]*([0-9.,]+)',
        # Table format with triệu/lượng
        r'(\d{2,3}[.,]\d{3}[.,]\d{3})',
    ]

    DEFAULT_TIMEOUT = 30
    MAX_CONTENT_LENGTH = 100000  # 100KB max

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        # Setup requests session with retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Cloudscraper for protected sites
        self.scraper = cloudscraper.create_scraper() if CLOUDSCRAPER_AVAILABLE else None

        # Define tool schema
        self.schema = ToolSchema(
            name="smartFetch",
            category="web",
            description=(
                "Fetch and parse a web page directly with timestamp validation. "
                "Use for real-time data that must be current (prices, rates, etc.). "
                "Unlike web search, this fetches the actual page content and validates freshness. "
                "Returns structured data with timestamp, tables, and quality score."
            ),
            capabilities=[
                "Direct page fetching (bypasses search index)",
                "Timestamp extraction and validation",
                "Table/price data extraction",
                "Source quality scoring",
                "Freshness gate (warns if data is stale)",
            ],
            limitations=[
                "Single URL at a time",
                "Cannot access login-required pages",
                "May be blocked by some anti-bot protection",
            ],
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="The URL to fetch and parse",
                    required=True,
                ),
                ToolParameter(
                    name="expected_date",
                    type="string",
                    description=(
                        "Expected date of the data (format: YYYY-MM-DD or DD/MM/YYYY). "
                        "If provided, will validate that page data matches this date. "
                        "Default: today's date."
                    ),
                    required=False,
                ),
                ToolParameter(
                    name="max_age_hours",
                    type="integer",
                    description=(
                        "Maximum age of data in hours. If page timestamp is older, "
                        "will flag as stale. Default: 24 hours."
                    ),
                    required=False,
                    default=24,
                ),
                ToolParameter(
                    name="extract_tables",
                    type="boolean",
                    description="Whether to extract HTML tables as structured data. Default: true.",
                    required=False,
                    default=True,
                ),
            ],
            returns={
                "url": "string - The fetched URL",
                "title": "string - Page title",
                "content": "string - Extracted text content",
                "timestamp": "string - Detected timestamp on page (ISO format)",
                "timestamp_source": "string - Where timestamp was found",
                "is_fresh": "boolean - Whether data meets freshness requirements",
                "freshness_message": "string - Human-readable freshness status",
                "tables": "array - Extracted tables as structured data",
                "source_tier": "string - Source quality tier (A/B/C/D)",
                "source_score": "number - Source quality score (0-1)",
                "prices": "object - Extracted price data (if detected)",
            },
            requires_symbol=False,
        )

    async def execute(
        self,
        url: str,
        expected_date: Optional[str] = None,
        max_age_hours: int = 24,
        extract_tables: bool = True,
        **kwargs,
    ) -> ToolOutput:
        """
        Fetch and parse a URL with timestamp validation.
        """
        start_time = datetime.now()

        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return create_error_output(
                    tool_name="smartFetch",
                    error=f"Invalid URL: {url}",
                    metadata={},
                )
        except Exception as e:
            return create_error_output(
                tool_name="smartFetch",
                error=f"URL parsing error: {e}",
                metadata={},
            )

        # Parse expected date
        target_date = self._parse_date(expected_date) if expected_date else datetime.now().date()

        self.logger.info(f"[smartFetch] Fetching: {url}")
        self.logger.info(f"[smartFetch] Expected date: {target_date}, Max age: {max_age_hours}h")

        try:
            # Run fetch in thread pool
            loop = asyncio.get_event_loop()
            fetch_result = await asyncio.wait_for(
                loop.run_in_executor(
                    _fetch_executor,
                    self._fetch_page,
                    url,
                ),
                timeout=self.DEFAULT_TIMEOUT,
            )

            if not fetch_result.success:
                return create_error_output(
                    tool_name="smartFetch",
                    error=fetch_result.error,
                    metadata={"url": url},
                )

            # Extract tables if requested
            tables = []
            if extract_tables and fetch_result.content:
                tables = self._extract_tables_from_html(fetch_result.metadata.get("html", ""))

            # Calculate source quality
            domain = parsed.netloc.lower().replace("www.", "")
            source_score = self.TRUSTED_DOMAINS.get(domain, self.TIER_D_OTHER)
            source_tier = self._score_to_tier(source_score)

            # Validate freshness
            is_fresh, freshness_msg = self._validate_freshness(
                fetch_result.timestamp,
                target_date,
                max_age_hours,
            )

            # Extract prices (for Vietnamese gold price pages)
            prices = self._extract_prices(fetch_result.content, fetch_result.metadata.get("html", ""))

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            result_data = {
                "url": url,
                "title": fetch_result.title,
                "content": fetch_result.content[:10000] if fetch_result.content else "",
                "timestamp": fetch_result.timestamp.isoformat() if fetch_result.timestamp else None,
                "timestamp_source": fetch_result.timestamp_source,
                "is_fresh": is_fresh,
                "freshness_message": freshness_msg,
                "tables": tables[:5],  # Limit to 5 tables
                "source_tier": source_tier,
                "source_score": source_score,
                "prices": prices,
                "fetch_method": fetch_result.fetch_method,
                "execution_time_ms": int(execution_time),
            }

            self.logger.info(f"[smartFetch] SUCCESS in {int(execution_time)}ms")
            self.logger.info(f"[smartFetch] Timestamp: {result_data['timestamp']}, Fresh: {is_fresh}")
            self.logger.info(f"[smartFetch] Source: {source_tier} ({source_score})")

            return create_success_output(
                tool_name="smartFetch",
                data=result_data,
                formatted_context=self._create_llm_context(result_data),
                metadata={
                    "url": url,
                    "execution_time_ms": int(execution_time),
                    "is_fresh": is_fresh,
                    "source_tier": source_tier,
                },
            )

        except asyncio.TimeoutError:
            return create_error_output(
                tool_name="smartFetch",
                error=f"Timeout after {self.DEFAULT_TIMEOUT}s",
                metadata={"url": url},
            )
        except Exception as e:
            self.logger.error(f"[smartFetch] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name="smartFetch",
                error=str(e),
                metadata={"url": url},
            )

    def _fetch_page(self, url: str) -> FetchResult:
        """Fetch page content using multiple methods."""
        methods = [
            ("requests", self._fetch_with_requests),
            ("cloudscraper", self._fetch_with_cloudscraper),
        ]

        last_error = ""
        for method_name, fetch_func in methods:
            try:
                html = fetch_func(url)
                if html:
                    return self._parse_page(url, html, method_name)
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"[smartFetch] {method_name} failed: {e}")
                continue

        return FetchResult(
            success=False,
            url=url,
            error=f"All fetch methods failed. Last error: {last_error}",
        )

    def _fetch_with_requests(self, url: str) -> str:
        """Fetch using requests library."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        response = self.session.get(url, headers=headers, timeout=self.DEFAULT_TIMEOUT)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        return response.text

    def _fetch_with_cloudscraper(self, url: str) -> str:
        """Fetch using cloudscraper for protected sites."""
        if not self.scraper:
            raise RuntimeError("Cloudscraper not available")

        response = self.scraper.get(url, timeout=self.DEFAULT_TIMEOUT)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        return response.text

    def _parse_page(self, url: str, html: str, method: str) -> FetchResult:
        """Parse HTML and extract content, title, timestamp."""
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Extract title
        title = ""
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Extract main content
        content = ""
        main_selectors = ['article', 'main', '.content', '#content', '.article-content', '.post-content']

        for selector in main_selectors:
            if selector.startswith('.') or selector.startswith('#'):
                main_elem = soup.select_one(selector)
            else:
                main_elem = soup.find(selector)

            if main_elem:
                content = main_elem.get_text(separator='\n', strip=True)
                break

        if not content:
            content = soup.get_text(separator='\n', strip=True)

        # Clean content
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content[:self.MAX_CONTENT_LENGTH]

        # Extract timestamp
        timestamp, timestamp_source = self._extract_timestamp(html, soup)

        return FetchResult(
            success=True,
            url=url,
            content=content,
            title=title,
            timestamp=timestamp,
            timestamp_source=timestamp_source,
            fetch_method=method,
            metadata={"html": html[:50000]},  # Keep HTML for table extraction
        )

    def _extract_timestamp(self, html: str, soup: BeautifulSoup) -> Tuple[Optional[datetime], str]:
        """Extract timestamp from page."""

        # Method 1: Check meta tags
        meta_tags = [
            ('property', 'article:modified_time'),
            ('property', 'article:published_time'),
            ('name', 'date'),
            ('name', 'last-modified'),
            ('itemprop', 'dateModified'),
            ('itemprop', 'datePublished'),
        ]

        for attr, value in meta_tags:
            meta = soup.find('meta', {attr: value})
            if meta and meta.get('content'):
                try:
                    dt = self._parse_datetime(meta['content'])
                    if dt:
                        return dt, f"meta[{attr}={value}]"
                except:
                    pass

        # Method 2: Check time/datetime elements
        time_elem = soup.find('time', {'datetime': True})
        if time_elem:
            try:
                dt = self._parse_datetime(time_elem['datetime'])
                if dt:
                    return dt, "time[datetime]"
            except:
                pass

        # Method 3: Search for Vietnamese date patterns in content
        for pattern in self.VN_DATE_PATTERNS:
            matches = re.findall(pattern, html)
            if matches:
                for match in matches:
                    try:
                        if len(match) == 3:
                            if len(match[0]) == 4:  # ISO format YYYY-MM-DD
                                year, month, day = match
                            else:  # DD/MM/YYYY
                                day, month, year = match
                            dt = datetime(int(year), int(month), int(day))
                            return dt, f"pattern:{pattern[:30]}"
                    except:
                        continue

        # Method 4: Look for "Cập nhật" text near timestamps
        update_patterns = [
            r'[Cc]ập\s*nhật[:\s]*(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\s*(\d{1,2}):(\d{2})',
            r'[Ll]ần\s*cuối[:\s]*(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',
        ]

        for pattern in update_patterns:
            match = re.search(pattern, html)
            if match:
                groups = match.groups()
                try:
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                    if len(groups) > 3:
                        hour, minute = int(groups[3]), int(groups[4])
                        dt = datetime(year, month, day, hour, minute)
                    else:
                        dt = datetime(year, month, day)
                    return dt, "update_pattern"
                except:
                    pass

        return None, "not_found"

    def _parse_datetime(self, date_str: str) -> Optional[datetime]:
        """Parse various datetime formats."""
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%d-%m-%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except:
                continue
        return None

    def _parse_date(self, date_str: str):
        """Parse date string to date object."""
        if not date_str:
            return datetime.now().date()

        formats = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except:
                continue
        return datetime.now().date()

    def _validate_freshness(
        self,
        page_timestamp: Optional[datetime],
        target_date,
        max_age_hours: int,
    ) -> Tuple[bool, str]:
        """Validate data freshness."""
        now = datetime.now()

        if page_timestamp is None:
            return False, (
                "KHÔNG TÌM THẤY TIMESTAMP trên trang. "
                "Không thể xác minh độ mới của dữ liệu. "
                "Vui lòng kiểm tra trực tiếp nguồn hoặc liên hệ nhà cung cấp."
            )

        # Check if timestamp is today
        if page_timestamp.date() == target_date:
            age_hours = (now - page_timestamp).total_seconds() / 3600
            if age_hours <= max_age_hours:
                return True, (
                    f"DỮ LIỆU MỚI: Cập nhật lúc {page_timestamp.strftime('%H:%M %d/%m/%Y')} "
                    f"({int(age_hours)} giờ trước)"
                )

        # Check age
        age = now - page_timestamp
        age_hours = age.total_seconds() / 3600
        age_days = age.days

        if age_hours <= max_age_hours:
            return True, (
                f"Dữ liệu từ {page_timestamp.strftime('%d/%m/%Y %H:%M')} "
                f"({int(age_hours)} giờ trước)"
            )

        # Data is stale
        return False, (
            f"⚠️ DỮ LIỆU CŨ: Trang hiển thị dữ liệu từ {page_timestamp.strftime('%d/%m/%Y')} "
            f"(cách đây {age_days} ngày, {int(age_hours)} giờ). "
            f"Yêu cầu dữ liệu ngày {target_date.strftime('%d/%m/%Y')} nhưng trang chưa cập nhật. "
            "Vui lòng kiểm tra nguồn khác hoặc liên hệ trực tiếp nhà cung cấp."
        )

    def _extract_tables_from_html(self, html: str) -> List[Dict[str, Any]]:
        """Extract HTML tables as structured data."""
        tables = []
        soup = BeautifulSoup(html, 'html.parser')

        for table in soup.find_all('table')[:10]:  # Limit to 10 tables
            headers = []
            rows = []

            # Extract headers
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            else:
                first_row = table.find('tr')
                if first_row:
                    ths = first_row.find_all('th')
                    if ths:
                        headers = [th.get_text(strip=True) for th in ths]

            # Extract rows
            tbody = table.find('tbody') or table
            for tr in tbody.find_all('tr')[:50]:  # Limit rows
                cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if cells and (not headers or cells != headers):
                    rows.append(cells)

            if rows:
                tables.append({
                    "headers": headers,
                    "rows": rows[:20],  # Limit to 20 rows
                    "row_count": len(rows),
                })

        return tables

    def _extract_prices(self, content: str, html: str) -> Dict[str, Any]:
        """Extract price data from content."""
        prices = {}

        # Look for Vietnamese gold price patterns
        buy_match = re.search(r'[Mm]ua(?:\s*vào)?[:\s]*([0-9.,]+)', content)
        if buy_match:
            prices['buy'] = self._clean_price(buy_match.group(1))

        sell_match = re.search(r'[Bb]án(?:\s*ra)?[:\s]*([0-9.,]+)', content)
        if sell_match:
            prices['sell'] = self._clean_price(sell_match.group(1))

        # Try to extract from tables
        soup = BeautifulSoup(html, 'html.parser')
        for table in soup.find_all('table')[:5]:
            text = table.get_text()
            if 'SJC' in text or 'Mi Hồng' in text or 'vàng' in text.lower():
                prices['source_table'] = True
                # Extract all price-like numbers
                price_matches = re.findall(r'(\d{2,3}[.,]\d{3}[.,]\d{3})', text)
                if price_matches:
                    prices['all_prices'] = list(set(price_matches))[:10]
                break

        return prices if prices else None

    def _clean_price(self, price_str: str) -> str:
        """Clean and normalize price string."""
        return price_str.replace(',', '.').replace(' ', '')

    def _score_to_tier(self, score: float) -> str:
        """Convert score to tier letter."""
        if score >= self.TIER_A_OFFICIAL:
            return "A (Official)"
        elif score >= self.TIER_B_AGGREGATOR:
            return "B (Aggregator)"
        elif score >= self.TIER_C_NEWS:
            return "C (News)"
        return "D (Other)"

    def _create_llm_context(self, data: Dict[str, Any]) -> str:
        """Create formatted context for LLM."""
        lines = [
            "═══════════════════════════════════════════════════════════",
            "📄 SMART FETCH RESULT",
            "═══════════════════════════════════════════════════════════",
            "",
            f"🔗 URL: {data['url']}",
            f"📰 Title: {data.get('title', 'N/A')}",
            f"🏷️ Source Tier: {data['source_tier']}",
            "",
        ]

        # Freshness status (CRITICAL)
        is_fresh = data.get('is_fresh', False)
        freshness_icon = "✅" if is_fresh else "⚠️"
        lines.extend([
            "─────────────────────────────────────────────────────────",
            f"{freshness_icon} FRESHNESS STATUS",
            "─────────────────────────────────────────────────────────",
            data.get('freshness_message', 'Unknown'),
            "",
        ])

        if data.get('timestamp'):
            lines.append(f"📅 Timestamp: {data['timestamp']}")
            lines.append(f"   (Found via: {data.get('timestamp_source', 'unknown')})")
            lines.append("")

        # Prices if found
        if data.get('prices'):
            lines.extend([
                "─────────────────────────────────────────────────────────",
                "💰 EXTRACTED PRICES",
                "─────────────────────────────────────────────────────────",
            ])
            prices = data['prices']
            if prices.get('buy'):
                lines.append(f"   Mua vào: {prices['buy']}")
            if prices.get('sell'):
                lines.append(f"   Bán ra: {prices['sell']}")
            if prices.get('all_prices'):
                lines.append(f"   All prices found: {', '.join(prices['all_prices'][:5])}")
            lines.append("")

        # Tables
        if data.get('tables'):
            lines.extend([
                "─────────────────────────────────────────────────────────",
                "📊 EXTRACTED TABLES",
                "─────────────────────────────────────────────────────────",
            ])
            for i, table in enumerate(data['tables'][:3], 1):
                lines.append(f"\nTable {i} ({table.get('row_count', 0)} rows):")
                if table.get('headers'):
                    lines.append(f"  Headers: {' | '.join(table['headers'][:5])}")
                for row in table.get('rows', [])[:5]:
                    lines.append(f"  Row: {' | '.join(str(c)[:20] for c in row[:5])}")
            lines.append("")

        # Content preview
        content = data.get('content', '')
        if content:
            lines.extend([
                "─────────────────────────────────────────────────────────",
                "📝 CONTENT PREVIEW",
                "─────────────────────────────────────────────────────────",
                content[:2000],
                "..." if len(content) > 2000 else "",
            ])

        lines.extend([
            "",
            "═══════════════════════════════════════════════════════════",
        ])

        return "\n".join(lines)


# Standalone test
if __name__ == "__main__":
    import asyncio

    async def test():
        tool = SmartFetchTool()

        print("Testing SmartFetchTool...")

        # Test with a Vietnamese gold price page
        result = await tool.execute(
            url="https://webgia.com/gia-vang/mi-hong/",
            expected_date="2026-01-28",
            max_age_hours=24,
        )

        if result.status == "success":
            print(f"Success!")
            print(f"Fresh: {result.data.get('is_fresh')}")
            print(f"Message: {result.data.get('freshness_message')}")
            print(f"Timestamp: {result.data.get('timestamp')}")
            print(f"Source Tier: {result.data.get('source_tier')}")
            if result.data.get('prices'):
                print(f"Prices: {result.data['prices']}")
        else:
            print(f"Error: {result.error}")

    asyncio.run(test())
