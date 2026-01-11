"""
HTML Preprocessor Utility
=========================

Utilities for cleaning and simplifying HTML content before LLM processing.
Reduces token usage by removing unnecessary elements and attributes.

Features:
- Remove scripts, styles, SVGs, and media elements
- Strip unnecessary HTML attributes
- Extract clean text content
- Minify HTML output
- Link extraction for citations

Usage:
    from src.news_aggregator.utils.html_preprocessor import HTMLPreprocessor

    preprocessor = HTMLPreprocessor()
    result = preprocessor.process(html_content, url)

    # Access clean content
    clean_text = result.clean_text
    title = result.title
    links = result.links
"""

import logging
import re
from typing import Generator, List, Optional, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Comment
from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger(__name__)


class ExtractedContent(BaseModel):
    """
    Structured model for extracted and preprocessed web content.

    Separates raw extraction from LLM summarization for better caching.
    """
    # Source info
    url: str = Field(..., description="Original URL")

    # Extracted content
    clean_text: str = Field(default="", description="Clean text without HTML tags")
    html_slim: str = Field(default="", description="Simplified HTML (optional)")
    title: Optional[str] = Field(None, description="Page title")

    # Metadata
    links: List[str] = Field(default_factory=list, description="Extracted links")
    word_count: int = Field(default=0, description="Word count of clean text")
    char_count: int = Field(default=0, description="Character count of clean text")

    # Processing info
    extraction_success: bool = Field(default=True)
    extraction_method: Optional[str] = Field(None, description="Method used for extraction")
    original_length: int = Field(default=0, description="Original HTML length")
    reduced_length: int = Field(default=0, description="Reduced content length")
    reduction_percent: float = Field(default=0.0, description="Size reduction percentage")

    # Cache key for deduplication
    content_hash: Optional[str] = Field(None, description="Hash for content deduplication")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/news/article",
                "clean_text": "Tesla stock rises 5% after earnings...",
                "title": "Tesla Stock News",
                "word_count": 500,
                "char_count": 3000,
                "reduction_percent": 85.5
            }
        }


class WebPage(BaseModel):
    """
    Raw web page content model (compatible with MetaGPT pattern).
    """
    inner_text: str = Field(default="", description="Text content from page")
    html: str = Field(default="", description="Raw HTML content")
    url: str = Field(..., description="Page URL")

    _soup: Optional[BeautifulSoup] = PrivateAttr(default=None)
    _title: Optional[str] = PrivateAttr(default=None)

    @property
    def soup(self) -> BeautifulSoup:
        """Lazy-loaded BeautifulSoup instance."""
        if self._soup is None:
            self._soup = BeautifulSoup(self.html, "html.parser")
        return self._soup

    @property
    def title(self) -> str:
        """Extract page title."""
        if self._title is None:
            title_tag = self.soup.find("title")
            self._title = title_tag.text.strip() if title_tag else ""
        return self._title

    def get_links(self) -> Generator[str, None, None]:
        """Extract all links from the page."""
        for a_tag in self.soup.find_all("a", href=True):
            href = a_tag["href"]
            parsed = urlparse(href)
            if not parsed.scheme and parsed.path:
                yield urljoin(self.url, href)
            elif href.startswith(("http://", "https://")):
                yield urljoin(self.url, href)


class HTMLPreprocessor:
    """
    HTML preprocessor for reducing content before LLM processing.

    Removes unnecessary elements to reduce token usage while
    preserving meaningful content for summarization.
    """

    # Elements to completely remove
    REMOVE_TAGS: Set[str] = {
        "script", "style", "noscript", "iframe", "frame",
        "svg", "img", "video", "audio", "canvas", "map",
        "object", "embed", "source", "track",
        "footer", "nav", "aside", "header",
        "form", "input", "button", "select", "textarea",
        "advertisement", "ad", "ads",
    }

    # Attributes to keep (remove all others)
    KEEP_ATTRS: Set[str] = {"class", "id", "href", "title", "alt"}

    # Ad-related class/id patterns
    AD_PATTERNS: List[str] = [
        r"ad[-_]?", r"advertisement", r"sponsor", r"promo",
        r"banner", r"popup", r"modal", r"cookie", r"consent",
        r"newsletter", r"subscribe", r"sidebar", r"widget",
        r"social[-_]?share", r"comment", r"related[-_]?post",
    ]

    # Maximum content length (chars) to send to LLM
    MAX_CONTENT_LENGTH: int = 15000

    def __init__(
        self,
        remove_links: bool = False,
        max_content_length: int = None,
    ):
        """
        Initialize preprocessor.

        Args:
            remove_links: Whether to remove href attributes
            max_content_length: Maximum content length to return
        """
        self.remove_links = remove_links
        self.max_content_length = max_content_length or self.MAX_CONTENT_LENGTH
        self._ad_pattern = re.compile(
            "|".join(self.AD_PATTERNS),
            re.IGNORECASE
        )

    def process(self, html: str, url: str) -> ExtractedContent:
        """
        Process HTML content and return structured extraction result.

        Args:
            html: Raw HTML content
            url: Source URL

        Returns:
            ExtractedContent with clean text and metadata
        """
        if not html:
            return ExtractedContent(
                url=url,
                extraction_success=False,
                extraction_method="empty_input"
            )

        original_length = len(html)

        try:
            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Extract title before cleaning
            title = self._extract_title(soup)

            # Extract links before cleaning
            links = self._extract_links(soup, url)

            # Clean the HTML
            clean_soup = self._clean_soup(soup)

            # Get clean text
            clean_text = self._get_clean_text(clean_soup)

            # Get simplified HTML (optional)
            html_slim = self._get_slim_html(clean_soup)

            # Truncate if needed
            if len(clean_text) > self.max_content_length:
                clean_text = clean_text[:self.max_content_length] + "..."

            # Calculate metrics
            word_count = len(clean_text.split())
            char_count = len(clean_text)
            reduced_length = len(clean_text)
            reduction_percent = (
                (original_length - reduced_length) / original_length * 100
                if original_length > 0 else 0
            )

            # Generate content hash for caching
            import hashlib
            content_hash = hashlib.md5(clean_text.encode()).hexdigest()[:16]

            return ExtractedContent(
                url=url,
                clean_text=clean_text,
                html_slim=html_slim,
                title=title,
                links=links[:10],  # Limit to 10 links
                word_count=word_count,
                char_count=char_count,
                extraction_success=True,
                extraction_method="html_preprocessor",
                original_length=original_length,
                reduced_length=reduced_length,
                reduction_percent=round(reduction_percent, 2),
                content_hash=content_hash,
            )

        except Exception as e:
            logger.error(f"[HTMLPreprocessor] Error processing {url}: {e}")
            return ExtractedContent(
                url=url,
                extraction_success=False,
                extraction_method="error",
            )

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract page title from various sources."""
        # Try meta og:title first (usually cleaner)
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

        # Try title tag
        title_tag = soup.find("title")
        if title_tag:
            return title_tag.get_text(strip=True)

        # Try h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)

        return None

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract relevant links from the page."""
        links = []
        seen = set()

        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]

            # Skip anchors and javascript
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Resolve relative URLs
            full_url = urljoin(base_url, href)

            # Skip if already seen
            if full_url in seen:
                continue
            seen.add(full_url)

            # Only include http(s) links
            if full_url.startswith(("http://", "https://")):
                links.append(full_url)

        return links

    def _clean_soup(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Clean soup by removing unnecessary elements."""
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove unwanted tags
        for tag in soup.find_all(self.REMOVE_TAGS):
            tag.decompose()

        # Remove ad-related elements by class/id
        for element in soup.find_all(True):
            classes = element.get("class", [])
            element_id = element.get("id", "")

            # Check class names
            class_str = " ".join(classes) if isinstance(classes, list) else str(classes)
            if self._ad_pattern.search(class_str) or self._ad_pattern.search(element_id):
                element.decompose()
                continue

            # Remove unnecessary attributes
            attrs_to_remove = [
                attr for attr in element.attrs
                if attr not in self.KEEP_ATTRS
            ]
            for attr in attrs_to_remove:
                del element[attr]

            # Remove href if configured
            if self.remove_links and "href" in element.attrs:
                del element["href"]

        return soup

    def _get_clean_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from soup."""
        # Get text with separator
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line]  # Remove empty lines

        # Join and normalize whitespace
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 newlines
        text = re.sub(r" {2,}", " ", text)  # Max 1 space

        return text.strip()

    def _get_slim_html(self, soup: BeautifulSoup) -> str:
        """Get minified HTML representation."""
        try:
            # Try to use htmlmin if available
            import htmlmin
            html = soup.decode()
            return htmlmin.minify(
                html,
                remove_comments=True,
                remove_empty_space=True,
            )
        except ImportError:
            # Fallback to simple string representation
            return str(soup)[:5000]  # Limit size


# Singleton instance
_preprocessor: Optional[HTMLPreprocessor] = None


def get_html_preprocessor() -> HTMLPreprocessor:
    """Get singleton preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = HTMLPreprocessor()
    return _preprocessor


def preprocess_html(html: str, url: str) -> ExtractedContent:
    """
    Convenience function to preprocess HTML content.

    Args:
        html: Raw HTML content
        url: Source URL

    Returns:
        ExtractedContent with clean text and metadata
    """
    return get_html_preprocessor().process(html, url)
