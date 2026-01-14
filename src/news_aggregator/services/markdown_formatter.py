"""
Markdown Formatter
==================

Formats TaskResult into readable markdown text for BE .NET callback.

Features:
- Clean, readable markdown output
- Structured sections for each symbol
- Citations with clickable links
- Sentiment indicators with emoji
- Market data summary

Usage:
    from src.news_aggregator.services.markdown_formatter import format_task_result_markdown

    markdown_content = format_task_result_markdown(task_result)
"""

from datetime import datetime
from typing import List, Optional

from src.news_aggregator.schemas.task import (
    TaskResult,
    SymbolAnalysis,
    SymbolInsight,
    NewsSource,
    MarketData,
    PriceChange,
)


def format_task_result_markdown(result: TaskResult) -> str:
    """
    Format TaskResult as markdown text.

    Args:
        result: TaskResult with analyses

    Returns:
        Formatted markdown string
    """
    lines = []

    # Title
    lines.append(f"# {result.title}")
    lines.append("")
    lines.append(f"*{result.generated_at.strftime('%d/%m/%Y %H:%M')} UTC*")
    lines.append("")

    # User prompt if provided
    if result.prompt:
        lines.append(f"> **Yêu cầu:** {result.prompt}")
        lines.append("")

    # Executive Summary
    if result.summary:
        lines.append("## Tóm Tắt")
        lines.append("")
        lines.append(result.summary)
        lines.append("")

    # Overall Sentiment
    sentiment_emoji = _get_sentiment_emoji(result.overall_sentiment)
    lines.append(f"**Xu hướng chung:** {sentiment_emoji} {result.overall_sentiment}")
    lines.append("")

    # Key Themes
    if result.key_themes:
        lines.append("**Chủ đề chính:** " + ", ".join(result.key_themes))
        lines.append("")

    lines.append("---")
    lines.append("")

    # Symbol Analyses
    for analysis in result.analyses:
        lines.extend(_format_symbol_analysis(analysis))
        lines.append("")

    # Error if any
    if result.error:
        lines.append("## Lỗi")
        lines.append("")
        lines.append(f"```\n{result.error}\n```")
        lines.append("")

    # Metadata footer
    if result.processing_time_ms:
        lines.append("---")
        lines.append(f"*Thời gian xử lý: {result.processing_time_ms}ms*")

    return "\n".join(lines)


def _format_symbol_analysis(analysis: SymbolAnalysis) -> List[str]:
    """Format single symbol analysis to markdown lines."""
    lines = []

    # Symbol Header
    sentiment_emoji = _get_sentiment_emoji(analysis.sentiment)
    lines.append(f"## {analysis.symbol} - {analysis.display_name}")
    lines.append("")

    # Market Data
    if analysis.market_data:
        lines.extend(_format_market_data(analysis.market_data))
        lines.append("")

    # Sentiment
    score_bar = _format_score_bar(analysis.sentiment_score)
    lines.append(f"**Sentiment:** {sentiment_emoji} {analysis.sentiment} ({score_bar})")
    lines.append("")

    # Key Insights
    if analysis.key_insights:
        lines.append("### Điểm Nổi Bật")
        lines.append("")
        for insight in analysis.key_insights:
            lines.append(_format_insight(insight))
        lines.append("")

    # Outlook
    if analysis.short_term_outlook or analysis.long_term_outlook:
        lines.append("### Triển Vọng")
        lines.append("")
        if analysis.short_term_outlook:
            lines.append(f"- **Ngắn hạn (1-7 ngày):** {analysis.short_term_outlook}")
        if analysis.long_term_outlook:
            lines.append(f"- **Dài hạn (1-3 tháng):** {analysis.long_term_outlook}")
        lines.append("")

    # Risk Factors
    if analysis.risk_factors:
        lines.append("### Rủi Ro Cần Lưu Ý")
        lines.append("")
        for risk in analysis.risk_factors:
            lines.append(f"- ⚠️ {risk}")
        lines.append("")

    # Sources
    if analysis.sources:
        lines.append("### Nguồn Tham Khảo")
        lines.append("")
        for source in analysis.sources:
            lines.append(_format_source(source))
        lines.append("")

    return lines


def _format_market_data(data: MarketData) -> List[str]:
    """Format market data to markdown lines."""
    lines = []

    # Current price
    lines.append(f"**Giá hiện tại:** ${data.current_price:,.2f} {data.currency}")

    # Price changes
    if data.changes:
        changes_text = []
        for change in data.changes:
            emoji = "📈" if change.change_percent >= 0 else "📉"
            sign = "+" if change.change_percent >= 0 else ""
            changes_text.append(f"{change.period}: {emoji} {sign}{change.change_percent:.2f}%")
        lines.append(f"**Biến động:** {' | '.join(changes_text)}")

    # Volume
    if data.volume:
        lines.append(f"**Khối lượng:** {data.volume:,}")

    # Market cap
    if data.market_cap:
        if data.market_cap >= 1e12:
            cap_str = f"${data.market_cap/1e12:.2f}T"
        elif data.market_cap >= 1e9:
            cap_str = f"${data.market_cap/1e9:.2f}B"
        else:
            cap_str = f"${data.market_cap/1e6:.2f}M"
        lines.append(f"**Vốn hóa:** {cap_str}")

    return lines


def _format_insight(insight: SymbolInsight) -> str:
    """Format single insight with citations."""
    # Sentiment icon
    sentiment_icon = ""
    if insight.sentiment:
        sentiment_map = {
            "bullish": "🟢",
            "bearish": "🔴",
            "neutral": "⚪"
        }
        sentiment_icon = sentiment_map.get(insight.sentiment.lower(), "")

    # Citations
    citations = ""
    if insight.source_indices:
        citations = " " + " ".join([f"[{i}]" for i in insight.source_indices])

    return f"- {sentiment_icon} {insight.text}{citations}"


def _format_source(source: NewsSource) -> str:
    """Format source as markdown link."""
    date_str = ""
    if source.published_at:
        date_str = f" ({source.published_at.strftime('%d/%m')})"

    return f"[{source.index}] [{source.title}]({source.url}) - *{source.source}*{date_str}"


def _get_sentiment_emoji(sentiment: str) -> str:
    """Get emoji for sentiment."""
    sentiment_map = {
        "BULLISH": "🟢",
        "BEARISH": "🔴",
        "NEUTRAL": "⚪",
        "MIXED": "🟡"
    }
    return sentiment_map.get(sentiment.upper(), "⚪")


def _format_score_bar(score: float) -> str:
    """
    Format sentiment score as visual bar.
    Score: -1.0 (bearish) to +1.0 (bullish)
    """
    # Normalize to 0-10 scale
    normalized = int((score + 1) * 5)
    normalized = max(0, min(10, normalized))

    if score >= 0:
        return f"+{score:.1f}"
    else:
        return f"{score:.1f}"


def format_error_markdown(
    request_id: int,
    error_message: str,
    job_id: Optional[str] = None
) -> str:
    """
    Format error result as markdown.

    Args:
        request_id: Original request ID
        error_message: Error description
        job_id: Optional job ID

    Returns:
        Formatted markdown error message
    """
    lines = [
        "# Lỗi Xử Lý Yêu Cầu",
        "",
        f"**Request ID:** {request_id}",
    ]

    if job_id:
        lines.append(f"**Job ID:** {job_id}")

    lines.extend([
        "",
        "## Chi Tiết Lỗi",
        "",
        f"```\n{error_message}\n```",
        "",
        f"*{datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC*"
    ])

    return "\n".join(lines)
