"""
Markdown Formatter
==================

Formats TaskResult into readable markdown using LLM for natural language output.

Features:
- Uses LLM to generate natural markdown in target language
- Supports multiple languages (vi, en, zh, ja, ko, etc.)
- Structured sections for each symbol
- Citations with clickable links

Usage:
    from src.news_aggregator.services.markdown_formatter import MarkdownFormatter

    formatter = MarkdownFormatter()
    markdown_content = await formatter.format_result(task_result)
"""

import json
import logging
from datetime import datetime
from typing import Optional

from src.news_aggregator.schemas.task import TaskResult

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Prompts
# =============================================================================

FORMAT_SYSTEM_PROMPT = """You are a professional financial report formatter.
Your task is to convert structured analysis data into a beautiful, readable markdown report.

Output Guidelines:
- Write in the specified target language (natural, professional tone)
- Use markdown formatting: headers, bullet points, bold, links
- Include emoji indicators for sentiment (🟢 bullish, 🔴 bearish, ⚪ neutral, 🟡 mixed)
- Format prices with proper currency symbols and separations
- Include all citations as markdown links [title](url)
- Structure: Title → Summary → Overall Sentiment → Per-Symbol Analysis → Sources

IMPORTANT: Output ONLY the markdown text, no code blocks or explanations."""

FORMAT_USER_PROMPT = """Convert this analysis result to a beautiful markdown report.

**Target Language:** {target_language}
**Date:** {date}

**Analysis Data (JSON):**
```json
{data_json}
```

Generate a professional markdown report in {language_name}. Include:
1. Report title
2. Executive summary (if available)
3. Overall market sentiment with emoji
4. Key themes
5. For each symbol:
   - Price and changes (with 📈/📉 emoji)
   - Sentiment indicator
   - Key insights with citation numbers [1], [2], etc.
   - Short-term and long-term outlook
   - Risk factors (with ⚠️ emoji)
   - Sources as markdown links
6. Processing time footer

Output only the markdown text."""


# Language names for prompt
LANGUAGE_NAMES = {
    "vi": "Vietnamese (Tiếng Việt)",
    "en": "English",
    "zh": "Chinese (中文)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
    "th": "Thai (ภาษาไทย)",
    "id": "Indonesian (Bahasa Indonesia)",
    "ms": "Malay (Bahasa Melayu)",
}


class MarkdownFormatter:
    """
    Formats TaskResult as markdown using LLM for natural language output.

    Uses the configured LLM to generate reports in the target language.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: Optional[str] = None,
    ):
        """
        Initialize formatter.

        Args:
            model_name: LLM model to use (default from settings)
            provider_type: LLM provider (default from settings)
        """
        self.logger = logger
        self._llm_provider = None

        # Load from settings if not provided
        try:
            from src.utils.config import settings
            self.model_name = model_name or settings.AGENT_MODEL or "gpt-4o-mini"
            self.provider_type = provider_type or settings.AGENT_PROVIDER or "openai"
        except ImportError:
            self.model_name = model_name or "gpt-4o-mini"
            self.provider_type = provider_type or "openai"

    def _get_llm_provider(self):
        """Lazy load LLM provider."""
        if self._llm_provider is None:
            try:
                from src.helpers.llm_helper import LLMGeneratorProvider
                self._llm_provider = LLMGeneratorProvider()
            except ImportError:
                self.logger.error("[MarkdownFormatter] LLM provider not available")
                raise
        return self._llm_provider

    def _get_api_key(self) -> Optional[str]:
        """Get API key based on provider type."""
        try:
            from src.utils.config import settings
            provider = self.provider_type.lower()

            if provider == "openai":
                return settings.OPENAI_API_KEY
            elif provider in ("gemini", "google"):
                return settings.GEMINI_API_KEY
            elif provider == "openrouter":
                return settings.OPENROUTER_API_KEY
            else:
                return getattr(settings, f"{provider.upper()}_API_KEY", None)
        except ImportError:
            import os
            return os.getenv("OPENAI_API_KEY")

    def _prepare_data_for_llm(self, result: TaskResult) -> str:
        """
        Prepare TaskResult data as JSON for LLM.

        Args:
            result: TaskResult to format

        Returns:
            JSON string with essential data
        """
        # Build simplified data structure for LLM
        data = {
            "title": result.title,
            "generated_at": result.generated_at.isoformat() if result.generated_at else datetime.utcnow().isoformat(),
            "target_language": result.target_language,
            "overall_sentiment": result.overall_sentiment,
            "key_themes": result.key_themes,
            "summary": result.summary,
            "processing_time_ms": result.processing_time_ms,
            "user_prompt": result.prompt,
            "analyses": []
        }

        # Add symbol analyses
        for analysis in result.analyses:
            symbol_data = {
                "symbol": analysis.symbol,
                "display_name": analysis.display_name,
                "sentiment": analysis.sentiment,
                "sentiment_score": analysis.sentiment_score,
                "short_term_outlook": analysis.short_term_outlook,
                "long_term_outlook": analysis.long_term_outlook,
                "risk_factors": analysis.risk_factors,
                "key_insights": [
                    {
                        "text": ins.text,
                        "source_indices": ins.source_indices,
                        "sentiment": ins.sentiment
                    }
                    for ins in analysis.key_insights
                ],
                "sources": [
                    {
                        "index": src.index,
                        "title": src.title,
                        "url": src.url,
                        "source": src.source,
                        "published_at": src.published_at.strftime("%d/%m/%Y") if src.published_at else None
                    }
                    for src in analysis.sources
                ]
            }

            # Add market data if available
            if analysis.market_data:
                md = analysis.market_data
                symbol_data["market_data"] = {
                    "current_price": md.current_price,
                    "currency": md.currency,
                    "volume": md.volume,
                    "market_cap": md.market_cap,
                    "changes": [
                        {
                            "period": c.period,
                            "change_percent": c.change_percent
                        }
                        for c in md.changes
                    ]
                }

            data["analyses"].append(symbol_data)

        return json.dumps(data, ensure_ascii=False, indent=2)

    async def format_result(self, result: TaskResult) -> str:
        """
        Format TaskResult as markdown using LLM.

        Args:
            result: TaskResult with analyses

        Returns:
            Formatted markdown string in target language
        """
        if not result.analyses:
            return self._format_empty_result(result)

        target_language = result.target_language or "vi"
        language_name = LANGUAGE_NAMES.get(target_language, f"Language code: {target_language}")

        # Prepare data for LLM
        data_json = self._prepare_data_for_llm(result)

        # Build prompt
        user_prompt = FORMAT_USER_PROMPT.format(
            target_language=target_language.upper(),
            language_name=language_name,
            date=datetime.utcnow().strftime("%d/%m/%Y"),
            data_json=data_json,
        )

        try:
            # Get LLM provider
            llm_provider = self._get_llm_provider()
            api_key = self._get_api_key()

            messages = [
                {"role": "system", "content": FORMAT_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            response = await llm_provider.generate_response(
                model_name=self.model_name,
                messages=messages,
                provider_type=self.provider_type,
                api_key=api_key,
                temperature=0.3,
                enable_thinking=False,
            )

            # Extract content
            if isinstance(response, dict):
                content = response.get("content") or response.get("text") or ""
            else:
                content = str(response)

            # Clean up response (remove code blocks if present)
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            self.logger.info(
                f"[MarkdownFormatter] Generated markdown | "
                f"lang={target_language} | length={len(content)}"
            )

            return content

        except Exception as e:
            self.logger.error(f"[MarkdownFormatter] LLM error: {e}")
            # Fallback to simple format
            return self._format_fallback(result)

    def _format_empty_result(self, result: TaskResult) -> str:
        """Format result when no analyses available."""
        messages = {
            "vi": "Không có dữ liệu phân tích.",
            "en": "No analysis data available.",
            "zh": "没有可用的分析数据。",
            "ja": "分析データがありません。",
            "ko": "분석 데이터가 없습니다.",
        }

        lang = result.target_language or "vi"
        msg = messages.get(lang, messages["en"])

        return f"# {result.title}\n\n{msg}"

    def _format_fallback(self, result: TaskResult) -> str:
        """Simple fallback format when LLM fails."""
        lines = [f"# {result.title}", ""]

        if result.summary:
            lines.extend([result.summary, ""])

        lines.append(f"**Sentiment:** {result.overall_sentiment}")

        if result.key_themes:
            lines.append(f"**Themes:** {', '.join(result.key_themes)}")

        lines.append("")

        for analysis in result.analyses:
            lines.append(f"## {analysis.symbol} - {analysis.display_name}")
            lines.append(f"**Sentiment:** {analysis.sentiment}")

            if analysis.key_insights:
                lines.append("")
                for insight in analysis.key_insights:
                    lines.append(f"- {insight.text}")

            lines.append("")

        if result.processing_time_ms:
            lines.append(f"*Processing time: {result.processing_time_ms}ms*")

        return "\n".join(lines)


async def format_error_markdown(
    request_id: int,
    error_message: str,
    job_id: Optional[str] = None,
    target_language: str = "vi",
) -> str:
    """
    Format error result as markdown.

    Args:
        request_id: Original request ID
        error_message: Error description
        job_id: Optional job ID
        target_language: Target language for error message

    Returns:
        Formatted markdown error message
    """
    error_titles = {
        "vi": "Lỗi Xử Lý Yêu Cầu",
        "en": "Request Processing Error",
        "zh": "请求处理错误",
        "ja": "リクエスト処理エラー",
        "ko": "요청 처리 오류",
    }

    error_labels = {
        "vi": {"request": "Mã yêu cầu", "job": "Mã công việc", "detail": "Chi tiết lỗi", "time": "Thời gian"},
        "en": {"request": "Request ID", "job": "Job ID", "detail": "Error Details", "time": "Time"},
        "zh": {"request": "请求ID", "job": "作业ID", "detail": "错误详情", "time": "时间"},
        "ja": {"request": "リクエストID", "job": "ジョブID", "detail": "エラー詳細", "time": "時間"},
        "ko": {"request": "요청 ID", "job": "작업 ID", "detail": "오류 세부정보", "time": "시간"},
    }

    title = error_titles.get(target_language, error_titles["en"])
    labels = error_labels.get(target_language, error_labels["en"])

    lines = [
        f"# ❌ {title}",
        "",
        f"**{labels['request']}:** {request_id}",
    ]

    if job_id:
        lines.append(f"**{labels['job']}:** {job_id}")

    lines.extend([
        "",
        f"## {labels['detail']}",
        "",
        f"```\n{error_message}\n```",
        "",
        f"*{labels['time']}: {datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC*"
    ])

    return "\n".join(lines)


# =============================================================================
# Convenience functions
# =============================================================================

_formatter: Optional[MarkdownFormatter] = None


def get_markdown_formatter() -> MarkdownFormatter:
    """Get singleton formatter instance."""
    global _formatter
    if _formatter is None:
        _formatter = MarkdownFormatter()
    return _formatter


async def format_task_result_markdown(result: TaskResult) -> str:
    """
    Format TaskResult as markdown (convenience function).

    Args:
        result: TaskResult to format

    Returns:
        Formatted markdown string
    """
    formatter = get_markdown_formatter()
    return await formatter.format_result(result)
