"""
Synthesis Handler - Combines 5 analysis steps into comprehensive investment report

Multi-LLM Call Strategy:
1. LLM Call 1: Executive Summary
2. LLM Call 2-4: Detailed Sections (parallel)
3. LLM Call 5: Web Search Enrichment (optional)
4. LLM Call 6: Final Recommendation with Scoring

Output: Markdown report with quantitative scoring

Usage:
    from src.handlers.synthesis_handler import synthesis_handler

    async for event in synthesis_handler.synthesize(symbol, model_name, provider_type, api_key):
        if event["type"] == "content":
            print(event["chunk"])
        elif event["type"] == "data":
            print(event["data"])  # Scoring data
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, AsyncGenerator

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.scanner_cache_helper import (
    get_all_scanner_results,
    save_scanner_result,
    SCANNER_STEPS
)
from src.services.scoring_service import scoring_service
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory
from src.agents.tools.web.web_search import WebSearchTool

logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPTS FOR SYNTHESIS
# =============================================================================

EXEC_SUMMARY_SYSTEM_PROMPT = """You are a senior investment analyst creating an executive summary.

Your task is to synthesize 5 analysis sections into a clear, actionable executive summary.

RULES:
- Be concise (3-4 paragraphs max)
- Lead with the main conclusion/recommendation
- Highlight key findings from each analysis type
- Mention critical risk factors
- Use specific numbers when available
- Match the user's language if specified

OUTPUT FORMAT:
## Executive Summary

[Opening paragraph with main recommendation]

[Key bullish/bearish factors paragraph]

[Risk and considerations paragraph]

[Action items / next steps paragraph]
"""

DETAIL_SECTION_SYSTEM_PROMPT = """You are a financial analyst writing a detailed section of an investment report.

Based on the provided analysis data, write a focused section with:
1. Key findings (bullet points)
2. Supporting data/numbers
3. Implications for investors
4. Limitations or caveats

Keep the section focused and evidence-based. Cite specific numbers from the data.
Match the user's language if specified.
"""

FINAL_RECOMMENDATION_SYSTEM_PROMPT = """You are a senior portfolio manager making a final investment recommendation.

Based on all the analysis provided, create a clear recommendation with:

1. **RECOMMENDATION**: BUY / HOLD / SELL with percentage distribution
2. **CONFIDENCE LEVEL**: High / Medium / Low with explanation
3. **TIME HORIZON**: Short (1-4 weeks) / Medium (1-3 months) / Long (6-12 months)
4. **PRICE TARGETS**: Entry zone, Target 1, Target 2, Stop Loss
5. **KEY RISKS**: Top 3 risks to monitor
6. **CATALYSTS**: Upcoming events that could impact the stock

Be specific and actionable. Use numbers from the analysis.
Match the user's language if specified.
"""

WEB_ENRICHMENT_SYSTEM_PROMPT = """You are a financial news analyst synthesizing web search results.

Your task is to create a "Latest News & Catalysts" section for an investment report.

RULES:
1. Focus on news that impacts investment decisions
2. Identify upcoming catalysts (earnings, product launches, regulatory events)
3. Note any Fed/macro factors that could affect the stock
4. ALWAYS include markdown links to sources: [Title](URL)
5. Be concise - bullet points preferred
6. Distinguish between factual news and speculation/opinion
7. Match the user's language if specified

OUTPUT FORMAT:
## Latest News & Catalysts

### Recent Developments
- Key news point 1 [Source](URL)
- Key news point 2 [Source](URL)

### Upcoming Catalysts
- Event 1 (date if known)
- Event 2

### Macro/Sector Factors (if relevant)
- Factor 1
- Factor 2

### Market Sentiment
Brief summary of overall sentiment from news sources.
"""


# =============================================================================
# SYNTHESIS HANDLER
# =============================================================================

class SynthesisHandler(LoggerMixin):
    """
    Orchestrates multi-LLM synthesis from 5 analysis steps.

    Flow:
    1. Check cache for available step results
    2. Run missing steps if needed
    3. Execute 6 LLM calls for comprehensive synthesis
    4. Generate Markdown report with scoring
    """

    def __init__(self):
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()
        self._market_scanner_handler = None

    @property
    def market_scanner_handler(self):
        """Lazy load market scanner handler to avoid circular imports."""
        if self._market_scanner_handler is None:
            from src.handlers.market_scanner_handler import market_scanner_handler
            self._market_scanner_handler = market_scanner_handler
        return self._market_scanner_handler

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    async def synthesize(
        self,
        symbol: str,
        model_name: str,
        provider_type: str,
        api_key: str,
        target_language: Optional[str] = None,
        run_missing_steps: bool = True,
        include_web_search: bool = False,
        timeframe: str = "1Y",
        benchmark: str = "SPY"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main synthesis method with streaming support.

        Args:
            symbol: Stock symbol to analyze
            model_name: LLM model name
            provider_type: LLM provider (openai, gemini)
            api_key: API key for LLM
            target_language: Response language (vi, en)
            run_missing_steps: Auto-run missing analysis steps
            include_web_search: Include web search enrichment
            timeframe: Technical analysis timeframe
            benchmark: RS benchmark symbol

        Yields:
            Dict with type: "progress", "content", "data", "error", "done"
        """
        symbol = symbol.upper().strip()
        start_time = datetime.now()

        try:
            # =================================================================
            # PHASE 1: Check cache for available results
            # =================================================================
            yield {
                "type": "progress",
                "step": "cache_check",
                "message": f"Checking cached analysis for {symbol}..."
            }

            cache_status = await get_all_scanner_results(symbol)
            available = cache_status.get("available", [])
            missing = cache_status.get("missing", [])
            step_data = cache_status.get("data", {})

            yield {
                "type": "progress",
                "step": "cache_status",
                "message": f"Found {len(available)}/5 cached steps",
                "available": available,
                "missing": missing
            }

            # =================================================================
            # PHASE 2: Run missing steps if enabled
            # =================================================================
            if missing and run_missing_steps:
                yield {
                    "type": "progress",
                    "step": "running_missing",
                    "message": f"Running {len(missing)} missing analyses..."
                }

                missing_results = await self._run_missing_steps(
                    symbol=symbol,
                    missing_steps=missing,
                    model_name=model_name,
                    provider_type=provider_type,
                    api_key=api_key,
                    target_language=target_language,
                    timeframe=timeframe,
                    benchmark=benchmark
                )

                # Merge with cached data
                for step_name, result in missing_results.items():
                    step_data[step_name] = result

                yield {
                    "type": "progress",
                    "step": "missing_complete",
                    "message": f"Completed running missing steps"
                }

            # Check if we have minimum data
            if len(step_data) < 2:
                yield {
                    "type": "error",
                    "error": "Insufficient data for synthesis. Run at least 2 analysis steps first."
                }
                return

            # =================================================================
            # PHASE 3: Calculate Scoring
            # =================================================================
            yield {
                "type": "progress",
                "step": "scoring",
                "message": "Calculating investment score..."
            }

            scoring_result = scoring_service.calculate_composite_score(step_data)

            yield {
                "type": "data",
                "section": "scoring",
                "data": scoring_result
            }

            # =================================================================
            # PHASE 4: Generate Report Header
            # =================================================================
            report_header = self._generate_report_header(
                symbol=symbol,
                scoring=scoring_result,
                available_steps=list(step_data.keys())
            )

            yield {
                "type": "content",
                "section": "header",
                "content": report_header
            }

            # =================================================================
            # PHASE 5: Multi-LLM Synthesis
            # =================================================================
            yield {
                "type": "progress",
                "step": "synthesis_start",
                "message": "Starting AI synthesis..."
            }

            # LLM Call 1: Executive Summary
            yield {
                "type": "progress",
                "step": "llm_exec_summary",
                "message": "Generating executive summary..."
            }

            exec_summary = await self._generate_executive_summary(
                symbol=symbol,
                step_data=step_data,
                scoring=scoring_result,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key,
                target_language=target_language
            )

            yield {
                "type": "content",
                "section": "executive_summary",
                "content": exec_summary
            }

            # LLM Calls 2-4: Detailed Sections (parallel)
            yield {
                "type": "progress",
                "step": "llm_details",
                "message": "Generating detailed analysis sections..."
            }

            detail_sections = await self._generate_detail_sections(
                symbol=symbol,
                step_data=step_data,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key,
                target_language=target_language
            )

            for section_name, content in detail_sections.items():
                yield {
                    "type": "content",
                    "section": section_name,
                    "content": content
                }

            # LLM Call 5: Web Search Enrichment (optional)
            if include_web_search:
                yield {
                    "type": "progress",
                    "step": "llm_web_search",
                    "message": "Searching for latest news..."
                }

                web_content = await self._generate_web_enrichment(
                    symbol=symbol,
                    step_data=step_data,
                    scoring=scoring_result,
                    model_name=model_name,
                    provider_type=provider_type,
                    api_key=api_key,
                    target_language=target_language
                )

                yield {
                    "type": "content",
                    "section": "web_enrichment",
                    "content": web_content
                }

            # LLM Call 6: Final Recommendation
            yield {
                "type": "progress",
                "step": "llm_final",
                "message": "Generating final recommendation..."
            }

            final_rec = await self._generate_final_recommendation(
                symbol=symbol,
                step_data=step_data,
                scoring=scoring_result,
                exec_summary=exec_summary,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key,
                target_language=target_language
            )

            yield {
                "type": "content",
                "section": "final_recommendation",
                "content": final_rec
            }

            # =================================================================
            # PHASE 6: Generate Report Footer
            # =================================================================
            elapsed = (datetime.now() - start_time).total_seconds()
            report_footer = self._generate_report_footer(
                symbol=symbol,
                elapsed_seconds=elapsed,
                step_data=step_data
            )

            yield {
                "type": "content",
                "section": "footer",
                "content": report_footer
            }

            yield {"type": "done"}

        except Exception as e:
            self.logger.error(f"[Synthesis] Error for {symbol}: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }

    # =========================================================================
    # PHASE 2: RUN MISSING STEPS
    # =========================================================================

    async def _run_missing_steps(
        self,
        symbol: str,
        missing_steps: List[str],
        model_name: str,
        provider_type: str,
        api_key: str,
        target_language: Optional[str],
        timeframe: str,
        benchmark: str
    ) -> Dict[str, Any]:
        """
        Run missing analysis steps in parallel.
        Collects full response (non-streaming) for synthesis.
        """
        results = {}

        # Build tasks for parallel execution
        tasks = []
        step_names = []

        for step in missing_steps:
            if step == "technical":
                task = self._collect_technical(
                    symbol, timeframe, model_name, provider_type, api_key, target_language
                )
            elif step == "position":
                task = self._collect_position(
                    symbol, benchmark, model_name, provider_type, api_key, target_language
                )
            elif step == "risk":
                task = self._collect_risk(
                    symbol, model_name, provider_type, api_key, target_language
                )
            elif step == "sentiment":
                task = self._collect_sentiment(
                    symbol, model_name, provider_type, api_key, target_language
                )
            elif step == "fundamental":
                task = self._collect_fundamental(
                    symbol, model_name, provider_type, api_key, target_language
                )
            else:
                continue

            tasks.append(task)
            step_names.append(step)

        # Execute all tasks in parallel
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, step_name in enumerate(step_names):
                result = task_results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"[Synthesis] Step {step_name} failed: {result}")
                    results[step_name] = {
                        "content": f"Error: {str(result)}",
                        "raw_data": None,
                        "error": True
                    }
                else:
                    results[step_name] = result
                    # Also save to cache for future use
                    await save_scanner_result(symbol, step_name, result)

        return results

    async def _collect_technical(
        self, symbol: str, timeframe: str, model_name: str,
        provider_type: str, api_key: str, target_language: Optional[str]
    ) -> Dict[str, Any]:
        """Collect technical analysis (non-streaming)."""
        chunks = []
        async for chunk in self.market_scanner_handler.stream_technical_analysis(
            symbol=symbol,
            timeframe=timeframe,
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key,
            target_language=target_language
        ):
            chunks.append(chunk)

        content = "".join(chunks)
        raw_data = await self.market_scanner_handler.get_technical_analysis(symbol, timeframe)

        return {
            "content": content,
            "raw_data": raw_data.get("raw_data"),
            "cached_at": datetime.now().isoformat()
        }

    async def _collect_position(
        self, symbol: str, benchmark: str, model_name: str,
        provider_type: str, api_key: str, target_language: Optional[str]
    ) -> Dict[str, Any]:
        """Collect market position analysis (non-streaming)."""
        chunks = []
        async for chunk in self.market_scanner_handler.stream_market_position(
            symbol=symbol,
            benchmark=benchmark,
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key,
            target_language=target_language
        ):
            chunks.append(chunk)

        content = "".join(chunks)
        raw_data = await self.market_scanner_handler.get_market_position(symbol, benchmark)

        return {
            "content": content,
            "raw_data": raw_data.get("raw_data"),
            "sector_context": raw_data.get("sector_context"),  # Include sector_context
            "cached_at": datetime.now().isoformat()
        }

    async def _collect_risk(
        self, symbol: str, model_name: str,
        provider_type: str, api_key: str, target_language: Optional[str]
    ) -> Dict[str, Any]:
        """Collect risk analysis (non-streaming)."""
        chunks = []
        async for chunk in self.market_scanner_handler.stream_risk_analysis(
            symbol=symbol,
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key,
            target_language=target_language
        ):
            chunks.append(chunk)

        content = "".join(chunks)
        raw_data = await self.market_scanner_handler.get_risk_analysis(symbol)

        return {
            "content": content,
            "raw_data": raw_data.get("raw_data"),
            "cached_at": datetime.now().isoformat()
        }

    async def _collect_sentiment(
        self, symbol: str, model_name: str,
        provider_type: str, api_key: str, target_language: Optional[str]
    ) -> Dict[str, Any]:
        """Collect sentiment analysis (non-streaming)."""
        chunks = []
        async for chunk in self.market_scanner_handler.stream_sentiment_news(
            symbol=symbol,
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key,
            target_language=target_language
        ):
            chunks.append(chunk)

        content = "".join(chunks)
        raw_data = await self.market_scanner_handler.get_sentiment_news(symbol)

        return {
            "content": content,
            "raw_data": raw_data.get("raw_data"),
            "cached_at": datetime.now().isoformat()
        }

    async def _collect_fundamental(
        self, symbol: str, model_name: str,
        provider_type: str, api_key: str, target_language: Optional[str]
    ) -> Dict[str, Any]:
        """Collect fundamental analysis (non-streaming)."""
        # Fundamental analysis is in a different handler
        from src.handlers.fundamental_analysis_handler import FundamentalAnalysisHandler
        from src.services.tool_call_service import ToolCallService

        fund_handler = FundamentalAnalysisHandler()
        tool_service = ToolCallService()

        # Get comprehensive data
        comprehensive_data = await fund_handler.generate_comprehensive_fundamental_data(
            symbol=symbol,
            tool_service=tool_service
        )

        # Stream analysis
        chunks = []
        async for chunk in fund_handler.stream_comprehensive_analysis(
            symbol=symbol,
            report=comprehensive_data.get("fundamental_report", {}),
            growth_data=comprehensive_data.get("growth_data", []),
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key,
            target_language=target_language
        ):
            chunks.append(chunk)

        content = "".join(chunks)

        return {
            "content": content,
            "raw_data": comprehensive_data,
            "cached_at": datetime.now().isoformat()
        }

    # =========================================================================
    # PHASE 4-5: LLM GENERATION METHODS
    # =========================================================================

    async def _generate_executive_summary(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any],
        model_name: str,
        provider_type: str,
        api_key: str,
        target_language: Optional[str]
    ) -> str:
        """Generate executive summary using LLM."""
        # Build summary of all steps
        step_summaries = []
        for step_name in ["technical", "position", "risk", "sentiment", "fundamental"]:
            data = step_data.get(step_name, {})
            content = data.get("content", "")
            if content:
                # Truncate each section to keep prompt manageable
                truncated = content[:2000] + "..." if len(content) > 2000 else content
                step_summaries.append(f"### {step_name.upper()}\n{truncated}")

        prompt_parts = [
            f"# SYNTHESIS REQUEST FOR {symbol}",
            "",
            "## SCORING SUMMARY",
            f"Composite Score: {scoring.get('composite_score', 'N/A')}/100",
            f"Recommendation: {scoring.get('recommendation', {}).get('action', 'N/A')}",
            f"Confidence: {scoring.get('recommendation', {}).get('confidence', 'N/A')}%",
            "",
            "## ANALYSIS DATA",
            "\n\n".join(step_summaries),
            "",
            "## YOUR TASK",
            "Create an executive summary based on all the above analysis.",
        ]

        if target_language:
            prompt_parts.append(f"\nIMPORTANT: Respond entirely in {target_language}.")

        prompt = "\n".join(prompt_parts)

        messages = [
            {"role": "system", "content": EXEC_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        # Collect full response
        response_chunks = []
        async for chunk in self.llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key
        ):
            response_chunks.append(chunk)

        return "".join(response_chunks)

    async def _generate_detail_sections(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        model_name: str,
        provider_type: str,
        api_key: str,
        target_language: Optional[str]
    ) -> Dict[str, str]:
        """Generate detailed analysis sections in parallel."""
        sections = {}

        # Group data for 3 parallel calls
        # Call 2: Technical Deep Dive
        # Call 3: Risk & Sentiment
        # Call 4: Fundamental Value

        async def generate_technical_detail():
            tech_data = step_data.get("technical", {}).get("content", "")
            pos_data = step_data.get("position", {}).get("content", "")

            prompt = f"""# TECHNICAL & MARKET POSITION DETAIL FOR {symbol}

## TECHNICAL ANALYSIS
{tech_data[:3000] if tech_data else 'Not available'}

## MARKET POSITION
{pos_data[:2000] if pos_data else 'Not available'}

## YOUR TASK
Write a detailed "Technical & Market Position" section combining trend analysis, momentum, and relative strength insights.
Focus on actionable trading implications."""

            if target_language:
                prompt += f"\n\nIMPORTANT: Respond in {target_language}."

            messages = [
                {"role": "system", "content": DETAIL_SECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            chunks = []
            async for chunk in self.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            ):
                chunks.append(chunk)
            return "".join(chunks)

        async def generate_risk_sentiment_detail():
            risk_data = step_data.get("risk", {}).get("content", "")
            sent_data = step_data.get("sentiment", {}).get("content", "")

            prompt = f"""# RISK & SENTIMENT DETAIL FOR {symbol}

## RISK ANALYSIS
{risk_data[:3000] if risk_data else 'Not available'}

## SENTIMENT & NEWS
{sent_data[:3000] if sent_data else 'Not available'}

## YOUR TASK
Write a detailed "Risk & Sentiment" section combining volatility analysis, stop loss recommendations, and news impact.
Emphasize risk management implications."""

            if target_language:
                prompt += f"\n\nIMPORTANT: Respond in {target_language}."

            messages = [
                {"role": "system", "content": DETAIL_SECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            chunks = []
            async for chunk in self.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            ):
                chunks.append(chunk)
            return "".join(chunks)

        async def generate_fundamental_detail():
            fund_data = step_data.get("fundamental", {}).get("content", "")

            prompt = f"""# FUNDAMENTAL VALUE ASSESSMENT FOR {symbol}

## FUNDAMENTAL ANALYSIS
{fund_data[:4000] if fund_data else 'Not available'}

## YOUR TASK
Write a detailed "Fundamental Value" section covering valuation, growth, profitability, and financial health.
Include specific metrics and how they compare to peers/industry."""

            if target_language:
                prompt += f"\n\nIMPORTANT: Respond in {target_language}."

            messages = [
                {"role": "system", "content": DETAIL_SECTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

            chunks = []
            async for chunk in self.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            ):
                chunks.append(chunk)
            return "".join(chunks)

        # Run all 3 in parallel
        results = await asyncio.gather(
            generate_technical_detail(),
            generate_risk_sentiment_detail(),
            generate_fundamental_detail(),
            return_exceptions=True
        )

        section_names = ["technical_detail", "risk_sentiment", "fundamental_detail"]
        for i, name in enumerate(section_names):
            if isinstance(results[i], Exception):
                sections[name] = f"Error generating section: {results[i]}"
            else:
                sections[name] = results[i]

        return sections

    async def _generate_web_enrichment(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any],
        model_name: str,
        provider_type: str,
        api_key: str,
        target_language: Optional[str]
    ) -> str:
        """
        Generate web search enrichment with smart queries.

        Searches for:
        1. Latest news for the specific symbol
        2. Sector/industry news if sentiment data suggests importance
        3. Fed/macro events that could impact the stock
        4. Upcoming catalysts and events

        Returns formatted markdown with citations.
        """
        web_search = WebSearchTool()
        all_citations = []
        all_answers = []

        # Build smart queries based on available data
        queries = self._build_smart_queries(symbol, step_data, scoring)

        self.logger.info(f"[Synthesis] Web enrichment with {len(queries)} queries for {symbol}")

        # Execute searches in parallel (max 3 queries for performance)
        search_tasks = []
        for query_info in queries[:3]:
            task = web_search.execute(
                query=query_info["query"],
                max_results=query_info.get("max_results", 3),
                use_finance_domains=True
            )
            search_tasks.append((query_info["category"], task))

        # Gather results
        search_results = {}
        for category, task in search_tasks:
            try:
                result = await task
                if result.status == "success" and result.data:
                    search_results[category] = result.data
                    # Collect citations
                    for citation in result.data.get("citations", []):
                        if citation not in all_citations:
                            all_citations.append(citation)
                    # Collect answers
                    if result.data.get("answer"):
                        all_answers.append({
                            "category": category,
                            "answer": result.data.get("answer", "")
                        })
            except Exception as e:
                self.logger.warning(f"[Synthesis] Web search failed for {category}: {e}")

        # If no results, return minimal section
        if not search_results:
            return self._format_no_web_results(symbol)

        # Use LLM to synthesize web search results
        synthesis_prompt = self._build_web_synthesis_prompt(
            symbol=symbol,
            search_results=search_results,
            all_answers=all_answers,
            all_citations=all_citations,
            target_language=target_language
        )

        messages = [
            {"role": "system", "content": WEB_ENRICHMENT_SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt}
        ]

        # Generate synthesis
        chunks = []
        try:
            async for chunk in self.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            ):
                chunks.append(chunk)
        except Exception as e:
            self.logger.error(f"[Synthesis] Web synthesis LLM error: {e}")
            return self._format_raw_web_results(symbol, search_results, all_citations)

        synthesized_content = "".join(chunks)

        # Append citation references
        citation_section = self._format_citations(all_citations)

        return f"{synthesized_content}\n\n{citation_section}"

    def _build_smart_queries(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Build smart search queries based on synthesis context.

        Returns list of queries with categories and priorities.
        """
        queries = []

        # Query 1: Always search for latest symbol news
        queries.append({
            "category": "symbol_news",
            "query": f"{symbol} stock latest news today",
            "max_results": 4,
            "priority": 1
        })

        # Query 2: Based on sentiment data - check for catalysts
        sentiment_content = step_data.get("sentiment", {}).get("content", "")
        if sentiment_content:
            # Check if there are earnings, product launches, or other events mentioned
            if any(word in sentiment_content.lower() for word in ["earnings", "report", "quarter", "guidance"]):
                queries.append({
                    "category": "earnings",
                    "query": f"{symbol} earnings results guidance analyst estimates 2025",
                    "max_results": 3,
                    "priority": 2
                })
            elif any(word in sentiment_content.lower() for word in ["launch", "product", "announcement", "release"]):
                queries.append({
                    "category": "catalyst",
                    "query": f"{symbol} new product launch announcement 2025",
                    "max_results": 3,
                    "priority": 2
                })

        # Query 3: Fed/macro if risk score indicates high macro sensitivity
        risk_score = scoring.get("component_scores", {}).get("risk", {}).get("score", 50)
        if risk_score < 45:  # Higher risk = more macro sensitive
            queries.append({
                "category": "macro",
                "query": "Federal Reserve interest rate decision 2025 stock market impact",
                "max_results": 3,
                "priority": 3
            })

        # Query 4: Sector news if position shows sector relevance
        position_content = step_data.get("position", {}).get("content", "")
        if position_content:
            # Extract sector if mentioned
            for sector in ["technology", "healthcare", "financial", "energy", "consumer", "industrial"]:
                if sector in position_content.lower():
                    queries.append({
                        "category": "sector",
                        "query": f"{sector} sector stocks news outlook 2025",
                        "max_results": 2,
                        "priority": 3
                    })
                    break

        # Sort by priority
        queries.sort(key=lambda x: x.get("priority", 99))

        return queries

    def _build_web_synthesis_prompt(
        self,
        symbol: str,
        search_results: Dict[str, Any],
        all_answers: List[Dict[str, str]],
        all_citations: List[Dict[str, str]],
        target_language: Optional[str]
    ) -> str:
        """Build prompt for LLM to synthesize web search results."""
        prompt_parts = [
            f"# WEB SEARCH RESULTS FOR {symbol}",
            "",
            "Synthesize these search results into a coherent 'Latest News & Catalysts' section.",
            "",
        ]

        # Add answers by category
        for item in all_answers:
            prompt_parts.extend([
                f"## {item['category'].upper().replace('_', ' ')}",
                item["answer"][:2000],
                ""
            ])

        # Add citation list
        if all_citations:
            prompt_parts.extend([
                "## AVAILABLE SOURCES (use markdown links when citing)",
                ""
            ])
            for i, c in enumerate(all_citations[:10], 1):
                title = c.get("title", "Untitled")[:60]
                url = c.get("url", "")
                prompt_parts.append(f"[{i}] [{title}]({url})")
            prompt_parts.append("")

        prompt_parts.extend([
            "## YOUR TASK",
            "Write a 'Latest News & Catalysts' section that:",
            "1. Summarizes key recent news impacting the stock",
            "2. Identifies upcoming catalysts or events",
            "3. Notes any Fed/macro factors if relevant",
            "4. INCLUDES markdown links to sources inline",
            "",
            "Format: Use bullet points and include [Source Title](URL) links.",
        ])

        if target_language:
            prompt_parts.append(f"\nIMPORTANT: Respond entirely in {target_language}.")

        return "\n".join(prompt_parts)

    def _format_citations(self, citations: List[Dict[str, str]]) -> str:
        """Format citations as markdown reference list."""
        if not citations:
            return ""

        lines = [
            "### Sources",
            ""
        ]

        for i, c in enumerate(citations[:10], 1):
            title = c.get("title", "Untitled")[:80]
            url = c.get("url", "")
            if url:
                lines.append(f"{i}. [{title}]({url})")

        return "\n".join(lines)

    def _format_no_web_results(self, symbol: str) -> str:
        """Format section when no web results available."""
        return f"""
## Latest News & Catalysts

*Web search did not return results. For the latest news on {symbol}, check:*

- [Bloomberg](https://www.bloomberg.com/quote/{symbol}:US)
- [Reuters](https://www.reuters.com/markets/companies/{symbol}.O/)
- [Yahoo Finance](https://finance.yahoo.com/quote/{symbol})
- Company Investor Relations page
"""

    def _format_raw_web_results(
        self,
        symbol: str,
        search_results: Dict[str, Any],
        citations: List[Dict[str, str]]
    ) -> str:
        """Format raw web results when LLM synthesis fails."""
        lines = [
            "## Latest News & Catalysts",
            "",
            f"Recent news and updates for {symbol}:",
            ""
        ]

        # Add citations as bullet points
        for c in citations[:8]:
            title = c.get("title", "Untitled")[:80]
            url = c.get("url", "")
            if url:
                lines.append(f"- [{title}]({url})")

        lines.append("")
        lines.append("*Review these sources for detailed information.*")

        return "\n".join(lines)

    async def _generate_final_recommendation(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any],
        exec_summary: str,
        model_name: str,
        provider_type: str,
        api_key: str,
        target_language: Optional[str]
    ) -> str:
        """Generate final recommendation using LLM."""
        # Build comprehensive data summary
        rec = scoring.get("recommendation", {})
        components = scoring.get("component_scores", {})
        key_factors = scoring.get("key_factors", [])

        # Format component scores
        component_lines = []
        for name, data in components.items():
            component_lines.append(f"- {name}: {data.get('score', 'N/A')}/100 ({data.get('confidence', 'N/A')} confidence)")

        # Format key factors
        bullish = [f["factor"] for f in key_factors if f.get("impact") == "bullish"]
        bearish = [f["factor"] for f in key_factors if f.get("impact") == "bearish"]

        prompt = f"""# FINAL RECOMMENDATION REQUEST FOR {symbol}

## SCORING DATA
- Composite Score: {scoring.get('composite_score', 'N/A')}/100
- Current Recommendation: {rec.get('action', 'N/A')}
- Distribution: BUY {rec.get('distribution', {}).get('buy', 0)}% | HOLD {rec.get('distribution', {}).get('hold', 0)}% | SELL {rec.get('distribution', {}).get('sell', 0)}%
- Confidence: {rec.get('confidence', 'N/A')}%
- Suggested Time Horizon: {rec.get('time_horizon', 'N/A')}

## COMPONENT SCORES
{chr(10).join(component_lines)}

## KEY FACTORS
Bullish: {', '.join(bullish) if bullish else 'None identified'}
Bearish: {', '.join(bearish) if bearish else 'None identified'}

## EXECUTIVE SUMMARY (for context)
{exec_summary[:2000]}

## YOUR TASK
Based on all the above, write a clear FINAL RECOMMENDATION section with:
1. Your recommendation (BUY/HOLD/SELL with confidence)
2. Price targets (Entry, TP1, TP2, Stop Loss) - use realistic estimates
3. Time horizon recommendation
4. Top 3 risks to monitor
5. Key catalysts to watch

Be specific and actionable."""

        if target_language:
            prompt += f"\n\nIMPORTANT: Respond entirely in {target_language}."

        messages = [
            {"role": "system", "content": FINAL_RECOMMENDATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        chunks = []
        async for chunk in self.llm_provider.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key
        ):
            chunks.append(chunk)

        return "".join(chunks)

    # =========================================================================
    # REPORT GENERATION HELPERS
    # =========================================================================

    def _generate_report_header(
        self,
        symbol: str,
        scoring: Dict[str, Any],
        available_steps: List[str]
    ) -> str:
        """Generate markdown report header."""
        rec = scoring.get("recommendation", {})
        dist = rec.get("distribution", {})

        header = f"""# Comprehensive Investment Report: {symbol}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Steps:** {', '.join(available_steps)}

---

## Investment Score Summary

| Metric | Value |
|--------|-------|
| **Composite Score** | {scoring.get('composite_score', 'N/A')}/100 |
| **Recommendation** | {rec.get('action', 'N/A')} {rec.get('emoji', '')} |
| **Distribution** | BUY {dist.get('buy', 0)}% · HOLD {dist.get('hold', 0)}% · SELL {dist.get('sell', 0)}% |
| **Confidence** | {rec.get('confidence', 'N/A')}% |
| **Time Horizon** | {rec.get('time_horizon', 'N/A')} |

### Component Scores
"""
        # Add component scores table
        components = scoring.get("component_scores", {})
        header += "\n| Component | Score | Weight | Confidence |\n|-----------|-------|--------|------------|\n"
        for name, data in components.items():
            header += f"| {name.title()} | {data.get('score', 'N/A')}/100 | {data.get('weight', 'N/A')} | {data.get('confidence', 'N/A')} |\n"

        # Add key factors
        key_factors = scoring.get("key_factors", [])
        if key_factors:
            header += "\n### Key Factors\n"
            bullish = [f for f in key_factors if f.get("impact") == "bullish"]
            bearish = [f for f in key_factors if f.get("impact") == "bearish"]

            if bullish:
                header += "\n**Bullish:**\n"
                for f in bullish:
                    header += f"- {f.get('factor', 'N/A')} ({f.get('component', '')})\n"

            if bearish:
                header += "\n**Bearish:**\n"
                for f in bearish:
                    header += f"- {f.get('factor', 'N/A')} ({f.get('component', '')})\n"

        header += "\n---\n"
        return header

    def _generate_report_footer(
        self,
        symbol: str,
        elapsed_seconds: float,
        step_data: Dict[str, Any]
    ) -> str:
        """Generate markdown report footer."""
        # Calculate data freshness
        freshness_lines = []
        for step_name in SCANNER_STEPS:
            data = step_data.get(step_name, {})
            cached_at = data.get("cached_at", "N/A")
            if cached_at and cached_at != "N/A":
                freshness_lines.append(f"- {step_name}: {cached_at}")
            else:
                freshness_lines.append(f"- {step_name}: Not available")

        footer = f"""
---

## Disclaimer

This report is generated by AI analysis and should not be considered as financial advice.
Always conduct your own research and consult with a qualified financial advisor before making investment decisions.

Past performance is not indicative of future results. Investments involve risk, including the possible loss of principal.

---

## Report Metadata

- **Symbol:** {symbol}
- **Generation Time:** {elapsed_seconds:.1f} seconds
- **LLM Calls:** 6 (1 summary + 3 parallel details + 1 web + 1 recommendation)

### Data Freshness
{chr(10).join(freshness_lines)}

---
*Report generated by HealerAgent Market Scanner v2.0*
"""
        return footer


# =============================================================================
# MODULE-LEVEL INSTANCE
# =============================================================================

synthesis_handler = SynthesisHandler()
