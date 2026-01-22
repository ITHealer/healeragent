"""
Synthesis Handler V2 - Single LLM Call Architecture (Production Ready)

Key Features:
1. SINGLE LLM CALL - All data in one context for 100% consistency
2. RAW DATA METRICS - Uses structured data, not truncated LLM text
3. DYNAMIC PEER COMPARISON - FMP stock_peers API instead of hardcoded mapping
4. INVESTOR-FOCUSED SECTIONS - Scenario Analysis, Fair Value Assessment
5. BINDING SCORING - LLM must follow the calculated score
6. SPECIFIC CATALYST DATES - Earnings calendar with exact dates
7. HOLDER-SPECIFIC RULES - Exit/reduce triggers for existing positions

Usage:
    from src.handlers.synthesis_handler_v2 import synthesis_handler_v2

    async for event in synthesis_handler_v2.synthesize(symbol, model_name, provider_type, api_key):
        if event["type"] == "content":
            print(event["chunk"])
"""

import asyncio
import logging
import os
import httpx
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, AsyncGenerator

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.scanner_cache_helper import (
    get_all_scanner_results,
    save_scanner_result,
    SCANNER_STEPS
)
from src.services.scoring_service import scoring_service
from src.helpers.llm_helper import LLMGeneratorProvider
from src.agents.tools.web.web_search import WebSearchTool

logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT (ENGLISH ONLY - Production)
# =============================================================================

CONSOLIDATED_SYSTEM_PROMPT = """You are a senior investment analyst creating a comprehensive investment report.

## CRITICAL RULES (MUST FOLLOW)

### 1. SCORING IS BINDING
The composite score and recommendation are PRE-CALCULATED and BINDING. You MUST:
- Use the EXACT recommendation (BUY/HOLD/SELL) from the scoring data
- NOT contradict the scoring in any section
- Align all analysis with the given score

### 2. TRADING PLAN CONSISTENCY
Entry, Stop-Loss, and Targets MUST be mathematically consistent:

**If Score >= 65 (BUY/STRONG BUY) - Use BREAKOUT SYSTEM:**
- Entry: ABOVE current price (breakout confirmation)
- Stop: Based on ATR or swing low FROM ENTRY PRICE
- Calculate risk % from ENTRY price, not current price

**If Score 45-64 (HOLD) - Use WAIT & WATCH:**
- NO entry recommendation (watchlist only)
- Define CONDITIONS for future entry
- Provide specific price levels for upgrade/downgrade

**If Score < 45 (SELL) - Use EXIT/AVOID:**
- For holders: Exit strategy with specific levels
- For new investors: AVOID
- Explain reasoning with data

### 3. STOP-LOSS CALCULATION (CRITICAL)
Stop-loss percentage MUST be calculated FROM ENTRY PRICE, not current price:
- Example: If Entry = $185 and Stop = $169, then risk = (185-169)/185 = 8.6%
- NEVER say "5% from current price" if entry is different from current price

### 4. CATALYST DATES (CRITICAL)
- Include SPECIFIC dates for earnings from provided data
- Format: "Earnings: Feb 25, 2026 (AMC)" - use EXACT date from data
- If no date available, state "Not yet announced"
- Source: FMP Earnings Calendar API

### 5. PEER COMPARISON
- Include comparison table when peer data is provided
- Show P/E, P/S, Revenue Growth, Market Cap
- ADD NOTE: "P/E may be distorted for companies with low/volatile EPS"
- Compare target's valuation position vs peers

### 6. TECHNICAL SCORE CONSISTENCY
Technical score MUST match the indicators:
- ADX < 15 (weak trend): Technical score should be 50-65 max
- ADX 15-25 (moderate trend): Technical score 60-75
- ADX > 25 (strong trend): Technical score can be 70-85
- If MACD bearish + ADX < 15: Technical should be < 60
- Explain score rationale if indicators seem contradictory

### 7. TRADING PLAN FOR HOLDERS (CRITICAL)
Always include specific rules for people ALREADY holding the stock:
- Reduce trigger: specific price level to reduce 50% position
- Exit trigger: specific price level to exit completely
- Trailing stop: how to adjust stop as price moves
- Add trigger: when it's OK to add to position (if applicable)

### 8. SECTOR RANKING METHODOLOGY
When mentioning sector rank:
- State the ranking system: "FMP Sector Performance (1-day ranking)"
- Note the limitation: "This is 1-DAY ranking, different from multi-timeframe RS"
- Provide context: is sector leading, lagging, or neutral?

### 9. WEB CITATIONS (MANDATORY)
When using web search data:
- INLINE citations: "Statement [Source Name](URL)"
- Every claim from web search needs a citation
- Include "## Sources" section at end with numbered list

### 10. SCENARIO ANALYSIS (REQUIRED)
Include Bull/Base/Bear scenarios with:
- Price targets for each scenario
- Probability estimates
- Key triggers for each scenario

### 11. FAIR VALUE ASSESSMENT (REQUIRED)
Compare current price to intrinsic value:
- Graham Value (if available)
- DCF Value (if available)
- State: Premium/Discount to fair value
- Margin of safety calculation

## OUTPUT FORMAT (STRUCTURED)

Generate report with these sections IN ORDER:

### PART 1: DATA ANALYSIS (5 Steps)
1. **Technical Analysis** - Trend, momentum, key levels, signals
2. **Market Position** - RS vs benchmark, sector context
3. **Risk Analysis** - Volatility, VaR, risk metrics
4. **Sentiment Analysis** - Sentiment score, news themes
5. **Fundamental Analysis** - Valuation, growth, peer comparison

### PART 2: INVESTOR-FOCUSED ANALYSIS
6. **Fair Value Assessment** - Intrinsic value vs current price
7. **Scenario Analysis** - Bull/Base/Bear with probabilities

### PART 3: NEWS & CATALYSTS
8. **News & Catalysts** - Latest news with inline citations + Sources section

### PART 4: CONCLUSION & ACTION
9. **Executive Summary** - Key highlights from each step
10. **Final Recommendation** - Action strategy for:
    - NEW investors (no position)
    - EXISTING holders (have position)
    - Conditions to upgrade/downgrade recommendation

RESPOND IN THE LANGUAGE SPECIFIED BY target_language PARAMETER.
"""


# =============================================================================
# SYNTHESIS HANDLER V2 CLASS
# =============================================================================

class SynthesisHandlerV2(LoggerMixin):
    """
    Consolidated single-LLM-call synthesis handler.

    Production-ready with:
    - Raw data metrics instead of truncated LLM text
    - Dynamic FMP stock_peers API
    - Investor-focused sections
    - Holder-specific trading rules
    """

    def __init__(self):
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()
        self._market_scanner_handler = None
        self.fmp_api_key = os.environ.get("FMP_API_KEY")

    @property
    def market_scanner_handler(self):
        """Lazy load market scanner handler."""
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
        target_language: Optional[str] = "vi",
        run_missing_steps: bool = True,
        include_web_search: bool = True,
        timeframe: str = "1Y",
        benchmark: str = "SPY"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Main synthesis method with single consolidated LLM call.

        Uses RAW DATA metrics from each step instead of truncated LLM content.
        """
        symbol = symbol.upper().strip()
        start_time = datetime.now()

        try:
            # =================================================================
            # PHASE 1: Gather all data
            # =================================================================
            yield {
                "type": "progress",
                "step": "gathering_data",
                "message": f"Gathering analysis data for {symbol}..."
            }

            cache_status = await get_all_scanner_results(symbol)
            step_data = cache_status.get("data", {})
            missing = cache_status.get("missing", [])

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

                for step_name, result in missing_results.items():
                    step_data[step_name] = result

            if len(step_data) < 2:
                yield {
                    "type": "error",
                    "error": "Insufficient data for synthesis. Run at least 2 analysis steps first."
                }
                return

            # =================================================================
            # PHASE 2: Calculate scoring (BINDING)
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
            # PHASE 3: Fetch additional data in parallel
            # =================================================================
            yield {
                "type": "progress",
                "step": "enrichment",
                "message": "Fetching earnings, peers, and news..."
            }

            # Parallel fetch: earnings, peers (FMP API), web search
            earnings_task = self._fetch_earnings_calendar(symbol)
            peers_task = self._fetch_peer_comparison_dynamic(symbol)  # NEW: Dynamic FMP API

            web_task = None
            if include_web_search:
                web_task = self._fetch_web_enrichment(symbol, step_data, scoring_result)

            tasks = [earnings_task, peers_task]
            if web_task:
                tasks.append(web_task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            earnings_data = results[0] if not isinstance(results[0], Exception) else None
            peer_data = results[1] if not isinstance(results[1], Exception) else None
            web_data = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None

            # =================================================================
            # PHASE 4: Generate report header
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
            # PHASE 5: Build consolidated prompt with RAW DATA
            # =================================================================
            yield {
                "type": "progress",
                "step": "synthesis",
                "message": "Generating comprehensive report..."
            }

            # Build prompt using RAW DATA metrics (not truncated content)
            consolidated_prompt = self._build_consolidated_prompt_v2(
                symbol=symbol,
                step_data=step_data,
                scoring=scoring_result,
                earnings_data=earnings_data,
                peer_data=peer_data,
                web_data=web_data,
                target_language=target_language
            )

            # Calculate trading plan
            trading_plan = self._calculate_trading_plan(
                symbol=symbol,
                step_data=step_data,
                scoring=scoring_result
            )

            # Calculate scenario analysis
            scenario_analysis = self._calculate_scenario_analysis(
                symbol=symbol,
                step_data=step_data,
                scoring=scoring_result
            )

            # Add trading plan and scenarios to prompt
            consolidated_prompt += f"\n\n{self._format_trading_plan_section(trading_plan)}"
            consolidated_prompt += f"\n\n{self._format_scenario_section(scenario_analysis)}"

            messages = [
                {"role": "system", "content": CONSOLIDATED_SYSTEM_PROMPT},
                {"role": "user", "content": consolidated_prompt}
            ]

            # Stream response
            full_content = []
            async for chunk in self.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            ):
                full_content.append(chunk)
                yield {
                    "type": "content",
                    "section": "report_body",
                    "content": chunk
                }

            # =================================================================
            # PHASE 6: Generate footer
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
            self.logger.error(f"[SynthesisV2] Error for {symbol}: {e}")
            yield {
                "type": "error",
                "error": str(e)
            }

    # =========================================================================
    # DATA FETCHING - EARNINGS CALENDAR
    # =========================================================================

    async def _fetch_earnings_calendar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch earnings calendar with next earnings date.

        Uses TWO FMP endpoints:
        1. /api/v3/earning_calendar - UPCOMING earnings
        2. /stable/earnings - HISTORICAL earnings (beat rate)
        """
        if not self.fmp_api_key:
            self.logger.warning("[Earnings] No FMP API key available")
            return None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                today = datetime.now().date()
                from_date = today.strftime("%Y-%m-%d")
                to_date = (today + timedelta(days=120)).strftime("%Y-%m-%d")

                # Fetch UPCOMING earnings
                calendar_url = "https://financialmodelingprep.com/api/v3/earning_calendar"
                calendar_params = {
                    "from": from_date,
                    "to": to_date,
                    "apikey": self.fmp_api_key
                }

                next_earnings = None
                calendar_response = await client.get(calendar_url, params=calendar_params)

                if calendar_response.status_code == 200:
                    calendar_data = calendar_response.json()
                    if isinstance(calendar_data, list):
                        for item in calendar_data:
                            if item.get("symbol", "").upper() == symbol.upper():
                                next_earnings = {
                                    "date": item.get("date"),
                                    "time": item.get("time", "TBD"),
                                    "eps_estimated": item.get("epsEstimated"),
                                    "revenue_estimated": item.get("revenueEstimated")
                                }
                                break

                # Fetch HISTORICAL earnings for beat rate
                historical_url = "https://financialmodelingprep.com/stable/earnings"
                historical_params = {"symbol": symbol, "apikey": self.fmp_api_key}

                historical = []
                beat_rate = None

                historical_response = await client.get(historical_url, params=historical_params)

                if historical_response.status_code == 200:
                    historical_data = historical_response.json()
                    if isinstance(historical_data, list) and historical_data:
                        sorted_history = sorted(
                            historical_data,
                            key=lambda x: x.get("date", ""),
                            reverse=True
                        )
                        historical = sorted_history[:8]

                        beats = sum(1 for h in historical
                                   if h.get("epsActual") and h.get("epsEstimated")
                                   and h["epsActual"] > h["epsEstimated"])
                        total = len([h for h in historical
                                    if h.get("epsActual") and h.get("epsEstimated")])
                        beat_rate = round(beats / total, 2) if total > 0 else None

                return {
                    "next_earnings_date": next_earnings.get("date") if next_earnings else None,
                    "earnings_time": next_earnings.get("time", "TBD") if next_earnings else "TBD",
                    "eps_estimated": next_earnings.get("eps_estimated") if next_earnings else None,
                    "revenue_estimated": next_earnings.get("revenue_estimated") if next_earnings else None,
                    "fiscal_quarter": self._determine_fiscal_quarter(next_earnings.get("date") if next_earnings else None),
                    "historical": historical[:4],
                    "beat_rate": beat_rate,
                    "source": "FMP Earnings Calendar API"
                }

        except Exception as e:
            self.logger.error(f"[Earnings] Error fetching for {symbol}: {e}")
            return None

    def _determine_fiscal_quarter(self, date_str: Optional[str]) -> str:
        """Determine fiscal quarter from date."""
        if not date_str:
            return "N/A"
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            month = date.month
            year = date.year
            if month in [1, 2, 3]:
                return f"Q4 FY{year}"
            elif month in [4, 5, 6]:
                return f"Q1 FY{year+1}"
            elif month in [7, 8, 9]:
                return f"Q2 FY{year+1}"
            else:
                return f"Q3 FY{year+1}"
        except:
            return "N/A"

    # =========================================================================
    # DATA FETCHING - DYNAMIC PEER COMPARISON (FMP API)
    # =========================================================================

    async def _fetch_peer_comparison_dynamic(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch peer comparison using FMP stock_peers API (dynamic, not hardcoded).

        Endpoint: GET /api/v4/stock_peers?symbol=NVDA
        """
        if not self.fmp_api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                # Step 1: Get peers from FMP API
                peers_url = f"https://financialmodelingprep.com/api/v4/stock_peers"
                peers_params = {"symbol": symbol, "apikey": self.fmp_api_key}

                peers_response = await client.get(peers_url, params=peers_params)

                peers = []
                if peers_response.status_code == 200:
                    peers_data = peers_response.json()
                    if isinstance(peers_data, list) and peers_data:
                        # API returns [{"symbol": "NVDA", "peersList": ["AMD", "INTC", ...]}]
                        peers = peers_data[0].get("peersList", [])[:5]  # Max 5 peers
                        self.logger.info(f"[Peers] FMP API returned peers for {symbol}: {peers}")

                if not peers:
                    self.logger.warning(f"[Peers] No peers found from FMP API for {symbol}")
                    return None

                # Step 2: Fetch metrics for target and peers
                all_symbols = [symbol] + peers

                tasks = []
                for sym in all_symbols:
                    tasks.append(self._fetch_single_company_metrics(client, sym))

                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Parse results
                peer_metrics = []
                target_metrics = None

                for i, result in enumerate(results):
                    if isinstance(result, Exception) or result is None:
                        continue
                    if all_symbols[i] == symbol:
                        target_metrics = result
                    else:
                        peer_metrics.append(result)

                if not target_metrics:
                    return None

                return {
                    "symbol": symbol,
                    "sector": target_metrics.get("sector", "N/A"),
                    "industry": target_metrics.get("industry", "N/A"),
                    "peers": peer_metrics,
                    "target_metrics": target_metrics,
                    "source": "FMP stock_peers API + key-metrics-ttm"
                }

        except Exception as e:
            self.logger.error(f"[Peers] Error fetching for {symbol}: {e}")
            return None

    async def _fetch_single_company_metrics(
        self,
        client: httpx.AsyncClient,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive metrics for a single symbol."""
        try:
            # Fetch key-metrics-ttm
            metrics_url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
            params = {"apikey": self.fmp_api_key}

            metrics_response = await client.get(metrics_url, params=params)

            if metrics_response.status_code != 200:
                return None

            metrics_data = metrics_response.json()
            if not isinstance(metrics_data, list) or not metrics_data:
                return None

            metrics = metrics_data[0]

            # Fetch profile for sector/industry
            profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
            profile_response = await client.get(profile_url, params=params)
            profile = {}
            if profile_response.status_code == 200:
                profile_data = profile_response.json()
                profile = profile_data[0] if profile_data else {}

            # Fetch ratios for additional metrics
            ratios_url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol}"
            ratios_response = await client.get(ratios_url, params=params)
            ratios = {}
            if ratios_response.status_code == 200:
                ratios_data = ratios_response.json()
                ratios = ratios_data[0] if ratios_data else {}

            return {
                "symbol": symbol,
                "company_name": profile.get("companyName", symbol),
                "sector": profile.get("sector", "N/A"),
                "industry": profile.get("industry", "N/A"),
                "current_price": profile.get("price"),
                "market_cap": profile.get("mktCap", 0),
                # Valuation metrics
                "pe_ttm": round(metrics.get("peRatioTTM", 0), 2) if metrics.get("peRatioTTM") else None,
                "pe_forward": round(ratios.get("priceEarningsToGrowthRatioTTM", 0), 2) if ratios.get("priceEarningsToGrowthRatioTTM") else None,
                "ps_ttm": round(metrics.get("priceToSalesRatioTTM", 0), 2) if metrics.get("priceToSalesRatioTTM") else None,
                "pb_ttm": round(metrics.get("pbRatioTTM", 0), 2) if metrics.get("pbRatioTTM") else None,
                "ev_ebitda": round(metrics.get("enterpriseValueOverEBITDATTM", 0), 2) if metrics.get("enterpriseValueOverEBITDATTM") else None,
                # Profitability
                "roe_ttm": round(metrics.get("roeTTM", 0) * 100, 1) if metrics.get("roeTTM") else None,
                "net_margin": round(metrics.get("netIncomePerShareTTM", 0) / metrics.get("revenuePerShareTTM", 1) * 100, 1) if metrics.get("revenuePerShareTTM") else None,
                # Growth (from ratios)
                "revenue_growth": round(ratios.get("revenueGrowthTTM", 0) * 100, 1) if ratios.get("revenueGrowthTTM") else None,
                # Dividend
                "dividend_yield": round(metrics.get("dividendYieldTTM", 0) * 100, 2) if metrics.get("dividendYieldTTM") else None,
            }

        except Exception as e:
            self.logger.debug(f"[Peers] Error fetching metrics for {symbol}: {e}")
            return None

    # =========================================================================
    # DATA FETCHING - WEB ENRICHMENT
    # =========================================================================

    async def _fetch_web_enrichment(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Fetch web search results with smart queries and deduplication."""
        try:
            web_search = WebSearchTool()

            current_date = datetime.now()
            month_year = current_date.strftime("%B %Y")

            queries = [
                {
                    "category": "latest_news",
                    "query": f"{symbol} stock news {month_year}",
                    "max_results": 4
                },
                {
                    "category": "earnings",
                    "query": f"{symbol} earnings guidance analyst estimates 2026",
                    "max_results": 3
                },
                {
                    "category": "analyst",
                    "query": f"{symbol} analyst rating price target {month_year}",
                    "max_results": 3
                }
            ]

            all_citations = []
            all_answers = []

            for query_info in queries:
                try:
                    result = await web_search.execute(
                        query=query_info["query"],
                        max_results=query_info.get("max_results", 3),
                        use_finance_domains=True
                    )

                    if result.status == "success" and result.data:
                        for citation in result.data.get("citations", []):
                            all_citations.append(citation)

                        if result.data.get("answer"):
                            all_answers.append({
                                "category": query_info["category"],
                                "answer": result.data.get("answer", "")
                            })
                except Exception as e:
                    self.logger.warning(f"[Web] Search failed for {query_info['category']}: {e}")

            # Deduplicate by URL
            seen_urls = set()
            unique_citations = []
            for citation in all_citations:
                url = citation.get("url", "")
                base_url = url.split("?")[0] if url else ""
                if base_url and base_url not in seen_urls:
                    seen_urls.add(base_url)
                    unique_citations.append(citation)

            return {
                "news": all_answers,
                "citations": unique_citations[:8],
                "search_date": current_date.strftime("%Y-%m-%d")
            }

        except Exception as e:
            self.logger.error(f"[Web] Error fetching enrichment for {symbol}: {e}")
            return None

    # =========================================================================
    # TRADING PLAN CALCULATION
    # =========================================================================

    def _calculate_trading_plan(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate consistent trading plan with holder-specific rules."""
        composite_score = scoring.get("composite_score", 50)

        # Extract data from raw_data
        risk_raw = step_data.get("risk", {}).get("raw_data", {})

        current_price = risk_raw.get("current_price", 0)
        atr = risk_raw.get("atr", {}).get("value", 0) if isinstance(risk_raw.get("atr"), dict) else 0

        # Support/Resistance levels
        support_1 = round(current_price * 0.95, 2) if current_price else 0
        support_2 = round(current_price * 0.90, 2) if current_price else 0
        resistance_1 = round(current_price * 1.05, 2) if current_price else 0
        resistance_2 = round(current_price * 1.10, 2) if current_price else 0

        plan = {
            "trading_system": "wait_and_watch",
            "current_price": current_price,
            "entry_zone": None,
            "stop_loss": None,
            "targets": [],
            "risk_reward_ratio": None,
            "position_sizing_note": "",
            "holder_rules": {
                "reduce_trigger": None,
                "exit_trigger": None,
                "trailing_stop": None,
                "add_trigger": None,
                "description": ""
            }
        }

        if composite_score >= 65:
            # BREAKOUT SYSTEM
            plan["trading_system"] = "breakout"

            entry_low = round(current_price * 1.03, 2)
            entry_high = round(current_price * 1.07, 2)
            plan["entry_zone"] = {"low": entry_low, "high": entry_high}

            avg_entry = (entry_low + entry_high) / 2
            stop_price = round(avg_entry - (2 * atr if atr > 0 else avg_entry * 0.08), 2)
            stop_pct = round(((avg_entry - stop_price) / avg_entry) * 100, 1) if avg_entry > 0 else 8.0
            plan["stop_loss"] = {"price": stop_price, "pct_from_entry": stop_pct}

            risk_per_share = avg_entry - stop_price
            target_1 = round(avg_entry + (risk_per_share * 2), 2)
            target_2 = round(avg_entry + (risk_per_share * 3), 2)

            plan["targets"] = [
                {"price": target_1, "pct_gain": round(((target_1 - avg_entry) / avg_entry) * 100, 1) if avg_entry > 0 else 0},
                {"price": target_2, "pct_gain": round(((target_2 - avg_entry) / avg_entry) * 100, 1) if avg_entry > 0 else 0}
            ]
            plan["risk_reward_ratio"] = 2.0
            plan["position_sizing_note"] = f"Risk {stop_pct}% per position. For 2% account risk: position size = (Account * 0.02) / (Entry * {stop_pct/100:.3f})"

            plan["holder_rules"] = {
                "reduce_trigger": None,
                "exit_trigger": stop_price,
                "trailing_stop": f"Move stop up by ATR (${atr:.2f}) as price rises" if atr else "Move stop up as price rises",
                "add_trigger": round(target_1 * 0.98, 2),
                "description": "Positive trend. Hold position, can add on pullback to support."
            }

        elif composite_score >= 45:
            # WAIT & WATCH
            plan["trading_system"] = "wait_and_watch"
            plan["position_sizing_note"] = "Watchlist only. Wait for score > 65 or clear technical breakout."

            plan["holder_rules"] = {
                "reduce_trigger": support_1,
                "exit_trigger": support_2,
                "trailing_stop": f"If price rises above ${resistance_1:.2f}, move stop to ${current_price:.2f}",
                "add_trigger": None,
                "description": f"Mixed signals. Holders: Reduce 50% if price breaks ${support_1:.2f}, exit fully below ${support_2:.2f}. Do NOT add until score > 65."
            }

        else:
            # EXIT / AVOID
            plan["trading_system"] = "exit_or_avoid"
            plan["position_sizing_note"] = "Reduce or exit position. Do not add new capital."

            plan["holder_rules"] = {
                "reduce_trigger": current_price,
                "exit_trigger": support_1,
                "trailing_stop": None,
                "add_trigger": None,
                "description": f"Negative signals. Holders: Reduce at least 50% NOW, exit fully below ${support_1:.2f}."
            }

        return plan

    def _format_trading_plan_section(self, plan: Dict[str, Any]) -> str:
        """Format trading plan as prompt section."""
        lines = [
            "## PRE-CALCULATED TRADING PLAN (USE THESE EXACT VALUES)",
            "",
            f"Trading System: {plan['trading_system'].upper().replace('_', ' ')}",
            f"Current Price: ${plan['current_price']:.2f}" if plan['current_price'] else "Current Price: N/A",
        ]

        lines.append("")
        lines.append("### FOR NEW INVESTORS:")
        if plan.get("entry_zone"):
            lines.append(f"- Entry Zone: ${plan['entry_zone']['low']:.2f} - ${plan['entry_zone']['high']:.2f} (BREAKOUT CONFIRMATION REQUIRED)")
        else:
            lines.append("- Entry Zone: NO ENTRY - Watchlist only")

        if plan.get("stop_loss"):
            lines.append(f"- Stop Loss: ${plan['stop_loss']['price']:.2f} ({plan['stop_loss']['pct_from_entry']}% below ENTRY price)")

        if plan.get("targets"):
            for i, target in enumerate(plan["targets"], 1):
                lines.append(f"- Target {i}: ${target['price']:.2f} (+{target['pct_gain']}%)")

        if plan.get("risk_reward_ratio"):
            lines.append(f"- Risk:Reward: 1:{plan['risk_reward_ratio']}")

        holder_rules = plan.get("holder_rules", {})
        if holder_rules:
            lines.append("")
            lines.append("### FOR EXISTING HOLDERS:")
            if holder_rules.get("description"):
                lines.append(f"- Strategy: {holder_rules['description']}")
            if holder_rules.get("reduce_trigger"):
                lines.append(f"- Reduce 50%: If price breaks ${holder_rules['reduce_trigger']:.2f}")
            if holder_rules.get("exit_trigger"):
                lines.append(f"- Exit fully: If price breaks ${holder_rules['exit_trigger']:.2f}")
            if holder_rules.get("trailing_stop"):
                lines.append(f"- Trailing Stop: {holder_rules['trailing_stop']}")

        return "\n".join(lines)

    # =========================================================================
    # SCENARIO ANALYSIS CALCULATION
    # =========================================================================

    def _calculate_scenario_analysis(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate Bull/Base/Bear scenarios with price targets and probabilities."""
        risk_raw = step_data.get("risk", {}).get("raw_data", {})
        fund_raw = step_data.get("fundamental", {}).get("raw_data", {})

        current_price = risk_raw.get("current_price", 0)
        atr = risk_raw.get("atr", {}).get("value", 0) if isinstance(risk_raw.get("atr"), dict) else 0
        volatility = risk_raw.get("volatility", {}).get("annualized", 0) if isinstance(risk_raw.get("volatility"), dict) else 0

        # Get intrinsic values if available
        intrinsic = fund_raw.get("fundamental_report", {}).get("intrinsic_value", {}) if fund_raw else {}
        graham_value = intrinsic.get("graham_value")
        dcf_value = intrinsic.get("dcf_value")

        composite_score = scoring.get("composite_score", 50)

        # Calculate scenario targets based on volatility and score
        if current_price > 0:
            # Bull case: +20-30% depending on score
            bull_multiplier = 1.25 if composite_score >= 65 else 1.20
            bull_target = round(current_price * bull_multiplier, 2)

            # Base case: +5-10%
            base_target = round(current_price * 1.08, 2)

            # Bear case: -15-25%
            bear_multiplier = 0.80 if composite_score < 45 else 0.85
            bear_target = round(current_price * bear_multiplier, 2)
        else:
            bull_target = base_target = bear_target = 0

        # Probability based on composite score
        if composite_score >= 70:
            bull_prob, base_prob, bear_prob = 40, 45, 15
        elif composite_score >= 55:
            bull_prob, base_prob, bear_prob = 30, 50, 20
        elif composite_score >= 45:
            bull_prob, base_prob, bear_prob = 25, 45, 30
        else:
            bull_prob, base_prob, bear_prob = 15, 40, 45

        return {
            "current_price": current_price,
            "bull": {
                "target": bull_target,
                "return_pct": round((bull_target - current_price) / current_price * 100, 1) if current_price else 0,
                "probability": bull_prob,
                "triggers": ["Earnings beat + raised guidance", "Sector rotation into stock", "New product/contract announcement"]
            },
            "base": {
                "target": base_target,
                "return_pct": round((base_target - current_price) / current_price * 100, 1) if current_price else 0,
                "probability": base_prob,
                "triggers": ["In-line earnings", "Stable macro environment", "No major surprises"]
            },
            "bear": {
                "target": bear_target,
                "return_pct": round((bear_target - current_price) / current_price * 100, 1) if current_price else 0,
                "probability": bear_prob,
                "triggers": ["Earnings miss", "Macro deterioration", "Competition pressure", "Guidance cut"]
            },
            "fair_value": {
                "graham_value": graham_value,
                "dcf_value": dcf_value,
                "premium_discount": self._calculate_premium_discount(current_price, graham_value, dcf_value)
            },
            "time_horizon": "1-3 months"
        }

    def _calculate_premium_discount(
        self,
        current_price: float,
        graham_value: Optional[float],
        dcf_value: Optional[float]
    ) -> Dict[str, Any]:
        """Calculate premium/discount to fair value."""
        result = {"assessment": "N/A", "details": {}}

        if not current_price:
            return result

        fair_values = []
        if graham_value and graham_value > 0:
            graham_diff = round((current_price - graham_value) / graham_value * 100, 1)
            result["details"]["graham"] = {
                "value": graham_value,
                "diff_pct": graham_diff,
                "status": "Premium" if graham_diff > 0 else "Discount"
            }
            fair_values.append(graham_value)

        if dcf_value and dcf_value > 0:
            dcf_diff = round((current_price - dcf_value) / dcf_value * 100, 1)
            result["details"]["dcf"] = {
                "value": dcf_value,
                "diff_pct": dcf_diff,
                "status": "Premium" if dcf_diff > 0 else "Discount"
            }
            fair_values.append(dcf_value)

        if fair_values:
            avg_fair_value = sum(fair_values) / len(fair_values)
            avg_diff = round((current_price - avg_fair_value) / avg_fair_value * 100, 1)

            if avg_diff > 20:
                result["assessment"] = "Significantly Overvalued"
            elif avg_diff > 5:
                result["assessment"] = "Moderately Overvalued"
            elif avg_diff > -5:
                result["assessment"] = "Fairly Valued"
            elif avg_diff > -20:
                result["assessment"] = "Moderately Undervalued"
            else:
                result["assessment"] = "Significantly Undervalued"

            result["avg_fair_value"] = round(avg_fair_value, 2)
            result["avg_diff_pct"] = avg_diff

        return result

    def _format_scenario_section(self, scenarios: Dict[str, Any]) -> str:
        """Format scenario analysis as prompt section."""
        lines = [
            "## PRE-CALCULATED SCENARIO ANALYSIS (USE THESE VALUES)",
            "",
            f"Current Price: ${scenarios['current_price']:.2f}" if scenarios['current_price'] else "Current Price: N/A",
            f"Time Horizon: {scenarios['time_horizon']}",
            "",
            "### SCENARIOS:",
            "",
            f"BULL CASE ({scenarios['bull']['probability']}% probability):",
            f"- Target: ${scenarios['bull']['target']:.2f} ({scenarios['bull']['return_pct']:+.1f}%)",
            f"- Triggers: {', '.join(scenarios['bull']['triggers'][:2])}",
            "",
            f"BASE CASE ({scenarios['base']['probability']}% probability):",
            f"- Target: ${scenarios['base']['target']:.2f} ({scenarios['base']['return_pct']:+.1f}%)",
            f"- Triggers: {', '.join(scenarios['base']['triggers'][:2])}",
            "",
            f"BEAR CASE ({scenarios['bear']['probability']}% probability):",
            f"- Target: ${scenarios['bear']['target']:.2f} ({scenarios['bear']['return_pct']:+.1f}%)",
            f"- Triggers: {', '.join(scenarios['bear']['triggers'][:2])}",
        ]

        fair_value = scenarios.get("fair_value", {})
        if fair_value.get("assessment") != "N/A":
            lines.extend([
                "",
                "### FAIR VALUE ASSESSMENT:",
                f"- Assessment: {fair_value['assessment']}",
            ])
            if fair_value.get("avg_fair_value"):
                lines.append(f"- Average Fair Value: ${fair_value['avg_fair_value']:.2f}")
                lines.append(f"- Premium/Discount: {fair_value['avg_diff_pct']:+.1f}%")

            details = fair_value.get("details", {})
            if details.get("graham"):
                lines.append(f"- Graham Value: ${details['graham']['value']:.2f} ({details['graham']['status']} {abs(details['graham']['diff_pct']):.1f}%)")
            if details.get("dcf"):
                lines.append(f"- DCF Value: ${details['dcf']['value']:.2f} ({details['dcf']['status']} {abs(details['dcf']['diff_pct']):.1f}%)")

        return "\n".join(lines)

    # =========================================================================
    # CONSOLIDATED PROMPT BUILDER (USES RAW DATA)
    # =========================================================================

    def _build_consolidated_prompt_v2(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any],
        earnings_data: Optional[Dict[str, Any]],
        peer_data: Optional[Dict[str, Any]],
        web_data: Optional[Dict[str, Any]],
        target_language: str
    ) -> str:
        """
        Build comprehensive prompt using RAW DATA metrics instead of truncated LLM content.
        """
        parts = [
            f"# COMPREHENSIVE INVESTMENT ANALYSIS: {symbol}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Target Language: {target_language.upper()}",
            "",
        ]

        # =====================================================================
        # SECTION 1: BINDING SCORING
        # =====================================================================
        rec = scoring.get("recommendation", {})
        dist = rec.get("distribution", {})

        parts.extend([
            "=" * 60,
            "## BINDING SCORING DATA (DO NOT CONTRADICT)",
            "=" * 60,
            "",
            f"Composite Score: {scoring.get('composite_score', 'N/A')}/100",
            f"Recommendation: {rec.get('action', 'HOLD')} (BINDING)",
            f"Distribution: BUY {dist.get('buy', 0)}% | HOLD {dist.get('hold', 0)}% | SELL {dist.get('sell', 0)}%",
            f"Confidence: {rec.get('confidence', 'N/A')}%",
            f"Time Horizon: {rec.get('time_horizon', 'N/A')}",
            "",
        ])

        # Component scores
        parts.append("### Component Scores:")
        components = scoring.get("component_scores", {})
        for name, data in components.items():
            parts.append(f"- {name.title()}: {data.get('score', 'N/A')}/100 (weight: {data.get('weight', 'N/A')}, confidence: {data.get('confidence', 'N/A')})")
        parts.append("")

        # Key factors
        key_factors = scoring.get("key_factors", [])
        if key_factors:
            bullish = [f["factor"] for f in key_factors if f.get("impact") == "bullish"]
            bearish = [f["factor"] for f in key_factors if f.get("impact") == "bearish"]
            parts.append("### Key Factors:")
            parts.append(f"Bullish: {', '.join(bullish) if bullish else 'None'}")
            parts.append(f"Bearish: {', '.join(bearish) if bearish else 'None'}")
            parts.append("")

        # =====================================================================
        # SECTION 2: TECHNICAL DATA (RAW METRICS)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## TECHNICAL ANALYSIS DATA",
            "=" * 60,
            "",
        ])

        tech_raw = step_data.get("technical", {}).get("raw_data", {})
        if tech_raw:
            parts.extend(self._format_technical_raw_data(tech_raw))
        else:
            parts.append("Technical data not available.")
        parts.append("")

        # =====================================================================
        # SECTION 3: POSITION DATA (RAW METRICS)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## MARKET POSITION DATA",
            "=" * 60,
            "",
        ])

        pos_raw = step_data.get("position", {}).get("raw_data", {})
        sector_context = step_data.get("position", {}).get("sector_context", {})
        if pos_raw or sector_context:
            parts.extend(self._format_position_raw_data(pos_raw, sector_context))
        else:
            parts.append("Position data not available.")
        parts.append("")

        # =====================================================================
        # SECTION 4: RISK DATA (RAW METRICS)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## RISK ANALYSIS DATA",
            "=" * 60,
            "",
        ])

        risk_raw = step_data.get("risk", {}).get("raw_data", {})
        if risk_raw:
            parts.extend(self._format_risk_raw_data(risk_raw))
        else:
            parts.append("Risk data not available.")
        parts.append("")

        # =====================================================================
        # SECTION 5: SENTIMENT DATA (RAW METRICS)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## SENTIMENT DATA",
            "=" * 60,
            "",
        ])

        sent_raw = step_data.get("sentiment", {}).get("raw_data", {})
        if sent_raw:
            parts.extend(self._format_sentiment_raw_data(sent_raw))
        else:
            parts.append("Sentiment data not available.")
        parts.append("")

        # =====================================================================
        # SECTION 6: FUNDAMENTAL DATA (RAW METRICS)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## FUNDAMENTAL DATA",
            "=" * 60,
            "",
        ])

        fund_raw = step_data.get("fundamental", {}).get("raw_data", {})
        if fund_raw:
            parts.extend(self._format_fundamental_raw_data(fund_raw))
        else:
            parts.append("Fundamental data not available.")
        parts.append("")

        # =====================================================================
        # SECTION 7: PEER COMPARISON (DYNAMIC FMP API)
        # =====================================================================
        if peer_data and peer_data.get("peers"):
            parts.extend([
                "=" * 60,
                "## PEER COMPARISON (FMP stock_peers API)",
                "=" * 60,
                "",
                f"Target: {symbol} | Sector: {peer_data.get('sector', 'N/A')} | Industry: {peer_data.get('industry', 'N/A')}",
                "",
                "| Symbol | P/E (TTM) | P/S (TTM) | EV/EBITDA | ROE % | Rev Growth % | Market Cap |",
                "|--------|-----------|-----------|-----------|-------|--------------|------------|",
            ])

            target = peer_data.get("target_metrics", {})
            parts.append(
                f"| **{symbol}** | {target.get('pe_ttm') or 'N/A'} | {target.get('ps_ttm') or 'N/A'} | "
                f"{target.get('ev_ebitda') or 'N/A'} | {target.get('roe_ttm') or 'N/A'} | "
                f"{target.get('revenue_growth') or 'N/A'} | ${target.get('market_cap', 0)/1e9:.1f}B |"
            )

            for peer in peer_data.get("peers", [])[:5]:
                parts.append(
                    f"| {peer.get('symbol', 'N/A')} | {peer.get('pe_ttm') or 'N/A'} | {peer.get('ps_ttm') or 'N/A'} | "
                    f"{peer.get('ev_ebitda') or 'N/A'} | {peer.get('roe_ttm') or 'N/A'} | "
                    f"{peer.get('revenue_growth') or 'N/A'} | ${peer.get('market_cap', 0)/1e9:.1f}B |"
                )

            parts.extend([
                "",
                "NOTE: P/E ratios may be distorted for companies with low or volatile EPS.",
                f"Source: {peer_data.get('source', 'FMP API')}",
                "",
            ])

        # =====================================================================
        # SECTION 8: EARNINGS CALENDAR
        # =====================================================================
        if earnings_data:
            parts.extend([
                "=" * 60,
                "## EARNINGS CALENDAR (FMP API)",
                "=" * 60,
                "",
            ])

            if earnings_data.get("next_earnings_date"):
                parts.append(f"NEXT EARNINGS: {earnings_data['next_earnings_date']} ({earnings_data.get('earnings_time', 'TBD')})")
                parts.append(f"Fiscal Quarter: {earnings_data.get('fiscal_quarter', 'N/A')}")
                if earnings_data.get("eps_estimated"):
                    parts.append(f"EPS Estimate: ${earnings_data['eps_estimated']:.2f}")
            else:
                parts.append("Next earnings date: Not yet announced")

            if earnings_data.get("beat_rate") is not None:
                parts.append(f"Historical Beat Rate: {earnings_data['beat_rate']*100:.0f}%")

            parts.append(f"Source: {earnings_data.get('source', 'FMP API')}")
            parts.append("")

        # =====================================================================
        # SECTION 9: WEB SEARCH RESULTS
        # =====================================================================
        if web_data and web_data.get("citations"):
            parts.extend([
                "=" * 60,
                "## WEB SEARCH RESULTS (Use Inline Citations)",
                "=" * 60,
                "",
            ])

            for item in web_data.get("news", []):
                parts.append(f"### {item['category'].upper()}")
                parts.append(item["answer"][:1500])
                parts.append("")

            parts.append("### Available Sources (CITE INLINE):")
            for i, citation in enumerate(web_data.get("citations", [])[:8], 1):
                title = citation.get("title", "Untitled")[:60]
                url = citation.get("url", "")
                parts.append(f"[{i}] [{title}]({url})")

            parts.append("")
            parts.append("INSTRUCTION: Cite sources INLINE when making claims, e.g., 'Stock fell 5% [Source Name](URL)'")
            parts.append("Include a '## Sources' section at the end with all sources used.")
            parts.append("")

        # =====================================================================
        # SECTION 10: OUTPUT INSTRUCTIONS
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## OUTPUT INSTRUCTIONS",
            "=" * 60,
            "",
            f"Generate a comprehensive investment report in {target_language.upper()}.",
            "",
            "Required sections in order:",
            "1. Technical Analysis - Use metrics above",
            "2. Market Position - Include sector ranking methodology",
            "3. Risk Analysis - Include VaR, ATR, volatility",
            "4. Sentiment Analysis - Include sentiment score interpretation",
            "5. Fundamental Analysis - Include peer comparison table",
            "6. Fair Value Assessment - Compare current price to Graham/DCF values",
            "7. Scenario Analysis - Bull/Base/Bear with probabilities",
            "8. News & Catalysts - With inline citations + Sources section",
            "9. Executive Summary - Key highlights from each section",
            "10. Final Recommendation - For NEW and EXISTING investors",
            "",
            "CRITICAL REMINDERS:",
            f"- Recommendation is {rec.get('action', 'HOLD')} - DO NOT contradict",
            "- Use EXACT values from trading plan and scenarios",
            "- Cite web sources inline",
            "- Include holder-specific rules (reduce/exit triggers)",
        ])

        return "\n".join(parts)

    # =========================================================================
    # RAW DATA FORMATTERS
    # =========================================================================

    def _format_technical_raw_data(self, raw: Dict[str, Any]) -> List[str]:
        """Format technical raw data as structured metrics."""
        lines = []

        # Price data
        if raw.get("price_data"):
            pd = raw["price_data"]
            lines.append("### Price Data:")
            lines.append(f"- Current Price: ${pd.get('close', 'N/A')}")
            lines.append(f"- Day Change: {pd.get('change_percent', 'N/A')}%")
            lines.append(f"- 52W High: ${pd.get('high_52w', 'N/A')}")
            lines.append(f"- 52W Low: ${pd.get('low_52w', 'N/A')}")
            lines.append("")

        # Moving Averages
        if raw.get("moving_averages"):
            ma = raw["moving_averages"]
            lines.append("### Moving Averages:")
            lines.append(f"- SMA20: ${ma.get('sma20', 'N/A')}")
            lines.append(f"- SMA50: ${ma.get('sma50', 'N/A')}")
            lines.append(f"- SMA200: ${ma.get('sma200', 'N/A')}")
            lines.append(f"- Price vs SMA200: {'Above' if ma.get('above_sma200') else 'Below'}")
            lines.append("")

        # Momentum Indicators
        if raw.get("momentum"):
            m = raw["momentum"]
            lines.append("### Momentum Indicators:")
            lines.append(f"- RSI(14): {m.get('rsi', 'N/A')}")
            lines.append(f"- MACD: {m.get('macd', 'N/A')}")
            lines.append(f"- MACD Signal: {m.get('macd_signal', 'N/A')}")
            lines.append(f"- MACD Histogram: {m.get('macd_histogram', 'N/A')}")
            lines.append("")

        # Trend
        if raw.get("trend"):
            t = raw["trend"]
            lines.append("### Trend Indicators:")
            lines.append(f"- ADX: {t.get('adx', 'N/A')} (Trend Strength: {'Strong' if t.get('adx', 0) > 25 else 'Weak' if t.get('adx', 0) < 15 else 'Moderate'})")
            lines.append(f"- +DI: {t.get('plus_di', 'N/A')}")
            lines.append(f"- -DI: {t.get('minus_di', 'N/A')}")
            lines.append("")

        # Volume
        if raw.get("volume"):
            v = raw["volume"]
            lines.append("### Volume:")
            lines.append(f"- Current Volume: {v.get('volume', 'N/A'):,}" if isinstance(v.get('volume'), (int, float)) else f"- Current Volume: {v.get('volume', 'N/A')}")
            lines.append(f"- Average Volume: {v.get('avg_volume', 'N/A'):,}" if isinstance(v.get('avg_volume'), (int, float)) else f"- Average Volume: {v.get('avg_volume', 'N/A')}")
            lines.append(f"- RVOL: {v.get('rvol', 'N/A')}x")
            lines.append("")

        # Signals
        if raw.get("signals"):
            lines.append("### Signals:")
            for signal in raw["signals"][:5]:
                lines.append(f"- {signal}")
            lines.append("")

        return lines

    def _format_position_raw_data(self, raw: Dict[str, Any], sector_ctx: Dict[str, Any]) -> List[str]:
        """Format position raw data as structured metrics."""
        lines = []

        # RS Data
        if raw:
            lines.append("### Relative Strength vs SPY:")
            if raw.get("excess_return_21d") is not None:
                lines.append(f"- 21-day Excess Return: {raw['excess_return_21d']:+.2f}%")
            if raw.get("excess_return_63d") is not None:
                lines.append(f"- 63-day Excess Return: {raw['excess_return_63d']:+.2f}%")
            if raw.get("excess_return_126d") is not None:
                lines.append(f"- 126-day Excess Return: {raw['excess_return_126d']:+.2f}%")
            if raw.get("rs_rating"):
                lines.append(f"- RS Rating: {raw['rs_rating']}")
            if raw.get("classification"):
                lines.append(f"- Classification: {raw['classification']}")
            lines.append("")

        # Sector Context
        if sector_ctx:
            lines.append("### Sector Context:")
            lines.append(f"- Sector: {sector_ctx.get('stock_sector', 'N/A')}")
            lines.append(f"- Industry: {sector_ctx.get('stock_industry', 'N/A')}")
            lines.append(f"- Sector Rank: #{sector_ctx.get('sector_rank', 'N/A')}/{sector_ctx.get('total_sectors', 11)}")
            lines.append(f"- Sector Change (1-day): {sector_ctx.get('sector_change_percent', 0):+.2f}%")
            lines.append(f"- Sector Status: {sector_ctx.get('sector_status', 'N/A')}")
            lines.append("")
            lines.append("NOTE: Sector ranking is 1-DAY performance from FMP Sector Performance API.")
            lines.append("This is different from multi-timeframe Relative Strength analysis.")
            lines.append("")

        return lines

    def _format_risk_raw_data(self, raw: Dict[str, Any]) -> List[str]:
        """Format risk raw data as structured metrics."""
        lines = []

        lines.append(f"### Current Price: ${raw.get('current_price', 'N/A')}")
        lines.append("")

        # Volatility
        if raw.get("volatility"):
            v = raw["volatility"]
            lines.append("### Volatility:")
            lines.append(f"- Daily Volatility: {v.get('daily', 'N/A')}%")
            lines.append(f"- Annualized Volatility: {v.get('annualized', 'N/A')}%")
            lines.append(f"- Classification: {v.get('classification', 'N/A')}")
            lines.append("")

        # ATR
        if raw.get("atr"):
            a = raw["atr"]
            lines.append("### ATR (Average True Range):")
            lines.append(f"- ATR Value: ${a.get('value', 'N/A')}")
            lines.append(f"- ATR %: {a.get('percent', 'N/A')}%")
            lines.append("")

        # VaR
        if raw.get("var"):
            v = raw["var"]
            lines.append("### Value at Risk (VaR):")
            lines.append(f"- VaR 95% (1-day): {v.get('var_95', 'N/A')}%")
            lines.append(f"- VaR 99% (1-day): {v.get('var_99', 'N/A')}%")
            lines.append("")

        # Stop Loss Recommendations
        if raw.get("stop_loss"):
            sl = raw["stop_loss"]
            lines.append("### Stop Loss Recommendations:")
            if sl.get("atr_based"):
                lines.append(f"- ATR-based (2x ATR): ${sl['atr_based'].get('price', 'N/A')} ({sl['atr_based'].get('percent', 'N/A')}%)")
            if sl.get("percent_based"):
                lines.append(f"- Percentage-based (5%): ${sl['percent_based'].get('price', 'N/A')}")
            lines.append("")

        return lines

    def _format_sentiment_raw_data(self, raw: Dict[str, Any]) -> List[str]:
        """Format sentiment raw data as structured metrics."""
        lines = []

        if raw.get("sentiment_score") is not None:
            score = raw["sentiment_score"]
            classification = "Bullish" if score > 0.2 else "Bearish" if score < -0.2 else "Neutral"
            lines.append(f"### Sentiment Score: {score:.3f} ({classification})")
            lines.append("")

        if raw.get("social_sentiment"):
            ss = raw["social_sentiment"]
            lines.append("### Social Sentiment:")
            lines.append(f"- Score: {ss.get('score', 'N/A')}")
            lines.append(f"- Posts Analyzed: {ss.get('post_count', 'N/A')}")
            lines.append("")

        if raw.get("news_sentiment"):
            ns = raw["news_sentiment"]
            lines.append("### News Sentiment:")
            lines.append(f"- Overall: {ns.get('overall', 'N/A')}")
            lines.append(f"- Articles Analyzed: {ns.get('article_count', 'N/A')}")
            lines.append("")

        if raw.get("key_themes"):
            lines.append("### Key Themes:")
            for theme in raw["key_themes"][:5]:
                lines.append(f"- {theme}")
            lines.append("")

        return lines

    def _format_fundamental_raw_data(self, raw: Dict[str, Any]) -> List[str]:
        """Format fundamental raw data as structured metrics."""
        lines = []

        report = raw.get("fundamental_report", {})

        # Valuation
        if report.get("valuation"):
            v = report["valuation"]
            lines.append("### Valuation Metrics:")
            lines.append(f"- P/E (TTM): {v.get('pe_ttm', 'N/A')}")
            lines.append(f"- P/E (FY): {v.get('pe_fy', 'N/A')}")
            lines.append(f"- P/S (TTM): {v.get('ps_ttm', 'N/A')}")
            lines.append(f"- P/B (TTM): {v.get('pb_ttm', 'N/A')}")
            lines.append(f"- EV/EBITDA: {v.get('ev_ebitda', 'N/A')}")
            lines.append("")

        # Profitability
        if report.get("profitability"):
            p = report["profitability"]
            lines.append("### Profitability:")
            lines.append(f"- Gross Margin: {p.get('gross_margin', 'N/A')}%")
            lines.append(f"- Operating Margin: {p.get('operating_margin', 'N/A')}%")
            lines.append(f"- Net Margin: {p.get('net_margin', 'N/A')}%")
            lines.append(f"- ROE: {p.get('roe', 'N/A')}%")
            lines.append(f"- ROA: {p.get('roa', 'N/A')}%")
            lines.append("")

        # Growth
        if report.get("growth"):
            g = report["growth"]
            lines.append("### Growth Metrics:")
            lines.append(f"- Revenue Growth (YoY): {g.get('revenue_growth_yoy', 'N/A')}%")
            lines.append(f"- EPS Growth (YoY): {g.get('eps_growth_yoy', 'N/A')}%")
            lines.append(f"- Revenue Growth (5Y CAGR): {g.get('revenue_cagr_5y', 'N/A')}%")
            lines.append(f"- EPS Growth (5Y CAGR): {g.get('eps_cagr_5y', 'N/A')}%")
            lines.append("")

        # Intrinsic Value
        if report.get("intrinsic_value"):
            iv = report["intrinsic_value"]
            lines.append("### Intrinsic Value (Fair Value):")
            if iv.get("graham_value"):
                lines.append(f"- Graham Value: ${iv['graham_value']:.2f}")
            if iv.get("dcf_value"):
                lines.append(f"- DCF Value: ${iv['dcf_value']:.2f}")
            if iv.get("current_price"):
                lines.append(f"- Current Price: ${iv['current_price']:.2f}")
            if iv.get("verdict"):
                lines.append(f"- Verdict: {iv['verdict']}")
            lines.append("")

        # Dividend
        if report.get("dividend"):
            d = report["dividend"]
            lines.append("### Dividend:")
            lines.append(f"- Dividend Yield: {d.get('yield', 'N/A')}%")
            lines.append(f"- Payout Ratio: {d.get('payout_ratio', 'N/A')}%")
            lines.append("")

        return lines

    # =========================================================================
    # REPORT HEADER/FOOTER
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
**Version:** Synthesis V2 (Production)

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
        components = scoring.get("component_scores", {})
        header += "\n| Component | Score | Weight | Confidence |\n|-----------|-------|--------|------------|\n"
        for name, data in components.items():
            header += f"| {name.title()} | {data.get('score', 'N/A')}/100 | {data.get('weight', 'N/A')} | {data.get('confidence', 'N/A')} |\n"

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
        freshness_lines = []
        for step_name in SCANNER_STEPS:
            data = step_data.get(step_name, {})
            cached_at = data.get("cached_at", "N/A")
            if cached_at and cached_at != "N/A":
                freshness_lines.append(f"- {step_name}: {cached_at}")
            else:
                freshness_lines.append(f"- {step_name}: Not available")

        return f"""
---

## Disclaimer

This report is generated by AI analysis and should not be considered as financial advice.
Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
Past performance is not indicative of future results. Investments involve risk, including the possible loss of principal.

---

## Report Metadata

- **Symbol:** {symbol}
- **Generation Time:** {elapsed_seconds:.1f} seconds
- **Architecture:** Single LLM Call (V2 Production)
- **Data Source:** FMP API (earnings, peers, metrics)

### Data Freshness
{chr(10).join(freshness_lines)}

---
*Report generated by HealerAgent Market Scanner V2*
"""

    # =========================================================================
    # RUN MISSING STEPS
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
        """Run missing analysis steps in parallel."""
        results = {}
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

        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, step_name in enumerate(step_names):
                result = task_results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"[SynthesisV2] Step {step_name} failed: {result}")
                    results[step_name] = {
                        "content": f"Error: {str(result)}",
                        "raw_data": None,
                        "error": True
                    }
                else:
                    results[step_name] = result
                    await save_scanner_result(symbol, step_name, result)

        return results

    async def _collect_technical(
        self, symbol: str, timeframe: str, model_name: str,
        provider_type: str, api_key: str, target_language: Optional[str]
    ) -> Dict[str, Any]:
        """Collect technical analysis."""
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
        """Collect market position analysis."""
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
            "sector_context": raw_data.get("sector_context"),
            "cached_at": datetime.now().isoformat()
        }

    async def _collect_risk(
        self, symbol: str, model_name: str,
        provider_type: str, api_key: str, target_language: Optional[str]
    ) -> Dict[str, Any]:
        """Collect risk analysis."""
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
        """Collect sentiment analysis."""
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
        """Collect fundamental analysis."""
        from src.handlers.fundamental_analysis_handler import FundamentalAnalysisHandler
        from src.services.tool_call_service import ToolCallService

        fund_handler = FundamentalAnalysisHandler()
        tool_service = ToolCallService()

        comprehensive_data = await fund_handler.generate_comprehensive_fundamental_data(
            symbol=symbol,
            tool_service=tool_service
        )

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


# =============================================================================
# MODULE-LEVEL INSTANCE
# =============================================================================

synthesis_handler_v2 = SynthesisHandlerV2()
