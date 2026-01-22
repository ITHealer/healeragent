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
# GICS SECTOR CLASSIFICATION (Phase 1: Critical Fix)
# =============================================================================
# Official 11 GICS Sectors - used for validation
GICS_SECTORS = {
    "Information Technology": ["Semiconductors", "Software", "Hardware", "IT Services", "Electronic Equipment"],
    "Health Care": ["Pharmaceuticals", "Biotechnology", "Health Care Equipment", "Health Care Services"],
    "Financials": ["Banks", "Insurance", "Capital Markets", "Consumer Finance"],
    "Consumer Discretionary": ["Automobiles", "Hotels Restaurants & Leisure", "Retail", "Household Durables"],
    "Consumer Staples": ["Food Products", "Beverages", "Household Products", "Personal Products"],
    "Industrials": ["Aerospace & Defense", "Machinery", "Transportation", "Commercial Services"],
    "Energy": ["Oil & Gas", "Energy Equipment & Services"],
    "Materials": ["Chemicals", "Metals & Mining", "Construction Materials", "Containers & Packaging"],
    "Utilities": ["Electric Utilities", "Gas Utilities", "Multi-Utilities", "Water Utilities"],
    "Real Estate": ["REITs", "Real Estate Management & Development"],
    "Communication Services": ["Diversified Telecommunication", "Media", "Entertainment", "Interactive Media"]
}

# Sector name normalization mapping
SECTOR_ALIASES = {
    "Technology": "Information Technology",
    "Tech": "Information Technology",
    "Healthcare": "Health Care",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Communication": "Communication Services",
    "Telecom": "Communication Services",
    "Financial Services": "Financials",
    "Finance": "Financials"
}

# Cache TTL configurations (in seconds)
CACHE_TTL = {
    "analyst_consensus": 6 * 3600,     # 6 hours - analyst data updates infrequently
    "insider_trading": 1 * 3600,        # 1 hour - insider data can change quickly
    "seasonal_analysis": 24 * 3600,     # 24 hours - seasonal patterns are stable
    "earnings_calendar": 12 * 3600,     # 12 hours - earnings dates are stable
}


def normalize_gics_sector(sector: str) -> str:
    """
    Normalize sector name to official GICS sector.

    Args:
        sector: Raw sector name from API

    Returns:
        Normalized GICS sector name
    """
    if not sector:
        return "Unknown"

    # Check if already a valid GICS sector
    if sector in GICS_SECTORS:
        return sector

    # Check aliases
    return SECTOR_ALIASES.get(sector, sector)


def validate_gics_classification(sector: str, industry: str) -> Dict[str, Any]:
    """
    Validate and return proper GICS classification.

    Args:
        sector: Sector name
        industry: Industry name

    Returns:
        Dict with validated sector, industry, and any warnings
    """
    normalized_sector = normalize_gics_sector(sector)
    is_valid_sector = normalized_sector in GICS_SECTORS

    result = {
        "sector": normalized_sector,
        "industry": industry,
        "is_valid_gics": is_valid_sector,
        "warnings": []
    }

    if not is_valid_sector:
        result["warnings"].append(f"'{sector}' is not a standard GICS sector")
        # Try to find best match
        for gics_sector in GICS_SECTORS:
            if sector.lower() in gics_sector.lower() or gics_sector.lower() in sector.lower():
                result["sector"] = gics_sector
                result["is_valid_gics"] = True
                result["warnings"].append(f"Normalized to '{gics_sector}'")
                break

    # Validate industry belongs to sector
    if result["is_valid_gics"] and industry:
        sector_industries = GICS_SECTORS.get(result["sector"], [])
        industry_match = any(ind.lower() in industry.lower() for ind in sector_industries)
        if not industry_match:
            result["warnings"].append(f"Industry '{industry}' may not belong to sector '{result['sector']}'")

    return result


# =============================================================================
# SYSTEM PROMPT (ENGLISH ONLY - Production)
# =============================================================================

CONSOLIDATED_SYSTEM_PROMPT = """You are a senior investment analyst creating a comprehensive, data-driven investment report.

## CRITICAL RULES - DATA ACCURACY FIRST

### 1. NO SCORING - ONLY RAW DATA
Do NOT invent or display any composite scores, ratings, or numeric scoring.
Present ONLY the RAW DATA metrics provided and let the reader interpret.
If scoring data appears in input, IGNORE IT and focus on actual metrics.

### 2. INPUT → OUTPUT FORMAT (REQUIRED FOR EACH SECTION)
Each analysis section MUST show:
- **INPUTS**: List the actual data values used (e.g., RSI=45.2, ADX=18.3)
- **ANALYSIS**: Your interpretation of the data
- **SIGNALS**: What the data indicates (bullish/bearish/neutral and WHY)

Example:
```
### Technical Analysis
**Inputs:** RSI(14)=45.2, MACD=-0.85, ADX=18.3, SMA50=$142, Current=$138
**Analysis:** RSI in neutral zone (30-70), MACD below signal line, ADX shows weak trend
**Signal:** NEUTRAL - Conflicting signals with weak trend strength
```

### 3. STOP-LOSS WITH STRUCTURE/ATR LOGIC (CRITICAL)
When recommending stop-loss, ALWAYS explain the logic:
- Show the calculation: "Stop = Entry - (2 × ATR)" or "Stop = below swing low at $X"
- State the ATR value used
- Calculate percentage from ENTRY price, not current price
- Example: "ATR=$5.20, Entry=$185, Stop=$185-(2×5.20)=$174.60 (5.6% risk)"

### 4. SECTOR vs INDUSTRY (CRITICAL DISTINCTION)
- **Sector**: One of 11 GICS sectors (e.g., Information Technology, Healthcare)
- **Industry**: Sub-classification (e.g., Semiconductors, Software)
- State ranking methodology: "FMP Sector Performance (1-day)"
- Note limitation: "1-day ranking ≠ multi-timeframe RS analysis"
- For RS, prefer multi-day data if available (21d/63d/126d)

### 5. PEER COMPARISON - USE CORRECT PEERS
For Semiconductors: Compare with AMD, AVGO, TSM, INTC, MRVL, QCOM
For other sectors: Use same industry/sub-industry peers
Do NOT mix mega-cap tech (AAPL, MSFT, GOOGL) as "peers" for semis.
Note: "P/E may be distorted for companies with low/negative EPS"

### 6. SENTIMENT - SHOW SAMPLE SIZE
Always state:
- Number of articles/posts analyzed
- Data source (e.g., "FMP News API", "Social Sentiment API")
- Time period (e.g., "last 7 days")

### 7. VALUATION - SHOW ASSUMPTIONS
For Graham/DCF values, state assumptions if available:
- Graham: EPS used, growth rate assumed
- DCF: WACC, terminal growth, FCF base
- Note: "Intrinsic value is model-dependent; actual value may differ"

### 8. EARNINGS CALENDAR
- Use EXACT date from provided data
- Format: "Feb 25, 2026 (AMC)" or "Not yet announced"
- Show historical beat rate with sample size: "Beat rate: 80% (last 8 quarters)"

### 9. WEB CITATIONS (MANDATORY)
- INLINE citations: "Statement [Source Name](URL)"
- Include "## Sources" section at end

### 10. TRADING RULES - HOLDER vs NEW INVESTOR
Separate recommendations:
**NEW INVESTORS:**
- Entry conditions (what must happen before entering)
- Entry price zone with rationale

**EXISTING HOLDERS:**
- Reduce trigger: price level + ATR/structure logic
- Exit trigger: price level + logic
- Trailing stop methodology

### 11. INVESTOR PROFILE SCENARIOS (REQUIRED)
Include TWO separate analysis perspectives:

**SCENARIO 1 - GROWTH INVESTOR:**
Focus: Capital appreciation, willing to accept higher valuation
Key metrics to emphasize:
- Revenue growth rate (YoY, 5Y CAGR)
- EPS growth rate
- Market position and competitive moat
- Total Addressable Market (TAM) expansion
- P/E premium justified by growth? (PEG ratio if available)
Verdict: Is this attractive for a growth investor? Why/why not?

**SCENARIO 2 - DIVIDEND/VALUE INVESTOR:**
Focus: Steady income, valuation discipline
Key metrics to emphasize:
- Dividend yield vs sector average
- Payout ratio sustainability
- Free cash flow yield
- P/E vs historical average and peers
- Graham/DCF value vs current price (margin of safety)
Verdict: Is this attractive for a dividend/value investor? Why/why not?

## OUTPUT FORMAT

Generate report with INPUT → OUTPUT format for each section:

### PART 1: 5-STEP DATA ANALYSIS
1. **Technical Analysis** - Show: RSI, MACD, ADX, MAs, Volume with actual values
2. **Market Position** - Show: RS values (21d/63d/126d), Sector(GICS)/Industry separation
3. **Risk Analysis** - Show: ATR, VaR, Volatility with calculations for stops
4. **Sentiment Analysis** - Show: Score, sample size, source, time period
5. **Fundamental Analysis** - Show: Valuation metrics, peer table (correct peers)

### PART 2: INVESTOR PROFILE ANALYSIS
6. **Growth Investor Perspective** - Revenue/EPS growth, TAM, competitive position
7. **Dividend/Value Investor Perspective** - Yield, payout ratio, margin of safety
8. **Fair Value Assessment** - Show assumptions, note model limitations
9. **Scenario Analysis** - Bull/Base/Bear with price targets and triggers

### PART 3: NEWS & CATALYSTS
10. **News & Catalysts** - Inline citations + Sources section

### PART 4: CONCLUSION
11. **Executive Summary** - Key data points from each section
12. **Action Plan** - Separate for NEW vs EXISTING investors with ATR/structure logic

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
            # PHASE 3: Fetch additional data in parallel (Enhanced with Phase 2/3 data)
            # =================================================================
            yield {
                "type": "progress",
                "step": "enrichment",
                "message": "Fetching earnings, peers, analyst consensus, and news..."
            }

            # Parallel fetch: earnings, peers (FMP API), analyst consensus, insider trading, web search
            earnings_task = self._fetch_earnings_calendar(symbol)
            peers_task = self._fetch_peer_comparison_dynamic(symbol)
            analyst_task = self._fetch_analyst_consensus(symbol)  # Phase 2: Analyst consensus
            insider_task = self._fetch_insider_trading(symbol)    # Phase 2: Insider trading
            seasonal_task = self._fetch_seasonal_analysis(symbol) # Phase 3: Seasonal patterns

            web_task = None
            if include_web_search:
                web_task = self._fetch_web_enrichment(symbol, step_data, scoring_result)

            tasks = [earnings_task, peers_task, analyst_task, insider_task, seasonal_task]
            if web_task:
                tasks.append(web_task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            earnings_data = results[0] if not isinstance(results[0], Exception) else None
            peer_data = results[1] if not isinstance(results[1], Exception) else None
            analyst_data = results[2] if not isinstance(results[2], Exception) else None
            insider_data = results[3] if not isinstance(results[3], Exception) else None
            seasonal_data = results[4] if not isinstance(results[4], Exception) else None
            web_data = results[5] if len(results) > 5 and not isinstance(results[5], Exception) else None

            # Log what data was fetched successfully
            self.logger.info(f"[SynthesisV2] Data fetch results for {symbol}:")
            self.logger.info(f"  - Earnings: {'✓' if earnings_data else '✗'}")
            self.logger.info(f"  - Peers: {'✓' if peer_data else '✗'}")
            self.logger.info(f"  - Analyst Consensus: {'✓' if analyst_data else '✗'}")
            self.logger.info(f"  - Insider Trading: {'✓' if insider_data else '✗'}")
            self.logger.info(f"  - Seasonal Analysis: {'✓' if seasonal_data else '✗'}")
            self.logger.info(f"  - Web Search: {'✓' if web_data else '✗'}")

            # =================================================================
            # Phase 1: Data Consistency Check
            # =================================================================
            consistency_check = self._check_data_consistency(step_data, symbol)
            if consistency_check["issues"]:
                self.logger.warning(f"[SynthesisV2] Data consistency issues: {consistency_check['issues']}")
            if consistency_check["warnings"]:
                self.logger.info(f"[SynthesisV2] Data consistency warnings: {consistency_check['warnings']}")

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
                target_language=target_language,
                analyst_data=analyst_data,    # Phase 2: Pass analyst data
                insider_data=insider_data,    # Phase 2: Pass insider data
                seasonal_data=seasonal_data   # Phase 3: Pass seasonal data
            )

            # Calculate trading plan
            trading_plan = self._calculate_trading_plan(
                symbol=symbol,
                step_data=step_data,
                scoring=scoring_result
            )

            # Calculate scenario analysis with analyst data for better probabilities
            scenario_analysis = self._calculate_scenario_analysis(
                symbol=symbol,
                step_data=step_data,
                scoring=scoring_result,
                analyst_data=analyst_data  # Phase 2: Use analyst consensus for probabilities
            )

            # Add trading plan and scenarios to prompt
            consolidated_prompt += f"\n\n{self._format_trading_plan_section(trading_plan)}"
            consolidated_prompt += f"\n\n{self._format_scenario_section(scenario_analysis)}"

            # =================================================================
            # LOG STEP DATA FOR DEBUGGING/AUDIT
            # =================================================================
            self._log_step_data_summary(symbol, step_data, earnings_data, peer_data)

            messages = [
                {"role": "system", "content": CONSOLIDATED_SYSTEM_PROMPT},
                {"role": "user", "content": consolidated_prompt}
            ]

            # Log prompt length for monitoring
            self.logger.info(f"[SynthesisV2] {symbol} - Prompt length: {len(consolidated_prompt)} chars")

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
        Fetch earnings calendar with next earnings date (Phase 1: Improved).

        Uses TWO FMP endpoints:
        1. /api/v3/earning_calendar - UPCOMING earnings
        2. /stable/earnings - HISTORICAL earnings (beat rate)

        Phase 1 Improvements:
        - Fiscal year validation based on company fiscal year end
        - Confirmation status (confirmed vs estimated)
        - Historical beat/miss patterns with magnitude
        """
        if not self.fmp_api_key:
            self.logger.warning("[Earnings] No FMP API key available")
            return None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                today = datetime.now().date()
                from_date = today.strftime("%Y-%m-%d")
                to_date = (today + timedelta(days=120)).strftime("%Y-%m-%d")

                # Fetch company profile for fiscal year end month
                profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
                profile_params = {"apikey": self.fmp_api_key}
                fiscal_year_end_month = None

                try:
                    profile_response = await client.get(profile_url, params=profile_params)
                    if profile_response.status_code == 200:
                        profile_data = profile_response.json()
                        if profile_data and isinstance(profile_data, list):
                            # FMP returns fiscalYearEnd as month name (e.g., "December")
                            fy_end = profile_data[0].get("fiscalYearEnd", "")
                            fiscal_year_end_month = self._month_name_to_number(fy_end)
                except Exception as e:
                    self.logger.debug(f"[Earnings] Could not fetch fiscal year end: {e}")

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
                                # Determine confirmation status
                                eps_estimated = item.get("epsEstimated")
                                revenue_estimated = item.get("revenueEstimated")
                                # If we have estimates, consider it more "confirmed"
                                is_confirmed = eps_estimated is not None and revenue_estimated is not None

                                next_earnings = {
                                    "date": item.get("date"),
                                    "time": item.get("time", "TBD"),
                                    "eps_estimated": eps_estimated,
                                    "revenue_estimated": revenue_estimated,
                                    "is_confirmed": is_confirmed,
                                    "confirmation_status": "Confirmed (estimates available)" if is_confirmed else "Estimated (no analyst estimates)"
                                }
                                break

                # Fetch HISTORICAL earnings for beat rate
                historical_url = "https://financialmodelingprep.com/stable/earnings"
                historical_params = {"symbol": symbol, "apikey": self.fmp_api_key}

                historical = []
                beat_rate = None
                avg_surprise_pct = None

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

                        # Calculate beat rate and average surprise
                        beats = 0
                        total = 0
                        surprises = []

                        for h in historical:
                            actual = h.get("epsActual")
                            estimated = h.get("epsEstimated")
                            if actual is not None and estimated is not None:
                                total += 1
                                if actual > estimated:
                                    beats += 1
                                # Calculate surprise percentage
                                if estimated != 0:
                                    surprise_pct = ((actual - estimated) / abs(estimated)) * 100
                                    surprises.append(surprise_pct)

                        beat_rate = round(beats / total, 2) if total > 0 else None
                        avg_surprise_pct = round(sum(surprises) / len(surprises), 1) if surprises else None

                # Determine fiscal quarter with company-specific FY end
                fiscal_quarter = self._determine_fiscal_quarter_v2(
                    date_str=next_earnings.get("date") if next_earnings else None,
                    fiscal_year_end_month=fiscal_year_end_month
                )

                return {
                    "next_earnings_date": next_earnings.get("date") if next_earnings else None,
                    "earnings_time": next_earnings.get("time", "TBD") if next_earnings else "TBD",
                    "eps_estimated": next_earnings.get("eps_estimated") if next_earnings else None,
                    "revenue_estimated": next_earnings.get("revenue_estimated") if next_earnings else None,
                    "is_confirmed": next_earnings.get("is_confirmed", False) if next_earnings else False,
                    "confirmation_status": next_earnings.get("confirmation_status", "Unknown") if next_earnings else "Unknown",
                    "fiscal_quarter": fiscal_quarter,
                    "fiscal_year_end_month": fiscal_year_end_month,
                    "historical": historical[:4],
                    "beat_rate": beat_rate,
                    "avg_surprise_pct": avg_surprise_pct,
                    "quarters_analyzed": len([h for h in historical if h.get("epsActual") and h.get("epsEstimated")]),
                    "source": "FMP Earnings Calendar API"
                }

        except Exception as e:
            self.logger.error(f"[Earnings] Error fetching for {symbol}: {e}")
            return None

    def _month_name_to_number(self, month_name: str) -> Optional[int]:
        """Convert month name to number (e.g., 'December' -> 12)."""
        month_map = {
            "january": 1, "february": 2, "march": 3, "april": 4,
            "may": 5, "june": 6, "july": 7, "august": 8,
            "september": 9, "october": 10, "november": 11, "december": 12
        }
        if not month_name:
            return None
        return month_map.get(month_name.lower())

    def _determine_fiscal_quarter_v2(
        self,
        date_str: Optional[str],
        fiscal_year_end_month: Optional[int] = None
    ) -> str:
        """
        Determine fiscal quarter from date with company-specific fiscal year end.

        Most companies use calendar year (Dec end), but some use different FY ends:
        - Apple: Sep end (FY2024 = Oct 2023 - Sep 2024)
        - Microsoft: Jun end
        - Walmart: Jan end

        Args:
            date_str: Earnings date in YYYY-MM-DD format
            fiscal_year_end_month: Month number when fiscal year ends (1-12)

        Returns:
            Fiscal quarter string (e.g., "Q1 FY2026")
        """
        if not date_str:
            return "N/A"

        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            month = date.month
            year = date.year

            # Default to calendar year (December end) if not specified
            fy_end = fiscal_year_end_month or 12

            # Calculate which quarter the earnings date falls in
            # Earnings are reported AFTER the quarter ends, typically 4-6 weeks later
            # So Feb earnings = Q4 (Oct-Dec), May earnings = Q1 (Jan-Mar), etc.

            if fy_end == 12:
                # Standard calendar year
                if month in [1, 2, 3]:
                    return f"Q4 FY{year - 1}"  # Reporting Q4 of previous year
                elif month in [4, 5, 6]:
                    return f"Q1 FY{year}"
                elif month in [7, 8, 9]:
                    return f"Q2 FY{year}"
                else:
                    return f"Q3 FY{year}"
            else:
                # Non-standard fiscal year
                # Calculate fiscal year start month
                fy_start = (fy_end % 12) + 1

                # Determine which fiscal quarter
                months_from_fy_start = (month - fy_start) % 12
                quarter = (months_from_fy_start // 3) + 1

                # Determine fiscal year
                if month > fy_end:
                    fiscal_year = year + 1
                else:
                    fiscal_year = year

                # Adjust for reporting lag (earnings report previous quarter)
                if quarter == 1:
                    return f"Q4 FY{fiscal_year - 1}"
                else:
                    return f"Q{quarter - 1} FY{fiscal_year}"

        except Exception as e:
            self.logger.debug(f"[Earnings] Error determining fiscal quarter: {e}")
            return "N/A"

    def _determine_fiscal_quarter(self, date_str: Optional[str]) -> str:
        """Determine fiscal quarter from date (legacy method for backward compatibility)."""
        return self._determine_fiscal_quarter_v2(date_str)

    # =========================================================================
    # PHASE 2: ANALYST CONSENSUS API
    # =========================================================================

    async def _fetch_analyst_consensus(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch analyst consensus data from FMP API.

        Endpoints:
        1. /api/v4/price-target-consensus - Price target consensus
        2. /api/v4/grades-consensus - Buy/Hold/Sell consensus

        Returns:
            Dict with analyst consensus data
        """
        if not self.fmp_api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Fetch price target consensus
                pt_url = f"https://financialmodelingprep.com/api/v4/price-target-consensus"
                pt_params = {"symbol": symbol, "apikey": self.fmp_api_key}

                price_target = None
                pt_response = await client.get(pt_url, params=pt_params)

                if pt_response.status_code == 200:
                    pt_data = pt_response.json()
                    if pt_data and isinstance(pt_data, list) and pt_data:
                        price_target = pt_data[0]

                # Fetch grades consensus
                grades_url = f"https://financialmodelingprep.com/api/v4/grades-consensus"
                grades_params = {"symbol": symbol, "apikey": self.fmp_api_key}

                grades = None
                grades_response = await client.get(grades_url, params=grades_params)

                if grades_response.status_code == 200:
                    grades_data = grades_response.json()
                    if grades_data and isinstance(grades_data, list) and grades_data:
                        grades = grades_data[0]

                # Calculate consensus metrics
                result = {
                    "symbol": symbol,
                    "price_target": {},
                    "ratings": {},
                    "analyst_count": 0,
                    "source": "FMP Analyst Consensus API"
                }

                if price_target:
                    result["price_target"] = {
                        "high": price_target.get("targetHigh"),
                        "low": price_target.get("targetLow"),
                        "consensus": price_target.get("targetConsensus"),
                        "median": price_target.get("targetMedian")
                    }

                if grades:
                    strong_buy = grades.get("strongBuy", 0)
                    buy = grades.get("buy", 0)
                    hold = grades.get("hold", 0)
                    sell = grades.get("sell", 0)
                    strong_sell = grades.get("strongSell", 0)

                    total = strong_buy + buy + hold + sell + strong_sell

                    result["ratings"] = {
                        "strong_buy": strong_buy,
                        "buy": buy,
                        "hold": hold,
                        "sell": sell,
                        "strong_sell": strong_sell,
                        "total": total,
                        "bullish_pct": round((strong_buy + buy) / total * 100, 1) if total > 0 else 0,
                        "bearish_pct": round((sell + strong_sell) / total * 100, 1) if total > 0 else 0
                    }
                    result["analyst_count"] = total

                    # Calculate consensus rating
                    if total > 0:
                        bullish_pct = (strong_buy + buy) / total
                        bearish_pct = (sell + strong_sell) / total

                        if bullish_pct > 0.7:
                            result["consensus_rating"] = "Strong Buy"
                        elif bullish_pct > 0.5:
                            result["consensus_rating"] = "Buy"
                        elif bearish_pct > 0.5:
                            result["consensus_rating"] = "Sell"
                        elif bearish_pct > 0.7:
                            result["consensus_rating"] = "Strong Sell"
                        else:
                            result["consensus_rating"] = "Hold"

                return result if result["analyst_count"] > 0 or result["price_target"] else None

        except Exception as e:
            self.logger.error(f"[Analyst] Error fetching consensus for {symbol}: {e}")
            return None

    # =========================================================================
    # PHASE 2: INSIDER TRADING API
    # =========================================================================

    async def _fetch_insider_trading(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch insider trading data from FMP API.

        Endpoints:
        1. /api/v4/insider-trading - Recent insider transactions
        2. /api/v4/insider-trading-statistics - Aggregated statistics

        Returns:
            Dict with insider trading data
        """
        if not self.fmp_api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Fetch recent insider trades
                trades_url = f"https://financialmodelingprep.com/api/v4/insider-trading"
                trades_params = {"symbol": symbol, "limit": 50, "apikey": self.fmp_api_key}

                trades = []
                trades_response = await client.get(trades_url, params=trades_params)

                if trades_response.status_code == 200:
                    trades_data = trades_response.json()
                    if isinstance(trades_data, list):
                        trades = trades_data

                # Fetch aggregated statistics (if available)
                stats_url = f"https://financialmodelingprep.com/api/v4/insider-trading-statistics"
                stats_params = {"symbol": symbol, "apikey": self.fmp_api_key}

                stats = None
                try:
                    stats_response = await client.get(stats_url, params=stats_params)
                    if stats_response.status_code == 200:
                        stats_data = stats_response.json()
                        if stats_data and isinstance(stats_data, list) and stats_data:
                            stats = stats_data[0]
                except:
                    pass  # Stats endpoint may not be available for all plans

                # Analyze recent trades (last 90 days)
                ninety_days_ago = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                recent_trades = [t for t in trades if t.get("filingDate", "") >= ninety_days_ago]

                # Aggregate buy/sell activity
                total_bought = 0
                total_sold = 0
                buy_value = 0
                sell_value = 0
                notable_trades = []

                for trade in recent_trades[:20]:  # Limit to most recent 20
                    shares = trade.get("securitiesTransacted", 0) or 0
                    price = trade.get("price", 0) or 0
                    value = shares * price
                    tx_type = trade.get("transactionType", "").lower()

                    if "purchase" in tx_type or "buy" in tx_type or tx_type == "p":
                        total_bought += shares
                        buy_value += value
                    elif "sale" in tx_type or "sell" in tx_type or tx_type == "s":
                        total_sold += shares
                        sell_value += value

                    # Track notable trades (> $100k or by C-suite)
                    reporter = trade.get("reportingName", "")
                    is_notable = (
                        value > 100000 or
                        any(title in reporter.lower() for title in ["ceo", "cfo", "coo", "cto", "president", "director"])
                    )
                    if is_notable:
                        notable_trades.append({
                            "date": trade.get("filingDate"),
                            "insider": reporter,
                            "type": "Buy" if "purchase" in tx_type or "buy" in tx_type else "Sell",
                            "shares": shares,
                            "price": price,
                            "value": value
                        })

                # Calculate net activity
                net_shares = total_bought - total_sold
                net_value = buy_value - sell_value

                # Determine insider sentiment
                if net_value > 1000000:
                    sentiment = "Strongly Bullish"
                elif net_value > 100000:
                    sentiment = "Bullish"
                elif net_value < -1000000:
                    sentiment = "Strongly Bearish"
                elif net_value < -100000:
                    sentiment = "Bearish"
                else:
                    sentiment = "Neutral"

                result = {
                    "symbol": symbol,
                    "period": "90 days",
                    "total_trades": len(recent_trades),
                    "buy_activity": {
                        "shares": total_bought,
                        "value": buy_value
                    },
                    "sell_activity": {
                        "shares": total_sold,
                        "value": sell_value
                    },
                    "net_activity": {
                        "shares": net_shares,
                        "value": net_value,
                        "sentiment": sentiment
                    },
                    "notable_trades": notable_trades[:5],  # Top 5 notable
                    "source": "FMP Insider Trading API"
                }

                if stats:
                    result["statistics"] = {
                        "total_insiders": stats.get("totalInsiderBought", 0) + stats.get("totalInsiderSold", 0),
                        "buyers": stats.get("totalInsiderBought", 0),
                        "sellers": stats.get("totalInsiderSold", 0)
                    }

                return result if recent_trades else None

        except Exception as e:
            self.logger.error(f"[Insider] Error fetching trading data for {symbol}: {e}")
            return None

    # =========================================================================
    # PHASE 3: SEASONAL ANALYSIS
    # =========================================================================

    async def _fetch_seasonal_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Calculate seasonal performance patterns from historical data.

        Analyzes monthly and quarterly performance patterns over multiple years.

        Returns:
            Dict with seasonal analysis data
        """
        if not self.fmp_api_key:
            return None

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                # Fetch 5 years of historical data
                url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
                params = {"apikey": self.fmp_api_key}

                response = await client.get(url, params=params)

                if response.status_code != 200:
                    return None

                data = response.json()
                historical = data.get("historical", [])

                if len(historical) < 252:  # Need at least 1 year of data
                    return None

                # Calculate monthly returns
                monthly_returns = {}
                for i in range(1, 12 + 1):
                    monthly_returns[i] = []

                # Sort by date ascending
                historical = sorted(historical, key=lambda x: x.get("date", ""))

                # Group by month and calculate returns
                for i in range(1, len(historical)):
                    try:
                        curr_date = datetime.strptime(historical[i]["date"], "%Y-%m-%d")
                        prev_date = datetime.strptime(historical[i-1]["date"], "%Y-%m-%d")

                        # Only calculate for different months
                        if curr_date.month != prev_date.month:
                            curr_price = historical[i]["close"]
                            prev_price = historical[i-1]["close"]

                            if prev_price > 0:
                                monthly_return = ((curr_price - prev_price) / prev_price) * 100
                                monthly_returns[curr_date.month].append(monthly_return)
                    except:
                        continue

                # Calculate statistics for each month
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

                seasonal_stats = {}
                for month, returns in monthly_returns.items():
                    if len(returns) >= 3:  # Need at least 3 data points
                        avg_return = sum(returns) / len(returns)
                        win_rate = len([r for r in returns if r > 0]) / len(returns) * 100
                        seasonal_stats[month_names[month - 1]] = {
                            "avg_return": round(avg_return, 2),
                            "win_rate": round(win_rate, 1),
                            "sample_size": len(returns)
                        }

                # Find best and worst months
                best_month = max(seasonal_stats.items(), key=lambda x: x[1]["avg_return"]) if seasonal_stats else None
                worst_month = min(seasonal_stats.items(), key=lambda x: x[1]["avg_return"]) if seasonal_stats else None

                # Calculate quarterly patterns
                quarterly_stats = {}
                quarter_map = {
                    "Q1": [1, 2, 3],
                    "Q2": [4, 5, 6],
                    "Q3": [7, 8, 9],
                    "Q4": [10, 11, 12]
                }

                for quarter, months in quarter_map.items():
                    quarter_returns = []
                    for m in months:
                        quarter_returns.extend(monthly_returns.get(m, []))
                    if quarter_returns:
                        quarterly_stats[quarter] = {
                            "avg_return": round(sum(quarter_returns) / len(quarter_returns), 2),
                            "win_rate": round(len([r for r in quarter_returns if r > 0]) / len(quarter_returns) * 100, 1)
                        }

                return {
                    "symbol": symbol,
                    "years_analyzed": len(historical) // 252,
                    "monthly_patterns": seasonal_stats,
                    "quarterly_patterns": quarterly_stats,
                    "best_month": {"month": best_month[0], **best_month[1]} if best_month else None,
                    "worst_month": {"month": worst_month[0], **worst_month[1]} if worst_month else None,
                    "source": "Calculated from FMP Historical Data"
                }

        except Exception as e:
            self.logger.error(f"[Seasonal] Error calculating for {symbol}: {e}")
            return None

    # =========================================================================
    # PHASE 1: DATA CONSISTENCY CHECKER
    # =========================================================================

    def _check_data_consistency(
        self,
        step_data: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Check data consistency across all steps.

        Validates:
        - Price consistency across steps
        - Timestamp freshness
        - Data completeness

        Returns:
            Dict with consistency check results
        """
        issues = []
        warnings = []
        prices = {}
        timestamps = {}

        # Collect prices from each step
        for step_name, data in step_data.items():
            raw = data.get("raw_data", {})
            cached_at = data.get("cached_at")

            if cached_at:
                timestamps[step_name] = cached_at

            # Extract price from different step structures
            if step_name == "risk":
                price = raw.get("current_price")
            elif step_name == "technical":
                price = raw.get("current_price") or raw.get("price_context", {}).get("current_price")
            elif step_name == "fundamental":
                report = raw.get("fundamental_report", {})
                price = report.get("intrinsic_value", {}).get("current_price")
            else:
                price = None

            if price:
                prices[step_name] = price

        # Check price consistency (allow 2% variance for timing differences)
        if len(prices) >= 2:
            price_values = list(prices.values())
            avg_price = sum(price_values) / len(price_values)
            for step, price in prices.items():
                variance = abs(price - avg_price) / avg_price * 100
                if variance > 2:
                    warnings.append(
                        f"Price variance in {step}: ${price:.2f} vs avg ${avg_price:.2f} ({variance:.1f}% diff)"
                    )

        # Check timestamp freshness
        now = datetime.now()
        for step, ts_str in timestamps.items():
            try:
                # Parse timestamp (assuming ISO format)
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                age_hours = (now - ts.replace(tzinfo=None)).total_seconds() / 3600

                if age_hours > 24:
                    issues.append(f"{step} data is {age_hours:.1f} hours old (stale)")
                elif age_hours > 6:
                    warnings.append(f"{step} data is {age_hours:.1f} hours old")
            except:
                pass

        # Check data completeness
        required_steps = ["technical", "risk", "fundamental"]
        for step in required_steps:
            if step not in step_data:
                issues.append(f"Missing required step: {step}")
            elif not step_data[step].get("raw_data"):
                warnings.append(f"No raw_data in {step} step")

        return {
            "symbol": symbol,
            "is_consistent": len(issues) == 0,
            "prices_found": prices,
            "issues": issues,
            "warnings": warnings,
            "timestamp_check": timestamps
        }

    # =========================================================================
    # LEGACY COMPATIBILITY
    # =========================================================================

    def _determine_fiscal_quarter_legacy(self, date_str: Optional[str]) -> str:
        """Determine fiscal quarter from date (legacy)."""
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
        scoring: Dict[str, Any],
        analyst_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate Bull/Base/Bear scenarios with price targets and probabilities.

        Phase 2 Improvement: Use analyst consensus + volatility for probability estimation
        instead of hardcoded composite score thresholds.

        Methodology:
        1. Base targets on analyst price targets when available
        2. Adjust probabilities based on:
           - Analyst consensus rating (bullish% vs bearish%)
           - Historical volatility (higher vol = more uncertainty)
           - Earnings beat rate (higher beat rate = more bullish)
        """
        risk_raw = step_data.get("risk", {}).get("raw_data", {})
        fund_raw = step_data.get("fundamental", {}).get("raw_data", {})

        current_price = risk_raw.get("current_price", 0)
        atr = risk_raw.get("atr", {}).get("value", 0) if isinstance(risk_raw.get("atr"), dict) else 0
        volatility_data = risk_raw.get("volatility", {})
        volatility = volatility_data.get("annualized", 0) if isinstance(volatility_data, dict) else 0

        # Get intrinsic values if available
        intrinsic = fund_raw.get("fundamental_report", {}).get("intrinsic_value", {}) if fund_raw else {}
        graham_value = intrinsic.get("graham_value")
        dcf_value = intrinsic.get("dcf_value")

        composite_score = scoring.get("composite_score", 50)

        # =================================================================
        # PHASE 2: Improved target calculation using analyst data
        # =================================================================
        bull_target = base_target = bear_target = 0

        if current_price > 0:
            # Try to use analyst price targets if available
            if analyst_data and analyst_data.get("price_target"):
                pt = analyst_data["price_target"]
                pt_high = pt.get("high")
                pt_low = pt.get("low")
                pt_consensus = pt.get("consensus") or pt.get("median")

                if pt_high and pt_low and pt_consensus:
                    # Use analyst targets with some adjustment
                    bull_target = round(pt_high * 0.95, 2)  # Slightly conservative
                    base_target = round(pt_consensus, 2)
                    bear_target = round(pt_low * 1.05, 2)  # Slightly less pessimistic

                    self.logger.debug(f"[Scenario] Using analyst targets: Bull=${bull_target}, Base=${base_target}, Bear=${bear_target}")

            # Fallback to volatility-based calculation
            if not bull_target:
                # Use volatility to scale targets (higher vol = wider range)
                vol_factor = max(volatility / 30, 0.5)  # Normalize to 30% vol baseline
                vol_factor = min(vol_factor, 2.0)  # Cap at 2x

                bull_multiplier = 1.15 + (0.10 * vol_factor)  # 1.15-1.35
                bear_multiplier = 0.90 - (0.10 * vol_factor)  # 0.70-0.90

                bull_target = round(current_price * bull_multiplier, 2)
                base_target = round(current_price * 1.05, 2)  # Conservative 5% base
                bear_target = round(current_price * bear_multiplier, 2)

        # =================================================================
        # PHASE 2: Improved probability calculation
        # =================================================================

        # Start with base probabilities
        bull_prob, base_prob, bear_prob = 25, 50, 25

        # Adjust based on analyst consensus (if available)
        if analyst_data and analyst_data.get("ratings"):
            ratings = analyst_data["ratings"]
            bullish_pct = ratings.get("bullish_pct", 50)
            bearish_pct = ratings.get("bearish_pct", 50)

            # Weight analyst consensus heavily (60% weight)
            bull_prob = round(25 + (bullish_pct - 50) * 0.6, 0)
            bear_prob = round(25 + (bearish_pct - 50) * 0.6, 0)
            base_prob = 100 - bull_prob - bear_prob

            # Log the methodology
            self.logger.debug(f"[Scenario] Analyst-adjusted probs: Bull={bull_prob}%, Bear={bear_prob}%")

        else:
            # Fallback: Use composite score (legacy method)
            if composite_score >= 70:
                bull_prob, base_prob, bear_prob = 35, 45, 20
            elif composite_score >= 55:
                bull_prob, base_prob, bear_prob = 30, 45, 25
            elif composite_score >= 45:
                bull_prob, base_prob, bear_prob = 25, 45, 30
            else:
                bull_prob, base_prob, bear_prob = 20, 40, 40

        # Adjust for volatility (higher vol = less certainty in extreme outcomes)
        if volatility > 40:
            # High volatility: flatten probabilities toward 33/33/33
            flatten_factor = min((volatility - 40) / 40, 0.3)  # Max 30% flattening
            bull_prob = round(bull_prob * (1 - flatten_factor) + 33 * flatten_factor)
            bear_prob = round(bear_prob * (1 - flatten_factor) + 33 * flatten_factor)
            base_prob = 100 - bull_prob - bear_prob

        # Ensure probabilities are valid (sum to 100, all positive)
        bull_prob = max(10, min(50, int(bull_prob)))
        bear_prob = max(10, min(50, int(bear_prob)))
        base_prob = 100 - bull_prob - bear_prob

        # Build methodology explanation
        methodology = []
        if analyst_data and analyst_data.get("ratings"):
            methodology.append(f"Analyst consensus: {analyst_data['ratings'].get('bullish_pct', 0):.0f}% bullish")
            methodology.append(f"Analyst count: {analyst_data.get('analyst_count', 0)}")
        else:
            methodology.append("No analyst data - using technical indicators")

        if volatility:
            methodology.append(f"Volatility factor: {volatility:.1f}% annualized")

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
            "probability_methodology": methodology,
            "time_horizon": "3-6 months"
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
        premium_discount = fair_value.get("premium_discount", {})
        if premium_discount.get("assessment") and premium_discount.get("assessment") != "N/A":
            lines.extend([
                "",
                "### FAIR VALUE ASSESSMENT:",
                f"- Assessment: {premium_discount['assessment']}",
            ])
            if premium_discount.get("avg_fair_value"):
                lines.append(f"- Average Fair Value: ${premium_discount['avg_fair_value']:.2f}")
                lines.append(f"- Premium/Discount: {premium_discount['avg_diff_pct']:+.1f}%")

            details = premium_discount.get("details", {})
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
        target_language: str,
        analyst_data: Optional[Dict[str, Any]] = None,
        insider_data: Optional[Dict[str, Any]] = None,
        seasonal_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build comprehensive prompt using RAW DATA metrics instead of truncated LLM content.

        Phase 2/3 Enhancement: Includes analyst consensus, insider trading, and seasonal data.
        """
        parts = [
            f"# COMPREHENSIVE INVESTMENT ANALYSIS: {symbol}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Target Language: {target_language.upper()}",
            "",
            "IMPORTANT: Present RAW DATA with INPUT → OUTPUT format.",
            "DO NOT create or display any composite scores or ratings.",
            "Let the data speak for itself - reader interprets.",
            "",
        ]

        # =====================================================================
        # SECTION 1: TECHNICAL ANALYSIS (LLM Summary + Raw Data)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## TECHNICAL ANALYSIS DATA",
            "=" * 60,
            "",
        ])

        tech_data = step_data.get("technical", {})
        tech_content = tech_data.get("content", "")
        tech_raw = tech_data.get("raw_data", {})

        # Include LLM summary first (comprehensive analysis)
        if tech_content:
            parts.append("### LLM ANALYSIS SUMMARY:")
            parts.append(tech_content[:3000])  # Limit to avoid token overflow
            parts.append("")

        # Then include raw metrics for verification
        if tech_raw:
            parts.extend(self._format_technical_raw_data(tech_raw))
        parts.append("")

        # =====================================================================
        # SECTION 2: MARKET POSITION (LLM Summary + Raw Data)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## MARKET POSITION DATA",
            "=" * 60,
            "",
        ])

        pos_data = step_data.get("position", {})
        pos_content = pos_data.get("content", "")
        pos_raw = pos_data.get("raw_data", {})
        sector_context = pos_data.get("sector_context", {})

        if pos_content:
            parts.append("### LLM ANALYSIS SUMMARY:")
            parts.append(pos_content[:2000])
            parts.append("")

        if pos_raw or sector_context:
            parts.extend(self._format_position_raw_data(pos_raw, sector_context))
        parts.append("")

        # =====================================================================
        # SECTION 3: RISK ANALYSIS (LLM Summary + Raw Data)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## RISK ANALYSIS DATA",
            "=" * 60,
            "",
        ])

        risk_data = step_data.get("risk", {})
        risk_content = risk_data.get("content", "")
        risk_raw = risk_data.get("raw_data", {})

        if risk_content:
            parts.append("### LLM ANALYSIS SUMMARY:")
            parts.append(risk_content[:2000])
            parts.append("")

        if risk_raw:
            parts.extend(self._format_risk_raw_data(risk_raw))
        parts.append("")

        # =====================================================================
        # SECTION 4: SENTIMENT DATA (LLM Summary + Raw Data)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## SENTIMENT DATA",
            "=" * 60,
            "",
        ])

        sent_data = step_data.get("sentiment", {})
        sent_content = sent_data.get("content", "")
        sent_raw = sent_data.get("raw_data", {})

        if sent_content:
            parts.append("### LLM ANALYSIS SUMMARY:")
            parts.append(sent_content[:2000])
            parts.append("")

        if sent_raw:
            parts.extend(self._format_sentiment_raw_data(sent_raw))
        parts.append("")

        # =====================================================================
        # SECTION 5: FUNDAMENTAL DATA (LLM Summary + Raw Data)
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## FUNDAMENTAL DATA",
            "=" * 60,
            "",
        ])

        fund_data = step_data.get("fundamental", {})
        fund_content = fund_data.get("content", "")
        fund_raw = fund_data.get("raw_data", {})

        if fund_content:
            parts.append("### LLM ANALYSIS SUMMARY:")
            parts.append(fund_content[:3000])
            parts.append("")

        if fund_raw:
            parts.extend(self._format_fundamental_raw_data(fund_raw))
        parts.append("")

        # =====================================================================
        # SECTION 7: PEER COMPARISON (DYNAMIC FMP API)
        # =====================================================================
        if peer_data and peer_data.get("peers"):
            industry = peer_data.get('industry', 'N/A')
            sector = peer_data.get('sector', 'N/A')

            parts.extend([
                "=" * 60,
                "## PEER COMPARISON (FMP stock_peers API)",
                "=" * 60,
                "",
                f"**Target:** {symbol}",
                f"**GICS Sector:** {sector}",
                f"**Industry:** {industry}",
                "",
            ])

            # Add industry-specific peer guidance
            if "semiconductor" in industry.lower():
                parts.append("**CORRECT SEMICONDUCTOR PEERS:** AMD, AVGO, TSM, INTC, MRVL, QCOM")
                parts.append("⚠️ Do NOT compare with mega-cap tech (AAPL, MSFT, GOOGL) as 'peers'")
                parts.append("")

            parts.extend([
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
                "**IMPORTANT NOTES:**",
                "- P/E ratios may be distorted for companies with low or volatile EPS",
                "- ROE > 50% may be distorted by buybacks, leverage, or one-time items",
                "- Compare within SAME INDUSTRY, not just same sector",
                f"Source: {peer_data.get('source', 'FMP stock_peers API')}",
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
                parts.append(f"**NEXT EARNINGS: {earnings_data['next_earnings_date']} ({earnings_data.get('earnings_time', 'TBD')})**")
                parts.append(f"Fiscal Quarter: {earnings_data.get('fiscal_quarter', 'N/A')}")
                if earnings_data.get("eps_estimated"):
                    parts.append(f"EPS Estimate: ${earnings_data['eps_estimated']:.2f}")
            else:
                parts.append("Next earnings date: Not yet announced")

            # Beat rate WITH SAMPLE SIZE
            if earnings_data.get("beat_rate") is not None:
                beat_rate = earnings_data['beat_rate'] * 100
                quarters_analyzed = earnings_data.get('quarters_analyzed', 8)
                parts.append(f"Historical Beat Rate: {beat_rate:.0f}% (last {quarters_analyzed} quarters)")
                parts.append(f"  → Sample: {quarters_analyzed} earnings reports analyzed")

            parts.append(f"Source: {earnings_data.get('source', 'FMP Earnings Calendar API')}")
            parts.append("")
            parts.append("⚠️ Use EXACT date from data. Do not invent or estimate earnings dates.")
            parts.append("")

        # =====================================================================
        # SECTION 9: ANALYST CONSENSUS (Phase 2)
        # =====================================================================
        if analyst_data:
            parts.extend([
                "=" * 60,
                "## ANALYST CONSENSUS (FMP API)",
                "=" * 60,
                "",
            ])

            # Price targets
            pt = analyst_data.get("price_target", {})
            if pt.get("consensus"):
                parts.append("**PRICE TARGETS:**")
                parts.append(f"- Consensus Target: ${pt.get('consensus'):.2f}")
                parts.append(f"- Median Target: ${pt.get('median'):.2f}" if pt.get("median") else "")
                parts.append(f"- High: ${pt.get('high'):.2f}" if pt.get("high") else "")
                parts.append(f"- Low: ${pt.get('low'):.2f}" if pt.get("low") else "")
                parts.append("")

            # Ratings breakdown
            ratings = analyst_data.get("ratings", {})
            if ratings.get("total"):
                parts.append(f"**ANALYST RATINGS ({ratings.get('total')} analysts):**")
                parts.append(f"- Strong Buy: {ratings.get('strong_buy', 0)}")
                parts.append(f"- Buy: {ratings.get('buy', 0)}")
                parts.append(f"- Hold: {ratings.get('hold', 0)}")
                parts.append(f"- Sell: {ratings.get('sell', 0)}")
                parts.append(f"- Strong Sell: {ratings.get('strong_sell', 0)}")
                parts.append("")
                parts.append(f"**Summary:** {ratings.get('bullish_pct', 0):.0f}% Bullish, {ratings.get('bearish_pct', 0):.0f}% Bearish")
                if analyst_data.get("consensus_rating"):
                    parts.append(f"**Consensus Rating:** {analyst_data['consensus_rating']}")
                parts.append("")

            parts.append(f"Source: {analyst_data.get('source', 'FMP Analyst Consensus API')}")
            parts.append("")

        # =====================================================================
        # SECTION 10: INSIDER TRADING (Phase 2)
        # =====================================================================
        if insider_data:
            parts.extend([
                "=" * 60,
                "## INSIDER TRADING (FMP API)",
                "=" * 60,
                "",
            ])

            parts.append(f"**Period:** Last {insider_data.get('period', '90 days')}")
            parts.append(f"**Total Trades:** {insider_data.get('total_trades', 0)}")
            parts.append("")

            # Buy/Sell activity
            buy = insider_data.get("buy_activity", {})
            sell = insider_data.get("sell_activity", {})
            net = insider_data.get("net_activity", {})

            parts.append("**ACTIVITY SUMMARY:**")
            parts.append(f"- Shares Bought: {buy.get('shares', 0):,.0f} (${buy.get('value', 0):,.0f})")
            parts.append(f"- Shares Sold: {sell.get('shares', 0):,.0f} (${sell.get('value', 0):,.0f})")
            parts.append(f"- Net Shares: {net.get('shares', 0):+,.0f}")
            parts.append(f"- Net Value: ${net.get('value', 0):+,.0f}")
            parts.append(f"- **Insider Sentiment: {net.get('sentiment', 'Neutral')}**")
            parts.append("")

            # Notable trades
            notable = insider_data.get("notable_trades", [])
            if notable:
                parts.append("**NOTABLE TRADES:**")
                for trade in notable[:3]:
                    parts.append(f"- {trade.get('date')}: {trade.get('insider', 'Unknown')} "
                               f"{trade.get('type')} {trade.get('shares', 0):,.0f} shares at ${trade.get('price', 0):.2f}")
                parts.append("")

            parts.append(f"Source: {insider_data.get('source', 'FMP Insider Trading API')}")
            parts.append("")

        # =====================================================================
        # SECTION 11: SEASONAL ANALYSIS (Phase 3)
        # =====================================================================
        if seasonal_data:
            parts.extend([
                "=" * 60,
                "## SEASONAL ANALYSIS (Historical Patterns)",
                "=" * 60,
                "",
            ])

            years = seasonal_data.get("years_analyzed", 0)
            parts.append(f"**Analysis Period:** {years} years of historical data")
            parts.append("")

            # Best and worst months
            best = seasonal_data.get("best_month")
            worst = seasonal_data.get("worst_month")

            if best:
                parts.append(f"**BEST MONTH:** {best.get('month')} (Avg: {best.get('avg_return', 0):+.1f}%, Win Rate: {best.get('win_rate', 0):.0f}%)")
            if worst:
                parts.append(f"**WORST MONTH:** {worst.get('month')} (Avg: {worst.get('avg_return', 0):+.1f}%, Win Rate: {worst.get('win_rate', 0):.0f}%)")
            parts.append("")

            # Quarterly patterns
            quarterly = seasonal_data.get("quarterly_patterns", {})
            if quarterly:
                parts.append("**QUARTERLY PATTERNS:**")
                for q, data in quarterly.items():
                    parts.append(f"- {q}: Avg {data.get('avg_return', 0):+.1f}%, Win Rate {data.get('win_rate', 0):.0f}%")
                parts.append("")

            parts.append("⚠️ Note: Past seasonal patterns do not guarantee future performance.")
            parts.append(f"Source: {seasonal_data.get('source', 'Calculated from FMP Historical Data')}")
            parts.append("")

        # =====================================================================
        # SECTION 12: WEB SEARCH RESULTS (Title + URL + Content format)
        # =====================================================================
        if web_data and web_data.get("citations"):
            parts.extend([
                "=" * 60,
                "## WEB SEARCH RESULTS (CITE INLINE WITH TITLE + URL)",
                "=" * 60,
                "",
            ])

            # Include search summaries with citations
            for item in web_data.get("news", []):
                parts.append(f"### {item['category'].upper()}")
                parts.append(item["answer"][:2000])
                parts.append("")

            # List all sources with FULL info (Title + URL + snippet)
            parts.append("### SOURCES LIST (Use these for inline citations):")
            parts.append("")
            for i, citation in enumerate(web_data.get("citations", [])[:10], 1):
                title = citation.get("title", "Untitled")
                url = citation.get("url", "")
                snippet = citation.get("snippet", citation.get("content", ""))[:200]

                parts.append(f"**[{i}] {title}**")
                parts.append(f"URL: {url}")
                if snippet:
                    parts.append(f"Snippet: {snippet}...")
                parts.append("")

            parts.extend([
                "### CITATION INSTRUCTIONS:",
                "When using information from web search, ALWAYS cite inline like this:",
                "  'NVIDIA stock dropped 4.4% ([Barron's](https://www.barrons.com/...))'",
                "",
                "Include a '## Sources' section at the END of your report listing all sources used.",
                "",
            ])

        # =====================================================================
        # OUTPUT INSTRUCTIONS
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## OUTPUT INSTRUCTIONS",
            "=" * 60,
            "",
            f"Generate a comprehensive investment report in {target_language.upper()}.",
            "",
            "CRITICAL: Use INPUT → OUTPUT format for each section:",
            "- Show actual data values (RSI=X, ADX=Y, ATR=$Z)",
            "- Then provide analysis based on those values",
            "- DO NOT invent any composite scores or ratings",
            "",
            "Required sections in order:",
            "1. Technical Analysis - Show: RSI, MACD, ADX, MAs with values → interpretation",
            "2. Market Position - Show: RS (21d/63d/126d), Sector vs Industry (GICS distinction)",
            "3. Risk Analysis - Show: ATR, VaR, Volatility → stop-loss calculation with formula",
            "4. Sentiment Analysis - Show: Score, sample size, confidence level, source",
            "5. Fundamental Analysis - Show: Valuation metrics, peer table (use correct industry peers)",
            "6. Analyst Consensus - Price targets, buy/sell ratings, analyst count",
            "7. Insider Trading - Net buy/sell activity, notable trades, insider sentiment",
            "8. Growth Investor Perspective - Revenue/EPS growth focus, TAM, competitive position",
            "9. Dividend/Value Investor Perspective - Yield, payout ratio, margin of safety",
            "10. Fair Value Assessment - Show: Graham/DCF with assumptions, note model limitations",
            "11. Scenario Analysis - Show: Bull/Base/Bear with probability methodology explained",
            "12. Seasonal Patterns - Historical monthly/quarterly performance patterns",
            "13. News & Catalysts - With inline citations (Title + URL) + Sources section",
            "14. Executive Summary - Key DATA POINTS from each section",
            "15. Action Plan - Separate for NEW vs EXISTING investors with ATR/structure logic",
            "",
            "REMINDERS:",
            "- NO scoring, NO ratings - only raw data and interpretation",
            "- Stop-loss MUST show calculation: 'Stop = Entry - (2×ATR=$X) = $Y'",
            "- Sector is GICS (11 sectors), Industry is sub-classification",
            "- Cite web sources inline with [Title](URL) format",
            "- Include holder-specific rules (reduce/exit triggers with logic)",
            "- Show sample sizes for sentiment and analyst data",
            "- Explain probability methodology for scenario analysis",
        ])

        return "\n".join(parts)

    # =========================================================================
    # RAW DATA FORMATTERS
    # =========================================================================

    def _format_technical_raw_data(self, raw: Dict[str, Any]) -> List[str]:
        """Format technical raw data using CORRECT structure from get_technical_indicators."""
        lines = ["### RAW METRICS (Extracted from technical indicators):"]
        lines.append("")

        # Get indicators dict - this is the actual structure
        indicators = raw.get("indicators", {})

        # Current price from price_context
        price_ctx = raw.get("price_context", {})
        current_price = raw.get("current_price") or price_ctx.get("current_price", 0)
        if current_price:
            lines.append(f"**Current Price: ${current_price}**")
            lines.append("")

        # RSI from indicators.rsi
        rsi_data = indicators.get("rsi", {})
        rsi_value = rsi_data.get("value")
        if rsi_value is not None:
            rsi_zone = "Overbought (>70)" if rsi_value > 70 else "Oversold (<30)" if rsi_value < 30 else "Neutral (30-70)"
            lines.append(f"**RSI(14): {rsi_value:.1f}** → Zone: {rsi_zone}")
            if rsi_data.get("signal"):
                lines.append(f"  Signal: {rsi_data['signal']}")
            lines.append("")

        # MACD from indicators.macd
        macd_data = indicators.get("macd", {})
        macd_line = macd_data.get("macd_line")
        macd_signal = macd_data.get("signal_line")
        macd_hist = macd_data.get("histogram")
        if macd_line is not None:
            macd_trend = "Bullish (MACD > Signal)" if macd_line > (macd_signal or 0) else "Bearish (MACD < Signal)"
            lines.append(f"**MACD Line: {macd_line:.4f}**")
            lines.append(f"  Signal Line: {macd_signal:.4f}" if macd_signal else "  Signal Line: N/A")
            lines.append(f"  Histogram: {macd_hist:.4f} → {macd_trend}" if macd_hist else "  Histogram: N/A")
            if macd_data.get("signal"):
                lines.append(f"  Signal: {macd_data['signal']}")
            lines.append("")

        # ADX from indicators.adx
        adx_data = indicators.get("adx", {})
        adx_value = adx_data.get("value")
        if adx_value is not None:
            if adx_value < 15:
                trend_strength = "WEAK (ADX<15: No clear trend)"
            elif adx_value < 25:
                trend_strength = "MODERATE (ADX 15-25: Developing trend)"
            else:
                trend_strength = "STRONG (ADX>25: Established trend)"
            lines.append(f"**ADX(14): {adx_value:.1f}** → {trend_strength}")
            plus_di = adx_data.get("plus_di")
            minus_di = adx_data.get("minus_di")
            if plus_di is not None and minus_di is not None:
                di_direction = "Bullish (+DI > -DI)" if plus_di > minus_di else "Bearish (-DI > +DI)"
                lines.append(f"  +DI: {plus_di:.1f}, -DI: {minus_di:.1f} → {di_direction}")
            lines.append("")

        # Moving Averages from indicators.moving_averages
        ma_data = indicators.get("moving_averages", {})
        sma_data = ma_data.get("sma", {})
        if sma_data:
            lines.append("**Moving Averages:**")
            sma20 = sma_data.get("sma_20", {}).get("value")
            sma50 = sma_data.get("sma_50", {}).get("value")
            sma200 = sma_data.get("sma_200", {}).get("value")
            if sma20: lines.append(f"  SMA20: ${sma20:.2f}")
            if sma50: lines.append(f"  SMA50: ${sma50:.2f}")
            if sma200: lines.append(f"  SMA200: ${sma200:.2f}")
            price_pos = ma_data.get("price_position", {})
            if price_pos:
                above_200 = "Above" if price_pos.get("above_sma_200") else "Below"
                lines.append(f"  Price vs SMA200: {above_200}")
            lines.append("")

        # Bollinger Bands from indicators.bollinger_bands
        bb_data = indicators.get("bollinger_bands", {})
        if bb_data.get("upper"):
            lines.append("**Bollinger Bands:**")
            lines.append(f"  Upper: ${bb_data.get('upper', 0):.2f}")
            lines.append(f"  Middle: ${bb_data.get('middle', 0):.2f}")
            lines.append(f"  Lower: ${bb_data.get('lower', 0):.2f}")
            if bb_data.get("position"):
                lines.append(f"  Position: {bb_data['position']}")
            lines.append("")

        # Volume from indicators.volume
        vol_data = indicators.get("volume", {})
        if vol_data:
            rvol = vol_data.get("rvol")
            if rvol:
                vol_status = "High (RVOL>1.5)" if rvol > 1.5 else "Low (RVOL<0.7)" if rvol < 0.7 else "Normal"
                lines.append(f"**RVOL: {rvol:.2f}x** → {vol_status}")
                lines.append(f"  Volume Trend: {vol_data.get('volume_trend', 'N/A')}")
                lines.append("")

        # Stochastic from indicators.stochastic
        stoch_data = indicators.get("stochastic", {})
        if stoch_data.get("k_value"):
            lines.append(f"**Stochastic:** %K={stoch_data.get('k_value', 0):.1f}, %D={stoch_data.get('d_value', 0):.1f}")
            if stoch_data.get("signal"):
                lines.append(f"  Signal: {stoch_data['signal']}")
            lines.append("")

        return lines

    def _format_position_raw_data(self, raw: Dict[str, Any], sector_ctx: Dict[str, Any]) -> List[str]:
        """
        Format position raw data using CORRECT structure from get_relative_strength.

        Phase 1 Improvement: Uses validate_gics_classification for proper sector validation.
        """
        lines = ["### RAW POSITION METRICS:"]
        lines.append("")

        # GICS Classification - CRITICAL DISTINCTION with validation
        if sector_ctx:
            sector = sector_ctx.get('stock_sector', 'N/A')
            industry = sector_ctx.get('stock_industry', 'N/A')

            # Validate GICS classification
            gics_result = validate_gics_classification(sector, industry)
            normalized_sector = gics_result["sector"]
            is_valid = gics_result["is_valid_gics"]

            lines.append("**GICS Classification:**")
            lines.append(f"- SECTOR (1 of 11 GICS): {normalized_sector}")
            if not is_valid:
                lines.append(f"  ⚠️ Original: '{sector}' (not standard GICS)")
            lines.append(f"- INDUSTRY (sub-category): {industry}")

            # Add validation warnings if any
            if gics_result["warnings"]:
                lines.append("")
                lines.append("⚠️ **Classification Notes:**")
                for warning in gics_result["warnings"]:
                    lines.append(f"  - {warning}")

            lines.append("")

            # Sector ranking
            sector_rank = sector_ctx.get('sector_rank', 'N/A')
            total_sectors = sector_ctx.get('total_sectors', 11)
            sector_change = sector_ctx.get('sector_change_percent', 0)

            lines.append("**Sector Performance (1-DAY):**")
            lines.append(f"- Rank: #{sector_rank}/{total_sectors}")
            lines.append(f"- Change: {sector_change:+.2f}%")
            lines.append(f"- Status: {sector_ctx.get('sector_status', 'N/A')}")
            lines.append("")
            lines.append("⚠️ Note: 1-day ranking ≠ multi-timeframe RS analysis")
            lines.append("")

        # RS Data - from raw which has the structure from get_relative_strength
        if raw:
            # Multi-timeframe RS metrics
            rs_metrics = raw.get("rs_metrics", {})
            if not rs_metrics:
                # Try direct access (different structure)
                rs_metrics = raw

            rs_21d = rs_metrics.get('excess_return_21d') or raw.get('excess_return_21d')
            rs_63d = rs_metrics.get('excess_return_63d') or raw.get('excess_return_63d')
            rs_126d = rs_metrics.get('excess_return_126d') or raw.get('excess_return_126d')

            lines.append("**Relative Strength vs SPY (Multi-Timeframe):**")
            if rs_21d is not None:
                lines.append(f"- 21-day RS: {rs_21d:+.2f}%")
            if rs_63d is not None:
                lines.append(f"- 63-day RS: {rs_63d:+.2f}%")
            if rs_126d is not None:
                lines.append(f"- 126-day RS: {rs_126d:+.2f}%")

            # RS Rating and classification
            rs_rating = raw.get("rs_rating") or rs_metrics.get("rs_rating")
            classification = raw.get("classification") or rs_metrics.get("classification")
            trend = raw.get("trend") or rs_metrics.get("trend")

            if rs_rating:
                lines.append(f"- RS Rating: {rs_rating}")
            if classification:
                lines.append(f"- Classification: {classification}")
            if trend:
                lines.append(f"- Trend: {trend}")
            lines.append("")

        return lines

    def _format_risk_raw_data(self, raw: Dict[str, Any]) -> List[str]:
        """Format risk raw data using CORRECT structure from suggest_stop_loss and risk_analysis."""
        lines = ["### RAW RISK METRICS:"]
        lines.append("")

        # Current price
        current_price = raw.get('current_price', 0)
        if current_price:
            lines.append(f"**Current Price: ${current_price}**")
            lines.append("")

        # ATR from stop_loss_recommendations
        stop_recs = raw.get("stop_loss_recommendations", {})
        atr_data = stop_recs.get("atr_based", {})
        atr_value = atr_data.get("atr_value", 0)
        atr_stop = atr_data.get("stop_price", 0)
        atr_pct = atr_data.get("risk_percent", 0)

        # Also try direct atr key
        if not atr_value:
            atr_value = raw.get("atr", 0)

        if atr_value or atr_stop:
            lines.append("**ATR-Based Stop Loss:**")
            if atr_value:
                lines.append(f"- ATR Value: ${atr_value:.2f}")
            if atr_stop:
                lines.append(f"- ATR Stop Price: ${atr_stop:.2f}")
            if atr_pct:
                lines.append(f"- Risk %: {atr_pct:.1f}%")
            lines.append("")

        # Pre-calculate stop levels if we have the data
        if current_price and atr_value:
            stop_2atr = round(current_price - (2 * atr_value), 2)
            stop_2atr_pct = round((2 * atr_value / current_price) * 100, 1)
            lines.append("**PRE-CALCULATED STOPS:**")
            lines.append(f"- 2× ATR Stop: ${stop_2atr} ({stop_2atr_pct}% risk)")
            lines.append(f"  Formula: ${current_price:.2f} - (2 × ${atr_value:.2f}) = ${stop_2atr}")
            lines.append("")

        # Percentage-based stop
        pct_data = stop_recs.get("percentage_based", {})
        if pct_data:
            lines.append("**Percentage-Based Stop:**")
            lines.append(f"- Stop Price: ${pct_data.get('stop_price', 'N/A')}")
            lines.append(f"- Risk %: {pct_data.get('risk_percent', 'N/A')}%")
            lines.append("")

        # Volatility from volatility key
        vol_data = raw.get("volatility", {})
        if vol_data:
            lines.append("**Volatility:**")
            lines.append(f"- Daily: {vol_data.get('daily', 'N/A')}%")
            lines.append(f"- Annualized: {vol_data.get('annualized', 'N/A')}%")
            lines.append(f"- Classification: {vol_data.get('classification', 'N/A')}")
            lines.append("")

        # VaR from var key
        var_data = raw.get("var", {})
        if var_data:
            var_pct = var_data.get("var_percent", 0)
            lines.append("**Value at Risk (VaR):**")
            lines.append(f"- VaR: {var_pct:.2f}% ({var_data.get('confidence_level', 95)}% confidence)")
            lines.append(f"- Method: {var_data.get('method', 'historical')}")
            lines.append("")

        # Max Drawdown
        dd_data = raw.get("max_drawdown", {})
        if dd_data:
            lines.append("**Max Drawdown:**")
            lines.append(f"- Max DD: {dd_data.get('max_drawdown', 0):.1f}%")
            lines.append(f"- Current DD: {dd_data.get('current_drawdown', 0):.1f}%")
            lines.append("")

        return lines

    def _format_sentiment_raw_data(self, raw: Dict[str, Any]) -> List[str]:
        """
        Format sentiment raw data using CORRECT structure from get_sentiment.

        Phase 3 Improvement: Includes sample size weighting and confidence levels.
        """
        lines = ["### RAW SENTIMENT METRICS:"]
        lines.append("")

        # Sentiment data can be nested
        sentiment_data = raw.get("sentiment", {}) or raw
        news_data = raw.get("news", {})

        # Overall sentiment score
        score = sentiment_data.get("sentiment_score") or sentiment_data.get("score")
        label = sentiment_data.get("sentiment_label") or sentiment_data.get("label")

        # Get sample size for confidence calculation
        data_points = sentiment_data.get("data_points_analyzed") or sentiment_data.get("data_count", 0)

        if score is not None:
            # Classification with score range
            if score > 0.5:
                classification = "STRONGLY BULLISH (>0.5)"
            elif score > 0.2:
                classification = "BULLISH (0.2-0.5)"
            elif score > -0.2:
                classification = "NEUTRAL (-0.2 to 0.2)"
            elif score > -0.5:
                classification = "BEARISH (-0.5 to -0.2)"
            else:
                classification = "STRONGLY BEARISH (<-0.5)"

            lines.append(f"**Sentiment Score: {score:.3f}**")
            lines.append(f"- Classification: {classification}")
            if label:
                lines.append(f"- Label: {label}")
            lines.append("")

            # Phase 3: Sample size weighting and confidence
            confidence = self._calculate_sentiment_confidence(data_points, score)
            lines.append(f"**Confidence Level: {confidence['level']}** ({confidence['score']:.0f}/100)")
            lines.append(f"- Sample Size: {data_points} data points")
            lines.append(f"- {confidence['explanation']}")
            lines.append("")

            # Warning for low sample size
            if data_points < 10:
                lines.append("⚠️ **LOW SAMPLE SIZE WARNING:**")
                lines.append("   - Less than 10 data points analyzed")
                lines.append("   - Sentiment may not be statistically significant")
                lines.append("   - Consider this as directional indicator only")
                lines.append("")
            elif data_points < 30:
                lines.append("⚠️ **MODERATE SAMPLE SIZE:**")
                lines.append("   - 10-30 data points may show sentiment direction")
                lines.append("   - Confidence increases with more data points")
                lines.append("")

        # Social sentiment breakdown
        social = sentiment_data.get("social_breakdown", {})
        if social:
            lines.append("**Social Media Breakdown:**")
            stocktwits = social.get("stocktwits")
            twitter = social.get("twitter")

            if stocktwits is not None:
                st_label = "Bullish" if stocktwits > 0.1 else "Bearish" if stocktwits < -0.1 else "Neutral"
                lines.append(f"- StockTwits: {stocktwits:.3f} ({st_label})")
            if twitter is not None:
                tw_label = "Bullish" if twitter > 0.1 else "Bearish" if twitter < -0.1 else "Neutral"
                lines.append(f"- Twitter/X: {twitter:.3f} ({tw_label})")

            # Source attribution
            lines.append("")
            lines.append("Source: Social sentiment APIs (last 7 days)")
            lines.append("")

        # News data
        if news_data:
            articles = news_data if isinstance(news_data, list) else news_data.get("articles", [])
            if articles:
                lines.append(f"**News Articles: {len(articles)} analyzed**")

                # Calculate news sentiment if available
                positive = sum(1 for a in articles if a.get("sentiment", "").lower() == "positive")
                negative = sum(1 for a in articles if a.get("sentiment", "").lower() == "negative")
                neutral = len(articles) - positive - negative

                if len(articles) > 0:
                    lines.append(f"- Positive: {positive} ({positive/len(articles)*100:.0f}%)")
                    lines.append(f"- Negative: {negative} ({negative/len(articles)*100:.0f}%)")
                    lines.append(f"- Neutral: {neutral} ({neutral/len(articles)*100:.0f}%)")
                lines.append("")

        return lines

    def _calculate_sentiment_confidence(self, sample_size: int, score: float) -> Dict[str, Any]:
        """
        Calculate confidence level for sentiment based on sample size.

        Phase 3: Proper statistical weighting for sentiment confidence.

        Args:
            sample_size: Number of data points analyzed
            score: Sentiment score (-1 to 1)

        Returns:
            Dict with confidence level, score, and explanation
        """
        # Base confidence from sample size (0-70 points)
        if sample_size >= 100:
            size_score = 70
        elif sample_size >= 50:
            size_score = 55
        elif sample_size >= 30:
            size_score = 40
        elif sample_size >= 10:
            size_score = 25
        else:
            size_score = 10

        # Bonus for strong/consistent signal (0-30 points)
        # Strong signals (far from 0) with sufficient sample size get bonus
        signal_strength = abs(score)
        if sample_size >= 20 and signal_strength > 0.3:
            signal_bonus = min(30, int(signal_strength * 50))
        else:
            signal_bonus = 0

        total_score = size_score + signal_bonus

        # Determine confidence level
        if total_score >= 80:
            level = "HIGH"
            explanation = f"Strong signal ({score:.2f}) with adequate sample ({sample_size})"
        elif total_score >= 50:
            level = "MODERATE"
            explanation = f"Reasonable sample size ({sample_size}), signal strength: {signal_strength:.2f}"
        elif total_score >= 30:
            level = "LOW"
            explanation = f"Limited data ({sample_size} points) - treat as directional only"
        else:
            level = "VERY LOW"
            explanation = f"Insufficient data ({sample_size} points) - sentiment unreliable"

        return {
            "level": level,
            "score": total_score,
            "explanation": explanation,
            "sample_size": sample_size,
            "signal_strength": signal_strength
        }

    def _format_fundamental_raw_data(self, raw: Dict[str, Any]) -> List[str]:
        """Format fundamental raw data with VALUATION ASSUMPTIONS."""
        lines = ["### FUNDAMENTAL INPUTS (Show assumptions for intrinsic values):"]
        lines.append("")

        report = raw.get("fundamental_report", {})

        # Valuation
        if report.get("valuation"):
            v = report["valuation"]
            pe_ttm = v.get('pe_ttm', 'N/A')
            ps_ttm = v.get('ps_ttm', 'N/A')
            pb_ttm = v.get('pb_ttm', 'N/A')
            ev_ebitda = v.get('ev_ebitda', 'N/A')

            lines.append("**Valuation Metrics:**")
            lines.append(f"- P/E (TTM): {pe_ttm}")
            lines.append(f"- P/E (Forward): {v.get('pe_fy', 'N/A')}")
            lines.append(f"- P/S (TTM): {ps_ttm}")
            lines.append(f"- P/B (TTM): {pb_ttm}")
            lines.append(f"- EV/EBITDA: {ev_ebitda}")
            lines.append("")

            # P/E distortion warning
            if pe_ttm and pe_ttm != 'N/A':
                try:
                    pe_val = float(pe_ttm)
                    if pe_val > 50:
                        lines.append("⚠️ P/E > 50 may indicate high growth expectations or earnings distortion")
                    elif pe_val < 0:
                        lines.append("⚠️ Negative P/E indicates losses - use P/S or EV/EBITDA instead")
                except (ValueError, TypeError):
                    pass
            lines.append("")

        # Profitability
        if report.get("profitability"):
            p = report["profitability"]
            roe = p.get('roe', 0)

            lines.append("**Profitability:**")
            lines.append(f"- Gross Margin: {p.get('gross_margin', 'N/A')}%")
            lines.append(f"- Operating Margin: {p.get('operating_margin', 'N/A')}%")
            lines.append(f"- Net Margin: {p.get('net_margin', 'N/A')}%")
            lines.append(f"- ROE: {roe}%")
            lines.append(f"- ROA: {p.get('roa', 'N/A')}%")

            # ROE distortion warning
            if roe and isinstance(roe, (int, float)) and roe > 50:
                lines.append("")
                lines.append("⚠️ ROE > 50% may be distorted by:")
                lines.append("   - Share buybacks reducing equity")
                lines.append("   - High debt leverage")
                lines.append("   - One-time gains")
            lines.append("")

        # Growth
        if report.get("growth"):
            g = report["growth"]
            lines.append("**Growth Metrics:**")
            lines.append(f"- Revenue Growth (YoY): {g.get('revenue_growth_yoy', 'N/A')}%")
            lines.append(f"- EPS Growth (YoY): {g.get('eps_growth_yoy', 'N/A')}%")
            lines.append(f"- Revenue Growth (5Y CAGR): {g.get('revenue_cagr_5y', 'N/A')}%")
            lines.append(f"- EPS Growth (5Y CAGR): {g.get('eps_cagr_5y', 'N/A')}%")
            lines.append("")

        # Intrinsic Value - WITH ASSUMPTIONS
        if report.get("intrinsic_value"):
            iv = report["intrinsic_value"]
            graham = iv.get("graham_value")
            dcf = iv.get("dcf_value")
            current = iv.get("current_price")

            lines.append("**Intrinsic Value (Fair Value) - MODEL-DEPENDENT:**")

            if graham:
                lines.append(f"- Graham Value: ${graham:.2f}")
                lines.append("  Assumptions: Graham formula = √(22.5 × EPS × BVPS)")
                lines.append(f"  Used EPS: {iv.get('eps_used', 'N/A')}, BVPS: {iv.get('bvps_used', 'N/A')}")

            if dcf:
                lines.append(f"- DCF Value: ${dcf:.2f}")
                lines.append(f"  Assumptions: WACC={iv.get('wacc', '10%')}, Terminal Growth={iv.get('terminal_growth', '3%')}")
                lines.append(f"  FCF Base: {iv.get('fcf_base', 'N/A')}")

            if current:
                lines.append(f"- Current Price: ${current:.2f}")

            if graham and dcf and current:
                avg_fair = (graham + dcf) / 2
                premium_discount = round((current - avg_fair) / avg_fair * 100, 1)
                if premium_discount > 0:
                    lines.append(f"  → Trading at {premium_discount}% PREMIUM to avg fair value")
                else:
                    lines.append(f"  → Trading at {abs(premium_discount)}% DISCOUNT to avg fair value")

            if iv.get("verdict"):
                lines.append(f"- Model Verdict: {iv['verdict']}")

            lines.append("")
            lines.append("⚠️ IMPORTANT: Intrinsic values are model-dependent.")
            lines.append("   Different assumptions (WACC ±1%, growth ±1%) can change values significantly.")
            lines.append("   Use as REFERENCE, not absolute truth.")
            lines.append("")

        # Dividend
        if report.get("dividend"):
            d = report["dividend"]
            lines.append("**Dividend:**")
            lines.append(f"- Dividend Yield: {d.get('yield', 'N/A')}%")
            lines.append(f"- Payout Ratio: {d.get('payout_ratio', 'N/A')}%")
            lines.append("")

        # PEER COMPARISON GUIDANCE
        lines.append("**PEER COMPARISON GUIDANCE:**")
        lines.append("For Semiconductors: Compare with AMD, AVGO, TSM, INTC, MRVL, QCOM")
        lines.append("Do NOT use mega-cap tech (AAPL, MSFT, GOOGL) as semiconductor peers.")
        lines.append("Note: P/E may be distorted for companies with low/volatile EPS.")
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
        """Generate markdown report header - DATA FOCUSED, NO SCORING."""
        header = f"""# Comprehensive Investment Report: {symbol}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Steps:** {', '.join(available_steps)}
**Version:** Synthesis V2 (Data-Driven)

---

*This report presents raw data and analysis. No composite scoring is used to avoid contradictions.*
*Each section shows INPUT → OUTPUT format for auditability.*

---
"""
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
    # STEP DATA LOGGING (FOR DEBUGGING/AUDIT)
    # =========================================================================

    def _log_step_data_summary(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        earnings_data: Optional[Dict[str, Any]],
        peer_data: Optional[Dict[str, Any]]
    ) -> None:
        """Log summary of step data going into synthesis LLM for audit purposes."""
        self.logger.info(f"[SynthesisV2] ========== STEP DATA SUMMARY: {symbol} ==========")

        # Technical - use correct path: raw['indicators']['rsi']['value']
        tech_data = step_data.get("technical", {})
        tech_raw = tech_data.get("raw_data", {})
        has_tech_content = bool(tech_data.get("content", ""))
        if tech_raw:
            indicators = tech_raw.get("indicators", {})
            rsi_data = indicators.get("rsi", {})
            macd_data = indicators.get("macd", {})
            adx_data = indicators.get("adx", {})
            self.logger.info(f"[SynthesisV2] TECHNICAL: RSI={rsi_data.get('value', 'N/A')}, "
                           f"MACD={macd_data.get('macd_line', 'N/A')}, ADX={adx_data.get('value', 'N/A')}, "
                           f"has_content={has_tech_content}")
        else:
            self.logger.info(f"[SynthesisV2] TECHNICAL: raw_data=None, has_content={has_tech_content}")

        # Position
        pos_data = step_data.get("position", {})
        pos_raw = pos_data.get("raw_data", {})
        sector_ctx = pos_data.get("sector_context", {})
        has_pos_content = bool(pos_data.get("content", ""))
        rs_metrics = pos_raw.get("rs_metrics", pos_raw)
        self.logger.info(f"[SynthesisV2] POSITION: RS_21d={rs_metrics.get('excess_return_21d', 'N/A')}, "
                       f"RS_63d={rs_metrics.get('excess_return_63d', 'N/A')}, "
                       f"Sector={sector_ctx.get('stock_sector', 'N/A')}, "
                       f"Industry={sector_ctx.get('stock_industry', 'N/A')}, "
                       f"has_content={has_pos_content}")

        # Risk - use correct path: raw['stop_loss_recommendations']['atr_based']
        risk_data = step_data.get("risk", {})
        risk_raw = risk_data.get("raw_data", {})
        has_risk_content = bool(risk_data.get("content", ""))
        if risk_raw:
            stop_recs = risk_raw.get("stop_loss_recommendations", {})
            atr_data = stop_recs.get("atr_based", {})
            var_data = risk_raw.get("var", {})
            self.logger.info(f"[SynthesisV2] RISK: ATR=${atr_data.get('atr_value', risk_raw.get('atr', 'N/A'))}, "
                           f"VaR={var_data.get('var_percent', 'N/A')}%, "
                           f"Price=${risk_raw.get('current_price', 'N/A')}, "
                           f"has_content={has_risk_content}")
        else:
            self.logger.info(f"[SynthesisV2] RISK: raw_data=None, has_content={has_risk_content}")

        # Sentiment - use correct path: raw['sentiment']
        sent_data = step_data.get("sentiment", {})
        sent_raw = sent_data.get("raw_data", {})
        has_sent_content = bool(sent_data.get("content", ""))
        if sent_raw:
            sentiment = sent_raw.get("sentiment", sent_raw)
            self.logger.info(f"[SynthesisV2] SENTIMENT: Score={sentiment.get('sentiment_score', sentiment.get('score', 'N/A'))}, "
                           f"Label={sentiment.get('sentiment_label', sentiment.get('label', 'N/A'))}, "
                           f"has_content={has_sent_content}")
        else:
            self.logger.info(f"[SynthesisV2] SENTIMENT: raw_data=None, has_content={has_sent_content}")

        # Fundamental - this path is correct
        fund_data = step_data.get("fundamental", {})
        fund_raw = fund_data.get("raw_data", {})
        has_fund_content = bool(fund_data.get("content", ""))
        if fund_raw:
            report = fund_raw.get("fundamental_report", {})
            valuation = report.get("valuation", {})
            intrinsic = report.get("intrinsic_value", {})
            self.logger.info(f"[SynthesisV2] FUNDAMENTAL: P/E={valuation.get('pe_ttm', 'N/A')}, "
                           f"P/S={valuation.get('ps_ttm', 'N/A')}, "
                           f"Graham=${intrinsic.get('graham_value', 'N/A')}, "
                           f"DCF=${intrinsic.get('dcf_value', 'N/A')}, "
                           f"has_content={has_fund_content}")
        else:
            self.logger.info(f"[SynthesisV2] FUNDAMENTAL: raw_data=None, has_content={has_fund_content}")

        # Earnings
        if earnings_data:
            self.logger.info(f"[SynthesisV2] EARNINGS: Date={earnings_data.get('next_earnings_date', 'N/A')}, "
                           f"Beat_rate={earnings_data.get('beat_rate', 'N/A')}")

        # Peers
        if peer_data:
            peers = peer_data.get("peers", [])
            peer_symbols = [p.get("symbol", "") for p in peers[:5]]
            self.logger.info(f"[SynthesisV2] PEERS: {', '.join(peer_symbols)}")

        self.logger.info(f"[SynthesisV2] ========== END STEP DATA SUMMARY ==========")

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
