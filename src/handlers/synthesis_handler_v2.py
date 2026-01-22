"""
Synthesis Handler V2 - Single LLM Call Architecture

Key Improvements over V1:
1. SINGLE LLM CALL - All data in one context for 100% consistency
2. BINDING SCORING - LLM must follow the calculated score
3. CONSISTENT Entry/Stop/Target logic based on trading system
4. SPECIFIC CATALYST DATES - Earnings calendar with exact dates
5. PEER COMPARISON - Comparison table with 3-5 peers
6. QUALITY WEB ENRICHMENT - Deduplicated, date-filtered, inline citations
7. CLEAR SECTOR RANKING - Methodology explanation

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
from typing import Dict, Any, Optional, List, AsyncGenerator, Tuple
from collections import OrderedDict

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
# CONFIGURATION
# =============================================================================

# Peer mapping by sector/industry
PEER_MAPPING = {
    # Semiconductors
    "NVDA": ["AMD", "AVGO", "TSM", "INTC", "QCOM"],
    "AMD": ["NVDA", "INTC", "QCOM", "AVGO", "TSM"],
    "INTC": ["AMD", "NVDA", "QCOM", "TXN", "ADI"],
    "TSM": ["NVDA", "AMD", "INTC", "ASML", "AVGO"],
    "AVGO": ["NVDA", "QCOM", "TXN", "ADI", "MRVL"],

    # Big Tech
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL"],
    "GOOGL": ["META", "MSFT", "AMZN", "AAPL"],
    "META": ["GOOGL", "SNAP", "PINS", "MSFT"],
    "AMZN": ["MSFT", "GOOGL", "WMT", "BABA"],

    # EV / Auto
    "TSLA": ["RIVN", "NIO", "F", "GM", "LCID"],
    "RIVN": ["TSLA", "LCID", "NIO", "F", "GM"],

    # Financials
    "JPM": ["BAC", "WFC", "C", "GS", "MS"],
    "BAC": ["JPM", "WFC", "C", "USB", "PNC"],

    # Healthcare
    "JNJ": ["PFE", "MRK", "ABBV", "LLY", "UNH"],
    "PFE": ["JNJ", "MRK", "ABBV", "BMY", "LLY"],

    # Energy
    "XOM": ["CVX", "COP", "SLB", "EOG", "OXY"],
    "CVX": ["XOM", "COP", "SLB", "EOG", "PXD"],
}

# Default peers by sector (when symbol not in mapping)
DEFAULT_SECTOR_PEERS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META"],
    "Healthcare": ["JNJ", "PFE", "UNH", "MRK"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS"],
    "Energy": ["XOM", "CVX", "COP", "SLB"],
    "Industrials": ["CAT", "BA", "UNP", "HON"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP"],
    "Basic Materials": ["LIN", "APD", "ECL", "NEM"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG"],
    "Utilities": ["NEE", "DUK", "SO", "D"],
}

# Trading system thresholds
TRADING_SYSTEM_THRESHOLDS = {
    "strong_buy": {"min_score": 75, "system": "aggressive_breakout", "risk_pct": 7},
    "buy": {"min_score": 65, "system": "breakout", "risk_pct": 5},
    "hold": {"min_score": 45, "system": "wait_and_watch", "risk_pct": 0},
    "sell": {"min_score": 30, "system": "mean_reversion_exit", "risk_pct": 3},
    "strong_sell": {"min_score": 0, "system": "avoid", "risk_pct": 0},
}


# =============================================================================
# CONSOLIDATED SYSTEM PROMPT
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
- Explain: "Mua khi giá xác nhận xu hướng tăng, không bắt đáy"

**If Score 45-64 (HOLD) - Use WAIT & WATCH:**
- NO entry recommendation (watchlist only)
- Define CONDITIONS for future entry
- Explain: "Chờ tín hiệu xác nhận trước khi mở vị thế"

**If Score < 45 (SELL) - Use EXIT/AVOID:**
- For holders: Exit strategy
- For new investors: AVOID
- Explain reasoning

### 3. STOP-LOSS CALCULATION (CRITICAL)
Stop-loss percentage MUST be calculated FROM ENTRY PRICE, not current price:
- If Entry = $185 and Stop = $169, then risk = (185-169)/185 = 8.6%
- NEVER say "5% from current price" if entry is different from current price

### 4. CATALYST DATES (CRITICAL)
- Include SPECIFIC dates for earnings from provided data
- Format: "Earnings: Feb 25, 2026 (AMC)" - use EXACT date from data
- If no date available, state "Chưa công bố" not "dự kiến sắp tới"
- Source: FMP Earnings Calendar API

### 5. PEER COMPARISON
- Include a mini comparison table when peer data is provided
- Show P/E, P/S, Growth metrics vs peers
- ADD WARNING: "Lưu ý: P/E có thể bị méo do EPS thấp/biến động (đặc biệt AMD, INTC)"
- Consider adding Forward P/E or EV/Sales note if P/E looks distorted

### 6. TECHNICAL SCORE CONSISTENCY (NEW - CRITICAL)
Technical score MUST match the narrative. Guidelines:
- ADX < 15 (weak trend) → Technical score should be 50-65 max, not 75+
- ADX 15-25 (moderate trend) → Technical score 60-75
- ADX > 25 (strong trend) → Technical score can be 70-85
- If MACD bearish + ADX < 15 → Technical should be < 60
- Explain WHY the score despite weak indicators if giving high score

### 7. TRADING PLAN FOR HOLDERS (NEW - CRITICAL)
Always include rules for people ALREADY holding the stock:
- "Nếu giá thủng vùng $X → giảm 50% vị thế"
- "Nếu breakout fail (giá quay về dưới $Y) → thoát hoàn toàn"
- "Trailing stop: di chuyển stop lên theo ATR khi giá tăng"

### 8. SECTOR RANKING CLARITY
When mentioning sector rank, explain the methodology:
- "Ngành Công nghệ #10/11 theo FMP Sector Performance (biến động 1 ngày)"
- "Lưu ý: Đây là ranking 1 NGÀY, khác với RS đa khung thời gian"

### 9. WEB CITATIONS (MANDATORY - ChatGPT Style)

**CRITICAL**: You MUST include clickable source links in TWO ways:

#### A. INLINE CITATIONS (Throughout your response)
When referencing information from web search, ALWAYS include the source as a clickable markdown link INLINE.

**Format**: Statement + [Source Name](URL)

**Examples**:
- "NVIDIA đạt doanh thu kỷ lục $35.1 tỷ trong Q3 2024. [NVIDIA Investor Relations](https://investor.nvidia.com/...)"
- "Theo [Reuters](https://reuters.com/...), chip H200 đang gặp vấn đề về chuỗi cung ứng."
- "TSMC dành 70% công suất CoWoS-L cho NVIDIA. [TechPowerUp](https://www.techpowerup.com/...)"

**DO NOT** write statements without source links. Every claim from web search needs a citation.

#### B. SOURCES SECTION (At the end of News section)
After the news analysis, include a "## 📚 Sources" section listing all sources used.

**Example Format**:
```
## 📚 Sources

1. [NVIDIA Announces Financial Results for Q3 2026](https://investor.nvidia.com/news/...)
2. [TSMC Reserves 70% of CoWoS-L Capacity for NVIDIA](https://www.techpowerup.com/...)
3. [AMD launches MI325X to rival Nvidia's Blackwell](https://www.cnbc.com/...)
```

**WARNING**: Responses without inline source citations AND source section will be considered INCOMPLETE.

## OUTPUT FORMAT (RESTRUCTURED)

Generate report in this EXACT order:

### PART 1: DATA ANALYSIS (5 Steps - Present findings first)
1. **Phân tích Kỹ thuật** - Trend, momentum, key levels, signals
2. **Vị thế Thị trường** - RS vs SPY, sector context with methodology
3. **Phân tích Rủi ro** - Volatility, VaR, risk metrics
4. **Tâm lý Thị trường** - Sentiment score, news themes
5. **Phân tích Cơ bản** - Valuation, growth, peer comparison

### PART 2: NEWS & CATALYSTS (With Citations)
6. **Tin tức & Chất xúc tác** - Latest news WITH inline citations + Sources section

### PART 3: CONCLUSION & ACTION (Highlights + Strategy)
7. **Tóm tắt Điều hành** - Key highlights from each step
8. **Khuyến nghị Cuối cùng** - Action strategy for:
   - Nhà đầu tư MỚI (chưa có vị thế)
   - Nhà đầu tư đang NẮM GIỮ (có vị thế rồi)
   - Điều kiện nâng/hạ khuyến nghị

This structure ensures:
- Data presented first → Reader understands context
- Conclusion AFTER all data → Consistent with evidence
- Action for BOTH new and existing investors
"""


# =============================================================================
# SYNTHESIS HANDLER V2
# =============================================================================

class SynthesisHandlerV2(LoggerMixin):
    """
    Consolidated single-LLM-call synthesis handler.

    Key improvements:
    - Single LLM call for 100% consistency
    - Binding scoring system
    - Consistent Entry/Stop/Target logic
    - Peer comparison
    - Specific catalyst dates
    - Quality web enrichment
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

        Yields:
            Dict with type: "progress", "content", "data", "error", "done"
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

            # Get cached step data
            cache_status = await get_all_scanner_results(symbol)
            step_data = cache_status.get("data", {})
            missing = cache_status.get("missing", [])

            # Run missing steps if needed
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

            # Check minimum data
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
                "message": "Fetching earnings calendar, peer data, and news..."
            }

            # Parallel fetch: earnings, peers, web search
            earnings_task = self._fetch_earnings_calendar(symbol)
            peers_task = self._fetch_peer_comparison(symbol, step_data)

            web_data = None
            if include_web_search:
                web_task = self._fetch_web_enrichment(symbol, step_data, scoring_result)
            else:
                web_task = asyncio.sleep(0)  # Dummy task

            results = await asyncio.gather(
                earnings_task,
                peers_task,
                web_task,
                return_exceptions=True
            )

            earnings_data = results[0] if not isinstance(results[0], Exception) else None
            peer_data = results[1] if not isinstance(results[1], Exception) else None
            web_data = results[2] if include_web_search and not isinstance(results[2], Exception) else None

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
            # PHASE 5: Single consolidated LLM call
            # =================================================================
            yield {
                "type": "progress",
                "step": "synthesis",
                "message": "Generating comprehensive report (single LLM call)..."
            }

            # Build consolidated prompt with ALL data
            consolidated_prompt = self._build_consolidated_prompt(
                symbol=symbol,
                step_data=step_data,
                scoring=scoring_result,
                earnings_data=earnings_data,
                peer_data=peer_data,
                web_data=web_data,
                target_language=target_language
            )

            # Calculate trading plan parameters
            trading_plan = self._calculate_trading_plan(
                symbol=symbol,
                step_data=step_data,
                scoring=scoring_result
            )

            # Add trading plan to prompt
            consolidated_prompt += f"\n\n{self._format_trading_plan_section(trading_plan)}"

            messages = [
                {"role": "system", "content": CONSOLIDATED_SYSTEM_PROMPT},
                {"role": "user", "content": consolidated_prompt}
            ]

            # Stream the consolidated response
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
    # DATA FETCHING METHODS
    # =========================================================================

    async def _fetch_earnings_calendar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch earnings calendar with next earnings date.

        Uses TWO endpoints:
        1. /api/v3/earning_calendar - For UPCOMING earnings (next date)
        2. /stable/earnings - For HISTORICAL earnings (beat rate)

        Returns:
            {
                "next_earnings_date": "2026-02-25",
                "earnings_time": "AMC",
                "fiscal_quarter": "Q4 FY26",
                "historical": [...],
                "beat_rate": 0.75,
                "source": "FMP Earnings Calendar API"
            }
        """
        if not self.fmp_api_key:
            self.logger.warning("[Earnings] No FMP API key available")
            return None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                today = datetime.now().date()
                from_date = today.strftime("%Y-%m-%d")
                to_date = (today + timedelta(days=120)).strftime("%Y-%m-%d")  # Look 4 months ahead

                # =============================================================
                # STEP 1: Fetch UPCOMING earnings from earning_calendar
                # =============================================================
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
                        # Find this symbol's next earnings
                        for item in calendar_data:
                            if item.get("symbol", "").upper() == symbol.upper():
                                next_earnings = {
                                    "date": item.get("date"),
                                    "time": item.get("time", "TBD"),  # BMO, AMC, or TBD
                                    "eps_estimated": item.get("epsEstimated"),
                                    "revenue_estimated": item.get("revenueEstimated")
                                }
                                self.logger.info(f"[Earnings] Found upcoming: {symbol} on {next_earnings['date']} ({next_earnings['time']})")
                                break

                # =============================================================
                # STEP 2: Fetch HISTORICAL earnings for beat rate
                # =============================================================
                historical_url = "https://financialmodelingprep.com/stable/earnings"
                historical_params = {"symbol": symbol, "apikey": self.fmp_api_key}

                historical = []
                beat_rate = None

                historical_response = await client.get(historical_url, params=historical_params)

                if historical_response.status_code == 200:
                    historical_data = historical_response.json()
                    if isinstance(historical_data, list) and historical_data:
                        # Sort by date descending and get recent history
                        sorted_history = sorted(
                            historical_data,
                            key=lambda x: x.get("date", ""),
                            reverse=True
                        )
                        historical = sorted_history[:8]

                        # Calculate beat rate
                        beats = sum(1 for h in historical
                                   if h.get("epsActual") and h.get("epsEstimated")
                                   and h["epsActual"] > h["epsEstimated"])
                        total = len([h for h in historical
                                    if h.get("epsActual") and h.get("epsEstimated")])
                        beat_rate = round(beats / total, 2) if total > 0 else None

                # =============================================================
                # STEP 3: Build result
                # =============================================================
                result = {
                    "next_earnings_date": next_earnings.get("date") if next_earnings else None,
                    "earnings_time": next_earnings.get("time", "TBD") if next_earnings else "TBD",
                    "eps_estimated": next_earnings.get("eps_estimated") if next_earnings else None,
                    "revenue_estimated": next_earnings.get("revenue_estimated") if next_earnings else None,
                    "fiscal_quarter": self._determine_fiscal_quarter(next_earnings.get("date") if next_earnings else None),
                    "historical": historical[:4],
                    "beat_rate": beat_rate,
                    "source": "FMP Earnings Calendar API (earning_calendar + stable/earnings)"
                }

                if result["next_earnings_date"]:
                    self.logger.info(
                        f"[Earnings] {symbol}: Next={result['next_earnings_date']} ({result['earnings_time']}), "
                        f"Beat Rate={result['beat_rate']}"
                    )
                else:
                    self.logger.warning(f"[Earnings] {symbol}: No upcoming earnings found in next 120 days")

                return result

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
                return f"Q1 FY{year}"
            elif month in [7, 8, 9]:
                return f"Q2 FY{year}"
            else:
                return f"Q3 FY{year}"
        except:
            return "N/A"

    async def _fetch_peer_comparison(
        self,
        symbol: str,
        step_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch peer comparison data.

        Returns:
            {
                "symbol": "NVDA",
                "sector": "Technology",
                "peers": [
                    {"symbol": "AMD", "pe_ttm": 120.5, "ps": 8.2, "revenue_growth": 45%},
                    ...
                ],
                "target_metrics": {"pe_ttm": 43.67, ...}
            }
        """
        if not self.fmp_api_key:
            return None

        try:
            # Determine peers
            peers = PEER_MAPPING.get(symbol, [])

            if not peers:
                # Try to get from sector in position data
                position_content = step_data.get("position", {}).get("content", "")
                for sector, sector_peers in DEFAULT_SECTOR_PEERS.items():
                    if sector.lower() in position_content.lower():
                        peers = [p for p in sector_peers if p != symbol][:4]
                        break

            if not peers:
                self.logger.info(f"[Peers] No peers found for {symbol}")
                return None

            # Fetch metrics for target and peers in parallel
            all_symbols = [symbol] + peers[:4]

            async with httpx.AsyncClient(timeout=20.0) as client:
                tasks = []
                for sym in all_symbols:
                    tasks.append(self._fetch_single_peer_metrics(client, sym))

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

            if not target_metrics or not peer_metrics:
                return None

            return {
                "symbol": symbol,
                "sector": target_metrics.get("sector", "N/A"),
                "peers": peer_metrics,
                "target_metrics": target_metrics
            }

        except Exception as e:
            self.logger.error(f"[Peers] Error fetching for {symbol}: {e}")
            return None

    async def _fetch_single_peer_metrics(
        self,
        client: httpx.AsyncClient,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Fetch metrics for a single symbol."""
        try:
            # Fetch key-metrics-ttm
            url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{symbol}"
            params = {"apikey": self.fmp_api_key}

            response = await client.get(url, params=params)

            if response.status_code != 200:
                return None

            data = response.json()

            if not isinstance(data, list) or not data:
                return None

            metrics = data[0]

            # Also get profile for sector
            profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
            profile_response = await client.get(profile_url, params=params)
            profile_data = profile_response.json() if profile_response.status_code == 200 else []
            profile = profile_data[0] if profile_data else {}

            return {
                "symbol": symbol,
                "sector": profile.get("sector", "N/A"),
                "pe_ttm": round(metrics.get("peRatioTTM", 0), 2) if metrics.get("peRatioTTM") else None,
                "ps_ttm": round(metrics.get("priceToSalesRatioTTM", 0), 2) if metrics.get("priceToSalesRatioTTM") else None,
                "pb_ttm": round(metrics.get("pbRatioTTM", 0), 2) if metrics.get("pbRatioTTM") else None,
                "roe_ttm": round(metrics.get("roeTTM", 0) * 100, 1) if metrics.get("roeTTM") else None,
                "revenue_growth": round(metrics.get("revenueGrowthTTM", 0) * 100, 1) if metrics.get("revenueGrowthTTM") else None,
                "market_cap": profile.get("mktCap", 0)
            }

        except Exception as e:
            self.logger.debug(f"[Peers] Error fetching metrics for {symbol}: {e}")
            return None

    async def _fetch_web_enrichment(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch web search results with smart queries and deduplication.

        Returns:
            {
                "news": [...],
                "citations": [...],  # Deduplicated
                "search_date": "2026-01-21"
            }
        """
        try:
            web_search = WebSearchTool()

            # Build date-aware smart queries
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
                    "query": f"{symbol} earnings date guidance Q4 2026",
                    "max_results": 3
                },
                {
                    "category": "analyst",
                    "query": f"{symbol} analyst price target upgrade downgrade {month_year}",
                    "max_results": 3
                }
            ]

            # Execute searches
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
                        # Collect citations
                        for citation in result.data.get("citations", []):
                            all_citations.append(citation)

                        # Collect answers
                        if result.data.get("answer"):
                            all_answers.append({
                                "category": query_info["category"],
                                "answer": result.data.get("answer", "")
                            })
                except Exception as e:
                    self.logger.warning(f"[Web] Search failed for {query_info['category']}: {e}")

            # Deduplicate citations by URL
            seen_urls = set()
            unique_citations = []
            for citation in all_citations:
                url = citation.get("url", "")
                # Extract base URL for dedup
                base_url = url.split("?")[0] if url else ""
                if base_url and base_url not in seen_urls:
                    seen_urls.add(base_url)
                    unique_citations.append(citation)

            # Filter for freshness (last 30 days preferred)
            fresh_citations = []
            for citation in unique_citations[:8]:
                # Keep citation, mark as fresh/stale if date available
                fresh_citations.append(citation)

            return {
                "news": all_answers,
                "citations": fresh_citations[:6],  # Max 6 unique sources
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
        """
        Calculate consistent trading plan based on score and technical data.

        Includes HOLDER-specific rules (exit/reduce triggers).

        Returns:
            {
                "trading_system": "breakout" | "wait_and_watch" | "exit",
                "current_price": 178.07,
                "entry_zone": {"low": 185, "high": 190},
                "stop_loss": {"price": 169, "pct_from_entry": 8.6},
                "targets": [{"price": 205, "pct_gain": 10}, ...],
                "risk_reward_ratio": 2.5,
                "position_sizing_note": "...",
                "holder_rules": {...}  # NEW: Rules for existing holders
            }
        """
        composite_score = scoring.get("composite_score", 50)
        recommendation = scoring.get("recommendation", {}).get("action", "HOLD")

        # Extract technical data
        risk_data = step_data.get("risk", {}).get("raw_data", {})
        tech_data = step_data.get("technical", {}).get("raw_data", {})

        # Get current price and ATR
        current_price = risk_data.get("current_price", 0)
        atr = risk_data.get("atr", {}).get("value", 0)
        atr_pct = risk_data.get("atr", {}).get("percent", 2.5)

        # Calculate key support/resistance levels for holder rules
        support_1 = round(current_price * 0.95, 2)  # ~5% below current
        support_2 = round(current_price * 0.90, 2)  # ~10% below current
        resistance_1 = round(current_price * 1.05, 2)  # ~5% above current

        # Default plan
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

        # Determine trading system based on score
        if composite_score >= 65:
            # BREAKOUT SYSTEM
            plan["trading_system"] = "breakout"

            # Entry above current price (breakout confirmation)
            entry_low = round(current_price * 1.03, 2)  # 3% above current
            entry_high = round(current_price * 1.07, 2)  # 7% above current
            plan["entry_zone"] = {"low": entry_low, "high": entry_high}

            # Stop loss using ATR from entry (2x ATR below entry)
            avg_entry = (entry_low + entry_high) / 2
            stop_price = round(avg_entry - (2 * atr if atr > 0 else avg_entry * 0.08), 2)
            stop_pct = round(((avg_entry - stop_price) / avg_entry) * 100, 1)
            plan["stop_loss"] = {"price": stop_price, "pct_from_entry": stop_pct}

            # Targets: 1:2 and 1:3 R:R
            risk_per_share = avg_entry - stop_price
            target_1 = round(avg_entry + (risk_per_share * 2), 2)
            target_2 = round(avg_entry + (risk_per_share * 3), 2)

            plan["targets"] = [
                {"price": target_1, "pct_gain": round(((target_1 - avg_entry) / avg_entry) * 100, 1)},
                {"price": target_2, "pct_gain": round(((target_2 - avg_entry) / avg_entry) * 100, 1)}
            ]

            plan["risk_reward_ratio"] = 2.0
            plan["position_sizing_note"] = f"Risk {stop_pct}% per position. For 2% account risk, position size = (Account * 0.02) / (Entry * {stop_pct/100:.3f})"

            # Holder rules for BUY scenario
            plan["holder_rules"] = {
                "reduce_trigger": None,  # No reduce needed for BUY
                "exit_trigger": stop_price,
                "trailing_stop": f"Di chuyển stop lên theo ATR khi giá tăng. ATR hiện tại = ${atr:.2f}" if atr else None,
                "add_trigger": round(target_1 * 0.98, 2),  # Add on pullback to target 1
                "description": "Xu hướng tích cực. Giữ vị thế, có thể thêm khi giá pullback về vùng hỗ trợ."
            }

        elif composite_score >= 45:
            # WAIT & WATCH - No entry for new investors
            plan["trading_system"] = "wait_and_watch"
            plan["entry_zone"] = None
            plan["stop_loss"] = None
            plan["targets"] = []
            plan["position_sizing_note"] = "Watchlist only. Wait for score > 65 or clear technical breakout."

            # Holder rules for HOLD scenario (CRITICAL)
            plan["holder_rules"] = {
                "reduce_trigger": support_1,
                "exit_trigger": support_2,
                "trailing_stop": f"Nếu giá tăng vượt ${resistance_1:.2f}, di chuyển stop lên ${current_price:.2f}",
                "add_trigger": None,  # Don't add in HOLD mode
                "description": (
                    f"Tín hiệu trái chiều. Holder nên:\n"
                    f"  • Giảm 50% vị thế nếu giá thủng ${support_1:.2f}\n"
                    f"  • Thoát hoàn toàn nếu giá thủng ${support_2:.2f}\n"
                    f"  • KHÔNG thêm vị thế mới cho đến khi score > 65"
                )
            }

        else:
            # EXIT / AVOID
            plan["trading_system"] = "exit_or_avoid"

            # For holders: exit strategy
            if current_price > 0:
                exit_price = round(current_price * 0.95, 2)  # Exit on 5% bounce
                plan["exit_zone"] = {"price": exit_price}
                plan["position_sizing_note"] = "Reduce or exit position. Do not add new capital."

            # Holder rules for SELL scenario
            plan["holder_rules"] = {
                "reduce_trigger": current_price,  # Reduce NOW
                "exit_trigger": support_1,
                "trailing_stop": None,
                "add_trigger": None,
                "description": (
                    f"Tín hiệu tiêu cực. Holder nên:\n"
                    f"  • Giảm ít nhất 50% vị thế NGAY\n"
                    f"  • Thoát hoàn toàn nếu giá thủng ${support_1:.2f}\n"
                    f"  • Nếu muốn giữ: đặt stop loss chặt tại ${support_1:.2f}"
                )
            }

        return plan

    def _format_trading_plan_section(self, plan: Dict[str, Any]) -> str:
        """Format trading plan as prompt section with holder rules."""
        lines = [
            "## PRE-CALCULATED TRADING PLAN (USE THIS EXACTLY)",
            "",
            f"Trading System: {plan['trading_system'].upper().replace('_', ' ')}",
            f"Current Price: ${plan['current_price']:.2f}" if plan['current_price'] else "Current Price: N/A",
        ]

        # New investor entry rules
        lines.append("")
        lines.append("### FOR NEW INVESTORS (Chưa có vị thế):")
        if plan.get("entry_zone"):
            lines.append(f"Entry Zone: ${plan['entry_zone']['low']:.2f} - ${plan['entry_zone']['high']:.2f} (BREAKOUT CONFIRMATION REQUIRED)")
        else:
            lines.append("Entry Zone: KHÔNG VÀO LỆNH - Watchlist only until conditions met")

        if plan.get("stop_loss"):
            lines.append(f"Stop Loss: ${plan['stop_loss']['price']:.2f} ({plan['stop_loss']['pct_from_entry']}% below ENTRY, not current price)")

        if plan.get("targets"):
            for i, target in enumerate(plan["targets"], 1):
                lines.append(f"Target {i}: ${target['price']:.2f} (+{target['pct_gain']}%)")

        if plan.get("risk_reward_ratio"):
            lines.append(f"Risk:Reward Ratio: 1:{plan['risk_reward_ratio']}")

        if plan.get("position_sizing_note"):
            lines.append(f"Position Sizing: {plan['position_sizing_note']}")

        # Holder rules (CRITICAL NEW SECTION)
        holder_rules = plan.get("holder_rules", {})
        if holder_rules:
            lines.append("")
            lines.append("### FOR EXISTING HOLDERS (Đã có vị thế) - CRITICAL:")

            if holder_rules.get("description"):
                lines.append(holder_rules["description"])

            if holder_rules.get("reduce_trigger"):
                lines.append(f"• Giảm 50% vị thế: Nếu giá thủng ${holder_rules['reduce_trigger']:.2f}")

            if holder_rules.get("exit_trigger"):
                lines.append(f"• Thoát hoàn toàn: Nếu giá thủng ${holder_rules['exit_trigger']:.2f}")

            if holder_rules.get("trailing_stop"):
                lines.append(f"• Trailing Stop: {holder_rules['trailing_stop']}")

            if holder_rules.get("add_trigger"):
                lines.append(f"• Thêm vị thế: Có thể add nếu giá pullback về ${holder_rules['add_trigger']:.2f}")

        return "\n".join(lines)

    # =========================================================================
    # PROMPT BUILDING
    # =========================================================================

    def _build_consolidated_prompt(
        self,
        symbol: str,
        step_data: Dict[str, Any],
        scoring: Dict[str, Any],
        earnings_data: Optional[Dict[str, Any]],
        peer_data: Optional[Dict[str, Any]],
        web_data: Optional[Dict[str, Any]],
        target_language: str
    ) -> str:
        """Build comprehensive prompt with all data."""

        parts = [
            f"# COMPREHENSIVE INVESTMENT ANALYSIS REQUEST: {symbol}",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        # =====================================================================
        # SECTION 1: BINDING SCORING (CRITICAL)
        # =====================================================================
        rec = scoring.get("recommendation", {})
        dist = rec.get("distribution", {})

        parts.extend([
            "=" * 60,
            "## BINDING SCORING DATA (DO NOT CONTRADICT)",
            "=" * 60,
            "",
            f"Composite Score: {scoring.get('composite_score', 'N/A')}/100",
            f"Recommendation: {rec.get('action', 'HOLD')} (THIS IS BINDING)",
            f"Distribution: BUY {dist.get('buy', 0)}% | HOLD {dist.get('hold', 0)}% | SELL {dist.get('sell', 0)}%",
            f"Confidence: {rec.get('confidence', 'N/A')}%",
            f"Time Horizon: {rec.get('time_horizon', 'N/A')}",
            "",
        ])

        # Component scores
        parts.append("### Component Scores")
        components = scoring.get("component_scores", {})
        for name, data in components.items():
            parts.append(f"- {name.title()}: {data.get('score', 'N/A')}/100 ({data.get('confidence', 'N/A')} confidence)")
        parts.append("")

        # Key factors
        key_factors = scoring.get("key_factors", [])
        if key_factors:
            bullish = [f["factor"] for f in key_factors if f.get("impact") == "bullish"]
            bearish = [f["factor"] for f in key_factors if f.get("impact") == "bearish"]

            parts.append("### Key Factors")
            parts.append(f"Bullish: {', '.join(bullish) if bullish else 'None'}")
            parts.append(f"Bearish: {', '.join(bearish) if bearish else 'None'}")
            parts.append("")

        # =====================================================================
        # SECTION 2: TECHNICAL & POSITION DATA
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## TECHNICAL ANALYSIS DATA",
            "=" * 60,
            "",
        ])

        tech_content = step_data.get("technical", {}).get("content", "")
        if tech_content:
            parts.append(tech_content[:4000])
        else:
            parts.append("Technical data not available.")
        parts.append("")

        # Position data with sector ranking explanation
        parts.extend([
            "## MARKET POSITION DATA",
            "",
        ])

        pos_content = step_data.get("position", {}).get("content", "")
        pos_raw = step_data.get("position", {}).get("raw_data", {})

        if pos_content:
            parts.append(pos_content[:3000])

        # Add sector ranking methodology
        pos_data = step_data.get("position", {})
        sector_context = pos_data.get("sector_context", {})
        if sector_context:
            sector_rank = sector_context.get("sector_rank")
            total_sectors = sector_context.get("total_sectors", 11)
            sector_change = sector_context.get("sector_change_percent", 0)
            sector_status = sector_context.get("sector_status", "N/A")
            stock_sector = sector_context.get("stock_sector", "N/A")

            parts.extend([
                "",
                "### Sector Ranking Methodology",
                f"- Sector: {stock_sector}",
                f"- Ranking System: Daily performance ranking (FMP Sector Performance API)",
                f"- Ranking: #{sector_rank}/{total_sectors} sectors today",
                f"- Today's Change: {sector_change:+.2f}%" if sector_change else "- Today's Change: N/A",
                f"- Status: {sector_status}",
                f"- Note: Sector rank is 1-DAY data; RS analysis uses multi-timeframe (21d, 63d, 126d)",
                f"- Interpretation: Lower rank number = better performing sector today",
            ])
        parts.append("")

        # =====================================================================
        # SECTION 3: RISK & SENTIMENT DATA
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## RISK ANALYSIS DATA",
            "=" * 60,
            "",
        ])

        risk_content = step_data.get("risk", {}).get("content", "")
        if risk_content:
            parts.append(risk_content[:3000])
        parts.append("")

        parts.extend([
            "## SENTIMENT DATA",
            "",
        ])

        sent_content = step_data.get("sentiment", {}).get("content", "")
        if sent_content:
            parts.append(sent_content[:2500])
        parts.append("")

        # =====================================================================
        # SECTION 4: FUNDAMENTAL DATA
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## FUNDAMENTAL ANALYSIS DATA",
            "=" * 60,
            "",
        ])

        fund_content = step_data.get("fundamental", {}).get("content", "")
        if fund_content:
            parts.append(fund_content[:4000])
        parts.append("")

        # =====================================================================
        # SECTION 5: PEER COMPARISON (NEW)
        # =====================================================================
        if peer_data and peer_data.get("peers"):
            parts.extend([
                "=" * 60,
                "## PEER COMPARISON DATA (INCLUDE TABLE IN REPORT)",
                "=" * 60,
                "",
                f"Target: {symbol} ({peer_data.get('sector', 'N/A')})",
                "",
                "| Symbol | P/E (TTM) | P/S (TTM) | Revenue Growth | Market Cap |",
                "|--------|-----------|-----------|----------------|------------|",
            ])

            # Add target row
            target = peer_data.get("target_metrics", {})
            parts.append(
                f"| **{symbol}** | {target.get('pe_ttm', 'N/A')} | {target.get('ps_ttm', 'N/A')} | "
                f"{target.get('revenue_growth', 'N/A')}% | ${target.get('market_cap', 0)/1e9:.1f}B |"
            )

            # Add peer rows
            for peer in peer_data.get("peers", [])[:4]:
                parts.append(
                    f"| {peer.get('symbol', 'N/A')} | {peer.get('pe_ttm', 'N/A')} | {peer.get('ps_ttm', 'N/A')} | "
                    f"{peer.get('revenue_growth', 'N/A')}% | ${peer.get('market_cap', 0)/1e9:.1f}B |"
                )

            parts.append("")
            parts.append("Use this table in the Fundamental section. Comment on how target compares to peers.")
            parts.append("")

        # =====================================================================
        # SECTION 6: EARNINGS CALENDAR (NEW - SPECIFIC DATES)
        # =====================================================================
        if earnings_data:
            parts.extend([
                "=" * 60,
                "## EARNINGS CALENDAR (USE EXACT DATES)",
                "=" * 60,
                "",
            ])

            if earnings_data.get("next_earnings_date"):
                parts.append(f"NEXT EARNINGS DATE: {earnings_data['next_earnings_date']} ({earnings_data.get('earnings_time', 'TBD')})")
                parts.append(f"Fiscal Quarter: {earnings_data.get('fiscal_quarter', 'N/A')}")
            else:
                parts.append("Next earnings date: Not yet announced")

            if earnings_data.get("beat_rate") is not None:
                parts.append(f"Historical EPS Beat Rate: {earnings_data['beat_rate']*100:.0f}%")

            parts.append("")
            parts.append("IMPORTANT: Use this EXACT date in the Catalysts section. Do not write 'dự kiến sắp tới'.")
            parts.append("")

        # =====================================================================
        # SECTION 7: WEB SEARCH RESULTS (WITH CITATION INSTRUCTION)
        # =====================================================================
        if web_data and web_data.get("citations"):
            parts.extend([
                "=" * 60,
                "## WEB SEARCH RESULTS (CITE INLINE)",
                "=" * 60,
                "",
            ])

            # Add answers
            for item in web_data.get("news", []):
                parts.append(f"### {item['category'].upper()}")
                parts.append(item["answer"][:1500])
                parts.append("")

            # Add citations for reference
            parts.append("### Available Sources (USE INLINE CITATIONS)")
            for i, citation in enumerate(web_data.get("citations", [])[:6], 1):
                title = citation.get("title", "Untitled")[:60]
                url = citation.get("url", "")
                parts.append(f"[{i}] [{title}]({url})")

            parts.append("")
            parts.append("INSTRUCTION: When making claims from web search, cite source INLINE like this: '...sell-off due to geopolitics [Barrons](url)'")
            parts.append("")

        # =====================================================================
        # SECTION 8: OUTPUT INSTRUCTIONS
        # =====================================================================
        parts.extend([
            "=" * 60,
            "## YOUR TASK",
            "=" * 60,
            "",
            f"Generate a complete investment report in {target_language.upper()} with these sections:",
            "",
            "1. **Tóm tắt Điều hành (Executive Summary)** - 3-4 paragraphs",
            "2. **Phân tích Kỹ thuật & Vị thế Thị trường** - Include sector rank methodology",
            "3. **Phân tích Rủi ro & Tâm lý** - Include specific risk metrics",
            "4. **Đánh giá Giá trị Cơ bản** - Include peer comparison table",
            "5. **Tin tức & Chất xúc tác** - Include EXACT earnings date, inline citations",
            "6. **Khuyến nghị Cuối cùng** - MUST match scoring, consistent Entry/Stop/Target",
            "",
            "REMEMBER:",
            f"- Recommendation is {rec.get('action', 'HOLD')} - DO NOT contradict this",
            "- Entry/Stop/Target must be mathematically consistent (stop % calculated from entry, not current price)",
            "- Include peer comparison table if data provided",
            f"- Use exact earnings date: {earnings_data.get('next_earnings_date', 'N/A') if earnings_data else 'N/A'}",
            "- Cite web sources inline with claims",
        ])

        return "\n".join(parts)

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
**Version:** Synthesis V2 (Single LLM Call)

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
- **Architecture:** Single LLM Call (V2) - 100% consistency guaranteed
- **LLM Calls:** 1 (consolidated)

### Data Freshness
{chr(10).join(freshness_lines)}

---
*Report generated by HealerAgent Market Scanner v2.0 (Synthesis V2)*
"""
        return footer

    # =========================================================================
    # RUN MISSING STEPS (inherited from V1)
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
            "sector_context": raw_data.get("sector_context"),  # Include sector_context
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
