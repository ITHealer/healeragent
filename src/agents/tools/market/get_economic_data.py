# src/agents/tools/market/get_economic_data.py

"""
Tool: Get Economic Data (Macro Context)

Provides macroeconomic data for market context:
- Treasury rates (Fed rates proxy)
- GDP growth
- CPI/Inflation
- Unemployment rate

FMP Stable APIs:
- /stable/treasury-rates - Treasury rates
- /stable/economic-indicators?name=GDP - GDP data
- /stable/economic-indicators?name=CPI - Inflation data
- /stable/economic-indicators?name=unemploymentRate - Unemployment
"""

import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

import httpx

from src.agents.tools.base import BaseTool, ToolOutput, ToolSchema, ToolParameter
from src.helpers.redis_cache import get_redis_client_llm
from src.utils.logger.custom_logging import LoggerMixin


class GetEconomicDataTool(BaseTool, LoggerMixin):
    """
    Tool: Get Economic/Macro Data

    Fetches macroeconomic indicators for market context analysis:
    - Treasury rates (proxy for Fed rates)
    - GDP growth rate
    - CPI inflation
    - Unemployment rate

    Use case: Add macro context to stock/crypto analysis
    """

    CACHE_TTL = 3600  # 1 hour - macro data doesn't change frequently
    FMP_STABLE_URL = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: str):
        """
        Initialize GetEconomicDataTool

        Args:
            api_key: FMP API key
        """
        super().__init__()
        self.api_key = api_key

        self.schema = ToolSchema(
            name="getEconomicData",
            category="market",
            description=(
                "Fetch macroeconomic data for market context analysis. "
                "Returns Treasury rates (Fed rates proxy), GDP growth, CPI inflation, unemployment. "
                "NO SYMBOL REQUIRED - returns overall economic indicators. "
                "Use when analyzing macro environment impact on stocks/crypto."
            ),
            capabilities=[
                "✅ Treasury rates (10Y, 2Y, 30Y, 5Y) - Fed rate proxy",
                "✅ GDP growth rate (quarterly)",
                "✅ CPI inflation rate",
                "✅ Unemployment rate",
                "✅ Yield curve analysis (2Y-10Y spread)",
                "✅ Market context for investment decisions"
            ],
            limitations=[
                "❌ Economic data updated monthly/quarterly",
                "❌ May have 1-2 week delay from official release",
                "❌ Does not include Fed meeting dates"
            ],
            usage_hints=[
                # English
                "User asks: 'What's the current interest rate?' → USE THIS",
                "User asks: 'Fed rate environment' → USE THIS",
                "User asks: 'Macro outlook' → USE THIS",
                "User asks: 'Economic conditions' → USE THIS",
                "User asks: 'Treasury yield' → USE THIS",
                # Vietnamese
                "User asks: 'Lãi suất hiện tại?' → USE THIS",
                "User asks: 'Tình hình kinh tế vĩ mô' → USE THIS",
                "User asks: 'Lạm phát bao nhiêu?' → USE THIS",
                # Context
                "For stock analysis → Combine with getStockPrice for macro context",
                "For crypto analysis → Use to assess risk-on/risk-off environment"
            ],
            parameters=[
                ToolParameter(
                    name="indicators",
                    type="array",
                    description="Economic indicators to fetch (default: all). Options: treasury, gdp, cpi, unemployment",
                    required=False,
                    default=["treasury", "gdp", "cpi", "unemployment"]
                )
            ],
            returns={
                "treasury_rates": "object - Current treasury rates (2Y, 5Y, 10Y, 30Y)",
                "yield_curve": "object - Yield curve analysis",
                "gdp": "object - GDP growth rate (%) not raw GDP value",
                "inflation": "object - Inflation rate (%) not CPI index",
                "unemployment": "object - Unemployment rate (%)",
                "summary": "string - Quick macro summary",
                "timestamp": "string"
            },
            typical_execution_time_ms=1500,
            requires_symbol=False
        )

    async def execute(
        self,
        indicators: Optional[List[str]] = None,
        **kwargs
    ) -> ToolOutput:
        """
        Execute getEconomicData

        Args:
            indicators: List of indicators to fetch (default: all)

        Returns:
            ToolOutput with economic data
        """
        start_time = time.time()

        # Default to all indicators
        if not indicators:
            indicators = ["treasury", "gdp", "cpi", "unemployment"]

        try:
            self.logger.info(f"[{self.schema.name}] Fetching economic data: {indicators}")

            # Check cache
            cache_key = f"getEconomicData_{'_'.join(sorted(indicators))}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"[{self.schema.name}] Cache HIT")
                return ToolOutput(
                    tool_name=self.schema.name,
                    status="success",
                    data=cached_result,
                    formatted_context=self._format_for_llm(cached_result)
                )

            # Fetch data for each indicator
            result = {
                "timestamp": datetime.now().isoformat(),
                "data_source": "FMP API"
            }

            # Treasury rates
            if "treasury" in indicators:
                treasury_data = await self._fetch_treasury_rates()
                if treasury_data:
                    result["treasury_rates"] = treasury_data
                    result["yield_curve"] = self._analyze_yield_curve(treasury_data)

            # GDP Growth Rate (try realGDPGrowth first, then calculate from GDP)
            if "gdp" in indicators:
                gdp_data = await self._fetch_economic_indicator("realGDPGrowth")
                if gdp_data and gdp_data.get("latest_value") is not None:
                    gdp_data["type"] = "growth_rate"
                    result["gdp"] = gdp_data
                else:
                    # Fallback: fetch GDP and calculate YoY growth
                    gdp_raw = await self._fetch_economic_indicator("GDP")
                    if gdp_raw and gdp_raw.get("recent_data") and len(gdp_raw["recent_data"]) >= 4:
                        current = gdp_raw["latest_value"]
                        previous = gdp_raw["recent_data"][3]["value"] if len(gdp_raw["recent_data"]) > 3 else gdp_raw["previous_value"]
                        if current and previous and previous > 0:
                            yoy_growth = ((current - previous) / previous) * 100
                            result["gdp"] = {
                                "indicator": "GDP_YoY",
                                "latest_value": round(yoy_growth, 2),
                                "latest_date": gdp_raw["latest_date"],
                                "type": "calculated_yoy",
                                "trend": "increasing" if yoy_growth > 0 else "decreasing"
                            }

            # Inflation Rate (try inflationRate first, then calculate from CPI)
            if "cpi" in indicators:
                inflation_data = await self._fetch_economic_indicator("inflationRate")
                if inflation_data and inflation_data.get("latest_value") is not None:
                    inflation_data["type"] = "inflation_rate"
                    result["inflation"] = inflation_data
                else:
                    # Fallback: fetch CPI and calculate YoY inflation
                    cpi_raw = await self._fetch_economic_indicator("CPI")
                    if cpi_raw and cpi_raw.get("recent_data"):
                        # Need 12+ months of data for YoY
                        self.logger.info("inflationRate not available, CPI index returned (need YoY calculation)")
                        result["inflation"] = {
                            "indicator": "CPI_index",
                            "latest_value": cpi_raw.get("latest_value"),
                            "latest_date": cpi_raw.get("latest_date"),
                            "type": "index_value",
                            "note": "This is CPI index, not inflation rate. Need 12-month data for YoY calculation."
                        }

            # Unemployment
            if "unemployment" in indicators:
                unemployment_data = await self._fetch_economic_indicator("unemploymentRate")
                if unemployment_data:
                    result["unemployment"] = unemployment_data

            # Generate summary
            result["summary"] = self._generate_summary(result)

            # Cache result
            await self._cache_result(cache_key, result)

            execution_time = (time.time() - start_time) * 1000
            self.logger.info(
                f"[{self.schema.name}] SUCCESS ({execution_time:.0f}ms)"
            )

            return ToolOutput(
                tool_name=self.schema.name,
                status="success",
                data=result,
                formatted_context=self._format_for_llm(result)
            )

        except Exception as e:
            self.logger.error(f"[{self.schema.name}] Error: {e}", exc_info=True)
            return ToolOutput(
                tool_name=self.schema.name,
                status="error",
                error=str(e),
                metadata={"type": type(e).__name__}
            )

    async def _fetch_treasury_rates(self) -> Optional[Dict[str, Any]]:
        """Fetch current treasury rates"""
        try:
            url = f"{self.FMP_STABLE_URL}/treasury-rates"
            params = {"apikey": self.api_key}

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if not data or not isinstance(data, list):
                return None

            # Get latest data point
            latest = data[0] if data else {}

            return {
                "date": latest.get("date"),
                "month_1": latest.get("month1"),
                "month_3": latest.get("month3"),
                "month_6": latest.get("month6"),
                "year_1": latest.get("year1"),
                "year_2": latest.get("year2"),
                "year_5": latest.get("year5"),
                "year_10": latest.get("year10"),
                "year_30": latest.get("year30"),
            }

        except Exception as e:
            self.logger.error(f"Error fetching treasury rates: {e}")
            return None

    async def _fetch_economic_indicator(self, indicator_name: str) -> Optional[Dict[str, Any]]:
        """Fetch economic indicator data"""
        try:
            url = f"{self.FMP_STABLE_URL}/economic-indicators"
            params = {
                "name": indicator_name,
                "apikey": self.api_key
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

            if not data or not isinstance(data, list):
                return None

            # Get recent data points (last 4 quarters/months)
            recent = data[:4] if len(data) >= 4 else data

            return {
                "indicator": indicator_name,
                "latest_value": recent[0].get("value") if recent else None,
                "latest_date": recent[0].get("date") if recent else None,
                "previous_value": recent[1].get("value") if len(recent) > 1 else None,
                "trend": self._calculate_trend(recent),
                "recent_data": [
                    {"date": r.get("date"), "value": r.get("value")}
                    for r in recent
                ]
            }

        except Exception as e:
            self.logger.error(f"Error fetching {indicator_name}: {e}")
            return None

    def _analyze_yield_curve(self, treasury: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze yield curve for recession signals"""
        year_2 = treasury.get("year_2")
        year_10 = treasury.get("year_10")

        if year_2 is None or year_10 is None:
            return {"status": "unavailable"}

        spread = year_10 - year_2

        if spread < 0:
            status = "inverted"
            signal = "Recession warning - yield curve inverted"
        elif spread < 0.25:
            status = "flat"
            signal = "Caution - yield curve flattening"
        else:
            status = "normal"
            signal = "Normal yield curve - no recession signal"

        return {
            "spread_10y_2y": round(spread, 3),
            "status": status,
            "signal": signal
        }

    def _calculate_trend(self, data: List[Dict]) -> str:
        """Calculate trend from recent data"""
        if len(data) < 2:
            return "insufficient_data"

        latest = data[0].get("value")
        previous = data[1].get("value")

        if latest is None or previous is None:
            return "unknown"

        if latest > previous:
            return "increasing"
        elif latest < previous:
            return "decreasing"
        else:
            return "stable"

    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """Generate quick macro summary"""
        parts = []

        # Treasury summary
        treasury = result.get("treasury_rates", {})
        if treasury.get("year_10"):
            parts.append(f"10Y Treasury: {treasury['year_10']:.2f}%")

        # Yield curve
        yc = result.get("yield_curve", {})
        if yc.get("status"):
            parts.append(f"Yield curve: {yc['status']}")

        # GDP Growth Rate
        gdp = result.get("gdp", {})
        if gdp.get("latest_value") is not None:
            parts.append(f"GDP Growth: {gdp['latest_value']:.1f}%")

        # Inflation Rate (use 'inflation' or fallback to 'cpi')
        inflation = result.get("inflation", {}) or result.get("cpi", {})
        if inflation.get("latest_value") is not None:
            parts.append(f"Inflation: {inflation['latest_value']:.1f}%")

        # Unemployment
        unemp = result.get("unemployment", {})
        if unemp.get("latest_value") is not None:
            parts.append(f"Unemployment: {unemp['latest_value']:.1f}%")

        return " | ".join(parts) if parts else "No data available"

    def _format_for_llm(self, result: Dict[str, Any]) -> str:
        """Format data for LLM context"""
        lines = [
            "## Economic Data (Macro Context)",
            f"**Summary**: {result.get('summary', 'N/A')}",
            ""
        ]

        # Treasury rates
        treasury = result.get("treasury_rates", {})
        if treasury:
            lines.append("### Treasury Rates (Fed Rate Proxy)")
            lines.append(f"- 2-Year: {treasury.get('year_2', 'N/A')}%")
            lines.append(f"- 5-Year: {treasury.get('year_5', 'N/A')}%")
            lines.append(f"- 10-Year: {treasury.get('year_10', 'N/A')}%")
            lines.append(f"- 30-Year: {treasury.get('year_30', 'N/A')}%")
            lines.append("")

        # Yield curve
        yc = result.get("yield_curve", {})
        if yc.get("status"):
            lines.append("### Yield Curve Analysis")
            lines.append(f"- 10Y-2Y Spread: {yc.get('spread_10y_2y', 'N/A')}%")
            lines.append(f"- Status: {yc.get('status', 'N/A')}")
            lines.append(f"- Signal: {yc.get('signal', 'N/A')}")
            lines.append("")

        # GDP Growth Rate
        gdp = result.get("gdp", {})
        if gdp.get("latest_value") is not None:
            gdp_type = gdp.get('type', 'unknown')
            lines.append("### GDP Growth Rate")
            lines.append(f"- Latest: {gdp['latest_value']:.1f}% ({gdp.get('latest_date', 'N/A')}) [{gdp_type}]")
            lines.append(f"- Trend: {gdp.get('trend', 'N/A')}")
            lines.append("")

        # Inflation Rate (use 'inflation' or fallback to 'cpi')
        inflation = result.get("inflation", {}) or result.get("cpi", {})
        if inflation.get("latest_value") is not None:
            inflation_type = inflation.get('type', 'unknown')
            lines.append("### Inflation Rate")
            lines.append(f"- Latest: {inflation['latest_value']:.1f}% ({inflation.get('latest_date', 'N/A')}) [{inflation_type}]")
            lines.append(f"- Trend: {inflation.get('trend', 'N/A')}")
            if inflation.get('note'):
                lines.append(f"- Note: {inflation['note']}")
            lines.append("")

        # Unemployment
        unemp = result.get("unemployment", {})
        if unemp.get("latest_value"):
            lines.append("### Unemployment Rate")
            lines.append(f"- Latest: {unemp['latest_value']:.1f}% ({unemp.get('latest_date', 'N/A')})")
            lines.append(f"- Trend: {unemp.get('trend', 'N/A')}")
            lines.append("")

        lines.append(f"*Source: FMP API | Updated: {result.get('timestamp', 'N/A')}*")

        return "\n".join(lines)

    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result from Redis"""
        try:
            redis_client = await get_redis_client_llm()
            cached_bytes = await redis_client.get(cache_key)
            await redis_client.close()

            if cached_bytes:
                if isinstance(cached_bytes, bytes):
                    return json.loads(cached_bytes.decode('utf-8'))
                return json.loads(cached_bytes)
            return None
        except Exception as e:
            self.logger.warning(f"Cache read error: {e}")
            return None

    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result to Redis"""
        try:
            redis_client = await get_redis_client_llm()
            json_string = json.dumps(result)
            await redis_client.set(cache_key, json_string, ex=self.CACHE_TTL)
            await redis_client.close()
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")


# Standalone test
if __name__ == "__main__":
    import asyncio
    import os

    async def test():
        api_key = os.getenv("FMP_API_KEY")
        if not api_key:
            print("❌ FMP_API_KEY not set")
            return

        tool = GetEconomicDataTool(api_key=api_key)

        print("\n" + "="*60)
        print("Testing getEconomicData Tool")
        print("="*60)

        result = await tool.execute()

        if result.status == "success":
            print("✅ SUCCESS")
            print(f"\nSummary: {result.data.get('summary')}")
            print(f"\nFormatted:\n{result.formatted_context}")
        else:
            print(f"❌ Error: {result.error}")

    asyncio.run(test())
