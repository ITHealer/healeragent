"""
Finance Guru - Valuation Tools (Phase 1)

Layer 3 of the 3-layer architecture: Agent-callable tool interfaces.

These tools provide the agent-facing interface for stock valuation:
- CalculateDCFTool: Discounted Cash Flow valuation
- CalculateGrahamTool: Graham Formula valuation
- CalculateDDMTool: Dividend Discount Model valuation
- CalculateComparablesTool: Comparable company analysis
- GetValuationSummaryTool: Combined valuation from multiple methods

WHAT: Agent-callable tools for intrinsic value calculations
WHY: Provides standardized interface for LLM agents to assess stock valuations
ARCHITECTURE: Layer 3 of 3-layer type-safe architecture

Author: HealerAgent Development Team
"""

import logging
from typing import Any, Dict, List, Optional

from src.agents.tools.base import (
    BaseTool,
    ToolOutput,
    ToolParameter,
    ToolSchema,
)
from src.agents.tools.finance_guru.calculators.valuation import (
    DCFCalculator,
    GrahamCalculator,
    DDMCalculator,
    ComparableCalculator,
    ValuationCalculator,
)
from src.agents.tools.finance_guru.models.valuation import (
    # Enums
    ValuationMethod,
    DDMType,
    # DCF
    DCFInputData,
    DCFConfig,
    # Graham
    GrahamInputData,
    GrahamConfig,
    # DDM
    DDMInputData,
    DDMConfig,
    # Comparable
    ComparableCompany,
    ComparableInputData,
    # NEW: Data transparency models
    DataSourceAttribution,
    ValuationBridge,
    # Risk framework
    RiskFramework,
    ScenarioCase,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_success_output(
    tool_name: str,
    data: Dict[str, Any],
    formatted_context: Optional[str] = None,
    execution_time_ms: int = 0,
) -> ToolOutput:
    """Create a successful ToolOutput."""
    return ToolOutput(
        tool_name=tool_name,
        status="success",
        data=data,
        formatted_context=formatted_context,
        execution_time_ms=execution_time_ms,
    )


def create_error_output(
    tool_name: str,
    error: str,
    execution_time_ms: int = 0,
) -> ToolOutput:
    """Create an error ToolOutput with formatted_context for LLM."""
    formatted_context = f"Error in {tool_name}: {error}"
    return ToolOutput(
        tool_name=tool_name,
        status="error",
        error=error,
        formatted_context=formatted_context,
        execution_time_ms=execution_time_ms,
    )


async def fetch_fundamental_data(
    symbol: str,
    fmp_api_key: Optional[str] = None,
    fetch_historical_fcf: bool = False,
) -> Dict[str, Any]:
    """
    Fetch fundamental data from FMP API for valuation.

    Enhanced with data attribution for institutional-grade transparency.

    Args:
        symbol: Stock symbol
        fmp_api_key: FMP API key
        fetch_historical_fcf: If True, fetch 3-year historical FCF for normalization

    Returns:
        Dict with fundamental metrics AND data_sources attribution
    """
    from datetime import datetime
    import uuid

    try:
        from src.agents.tools.finance_guru.services.fmp_service import FMPService

        fmp = FMPService(api_key=fmp_api_key)

        # Track API endpoints called for attribution
        api_endpoints_called = []
        data_sources = {}
        fetch_timestamp = datetime.utcnow().isoformat() + "Z"
        snapshot_id = str(uuid.uuid4())[:8]

        # Fetch key metrics and profile
        profile = await fmp.get_company_profile(symbol)
        api_endpoints_called.append(f"/v3/profile/{symbol}")

        key_metrics = await fmp.get_key_metrics(symbol)
        api_endpoints_called.append(f"/v3/key-metrics/{symbol}")

        income = await fmp.get_income_statement(symbol)
        api_endpoints_called.append(f"/v3/income-statement/{symbol}")

        cash_flow = await fmp.get_cash_flow_statement(symbol)
        api_endpoints_called.append(f"/v3/cash-flow-statement/{symbol}")

        if not profile:
            raise ValueError(f"Could not fetch data for {symbol}")

        # Extract fiscal year from cash flow statement
        fiscal_year = None
        fiscal_period = None
        if cash_flow:
            fiscal_year = cash_flow.get("calendarYear") or cash_flow.get("date", "")[:4]
            fiscal_period = cash_flow.get("period", "FY")

        # Track price date from profile
        price_date = profile.get("date") or fetch_timestamp[:10]

        # Resolve shares_outstanding with multiple fallback sources
        shares_outstanding_raw = profile.get("sharesOutstanding") or 0
        shares_source = "FMP Company Profile"
        if not shares_outstanding_raw and income:
            # Fallback 1: weighted average shares from income statement
            shares_outstanding_raw = (
                income.get("weightedAverageShsOut")
                or income.get("weightedAverageShsOutDil")
                or 0
            )
            shares_source = "FMP Income Statement (weighted avg)"
        if not shares_outstanding_raw:
            # Fallback 2: calculate from market cap / price
            mkt_cap = profile.get("mktCap") or profile.get("marketCap") or 0
            price = profile.get("price") or 0
            if mkt_cap > 0 and price > 0:
                shares_outstanding_raw = mkt_cap / price
                shares_source = "Calculated (Market Cap / Price)"

        shares_outstanding_m = shares_outstanding_raw / 1e6 if shares_outstanding_raw else 0

        # Get current FCF
        current_fcf = cash_flow.get("freeCashFlow", 0) / 1e6 if cash_flow else 0

        # Build data sources attribution
        data_sources["price"] = f"FMP Company Profile, as of {price_date}"
        data_sources["shares_outstanding"] = shares_source
        data_sources["eps"] = f"FMP Key Metrics, {fiscal_period} {fiscal_year}" if key_metrics else "N/A"
        data_sources["book_value_per_share"] = f"FMP Key Metrics, {fiscal_period} {fiscal_year}" if key_metrics else "N/A"
        data_sources["fcf"] = f"FMP Cash Flow Statement, {fiscal_period} {fiscal_year}" if cash_flow else "N/A"
        data_sources["cash"] = f"FMP Company Profile, as of {price_date}"
        data_sources["debt"] = f"FMP Company Profile, as of {price_date}"

        # Historical FCF for normalization
        historical_fcf = None
        normalized_fcf = None
        fcf_volatility = None
        historical_fcf_years = []

        if fetch_historical_fcf:
            try:
                # Fetch 3 years of cash flow data
                historical_cf = await fmp.get_cash_flow_statement(symbol, limit=3)
                api_endpoints_called.append(f"/v3/cash-flow-statement/{symbol}?limit=3")

                if isinstance(historical_cf, list) and len(historical_cf) >= 2:
                    fcf_values = []
                    for cf in historical_cf:
                        fcf_val = cf.get("freeCashFlow", 0)
                        if fcf_val:
                            fcf_values.append(fcf_val / 1e6)
                            year = cf.get("calendarYear") or cf.get("date", "")[:4]
                            historical_fcf_years.append(year)

                    if fcf_values:
                        historical_fcf = fcf_values
                        # Calculate normalized FCF (3-year average)
                        normalized_fcf = sum(fcf_values) / len(fcf_values)
                        # Calculate volatility (coefficient of variation)
                        if len(fcf_values) >= 2:
                            mean_fcf = normalized_fcf
                            variance = sum((x - mean_fcf) ** 2 for x in fcf_values) / len(fcf_values)
                            std_dev = variance ** 0.5
                            fcf_volatility = (std_dev / abs(mean_fcf) * 100) if mean_fcf != 0 else None

                        # Update FCF source with historical years
                        data_sources["fcf_historical"] = f"FMP Cash Flow Statement, FY{historical_fcf_years[-1]}-FY{historical_fcf_years[0]}"
            except Exception as e:
                logger.warning(f"Failed to fetch historical FCF: {e}")

        return {
            "symbol": symbol,
            "price": profile.get("price", 0),
            "eps": key_metrics.get("eps", 0) if key_metrics else 0,
            "book_value_per_share": key_metrics.get("bookValuePerShare", 0) if key_metrics else 0,
            "revenue_per_share": key_metrics.get("revenuePerShare", 0) if key_metrics else 0,
            "dividend_per_share": key_metrics.get("dividendPerShare", 0) if key_metrics else 0,
            "shares_outstanding": shares_outstanding_m,
            "free_cash_flow": current_fcf,
            "cash": profile.get("cash", 0) / 1e6,
            "debt": profile.get("totalDebt", 0) / 1e6,
            # Historical FCF data
            "historical_fcf": historical_fcf,
            "normalized_fcf": normalized_fcf,
            "fcf_volatility": fcf_volatility,
            # NEW: Data attribution for transparency
            "data_attribution": {
                "data_as_of_date": fetch_timestamp,
                "price_date": price_date,
                "fiscal_year": f"FY{fiscal_year}" if fiscal_year else None,
                "api_endpoints": api_endpoints_called,
                "data_sources": data_sources,
                "snapshot_id": snapshot_id,
            },
        }

    except ImportError:
        logger.warning("FMPService not available, using mock data")
        return _create_mock_fundamental_data(symbol)


def _normalize_growth_rates(growth_rates: List[Any]) -> List[float]:
    """Normalize growth rates to floats.

    Handles various input formats:
    - Float: 0.10 → 0.10
    - String decimal: "0.10" → 0.10
    - String percentage: "10%" → 0.10
    - Integer: 10 → 0.10 (assumes percentage if > 1)
    """
    normalized = []
    for g in growth_rates:
        if isinstance(g, (int, float)):
            # If value > 1, assume it's a percentage (e.g., 10 means 10%)
            val = float(g)
            if val > 1:
                val = val / 100
            normalized.append(val)
        elif isinstance(g, str):
            # Remove whitespace and percentage sign
            g_clean = g.strip().rstrip('%')
            try:
                val = float(g_clean)
                # If original had % or value > 1, treat as percentage
                if '%' in g or val > 1:
                    val = val / 100
                normalized.append(val)
            except ValueError:
                # Skip invalid values
                continue
        # Skip None or other types
    return normalized if normalized else [0.10, 0.08, 0.06, 0.05, 0.04]


def _create_mock_fundamental_data(symbol: str) -> Dict[str, Any]:
    """Create mock fundamental data for testing."""
    mock_data = {
        "AAPL": {
            "price": 185.0,
            "eps": 6.15,
            "book_value_per_share": 4.25,
            "revenue_per_share": 24.35,
            "dividend_per_share": 0.96,
            "shares_outstanding": 15400.0,  # millions
            "free_cash_flow": 99000.0,  # millions
            "cash": 62000.0,
            "debt": 109000.0,
        },
        "MSFT": {
            "price": 380.0,
            "eps": 11.05,
            "book_value_per_share": 28.50,
            "revenue_per_share": 29.80,
            "dividend_per_share": 3.00,
            "shares_outstanding": 7450.0,
            "free_cash_flow": 59000.0,
            "cash": 104000.0,
            "debt": 47000.0,
        },
    }

    default = {
        "price": 150.0,
        "eps": 5.0,
        "book_value_per_share": 20.0,
        "revenue_per_share": 30.0,
        "dividend_per_share": 2.0,
        "shares_outstanding": 1000.0,
        "free_cash_flow": 5000.0,
        "cash": 10000.0,
        "debt": 5000.0,
    }

    data = mock_data.get(symbol.upper(), default)
    data["symbol"] = symbol.upper()
    return data


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================


class CalculateDCFTool(BaseTool):
    """Tool for DCF (Discounted Cash Flow) valuation.

    Values a stock based on projected future free cash flows
    discounted to present value.

    Usage by agent:
        calculateDCF(
            symbol="AAPL",
            current_fcf=99000,
            growth_rates=[0.12, 0.10, 0.08, 0.06, 0.04],
            discount_rate=0.10
        )
    """

    def __init__(self):
        super().__init__()
        self.calculator = DCFCalculator()

        self.schema = ToolSchema(
            name="calculateDCF",
            category="valuation",
            description=(
                "Calculate intrinsic value using Discounted Cash Flow analysis. "
                "Projects future free cash flows and discounts them to present value. "
                "Best for companies with predictable cash flows."
            ),
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock symbol (e.g., 'AAPL')",
                    required=True,
                ),
                ToolParameter(
                    name="current_fcf",
                    type="number",
                    description="Current free cash flow in millions (e.g., 99000)",
                    required=False,
                ),
                ToolParameter(
                    name="growth_rates",
                    type="array",
                    description="Projected annual growth rates (e.g., [0.12, 0.10, 0.08])",
                    required=False,
                ),
                ToolParameter(
                    name="discount_rate",
                    type="number",
                    description="WACC/discount rate (e.g., 0.10 for 10%)",
                    required=False,
                    default=0.10,
                ),
                ToolParameter(
                    name="terminal_growth",
                    type="number",
                    description="Long-term growth rate (typically 2-3%)",
                    required=False,
                    default=0.025,
                ),
                ToolParameter(
                    name="normalize_fcf",
                    type="boolean",
                    description=(
                        "If true, use 3-year average FCF instead of current FCF. "
                        "Recommended for companies with volatile cash flows."
                    ),
                    required=False,
                    default=False,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute DCF valuation."""
        try:
            symbol = kwargs.get("symbol", "").upper()
            current_fcf = kwargs.get("current_fcf")
            growth_rates_raw = kwargs.get("growth_rates", [0.10, 0.08, 0.06, 0.05, 0.04])
            growth_rates = _normalize_growth_rates(growth_rates_raw)
            discount_rate = kwargs.get("discount_rate", 0.10)
            terminal_growth = kwargs.get("terminal_growth", 0.025)
            normalize_fcf = kwargs.get("normalize_fcf", False)

            if not symbol:
                return create_error_output(self.schema.name, "Symbol is required")

            # Fetch fundamental data, including historical FCF if normalization requested
            fundamentals = await fetch_fundamental_data(
                symbol,
                fetch_historical_fcf=normalize_fcf or current_fcf is None,
            )

            # Determine which FCF to use
            fcf_source = "manual" if current_fcf is not None else "api"
            fcf_note = None

            if current_fcf is None:
                # Check if normalization is requested or recommended
                historical_fcf = fundamentals.get("historical_fcf")
                normalized_fcf = fundamentals.get("normalized_fcf")
                fcf_volatility = fundamentals.get("fcf_volatility")
                current_fcf_raw = fundamentals.get("free_cash_flow", 0)

                if normalize_fcf and normalized_fcf:
                    # Use normalized FCF
                    current_fcf = normalized_fcf
                    fcf_source = "normalized_3yr_avg"
                    fcf_note = (
                        f"Using 3-year average FCF (${normalized_fcf:,.0f}M) "
                        f"instead of current (${current_fcf_raw:,.0f}M). "
                        f"Historical: {[f'${f:,.0f}M' for f in (historical_fcf or [])]}"
                    )
                elif fcf_volatility and fcf_volatility > 30 and normalized_fcf:
                    # Auto-suggest normalization for high volatility
                    current_fcf = current_fcf_raw
                    fcf_source = "api"
                    fcf_note = (
                        f"⚠️ FCF volatility is high ({fcf_volatility:.0f}%). "
                        f"Consider using normalize_fcf=true for more stable valuation. "
                        f"Normalized FCF would be ${normalized_fcf:,.0f}M."
                    )
                else:
                    current_fcf = current_fcf_raw
                    fcf_source = "api"

                if current_fcf <= 0:
                    return create_error_output(
                        self.schema.name,
                        f"Cannot perform DCF: {symbol} has no positive free cash flow"
                    )

            shares_outstanding = fundamentals.get("shares_outstanding", 0)
            if not shares_outstanding or shares_outstanding <= 0:
                # Last-resort fallback: estimate from market cap / price
                current_price_check = fundamentals.get("price", 0)
                if current_price_check and current_price_check > 0:
                    # Try to compute from market cap (note: market cap from FMP is in raw units)
                    logger.warning(
                        f"[calculateDCF] shares_outstanding is 0 for {symbol}, "
                        f"cannot perform per-share valuation without share count"
                    )
                    return create_error_output(
                        self.schema.name,
                        f"Cannot perform DCF: shares outstanding data unavailable for {symbol}. "
                        f"Please provide shares_outstanding parameter manually."
                    )
            current_price = fundamentals.get("price")
            cash = fundamentals.get("cash", 0)
            debt = fundamentals.get("debt", 0)

            # Build input data
            data = DCFInputData(
                symbol=symbol,
                current_fcf=current_fcf,
                growth_rates=growth_rates,
                terminal_growth=terminal_growth,
                discount_rate=discount_rate,
                shares_outstanding=shares_outstanding,
                current_price=current_price,
                cash=cash,
                debt=debt,
            )

            config = DCFConfig(margin_of_safety=0.25)
            result = self.calculator.calculate(data, config, fcf_source=fcf_source)

            # Build data attribution for institutional-grade transparency
            raw_attribution = fundamentals.get("data_attribution", {})
            data_attribution = None
            if raw_attribution:
                data_attribution = DataSourceAttribution(
                    data_as_of_date=raw_attribution.get("data_as_of_date", ""),
                    price_date=raw_attribution.get("price_date"),
                    fiscal_year=raw_attribution.get("fiscal_year"),
                    api_endpoints=raw_attribution.get("api_endpoints", []),
                    data_sources=raw_attribution.get("data_sources", {}),
                    snapshot_id=raw_attribution.get("snapshot_id"),
                )

            # Build valuation bridge for calculation transparency
            valuation_bridge = ValuationBridge(
                sum_of_pv_fcf=result.sum_of_pv_fcf,
                pv_terminal_value=result.pv_terminal_value,
                enterprise_value=result.enterprise_value,
                plus_cash=cash,
                minus_debt=debt,
                equity_value=result.equity_value,
                shares_outstanding=shares_outstanding,
                intrinsic_value_per_share=result.intrinsic_value_per_share,
            )

            # Build enhanced formatted_context
            formatted = self._build_formatted_context(
                symbol=symbol,
                result=result,
                current_fcf=current_fcf,
                growth_rates=growth_rates,
                discount_rate=discount_rate,
                terminal_growth=terminal_growth,
                shares_outstanding=shares_outstanding,
                cash=cash,
                debt=debt,
                fcf_source=fcf_source,
                fcf_note=fcf_note,
                data_attribution=data_attribution,
                valuation_bridge=valuation_bridge,
                risk_framework=result.risk_framework,
            )

            return create_success_output(
                self.schema.name,
                data={
                    "symbol": symbol,
                    "intrinsic_value": result.intrinsic_value_per_share,
                    "current_price": result.current_price,
                    "upside_potential": result.upside_potential,
                    "margin_of_safety_price": result.margin_of_safety_price,
                    "enterprise_value": result.enterprise_value,
                    "equity_value": result.equity_value,
                    "verdict": result.verdict,
                    "discount_rate": discount_rate,
                    "terminal_growth": terminal_growth,
                    "fcf_source": fcf_source,
                    "fcf_note": fcf_note,
                    "tv_as_pct_of_ev": result.tv_as_pct_of_ev,
                    "implied_growth_rate": result.implied_growth_rate,
                    "sensitivity_2d": result.sensitivity_2d,
                    "validation_warnings": result.validation_warnings,
                    # NEW: Data transparency fields
                    "data_attribution": data_attribution.model_dump() if data_attribution else None,
                    "valuation_bridge": valuation_bridge.model_dump(),
                    # Risk framework
                    "risk_framework": result.risk_framework.model_dump() if result.risk_framework else None,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"DCF calculation failed: {e}")
            return create_error_output(self.schema.name, str(e))

    def _build_formatted_context(
        self,
        symbol: str,
        result,
        current_fcf: float,
        growth_rates: List[float],
        discount_rate: float,
        terminal_growth: float,
        shares_outstanding: float,
        cash: float,
        debt: float,
        fcf_source: str = "api",
        fcf_note: Optional[str] = None,
        data_attribution: Optional[DataSourceAttribution] = None,
        valuation_bridge: Optional[ValuationBridge] = None,
        risk_framework: Optional[RiskFramework] = None,
    ) -> str:
        """Build enhanced formatted_context for DCF output with full transparency."""
        sections = []

        # Title
        sections.append(f"═══════════════════════════════════════════════════")
        sections.append(f"DCF VALUATION ANALYSIS: {symbol}")
        sections.append(f"═══════════════════════════════════════════════════")

        # NEW Section 0: Data Attribution (for transparency)
        if data_attribution:
            sections.append(f"\n📋 DATA SOURCES & ATTRIBUTION:")
            sections.append(f"  • Data As Of: {data_attribution.data_as_of_date[:10]}")
            sections.append(f"  • Price Date: {data_attribution.price_date or 'N/A'}")
            sections.append(f"  • Fiscal Year: {data_attribution.fiscal_year or 'N/A'}")
            sections.append(f"  • Snapshot ID: {data_attribution.snapshot_id or 'N/A'}")
            if data_attribution.data_sources:
                sections.append(f"  • Sources:")
                for metric, source in data_attribution.data_sources.items():
                    sections.append(f"    - {metric}: {source}")

        # Section 1: Input Parameters
        growth_str = ", ".join(f"{g:.1%}" for g in growth_rates[:5])
        if len(growth_rates) > 5:
            growth_str += "..."

        # FCF source label
        fcf_source_label = {
            "api": "Current (API)",
            "manual": "Manual Input",
            "normalized_3yr_avg": "Normalized (3-Year Avg)",
        }.get(fcf_source, fcf_source)

        sections.append(f"\n📊 INPUT PARAMETERS:")
        sections.append(f"  • FCF: ${current_fcf:,.0f}M ({fcf_source_label})")
        if fcf_note:
            sections.append(f"    └─ {fcf_note}")
        sections.append(f"  • Growth Rates: [{growth_str}]")
        sections.append(f"  • WACC (Discount Rate): {discount_rate:.2%}")  # Changed to .2% for precision
        sections.append(f"  • Terminal Growth: {terminal_growth:.2%}")  # Changed to .2% for precision
        sections.append(f"  • Shares Outstanding: {shares_outstanding:,.0f}M")
        sections.append(f"  • Cash: ${cash:,.0f}M | Debt: ${debt:,.0f}M")

        # Section 2: Cash Flow Projections
        sections.append(f"\n📈 CASH FLOW PROJECTIONS:")
        sections.append(f"  {'Year':<6} {'FCF ($M)':<12} {'Growth':<10} {'PV ($M)':<12}")
        sections.append(f"  {'-'*40}")
        for p in result.projections:
            sections.append(
                f"  {p.year:<6} {p.fcf:>10,.0f}  {p.growth_rate:>8.1%}  {p.present_value:>10,.0f}"
            )

        # Section 3: Valuation Summary
        tv_pct = result.tv_as_pct_of_ev or 0
        sections.append(f"\n💰 VALUATION SUMMARY:")
        sections.append(f"  • Sum of PV (FCFs): ${result.sum_of_pv_fcf:,.0f}M")
        sections.append(f"  • Terminal Value: ${result.terminal_value:,.0f}M")
        sections.append(f"  • PV of Terminal: ${result.pv_terminal_value:,.0f}M ({tv_pct:.1f}% of EV)")
        sections.append(f"  • Enterprise Value: ${result.enterprise_value:,.0f}M")
        sections.append(f"  • Equity Value: ${result.equity_value:,.0f}M")

        # NEW Section 3.5: Valuation Bridge (EV → Equity → Per Share)
        if valuation_bridge:
            sections.append(f"\n📐 VALUATION BRIDGE (EV → Equity → Per Share):")
            sections.append(f"  ┌{'─'*52}┐")
            sections.append(f"  │ {'Step':<30} {'Amount ($M)':>18} │")
            sections.append(f"  ├{'─'*52}┤")
            sections.append(f"  │ {'Sum of PV (FCFs)':<30} {valuation_bridge.sum_of_pv_fcf:>15,.0f}   │")
            sections.append(f"  │ {'+ PV of Terminal Value':<30} {valuation_bridge.pv_terminal_value:>15,.0f}   │")
            sections.append(f"  ├{'─'*52}┤")
            sections.append(f"  │ {'= Enterprise Value':<30} {valuation_bridge.enterprise_value:>15,.0f}   │")
            sections.append(f"  │ {'+ Cash & Equivalents':<30} {valuation_bridge.plus_cash:>15,.0f}   │")
            sections.append(f"  │ {'- Total Debt':<30} {valuation_bridge.minus_debt:>15,.0f}   │")
            sections.append(f"  ├{'─'*52}┤")
            sections.append(f"  │ {'= Equity Value':<30} {valuation_bridge.equity_value:>15,.0f}   │")
            sections.append(f"  │ {'÷ Shares Outstanding (M)':<30} {valuation_bridge.shares_outstanding:>15,.0f}   │")
            sections.append(f"  ├{'─'*52}┤")
            sections.append(f"  │ {'= INTRINSIC VALUE/SHARE':<30} ${valuation_bridge.intrinsic_value_per_share:>14.2f}   │")
            sections.append(f"  └{'─'*52}┘")

        # Section 4: Per Share Analysis
        sections.append(f"\n📍 PER SHARE ANALYSIS:")
        sections.append(f"  ┌{'─'*40}┐")
        sections.append(f"  │ Intrinsic Value:      ${result.intrinsic_value_per_share:>12.2f} │")
        sections.append(f"  │ Current Price:        ${result.current_price:>12.2f} │")
        if result.upside_potential is not None:
            upside_sign = "+" if result.upside_potential > 0 else ""
            sections.append(f"  │ Upside Potential:     {upside_sign}{result.upside_potential:>11.1f}% │")
        sections.append(f"  │ Margin of Safety:     ${result.margin_of_safety_price:>12.2f} │")
        sections.append(f"  └{'─'*40}┘")

        # Section 5: Reverse DCF (Implied Growth)
        if result.implied_growth_rate is not None:
            sections.append(f"\n🔄 REVERSE DCF (Market Expectations):")
            sections.append(f"  • Implied Perpetual Growth: {result.implied_growth_rate:.2%}")
            sections.append(f"  • Your Assumption: {terminal_growth:.2%}")
            diff = result.implied_growth_rate - terminal_growth
            if diff > 0.005:
                sections.append(f"  → Market expects HIGHER growth than your model")
            elif diff < -0.005:
                sections.append(f"  → Market expects LOWER growth than your model")
            else:
                sections.append(f"  → Market expectations align with your model")

        # Section 6: 2D Sensitivity Matrix
        if result.sensitivity_2d:
            sens = result.sensitivity_2d
            base_wacc_idx = sens.get("base_wacc_idx")
            base_tgr_idx = sens.get("base_tgr_idx")
            consistency = sens.get("consistency_check", True)

            sections.append(f"\n📐 SENSITIVITY MATRIX (WACC × Terminal Growth):")
            sections.append(f"  Intrinsic Value at different assumptions:")
            if not consistency:
                sections.append(f"  ⚠️ Warning: Matrix base case may not match calculated value")
            sections.append(f"")

            # Header row
            header = "  WACC \\ TGR │"
            for j, g in enumerate(sens.get("growth_rates", [])):
                # Mark base TGR column
                if j == base_tgr_idx:
                    header += f" [{g:>5}] │"
                else:
                    header += f" {g:>7} │"
            sections.append(header)
            sections.append(f"  {'─'*12}┼" + "─"*9*len(sens.get("growth_rates", [])))

            # Data rows
            wacc_rates = sens.get("wacc_rates", [])
            values = sens.get("values", [])
            for i, wacc in enumerate(wacc_rates):
                # Mark base WACC row
                if i == base_wacc_idx:
                    row = f" [{wacc:>9}] │"
                else:
                    row = f"  {wacc:>11} │"

                if i < len(values):
                    for j, v in enumerate(values[i]):
                        if v is not None:
                            # Highlight base case cell with [brackets]
                            if i == base_wacc_idx and j == base_tgr_idx:
                                row += f" [${v:>4.0f}] │"  # BASE CASE
                            # Highlight current price range with *
                            elif result.current_price and abs(v - result.current_price) / result.current_price < 0.1:
                                row += f" *{v:>5.0f}* │"
                            else:
                                row += f" ${v:>5.0f} │"
                        else:
                            row += f"    N/A │"
                sections.append(row)

            sections.append(f"")
            sections.append(f"  Legend:")
            sections.append(f"    [brackets] = Base case (your input assumptions)")
            sections.append(f"    *asterisk* = Values within 10% of current price (${result.current_price:.2f})")

        # Section 7: Validation Warnings
        if result.validation_warnings:
            sections.append(f"\n⚠️  VALIDATION WARNINGS:")
            for warning in result.validation_warnings:
                sections.append(f"  {warning}")

        # NEW Section 8: Risk Framework & Scenario Analysis
        if risk_framework:
            sections.append(f"\n🎯 RISK FRAMEWORK & SCENARIO ANALYSIS:")

            # Scenarios table
            sections.append(f"")
            sections.append(f"  {'Scenario':<10} {'Prob':<8} {'Value':<12} {'Upside':<12} {'Key Assumption':<25}")
            sections.append(f"  {'-'*70}")
            for scenario in risk_framework.scenarios:
                prob_str = f"{scenario.probability:.0%}" if scenario.probability else "N/A"
                upside_str = f"{scenario.upside_potential:+.1f}%" if scenario.upside_potential is not None else "N/A"
                key_assump = ""
                if scenario.key_assumptions:
                    if "fcf_growth" in scenario.key_assumptions:
                        key_assump = scenario.key_assumptions["fcf_growth"]
                sections.append(
                    f"  {scenario.name:<10} {prob_str:<8} ${scenario.intrinsic_value:>9,.2f}  {upside_str:<12} {key_assump:<25}"
                )

            # Summary metrics
            sections.append(f"")
            if risk_framework.probability_weighted_value:
                sections.append(f"  📊 Probability-Weighted Value: ${risk_framework.probability_weighted_value:,.2f}")
            if risk_framework.risk_reward_ratio:
                rr = risk_framework.risk_reward_ratio
                rr_str = f"{rr:.2f}:1" if rr < 100 else "Very High"
                sections.append(f"  📈 Risk/Reward Ratio: {rr_str}")
            if risk_framework.invalidation_price:
                sections.append(f"  🛑 Invalidation Price: ${risk_framework.invalidation_price:,.2f}")
            if risk_framework.position_size_suggestion:
                sections.append(f"  💼 Position Sizing: {risk_framework.position_size_suggestion}")

            # Quantitative triggers for each scenario
            sections.append(f"")
            sections.append(f"  📋 QUANTITATIVE TRIGGERS:")
            for scenario in risk_framework.scenarios:
                if scenario.quantitative_triggers:
                    sections.append(f"    {scenario.name}:")
                    for trigger in scenario.quantitative_triggers[:2]:  # Limit to 2 triggers
                        sections.append(f"      • {trigger}")

            # Key risks
            if risk_framework.key_risks:
                sections.append(f"")
                sections.append(f"  ⚠️ KEY RISKS TO MONITOR:")
                for risk in risk_framework.key_risks[:4]:
                    sections.append(f"    • {risk}")

        # Section 9: Final Verdict
        verdict_emoji = {
            "undervalued": "✅",
            "overvalued": "❌",
            "fairly_valued": "⚖️",
            "unknown": "❓",
        }
        emoji = verdict_emoji.get(result.verdict, "❓")
        sections.append(f"\n{'═'*53}")
        sections.append(f"VERDICT: {emoji} {result.verdict.upper()}")
        if risk_framework and risk_framework.probability_weighted_value:
            sections.append(f"Probability-Weighted Fair Value: ${risk_framework.probability_weighted_value:,.2f}")
        sections.append(f"{'═'*53}")

        return "\n".join(sections)


class CalculateGrahamTool(BaseTool):
    """Tool for Graham Formula valuation.

    Benjamin Graham's classic intrinsic value formula based on EPS and growth.

    Usage by agent:
        calculateGraham(symbol="AAPL", eps=6.15, growth_rate=0.10)
    """

    def __init__(self):
        super().__init__()
        self.calculator = GrahamCalculator()

        self.schema = ToolSchema(
            name="calculateGraham",
            category="valuation",
            description=(
                "Calculate intrinsic value using Benjamin Graham's formula: "
                "V = EPS × (8.5 + 2g) × (4.4/Y). Simple but effective for "
                "profitable, stable companies."
            ),
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock symbol",
                    required=True,
                ),
                ToolParameter(
                    name="eps",
                    type="number",
                    description="Earnings per share (trailing 12 months)",
                    required=False,
                ),
                ToolParameter(
                    name="growth_rate",
                    type="number",
                    description="Expected 5-year growth rate (e.g., 0.10 for 10%)",
                    required=False,
                    default=0.10,
                ),
                ToolParameter(
                    name="aaa_yield",
                    type="number",
                    description="AAA corporate bond yield (default 4.4%)",
                    required=False,
                    default=0.044,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute Graham valuation."""
        try:
            symbol = kwargs.get("symbol", "").upper()
            eps = kwargs.get("eps")
            growth_rate = kwargs.get("growth_rate", 0.10)
            aaa_yield = kwargs.get("aaa_yield", 0.044)

            if not symbol:
                return create_error_output(self.schema.name, "Symbol is required")

            # Fetch data if not provided
            fundamentals = await fetch_fundamental_data(symbol)

            if eps is None:
                eps = fundamentals.get("eps", 0)
                if eps <= 0:
                    return create_error_output(
                        self.schema.name,
                        f"Cannot use Graham formula: {symbol} has no positive EPS"
                    )

            current_price = fundamentals.get("price", 100)

            # Build input data
            data = GrahamInputData(
                symbol=symbol,
                eps=eps,
                growth_rate=growth_rate,
                current_price=current_price,
                aaa_yield=aaa_yield,
            )

            config = GrahamConfig(margin_of_safety=0.33)
            result = self.calculator.calculate(data, config)

            formatted = (
                f"Graham Formula Valuation for {symbol}:\n\n"
                f"Formula: V = EPS × (8.5 + 2g) × (4.4/Y)\n\n"
                f"Inputs:\n"
                f"  EPS: ${result.eps:.2f}\n"
                f"  Growth Rate (g): {result.growth_rate:.1%}\n"
                f"  AAA Bond Yield (Y): {result.aaa_yield:.1%}\n\n"
                f"Calculation:\n"
                f"  P/E Base (no growth): {result.pe_no_growth:.1f}\n"
                f"  Growth Premium (2g): {result.growth_premium:.1f}\n"
                f"  Bond Adjustment (4.4/Y): {result.bond_adjustment:.2f}\n"
                f"  Graham Value: ${result.graham_value:.2f}\n\n"
                f"Valuation:\n"
                f"  Intrinsic Value: ${result.intrinsic_value:.2f}\n"
                f"  Current Price: ${result.current_price:.2f}\n"
                f"  Upside Potential: {result.upside_potential:.1f}%\n"
                f"  Margin of Safety Price: ${result.margin_of_safety_price:.2f}\n\n"
                f"Verdict: {result.verdict.upper()}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "symbol": symbol,
                    "intrinsic_value": result.intrinsic_value,
                    "graham_value": result.graham_value,
                    "current_price": result.current_price,
                    "upside_potential": result.upside_potential,
                    "margin_of_safety_price": result.margin_of_safety_price,
                    "verdict": result.verdict,
                    "eps": eps,
                    "growth_rate": growth_rate,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Graham calculation failed: {e}")
            return create_error_output(self.schema.name, str(e))


class CalculateDDMTool(BaseTool):
    """Tool for DDM (Dividend Discount Model) valuation.

    Values dividend-paying stocks based on future dividend payments.

    Usage by agent:
        calculateDDM(symbol="MSFT", current_dividend=3.0, dividend_growth=0.10)
    """

    def __init__(self):
        super().__init__()
        self.calculator = DDMCalculator()

        self.schema = ToolSchema(
            name="calculateDDM",
            category="valuation",
            description=(
                "Calculate intrinsic value using Dividend Discount Model. "
                "Values stock as present value of all future dividends. "
                "Best for mature, dividend-paying companies."
            ),
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock symbol",
                    required=True,
                ),
                ToolParameter(
                    name="current_dividend",
                    type="number",
                    description="Current annual dividend per share",
                    required=False,
                ),
                ToolParameter(
                    name="dividend_growth",
                    type="number",
                    description="Expected dividend growth rate (e.g., 0.06 for 6%)",
                    required=False,
                    default=0.06,
                ),
                ToolParameter(
                    name="required_return",
                    type="number",
                    description="Required rate of return (e.g., 0.10 for 10%)",
                    required=False,
                    default=0.10,
                ),
                ToolParameter(
                    name="model_type",
                    type="string",
                    description="DDM model: gordon, two_stage, h_model, three_stage",
                    required=False,
                    default="gordon",
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute DDM valuation."""
        try:
            symbol = kwargs.get("symbol", "").upper()
            current_dividend = kwargs.get("current_dividend")
            dividend_growth = kwargs.get("dividend_growth", 0.06)
            required_return = kwargs.get("required_return", 0.10)
            model_type_str = kwargs.get("model_type", "gordon")

            if not symbol:
                return create_error_output(self.schema.name, "Symbol is required")

            # Fetch data if not provided
            fundamentals = await fetch_fundamental_data(symbol)

            if current_dividend is None:
                current_dividend = fundamentals.get("dividend_per_share", 0)
                if current_dividend <= 0:
                    return create_error_output(
                        self.schema.name,
                        f"Cannot use DDM: {symbol} does not pay dividends"
                    )

            current_price = fundamentals.get("price", 100)

            # Map model type
            model_map = {
                "gordon": DDMType.GORDON,
                "two_stage": DDMType.TWO_STAGE,
                "h_model": DDMType.H_MODEL,
                "three_stage": DDMType.THREE_STAGE,
            }
            model_type = model_map.get(model_type_str, DDMType.GORDON)

            # Validate growth < required return
            if dividend_growth >= required_return:
                return create_error_output(
                    self.schema.name,
                    f"Dividend growth ({dividend_growth:.1%}) must be less than required return ({required_return:.1%})"
                )

            # Build input data
            data = DDMInputData(
                symbol=symbol,
                current_dividend=current_dividend,
                dividend_growth=dividend_growth,
                required_return=required_return,
                current_price=current_price,
            )

            config = DDMConfig(model_type=model_type)
            result = self.calculator.calculate(data, config)

            model_desc = {
                DDMType.GORDON: "Gordon Growth (constant growth forever)",
                DDMType.TWO_STAGE: "Two-Stage (high growth then stable)",
                DDMType.H_MODEL: "H-Model (declining growth)",
                DDMType.THREE_STAGE: "Three-Stage (growth, transition, mature)",
            }

            formatted = (
                f"DDM Valuation for {symbol}:\n\n"
                f"Model: {model_desc.get(model_type, model_type_str)}\n\n"
                f"Inputs:\n"
                f"  Current Dividend: ${result.current_dividend:.2f}\n"
                f"  Expected Dividend (D1): ${result.expected_dividend:.2f}\n"
                f"  Dividend Growth: {result.dividend_growth:.1%}\n"
                f"  Required Return: {result.required_return:.1%}\n\n"
                f"Valuation:\n"
                f"  Intrinsic Value: ${result.intrinsic_value:.2f}\n"
                f"  Current Price: ${result.current_price:.2f}\n"
                f"  Dividend Yield: {result.dividend_yield:.2%}\n"
                f"  Upside Potential: {result.upside_potential:.1f}%\n\n"
                f"Verdict: {result.verdict.upper()}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "symbol": symbol,
                    "intrinsic_value": result.intrinsic_value,
                    "current_price": result.current_price,
                    "upside_potential": result.upside_potential,
                    "dividend_yield": result.dividend_yield,
                    "verdict": result.verdict,
                    "model_type": model_type_str,
                    "current_dividend": current_dividend,
                    "dividend_growth": dividend_growth,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"DDM calculation failed: {e}")
            return create_error_output(self.schema.name, str(e))


class GetValuationSummaryTool(BaseTool):
    """Tool for combined valuation from multiple methods.

    Runs multiple valuation methods and provides aggregate assessment.

    Usage by agent:
        getValuationSummary(symbol="AAPL")
    """

    def __init__(self):
        super().__init__()
        self.dcf_calc = DCFCalculator()
        self.graham_calc = GrahamCalculator()
        self.ddm_calc = DDMCalculator()
        self.summary_calc = ValuationCalculator()

        self.schema = ToolSchema(
            name="getValuationSummary",
            category="valuation",
            description=(
                "Comprehensive valuation using multiple methods (DCF, Graham, DDM). "
                "Provides average value, confidence level, and overall recommendation."
            ),
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock symbol",
                    required=True,
                ),
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute multi-method valuation."""
        try:
            symbol = kwargs.get("symbol", "").upper()

            if not symbol:
                return create_error_output(self.schema.name, "Symbol is required")

            # Fetch fundamental data
            fundamentals = await fetch_fundamental_data(symbol)
            current_price = fundamentals.get("price", 0)

            if current_price <= 0:
                return create_error_output(
                    self.schema.name,
                    f"Could not fetch price data for {symbol}"
                )

            dcf_result = None
            graham_result = None
            ddm_result = None

            # Try DCF
            fcf = fundamentals.get("free_cash_flow", 0)
            if fcf > 0:
                try:
                    data = DCFInputData(
                        symbol=symbol,
                        current_fcf=fcf,
                        growth_rates=[0.10, 0.08, 0.06, 0.05, 0.04],
                        terminal_growth=0.025,
                        discount_rate=0.10,
                        shares_outstanding=fundamentals.get("shares_outstanding", 1000),
                        current_price=current_price,
                        cash=fundamentals.get("cash", 0),
                        debt=fundamentals.get("debt", 0),
                    )
                    config = DCFConfig()
                    dcf_result = self.dcf_calc.calculate(data, config)
                except Exception as e:
                    logger.warning(f"DCF calculation failed: {e}")

            # Try Graham
            eps = fundamentals.get("eps", 0)
            if eps > 0:
                try:
                    data = GrahamInputData(
                        symbol=symbol,
                        eps=eps,
                        growth_rate=0.10,
                        current_price=current_price,
                    )
                    config = GrahamConfig()
                    graham_result = self.graham_calc.calculate(data, config)
                except Exception as e:
                    logger.warning(f"Graham calculation failed: {e}")

            # Try DDM
            dividend = fundamentals.get("dividend_per_share", 0)
            if dividend > 0:
                try:
                    data = DDMInputData(
                        symbol=symbol,
                        current_dividend=dividend,
                        dividend_growth=0.06,
                        required_return=0.10,
                        current_price=current_price,
                    )
                    config = DDMConfig()
                    ddm_result = self.ddm_calc.calculate(data, config)
                except Exception as e:
                    logger.warning(f"DDM calculation failed: {e}")

            # Create summary
            summary = self.summary_calc.calculate_summary(
                symbol=symbol,
                current_price=current_price,
                dcf_result=dcf_result,
                graham_result=graham_result,
                ddm_result=ddm_result,
            )

            # Format valuations
            val_lines = []
            for method, value in summary.valuations.items():
                diff = (value - current_price) / current_price * 100
                val_lines.append(f"  {method}: ${value:.2f} ({diff:+.1f}%)")

            recs = "\n".join(f"  • {r}" for r in summary.recommendations)

            formatted = (
                f"Valuation Summary for {symbol}:\n\n"
                f"Current Price: ${summary.current_price:.2f}\n\n"
                f"Intrinsic Values by Method:\n" + "\n".join(val_lines) + "\n\n"
                f"Aggregate Values:\n"
                f"  Average: ${summary.average_value:.2f}\n"
                f"  Median: ${summary.median_value:.2f}\n"
                f"  Range: ${summary.range[0]:.2f} - ${summary.range[1]:.2f}\n\n"
                f"Assessment:\n"
                f"  Verdict: {summary.overall_verdict.upper()}\n"
                f"  Confidence: {summary.confidence.upper()}\n\n"
                f"Recommendations:\n{recs}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "symbol": symbol,
                    "current_price": summary.current_price,
                    "valuations": summary.valuations,
                    "average_value": summary.average_value,
                    "median_value": summary.median_value,
                    "range": summary.range,
                    "overall_verdict": summary.overall_verdict,
                    "confidence": summary.confidence,
                    "recommendations": summary.recommendations,
                    "methods_used": [m.value for m in summary.methods_used],
                },
                formatted_context=formatted,
                execution_time_ms=summary.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Valuation summary failed: {e}")
            return create_error_output(self.schema.name, str(e))


class CalculateComparablesTool(BaseTool):
    """Tool for Comparable Company (Peer Multiples) valuation.

    Automatically fetches financial ratios for the target and all peer symbols,
    then calculates implied fair values using P/E, P/B, P/S, and EV/EBITDA multiples.

    This solves the problem of missing peer data ("-" values) by making actual
    API calls for each peer symbol instead of relying on the agent to manually
    extract and pass the data.

    Usage by agent:
        calculateComparables(
            symbol="NVDA",
            peers=["AMD", "INTC", "AVGO"]
        )
    """

    def __init__(self):
        super().__init__()
        self.calculator = ComparableCalculator()

        self.schema = ToolSchema(
            name="calculateComparables",
            category="valuation",
            description=(
                "Calculate comparable company (peer multiples) valuation. "
                "Automatically fetches P/E, P/B, P/S, EV/EBITDA ratios for the target "
                "and all peer companies, then computes implied fair values. "
                "Use for relative valuation and peer comparison analysis."
            ),
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Target stock symbol (e.g., 'NVDA')",
                    required=True,
                ),
                ToolParameter(
                    name="peers",
                    type="array",
                    description=(
                        "List of 2-5 peer company symbols from the same sector/industry "
                        "(e.g., ['AMD', 'INTC', 'AVGO'] for NVDA)"
                    ),
                    required=True,
                ),
            ],
        )

    async def _fetch_ratios_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch valuation ratios for a single symbol using FMP API directly.

        Returns dict with pe_ratio (TTM), pe_forward, pb_ratio, ps_ratio, ev_ebitda,
        along with pe_type indicator, or None on failure.
        """
        import os
        import httpx

        api_key = os.environ.get("FMP_API_KEY")
        if not api_key:
            logger.warning("[calculateComparables] FMP_API_KEY not set")
            return None

        base_url = "https://financialmodelingprep.com/api"
        sym = symbol.upper()

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                result = {
                    "symbol": sym,
                    "pe_ratio": None,
                    "pe_type": None,  # "TTM" or "Forward"
                    "pe_forward": None,
                    "pb_ratio": None,
                    "ps_ratio": None,
                    "ev_ebitda": None,
                    "price_to_cash_flow": None,
                }

                # Fetch TTM ratios (primary source for P/E TTM)
                url_ttm = f"{base_url}/v3/ratios-ttm/{sym}"
                resp_ttm = await client.get(url_ttm, params={"apikey": api_key})
                if resp_ttm.status_code == 200:
                    raw_ttm = resp_ttm.json()
                    if raw_ttm and len(raw_ttm) > 0:
                        item = raw_ttm[0]
                        result["pe_ratio"] = item.get("peRatioTTM")
                        result["pe_type"] = "TTM"
                        result["pb_ratio"] = item.get("priceToBookRatioTTM")
                        result["ps_ratio"] = item.get("priceToSalesRatioTTM")
                        result["ev_ebitda"] = item.get("enterpriseValueMultipleTTM")
                        result["price_to_cash_flow"] = item.get("priceCashFlowRatioTTM")

                # Try to get Forward P/E from key metrics or analyst estimates
                url_profile = f"{base_url}/v3/profile/{sym}"
                resp_profile = await client.get(url_profile, params={"apikey": api_key})
                if resp_profile.status_code == 200:
                    profile_data = resp_profile.json()
                    if profile_data and len(profile_data) > 0:
                        price = profile_data[0].get("price", 0)
                        # Try to get forward EPS estimate
                        url_estimates = f"{base_url}/v3/analyst-estimates/{sym}"
                        resp_est = await client.get(
                            url_estimates,
                            params={"apikey": api_key, "limit": 1}
                        )
                        if resp_est.status_code == 200:
                            est_data = resp_est.json()
                            if est_data and len(est_data) > 0:
                                fwd_eps = est_data[0].get("estimatedEpsAvg")
                                if fwd_eps and fwd_eps > 0 and price > 0:
                                    result["pe_forward"] = round(price / fwd_eps, 2)

                # Fallback: annual ratios if TTM not available
                if result["pe_ratio"] is None:
                    url = f"{base_url}/v3/ratios/{sym}"
                    resp = await client.get(url, params={"apikey": api_key, "period": "annual", "limit": 1})
                    if resp.status_code == 200:
                        raw_data = resp.json()
                        if raw_data and len(raw_data) > 0:
                            item = raw_data[0]
                            result["pe_ratio"] = item.get("priceEarningsRatio")
                            result["pe_type"] = "Annual"
                            if not result["pb_ratio"]:
                                result["pb_ratio"] = item.get("priceToBookRatio")
                            if not result["ps_ratio"]:
                                result["ps_ratio"] = item.get("priceToSalesRatio")
                            if not result["ev_ebitda"]:
                                result["ev_ebitda"] = item.get("enterpriseValueMultiple")

                # Return result if we have at least some data
                if any([result["pe_ratio"], result["pb_ratio"], result["ps_ratio"], result["ev_ebitda"]]):
                    return result
                return None

        except Exception as e:
            logger.warning(f"[calculateComparables] Failed to fetch ratios for {sym}: {e}")
            return None

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute comparable company valuation."""
        import asyncio

        try:
            symbol = kwargs.get("symbol", "").upper()
            peers_raw = kwargs.get("peers", [])

            if not symbol:
                return create_error_output(self.schema.name, "Symbol is required")
            if not peers_raw or len(peers_raw) < 2:
                return create_error_output(
                    self.schema.name,
                    "At least 2 peer symbols are required (e.g., peers=['AMD', 'INTC', 'AVGO'])"
                )

            peer_symbols = [p.upper() for p in peers_raw[:5]]  # Cap at 5 peers

            # Fetch fundamental data for target company
            fundamentals = await fetch_fundamental_data(symbol)
            current_price = fundamentals.get("price", 0)
            eps = fundamentals.get("eps", 0)
            book_value_ps = fundamentals.get("book_value_per_share", 0)
            revenue_ps = fundamentals.get("revenue_per_share", 0)

            if current_price <= 0:
                return create_error_output(
                    self.schema.name,
                    f"Could not fetch price data for {symbol}"
                )

            # Fetch ratios for target + all peers in parallel
            fetch_tasks = [self._fetch_ratios_for_symbol(symbol)]
            for peer in peer_symbols:
                fetch_tasks.append(self._fetch_ratios_for_symbol(peer))

            results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

            # Process target ratios
            target_ratios = results[0] if not isinstance(results[0], Exception) else None

            # Process peer ratios
            peer_companies = []
            peer_details = []  # For formatted output
            pe_types_used = set()  # Track what P/E types we're using

            for i, peer_sym in enumerate(peer_symbols):
                peer_result = results[i + 1]
                if isinstance(peer_result, Exception) or peer_result is None:
                    logger.warning(f"[calculateComparables] No data for peer {peer_sym}")
                    peer_details.append({
                        "symbol": peer_sym,
                        "pe_ratio": None,
                        "pe_type": None,
                        "pe_forward": None,
                        "pb_ratio": None,
                        "ps_ratio": None,
                        "ev_ebitda": None,
                        "status": "no_data",
                    })
                    continue

                pe = peer_result.get("pe_ratio")
                pe_type = peer_result.get("pe_type", "TTM")
                pe_forward = peer_result.get("pe_forward")
                pb = peer_result.get("pb_ratio")
                ps = peer_result.get("ps_ratio")
                ev = peer_result.get("ev_ebitda")

                # Filter out negative or zero values (e.g., negative P/E means losses)
                if pe is not None and pe <= 0:
                    pe = None
                if pe_forward is not None and pe_forward <= 0:
                    pe_forward = None
                if pb is not None and pb <= 0:
                    pb = None
                if ps is not None and ps <= 0:
                    ps = None
                if ev is not None and ev <= 0:
                    ev = None

                if pe_type:
                    pe_types_used.add(pe_type)

                peer_companies.append(ComparableCompany(
                    symbol=peer_sym,
                    pe_ratio=pe,
                    pb_ratio=pb,
                    ps_ratio=ps,
                    ev_ebitda=ev,
                ))

                peer_details.append({
                    "symbol": peer_sym,
                    "pe_ratio": round(pe, 2) if pe else None,
                    "pe_type": pe_type,
                    "pe_forward": round(pe_forward, 2) if pe_forward else None,
                    "pb_ratio": round(pb, 2) if pb else None,
                    "ps_ratio": round(ps, 2) if ps else None,
                    "ev_ebitda": round(ev, 2) if ev else None,
                    "status": "ok",
                })

            if len(peer_companies) < 2:
                return create_error_output(
                    self.schema.name,
                    f"Could not fetch sufficient peer data. Only {len(peer_companies)} peers had valid data. "
                    f"Need at least 2. Check if peer symbols are valid."
                )

            # Build ComparableInputData
            # Estimate EBITDA per share from target ratios if available
            ebitda_per_share = None
            if target_ratios and target_ratios.get("ev_ebitda"):
                # Reverse-calculate: if we have EV/EBITDA and price, approximate EBITDA/share
                # This is simplified; proper calc would use enterprise value
                ev_multiple = target_ratios["ev_ebitda"]
                if ev_multiple > 0:
                    ebitda_per_share = current_price / ev_multiple

            comp_data = ComparableInputData(
                symbol=symbol,
                eps=eps if eps > 0 else 0.01,  # Avoid zero
                book_value_per_share=book_value_ps if book_value_ps > 0 else 0.01,
                revenue_per_share=revenue_ps if revenue_ps > 0 else 0.01,
                ebitda_per_share=ebitda_per_share,
                current_price=current_price,
                peers=peer_companies,
            )

            # Run calculation
            result = self.calculator.calculate(comp_data)

            # Build formatted output with actual values
            # Determine P/E type label
            pe_type_label = "TTM"
            if pe_types_used:
                pe_type_label = list(pe_types_used)[0] if len(pe_types_used) == 1 else "Mixed"

            # Target ratios display
            t_pe = f"{target_ratios['pe_ratio']:.2f}x" if target_ratios and target_ratios.get('pe_ratio') else "N/A"
            t_pe_fwd = f"{target_ratios['pe_forward']:.2f}x" if target_ratios and target_ratios.get('pe_forward') else "N/A"
            t_pb = f"{target_ratios['pb_ratio']:.2f}x" if target_ratios and target_ratios.get('pb_ratio') else "N/A"
            t_ps = f"{target_ratios['ps_ratio']:.2f}x" if target_ratios and target_ratios.get('ps_ratio') else "N/A"
            t_ev = f"{target_ratios['ev_ebitda']:.2f}x" if target_ratios and target_ratios.get('ev_ebitda') else "N/A"

            # Check if we have any forward P/E data
            has_forward_pe = any(
                pd.get('pe_forward') is not None
                for pd in peer_details
            ) or (target_ratios and target_ratios.get('pe_forward'))

            # Build peer table rows
            table_rows = []
            if has_forward_pe:
                table_rows.append(
                    f"| **{symbol}** (Target) | ${current_price:.2f} | {t_pe} | {t_pe_fwd} | {t_pb} | {t_ps} | {t_ev} |"
                )
            else:
                table_rows.append(
                    f"| **{symbol}** (Target) | ${current_price:.2f} | {t_pe} | {t_pb} | {t_ps} | {t_ev} | - |"
                )

            for pd_item in peer_details:
                s = pd_item["symbol"]
                pe_str = f"{pd_item['pe_ratio']:.2f}x" if pd_item.get('pe_ratio') else "N/A"
                pe_fwd_str = f"{pd_item['pe_forward']:.2f}x" if pd_item.get('pe_forward') else "N/A"
                pb_str = f"{pd_item['pb_ratio']:.2f}x" if pd_item.get('pb_ratio') else "N/A"
                ps_str = f"{pd_item['ps_ratio']:.2f}x" if pd_item.get('ps_ratio') else "N/A"
                ev_str = f"{pd_item['ev_ebitda']:.2f}x" if pd_item.get('ev_ebitda') else "N/A"
                status = pd_item.get("status", "")
                note = " (no data)" if status == "no_data" else ""

                if has_forward_pe:
                    table_rows.append(
                        f"| {s}{note} | - | {pe_str} | {pe_fwd_str} | {pb_str} | {ps_str} | {ev_str} |"
                    )
                else:
                    table_rows.append(
                        f"| {s}{note} | - | {pe_str} | {pb_str} | {ps_str} | {ev_str} | - |"
                    )

            # Add peer average/median row
            stats = result.peer_stats
            avg_pe = f"{stats['P/E']['avg']:.2f}x" if 'P/E' in stats else "N/A"
            avg_pb = f"{stats['P/B']['avg']:.2f}x" if 'P/B' in stats else "N/A"
            avg_ps = f"{stats['P/S']['avg']:.2f}x" if 'P/S' in stats else "N/A"
            avg_ev = f"{stats['EV/EBITDA']['avg']:.2f}x" if 'EV/EBITDA' in stats else "N/A"

            if has_forward_pe:
                table_rows.append(
                    f"| **Peer Average** | - | {avg_pe} | - | {avg_pb} | {avg_ps} | {avg_ev} |"
                )
                table_header = (
                    f"| Company | Price | P/E ({pe_type_label}) | P/E (Fwd) | P/B | P/S | EV/EBITDA |\n"
                    "|---------|-------|----------|-----------|-----|-----|----------|\n"
                )
            else:
                table_rows.append(
                    f"| **Peer Average** | - | {avg_pe} | {avg_pb} | {avg_ps} | {avg_ev} | - |"
                )
                table_header = (
                    f"| Company | Price | P/E ({pe_type_label}) | P/B | P/S | EV/EBITDA | Note |\n"
                    "|---------|-------|----------|-----|-----|-----------|------|\n"
                )

            table_body = "\n".join(table_rows)

            # Implied values section
            implied_lines = []
            for v in result.valuations:
                multiple_label = v.multiple_name
                if v.multiple_name == "P/E":
                    multiple_label = f"P/E ({pe_type_label})"
                implied_lines.append(
                    f"  {multiple_label}: Peer Avg={v.peer_average:.2f}x → "
                    f"Implied Price=${v.implied_value:.2f} "
                    f"(Median={v.peer_median:.2f}x → ${v.implied_value_median:.2f})"
                )
            implied_text = "\n".join(implied_lines) if implied_lines else "  No valid multiples available"

            # P/E type note
            pe_note = ""
            if pe_type_label == "TTM":
                pe_note = "\n📝 Note: P/E ratios are Trailing Twelve Months (TTM) based on historical earnings."
            elif pe_type_label == "Forward":
                pe_note = "\n📝 Note: P/E ratios are Forward P/E based on analyst earnings estimates."
            elif pe_type_label == "Mixed":
                pe_note = "\n📝 Note: P/E ratios may be mixed (TTM/Forward). Check individual sources."

            formatted = (
                f"═══════════════════════════════════════════════════════\n"
                f"COMPARABLE COMPANY VALUATION: {symbol}\n"
                f"═══════════════════════════════════════════════════════\n\n"
                f"📊 Peer Comparison Table:\n"
                f"{table_header}{table_body}\n"
                f"{pe_note}\n\n"
                f"💰 Implied Fair Values (peer multiples × {symbol} metrics):\n"
                f"{implied_text}\n\n"
                f"📍 Summary:\n"
                f"  ┌{'─'*44}┐\n"
                f"  │ Average Fair Value:     ${result.average_intrinsic_value:>14.2f} │\n"
                f"  │ Median Fair Value:      ${result.median_intrinsic_value:>14.2f} │\n"
                f"  │ Current Price:          ${result.current_price:>14.2f} │\n"
                f"  │ Upside Potential:       {result.upside_potential:>13.1f}% │\n"
                f"  └{'─'*44}┘\n\n"
                f"{'═'*57}\n"
                f"VERDICT: {result.verdict.upper()}\n"
                f"{'═'*57}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "symbol": symbol,
                    "current_price": current_price,
                    "pe_type": pe_type_label,
                    "target_ratios": {
                        "pe_ratio": target_ratios.get("pe_ratio") if target_ratios else None,
                        "pe_type": target_ratios.get("pe_type") if target_ratios else None,
                        "pe_forward": target_ratios.get("pe_forward") if target_ratios else None,
                        "pb_ratio": target_ratios.get("pb_ratio") if target_ratios else None,
                        "ps_ratio": target_ratios.get("ps_ratio") if target_ratios else None,
                        "ev_ebitda": target_ratios.get("ev_ebitda") if target_ratios else None,
                    },
                    "peers": peer_details,
                    "peer_stats": result.peer_stats,
                    "valuations": [
                        {
                            "multiple": v.multiple_name,
                            "peer_average": v.peer_average,
                            "peer_median": v.peer_median,
                            "implied_value_avg": v.implied_value,
                            "implied_value_median": v.implied_value_median,
                        }
                        for v in result.valuations
                    ],
                    "average_intrinsic_value": result.average_intrinsic_value,
                    "median_intrinsic_value": result.median_intrinsic_value,
                    "upside_potential": result.upside_potential,
                    "verdict": result.verdict,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"Comparable calculation failed: {e}")
            return create_error_output(self.schema.name, str(e))
