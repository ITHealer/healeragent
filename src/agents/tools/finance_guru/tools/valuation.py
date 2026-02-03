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
) -> Dict[str, Any]:
    """
    Fetch fundamental data from FMP API for valuation.

    Args:
        symbol: Stock symbol
        fmp_api_key: FMP API key

    Returns:
        Dict with fundamental metrics
    """
    try:
        from src.agents.tools.finance_guru.services.fmp_service import FMPService

        fmp = FMPService(api_key=fmp_api_key)

        # Fetch key metrics and profile
        profile = await fmp.get_company_profile(symbol)
        key_metrics = await fmp.get_key_metrics(symbol)
        income = await fmp.get_income_statement(symbol)
        cash_flow = await fmp.get_cash_flow_statement(symbol)

        if not profile:
            raise ValueError(f"Could not fetch data for {symbol}")

        # Resolve shares_outstanding with multiple fallback sources
        shares_outstanding_raw = profile.get("sharesOutstanding") or 0
        if not shares_outstanding_raw and income:
            # Fallback 1: weighted average shares from income statement
            shares_outstanding_raw = (
                income.get("weightedAverageShsOut")
                or income.get("weightedAverageShsOutDil")
                or 0
            )
        if not shares_outstanding_raw:
            # Fallback 2: calculate from market cap / price
            mkt_cap = profile.get("mktCap") or profile.get("marketCap") or 0
            price = profile.get("price") or 0
            if mkt_cap > 0 and price > 0:
                shares_outstanding_raw = mkt_cap / price

        shares_outstanding_m = shares_outstanding_raw / 1e6 if shares_outstanding_raw else 0

        return {
            "symbol": symbol,
            "price": profile.get("price", 0),
            "eps": key_metrics.get("eps", 0) if key_metrics else 0,
            "book_value_per_share": key_metrics.get("bookValuePerShare", 0) if key_metrics else 0,
            "revenue_per_share": key_metrics.get("revenuePerShare", 0) if key_metrics else 0,
            "dividend_per_share": key_metrics.get("dividendPerShare", 0) if key_metrics else 0,
            "shares_outstanding": shares_outstanding_m,
            "free_cash_flow": cash_flow.get("freeCashFlow", 0) / 1e6 if cash_flow else 0,
            "cash": profile.get("cash", 0) / 1e6,
            "debt": profile.get("totalDebt", 0) / 1e6,
        }

    except ImportError:
        logger.warning("FMPService not available, using mock data")
        return _create_mock_fundamental_data(symbol)


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
            ],
        )

    async def execute(self, **kwargs) -> ToolOutput:
        """Execute DCF valuation."""
        try:
            symbol = kwargs.get("symbol", "").upper()
            current_fcf = kwargs.get("current_fcf")
            growth_rates = kwargs.get("growth_rates", [0.10, 0.08, 0.06, 0.05, 0.04])
            discount_rate = kwargs.get("discount_rate", 0.10)
            terminal_growth = kwargs.get("terminal_growth", 0.025)

            if not symbol:
                return create_error_output(self.schema.name, "Symbol is required")

            # Fetch fundamental data if not provided
            fundamentals = await fetch_fundamental_data(symbol)

            if current_fcf is None:
                current_fcf = fundamentals.get("free_cash_flow", 0)
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
            result = self.calculator.calculate(data, config)

            # Format projections
            proj_text = "\n".join(
                f"  Year {p.year}: FCF ${p.fcf:,.0f}M (growth: {p.growth_rate:.1%}) → PV ${p.present_value:,.0f}M"
                for p in result.projections[:3]
            )

            formatted = (
                f"DCF Valuation for {symbol}:\n\n"
                f"Cash Flow Projections:\n{proj_text}\n  ...\n\n"
                f"Valuation Summary:\n"
                f"  Sum of PV (FCFs): ${result.sum_of_pv_fcf:,.0f}M\n"
                f"  Terminal Value: ${result.terminal_value:,.0f}M\n"
                f"  PV of Terminal: ${result.pv_terminal_value:,.0f}M\n"
                f"  Enterprise Value: ${result.enterprise_value:,.0f}M\n"
                f"  Equity Value: ${result.equity_value:,.0f}M\n\n"
                f"Per Share:\n"
                f"  Intrinsic Value: ${result.intrinsic_value_per_share:.2f}\n"
                f"  Current Price: ${result.current_price:.2f}\n"
                f"  Upside Potential: {result.upside_potential:.1f}%\n"
                f"  Margin of Safety Price: ${result.margin_of_safety_price:.2f}\n\n"
                f"Verdict: {result.verdict.upper()}"
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
                    "verdict": result.verdict,
                    "discount_rate": discount_rate,
                    "terminal_growth": terminal_growth,
                },
                formatted_context=formatted,
                execution_time_ms=result.calculation_time_ms or 0,
            )

        except Exception as e:
            logger.exception(f"DCF calculation failed: {e}")
            return create_error_output(self.schema.name, str(e))


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

        Returns dict with pe_ratio, pb_ratio, ps_ratio, ev_ebitda, or None on failure.
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
                # Primary: annual ratios (latest period)
                url = f"{base_url}/v3/ratios/{sym}"
                resp = await client.get(url, params={"apikey": api_key, "period": "annual", "limit": 1})
                resp.raise_for_status()
                raw_data = resp.json()

                if raw_data and len(raw_data) > 0:
                    item = raw_data[0]
                    return {
                        "symbol": sym,
                        "pe_ratio": item.get("priceEarningsRatio"),
                        "pb_ratio": item.get("priceToBookRatio"),
                        "ps_ratio": item.get("priceToSalesRatio"),
                        "ev_ebitda": item.get("enterpriseValueMultiple"),
                        "price_to_cash_flow": item.get("priceCashFlowRatio"),
                    }

                # Fallback: TTM ratios
                url_ttm = f"{base_url}/v3/ratios-ttm/{sym}"
                resp_ttm = await client.get(url_ttm, params={"apikey": api_key})
                resp_ttm.raise_for_status()
                raw_ttm = resp_ttm.json()

                if raw_ttm and len(raw_ttm) > 0:
                    item = raw_ttm[0]
                    return {
                        "symbol": sym,
                        "pe_ratio": item.get("peRatioTTM"),
                        "pb_ratio": item.get("priceToBookRatioTTM"),
                        "ps_ratio": item.get("priceToSalesRatioTTM"),
                        "ev_ebitda": item.get("enterpriseValueMultipleTTM"),
                        "price_to_cash_flow": item.get("priceCashFlowRatioTTM"),
                    }

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
            for i, peer_sym in enumerate(peer_symbols):
                peer_result = results[i + 1]
                if isinstance(peer_result, Exception) or peer_result is None:
                    logger.warning(f"[calculateComparables] No data for peer {peer_sym}")
                    peer_details.append({
                        "symbol": peer_sym,
                        "pe_ratio": None,
                        "pb_ratio": None,
                        "ps_ratio": None,
                        "ev_ebitda": None,
                        "status": "no_data",
                    })
                    continue

                pe = peer_result.get("pe_ratio")
                pb = peer_result.get("pb_ratio")
                ps = peer_result.get("ps_ratio")
                ev = peer_result.get("ev_ebitda")

                # Filter out negative or zero values (e.g., negative P/E means losses)
                if pe is not None and pe <= 0:
                    pe = None
                if pb is not None and pb <= 0:
                    pb = None
                if ps is not None and ps <= 0:
                    ps = None
                if ev is not None and ev <= 0:
                    ev = None

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
            # Target ratios display
            t_pe = f"{target_ratios['pe_ratio']:.2f}x" if target_ratios and target_ratios.get('pe_ratio') else "N/A"
            t_pb = f"{target_ratios['pb_ratio']:.2f}x" if target_ratios and target_ratios.get('pb_ratio') else "N/A"
            t_ps = f"{target_ratios['ps_ratio']:.2f}x" if target_ratios and target_ratios.get('ps_ratio') else "N/A"
            t_ev = f"{target_ratios['ev_ebitda']:.2f}x" if target_ratios and target_ratios.get('ev_ebitda') else "N/A"

            # Build peer table rows
            table_rows = []
            table_rows.append(
                f"| **{symbol}** (Target) | ${current_price:.2f} | {t_pe} | {t_pb} | {t_ps} | {t_ev} | - |"
            )

            for pd_item in peer_details:
                s = pd_item["symbol"]
                pe_str = f"{pd_item['pe_ratio']:.2f}x" if pd_item.get('pe_ratio') else "N/A"
                pb_str = f"{pd_item['pb_ratio']:.2f}x" if pd_item.get('pb_ratio') else "N/A"
                ps_str = f"{pd_item['ps_ratio']:.2f}x" if pd_item.get('ps_ratio') else "N/A"
                ev_str = f"{pd_item['ev_ebitda']:.2f}x" if pd_item.get('ev_ebitda') else "N/A"
                status = pd_item.get("status", "")
                note = "" if status == "ok" else " (no data)"
                table_rows.append(
                    f"| {s}{note} | - | {pe_str} | {pb_str} | {ps_str} | {ev_str} | - |"
                )

            # Add peer average/median row
            stats = result.peer_stats
            avg_pe = f"{stats['P/E']['avg']:.2f}x" if 'P/E' in stats else "N/A"
            avg_pb = f"{stats['P/B']['avg']:.2f}x" if 'P/B' in stats else "N/A"
            avg_ps = f"{stats['P/S']['avg']:.2f}x" if 'P/S' in stats else "N/A"
            avg_ev = f"{stats['EV/EBITDA']['avg']:.2f}x" if 'EV/EBITDA' in stats else "N/A"
            table_rows.append(
                f"| **Peer Average** | - | {avg_pe} | {avg_pb} | {avg_ps} | {avg_ev} | - |"
            )

            table_header = (
                "| Company | Price | P/E | P/B | P/S | EV/EBITDA | Note |\n"
                "|---------|-------|-----|-----|-----|-----------|------|\n"
            )
            table_body = "\n".join(table_rows)

            # Implied values section
            implied_lines = []
            for v in result.valuations:
                implied_lines.append(
                    f"  {v.multiple_name}: Peer Avg={v.peer_average:.2f}x → "
                    f"Implied Price=${v.implied_value:.2f} "
                    f"(Median={v.peer_median:.2f}x → ${v.implied_value_median:.2f})"
                )
            implied_text = "\n".join(implied_lines) if implied_lines else "  No valid multiples available"

            formatted = (
                f"Comparable Company Valuation for {symbol}:\n\n"
                f"Peer Comparison Table:\n"
                f"{table_header}{table_body}\n\n"
                f"Implied Fair Values (using peer multiples × {symbol} metrics):\n"
                f"{implied_text}\n\n"
                f"Summary:\n"
                f"  Average Implied Fair Value: ${result.average_intrinsic_value:.2f}\n"
                f"  Median Implied Fair Value: ${result.median_intrinsic_value:.2f}\n"
                f"  Current Price: ${result.current_price:.2f}\n"
                f"  Upside Potential: {result.upside_potential:.1f}%\n\n"
                f"Verdict: {result.verdict.upper()}"
            )

            return create_success_output(
                self.schema.name,
                data={
                    "symbol": symbol,
                    "current_price": current_price,
                    "target_ratios": {
                        "pe_ratio": target_ratios.get("pe_ratio") if target_ratios else None,
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
