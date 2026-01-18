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
    """Create an error ToolOutput."""
    return ToolOutput(
        tool_name=tool_name,
        status="error",
        error=error,
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
        from src.services.fmp_service import FMPService

        fmp = FMPService(api_key=fmp_api_key)

        # Fetch key metrics and profile
        profile = await fmp.get_company_profile(symbol)
        key_metrics = await fmp.get_key_metrics(symbol)
        income = await fmp.get_income_statement(symbol)
        cash_flow = await fmp.get_cash_flow_statement(symbol)

        if not profile:
            raise ValueError(f"Could not fetch data for {symbol}")

        return {
            "symbol": symbol,
            "price": profile.get("price", 0),
            "eps": key_metrics.get("eps", 0) if key_metrics else 0,
            "book_value_per_share": key_metrics.get("bookValuePerShare", 0) if key_metrics else 0,
            "revenue_per_share": key_metrics.get("revenuePerShare", 0) if key_metrics else 0,
            "dividend_per_share": key_metrics.get("dividendPerShare", 0) if key_metrics else 0,
            "shares_outstanding": profile.get("sharesOutstanding", 0) / 1e6,  # Convert to millions
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

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
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

            shares_outstanding = fundamentals.get("shares_outstanding", 1000)
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

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
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

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
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

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
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
