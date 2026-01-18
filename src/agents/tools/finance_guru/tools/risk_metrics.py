"""
Finance Guru - Risk Metrics Tools

Layer 3 of the 3-layer architecture: Agent-callable tool interfaces.

These tools provide the agent-facing interface for risk metrics:
- GetRiskMetricsTool (comprehensive risk analysis)
- GetVaRTool (Value at Risk)
- GetSharpeRatioTool (Sharpe Ratio)
- GetMaxDrawdownTool (Maximum Drawdown)
- GetBetaAlphaTool (Beta/Alpha vs benchmark)

WHAT: Agent-callable tools for risk analysis
WHY: Provides standardized interface for LLM agents to access risk metrics
ARCHITECTURE: Layer 3 of 3-layer type-safe architecture

Author: HealerAgent Development Team
Created: 2025-01-18
"""

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from src.agents.tools.base import (
    BaseTool,
    ToolOutput,
    ToolParameter,
    ToolSchema,
)
from src.agents.tools.finance_guru.calculators.risk_metrics import (
    RiskMetricsCalculator,
)
from src.agents.tools.finance_guru.models.risk_metrics import (
    RiskDataInput,
    BenchmarkDataInput,
    RiskCalculationConfig,
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


async def fetch_price_data(
    symbol: str,
    days: int = 252,
    fmp_api_key: Optional[str] = None,
) -> RiskDataInput:
    """
    Fetch price data from FMP API.

    Args:
        symbol: Stock symbol
        days: Number of days of data
        fmp_api_key: FMP API key

    Returns:
        RiskDataInput with price data
    """
    try:
        from src.services.fmp_service import FMPService

        fmp = FMPService(api_key=fmp_api_key)
        historical = await fmp.get_historical_price(symbol, days=days)

        if not historical or len(historical) < 30:
            raise ValueError(f"Insufficient data for {symbol}: need 30+ days")

        # Reverse for chronological order
        historical = list(reversed(historical))

        return RiskDataInput(
            ticker=symbol.upper(),
            dates=[date.fromisoformat(d["date"]) for d in historical],
            prices=[float(d["close"]) for d in historical],
            volumes=[float(d.get("volume", 0)) for d in historical],
        )

    except ImportError:
        logger.warning("FMPService not available, using mock data")
        return _create_mock_price_data(symbol, days)


def _create_mock_price_data(symbol: str, days: int = 252) -> RiskDataInput:
    """Create mock price data for testing."""
    import random

    base_price = 150.0
    dates = []
    prices = []
    volumes = []

    today = date.today()
    for i in range(days):
        d = today - timedelta(days=days - i - 1)
        if d.weekday() < 5:
            dates.append(d)
            change = random.uniform(-0.03, 0.035)
            base_price = base_price * (1 + change)
            prices.append(round(base_price, 2))
            volumes.append(random.randint(500000, 2000000))

    return RiskDataInput(
        ticker=symbol.upper(),
        dates=dates,
        prices=prices,
        volumes=volumes,
    )


# =============================================================================
# COMPREHENSIVE RISK METRICS TOOL
# =============================================================================

class GetRiskMetricsTool(BaseTool):
    """
    Tool to calculate comprehensive risk metrics.

    Includes: VaR, CVaR, Sharpe, Sortino, Calmar, Max Drawdown,
    Beta/Alpha (with benchmark), and risk score.
    """

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key

        self.schema = ToolSchema(
            name="getRiskMetrics",
            category="risk_enhanced",
            description="Calculate comprehensive risk metrics including VaR, Sharpe, Sortino, Max Drawdown, and more",
            capabilities=[
                "✅ Value at Risk (VaR) - Historical & Parametric",
                "✅ Conditional VaR (Expected Shortfall)",
                "✅ Sharpe Ratio - Risk-adjusted return",
                "✅ Sortino Ratio - Downside risk-adjusted",
                "✅ Calmar Ratio - Return per max drawdown",
                "✅ Maximum Drawdown with recovery analysis",
                "✅ Omega Ratio - Probability-weighted gains/losses",
                "✅ Beta/Alpha vs benchmark (optional)",
                "✅ Treynor & Information Ratios (with benchmark)",
                "✅ Overall risk score (0-100)",
            ],
            limitations=[
                "❌ Historical metrics only (no forward predictions)",
                "❌ Needs minimum 30 days of data",
                "❌ VaR assumes past patterns continue",
            ],
            usage_hints=[
                "Use for comprehensive portfolio risk assessment",
                "Provide benchmark symbol for relative metrics",
                "Check risk score for quick risk classification",
                "Review max drawdown for worst-case scenario",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                    pattern=r"^[A-Z]{1,10}$",
                ),
                ToolParameter(
                    name="benchmark_symbol",
                    type="string",
                    description="Benchmark symbol for Beta/Alpha (e.g., 'SPY')",
                    required=False,
                    default=None,
                ),
                ToolParameter(
                    name="days",
                    type="integer",
                    description="Historical days to analyze (default: 252 = 1 year)",
                    required=False,
                    default=252,
                    min_value=30,
                    max_value=1260,
                ),
                ToolParameter(
                    name="confidence_level",
                    type="number",
                    description="VaR confidence level (default: 0.95)",
                    required=False,
                    default=0.95,
                    min_value=0.90,
                    max_value=0.99,
                ),
                ToolParameter(
                    name="portfolio_value",
                    type="number",
                    description="Portfolio value for dollar VaR (optional)",
                    required=False,
                    default=None,
                ),
            ],
            returns={
                "ticker": "string",
                "risk_score": "number (0-100)",
                "risk_level": "string (conservative/moderate/aggressive/speculative)",
                "var": "VaR analysis object",
                "cvar": "CVaR analysis object",
                "sharpe_ratio": "Sharpe analysis object",
                "sortino_ratio": "Sortino analysis object",
                "max_drawdown": "Drawdown analysis object",
                "beta_alpha": "Beta/Alpha object (if benchmark provided)",
                "summary": "Human-readable summary",
            },
            typical_execution_time_ms=3000,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute comprehensive risk metrics calculation."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]
            days = validated.get("days", 252)
            benchmark_symbol = validated.get("benchmark_symbol")
            confidence = validated.get("confidence_level", 0.95)
            portfolio_value = validated.get("portfolio_value")

            # Fetch data
            data = await fetch_price_data(symbol, days, self.fmp_api_key)

            # Fetch benchmark if provided
            benchmark = None
            if benchmark_symbol:
                benchmark_data = await fetch_price_data(benchmark_symbol, days, self.fmp_api_key)
                benchmark = BenchmarkDataInput(
                    ticker=benchmark_data.ticker,
                    dates=benchmark_data.dates,
                    prices=benchmark_data.prices,
                )

            # Configure calculator
            config = RiskCalculationConfig(
                confidence_level=confidence,
                var_method="historical",
            )
            calculator = RiskMetricsCalculator(config)

            # Calculate
            result = calculator.safe_calculate(
                data,
                benchmark=benchmark,
                portfolio_value=portfolio_value,
            )

            result_dict = result.to_dict()

            # Format context
            formatted = f"""
**Risk Metrics for {symbol}**

📊 **Risk Score: {result.risk_score}/100 ({result.risk_level.upper()})**

**Value at Risk (95%):**
- VaR: {result.var.var_percent:.2f}%
- CVaR (Expected Shortfall): {result.cvar.cvar_percent:.2f}%

**Risk-Adjusted Returns:**
- Sharpe Ratio: {result.sharpe_ratio.sharpe_ratio:.2f} ({result.sharpe_ratio.quality})
- Sortino Ratio: {result.sortino_ratio.sortino_ratio:.2f} ({result.sortino_ratio.quality})
- Calmar Ratio: {result.calmar_ratio.calmar_ratio:.2f}
- Omega Ratio: {result.omega_ratio.omega_ratio:.2f}

**Drawdown Analysis:**
- Max Drawdown: {result.max_drawdown.max_drawdown:.1%}
- Current Drawdown: {result.max_drawdown.current_drawdown:.1%}
- Recovery: {'Yes' if result.max_drawdown.recovery_date else 'Not yet'}

**Volatility:**
- Annual: {result.volatility.annual_volatility:.1%}
- Regime: {result.volatility.volatility_regime.upper()}
"""
            if result.beta_alpha:
                formatted += f"""
**Benchmark Relative ({result.beta_alpha.benchmark_ticker}):**
- Beta: {result.beta_alpha.beta:.2f}
- Alpha: {result.beta_alpha.alpha:.1%}
- Correlation: {result.beta_alpha.correlation:.2f}
"""

            formatted += f"\n**Summary:** {result.summary}"

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            logger.error(f"Risk metrics calculation failed: {e}")
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


# =============================================================================
# INDIVIDUAL METRIC TOOLS
# =============================================================================

class GetVaRTool(BaseTool):
    """Tool to calculate Value at Risk."""

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key

        self.schema = ToolSchema(
            name="getVaR",
            category="risk_enhanced",
            description="Calculate Value at Risk (VaR) and Conditional VaR (CVaR)",
            capabilities=[
                "✅ Historical VaR calculation",
                "✅ Parametric VaR (normal distribution)",
                "✅ CVaR (Expected Shortfall)",
                "✅ Dollar VaR with portfolio value",
            ],
            limitations=[
                "❌ Assumes past patterns predict future risk",
                "❌ Parametric method assumes normality",
            ],
            usage_hints=[
                "Use historical method for fat-tailed assets",
                "95% VaR means 1-in-20 days may exceed this loss",
                "CVaR shows average loss when VaR is breached",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                ),
                ToolParameter(
                    name="confidence_level",
                    type="number",
                    description="Confidence level (default: 0.95)",
                    required=False,
                    default=0.95,
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="VaR method: 'historical' or 'parametric'",
                    required=False,
                    default="historical",
                    enum=["historical", "parametric"],
                ),
                ToolParameter(
                    name="portfolio_value",
                    type="number",
                    description="Portfolio value for dollar VaR",
                    required=False,
                ),
            ],
            returns={
                "var_percent": "VaR as percentage",
                "var_dollar": "VaR in dollars (if portfolio_value provided)",
                "cvar_percent": "CVaR as percentage",
                "confidence_level": "Confidence level used",
            },
            typical_execution_time_ms=1500,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute VaR calculation."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]
            confidence = validated.get("confidence_level", 0.95)
            method = validated.get("method", "historical")
            portfolio_value = validated.get("portfolio_value")

            data = await fetch_price_data(symbol, 252, self.fmp_api_key)

            config = RiskCalculationConfig(
                confidence_level=confidence,
                var_method=method,
            )
            calculator = RiskMetricsCalculator(config)

            import pandas as pd
            df = pd.DataFrame({"price": data.prices})
            returns = df["price"].pct_change().dropna()

            var_result = calculator._calculate_var(returns, portfolio_value)
            cvar_result = calculator._calculate_cvar(returns, var_result.var_value, portfolio_value)

            result_dict = {
                "ticker": symbol,
                "var": var_result.model_dump(),
                "cvar": cvar_result.model_dump(),
            }

            formatted = f"""
**Value at Risk for {symbol}**

VaR ({confidence:.0%} confidence, {method}):
- Daily VaR: {var_result.var_percent:.2f}%
{'- Dollar VaR: $' + f"{var_result.var_dollar:,.2f}" if var_result.var_dollar else ''}

CVaR (Expected Shortfall):
- Daily CVaR: {cvar_result.cvar_percent:.2f}%
{'- Dollar CVaR: $' + f"{cvar_result.cvar_dollar:,.2f}" if cvar_result.cvar_dollar else ''}
- Tail observations: {cvar_result.tail_observations}

**Interpretation:**
{confidence:.0%} of days, losses won't exceed {abs(var_result.var_percent):.2f}%
When losses DO exceed VaR, average loss is {abs(cvar_result.cvar_percent):.2f}%
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


class GetSharpeRatioTool(BaseTool):
    """Tool to calculate Sharpe and Sortino ratios."""

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key

        self.schema = ToolSchema(
            name="getSharpeRatio",
            category="risk_enhanced",
            description="Calculate Sharpe and Sortino risk-adjusted return ratios",
            capabilities=[
                "✅ Sharpe Ratio (total volatility)",
                "✅ Sortino Ratio (downside volatility only)",
                "✅ Quality assessment (poor/acceptable/good/excellent)",
            ],
            limitations=[
                "❌ Assumes normal distribution for Sharpe",
                "❌ Historical returns may not predict future",
            ],
            usage_hints=[
                "Sharpe > 1.0 is generally good",
                "Sortino > Sharpe suggests positive skew",
                "Compare ratios across similar assets",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                ),
                ToolParameter(
                    name="risk_free_rate",
                    type="number",
                    description="Annual risk-free rate (default: 0.045 = 4.5%)",
                    required=False,
                    default=0.045,
                ),
                ToolParameter(
                    name="days",
                    type="integer",
                    description="Days of data (default: 252)",
                    required=False,
                    default=252,
                ),
            ],
            returns={
                "sharpe_ratio": "Annualized Sharpe Ratio",
                "sortino_ratio": "Annualized Sortino Ratio",
                "sharpe_quality": "Quality assessment",
                "sortino_quality": "Quality assessment",
            },
            typical_execution_time_ms=1500,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute Sharpe/Sortino calculation."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]
            rf = validated.get("risk_free_rate", 0.045)
            days = validated.get("days", 252)

            data = await fetch_price_data(symbol, days, self.fmp_api_key)

            config = RiskCalculationConfig(risk_free_rate=rf)
            calculator = RiskMetricsCalculator(config)

            import pandas as pd
            df = pd.DataFrame({"price": data.prices})
            returns = df["price"].pct_change().dropna()

            sharpe = calculator._calculate_sharpe(returns)
            sortino = calculator._calculate_sortino(returns)

            result_dict = {
                "ticker": symbol,
                "sharpe": sharpe.model_dump(),
                "sortino": sortino.model_dump(),
            }

            formatted = f"""
**Risk-Adjusted Returns for {symbol}**

Sharpe Ratio: {sharpe.sharpe_ratio:.2f} ({sharpe.quality.upper()})
- Excess return: {sharpe.excess_return:.2%}
- Volatility: {sharpe.volatility:.2%}

Sortino Ratio: {sortino.sortino_ratio:.2f} ({sortino.quality.upper()})
- Downside deviation: {sortino.downside_deviation:.2%}
- Negative days: {sortino.negative_returns_count}

Risk-free rate used: {rf:.1%}
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


class GetMaxDrawdownTool(BaseTool):
    """Tool to calculate Maximum Drawdown analysis."""

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key

        self.schema = ToolSchema(
            name="getMaxDrawdown",
            category="risk_enhanced",
            description="Calculate Maximum Drawdown with detailed peak-to-trough analysis",
            capabilities=[
                "✅ Maximum drawdown percentage",
                "✅ Peak and trough dates/prices",
                "✅ Recovery analysis",
                "✅ Current drawdown status",
                "✅ Calmar ratio",
            ],
            limitations=[
                "❌ Historical worst case only",
                "❌ Future drawdowns may be worse",
            ],
            usage_hints=[
                "Check pain tolerance against max drawdown",
                "-20% is bear market, -50% is catastrophic",
                "Current drawdown shows if already in decline",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                ),
                ToolParameter(
                    name="days",
                    type="integer",
                    description="Days of data (default: 252)",
                    required=False,
                    default=252,
                ),
            ],
            returns={
                "max_drawdown": "Maximum drawdown percentage",
                "peak_date": "Date of peak before drawdown",
                "trough_date": "Date of trough (bottom)",
                "recovery_date": "Date recovered (if any)",
                "current_drawdown": "Current drawdown from recent peak",
                "calmar_ratio": "Annual return / max drawdown",
            },
            typical_execution_time_ms=1500,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute Max Drawdown calculation."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]
            days = validated.get("days", 252)

            data = await fetch_price_data(symbol, days, self.fmp_api_key)

            calculator = RiskMetricsCalculator()

            import pandas as pd
            df = pd.DataFrame({"date": data.dates, "price": data.prices})
            df = df.set_index("date")
            returns = df["price"].pct_change().dropna()

            max_dd = calculator._calculate_max_drawdown(df["price"])
            calmar = calculator._calculate_calmar(returns, max_dd.max_drawdown)

            result_dict = {
                "ticker": symbol,
                "max_drawdown": max_dd.model_dump(),
                "calmar_ratio": calmar.model_dump(),
            }

            formatted = f"""
**Maximum Drawdown Analysis for {symbol}**

Max Drawdown: {max_dd.max_drawdown:.1%}
- Peak: ${max_dd.peak_price:.2f} on {max_dd.peak_date}
- Trough: ${max_dd.trough_price:.2f} on {max_dd.trough_date}
- Duration: {max_dd.drawdown_duration_days} days

Recovery: {'Yes - ' + str(max_dd.recovery_date) + f' ({max_dd.recovery_duration_days} days)' if max_dd.recovery_date else 'Not yet recovered'}

Current Drawdown: {max_dd.current_drawdown:.1%} from recent peak

Calmar Ratio: {calmar.calmar_ratio:.2f}
(Annual return {calmar.annual_return:.1%} / Max DD {abs(calmar.max_drawdown):.1%})
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


class GetBetaAlphaTool(BaseTool):
    """Tool to calculate Beta and Alpha vs benchmark."""

    def __init__(self, fmp_api_key: Optional[str] = None):
        super().__init__()
        self.fmp_api_key = fmp_api_key

        self.schema = ToolSchema(
            name="getBetaAlpha",
            category="risk_enhanced",
            description="Calculate Beta (market sensitivity) and Alpha (excess return) vs benchmark",
            capabilities=[
                "✅ Beta calculation",
                "✅ Jensen's Alpha",
                "✅ R-squared (variance explained)",
                "✅ Correlation with benchmark",
                "✅ Treynor Ratio",
                "✅ Information Ratio",
            ],
            limitations=[
                "❌ Requires benchmark data",
                "❌ Historical relationship may not persist",
            ],
            usage_hints=[
                "Beta > 1 means more volatile than market",
                "Positive alpha indicates outperformance",
                "Use SPY for US market benchmark",
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol",
                    required=True,
                ),
                ToolParameter(
                    name="benchmark_symbol",
                    type="string",
                    description="Benchmark symbol (default: SPY)",
                    required=False,
                    default="SPY",
                ),
                ToolParameter(
                    name="days",
                    type="integer",
                    description="Days of data (default: 252)",
                    required=False,
                    default=252,
                ),
            ],
            returns={
                "beta": "Portfolio beta",
                "alpha": "Jensen's alpha (annualized)",
                "r_squared": "R-squared",
                "correlation": "Correlation with benchmark",
                "treynor_ratio": "Treynor ratio",
                "information_ratio": "Information ratio",
            },
            typical_execution_time_ms=2500,
        )

    async def execute(self, **params) -> ToolOutput:
        """Execute Beta/Alpha calculation."""
        import time
        start_time = time.time()

        try:
            validated = self.validate_input(params)
            symbol = validated["symbol"]
            benchmark_symbol = validated.get("benchmark_symbol", "SPY")
            days = validated.get("days", 252)

            # Fetch both data series
            data = await fetch_price_data(symbol, days, self.fmp_api_key)
            bench_data = await fetch_price_data(benchmark_symbol, days, self.fmp_api_key)

            benchmark = BenchmarkDataInput(
                ticker=bench_data.ticker,
                dates=bench_data.dates,
                prices=bench_data.prices,
            )

            calculator = RiskMetricsCalculator()

            import pandas as pd
            df = pd.DataFrame({"price": data.prices})
            returns = df["price"].pct_change().dropna()

            beta_alpha = calculator._calculate_beta_alpha(returns, benchmark)
            treynor = calculator._calculate_treynor(returns, beta_alpha.beta) if beta_alpha else None
            info_ratio = calculator._calculate_information_ratio(returns, benchmark)

            result_dict = {
                "ticker": symbol,
                "benchmark": benchmark_symbol,
                "beta_alpha": beta_alpha.model_dump() if beta_alpha else None,
                "treynor": treynor.model_dump() if treynor else None,
                "information_ratio": info_ratio.model_dump() if info_ratio else None,
            }

            formatted = f"""
**Beta/Alpha Analysis for {symbol} vs {benchmark_symbol}**

Beta: {beta_alpha.beta:.2f} ({beta_alpha.beta_category.replace('_', ' ').title()})
Alpha (Annual): {beta_alpha.alpha:.2%}
Correlation: {beta_alpha.correlation:.2f}
R-squared: {beta_alpha.r_squared:.2%}
"""
            if treynor:
                formatted += f"\nTreynor Ratio: {treynor.treynor_ratio:.4f}"

            if info_ratio:
                formatted += f"""
Information Ratio: {info_ratio.information_ratio:.2f} ({info_ratio.quality.upper()})
- Active Return: {info_ratio.active_return:.2%}
- Tracking Error: {info_ratio.tracking_error:.2%}
"""

            formatted += f"""
**Interpretation:**
- Beta {beta_alpha.beta:.2f} means when {benchmark_symbol} moves 1%, {symbol} typically moves {beta_alpha.beta:.2f}%
- Alpha {beta_alpha.alpha:.2%} means {symbol} {"outperformed" if beta_alpha.alpha > 0 else "underperformed"} by {abs(beta_alpha.alpha):.2%} annually after adjusting for beta risk
"""

            execution_time = int((time.time() - start_time) * 1000)

            return create_success_output(
                tool_name=self.schema.name,
                data=result_dict,
                formatted_context=formatted,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            return create_error_output(
                tool_name=self.schema.name,
                error=str(e),
                execution_time_ms=execution_time,
            )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "GetRiskMetricsTool",
    "GetVaRTool",
    "GetSharpeRatioTool",
    "GetMaxDrawdownTool",
    "GetBetaAlphaTool",
]
