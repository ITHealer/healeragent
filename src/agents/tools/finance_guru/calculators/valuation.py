"""
Finance Guru - Valuation Calculators (Phase 1)

Layer 2: Pure computation functions for intrinsic value calculations.

This module implements:
- DCF (Discounted Cash Flow) valuation
- Graham Formula valuation
- DDM (Dividend Discount Model) valuation
- Comparable company analysis

EDUCATIONAL NOTES:
- Intrinsic value: The "true" worth of a stock based on fundamentals
- Margin of safety: Buffer between intrinsic value and purchase price
- Discount rate: Rate used to convert future cash flows to present value

Author: HealerAgent Development Team
"""

from typing import Optional

from src.agents.tools.finance_guru.calculators.base import (
    BaseCalculator,
    CalculationContext,
    CalculationError,
)
from src.agents.tools.finance_guru.models.valuation import (
    # Enums
    ValuationMethod,
    DDMType,
    # DCF
    DCFInputData,
    DCFConfig,
    DCFProjection,
    DCFOutput,
    # Graham
    GrahamInputData,
    GrahamConfig,
    GrahamOutput,
    # DDM
    DDMInputData,
    DDMConfig,
    DDMOutput,
    # Comparable
    ComparableInputData,
    MultipleValuation,
    ComparableOutput,
    # Summary
    ValuationSummary,
    # Risk Framework
    ScenarioCase,
    RiskFramework,
)


class DCFCalculator(BaseCalculator):
    """Discounted Cash Flow valuation calculator.

    DCF values a company by:
    1. Projecting future free cash flows
    2. Discounting them to present value using WACC
    3. Adding terminal value (perpetual growth assumption)
    4. Adjusting for cash and debt to get equity value

    Formula:
        Enterprise Value = Σ(FCFt / (1 + r)^t) + TV / (1 + r)^n
        TV = FCFn × (1 + g) / (r - g)  [Gordon Growth for terminal value]
        Equity Value = EV + Cash - Debt
    """

    def __init__(self):
        super().__init__()
        self.context = CalculationContext(calculator_name="DCFCalculator")

    def calculate(
        self,
        data: DCFInputData,
        config: DCFConfig,
        fcf_source: str = "api",
    ) -> DCFOutput:
        """Calculate DCF intrinsic value.

        Args:
            data: Input data with FCF, growth rates, discount rate
            config: Configuration with projection years, margin of safety
            fcf_source: Source of FCF data (api, manual, normalized)

        Returns:
            DCFOutput with intrinsic value and projections
        """
        self.context.start()

        try:
            projections = []
            fcf = data.current_fcf
            sum_pv_fcf = 0.0

            # Project cash flows
            for year in range(1, config.projection_years + 1):
                # Get growth rate for this year
                if year <= len(data.growth_rates):
                    growth = data.growth_rates[year - 1]
                else:
                    growth = data.growth_rates[-1]  # Use last rate

                fcf = fcf * (1 + growth)
                discount_factor = 1 / ((1 + data.discount_rate) ** year)
                pv = fcf * discount_factor

                projections.append(DCFProjection(
                    year=year,
                    fcf=fcf,
                    growth_rate=growth,
                    discount_factor=discount_factor,
                    present_value=pv,
                ))

                sum_pv_fcf += pv

            # Terminal value (Gordon Growth)
            terminal_fcf = fcf * (1 + data.terminal_growth)
            terminal_value = terminal_fcf / (data.discount_rate - data.terminal_growth)

            # Present value of terminal value
            terminal_discount = 1 / ((1 + data.discount_rate) ** config.projection_years)
            pv_terminal = terminal_value * terminal_discount

            # Enterprise value
            enterprise_value = sum_pv_fcf + pv_terminal

            # Equity value
            equity_value = enterprise_value + data.cash - data.debt

            # Intrinsic value per share
            intrinsic_value = equity_value / data.shares_outstanding

            # Margin of safety price
            margin_price = intrinsic_value * (1 - config.margin_of_safety)

            # Calculate TV as % of EV (important for reliability assessment)
            tv_as_pct_of_ev = (pv_terminal / enterprise_value * 100) if enterprise_value > 0 else None

            # Calculate upside and verdict
            upside = None
            verdict = "unknown"
            if data.current_price:
                upside = (intrinsic_value - data.current_price) / data.current_price * 100
                if upside > 20:
                    verdict = "undervalued"
                elif upside < -10:
                    verdict = "overvalued"
                else:
                    verdict = "fairly_valued"

            # Sensitivity analysis (1D)
            sensitivity = self._sensitivity_analysis(data, config)

            # 2D Sensitivity Matrix (WACC × Terminal Growth)
            # Pass intrinsic_value to ensure consistency between base case and matrix
            sensitivity_2d = self._sensitivity_2d_matrix(
                data, config, base_intrinsic_value=intrinsic_value
            )

            # Reverse DCF: Calculate implied perpetual growth rate given current price
            implied_growth = None
            if data.current_price and data.current_price > 0:
                implied_growth = self._calculate_implied_growth(data, config)

            # Validation warnings
            validation_warnings = self._generate_validation_warnings(
                data=data,
                tv_pct=tv_as_pct_of_ev,
                upside=upside,
                implied_growth=implied_growth,
            )

            # Risk framework with scenario analysis
            risk_framework = self._generate_risk_framework(
                data=data,
                config=config,
                base_intrinsic_value=intrinsic_value,
            )

            self.context.complete()

            return DCFOutput(
                symbol=data.symbol,
                projections=projections,
                sum_of_pv_fcf=sum_pv_fcf,
                terminal_value=terminal_value,
                pv_terminal_value=pv_terminal,
                enterprise_value=enterprise_value,
                equity_value=equity_value,
                intrinsic_value_per_share=intrinsic_value,
                current_price=data.current_price,
                upside_potential=upside,
                margin_of_safety_price=margin_price,
                verdict=verdict,
                # Enhanced fields
                discount_rate=data.discount_rate,
                terminal_growth=data.terminal_growth,
                fcf_source=fcf_source,
                shares_outstanding=data.shares_outstanding,
                cash=data.cash,
                debt=data.debt,
                tv_as_pct_of_ev=tv_as_pct_of_ev,
                implied_growth_rate=implied_growth,
                sensitivity=sensitivity,
                sensitivity_2d=sensitivity_2d,
                validation_warnings=validation_warnings if validation_warnings else None,
                # Risk framework
                risk_framework=risk_framework,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"DCF calculation failed: {e}")

    def _sensitivity_analysis(
        self,
        data: DCFInputData,
        config: DCFConfig,
    ) -> dict[str, dict[str, float]]:
        """Perform sensitivity analysis on key inputs.

        Tests intrinsic value across different discount rates and terminal growth rates.
        """
        sensitivity = {}

        # Discount rate sensitivity
        discount_rates = [
            data.discount_rate - 0.02,
            data.discount_rate - 0.01,
            data.discount_rate,
            data.discount_rate + 0.01,
            data.discount_rate + 0.02,
        ]

        for dr in discount_rates:
            if dr <= data.terminal_growth:
                continue

            try:
                modified_data = DCFInputData(
                    symbol=data.symbol,
                    current_fcf=data.current_fcf,
                    growth_rates=data.growth_rates,
                    terminal_growth=data.terminal_growth,
                    discount_rate=dr,
                    shares_outstanding=data.shares_outstanding,
                    cash=data.cash,
                    debt=data.debt,
                )
                result = self._calculate_simple(modified_data, config)
                dr_key = f"{dr:.1%}"
                if "discount_rate" not in sensitivity:
                    sensitivity["discount_rate"] = {}
                sensitivity["discount_rate"][dr_key] = result
            except Exception:
                pass

        # Terminal growth sensitivity
        terminal_rates = [0.01, 0.02, 0.025, 0.03, 0.035]

        for tg in terminal_rates:
            if tg >= data.discount_rate:
                continue

            try:
                modified_data = DCFInputData(
                    symbol=data.symbol,
                    current_fcf=data.current_fcf,
                    growth_rates=data.growth_rates,
                    terminal_growth=tg,
                    discount_rate=data.discount_rate,
                    shares_outstanding=data.shares_outstanding,
                    cash=data.cash,
                    debt=data.debt,
                )
                result = self._calculate_simple(modified_data, config)
                tg_key = f"{tg:.1%}"
                if "terminal_growth" not in sensitivity:
                    sensitivity["terminal_growth"] = {}
                sensitivity["terminal_growth"][tg_key] = result
            except Exception:
                pass

        return sensitivity

    def _calculate_simple(self, data: DCFInputData, config: DCFConfig) -> float:
        """Simple DCF calculation returning just intrinsic value."""
        fcf = data.current_fcf
        sum_pv_fcf = 0.0

        for year in range(1, config.projection_years + 1):
            growth = data.growth_rates[min(year - 1, len(data.growth_rates) - 1)]
            fcf = fcf * (1 + growth)
            discount_factor = 1 / ((1 + data.discount_rate) ** year)
            sum_pv_fcf += fcf * discount_factor

        terminal_fcf = fcf * (1 + data.terminal_growth)
        terminal_value = terminal_fcf / (data.discount_rate - data.terminal_growth)
        terminal_discount = 1 / ((1 + data.discount_rate) ** config.projection_years)
        pv_terminal = terminal_value * terminal_discount

        enterprise_value = sum_pv_fcf + pv_terminal
        equity_value = enterprise_value + data.cash - data.debt

        return equity_value / data.shares_outstanding

    def _sensitivity_2d_matrix(
        self,
        data: DCFInputData,
        config: DCFConfig,
        base_intrinsic_value: Optional[float] = None,
    ) -> dict:
        """Generate 2D sensitivity matrix (WACC × Terminal Growth).

        IMPORTANT: Ensures consistency between base case result and matrix[base_wacc][base_tgr].

        Returns a dict with:
        - wacc_rates: list of WACC values (formatted strings)
        - growth_rates: list of terminal growth values (formatted strings)
        - values: 2D grid of intrinsic values [wacc_idx][growth_idx]
        - base_wacc_idx: index of base WACC in the matrix
        - base_tgr_idx: index of base terminal growth in the matrix
        - consistency_check: True if matrix[base][base] matches base_intrinsic_value
        """
        # Define ranges centered on actual inputs (NO ARBITRARY ROUNDING)
        base_wacc = data.discount_rate
        base_tgr = data.terminal_growth

        wacc_rates = [
            base_wacc - 0.02,
            base_wacc - 0.01,
            base_wacc,  # Base case - MUST use actual input value
            base_wacc + 0.01,
            base_wacc + 0.02,
        ]

        # Include the actual terminal growth in the matrix
        growth_rates = sorted(set([0.015, 0.020, base_tgr, 0.030, 0.035]))
        if len(growth_rates) > 5:
            # Keep base_tgr and closest values
            growth_rates = [g for g in growth_rates if abs(g - base_tgr) <= 0.015]
            if len(growth_rates) < 5:
                growth_rates = [0.015, 0.020, base_tgr, 0.030, 0.035][:5]

        # Filter out invalid combinations
        wacc_rates = [w for w in wacc_rates if w > 0.04]  # Min WACC 4%
        growth_rates = [g for g in growth_rates if g < max(wacc_rates)]

        # Find indices of base case in matrix
        base_wacc_idx = None
        base_tgr_idx = None
        for i, w in enumerate(wacc_rates):
            if abs(w - base_wacc) < 0.0001:
                base_wacc_idx = i
        for j, g in enumerate(growth_rates):
            if abs(g - base_tgr) < 0.0001:
                base_tgr_idx = j

        values = []
        for i, wacc in enumerate(wacc_rates):
            row = []
            for j, tg in enumerate(growth_rates):
                if tg >= wacc:
                    row.append(None)  # Invalid combination
                    continue
                try:
                    # For base case, use the pre-calculated value for CONSISTENCY
                    if (i == base_wacc_idx and j == base_tgr_idx
                            and base_intrinsic_value is not None):
                        row.append(round(base_intrinsic_value, 2))
                    else:
                        modified_data = DCFInputData(
                            symbol=data.symbol,
                            current_fcf=data.current_fcf,
                            growth_rates=data.growth_rates,
                            terminal_growth=tg,
                            discount_rate=wacc,
                            shares_outstanding=data.shares_outstanding,
                            cash=data.cash,
                            debt=data.debt,
                        )
                        iv = self._calculate_simple(modified_data, config)
                        row.append(round(iv, 2))
                except Exception:
                    row.append(None)
            values.append(row)

        # Consistency check
        consistency_check = True
        if (base_wacc_idx is not None and base_tgr_idx is not None
                and base_intrinsic_value is not None):
            matrix_base_value = values[base_wacc_idx][base_tgr_idx]
            if matrix_base_value is not None:
                consistency_check = abs(matrix_base_value - base_intrinsic_value) < 0.01

        return {
            "wacc_rates": [f"{w:.2%}" for w in wacc_rates],  # Use .2% for precision
            "growth_rates": [f"{g:.2%}" for g in growth_rates],  # Use .2% for precision
            "values": values,
            "base_wacc_idx": base_wacc_idx,
            "base_tgr_idx": base_tgr_idx,
            "consistency_check": consistency_check,
        }

    def _calculate_implied_growth(
        self,
        data: DCFInputData,
        config: DCFConfig,
    ) -> Optional[float]:
        """Reverse DCF: Calculate implied perpetual growth rate given current price.

        This answers: "What growth rate is the market pricing in?"

        Uses binary search to find the terminal growth rate that produces
        an intrinsic value equal to the current price.
        """
        if not data.current_price or data.current_price <= 0:
            return None

        target_price = data.current_price
        low, high = -0.02, data.discount_rate - 0.005  # Growth must be < WACC

        # Binary search for implied growth
        for _ in range(50):  # Max iterations
            mid = (low + high) / 2
            try:
                modified_data = DCFInputData(
                    symbol=data.symbol,
                    current_fcf=data.current_fcf,
                    growth_rates=data.growth_rates,
                    terminal_growth=mid,
                    discount_rate=data.discount_rate,
                    shares_outstanding=data.shares_outstanding,
                    cash=data.cash,
                    debt=data.debt,
                )
                iv = self._calculate_simple(modified_data, config)

                if abs(iv - target_price) < 0.01:
                    return round(mid, 4)
                elif iv > target_price:
                    high = mid
                else:
                    low = mid
            except Exception:
                high = mid  # Reduce range on error

        # Return best estimate
        return round((low + high) / 2, 4)

    def _generate_validation_warnings(
        self,
        data: DCFInputData,
        tv_pct: Optional[float],
        upside: Optional[float],
        implied_growth: Optional[float],
    ) -> list[str]:
        """Generate validation warnings for DCF analysis.

        Warnings indicate potential issues with the valuation that
        should be considered when interpreting results.
        """
        warnings = []

        # Terminal value as % of EV warning
        if tv_pct is not None and tv_pct > 75:
            warnings.append(
                f"⚠️ Terminal Value = {tv_pct:.1f}% of EV (>75%). "
                "Valuation heavily depends on long-term assumptions."
            )

        # Terminal growth vs typical GDP growth
        if data.terminal_growth > 0.03:
            warnings.append(
                f"⚠️ Terminal growth ({data.terminal_growth:.1%}) exceeds typical GDP growth (2-3%). "
                "Consider using more conservative assumption."
            )

        # Negative FCF warning (already blocked but check anyway)
        if data.current_fcf <= 0:
            warnings.append(
                "⚠️ Negative or zero FCF. DCF not suitable for this company currently."
            )

        # Extreme upside/downside
        if upside is not None:
            if upside > 100:
                warnings.append(
                    f"⚠️ Extremely high upside ({upside:.0f}%). "
                    "Verify inputs and consider if assumptions are realistic."
                )
            elif upside < -50:
                warnings.append(
                    f"⚠️ Extremely negative valuation ({upside:.0f}%). "
                    "Stock may be priced for different growth expectations."
                )

        # Implied growth warning
        if implied_growth is not None:
            if implied_growth > 0.04:
                warnings.append(
                    f"⚠️ Market implies {implied_growth:.1%} perpetual growth, "
                    "higher than typical mature company growth."
                )
            elif implied_growth < 0:
                warnings.append(
                    f"⚠️ Market implies negative perpetual growth ({implied_growth:.1%}). "
                    "Stock may be distressed or facing structural decline."
                )

        # Discount rate sanity check
        if data.discount_rate < 0.06:
            warnings.append(
                f"⚠️ Low discount rate ({data.discount_rate:.1%}). "
                "May underestimate risk for equity investment."
            )
        elif data.discount_rate > 0.15:
            warnings.append(
                f"⚠️ High discount rate ({data.discount_rate:.1%}). "
                "Typical range is 8-12% for mature companies."
            )

        return warnings

    def _generate_risk_framework(
        self,
        data: DCFInputData,
        config: DCFConfig,
        base_intrinsic_value: float,
    ) -> RiskFramework:
        """Generate risk framework with scenario analysis.

        Creates bull, base, and bear cases with quantitative triggers
        and calculates probability-weighted expected value.
        """
        current_price = data.current_price or base_intrinsic_value

        # Bull Case: Lower WACC (-1%), Higher Growth (+2% each year)
        try:
            bull_growth = [min(g + 0.02, 0.25) for g in data.growth_rates]
            bull_data = DCFInputData(
                symbol=data.symbol,
                current_fcf=data.current_fcf,
                growth_rates=bull_growth,
                terminal_growth=min(data.terminal_growth + 0.005, 0.035),
                discount_rate=max(data.discount_rate - 0.01, 0.06),
                shares_outstanding=data.shares_outstanding,
                cash=data.cash,
                debt=data.debt,
            )
            bull_value = self._calculate_simple(bull_data, config)
            bull_upside = (bull_value - current_price) / current_price * 100
        except Exception:
            bull_value = base_intrinsic_value * 1.3
            bull_upside = 30.0

        # Bear Case: Higher WACC (+1%), Lower Growth (-3% each year)
        try:
            bear_growth = [max(g - 0.03, 0.0) for g in data.growth_rates]
            bear_data = DCFInputData(
                symbol=data.symbol,
                current_fcf=data.current_fcf,
                growth_rates=bear_growth,
                terminal_growth=max(data.terminal_growth - 0.005, 0.015),
                discount_rate=min(data.discount_rate + 0.01, 0.15),
                shares_outstanding=data.shares_outstanding,
                cash=data.cash,
                debt=data.debt,
            )
            bear_value = self._calculate_simple(bear_data, config)
            bear_upside = (bear_value - current_price) / current_price * 100
        except Exception:
            bear_value = base_intrinsic_value * 0.7
            bear_upside = -30.0

        base_upside = (base_intrinsic_value - current_price) / current_price * 100

        # Scenarios with probabilities
        scenarios = [
            ScenarioCase(
                name="Bull",
                probability=0.25,
                intrinsic_value=bull_value,
                upside_potential=bull_upside,
                key_assumptions={
                    "wacc": data.discount_rate - 0.01,
                    "terminal_growth": data.terminal_growth + 0.005,
                    "fcf_growth": "+2% annually",
                },
                quantitative_triggers=[
                    "Revenue growth exceeds analyst estimates by >5%",
                    "Operating margin expansion >100bps YoY",
                    "FCF conversion rate >25%",
                ],
            ),
            ScenarioCase(
                name="Base",
                probability=0.50,
                intrinsic_value=base_intrinsic_value,
                upside_potential=base_upside,
                key_assumptions={
                    "wacc": data.discount_rate,
                    "terminal_growth": data.terminal_growth,
                    "fcf_growth": "As projected",
                },
                quantitative_triggers=[
                    "Revenue growth in line with consensus",
                    "Margins stable within historical range",
                    "No major business model disruption",
                ],
            ),
            ScenarioCase(
                name="Bear",
                probability=0.25,
                intrinsic_value=bear_value,
                upside_potential=bear_upside,
                key_assumptions={
                    "wacc": data.discount_rate + 0.01,
                    "terminal_growth": data.terminal_growth - 0.005,
                    "fcf_growth": "-3% annually",
                },
                quantitative_triggers=[
                    "Revenue growth misses estimates by >5%",
                    "Operating margin compression >100bps",
                    "Competitive pressure from new entrants",
                ],
            ),
        ]

        # Probability-weighted expected value
        prob_weighted_value = sum(s.probability * s.intrinsic_value for s in scenarios)

        # Invalidation price (20% below bear case)
        invalidation_price = bear_value * 0.8

        # Downside risk (current price to bear case)
        downside_risk = abs(bear_upside) if bear_upside < 0 else 0

        # Risk-reward ratio
        upside = max(bull_upside, 0)
        risk_reward = upside / downside_risk if downside_risk > 0 else float('inf')

        # Position sizing suggestion based on conviction
        if risk_reward > 3:
            position_suggestion = "High conviction: Standard position size (2-5% of portfolio)"
        elif risk_reward > 1.5:
            position_suggestion = "Moderate conviction: Reduced position (1-2% of portfolio)"
        else:
            position_suggestion = "Low conviction: Small starter position (<1%) or pass"

        # Key risks
        key_risks = [
            "Macro environment: Interest rate changes affect WACC",
            "Competitive dynamics: Market share erosion risk",
            "Execution risk: Growth assumptions may not materialize",
            "Valuation risk: Multiple compression in risk-off environment",
        ]

        return RiskFramework(
            scenarios=scenarios,
            probability_weighted_value=prob_weighted_value,
            invalidation_price=invalidation_price,
            downside_risk=downside_risk,
            risk_reward_ratio=risk_reward,
            position_size_suggestion=position_suggestion,
            key_risks=key_risks,
        )


class GrahamCalculator(BaseCalculator):
    """Graham Formula valuation calculator.

    Benjamin Graham's intrinsic value formula:
        V = EPS × (8.5 + 2g) × (4.4 / Y)

    Where:
    - V = Intrinsic value
    - EPS = Trailing 12-month earnings per share
    - g = Expected 5-year growth rate (as whole number, e.g., 10 for 10%)
    - 8.5 = P/E base for a no-growth company
    - Y = Current yield on AAA corporate bonds

    The formula was later revised by Graham to include margin of safety:
        V = EPS × (8.5 + 2g) × (4.4 / Y) × (Margin)
    """

    def __init__(self):
        super().__init__()
        self.context = CalculationContext(calculator_name="GrahamCalculator")

    def calculate(
        self,
        data: GrahamInputData,
        config: GrahamConfig,
    ) -> GrahamOutput:
        """Calculate Graham intrinsic value.

        Args:
            data: Input with EPS, growth rate, AAA yield
            config: Configuration with P/E base, margin of safety

        Returns:
            GrahamOutput with intrinsic value
        """
        self.context.start()

        try:
            # Convert growth rate to whole number (Graham used 10 for 10%)
            growth_whole = data.growth_rate * 100

            # P/E components
            pe_no_growth = config.pe_base
            growth_premium = config.growth_multiplier * growth_whole

            # Bond yield adjustment (original formula used 4.4%)
            bond_adjustment = 4.4 / (data.aaa_yield * 100)

            # Graham formula
            graham_value = data.eps * (pe_no_growth + growth_premium) * bond_adjustment

            # Apply margin of safety
            intrinsic_value = graham_value
            margin_price = intrinsic_value * (1 - config.margin_of_safety)

            # Calculate upside
            upside = (intrinsic_value - data.current_price) / data.current_price * 100

            # Verdict
            if upside > 20:
                verdict = "undervalued"
            elif upside < -10:
                verdict = "overvalued"
            else:
                verdict = "fairly_valued"

            self.context.complete()

            return GrahamOutput(
                symbol=data.symbol,
                eps=data.eps,
                growth_rate=data.growth_rate,
                aaa_yield=data.aaa_yield,
                graham_value=graham_value,
                intrinsic_value=intrinsic_value,
                margin_of_safety_price=margin_price,
                current_price=data.current_price,
                upside_potential=upside,
                verdict=verdict,
                pe_no_growth=pe_no_growth,
                growth_premium=growth_premium,
                bond_adjustment=bond_adjustment,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Graham calculation failed: {e}")


class DDMCalculator(BaseCalculator):
    """Dividend Discount Model calculator.

    Implements several DDM variants:

    1. Gordon Growth Model (constant growth):
       P = D1 / (r - g)
       Where D1 = D0 × (1 + g)

    2. Two-Stage Model:
       P = Σ(Dt / (1 + r)^t) + Pn / (1 + r)^n
       Where Pn = Dn+1 / (r - g_stable)

    3. H-Model (declining growth):
       P = D0 × (1 + g_stable) / (r - g_stable) + D0 × H × (g_high - g_stable) / (r - g_stable)
       Where H = half-life of high growth period
    """

    def __init__(self):
        super().__init__()
        self.context = CalculationContext(calculator_name="DDMCalculator")

    def calculate(
        self,
        data: DDMInputData,
        config: DDMConfig,
    ) -> DDMOutput:
        """Calculate DDM intrinsic value.

        Args:
            data: Input with dividend, growth, required return
            config: Configuration with model type

        Returns:
            DDMOutput with intrinsic value
        """
        self.context.start()

        try:
            # Expected dividend next year
            expected_dividend = data.current_dividend * (1 + data.dividend_growth)

            # Calculate based on model type
            if config.model_type == DDMType.GORDON:
                intrinsic_value, stages = self._gordon_growth(data)
            elif config.model_type == DDMType.TWO_STAGE:
                intrinsic_value, stages = self._two_stage(data, config)
            elif config.model_type == DDMType.H_MODEL:
                intrinsic_value, stages = self._h_model(data, config)
            else:  # THREE_STAGE
                intrinsic_value, stages = self._three_stage(data, config)

            # Dividend yield
            dividend_yield = data.current_dividend / data.current_price

            # Upside potential
            upside = (intrinsic_value - data.current_price) / data.current_price * 100

            # Verdict
            if upside > 15:
                verdict = "undervalued"
            elif upside < -10:
                verdict = "overvalued"
            else:
                verdict = "fairly_valued"

            self.context.complete()

            return DDMOutput(
                symbol=data.symbol,
                model_type=config.model_type,
                current_dividend=data.current_dividend,
                expected_dividend=expected_dividend,
                dividend_growth=data.dividend_growth,
                required_return=data.required_return,
                intrinsic_value=intrinsic_value,
                current_price=data.current_price,
                dividend_yield=dividend_yield,
                upside_potential=upside,
                verdict=verdict,
                stages=stages,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"DDM calculation failed: {e}")

    def _gordon_growth(self, data: DDMInputData) -> tuple[float, None]:
        """Gordon Growth Model: P = D1 / (r - g)"""
        d1 = data.current_dividend * (1 + data.dividend_growth)
        intrinsic_value = d1 / (data.required_return - data.dividend_growth)
        return intrinsic_value, None

    def _two_stage(
        self, data: DDMInputData, config: DDMConfig
    ) -> tuple[float, list[dict]]:
        """Two-stage DDM with high growth then stable growth."""
        high_growth = config.high_growth_rate or data.dividend_growth
        stable_growth = config.stable_growth_rate
        years = config.high_growth_years

        stages = []
        pv_dividends = 0.0
        dividend = data.current_dividend

        # High growth phase
        for year in range(1, years + 1):
            dividend = dividend * (1 + high_growth)
            pv = dividend / ((1 + data.required_return) ** year)
            pv_dividends += pv
            stages.append({
                "year": year,
                "phase": "high_growth",
                "dividend": dividend,
                "present_value": pv,
            })

        # Terminal value at stable growth
        terminal_dividend = dividend * (1 + stable_growth)
        terminal_value = terminal_dividend / (data.required_return - stable_growth)
        pv_terminal = terminal_value / ((1 + data.required_return) ** years)

        stages.append({
            "year": years + 1,
            "phase": "terminal",
            "terminal_value": terminal_value,
            "present_value": pv_terminal,
        })

        intrinsic_value = pv_dividends + pv_terminal
        return intrinsic_value, stages

    def _h_model(
        self, data: DDMInputData, config: DDMConfig
    ) -> tuple[float, list[dict]]:
        """H-Model: declining growth from high to stable over 2H years."""
        high_growth = config.high_growth_rate or data.dividend_growth
        stable_growth = config.stable_growth_rate
        h = config.high_growth_years / 2  # H = half-life

        # H-Model formula
        d0 = data.current_dividend
        r = data.required_return

        # Stable growth component
        stable_component = d0 * (1 + stable_growth) / (r - stable_growth)

        # Growth premium component
        premium_component = d0 * h * (high_growth - stable_growth) / (r - stable_growth)

        intrinsic_value = stable_component + premium_component

        stages = [
            {
                "component": "stable_growth",
                "value": stable_component,
                "description": f"D0(1+g_s)/(r-g_s) where g_s={stable_growth:.1%}",
            },
            {
                "component": "growth_premium",
                "value": premium_component,
                "description": f"D0×H×(g_h-g_s)/(r-g_s) where H={h}, g_h={high_growth:.1%}",
            },
        ]

        return intrinsic_value, stages

    def _three_stage(
        self, data: DDMInputData, config: DDMConfig
    ) -> tuple[float, list[dict]]:
        """Three-stage DDM: high growth, transition, stable."""
        high_growth = config.high_growth_rate or data.dividend_growth
        stable_growth = config.stable_growth_rate
        high_years = config.high_growth_years
        transition_years = high_years  # Same length for transition

        stages = []
        pv_total = 0.0
        dividend = data.current_dividend

        # Phase 1: High growth
        for year in range(1, high_years + 1):
            dividend = dividend * (1 + high_growth)
            pv = dividend / ((1 + data.required_return) ** year)
            pv_total += pv
            stages.append({
                "year": year,
                "phase": "high_growth",
                "growth_rate": high_growth,
                "dividend": dividend,
                "present_value": pv,
            })

        # Phase 2: Transition (linearly declining growth)
        current_year = high_years
        for i in range(1, transition_years + 1):
            current_year += 1
            # Linear interpolation of growth rate
            growth = high_growth - (high_growth - stable_growth) * (i / transition_years)
            dividend = dividend * (1 + growth)
            pv = dividend / ((1 + data.required_return) ** current_year)
            pv_total += pv
            stages.append({
                "year": current_year,
                "phase": "transition",
                "growth_rate": growth,
                "dividend": dividend,
                "present_value": pv,
            })

        # Phase 3: Terminal value at stable growth
        terminal_dividend = dividend * (1 + stable_growth)
        terminal_value = terminal_dividend / (data.required_return - stable_growth)
        pv_terminal = terminal_value / ((1 + data.required_return) ** current_year)
        pv_total += pv_terminal

        stages.append({
            "year": current_year + 1,
            "phase": "terminal",
            "growth_rate": stable_growth,
            "terminal_value": terminal_value,
            "present_value": pv_terminal,
        })

        return pv_total, stages


class ComparableCalculator(BaseCalculator):
    """Comparable company valuation calculator.

    Relative valuation using peer multiples:
    - P/E (Price-to-Earnings)
    - P/B (Price-to-Book)
    - P/S (Price-to-Sales)
    - EV/EBITDA

    Implied value = Target metric × Peer average multiple
    """

    def __init__(self):
        super().__init__()
        self.context = CalculationContext(calculator_name="ComparableCalculator")

    def calculate(
        self,
        data: ComparableInputData,
    ) -> ComparableOutput:
        """Calculate comparable company valuation.

        Args:
            data: Input with target company data and peer multiples

        Returns:
            ComparableOutput with implied values from each multiple
        """
        self.context.start()

        try:
            valuations = []
            implied_values = []
            peer_stats = {}

            # P/E valuation
            pe_ratios = [p.pe_ratio for p in data.peers if p.pe_ratio]
            if pe_ratios and data.eps > 0:
                avg_pe = sum(pe_ratios) / len(pe_ratios)
                med_pe = sorted(pe_ratios)[len(pe_ratios) // 2]
                implied_avg = data.eps * avg_pe
                implied_med = data.eps * med_pe

                valuations.append(MultipleValuation(
                    multiple_name="P/E",
                    peer_average=avg_pe,
                    peer_median=med_pe,
                    implied_value=implied_avg,
                    implied_value_median=implied_med,
                ))
                implied_values.append(implied_avg)
                peer_stats["P/E"] = {
                    "min": min(pe_ratios),
                    "max": max(pe_ratios),
                    "avg": avg_pe,
                    "median": med_pe,
                }

            # P/B valuation
            pb_ratios = [p.pb_ratio for p in data.peers if p.pb_ratio]
            if pb_ratios:
                avg_pb = sum(pb_ratios) / len(pb_ratios)
                med_pb = sorted(pb_ratios)[len(pb_ratios) // 2]
                implied_avg = data.book_value_per_share * avg_pb
                implied_med = data.book_value_per_share * med_pb

                valuations.append(MultipleValuation(
                    multiple_name="P/B",
                    peer_average=avg_pb,
                    peer_median=med_pb,
                    implied_value=implied_avg,
                    implied_value_median=implied_med,
                ))
                implied_values.append(implied_avg)
                peer_stats["P/B"] = {
                    "min": min(pb_ratios),
                    "max": max(pb_ratios),
                    "avg": avg_pb,
                    "median": med_pb,
                }

            # P/S valuation
            ps_ratios = [p.ps_ratio for p in data.peers if p.ps_ratio]
            if ps_ratios:
                avg_ps = sum(ps_ratios) / len(ps_ratios)
                med_ps = sorted(ps_ratios)[len(ps_ratios) // 2]
                implied_avg = data.revenue_per_share * avg_ps
                implied_med = data.revenue_per_share * med_ps

                valuations.append(MultipleValuation(
                    multiple_name="P/S",
                    peer_average=avg_ps,
                    peer_median=med_ps,
                    implied_value=implied_avg,
                    implied_value_median=implied_med,
                ))
                implied_values.append(implied_avg)
                peer_stats["P/S"] = {
                    "min": min(ps_ratios),
                    "max": max(ps_ratios),
                    "avg": avg_ps,
                    "median": med_ps,
                }

            # EV/EBITDA valuation
            ev_ebitda = [p.ev_ebitda for p in data.peers if p.ev_ebitda]
            if ev_ebitda and data.ebitda_per_share:
                avg_ev = sum(ev_ebitda) / len(ev_ebitda)
                med_ev = sorted(ev_ebitda)[len(ev_ebitda) // 2]
                # Note: This is simplified - proper EV/EBITDA needs enterprise value adjustment
                implied_avg = data.ebitda_per_share * avg_ev
                implied_med = data.ebitda_per_share * med_ev

                valuations.append(MultipleValuation(
                    multiple_name="EV/EBITDA",
                    peer_average=avg_ev,
                    peer_median=med_ev,
                    implied_value=implied_avg,
                    implied_value_median=implied_med,
                ))
                implied_values.append(implied_avg)
                peer_stats["EV/EBITDA"] = {
                    "min": min(ev_ebitda),
                    "max": max(ev_ebitda),
                    "avg": avg_ev,
                    "median": med_ev,
                }

            if not implied_values:
                raise CalculationError("No valid peer multiples available")

            # Calculate summary statistics
            avg_value = sum(implied_values) / len(implied_values)
            med_value = sorted(implied_values)[len(implied_values) // 2]

            # Upside potential
            upside = (avg_value - data.current_price) / data.current_price * 100

            # Verdict
            if upside > 15:
                verdict = "undervalued"
            elif upside < -10:
                verdict = "overvalued"
            else:
                verdict = "fairly_valued"

            self.context.complete()

            return ComparableOutput(
                symbol=data.symbol,
                peers=[p.symbol for p in data.peers],
                valuations=valuations,
                average_intrinsic_value=avg_value,
                median_intrinsic_value=med_value,
                current_price=data.current_price,
                upside_potential=upside,
                verdict=verdict,
                peer_stats=peer_stats,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Comparable calculation failed: {e}")


class ValuationCalculator(BaseCalculator):
    """Combined valuation calculator using multiple methods.

    Aggregates results from DCF, Graham, DDM, and Comparables
    to provide a comprehensive valuation assessment.
    """

    def __init__(self):
        super().__init__()
        self.context = CalculationContext(calculator_name="ValuationCalculator")
        self.dcf_calc = DCFCalculator()
        self.graham_calc = GrahamCalculator()
        self.ddm_calc = DDMCalculator()
        self.comparable_calc = ComparableCalculator()

    def calculate(self, data, **kwargs):
        """Implement abstract method - routes to calculate_summary()."""
        return self.calculate_summary(
            symbol=kwargs.get("symbol", ""),
            current_price=kwargs.get("current_price", 0),
            dcf_result=kwargs.get("dcf_result"),
            graham_result=kwargs.get("graham_result"),
            ddm_result=kwargs.get("ddm_result"),
            comparable_result=kwargs.get("comparable_result"),
        )

    def calculate_summary(
        self,
        symbol: str,
        current_price: float,
        dcf_result: Optional[DCFOutput] = None,
        graham_result: Optional[GrahamOutput] = None,
        ddm_result: Optional[DDMOutput] = None,
        comparable_result: Optional[ComparableOutput] = None,
    ) -> ValuationSummary:
        """Create combined valuation summary.

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            dcf_result: Optional DCF calculation result
            graham_result: Optional Graham calculation result
            ddm_result: Optional DDM calculation result
            comparable_result: Optional comparable calculation result

        Returns:
            ValuationSummary with aggregate assessment
        """
        self.context.start()

        try:
            methods_used = []
            valuations = {}

            if dcf_result:
                methods_used.append(ValuationMethod.DCF)
                valuations["DCF"] = dcf_result.intrinsic_value_per_share

            if graham_result:
                methods_used.append(ValuationMethod.GRAHAM)
                valuations["Graham"] = graham_result.intrinsic_value

            if ddm_result:
                methods_used.append(ValuationMethod.DDM)
                valuations["DDM"] = ddm_result.intrinsic_value

            if comparable_result:
                methods_used.append(ValuationMethod.COMPARABLE)
                valuations["Comparable"] = comparable_result.average_intrinsic_value

            if not valuations:
                raise CalculationError("At least one valuation method result is required")

            values = list(valuations.values())
            avg_value = sum(values) / len(values)
            sorted_values = sorted(values)
            med_value = sorted_values[len(sorted_values) // 2]
            value_range = (min(values), max(values))

            # Calculate spread to determine confidence
            spread = (max(values) - min(values)) / avg_value if avg_value else 0

            if spread < 0.15:
                confidence = "high"
            elif spread < 0.30:
                confidence = "medium"
            else:
                confidence = "low"

            # Overall verdict
            upside = (avg_value - current_price) / current_price * 100
            if upside > 20:
                overall_verdict = "undervalued"
            elif upside < -15:
                overall_verdict = "overvalued"
            else:
                overall_verdict = "fairly_valued"

            # Generate recommendations
            recommendations = []
            if overall_verdict == "undervalued":
                recommendations.append(
                    f"Stock appears undervalued with {upside:.1f}% potential upside."
                )
                if confidence == "high":
                    recommendations.append(
                        "Multiple valuation methods agree - higher confidence in assessment."
                    )
            elif overall_verdict == "overvalued":
                recommendations.append(
                    f"Stock appears overvalued, trading {-upside:.1f}% above intrinsic value."
                )
            else:
                recommendations.append("Stock appears to be trading near fair value.")

            if confidence == "low":
                recommendations.append(
                    "Large spread between methods suggests higher uncertainty in valuation."
                )

            self.context.complete()

            return ValuationSummary(
                symbol=symbol,
                methods_used=methods_used,
                valuations=valuations,
                average_value=avg_value,
                median_value=med_value,
                range=value_range,
                current_price=current_price,
                overall_verdict=overall_verdict,
                confidence=confidence,
                recommendations=recommendations,
                calculation_time_ms=self.context.elapsed_ms,
            )

        except Exception as e:
            self.context.fail(str(e))
            raise CalculationError(f"Valuation summary failed: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def calculate_dcf(
    symbol: str,
    current_fcf: float,
    growth_rates: list[float],
    discount_rate: float,
    shares_outstanding: float,
    terminal_growth: float = 0.025,
    current_price: Optional[float] = None,
    cash: float = 0.0,
    debt: float = 0.0,
) -> DCFOutput:
    """Convenience function for DCF valuation.

    Args:
        symbol: Stock symbol
        current_fcf: Current free cash flow (millions)
        growth_rates: Projected growth rates
        discount_rate: WACC/discount rate
        shares_outstanding: Shares outstanding (millions)
        terminal_growth: Terminal growth rate
        current_price: Current stock price
        cash: Cash and equivalents
        debt: Total debt

    Returns:
        DCFOutput
    """
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
    config = DCFConfig()
    calc = DCFCalculator()
    return calc.calculate(data, config)


def calculate_graham(
    symbol: str,
    eps: float,
    growth_rate: float,
    current_price: float,
    aaa_yield: float = 0.044,
) -> GrahamOutput:
    """Convenience function for Graham Formula valuation.

    Args:
        symbol: Stock symbol
        eps: Earnings per share
        growth_rate: Expected 5-year growth
        current_price: Current stock price
        aaa_yield: AAA bond yield

    Returns:
        GrahamOutput
    """
    data = GrahamInputData(
        symbol=symbol,
        eps=eps,
        growth_rate=growth_rate,
        current_price=current_price,
        aaa_yield=aaa_yield,
    )
    config = GrahamConfig()
    calc = GrahamCalculator()
    return calc.calculate(data, config)


def calculate_ddm(
    symbol: str,
    current_dividend: float,
    dividend_growth: float,
    required_return: float,
    current_price: float,
    model_type: DDMType = DDMType.GORDON,
) -> DDMOutput:
    """Convenience function for DDM valuation.

    Args:
        symbol: Stock symbol
        current_dividend: Annual dividend
        dividend_growth: Dividend growth rate
        required_return: Required return
        current_price: Current stock price
        model_type: DDM model type

    Returns:
        DDMOutput
    """
    data = DDMInputData(
        symbol=symbol,
        current_dividend=current_dividend,
        dividend_growth=dividend_growth,
        required_return=required_return,
        current_price=current_price,
    )
    config = DDMConfig(model_type=model_type)
    calc = DDMCalculator()
    return calc.calculate(data, config)
