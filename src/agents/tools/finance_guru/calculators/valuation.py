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
    ) -> DCFOutput:
        """Calculate DCF intrinsic value.

        Args:
            data: Input data with FCF, growth rates, discount rate
            config: Configuration with projection years, margin of safety

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

            # Sensitivity analysis
            sensitivity = self._sensitivity_analysis(data, config)

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
                sensitivity=sensitivity,
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
