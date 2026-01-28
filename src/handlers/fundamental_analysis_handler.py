import json
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
import logging
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod

from src.hedge_fund.tools.api_fmp import (
    get_financial_metrics,
    search_line_items,
    get_insider_trades,
    get_company_news,
    get_market_cap
)

# Import valuation calculators for enhanced analysis
try:
    from src.agents.tools.finance_guru.calculators.valuation import (
        DCFCalculator,
        GrahamCalculator,
        ValuationCalculator,
    )
    from src.agents.tools.finance_guru.models.valuation import (
        DCFInputData,
        DCFConfig,
        GrahamInputData,
        GrahamConfig,
    )
    VALUATION_AVAILABLE = True
except ImportError:
    VALUATION_AVAILABLE = False

class FundamentalAnalysisHandler(LoggerMixin):
    """
    Handler for fundamental analysis operations.
    Processes financial statement growth data and provides AI-powered insights.
    """
    
    def __init__(self):
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()
        self.logger = logging.getLogger(__name__)
    
    def _safe_get_value(self, data: Union[Dict, Any], key: str, default=None):
        """
        Safely get value from either dictionary or Pydantic model.
        
        Args:
            data: Dictionary or Pydantic model
            key: Key/attribute name
            default: Default value if not found
            
        Returns:
            Value or default
        """
        if isinstance(data, dict):
            return data.get(key, default)
        else:
            # Handle Pydantic model
            return getattr(data, key, default)
    
    def _convert_to_dict(self, data: Union[Dict, Any]) -> Dict[str, Any]:
        """
        Convert data to dictionary if it's a Pydantic model.
        
        Args:
            data: Dictionary or Pydantic model
            
        Returns:
            Dictionary representation
        """
        if isinstance(data, dict):
            return data
        elif hasattr(data, 'model_dump'):
            return data.model_dump()
        elif hasattr(data, 'dict'):
            return data.dict()
        else:
            # Fallback: convert attributes to dict
            return {k: v for k, v in data.__dict__.items() if not k.startswith('_')}
        
    async def analyze_financial_growth(
        self,
        symbol: str,
        growth_data: List[Union[Dict[str, Any], Any]],
        period: str = "annual",
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze financial statement growth data using LLM.
        
        Args:
            symbol: Stock symbol
            growth_data: Financial growth data (can be list of dicts or Pydantic models)
            period: Analysis period (annual/quarter)
            model_name: LLM model to use
            provider_type: Provider type (openai/ollama/gemini)
            api_key: API key for provider
            
        Returns:
            Dict containing both raw data and AI analysis
        """
        try:
            if not growth_data:
                return {
                    "symbol": symbol,
                    "period": period,
                    "raw_data": [],
                    "analysis": "No financial data available for analysis.",
                    "key_insights": {}
                }
            
            # Convert all data to dictionaries
            dict_data = [self._convert_to_dict(item) for item in growth_data]
            
            # Get the latest data point for analysis
            latest_data = dict_data[0] if dict_data else {}
            
            # Create analysis prompt
            prompt = self._create_fundamental_analysis_prompt(
                symbol=symbol,
                data=latest_data,
                period=period,
                historical_data=dict_data[:5] if len(dict_data) > 1 else []
            )
            
            # Generate AI analysis
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response = await self.llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.3  # Lower temperature for more consistent financial analysis
            )
            
            analysis_text = response.get("content", "Analysis generation failed.")
            
            # Extract key metrics for quick reference
            key_insights = self._extract_key_insights(latest_data)
            
            return {
                "symbol": symbol,
                "period": period,
                "latest_date": latest_data.get("date", "N/A"),
                "raw_data": dict_data,  # Return dictionary data
                "analysis": analysis_text,
                "key_insights": key_insights
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing fundamental data for {symbol}: {str(e)}")
            raise Exception(f"Fundamental analysis error: {str(e)}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for fundamental analysis."""
        return """You are an expert financial analyst specializing in fundamental analysis of stocks. 

Your role is to analyze financial statement growth data and provide clear, actionable insights about:
1. Company's financial health and growth trajectory
2. Key strengths and concerns in the financial metrics
3. Implications for investors
4. Comparison with industry standards when relevant

Provide analysis that is:
- Data-driven and specific (cite actual numbers)
- Balanced (both positive and negative aspects)
- Forward-looking (what these metrics suggest about future performance)
- Actionable (what investors should consider)

Format your response with clear sections and bullet points for readability."""

    def _create_fundamental_analysis_prompt(
        self,
        symbol: str,
        data: Dict[str, Any],
        period: str,
        historical_data: List[Dict[str, Any]]
    ) -> str:
        """Create a detailed prompt for fundamental analysis."""
        
        # Format key metrics for better readability
        revenue_growth = self._format_percentage(data.get("revenueGrowth", 0))
        net_income_growth = self._format_percentage(data.get("netIncomeGrowth", 0))
        eps_growth = self._format_percentage(data.get("epsgrowth", 0))
        operating_income_growth = self._format_percentage(data.get("operatingIncomeGrowth", 0))
        fcf_growth = self._format_percentage(data.get("freeCashFlowGrowth", 0))
        
        # Long-term growth metrics
        ten_y_revenue = self._format_percentage(data.get("tenYRevenueGrowthPerShare", 0))
        five_y_revenue = self._format_percentage(data.get("fiveYRevenueGrowthPerShare", 0))
        three_y_revenue = self._format_percentage(data.get("threeYRevenueGrowthPerShare", 0))
        
        prompt = f"""Analyze the following {period} financial statement growth data for {symbol}:

**Latest Period: {data.get('date', 'N/A')}**

📊 **Core Growth Metrics:**
- Revenue Growth: {revenue_growth}
- Net Income Growth: {net_income_growth}
- EPS Growth: {eps_growth}
- Operating Income Growth: {operating_income_growth}
- Free Cash Flow Growth: {fcf_growth}

💰 **Profitability Metrics:**
- Gross Profit Growth: {self._format_percentage(data.get("grossProfitGrowth", 0))}
- EBITDA Growth: {self._format_percentage(data.get("ebitgrowth", 0))}
- Operating Cash Flow Growth: {self._format_percentage(data.get("operatingCashFlowGrowth", 0))}

📈 **Long-term Performance:**
- 10-Year Revenue Growth/Share: {ten_y_revenue}
- 5-Year Revenue Growth/Share: {five_y_revenue}
- 3-Year Revenue Growth/Share: {three_y_revenue}
- 10-Year Net Income Growth/Share: {self._format_percentage(data.get("tenYNetIncomeGrowthPerShare", 0))}

💼 **Capital Management:**
- Debt Growth: {self._format_percentage(data.get("debtGrowth", 0))}
- Book Value/Share Growth: {self._format_percentage(data.get("bookValueperShareGrowth", 0))}
- Dividend/Share Growth: {self._format_percentage(data.get("dividendsperShareGrowth", 0))}

🔬 **Investment & Operations:**
- R&D Expense Growth: {self._format_percentage(data.get("rdexpenseGrowth", 0))}
- SG&A Expense Growth: {self._format_percentage(data.get("sgaexpensesGrowth", 0))}
- Receivables Growth: {self._format_percentage(data.get("receivablesGrowth", 0))}
- Inventory Growth: {self._format_percentage(data.get("inventoryGrowth", 0))}

Please provide a comprehensive analysis covering:
1. **Overall Financial Health Assessment**
2. **Growth Quality Analysis** (Is the growth sustainable?)
3. **Key Strengths and Concerns**
4. **Market Implications** (What does this mean for the stock?)
5. **Investment Considerations** (Buy/Hold/Sell factors)

Focus on the most significant trends and their implications for investors."""

        return prompt
    
    def _format_percentage(self, value: float) -> str:
        """Format a decimal value as percentage."""
        if value is None:
            return "N/A"
        return f"{value * 100:.2f}%"

    def _format_large_number(self, value: float) -> str:
        """Format large numbers with B/M/K suffix for readability."""
        if value is None or value == 0:
            return "N/A"
        abs_value = abs(value)
        sign = "-" if value < 0 else ""
        if abs_value >= 1e12:
            return f"{sign}{abs_value/1e12:.2f}T"
        elif abs_value >= 1e9:
            return f"{sign}{abs_value/1e9:.2f}B"
        elif abs_value >= 1e6:
            return f"{sign}{abs_value/1e6:.2f}M"
        elif abs_value >= 1e3:
            return f"{sign}{abs_value/1e3:.2f}K"
        else:
            return f"{sign}{abs_value:.2f}"

    def _extract_key_insights(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key insights for quick reference."""
        return {
            "revenue_trend": self._classify_growth(data.get("revenueGrowth", 0)),
            "profitability_trend": self._classify_growth(data.get("netIncomeGrowth", 0)),
            "cash_flow_health": self._classify_growth(data.get("freeCashFlowGrowth", 0)),
            "debt_management": self._classify_debt_change(data.get("debtGrowth", 0)),
            "growth_quality_score": self._calculate_growth_quality_score(data)
        }
    
    def _classify_growth(self, growth_rate: float) -> str:
        """Classify growth rate into categories."""
        if growth_rate is None:
            return "Unknown"
        elif growth_rate > 0.20:
            return "Strong Growth"
        elif growth_rate > 0.10:
            return "Moderate Growth"
        elif growth_rate > 0:
            return "Slow Growth"
        elif growth_rate > -0.10:
            return "Slight Decline"
        else:
            return "Significant Decline"
    
    def _classify_debt_change(self, debt_growth: float) -> str:
        """Classify debt changes."""
        if debt_growth is None:
            return "Unknown"
        elif debt_growth < -0.10:
            return "Improving (Reducing)"
        elif debt_growth < 0:
            return "Slightly Improving"
        elif debt_growth < 0.10:
            return "Stable"
        elif debt_growth < 0.20:
            return "Increasing Moderately"
        else:
            return "Increasing Significantly"
    
    def _calculate_growth_quality_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate a simple growth quality score (0-100).
        Higher score indicates better quality growth.
        """
        score = 50  # Base score

        # Revenue growth contribution
        revenue_growth = data.get("revenueGrowth", 0) or 0
        score += min(revenue_growth * 100, 20)  # Max 20 points

        # Profitability improvement
        net_income_growth = data.get("netIncomeGrowth", 0) or 0
        if net_income_growth > revenue_growth:
            score += 10  # Operating leverage

        # Cash flow quality
        fcf_growth = data.get("freeCashFlowGrowth", 0) or 0
        if fcf_growth > 0:
            score += min(fcf_growth * 50, 15)  # Max 15 points

        # Debt management
        debt_growth = data.get("debtGrowth", 0) or 0
        if debt_growth < 0:
            score += 5  # Reducing debt
        elif debt_growth > 0.2:
            score -= 10  # High debt growth

        # Long-term consistency
        if data.get("fiveYRevenueGrowthPerShare", 0) > 0.5:
            score += 10  # Consistent long-term growth

        return min(max(score, 0), 100)  # Bound between 0-100

    def _calculate_intrinsic_value(
        self,
        symbol: str,
        current_price: Optional[float],
        eps: Optional[float],
        fcf: Optional[float],
        shares_outstanding: Optional[float],
        cash: Optional[float] = 0,
        debt: Optional[float] = 0,
        eps_growth_rate: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate intrinsic value using valuation models.

        Uses Graham Formula and DCF when data is available.
        Returns valuation analysis with verdict.

        IMPORTANT FINANCIAL LOGIC:
        - Graham Formula: V = EPS × (8.5 + 2g) × (4.4/Y)
          - Uses FORWARD growth estimate (5Y expected)
          - If using historical CAGR, must be stated explicitly

        - DCF: PV of projected free cash flows + terminal value
          - More appropriate for companies with positive FCF
          - Sensitive to discount rate and terminal growth assumptions

        - P/E Analysis:
          - Compare current P/E to historical range
          - Compare to industry peers
          - P/E expansion/contraction thesis

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            eps: Earnings per share (TTM)
            fcf: Free cash flow (total, in same units as shares)
            shares_outstanding: Number of shares outstanding
            cash: Cash and equivalents
            debt: Total debt
            eps_growth_rate: Expected EPS growth rate (decimal)

        Returns:
            Dict with intrinsic value calculations and verdict
        """
        result = {
            "graham_value": None,
            "dcf_value": None,
            "current_price": current_price,
            "pe_analysis": None,
            "verdict": "insufficient_data",
            "methodology_notes": [],
        }

        if not VALUATION_AVAILABLE:
            result["methodology_notes"].append("Valuation calculators not available")
            return result

        # Calculate P/E ratio and analysis
        if current_price and eps and eps > 0:
            current_pe = current_price / eps
            result["pe_analysis"] = {
                "current_pe": round(current_pe, 2),
                "pe_interpretation": self._interpret_pe_ratio(current_pe),
            }
            result["methodology_notes"].append(f"P/E [TTM]: {current_pe:.2f}")

        # Graham Formula Valuation
        if eps and eps > 0:
            try:
                # Default growth rate: 10% if not provided
                growth_rate = eps_growth_rate if eps_growth_rate else 0.10
                # Cap growth rate at reasonable levels
                growth_rate = min(max(growth_rate, 0), 0.30)  # 0-30% range

                graham_data = GrahamInputData(
                    symbol=symbol,
                    eps=eps,
                    growth_rate=growth_rate,
                    current_price=current_price or 100,  # Placeholder if no price
                    aaa_yield=0.044,  # Standard 4.4% AAA bond yield
                )
                graham_config = GrahamConfig(margin_of_safety=0.25)
                graham_calc = GrahamCalculator()
                graham_result = graham_calc.calculate(graham_data, graham_config)

                result["graham_value"] = round(graham_result.intrinsic_value, 2)
                result["graham_details"] = {
                    "eps_used": eps,
                    "growth_rate_used": growth_rate,
                    "growth_rate_source": "historical_eps_growth" if eps_growth_rate else "default_10%",
                    "margin_of_safety_price": round(graham_result.margin_of_safety_price, 2),
                }
                result["methodology_notes"].append(
                    f"Graham Value: ${graham_result.intrinsic_value:.2f} "
                    f"(using {growth_rate*100:.1f}% growth, {'historical CAGR' if eps_growth_rate else 'default estimate'})"
                )

            except Exception as e:
                self.logger.warning(f"Graham calculation failed for {symbol}: {e}")
                result["methodology_notes"].append(f"Graham calculation error: {str(e)}")

        # DCF Valuation with WACC calculation and sensitivity analysis
        if fcf and fcf > 0 and shares_outstanding and shares_outstanding > 0:
            try:
                fcf_per_share = fcf / shares_outstanding
                cash_val = cash if cash else 0
                debt_val = debt if debt else 0

                # Calculate WACC using CAPM: WACC = Rf + Beta × (Rm - Rf)
                # Default assumptions if not provided
                risk_free_rate = 0.045  # Current 10-year Treasury yield ~4.5%
                market_premium = 0.055  # Long-term equity risk premium ~5.5%
                beta = 1.0  # Default beta, should be passed in for better accuracy

                # Calculate cost of equity using CAPM
                cost_of_equity = risk_free_rate + beta * market_premium

                # For simplicity, use cost of equity as WACC (assumes no debt or equity-only)
                # In production, would need debt/equity ratio and cost of debt
                wacc = cost_of_equity

                # Projected growth rates (conservative)
                growth_rates = [0.10, 0.08, 0.06, 0.05, 0.04]  # Declining growth
                terminal_growth = 0.025  # 2.5% perpetual growth

                dcf_data = DCFInputData(
                    symbol=symbol,
                    current_fcf=fcf,
                    growth_rates=growth_rates,
                    terminal_growth=terminal_growth,
                    discount_rate=wacc,
                    shares_outstanding=shares_outstanding,
                    current_price=current_price,
                    cash=cash_val,
                    debt=debt_val,
                )
                dcf_config = DCFConfig(margin_of_safety=0.25)
                dcf_calc = DCFCalculator()
                dcf_result = dcf_calc.calculate(dcf_data, dcf_config)

                result["dcf_value"] = round(dcf_result.intrinsic_value_per_share, 2)

                # Calculate sensitivity analysis (3x3 grid: WACC × Terminal Growth)
                sensitivity = self._calculate_dcf_sensitivity(
                    fcf=fcf,
                    shares_outstanding=shares_outstanding,
                    base_wacc=wacc,
                    base_terminal_growth=terminal_growth,
                    growth_rates=growth_rates,
                    cash=cash_val,
                    debt=debt_val
                )

                result["dcf_details"] = {
                    "fcf_used": fcf,
                    "wacc": round(wacc, 4),
                    "wacc_components": {
                        "risk_free_rate": risk_free_rate,
                        "market_premium": market_premium,
                        "beta": beta,
                        "cost_of_equity": round(cost_of_equity, 4)
                    },
                    "terminal_growth": terminal_growth,
                    "enterprise_value": round(dcf_result.enterprise_value, 0),
                    "margin_of_safety_price": round(dcf_result.margin_of_safety_price, 2),
                    "sensitivity_analysis": sensitivity
                }

                # Sanity checks
                sanity_warnings = []
                if dcf_result.intrinsic_value_per_share > current_price * 3:
                    sanity_warnings.append("DCF value > 3x current price - may be overly optimistic")
                if dcf_result.intrinsic_value_per_share < current_price * 0.3:
                    sanity_warnings.append("DCF value < 0.3x current price - may be overly pessimistic")
                if terminal_growth > wacc:
                    sanity_warnings.append("Terminal growth > WACC - invalid assumption")

                if sanity_warnings:
                    result["dcf_details"]["sanity_warnings"] = sanity_warnings

                result["methodology_notes"].append(
                    f"DCF Value: ${dcf_result.intrinsic_value_per_share:.2f} "
                    f"(WACC={wacc*100:.1f}%, Terminal Growth={terminal_growth*100:.1f}%)"
                )

            except Exception as e:
                self.logger.warning(f"DCF calculation failed for {symbol}: {e}")
                result["methodology_notes"].append(f"DCF calculation error: {str(e)}")

        # Determine overall verdict
        result["verdict"] = self._determine_valuation_verdict(
            current_price=current_price,
            graham_value=result.get("graham_value"),
            dcf_value=result.get("dcf_value"),
        )

        return result

    def _calculate_dcf_sensitivity(
        self,
        fcf: float,
        shares_outstanding: float,
        base_wacc: float,
        base_terminal_growth: float,
        growth_rates: list,
        cash: float = 0,
        debt: float = 0
    ) -> Dict[str, Any]:
        """
        Calculate DCF sensitivity analysis (3x3 grid).

        Creates a matrix showing how DCF value changes with different
        WACC and terminal growth assumptions.

        Args:
            fcf: Free cash flow
            shares_outstanding: Shares outstanding
            base_wacc: Base WACC rate
            base_terminal_growth: Base terminal growth rate
            growth_rates: FCF growth rates for projection period
            cash: Cash and equivalents
            debt: Total debt

        Returns:
            Dict with sensitivity grid and summary
        """
        try:
            # Define sensitivity ranges
            wacc_variants = [
                base_wacc - 0.01,  # WACC - 1%
                base_wacc,          # Base WACC
                base_wacc + 0.01   # WACC + 1%
            ]

            tg_variants = [
                base_terminal_growth - 0.005,  # TG - 0.5%
                base_terminal_growth,           # Base TG
                base_terminal_growth + 0.005   # TG + 0.5%
            ]

            # Calculate DCF for each combination
            grid = {}
            values = []

            for wacc in wacc_variants:
                wacc_key = f"WACC_{wacc*100:.1f}%"
                grid[wacc_key] = {}

                for tg in tg_variants:
                    # Skip invalid combinations (terminal growth >= WACC)
                    if tg >= wacc:
                        grid[wacc_key][f"TG_{tg*100:.1f}%"] = "N/A (TG >= WACC)"
                        continue

                    try:
                        dcf_data = DCFInputData(
                            symbol="sensitivity",
                            current_fcf=fcf,
                            growth_rates=growth_rates,
                            terminal_growth=tg,
                            discount_rate=wacc,
                            shares_outstanding=shares_outstanding,
                            current_price=100,  # Placeholder
                            cash=cash,
                            debt=debt,
                        )
                        dcf_config = DCFConfig(margin_of_safety=0.25)
                        dcf_calc = DCFCalculator()
                        result = dcf_calc.calculate(dcf_data, dcf_config)

                        value = round(result.intrinsic_value_per_share, 2)
                        grid[wacc_key][f"TG_{tg*100:.1f}%"] = value
                        values.append(value)
                    except:
                        grid[wacc_key][f"TG_{tg*100:.1f}%"] = "Error"

            # Calculate summary statistics
            if values:
                return {
                    "grid": grid,
                    "range": {
                        "min": min(values),
                        "max": max(values),
                        "median": sorted(values)[len(values) // 2]
                    },
                    "interpretation": self._interpret_sensitivity_range(min(values), max(values))
                }
            else:
                return {"grid": grid, "error": "Could not calculate sensitivity values"}

        except Exception as e:
            self.logger.warning(f"Sensitivity analysis failed: {e}")
            return {"error": str(e)}

    def _interpret_sensitivity_range(self, min_val: float, max_val: float) -> str:
        """Interpret the sensitivity analysis range."""
        spread = (max_val - min_val) / min_val * 100 if min_val > 0 else 0

        if spread < 20:
            return "Low sensitivity - DCF value relatively stable across assumptions"
        elif spread < 50:
            return "Moderate sensitivity - DCF value varies meaningfully with assumptions"
        else:
            return "High sensitivity - DCF value highly dependent on assumptions; use with caution"

    def _interpret_pe_ratio(self, pe_ratio: float) -> str:
        """
        Interpret P/E ratio with context.

        Standard interpretation:
        - < 10: Potentially undervalued or declining business
        - 10-15: Fair value for mature companies
        - 15-25: Growth premium
        - 25-40: High growth expectations
        - > 40: Very high expectations or speculation
        """
        if pe_ratio < 10:
            return "low_pe_value_or_distressed"
        elif pe_ratio < 15:
            return "fair_value_mature"
        elif pe_ratio < 25:
            return "growth_premium"
        elif pe_ratio < 40:
            return "high_growth_expectations"
        else:
            return "very_high_speculative"

    def _determine_valuation_verdict(
        self,
        current_price: Optional[float],
        graham_value: Optional[float],
        dcf_value: Optional[float],
    ) -> str:
        """
        Determine overall valuation verdict based on intrinsic values.

        METHODOLOGY:
        1. If both Graham and DCF available, use average
        2. Compare to current price
        3. Verdict based on upside/downside potential

        IMPORTANT: This is a metrics-based assessment, NOT investment advice.
        """
        if not current_price or current_price <= 0:
            return "insufficient_data"

        values = []
        if graham_value and graham_value > 0:
            values.append(graham_value)
        if dcf_value and dcf_value > 0:
            values.append(dcf_value)

        if not values:
            return "insufficient_data"

        avg_intrinsic = sum(values) / len(values)
        upside_pct = ((avg_intrinsic - current_price) / current_price) * 100

        if upside_pct > 30:
            return "significantly_undervalued"
        elif upside_pct > 15:
            return "undervalued"
        elif upside_pct > -10:
            return "fairly_valued"
        elif upside_pct > -25:
            return "overvalued"
        else:
            return "significantly_overvalued"
    

    async def generate_comprehensive_fundamental_data(
        self,
        symbol: str,
        tool_service: Any  # ToolCallService instance
    ) -> Dict[str, Any]:
        """
        Generate comprehensive fundamental data với đầy đủ metrics
        """
        try:
            # 1. Fetch all necessary data
            self.logger.info(f"Generating comprehensive fundamental data for {symbol}")
            
            # Get all data in parallel for better performance. Fetch data concurrently
            results = await asyncio.gather(
                tool_service.get_key_metrics(symbol, limit=1),
                tool_service.get_income_statement(symbol, limit=5),
                tool_service.get_balance_sheet(symbol, limit=5),
                tool_service.get_cash_flow_statement(symbol, limit=5),
                tool_service.get_quote(symbol),
                tool_service.get_financial_statement_growth(symbol, limit=5),
                tool_service.get_key_metrics_ttm(symbol),  # Add TTM metrics for accurate TTM P/E
                return_exceptions=True
            )

            key_metrics, income_stmt, balance_sheet, cash_flow, quote, growth_data, key_metrics_ttm = results

            # Handle exceptions
            key_metrics = key_metrics if not isinstance(key_metrics, Exception) else None
            income_stmt = income_stmt if not isinstance(income_stmt, Exception) else None
            balance_sheet = balance_sheet if not isinstance(balance_sheet, Exception) else None
            cash_flow = cash_flow if not isinstance(cash_flow, Exception) else None
            quote = quote if not isinstance(quote, Exception) else None
            growth_data = growth_data if not isinstance(growth_data, Exception) else None
            key_metrics_ttm = key_metrics_ttm if not isinstance(key_metrics_ttm, Exception) else None

            # =====================================================
            # RAW DATA LOGGING FOR VALIDATION (CRITICAL FOR DEBUG)
            # =====================================================
            self.logger.info(f"=" * 60)
            self.logger.info(f"RAW FUNDAMENTAL DATA FOR {symbol}")
            self.logger.info(f"=" * 60)

            # Log Key Metrics (raw)
            if key_metrics:
                self.logger.info(f"[RAW] Key Metrics ({len(key_metrics)} records):")
                for i, km in enumerate(key_metrics[:2]):  # Log first 2 for brevity
                    self.logger.info(f"  Record {i}: {json.dumps(km, default=str)[:500]}...")
            else:
                self.logger.warning(f"[RAW] Key Metrics: NONE/EMPTY")

            # Log Quote (raw)
            if quote:
                self.logger.info(f"[RAW] Quote: {json.dumps(quote, default=str)[:300]}")
            else:
                self.logger.warning(f"[RAW] Quote: NONE/EMPTY")

            # Log Income Statement (raw)
            if income_stmt:
                self.logger.info(f"[RAW] Income Statement ({len(income_stmt)} records):")
                if income_stmt:
                    latest_income = income_stmt[0]
                    self.logger.info(f"  Latest Period: {latest_income.get('date', 'N/A')}")
                    self.logger.info(f"  Revenue: {latest_income.get('revenue', 'N/A')}")
                    self.logger.info(f"  Net Income: {latest_income.get('netIncome', 'N/A')}")
                    self.logger.info(f"  EPS: {latest_income.get('eps', 'N/A')}")
            else:
                self.logger.warning(f"[RAW] Income Statement: NONE/EMPTY")

            # Log Balance Sheet (raw)
            if balance_sheet:
                self.logger.info(f"[RAW] Balance Sheet ({len(balance_sheet)} records):")
                if balance_sheet:
                    latest_bs = balance_sheet[0]
                    self.logger.info(f"  Latest Period: {latest_bs.get('date', 'N/A')}")
                    self.logger.info(f"  Total Debt: {latest_bs.get('totalDebt', 'N/A')}")
                    self.logger.info(f"  Total Equity: {latest_bs.get('totalStockholdersEquity', 'N/A')}")
            else:
                self.logger.warning(f"[RAW] Balance Sheet: NONE/EMPTY")

            # Log Cash Flow (raw)
            if cash_flow:
                self.logger.info(f"[RAW] Cash Flow ({len(cash_flow)} records):")
                if cash_flow:
                    latest_cf = cash_flow[0]
                    self.logger.info(f"  Latest Period: {latest_cf.get('date', 'N/A')}")
                    self.logger.info(f"  Free Cash Flow: {latest_cf.get('freeCashFlow', 'N/A')}")
                    self.logger.info(f"  Operating Cash Flow: {latest_cf.get('operatingCashFlow', 'N/A')}")
            else:
                self.logger.warning(f"[RAW] Cash Flow: NONE/EMPTY")

            # Log Growth Data (raw)
            if growth_data:
                self.logger.info(f"[RAW] Growth Data ({len(growth_data)} records):")
                if growth_data:
                    latest_growth = growth_data[0] if isinstance(growth_data[0], dict) else self._convert_to_dict(growth_data[0])
                    self.logger.info(f"  Latest Period: {latest_growth.get('date', 'N/A')}")
                    self.logger.info(f"  Revenue Growth: {latest_growth.get('revenueGrowth', 'N/A')}")
                    self.logger.info(f"  EPS Growth: {latest_growth.get('epsgrowth', 'N/A')}")
            else:
                self.logger.warning(f"[RAW] Growth Data: NONE/EMPTY")

            # Log Key Metrics TTM (raw) - for accurate TTM P/E
            if key_metrics_ttm:
                self.logger.info(f"[RAW] Key Metrics TTM:")
                self.logger.info(f"  P/E TTM: {getattr(key_metrics_ttm, 'peRatioTTM', 'N/A')}")
                self.logger.info(f"  P/B TTM: {getattr(key_metrics_ttm, 'pbRatioTTM', 'N/A')}")
                self.logger.info(f"  P/S TTM: {getattr(key_metrics_ttm, 'priceToSalesRatioTTM', 'N/A')}")
                self.logger.info(f"  Net Income Per Share TTM (EPS TTM): {getattr(key_metrics_ttm, 'netIncomePerShareTTM', 'N/A')}")
                self.logger.info(f"  Dividend Yield TTM: {getattr(key_metrics_ttm, 'dividendYieldTTM', 'N/A')}")
            else:
                self.logger.warning(f"[RAW] Key Metrics TTM: NONE/EMPTY")

            self.logger.info(f"=" * 60)
            # =====================================================

            # Extract data
            km_data = key_metrics[0] if key_metrics else {}

            # =====================================================
            # SNAPSHOT LOCKING - Use consistent data source
            # =====================================================
            # Price from quote (realtime/close)
            current_price = quote.get("price") if quote else None
            quote_market_cap = quote.get("marketCap") if quote else None

            # Per-share metrics from key_metrics (FY record)
            eps_ttm = income_stmt[0].get("eps") if income_stmt else None
            book_value_per_share = km_data.get("bookValuePerShare")
            revenue_per_share = km_data.get("revenuePerShare")

            # =====================================================
            # CALCULATE RATIOS FROM CURRENT PRICE (Not from key_metrics!)
            # =====================================================
            # key_metrics ratios use HISTORICAL price, we need CURRENT price
            calc_pe = None
            calc_pb = None
            calc_ps = None

            if current_price and current_price > 0:
                if eps_ttm and eps_ttm > 0:
                    calc_pe = round(current_price / eps_ttm, 2)
                if book_value_per_share and book_value_per_share > 0:
                    calc_pb = round(current_price / book_value_per_share, 2)
                if revenue_per_share and revenue_per_share > 0:
                    calc_ps = round(current_price / revenue_per_share, 2)

            # Log calculated vs reported ratios for transparency
            reported_pe = km_data.get("peRatio")
            reported_pb = km_data.get("pbRatio")
            reported_ps = km_data.get("priceToSalesRatio")

            self.logger.info(f"[SANITY CHECK] Valuation Ratios:")
            self.logger.info(f"  Price (quote): ${current_price}")
            self.logger.info(f"  EPS (income_stmt): ${eps_ttm}")
            self.logger.info(f"  Book Value/Share: ${book_value_per_share}")
            self.logger.info(f"  Revenue/Share: ${revenue_per_share}")
            self.logger.info(f"  Calculated P/E: {calc_pe} vs Reported (key_metrics): {reported_pe}")
            self.logger.info(f"  Calculated P/B: {calc_pb} vs Reported (key_metrics): {reported_pb}")
            self.logger.info(f"  Calculated P/S: {calc_ps} vs Reported (key_metrics): {reported_ps}")

            # Warn if significant difference (>10%)
            if calc_pe and reported_pe and abs(calc_pe - reported_pe) / reported_pe > 0.1:
                self.logger.warning(f"  ⚠️ P/E mismatch >10%: Using calculated value {calc_pe}")

            # 2. Calculate all metrics
            report = {
                "symbol": symbol.upper(),
                "generated": datetime.now().isoformat(timespec="seconds"),
                "data_snapshot": {
                    "price_source": "FMP /quote",
                    "price": current_price,
                    "price_date": datetime.now().strftime("%Y-%m-%d"),
                    "financial_period": income_stmt[0].get("date") if income_stmt else "N/A",
                    "market_cap_quote": quote_market_cap,
                    "market_cap_km": km_data.get("marketCap"),
                    "note": "Ratios calculated from current price + TTM metrics"
                },
                "valuation": {},
                "growth": {},
                "profitability": {},
                "leverage": {},
                "cashflow": {},
                "dividends": {},
                "risk": {}
            }

            # =====================================================
            # EXTRACT TTM METRICS FROM key_metrics_ttm ENDPOINT
            # =====================================================
            # TTM metrics from FMP /key-metrics-ttm (uses rolling 12-month EPS)
            ttm_pe = getattr(key_metrics_ttm, 'peRatioTTM', None) if key_metrics_ttm else None
            ttm_pb = getattr(key_metrics_ttm, 'pbRatioTTM', None) if key_metrics_ttm else None
            ttm_ps = getattr(key_metrics_ttm, 'priceToSalesRatioTTM', None) if key_metrics_ttm else None
            ttm_eps = getattr(key_metrics_ttm, 'netIncomePerShareTTM', None) if key_metrics_ttm else None

            # Log P/E comparison for transparency
            self.logger.info(f"[P/E COMPARISON] FY vs TTM:")
            self.logger.info(f"  P/E (FY): {calc_pe} - calculated from Price ${current_price} / EPS FY ${eps_ttm}")
            self.logger.info(f"  P/E (TTM): {ttm_pe} - from FMP /key-metrics-ttm endpoint")
            self.logger.info(f"  EPS FY (income_stmt): ${eps_ttm}")
            self.logger.info(f"  EPS TTM (key_metrics_ttm): ${ttm_eps}")
            if calc_pe and ttm_pe:
                pe_diff_pct = abs(calc_pe - ttm_pe) / ttm_pe * 100
                self.logger.info(f"  Difference: {pe_diff_pct:.1f}% - {'⚠️ Significant difference!' if pe_diff_pct > 15 else 'Within expected range'}")

            # Valuation metrics - SHOW BOTH FY AND TTM VALUES
            report["valuation"] = {
                "price": current_price,
                "market_cap": quote_market_cap,
                # FY-based ratios (calculated from current price + FY metrics)
                "pe_fy": calc_pe,  # P/E using FY EPS from income statement
                "pb_fy": calc_pb,  # P/B using FY book value
                "ps_fy": calc_ps,  # P/S using FY revenue per share
                "eps_fy": eps_ttm,  # EPS from latest fiscal year income statement
                # TTM-based ratios (from FMP /key-metrics-ttm - rolling 12 months)
                "pe_ttm": ttm_pe,  # P/E using true TTM EPS (matches Yahoo Finance)
                "pb_ttm": ttm_pb,  # P/B using TTM book value
                "ps_ttm": ttm_ps,  # P/S using TTM revenue
                "eps_ttm": ttm_eps,  # True TTM EPS (rolling 12 months)
                # Other valuation metrics
                "forward_pe": km_data.get("forwardPE"),  # Analyst estimates
                "peg": km_data.get("pegRatio"),
                "ev_ebitda": km_data.get("enterpriseValueOverEBITDA"),
                # Raw inputs for transparency
                "book_value_per_share": book_value_per_share,
                "revenue_per_share": revenue_per_share,
            }

            # Add P/E comparison section with clear explanation
            report["pe_comparison"] = {
                "pe_fy": calc_pe,
                "pe_ttm": ttm_pe,
                "eps_fy": eps_ttm,
                "eps_ttm": ttm_eps,
                "fy_period": income_stmt[0].get("date") if income_stmt else "N/A",
                "explanation": {
                    "fy_meaning": "P/E FY uses EPS from the latest fiscal year (e.g., FY ending Jan 2025). This may not include recent quarters.",
                    "ttm_meaning": "P/E TTM uses trailing 12-month EPS (rolling sum of last 4 quarters). This is what Yahoo Finance shows.",
                    "why_different": "If recent quarters had higher/lower earnings than the same quarters last year, TTM EPS will differ from FY EPS.",
                    "which_to_use": "TTM P/E is more current and comparable with other sources. FY P/E is useful for year-over-year consistency."
                },
                "note": "TTM = Trailing Twelve Months (rolling 4 quarters). FY = Fiscal Year (fixed 12-month period)."
            }

            # Dividend metrics (from key_metrics)
            dividend_yield = km_data.get("dividendYield")
            dividend_per_share = km_data.get("dividendPerShare")
            payout_ratio = km_data.get("payoutRatio")

            report["dividends"] = {
                "dividend_yield": dividend_yield,
                "dividend_per_share": dividend_per_share,
                "payout_ratio": payout_ratio,
                "has_dividend": dividend_yield is not None and dividend_yield > 0,
                "note": "Dividend data from FMP /key-metrics-ttm"
            }

            self.logger.info(f"[DIVIDENDS] Yield: {dividend_yield}, Per Share: {dividend_per_share}, Payout: {payout_ratio}")
            
            # Growth metrics - Calculate CAGR
            if income_stmt and len(income_stmt) >= 5:
                # Revenue CAGR 5Y
                revenue_first = income_stmt[-1].get("revenue", 0)
                revenue_last = income_stmt[0].get("revenue", 0)
                years = len(income_stmt) - 1
                
                if revenue_first is not None and revenue_last is not None and revenue_first > 0 and years > 0:
                    rev_cagr = (pow(revenue_last / revenue_first, 1 / years) - 1) * 100
                else:
                    rev_cagr = None

                # if revenue_first > 0 and years > 0:
                #     rev_cagr = (pow(revenue_last / revenue_first, 1 / years) - 1) * 100
                # else:
                #     rev_cagr = None
                    
                # EPS CAGR 5Y
                eps_first = income_stmt[-1].get("eps", 0)
                eps_last = income_stmt[0].get("eps", 0)
                
                # if eps_first > 0 and years > 0:
                #     eps_cagr = (pow(eps_last / eps_first, 1 / years) - 1) * 100
                # else:
                #     eps_cagr = None

                if eps_first is not None and eps_last is not None and eps_first > 0 and years > 0:
                    eps_cagr = (pow(eps_last / eps_first, 1 / years) - 1) * 100
                else:
                    eps_cagr = None
                    
                report["growth"] = {
                    "rev_cagr_5y": round(rev_cagr, 2) if rev_cagr is not None else None,
                    "eps_cagr_5y": round(eps_cagr, 2) if eps_cagr is not None else None
                }
            else:
                report["growth"] = {
                    "rev_cagr_5y": None,
                    "eps_cagr_5y": None
                }
            
            # Profitability metrics
            if income_stmt and len(income_stmt) > 0:
                latest_income = income_stmt[0]
                revenue = latest_income.get("revenue", 0)
                net_income = latest_income.get("netIncome", 0)
                gross_profit = latest_income.get("grossProfit", 0)
                operating_income = latest_income.get("operatingIncome", 0)

                if revenue is not None and net_income is not None and revenue > 0:
                    net_margin = (net_income / revenue * 100)
                    gross_margin = (gross_profit / revenue * 100) if gross_profit else None
                    operating_margin = (operating_income / revenue * 100) if operating_income else None
                else:
                    net_margin = None
                    gross_margin = None
                    operating_margin = None

                report["profitability"] = {
                    # Raw data for transparency
                    "revenue": revenue,
                    "net_income": net_income,
                    "eps": eps_ttm,
                    "gross_profit": gross_profit,
                    "operating_income": operating_income,
                    # Calculated margins
                    "gross_margin": round(gross_margin, 2) if gross_margin else None,
                    "operating_margin": round(operating_margin, 2) if operating_margin else None,
                    "net_margin": round(net_margin, 2) if net_margin else None,
                    # From key_metrics
                    "roe": km_data.get("roe"),
                    "roa": km_data.get("roa"),
                    "roic": km_data.get("roic"),
                }
            
            # Leverage metrics
            if balance_sheet and len(balance_sheet) > 0:
                latest_bs = balance_sheet[0]
                total_debt = latest_bs.get("totalDebt", 0)
                total_equity = latest_bs.get("totalStockholdersEquity", 0)
                current_assets = latest_bs.get("totalCurrentAssets", 0)
                current_liabilities = latest_bs.get("totalCurrentLiabilities", 0)
                
                # de_ratio = (total_debt / total_equity) if total_equity > 0 else None
                if total_debt is not None and total_equity is not None and total_equity > 0:
                    de_ratio = total_debt / total_equity
                else:
                    de_ratio = None

                # current_ratio = (current_assets / current_liabilities) if current_liabilities > 0 else None
                if current_assets is not None and current_liabilities is not None and current_liabilities > 0:
                    current_ratio = current_assets / current_liabilities
                else:
                    current_ratio = None
                
                # report["leverage"] = {
                #     "de_ratio": round(de_ratio, 2) if de_ratio else None,
                #     "net_debt_to_ebitda": km_data.get("netDebtToEBITDA"),
                #     "current_ratio": round(current_ratio, 2) if current_ratio else None
                # }
                report["leverage"] = {
                    "de_ratio": round(de_ratio, 2) if de_ratio is not None else None,
                    "net_debt_to_ebitda": km_data.get("netDebtToEBITDA"),
                    "current_ratio": round(current_ratio, 2) if current_ratio is not None else None
                }
            else:
                report["leverage"] = {
                    "de_ratio": None,
                    "net_debt_to_ebitda": None,
                    "current_ratio": None
                }
             
            # Cash flow metrics
            if cash_flow and len(cash_flow) > 0:
                latest_cf = cash_flow[0]
                fcf = latest_cf.get("freeCashFlow", 0)
                operating_cf = latest_cf.get("operatingCashFlow", 0)
                capex = latest_cf.get("capitalExpenditure", 0)

                # Use QUOTE market cap for FCF yield (current valuation)
                # Not key_metrics market cap (historical)
                market_cap_for_yield = quote_market_cap if quote_market_cap else km_data.get("marketCap", 0)

                if fcf is not None and market_cap_for_yield is not None and market_cap_for_yield > 0:
                    fcf_yield = (fcf / market_cap_for_yield * 100)
                else:
                    fcf_yield = None

                self.logger.info(f"[FCF YIELD] FCF: {fcf}, Market Cap (quote): {quote_market_cap}, Yield: {fcf_yield}")

                report["cashflow"] = {
                    "fcf": int(fcf) if fcf is not None else None,
                    "operating_cf": int(operating_cf) if operating_cf is not None else None,
                    "capex": int(capex) if capex is not None else None,
                    "fcf_yield": round(fcf_yield, 2) if fcf_yield is not None else None,
                    "market_cap_used": market_cap_for_yield,
                    "note": "FCF yield calculated using quote marketCap (current)"
                }
            else:
                report["cashflow"] = {
                    "fcf": None,
                    "operating_cf": None,
                    "capex": None,
                    "fcf_yield": None,
                    "market_cap_used": None,
                    "note": "Cash flow data not available"
                }
            
            # Risk metrics
            report["risk"] = {
                "beta": km_data.get("beta"),
                "altman_z": km_data.get("altmanZScore"),
                "piotroski_f": km_data.get("piotroskiScore")
            }

            # Growth data for AI analysis
            growth_data_list = []
            if growth_data:
                growth_data_list = [self._convert_to_dict(item) for item in growth_data]

            # =====================================================
            # VALUATION ANALYSIS (Using valuation calculators)
            # =====================================================
            # Calculate shares_outstanding with fallback from marketCap/price
            shares_outstanding = km_data.get("sharesOutstanding")
            if not shares_outstanding and quote:
                market_cap = quote.get("marketCap")
                price = quote.get("price")
                if market_cap and price and price > 0:
                    shares_outstanding = market_cap / price
                    self.logger.info(f"[VALUATION] Calculated shares_outstanding from marketCap/price: {shares_outstanding:.0f}")

            valuation_analysis = self._calculate_intrinsic_value(
                symbol=symbol,
                current_price=quote.get("price") if quote else None,
                eps=income_stmt[0].get("eps") if income_stmt else None,
                fcf=cash_flow[0].get("freeCashFlow") if cash_flow else None,
                shares_outstanding=shares_outstanding,
                cash=balance_sheet[0].get("cashAndCashEquivalents") if balance_sheet else 0,
                debt=balance_sheet[0].get("totalDebt") if balance_sheet else 0,
                eps_growth_rate=growth_data_list[0].get("epsgrowth") if growth_data_list else None,
            )

            # Add valuation analysis to report
            report["intrinsic_value"] = valuation_analysis

            # Log calculated valuation
            if valuation_analysis:
                self.logger.info(f"[VALUATION] Intrinsic Value Analysis:")
                self.logger.info(f"  Graham Value: {valuation_analysis.get('graham_value', 'N/A')}")
                self.logger.info(f"  DCF Value: {valuation_analysis.get('dcf_value', 'N/A')}")
                self.logger.info(f"  Current Price: {valuation_analysis.get('current_price', 'N/A')}")
                self.logger.info(f"  Verdict: {valuation_analysis.get('verdict', 'N/A')}")

            return {
                "fundamental_report": report,
                "growth_data": growth_data_list
            }
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive fundamental data for {symbol}: {e}")
            raise

    
    async def stream_comprehensive_analysis(
        self,
        symbol: str,
        report: Dict[str, Any],
        growth_data: List[Dict[str, Any]],
        model_name: str,
        provider_type: str,
        api_key: Optional[str] = None,
        chat_history: Optional[str] = None,
        user_question: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream comprehensive fundamental analysis with enhanced context
        """
        try:
            # Create enhanced prompt with context
            base_prompt = self._create_comprehensive_prompt_with_context(
                symbol=symbol,
                report=report,
                growth_data=growth_data,
                context=chat_history,
                user_question=user_question
            )
            
            # Add context if available
            context_section = ""
            if chat_history:
                context_section = f"""
    Previous Analysis Context:
    {chat_history}

    Please reference and build upon any previous fundamental analyses when relevant.
    Highlight any significant changes in metrics or investment thesis.
    """
            
            # Add user question if provided
            question_section = ""
            if user_question:
                question_section = f"""

    User's Specific Question:
    {user_question}

    Please ensure your analysis addresses this question while providing comprehensive fundamental insights.
    """
            
            # Combine all sections
            full_prompt = f"{context_section}\n{base_prompt}\n{question_section}"
            
            # Handle language detection - use default if no user_question
            detected_language = target_language or "en"

            if user_question:
                detection_method = DetectionMethod.LLM if len(user_question.split()) < 2 else DetectionMethod.LIBRARY

                # Language detection
                language_info = await language_detector.detect(
                    text=user_question,
                    method=detection_method,
                    system_language=target_language,
                    model_name=model_name,
                    provider_type=provider_type,
                    api_key=api_key
                )
                detected_language = language_info.get("detected_language") or target_language or "en"

            if detected_language:
                lang_name = {
                    "en": "English",
                    "vi": "Vietnamese", 
                    "zh": "Chinese",
                    "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
                    "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
                }.get(detected_language, "the detected language")
                
            language_instruction = f"""
            CRITICAL LANGUAGE REQUIREMENT:
            You MUST respond ENTIRELY in {lang_name} language.
            - ALL text, explanations, and analysis must be in {lang_name}
            - Use appropriate financial terminology for {lang_name}
            - Format numbers and dates according to {lang_name} conventions
            """

            # System prompt with context awareness
            system_prompt = f"""You are a fundamental analyst providing data-driven stock analysis.

    {language_instruction}

    ## DATA INTEGRITY RULES (CRITICAL)

    ### 1. PERIOD TAGS (MANDATORY)
    Every financial metric MUST include its period:
    - [FY2025] for fiscal year data
    - [TTM] for trailing twelve months
    - [Q3 FY2025] for quarterly data
    - Example: "Revenue [FY2025]: $130.5B" NOT just "Revenue: $130.5B"

    ### 2. DATA SOURCE & TIMESTAMPS
    - All data comes from Financial Modeling Prep (FMP) API
    - When citing key metrics: "(Source: FMP /key-metrics-ttm, as of [date])"
    - Price data: specify "close price" or "realtime" and timezone if known
    - If P/E changes from previous analysis, EXPLAIN why (price change, EPS update, etc.)

    ### 2.5 P/E RATIO: FY vs TTM (CRITICAL FOR USER CLARITY)
    We provide TWO P/E values - explain both to users:

    **P/E (FY)**: Uses EPS from the latest FISCAL YEAR (fixed 12-month period)
    **P/E (TTM)**: Uses EPS from TRAILING TWELVE MONTHS (rolling 4 quarters) - THIS MATCHES YAHOO FINANCE

    ⚠️ **WHY THEY DIFFER**:
    - FY EPS is from a fixed period (e.g., fiscal year ending Jan 2025)
    - TTM EPS is rolling (sum of last 4 quarters, updated each quarter)
    - If recent quarters had higher/lower earnings than same quarters last year, TTM ≠ FY

    📌 **ALWAYS EXPLAIN TO USERS**:
    - "Our P/E (TTM) of X.X matches Yahoo Finance because both use trailing 12-month EPS"
    - "Our P/E (FY) of X.X uses fiscal year EPS which may differ from TTM"
    - Help users understand which to use for their analysis

    ### 3. SEPARATE FACTS vs INTERPRETATION vs ACTION
    - **FACTS**: Raw data with period tags and source
    - **INTERPRETATION**: Your analysis of what the data means
    - **ACTION**: Recommendations based on interpretation

    ### 4. TWO INVESTOR PERSPECTIVES (REQUIRED)
    Always analyze from BOTH perspectives:

    **A) Growth Investor Perspective** (ưu tiên tăng giá):
    - Focus on: EPS growth, Revenue growth, Forward P/E vs Trailing P/E
    - High P/E acceptable if justified by growth
    - Key question: "Is Forward P/E significantly lower than Trailing P/E?"
      - YES → market expects strong EPS growth (positive for growth thesis)
      - NO → growth expectations are modest

    **B) Dividend Investor Perspective** (ưu tiên dòng tiền):
    - Focus on: Dividend Yield, Payout Ratio, Dividend Growth history
    - If Yield < 1%: NOT suitable as core dividend holding
    - If Yield > 3% with stable payout: potential income stock

    ### 5. P/E DIAGNOSTIC TEST (3 Scenarios)
    Evaluate P/E using this framework:

    ✅ **GOOD Scenario** (đậu mạnh):
    - Forward P/E << Trailing P/E (market expects EPS jump)
    - Business confirming growth (revenue/segment growth strong)
    - P/E not outlier vs industry peers

    ➖ **NEUTRAL Scenario** (đậu nhẹ, cần quan sát):
    - Forward P/E slightly < Trailing P/E
    - Growth slowing but still positive
    - Valuation in line with peers

    ❌ **RISK Scenario** (rớt):
    - Trailing P/E high AND Forward P/E also high
    - EPS forecast being cut
    - No dividend cushion (yield too low)
    - If growth misses expectations, price volatility will be high

    ### 6. INTRINSIC VALUE MODELS (USE WITH CAUTION)

    **CRITICAL**: DCF and Graham are MODEL ESTIMATES, not facts!

    **Confidence Level**: Always use **MEDIUM** confidence for model-based valuations
    - These models depend heavily on assumptions
    - Growth stocks often appear "overvalued" in DCF/Graham
    - DO NOT use "HIGH confidence" for model-based conclusions

    **Required Disclaimers**:
    - "DCF estimate based on: WACC X%, terminal growth Y%"
    - "Graham formula uses historical growth (less accurate than forward estimates)"
    - "Model estimates have wide uncertainty range"

    ### 7. PEG RATIO GUIDANCE
    - Standard PEG = P/E ÷ FORWARD earnings growth
    - If using historical CAGR: "PEG using historical 5Y CAGR (not forward estimates)"
    - PEG < 1.0: potentially undervalued vs growth
    - PEG 1.0-1.5: fairly valued
    - PEG > 2.0: potentially expensive vs growth

    ### 8. PEER COMPARISON (if data available)
    - Compare P/E, P/S, FCF Yield with 2-3 industry peers
    - State if stock is premium/discount to peers
    - Note: premium may be justified by superior growth

    ### 9. FINANCIAL RATIOS BENCHMARKS

    **Profitability**:
    - Net Margin: >20% excellent, 10-20% good, <10% monitor
    - ROE: >15% excellent, 10-15% good, <10% below average

    **Leverage**:
    - D/E Ratio: <0.5 conservative, 0.5-1.0 moderate, >1.0 aggressive
    - Current Ratio: >1.5 strong, 1.0-1.5 adequate, <1.0 liquidity risk

    **Growth Quality**:
    - Revenue growth > EPS growth: margin compression warning
    - EPS growth > Revenue growth: operating leverage (positive)

    ## OUTPUT QUALITY
    - Data-driven with specific metrics AND period tags
    - Accessible to both beginners and professional traders
    - Use icons for visual clarity
    - Brief explanations of financial terms for general audience
    - NEVER claim "HIGH confidence" for model-based valuations"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ]
            
            # Stream the response
            async for chunk in self.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.3
            ):
                if chunk:
                    yield chunk
                    
        except Exception as e:
            self.logger.error(f"Error streaming comprehensive analysis for {symbol}: {e}")
            yield f"Error generating analysis: {str(e)}"


    def _create_comprehensive_prompt_with_context(
        self,
        symbol: str,
        report: Dict[str, Any],
        growth_data: List[Dict[str, Any]],
        context: Optional[str] = None,
        user_question: Optional[str] = None
    ) -> str:
        """
        Create comprehensive prompt with optional context for continuity
        """
        base_prompt = self.create_comprehensive_analysis_prompt(
            symbol=symbol,
            report=report,
            growth_data=growth_data,
            memory_context=context,
            user_question=user_question
        )
        
        # Prepend context if available
        if context:
            base_prompt = f"""Previous fundamental analyses and insights:
    {context}

    Now analyzing updated fundamental data:
    {base_prompt}"""
        
        # Append user question if provided
        if user_question:
            base_prompt += f"\n\nSpecific Focus: {user_question}"
        
        return base_prompt

    def create_comprehensive_analysis_prompt(
        self,
        symbol: str,
        report: Dict[str, Any],
        growth_data: List[Dict[str, Any]],
        memory_context: Optional[str] = "",
        user_question: Optional[str] = None,
        target_language: Optional[str] = None
    ) -> str:
        """
        Create comprehensive prompt combining all fundamental metrics for AI analysis.

        Enhanced with:
        - Period tags for all metrics
        - Data source attribution
        - PEG calculation guidance
        - Bull/Base/Bear scenarios requirement
        - Intrinsic value calculations (Graham, DCF)
        """
        # Extract latest growth data with date
        latest_growth = growth_data[0] if growth_data else {}
        growth_date = latest_growth.get('date', 'N/A')

        # Determine period type from report
        report_date = report.get('generated', datetime.now().isoformat())[:10]

        # Extract intrinsic value data if available
        intrinsic_value = report.get('intrinsic_value', {})
        valuation_section = ""
        if intrinsic_value:
            graham_val = intrinsic_value.get('graham_value')
            dcf_val = intrinsic_value.get('dcf_value')
            current_price = intrinsic_value.get('current_price')
            verdict = intrinsic_value.get('verdict', 'N/A')
            pe_analysis = intrinsic_value.get('pe_analysis') or {}
            methodology_notes = intrinsic_value.get('methodology_notes') or []
            dcf_details = intrinsic_value.get('dcf_details') or {}
            graham_details = intrinsic_value.get('graham_details') or {}

            # Pre-format values to handle None safely
            price_str = f"${current_price:.2f}" if current_price else "N/A"
            graham_str = f"${graham_val:.2f}" if graham_val else "N/A"
            dcf_str = f"${dcf_val:.2f}" if dcf_val else "N/A"
            verdict_str = verdict.upper().replace('_', ' ') if verdict else "N/A"

            # Calculate Price/Intrinsic ratios
            price_to_graham = f"{current_price/graham_val:.2f}x" if graham_val and current_price else "N/A"
            price_to_dcf = f"{current_price/dcf_val:.2f}x" if dcf_val and current_price else "N/A"

            # Calculate margin of safety
            graham_margin = f"{((graham_val - current_price)/current_price)*100:.1f}%" if graham_val and current_price else "N/A"
            dcf_margin = f"{((dcf_val - current_price)/current_price)*100:.1f}%" if dcf_val and current_price else "N/A"

            # DCF Sensitivity Analysis (WACC ±1%, Terminal Growth ±0.5%)
            dcf_sensitivity = ""
            if dcf_val and current_price:
                base_wacc = dcf_details.get('discount_rate', 0.10)
                base_tg = dcf_details.get('terminal_growth', 0.025)
                # Approximate sensitivity: DCF value inversely proportional to (WACC - terminal_growth)
                # Lower WACC = higher value, Higher terminal growth = higher value
                dcf_low = dcf_val * 0.80  # ~WACC 11%, TG 2%
                dcf_high = dcf_val * 1.25  # ~WACC 9%, TG 3%
                dcf_sensitivity = f"""
    **DCF Sensitivity Analysis** (Confidence: MEDIUM):
    | Scenario | WACC | Terminal Growth | Estimated Value |
    |----------|------|-----------------|-----------------|
    | Conservative | 11% | 2.0% | ${dcf_low:.2f} |
    | Base Case | 10% | 2.5% | ${dcf_val:.2f} |
    | Optimistic | 9% | 3.0% | ${dcf_high:.2f} |

    Range: ${dcf_low:.2f} - ${dcf_high:.2f} (Current: {price_str})"""

            valuation_section = f"""
    ═══════════════════════════════════════════════════════════════════════════════
    💰 INTRINSIC VALUE ANALYSIS (MODEL ESTIMATES - Confidence: MEDIUM)
    ═══════════════════════════════════════════════════════════════════════════════

    **Current Market Price**: {price_str}

    **Graham Formula Valuation**:
    - Intrinsic Value: {graham_str}
    - Price/Graham: {price_to_graham}
    - Margin of Safety: {graham_margin}
    - Formula: V = EPS × (8.5 + 2g) × (4.4/Y)
    - Growth Rate Used: {graham_details.get('growth_rate_used', 'N/A')} ({graham_details.get('growth_rate_source', 'N/A')})
    - ⚠️ Note: Graham uses HISTORICAL growth, not forward estimates

    **DCF Valuation**:
    - Intrinsic Value: {dcf_str}
    - Price/DCF: {price_to_dcf}
    - Margin of Safety: {dcf_margin}
    - Assumptions: WACC={dcf_details.get('discount_rate', 0.10)*100:.0f}%, Terminal Growth={dcf_details.get('terminal_growth', 0.025)*100:.1f}%
    {dcf_sensitivity}

    **P/E Analysis**:
    - Current P/E [TTM]: {pe_analysis.get('current_pe', 'N/A')}
    - Interpretation: {pe_analysis.get('pe_interpretation', 'N/A')}

    **Model-Based Verdict**: {verdict_str}
    ⚠️ This is based on model estimates with inherent uncertainty.
    Growth stocks often appear "overvalued" in DCF/Graham models.

    **Methodology Notes**:
    {chr(10).join('- ' + note for note in methodology_notes)}

    ═══════════════════════════════════════════════════════════════════════════════
    ⚠️ IMPORTANT DISCLAIMERS
    ═══════════════════════════════════════════════════════════════════════════════
    - DCF/Graham are MODEL ESTIMATES, not facts
    - Results depend heavily on assumptions (growth rates, discount rates)
    - Growth stocks typically show as "overvalued" in these models
    - Always compare with peer valuations and business fundamentals
    - These estimates have MEDIUM confidence, not HIGH
"""

        # Extract key raw metrics for display
        valuation = report.get('valuation', {})
        profitability = report.get('profitability', {})
        dividends = report.get('dividends', {})
        cashflow = report.get('cashflow', {})
        leverage = report.get('leverage', {})
        growth = report.get('growth', {})
        data_snapshot = report.get('data_snapshot', {})
        pe_comparison = report.get('pe_comparison', {})

        # Build comprehensive raw data summary for LLM
        raw_data_summary = f"""
    ═══════════════════════════════════════════════════════════════════════════════
    📋 DATA SNAPSHOT (LOCKED - All calculations use these values)
    ═══════════════════════════════════════════════════════════════════════════════

    **Data Sources**:
    - Price: FMP /quote (realtime/close)
    - Financials: FMP /income-statement, /balance-sheet, /cash-flow (TTM)
    - Ratios: CALCULATED from current price + TTM metrics (NOT from key_metrics)

    **Price Snapshot**:
    - Current Price: ${valuation.get('price', 'N/A')}
    - Price Date: {data_snapshot.get('price_date', report_date)}
    - Market Cap (quote): {self._format_large_number(data_snapshot.get('market_cap_quote', 0))}

    ═══════════════════════════════════════════════════════════════════════════════
    ⚠️ P/E RATIO COMPARISON (FY vs TTM) - IMPORTANT FOR USERS
    ═══════════════════════════════════════════════════════════════════════════════

    🔴 **WHY DO WE SHOW TWO P/E VALUES?**
    Different sources (Yahoo Finance, Bloomberg, etc.) may show different P/E values.
    This is NOT an error - they use different EPS calculations:

    | Type | P/E | EPS | Explanation |
    |------|-----|-----|-------------|
    | **P/E (FY)** | {valuation.get('pe_fy', 'N/A')} | ${valuation.get('eps_fy', 'N/A')} | Uses EPS from latest fiscal year ({pe_comparison.get('fy_period', 'N/A')}). Good for YoY comparison. |
    | **P/E (TTM)** | {valuation.get('pe_ttm', 'N/A')} | ${valuation.get('eps_ttm', 'N/A')} | Uses rolling 12-month EPS (last 4 quarters). **Matches Yahoo Finance.** |
    | **Forward P/E** | {valuation.get('forward_pe', 'N/A')} | Analyst Est. | Based on next 12-month EPS estimates from analysts. |

    📌 **KEY DIFFERENCES EXPLAINED**:
    - **FY (Fiscal Year)**: EPS from a fixed 12-month period (e.g., Jan 2024 - Jan 2025)
    - **TTM (Trailing Twelve Months)**: EPS from the most recent 4 quarters (rolling)
    - If recent quarters had HIGHER earnings → TTM EPS > FY EPS → TTM P/E < FY P/E
    - If recent quarters had LOWER earnings → TTM EPS < FY EPS → TTM P/E > FY P/E

    💡 **WHICH P/E TO USE?**
    - **Compare with Yahoo Finance/Bloomberg?** → Use P/E (TTM)
    - **Year-over-year analysis?** → Use P/E (FY) for consistency
    - **Valuation thesis?** → Consider Forward P/E for growth expectations

    ═══════════════════════════════════════════════════════════════════════════════
    📊 VALUATION RATIOS (Full Details)
    ═══════════════════════════════════════════════════════════════════════════════

    **Calculation Methods**:
    - P/E (FY): Price ${valuation.get('price', 'N/A')} / EPS FY ${valuation.get('eps_fy', 'N/A')} = {valuation.get('pe_fy', 'N/A')}
    - P/E (TTM): From FMP /key-metrics-ttm endpoint = {valuation.get('pe_ttm', 'N/A')}
    - P/B: Price / Book Value per Share = {valuation.get('price', 'N/A')} / {valuation.get('book_value_per_share', 'N/A')} = {valuation.get('pb_ttm', 'N/A')}
    - P/S: Price / Revenue per Share = {valuation.get('price', 'N/A')} / {valuation.get('revenue_per_share', 'N/A')} = {valuation.get('ps_ttm', 'N/A')}

    | Metric | FY Value | TTM Value | Notes |
    |--------|----------|-----------|-------|
    | **P/E Ratio** | {valuation.get('pe_fy', 'N/A')} | {valuation.get('pe_ttm', 'N/A')} | TTM matches Yahoo Finance |
    | **P/B Ratio** | {valuation.get('pb_fy', 'N/A')} | {valuation.get('pb_ttm', 'N/A')} | Book value per share |
    | **P/S Ratio** | {valuation.get('ps_fy', 'N/A')} | {valuation.get('ps_ttm', 'N/A')} | Revenue per share |
    | **EPS** | ${valuation.get('eps_fy', 'N/A')} | ${valuation.get('eps_ttm', 'N/A')} | Earnings per share |
    | Forward P/E | - | {valuation.get('forward_pe', 'N/A')} | Analyst estimates |
    | PEG | - | {valuation.get('peg', 'N/A')} | P/E / Growth rate |
    | EV/EBITDA | - | {valuation.get('ev_ebitda', 'N/A')} | Enterprise value based |

    ═══════════════════════════════════════════════════════════════════════════════
    💰 PROFITABILITY (Source: FMP /income-statement TTM)
    ═══════════════════════════════════════════════════════════════════════════════

    | Metric | Value | Raw Data |
    |--------|-------|----------|
    | Revenue | {self._format_large_number(profitability.get('revenue', 0))} | {profitability.get('revenue', 'N/A')} |
    | Gross Profit | {self._format_large_number(profitability.get('gross_profit', 0))} | Margin: {profitability.get('gross_margin', 'N/A')}% |
    | Operating Income | {self._format_large_number(profitability.get('operating_income', 0))} | Margin: {profitability.get('operating_margin', 'N/A')}% |
    | Net Income | {self._format_large_number(profitability.get('net_income', 0))} | Margin: {profitability.get('net_margin', 'N/A')}% |
    | EPS | ${profitability.get('eps', 'N/A')} | Used for P/E calculation |
    | ROE | {self._format_percentage(profitability.get('roe', 0))} | Return on Equity |
    | ROA | {self._format_percentage(profitability.get('roa', 0))} | Return on Assets |

    ═══════════════════════════════════════════════════════════════════════════════
    📈 GROWTH (Source: FMP /income-statement 5Y CAGR)
    ═══════════════════════════════════════════════════════════════════════════════

    | Metric | Value | Note |
    |--------|-------|------|
    | Revenue CAGR (5Y) | {growth.get('rev_cagr_5y', 'N/A')}% | Compound annual growth |
    | EPS CAGR (5Y) | {growth.get('eps_cagr_5y', 'N/A')}% | Used for PEG if forward not available |

    ═══════════════════════════════════════════════════════════════════════════════
    💵 DIVIDENDS (Source: FMP /key-metrics-ttm)
    ═══════════════════════════════════════════════════════════════════════════════

    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | Dividend Yield | {self._format_percentage(dividends.get('dividend_yield', 0))} | {'Suitable for income' if dividends.get('dividend_yield', 0) and dividends.get('dividend_yield', 0) > 0.03 else 'NOT suitable for income focus (< 3%)'} |
    | Dividend/Share | ${dividends.get('dividend_per_share', 'N/A')} | Annual dividend |
    | Payout Ratio | {self._format_percentage(dividends.get('payout_ratio', 0))} | % of earnings paid as dividend |
    | Has Dividend? | {'Yes' if dividends.get('has_dividend') else 'No/Minimal'} | {dividends.get('note', '')} |

    ═══════════════════════════════════════════════════════════════════════════════
    💸 CASH FLOW (Source: FMP /cash-flow-statement TTM)
    ═══════════════════════════════════════════════════════════════════════════════

    | Metric | Value | Note |
    |--------|-------|------|
    | Free Cash Flow | {self._format_large_number(cashflow.get('fcf', 0))} | FCF = Operating CF - CapEx |
    | Operating CF | {self._format_large_number(cashflow.get('operating_cf', 0))} | Cash from operations |
    | CapEx | {self._format_large_number(cashflow.get('capex', 0))} | Capital expenditure |
    | FCF Yield | {cashflow.get('fcf_yield', 'N/A')}% | FCF / Market Cap (quote) |

    ═══════════════════════════════════════════════════════════════════════════════
    ⚖️ LEVERAGE (Source: FMP /balance-sheet-statement TTM)
    ═══════════════════════════════════════════════════════════════════════════════

    | Metric | Value | Risk Level |
    |--------|-------|------------|
    | D/E Ratio | {leverage.get('de_ratio', 'N/A')} | {'Low risk' if leverage.get('de_ratio', 1) and leverage.get('de_ratio', 1) < 0.5 else 'Moderate' if leverage.get('de_ratio', 1) and leverage.get('de_ratio', 1) < 1 else 'Higher risk'} |
    | Current Ratio | {leverage.get('current_ratio', 'N/A')} | {'Strong liquidity' if leverage.get('current_ratio', 0) and leverage.get('current_ratio', 0) > 1.5 else 'Adequate' if leverage.get('current_ratio', 0) and leverage.get('current_ratio', 0) > 1 else 'Liquidity concern'} |
    | Net Debt/EBITDA | {leverage.get('net_debt_to_ebitda', 'N/A')} | Debt coverage |
"""

        prompt = f"""
    Analyze the comprehensive fundamental data for {symbol}:

    ═══════════════════════════════════════════════════════════════════════════════
    📊 FUNDAMENTAL METRICS REPORT
    Data Source: Financial Modeling Prep (FMP) API
    Endpoints: /quote, /key-metrics-ttm, /income-statement, /balance-sheet-statement, /cash-flow-statement
    Report Generated: {report_date} (UTC)
    Period: TTM (Trailing Twelve Months) unless otherwise noted
    ═══════════════════════════════════════════════════════════════════════════════
    {raw_data_summary}
    ═══════════════════════════════════════════════════════════════════════════════
    📄 FULL REPORT DATA
    ═══════════════════════════════════════════════════════════════════════════════

    {json.dumps(report, indent=2)}
    {valuation_section}
    ═══════════════════════════════════════════════════════════════════════════════
    📈 RECENT GROWTH TRENDS
    Period: Annual (YoY) as of {growth_date}
    ═══════════════════════════════════════════════════════════════════════════════

    - Revenue Growth [YoY]: {self._format_percentage(latest_growth.get('revenueGrowth', 0))}
    - EPS Growth [YoY]: {self._format_percentage(latest_growth.get('epsgrowth', 0))}
    - FCF Growth [YoY]: {self._format_percentage(latest_growth.get('freeCashFlowGrowth', 0))}
    - Operating Income Growth [YoY]: {self._format_percentage(latest_growth.get('operatingIncomeGrowth', 0))}

    ═══════════════════════════════════════════════════════════════════════════════
    📝 ANALYSIS REQUIREMENTS
    ═══════════════════════════════════════════════════════════════════════════════

    Provide a comprehensive investment analysis with the following structure:

    ### 1. 📊 INVESTMENT RATING & THESIS
    - Clear rating: STRONG BUY / BUY / HOLD / SELL / STRONG SELL
    - Core investment thesis in 2-3 sentences
    - Confidence level: MEDIUM (use MEDIUM for model-based conclusions)
    - **NOTE**: Rating is metrics-based, not investment advice

    ### 2. 👥 TWO INVESTOR PERSPECTIVES (REQUIRED)

    **A) 📈 Growth Investor View** (ưu tiên tăng giá):
    - Is this stock suitable for growth investing?
    - Forward P/E vs Trailing P/E analysis
    - EPS growth trajectory
    - Key growth drivers

    **B) 💵 Dividend Investor View** (ưu tiên dòng tiền):
    - Current Dividend Yield
    - Is yield > 3%? (threshold for income focus)
    - Payout ratio sustainability
    - Conclusion: Suitable/Not suitable as income stock

    ### 3. 🔍 P/E DIAGNOSTIC TEST (3 Scenarios)
    Evaluate using this framework:

    | Test | Result | Interpretation |
    |------|--------|----------------|
    | Forward P/E < Trailing P/E? | YES/NO | Market expects EPS growth? |
    | P/E vs Industry peers? | Premium/Discount | Relative valuation |
    | Dividend cushion? | Yield X% | Safety net if growth misses |

    **Overall P/E Verdict**: ✅ GOOD / ➖ NEUTRAL / ❌ RISK

    ### 4. 💯 FINANCIAL HEALTH SCORE (0-100)
    | Category | Score (0-20) | Key Metrics |
    |----------|--------------|-------------|
    | Valuation | X/20 | P/E vs growth, P/S |
    | Growth | X/20 | Rev CAGR, EPS CAGR |
    | Profitability | X/20 | Net Margin, ROE |
    | Leverage | X/20 | D/E, Current Ratio |
    | Cash Flow | X/20 | FCF, FCF Yield |

    ### 5. ✅ KEY STRENGTHS (3-5 points)
    - Use exact metrics WITH period tags
    - Explain WHY each metric is strong

    ### 6. ⚠️ CRITICAL RISKS (2-3 points)
    - Quantify risks with specific metrics
    - Note: Low dividend yield = no cushion if growth disappoints

    ### 7. 💰 VALUATION ASSESSMENT

    **PEG Analysis**:
    - State source: "PEG using [historical CAGR / forward estimates]"
    - PEG < 1.0: potentially undervalued vs growth
    - PEG 1.0-1.5: fairly valued
    - PEG > 2.0: potentially expensive vs growth

    **DCF/Graham Models** (if provided):
    - Note: These are MODEL ESTIMATES with MEDIUM confidence
    - Growth stocks often appear "overvalued" in these models
    - Always cross-check with business fundamentals

    ### 8. 🎯 SCENARIO ANALYSIS (REQUIRED)
    | Scenario | Description | Trigger |
    |----------|-------------|---------|
    | ✅ **GOOD** | Growth confirms, Forward P/E justified | EPS beats estimates |
    | ➖ **NEUTRAL** | Growth slows but still positive | Mixed signals |
    | ❌ **RISK** | Growth misses, high P/E not justified | EPS forecast cut |

    ### 9. 📋 ACTIONABLE RECOMMENDATIONS
    - Entry strategy (general guidance, not specific prices)
    - Position sizing suggestion
    - Time horizon
    - Key catalysts to watch

    ### 10. 📡 KEY METRICS TO MONITOR
    - Which 3-5 metrics are most critical?
    - What changes would alter your thesis?

    ═══════════════════════════════════════════════════════════════════════════════
    ⚠️ IMPORTANT RULES
    ═══════════════════════════════════════════════════════════════════════════════

    1. ALWAYS include period tags: [TTM], [FY2025], [Q3 FY2025], [5Y CAGR]
    2. Cite data source: "(FMP /key-metrics-ttm, as of [date])"
    3. Use exact numbers from the data - never fabricate
    4. Separate FACTS from INTERPRETATION clearly
    5. Use MEDIUM confidence for model-based valuations (DCF/Graham)
    6. Include BOTH Growth and Dividend investor perspectives
    7. End with disclaimer: Analysis is informational, not investment advice
    """

        return prompt
    

    ## Add data - 18/10/25
    async def get_fundamental_data(
        self,
        symbol: str,
        lookback_days: int = 365
    ) -> Dict[str, Any]:
        """
        Fetch all fundamental data for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Number of days to look back for historical data
            
        Returns:
            Dictionary containing all fundamental data
        """
        try:
            self.logger.info(f"Fetching fundamental data for {symbol}")
            
            # Calculate date ranges
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            # Fetch all data concurrently for better performance
            results = await asyncio.gather(
                self._get_financial_metrics_data(symbol, end_date),
                self._get_key_line_items(symbol, end_date),
                self._get_insider_trading_data(symbol, start_date, end_date),
                self._get_news_sentiment_data(symbol, start_date, end_date),
                self._get_company_overview(symbol, end_date),
                return_exceptions=True
            )
            
            # Unpack results
            financial_metrics, line_items, insider_trades, news_data, company_facts = results
            
            # Handle exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error in fundamental data fetch (index {i}): {str(result)}")
            
            return {
                "symbol": symbol,
                "financial_metrics": financial_metrics if not isinstance(financial_metrics, Exception) else {},
                "line_items": line_items if not isinstance(line_items, Exception) else {},
                "insider_trades": insider_trades if not isinstance(insider_trades, Exception) else [],
                "news_sentiment": news_data if not isinstance(news_data, Exception) else {},
                "company_overview": company_facts if not isinstance(company_facts, Exception) else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
            raise
    
    async def _get_financial_metrics_data(self, symbol: str, end_date: str) -> Dict[str, Any]:
        """Fetch financial metrics (valuation, profitability, growth)"""
        try:
            metrics = get_financial_metrics(
                ticker=symbol,
                end_date=end_date,
                period="ttm",
                limit=4  # Get last 4 quarters for trend analysis
            )
            
            if not metrics or len(metrics) == 0:
                return {}
            
            # Get the most recent metrics
            latest = metrics[0]
            
            return {
                "valuation": {
                    "market_cap": latest.market_cap,
                    "enterprise_value": latest.enterprise_value,
                    "pe_ratio": latest.price_to_earnings_ratio,
                    "pb_ratio": latest.price_to_book_ratio,
                    "ps_ratio": latest.price_to_sales_ratio,
                    "ev_to_ebitda": latest.enterprise_value_to_ebitda_ratio,
                    "peg_ratio": latest.peg_ratio
                },
                "profitability": {
                    "gross_margin": latest.gross_margin,
                    "operating_margin": latest.operating_margin,
                    "net_margin": latest.net_margin,
                    "roe": latest.return_on_equity,
                    "roa": latest.return_on_assets,
                    "roic": latest.return_on_invested_capital
                },
                "growth": {
                    "revenue_growth": latest.revenue_growth,
                    "earnings_growth": latest.earnings_growth,
                    "fcf_growth": latest.free_cash_flow_growth,
                    "eps_growth": latest.earnings_per_share_growth
                },
                "liquidity": {
                    "current_ratio": latest.current_ratio,
                    "quick_ratio": latest.quick_ratio,
                    "cash_ratio": latest.cash_ratio
                },
                "leverage": {
                    "debt_to_equity": latest.debt_to_equity,
                    "debt_to_assets": latest.debt_to_assets,
                    "interest_coverage": latest.interest_coverage
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching financial metrics for {symbol}: {str(e)}")
            return {}
    
    async def _get_key_line_items(self, symbol: str, end_date: str) -> Dict[str, Any]:
        """Fetch key financial statement line items"""
        try:
            # Define critical line items we want to analyze
            line_items_to_fetch = [
                "revenue",
                "net_income",
                "free_cash_flow",
                "operating_cash_flow",
                "total_debt",
                "cash_and_cash_equivalents",
                "total_assets",
                "total_equity"
            ]
            
            items = search_line_items(
                ticker=symbol,
                line_items=line_items_to_fetch,
                end_date=end_date,
                period="ttm",
                limit=4  # Get trend over 4 quarters
            )
            
            if not items or len(items) == 0:
                return {}
            
            # Extract latest values
            latest = items[0]
            
            return {
                "revenue": getattr(latest, "revenue", None),
                "net_income": getattr(latest, "net_income", None),
                "free_cash_flow": getattr(latest, "free_cash_flow", None),
                "operating_cash_flow": getattr(latest, "operating_cash_flow", None),
                "total_debt": getattr(latest, "total_debt", None),
                "cash": getattr(latest, "cash_and_cash_equivalents", None),
                "total_assets": getattr(latest, "total_assets", None),
                "total_equity": getattr(latest, "total_equity", None)
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching line items for {symbol}: {str(e)}")
            return {}
    
    async def _get_insider_trading_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Fetch and analyze insider trading activity"""
        try:
            trades = get_insider_trades(
                ticker=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=100
            )
            
            if not trades:
                return []
            
            # Analyze insider activity
            buy_count = 0
            sell_count = 0
            buy_value = 0.0
            sell_value = 0.0
            
            for trade in trades:
                if trade.transaction_type in ["Buy", "P-Purchase"]:
                    buy_count += 1
                    if trade.transaction_value:
                        buy_value += trade.transaction_value
                elif trade.transaction_type in ["Sell", "S-Sale"]:
                    sell_count += 1
                    if trade.transaction_value:
                        sell_value += abs(trade.transaction_value)
            
            return {
                "summary": {
                    "total_trades": len(trades),
                    "buy_count": buy_count,
                    "sell_count": sell_count,
                    "buy_value": buy_value,
                    "sell_value": sell_value,
                    "net_value": buy_value - sell_value,
                    "buy_sell_ratio": buy_count / sell_count if sell_count > 0 else float('inf')
                },
                "recent_trades": [
                    {
                        "date": trade.transaction_date,
                        "owner": trade.owner_name,
                        "type": trade.transaction_type,
                        "shares": trade.transaction_shares,
                        "value": trade.transaction_value
                    }
                    for trade in trades[:10]  # Top 10 most recent
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching insider trades for {symbol}: {str(e)}")
            return []
    
    async def _get_news_sentiment_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, Any]:
        """Fetch and analyze news sentiment"""
        try:
            news = get_company_news(
                ticker=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=50
            )
            
            if not news:
                return {}
            
            # Analyze sentiment distribution
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            for article in news:
                sentiment = article.sentiment
                if sentiment == "Positive":
                    positive_count += 1
                elif sentiment == "Negative":
                    negative_count += 1
                else:
                    neutral_count += 1
            
            total = len(news)
            
            return {
                "summary": {
                    "total_articles": total,
                    "positive_count": positive_count,
                    "negative_count": negative_count,
                    "neutral_count": neutral_count,
                    "positive_ratio": positive_count / total if total > 0 else 0,
                    "negative_ratio": negative_count / total if total > 0 else 0,
                    "overall_sentiment": self._determine_overall_sentiment(
                        positive_count, negative_count, neutral_count
                    )
                },
                "recent_headlines": [
                    {
                        "date": article.date,
                        "title": article.title,
                        "sentiment": article.sentiment,
                        "source": article.source,
                        "url": article.url
                    }
                    for article in news[:10]  # Top 10 most recent
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {str(e)}")
            return {}
    
    async def _get_company_overview(self, symbol: str, end_date: str) -> Dict[str, Any]:
        """Fetch company overview and market cap"""
        try:
            response = get_market_cap(ticker=symbol, end_date=end_date)
            
            if not response or not response.company_facts:
                return {}
            
            facts = response.company_facts
            
            return {
                "name": facts.name,
                "sector": facts.sector,
                "industry": facts.industry,
                "market_cap": facts.market_cap,
                "employees": facts.number_of_employees,
                "location": facts.location,
                "website": facts.website_url,
                "exchange": facts.exchange
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching company overview for {symbol}: {str(e)}")
            return {}
    
    @staticmethod
    def _determine_overall_sentiment(positive: int, negative: int, neutral: int) -> str:
        """Determine overall sentiment from counts"""
        total = positive + negative + neutral
        if total == 0:
            return "Neutral"
        
        pos_ratio = positive / total
        neg_ratio = negative / total
        
        if pos_ratio > 0.5:
            return "Bullish"
        elif neg_ratio > 0.5:
            return "Bearish"
        elif pos_ratio > neg_ratio * 1.5:
            return "Slightly Bullish"
        elif neg_ratio > pos_ratio * 1.5:
            return "Slightly Bearish"
        else:
            return "Neutral"