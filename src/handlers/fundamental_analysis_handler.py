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
                return_exceptions=True
            )
            
            key_metrics, income_stmt, balance_sheet, cash_flow, quote, growth_data = results
            
            # Handle exceptions
            key_metrics = key_metrics if not isinstance(key_metrics, Exception) else None
            income_stmt = income_stmt if not isinstance(income_stmt, Exception) else None
            balance_sheet = balance_sheet if not isinstance(balance_sheet, Exception) else None
            cash_flow = cash_flow if not isinstance(cash_flow, Exception) else None
            quote = quote if not isinstance(quote, Exception) else None
            growth_data = growth_data if not isinstance(growth_data, Exception) else None
            
            # Extract data
            km_data = key_metrics[0] if key_metrics else {}
            
            # 2. Calculate all metrics
            report = {
                "symbol": symbol.upper(),
                "generated": datetime.now().isoformat(timespec="seconds"),
                "valuation": {},
                "growth": {},
                "profitability": {},
                "leverage": {},
                "cashflow": {},
                "risk": {}
            }
            
            # Valuation metrics
            report["valuation"] = {
                "price": quote.get("price") if quote else None,
                "pe": km_data.get("peRatio"),
                "pb": km_data.get("pbRatio"),
                "ps": km_data.get("priceToSalesRatio"),
                "peg": km_data.get("pegRatio")
            }
            
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
                
                if revenue is not None and net_income is not None and revenue > 0:
                    net_margin = (net_income / revenue * 100)
                else:
                    net_margin = None
                
                report["profitability"] = {
                    "net_margin": round(net_margin, 2) if net_margin else None,
                    "roe": km_data.get("roe"),
                    "roa": km_data.get("roa")
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
                market_cap = km_data.get("marketCap", 0)
                
                # fcf_yield = (fcf / market_cap * 100) if market_cap > 0 else None
                if fcf is not None and market_cap is not None and market_cap > 0:
                    fcf_yield = (fcf / market_cap * 100)
                else:
                    fcf_yield = None

                # report["cashflow"] = {
                #     "fcf": int(fcf) if fcf else None,
                #     "fcf_yield": round(fcf_yield, 2) if fcf_yield else None
                # }
                report["cashflow"] = {
                    "fcf": int(fcf) if fcf is not None else None,
                    "fcf_yield": round(fcf_yield, 2) if fcf_yield is not None else None
                }
            else:
                report["cashflow"] = {
                    "fcf": None,
                    "fcf_yield": None
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
            
            detection_method = ""
            if len(user_question.split()) < 2:
                detection_method = DetectionMethod.LLM
            else:
                detection_method = DetectionMethod.LIBRARY

            # Language detection
            language_info = await language_detector.detect(
                text=user_question,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )

            detected_language = language_info["detected_language"]

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
            system_prompt = f"""You are an expert fundamental analyst with 20+ years experience.

    {language_instruction}

    ## DATA INTEGRITY RULES (CRITICAL)

    ### 1. PERIOD TAGS (MANDATORY)
    Every financial metric MUST include its period:
    - [FY2025] for fiscal year data
    - [TTM] for trailing twelve months
    - [Q3 FY2025] for quarterly data
    - Example: "Revenue [FY2025]: $130.5B" NOT just "Revenue: $130.5B"

    ### 2. DATA SOURCE
    All data comes from Financial Modeling Prep (FMP) API.
    When citing key metrics, mention: "(Source: FMP API, [period])"

    ### 3. SEPARATE FACTS vs INTERPRETATION vs ACTION
    Structure your analysis with clear separation:
    - **FACTS**: Raw data with period tags and source
    - **INTERPRETATION**: Your analysis of what the data means
    - **ACTION**: Recommendations based on interpretation

    ### 4. PEG RATIO CALCULATION (IMPORTANT)
    - Standard PEG = P/E ÷ FORWARD earnings growth estimate (not historical CAGR)
    - If using historical CAGR, explicitly state: "Using historical 5Y EPS CAGR (not forward estimates)"
    - PEG < 1.0 suggests undervalued relative to growth
    - PEG > 2.0 suggests overvalued relative to growth

    ### 5. BULL/BASE/BEAR SCENARIOS (REQUIRED)
    Always include a brief scenario analysis:
    - **Bull Case**: What needs to happen for upside (1-2 sentences)
    - **Base Case**: Most likely outcome (1-2 sentences)
    - **Bear Case**: Key risks and downside triggers (1-2 sentences)

    ## CONTEXT AWARENESS
    When previous analyses are provided in context:
    - Reference significant changes in fundamental metrics
    - Update investment thesis based on new data
    - Highlight improvement or deterioration in financial health
    - Compare current valuation with previous assessments

    ## OUTPUT QUALITY
    - Data-driven analysis with specific metrics AND period tags
    - Clear investment recommendations with confidence level
    - Risk/reward assessment with quantified scenarios
    - Professional tone but easy to understand
    - Use appropriate icons for visual appeal
    - Explain financial terms briefly for general audience"""

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
        """
        # Extract latest growth data with date
        latest_growth = growth_data[0] if growth_data else {}
        growth_date = latest_growth.get('date', 'N/A')

        # Determine period type from report
        report_date = report.get('generated', datetime.now().isoformat())[:10]

        prompt = f"""
    Analyze the comprehensive fundamental data for {symbol}:

    ═══════════════════════════════════════════════════════════════════════════════
    📊 FUNDAMENTAL METRICS REPORT
    Data Source: Financial Modeling Prep (FMP) API
    Report Generated: {report_date}
    Period: TTM (Trailing Twelve Months) unless otherwise noted
    ═══════════════════════════════════════════════════════════════════════════════

    {json.dumps(report, indent=2)}

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
    - Confidence level: HIGH / MEDIUM / LOW (with justification)
    - **NOTE**: Rating is metrics-based, not investment advice

    ### 2. 💯 FINANCIAL HEALTH SCORE (0-100)
    - Calculate weighted score based on all metrics
    - Breakdown by category with individual scores:
      | Category | Score (0-20) | Key Metrics |
      |----------|--------------|-------------|
      | Valuation | X/20 | P/E, P/S, P/B |
      | Growth | X/20 | Rev CAGR, EPS CAGR |
      | Profitability | X/20 | Net Margin, ROE |
      | Leverage | X/20 | D/E, Current Ratio |
      | Cash Flow | X/20 | FCF, FCF Yield |

    ### 3. ✅ KEY STRENGTHS (3-5 points)
    - Use exact metrics WITH period tags: "Net Margin [TTM]: 55.8%"
    - Explain WHY each metric is strong

    ### 4. ⚠️ CRITICAL RISKS (2-3 points)
    - Quantify risks with specific metrics
    - Include both financial and business risks

    ### 5. 💰 VALUATION ASSESSMENT
    - Is it overvalued/fairly valued/undervalued?
    - Compare P/E, P/B, P/S to growth rates

    **PEG Analysis** (IMPORTANT):
    - If PEG is available in data, use it directly
    - If calculating manually:
      - State clearly: "PEG calculated using [5Y historical CAGR / forward estimates]"
      - PEG = P/E ÷ EPS Growth Rate
      - Note: Using historical CAGR is less accurate than forward estimates
    - Interpretation: PEG < 1.0 = potentially undervalued, > 2.0 = potentially overvalued

    ### 6. 🎯 SCENARIO ANALYSIS (REQUIRED)
    Provide brief scenarios (1-2 sentences each):

    | Scenario | Description | Trigger |
    |----------|-------------|---------|
    | **Bull** 🐂 | Upside case | What needs to happen |
    | **Base** ⚖️ | Most likely | Expected outcome |
    | **Bear** 🐻 | Downside risk | Key risk triggers |

    ### 7. 📋 ACTIONABLE RECOMMENDATIONS
    - Entry strategy (general guidance, not specific prices)
    - Position sizing suggestion (% of portfolio)
    - Time horizon
    - Key catalysts to watch

    ### 8. 📡 KEY METRICS TO MONITOR
    - Which 3-5 metrics are most critical for this stock?
    - What changes would alter your thesis?

    ═══════════════════════════════════════════════════════════════════════════════
    ⚠️ IMPORTANT FORMATTING RULES
    ═══════════════════════════════════════════════════════════════════════════════

    1. ALWAYS include period tags: [TTM], [FY2025], [Q3 FY2025], [5Y CAGR]
    2. Cite data source when using key numbers: "(FMP API, TTM)"
    3. Use exact numbers from the data - never fabricate
    4. Separate FACTS from INTERPRETATION clearly
    5. Include Bull/Base/Bear scenarios
    6. End with disclaimer: Analysis is informational, not investment advice
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