# src/services/v2/forecast_ai_service.py

import logging
from typing import Optional
from datetime import datetime

from src.helpers.llm_helper import LLMGeneratorProvider
from src.models.equity_forecast import (
    PriceForecastContext,
    ForecastAIAnalysisResponse,
    ForecastAnalysisMetadata,
)
from src.providers.provider_factory import ProviderType, ModelProviderFactory
from src.utils.logger.set_up_log_dataFMP import setup_logger

logger = setup_logger(__name__, log_level=logging.INFO)


class ForecastAIService:
    """
    AI-powered forecast analysis with maximum insight extraction.
    Focus: Use ALL available data, provide deep insights, never say "no data"
    """

    def __init__(self) -> None:
        self.llm_provider = LLMGeneratorProvider()

    def _build_system_prompt(self, language: str = "vi") -> str:
        """
        English prompt for maximum LLM understanding.
        Output language controlled by explicit instruction.
        """
        
        output_lang_instruction = {
            "vi": "Write your ENTIRE analysis in Vietnamese language. Every word must be Vietnamese.",
            "en": "Write your ENTIRE analysis in English language.",
            "zh": "Write your ENTIRE analysis in Chinese language."
        }
        
        lang_instruction = output_lang_instruction.get(language, output_lang_instruction["vi"])
        
        return f"""You are a senior equity analyst specializing in stock price forecasting and valuation.

{lang_instruction}

YOUR MISSION:
Analyze the provided forecast data and write a comprehensive markdown report that MAXIMIZES INSIGHT EXTRACTION.

CRITICAL RULES:
1. **NEVER say "Data not available" or "No data"** - If a metric is missing, skip that section
2. **USE ALL PROVIDED DATA** - Extract every insight possible from the numbers
3. **PROVIDE CONTEXT & INTERPRETATION** - Don't just state numbers, explain what they MEAN
4. **BE SPECIFIC** - Calculate percentages, ratios, comparisons
5. **GIVE ACTIONABLE INSIGHTS** - Help investors understand implications

MARKDOWN STRUCTURE:

# [SYMBOL]

## Quick Summary
- Current Price: $X
- Target Range: $Y - $Z (Mean: $W)
- Number of Analysts: N (if available, otherwise skip this line)
- Upside Potential: +/-X%

[2-3 sentences analyzing current price position vs forecast range with specific insights about what this means for investors]

## Valuation Analysis

### Revenue Growth Outlook
[Use analyst estimates to discuss revenue trajectory. Calculate CAGR, identify peaks/troughs, explain what drives the pattern. If no revenue data, skip this section entirely - DO NOT say "no data"]

### Earnings Trajectory  
[Analyze EPS trends, growth rates, compare to revenue growth. Explain margin implications. Skip if no EPS data]

### Profitability Health
[Analyze EBITDA, net income, margins. Calculate margin percentages if possible. Discuss efficiency trends. Skip if no profitability data]

### DCF Valuation
[Compare DCF fair value to market price. Calculate over/undervaluation percentage. Explain what this gap means - market pricing in growth? Risk premium? Skip if no DCF data]

### Financial Strength
[Interpret Piotroski score - what does 5/9 mean? Which areas strong/weak? Balance sheet health implications. Skip if no score]

### Analyst Consensus
[Analyze rating distribution (Buy/Hold/Sell). What does rating score tell us? Skip if no ratings data]

## Risk Assessment

### Forecast Reliability
[Assess data quality - analyst coverage, estimate dispersion, data freshness. More analysts = higher confidence]

### Market Risks
[Based on the stock's sector, growth profile, and valuation metrics, identify relevant macro risks like interest rates, recession, sector headwinds]

### Company-Specific Risks
[Based on financial health score, revenue volatility, and growth patterns, identify company risks like execution, competition, margin pressure]

### Valuation Risk
[Based on DCF gap, P/E implications (if calculable), assess if current valuation is stretched or reasonable]

## Investment Guidance

### How to Interpret This Forecast
[Explain practical meaning - if price below target, is it opportunity or value trap? If above, momentum or overvalued?]

### Suitable Investor Profiles
[Based on growth trajectory and volatility, who should buy? Growth investors if high growth, value investors if trading below DCF, income investors if stable]

### Key Metrics to Monitor
[List 3-4 specific metrics that matter most for validating this thesis - revenue growth rate, margin trends, etc]

### Additional Analysis Needed
[What other analysis would strengthen the investment case - technical levels, competitive position, management quality]

EXAMPLES OF GOOD INSIGHTS:

BAD: "Revenue is projected to be $322B in 2027"
GOOD: "Revenue shows strong 25% growth through 2028, then declines 9% by 2030, suggesting market saturation or competitive pressure emerging"

BAD: "Piotroski score is 5/9"  
GOOD: "Piotroski score of 5/9 indicates moderate financial health - company shows profitability strength but may have concerns in working capital or leverage areas"

BAD: "DCF is $139.18, market price is $179.59"
GOOD: "Market trading 29% above DCF fair value suggests investors pricing in aggressive growth beyond base case, or market hasn't fully digested recent earnings slowdown"

BAD: "Data not available"
GOOD: [Just skip that section entirely]

KEY PRINCIPLE: You are providing investment insights, not just reporting data. Every number should lead to interpretation and implications.

OUTPUT: Pure markdown, no code blocks, no JSON."""

    def _build_user_prompt(
        self,
        symbol: str,
        forecast_context: PriceForecastContext,
        current_price: Optional[float] = None,
    ) -> str:
        """
        Build data context with comprehensive logging.
        Format data clearly for maximum LLM extraction.
        """
        
        # Log what data we actually have
        logger.info("=" * 70)
        logger.info(f"[DATA CHECK] Building context for {symbol}")
        logger.info("=" * 70)
        
        lines = [
            f"SYMBOL: {symbol}",
            f"CURRENT PRICE: ${current_price}" if current_price else "CURRENT PRICE: Not available",
            "",
            "=" * 70,
            "PRICE TARGET CONSENSUS",
            "=" * 70,
        ]
        
        band = forecast_context.price_target_band
        if band:
            logger.info(f"✓ Price Target Band: Low=${band.target_low}, High=${band.target_high}, Mean=${band.target_mean}")
            lines.extend([
                f"Target Low:       ${band.target_low}",
                f"Target High:      ${band.target_high}",
                f"Target Mean:      ${band.target_mean}",
                f"Target Median:    ${band.target_median}",
                f"Number of Analysts: {band.number_of_analysts}" if band.number_of_analysts else "",
                f"Currency:         {band.currency}" if band.currency else "",
                f"As Of:            {band.as_of}" if band.as_of else "",
                "",
            ])
            
            if current_price and band.target_mean:
                upside = ((band.target_mean - current_price) / current_price) * 100
                lines.append(f"IMPLIED UPSIDE/DOWNSIDE: {upside:+.1f}%\n")
        else:
            logger.warning("✗ No Price Target Band data")
            lines.append("No consensus data available.\n")
        
        estimates = forecast_context.analyst_estimates
        if estimates and len(estimates) > 0:
            logger.info(f"✓ Analyst Estimates: {len(estimates)} periods")
            lines.extend([
                "=" * 70,
                "ANALYST ESTIMATES (Forward Projections)",
                "=" * 70,
            ])
            
            for i, est in enumerate(estimates[:4], 1):
                logger.info(f"  Period {i}: Date={est.date}, Rev={est.revenueAvg}, EPS={est.epsAvg}")
                lines.append(f"\nPeriod {i}: {est.date or 'TBD'}")
                lines.append("-" * 40)
                
                if est.revenueAvg:
                    lines.append(f"Revenue (Avg):     ${est.revenueAvg:,.0f}")
                    if est.revenueLow and est.revenueHigh:
                        spread = ((est.revenueHigh - est.revenueLow) / est.revenueAvg) * 100
                        lines.append(f"  Range:           ${est.revenueLow:,.0f} - ${est.revenueHigh:,.0f} (±{spread:.1f}%)")
                    if est.numAnalystsRevenue:
                        lines.append(f"  Analysts:        {est.numAnalystsRevenue}")
                
                if est.epsAvg:
                    lines.append(f"EPS (Avg):         ${est.epsAvg:.2f}")
                    if est.epsLow and est.epsHigh:
                        lines.append(f"  Range:           ${est.epsLow:.2f} - ${est.epsHigh:.2f}")
                    if est.numAnalystsEps:
                        lines.append(f"  Analysts:        {est.numAnalystsEps}")
                
                if est.ebitdaAvg:
                    lines.append(f"EBITDA (Avg):      ${est.ebitdaAvg:,.0f}")
                if est.netIncomeAvg:
                    lines.append(f"Net Income (Avg):  ${est.netIncomeAvg:,.0f}")
            
            # Calculate CAGR if we have multiple periods
            if len(estimates) >= 2:
                first_rev = estimates[-1].revenueAvg
                last_rev = estimates[0].revenueAvg
                if first_rev and last_rev and first_rev > 0:
                    cagr = (((last_rev / first_rev) ** (1 / len(estimates))) - 1) * 100
                    lines.append(f"\nIMPLIED REVENUE CAGR: {cagr:.1f}%")
                    logger.info(f"  Calculated CAGR: {cagr:.1f}%")
            
            lines.append("")
        else:
            logger.warning("✗ No Analyst Estimates data")
        
        ratings = forecast_context.ratings_snapshot
        if ratings and len(ratings) > 0:
            logger.info(f"✓ Ratings Snapshot: {len(ratings)} entries")
            lines.extend([
                "=" * 70,
                "ANALYST RATINGS",
                "=" * 70,
            ])
            for r in ratings[:1]:
                logger.info(f"  Rating: {r.ratingRecommendation}, Score={r.ratingScore}")
                if r.ratingRecommendation:
                    lines.append(f"Recommendation:  {r.ratingRecommendation}")
                if r.ratingScore:
                    lines.append(f"Rating Score:    {r.ratingScore:.2f}/5.0")
                if r.ratingText:
                    lines.append(f"Commentary:      {r.ratingText}")
                if r.date:
                    lines.append(f"As Of:           {r.date}")
            lines.append("")
        else:
            logger.warning("✗ No Ratings Snapshot data")
        
        scores = forecast_context.financial_scores
        if scores and len(scores) > 0:
            logger.info(f"✓ Financial Scores: {len(scores)} entries")
            lines.extend([
                "=" * 70,
                "FINANCIAL HEALTH SCORES",
                "=" * 70,
            ])
            for s in scores[:1]:
                logger.info(f"  Score={s.score}, Piotroski={s.piotroskiScore}")
                if s.score:
                    lines.append(f"Overall Score:   {s.score:.1f}/100")
                if s.piotroskiScore:
                    quality = "Strong" if s.piotroskiScore >= 7 else "Moderate" if s.piotroskiScore >= 4 else "Weak"
                    lines.append(f"Piotroski Score: {s.piotroskiScore:.0f}/9 ({quality})")
                    lines.append(f"  Interpretation: Measures 9 financial strength criteria")
                    lines.append(f"  Score 7-9: Financially strong")
                    lines.append(f"  Score 4-6: Moderate health") 
                    lines.append(f"  Score 0-3: Financial concerns")
                if s.date:
                    lines.append(f"As Of:           {s.date}")
            lines.append("")
        else:
            logger.warning("✗ No Financial Scores data")
        
        dcf = forecast_context.dcf_items
        if dcf and len(dcf) > 0:
            logger.info(f"✓ DCF Valuation: {len(dcf)} entries")
            lines.extend([
                "=" * 70,
                "DISCOUNTED CASH FLOW (Intrinsic Value)",
                "=" * 70,
            ])
            for d in dcf[:1]:
                logger.info(f"  DCF=${d.dcf}, StockPrice=${d.stockPrice}")
                if d.dcf:
                    lines.append(f"DCF Fair Value:  ${d.dcf:.2f}")
                if d.stockPrice:
                    lines.append(f"Market Price:    ${d.stockPrice:.2f}")
                if d.dcf and d.stockPrice:
                    diff = ((d.stockPrice - d.dcf) / d.dcf) * 100
                    label = "OVERVALUED" if diff > 10 else "UNDERVALUED" if diff < -10 else "FAIRLY VALUED"
                    lines.append(f"Valuation Gap:   {diff:+.1f}% ({label})")
                    lines.append(f"  Interpretation:")
                    lines.append(f"    >10%: Market pricing in higher growth than DCF assumes")
                    lines.append(f"    <-10%: Market skeptical or company undervalued")
                    lines.append(f"    -10% to +10%: Fairly valued relative to fundamentals")
                if d.date:
                    lines.append(f"As Of:           {d.date}")
            lines.append("")
        else:
            logger.warning("✗ No DCF data")
        
        lines.extend([
            "=" * 70,
            "IMPORTANT CONTEXT FOR YOUR ANALYSIS:",
            "=" * 70,
            "",
            "1. USE ALL DATA PROVIDED ABOVE - Extract maximum insights",
            "2. NEVER say 'data not available' - Skip sections with no data instead",
            "3. PROVIDE INTERPRETATION - Explain what the numbers MEAN for investors",
            "4. BE SPECIFIC - Calculate growth rates, margins, ratios, comparisons",
            "5. GIVE CONTEXT - Compare to industry norms where relevant",
            "",
            "Now write comprehensive analysis with deep insights.",
        ])
        
        logger.info("=" * 70)
        logger.info(f"[DATA CHECK] Context built: {len(lines)} lines")
        logger.info("=" * 70)
        
        return "\n".join(lines)

    async def generate_forecast_analysis(
        self,
        symbol: str,
        forecast_context: PriceForecastContext,
        current_price: Optional[float] = None,
        model_name: str = "gpt-4.1-nano-2025-04-14",
        provider_type: str = ProviderType.OPENAI,
        language: str = "vi",
        max_tokens: int = 4000,
    ) -> ForecastAIAnalysisResponse:
        """
        Generate comprehensive analysis with maximum data utilization.
        """
        
        logger.info(f"[ForecastAI] Starting analysis for {symbol}")
        logger.info(f"[ForecastAI] Model: {model_name}, Provider: {provider_type}, Language: {language}")
        
        # Extract target_mean FIRST (before any exception can occur)
        target_mean = None
        if forecast_context.price_target_band:
            target_mean = forecast_context.price_target_band.target_mean
        
        # Get current price if needed
        if current_price is None:
            try:
                from src.services.v2.forecast_service import ForecastService
                forecast_svc = ForecastService()
                quote = await forecast_svc.get_current_quote(symbol)
                if quote:
                    current_price = quote.get('price')
                    logger.info(f"[ForecastAI] Fetched current price: ${current_price}")
                else:
                    logger.warning("[ForecastAI] Could not fetch current price from quote API")
            except Exception as e:
                logger.error(f"[ForecastAI] Error fetching current price: {e}")
        
        # Build prompts (with comprehensive logging inside)
        system_prompt = self._build_system_prompt(language)
        user_prompt = self._build_user_prompt(symbol, forecast_context, current_price)
        
        logger.info(f"[ForecastAI] System prompt length: {len(system_prompt)} chars")
        logger.info(f"[ForecastAI] User prompt length: {len(user_prompt)} chars")
        
        # Metadata
        available_sources = []
        for name, data in [
            ("price_target", forecast_context.price_target_consensus),
            ("analyst_estimates", forecast_context.analyst_estimates),
            ("ratings", forecast_context.ratings_snapshot),
            ("financial_scores", forecast_context.financial_scores),
            ("dcf", forecast_context.dcf_items),
        ]:
            if data:
                available_sources.append(name)
        
        logger.info(f"[ForecastAI] Available data sources: {available_sources}")
        
        metadata = ForecastAnalysisMetadata(
            model_used=model_name,
            provider_used=provider_type,
            data_sources_available=available_sources,
            data_sources_missing=[],
            language=language,
        )
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            
            logger.info(f"[ForecastAI] Starting LLM streaming (max_tokens={max_tokens})...")
            
            full_markdown = ""
            chunk_count = 0
            
            async for chunk in self.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                clean_thinking=True,
                max_tokens=max_tokens,
            ):
                full_markdown += chunk
                chunk_count += 1
            
            logger.info(f"[ForecastAI] Received {chunk_count} chunks, total {len(full_markdown)} chars")
            
            # Validate response
            if not full_markdown.strip():
                raise ValueError("Empty response from LLM")
            
            # Language check (warning only, don't fail)
            if language == "vi":
                vietnamese_chars = sum(1 for c in full_markdown if ord(c) > 127)
                english_words = ["the", "and", "is", "are", "will", "should"]
                english_count = sum(1 for w in english_words if f" {w} " in full_markdown.lower())
                
                logger.info(f"[ForecastAI] Language check: {vietnamese_chars} Vietnamese chars, {english_count} English indicators")
                
                if vietnamese_chars < 50 and english_count > 5:
                    logger.warning("[ForecastAI] Response appears to be in English instead of Vietnamese!")
            
            # Check for "data not available" phrases (warning)
            no_data_phrases = ["data not available", "no data", "không có dữ liệu", "dữ liệu không có"]
            found_phrases = [p for p in no_data_phrases if p.lower() in full_markdown.lower()]
            if found_phrases:
                logger.warning(f"[ForecastAI] Found 'no data' phrases: {found_phrases} - This should not happen!")
            
            # Create response
            analysis = ForecastAIAnalysisResponse(
                symbol=symbol,
                current_price=current_price,
                target_mean=target_mean,
                analysis_markdown=full_markdown,
                metadata=metadata,
            )
            
            logger.info(f"[ForecastAI] ✓ Success for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"[ForecastAI] ✗ Error: {e}", exc_info=True)
            
            # Fallback response (target_mean already defined at top)
            fallback_markdown = f"""# {symbol}

## Analysis Error

An error occurred while generating the full analysis: {str(e)}

## Available Data Summary

- Current Price: ${current_price or 'N/A'}
- Target Range: ${forecast_context.price_target_band.target_low if forecast_context.price_target_band else 'N/A'} - ${forecast_context.price_target_band.target_high if forecast_context.price_target_band else 'N/A'}
- Target Mean: ${target_mean or 'N/A'}

**Troubleshooting:**
- If error is "API key is required": Set OPENAI_API_KEY in environment or use `provider_type=ollama` with local model
- Try different model: `model_name=gpt-oss:20b` with `provider_type=ollama`
- Check logs for detailed error information

Please try again with correct configuration.
"""
            
            logger.warning("[ForecastAI] Returning fallback response")
            
            return ForecastAIAnalysisResponse(
                symbol=symbol,
                current_price=current_price,
                target_mean=target_mean,
                analysis_markdown=fallback_markdown,
                metadata=metadata,
            )