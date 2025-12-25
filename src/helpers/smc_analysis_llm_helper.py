from typing import Optional, AsyncGenerator

from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType
from src.schemas.smc_models import SMCAnalysisResult


class SMCAnalysisLLMHelper(LoggerMixin):
    """
    Helper class for generating LLM-based SMC analysis interpretations.
    Uses professional prompt engineering techniques for high-quality trading analysis.
    """
    
    def __init__(self):
        super().__init__()
        self.llm_provider = LLMGeneratorProvider()
        self.logger.info("[SMC-LLM] SMCAnalysisLLMHelper initialized")
    
    def build_system_prompt(self, target_language: str = "en") -> str:
        """
        Build system prompt for SMC analysis interpretation.
        
        Follows prompt engineering best practices:
        - Compact, data-driven format
        - Numbers > Text, Action > Analysis
        - Every claim backed by specific data
        """
        
        lang_instruction = {
            "vi": """Responses must be entirely in Vietnamese. 
- Translate ALL section headers, labels, and explanations to Vietnamese
- Keep technical terms in English with Vietnamese explanation on first use. 
  Example: "BOS (Break of Structure - Phá vỡ cấu trúc)"
- Tone: direct, confident, like a senior trader briefing a colleague    
""",
            "en": """Responses must be entirely in English.
- Translate ALL section headers, labels, and explanations to English
- Define technical terms on first use
- Clear and actionable
""",
            "zh": """Responses must be entirely in Chinese.
- Translate ALL headers and explanations to Simplified Chinese
- Technical terms: English with Chinese explanation
  Example: "BOS (Break of Structure - 结构突破)"
- Professional but accessible tone"""
        }
        
        language_rule = lang_instruction.get(target_language, lang_instruction["en"])
        
        return f"""<role>
You are an elite SMC (Smart Money Concepts) analyst.
Goal: Make analysis EASY TO UNDERSTAND and ACTIONABLE.
</role>

<language_rule>
{language_rule}
</language_rule>

<output_format>
Follow this structure. Translate headers naturally based on the requested language.

---

✅ **1. Market Context**

Current structure: [Bullish/Bearish/Ranging]
- [Key observation 1 about structure - HL, HH, BOS, CHoCH]
- [Key observation 2 about recent price action]
- [Key observation 3 if relevant]

➡️ Overall Bias: [LONG/SHORT/NEUTRAL] - briefly why

---

✅ **2. Key Levels**

**Resistance (targets for longs / entries for shorts):**
1. [price range] → [what this level is - OB, liquidity, etc.]
2. [price range] → [description]

**Support (entries for longs / targets for shorts):**
1. [price range] → [what this level is - OB, FVG, demand zone]
2. [price range] → [description]
3. [price range] → [deep support if relevant]

---

✅ **3. Liquidity Analysis**

- [Liquidity observation 1 - where are the stops/orders sitting]
- [Liquidity observation 2 - weak highs/lows that may get hunted]
- [Liquidity observation 3 if relevant]

➡️ [Conclusion about likely price movement based on liquidity]

---

✅ **4. Price Action Read**

Current situation:
- [Where price is now relative to key levels]
- [What price is doing - consolidating, pushing, rejecting]

Key signal:
👉 [Most important observation that determines trade direction]

---

⚡ **5. Trading Scenarios**

🟢 **SCENARIO 1 - [LONG/SHORT] (Preferred)**

**Entry Zone:**
- [price range]
- Why: [2-3 bullet points explaining confluence]

**Targets:**
1. [price] → [reason - liquidity, OB, etc.]
2. [price] → [reason]
3. [price] → [extended target]

**Stop Loss:**
- Below/Above [price] - [reason]

➡️ R:R = [ratio] - [assessment: excellent/good/acceptable]

---

🔴 **SCENARIO 2 - [Opposite direction] (Counter-trend / Backup)**

**Entry Zone:**
- [price range]

**Targets:**
- [price range]

**Stop Loss:**
- [price]

⚠️ This is counter-trend. Only trade if:
- [Confirmation 1 required]
- [Confirmation 2 required]

---

❌ **Invalidation - When is this analysis wrong?**

If:
- [Condition 1 - specific price level break]
- [Condition 2 - structure change]

→ [What happens next if invalidated]

---

⭐ **Conclusion**

➡️ [1 sentence: overall bias and confidence]
➡️ [1 sentence: best entry zone]
➡️ [1 sentence: targets summary]

---

</output_format>

<writing_rules>
1. Use these emojis consistently: ✅ for sections, ➡️ for conclusions, 🟢 for bullish/preferred, 🔴 for bearish/counter, ⚠️ for warnings, ⭐ for summary, ❌ for invalidation, 👉 for key points
2. Keep bullet points short - one line each
3. Every price level needs context (what it is, why it matters)
4. Show R:R ratio for main scenario
5. Always include invalidation conditions
6. End with clear, actionable conclusion
7. Terms on first use: "CHoCH (Change of Character - trend reversal signal)"
8. Be confident but honest about risk
</writing_rules>

<quality_check>
Before sending, verify:
- Headers translated naturally?
- Each level has explanation?
- R:R calculated?
- Invalidation conditions clear?
- Conclusion is actionable?
</quality_check>"""

    def build_analysis_prompt(
        self,
        analysis: SMCAnalysisResult,
        user_question: Optional[str] = None
    ) -> str:
        """Build user prompt containing analysis data in structured format."""
        
        # Format entry zones
        entry_zones_text = ""
        for i, ez in enumerate(analysis.trading_plan.entry_zones, 1):
            entry_zones_text += f"""
  Entry Zone {i}:
    - Type: {ez.zone_type}
    - Range: {ez.price_range.get('low', 'N/A')} - {ez.price_range.get('high', 'N/A')}
    - Confidence: {ez.confidence}
    - Reasons: {', '.join(ez.reasons)}"""
        
        # Format targets
        targets_text = ""
        for i, t in enumerate(analysis.trading_plan.targets, 1):
            targets_text += f"""
  Target {i}: {t.price} ({t.target_type}) - {t.reasoning}"""
        
        # Format Order Blocks
        bullish_obs_text = ""
        for i, ob in enumerate(analysis.order_block_analysis.active_bullish_obs, 1):
            bullish_obs_text += f"""
  Bullish OB {i}: {ob['low']:.2f} - {ob['high']:.2f} | Distance: {ob['distance_pct']:.2f}% | Size: {ob['zone_size']:.2f}"""
        
        bearish_obs_text = ""
        for i, ob in enumerate(analysis.order_block_analysis.active_bearish_obs, 1):
            bearish_obs_text += f"""
  Bearish OB {i}: {ob['low']:.2f} - {ob['high']:.2f} | Distance: {ob['distance_pct']:.2f}% | Size: {ob['zone_size']:.2f}"""
        
        # Format Liquidity
        buy_liq_text = ""
        for liq in analysis.liquidity_analysis.buy_side_liquidity:
            buy_liq_text += f"\n  EQH: {liq['level']:.2f} | Distance: {liq['distance_pct']:.2f}%"
        
        sell_liq_text = ""
        for liq in analysis.liquidity_analysis.sell_side_liquidity:
            sell_liq_text += f"\n  EQL: {liq['level']:.2f} | Distance: {liq['distance_pct']:.2f}%"
        
        # Format FVGs
        bullish_fvg_text = ""
        for i, fvg in enumerate(analysis.fvg_analysis.active_bullish_fvgs, 1):
            bullish_fvg_text += f"""
  Bullish FVG {i}: {fvg['bottom_price']:.2f} - {fvg['top_price']:.2f} | Gap: {fvg['gap_size']:.2f} | Distance: {fvg['distance_pct']:.2f}%"""
        
        bearish_fvg_text = ""
        for i, fvg in enumerate(analysis.fvg_analysis.active_bearish_fvgs, 1):
            bearish_fvg_text += f"""
  Bearish FVG {i}: {fvg['bottom_price']:.2f} - {fvg['top_price']:.2f} | Gap: {fvg['gap_size']:.2f} | Distance: {fvg['distance_pct']:.2f}%"""

        prompt = f"""<smc_analysis_data>

=== SYMBOL & TIMEFRAME ===
Symbol: {analysis.symbol}
Interval: {analysis.interval}
Current Price: {analysis.current_price:.2f}
Analysis Time: {analysis.timestamp}
Data Quality: {analysis.data_quality_score:.1f}/100
Analysis Confidence: {analysis.analysis_confidence.upper()}

=== TREND ANALYSIS ===
Direction: {analysis.trend_analysis.direction.upper()}
Strength: {analysis.trend_analysis.strength.upper()}
Last Structure Event: {analysis.trend_analysis.last_structure_event}
Confirmation Level: {analysis.trend_analysis.confirmation_level.upper()}
Evidence: {analysis.trend_analysis.reasoning}

=== MARKET STRUCTURE ===
Total BOS Events: {analysis.structure_analysis.bos_events}
Total CHoCH Events: {analysis.structure_analysis.choch_events}
Last BOS Direction: {analysis.structure_analysis.last_bos_direction or 'N/A'}
Last CHoCH Direction: {analysis.structure_analysis.last_choch_direction or 'N/A'}
Structure Bias: {analysis.structure_analysis.structure_bias.upper()}

Swing Highs: {', '.join([f'{h:.2f}' for h in analysis.structure_analysis.swing_highs]) or 'None'}
Swing Lows: {', '.join([f'{l:.2f}' for l in analysis.structure_analysis.swing_lows]) or 'None'}

=== ORDER BLOCKS ===
Total Active: {analysis.order_block_analysis.total_active}
Total Mitigated: {analysis.order_block_analysis.total_mitigated}

Active Bullish OBs (Demand Zones):{bullish_obs_text or ' None'}

Active Bearish OBs (Supply Zones):{bearish_obs_text or ' None'}

Strongest Demand Zone: {f"{analysis.order_block_analysis.strongest_demand_zone['low']:.2f} - {analysis.order_block_analysis.strongest_demand_zone['high']:.2f}" if analysis.order_block_analysis.strongest_demand_zone else 'None'}
Strongest Supply Zone: {f"{analysis.order_block_analysis.strongest_supply_zone['low']:.2f} - {analysis.order_block_analysis.strongest_supply_zone['high']:.2f}" if analysis.order_block_analysis.strongest_supply_zone else 'None'}

=== LIQUIDITY ZONES ===
Buy-Side Liquidity (EQH - Upside Targets):{buy_liq_text or ' None'}

Sell-Side Liquidity (EQL - Downside Targets):{sell_liq_text or ' None'}

Nearest EQH: {f'{analysis.liquidity_analysis.nearest_eqh:.2f}' if analysis.liquidity_analysis.nearest_eqh else 'None'}
Nearest EQL: {f'{analysis.liquidity_analysis.nearest_eql:.2f}' if analysis.liquidity_analysis.nearest_eql else 'None'}
Potential Sweep Targets: {', '.join([f'{t:.2f}' for t in analysis.liquidity_analysis.potential_sweep_targets]) or 'None'}

=== FAIR VALUE GAPS ===
Total Active FVGs: {analysis.fvg_analysis.total_active}
Total Mitigated FVGs: {analysis.fvg_analysis.total_mitigated}

Active Bullish FVGs:{bullish_fvg_text or ' None'}

Active Bearish FVGs:{bearish_fvg_text or ' None'}

Nearest Unfilled FVG: {f"{analysis.fvg_analysis.nearest_unfilled_fvg['type']} at {analysis.fvg_analysis.nearest_unfilled_fvg['bottom_price']:.2f} - {analysis.fvg_analysis.nearest_unfilled_fvg['top_price']:.2f}" if analysis.fvg_analysis.nearest_unfilled_fvg else 'None'}

=== PREMIUM/DISCOUNT ANALYSIS ===
Current Zone: {analysis.premium_discount_analysis.current_zone.upper()}
Equilibrium Price: {analysis.premium_discount_analysis.equilibrium_price:.2f}
Premium Threshold: {analysis.premium_discount_analysis.premium_threshold:.2f}
Discount Threshold: {analysis.premium_discount_analysis.discount_threshold:.2f}
Distance from Equilibrium: {abs(analysis.premium_discount_analysis.distance_to_equilibrium_pct):.2f}% {'above' if analysis.premium_discount_analysis.distance_to_equilibrium_pct > 0 else 'below'}
Zone Recommendation: {analysis.premium_discount_analysis.trade_recommendation}

=== TRADING PLAN (Pre-computed) ===
Bias: {analysis.trading_plan.bias.upper()}
Signal Strength: {analysis.trading_plan.signal_strength.upper()}
Recommended Action: {analysis.trading_plan.recommended_action}

Entry Zones:{entry_zones_text or ' None defined'}

Stop Loss: {analysis.trading_plan.stop_loss:.2f}
Stop Loss Reasoning: {analysis.trading_plan.stop_loss_reasoning}

Targets:{targets_text or ' None defined'}

Risk:Reward Ratio: {analysis.trading_plan.risk_reward_ratio:.2f}
Invalidation Level: {analysis.trading_plan.invalidation_level:.2f}

Key Warnings:
{chr(10).join([f'  - {w}' for w in analysis.trading_plan.key_warnings]) or '  None'}

=== EXECUTIVE SUMMARY ===
{analysis.executive_summary}

</smc_analysis_data>

Based on the structured SMC data above, provide your professional analysis following the exact 11-section output format specified in your instructions. Be specific with price levels and actionable in your recommendations."""

        if user_question:
            prompt += f"""

<user_question>
The trader has an additional question:
{user_question}

Address this question within your analysis, particularly in the relevant sections.
</user_question>"""

        return prompt

    def _get_api_key(self, provider_type: str) -> Optional[str]:
        """Get API key for provider."""
        if provider_type in [ProviderType.OPENAI, "openai"]:
            return settings.OPENAI_API_KEY
        elif provider_type in [ProviderType.GEMINI, "gemini"]:
            return settings.GEMINI_API_KEY
        return None

    async def generate_interpretation(
        self,
        analysis: SMCAnalysisResult,
        user_question: Optional[str] = None,
        target_language: str = "en",
        model_name: str = "gpt-4.1-nano",
        provider_type: str = "openai",
    ) -> str:
        """Generate LLM interpretation of SMC analysis."""
        
        system_prompt = self.build_system_prompt(target_language)
        user_prompt = self.build_analysis_prompt(analysis, user_question)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        self.logger.info(f"[SMC-LLM] Generating interpretation with {model_name}")
        
        try:
            api_key = self._get_api_key(provider_type)
            
            response = await self.llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.3,
                max_tokens=4000  # Increased for detailed output
            )
            
            if isinstance(response, dict):
                content = response.get("content", "")
                if isinstance(content, list) and len(content) > 0:
                    content = content[0].get("text", "")
            else:
                content = str(response)
            
            self.logger.info("[SMC-LLM] Interpretation generated successfully")
            return content
            
        except Exception as e:
            self.logger.error(f"[SMC-LLM] Error: {e}")
            return self._generate_fallback_analysis(analysis)

    async def stream_interpretation(
        self,
        analysis: SMCAnalysisResult,
        user_question: Optional[str] = None,
        target_language: str = "en",
        model_name: str = "gpt-4.1-nano",
        provider_type: str = "openai",
    ) -> AsyncGenerator[str, None]:
        """Stream LLM interpretation."""
        
        system_prompt = self.build_system_prompt(target_language)
        user_prompt = self.build_analysis_prompt(analysis, user_question)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        self.logger.info(f"[SMC-LLM] Streaming interpretation with {model_name}")
        
        try:
            api_key = self._get_api_key(provider_type)
            
            async for chunk in self.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.3,
                max_tokens=4000
            ):
                if chunk:
                    yield chunk
                    
        except Exception as e:
            self.logger.error(f"[SMC-LLM] Stream error: {e}")
            yield self._generate_fallback_analysis(analysis)

    def _generate_fallback_analysis(self, analysis: SMCAnalysisResult) -> str:
        """Generate fallback analysis when LLM fails."""
        
        # Build basic entry zones text
        entry_text = ""
        for ez in analysis.trading_plan.entry_zones:
            entry_text += f"\n- {ez.zone_type.upper()}: {ez.price_range.get('low', 'N/A')} - {ez.price_range.get('high', 'N/A')} ({ez.confidence} confidence)"
        
        # Build targets text
        targets_text = ""
        for t in analysis.trading_plan.targets:
            targets_text += f"\n- {t.price:.2f} ({t.target_type})"
        
        return f"""## SMC Analysis Summary (Fallback Mode)

**Symbol:** {analysis.symbol} | **Timeframe:** {analysis.interval}
**Current Price:** {analysis.current_price:.2f}

### 1. MARKET BIAS & TREND DIRECTION
- **Direction:** {analysis.trend_analysis.direction.upper()}
- **Strength:** {analysis.trend_analysis.strength.upper()}
- **Confirmation:** {analysis.trend_analysis.confirmation_level.upper()}

### 2. MARKET STRUCTURE ANALYSIS
- BOS Events: {analysis.structure_analysis.bos_events}
- CHoCH Events: {analysis.structure_analysis.choch_events}
- Structure Bias: {analysis.structure_analysis.structure_bias.upper()}

### 3. SWING POINTS IDENTIFICATION
- Swing Highs: {', '.join([f'{h:.2f}' for h in analysis.structure_analysis.swing_highs]) or 'None'}
- Swing Lows: {', '.join([f'{l:.2f}' for l in analysis.structure_analysis.swing_lows]) or 'None'}

### 4. PREMIUM/DISCOUNT ZONE
- Current Zone: **{analysis.premium_discount_analysis.current_zone.upper()}**
- Equilibrium: {analysis.premium_discount_analysis.equilibrium_price:.2f}

### 5. ORDER BLOCK ANALYSIS
- Active Demand Zones: {len(analysis.order_block_analysis.active_bullish_obs)}
- Active Supply Zones: {len(analysis.order_block_analysis.active_bearish_obs)}
- Strongest Demand: {f"{analysis.order_block_analysis.strongest_demand_zone['low']:.2f} - {analysis.order_block_analysis.strongest_demand_zone['high']:.2f}" if analysis.order_block_analysis.strongest_demand_zone else 'None'}

### 6. LIQUIDITY MAPPING
- Nearest EQH (Target): {f'{analysis.liquidity_analysis.nearest_eqh:.2f}' if analysis.liquidity_analysis.nearest_eqh else 'None'}
- Nearest EQL (Support): {f'{analysis.liquidity_analysis.nearest_eql:.2f}' if analysis.liquidity_analysis.nearest_eql else 'None'}

### 7. FAIR VALUE GAPS
- Active Bullish FVGs: {len(analysis.fvg_analysis.active_bullish_fvgs)}
- Active Bearish FVGs: {len(analysis.fvg_analysis.active_bearish_fvgs)}
- Nearest: {f"{analysis.fvg_analysis.nearest_unfilled_fvg['type']} at {analysis.fvg_analysis.nearest_unfilled_fvg['bottom_price']:.2f} - {analysis.fvg_analysis.nearest_unfilled_fvg['top_price']:.2f}" if analysis.fvg_analysis.nearest_unfilled_fvg else 'None'}

### 8. TRADING PLAN
```
BIAS: {analysis.trading_plan.bias.upper()}
ACTION: {analysis.trading_plan.recommended_action}
ENTRY ZONES: {entry_text or 'None'}
STOP LOSS: {analysis.trading_plan.stop_loss:.2f}
TARGETS: {targets_text or 'None'}
R:R = {analysis.trading_plan.risk_reward_ratio:.2f}
```

### 9. RISK ASSESSMENT
- R:R Quality: {'OPTIMAL' if analysis.trading_plan.risk_reward_ratio >= 2 else 'ACCEPTABLE' if analysis.trading_plan.risk_reward_ratio >= 1.5 else 'POOR'}
- Warnings: {', '.join(analysis.trading_plan.key_warnings) or 'None'}

### 10. NEXT MOVE PREDICTION
- Most likely: {'Continuation to EQH' if analysis.trend_analysis.direction == 'bullish' else 'Continuation to EQL' if analysis.trend_analysis.direction == 'bearish' else 'Sideways consolidation'}

### 11. ACTIONABLE SUMMARY
{analysis.executive_summary}

---
*Note: This is a fallback analysis. Full LLM interpretation unavailable.*"""


# Singleton instance
smc_llm_helper = SMCAnalysisLLMHelper()