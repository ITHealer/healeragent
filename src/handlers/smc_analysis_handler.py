from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.utils.logger.custom_logging import LoggerMixin
from src.schemas.smc_models import (
    SMCAnalyzeRequest,
    SMCAnalysisResult,
    TrendAnalysisResult,
    StructureAnalysisResult,
    OrderBlockAnalysisResult,
    LiquidityAnalysisResult,
    FVGAnalysisResult,
    PremiumDiscountResult,
    TradingPlanResult,
    EntryZoneResult,
    TargetLevelResult,
)


class SMCAnalysisHandler(LoggerMixin):
    """
    Handler for processing SMC indicator data and generating analysis.
    
    Performs deterministic analysis based on SMC rules:
    1. Trend detection from BOS/CHoCH events
    2. Order block validity and ranking
    3. Liquidity zone mapping
    4. FVG identification
    5. Premium/Discount calculation
    6. Trading plan generation
    """
    
    # Timeframe classification
    SHORT_TERM_INTERVALS = {"1m", "3m", "5m", "15m"}
    LONG_TERM_INTERVALS = {"30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
    
    def __init__(self):
        super().__init__()
        self.logger.info("[SMC] SMCAnalysisHandler initialized")
    
    def _get_timeframe_context(self, interval: str, mode: str) -> dict:
        """
        Determine trading timeframe context based on interval and mode.
        
        Rules:
        - interval <= 15m = short-term
        - interval > 15m = long-term
        - mode = "internal" = short-term structure
        - mode = "swing" = long-term structure
        
        Returns dict with timeframe classification and trading implications.
        """
        # Determine by interval
        interval_is_short = interval.lower() in self.SHORT_TERM_INTERVALS
        
        # Determine by mode
        mode_is_short = mode.lower() == "internal"
        
        # Combined assessment
        if interval_is_short and mode_is_short:
            timeframe_type = "scalping"
            hold_duration = "minutes to hours"
            sl_multiplier = 0.005  # Tighter SL for scalping (0.5%)
            tp_multiplier = 0.01   # Smaller TP (1%)
            risk_profile = "high"
            structure_focus = "internal structure, minor swing points"
        elif interval_is_short and not mode_is_short:
            timeframe_type = "intraday"
            hold_duration = "hours"
            sl_multiplier = 0.01   # 1% SL
            tp_multiplier = 0.02   # 2% TP
            risk_profile = "medium-high"
            structure_focus = "swing structure on lower timeframe"
        elif not interval_is_short and mode_is_short:
            timeframe_type = "day_trading"
            hold_duration = "hours to 1-2 days"
            sl_multiplier = 0.015  # 1.5% SL
            tp_multiplier = 0.03   # 3% TP
            risk_profile = "medium"
            structure_focus = "internal structure on higher timeframe"
        else:  # Long interval + swing mode
            timeframe_type = "swing_trading"
            hold_duration = "days to weeks"
            sl_multiplier = 0.03   # 3% SL
            tp_multiplier = 0.05   # 5% TP
            risk_profile = "lower"
            structure_focus = "major swing structure, key levels"
        
        return {
            "timeframe_type": timeframe_type,
            "is_short_term": interval_is_short or mode_is_short,
            "hold_duration": hold_duration,
            "sl_multiplier": sl_multiplier,
            "tp_multiplier": tp_multiplier,
            "risk_profile": risk_profile,
            "structure_focus": structure_focus,
            "interval": interval,
            "mode": mode
        }
    
    def analyze(self, request: SMCAnalyzeRequest) -> SMCAnalysisResult:
        """
        Main analysis method - processes SMC data and returns structured analysis.
        
        Args:
            request: SMCAnalyzeRequest containing all indicator data
            
        Returns:
            SMCAnalysisResult with complete analysis
        """
        self.logger.info(f"[SMC] Starting analysis for {request.symbol} @ {request.interval} (mode: {request.mode})")
        
        current_price = request.currentPrice
        smc_data = request.smcData
        metadata = request.metadata
        
        # Get timeframe context for appropriate SL/TP and analysis style
        timeframe_context = self._get_timeframe_context(request.interval, request.mode)
        self.logger.info(f"[SMC] Timeframe type: {timeframe_context['timeframe_type']}, Hold: {timeframe_context['hold_duration']}")
        
        # Perform individual analyses
        trend_analysis = self._analyze_trend(
            structure_events=smc_data.structureEvents,
            metadata=metadata
        )
        
        structure_analysis = self._analyze_structure(
            structure_events=smc_data.structureEvents,
            swing_points=smc_data.swingsSwing
        )
        
        ob_analysis = self._analyze_order_blocks(
            order_blocks=smc_data.orderBlocks,
            current_price=current_price,
            trend=trend_analysis.direction
        )
        
        liquidity_analysis = self._analyze_liquidity(
            liquidity_zones=smc_data.liquidityZones,
            current_price=current_price,
            trend=trend_analysis.direction
        )
        
        fvg_analysis = self._analyze_fvg(
            fvgs=smc_data.fairValueGaps,
            current_price=current_price
        )
        
        pd_analysis = self._analyze_premium_discount(
            premium_discount=smc_data.premiumDiscount,
            current_price=current_price,
            trend=trend_analysis.direction
        )
        
        # Generate trading plan with timeframe context
        trading_plan = self._generate_trading_plan(
            trend=trend_analysis,
            ob_analysis=ob_analysis,
            liquidity=liquidity_analysis,
            fvg=fvg_analysis,
            pd_analysis=pd_analysis,
            current_price=current_price,
            timeframe_context=timeframe_context
        )
        
        # Calculate data quality score
        data_quality = self._calculate_data_quality(metadata)
        
        # Generate executive summary with timeframe context
        executive_summary = self._generate_executive_summary(
            symbol=request.symbol,
            interval=request.interval,
            mode=request.mode,
            trend=trend_analysis,
            trading_plan=trading_plan,
            pd_analysis=pd_analysis,
            current_price=current_price,
            timeframe_context=timeframe_context
        )
        
        # Determine overall confidence
        analysis_confidence = self._determine_confidence(
            trend=trend_analysis,
            ob_analysis=ob_analysis,
            data_quality=data_quality
        )
        
        result = SMCAnalysisResult(
            symbol=request.symbol,
            interval=request.interval,
            mode=request.mode,
            timestamp=datetime.now().isoformat(),
            current_price=current_price,
            timeframe_type=timeframe_context['timeframe_type'],
            is_short_term=timeframe_context['is_short_term'],
            trend_analysis=trend_analysis,
            structure_analysis=structure_analysis,
            order_block_analysis=ob_analysis,
            liquidity_analysis=liquidity_analysis,
            fvg_analysis=fvg_analysis,
            premium_discount_analysis=pd_analysis,
            trading_plan=trading_plan,
            executive_summary=executive_summary,
            analysis_confidence=analysis_confidence,
            data_quality_score=data_quality
        )
        
        self.logger.info(f"[SMC] Analysis complete: {trading_plan.recommended_action} ({timeframe_context['timeframe_type']}) with {analysis_confidence} confidence")
        
        return result
    
    def _analyze_trend(self, structure_events: List, metadata: Any) -> TrendAnalysisResult:
        """Analyze market trend from structure events."""
        
        if not structure_events:
            return TrendAnalysisResult(
                direction="neutral",
                strength="weak",
                last_structure_event="BOS",
                confirmation_level="low",
                reasoning="No structure events available for trend analysis"
            )
        
        # Sort events by time (most recent first)
        sorted_events = sorted(structure_events, key=lambda x: x.time, reverse=True)
        last_event = sorted_events[0]
        
        # Count bullish vs bearish events
        bullish_count = sum(1 for e in structure_events if e.direction == "bullish")
        bearish_count = sum(1 for e in structure_events if e.direction == "bearish")
        
        # Count BOS vs CHoCH
        bos_count = sum(1 for e in structure_events if e.type == "BOS")
        choch_count = sum(1 for e in structure_events if e.type == "CHoCH")
        
        # Determine trend
        direction = last_event.direction
        
        if last_event.type == "BOS":
            confirmation = "high" if bos_count > choch_count else "medium"
        else:
            confirmation = "medium"
        
        # Determine strength
        if bullish_count > bearish_count + 1:
            strength = "strong" if direction == "bullish" else "weak"
        elif bearish_count > bullish_count + 1:
            strength = "strong" if direction == "bearish" else "weak"
        else:
            strength = "moderate"
        
        reasoning = (
            f"Last structure event: {last_event.type} {last_event.direction} at {last_event.level:.2f}. "
            f"Structure count: {bos_count} BOS, {choch_count} CHoCH. "
            f"Direction count: {bullish_count} bullish, {bearish_count} bearish events."
        )
        
        return TrendAnalysisResult(
            direction=direction,
            strength=strength,
            last_structure_event=last_event.type,
            confirmation_level=confirmation,
            reasoning=reasoning
        )
    
    def _analyze_structure(self, structure_events: List, swing_points: List) -> StructureAnalysisResult:
        """Analyze market structure including swing points."""
        
        swing_highs = [p.price for p in swing_points if p.type == "high"]
        swing_lows = [p.price for p in swing_points if p.type == "low"]
        
        bos_events = [e for e in structure_events if e.type == "BOS"]
        choch_events = [e for e in structure_events if e.type == "CHoCH"]
        
        last_bos_direction = None
        last_choch_direction = None
        
        if bos_events:
            sorted_bos = sorted(bos_events, key=lambda x: x.time, reverse=True)
            last_bos_direction = sorted_bos[0].direction
        if choch_events:
            sorted_choch = sorted(choch_events, key=lambda x: x.time, reverse=True)
            last_choch_direction = sorted_choch[0].direction
        
        if last_bos_direction and not last_choch_direction:
            structure_bias = last_bos_direction
        elif last_choch_direction:
            structure_bias = last_choch_direction
        else:
            structure_bias = "neutral"
        
        reasoning = ""
        if swing_highs:
            reasoning += f"Recent swing highs: {', '.join([f'{h:.2f}' for h in swing_highs[:3]])}. "
        if swing_lows:
            reasoning += f"Recent swing lows: {', '.join([f'{l:.2f}' for l in swing_lows[:3]])}."
        
        return StructureAnalysisResult(
            swing_highs=swing_highs[:5],
            swing_lows=swing_lows[:5],
            bos_events=len(bos_events),
            choch_events=len(choch_events),
            last_bos_direction=last_bos_direction,
            last_choch_direction=last_choch_direction,
            structure_bias=structure_bias,
            reasoning=reasoning or "Structure analysis based on swing points"
        )
    
    def _analyze_order_blocks(
        self,
        order_blocks: List,
        current_price: float,
        trend: str
    ) -> OrderBlockAnalysisResult:
        """Analyze order blocks and rank by relevance."""
        
        active_bullish = []
        active_bearish = []
        mitigated_count = 0
        
        for ob in order_blocks:
            midpoint = (ob.high + ob.low) / 2
            zone_size = abs(ob.high - ob.low)
            
            ob_data = {
                "high": ob.high,
                "low": ob.low,
                "midpoint": midpoint,
                "zone_size": zone_size,
                "mitigated": ob.mitigated,
                "distance_from_price": abs(current_price - midpoint),
                "distance_pct": abs(current_price - midpoint) / current_price * 100
            }
            
            if ob.mitigated:
                mitigated_count += 1
            elif ob.type == "bullish":
                active_bullish.append(ob_data)
            elif ob.type == "bearish":
                active_bearish.append(ob_data)
        
        active_bullish.sort(key=lambda x: x["distance_from_price"])
        active_bearish.sort(key=lambda x: x["distance_from_price"])
        
        strongest_demand = active_bullish[0] if active_bullish else None
        strongest_supply = active_bearish[0] if active_bearish else None
        
        reasoning = f"Found {len(active_bullish)} active bullish OBs (demand zones), {len(active_bearish)} active bearish OBs (supply zones)."
        if strongest_demand:
            reasoning += f" Strongest demand: {strongest_demand['low']:.2f}-{strongest_demand['high']:.2f}."
        if strongest_supply:
            reasoning += f" Strongest supply: {strongest_supply['low']:.2f}-{strongest_supply['high']:.2f}."
        
        return OrderBlockAnalysisResult(
            active_bullish_obs=active_bullish[:5],
            active_bearish_obs=active_bearish[:5],
            total_active=len(active_bullish) + len(active_bearish),
            total_mitigated=mitigated_count,
            strongest_demand_zone=strongest_demand,
            strongest_supply_zone=strongest_supply,
            reasoning=reasoning
        )
    
    def _analyze_liquidity(
        self,
        liquidity_zones: List,
        current_price: float,
        trend: str
    ) -> LiquidityAnalysisResult:
        """Analyze liquidity zones (EQH/EQL)."""
        
        buy_side = []
        sell_side = []
        
        for zone in liquidity_zones:
            zone_data = {
                "level": zone.level,
                "distance_from_price": abs(current_price - zone.level),
                "distance_pct": abs(current_price - zone.level) / current_price * 100,
                "above_price": zone.level > current_price
            }
            
            if zone.type == "EQH":
                buy_side.append(zone_data)
            else:
                sell_side.append(zone_data)
        
        buy_side.sort(key=lambda x: x["distance_from_price"])
        sell_side.sort(key=lambda x: x["distance_from_price"])
        
        nearest_eqh = buy_side[0]["level"] if buy_side else None
        nearest_eql = sell_side[0]["level"] if sell_side else None
        
        sweep_targets = []
        if trend == "bullish" and nearest_eqh:
            sweep_targets.append(nearest_eqh)
        if trend == "bearish" and nearest_eql:
            sweep_targets.append(nearest_eql)
        
        reasoning = ""
        if nearest_eqh:
            reasoning += f"Buy side liquidity (EQH) at {nearest_eqh:.2f}. "
        if nearest_eql:
            reasoning += f"Sell side liquidity (EQL) at {nearest_eql:.2f}. "
        if sweep_targets:
            reasoning += f"Potential sweep targets: {', '.join([f'{t:.2f}' for t in sweep_targets])}."
        
        return LiquidityAnalysisResult(
            buy_side_liquidity=buy_side[:3],
            sell_side_liquidity=sell_side[:3],
            nearest_eqh=nearest_eqh,
            nearest_eql=nearest_eql,
            potential_sweep_targets=sweep_targets,
            reasoning=reasoning or "No significant liquidity zones identified"
        )
    
    def _analyze_fvg(self, fvgs: List, current_price: float) -> FVGAnalysisResult:
        """Analyze Fair Value Gaps."""
        
        active_bullish = []
        active_bearish = []
        mitigated_count = 0
        
        for fvg in fvgs:
            if fvg.mitigated:
                mitigated_count += 1
                continue
            
            midpoint = (fvg.topPrice + fvg.bottomPrice) / 2
            gap_size = abs(fvg.topPrice - fvg.bottomPrice)
            
            fvg_data = {
                "top_price": fvg.topPrice,
                "bottom_price": fvg.bottomPrice,
                "midpoint": midpoint,
                "gap_size": gap_size,
                "distance_from_price": min(
                    abs(current_price - fvg.topPrice),
                    abs(current_price - fvg.bottomPrice)
                ),
                "distance_pct": min(
                    abs(current_price - fvg.topPrice),
                    abs(current_price - fvg.bottomPrice)
                ) / current_price * 100
            }
            
            if fvg.type == "bullish":
                active_bullish.append(fvg_data)
            else:
                active_bearish.append(fvg_data)
        
        active_bullish.sort(key=lambda x: x["distance_from_price"])
        active_bearish.sort(key=lambda x: x["distance_from_price"])
        
        all_active = [
            {**fvg, "type": "bullish"} for fvg in active_bullish
        ] + [
            {**fvg, "type": "bearish"} for fvg in active_bearish
        ]
        all_active.sort(key=lambda x: x["distance_from_price"])
        nearest_fvg = all_active[0] if all_active else None
        
        reasoning = f"Active bullish FVGs: {len(active_bullish)}. Active bearish FVGs: {len(active_bearish)}."
        if nearest_fvg:
            reasoning += f" Nearest unfilled: {nearest_fvg['type']} at {nearest_fvg['bottom_price']:.2f}-{nearest_fvg['top_price']:.2f}."
        
        return FVGAnalysisResult(
            active_bullish_fvgs=active_bullish[:3],
            active_bearish_fvgs=active_bearish[:3],
            total_active=len(active_bullish) + len(active_bearish),
            total_mitigated=mitigated_count,
            nearest_unfilled_fvg=nearest_fvg,
            reasoning=reasoning
        )
    
    def _analyze_premium_discount(
        self,
        premium_discount: List,
        current_price: float,
        trend: str
    ) -> PremiumDiscountResult:
        """Analyze Premium/Discount zones."""
        
        if not premium_discount:
            return PremiumDiscountResult(
                current_zone="equilibrium",
                equilibrium_price=current_price,
                premium_threshold=current_price,
                discount_threshold=current_price,
                distance_to_equilibrium_pct=0.0,
                trade_recommendation="No premium/discount data available",
                reasoning="Premium/Discount zones not calculated"
            )
        
        latest_pd = premium_discount[0]
        equilibrium = latest_pd.midpoint
        
        range_size = equilibrium * 0.02
        premium_threshold = equilibrium + (range_size * 0.5)
        discount_threshold = equilibrium - (range_size * 0.5)
        
        if current_price > premium_threshold:
            current_zone = "premium"
        elif current_price < discount_threshold:
            current_zone = "discount"
        else:
            current_zone = "equilibrium"
        
        distance_pct = (current_price - equilibrium) / equilibrium * 100
        
        if current_zone == "premium":
            if trend == "bullish":
                recommendation = "Price in premium zone. Consider waiting for pullback to discount for LONG entries"
            else:
                recommendation = "Price in premium zone. Favorable for SHORT entries"
        elif current_zone == "discount":
            if trend == "bullish":
                recommendation = "Price in discount zone. Favorable for LONG entries"
            else:
                recommendation = "Price in discount zone. Consider waiting for rally to premium for SHORT entries"
        else:
            recommendation = "Price at equilibrium. Wait for price to reach extreme zones for optimal entries"
        
        reasoning = (
            f"Current price {current_price:.2f} is in {current_zone} zone. "
            f"Equilibrium at {equilibrium:.2f}. "
            f"Distance: {abs(distance_pct):.2f}% {'above' if distance_pct > 0 else 'below'} equilibrium."
        )
        
        return PremiumDiscountResult(
            current_zone=current_zone,
            equilibrium_price=equilibrium,
            premium_threshold=premium_threshold,
            discount_threshold=discount_threshold,
            distance_to_equilibrium_pct=distance_pct,
            trade_recommendation=recommendation,
            reasoning=reasoning
        )
    
    def _generate_trading_plan(
        self,
        trend: TrendAnalysisResult,
        ob_analysis: OrderBlockAnalysisResult,
        liquidity: LiquidityAnalysisResult,
        fvg: FVGAnalysisResult,
        pd_analysis: PremiumDiscountResult,
        current_price: float,
        timeframe_context: dict = None
    ) -> TradingPlanResult:
        """
        Generate complete trading plan with timeframe-aware SL/TP.
        
        Timeframe adjustments:
        - Short-term (scalping/intraday): tighter SL/TP, faster invalidation
        - Long-term (swing): wider SL/TP, more room for price movement
        """
        
        # Default timeframe context if not provided
        if timeframe_context is None:
            timeframe_context = {
                "timeframe_type": "swing_trading",
                "is_short_term": False,
                "sl_multiplier": 0.03,
                "tp_multiplier": 0.05,
                "hold_duration": "days to weeks"
            }
        
        sl_multiplier = timeframe_context.get("sl_multiplier", 0.03)
        tp_multiplier = timeframe_context.get("tp_multiplier", 0.05)
        is_short_term = timeframe_context.get("is_short_term", False)
        timeframe_type = timeframe_context.get("timeframe_type", "swing_trading")
        
        # Determine action - adjust for timeframe
        if trend.direction == "bullish":
            if pd_analysis.current_zone == "discount":
                action = "LONG"
            elif pd_analysis.current_zone == "premium":
                # Short-term có thể scalp ngay, long-term nên đợi
                action = "SCALP_LONG" if is_short_term else "WAIT_FOR_PULLBACK"
            else:
                action = "LONG_ON_CONFIRMATION"
        elif trend.direction == "bearish":
            if pd_analysis.current_zone == "premium":
                action = "SHORT"
            elif pd_analysis.current_zone == "discount":
                action = "SCALP_SHORT" if is_short_term else "WAIT_FOR_RALLY"
            else:
                action = "SHORT_ON_CONFIRMATION"
        else:
            action = "WAIT"
        
        # Build entry zones
        entry_zones = []
        
        if trend.direction == "bullish" and ob_analysis.strongest_demand_zone:
            demand = ob_analysis.strongest_demand_zone
            entry_zones.append(EntryZoneResult(
                zone_type="order_block",
                price_range={"low": demand["low"], "high": demand["high"]},
                confidence="high",
                reasons=["Unmitigated bullish order block", "Aligned with bullish trend"]
            ))
        
        if trend.direction == "bearish" and ob_analysis.strongest_supply_zone:
            supply = ob_analysis.strongest_supply_zone
            entry_zones.append(EntryZoneResult(
                zone_type="order_block",
                price_range={"low": supply["low"], "high": supply["high"]},
                confidence="high",
                reasons=["Unmitigated bearish order block", "Aligned with bearish trend"]
            ))
        
        if fvg.nearest_unfilled_fvg:
            nearest = fvg.nearest_unfilled_fvg
            if (trend.direction == "bullish" and nearest.get("type") == "bullish") or \
               (trend.direction == "bearish" and nearest.get("type") == "bearish"):
                entry_zones.append(EntryZoneResult(
                    zone_type="fvg",
                    price_range={"low": nearest["bottom_price"], "high": nearest["top_price"]},
                    confidence="medium",
                    reasons=[f"Unfilled {nearest.get('type')} FVG", "Gap likely to be filled"]
                ))
        
        # Calculate stop loss with timeframe-aware multiplier
        stop_loss = 0.0
        sl_reasoning = ""
        sl_buffer = 1 - sl_multiplier if trend.direction == "bullish" else 1 + sl_multiplier
        
        if trend.direction == "bullish":
            if ob_analysis.strongest_demand_zone:
                # Use timeframe-appropriate buffer below demand zone
                stop_loss = ob_analysis.strongest_demand_zone["low"] * (1 - sl_multiplier * 0.5)
                sl_reasoning = f"Stop loss {sl_multiplier*50:.1f}% below demand zone ({timeframe_type})"
            elif liquidity.nearest_eql:
                stop_loss = liquidity.nearest_eql * (1 - sl_multiplier * 0.5)
                sl_reasoning = f"Stop loss below EQL ({timeframe_type})"
            else:
                stop_loss = current_price * (1 - sl_multiplier)
                sl_reasoning = f"Default {sl_multiplier*100:.1f}% stop loss ({timeframe_type})"
        elif trend.direction == "bearish":
            if ob_analysis.strongest_supply_zone:
                stop_loss = ob_analysis.strongest_supply_zone["high"] * (1 + sl_multiplier * 0.5)
                sl_reasoning = f"Stop loss {sl_multiplier*50:.1f}% above supply zone ({timeframe_type})"
            elif liquidity.nearest_eqh:
                stop_loss = liquidity.nearest_eqh * (1 + sl_multiplier * 0.5)
                sl_reasoning = f"Stop loss above EQH ({timeframe_type})"
            else:
                stop_loss = current_price * (1 + sl_multiplier)
                sl_reasoning = f"Default {sl_multiplier*100:.1f}% stop loss ({timeframe_type})"
        else:
            stop_loss = current_price * (1 - sl_multiplier)
            sl_reasoning = f"Neutral trend - conservative {sl_multiplier*100:.1f}% stop loss"
        
        # Build targets with timeframe-aware TP
        targets = []
        
        if trend.direction == "bullish" and liquidity.nearest_eqh:
            targets.append(TargetLevelResult(
                price=liquidity.nearest_eqh,
                target_type="liquidity_sweep",
                reasoning=f"EQH liquidity target ({timeframe_type})"
            ))
        elif trend.direction == "bearish" and liquidity.nearest_eql:
            targets.append(TargetLevelResult(
                price=liquidity.nearest_eql,
                target_type="liquidity_sweep",
                reasoning=f"EQL liquidity target ({timeframe_type})"
            ))
        
        # Add percentage-based targets using timeframe TP multiplier
        if not targets:
            target_pct = 1 + tp_multiplier if trend.direction == "bullish" else 1 - tp_multiplier
            targets.append(TargetLevelResult(
                price=current_price * target_pct,
                target_type="percentage",
                reasoning=f"Default {tp_multiplier*100:.1f}% target ({timeframe_type})"
            ))
        
        # For short-term, add intermediate target
        if is_short_term and len(targets) == 1:
            intermediate_pct = 1 + (tp_multiplier * 0.5) if trend.direction == "bullish" else 1 - (tp_multiplier * 0.5)
            targets.insert(0, TargetLevelResult(
                price=current_price * intermediate_pct,
                target_type="intermediate",
                reasoning=f"Quick {tp_multiplier*50:.1f}% scalp target"
            ))
        
        # Calculate R:R
        rr_ratio = 0.0
        if entry_zones and targets and stop_loss > 0:
            avg_entry = sum(
                (ez.price_range.get("low", 0) + ez.price_range.get("high", 0)) / 2 
                for ez in entry_zones
            ) / len(entry_zones)
            
            first_target = targets[0].price
            risk = abs(avg_entry - stop_loss)
            reward = abs(first_target - avg_entry)
            rr_ratio = reward / risk if risk > 0 else 0
        
        # Determine signal strength
        score = 0
        if trend.confirmation_level == "high":
            score += 3
        elif trend.confirmation_level == "medium":
            score += 2
        else:
            score += 1
        
        high_conf_entries = sum(1 for ez in entry_zones if ez.confidence == "high")
        score += high_conf_entries * 2
        
        if rr_ratio >= 3:
            score += 3
        elif rr_ratio >= 2:
            score += 2
        elif rr_ratio >= 1:
            score += 1
        
        if score >= 7:
            signal_strength = "strong"
        elif score >= 4:
            signal_strength = "moderate"
        else:
            signal_strength = "weak"
        
        # Warnings with timeframe context
        warnings = []
        
        # Zone warnings
        if trend.direction == "bullish" and pd_analysis.current_zone == "premium":
            if is_short_term:
                warnings.append("Price in premium zone - scalp only with tight SL")
            else:
                warnings.append("Price in premium zone - wait for discount entry for better R:R")
        elif trend.direction == "bearish" and pd_analysis.current_zone == "discount":
            if is_short_term:
                warnings.append("Price in discount zone - scalp only with tight SL")
            else:
                warnings.append("Price in discount zone - wait for premium entry for better R:R")
        
        # R:R warnings - adjust threshold by timeframe
        min_rr = 1.0 if is_short_term else 1.5
        if rr_ratio and rr_ratio < min_rr:
            warnings.append(f"R:R {rr_ratio:.2f} below {min_rr}:1 minimum for {timeframe_type}")
        
        # Structure warnings
        if trend.last_structure_event == "CHoCH":
            warnings.append("Recent CHoCH indicates potential trend reversal - use caution")
        
        # Timeframe-specific warnings
        if is_short_term:
            warnings.append(f"Short-term {timeframe_type}: Use tight risk management, quick entries/exits")
        
        invalidation = None
        if stop_loss and stop_loss > 0:
            invalidation = round(stop_loss * 0.99 if trend.direction == "bullish" else stop_loss * 1.01, 2)
        
        return TradingPlanResult(
            bias=trend.direction,
            signal_strength=signal_strength,
            recommended_action=action,
            entry_zones=entry_zones[:3] if entry_zones else [],
            stop_loss=round(stop_loss, 2) if stop_loss and stop_loss > 0 else None,
            stop_loss_reasoning=sl_reasoning,
            targets=targets[:3] if targets else [],
            risk_reward_ratio=round(rr_ratio, 2) if rr_ratio and rr_ratio > 0 else None,
            invalidation_level=invalidation,
            key_warnings=warnings,
            timeframe_type=timeframe_type,
            hold_duration=timeframe_context.get("hold_duration", "days to weeks")
        )
    
    def _calculate_data_quality(self, metadata: Any) -> float:
        """Calculate data quality score (0-100). Handles None values safely."""
        
        score = 0.0
        
        # Safe get with default 0
        total_structure = metadata.totalStructureEvents or 0
        active_obs = metadata.activeOrderBlocks or 0
        total_liq = metadata.totalLiquidityZones or 0
        active_fvg = metadata.activeFvgCount or 0
        current_trend = metadata.currentTrend or "neutral"
        
        if total_structure >= 5:
            score += 30
        elif total_structure >= 3:
            score += 20
        elif total_structure >= 1:
            score += 10
        
        if active_obs >= 3:
            score += 25
        elif active_obs >= 2:
            score += 15
        elif active_obs >= 1:
            score += 10
        
        if total_liq >= 4:
            score += 20
        elif total_liq >= 2:
            score += 15
        elif total_liq >= 1:
            score += 10
        
        if active_fvg >= 3:
            score += 15
        elif active_fvg >= 2:
            score += 10
        elif active_fvg >= 1:
            score += 5
        
        if current_trend and current_trend != "neutral":
            score += 10
        
        return min(score, 100.0)
    
    def _generate_executive_summary(
        self,
        symbol: str,
        interval: str,
        mode: str,
        trend: TrendAnalysisResult,
        trading_plan: TradingPlanResult,
        pd_analysis: PremiumDiscountResult,
        current_price: float,
        timeframe_context: dict = None
    ) -> str:
        """Generate brief executive summary with timeframe context."""
        
        # Get timeframe info
        if timeframe_context:
            timeframe_type = timeframe_context.get("timeframe_type", "swing_trading")
            hold_duration = timeframe_context.get("hold_duration", "days to weeks")
        else:
            timeframe_type = "swing_trading"
            hold_duration = "days to weeks"
        
        # Handle None values for R:R ratio
        rr_display = f"{trading_plan.risk_reward_ratio:.2f}" if trading_plan.risk_reward_ratio else "N/A"
        
        # Format timeframe type nicely
        timeframe_label = timeframe_type.replace("_", " ").title()
        
        summary = (
            f"{symbol} {interval} ({mode.upper()} mode - {timeframe_label}): "
            f"Trend is {trend.direction} ({trend.strength} strength) "
            f"based on {trend.last_structure_event}. "
            f"Price at {current_price:.2f} in {pd_analysis.current_zone} zone. "
            f"Recommendation: {trading_plan.recommended_action} with R:R {rr_display}. "
            f"Expected hold: {hold_duration}."
        )
        
        if trading_plan.entry_zones:
            first = trading_plan.entry_zones[0]
            low = first.price_range.get('low', 0)
            high = first.price_range.get('high', 0)
            if low and high:
                summary += f" Primary entry: {low:.2f}-{high:.2f}."
        
        if trading_plan.targets:
            summary += f" Target: {trading_plan.targets[0].price:.2f}."
        
        # Handle None stop_loss
        if trading_plan.stop_loss:
            summary += f" Stop loss: {trading_plan.stop_loss:.2f}."
        
        return summary
    
    def _determine_confidence(
        self,
        trend: TrendAnalysisResult,
        ob_analysis: OrderBlockAnalysisResult,
        data_quality: float
    ) -> str:
        """Determine overall analysis confidence."""
        
        if data_quality >= 70 and trend.confirmation_level == "high":
            return "high"
        elif data_quality >= 50 and trend.confirmation_level != "low":
            return "medium"
        else:
            return "low"