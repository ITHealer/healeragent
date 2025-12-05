import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class FinancialDataFormatter:
    """
    Format financial data cho LLM consumption
    
    Design principles:
    - Clear section headers
    - Structured information hierarchy
    - Include units and context
    - Handle missing data gracefully
    """
    
    @staticmethod
    def format_price_data(price_data: Dict[str, Any]) -> str:
        """
        Format comprehensive stock price data
        
        FIXED: Handle both atomic tool format and legacy format
        
        Args:
            price_data: Dict containing all price-related data
            
        Returns:
            Formatted text for LLM
        """
        if not price_data:
            return "No price data available"
        
        sections = []
        
        # Extract current price (try multiple keys)
        current_price = (
            price_data.get('current_price') or 
            price_data.get('price') or
            price_data.get('last_price')
        )
        
        # Extract symbol and name
        symbol = price_data.get('symbol', 'N/A')
        name = price_data.get('name', '')
        
        # Extract change data (multiple formats)
        change = price_data.get('change')
        change_percent = price_data.get('change_percent')
        
        # Extract additional metrics from atomic tools
        volume = price_data.get('volume')
        day_high = price_data.get('day_high')
        day_low = price_data.get('day_low')
        previous_close = price_data.get('previous_close')
        market_cap = price_data.get('market_cap')
        
        # Legacy format
        performance = price_data.get('performance', {})
        analyst_targets = price_data.get('analyst_targets', {})
        
        # Header with Symbol & Name
        header = f"📊 STOCK PRICE - {symbol}"
        if name:
            header += f" ({name})"
        sections.append(header)
        sections.append("═" * 60)
        
        # Current Price & Change
        if current_price:
            price_section = []
            
            price_section.append(f"**Current Price:** ${current_price:,.2f}")
            
            if change is not None and change_percent is not None:
                emoji = "📈" if change >= 0 else "📉"
                price_section.append(
                    f"**Change:** {emoji} ${change:+,.2f} ({change_percent:+.2f}%)"
                )
            
            if previous_close:
                price_section.append(f"**Previous Close:** ${previous_close:,.2f}")
            
            sections.append("\n".join(price_section))
            sections.append("")
        
        # Day Range & Volume
        if day_high or day_low or volume:
            range_section = []
            
            if day_high and day_low:
                range_section.append(f"**Day Range:** ${day_low:,.2f} - ${day_high:,.2f}")
            
            if volume:
                if volume >= 1_000_000_000:
                    vol_str = f"{volume / 1_000_000_000:.2f}B"
                elif volume >= 1_000_000:
                    vol_str = f"{volume / 1_000_000:.2f}M"
                else:
                    vol_str = f"{volume:,}"
                range_section.append(f"**Volume:** {vol_str}")
            
            if range_section:
                sections.append("\n".join(range_section))
                sections.append("")
        
        # Market Cap
        if market_cap:
            if market_cap >= 1_000_000_000_000:
                cap_str = f"${market_cap / 1_000_000_000_000:.2f}T"
            elif market_cap >= 1_000_000_000:
                cap_str = f"${market_cap / 1_000_000_000:.2f}B"
            elif market_cap >= 1_000_000:
                cap_str = f"${market_cap / 1_000_000:.2f}M"
            else:
                cap_str = f"${market_cap:,.0f}"
            
            sections.append(f"**Market Cap:** {cap_str}")
            sections.append("")
        
        # Legacy Performance Data
        if performance:
            perf_section = ["📈 **PERFORMANCE**"]
            
            month_change = performance.get('month_change')
            three_month_change = performance.get('three_month_change')
            
            if month_change is not None:
                emoji = "📈" if month_change >= 0 else "📉"
                perf_section.append(f"1-Month: {emoji} {month_change:+.2f}%")
            
            if three_month_change is not None:
                emoji = "📈" if three_month_change >= 0 else "📉"
                perf_section.append(f"3-Month: {emoji} {three_month_change:+.2f}%")
            
            if len(perf_section) > 1:
                sections.append("\n".join(perf_section))
                sections.append("")
        
        # Analyst Targets
        if analyst_targets and any(v is not None for v in analyst_targets.values()):
            target_section = ["🎯 **ANALYST TARGETS**"]
            
            high = analyst_targets.get('priceTargetHigh')
            avg = analyst_targets.get('priceTargetAverage')
            low = analyst_targets.get('priceTargetLow')
            count = analyst_targets.get('numberOfAnalysts')
            
            if high and current_price:
                upside = ((high - current_price) / current_price * 100)
                target_section.append(f"High: ${high:,.2f} ({upside:+.1f}% upside)")
            
            if avg and current_price:
                upside = ((avg - current_price) / current_price * 100)
                target_section.append(f"Average: ${avg:,.2f} ({upside:+.1f}%)")
            
            if low and current_price:
                downside = ((low - current_price) / current_price * 100)
                target_section.append(f"Low: ${low:,.2f} ({downside:+.1f}%)")
            
            if count:
                target_section.append(f"Analysts: {count}")
            
            if len(target_section) > 1:
                sections.append("\n".join(target_section))
        
        if not sections:
            return "No price data available to format"
        
        return "\n".join(sections)
    
    @staticmethod
    def format_technical_indicators_data(tech_data: Dict[str, Any]) -> str:
        """Format technical indicators for LLM"""
        if not tech_data:
            return "No technical data available"
        
        sections = []
        
        symbol = tech_data.get('symbol', 'N/A')
        sections.append(f"📊 TECHNICAL INDICATORS - {symbol}")
        sections.append("═" * 60)
        
        # Current Price
        current_price = tech_data.get('current_price')
        if current_price:
            sections.append(f"\n**Current Price:** ${current_price:,.2f}")
        
        # RSI
        rsi = tech_data.get('rsi_14')
        if rsi is not None:
            emoji = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
            sections.append(f"\n**RSI (14):** {emoji} {rsi:.2f}")
            rsi_signal = tech_data.get('rsi_signal', 'N/A')
            sections.append(f"Signal: {rsi_signal}")
        
        # MACD
        macd = tech_data.get('macd_line')
        if macd is not None:
            sections.append(f"\n**MACD Line:** {macd:.4f}")
            macd_signal = tech_data.get('macd_signal')
            if macd_signal is not None:
                sections.append(f"Signal Line: {macd_signal:.4f}")
            macd_hist = tech_data.get('macd_histogram')
            if macd_hist is not None:
                sections.append(f"Histogram: {macd_hist:.4f}")
            macd_signal_text = tech_data.get('macd_signal_text')
            if macd_signal_text:
                sections.append(f"Trend: {macd_signal_text}")
        
        # Moving Averages
        sma_20 = tech_data.get('sma_20')
        sma_50 = tech_data.get('sma_50')
        sma_200 = tech_data.get('sma_200')
        
        if any([sma_20, sma_50, sma_200]):
            sections.append("\n**Moving Averages:**")
            if sma_20:
                sections.append(f"SMA (20): ${sma_20:,.2f}")
            if sma_50:
                sections.append(f"SMA (50): ${sma_50:,.2f}")
            if sma_200:
                sections.append(f"SMA (200): ${sma_200:,.2f}")
        
        # EMA
        ema_12 = tech_data.get('ema_12')
        ema_26 = tech_data.get('ema_26')
        
        if ema_12 or ema_26:
            if ema_12:
                sections.append(f"EMA (12): ${ema_12:,.2f}")
            if ema_26:
                sections.append(f"EMA (26): ${ema_26:,.2f}")
        
        # Bollinger Bands
        bb_upper = tech_data.get('bb_upper')
        bb_middle = tech_data.get('bb_middle')
        bb_lower = tech_data.get('bb_lower')
        
        if bb_upper or bb_middle or bb_lower:
            sections.append("\n**Bollinger Bands:**")
            if bb_upper:
                sections.append(f"Upper: ${bb_upper:,.2f}")
            if bb_middle:
                sections.append(f"Middle: ${bb_middle:,.2f}")
            if bb_lower:
                sections.append(f"Lower: ${bb_lower:,.2f}")
        
        # ATR
        atr = tech_data.get('atr_14')
        if atr:
            sections.append(f"\n**ATR (14):** ${atr:.2f}")
        
        # Timestamp
        timestamp = tech_data.get('timestamp')
        if timestamp:
            sections.append(f"\nTimestamp: {timestamp}")
        
        return "\n".join(sections)
    
    @staticmethod
    def format_risk_assessment_data(risk_data: Dict[str, Any]) -> str:
        """Format risk assessment for LLM"""
        if not risk_data:
            return "No risk assessment data available"
        
        sections = []
        
        symbol = risk_data.get('symbol', 'N/A')
        sections.append(f"⚠️ RISK ASSESSMENT - {symbol}")
        sections.append("═" * 60)
        
        current_price = risk_data.get('current_price')
        if current_price:
            sections.append(f"\n**Current Price:** ${current_price:,.2f}")
        
        volatility = risk_data.get('volatility')
        if volatility is not None:
            sections.append(f"**Volatility:** {volatility:.2f}%")
        
        # Stop loss recommendations
        rec = risk_data.get('recommendation', {})
        if rec:
            sections.append("\n🛡️ **STOP LOSS RECOMMENDATIONS:**")
            
            conservative = rec.get('conservative_stop_loss')
            if conservative:
                sections.append(f"Conservative: ${conservative:,.2f}")
            
            moderate = rec.get('moderate_stop_loss')
            if moderate:
                sections.append(f"Moderate: ${moderate:,.2f}")
            
            aggressive = rec.get('aggressive_stop_loss')
            if aggressive:
                sections.append(f"Aggressive: ${aggressive:,.2f}")
        
        # Risk level
        risk_level = risk_data.get('risk_level')
        if risk_level:
            sections.append(f"\n**Risk Level:** {risk_level}")
        
        return "\n".join(sections)
    
    @staticmethod
    def format_news_data(news_data: Dict[str, Any]) -> str:
        """
        Format news data for LLM
        
        Args:
            news_data: Dict containing news articles
            
        Returns:
            Formatted text
        """
        if not news_data:
            return "No news data available"
        
        sections = []
        
        # Extract articles
        articles = news_data.get('articles', [])
        if not articles:
            articles = news_data.get('news', [])
        
        if not articles:
            return "No news articles available"
        
        symbol = news_data.get('symbol', 'N/A')
        sections.append(f"📰 RECENT NEWS - {symbol}")
        sections.append("═" * 60)
        
        for idx, article in enumerate(articles[:5], 1):
            sections.append(f"\n**Article {idx}:**")
            
            title = article.get('title', 'No title')
            sections.append(f"Title: {title}")
            
            published = article.get('publishedDate') or article.get('published_at')
            if published:
                sections.append(f"Published: {published}")
            
            summary = article.get('text') or article.get('summary')
            if summary:
                summary_text = summary[:200] + "..." if len(summary) > 200 else summary
                sections.append(f"Summary: {summary_text}")
            
            url = article.get('url')
            if url:
                sections.append(f"URL: {url}")
            
            sections.append("")
        
        return "\n".join(sections)
    
    @staticmethod
    def format_chart_data(chart_data: Dict[str, Any]) -> str:
        """
        Format chart/technical data for LLM
        
        Args:
            chart_data: Dict containing chart data
            
        Returns:
            Formatted text
        """
        return FinancialDataFormatter.format_technical_indicators_data(chart_data)
    

    @staticmethod
    def format_bollinger_patterns(bb_patterns: Dict[str, Any]) -> str:
        """
        Format Bollinger Bands patterns for LLM
        
        Args:
            bb_patterns: Dict containing w_bottom, m_top, squeeze_breakout
            
        Returns:
            Formatted text for LLM
        """
        if not bb_patterns:
            return "No Bollinger Bands patterns detected"
        
        sections = []
        sections.append("🎯 BOLLINGER BANDS PATTERNS")
        sections.append("═" * 60)
        
        # W-Bottom Pattern
        if w_bottom := bb_patterns.get("w_bottom"):
            if w_bottom.get("detected"):
                sections.append("\n✅ **W-BOTTOM PATTERN DETECTED**")
                sections.append(f"Signal: {w_bottom.get('signal', 'N/A')}")
                sections.append(f"Confidence: {w_bottom.get('confidence', 0)*100:.0f}%")
                sections.append(f"Description: {w_bottom.get('description', 'N/A')}")
                sections.append("")
                sections.append("📈 **Trading Implications:**")
                sections.append("- Strong bullish reversal signal")
                sections.append("- Entry: Current price or pullback to middle band")
                sections.append("- Target: Previous highs or upper band extension")
                sections.append("- Stop: Below recent swing low")
        
        # M-Top Pattern
        if m_top := bb_patterns.get("m_top"):
            if m_top.get("detected"):
                sections.append("\n✅ **M-TOP PATTERN DETECTED**")
                sections.append(f"Signal: {m_top.get('signal', 'N/A')}")
                sections.append(f"Confidence: {m_top.get('confidence', 0)*100:.0f}%")
                sections.append(f"Description: {m_top.get('description', 'N/A')}")
                sections.append("")
                sections.append("📉 **Trading Implications:**")
                sections.append("- Strong bearish reversal signal")
                sections.append("- Consider profit taking or short positions")
                sections.append("- Stop: Above recent swing high")
                sections.append("- Target: Lower band or previous support")
        
        # Squeeze Breakout
        if squeeze := bb_patterns.get("squeeze_breakout"):
            if squeeze.get("detected"):
                direction = squeeze.get("direction", "unknown")
                expansion = squeeze.get("bandwidth_expansion", 0)
                
                sections.append("\n✅ **BOLLINGER SQUEEZE BREAKOUT**")
                sections.append(f"Direction: {direction.upper()}")
                sections.append(f"Bandwidth Expansion: {expansion:.2f}x")
                sections.append(f"Description: {squeeze.get('description', 'N/A')}")
                sections.append("")
                sections.append("⚡ **Trading Implications:**")
                
                if direction == "bullish":
                    sections.append("- Bullish momentum confirmed")
                    sections.append("- Entry: Breakout above upper band")
                    sections.append("- Trend likely to continue")
                else:
                    sections.append("- Bearish momentum confirmed")
                    sections.append("- Consider exit or short positions")
                    sections.append("- Downtrend likely to continue")
        
        if len(sections) <= 2:
            return "No significant Bollinger Bands patterns detected"
        
        return "\n".join(sections)


    @staticmethod
    def format_volume_profile(data: Dict[str, Any]) -> str:
        """
        Format volume profile for LLM with comprehensive null checks
        
        FIXED: Safe handling of None/0 values
        
        Args:
            data: Volume profile data
            
        Returns:
            Formatted text for LLM
        """
        if not data:
            return "No volume profile data available"
        
        sections = []
        
        symbol = data.get('symbol', 'N/A')
        sections.append(f"📊 VOLUME PROFILE - {symbol}")
        sections.append("═" * 60)
        
        # ════════════════════════════════════════════════════════════
        # Extract values with safe defaults
        # ════════════════════════════════════════════════════════════
        poc = data.get('poc') or 0.0
        poc_volume = data.get('poc_volume') or 0
        va_high = data.get('value_area_high') or 0.0
        va_low = data.get('value_area_low') or 0.0
        va_volume_pct = data.get('value_area_volume_pct') or 70.0
        data_quality = data.get('data_quality', 'unknown')
        
        # ════════════════════════════════════════════════════════════
        # Check data quality first
        # ════════════════════════════════════════════════════════════
        if data_quality == "insufficient":
            sections.append("\n⚠️ **DATA QUALITY: INSUFFICIENT**")
            sections.append("Not enough volume data to calculate reliable profile.")
            sections.append("\n💡 **Recommendations:**")
            sections.append("- Increase lookback period (try 90-120 days)")
            sections.append("- Check if symbol has sufficient trading history")
            sections.append("- Verify market hours (some symbols have low liquidity)")
            return "\n".join(sections)
        
        if data_quality == "partial":
            sections.append("\n⚠️ **DATA QUALITY: PARTIAL**")
            sections.append("Some volume metrics could not be calculated reliably.")
        
        # ════════════════════════════════════════════════════════════
        # Point of Control (POC)
        # ════════════════════════════════════════════════════════════
        sections.append("\n**🎯 Point of Control (POC):**")
        
        if poc > 0:
            sections.append(f"Price: ${poc:,.2f}")
            if poc_volume > 0:
                sections.append(f"Volume: {poc_volume:,} shares")
            sections.append("Significance: Highest volume traded at this price")
        else:
            sections.append("⚠️ POC could not be calculated (insufficient data)")
        
        # ════════════════════════════════════════════════════════════
        # Value Area
        # ════════════════════════════════════════════════════════════
        sections.append(f"\n**📈 Value Area ({va_volume_pct:.0f}% of volume):**")
        
        if va_high > 0 and va_low > 0:
            sections.append(f"High: ${va_high:,.2f}")
            sections.append(f"Low: ${va_low:,.2f}")
            range_width = va_high - va_low
            range_pct = (range_width / va_low * 100) if va_low > 0 else 0
            sections.append(f"Range: ${range_width:,.2f} ({range_pct:.1f}%)")
        else:
            sections.append("⚠️ Value Area could not be calculated")
        
        # ════════════════════════════════════════════════════════════
        # Volume Distribution
        # ════════════════════════════════════════════════════════════
        volume_distribution = data.get('volume_distribution', [])
        
        if volume_distribution and len(volume_distribution) > 0:
            sections.append("\n**📊 Volume Distribution by Price:**")
            
            # Show top 5 price levels by volume
            sorted_dist = sorted(
                volume_distribution,
                key=lambda x: x.get('volume_pct', 0),
                reverse=True
            )[:5]
            
            for idx, level in enumerate(sorted_dist, 1):
                price_range = level.get('price_range', 'N/A')
                volume_pct = level.get('volume_pct', 0)
                
                if volume_pct > 0:
                    sections.append(f"{idx}. {price_range}: {volume_pct:.1f}% of total volume")
        
        # ════════════════════════════════════════════════════════════
        # High Volume Nodes (HVN)
        # ════════════════════════════════════════════════════════════
        hvn = data.get('high_volume_nodes', [])
        
        if hvn and len(hvn) > 0:
            sections.append("\n**🔴 High Volume Nodes (Strong S/R):**")
            for node in hvn[:3]:  # Top 3
                price = node.get('price', 0)
                if price > 0:
                    sections.append(f"- ${price:,.2f}")
        
        # ════════════════════════════════════════════════════════════
        # Low Volume Nodes (LVN)
        # ════════════════════════════════════════════════════════════
        lvn = data.get('low_volume_nodes', [])
        
        if lvn and len(lvn) > 0:
            sections.append("\n**🟢 Low Volume Nodes (Breakout zones):**")
            for node in lvn[:3]:  # Top 3
                price = node.get('price', 0)
                if price > 0:
                    sections.append(f"- ${price:,.2f}")
        
        # ════════════════════════════════════════════════════════════
        # Trading Implications (only if we have data)
        # ════════════════════════════════════════════════════════════
        if poc > 0 and va_high > 0 and va_low > 0:
            sections.append("\n**💡 Trading Implications:**")
            sections.append(f"- POC at ${poc:,.2f} acts as strong support/resistance")
            sections.append(f"- Value Area ${va_low:,.2f}-${va_high:,.2f} is acceptance zone")
            sections.append("- Price above Value Area: Bullish sentiment")
            sections.append("- Price below Value Area: Bearish sentiment")
            sections.append("- Low volume areas indicate potential breakout zones")
        
        return "\n".join(sections)


    @staticmethod
    def format_chart_patterns(patterns: List[Dict[str, Any]]) -> str:
        """
        Format chart patterns for LLM
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Formatted text for LLM
        """
        if not patterns:
            return "No chart patterns detected"
        
        sections = []
        sections.append("📐 CHART PATTERNS DETECTED")
        sections.append("═" * 60)
        
        for pattern in patterns:
            pattern_type = pattern.get("type", "Unknown")
            confidence = pattern.get("confidence", "Unknown")
            signal = pattern.get("signal", "N/A")
            price_level = pattern.get("price_level")
            
            sections.append(f"\n✅ **{pattern_type}**")
            sections.append(f"Confidence: {confidence}")
            sections.append(f"Signal: {signal}")
            
            if price_level:
                sections.append(f"Price Level: ${price_level:,.2f}")
            
            if start_date := pattern.get("start_date"):
                end_date = pattern.get("end_date", "")
                sections.append(f"Formation: {start_date} to {end_date}")
            
            # Add trading implications based on pattern type
            if "Double Bottom" in pattern_type:
                sections.append("\n**Trading Implications:**")
                sections.append("- Bullish reversal pattern")
                sections.append("- Entry above neckline breakout")
                sections.append("- Target: Height of pattern added to breakout")
            
            elif "Double Top" in pattern_type:
                sections.append("\n**Trading Implications:**")
                sections.append("- Bearish reversal pattern")
                sections.append("- Entry below neckline breakdown")
                sections.append("- Target: Height of pattern subtracted from breakdown")
            
            elif "Breakout" in pattern_type:
                sections.append("\n**Trading Implications:**")
                sections.append("- Continuation pattern")
                sections.append("- Confirm with volume increase")
                sections.append("- Use previous resistance as new support")
        
        return "\n".join(sections)


    @staticmethod
    def format_support_resistance(data: Dict) -> str:
        """
        Format support and resistance levels data
        
        Args:
            data: Support/resistance data from getSupportResistance tool
            
        Returns:
            Formatted string for LLM consumption
        """
        symbol = data.get("symbol", "N/A")
        current_price = data.get("current_price", 0)
        
        lines = [f"📊 **SUPPORT & RESISTANCE LEVELS - {symbol}**\n"]
        lines.append(f"💰 **Current Price:** ${current_price:,.2f}\n")
        
        # Resistance levels
        resistance_levels = data.get("resistance_levels", [])
        if resistance_levels:
            lines.append("⬆️ **RESISTANCE LEVELS:**")
            for i, level in enumerate(resistance_levels, 1):
                price = level.get("price", 0) if isinstance(level, dict) else level
                distance = ((price - current_price) / current_price * 100) if current_price > 0 else 0
                
                if distance > 0:
                    lines.append(f"  R{i}: ${price:,.2f} ({distance:+.2f}% above)")
                else:
                    lines.append(f"  R{i}: ${price:,.2f} (broken)")
        
        # Support levels
        support_levels = data.get("support_levels", [])
        if support_levels:
            lines.append("\n⬇️ **SUPPORT LEVELS:**")
            for i, level in enumerate(support_levels, 1):
                price = level.get("price", 0) if isinstance(level, dict) else level
                distance = ((current_price - price) / current_price * 100) if current_price > 0 else 0
                
                if distance > 0:
                    lines.append(f"  S{i}: ${price:,.2f} ({distance:.2f}% below)")
                else:
                    lines.append(f"  S{i}: ${price:,.2f} (broken)")
        
        # Nearest levels
        nearest_support = data.get("nearest_support")
        nearest_resistance = data.get("nearest_resistance")
        
        if nearest_support or nearest_resistance:
            lines.append("\n🎯 **KEY LEVELS:**")
            if nearest_support:
                support_price = nearest_support.get("price", 0) if isinstance(nearest_support, dict) else nearest_support
                distance = ((current_price - support_price) / current_price * 100) if current_price > 0 else 0
                lines.append(f"  Nearest Support: ${support_price:,.2f} ({distance:.2f}% below)")
            
            if nearest_resistance:
                resistance_price = nearest_resistance.get("price", 0) if isinstance(nearest_resistance, dict) else nearest_resistance
                distance = ((resistance_price - current_price) / current_price * 100) if current_price > 0 else 0
                lines.append(f"  Nearest Resistance: ${resistance_price:,.2f} ({distance:+.2f}% above)")
        
        # Trading range
        if support_levels and resistance_levels:
            lowest_support = min([s.get("price", 0) if isinstance(s, dict) else s for s in support_levels])
            highest_resistance = max([r.get("price", 0) if isinstance(r, dict) else r for r in resistance_levels])
            trading_range = ((highest_resistance - lowest_support) / lowest_support * 100) if lowest_support > 0 else 0
            
            lines.append(f"\n📏 **Trading Range:** ${lowest_support:,.2f} - ${highest_resistance:,.2f} ({trading_range:.2f}% range)")
        
        # Strength/conviction if available
        strength = data.get("strength") or data.get("conviction")
        if strength:
            lines.append(f"\n💪 **Level Strength:** {strength}")
        
        # Notes or analysis
        analysis = data.get("analysis") or data.get("notes")
        if analysis:
            lines.append(f"\n📝 **Analysis:** {analysis}")
        
        return "\n".join(lines)


    @staticmethod
    def format_relative_strength(rs_data: Dict[str, Any]) -> str:
        """
        Format relative strength analysis for LLM
        
        Args:
            rs_data: Dict with RS scores and performance data
            
        Returns:
            Formatted text for LLM
        """
        if not rs_data:
            return "No relative strength data available"
        
        sections = []
        
        symbol = rs_data.get("symbol", "N/A")
        benchmark = rs_data.get("benchmark", "SPY")
        
        sections.append(f"📊 RELATIVE STRENGTH - {symbol} vs {benchmark}")
        sections.append("═" * 60)
        
        rs_scores = rs_data.get("relative_strength", {})
        
        if not rs_scores:
            return f"Insufficient data for relative strength analysis of {symbol}"
        
        # Display RS scores
        sections.append("\n**Relative Strength Scores:**")
        
        for period in ["21d", "63d", "126d", "252d"]:
            rs_key = f"RS_{period}"
            if rs_score := rs_scores.get(rs_key):
                period_name = {
                    "21d": "1-Month",
                    "63d": "3-Month",
                    "126d": "6-Month",
                    "252d": "1-Year"
                }.get(period, period)
                
                # Rating based on RS score
                if rs_score >= 80:
                    rating = "⭐⭐⭐ Strong Outperformance"
                elif rs_score >= 65:
                    rating = "⭐⭐ Moderate Outperformance"
                elif rs_score >= 50:
                    rating = "⭐ Slight Outperformance"
                elif rs_score >= 35:
                    rating = "⚠️ Slight Underperformance"
                elif rs_score >= 20:
                    rating = "⚠️⚠️ Moderate Underperformance"
                else:
                    rating = "⚠️⚠️⚠️ Strong Underperformance"
                
                sections.append(f"- {period_name}: {rs_score:.1f} ({rating})")
        
        # Performance details
        sections.append("\n**Performance Comparison:**")
        
        for period in ["21d", "63d", "126d", "252d"]:
            stock_ret = rs_scores.get(f"Return_{period}")
            bench_ret = rs_scores.get(f"Benchmark_{period}")
            excess = rs_scores.get(f"Excess_{period}")
            
            if stock_ret is not None and bench_ret is not None:
                period_name = {
                    "21d": "1-Month",
                    "63d": "3-Month",
                    "126d": "6-Month",
                    "252d": "1-Year"
                }.get(period, period)
                
                sections.append(
                    f"- {period_name}: {symbol} {stock_ret:+.1f}% vs "
                    f"{benchmark} {bench_ret:+.1f}% = {excess:+.1f}% excess"
                )
        
        # Summary
        avg_rs = sum(rs_scores.get(f"RS_{p}", 50) for p in ["21d", "63d", "126d", "252d"]) / 4
        
        sections.append("\n**📈 Summary:**")
        if avg_rs >= 65:
            sections.append(f"- {symbol} is strongly outperforming {benchmark}")
            sections.append("- Relative strength supports bullish outlook")
        elif avg_rs >= 50:
            sections.append(f"- {symbol} is moderately outperforming {benchmark}")
            sections.append("- Positive relative momentum")
        elif avg_rs >= 35:
            sections.append(f"- {symbol} is underperforming {benchmark}")
            sections.append("- Weak relative momentum")
        else:
            sections.append(f"- {symbol} is significantly underperforming {benchmark}")
            sections.append("- Consider rotating to stronger stocks")
        
        return "\n".join(sections)
    

    @staticmethod
    def format_sentiment_data(sentiment_data: Dict[str, Any]) -> str:
        """
        Format sentiment analysis data for LLM
        
        Args:
            sentiment_data: Dict containing sentiment metrics
            
        Returns:
            Formatted text for LLM
        """
        if not sentiment_data:
            return "No sentiment data available"
        
        sections = []
        
        symbol = sentiment_data.get('symbol', 'N/A')
        sections.append(f"💭 SENTIMENT ANALYSIS - {symbol}")
        sections.append("═" * 60)
        
        # Overall Sentiment
        overall = sentiment_data.get('overall_sentiment', 'N/A')
        score = sentiment_data.get('sentiment_score', 0.0)
        level = sentiment_data.get('sentiment_level', 'N/A')
        
        sections.append(f"\n**Overall Sentiment:** {overall} ({level})")
        sections.append(f"**Sentiment Score:** {score:.3f} (Range: -1 to +1)")
        
        # Score interpretation
        if score > 0.6:
            sections.append("📈 Very bullish sentiment - Strong positive investor mood")
        elif score > 0.2:
            sections.append("📊 Positive sentiment - Mild optimism")
        elif score > -0.2:
            sections.append("➡️ Neutral sentiment - Balanced market view")
        elif score > -0.6:
            sections.append("📉 Negative sentiment - Cautious outlook")
        else:
            sections.append("⚠️ Very bearish sentiment - Strong pessimism")
        
        # Trend
        trend = sentiment_data.get('sentiment_trend', 'N/A')
        if trend != 'N/A':
            sections.append(f"\n**Sentiment Trend:** {trend}")
            
            if trend == "improving":
                sections.append("✅ Sentiment is getting more positive")
            elif trend == "declining":
                sections.append("⚠️ Sentiment is deteriorating")
            else:
                sections.append("➡️ Sentiment is stable")
        
        # Recent Shift
        recent_shift = sentiment_data.get('recent_shift', 'N/A')
        if recent_shift != 'N/A':
            sections.append(f"**Recent Change:** {recent_shift}")
        
        # Volatility
        volatility = sentiment_data.get('volatility', 0.0)
        if volatility:
            sections.append(f"**Sentiment Volatility:** {volatility:.2f}")
            
            if volatility > 0.3:
                sections.append("⚡ High volatility - Rapidly changing opinions")
            elif volatility > 0.15:
                sections.append("📊 Moderate volatility - Some fluctuation")
            else:
                sections.append("🔒 Low volatility - Stable sentiment")
        
        # News Sentiment Breakdown
        news_sentiment = sentiment_data.get('news_sentiment', {})
        if news_sentiment and isinstance(news_sentiment, dict):
            sections.append("\n**📰 News Sentiment:**")
            
            news_score = news_sentiment.get('score', 0.0)
            news_count = news_sentiment.get('article_count', 0)
            
            if news_count > 0:
                sections.append(f"- {news_count} articles analyzed")
                sections.append(f"- Average score: {news_score:.2f}")
        
        # Social Media Sentiment Breakdown
        social_sentiment = sentiment_data.get('social_sentiment', {})
        if social_sentiment and isinstance(social_sentiment, dict):
            sections.append("\n**📱 Social Media Sentiment:**")
            
            twitter_score = social_sentiment.get('twitter_score', 0.0)
            stocktwits_score = social_sentiment.get('stocktwits_score', 0.0)
            
            if twitter_score:
                sections.append(f"- Twitter: {twitter_score:.2f}")
            if stocktwits_score:
                sections.append(f"- StockTwits: {stocktwits_score:.2f}")
        
        # Data Quality
        data_points = sentiment_data.get('data_points', 0)
        date_range = sentiment_data.get('date_range', 'N/A')
        
        if data_points > 0:
            sections.append(f"\n**Data Quality:**")
            sections.append(f"- {data_points} data points analyzed")
            sections.append(f"- Date range: {date_range}")
        
        # Trading Implications
        sections.append("\n**💡 Trading Implications:**")
        
        if score > 0.5:
            sections.append("- Strong positive sentiment may indicate overbought conditions")
            sections.append("- Consider taking profits or waiting for pullback")
        elif score > 0.2:
            sections.append("- Positive sentiment supports upward momentum")
            sections.append("- Good environment for holding long positions")
        elif score < -0.5:
            sections.append("- Extreme negative sentiment may signal capitulation")
            sections.append("- Potential contrarian buying opportunity")
        elif score < -0.2:
            sections.append("- Negative sentiment creates headwinds")
            sections.append("- Exercise caution with new long positions")
        else:
            sections.append("- Neutral sentiment - lack of strong directional conviction")
            sections.append("- Wait for clearer trend before major positions")
        
        return "\n".join(sections)
    
    
    @staticmethod
    def format_stop_loss_data(stop_loss_data: Dict[str, Any]) -> str:
        """
        Format stop loss suggestions for LLM
        
        Args:
            stop_loss_data: Dict containing stop loss recommendations
            
        Returns:
            Formatted text for LLM
        """
        if not stop_loss_data:
            return "No stop loss data available"
        
        sections = []
        
        symbol = stop_loss_data.get('symbol', 'N/A')
        sections.append(f"🛡️ STOP LOSS SUGGESTIONS - {symbol}")
        sections.append("═" * 60)
        
        current_price = stop_loss_data.get('current_price', 0.0)
        if current_price:
            sections.append(f"\n**Current Price:** ${current_price:,.2f}")
        
        stop_levels = stop_loss_data.get('stop_levels', {})
        
        # ATR-Based Stops
        if atr_stops := stop_levels.get('atr_based', {}):
            sections.append("\n**📊 ATR-Based Stop Loss:**")
            sections.append("(Based on Average True Range - adapts to volatility)")
            
            atr_2x = atr_stops.get('atr_2x', 0.0)
            atr_3x = atr_stops.get('atr_3x', 0.0)
            
            if atr_2x:
                risk_pct = ((current_price - atr_2x) / current_price * 100)
                sections.append(f"- ATR 2x: ${atr_2x:,.2f} ({risk_pct:.2f}% risk)")
                sections.append("  → Conservative, allows for normal volatility")
            
            if atr_3x:
                risk_pct = ((current_price - atr_3x) / current_price * 100)
                sections.append(f"- ATR 3x: ${atr_3x:,.2f} ({risk_pct:.2f}% risk)")
                sections.append("  → More room for price swings")
        
        # Percentage-Based Stops
        if pct_stops := stop_levels.get('percentage_based', {}):
            sections.append("\n**📉 Percentage-Based Stop Loss:**")
            sections.append("(Fixed percentage below entry)")
            
            pct_3 = pct_stops.get('percent_3', 0.0)
            pct_5 = pct_stops.get('percent_5', 0.0)
            pct_7 = pct_stops.get('percent_7', 0.0)
            
            if pct_3:
                sections.append(f"- 3% stop: ${pct_3:,.2f}")
            if pct_5:
                sections.append(f"- 5% stop: ${pct_5:,.2f}")
            if pct_7:
                sections.append(f"- 7% stop: ${pct_7:,.2f}")
        
        # SMA-Based Stops
        if sma_stops := stop_levels.get('sma_based', {}):
            sections.append("\n**📈 Moving Average-Based Stop Loss:**")
            sections.append("(Support from key moving averages)")
            
            sma_20 = sma_stops.get('sma_20', 0.0)
            sma_50 = sma_stops.get('sma_50', 0.0)
            
            if sma_20:
                risk_pct = ((current_price - sma_20) / current_price * 100)
                sections.append(f"- Below 20-day SMA: ${sma_20:,.2f} ({risk_pct:.2f}% risk)")
            
            if sma_50:
                risk_pct = ((current_price - sma_50) / current_price * 100)
                sections.append(f"- Below 50-day SMA: ${sma_50:,.2f} ({risk_pct:.2f}% risk)")
        
        # Technical Stops
        if tech_stops := stop_levels.get('technical', {}):
            sections.append("\n**🎯 Technical-Based Stop Loss:**")
            sections.append("(Based on price action)")
            
            swing_low = tech_stops.get('recent_swing', 0.0)
            if swing_low:
                risk_pct = ((current_price - swing_low) / current_price * 100)
                sections.append(f"- Recent Swing Low: ${swing_low:,.2f} ({risk_pct:.2f}% risk)")
                sections.append("  → Natural support from previous price action")
        
        # Recommendations
        recommendations = stop_loss_data.get('recommendations', {})
        if recommendations:
            sections.append("\n**💡 RECOMMENDATIONS:**")
            
            conservative = recommendations.get('conservative', {})
            if conservative:
                level = conservative.get('stop_level', 0.0)
                method = conservative.get('method', 'N/A')
                risk = conservative.get('risk_percent', 0.0)
                sections.append(f"\n🟢 **Conservative (Low Risk):**")
                sections.append(f"   Stop: ${level:,.2f} | Method: {method}")
                sections.append(f"   Risk: {risk:.2f}% | Best for: Capital preservation")
            
            moderate = recommendations.get('moderate', {})
            if moderate:
                level = moderate.get('stop_level', 0.0)
                method = moderate.get('method', 'N/A')
                risk = moderate.get('risk_percent', 0.0)
                sections.append(f"\n🟡 **Moderate (Balanced Risk):**")
                sections.append(f"   Stop: ${level:,.2f} | Method: {method}")
                sections.append(f"   Risk: {risk:.2f}% | Best for: Balanced approach")
            
            aggressive = recommendations.get('aggressive', {})
            if aggressive:
                level = aggressive.get('stop_level', 0.0)
                method = aggressive.get('method', 'N/A')
                risk = aggressive.get('risk_percent', 0.0)
                sections.append(f"\n🔴 **Aggressive (Higher Risk):**")
                sections.append(f"   Stop: ${level:,.2f} | Method: {method}")
                sections.append(f"   Risk: {risk:.2f}% | Best for: Higher risk tolerance")
        
        # Risk Per Share
        risk_per_share = stop_loss_data.get('risk_per_share', {})
        if risk_per_share:
            sections.append("\n**💰 Risk Per Share:**")
            for method, risk_amount in risk_per_share.items():
                if risk_amount:
                    sections.append(f"- {method}: ${risk_amount:.2f} per share")
        
        # General Guidelines
        sections.append("\n**📚 Stop Loss Guidelines:**")
        sections.append("1. Always use a stop loss to protect capital")
        sections.append("2. Place stops below key support levels")
        sections.append("3. Adjust stops as price moves in your favor (trailing stops)")
        sections.append("4. Never move stops away from entry (only tighter)")
        sections.append("5. Size positions based on stop distance to maintain consistent risk")
        
        return "\n".join(sections)
    

    @staticmethod
    def format_bollinger_patterns(bb_patterns: Dict[str, Any]) -> str:
        """
        Format Bollinger Bands patterns for LLM
        
        Enhanced version with all pattern types
        """
        if not bb_patterns:
            return "No Bollinger Bands patterns detected"
        
        sections = []
        sections.append("🎯 BOLLINGER BANDS PATTERNS")
        sections.append("═" * 60)
        
        # W-Bottom Pattern
        w_bottom = bb_patterns.get('w_bottom', {})
        if w_bottom.get('detected'):
            sections.append("\n✅ **W-BOTTOM PATTERN DETECTED** (Bullish)")
            sections.append(f"Confidence: {w_bottom.get('confidence', 0):.0%}")
            
            if 'first_bottom' in w_bottom:
                sections.append(f"First Bottom: ${w_bottom['first_bottom']['price']:,.2f} on {w_bottom['first_bottom']['date']}")
            if 'second_bottom' in w_bottom:
                sections.append(f"Second Bottom: ${w_bottom['second_bottom']['price']:,.2f} on {w_bottom['second_bottom']['date']}")
            
            if w_bottom.get('breakout_confirmed'):
                sections.append("🚀 Middle band breakout CONFIRMED")
            
            sections.append(f"\n📊 Signal: {w_bottom.get('description', 'Bullish reversal expected')}")
        
        # M-Top Pattern
        m_top = bb_patterns.get('m_top', {})
        if m_top.get('detected'):
            sections.append("\n⚠️ **M-TOP PATTERN DETECTED** (Bearish)")
            sections.append(f"Confidence: {m_top.get('confidence', 0):.0%}")
            
            if 'first_top' in m_top:
                sections.append(f"First Top: ${m_top['first_top']['price']:,.2f} on {m_top['first_top']['date']}")
            if 'second_top' in m_top:
                sections.append(f"Second Top: ${m_top['second_top']['price']:,.2f} on {m_top['second_top']['date']}")
            
            if m_top.get('breakdown_confirmed'):
                sections.append("📉 Middle band breakdown CONFIRMED")
            
            sections.append(f"\n📊 Signal: {m_top.get('description', 'Bearish reversal expected')}")
        
        # Squeeze Breakout
        squeeze = bb_patterns.get('squeeze_breakout', {})
        if squeeze.get('detected'):
            if squeeze.get('breakout_confirmed'):
                direction = squeeze.get('direction', 'unknown').upper()
                emoji = "🚀" if direction == "UP" else "⚠️"
                
                sections.append(f"\n{emoji} **SQUEEZE BREAKOUT** - {direction}")
                sections.append(f"Confidence: {squeeze.get('confidence', 0):.0%}")
                sections.append(f"Direction: {squeeze.get('signal', 'N/A').title()}")
                
                if squeeze.get('volume_confirmed'):
                    sections.append("✅ Volume confirmation: STRONG")
                else:
                    sections.append("⚠️ Volume confirmation: Moderate")
                
                sections.append(f"Bandwidth: {squeeze.get('current_bandwidth', 0):.4f}")
                sections.append(f"\n📊 {squeeze.get('description', 'High volatility expansion expected')}")
            
            elif squeeze.get('squeeze_active'):
                sections.append("\n⏳ **SQUEEZE ACTIVE**")
                sections.append(f"Bandwidth: {squeeze.get('current_bandwidth', 0):.4f}")
                sections.append("Status: Awaiting breakout direction")
                sections.append("\n💡 Watch for directional move with increased volume")
        
        if not any([
            w_bottom.get('detected'),
            m_top.get('detected'),
            squeeze.get('detected')
        ]):
            return "No significant Bollinger Bands patterns detected"
        
        return "\n".join(sections)
    

    @staticmethod
    def format_income_statement(data: Dict[str, Any]) -> str:
        """
        Format income statement data for LLM
        
        Args:
            data: Income statement data from GetIncomeStatementTool
            
        Returns:
            Formatted text for LLM consumption
        """
        if not data:
            return "No income statement data available"
        
        sections = []
        symbol = data.get('symbol', 'N/A')
        period_type = data.get('period_type', 'annual')
        
        sections.append(f"📊 INCOME STATEMENT - {symbol} ({period_type.upper()})")
        sections.append("═" * 60)
        
        latest = data.get('latest_period', {})
        
        if not latest:
            return "No income statement periods available"
        
        # ═══════════════════════════════════════════════════════════
        # Header Info
        # ═══════════════════════════════════════════════════════════
        sections.append(f"\n**Period:** {latest.get('date', 'N/A')}")
        sections.append(f"**Fiscal Period:** {latest.get('period', 'N/A')} {latest.get('calendar_year', '')}")
        
        # ═══════════════════════════════════════════════════════════
        # Revenue & Cost
        # ═══════════════════════════════════════════════════════════
        revenue = latest.get('revenue', 0)
        cost_of_revenue = latest.get('cost_of_revenue', 0)
        gross_profit = latest.get('gross_profit', 0)
        gross_margin = latest.get('gross_profit_ratio', 0)
        
        sections.append(f"\n**💰 Revenue & Cost:**")
        if revenue:
            sections.append(f"- Revenue: ${revenue:,.0f}")
        if cost_of_revenue:
            sections.append(f"- Cost of Revenue: ${cost_of_revenue:,.0f}")
        if gross_profit:
            sections.append(f"- Gross Profit: ${gross_profit:,.0f} ({gross_margin*100:.1f}% margin)")
        
        # ═══════════════════════════════════════════════════════════
        # Operating Performance
        # ═══════════════════════════════════════════════════════════
        operating_expenses = latest.get('operating_expenses', 0)
        rd_expenses = latest.get('rd_expenses', 0)
        operating_income = latest.get('operating_income', 0)
        operating_margin = latest.get('operating_income_ratio', 0)
        ebitda = latest.get('ebitda', 0)
        
        sections.append(f"\n**🏭 Operating Performance:**")
        if operating_expenses:
            sections.append(f"- Operating Expenses: ${operating_expenses:,.0f}")
        if rd_expenses:
            sections.append(f"- R&D Expenses: ${rd_expenses:,.0f}")
        if operating_income:
            sections.append(f"- Operating Income: ${operating_income:,.0f} ({operating_margin*100:.1f}% margin)")
        if ebitda:
            sections.append(f"- EBITDA: ${ebitda:,.0f}")
        
        # ═══════════════════════════════════════════════════════════
        # Net Income
        # ═══════════════════════════════════════════════════════════
        income_before_tax = latest.get('income_before_tax', 0)
        income_tax = latest.get('income_tax_expense', 0)
        net_income = latest.get('net_income', 0)
        net_margin = latest.get('net_income_ratio', 0)
        
        sections.append(f"\n**💵 Net Income:**")
        if income_before_tax:
            sections.append(f"- Income Before Tax: ${income_before_tax:,.0f}")
        if income_tax:
            sections.append(f"- Income Tax Expense: ${income_tax:,.0f}")
        if net_income:
            sections.append(f"- Net Income: ${net_income:,.0f} ({net_margin*100:.1f}% margin)")
        
        # ═══════════════════════════════════════════════════════════
        # Earnings Per Share
        # ═══════════════════════════════════════════════════════════
        eps = latest.get('eps', 0)
        eps_diluted = latest.get('eps_diluted', 0)
        
        sections.append(f"\n**📈 Earnings Per Share:**")
        if eps:
            sections.append(f"- EPS (Basic): ${eps:.2f}")
        if eps_diluted:
            sections.append(f"- EPS (Diluted): ${eps_diluted:.2f}")
        
        # ═══════════════════════════════════════════════════════════
        # Historical Comparison
        # ═══════════════════════════════════════════════════════════
        periods = data.get('periods', [])
        if len(periods) > 1:
            sections.append(f"\n**📊 Historical Trend ({len(periods)} periods):**")
            
            # Show recent periods
            for idx, period in enumerate(periods[:3]):
                date = period.get('date', 'N/A')
                rev = period.get('revenue', 0)
                ni = period.get('net_income', 0)
                sections.append(f"  {idx+1}. {date}: Revenue ${rev:,.0f}, Net Income ${ni:,.0f}")
        
        return "\n".join(sections)
    
    @staticmethod
    def format_balance_sheet(data: Dict[str, Any]) -> str:
        """
        Format balance sheet data for LLM
        
        Args:
            data: Balance sheet data from GetBalanceSheetTool
            
        Returns:
            Formatted text for LLM consumption
        """
        if not data:
            return "No balance sheet data available"
        
        sections = []
        symbol = data.get('symbol', 'N/A')
        period_type = data.get('period_type', 'annual')
        
        sections.append(f"🏦 BALANCE SHEET - {symbol} ({period_type.upper()})")
        sections.append("═" * 60)
        
        latest = data.get('latest_period', {})
        
        if not latest:
            return "No balance sheet periods available"
        
        # ═══════════════════════════════════════════════════════════
        # Header Info
        # ═══════════════════════════════════════════════════════════
        sections.append(f"\n**Period:** {latest.get('date', 'N/A')}")
        sections.append(f"**Fiscal Period:** {latest.get('period', 'N/A')} {latest.get('calendar_year', '')}")
        
        # ═══════════════════════════════════════════════════════════
        # Assets
        # ═══════════════════════════════════════════════════════════
        total_assets = latest.get('total_assets', 0)
        current_assets = latest.get('total_current_assets', 0)
        non_current_assets = latest.get('total_non_current_assets', 0)
        cash = latest.get('cash_and_cash_equivalents', 0)
        receivables = latest.get('net_receivables', 0)
        inventory = latest.get('inventory', 0)
        ppe = latest.get('property_plant_equipment', 0)
        
        sections.append(f"\n**💎 ASSETS:**")
        if total_assets:
            sections.append(f"- **Total Assets: ${total_assets:,.0f}**")
        
        sections.append(f"\n  Current Assets:")
        if current_assets:
            sections.append(f"  - Total Current: ${current_assets:,.0f}")
        if cash:
            sections.append(f"  - Cash & Equivalents: ${cash:,.0f}")
        if receivables:
            sections.append(f"  - Receivables: ${receivables:,.0f}")
        if inventory:
            sections.append(f"  - Inventory: ${inventory:,.0f}")
        
        sections.append(f"\n  Non-Current Assets:")
        if non_current_assets:
            sections.append(f"  - Total Non-Current: ${non_current_assets:,.0f}")
        if ppe:
            sections.append(f"  - Property, Plant & Equipment: ${ppe:,.0f}")
        
        # ═══════════════════════════════════════════════════════════
        # Liabilities
        # ═══════════════════════════════════════════════════════════
        total_liabilities = latest.get('total_liabilities', 0)
        current_liabilities = latest.get('total_current_liabilities', 0)
        non_current_liabilities = latest.get('total_non_current_liabilities', 0)
        accounts_payable = latest.get('accounts_payable', 0)
        short_term_debt = latest.get('short_term_debt', 0)
        long_term_debt = latest.get('long_term_debt', 0)
        total_debt = latest.get('total_debt', 0)
        net_debt = latest.get('net_debt', 0)
        
        sections.append(f"\n**⚖️ LIABILITIES:**")
        if total_liabilities:
            sections.append(f"- **Total Liabilities: ${total_liabilities:,.0f}**")
        
        sections.append(f"\n  Current Liabilities:")
        if current_liabilities:
            sections.append(f"  - Total Current: ${current_liabilities:,.0f}")
        if accounts_payable:
            sections.append(f"  - Accounts Payable: ${accounts_payable:,.0f}")
        if short_term_debt:
            sections.append(f"  - Short-Term Debt: ${short_term_debt:,.0f}")
        
        sections.append(f"\n  Non-Current Liabilities:")
        if non_current_liabilities:
            sections.append(f"  - Total Non-Current: ${non_current_liabilities:,.0f}")
        if long_term_debt:
            sections.append(f"  - Long-Term Debt: ${long_term_debt:,.0f}")
        
        if total_debt:
            sections.append(f"\n  Total Debt: ${total_debt:,.0f}")
        if net_debt:
            sections.append(f"  Net Debt: ${net_debt:,.0f}")
        
        # ═══════════════════════════════════════════════════════════
        # Equity
        # ═══════════════════════════════════════════════════════════
        total_equity = latest.get('total_equity', 0)
        retained_earnings = latest.get('retained_earnings', 0)
        
        sections.append(f"\n**🏛️ SHAREHOLDERS' EQUITY:**")
        if total_equity:
            sections.append(f"- **Total Equity: ${total_equity:,.0f}**")
        if retained_earnings:
            sections.append(f"- Retained Earnings: ${retained_earnings:,.0f}")
        
        # ═══════════════════════════════════════════════════════════
        # Key Metrics
        # ═══════════════════════════════════════════════════════════
        working_capital = latest.get('working_capital', 0)
        debt_to_equity = latest.get('debt_to_equity_ratio', 0)
        
        sections.append(f"\n**📊 Key Metrics:**")
        if working_capital:
            sections.append(f"- Working Capital: ${working_capital:,.0f}")
        if debt_to_equity:
            sections.append(f"- Debt/Equity Ratio: {debt_to_equity:.2f}")
        
        # ═══════════════════════════════════════════════════════════
        # Accounting Equation Check
        # ═══════════════════════════════════════════════════════════
        if total_assets and total_liabilities and total_equity:
            calculated_equity = total_assets - total_liabilities
            sections.append(f"\n**✅ Balance Check:**")
            sections.append(f"Assets = Liabilities + Equity")
            sections.append(f"${total_assets:,.0f} = ${total_liabilities:,.0f} + ${total_equity:,.0f}")
            if abs(calculated_equity - total_equity) < 1000:  # Allow small rounding difference
                sections.append("✓ Balance sheet is balanced")
        
        return "\n".join(sections)
    
    @staticmethod
    def format_cash_flow(data: Dict[str, Any]) -> str:
        """
        Format cash flow statement data for LLM
        
        Args:
            data: Cash flow data from GetCashFlowTool
            
        Returns:
            Formatted text for LLM consumption
        """
        if not data:
            return "No cash flow data available"
        
        sections = []
        symbol = data.get('symbol', 'N/A')
        period_type = data.get('period_type', 'annual')
        
        sections.append(f"💵 CASH FLOW STATEMENT - {symbol} ({period_type.upper()})")
        sections.append("═" * 60)
        
        latest = data.get('latest_period', {})
        
        if not latest:
            return "No cash flow periods available"
        
        # ═══════════════════════════════════════════════════════════
        # Header Info
        # ═══════════════════════════════════════════════════════════
        sections.append(f"\n**Period:** {latest.get('date', 'N/A')}")
        sections.append(f"**Fiscal Period:** {latest.get('period', 'N/A')} {latest.get('calendar_year', '')}")
        
        # ═══════════════════════════════════════════════════════════
        # Operating Activities
        # ═══════════════════════════════════════════════════════════
        operating_cf = latest.get('operating_cash_flow', 0)
        depreciation = latest.get('depreciation_and_amortization', 0)
        stock_comp = latest.get('stock_based_compensation', 0)
        working_capital_change = latest.get('change_in_working_capital', 0)
        
        sections.append(f"\n**🏭 Operating Activities:**")
        if operating_cf:
            cf_sign = "+" if operating_cf > 0 else "-"
            sections.append(f"- **Net Cash from Operations: {cf_sign}${abs(operating_cf):,.0f}**")
        if depreciation:
            sections.append(f"- Depreciation & Amortization: ${depreciation:,.0f}")
        if stock_comp:
            sections.append(f"- Stock-Based Compensation: ${stock_comp:,.0f}")
        if working_capital_change:
            sections.append(f"- Change in Working Capital: ${working_capital_change:+,.0f}")
        
        # ═══════════════════════════════════════════════════════════
        # Investing Activities
        # ═══════════════════════════════════════════════════════════
        investing_cf = latest.get('investing_cash_flow', 0)
        capex = latest.get('capital_expenditure', 0)
        acquisitions = latest.get('acquisitions', 0)
        investments_purchased = latest.get('purchases_of_investments', 0)
        investments_sold = latest.get('sales_of_investments', 0)
        
        sections.append(f"\n**💼 Investing Activities:**")
        if investing_cf:
            cf_sign = "+" if investing_cf > 0 else "-"
            sections.append(f"- **Net Cash from Investing: {cf_sign}${abs(investing_cf):,.0f}**")
        if capex:
            sections.append(f"- Capital Expenditures: ${capex:,.0f}")
        if acquisitions:
            sections.append(f"- Acquisitions: ${acquisitions:,.0f}")
        if investments_purchased:
            sections.append(f"- Investments Purchased: ${investments_purchased:,.0f}")
        if investments_sold:
            sections.append(f"- Investments Sold: ${investments_sold:,.0f}")
        
        # ═══════════════════════════════════════════════════════════
        # Financing Activities
        # ═══════════════════════════════════════════════════════════
        financing_cf = latest.get('financing_cash_flow', 0)
        debt_repayment = latest.get('debt_repayment', 0)
        stock_issued = latest.get('common_stock_issued', 0)
        stock_repurchased = latest.get('common_stock_repurchased', 0)
        dividends = latest.get('dividends_paid', 0)
        
        sections.append(f"\n**🏦 Financing Activities:**")
        if financing_cf:
            cf_sign = "+" if financing_cf > 0 else "-"
            sections.append(f"- **Net Cash from Financing: {cf_sign}${abs(financing_cf):,.0f}**")
        if debt_repayment:
            sections.append(f"- Debt Repayment: ${debt_repayment:,.0f}")
        if stock_issued:
            sections.append(f"- Stock Issued: ${stock_issued:,.0f}")
        if stock_repurchased:
            sections.append(f"- Stock Repurchased: ${stock_repurchased:,.0f}")
        if dividends:
            sections.append(f"- Dividends Paid: ${dividends:,.0f}")
        
        # ═══════════════════════════════════════════════════════════
        # Free Cash Flow
        # ═══════════════════════════════════════════════════════════
        fcf = latest.get('free_cash_flow', 0)
        fcf_margin = latest.get('fcf_margin', 0)
        
        sections.append(f"\n**🌟 Free Cash Flow:**")
        if fcf:
            fcf_sign = "+" if fcf > 0 else "-"
            sections.append(f"- **Free Cash Flow: {fcf_sign}${abs(fcf):,.0f}**")
        if fcf_margin:
            sections.append(f"- FCF Margin: {fcf_margin:.1f}%")
        
        # ═══════════════════════════════════════════════════════════
        # Net Change in Cash
        # ═══════════════════════════════════════════════════════════
        net_change = latest.get('net_change_in_cash', 0)
        cash_beginning = latest.get('cash_at_beginning', 0)
        cash_end = latest.get('cash_at_end', 0)
        
        sections.append(f"\n**💰 Net Change in Cash:**")
        if net_change:
            sections.append(f"- Net Change: ${net_change:+,.0f}")
        if cash_beginning:
            sections.append(f"- Cash at Beginning: ${cash_beginning:,.0f}")
        if cash_end:
            sections.append(f"- Cash at End: ${cash_end:,.0f}")
        
        # ═══════════════════════════════════════════════════════════
        # Cash Flow Analysis
        # ═══════════════════════════════════════════════════════════
        sections.append(f"\n**📊 Cash Flow Analysis:**")
        
        if operating_cf and operating_cf > 0:
            sections.append("✓ Positive operating cash flow - healthy core operations")
        elif operating_cf and operating_cf < 0:
            sections.append("⚠️ Negative operating cash flow - potential operational issues")
        
        if fcf and fcf > 0:
            sections.append("✓ Positive free cash flow - company can invest or return cash to shareholders")
        elif fcf and fcf < 0:
            sections.append("⚠️ Negative free cash flow - company may need external financing")
        
        if capex and operating_cf and abs(capex) > 0:
            capex_ratio = (abs(capex) / operating_cf * 100) if operating_cf > 0 else 0
            sections.append(f"- CapEx is {capex_ratio:.1f}% of operating cash flow")
        
        return "\n".join(sections)
    
    @staticmethod
    def format_financial_ratios(data: Dict[str, Any]) -> str:
        """
        Format financial ratios data for LLM
        
        Args:
            data: Financial ratios data from GetFinancialRatiosTool
            
        Returns:
            Formatted text for LLM consumption
        """
        if not data:
            return "No financial ratios data available"
        
        sections = []
        symbol = data.get('symbol', 'N/A')
        period_type = data.get('period_type', 'annual')
        
        sections.append(f"📊 FINANCIAL RATIOS - {symbol} ({period_type.upper()})")
        sections.append("═" * 60)
        
        latest = data.get('latest_period', {})
        
        if not latest:
            return "No financial ratios periods available"
        
        # ═══════════════════════════════════════════════════════════
        # Header Info
        # ═══════════════════════════════════════════════════════════
        sections.append(f"\n**Period:** {latest.get('date', 'N/A')}")
        
        # ═══════════════════════════════════════════════════════════
        # LIQUIDITY RATIOS
        # ═══════════════════════════════════════════════════════════
        current_ratio = latest.get('current_ratio', 0)
        quick_ratio = latest.get('quick_ratio', 0)
        cash_ratio = latest.get('cash_ratio', 0)
        
        sections.append(f"\n**💧 Liquidity Ratios:**")
        if current_ratio:
            status = "✓ Strong" if current_ratio >= 2.0 else "⚠️ Weak" if current_ratio < 1.0 else "○ Adequate"
            sections.append(f"- Current Ratio: {current_ratio:.2f} {status}")
        if quick_ratio:
            status = "✓ Strong" if quick_ratio >= 1.0 else "⚠️ Weak"
            sections.append(f"- Quick Ratio: {quick_ratio:.2f} {status}")
        if cash_ratio:
            sections.append(f"- Cash Ratio: {cash_ratio:.2f}")
        
        # ═══════════════════════════════════════════════════════════
        # PROFITABILITY RATIOS
        # ═══════════════════════════════════════════════════════════
        gross_margin = latest.get('gross_profit_margin', 0)
        operating_margin = latest.get('operating_profit_margin', 0)
        net_margin = latest.get('net_profit_margin', 0)
        ebitda_margin = latest.get('ebitda_margin', 0)
        roe = latest.get('return_on_equity', 0)
        roa = latest.get('return_on_assets', 0)
        roic = latest.get('return_on_capital_employed', 0)
        
        sections.append(f"\n**💰 Profitability Ratios:**")
        if gross_margin:
            sections.append(f"- Gross Profit Margin: {gross_margin*100:.1f}%")
        if operating_margin:
            sections.append(f"- Operating Profit Margin: {operating_margin*100:.1f}%")
        if net_margin:
            sections.append(f"- Net Profit Margin: {net_margin*100:.1f}%")
        if ebitda_margin:
            sections.append(f"- EBITDA Margin: {ebitda_margin*100:.1f}%")
        
        sections.append(f"\n  Return Metrics:")
        if roe:
            status = "✓ Excellent" if roe >= 0.15 else "○ Good" if roe >= 0.10 else "⚠️ Weak"
            sections.append(f"  - Return on Equity (ROE): {roe*100:.1f}% {status}")
        if roa:
            sections.append(f"  - Return on Assets (ROA): {roa*100:.1f}%")
        if roic:
            sections.append(f"  - Return on Capital Employed: {roic*100:.1f}%")
        
        # ═══════════════════════════════════════════════════════════
        # LEVERAGE RATIOS
        # ═══════════════════════════════════════════════════════════
        debt_equity = latest.get('debt_equity_ratio', 0)
        debt_ratio = latest.get('debt_ratio', 0)
        interest_coverage = latest.get('interest_coverage', 0)
        
        sections.append(f"\n**⚖️ Leverage Ratios:**")
        if debt_equity:
            status = "✓ Low leverage" if debt_equity < 0.5 else "⚠️ High leverage" if debt_equity > 2.0 else "○ Moderate"
            sections.append(f"- Debt/Equity Ratio: {debt_equity:.2f} {status}")
        if debt_ratio:
            sections.append(f"- Debt Ratio: {debt_ratio:.2f}")
        if interest_coverage:
            status = "✓ Strong" if interest_coverage >= 3.0 else "⚠️ Weak"
            sections.append(f"- Interest Coverage: {interest_coverage:.2f}x {status}")
        
        # ═══════════════════════════════════════════════════════════
        # VALUATION RATIOS
        # ═══════════════════════════════════════════════════════════
        pe_ratio = latest.get('price_earnings_ratio', 0)
        pb_ratio = latest.get('price_to_book_ratio', 0)
        ps_ratio = latest.get('price_to_sales_ratio', 0)
        peg_ratio = latest.get('price_earnings_to_growth_ratio', 0)
        ev_ebitda = latest.get('enterprise_value_multiple', 0)
        
        sections.append(f"\n**📈 Valuation Ratios:**")
        if pe_ratio:
            sections.append(f"- P/E Ratio: {pe_ratio:.2f}")
        if pb_ratio:
            sections.append(f"- P/B Ratio: {pb_ratio:.2f}")
        if ps_ratio:
            sections.append(f"- P/S Ratio: {ps_ratio:.2f}")
        if peg_ratio:
            status = "✓ Undervalued" if 0 < peg_ratio < 1 else "⚠️ Overvalued" if peg_ratio > 2 else "○ Fair"
            sections.append(f"- PEG Ratio: {peg_ratio:.2f} {status}")
        if ev_ebitda:
            sections.append(f"- EV/EBITDA: {ev_ebitda:.2f}")
        
        # ═══════════════════════════════════════════════════════════
        # EFFICIENCY RATIOS
        # ═══════════════════════════════════════════════════════════
        asset_turnover = latest.get('asset_turnover', 0)
        inventory_turnover = latest.get('inventory_turnover', 0)
        receivables_turnover = latest.get('receivables_turnover', 0)
        days_sales = latest.get('days_sales_outstanding', 0)
        days_inventory = latest.get('days_inventory_outstanding', 0)
        cash_conversion = latest.get('cash_conversion_cycle', 0)
        
        sections.append(f"\n**⚙️ Efficiency Ratios:**")
        if asset_turnover:
            sections.append(f"- Asset Turnover: {asset_turnover:.2f}x")
        if inventory_turnover:
            sections.append(f"- Inventory Turnover: {inventory_turnover:.2f}x")
        if receivables_turnover:
            sections.append(f"- Receivables Turnover: {receivables_turnover:.2f}x")
        
        sections.append(f"\n  Operating Cycle:")
        if days_sales:
            sections.append(f"  - Days Sales Outstanding: {days_sales:.0f} days")
        if days_inventory:
            sections.append(f"  - Days Inventory Outstanding: {days_inventory:.0f} days")
        if cash_conversion:
            status = "✓ Efficient" if cash_conversion < 60 else "⚠️ Slow"
            sections.append(f"  - Cash Conversion Cycle: {cash_conversion:.0f} days {status}")
        
        # ═══════════════════════════════════════════════════════════
        # DIVIDEND METRICS
        # ═══════════════════════════════════════════════════════════
        dividend_yield = latest.get('dividend_yield', 0)
        payout_ratio = latest.get('payout_ratio', 0)
        
        if dividend_yield or payout_ratio:
            sections.append(f"\n**💵 Dividend Metrics:**")
            if dividend_yield:
                sections.append(f"- Dividend Yield: {dividend_yield*100:.2f}%")
            if payout_ratio:
                status = "✓ Sustainable" if payout_ratio < 0.6 else "⚠️ High"
                sections.append(f"- Payout Ratio: {payout_ratio*100:.1f}% {status}")
        
        # ═══════════════════════════════════════════════════════════
        # Overall Financial Health Summary
        # ═══════════════════════════════════════════════════════════
        sections.append(f"\n**🏥 Financial Health Summary:**")
        
        health_score = 0
        max_score = 0
        
        # Check liquidity
        if current_ratio:
            max_score += 1
            if current_ratio >= 1.5:
                health_score += 1
                sections.append("✓ Strong liquidity position")
        
        # Check profitability
        if roe:
            max_score += 1
            if roe >= 0.10:
                health_score += 1
                sections.append("✓ Good return on equity")
        
        # Check leverage
        if debt_equity:
            max_score += 1
            if debt_equity < 1.0:
                health_score += 1
                sections.append("✓ Conservative debt levels")
        
        # Check interest coverage
        if interest_coverage:
            max_score += 1
            if interest_coverage >= 3.0:
                health_score += 1
                sections.append("✓ Strong debt service capability")
        
        if max_score > 0:
            health_pct = (health_score / max_score) * 100
            sections.append(f"\n**Overall Health Score: {health_pct:.0f}% ({health_score}/{max_score} criteria met)**")
        
        return "\n".join(sections)
    
    @staticmethod
    def format_growth_metrics(data: Dict[str, Any]) -> str:
        """
        Format growth metrics data for LLM
        
        Args:
            data: Growth metrics data from GetGrowthMetricsTool
            
        Returns:
            Formatted text for LLM consumption
        """
        if not data:
            return "No growth metrics data available"
        
        sections = []
        symbol = data.get('symbol', 'N/A')
        period_type = data.get('period_type', 'annual')
        
        sections.append(f"📈 GROWTH METRICS - {symbol} ({period_type.upper()})")
        sections.append("═" * 60)
        
        latest = data.get('latest_period', {})
        
        if not latest:
            return "No growth metrics periods available"
        
        # ═══════════════════════════════════════════════════════════
        # Header Info
        # ═══════════════════════════════════════════════════════════
        sections.append(f"\n**Period:** {latest.get('date', 'N/A')}")
        sections.append(f"**Growth Period:** {period_type.title()} Year-over-Year")
        
        # ═══════════════════════════════════════════════════════════
        # REVENUE & PROFIT GROWTH
        # ═══════════════════════════════════════════════════════════
        revenue_growth = latest.get('revenue_growth', 0)
        gross_profit_growth = latest.get('gross_profit_growth', 0)
        operating_income_growth = latest.get('operating_income_growth', 0)
        ebitda_growth = latest.get('ebitda_growth', 0)
        net_income_growth = latest.get('net_income_growth', 0)
        
        sections.append(f"\n**💰 Revenue & Profit Growth:**")
        
        if revenue_growth is not None:
            emoji = "🚀" if revenue_growth > 0.20 else "📈" if revenue_growth > 0 else "📉"
            sections.append(f"- Revenue Growth: {emoji} {revenue_growth*100:+.1f}%")
        
        if gross_profit_growth is not None:
            emoji = "📈" if gross_profit_growth > 0 else "📉"
            sections.append(f"- Gross Profit Growth: {emoji} {gross_profit_growth*100:+.1f}%")
        
        if operating_income_growth is not None:
            emoji = "📈" if operating_income_growth > 0 else "📉"
            sections.append(f"- Operating Income Growth: {emoji} {operating_income_growth*100:+.1f}%")
        
        if ebitda_growth is not None:
            emoji = "📈" if ebitda_growth > 0 else "📉"
            sections.append(f"- EBITDA Growth: {emoji} {ebitda_growth*100:+.1f}%")
        
        if net_income_growth is not None:
            emoji = "🚀" if net_income_growth > 0.25 else "📈" if net_income_growth > 0 else "📉"
            sections.append(f"- Net Income Growth: {emoji} {net_income_growth*100:+.1f}%")
        
        # ═══════════════════════════════════════════════════════════
        # EPS GROWTH
        # ═══════════════════════════════════════════════════════════
        eps_growth = latest.get('eps_growth', 0)
        eps_diluted_growth = latest.get('eps_diluted_growth', 0)
        
        sections.append(f"\n**📊 Earnings Per Share Growth:**")
        if eps_growth is not None:
            emoji = "🚀" if eps_growth > 0.20 else "📈" if eps_growth > 0 else "📉"
            sections.append(f"- EPS Growth: {emoji} {eps_growth*100:+.1f}%")
        if eps_diluted_growth is not None:
            emoji = "📈" if eps_diluted_growth > 0 else "📉"
            sections.append(f"- EPS Diluted Growth: {emoji} {eps_diluted_growth*100:+.1f}%")
        
        # ═══════════════════════════════════════════════════════════
        # CASH FLOW GROWTH
        # ═══════════════════════════════════════════════════════════
        operating_cf_growth = latest.get('operating_cash_flow_growth', 0)
        fcf_growth = latest.get('free_cash_flow_growth', 0)
        
        sections.append(f"\n**💵 Cash Flow Growth:**")
        if operating_cf_growth is not None:
            emoji = "📈" if operating_cf_growth > 0 else "📉"
            sections.append(f"- Operating Cash Flow Growth: {emoji} {operating_cf_growth*100:+.1f}%")
        if fcf_growth is not None:
            emoji = "🚀" if fcf_growth > 0.25 else "📈" if fcf_growth > 0 else "📉"
            sections.append(f"- Free Cash Flow Growth: {emoji} {fcf_growth*100:+.1f}%")
        
        # ═══════════════════════════════════════════════════════════
        # BALANCE SHEET GROWTH
        # ═══════════════════════════════════════════════════════════
        assets_growth = latest.get('assets_growth', 0)
        equity_growth = latest.get('shareholders_equity_growth', 0)
        debt_growth = latest.get('debt_growth', 0)
        
        sections.append(f"\n**🏦 Balance Sheet Growth:**")
        if assets_growth is not None:
            sections.append(f"- Total Assets Growth: {assets_growth*100:+.1f}%")
        if equity_growth is not None:
            sections.append(f"- Shareholders' Equity Growth: {equity_growth*100:+.1f}%")
        if debt_growth is not None:
            emoji = "⚠️" if debt_growth > 0.20 else "○"
            sections.append(f"- Debt Growth: {emoji} {debt_growth*100:+.1f}%")
        
        # ═══════════════════════════════════════════════════════════
        # RETURN METRICS GROWTH
        # ═══════════════════════════════════════════════════════════
        roe_growth = latest.get('roe_growth', 0)
        roa_growth = latest.get('roa_growth', 0)
        
        if roe_growth or roa_growth:
            sections.append(f"\n**📈 Return Metrics Growth:**")
            if roe_growth is not None:
                sections.append(f"- ROE Growth: {roe_growth*100:+.1f}%")
            if roa_growth is not None:
                sections.append(f"- ROA Growth: {roa_growth*100:+.1f}%")
        
        # ═══════════════════════════════════════════════════════════
        # AVERAGE GROWTH RATES (Multi-Period)
        # ═══════════════════════════════════════════════════════════
        avg_growth = data.get('average_growth_rates', {})
        period_count = data.get('period_count', 0)
        
        if avg_growth and period_count > 1:
            sections.append(f"\n**📊 Average Growth Rates (last {period_count} periods):**")
            
            if avg_growth.get('revenue_growth') is not None:
                sections.append(f"- Avg Revenue Growth: {avg_growth['revenue_growth']*100:+.1f}%")
            if avg_growth.get('net_income_growth') is not None:
                sections.append(f"- Avg Net Income Growth: {avg_growth['net_income_growth']*100:+.1f}%")
            if avg_growth.get('eps_growth') is not None:
                sections.append(f"- Avg EPS Growth: {avg_growth['eps_growth']*100:+.1f}%")
            if avg_growth.get('free_cash_flow_growth') is not None:
                sections.append(f"- Avg FCF Growth: {avg_growth['free_cash_flow_growth']*100:+.1f}%")
        
        # ═══════════════════════════════════════════════════════════
        # GROWTH QUALITY ASSESSMENT
        # ═══════════════════════════════════════════════════════════
        sections.append(f"\n**🏆 Growth Quality Assessment:**")
        
        quality_signals = []
        warning_signals = []
        
        # Check revenue vs earnings growth alignment
        if revenue_growth and net_income_growth:
            if net_income_growth > revenue_growth:
                quality_signals.append("✓ Earnings growing faster than revenue (improving margins)")
            elif revenue_growth > 0 and net_income_growth < 0:
                warning_signals.append("⚠️ Revenue growing but earnings declining (margin compression)")
        
        # Check cash flow vs earnings
        if net_income_growth and fcf_growth:
            if fcf_growth > 0 and net_income_growth > 0:
                quality_signals.append("✓ Both earnings and cash flow growing (high-quality growth)")
            elif net_income_growth > 0 and fcf_growth < 0:
                warning_signals.append("⚠️ Earnings growing but FCF declining (quality concern)")
        
        # Check debt growth
        if debt_growth and revenue_growth:
            if debt_growth > revenue_growth and debt_growth > 0.15:
                warning_signals.append("⚠️ Debt growing faster than revenue")
        
        # Check R&D investment
        rd_growth = latest.get('rd_expenses_growth', 0)
        if rd_growth and rd_growth > 0:
            quality_signals.append(f"✓ Investing in R&D (growth: {rd_growth*100:+.1f}%)")
        
        # Display signals
        if quality_signals:
            sections.append("\n**Positive Signals:**")
            for signal in quality_signals:
                sections.append(f"  {signal}")
        
        if warning_signals:
            sections.append("\n**Warning Signals:**")
            for signal in warning_signals:
                sections.append(f"  {signal}")
        
        if not quality_signals and not warning_signals:
            sections.append("No significant quality signals detected in current period.")
        
        return "\n".join(sections)
    

    @staticmethod
    def format_stock_news(data: Dict[str, Any]) -> str:
        """Format stock news data for LLM"""
        
        if not data or not isinstance(data, dict):
            return "No news data available."
        
        symbol = data.get('symbol', 'N/A')
        article_count = data.get('article_count', 0)
        articles = data.get('articles', [])
        
        if article_count == 0:
            return f"No recent news found for {symbol}."
        
        lines = [
            f"📰 **LATEST NEWS FOR {symbol}**",
            f"Found {article_count} recent articles:",
            ""
        ]
        
        for i, article in enumerate(articles[:10], 1):
            title = article.get('title', 'Untitled')
            date = article.get('published_date', '')
            site = article.get('site', 'Unknown')
            text = article.get('text', '')
            url = article.get('url', '')
            
            lines.append(f"**Article {i}:**")
            lines.append(f"📌 {title}")
            lines.append(f"🗓️  {date} | 📰 {site}")
            
            if text:
                snippet = text[:200] + '...' if len(text) > 200 else text
                lines.append(f"📝 {snippet}")
            
            if url:
                lines.append(f"🔗 [Read more]({url})")
            
            lines.append("")
        
        return "\n".join(lines)


    @staticmethod
    def format_earnings_calendar(data: Dict[str, Any]) -> str:
        """Format earnings calendar data for LLM"""
        
        if not data or not isinstance(data, dict):
            return "No earnings data available."
        
        symbol = data.get('symbol', 'N/A')
        report_count = data.get('report_count', 0)
        reports = data.get('reports', [])
        summary = data.get('summary', {})
        
        if report_count == 0:
            return f"No earnings history found for {symbol}."
        
        lines = [
            f"📊 **EARNINGS HISTORY FOR {symbol}**",
            ""
        ]
        
        # Summary statistics
        if summary:
            lines.append("**📈 Performance Summary:**")
            
            eps_beat_rate = summary.get('eps_beat_rate')
            if eps_beat_rate is not None:
                lines.append(f"  - EPS Beat Rate: {eps_beat_rate:.0%}")
            
            avg_eps_surprise = summary.get('avg_eps_surprise_pct')
            if avg_eps_surprise is not None:
                surprise_emoji = "🚀" if avg_eps_surprise > 5 else "✅" if avg_eps_surprise > 0 else "⚠️"
                lines.append(f"  - Avg EPS Surprise: {surprise_emoji} {avg_eps_surprise:+.2f}%")
            
            revenue_beat_rate = summary.get('revenue_beat_rate')
            if revenue_beat_rate is not None:
                lines.append(f"  - Revenue Beat Rate: {revenue_beat_rate:.0%}")
            
            lines.append("")
        
        # Recent reports
        lines.append("**Last 5 Earnings Reports:**")
        lines.append("")
        
        for i, report in enumerate(reports[:5], 1):
            date = report.get('date', 'N/A')
            eps_actual = report.get('eps_actual')
            eps_estimated = report.get('eps_estimated')
            eps_surprise_pct = report.get('eps_surprise_pct')
            revenue_actual = report.get('revenue_actual')
            revenue_estimated = report.get('revenue_estimated')
            
            lines.append(f"**Report {i} ({date}):**")
            
            # EPS
            if eps_actual is not None and eps_estimated is not None:
                beat_emoji = "✅" if eps_surprise_pct and eps_surprise_pct > 0 else "❌"
                lines.append(f"  - EPS: ${eps_actual:.2f} vs ${eps_estimated:.2f} est. {beat_emoji}")
                
                if eps_surprise_pct is not None:
                    surprise_emoji = "🚀" if eps_surprise_pct > 10 else "⚠️" if eps_surprise_pct < -10 else ""
                    lines.append(f"    Surprise: {surprise_emoji} {eps_surprise_pct:+.2f}%")
            
            # Revenue
            if revenue_actual is not None and revenue_estimated is not None:
                rev_actual_b = revenue_actual / 1_000_000_000
                rev_est_b = revenue_estimated / 1_000_000_000
                lines.append(f"  - Revenue: ${rev_actual_b:.2f}B vs ${rev_est_b:.2f}B est.")
            
            lines.append("")
        
        # Key insights
        lines.append("**🔍 Key Insights:**")
        if eps_beat_rate and eps_beat_rate >= 0.75:
            lines.append("  ✅ Strong earnings consistency - beats estimates 75%+ of the time")
        elif eps_beat_rate and eps_beat_rate >= 0.5:
            lines.append("  ○ Mixed earnings performance")
        elif eps_beat_rate is not None:
            lines.append("  ⚠️  Often misses earnings estimates")
        
        if avg_eps_surprise and avg_eps_surprise > 5:
            lines.append("  🚀 Tends to significantly beat expectations")
        elif avg_eps_surprise and avg_eps_surprise < -5:
            lines.append("  ⚠️  Tends to significantly miss expectations")
        
        return "\n".join(lines)


    @staticmethod
    def format_company_events(data: Dict[str, Any]) -> str:
        """Format company events (dividends) data for LLM"""
        
        if not data or not isinstance(data, dict):
            return "No company events data available."
        
        symbol = data.get('symbol', 'N/A')
        event_count = data.get('event_count', 0)
        events = data.get('events', [])
        summary = data.get('summary', {})
        
        if event_count == 0:
            return f"No dividend events found for {symbol} in the specified period."
        
        lines = [
            f"💰 **DIVIDEND HISTORY FOR {symbol}**",
            ""
        ]
        
        # Summary statistics
        if summary:
            lines.append("**📊 Dividend Summary:**")
            
            total_dividends = summary.get('total_dividends')
            if total_dividends is not None:
                lines.append(f"  - Total Paid: ${total_dividends:.4f}")
            
            avg_dividend = summary.get('average_dividend')
            if avg_dividend is not None:
                lines.append(f"  - Average Payment: ${avg_dividend:.4f}")
            
            payment_count = summary.get('payment_count')
            if payment_count:
                lines.append(f"  - Number of Payments: {payment_count}")
            
            frequency = summary.get('frequency', 'Unknown')
            lines.append(f"  - Payment Frequency: {frequency}")
            
            most_recent_yield = summary.get('most_recent_yield')
            if most_recent_yield is not None:
                lines.append(f"  - Current Yield: {most_recent_yield:.2f}%")
            
            lines.append("")
        
        # Recent events
        lines.append("**📅 Recent Dividend Events:**")
        lines.append("")
        
        for i, event in enumerate(events[:10], 1):
            ex_date = event.get('ex_dividend_date', 'N/A')
            payment_date = event.get('payment_date', 'N/A')
            dividend = event.get('dividend')
            div_yield = event.get('yield')
            
            lines.append(f"**Event {i}:**")
            lines.append(f"  - Ex-Date: {ex_date}")
            
            if dividend is not None:
                lines.append(f"  - Dividend: ${dividend:.4f}")
            
            if div_yield is not None:
                lines.append(f"  - Yield: {div_yield:.2f}%")
            
            if payment_date and payment_date != 'N/A':
                lines.append(f"  - Payment Date: {payment_date}")
            
            lines.append("")
        
        # Analysis
        lines.append("**🔍 Dividend Analysis:**")
        
        if payment_count:
            if payment_count >= 4:
                lines.append("  ✅ Consistent dividend payer")
            elif payment_count >= 2:
                lines.append("  ○ Regular dividend payer")
            else:
                lines.append("  ⚠️  Infrequent dividend payments")
        
        if frequency and frequency != 'Unknown':
            lines.append(f"  - Pays dividends: {frequency}")
        
        return "\n".join(lines)

   
    @staticmethod
    def format_market_indices(data: Dict) -> str:
        """Format market indices data"""
        indices = data.get("indices", [])
        major_indices = data.get("major_indices", {})
        
        lines = ["📊 **MARKET INDICES**\n"]
        
        # Major indices first
        lines.append("**Major Indices:**")
        for symbol, info in major_indices.items():
            # Handle potential missing/None data gracefully
            if not isinstance(info, dict):
                continue

            name = info.get("name", symbol)
            
            # 🛡️ FIX: Ensure values are numbers (handle NoneType)
            price = info.get("price")
            if price is None: 
                price = 0.0
                
            change_pct = info.get("changes_percentage")
            if change_pct is None: 
                change_pct = 0.0
            
            emoji = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"
            
            lines.append(f"{emoji} {name}: {price:,.2f} ({change_pct:+.2f}%)")
        
        lines.append(f"\n📈 Total indices tracked: {data.get('index_count', 0)}")
        
        return "\n".join(lines)

    @staticmethod
    def format_sector_performance(data: Dict) -> str:
        """Format sector performance data"""
        sectors = data.get("sectors", [])
        summary = data.get("summary", {})
        
        lines = ["🎯 **SECTOR PERFORMANCE**\n"]
        
        lines.append(f"**Best Sector:** {data.get('best_sector', 'N/A')}")
        lines.append(f"**Worst Sector:** {data.get('worst_sector', 'N/A')}")
        lines.append(f"**Market Sentiment:** {summary.get('market_sentiment', 'neutral').upper()}\n")
        
        lines.append("**Sectors:**")
        for sector in sectors:
            name = sector.get("sector", "Unknown")
            change = sector.get("changePercent", 0)
            emoji = "🟢" if change > 0 else "🔴"
            lines.append(f"{emoji} {name}: {change:+.2f}%")
        
        return "\n".join(lines)

    @staticmethod
    def format_market_movers(data: Dict) -> str:
        """Format market movers data"""
        mover_type = data.get("mover_type", "")
        stocks = data.get("stocks", [])[:10]  # Top 10
        
        title_map = {
            "gainers": "📈 TOP GAINERS",
            "losers": "📉 TOP LOSERS",
            "actives": "🔥 MOST ACTIVE"
        }
        
        lines = [f"**{title_map.get(mover_type, 'MARKET MOVERS')}**\n"]
        
        for i, stock in enumerate(stocks, 1):
            symbol = stock.get("symbol", "")
            price = stock.get("price", 0)
            change_pct = stock.get("changesPercentage", 0)
            volume = stock.get("volume", 0)
            
            lines.append(
                f"{i}. **{symbol}**: ${price:.2f} ({change_pct:+.2f}%) "
                f"| Vol: {volume:,.0f}"
            )
        
        return "\n".join(lines)

    @staticmethod
    def format_market_breadth(data: Dict) -> str:
        """Format market breadth data"""
        breadth = data.get("breadth", {})
        
        lines = ["📊 **MARKET BREADTH**\n"]
        
        advancers = breadth.get("advancers", 0)
        decliners = breadth.get("decliners", 0)
        ratio = breadth.get("advance_decline_ratio", 0)
        sentiment = breadth.get("market_sentiment", "neutral")
        
        sentiment_emoji = {
            "bullish": "🟢",
            "bearish": "🔴",
            "neutral": "⚪"
        }
        
        lines.append(f"**Advancers:** {advancers} ↑")
        lines.append(f"**Decliners:** {decliners} ↓")
        lines.append(f"**A/D Ratio:** {ratio:.2f}")
        lines.append(f"**Sentiment:** {sentiment_emoji.get(sentiment, '⚪')} {sentiment.upper()}")
        lines.append(f"\n📈 {breadth.get('percent_advancers', 0):.1f}% sectors advancing")
        
        return "\n".join(lines)

    @staticmethod
    def format_stock_heatmap(data: Dict) -> str:
        """Format stock heatmap data"""
        cells = data.get("cells", [])
        
        lines = ["🗺️ **MARKET HEATMAP**\n"]
        
        lines.append(f"**Range:** {data.get('min_change', 0):.2f}% to {data.get('max_change', 0):.2f}%\n")
        
        lines.append("**Sectors:**")
        for cell in cells[:10]:  # Top 10
            name = cell.get("name", "")
            change = cell.get("change_percent", 0)
            emoji = "🟢" if change > 2 else "🟡" if change > -2 else "🔴"
            lines.append(f"{emoji} {name}: {change:+.2f}%")
        
        return "\n".join(lines)

    @staticmethod
    def format_crypto_price(data: Dict) -> str:
        """Format crypto price data"""
        symbol = data.get("symbol", "")
        name = data.get("name", "")
        price = data.get("price", 0)
        change_pct = data.get("changes_percentage", 0)
        volume = data.get("volume", 0)
        market_cap = data.get("market_cap")
        
        emoji = "🟢" if change_pct > 0 else "🔴"
        
        lines = [f"₿ **{name} ({symbol})**\n"]
        lines.append(f"**Price:** ${price:,.2f} {emoji} {change_pct:+.2f}%")
        lines.append(f"**24h High:** ${data.get('day_high', 0):,.2f}")
        lines.append(f"**24h Low:** ${data.get('day_low', 0):,.2f}")
        lines.append(f"**Volume:** ${volume:,.0f}")
        
        if market_cap:
            lines.append(f"**Market Cap:** ${market_cap:,.0f}")
        
        return "\n".join(lines)

    @staticmethod
    def format_crypto_technicals(data: Dict) -> str:
        """Format crypto technical indicators"""
        symbol = data.get("symbol", "")
        timeframe = data.get("timeframe", "")
        indicators = data.get("indicators", {})
        
        lines = [f"📊 **{symbol} TECHNICAL ANALYSIS ({timeframe})**\n"]
        
        # RSI
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            if rsi >= 70:
                status = "🔴 OVERBOUGHT"
            elif rsi <= 30:
                status = "🟢 OVERSOLD"
            else:
                status = "⚪ NEUTRAL"
            lines.append(f"**RSI(14):** {rsi:.2f} {status}")
        
        # MACD
        if "macd" in indicators:
            macd = indicators["macd"]
            macd_val = macd.get("macd", 0)
            signal = macd.get("signal", 0)
            hist = macd.get("histogram", 0)
            trend = "🟢 BULLISH" if hist > 0 else "🔴 BEARISH"
            lines.append(f"**MACD:** {macd_val:.4f} {trend}")
        
        # Moving Averages
        if "sma_20" in indicators:
            lines.append(f"**SMA(20):** ${indicators['sma_20']:,.2f}")
        if "sma_50" in indicators:
            lines.append(f"**SMA(50):** ${indicators['sma_50']:,.2f}")
        
        lines.append(f"\n📈 Current Price: ${data.get('current_price', 0):,.2f}")
        
        return "\n".join(lines)
    

    @staticmethod
    def format_screener_results(screener_data: Dict[str, Any]) -> str:
        """
        Format stock screener results
        
        Args:
            screener_data: Dict containing screened stocks
            
        Returns:
            Formatted text
        """
        if not screener_data:
            return "No screener results available"
        
        sections = []
        
        # Header
        count = screener_data.get('count', 0)
        sections.append(f"🔍 STOCK SCREENER RESULTS - {count} stocks found")
        sections.append("═" * 60)
        
        # Criteria used
        criteria = screener_data.get('criteria', {})
        if criteria:
            sections.append("\n**Screening Criteria:**")
            for key, value in criteria.items():
                if value is not None:
                    sections.append(f"  • {key}: {value}")
        
        # Top symbols
        top_symbols = screener_data.get('top_symbols', [])
        if top_symbols:
            sections.append(f"\n**Top Symbols:** {', '.join(top_symbols[:10])}")
        
        # Detailed stocks
        stocks = screener_data.get('stocks', [])
        if stocks:
            sections.append(f"\n**Detailed Results (showing {min(len(stocks), 10)} of {count}):**\n")
            
            for idx, stock in enumerate(stocks[:10], 1):
                sections.append(f"{idx}. **{stock.get('symbol', 'N/A')}** - {stock.get('name', 'N/A')}")
                
                sector = stock.get('sector')
                industry = stock.get('industry')
                if sector:
                    sections.append(f"   Sector: {sector}")
                if industry:
                    sections.append(f"   Industry: {industry}")
                
                market_cap = stock.get('market_cap')
                if market_cap:
                    sections.append(f"   Market Cap: ${market_cap:,.0f}")
                
                price = stock.get('price')
                if price:
                    sections.append(f"   Price: ${price:,.2f}")
                
                sections.append("")
        
        # Summary
        summary = screener_data.get('summary', '')
        if summary:
            sections.append(f"\n**Summary:** {summary}")
        
        return "\n".join(sections)


    @staticmethod
    def format_chart_patterns_from_tool(pattern_data: Dict[str, Any]) -> str:
        """
        Format chart patterns from detectChartPatterns tool output
        
        ✅ NEW METHOD: Handles full tool output structure with bullish/bearish breakdown
        
        Tool output structure:
        {
            'symbol': 'NVDA',
            'patterns_detected': [16 patterns],
            'total_patterns': 16,
            'bullish_patterns': [9 patterns],
            'bearish_patterns': [7 patterns],
            'highest_confidence_pattern': {...},
            'summary': 'Pattern analysis completed',
            'timestamp': '2025-12-03...'
        }
        
        Args:
            pattern_data: Dict from detectChartPatterns tool
            
        Returns:
            Formatted text for LLM with complete pattern analysis
        """
        
        if not pattern_data:
            return "No chart pattern data available"
        
        # Extract data
        symbol = pattern_data.get('symbol', 'N/A')
        patterns_detected = pattern_data.get('patterns_detected', [])
        bullish = pattern_data.get('bullish_patterns', [])
        bearish = pattern_data.get('bearish_patterns', [])
        total = pattern_data.get('total_patterns', 0)
        highest_conf = pattern_data.get('highest_confidence_pattern', {})
        timestamp = pattern_data.get('timestamp', '')
        
        if not patterns_detected:
            return f"📊 CHART PATTERNS - {symbol}\\n\\nNo significant patterns detected in recent price action."
        
        # Build formatted output
        lines = [
            f"📊 CHART PATTERNS - {symbol}",
            "═" * 60,
            "",
            f"**Total Patterns Detected**: {total} ({len(bullish)} Bullish, {len(bearish)} Bearish)",
            ""
        ]
        
        # ================================================================
        # Bullish Patterns (show top 5 + count)
        # ================================================================
        if bullish:
            lines.append("🟢 **BULLISH PATTERNS**:")
            
            # Show top 5 most recent
            for i, pattern in enumerate(bullish[:5], 1):
                ptype = pattern.get('type', 'Unknown')
                start = pattern.get('start_date', '')[:10]  # YYYY-MM-DD
                end = pattern.get('end_date', '')[:10]
                price = pattern.get('price_level', 0)
                conf = pattern.get('confidence', 'Low')
                
                lines.append(
                    f"  {i}. {ptype} @ ${price:.2f} "
                    f"({start} → {end}) - {conf} confidence"
                )
            
            # Show count of remaining
            if len(bullish) > 5:
                lines.append(f"  ... and {len(bullish) - 5} more bullish patterns")
            
            lines.append("")
        
        # ================================================================
        # Bearish Patterns (show top 5 + count)
        # ================================================================
        if bearish:
            lines.append("🔴 **BEARISH PATTERNS**:")
            
            # Show top 5 most recent
            for i, pattern in enumerate(bearish[:5], 1):
                ptype = pattern.get('type', 'Unknown')
                start = pattern.get('start_date', '')[:10]
                end = pattern.get('end_date', '')[:10]
                price = pattern.get('price_level', 0)
                conf = pattern.get('confidence', 'Low')
                
                lines.append(
                    f"  {i}. {ptype} @ ${price:.2f} "
                    f"({start} → {end}) - {conf} confidence"
                )
            
            # Show count of remaining
            if len(bearish) > 5:
                lines.append(f"  ... and {len(bearish) - 5} more bearish patterns")
            
            lines.append("")
        
        # ================================================================
        # Highest Confidence Pattern (Key Pattern)
        # ================================================================
        if highest_conf:
            lines.append("⭐ **KEY PATTERN** (Highest Confidence):")
            hc_type = highest_conf.get('type', 'Unknown')
            hc_price = highest_conf.get('price_level', 0)
            hc_conf = highest_conf.get('confidence', 'Unknown')
            hc_start = highest_conf.get('start_date', '')[:10]
            hc_end = highest_conf.get('end_date', '')[:10]
            
            lines.append(
                f"  {hc_type} @ ${hc_price:.2f} "
                f"({hc_start} → {hc_end}) - {hc_conf} confidence"
            )
            lines.append("")
        
        # ================================================================
        # Pattern Interpretation (Synthesis)
        # ================================================================
        lines.append("💡 **INTERPRETATION**:")
        
        if bullish and bearish:
            ratio = len(bullish) / len(bearish) if bearish else float('inf')
            
            if ratio > 1.5:
                lines.append(
                    f"  Bullish patterns dominate ({len(bullish)} vs {len(bearish)}). "
                    "Multiple support formations suggest accumulation phase and potential upside."
                )
            elif ratio < 0.67:
                lines.append(
                    f"  Bearish patterns dominate ({len(bearish)} vs {len(bullish)}). "
                    "Multiple resistance formations suggest distribution phase and potential downside pressure."
                )
            else:
                lines.append(
                    f"  Mixed signals with {len(bullish)} bullish and {len(bearish)} bearish patterns. "
                    "Market appears to be in consolidation or indecision phase. "
                    "Wait for clearer directional bias before taking positions."
                )
        
        elif bullish:
            lines.append(
                f"  Strong bullish pattern dominance with {len(bullish)} formations detected. "
                "Multiple Double Bottom patterns suggest strong support building. "
                "Breakout above resistance could confirm uptrend."
            )
        
        elif bearish:
            lines.append(
                f"  Bearish patterns dominate with {len(bearish)} formations detected. "
                "Multiple Double Top patterns indicate significant resistance overhead. "
                "Breakdown below support would confirm downtrend."
            )
        
        # ================================================================
        # Trading Implications (Actionable)
        # ================================================================
        lines.append("")
        lines.append("📈 **TRADING IMPLICATIONS**:")
        
        if bullish and not bearish:
            lines.append("  • Watch for breakout above resistance levels")
            lines.append("  • Entry: On volume-confirmed breakout")
            lines.append("  • Target: Measured move from pattern height")
            lines.append("  • Stop: Below pattern support")
        
        elif bearish and not bullish:
            lines.append("  • Watch for breakdown below support levels")
            lines.append("  • Consider taking profits or reducing exposure")
            lines.append("  • Wait for reversal signals before re-entry")
        
        else:  # Mixed
            lines.append("  • Range-bound trading environment")
            lines.append("  • Trade between support and resistance")
            lines.append("  • Wait for breakout/breakdown confirmation")
            lines.append("  • Reduce position size due to uncertainty")
        
        # ================================================================
        # Metadata
        # ================================================================
        lines.append("")
        lines.append(f"Analysis Period: Last 90 days")
        if timestamp:
            lines.append(f"Timestamp: {timestamp}")
        
        return "\\n".join(lines)
    

    @staticmethod
    def _format_generic_data(tool_name: str, data: Dict[str, Any]) -> str:
        """
        Fallback formatter for tools without specific format method
        
        Provides clean JSON representation with tool context
        """
        import json
        
        # Extract key information
        symbol = data.get('symbol', data.get('symbols', 'N/A'))
        timestamp = data.get('timestamp', '')
        
        sections = []
        sections.append(f"📊 {tool_name.upper()}")
        sections.append("═" * 60)
        
        if symbol and symbol != 'N/A':
            sections.append(f"Symbol: {symbol}")
        
        if timestamp:
            sections.append(f"Timestamp: {timestamp}")
        
        sections.append("\nData:")
        
        # Format as clean JSON
        try:
            # Exclude metadata keys
            exclude_keys = {'status', 'error', 'tool_name', 'timestamp', 'symbols', 'symbol'}
            clean_data = {k: v for k, v in data.items() if k not in exclude_keys}
            
            json_str = json.dumps(clean_data, indent=2, ensure_ascii=False, default=str)
            sections.append(f"```json\n{json_str}\n```")
        except Exception as e:
            sections.append(f"[Error formatting data: {e}]")
            sections.append(str(data))
        
        return "\n".join(sections)
    

    @staticmethod
    def format_by_tool_name(tool_name: str, data: Dict[str, Any]) -> str:
        """
        🎯 MASTER ROUTER: Route tool output to appropriate formatter
        
        COMPREHENSIVE MAPPING for all 27 atomic tools → 29 formatters
        
        This is the SINGLE ENTRY POINT for all tool result formatting.
        Called by tool_execution_service._format_result_with_context()
        
        Args:
            tool_name: Name of the tool that generated the data
            data: Tool output data (Dict)
            
        Returns:
            Formatted string ready for LLM context
            
        Error Handling:
            - Unknown tool → _format_generic_data (JSON fallback)
            - Formatter exception → Returns raw data + error message
            - Never crashes, always returns a string
        """
        
        # ════════════════════════════════════════════════════════════════
        # TOOL → FORMATTER MAPPING
        # ════════════════════════════════════════════════════════════════
        
        FORMATTER_MAP = {
            # ═══ PRICE & PERFORMANCE (3 tools) ═══
            'getStockPrice': FinancialDataFormatter.format_price_data,
            'getStockPerformance': FinancialDataFormatter.format_price_data,
            'getPriceTargets': FinancialDataFormatter.format_price_data,
            
            # ═══ TECHNICAL ANALYSIS (4 tools) ═══
            'getTechnicalIndicators': FinancialDataFormatter.format_technical_indicators_data,
            'detectChartPatterns': FinancialDataFormatter.format_chart_patterns_from_tool,  # ✅ NEW!
            'getRelativeStrength': FinancialDataFormatter.format_relative_strength,
            'getSupportResistance': FinancialDataFormatter.format_support_resistance,  # ✅ NEW!
            
            # ═══ RISK & DECISION SUPPORT (3 tools) ═══
            'assessRisk': FinancialDataFormatter.format_risk_assessment_data,
            'getVolumeProfile': FinancialDataFormatter.format_volume_profile,
            'suggestStopLoss': FinancialDataFormatter.format_stop_loss_data,  # ✅ NEW!
            
            # ═══ FUNDAMENTALS (5 tools) ═══
            'getIncomeStatement': FinancialDataFormatter.format_income_statement,  # ✅ NEW!
            'getBalanceSheet': FinancialDataFormatter.format_balance_sheet,  # ✅ NEW!
            'getCashFlow': FinancialDataFormatter.format_cash_flow,  # ✅ NEW!
            'getFinancialRatios': FinancialDataFormatter.format_financial_ratios,  # ✅ NEW!
            'getGrowthMetrics': FinancialDataFormatter.format_growth_metrics,  # ✅ NEW!
            
            # ═══ NEWS & EVENTS (4 tools) ═══
            'getStockNews': FinancialDataFormatter.format_stock_news,  # ✅ NEW!
            'getSentiment': FinancialDataFormatter.format_sentiment_data,  # ✅ NEW!
            'getEarningsCalendar': FinancialDataFormatter.format_earnings_calendar,  # ✅ NEW!
            'getCompanyEvents': FinancialDataFormatter.format_company_events,  # ✅ NEW!
            
            # ═══ MARKET DATA (6 tools) ═══
            'getMarketIndices': FinancialDataFormatter.format_market_indices,  # ✅ NEW!
            'getSectorPerformance': FinancialDataFormatter.format_sector_performance,  # ✅ NEW!
            'getMarketMovers': FinancialDataFormatter.format_market_movers,  # ✅ NEW!
            'getMarketBreadth': FinancialDataFormatter.format_market_breadth,  # ✅ NEW!
            'getStockHeatmap': FinancialDataFormatter.format_stock_heatmap,  # ✅ NEW!
            'getMarketNews': FinancialDataFormatter.format_news_data,  # Reuse existing
            
            # ═══ SCREENING (1 tool) ═══
            'stockScreener': FinancialDataFormatter.format_screener_results,  # ✅ NEW!
            
            # ═══ CRYPTO (2 tools) ═══
            'getCryptoPrice': FinancialDataFormatter.format_crypto_price,  # ✅ NEW!
            'getCryptoTechnicals': FinancialDataFormatter.format_crypto_technicals,  # ✅ NEW!
        }
        
        # ════════════════════════════════════════════════════════════════
        # ROUTER LOGIC
        # ════════════════════════════════════════════════════════════════
        
        # Get formatter for tool
        formatter = FORMATTER_MAP.get(tool_name)
        
        if not formatter:
            # Fallback for unrecognized tools
            return FinancialDataFormatter._format_generic_data(data, tool_name)
        
        try:
            # Execute formatter
            return formatter(data)
            
        except Exception as e:
            return (
                f"⚠️ Formatting Error: {tool_name}\n"
                f"{'═' * 60}\n\n"
                f"**Error**: {str(e)}\n\n"
                f"**Raw Data**:\n"
                f"{json.dumps(data, indent=2, ensure_ascii=False, default=str)}"
            )
