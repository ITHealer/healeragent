"""
Context Preprocessing Layer - Extract Insights Before LLM Injection

File: src/helpers/analysis_insights_extractor.py
"""

from typing import Dict, Any, List
import logging


class AnalysisInsightsExtractor:
    """
    Extract key insights from tool results
    
    PURPOSE: Preprocess raw tool data → structured insights for LLM
    
    BENEFITS:
    - Reduce context size
    - Highlight important signals
    - Provide hierarchical information
    - Enable better LLM reasoning
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_insights(self, all_tool_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract insights from multiple tool results
        
        Args:
            all_tool_results: Dict of {task_id: {tool_name: result}}
            
        Returns:
            Structured insights dict
        """
        insights = {
            "symbol": None,
            "key_metrics": {},
            "signals": [],
            "patterns_detected": [],
            "risk_levels": {},
            "recommendations": []
        }
        
        # Extract from each tool type
        for task_id, task_results in all_tool_results.items():
            if not isinstance(task_results, dict):
                continue
            
            for tool_name, tool_result in task_results.items():
                if not isinstance(tool_result, dict):
                    continue
                
                # ✅ FIXED: Extract symbols from metadata
                symbols = []
                
                # Try metadata first (new atomic tools)
                if metadata := tool_result.get('metadata'):
                    symbols = metadata.get('symbols', [])
                
                # Fallback to direct symbols key (legacy tools)
                if not symbols:
                    symbols = tool_result.get('symbols', [])
                
                if symbols and isinstance(symbols, str):
                    symbols = [symbols]
                
                # Set symbol in insights
                if symbols and not insights["symbol"]:
                    insights["symbol"] = symbols[0]
                
                # Extract based on tool type
                if tool_name in ['getStockPrice', 'showStockPrice']:
                    self._extract_price_insights(tool_result, insights)
                
                elif tool_name in ['getTechnicalIndicators', 'showStockChart']:
                    self._extract_technical_insights(tool_result, insights)
                
                elif tool_name == 'assessRisk':
                    self._extract_risk_insights(tool_result, insights)
        
        return insights
    
    def _extract_price_insights(self, tool_result: Dict, insights: Dict):
        """Extract key insights from price data"""
        data = tool_result.get('data') or tool_result.get('raw_data', {})
        
        if not data:
            return
        
        # Set symbol
        if not insights["symbol"]:
            insights["symbol"] = data.get('symbol')
        
        # Key metrics
        price = data.get('price') or data.get('current_price')
        if price:
            insights["key_metrics"]["current_price"] = price
        
        change_pct = data.get('change_percent')
        if change_pct:
            insights["key_metrics"]["change_percent"] = change_pct
            
            # Generate signal
            if change_pct > 5:
                insights["signals"].append({
                    "type": "price_movement",
                    "severity": "high",
                    "message": f"Strong upward movement: +{change_pct:.1f}%"
                })
            elif change_pct < -5:
                insights["signals"].append({
                    "type": "price_movement",
                    "severity": "high",
                    "message": f"Sharp decline: {change_pct:.1f}%"
                })
    
    def _extract_technical_insights(self, tool_result: Dict, insights: Dict):
        """Extract key insights from technical indicators"""
        data = tool_result.get('data') or tool_result.get('raw_data', {})
        
        if not data:
            return
        
        # RSI insights
        rsi = data.get('rsi_14')
        if rsi is not None:
            insights["key_metrics"]["rsi"] = rsi
            
            if rsi > 70:
                insights["signals"].append({
                    "type": "momentum",
                    "severity": "medium",
                    "message": f"RSI overbought at {rsi:.1f} - potential reversal"
                })
            elif rsi < 30:
                insights["signals"].append({
                    "type": "momentum",
                    "severity": "medium",
                    "message": f"RSI oversold at {rsi:.1f} - potential bounce"
                })
        
        # MACD insights
        macd_signal = data.get('macd_signal_text')
        if macd_signal:
            insights["key_metrics"]["macd_trend"] = macd_signal
            
            if macd_signal == "Bullish":
                insights["signals"].append({
                    "type": "trend",
                    "severity": "medium",
                    "message": "MACD bullish crossover - uptrend momentum"
                })
        
        # Bollinger Patterns insights
        bb_patterns = data.get('bollinger_patterns', {})
        
        if w_bottom := bb_patterns.get('w_bottom'):
            if w_bottom.get('detected'):
                insights["patterns_detected"].append({
                    "pattern": "W-Bottom",
                    "confidence": w_bottom.get('confidence', 0),
                    "signal": w_bottom.get('signal'),
                    "description": w_bottom.get('description')
                })
                
                insights["signals"].append({
                    "type": "pattern",
                    "severity": "high",
                    "message": f"W-Bottom pattern detected - strong BUY signal (confidence: {w_bottom.get('confidence', 0)*100:.0f}%)"
                })
        
        if m_top := bb_patterns.get('m_top'):
            if m_top.get('detected'):
                insights["patterns_detected"].append({
                    "pattern": "M-Top",
                    "confidence": m_top.get('confidence', 0),
                    "signal": m_top.get('signal'),
                    "description": m_top.get('description')
                })
                
                insights["signals"].append({
                    "type": "pattern",
                    "severity": "high",
                    "message": f"M-Top pattern detected - strong SELL signal (confidence: {m_top.get('confidence', 0)*100:.0f}%)"
                })
        
        if squeeze := bb_patterns.get('squeeze_breakout'):
            if squeeze.get('detected'):
                direction = squeeze.get('direction', 'unknown')
                insights["patterns_detected"].append({
                    "pattern": "Bollinger Squeeze Breakout",
                    "direction": direction,
                    "description": squeeze.get('description')
                })
                
                insights["signals"].append({
                    "type": "volatility",
                    "severity": "high",
                    "message": f"Bollinger squeeze breakout - {direction} momentum"
                })
    
    def _extract_risk_insights(self, tool_result: Dict, insights: Dict):
        """Extract key insights from risk analysis"""
        data = tool_result.get('data') or tool_result.get('raw_data', {})
        
        if not data:
            return
        
        # Stop loss levels
        stop_levels = data.get('stop_levels', {})
        if stop_levels:
            insights["risk_levels"]["stop_loss_conservative"] = stop_levels.get('atr_based', {}).get('conservative_1x')
            insights["risk_levels"]["stop_loss_moderate"] = stop_levels.get('atr_based', {}).get('moderate_2x')
            insights["risk_levels"]["stop_loss_aggressive"] = stop_levels.get('atr_based', {}).get('aggressive_3x')
        
        # Recommendation
        if recommendation := data.get('recommendation'):
            rec_text = recommendation.get('rationale', '')
            if rec_text:
                insights["recommendations"].append({
                    "type": "risk_management",
                    "message": rec_text
                })
    
    def format_insights_for_llm(self, insights: Dict[str, Any]) -> str:
        """
        Format extracted insights for LLM context
        
        Args:
            insights: Extracted insights dict
            
        Returns:
            Formatted string for LLM
        """
        sections = []
        
        # Header
        symbol = insights.get("symbol", "N/A")
        sections.append(f"{'='*70}")
        sections.append(f"KEY INSIGHTS SUMMARY - {symbol}")
        sections.append(f"{'='*70}")
        
        # Key Metrics
        if metrics := insights.get("key_metrics"):
            sections.append("\n📊 KEY METRICS:")
            
            if price := metrics.get("current_price"):
                sections.append(f"- Current Price: ${price:,.2f}")
            
            if change := metrics.get("change_percent"):
                emoji = "📈" if change >= 0 else "📉"
                sections.append(f"- Change: {emoji} {change:+.2f}%")
            
            if rsi := metrics.get("rsi"):
                sections.append(f"- RSI: {rsi:.1f}")
            
            if macd := metrics.get("macd_trend"):
                sections.append(f"- MACD Trend: {macd}")
        
        # Signals
        if signals := insights.get("signals"):
            sections.append("\n🚨 CRITICAL SIGNALS:")
            
            # Sort by severity
            high_signals = [s for s in signals if s.get("severity") == "high"]
            medium_signals = [s for s in signals if s.get("severity") == "medium"]
            
            for signal in high_signals:
                sections.append(f"- [HIGH] {signal.get('message')}")
            
            for signal in medium_signals:
                sections.append(f"- [MEDIUM] {signal.get('message')}")
        
        # Patterns
        if patterns := insights.get("patterns_detected"):
            sections.append("\n🎯 PATTERNS DETECTED:")
            
            for pattern in patterns:
                pattern_name = pattern.get("pattern")
                confidence = pattern.get("confidence")
                signal = pattern.get("signal")
                
                if confidence:
                    sections.append(
                        f"- {pattern_name}: {signal} "
                        f"(Confidence: {confidence*100:.0f}%)"
                    )
                else:
                    sections.append(f"- {pattern_name}: {signal}")
        
        # Risk Levels
        if risk := insights.get("risk_levels"):
            sections.append("\n⚠️ RISK MANAGEMENT:")
            
            if stop_cons := risk.get("stop_loss_conservative"):
                sections.append(f"- Conservative Stop Loss: ${stop_cons:,.2f}")
            
            if stop_mod := risk.get("stop_loss_moderate"):
                sections.append(f"- Moderate Stop Loss: ${stop_mod:,.2f}")
        
        # Recommendations
        if recs := insights.get("recommendations"):
            sections.append("\n💡 RECOMMENDATIONS:")
            
            for rec in recs:
                sections.append(f"- {rec.get('message')}")
        
        sections.append(f"{'='*70}\n")
        
        return "\n".join(sections)