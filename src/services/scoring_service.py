"""
Investment Scoring Service

Provides weighted composite scoring system for synthesis reports.
Converts qualitative analysis from 5 steps into quantitative scores.

Component Weights:
    - Fundamental:  30%  (P/E, Growth, Profitability)
    - Technical:    20%  (Trend, Momentum, Volume)
    - Risk:         20%  (Volatility, Max DD, Stop Loss quality)
    - Position:     15%  (Relative Strength vs market)
    - Sentiment:    15%  (News tone, Social sentiment)

Score Range: 0-100
    80-100: Strong Buy
    65-79:  Buy
    45-64:  Hold
    30-44:  Sell
    0-29:   Strong Sell

Usage:
    from src.services.scoring_service import ScoringService

    service = ScoringService()
    result = service.calculate_composite_score(all_step_data)
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

COMPONENT_WEIGHTS = {
    "fundamental": 0.30,
    "technical":   0.20,
    "risk":        0.20,
    "position":    0.15,
    "sentiment":   0.15,
}

RECOMMENDATION_THRESHOLDS = {
    "strong_buy":  {"min": 80, "label": "STRONG BUY", "emoji": "🟢🟢"},
    "buy":         {"min": 65, "label": "BUY", "emoji": "🟢"},
    "hold":        {"min": 45, "label": "HOLD", "emoji": "🟡"},
    "sell":        {"min": 30, "label": "SELL", "emoji": "🔴"},
    "strong_sell": {"min": 0,  "label": "STRONG SELL", "emoji": "🔴🔴"},
}


@dataclass
class ComponentScore:
    """Score for a single analysis component."""
    name: str
    score: float  # 0-100
    weight: float
    signals: List[str]
    confidence: str  # "high", "medium", "low"


@dataclass
class CompositeScore:
    """Final composite investment score."""
    total_score: float
    recommendation: str
    confidence: float
    buy_pct: int
    hold_pct: int
    sell_pct: int
    components: Dict[str, ComponentScore]
    key_factors: List[Dict[str, str]]
    time_horizon: str


# =============================================================================
# SCORING SERVICE
# =============================================================================

class ScoringService:
    """
    Investment scoring service that converts analysis data into quantitative scores.
    """

    def __init__(self):
        self.weights = COMPONENT_WEIGHTS
        self.thresholds = RECOMMENDATION_THRESHOLDS

    def calculate_composite_score(
        self,
        step_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate composite investment score from all step data.

        Args:
            step_data: Dictionary containing data from all 5 steps:
                {
                    "technical": {"content": "...", "raw_data": {...}},
                    "position": {"content": "...", "raw_data": {...}},
                    "risk": {"content": "...", "raw_data": {...}},
                    "sentiment": {"content": "...", "raw_data": {...}},
                    "fundamental": {"content": "...", "raw_data": {...}}
                }

        Returns:
            Composite score result dictionary
        """
        component_scores = {}

        # Calculate each component score
        component_scores["fundamental"] = self._score_fundamental(
            step_data.get("fundamental", {})
        )
        component_scores["technical"] = self._score_technical(
            step_data.get("technical", {})
        )
        component_scores["risk"] = self._score_risk(
            step_data.get("risk", {})
        )
        component_scores["position"] = self._score_position(
            step_data.get("position", {})
        )
        component_scores["sentiment"] = self._score_sentiment(
            step_data.get("sentiment", {})
        )

        # Calculate weighted composite
        total_score = self._calculate_weighted_score(component_scores)

        # Determine recommendation
        recommendation = self._get_recommendation(total_score)

        # Calculate confidence
        confidence = self._calculate_confidence(component_scores)

        # Calculate distribution percentages
        buy_pct, hold_pct, sell_pct = self._calculate_distribution(
            total_score, confidence
        )

        # Extract key factors
        key_factors = self._extract_key_factors(component_scores)

        # Determine time horizon
        time_horizon = self._determine_time_horizon(component_scores)

        return {
            "composite_score": round(total_score, 1),
            "recommendation": {
                "action": recommendation["label"],
                "emoji": recommendation["emoji"],
                "distribution": {
                    "buy": buy_pct,
                    "hold": hold_pct,
                    "sell": sell_pct
                },
                "confidence": round(confidence, 1),
                "time_horizon": time_horizon
            },
            "component_scores": {
                name: {
                    "score": cs["score"],
                    "weight": f"{cs['weight']*100:.0f}%",
                    "signals": cs["signals"],
                    "confidence": cs["confidence"]
                }
                for name, cs in component_scores.items()
            },
            "key_factors": key_factors
        }

    # =========================================================================
    # COMPONENT SCORING METHODS
    # =========================================================================

    def _score_fundamental(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score fundamental analysis (0-100).

        Factors:
        - P/E ratio vs industry average
        - Revenue/EPS growth rate
        - ROE, ROA, margins
        - Debt levels
        """
        score = 50  # Default neutral
        signals = []
        confidence = "medium"

        content = data.get("content", "")
        raw_data = data.get("raw_data", {})

        # Extract signals from content analysis
        content_lower = content.lower()

        # Valuation signals
        if any(x in content_lower for x in ["undervalued", "cheap", "discount", "attractive valuation"]):
            score += 15
            signals.append("+Valuation hấp dẫn")
        elif any(x in content_lower for x in ["overvalued", "expensive", "premium"]):
            score -= 15
            signals.append("-Định giá cao")

        # Growth signals
        if any(x in content_lower for x in ["strong growth", "high growth", "accelerating"]):
            score += 12
            signals.append("+Tăng trưởng mạnh")
        elif any(x in content_lower for x in ["declining", "negative growth", "slowing"]):
            score -= 12
            signals.append("-Tăng trưởng chậm")

        # Profitability signals
        if any(x in content_lower for x in ["high roe", "strong margins", "profitable"]):
            score += 10
            signals.append("+Biên lợi nhuận tốt")
        elif any(x in content_lower for x in ["low margin", "unprofitable", "negative earnings"]):
            score -= 10
            signals.append("-Biên lợi nhuận thấp")

        # Debt signals
        if any(x in content_lower for x in ["low debt", "strong balance sheet", "cash rich"]):
            score += 8
            signals.append("+Tài chính lành mạnh")
        elif any(x in content_lower for x in ["high debt", "leverage", "debt concern"]):
            score -= 8
            signals.append("-Nợ cao")

        # Analyst recommendation signals
        if any(x in content_lower for x in ["strong buy", "outperform", "buy recommendation"]):
            score += 5
            signals.append("+Analyst khuyến nghị mua")
        elif any(x in content_lower for x in ["sell", "underperform", "downgrade"]):
            score -= 5
            signals.append("-Analyst khuyến nghị bán")

        # Clamp score to 0-100
        score = max(0, min(100, score))

        # Determine confidence based on data availability
        if raw_data and len(content) > 500:
            confidence = "high"
        elif not raw_data and len(content) < 200:
            confidence = "low"

        return {
            "score": score,
            "weight": self.weights["fundamental"],
            "signals": signals if signals else ["~Neutral"],
            "confidence": confidence
        }

    def _score_technical(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score technical analysis (0-100).

        Factors:
        - Trend direction (bullish/bearish/ranging)
        - Momentum indicators (RSI, MACD)
        - Volume confirmation
        - Support/Resistance levels
        """
        score = 50
        signals = []
        confidence = "medium"

        content = data.get("content", "")
        content_lower = content.lower()

        # Trend signals
        if any(x in content_lower for x in ["strong uptrend", "bullish trend", "golden cross"]):
            score += 15
            signals.append("+Xu hướng tăng mạnh")
        elif any(x in content_lower for x in ["uptrend", "bullish"]):
            score += 10
            signals.append("+Xu hướng tăng")
        elif any(x in content_lower for x in ["downtrend", "bearish", "death cross"]):
            score -= 15
            signals.append("-Xu hướng giảm")
        elif any(x in content_lower for x in ["ranging", "sideways", "consolidation"]):
            signals.append("~Sideway/Tích lũy")

        # RSI signals
        if "oversold" in content_lower or "rsi" in content_lower and "below 30" in content_lower:
            score += 10
            signals.append("+RSI oversold (cơ hội mua)")
        elif "overbought" in content_lower or "rsi" in content_lower and "above 70" in content_lower:
            score -= 5
            signals.append("-RSI overbought")

        # MACD signals
        if any(x in content_lower for x in ["macd bullish", "macd crossing up", "macd positive"]):
            score += 8
            signals.append("+MACD bullish")
        elif any(x in content_lower for x in ["macd bearish", "macd crossing down", "macd negative"]):
            score -= 8
            signals.append("-MACD bearish")

        # Volume signals
        if any(x in content_lower for x in ["high volume", "volume confirm", "accumulation"]):
            score += 7
            signals.append("+Volume xác nhận")
        elif any(x in content_lower for x in ["low volume", "weak volume", "distribution"]):
            score -= 5
            signals.append("-Volume yếu")

        # Support/Resistance
        if any(x in content_lower for x in ["near support", "bouncing", "support hold"]):
            score += 5
            signals.append("+Gần vùng hỗ trợ")
        elif any(x in content_lower for x in ["near resistance", "rejected"]):
            score -= 5
            signals.append("-Gần vùng kháng cự")

        # Breakout signals
        if any(x in content_lower for x in ["breakout", "breaking out"]):
            score += 10
            signals.append("+Breakout signal")
        elif any(x in content_lower for x in ["breakdown", "breaking down"]):
            score -= 10
            signals.append("-Breakdown signal")

        score = max(0, min(100, score))

        if len(content) > 800:
            confidence = "high"
        elif len(content) < 300:
            confidence = "low"

        return {
            "score": score,
            "weight": self.weights["technical"],
            "signals": signals if signals else ["~Neutral"],
            "confidence": confidence
        }

    def _score_risk(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score risk analysis (0-100).
        Higher score = Lower risk = Better.

        Factors:
        - Volatility level
        - Max drawdown
        - Risk/Reward ratio
        - Stop loss quality
        """
        score = 50
        signals = []
        confidence = "medium"

        content = data.get("content", "")
        content_lower = content.lower()

        # Volatility signals (lower = better for risk score)
        if any(x in content_lower for x in ["low volatility", "stable", "low atr"]):
            score += 15
            signals.append("+Biến động thấp")
        elif any(x in content_lower for x in ["high volatility", "volatile", "extreme"]):
            score -= 15
            signals.append("-Biến động cao")
        elif any(x in content_lower for x in ["moderate volatility", "normal"]):
            signals.append("~Biến động bình thường")

        # Max drawdown signals
        if any(x in content_lower for x in ["small drawdown", "limited downside"]):
            score += 10
            signals.append("+Drawdown thấp")
        elif any(x in content_lower for x in ["large drawdown", "significant decline", "deep pullback"]):
            score -= 10
            signals.append("-Drawdown cao")

        # Risk/Reward signals
        if any(x in content_lower for x in ["favorable risk", "good r:r", "2:1", "3:1"]):
            score += 12
            signals.append("+R:R hấp dẫn")
        elif any(x in content_lower for x in ["poor risk", "unfavorable", "0.5:1"]):
            score -= 10
            signals.append("-R:R không hấp dẫn")

        # Stop loss quality
        if any(x in content_lower for x in ["clear stop", "defined stop", "tight stop"]):
            score += 8
            signals.append("+Stop loss rõ ràng")
        elif any(x in content_lower for x in ["wide stop", "unclear stop"]):
            score -= 5
            signals.append("-Stop loss không rõ")

        # Gap risk
        if any(x in content_lower for x in ["gap risk", "earnings risk", "event risk"]):
            score -= 8
            signals.append("-Rủi ro gap/event")

        score = max(0, min(100, score))

        if len(content) > 500:
            confidence = "high"
        elif len(content) < 200:
            confidence = "low"

        return {
            "score": score,
            "weight": self.weights["risk"],
            "signals": signals if signals else ["~Neutral"],
            "confidence": confidence
        }

    def _score_position(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score market position / relative strength (0-100).

        Factors:
        - RS vs benchmark (SPY)
        - Sector strength
        - Leadership status
        """
        score = 50
        signals = []
        confidence = "medium"

        content = data.get("content", "")
        content_lower = content.lower()

        # RS vs benchmark
        if any(x in content_lower for x in ["leader", "outperform", "strong rs", "beating"]):
            score += 20
            signals.append("+Outperform thị trường")
        elif any(x in content_lower for x in ["underperform", "lagging", "weak rs", "trailing"]):
            score -= 20
            signals.append("-Underperform thị trường")
        elif any(x in content_lower for x in ["neutral", "in-line", "matching"]):
            signals.append("~RS trung tính")

        # RS trend
        if any(x in content_lower for x in ["improving rs", "strengthening", "gaining"]):
            score += 10
            signals.append("+RS đang cải thiện")
        elif any(x in content_lower for x in ["weakening rs", "declining rs", "deteriorating"]):
            score -= 10
            signals.append("-RS đang suy yếu")

        # Sector strength
        if any(x in content_lower for x in ["leading sector", "strong sector", "sector tailwind"]):
            score += 8
            signals.append("+Ngành đang mạnh")
        elif any(x in content_lower for x in ["lagging sector", "weak sector", "sector headwind"]):
            score -= 8
            signals.append("-Ngành đang yếu")

        # Multi-timeframe confirmation
        if any(x in content_lower for x in ["confirmed", "all timeframe", "consistent"]):
            score += 5
            signals.append("+Xác nhận đa khung")

        score = max(0, min(100, score))

        if len(content) > 500:
            confidence = "high"
        elif len(content) < 200:
            confidence = "low"

        return {
            "score": score,
            "weight": self.weights["position"],
            "signals": signals if signals else ["~Neutral"],
            "confidence": confidence
        }

    def _score_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score sentiment & news (0-100).

        Factors:
        - Social sentiment score
        - News tone
        - Analyst sentiment
        """
        score = 50
        signals = []
        confidence = "medium"

        content = data.get("content", "")
        content_lower = content.lower()

        # Social sentiment
        if any(x in content_lower for x in ["bullish sentiment", "positive sentiment", "optimistic"]):
            score += 15
            signals.append("+Sentiment tích cực")
        elif any(x in content_lower for x in ["bearish sentiment", "negative sentiment", "pessimistic"]):
            score -= 15
            signals.append("-Sentiment tiêu cực")
        elif any(x in content_lower for x in ["neutral sentiment", "mixed sentiment"]):
            signals.append("~Sentiment trung tính")

        # News tone
        if any(x in content_lower for x in ["positive news", "good news", "catalyst"]):
            score += 12
            signals.append("+Tin tức tích cực")
        elif any(x in content_lower for x in ["negative news", "bad news", "concern"]):
            score -= 12
            signals.append("-Tin tức tiêu cực")

        # News impact
        if any(x in content_lower for x in ["major catalyst", "significant", "game-changing"]):
            score += 8
            signals.append("+Catalyst quan trọng")
        elif any(x in content_lower for x in ["headwind", "risk", "uncertainty"]):
            score -= 8
            signals.append("-Rủi ro từ tin tức")

        # Analyst mentions
        if any(x in content_lower for x in ["upgrade", "raised target", "positive coverage"]):
            score += 5
            signals.append("+Analyst tích cực")
        elif any(x in content_lower for x in ["downgrade", "lowered target", "negative coverage"]):
            score -= 5
            signals.append("-Analyst tiêu cực")

        score = max(0, min(100, score))

        if len(content) > 500:
            confidence = "high"
        elif len(content) < 200:
            confidence = "low"

        return {
            "score": score,
            "weight": self.weights["sentiment"],
            "signals": signals if signals else ["~Neutral"],
            "confidence": confidence
        }

    # =========================================================================
    # CALCULATION HELPERS
    # =========================================================================

    def _calculate_weighted_score(
        self,
        component_scores: Dict[str, Dict]
    ) -> float:
        """Calculate weighted average of component scores."""
        total = 0
        for name, data in component_scores.items():
            total += data["score"] * data["weight"]
        return total

    def _get_recommendation(self, score: float) -> Dict[str, str]:
        """Get recommendation based on composite score."""
        for level, config in self.thresholds.items():
            if score >= config["min"]:
                return config
        return self.thresholds["strong_sell"]

    def _calculate_confidence(
        self,
        component_scores: Dict[str, Dict]
    ) -> float:
        """
        Calculate overall confidence based on:
        - Component confidence levels
        - Signal alignment
        """
        confidence_map = {"high": 90, "medium": 70, "low": 50}

        # Average component confidence
        confidence_values = [
            confidence_map.get(cs["confidence"], 70)
            for cs in component_scores.values()
        ]
        avg_confidence = sum(confidence_values) / len(confidence_values)

        # Adjust for signal alignment
        scores = [cs["score"] for cs in component_scores.values()]
        score_variance = self._calculate_variance(scores)

        # High variance = low alignment = lower confidence
        if score_variance > 400:  # High variance
            avg_confidence -= 15
        elif score_variance < 100:  # Low variance = good alignment
            avg_confidence += 10

        return max(40, min(95, avg_confidence))

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def _calculate_distribution(
        self,
        score: float,
        confidence: float
    ) -> Tuple[int, int, int]:
        """
        Calculate buy/hold/sell distribution percentages.

        Higher score -> higher buy %
        Higher confidence -> more extreme distribution
        """
        # Base distribution from score
        if score >= 70:
            buy_base, hold_base, sell_base = 70, 20, 10
        elif score >= 55:
            buy_base, hold_base, sell_base = 55, 30, 15
        elif score >= 45:
            buy_base, hold_base, sell_base = 30, 50, 20
        elif score >= 35:
            buy_base, hold_base, sell_base = 15, 35, 50
        else:
            buy_base, hold_base, sell_base = 10, 20, 70

        # Adjust based on confidence
        # High confidence -> more extreme
        # Low confidence -> more towards hold
        conf_factor = (confidence - 70) / 50  # -0.6 to +0.5

        if conf_factor > 0:
            # High confidence - make distribution more extreme
            if score >= 50:
                buy_adj = int(buy_base * (1 + conf_factor * 0.2))
                hold_adj = int(hold_base * (1 - conf_factor * 0.3))
                sell_adj = 100 - buy_adj - hold_adj
            else:
                sell_adj = int(sell_base * (1 + conf_factor * 0.2))
                hold_adj = int(hold_base * (1 - conf_factor * 0.3))
                buy_adj = 100 - sell_adj - hold_adj
        else:
            # Low confidence - shift towards hold
            shift = int(abs(conf_factor) * 15)
            hold_adj = hold_base + shift
            if score >= 50:
                buy_adj = buy_base - shift
                sell_adj = 100 - buy_adj - hold_adj
            else:
                sell_adj = sell_base - shift
                buy_adj = 100 - sell_adj - hold_adj

        # Ensure valid percentages
        buy_adj = max(5, min(85, buy_adj))
        sell_adj = max(5, min(85, sell_adj))
        hold_adj = 100 - buy_adj - sell_adj

        return buy_adj, hold_adj, sell_adj

    def _extract_key_factors(
        self,
        component_scores: Dict[str, Dict]
    ) -> List[Dict[str, str]]:
        """
        Extract key factors (bullish/bearish) from all components.
        Returns top 5 most impactful factors.
        """
        factors = []

        # Collect all signals with their impact
        for comp_name, data in component_scores.items():
            weight = data["weight"]
            score = data["score"]

            for signal in data["signals"]:
                if signal.startswith("+"):
                    impact = "bullish"
                    importance = "high" if weight >= 0.2 else "medium"
                elif signal.startswith("-"):
                    impact = "bearish"
                    importance = "high" if weight >= 0.2 else "medium"
                else:
                    impact = "neutral"
                    importance = "low"

                factors.append({
                    "factor": signal.lstrip("+-~"),
                    "impact": impact,
                    "component": comp_name,
                    "weight": importance
                })

        # Sort by importance and take top factors
        importance_order = {"high": 0, "medium": 1, "low": 2}
        factors.sort(key=lambda x: (importance_order.get(x["weight"], 2), x["impact"]))

        # Return top 6 factors (mix of bullish and bearish)
        bullish = [f for f in factors if f["impact"] == "bullish"][:3]
        bearish = [f for f in factors if f["impact"] == "bearish"][:3]

        return bullish + bearish

    def _determine_time_horizon(
        self,
        component_scores: Dict[str, Dict]
    ) -> str:
        """
        Determine recommended investment time horizon.

        - Strong technicals -> Short-term (swing)
        - Strong fundamentals -> Long-term (position)
        - Mixed -> Medium-term
        """
        tech_score = component_scores.get("technical", {}).get("score", 50)
        fund_score = component_scores.get("fundamental", {}).get("score", 50)

        # Calculate which analysis is stronger
        tech_strength = abs(tech_score - 50)
        fund_strength = abs(fund_score - 50)

        if fund_strength > tech_strength + 10:
            return "6-12 months (position)"
        elif tech_strength > fund_strength + 10:
            return "1-4 weeks (swing)"
        else:
            return "1-3 months (medium-term)"


# =============================================================================
# MODULE-LEVEL INSTANCE
# =============================================================================

scoring_service = ScoringService()
