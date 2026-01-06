"""
Smart Money Concepts (SMC) Calculator

Pure Python implementations of SMC analysis:
- Market Structure (HH/HL/LH/LL, BOS, CHoCH)
- Order Blocks
- Fair Value Gaps (FVG)
- Liquidity Zones
- Premium/Discount Zones
- Optimal Trade Entry (OTE)

All functions operate on OHLCV data arrays.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class StructureType(Enum):
    """Market structure point types"""
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"


class TrendDirection(Enum):
    """Trend direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


@dataclass
class StructurePoint:
    """A market structure point"""
    index: int
    price: float
    type: StructureType
    is_break: bool = False  # BOS/CHoCH


@dataclass
class OrderBlock:
    """Order Block zone"""
    type: str  # "bullish" or "bearish"
    top: float
    bottom: float
    start_index: int
    end_index: int
    strength: str  # "strong", "moderate", "weak"
    mitigated: bool = False
    mitigated_at: Optional[int] = None


@dataclass
class FairValueGap:
    """Fair Value Gap (Imbalance)"""
    type: str  # "bullish" or "bearish"
    top: float
    bottom: float
    size: float
    start_index: int
    filled: bool = False
    filled_percent: float = 0.0


@dataclass
class LiquidityZone:
    """Liquidity zone (Equal Highs/Lows)"""
    type: str  # "equal_highs" or "equal_lows"
    price: float
    count: int
    indices: List[int]
    swept: bool = False


# =============================================================================
# MARKET STRUCTURE DETECTION
# =============================================================================

def detect_market_structure(
    ohlcv: Dict[str, np.ndarray],
    swing_length: int = 5
) -> Dict[str, Any]:
    """
    Detect market structure including:
    - Swing Highs/Lows
    - HH/HL/LH/LL classification
    - Break of Structure (BOS)
    - Change of Character (CHoCH)

    Args:
        ohlcv: Dict with 'open', 'high', 'low', 'close', 'volume' arrays
        swing_length: Number of bars for swing detection

    Returns:
        Dict with structure analysis
    """
    highs = np.array(ohlcv['high'], dtype=float)
    lows = np.array(ohlcv['low'], dtype=float)
    closes = np.array(ohlcv['close'], dtype=float)

    if len(highs) < swing_length * 3:
        return {
            "trend": "unknown",
            "structure_points": [],
            "bos": [],
            "choch": [],
            "error": "insufficient_data"
        }

    # Find swing points
    swing_highs = []
    swing_lows = []

    for i in range(swing_length, len(highs) - swing_length):
        # Swing High
        is_swing_high = all(
            highs[i] > highs[i - j] and highs[i] > highs[i + j]
            for j in range(1, swing_length + 1)
        )
        if is_swing_high:
            swing_highs.append((i, float(highs[i])))

        # Swing Low
        is_swing_low = all(
            lows[i] < lows[i - j] and lows[i] < lows[i + j]
            for j in range(1, swing_length + 1)
        )
        if is_swing_low:
            swing_lows.append((i, float(lows[i])))

    # Combine and sort swing points
    all_swings = []
    for idx, price in swing_highs:
        all_swings.append({"index": idx, "price": price, "type": "high"})
    for idx, price in swing_lows:
        all_swings.append({"index": idx, "price": price, "type": "low"})

    all_swings.sort(key=lambda x: x["index"])

    # Classify structure (HH/HL/LH/LL)
    structure_points = []
    bos_events = []
    choch_events = []

    last_high = None
    last_low = None
    current_trend = TrendDirection.RANGING

    for swing in all_swings:
        if swing["type"] == "high":
            if last_high is not None:
                if swing["price"] > last_high["price"]:
                    structure = StructureType.HIGHER_HIGH
                    # Check for BOS (bullish)
                    if current_trend == TrendDirection.BULLISH:
                        bos_events.append({
                            "index": swing["index"],
                            "price": swing["price"],
                            "direction": "bullish",
                            "broken_level": last_high["price"]
                        })
                    elif current_trend == TrendDirection.BEARISH:
                        # CHoCH - trend reversal
                        choch_events.append({
                            "index": swing["index"],
                            "price": swing["price"],
                            "from": "bearish",
                            "to": "bullish"
                        })
                        current_trend = TrendDirection.BULLISH
                    else:
                        current_trend = TrendDirection.BULLISH
                else:
                    structure = StructureType.LOWER_HIGH
                    if current_trend == TrendDirection.BULLISH and last_low:
                        # Potential CHoCH warning
                        pass

                structure_points.append({
                    "index": swing["index"],
                    "price": swing["price"],
                    "structure": structure.value,
                    "type": "high"
                })
            last_high = swing

        else:  # low
            if last_low is not None:
                if swing["price"] < last_low["price"]:
                    structure = StructureType.LOWER_LOW
                    # Check for BOS (bearish)
                    if current_trend == TrendDirection.BEARISH:
                        bos_events.append({
                            "index": swing["index"],
                            "price": swing["price"],
                            "direction": "bearish",
                            "broken_level": last_low["price"]
                        })
                    elif current_trend == TrendDirection.BULLISH:
                        # CHoCH - trend reversal
                        choch_events.append({
                            "index": swing["index"],
                            "price": swing["price"],
                            "from": "bullish",
                            "to": "bearish"
                        })
                        current_trend = TrendDirection.BEARISH
                    else:
                        current_trend = TrendDirection.BEARISH
                else:
                    structure = StructureType.HIGHER_LOW
                    if current_trend == TrendDirection.BEARISH:
                        # Potential CHoCH warning
                        pass

                structure_points.append({
                    "index": swing["index"],
                    "price": swing["price"],
                    "structure": structure.value,
                    "type": "low"
                })
            last_low = swing

    # Determine overall trend
    if len(structure_points) >= 2:
        recent = structure_points[-4:] if len(structure_points) >= 4 else structure_points
        hh_count = sum(1 for p in recent if p["structure"] == "HH")
        hl_count = sum(1 for p in recent if p["structure"] == "HL")
        lh_count = sum(1 for p in recent if p["structure"] == "LH")
        ll_count = sum(1 for p in recent if p["structure"] == "LL")

        if hh_count + hl_count > lh_count + ll_count:
            trend = "bullish"
        elif lh_count + ll_count > hh_count + hl_count:
            trend = "bearish"
        else:
            trend = "ranging"
    else:
        trend = "unknown"

    return {
        "trend": trend,
        "structure_points": structure_points[-10:],  # Last 10 points
        "bos": bos_events[-5:],  # Last 5 BOS
        "choch": choch_events[-3:],  # Last 3 CHoCH
        "swing_highs": [{"index": s[0], "price": s[1]} for s in swing_highs[-5:]],
        "swing_lows": [{"index": s[0], "price": s[1]} for s in swing_lows[-5:]],
        "last_high": last_high,
        "last_low": last_low
    }


# =============================================================================
# ORDER BLOCKS
# =============================================================================

def detect_order_blocks(
    ohlcv: Dict[str, np.ndarray],
    lookback: int = 50,
    min_move_percent: float = 0.5
) -> Dict[str, List[Dict]]:
    """
    Detect bullish and bearish order blocks

    Order Block criteria:
    - Bullish: Last down candle before strong up move
    - Bearish: Last up candle before strong down move

    Args:
        ohlcv: Dict with OHLCV arrays
        lookback: Number of bars to look back
        min_move_percent: Minimum move % to qualify

    Returns:
        Dict with bullish and bearish order blocks
    """
    opens = np.array(ohlcv['open'], dtype=float)
    highs = np.array(ohlcv['high'], dtype=float)
    lows = np.array(ohlcv['low'], dtype=float)
    closes = np.array(ohlcv['close'], dtype=float)

    bullish_obs = []
    bearish_obs = []

    start_idx = max(0, len(closes) - lookback)

    for i in range(start_idx + 3, len(closes) - 1):
        current_close = closes[i]

        # Check for bullish order block
        # Condition: Down candle followed by strong up move
        if closes[i - 1] < opens[i - 1]:  # Previous candle is bearish
            # Check for strong up move after
            move_up = (closes[i] - closes[i - 1]) / closes[i - 1] * 100
            if move_up >= min_move_percent:
                ob = {
                    "type": "bullish",
                    "top": float(max(opens[i - 1], closes[i - 1])),
                    "bottom": float(lows[i - 1]),
                    "start_index": i - 1,
                    "move_percent": round(move_up, 2),
                    "mitigated": False
                }

                # Check if already mitigated
                for j in range(i, len(closes)):
                    if lows[j] <= ob["bottom"]:
                        ob["mitigated"] = True
                        ob["mitigated_at"] = j
                        break

                # Determine strength
                if move_up >= 2.0:
                    ob["strength"] = "strong"
                elif move_up >= 1.0:
                    ob["strength"] = "moderate"
                else:
                    ob["strength"] = "weak"

                bullish_obs.append(ob)

        # Check for bearish order block
        # Condition: Up candle followed by strong down move
        if closes[i - 1] > opens[i - 1]:  # Previous candle is bullish
            # Check for strong down move after
            move_down = (closes[i - 1] - closes[i]) / closes[i - 1] * 100
            if move_down >= min_move_percent:
                ob = {
                    "type": "bearish",
                    "top": float(highs[i - 1]),
                    "bottom": float(min(opens[i - 1], closes[i - 1])),
                    "start_index": i - 1,
                    "move_percent": round(move_down, 2),
                    "mitigated": False
                }

                # Check if already mitigated
                for j in range(i, len(closes)):
                    if highs[j] >= ob["top"]:
                        ob["mitigated"] = True
                        ob["mitigated_at"] = j
                        break

                # Determine strength
                if move_down >= 2.0:
                    ob["strength"] = "strong"
                elif move_down >= 1.0:
                    ob["strength"] = "moderate"
                else:
                    ob["strength"] = "weak"

                bearish_obs.append(ob)

    # Filter to keep unmitigated OBs and most recent mitigated ones
    active_bullish = [ob for ob in bullish_obs if not ob["mitigated"]][-5:]
    active_bearish = [ob for ob in bearish_obs if not ob["mitigated"]][-5:]

    return {
        "bullish_order_blocks": active_bullish,
        "bearish_order_blocks": active_bearish,
        "total_bullish": len(bullish_obs),
        "total_bearish": len(bearish_obs),
        "active_bullish_count": len(active_bullish),
        "active_bearish_count": len(active_bearish)
    }


# =============================================================================
# FAIR VALUE GAPS
# =============================================================================

def detect_fair_value_gaps(
    ohlcv: Dict[str, np.ndarray],
    lookback: int = 50,
    min_gap_percent: float = 0.1
) -> Dict[str, List[Dict]]:
    """
    Detect Fair Value Gaps (FVG) / Imbalances

    FVG criteria:
    - Bullish FVG: Low of current candle > High of candle 2 bars ago
    - Bearish FVG: High of current candle < Low of candle 2 bars ago

    Args:
        ohlcv: Dict with OHLCV arrays
        lookback: Number of bars to look back
        min_gap_percent: Minimum gap size in %

    Returns:
        Dict with bullish and bearish FVGs
    """
    highs = np.array(ohlcv['high'], dtype=float)
    lows = np.array(ohlcv['low'], dtype=float)
    closes = np.array(ohlcv['close'], dtype=float)

    bullish_fvgs = []
    bearish_fvgs = []

    start_idx = max(0, len(closes) - lookback)

    for i in range(start_idx + 2, len(closes)):
        current_price = closes[-1]

        # Bullish FVG: Gap up
        # Low[i] > High[i-2]
        if lows[i] > highs[i - 2]:
            gap_size = lows[i] - highs[i - 2]
            gap_percent = gap_size / highs[i - 2] * 100

            if gap_percent >= min_gap_percent:
                fvg = {
                    "type": "bullish",
                    "top": float(lows[i]),
                    "bottom": float(highs[i - 2]),
                    "size": round(float(gap_size), 4),
                    "size_percent": round(gap_percent, 2),
                    "index": i,
                    "filled": False,
                    "filled_percent": 0.0
                }

                # Check how much is filled
                for j in range(i + 1, len(closes)):
                    if lows[j] <= fvg["bottom"]:
                        fvg["filled"] = True
                        fvg["filled_percent"] = 100.0
                        break
                    elif lows[j] < fvg["top"]:
                        fill_pct = (fvg["top"] - lows[j]) / gap_size * 100
                        fvg["filled_percent"] = round(fill_pct, 1)

                bullish_fvgs.append(fvg)

        # Bearish FVG: Gap down
        # High[i] < Low[i-2]
        if highs[i] < lows[i - 2]:
            gap_size = lows[i - 2] - highs[i]
            gap_percent = gap_size / lows[i - 2] * 100

            if gap_percent >= min_gap_percent:
                fvg = {
                    "type": "bearish",
                    "top": float(lows[i - 2]),
                    "bottom": float(highs[i]),
                    "size": round(float(gap_size), 4),
                    "size_percent": round(gap_percent, 2),
                    "index": i,
                    "filled": False,
                    "filled_percent": 0.0
                }

                # Check how much is filled
                for j in range(i + 1, len(closes)):
                    if highs[j] >= fvg["top"]:
                        fvg["filled"] = True
                        fvg["filled_percent"] = 100.0
                        break
                    elif highs[j] > fvg["bottom"]:
                        fill_pct = (highs[j] - fvg["bottom"]) / gap_size * 100
                        fvg["filled_percent"] = round(fill_pct, 1)

                bearish_fvgs.append(fvg)

    # Keep only unfilled or partially filled FVGs
    active_bullish = [fvg for fvg in bullish_fvgs if not fvg["filled"]][-5:]
    active_bearish = [fvg for fvg in bearish_fvgs if not fvg["filled"]][-5:]

    return {
        "bullish_fvgs": active_bullish,
        "bearish_fvgs": active_bearish,
        "total_bullish": len(bullish_fvgs),
        "total_bearish": len(bearish_fvgs),
        "active_bullish_count": len(active_bullish),
        "active_bearish_count": len(active_bearish)
    }


# =============================================================================
# LIQUIDITY ZONES
# =============================================================================

def detect_liquidity_zones(
    ohlcv: Dict[str, np.ndarray],
    lookback: int = 100,
    tolerance_percent: float = 0.1,
    min_touches: int = 2
) -> Dict[str, List[Dict]]:
    """
    Detect liquidity zones (Equal Highs/Lows)

    Liquidity pools where stop losses accumulate:
    - Equal Highs: Multiple swing highs at similar price
    - Equal Lows: Multiple swing lows at similar price

    Args:
        ohlcv: Dict with OHLCV arrays
        lookback: Number of bars to analyze
        tolerance_percent: Price tolerance for "equal"
        min_touches: Minimum touches to qualify

    Returns:
        Dict with liquidity zones
    """
    highs = np.array(ohlcv['high'], dtype=float)
    lows = np.array(ohlcv['low'], dtype=float)
    closes = np.array(ohlcv['close'], dtype=float)

    current_price = closes[-1]

    # Find swing points first
    swing_highs = []
    swing_lows = []
    swing_len = 5

    start_idx = max(0, len(closes) - lookback)

    for i in range(start_idx + swing_len, len(highs) - swing_len):
        # Swing High
        is_swing_high = all(
            highs[i] >= highs[i - j] and highs[i] >= highs[i + j]
            for j in range(1, swing_len + 1)
        )
        if is_swing_high:
            swing_highs.append((i, float(highs[i])))

        # Swing Low
        is_swing_low = all(
            lows[i] <= lows[i - j] and lows[i] <= lows[i + j]
            for j in range(1, swing_len + 1)
        )
        if is_swing_low:
            swing_lows.append((i, float(lows[i])))

    # Find equal highs (liquidity above)
    equal_highs = []
    used_high_indices = set()

    for i, (idx1, price1) in enumerate(swing_highs):
        if i in used_high_indices:
            continue

        cluster = [(idx1, price1)]
        tolerance = price1 * tolerance_percent / 100

        for j, (idx2, price2) in enumerate(swing_highs[i + 1:], start=i + 1):
            if j in used_high_indices:
                continue
            if abs(price2 - price1) <= tolerance:
                cluster.append((idx2, price2))
                used_high_indices.add(j)

        if len(cluster) >= min_touches:
            used_high_indices.add(i)
            avg_price = np.mean([p for _, p in cluster])

            # Check if swept
            swept = False
            for k in range(max(idx for idx, _ in cluster) + 1, len(highs)):
                if highs[k] > max(p for _, p in cluster):
                    swept = True
                    break

            equal_highs.append({
                "type": "equal_highs",
                "price": round(avg_price, 4),
                "touches": len(cluster),
                "indices": [idx for idx, _ in cluster],
                "swept": swept,
                "distance_from_price": round((avg_price - current_price) / current_price * 100, 2)
            })

    # Find equal lows (liquidity below)
    equal_lows = []
    used_low_indices = set()

    for i, (idx1, price1) in enumerate(swing_lows):
        if i in used_low_indices:
            continue

        cluster = [(idx1, price1)]
        tolerance = price1 * tolerance_percent / 100

        for j, (idx2, price2) in enumerate(swing_lows[i + 1:], start=i + 1):
            if j in used_low_indices:
                continue
            if abs(price2 - price1) <= tolerance:
                cluster.append((idx2, price2))
                used_low_indices.add(j)

        if len(cluster) >= min_touches:
            used_low_indices.add(i)
            avg_price = np.mean([p for _, p in cluster])

            # Check if swept
            swept = False
            for k in range(max(idx for idx, _ in cluster) + 1, len(lows)):
                if lows[k] < min(p for _, p in cluster):
                    swept = True
                    break

            equal_lows.append({
                "type": "equal_lows",
                "price": round(avg_price, 4),
                "touches": len(cluster),
                "indices": [idx for idx, _ in cluster],
                "swept": swept,
                "distance_from_price": round((current_price - avg_price) / current_price * 100, 2)
            })

    # Filter to keep unswept zones
    active_eq_highs = [z for z in equal_highs if not z["swept"]]
    active_eq_lows = [z for z in equal_lows if not z["swept"]]

    return {
        "equal_highs": active_eq_highs[-5:],
        "equal_lows": active_eq_lows[-5:],
        "buy_side_liquidity": active_eq_highs,  # Above current price
        "sell_side_liquidity": active_eq_lows,  # Below current price
        "nearest_liquidity_above": active_eq_highs[0] if active_eq_highs else None,
        "nearest_liquidity_below": active_eq_lows[-1] if active_eq_lows else None
    }


# =============================================================================
# PREMIUM/DISCOUNT ZONES
# =============================================================================

def calculate_premium_discount(
    ohlcv: Dict[str, np.ndarray],
    lookback: int = 50
) -> Dict[str, Any]:
    """
    Calculate Premium/Discount zones based on recent range

    Premium Zone: Upper 50% of range (sell zone in downtrend)
    Discount Zone: Lower 50% of range (buy zone in uptrend)
    Equilibrium: 50% level

    Also calculates Optimal Trade Entry (OTE) zone.

    Args:
        ohlcv: Dict with OHLCV arrays
        lookback: Number of bars for range calculation

    Returns:
        Dict with zones and current position
    """
    highs = np.array(ohlcv['high'], dtype=float)
    lows = np.array(ohlcv['low'], dtype=float)
    closes = np.array(ohlcv['close'], dtype=float)

    if len(closes) < lookback:
        return {"error": "insufficient_data"}

    # Find range high and low in lookback period
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]

    range_high = float(np.max(recent_highs))
    range_low = float(np.min(recent_lows))
    range_size = range_high - range_low

    if range_size == 0:
        return {"error": "no_range"}

    current_price = float(closes[-1])

    # Calculate zones
    equilibrium = range_low + (range_size * 0.5)

    # Premium zone (upper 50%)
    premium_zone = {
        "top": range_high,
        "bottom": equilibrium,
        "description": "Sell zone / Bearish bias"
    }

    # Discount zone (lower 50%)
    discount_zone = {
        "top": equilibrium,
        "bottom": range_low,
        "description": "Buy zone / Bullish bias"
    }

    # Optimal Trade Entry (OTE) zone - 61.8% to 78.6% retracement
    # For bullish OTE (buy zone)
    ote_bullish = {
        "top": range_high - (range_size * 0.618),
        "bottom": range_high - (range_size * 0.786),
        "description": "Bullish OTE - Buy zone"
    }

    # For bearish OTE (sell zone)
    ote_bearish = {
        "top": range_low + (range_size * 0.786),
        "bottom": range_low + (range_size * 0.618),
        "description": "Bearish OTE - Sell zone"
    }

    # Fibonacci levels within range
    fib_levels = {
        "0.0": range_low,
        "0.236": range_low + (range_size * 0.236),
        "0.382": range_low + (range_size * 0.382),
        "0.5": equilibrium,
        "0.618": range_low + (range_size * 0.618),
        "0.786": range_low + (range_size * 0.786),
        "1.0": range_high
    }

    # Determine current position
    position_percent = ((current_price - range_low) / range_size) * 100

    if position_percent >= 50:
        current_zone = "premium"
        bias = "bearish"
    else:
        current_zone = "discount"
        bias = "bullish"

    # Check if in OTE
    in_bullish_ote = ote_bullish["bottom"] <= current_price <= ote_bullish["top"]
    in_bearish_ote = ote_bearish["bottom"] <= current_price <= ote_bearish["top"]

    return {
        "range_high": round(range_high, 4),
        "range_low": round(range_low, 4),
        "range_size": round(range_size, 4),
        "equilibrium": round(equilibrium, 4),
        "current_price": round(current_price, 4),
        "position_percent": round(position_percent, 1),
        "current_zone": current_zone,
        "bias": bias,
        "premium_zone": {k: round(v, 4) if isinstance(v, float) else v for k, v in premium_zone.items()},
        "discount_zone": {k: round(v, 4) if isinstance(v, float) else v for k, v in discount_zone.items()},
        "ote_bullish": {k: round(v, 4) if isinstance(v, float) else v for k, v in ote_bullish.items()},
        "ote_bearish": {k: round(v, 4) if isinstance(v, float) else v for k, v in ote_bearish.items()},
        "in_bullish_ote": in_bullish_ote,
        "in_bearish_ote": in_bearish_ote,
        "fibonacci_levels": {k: round(v, 4) for k, v in fib_levels.items()},
        "lookback_bars": lookback
    }


# =============================================================================
# COMPLETE SMC ANALYSIS
# =============================================================================

def analyze_smc(
    ohlcv: Dict[str, np.ndarray],
    lookback: int = 100
) -> Dict[str, Any]:
    """
    Perform complete SMC analysis

    Args:
        ohlcv: Dict with OHLCV arrays
        lookback: Number of bars to analyze

    Returns:
        Dict with complete SMC analysis
    """
    # Market Structure
    structure = detect_market_structure(ohlcv)

    # Order Blocks
    order_blocks = detect_order_blocks(ohlcv, lookback=lookback)

    # Fair Value Gaps
    fvgs = detect_fair_value_gaps(ohlcv, lookback=lookback)

    # Liquidity Zones
    liquidity = detect_liquidity_zones(ohlcv, lookback=lookback)

    # Premium/Discount
    pd_zones = calculate_premium_discount(ohlcv, lookback=lookback)

    # Trading bias based on all factors
    bias_score = 0

    # Structure contribution
    if structure["trend"] == "bullish":
        bias_score += 2
    elif structure["trend"] == "bearish":
        bias_score -= 2

    # Zone contribution
    if pd_zones.get("current_zone") == "discount":
        bias_score += 1
    elif pd_zones.get("current_zone") == "premium":
        bias_score -= 1

    # OTE contribution
    if pd_zones.get("in_bullish_ote"):
        bias_score += 1
    if pd_zones.get("in_bearish_ote"):
        bias_score -= 1

    # Order block contribution
    if order_blocks.get("active_bullish_count", 0) > order_blocks.get("active_bearish_count", 0):
        bias_score += 1
    elif order_blocks.get("active_bearish_count", 0) > order_blocks.get("active_bullish_count", 0):
        bias_score -= 1

    # Determine overall bias
    if bias_score >= 2:
        overall_bias = "strong_bullish"
    elif bias_score >= 1:
        overall_bias = "bullish"
    elif bias_score <= -2:
        overall_bias = "strong_bearish"
    elif bias_score <= -1:
        overall_bias = "bearish"
    else:
        overall_bias = "neutral"

    return {
        "overall_bias": overall_bias,
        "bias_score": bias_score,
        "market_structure": structure,
        "order_blocks": order_blocks,
        "fair_value_gaps": fvgs,
        "liquidity_zones": liquidity,
        "premium_discount": pd_zones,
        "trading_zones": {
            "buy_zones": [
                *[{"type": "bullish_ob", **ob} for ob in order_blocks.get("bullish_order_blocks", [])],
                *[{"type": "bullish_fvg", **fvg} for fvg in fvgs.get("bullish_fvgs", [])]
            ],
            "sell_zones": [
                *[{"type": "bearish_ob", **ob} for ob in order_blocks.get("bearish_order_blocks", [])],
                *[{"type": "bearish_fvg", **fvg} for fvg in fvgs.get("bearish_fvgs", [])]
            ]
        }
    }
