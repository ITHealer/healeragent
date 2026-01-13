"""
Technical Indicator Calculations

Centralized module for computing technical indicators using pandas_ta.
Provides calculation functions, analysis with signal interpretation,
pattern detection, and support/resistance identification.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, List, Optional, Tuple
from scipy.signal import argrelextrema


# =============================================================================
# CORE INDICATOR CALCULATIONS
# =============================================================================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators to DataFrame.

    Args:
        df: DataFrame with columns [open, high, low, close, volume]

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Trend Indicators
    df['sma_20'] = ta.sma(df['close'], length=20)
    df['sma_50'] = ta.sma(df['close'], length=50)
    df['sma_200'] = ta.sma(df['close'], length=200)
    df['ema_12'] = ta.ema(df['close'], length=12)
    df['ema_26'] = ta.ema(df['close'], length=26)

    # Momentum Indicators
    df['rsi'] = ta.rsi(df['close'], length=14)

    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        df.rename(columns={
            'MACD_12_26_9': 'macd_line',
            'MACDs_12_26_9': 'macd_signal',
            'MACDh_12_26_9': 'macd_histogram'
        }, inplace=True)

    stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3)
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
        df.rename(columns={
            'STOCHk_14_3_3': 'stoch_k',
            'STOCHd_14_3_3': 'stoch_d'
        }, inplace=True)

    # Volatility Indicators
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

    bbands = ta.bbands(df['close'], length=20, std=2.0)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
        df.rename(columns={
            'BBL_20_2.0': 'bb_lower',
            'BBM_20_2.0': 'bb_middle',
            'BBU_20_2.0': 'bb_upper',
            'BBB_20_2.0': 'bb_bandwidth',
            'BBP_20_2.0': 'bb_percent'
        }, inplace=True)

    # Trend Strength
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)
        df.rename(columns={
            'ADX_14': 'adx',
            'DMP_14': 'di_plus',
            'DMN_14': 'di_minus'
        }, inplace=True)

    # Volume Indicators
    df['volume_sma_20'] = ta.sma(df['volume'], length=20)
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # Price Range
    daily_range = df['high'] - df['low']
    df['adr'] = daily_range.rolling(window=20).mean()
    df['adr_pct'] = (df['adr'] / df['close']) * 100

    return df


# =============================================================================
# INDIVIDUAL INDICATOR CALCULATIONS
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI from price series."""
    return ta.rsi(prices, length=period)


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return ta.sma(prices, length=period)


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return ta.ema(prices, length=period)


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    return ta.atr(high, low, close, length=period)


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> Dict[str, float]:
    """Calculate MACD values for latest data point."""
    macd_df = ta.macd(prices, fast=fast, slow=slow, signal=signal)
    if macd_df is None or macd_df.empty:
        return {'macd_line': 0, 'macd_signal': 0, 'macd_histogram': 0}

    latest = macd_df.iloc[-1]
    return {
        'macd_line': round(float(latest.iloc[0]), 4),
        'macd_signal': round(float(latest.iloc[1]), 4),
        'macd_histogram': round(float(latest.iloc[2]), 4)
    }


def calculate_bollinger_bands(prices: pd.Series, period: int = 20,
                               std_dev: float = 2.0) -> Dict[str, float]:
    """Calculate Bollinger Bands for latest data point."""
    bbands = ta.bbands(prices, length=period, std=std_dev)
    if bbands is None or bbands.empty:
        return {'bb_upper': 0, 'bb_middle': 0, 'bb_lower': 0}

    latest = bbands.iloc[-1]
    return {
        'bb_lower': round(float(latest.iloc[0]), 2),
        'bb_middle': round(float(latest.iloc[1]), 2),
        'bb_upper': round(float(latest.iloc[2]), 2),
        'bb_bandwidth': round(float(latest.iloc[3]), 4) if len(latest) > 3 else 0,
        'bb_percent': round(float(latest.iloc[4]), 4) if len(latest) > 4 else 0
    }


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k: int = 14, d: int = 3) -> Dict[str, float]:
    """Calculate Stochastic Oscillator for latest data point."""
    stoch = ta.stoch(high, low, close, k=k, d=d)
    if stoch is None or stoch.empty:
        return {'stoch_k': 0, 'stoch_d': 0}

    latest = stoch.iloc[-1]
    return {
        'stoch_k': round(float(latest.iloc[0]), 2),
        'stoch_d': round(float(latest.iloc[1]), 2)
    }


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series,
                  period: int = 14) -> Dict[str, float]:
    """Calculate ADX and Directional Indicators for latest data point."""
    adx_df = ta.adx(high, low, close, length=period)
    if adx_df is None or adx_df.empty:
        return {'adx': 0, 'di_plus': 0, 'di_minus': 0}

    latest = adx_df.iloc[-1]
    return {
        'adx': round(float(latest.iloc[0]), 2),
        'di_plus': round(float(latest.iloc[1]), 2),
        'di_minus': round(float(latest.iloc[2]), 2)
    }


# =============================================================================
# INDICATOR ANALYSIS (with signal interpretation)
# =============================================================================

def analyze_rsi(rsi_value: float) -> Dict[str, Any]:
    """Analyze RSI value and return interpretation."""
    if pd.isna(rsi_value):
        return {'value': None, 'signal': 'NEUTRAL', 'condition': 'unknown'}

    rsi = round(rsi_value, 2)

    if rsi >= 70:
        return {'value': rsi, 'signal': 'SELL', 'condition': 'overbought'}
    elif rsi >= 60:
        return {'value': rsi, 'signal': 'NEUTRAL', 'condition': 'strong'}
    elif rsi <= 30:
        return {'value': rsi, 'signal': 'BUY', 'condition': 'oversold'}
    elif rsi <= 40:
        return {'value': rsi, 'signal': 'NEUTRAL', 'condition': 'weak'}
    else:
        return {'value': rsi, 'signal': 'NEUTRAL', 'condition': 'neutral'}


def analyze_macd(macd_line: float, signal_line: float,
                 histogram: float) -> Dict[str, Any]:
    """Analyze MACD and return interpretation."""
    if any(pd.isna([macd_line, signal_line, histogram])):
        return {'signal': 'NEUTRAL', 'trend': 'unknown'}

    result = {
        'macd_line': round(macd_line, 4),
        'signal_line': round(signal_line, 4),
        'histogram': round(histogram, 4)
    }

    # Signal based on histogram
    if histogram > 0:
        result['signal'] = 'BULLISH'
        result['trend'] = 'bullish momentum'
    else:
        result['signal'] = 'BEARISH'
        result['trend'] = 'bearish momentum'

    # Crossover detection (simplified)
    if macd_line > signal_line and histogram > 0:
        result['crossover'] = 'bullish'
    elif macd_line < signal_line and histogram < 0:
        result['crossover'] = 'bearish'
    else:
        result['crossover'] = 'none'

    return result


def analyze_stochastic(stoch_k: float, stoch_d: float) -> Dict[str, Any]:
    """Analyze Stochastic Oscillator and return interpretation."""
    if any(pd.isna([stoch_k, stoch_d])):
        return {'signal': 'NEUTRAL', 'condition': 'unknown'}

    result = {
        'stoch_k': round(stoch_k, 2),
        'stoch_d': round(stoch_d, 2)
    }

    if stoch_k >= 80 and stoch_d >= 80:
        result['signal'] = 'SELL'
        result['condition'] = 'overbought'
    elif stoch_k <= 20 and stoch_d <= 20:
        result['signal'] = 'BUY'
        result['condition'] = 'oversold'
    else:
        result['signal'] = 'NEUTRAL'
        result['condition'] = 'neutral'

    # Crossover
    if stoch_k > stoch_d:
        result['crossover'] = 'bullish'
    else:
        result['crossover'] = 'bearish'

    return result


def analyze_bollinger_bands(price: float, bb_upper: float, bb_middle: float,
                            bb_lower: float, bb_bandwidth: float = 0) -> Dict[str, Any]:
    """Analyze Bollinger Bands position and return interpretation."""
    if any(pd.isna([price, bb_upper, bb_middle, bb_lower])):
        return {'signal': 'NEUTRAL', 'position': 'unknown'}

    result = {
        'bb_upper': round(bb_upper, 2),
        'bb_middle': round(bb_middle, 2),
        'bb_lower': round(bb_lower, 2),
        'bb_bandwidth': round(bb_bandwidth, 4) if bb_bandwidth else 0
    }

    # Price position
    if price >= bb_upper:
        result['signal'] = 'SELL'
        result['position'] = 'above_upper'
        result['description'] = 'Price at or above upper band - potential pullback'
    elif price <= bb_lower:
        result['signal'] = 'BUY'
        result['position'] = 'below_lower'
        result['description'] = 'Price at or below lower band - potential bounce'
    else:
        result['signal'] = 'NEUTRAL'
        result['position'] = 'within_bands'
        # Determine relative position
        if price > bb_middle:
            result['description'] = 'Price above middle band - bullish bias'
        else:
            result['description'] = 'Price below middle band - bearish bias'

    # Squeeze detection (bandwidth < 50% of 50-day average typically)
    result['squeeze'] = bb_bandwidth < 0.05 if bb_bandwidth else False

    return result


def analyze_volume(current_volume: float, avg_volume: float) -> Dict[str, Any]:
    """Analyze volume relative to average."""
    if any(pd.isna([current_volume, avg_volume])) or avg_volume == 0:
        return {'signal': 'NEUTRAL', 'description': 'Unknown volume'}

    volume_ratio = current_volume / avg_volume

    result = {
        'current_volume': int(current_volume),
        'avg_volume': int(avg_volume),
        'volume_ratio': round(volume_ratio, 2)
    }

    if volume_ratio >= 2.0:
        result['signal'] = 'STRONG'
        result['description'] = 'Very high volume - significant activity'
    elif volume_ratio >= 1.5:
        result['signal'] = 'HIGH'
        result['description'] = 'Above average volume'
    elif volume_ratio <= 0.5:
        result['signal'] = 'LOW'
        result['description'] = 'Very low volume - lack of interest'
    else:
        result['signal'] = 'NORMAL'
        result['description'] = 'Normal volume'

    return result


def analyze_trend(price: float, sma_20: float, sma_50: float,
                  sma_200: float) -> Dict[str, Any]:
    """Analyze price trend relative to moving averages."""
    result = {
        'price': round(price, 2),
        'sma_20': round(sma_20, 2) if pd.notna(sma_20) else None,
        'sma_50': round(sma_50, 2) if pd.notna(sma_50) else None,
        'sma_200': round(sma_200, 2) if pd.notna(sma_200) else None
    }

    # Check price vs MAs
    above_20 = price > sma_20 if pd.notna(sma_20) else None
    above_50 = price > sma_50 if pd.notna(sma_50) else None
    above_200 = price > sma_200 if pd.notna(sma_200) else None

    result['above_sma_20'] = above_20
    result['above_sma_50'] = above_50
    result['above_sma_200'] = above_200

    # MA alignment
    if pd.notna(sma_20) and pd.notna(sma_50):
        result['sma_20_above_50'] = sma_20 > sma_50
    if pd.notna(sma_50) and pd.notna(sma_200):
        result['sma_50_above_200'] = sma_50 > sma_200

    # Overall trend determination
    bullish_count = sum([above_20 or False, above_50 or False, above_200 or False])

    if bullish_count == 3:
        result['trend'] = 'STRONG_UPTREND'
        result['signal'] = 'BULLISH'
    elif bullish_count == 2:
        result['trend'] = 'UPTREND'
        result['signal'] = 'BULLISH'
    elif bullish_count == 1:
        result['trend'] = 'MIXED'
        result['signal'] = 'NEUTRAL'
    else:
        result['trend'] = 'DOWNTREND'
        result['signal'] = 'BEARISH'

    return result


def analyze_adx(adx: float, di_plus: float, di_minus: float) -> Dict[str, Any]:
    """Analyze ADX and Directional Movement."""
    if any(pd.isna([adx, di_plus, di_minus])):
        return {'signal': 'NEUTRAL', 'trend_strength': 'unknown'}

    result = {
        'adx': round(adx, 2),
        'di_plus': round(di_plus, 2),
        'di_minus': round(di_minus, 2)
    }

    # Trend strength
    if adx >= 50:
        result['trend_strength'] = 'very_strong'
    elif adx >= 25:
        result['trend_strength'] = 'strong'
    elif adx >= 20:
        result['trend_strength'] = 'moderate'
    else:
        result['trend_strength'] = 'weak'

    # Direction
    if di_plus > di_minus:
        result['direction'] = 'bullish'
        result['signal'] = 'BULLISH' if adx >= 25 else 'NEUTRAL'
    else:
        result['direction'] = 'bearish'
        result['signal'] = 'BEARISH' if adx >= 25 else 'NEUTRAL'

    return result


# =============================================================================
# SUPPORT & RESISTANCE
# =============================================================================

def identify_support_levels(lows: np.ndarray, closes: np.ndarray,
                            current_price: float, order: int = 5,
                            num_levels: int = 5) -> List[Dict[str, Any]]:
    """Identify support levels using local minima."""
    if len(lows) < order * 2:
        return []

    # Find local minima
    minima_idx = argrelextrema(lows, np.less, order=order)[0]

    # Get support levels below current price
    supports = []
    for idx in minima_idx:
        level = float(lows[idx])
        if level < current_price:
            distance_pct = ((current_price - level) / current_price) * 100
            supports.append({
                'price': round(level, 2),
                'type': 'swing_low',
                'distance_pct': round(distance_pct, 2),
                'strength': 0.7
            })

    # Sort by distance (nearest first)
    supports.sort(key=lambda x: x['distance_pct'])

    # Cluster nearby levels
    return _cluster_levels(supports, current_price, tolerance_pct=1.0)[:num_levels]


def identify_resistance_levels(highs: np.ndarray, closes: np.ndarray,
                               current_price: float, order: int = 5,
                               num_levels: int = 5) -> List[Dict[str, Any]]:
    """Identify resistance levels using local maxima."""
    if len(highs) < order * 2:
        return []

    # Find local maxima
    maxima_idx = argrelextrema(highs, np.greater, order=order)[0]

    # Get resistance levels above current price
    resistances = []
    for idx in maxima_idx:
        level = float(highs[idx])
        if level > current_price:
            distance_pct = ((level - current_price) / current_price) * 100
            resistances.append({
                'price': round(level, 2),
                'type': 'swing_high',
                'distance_pct': round(distance_pct, 2),
                'strength': 0.7
            })

    # Sort by distance (nearest first)
    resistances.sort(key=lambda x: x['distance_pct'])

    # Cluster nearby levels
    return _cluster_levels(resistances, current_price, tolerance_pct=1.0)[:num_levels]


def calculate_pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
    """Calculate classic pivot points."""
    pivot = (high + low + close) / 3

    return {
        'pivot': round(pivot, 2),
        'r1': round(2 * pivot - low, 2),
        'r2': round(pivot + (high - low), 2),
        'r3': round(high + 2 * (pivot - low), 2),
        's1': round(2 * pivot - high, 2),
        's2': round(pivot - (high - low), 2),
        's3': round(low - 2 * (high - pivot), 2)
    }


def _cluster_levels(levels: List[Dict], current_price: float,
                    tolerance_pct: float = 1.0) -> List[Dict]:
    """Cluster nearby price levels."""
    if not levels:
        return []

    tolerance = current_price * (tolerance_pct / 100)
    clustered = []

    for level in levels:
        is_duplicate = False
        for existing in clustered:
            if abs(level['price'] - existing['price']) <= tolerance:
                # Merge - keep stronger
                if level.get('strength', 0) > existing.get('strength', 0):
                    existing.update(level)
                existing['strength'] = max(
                    existing.get('strength', 0.5),
                    level.get('strength', 0.5)
                ) + 0.1
                is_duplicate = True
                break

        if not is_duplicate:
            clustered.append(level.copy())

    return clustered


# =============================================================================
# CHART PATTERN DETECTION
# =============================================================================

def identify_chart_patterns(df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
    """Identify common chart patterns."""
    patterns = {
        'double_bottom': None,
        'double_top': None,
        'bull_flag': None,
        'bear_flag': None
    }

    if len(df) < lookback:
        return patterns

    df_subset = df.tail(lookback).copy()
    closes = df_subset['close'].values
    highs = df_subset['high'].values
    lows = df_subset['low'].values

    # Detect patterns
    patterns['double_bottom'] = _detect_double_bottom(lows, closes)
    patterns['double_top'] = _detect_double_top(highs, closes)
    patterns['bull_flag'] = _detect_bull_flag(closes, highs, lows)
    patterns['bear_flag'] = _detect_bear_flag(closes, highs, lows)

    return patterns


def _detect_double_bottom(lows: np.ndarray, closes: np.ndarray,
                          tolerance: float = 0.02) -> Optional[Dict[str, Any]]:
    """Detect double bottom pattern (W shape)."""
    if len(lows) < 20:
        return None

    # Find local minima
    minima_idx = argrelextrema(lows, np.less, order=5)[0]

    if len(minima_idx) < 2:
        return None

    # Check last two lows
    for i in range(len(minima_idx) - 1, 0, -1):
        idx1, idx2 = minima_idx[i-1], minima_idx[i]
        low1, low2 = lows[idx1], lows[idx2]

        # Check if lows are similar (within tolerance)
        if abs(low1 - low2) / low1 <= tolerance:
            # Check for higher high between lows
            middle_high = max(closes[idx1:idx2+1])
            neckline = middle_high

            # Pattern confirmed if price breaks above neckline
            current_price = closes[-1]
            if current_price > neckline:
                return {
                    'detected': True,
                    'signal': 'BUY',
                    'confidence': 0.75,
                    'neckline': round(neckline, 2),
                    'target': round(neckline + (neckline - low2), 2),
                    'description': 'Double bottom pattern - bullish reversal'
                }

    return None


def _detect_double_top(highs: np.ndarray, closes: np.ndarray,
                       tolerance: float = 0.02) -> Optional[Dict[str, Any]]:
    """Detect double top pattern (M shape)."""
    if len(highs) < 20:
        return None

    # Find local maxima
    maxima_idx = argrelextrema(highs, np.greater, order=5)[0]

    if len(maxima_idx) < 2:
        return None

    # Check last two highs
    for i in range(len(maxima_idx) - 1, 0, -1):
        idx1, idx2 = maxima_idx[i-1], maxima_idx[i]
        high1, high2 = highs[idx1], highs[idx2]

        # Check if highs are similar
        if abs(high1 - high2) / high1 <= tolerance:
            # Check for lower low between highs
            middle_low = min(closes[idx1:idx2+1])
            neckline = middle_low

            # Pattern confirmed if price breaks below neckline
            current_price = closes[-1]
            if current_price < neckline:
                return {
                    'detected': True,
                    'signal': 'SELL',
                    'confidence': 0.75,
                    'neckline': round(neckline, 2),
                    'target': round(neckline - (high2 - neckline), 2),
                    'description': 'Double top pattern - bearish reversal'
                }

    return None


def _detect_bull_flag(closes: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray) -> Optional[Dict[str, Any]]:
    """Detect bull flag pattern (consolidation after strong move up)."""
    if len(closes) < 20:
        return None

    # Check for strong uptrend in first half
    first_half = closes[:len(closes)//2]
    second_half = closes[len(closes)//2:]

    first_move = (first_half[-1] - first_half[0]) / first_half[0]
    second_range = (max(second_half) - min(second_half)) / min(second_half)

    # Bull flag: strong move up (>5%) followed by tight consolidation (<3%)
    if first_move > 0.05 and second_range < 0.03:
        return {
            'detected': True,
            'signal': 'BUY',
            'confidence': 0.70,
            'description': 'Bull flag pattern - continuation bullish'
        }

    return None


def _detect_bear_flag(closes: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray) -> Optional[Dict[str, Any]]:
    """Detect bear flag pattern (consolidation after strong move down)."""
    if len(closes) < 20:
        return None

    # Check for strong downtrend in first half
    first_half = closes[:len(closes)//2]
    second_half = closes[len(closes)//2:]

    first_move = (first_half[-1] - first_half[0]) / first_half[0]
    second_range = (max(second_half) - min(second_half)) / min(second_half)

    # Bear flag: strong move down (<-5%) followed by tight consolidation (<3%)
    if first_move < -0.05 and second_range < 0.03:
        return {
            'detected': True,
            'signal': 'SELL',
            'confidence': 0.70,
            'description': 'Bear flag pattern - continuation bearish'
        }

    return None


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(data: Dict[str, Any]) -> List[str]:
    """Generate trading signals from indicator values."""
    signals = []

    # RSI signals
    rsi = data.get('rsi_14') or data.get('rsi')
    if rsi:
        if rsi > 70:
            signals.append('RSI_OVERBOUGHT')
        elif rsi < 30:
            signals.append('RSI_OVERSOLD')
        else:
            signals.append('RSI_NEUTRAL')

    # MACD signals
    macd_hist = data.get('macd_histogram')
    if macd_hist is not None:
        signals.append('MACD_BULLISH' if macd_hist > 0 else 'MACD_BEARISH')

    # Trend signals (price vs SMA)
    price = data.get('current_price', 0)
    sma_200 = data.get('sma_200')
    sma_50 = data.get('sma_50')

    if sma_200 and price:
        signals.append(
            'TREND_LONG_TERM_BULLISH' if price > sma_200
            else 'TREND_LONG_TERM_BEARISH'
        )

    if sma_50 and price:
        signals.append(
            'TREND_MEDIUM_TERM_BULLISH' if price > sma_50
            else 'TREND_MEDIUM_TERM_BEARISH'
        )

    # Bollinger Bands signals
    bb_upper = data.get('bb_upper')
    bb_lower = data.get('bb_lower')

    if bb_upper and bb_lower and price:
        if price > bb_upper:
            signals.append('PRICE_ABOVE_UPPER_BAND')
        elif price < bb_lower:
            signals.append('PRICE_BELOW_LOWER_BAND')

    # Stochastic signals
    stoch_k = data.get('stoch_k')
    stoch_d = data.get('stoch_d')

    if stoch_k and stoch_d:
        if stoch_k > 80 and stoch_d > 80:
            signals.append('STOCH_OVERBOUGHT')
        elif stoch_k < 20 and stoch_d < 20:
            signals.append('STOCH_OVERSOLD')

    # ADX signals
    adx = data.get('adx')
    if adx:
        if adx > 25:
            signals.append('TREND_STRONG')
        else:
            signals.append('TREND_WEAK')

    return signals


def generate_outlook(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate overall market outlook from indicators."""
    if df.empty:
        return {'outlook': 'NEUTRAL', 'confidence': 0}

    latest = df.iloc[-1]
    bullish_signals = 0
    bearish_signals = 0
    total_signals = 0

    # Check RSI
    rsi = latest.get('rsi')
    if pd.notna(rsi):
        total_signals += 1
        if rsi < 40:
            bearish_signals += 1
        elif rsi > 60:
            bullish_signals += 1

    # Check MACD
    macd_hist = latest.get('macd_histogram')
    if pd.notna(macd_hist):
        total_signals += 1
        if macd_hist > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1

    # Check trend (price vs SMAs)
    price = latest.get('close')
    sma_20 = latest.get('sma_20')
    sma_50 = latest.get('sma_50')

    if pd.notna(price) and pd.notna(sma_20):
        total_signals += 1
        if price > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1

    if pd.notna(price) and pd.notna(sma_50):
        total_signals += 1
        if price > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1

    # Determine outlook
    if total_signals == 0:
        return {'outlook': 'NEUTRAL', 'confidence': 0}

    bullish_pct = bullish_signals / total_signals
    bearish_pct = bearish_signals / total_signals

    if bullish_pct >= 0.7:
        outlook = 'BULLISH'
        confidence = bullish_pct
    elif bearish_pct >= 0.7:
        outlook = 'BEARISH'
        confidence = bearish_pct
    elif bullish_pct > bearish_pct:
        outlook = 'SLIGHTLY_BULLISH'
        confidence = bullish_pct
    elif bearish_pct > bullish_pct:
        outlook = 'SLIGHTLY_BEARISH'
        confidence = bearish_pct
    else:
        outlook = 'NEUTRAL'
        confidence = 0.5

    return {
        'outlook': outlook,
        'confidence': round(confidence, 2),
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'total_signals': total_signals
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_indicator_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary of all indicators from latest data point."""
    if df.empty:
        return {}

    latest = df.iloc[-1]

    return {
        'price': round(float(latest['close']), 2),
        'rsi_14': round(float(latest['rsi']), 2) if pd.notna(latest.get('rsi')) else None,
        'macd_line': round(float(latest['macd_line']), 4) if pd.notna(latest.get('macd_line')) else None,
        'macd_signal': round(float(latest['macd_signal']), 4) if pd.notna(latest.get('macd_signal')) else None,
        'macd_histogram': round(float(latest['macd_histogram']), 4) if pd.notna(latest.get('macd_histogram')) else None,
        'sma_20': round(float(latest['sma_20']), 2) if pd.notna(latest.get('sma_20')) else None,
        'sma_50': round(float(latest['sma_50']), 2) if pd.notna(latest.get('sma_50')) else None,
        'sma_200': round(float(latest['sma_200']), 2) if pd.notna(latest.get('sma_200')) else None,
        'ema_12': round(float(latest['ema_12']), 2) if pd.notna(latest.get('ema_12')) else None,
        'ema_26': round(float(latest['ema_26']), 2) if pd.notna(latest.get('ema_26')) else None,
        'bb_upper': round(float(latest['bb_upper']), 2) if pd.notna(latest.get('bb_upper')) else None,
        'bb_middle': round(float(latest['bb_middle']), 2) if pd.notna(latest.get('bb_middle')) else None,
        'bb_lower': round(float(latest['bb_lower']), 2) if pd.notna(latest.get('bb_lower')) else None,
        'atr_14': round(float(latest['atr']), 2) if pd.notna(latest.get('atr')) else None,
        'stoch_k': round(float(latest['stoch_k']), 2) if pd.notna(latest.get('stoch_k')) else None,
        'stoch_d': round(float(latest['stoch_d']), 2) if pd.notna(latest.get('stoch_d')) else None,
        'adx': round(float(latest['adx']), 2) if pd.notna(latest.get('adx')) else None,
        'di_plus': round(float(latest['di_plus']), 2) if pd.notna(latest.get('di_plus')) else None,
        'di_minus': round(float(latest['di_minus']), 2) if pd.notna(latest.get('di_minus')) else None
    }
