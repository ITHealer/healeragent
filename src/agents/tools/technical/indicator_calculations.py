"""
Technical Indicator Calculations

Centralized module for computing technical indicators using pandas_ta.
Provides calculation functions, analysis with signal interpretation,
pattern detection, and support/resistance identification.

Configuration:
    All magic numbers are centralized in technical_constants.py
    Import TECHNICAL_CONFIG to access configuration values.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, List, Optional, Tuple
from scipy.signal import argrelextrema

from src.agents.tools.technical.technical_constants import TECHNICAL_CONFIG


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

    Note:
        All indicator periods are configured in TECHNICAL_CONFIG.
    """
    cfg = TECHNICAL_CONFIG
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Trend Indicators (SMAs and EMAs)
    df['sma_20'] = ta.sma(df['close'], length=cfg.SMA_SHORT_PERIOD)
    df['sma_50'] = ta.sma(df['close'], length=cfg.SMA_MEDIUM_PERIOD)
    df['sma_200'] = ta.sma(df['close'], length=cfg.SMA_LONG_PERIOD)
    df['ema_12'] = ta.ema(df['close'], length=cfg.EMA_FAST_PERIOD)
    df['ema_26'] = ta.ema(df['close'], length=cfg.EMA_SLOW_PERIOD)

    # Momentum Indicators
    df['rsi'] = ta.rsi(df['close'], length=cfg.RSI_PERIOD)

    # MACD
    macd = ta.macd(
        df['close'],
        fast=cfg.MACD_FAST_PERIOD,
        slow=cfg.MACD_SLOW_PERIOD,
        signal=cfg.MACD_SIGNAL_PERIOD
    )
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        macd_cols = {
            f'MACD_{cfg.MACD_FAST_PERIOD}_{cfg.MACD_SLOW_PERIOD}_{cfg.MACD_SIGNAL_PERIOD}': 'macd_line',
            f'MACDs_{cfg.MACD_FAST_PERIOD}_{cfg.MACD_SLOW_PERIOD}_{cfg.MACD_SIGNAL_PERIOD}': 'macd_signal',
            f'MACDh_{cfg.MACD_FAST_PERIOD}_{cfg.MACD_SLOW_PERIOD}_{cfg.MACD_SIGNAL_PERIOD}': 'macd_histogram'
        }
        df.rename(columns=macd_cols, inplace=True)

    # Stochastic Oscillator
    stoch = ta.stoch(
        df['high'], df['low'], df['close'],
        k=cfg.STOCH_K_PERIOD,
        d=cfg.STOCH_D_PERIOD
    )
    if stoch is not None:
        df = pd.concat([df, stoch], axis=1)
        stoch_cols = {
            f'STOCHk_{cfg.STOCH_K_PERIOD}_{cfg.STOCH_D_PERIOD}_{cfg.STOCH_SMOOTH}': 'stoch_k',
            f'STOCHd_{cfg.STOCH_K_PERIOD}_{cfg.STOCH_D_PERIOD}_{cfg.STOCH_SMOOTH}': 'stoch_d'
        }
        df.rename(columns=stoch_cols, inplace=True)

    # Volatility Indicators
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=cfg.ATR_PERIOD)

    # Bollinger Bands
    bbands = ta.bbands(df['close'], length=cfg.BOLLINGER_PERIOD, std=cfg.BOLLINGER_STD_DEV)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
        bb_cols = {
            f'BBL_{cfg.BOLLINGER_PERIOD}_{cfg.BOLLINGER_STD_DEV}': 'bb_lower',
            f'BBM_{cfg.BOLLINGER_PERIOD}_{cfg.BOLLINGER_STD_DEV}': 'bb_middle',
            f'BBU_{cfg.BOLLINGER_PERIOD}_{cfg.BOLLINGER_STD_DEV}': 'bb_upper',
            f'BBB_{cfg.BOLLINGER_PERIOD}_{cfg.BOLLINGER_STD_DEV}': 'bb_bandwidth',
            f'BBP_{cfg.BOLLINGER_PERIOD}_{cfg.BOLLINGER_STD_DEV}': 'bb_percent'
        }
        df.rename(columns=bb_cols, inplace=True)

    # Trend Strength (ADX)
    adx = ta.adx(df['high'], df['low'], df['close'], length=cfg.ADX_PERIOD)
    if adx is not None:
        df = pd.concat([df, adx], axis=1)
        adx_cols = {
            f'ADX_{cfg.ADX_PERIOD}': 'adx',
            f'DMP_{cfg.ADX_PERIOD}': 'di_plus',
            f'DMN_{cfg.ADX_PERIOD}': 'di_minus'
        }
        df.rename(columns=adx_cols, inplace=True)

    # Volume Indicators
    df['volume_sma_20'] = ta.sma(df['volume'], length=cfg.VOLUME_SMA_PERIOD)
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']

    # VWAP (Volume Weighted Average Price)
    # Calculate cumulative VWAP for the entire period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    # OBV (On-Balance Volume)
    # OBV adds volume on up days and subtracts on down days
    price_change = df['close'].diff()
    obv_direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    df['obv'] = (df['volume'] * obv_direction).cumsum()

    # Price Range (ADR)
    daily_range = df['high'] - df['low']
    df['adr'] = daily_range.rolling(window=cfg.ADR_PERIOD).mean()
    df['adr_pct'] = (df['adr'] / df['close']) * 100

    return df


# =============================================================================
# INDIVIDUAL INDICATOR CALCULATIONS
# =============================================================================

def calculate_rsi(
    prices: pd.Series,
    period: int = None
) -> pd.Series:
    """Calculate RSI from price series."""
    period = period or TECHNICAL_CONFIG.RSI_PERIOD
    return ta.rsi(prices, length=period)


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return ta.sma(prices, length=period)


def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return ta.ema(prices, length=period)


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = None
) -> pd.Series:
    """Calculate Average True Range."""
    period = period or TECHNICAL_CONFIG.ATR_PERIOD
    return ta.atr(high, low, close, length=period)


def calculate_macd(
    prices: pd.Series,
    fast: int = None,
    slow: int = None,
    signal: int = None
) -> Dict[str, float]:
    """Calculate MACD values for latest data point."""
    cfg = TECHNICAL_CONFIG
    fast = fast or cfg.MACD_FAST_PERIOD
    slow = slow or cfg.MACD_SLOW_PERIOD
    signal = signal or cfg.MACD_SIGNAL_PERIOD

    macd_df = ta.macd(prices, fast=fast, slow=slow, signal=signal)
    if macd_df is None or macd_df.empty:
        return {'macd_line': 0, 'macd_signal': 0, 'macd_histogram': 0}

    latest = macd_df.iloc[-1]
    return {
        'macd_line': round(float(latest.iloc[0]), 4),
        'macd_signal': round(float(latest.iloc[1]), 4),
        'macd_histogram': round(float(latest.iloc[2]), 4)
    }


def calculate_bollinger_bands(
    prices: pd.Series,
    period: int = None,
    std_dev: float = None
) -> Dict[str, float]:
    """Calculate Bollinger Bands for latest data point."""
    cfg = TECHNICAL_CONFIG
    period = period or cfg.BOLLINGER_PERIOD
    std_dev = std_dev or cfg.BOLLINGER_STD_DEV

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


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = None,
    d: int = None
) -> Dict[str, float]:
    """Calculate Stochastic Oscillator for latest data point."""
    cfg = TECHNICAL_CONFIG
    k = k or cfg.STOCH_K_PERIOD
    d = d or cfg.STOCH_D_PERIOD

    stoch = ta.stoch(high, low, close, k=k, d=d)
    if stoch is None or stoch.empty:
        return {'stoch_k': 0, 'stoch_d': 0}

    latest = stoch.iloc[-1]
    return {
        'stoch_k': round(float(latest.iloc[0]), 2),
        'stoch_d': round(float(latest.iloc[1]), 2)
    }


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = None
) -> Dict[str, float]:
    """Calculate ADX and Directional Indicators for latest data point."""
    period = period or TECHNICAL_CONFIG.ADX_PERIOD
    adx_df = ta.adx(high, low, close, length=period)
    if adx_df is None or adx_df.empty:
        return {'adx': 0, 'di_plus': 0, 'di_minus': 0}

    latest = adx_df.iloc[-1]
    return {
        'adx': round(float(latest.iloc[0]), 2),
        'di_plus': round(float(latest.iloc[1]), 2),
        'di_minus': round(float(latest.iloc[2]), 2)
    }


def calculate_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series
) -> Dict[str, float]:
    """
    Calculate Volume Weighted Average Price (VWAP).

    VWAP = Cumulative(Typical Price × Volume) / Cumulative(Volume)
    where Typical Price = (High + Low + Close) / 3

    Returns:
        Dict with vwap value for the latest data point
    """
    if any(s.empty for s in [high, low, close, volume]):
        return {'vwap': 0.0}

    typical_price = (high + low + close) / 3
    vwap_series = (typical_price * volume).cumsum() / volume.cumsum()

    if vwap_series.empty or pd.isna(vwap_series.iloc[-1]):
        return {'vwap': 0.0}

    return {'vwap': round(float(vwap_series.iloc[-1]), 2)}


# =============================================================================
# INDICATOR ANALYSIS (with signal interpretation)
# =============================================================================

def analyze_rsi(rsi_value: float) -> Dict[str, Any]:
    """Analyze RSI value and return interpretation."""
    cfg = TECHNICAL_CONFIG

    if pd.isna(rsi_value):
        return {'value': None, 'signal': 'NEUTRAL', 'condition': 'unknown'}

    rsi = round(rsi_value, 2)

    if rsi >= cfg.RSI_OVERBOUGHT:
        return {'value': rsi, 'signal': 'SELL', 'condition': 'overbought'}
    elif rsi >= cfg.RSI_STRONG_THRESHOLD:
        return {'value': rsi, 'signal': 'NEUTRAL', 'condition': 'strong'}
    elif rsi <= cfg.RSI_OVERSOLD:
        return {'value': rsi, 'signal': 'BUY', 'condition': 'oversold'}
    elif rsi <= cfg.RSI_WEAK_THRESHOLD:
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
    cfg = TECHNICAL_CONFIG

    if any(pd.isna([stoch_k, stoch_d])):
        return {'signal': 'NEUTRAL', 'condition': 'unknown'}

    result = {
        'stoch_k': round(stoch_k, 2),
        'stoch_d': round(stoch_d, 2)
    }

    if stoch_k >= cfg.STOCH_OVERBOUGHT and stoch_d >= cfg.STOCH_OVERBOUGHT:
        result['signal'] = 'SELL'
        result['condition'] = 'overbought'
    elif stoch_k <= cfg.STOCH_OVERSOLD and stoch_d <= cfg.STOCH_OVERSOLD:
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
    cfg = TECHNICAL_CONFIG

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

    # Squeeze detection
    result['squeeze'] = bb_bandwidth < cfg.BOLLINGER_SQUEEZE_THRESHOLD if bb_bandwidth else False

    return result


def analyze_volume(current_volume: float, avg_volume: float) -> Dict[str, Any]:
    """Analyze volume relative to average."""
    cfg = TECHNICAL_CONFIG

    if any(pd.isna([current_volume, avg_volume])) or avg_volume == 0:
        return {'signal': 'NEUTRAL', 'description': 'Unknown volume'}

    volume_ratio = current_volume / avg_volume

    result = {
        'current_volume': int(current_volume),
        'avg_volume': int(avg_volume),
        'volume_ratio': round(volume_ratio, 2)
    }

    if volume_ratio >= cfg.VOLUME_VERY_HIGH_THRESHOLD:
        result['signal'] = 'STRONG'
        result['description'] = 'Very high volume - significant activity'
    elif volume_ratio >= cfg.VOLUME_HIGH_THRESHOLD:
        result['signal'] = 'HIGH'
        result['description'] = 'Above average volume'
    elif volume_ratio <= cfg.VOLUME_LOW_THRESHOLD:
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
    cfg = TECHNICAL_CONFIG

    if any(pd.isna([adx, di_plus, di_minus])):
        return {'signal': 'NEUTRAL', 'trend_strength': 'unknown'}

    result = {
        'adx': round(adx, 2),
        'di_plus': round(di_plus, 2),
        'di_minus': round(di_minus, 2)
    }

    # Trend strength
    if adx >= cfg.ADX_VERY_STRONG_TREND:
        result['trend_strength'] = 'very_strong'
    elif adx >= cfg.ADX_STRONG_TREND:
        result['trend_strength'] = 'strong'
    elif adx >= cfg.ADX_MODERATE_TREND:
        result['trend_strength'] = 'moderate'
    else:
        result['trend_strength'] = 'weak'

    # Direction
    if di_plus > di_minus:
        result['direction'] = 'bullish'
        result['signal'] = 'BULLISH' if adx >= cfg.ADX_STRONG_TREND else 'NEUTRAL'
    else:
        result['direction'] = 'bearish'
        result['signal'] = 'BEARISH' if adx >= cfg.ADX_STRONG_TREND else 'NEUTRAL'

    return result


def analyze_vwap(price: float, vwap: float) -> Dict[str, Any]:
    """
    Analyze VWAP relative to current price.

    VWAP (Volume Weighted Average Price) is used by institutions to gauge
    whether they paid a fair price. Key interpretations:
    - Price > VWAP: Bullish intraday sentiment, buyers in control
    - Price < VWAP: Bearish intraday sentiment, sellers in control
    - Price near VWAP: Fair value zone, watch for breakout

    Args:
        price: Current price
        vwap: VWAP value

    Returns:
        Dict with analysis including signal and description
    """
    if any(pd.isna([price, vwap])) or vwap == 0:
        return {'signal': 'NEUTRAL', 'position': 'unknown', 'vwap': None}

    result = {
        'price': round(price, 2),
        'vwap': round(vwap, 2)
    }

    # Calculate percentage difference
    diff_pct = ((price - vwap) / vwap) * 100
    result['diff_pct'] = round(diff_pct, 2)

    # Determine signal based on position relative to VWAP
    if diff_pct > 2.0:  # Price significantly above VWAP
        result['signal'] = 'BULLISH'
        result['position'] = 'above_vwap'
        result['description'] = 'Price above VWAP - bullish momentum, institutional buying pressure'
    elif diff_pct < -2.0:  # Price significantly below VWAP
        result['signal'] = 'BEARISH'
        result['position'] = 'below_vwap'
        result['description'] = 'Price below VWAP - bearish momentum, institutional selling pressure'
    elif diff_pct > 0.5:  # Slightly above
        result['signal'] = 'SLIGHTLY_BULLISH'
        result['position'] = 'slightly_above_vwap'
        result['description'] = 'Price slightly above VWAP - mild bullish bias'
    elif diff_pct < -0.5:  # Slightly below
        result['signal'] = 'SLIGHTLY_BEARISH'
        result['position'] = 'slightly_below_vwap'
        result['description'] = 'Price slightly below VWAP - mild bearish bias'
    else:  # Near VWAP
        result['signal'] = 'NEUTRAL'
        result['position'] = 'at_vwap'
        result['description'] = 'Price at fair value (VWAP) - watch for directional breakout'

    return result


def analyze_obv(
    obv_series: pd.Series,
    price_series: pd.Series,
    lookback: int = 14
) -> Dict[str, Any]:
    """
    Analyze On-Balance Volume (OBV) for trend confirmation.

    OBV Theory:
    - OBV rising with price rising = Confirmed uptrend (accumulation)
    - OBV falling with price falling = Confirmed downtrend (distribution)
    - OBV rising while price falling = Bullish divergence (potential reversal up)
    - OBV falling while price rising = Bearish divergence (potential reversal down)

    Args:
        obv_series: OBV values series
        price_series: Close price series
        lookback: Period for trend analysis

    Returns:
        Dict with OBV analysis and divergence detection
    """
    if obv_series.empty or price_series.empty or len(obv_series) < lookback:
        return {'signal': 'NEUTRAL', 'trend': 'unknown', 'divergence': None}

    # Get recent values
    recent_obv = obv_series.tail(lookback)
    recent_price = price_series.tail(lookback)

    # Calculate trends
    obv_start, obv_end = recent_obv.iloc[0], recent_obv.iloc[-1]
    price_start, price_end = recent_price.iloc[0], recent_price.iloc[-1]

    obv_change_pct = ((obv_end - obv_start) / abs(obv_start) * 100) if obv_start != 0 else 0
    price_change_pct = ((price_end - price_start) / price_start * 100) if price_start != 0 else 0

    result = {
        'obv_current': int(obv_end),
        'obv_change_pct': round(obv_change_pct, 2),
        'price_change_pct': round(price_change_pct, 2),
        'lookback_days': lookback
    }

    # Determine OBV trend
    obv_trend = 'rising' if obv_change_pct > 2 else 'falling' if obv_change_pct < -2 else 'flat'
    price_trend = 'rising' if price_change_pct > 2 else 'falling' if price_change_pct < -2 else 'flat'

    result['obv_trend'] = obv_trend
    result['price_trend'] = price_trend

    # Detect divergence and signal
    if obv_trend == 'rising' and price_trend == 'rising':
        result['signal'] = 'BULLISH'
        result['divergence'] = None
        result['description'] = 'OBV confirms uptrend - accumulation phase, buyers in control'
    elif obv_trend == 'falling' and price_trend == 'falling':
        result['signal'] = 'BEARISH'
        result['divergence'] = None
        result['description'] = 'OBV confirms downtrend - distribution phase, sellers in control'
    elif obv_trend == 'rising' and price_trend == 'falling':
        result['signal'] = 'BULLISH_DIVERGENCE'
        result['divergence'] = 'bullish'
        result['description'] = 'BULLISH DIVERGENCE - OBV rising while price falls, potential reversal UP'
    elif obv_trend == 'falling' and price_trend == 'rising':
        result['signal'] = 'BEARISH_DIVERGENCE'
        result['divergence'] = 'bearish'
        result['description'] = 'BEARISH DIVERGENCE - OBV falling while price rises, potential reversal DOWN'
    else:
        result['signal'] = 'NEUTRAL'
        result['divergence'] = None
        result['description'] = 'OBV trend inconclusive - wait for clearer signal'

    return result


def detect_ma_crossovers(
    df: pd.DataFrame,
    lookback: int = 5
) -> Dict[str, Any]:
    """
    Detect Moving Average crossovers (Golden Cross / Death Cross).

    Golden Cross: Short-term MA (SMA_50) crosses ABOVE long-term MA (SMA_200)
                  → Bullish signal, suggests uptrend beginning
    Death Cross: Short-term MA (SMA_50) crosses BELOW long-term MA (SMA_200)
                 → Bearish signal, suggests downtrend beginning

    Also detects faster crossovers:
    - SMA_20 crossing SMA_50 for medium-term signals

    Args:
        df: DataFrame with sma_20, sma_50, sma_200 columns
        lookback: Number of days to look back for crossover detection

    Returns:
        Dict with crossover detection results
    """
    result = {
        'golden_cross': None,
        'death_cross': None,
        'sma_20_50_cross': None,
        'current_alignment': None,
        'signal': 'NEUTRAL',
        'description': 'No recent crossovers detected'
    }

    if df.empty or len(df) < lookback + 1:
        return result

    # Get recent data
    recent = df.tail(lookback + 1)

    sma_50 = recent.get('sma_50')
    sma_200 = recent.get('sma_200')
    sma_20 = recent.get('sma_20')

    # Check SMA_50 vs SMA_200 (Golden/Death Cross)
    if sma_50 is not None and sma_200 is not None:
        if not sma_50.isna().all() and not sma_200.isna().all():
            # Calculate crossover: when difference changes sign
            diff = sma_50 - sma_200

            for i in range(1, len(diff)):
                prev_diff = diff.iloc[i - 1]
                curr_diff = diff.iloc[i]

                if pd.notna(prev_diff) and pd.notna(curr_diff):
                    # Golden Cross: diff goes from negative to positive
                    if prev_diff <= 0 and curr_diff > 0:
                        result['golden_cross'] = {
                            'detected': True,
                            'days_ago': lookback - i,
                            'sma_50': round(float(sma_50.iloc[i]), 2),
                            'sma_200': round(float(sma_200.iloc[i]), 2)
                        }
                        result['signal'] = 'BULLISH'
                        result['description'] = f'GOLDEN CROSS detected {lookback - i} days ago - SMA_50 crossed above SMA_200, bullish signal'

                    # Death Cross: diff goes from positive to negative
                    elif prev_diff >= 0 and curr_diff < 0:
                        result['death_cross'] = {
                            'detected': True,
                            'days_ago': lookback - i,
                            'sma_50': round(float(sma_50.iloc[i]), 2),
                            'sma_200': round(float(sma_200.iloc[i]), 2)
                        }
                        result['signal'] = 'BEARISH'
                        result['description'] = f'DEATH CROSS detected {lookback - i} days ago - SMA_50 crossed below SMA_200, bearish signal'

            # Current alignment
            latest_sma_50 = sma_50.iloc[-1]
            latest_sma_200 = sma_200.iloc[-1]
            if pd.notna(latest_sma_50) and pd.notna(latest_sma_200):
                result['current_alignment'] = 'bullish' if latest_sma_50 > latest_sma_200 else 'bearish'

    # Check SMA_20 vs SMA_50 (faster signal)
    if sma_20 is not None and sma_50 is not None:
        if not sma_20.isna().all() and not sma_50.isna().all():
            diff_20_50 = sma_20 - sma_50

            for i in range(1, len(diff_20_50)):
                prev_diff = diff_20_50.iloc[i - 1]
                curr_diff = diff_20_50.iloc[i]

                if pd.notna(prev_diff) and pd.notna(curr_diff):
                    if prev_diff <= 0 and curr_diff > 0:
                        result['sma_20_50_cross'] = {
                            'type': 'bullish',
                            'days_ago': lookback - i,
                            'description': 'SMA_20 crossed above SMA_50 - short-term bullish'
                        }
                        # Only set signal if no major crossover (Golden/Death) detected
                        if result['signal'] == 'NEUTRAL':
                            result['signal'] = 'BULLISH'
                            result['description'] = f'SMA_20/50 bullish crossover {lookback - i} days ago'
                    elif prev_diff >= 0 and curr_diff < 0:
                        result['sma_20_50_cross'] = {
                            'type': 'bearish',
                            'days_ago': lookback - i,
                            'description': 'SMA_20 crossed below SMA_50 - short-term bearish'
                        }
                        # Only set signal if no major crossover (Golden/Death) detected
                        if result['signal'] == 'NEUTRAL':
                            result['signal'] = 'BEARISH'
                            result['description'] = f'SMA_20/50 bearish crossover {lookback - i} days ago'

    return result


# =============================================================================
# SUPPORT & RESISTANCE
# =============================================================================

def identify_support_levels(
    lows: np.ndarray,
    closes: np.ndarray,
    current_price: float,
    order: int = None,
    num_levels: int = None
) -> List[Dict[str, Any]]:
    """Identify support levels using local minima."""
    cfg = TECHNICAL_CONFIG
    order = order or cfg.SR_LOOKBACK
    num_levels = num_levels or cfg.SR_NUM_LEVELS

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
                'strength': cfg.SR_DEFAULT_STRENGTH
            })

    # Sort by distance (nearest first)
    supports.sort(key=lambda x: x['distance_pct'])

    # Cluster nearby levels
    return _cluster_levels(supports, current_price, tolerance_pct=cfg.SR_CLUSTER_TOLERANCE)[:num_levels]


def identify_resistance_levels(
    highs: np.ndarray,
    closes: np.ndarray,
    current_price: float,
    order: int = None,
    num_levels: int = None
) -> List[Dict[str, Any]]:
    """Identify resistance levels using local maxima."""
    cfg = TECHNICAL_CONFIG
    order = order or cfg.SR_LOOKBACK
    num_levels = num_levels or cfg.SR_NUM_LEVELS

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
                'strength': cfg.SR_DEFAULT_STRENGTH
            })

    # Sort by distance (nearest first)
    resistances.sort(key=lambda x: x['distance_pct'])

    # Cluster nearby levels
    return _cluster_levels(resistances, current_price, tolerance_pct=cfg.SR_CLUSTER_TOLERANCE)[:num_levels]


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


def _cluster_levels(
    levels: List[Dict],
    current_price: float,
    tolerance_pct: float = None
) -> List[Dict]:
    """Cluster nearby price levels."""
    cfg = TECHNICAL_CONFIG
    tolerance_pct = tolerance_pct or cfg.SR_CLUSTER_TOLERANCE

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
                    existing.get('strength', cfg.SR_DEFAULT_STRENGTH),
                    level.get('strength', cfg.SR_DEFAULT_STRENGTH)
                ) + cfg.SR_STRENGTH_INCREMENT
                is_duplicate = True
                break

        if not is_duplicate:
            clustered.append(level.copy())

    return clustered


# =============================================================================
# CHART PATTERN DETECTION
# =============================================================================

def identify_chart_patterns(df: pd.DataFrame, lookback: int = None) -> Dict[str, Any]:
    """Identify common chart patterns."""
    cfg = TECHNICAL_CONFIG
    lookback = lookback or cfg.PATTERN_LOOKBACK

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


def _detect_double_bottom(
    lows: np.ndarray,
    closes: np.ndarray,
    tolerance: float = None
) -> Optional[Dict[str, Any]]:
    """Detect double bottom pattern (W shape)."""
    cfg = TECHNICAL_CONFIG
    tolerance = tolerance or cfg.PATTERN_TOLERANCE

    if len(lows) < cfg.PATTERN_MIN_BARS:
        return None

    # Find local minima
    minima_idx = argrelextrema(lows, np.less, order=cfg.PATTERN_EXTREMA_ORDER)[0]

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
                    'confidence': cfg.PATTERN_CONFIDENCE_HIGH,
                    'neckline': round(neckline, 2),
                    'target': round(neckline + (neckline - low2), 2),
                    'description': 'Double bottom pattern - bullish reversal'
                }

    return None


def _detect_double_top(
    highs: np.ndarray,
    closes: np.ndarray,
    tolerance: float = None
) -> Optional[Dict[str, Any]]:
    """Detect double top pattern (M shape)."""
    cfg = TECHNICAL_CONFIG
    tolerance = tolerance or cfg.PATTERN_TOLERANCE

    if len(highs) < cfg.PATTERN_MIN_BARS:
        return None

    # Find local maxima
    maxima_idx = argrelextrema(highs, np.greater, order=cfg.PATTERN_EXTREMA_ORDER)[0]

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
                    'confidence': cfg.PATTERN_CONFIDENCE_HIGH,
                    'neckline': round(neckline, 2),
                    'target': round(neckline - (high2 - neckline), 2),
                    'description': 'Double top pattern - bearish reversal'
                }

    return None


def _detect_bull_flag(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray
) -> Optional[Dict[str, Any]]:
    """Detect bull flag pattern (consolidation after strong move up)."""
    cfg = TECHNICAL_CONFIG

    if len(closes) < cfg.PATTERN_MIN_BARS:
        return None

    # Check for strong uptrend in first half
    first_half = closes[:len(closes)//2]
    second_half = closes[len(closes)//2:]

    first_move = (first_half[-1] - first_half[0]) / first_half[0]
    second_range = (max(second_half) - min(second_half)) / min(second_half)

    # Bull flag: strong move up followed by tight consolidation
    if first_move > cfg.BULL_FLAG_MIN_MOVE and second_range < cfg.BULL_FLAG_MAX_CONSOLIDATION:
        return {
            'detected': True,
            'signal': 'BUY',
            'confidence': cfg.PATTERN_CONFIDENCE_MEDIUM,
            'description': 'Bull flag pattern - continuation bullish'
        }

    return None


def _detect_bear_flag(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray
) -> Optional[Dict[str, Any]]:
    """Detect bear flag pattern (consolidation after strong move down)."""
    cfg = TECHNICAL_CONFIG

    if len(closes) < cfg.PATTERN_MIN_BARS:
        return None

    # Check for strong downtrend in first half
    first_half = closes[:len(closes)//2]
    second_half = closes[len(closes)//2:]

    first_move = (first_half[-1] - first_half[0]) / first_half[0]
    second_range = (max(second_half) - min(second_half)) / min(second_half)

    # Bear flag: strong move down followed by tight consolidation
    if first_move < -cfg.BULL_FLAG_MIN_MOVE and second_range < cfg.BULL_FLAG_MAX_CONSOLIDATION:
        return {
            'detected': True,
            'signal': 'SELL',
            'confidence': cfg.PATTERN_CONFIDENCE_MEDIUM,
            'description': 'Bear flag pattern - continuation bearish'
        }

    return None


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals(data: Dict[str, Any]) -> List[str]:
    """
    Generate trading signals from indicator values.

    Supports both flat structure (backward compatible) and nested structure
    from the comprehensive technical indicators tool.
    """
    cfg = TECHNICAL_CONFIG
    signals = []

    # Helper to extract value from nested or flat structure
    def get_value(key: str, nested_path: tuple = None):
        """Get value from flat key or nested path."""
        # Try flat structure first
        if key in data:
            val = data.get(key)
            if isinstance(val, dict):
                return val.get('value')
            return val
        # Try nested structure
        if nested_path:
            obj = data
            for p in nested_path:
                if isinstance(obj, dict):
                    obj = obj.get(p, {})
                else:
                    return None
            return obj if not isinstance(obj, dict) else obj.get('value')
        return None

    # RSI signals
    rsi = get_value('rsi_14') or get_value('rsi', ('rsi', 'value'))
    if rsi:
        if rsi > cfg.RSI_OVERBOUGHT:
            signals.append('RSI_OVERBOUGHT')
        elif rsi < cfg.RSI_OVERSOLD:
            signals.append('RSI_OVERSOLD')
        else:
            signals.append('RSI_NEUTRAL')

    # MACD signals
    macd_hist = get_value('macd_histogram') or get_value('histogram', ('macd', 'histogram'))
    if macd_hist is not None:
        signals.append('MACD_BULLISH' if macd_hist > 0 else 'MACD_BEARISH')

    # Trend signals (price vs SMA)
    price = data.get('current_price', 0)

    # Try flat then nested for SMAs
    sma_200 = get_value('sma_200')
    if sma_200 is None:
        ma_data = data.get('moving_averages', {}).get('sma', {})
        sma_200 = ma_data.get('sma_200', {}).get('value') if isinstance(ma_data.get('sma_200'), dict) else None

    sma_50 = get_value('sma_50')
    if sma_50 is None:
        ma_data = data.get('moving_averages', {}).get('sma', {})
        sma_50 = ma_data.get('sma_50', {}).get('value') if isinstance(ma_data.get('sma_50'), dict) else None

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
    bb_upper = get_value('bb_upper') or get_value('upper', ('bollinger_bands', 'upper'))
    bb_lower = get_value('bb_lower') or get_value('lower', ('bollinger_bands', 'lower'))

    if bb_upper and bb_lower and price:
        if price > bb_upper:
            signals.append('PRICE_ABOVE_UPPER_BAND')
        elif price < bb_lower:
            signals.append('PRICE_BELOW_LOWER_BAND')

    # Stochastic signals
    stoch_k = get_value('stoch_k') or get_value('k', ('stochastic', 'k'))
    stoch_d = get_value('stoch_d') or get_value('d', ('stochastic', 'd'))

    if stoch_k and stoch_d:
        if stoch_k > cfg.STOCH_OVERBOUGHT and stoch_d > cfg.STOCH_OVERBOUGHT:
            signals.append('STOCH_OVERBOUGHT')
        elif stoch_k < cfg.STOCH_OVERSOLD and stoch_d < cfg.STOCH_OVERSOLD:
            signals.append('STOCH_OVERSOLD')

    # ADX signals
    adx = get_value('adx') or get_value('adx', ('adx', 'adx'))
    if adx:
        if adx > cfg.ADX_STRONG_TREND:
            signals.append('TREND_STRONG')
        else:
            signals.append('TREND_WEAK')

    # VWAP signals (new)
    vwap = get_value('vwap') or get_value('vwap', ('vwap', 'value'))
    if vwap and price:
        if price > vwap * 1.02:
            signals.append('PRICE_ABOVE_VWAP')
        elif price < vwap * 0.98:
            signals.append('PRICE_BELOW_VWAP')

    # Volume signals (new)
    volume_ratio = get_value('volume_ratio') or get_value('ratio', ('volume', 'ratio'))
    if volume_ratio:
        if volume_ratio >= cfg.VOLUME_VERY_HIGH_THRESHOLD:
            signals.append('VOLUME_VERY_HIGH')
        elif volume_ratio >= cfg.VOLUME_HIGH_THRESHOLD:
            signals.append('VOLUME_HIGH')
        elif volume_ratio <= cfg.VOLUME_LOW_THRESHOLD:
            signals.append('VOLUME_LOW')

    # OBV divergence signals (from nested structure)
    obv_data = data.get('obv', {})
    if isinstance(obv_data, dict):
        obv_divergence = obv_data.get('divergence')
        if obv_divergence == 'bullish':
            signals.append('OBV_BULLISH_DIVERGENCE')
        elif obv_divergence == 'bearish':
            signals.append('OBV_BEARISH_DIVERGENCE')

    # MA Crossover signals (from nested structure)
    ma_cross_data = data.get('ma_crossovers', {})
    if isinstance(ma_cross_data, dict):
        # Check golden_cross safely - value can be None or dict
        golden_cross = ma_cross_data.get('golden_cross')
        if isinstance(golden_cross, dict) and golden_cross.get('detected'):
            signals.append('GOLDEN_CROSS')

        # Check death_cross safely
        death_cross = ma_cross_data.get('death_cross')
        if isinstance(death_cross, dict) and death_cross.get('detected'):
            signals.append('DEATH_CROSS')

        # Check SMA 20/50 crossover
        sma_20_50 = ma_cross_data.get('sma_20_50_cross')
        if isinstance(sma_20_50, dict):
            if sma_20_50.get('type') == 'bullish':
                signals.append('SMA_20_50_BULLISH_CROSS')
            elif sma_20_50.get('type') == 'bearish':
                signals.append('SMA_20_50_BEARISH_CROSS')

    return signals


def generate_outlook(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate overall market outlook from indicators."""
    cfg = TECHNICAL_CONFIG

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
        if rsi < cfg.RSI_WEAK_THRESHOLD:
            bearish_signals += 1
        elif rsi > cfg.RSI_STRONG_THRESHOLD:
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

    if bullish_pct >= cfg.OUTLOOK_STRONG_THRESHOLD:
        outlook = 'BULLISH'
        confidence = bullish_pct
    elif bearish_pct >= cfg.OUTLOOK_STRONG_THRESHOLD:
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
        # Renamed from 'confidence' to 'consensus' for clarity
        # This is % of basic signals agreeing, NOT trend strength (use ADX for that)
        'confidence': round(confidence, 2),  # Keep for backwards compat
        'consensus': round(confidence, 2),   # New clearer name
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'total_signals': total_signals,
        'note': 'Consensus = % of RSI/MACD/MA signals agreeing. For trend strength, check ADX indicator.'
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
        'di_minus': round(float(latest['di_minus']), 2) if pd.notna(latest.get('di_minus')) else None,
        'vwap': round(float(latest['vwap']), 2) if pd.notna(latest.get('vwap')) else None,
        'obv': int(latest['obv']) if pd.notna(latest.get('obv')) else None,
        'volume': int(latest['volume']) if pd.notna(latest.get('volume')) else None,
        'volume_sma_20': int(latest['volume_sma_20']) if pd.notna(latest.get('volume_sma_20')) else None,
        'volume_ratio': round(float(latest['volume_ratio']), 2) if pd.notna(latest.get('volume_ratio')) else None
    }
