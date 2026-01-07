"""
Technical Indicators Calculator

Pure Python implementations of technical indicators.
Uses numpy for efficient array operations.

All functions expect numpy arrays or lists as input.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class SwingPoint:
    """Represents a swing high or low point"""
    index: int
    price: float
    type: str  # "high" or "low"


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

def calculate_rsi(closes: np.ndarray, period: int = 14) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index (RSI)

    Args:
        closes: Array of closing prices
        period: RSI period (default 14)

    Returns:
        Dict with rsi value and interpretation
    """
    closes = np.array(closes, dtype=float)

    if len(closes) < period + 1:
        return {"value": None, "signal": "insufficient_data", "period": period}

    # Calculate price changes
    deltas = np.diff(closes)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate average gains and losses (Wilder's smoothing)
    avg_gain = np.zeros(len(deltas))
    avg_loss = np.zeros(len(deltas))

    # First average
    avg_gain[period - 1] = np.mean(gains[:period])
    avg_loss[period - 1] = np.mean(losses[:period])

    # Wilder's smoothing for subsequent values
    for i in range(period, len(deltas)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    # Calculate RS and RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))

    current_rsi = round(float(rsi[-1]), 2)

    # Interpretation
    if current_rsi >= 70:
        signal = "overbought"
    elif current_rsi <= 30:
        signal = "oversold"
    elif current_rsi >= 50:
        signal = "bullish"
    else:
        signal = "bearish"

    return {
        "value": current_rsi,
        "signal": signal,
        "period": period,
        "history": rsi[-5:].tolist() if len(rsi) >= 5 else rsi.tolist()
    }


def calculate_stoch_rsi(
    closes: np.ndarray,
    rsi_period: int = 14,
    stoch_period: int = 14,
    k_smooth: int = 3,
    d_smooth: int = 3
) -> Dict[str, Any]:
    """
    Calculate Stochastic RSI

    Args:
        closes: Array of closing prices
        rsi_period: RSI calculation period
        stoch_period: Stochastic period applied to RSI
        k_smooth: %K smoothing period
        d_smooth: %D smoothing period (signal line)

    Returns:
        Dict with stoch_rsi_k, stoch_rsi_d, and signal
    """
    closes = np.array(closes, dtype=float)

    min_len = rsi_period + stoch_period + k_smooth + d_smooth
    if len(closes) < min_len:
        return {"k": None, "d": None, "signal": "insufficient_data"}

    # Calculate RSI first
    rsi_result = calculate_rsi(closes, rsi_period)
    if rsi_result["value"] is None:
        return {"k": None, "d": None, "signal": "insufficient_data"}

    # Get full RSI series
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(len(deltas))
    avg_loss = np.zeros(len(deltas))
    avg_gain[rsi_period - 1] = np.mean(gains[:rsi_period])
    avg_loss[rsi_period - 1] = np.mean(losses[:rsi_period])

    for i in range(rsi_period, len(deltas)):
        avg_gain[i] = (avg_gain[i - 1] * (rsi_period - 1) + gains[i]) / rsi_period
        avg_loss[i] = (avg_loss[i - 1] * (rsi_period - 1) + losses[i]) / rsi_period

    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))

    # Calculate Stochastic of RSI
    stoch_rsi = np.zeros(len(rsi))
    for i in range(stoch_period - 1, len(rsi)):
        rsi_window = rsi[i - stoch_period + 1:i + 1]
        rsi_min = np.min(rsi_window)
        rsi_max = np.max(rsi_window)
        if rsi_max - rsi_min != 0:
            stoch_rsi[i] = (rsi[i] - rsi_min) / (rsi_max - rsi_min) * 100
        else:
            stoch_rsi[i] = 50

    # Smooth %K
    k_line = np.convolve(stoch_rsi, np.ones(k_smooth) / k_smooth, mode='valid')

    # Calculate %D (signal line)
    d_line = np.convolve(k_line, np.ones(d_smooth) / d_smooth, mode='valid')

    current_k = round(float(k_line[-1]), 2) if len(k_line) > 0 else None
    current_d = round(float(d_line[-1]), 2) if len(d_line) > 0 else None

    # Signal interpretation
    signal = "neutral"
    if current_k is not None and current_d is not None:
        if current_k > 80 and current_d > 80:
            signal = "overbought"
        elif current_k < 20 and current_d < 20:
            signal = "oversold"
        elif current_k > current_d:
            signal = "bullish_crossover" if len(k_line) > 1 and k_line[-2] < d_line[-1] else "bullish"
        else:
            signal = "bearish_crossover" if len(k_line) > 1 and k_line[-2] > d_line[-1] else "bearish"

    return {
        "k": current_k,
        "d": current_d,
        "signal": signal,
        "periods": {"rsi": rsi_period, "stoch": stoch_period, "k_smooth": k_smooth, "d_smooth": d_smooth}
    }


def calculate_stochastic(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    k_period: int = 14,
    d_period: int = 3
) -> Dict[str, Any]:
    """
    Calculate Stochastic Oscillator (%K, %D)

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        k_period: %K period
        d_period: %D smoothing period

    Returns:
        Dict with %K, %D values and signal
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(closes) < k_period + d_period:
        return {"k": None, "d": None, "signal": "insufficient_data"}

    # Calculate %K
    k_values = np.zeros(len(closes))
    for i in range(k_period - 1, len(closes)):
        highest_high = np.max(highs[i - k_period + 1:i + 1])
        lowest_low = np.min(lows[i - k_period + 1:i + 1])
        if highest_high - lowest_low != 0:
            k_values[i] = (closes[i] - lowest_low) / (highest_high - lowest_low) * 100
        else:
            k_values[i] = 50

    # Calculate %D (SMA of %K)
    d_values = np.convolve(k_values, np.ones(d_period) / d_period, mode='valid')

    current_k = round(float(k_values[-1]), 2)
    current_d = round(float(d_values[-1]), 2) if len(d_values) > 0 else None

    # Signal
    signal = "neutral"
    if current_k > 80:
        signal = "overbought"
    elif current_k < 20:
        signal = "oversold"
    elif current_d and current_k > current_d:
        signal = "bullish"
    elif current_d:
        signal = "bearish"

    return {
        "k": current_k,
        "d": current_d,
        "signal": signal,
        "k_period": k_period,
        "d_period": d_period
    }


def calculate_cci(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20
) -> Dict[str, Any]:
    """
    Calculate Commodity Channel Index (CCI)

    CCI = (Typical Price - SMA of TP) / (0.015 × Mean Deviation)
    Typical Price = (High + Low + Close) / 3

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: CCI period (default 20)

    Returns:
        Dict with CCI value and interpretation
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(closes) < period:
        return {"value": None, "signal": "insufficient_data", "period": period}

    # Typical Price
    tp = (highs + lows + closes) / 3

    # SMA of Typical Price
    sma_tp = np.convolve(tp, np.ones(period) / period, mode='valid')

    # Mean Deviation
    mean_dev = np.zeros(len(sma_tp))
    for i in range(len(sma_tp)):
        mean_dev[i] = np.mean(np.abs(tp[i:i + period] - sma_tp[i]))

    # CCI calculation - guard against division by zero
    mean_dev = np.where(mean_dev == 0, 1e-10, mean_dev)
    cci = (tp[period - 1:] - sma_tp) / (0.015 * mean_dev)

    current_cci = round(float(cci[-1]), 2)

    # Interpretation
    if current_cci > 100:
        signal = "overbought"
    elif current_cci < -100:
        signal = "oversold"
    elif current_cci > 0:
        signal = "bullish"
    else:
        signal = "bearish"

    return {
        "value": current_cci,
        "signal": signal,
        "period": period,
        "interpretation": "OVERBOUGHT" if current_cci > 100 else ("OVERSOLD" if current_cci < -100 else "NEUTRAL")
    }


def calculate_williams_r(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> Dict[str, Any]:
    """
    Calculate Williams %R

    %R = (Highest High - Close) / (Highest High - Lowest Low) × -100

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: Williams %R period (default 14)

    Returns:
        Dict with Williams %R value and interpretation
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(closes) < period:
        return {"value": None, "signal": "insufficient_data", "period": period}

    # Calculate Williams %R
    williams_r = np.zeros(len(closes))
    for i in range(period - 1, len(closes)):
        highest_high = np.max(highs[i - period + 1:i + 1])
        lowest_low = np.min(lows[i - period + 1:i + 1])
        if highest_high - lowest_low != 0:
            williams_r[i] = ((highest_high - closes[i]) / (highest_high - lowest_low)) * -100
        else:
            williams_r[i] = -50

    current_wr = round(float(williams_r[-1]), 2)

    # Interpretation (note: Williams %R is inverted, -20 to 0 = overbought, -100 to -80 = oversold)
    if current_wr > -20:
        signal = "overbought"
    elif current_wr < -80:
        signal = "oversold"
    elif current_wr > -50:
        signal = "bullish"
    else:
        signal = "bearish"

    return {
        "value": current_wr,
        "signal": signal,
        "period": period,
        "interpretation": "OVERBOUGHT" if current_wr > -20 else ("OVERSOLD" if current_wr < -80 else "NEUTRAL")
    }


def calculate_awesome_oscillator(
    highs: np.ndarray,
    lows: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate Awesome Oscillator (AO)

    AO = SMA(Median Price, 5) - SMA(Median Price, 34)
    Median Price = (High + Low) / 2

    Args:
        highs: Array of high prices
        lows: Array of low prices

    Returns:
        Dict with AO value and interpretation
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)

    if len(highs) < 34:
        return {"value": None, "signal": "insufficient_data"}

    # Median Price
    median_price = (highs + lows) / 2

    # SMA 5 and SMA 34
    sma_5 = np.convolve(median_price, np.ones(5) / 5, mode='valid')
    sma_34 = np.convolve(median_price, np.ones(34) / 34, mode='valid')

    # Align arrays (sma_34 will be shorter)
    ao = sma_5[-(len(sma_34)):] - sma_34

    current_ao = round(float(ao[-1]), 4)
    prev_ao = round(float(ao[-2]), 4) if len(ao) > 1 else 0

    # Interpretation
    if current_ao > 0:
        if current_ao > prev_ao:
            signal = "bullish_increasing"
        else:
            signal = "bullish_decreasing"
    else:
        if current_ao < prev_ao:
            signal = "bearish_increasing"
        else:
            signal = "bearish_decreasing"

    return {
        "value": current_ao,
        "prev_value": prev_ao,
        "increasing": current_ao > prev_ao,
        "signal": signal,
        "interpretation": "BULLISH" if current_ao > 0 else "BEARISH"
    }


def calculate_macd(
    closes: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Dict[str, Any]:
    """
    Calculate MACD (Moving Average Convergence Divergence)

    Args:
        closes: Array of closing prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line EMA period

    Returns:
        Dict with MACD line, signal line, histogram
    """
    closes = np.array(closes, dtype=float)

    if len(closes) < slow_period + signal_period:
        return {
            "macd_line": None,
            "signal_line": None,
            "histogram": None,
            "signal": "insufficient_data"
        }

    # Calculate EMAs
    def ema(data, period):
        alpha = 2 / (period + 1)
        ema_vals = np.zeros(len(data))
        ema_vals[0] = data[0]
        for i in range(1, len(data)):
            ema_vals[i] = alpha * data[i] + (1 - alpha) * ema_vals[i - 1]
        return ema_vals

    fast_ema = ema(closes, fast_period)
    slow_ema = ema(closes, slow_period)

    # MACD line
    macd_line = fast_ema - slow_ema

    # Signal line (EMA of MACD)
    signal_line = ema(macd_line, signal_period)

    # Histogram
    histogram = macd_line - signal_line

    current_macd = round(float(macd_line[-1]), 4)
    current_signal = round(float(signal_line[-1]), 4)
    current_hist = round(float(histogram[-1]), 4)

    # Signal interpretation
    signal = "neutral"
    if current_hist > 0:
        if len(histogram) > 1 and histogram[-2] < 0:
            signal = "bullish_crossover"
        else:
            signal = "bullish"
    else:
        if len(histogram) > 1 and histogram[-2] > 0:
            signal = "bearish_crossover"
        else:
            signal = "bearish"

    return {
        "macd_line": current_macd,
        "signal_line": current_signal,
        "histogram": current_hist,
        "signal": signal,
        "periods": {"fast": fast_period, "slow": slow_period, "signal": signal_period}
    }


# =============================================================================
# TREND INDICATORS
# =============================================================================

def calculate_sma(closes: np.ndarray, period: int = 20) -> Dict[str, Any]:
    """
    Calculate Simple Moving Average
    """
    closes = np.array(closes, dtype=float)

    if len(closes) < period:
        return {"value": None, "period": period}

    sma = np.convolve(closes, np.ones(period) / period, mode='valid')

    return {
        "value": round(float(sma[-1]), 4),
        "period": period,
        "trend": "above" if closes[-1] > sma[-1] else "below"
    }


def calculate_ema(closes: np.ndarray, period: int = 20) -> Dict[str, Any]:
    """
    Calculate Exponential Moving Average
    """
    closes = np.array(closes, dtype=float)

    if len(closes) < period:
        return {"value": None, "period": period}

    alpha = 2 / (period + 1)
    ema = np.zeros(len(closes))
    ema[0] = closes[0]

    for i in range(1, len(closes)):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]

    return {
        "value": round(float(ema[-1]), 4),
        "period": period,
        "trend": "above" if closes[-1] > ema[-1] else "below"
    }


def calculate_adx(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> Dict[str, Any]:
    """
    Calculate Average Directional Index (ADX)

    Args:
        highs, lows, closes: Price arrays
        period: ADX period

    Returns:
        Dict with ADX, +DI, -DI values
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(closes) < period * 2:
        return {"adx": None, "plus_di": None, "minus_di": None, "signal": "insufficient_data"}

    # True Range
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )

    # +DM and -DM
    plus_dm = np.where(
        (highs[1:] - highs[:-1]) > (lows[:-1] - lows[1:]),
        np.maximum(highs[1:] - highs[:-1], 0),
        0
    )
    minus_dm = np.where(
        (lows[:-1] - lows[1:]) > (highs[1:] - highs[:-1]),
        np.maximum(lows[:-1] - lows[1:], 0),
        0
    )

    # Smoothed values using Wilder's method
    def wilder_smooth(data, period):
        smoothed = np.zeros(len(data))
        smoothed[period - 1] = np.sum(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i - 1] - smoothed[i - 1] / period + data[i]
        return smoothed

    atr = wilder_smooth(tr, period)
    plus_dm_smooth = wilder_smooth(plus_dm, period)
    minus_dm_smooth = wilder_smooth(minus_dm, period)

    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / np.where(atr != 0, atr, 1)
    minus_di = 100 * minus_dm_smooth / np.where(atr != 0, atr, 1)

    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, plus_di + minus_di, 1)
    adx = wilder_smooth(dx[period - 1:], period) / period

    current_adx = round(float(adx[-1]), 2) if len(adx) > 0 else None
    current_plus_di = round(float(plus_di[-1]), 2)
    current_minus_di = round(float(minus_di[-1]), 2)

    # Signal interpretation
    signal = "no_trend"
    if current_adx:
        if current_adx >= 25:
            if current_plus_di > current_minus_di:
                signal = "strong_uptrend"
            else:
                signal = "strong_downtrend"
        elif current_adx >= 20:
            signal = "weak_trend"
        else:
            signal = "no_trend"

    return {
        "adx": current_adx,
        "plus_di": current_plus_di,
        "minus_di": current_minus_di,
        "signal": signal,
        "trend_strength": "strong" if current_adx and current_adx >= 25 else "weak" if current_adx and current_adx >= 20 else "none"
    }


def calculate_parabolic_sar(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2
) -> Dict[str, Any]:
    """
    Calculate Parabolic SAR

    SAR(n+1) = SAR(n) + AF × (EP - SAR(n))
    AF starts at 0.02, increases by 0.02 each time EP is updated, max 0.2
    EP = Extreme Point (highest high in uptrend, lowest low in downtrend)

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        af_start: Initial acceleration factor (default 0.02)
        af_increment: AF increment (default 0.02)
        af_max: Maximum AF (default 0.2)

    Returns:
        Dict with SAR value and direction
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    length = len(closes)
    if length < 2:
        return {"value": None, "direction": None, "signal": "insufficient_data"}

    sar = np.zeros(length)
    direction = np.zeros(length, dtype=int)

    # Initialize - start with uptrend assumption
    sar[0] = lows[0]
    direction[0] = 1  # 1 = uptrend, -1 = downtrend
    af = af_start
    ep = highs[0]

    for i in range(1, length):
        if direction[i - 1] == 1:  # Uptrend
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            # SAR cannot be above prior two lows in uptrend
            sar[i] = min(sar[i], lows[i - 1])
            if i > 1:
                sar[i] = min(sar[i], lows[i - 2])

            if lows[i] < sar[i]:  # Reversal to downtrend
                direction[i] = -1
                sar[i] = ep
                ep = lows[i]
                af = af_start
            else:
                direction[i] = 1
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af_increment, af_max)
        else:  # Downtrend
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
            # SAR cannot be below prior two highs in downtrend
            sar[i] = max(sar[i], highs[i - 1])
            if i > 1:
                sar[i] = max(sar[i], highs[i - 2])

            if highs[i] > sar[i]:  # Reversal to uptrend
                direction[i] = 1
                sar[i] = ep
                ep = highs[i]
                af = af_start
            else:
                direction[i] = -1
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af_increment, af_max)

    current_sar = round(float(sar[-1]), 4)
    current_dir = int(direction[-1])
    trend_changed = direction[-1] != direction[-2] if length > 1 else False

    return {
        "value": current_sar,
        "direction": "bullish" if current_dir == 1 else "bearish",
        "trend_changed": trend_changed,
        "signal": "buy" if current_dir == 1 else "sell",
        "interpretation": "BUY" if current_dir == 1 else "SELL"
    }


def calculate_aroon(
    highs: np.ndarray,
    lows: np.ndarray,
    period: int = 25
) -> Dict[str, Any]:
    """
    Calculate Aroon Indicator

    Aroon Up = ((Period - Periods Since Highest High) / Period) × 100
    Aroon Down = ((Period - Periods Since Lowest Low) / Period) × 100

    Args:
        highs: Array of high prices
        lows: Array of low prices
        period: Aroon period (default 25)

    Returns:
        Dict with Aroon Up, Aroon Down, Oscillator and interpretation
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)

    if len(highs) < period + 1:
        return {"aroon_up": None, "aroon_down": None, "oscillator": None, "signal": "insufficient_data"}

    aroon_up = np.zeros(len(highs))
    aroon_down = np.zeros(len(lows))

    for i in range(period, len(highs)):
        high_window = highs[i - period:i + 1]
        low_window = lows[i - period:i + 1]

        # Find periods since highest high and lowest low
        periods_since_high = period - np.argmax(high_window)
        periods_since_low = period - np.argmin(low_window)

        aroon_up[i] = ((period - periods_since_high) / period) * 100
        aroon_down[i] = ((period - periods_since_low) / period) * 100

    current_up = round(float(aroon_up[-1]), 2)
    current_down = round(float(aroon_down[-1]), 2)
    oscillator = round(current_up - current_down, 2)

    # Interpretation
    if current_up > 70 and current_down < 30:
        signal = "strong_bullish"
    elif current_down > 70 and current_up < 30:
        signal = "strong_bearish"
    elif current_up > current_down:
        signal = "bullish"
    elif current_down > current_up:
        signal = "bearish"
    else:
        signal = "neutral"

    return {
        "aroon_up": current_up,
        "aroon_down": current_down,
        "oscillator": oscillator,
        "signal": signal,
        "period": period,
        "interpretation": "STRONG_BULLISH" if current_up > 70 and current_down < 30 else (
            "STRONG_BEARISH" if current_down > 70 and current_up < 30 else "NEUTRAL"
        )
    }


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

def calculate_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14
) -> Dict[str, Any]:
    """
    Calculate Average True Range (ATR)
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(closes) < period + 1:
        return {"value": None, "percent": None, "signal": "insufficient_data"}

    # True Range
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )

    # ATR using Wilder's smoothing
    atr = np.zeros(len(tr))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    current_atr = round(float(atr[-1]), 4)
    atr_percent = round((current_atr / closes[-1]) * 100, 2)

    return {
        "value": current_atr,
        "percent": atr_percent,
        "period": period,
        "volatility": "high" if atr_percent > 3 else "medium" if atr_percent > 1.5 else "low"
    }


def calculate_bollinger(
    closes: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0
) -> Dict[str, Any]:
    """
    Calculate Bollinger Bands
    """
    closes = np.array(closes, dtype=float)

    if len(closes) < period:
        return {"upper": None, "middle": None, "lower": None, "signal": "insufficient_data"}

    # Middle band (SMA)
    middle = np.convolve(closes, np.ones(period) / period, mode='valid')

    # Standard deviation
    std = np.zeros(len(middle))
    for i in range(len(middle)):
        std[i] = np.std(closes[i:i + period])

    # Upper and lower bands
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    current_price = closes[-1]
    current_upper = round(float(upper[-1]), 4)
    current_middle = round(float(middle[-1]), 4)
    current_lower = round(float(lower[-1]), 4)

    # Bandwidth and %B
    bandwidth = round((current_upper - current_lower) / current_middle * 100, 2)
    percent_b = round((current_price - current_lower) / (current_upper - current_lower) * 100, 2) if current_upper != current_lower else 50

    # Signal
    signal = "neutral"
    if current_price >= current_upper:
        signal = "overbought"
    elif current_price <= current_lower:
        signal = "oversold"
    elif current_price > current_middle:
        signal = "above_middle"
    else:
        signal = "below_middle"

    return {
        "upper": current_upper,
        "middle": current_middle,
        "lower": current_lower,
        "bandwidth": bandwidth,
        "percent_b": percent_b,
        "signal": signal,
        "period": period,
        "std_dev": std_dev
    }


def calculate_keltner_channels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0
) -> Dict[str, Any]:
    """
    Calculate Keltner Channels

    Middle = EMA(Close, 20)
    Upper = Middle + 2 × ATR(10)
    Lower = Middle - 2 × ATR(10)

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        ema_period: EMA period for middle line (default 20)
        atr_period: ATR period (default 10)
        multiplier: ATR multiplier (default 2.0)

    Returns:
        Dict with Keltner Channel values
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(closes) < max(ema_period, atr_period) + 1:
        return {"upper": None, "middle": None, "lower": None, "signal": "insufficient_data"}

    # Calculate EMA for middle line
    alpha = 2 / (ema_period + 1)
    ema = np.zeros(len(closes))
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]

    # Calculate ATR
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    atr = np.zeros(len(tr))
    atr[atr_period - 1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period

    # Keltner Channels (align arrays)
    middle = ema[1:]
    upper = middle + (multiplier * atr)
    lower = middle - (multiplier * atr)

    current_price = closes[-1]
    current_upper = round(float(upper[-1]), 4)
    current_middle = round(float(middle[-1]), 4)
    current_lower = round(float(lower[-1]), 4)

    # Signal
    if current_price > current_upper:
        signal = "overbought"
    elif current_price < current_lower:
        signal = "oversold"
    elif current_price > current_middle:
        signal = "above_middle"
    else:
        signal = "below_middle"

    return {
        "upper": current_upper,
        "middle": current_middle,
        "lower": current_lower,
        "signal": signal,
        "ema_period": ema_period,
        "atr_period": atr_period,
        "multiplier": multiplier,
        "interpretation": "OVERBOUGHT" if current_price > current_upper else (
            "OVERSOLD" if current_price < current_lower else "NEUTRAL"
        )
    }


def calculate_donchian_channels(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 20
) -> Dict[str, Any]:
    """
    Calculate Donchian Channels

    Upper = Highest High over N periods
    Lower = Lowest Low over N periods
    Middle = (Upper + Lower) / 2

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        period: Channel period (default 20)

    Returns:
        Dict with Donchian Channel values
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(highs) < period:
        return {"upper": None, "middle": None, "lower": None, "signal": "insufficient_data"}

    # Calculate channels
    upper = np.zeros(len(highs))
    lower = np.zeros(len(lows))
    middle = np.zeros(len(highs))

    for i in range(period - 1, len(highs)):
        upper[i] = np.max(highs[i - period + 1:i + 1])
        lower[i] = np.min(lows[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2

    current_price = closes[-1]
    current_upper = round(float(upper[-1]), 4)
    current_middle = round(float(middle[-1]), 4)
    current_lower = round(float(lower[-1]), 4)

    # Width calculation
    width = round((current_upper - current_lower) / current_middle, 4) if current_middle != 0 else 0

    # Signal - price near channel boundaries
    if current_price >= current_upper:
        signal = "breakout_high"
    elif current_price <= current_lower:
        signal = "breakout_low"
    elif current_price > current_middle:
        signal = "above_middle"
    else:
        signal = "below_middle"

    return {
        "upper": current_upper,
        "middle": current_middle,
        "lower": current_lower,
        "width": width,
        "signal": signal,
        "period": period,
        "interpretation": "BREAKOUT_HIGH" if current_price >= current_upper else (
            "BREAKOUT_LOW" if current_price <= current_lower else "NEUTRAL"
        )
    }


# =============================================================================
# COMPLEX INDICATORS
# =============================================================================

def calculate_ichimoku(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52
) -> Dict[str, Any]:
    """
    Calculate Ichimoku Cloud components
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(closes) < senkou_b_period:
        return {
            "tenkan": None, "kijun": None,
            "senkou_a": None, "senkou_b": None,
            "chikou": None, "signal": "insufficient_data"
        }

    def donchian_middle(h, l, period):
        """Calculate middle of Donchian channel"""
        result = np.zeros(len(h))
        for i in range(period - 1, len(h)):
            result[i] = (np.max(h[i - period + 1:i + 1]) + np.min(l[i - period + 1:i + 1])) / 2
        return result

    # Tenkan-sen (Conversion Line)
    tenkan = donchian_middle(highs, lows, tenkan_period)

    # Kijun-sen (Base Line)
    kijun = donchian_middle(highs, lows, kijun_period)

    # Senkou Span A (Leading Span A)
    senkou_a = (tenkan + kijun) / 2

    # Senkou Span B (Leading Span B)
    senkou_b = donchian_middle(highs, lows, senkou_b_period)

    # Chikou Span (Lagging Span) - current close shifted back 26 periods
    chikou = closes[-1]

    current_tenkan = round(float(tenkan[-1]), 4)
    current_kijun = round(float(kijun[-1]), 4)
    current_senkou_a = round(float(senkou_a[-1]), 4)
    current_senkou_b = round(float(senkou_b[-1]), 4)
    current_price = closes[-1]

    # Signal interpretation
    cloud_top = max(current_senkou_a, current_senkou_b)
    cloud_bottom = min(current_senkou_a, current_senkou_b)

    signal = "neutral"
    if current_price > cloud_top:
        if current_tenkan > current_kijun:
            signal = "strong_bullish"
        else:
            signal = "bullish"
    elif current_price < cloud_bottom:
        if current_tenkan < current_kijun:
            signal = "strong_bearish"
        else:
            signal = "bearish"
    else:
        signal = "in_cloud"

    return {
        "tenkan": current_tenkan,
        "kijun": current_kijun,
        "senkou_a": current_senkou_a,
        "senkou_b": current_senkou_b,
        "chikou": round(float(chikou), 4),
        "cloud_top": round(float(cloud_top), 4),
        "cloud_bottom": round(float(cloud_bottom), 4),
        "signal": signal,
        "price_vs_cloud": "above" if current_price > cloud_top else "below" if current_price < cloud_bottom else "inside"
    }


def calculate_vwap(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate Volume Weighted Average Price (VWAP)
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    volumes = np.array(volumes, dtype=float)

    if len(closes) < 2:
        return {"value": None, "signal": "insufficient_data"}

    # Typical price
    typical_price = (highs + lows + closes) / 3

    # VWAP
    cum_tp_vol = np.cumsum(typical_price * volumes)
    cum_vol = np.cumsum(volumes)
    vwap = cum_tp_vol / np.where(cum_vol != 0, cum_vol, 1)

    current_vwap = round(float(vwap[-1]), 4)
    current_price = closes[-1]

    return {
        "value": current_vwap,
        "price": round(float(current_price), 4),
        "deviation": round(float(current_price - current_vwap), 4),
        "deviation_percent": round((current_price - current_vwap) / current_vwap * 100, 2),
        "signal": "above_vwap" if current_price > current_vwap else "below_vwap"
    }


def calculate_obv(closes: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
    """
    Calculate On-Balance Volume (OBV)
    """
    closes = np.array(closes, dtype=float)
    volumes = np.array(volumes, dtype=float)

    if len(closes) < 2:
        return {"value": None, "signal": "insufficient_data"}

    # Calculate OBV
    obv = np.zeros(len(closes))
    obv[0] = volumes[0]

    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]

    # OBV trend (using 20-period SMA)
    if len(obv) >= 20:
        obv_sma = np.mean(obv[-20:])
        trend = "rising" if obv[-1] > obv_sma else "falling"
    else:
        trend = "unknown"

    return {
        "value": round(float(obv[-1]), 0),
        "trend": trend,
        "signal": "accumulation" if trend == "rising" else "distribution" if trend == "falling" else "neutral"
    }


def calculate_mfi(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 14
) -> Dict[str, Any]:
    """
    Calculate Money Flow Index (MFI)

    Typical Price = (High + Low + Close) / 3
    Raw Money Flow = TP × Volume
    Money Flow Ratio = Positive MF / Negative MF
    MFI = 100 - (100 / (1 + MFR))

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        volumes: Array of volumes
        period: MFI period (default 14)

    Returns:
        Dict with MFI value and interpretation
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    volumes = np.array(volumes, dtype=float)

    if len(closes) < period + 1:
        return {"value": None, "signal": "insufficient_data", "period": period}

    # Typical Price
    tp = (highs + lows + closes) / 3
    raw_money_flow = tp * volumes

    # Positive and Negative Money Flow
    positive_mf = np.zeros(len(tp))
    negative_mf = np.zeros(len(tp))

    for i in range(1, len(tp)):
        if tp[i] > tp[i - 1]:
            positive_mf[i] = raw_money_flow[i]
        elif tp[i] < tp[i - 1]:
            negative_mf[i] = raw_money_flow[i]
        # When TP unchanged: both stay 0 (neutral)

    # Sum over period
    mfi_values = np.zeros(len(tp))
    for i in range(period, len(tp)):
        positive_sum = np.sum(positive_mf[i - period + 1:i + 1])
        negative_sum = np.sum(negative_mf[i - period + 1:i + 1])

        if negative_sum == 0:
            mfi_values[i] = 100
        else:
            mf_ratio = positive_sum / negative_sum
            mfi_values[i] = 100 - (100 / (1 + mf_ratio))

    current_mfi = round(float(mfi_values[-1]), 2)

    # Interpretation
    if current_mfi > 80:
        signal = "overbought"
    elif current_mfi < 20:
        signal = "oversold"
    elif current_mfi > 50:
        signal = "bullish"
    else:
        signal = "bearish"

    return {
        "value": current_mfi,
        "signal": signal,
        "period": period,
        "interpretation": "OVERBOUGHT" if current_mfi > 80 else ("OVERSOLD" if current_mfi < 20 else "NEUTRAL")
    }


def calculate_cmf(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    period: int = 21
) -> Dict[str, Any]:
    """
    Calculate Chaikin Money Flow (CMF)

    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    Money Flow Volume = MFM × Volume
    CMF = Σ(MFV over period) / Σ(Volume over period)

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        volumes: Array of volumes
        period: CMF period (default 21)

    Returns:
        Dict with CMF value and interpretation
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    volumes = np.array(volumes, dtype=float)

    if len(closes) < period:
        return {"value": None, "signal": "insufficient_data", "period": period}

    # Money Flow Multiplier
    high_low_diff = highs - lows
    high_low_diff = np.where(high_low_diff == 0, 1e-10, high_low_diff)
    mfm = ((closes - lows) - (highs - closes)) / high_low_diff

    # Money Flow Volume
    mfv = mfm * volumes

    # CMF calculation (rolling sum)
    cmf_values = np.zeros(len(closes))
    for i in range(period - 1, len(closes)):
        mfv_sum = np.sum(mfv[i - period + 1:i + 1])
        vol_sum = np.sum(volumes[i - period + 1:i + 1])
        if vol_sum != 0:
            cmf_values[i] = mfv_sum / vol_sum

    current_cmf = round(float(cmf_values[-1]), 4)

    # Interpretation
    if current_cmf > 0.05:
        signal = "accumulation"
    elif current_cmf < -0.05:
        signal = "distribution"
    else:
        signal = "neutral"

    return {
        "value": current_cmf,
        "signal": signal,
        "period": period,
        "interpretation": "ACCUMULATION" if current_cmf > 0.05 else ("DISTRIBUTION" if current_cmf < -0.05 else "NEUTRAL")
    }


def calculate_ad_line(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate Accumulation/Distribution Line

    A/D = A/D(prev) + Money Flow Volume
    Money Flow Volume = ((Close - Low) - (High - Close)) / (High - Low) × Volume

    Args:
        highs: Array of high prices
        lows: Array of low prices
        closes: Array of closing prices
        volumes: Array of volumes

    Returns:
        Dict with A/D Line value and trend
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    volumes = np.array(volumes, dtype=float)

    if len(closes) < 2:
        return {"value": None, "signal": "insufficient_data"}

    # Money Flow Multiplier
    high_low_diff = highs - lows
    high_low_diff = np.where(high_low_diff == 0, 1e-10, high_low_diff)
    mfm = ((closes - lows) - (highs - closes)) / high_low_diff

    # Money Flow Volume
    mfv = mfm * volumes

    # A/D Line (cumulative sum)
    ad = np.cumsum(mfv)

    # Trend using 20-period comparison
    if len(ad) >= 20:
        ad_sma = np.mean(ad[-20:])
        trend = "rising" if ad[-1] > ad_sma else "falling"
    else:
        trend = "unknown"

    return {
        "value": int(ad[-1]),
        "trend": trend,
        "signal": "accumulation" if trend == "rising" else "distribution" if trend == "falling" else "neutral",
        "interpretation": "BULLISH" if trend == "rising" else "BEARISH"
    }


def calculate_supertrend(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 10,
    multiplier: float = 3.0
) -> Dict[str, Any]:
    """
    Calculate Supertrend indicator
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)

    if len(closes) < period + 1:
        return {"value": None, "direction": None, "signal": "insufficient_data"}

    # Calculate ATR
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )

    atr = np.zeros(len(tr))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    # Calculate bands
    hl2 = (highs[1:] + lows[1:]) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Supertrend
    supertrend = np.zeros(len(closes) - 1)
    direction = np.zeros(len(closes) - 1)

    supertrend[0] = upper_band[0]
    direction[0] = 1

    for i in range(1, len(supertrend)):
        if closes[i] > supertrend[i - 1]:
            supertrend[i] = lower_band[i]
            direction[i] = 1  # Bullish
        else:
            supertrend[i] = upper_band[i]
            direction[i] = -1  # Bearish

    current_st = round(float(supertrend[-1]), 4)
    current_dir = int(direction[-1])

    return {
        "value": current_st,
        "direction": "bullish" if current_dir == 1 else "bearish",
        "signal": "buy" if current_dir == 1 else "sell",
        "period": period,
        "multiplier": multiplier
    }


# =============================================================================
# FIBONACCI & SWING POINTS
# =============================================================================

def find_swing_points(
    highs: np.ndarray,
    lows: np.ndarray,
    left_bars: int = 5,
    right_bars: int = 5
) -> Dict[str, List[SwingPoint]]:
    """
    Find swing high and swing low points

    Args:
        highs: Array of high prices
        lows: Array of low prices
        left_bars: Number of bars to the left for confirmation
        right_bars: Number of bars to the right for confirmation

    Returns:
        Dict with lists of swing highs and swing lows
    """
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)

    swing_highs = []
    swing_lows = []

    for i in range(left_bars, len(highs) - right_bars):
        # Check for swing high
        is_swing_high = True
        for j in range(1, left_bars + 1):
            if highs[i] <= highs[i - j]:
                is_swing_high = False
                break
        if is_swing_high:
            for j in range(1, right_bars + 1):
                if highs[i] <= highs[i + j]:
                    is_swing_high = False
                    break

        if is_swing_high:
            swing_highs.append(SwingPoint(index=i, price=float(highs[i]), type="high"))

        # Check for swing low
        is_swing_low = True
        for j in range(1, left_bars + 1):
            if lows[i] >= lows[i - j]:
                is_swing_low = False
                break
        if is_swing_low:
            for j in range(1, right_bars + 1):
                if lows[i] >= lows[i + j]:
                    is_swing_low = False
                    break

        if is_swing_low:
            swing_lows.append(SwingPoint(index=i, price=float(lows[i]), type="low"))

    return {
        "swing_highs": swing_highs,
        "swing_lows": swing_lows,
        "last_swing_high": swing_highs[-1] if swing_highs else None,
        "last_swing_low": swing_lows[-1] if swing_lows else None
    }


def calculate_fibonacci(
    swing_high: float,
    swing_low: float,
    is_uptrend: bool = True
) -> Dict[str, Any]:
    """
    Calculate Fibonacci retracement and extension levels

    Args:
        swing_high: Swing high price
        swing_low: Swing low price
        is_uptrend: True for retracement in uptrend, False for downtrend

    Returns:
        Dict with Fibonacci levels
    """
    diff = swing_high - swing_low

    # Standard Fibonacci ratios
    fib_ratios = {
        "0.0": 0.0,
        "0.236": 0.236,
        "0.382": 0.382,
        "0.5": 0.5,
        "0.618": 0.618,
        "0.786": 0.786,
        "1.0": 1.0,
        "1.272": 1.272,
        "1.618": 1.618,
        "2.0": 2.0,
        "2.618": 2.618
    }

    levels = {}

    if is_uptrend:
        # Retracement levels from high
        for name, ratio in fib_ratios.items():
            levels[f"fib_{name.replace('.', '_')}"] = round(swing_high - (diff * ratio), 4)
    else:
        # Retracement levels from low
        for name, ratio in fib_ratios.items():
            levels[f"fib_{name.replace('.', '_')}"] = round(swing_low + (diff * ratio), 4)

    return {
        "swing_high": swing_high,
        "swing_low": swing_low,
        "trend": "uptrend" if is_uptrend else "downtrend",
        "levels": levels,
        "golden_pocket": {
            "upper": round(swing_high - (diff * 0.618), 4) if is_uptrend else round(swing_low + (diff * 0.618), 4),
            "lower": round(swing_high - (diff * 0.65), 4) if is_uptrend else round(swing_low + (diff * 0.65), 4)
        },
        "ote_zone": {  # Optimal Trade Entry
            "upper": round(swing_high - (diff * 0.618), 4) if is_uptrend else round(swing_low + (diff * 0.618), 4),
            "lower": round(swing_high - (diff * 0.786), 4) if is_uptrend else round(swing_low + (diff * 0.786), 4)
        }
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_all_indicators(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    indicators: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate multiple indicators at once

    Args:
        highs, lows, closes, volumes: OHLCV data
        indicators: List of indicators to calculate (None = all popular)

    Returns:
        Dict with all calculated indicators
    """
    # Default popular indicators
    if indicators is None:
        indicators = ["rsi", "macd", "sma", "ema", "bbands", "atr"]

    results = {}
    indicators_lower = [i.lower() for i in indicators]

    if "rsi" in indicators_lower:
        results["rsi"] = calculate_rsi(closes)

    if "stochrsi" in indicators_lower:
        results["stoch_rsi"] = calculate_stoch_rsi(closes)

    if "stochastic" in indicators_lower:
        results["stochastic"] = calculate_stochastic(highs, lows, closes)

    if "macd" in indicators_lower:
        results["macd"] = calculate_macd(closes)

    if "sma" in indicators_lower or "sma_20" in indicators_lower:
        results["sma_20"] = calculate_sma(closes, 20)
        results["sma_50"] = calculate_sma(closes, 50)
        results["sma_200"] = calculate_sma(closes, 200)

    if "ema" in indicators_lower or "ema_20" in indicators_lower:
        results["ema_20"] = calculate_ema(closes, 20)
        results["ema_50"] = calculate_ema(closes, 50)

    if "adx" in indicators_lower:
        results["adx"] = calculate_adx(highs, lows, closes)

    if "atr" in indicators_lower:
        results["atr"] = calculate_atr(highs, lows, closes)

    if "bbands" in indicators_lower or "bollinger" in indicators_lower:
        results["bollinger"] = calculate_bollinger(closes)

    if "ichimoku" in indicators_lower:
        results["ichimoku"] = calculate_ichimoku(highs, lows, closes)

    if "vwap" in indicators_lower:
        results["vwap"] = calculate_vwap(highs, lows, closes, volumes)

    if "obv" in indicators_lower:
        results["obv"] = calculate_obv(closes, volumes)

    if "supertrend" in indicators_lower:
        results["supertrend"] = calculate_supertrend(highs, lows, closes)

    if "fibonacci" in indicators_lower:
        swings = find_swing_points(highs, lows)
        if swings["last_swing_high"] and swings["last_swing_low"]:
            results["fibonacci"] = calculate_fibonacci(
                swings["last_swing_high"].price,
                swings["last_swing_low"].price,
                is_uptrend=swings["last_swing_high"].index > swings["last_swing_low"].index
            )
        else:
            results["fibonacci"] = {"error": "insufficient_swing_points"}

    return results
