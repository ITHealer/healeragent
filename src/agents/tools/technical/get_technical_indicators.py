"""
Enhanced getTechnicalIndicators - FIXED VERSION

File: src/agents/tools/technical/get_technical_indicators.py
"""

import httpx
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    create_success_output,
    create_error_output
)

# Import existing analysis classes
from src.stock.analysis.technical_analysis import TechnicalAnalysis


class GetTechnicalIndicatorsTool(BaseTool):
    """
    ENHANCED: Atomic tool với Bollinger Advanced Analysis
    
    FIXED: Correct usage of create_success_output and create_error_output
    """
    
    FMP_BASE_URL = "https://financialmodelingprep.com/api"
    DEFAULT_INDICATORS = ["RSI", "MACD", "SMA", "EMA", "BB", "ATR"]
    
    # Indicator aliases mapping
    INDICATOR_ALIASES = {
        "BOLLINGER": "BB",
        "BOLLINGER_BANDS": "BB",
        "BOLLINGERBANDS": "BB",
        "BOLL": "BB",
        "RSI_14": "RSI",
        "RSI14": "RSI",
        "SIMPLE_MOVING_AVERAGE": "SMA",
        "EXPONENTIAL_MOVING_AVERAGE": "EMA",
        "ATR_14": "ATR",
        "AVERAGE_TRUE_RANGE": "ATR"
    }
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        
        if api_key is None:
            import os
            api_key = os.environ.get("FMP_API_KEY")
        
        if not api_key:
            raise ValueError("FMP_API_KEY required")
        
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Define tool schema
        self.schema = ToolSchema(
            name="getTechnicalIndicators",
            category="technical",
            description=(
                "Calculate technical indicators (RSI, MACD, Moving Averages, Bollinger Bands) "
                "from historical price data. Returns indicator values with buy/sell signals. "
                "Supports multiple timeframes (1D to 1Y). "
                "Use when user asks for technical analysis, trading indicators, or chart analysis."
            ),
            capabilities=[
                "✅ RSI (Relative Strength Index) - Overbought/oversold detection",
                "✅ MACD (Moving Average Convergence Divergence) - Trend momentum",
                "✅ SMA & EMA (Simple/Exponential Moving Averages) - Trend direction",
                "✅ Bollinger Bands - Volatility analysis",
                "✅ Multiple timeframes (daily, weekly, monthly)",
                "✅ Buy/sell signal interpretation"
            ],
            limitations=[
                "❌ Requires minimum 50 data points for accurate calculations",
                "❌ Historical data may be delayed (15-min on free tier)",
                "❌ One symbol at a time",
                "❌ No real-time intraday indicators (use daily/weekly data)"
            ],
            usage_hints=[
                # English - Technical analysis requests
                "User asks: 'Apple technical indicators' → USE THIS with symbol=AAPL",
                "User asks: 'TSLA RSI' → USE THIS with symbol=TSLA, indicators=['RSI']",
                "User asks: 'Show me MACD for Microsoft' → USE THIS with symbol=MSFT, indicators=['MACD']",
                "User asks: 'Technical analysis of Amazon' → USE THIS with symbol=AMZN",
                "User asks: 'Is NVDA overbought?' → USE THIS with symbol=NVDA, indicators=['RSI']",
                
                # Vietnamese - Phân tích kỹ thuật
                "User asks: 'Chỉ báo kỹ thuật Apple' → USE THIS with symbol=AAPL",
                "User asks: 'RSI của Tesla' → USE THIS with symbol=TSLA, indicators=['RSI']",
                "User asks: 'Phân tích kỹ thuật Amazon' → USE THIS with symbol=AMZN",
                "User asks: 'NVDA có overbought không?' → USE THIS with symbol=NVDA, indicators=['RSI']",
                "User asks: 'MACD Microsoft' → USE THIS with symbol=MSFT, indicators=['MACD']",
                
                # Chinese - 技术分析
                "User asks: 'Apple技术指标' → USE THIS with symbol=AAPL",
                "User asks: '显示特斯拉RSI' → USE THIS with symbol=TSLA, indicators=['RSI']",
                
                # When NOT to use
                "User asks for CURRENT PRICE only → DO NOT USE (use getStockPrice)",
                "User asks about FUNDAMENTALS → DO NOT USE (use getFinancialRatios)",
                "User asks about NEWS → DO NOT USE (use getStockNews)",
                "User asks about MARKET overview → DO NOT USE (use getMarketIndices)"
            ],
            parameters=[
                ToolParameter(
                    name="symbol",
                    type="string",
                    description="Stock ticker symbol (e.g., AAPL, TSLA, NVDA)",
                    required=True,
                    pattern="^[A-Z]{1,7}$"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    description="Timeframe for analysis (1D, 5D, 1M, 3M, 6M, 1Y)",
                    required=False,
                    default="3M",
                    allowed_values=["1D", "5D", "1M", "3M", "6M", "1Y"]
                ),
                ToolParameter(
                    name="indicators",
                    type="array",
                    description="List of indicators to calculate",
                    required=False,
                    default=["RSI", "MACD", "SMA_20", "SMA_50", "EMA_12", "EMA_26"]
                )
            ],
            returns={
                "symbol": "string",
                "timeframe": "string",
                "indicators": "object - RSI, MACD, SMA, EMA values",
                "signals": "object - Buy/sell/hold signals",
                "current_price": "number",
                "data_points": "number",
                "timestamp": "string"
            },
            typical_execution_time_ms=1500,
            requires_symbol=True
        )

    async def execute(
        self,
        symbol: str,
        indicators: Optional[List[str]] = None,
        lookback_days: int = 200,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute technical indicators calculation with Bollinger advanced analysis
        
        Args:
            symbol: Stock symbol
            indicators: List of indicators to calculate (default: all)
            lookback_days: Days of historical data (default: 200, min: 60 for patterns)
            
        Returns:
            ToolOutput with indicator data and Bollinger patterns
        """
        start_time = datetime.now()
        symbol = symbol.upper()
        
        try:
            # ════════════════════════════════════════════════════════════
            # STEP 1: Normalize indicators using aliases
            # ════════════════════════════════════════════════════════════
            if indicators is None or len(indicators) == 0:
                indicators = self.DEFAULT_INDICATORS.copy()
            else:
                normalized = []
                for ind in indicators:
                    ind_upper = ind.upper().strip()
                    normalized_name = self.INDICATOR_ALIASES.get(ind_upper, ind_upper)
                    normalized.append(normalized_name)
                    
                    if normalized_name != ind_upper:
                        self.logger.info(
                            f"[INDICATOR ALIAS] Normalized '{ind}' → '{normalized_name}'"
                        )
                
                indicators = normalized
            
            self.logger.info(
                f"[getTechnicalIndicators] Executing: symbol={symbol}, "
                f"indicators={indicators}, lookback={lookback_days}"
            )
            
            # ════════════════════════════════════════════════════════════
            # STEP 2: Fetch historical data from FMP
            # ════════════════════════════════════════════════════════════
            historical_data = await self._fetch_historical_data(symbol, lookback_days)
            
            if not historical_data or len(historical_data) < 60:
                return create_error_output(
                    tool_name="getTechnicalIndicators",
                    error=f"Insufficient historical data for {symbol} (need 60+ days)",
                    metadata={
                        "symbol": symbol,
                        "lookback_days": lookback_days,
                        "data_points_received": len(historical_data) if historical_data else 0
                    }
                )
            
            # ════════════════════════════════════════════════════════════
            # STEP 3: Convert to DataFrame and prepare
            # ════════════════════════════════════════════════════════════
            df = pd.DataFrame(historical_data)
            df = df.rename(columns={
                "date": "timestamp",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })
            
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            df = df.reset_index(drop=True)
            
            self.logger.info(
                f"[{symbol}] Prepared DataFrame: {len(df)} rows, "
                f"date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
            )
            
            # ════════════════════════════════════════════════════════════
            # STEP 4: Calculate ALL core indicators using TechnicalAnalysis
            # ════════════════════════════════════════════════════════════
            df = TechnicalAnalysis.add_core_indicators(df)
            
            # ════════════════════════════════════════════════════════════
            # STEP 5: Build result dictionary
            # ════════════════════════════════════════════════════════════
            result_data = {
                "symbol": symbol,
                "current_price": float(df["close"].iloc[-1]),
                "timestamp": df["timestamp"].iloc[-1].isoformat()
            }
            
            latest = df.iloc[-1]
            
            # ════════════════════════════════════════════════════════════
            # STEP 6: Extract requested indicators
            # ════════════════════════════════════════════════════════════
            
            # ──────────────────────────────────────────────────────────
            # 🔧 FIX 1: Helper function to check indicator presence
            # ──────────────────────────────────────────────────────────
            def has_indicator(base_name: str, indicator_list: List[str]) -> bool:
                """
                Check if indicator or any variant is in the list.
                
                Examples:
                    has_indicator('RSI', ['RSI', 'MACD']) → True
                    has_indicator('SMA', ['SMA_20', 'SMA_50']) → True
                    has_indicator('EMA', ['EMA_12']) → True
                    has_indicator('BB', ['BOLLINGER', 'RSI']) → True (via alias)
                """
                base_upper = base_name.upper()
                
                for ind in indicator_list:
                    ind_upper = ind.upper()
                    
                    # Exact match
                    if ind_upper == base_upper:
                        return True
                    
                    # Starts with match (SMA_20 matches SMA)
                    if ind_upper.startswith(base_upper + '_'):
                        return True
                    
                    # Alias check for Bollinger Bands
                    if base_upper == 'BB' and ind_upper in ['BOLLINGER', 'BOLLINGER_BANDS']:
                        return True
                
                return False
            
            # ──────────────────────────────────────────────────────────
            # RSI
            # ──────────────────────────────────────────────────────────
            if has_indicator('RSI', indicators):
                rsi = float(latest["rsi"]) if pd.notna(latest["rsi"]) else None
                if rsi:
                    result_data["rsi_14"] = round(rsi, 2)
                    
                    if rsi > 70:
                        result_data["rsi_signal"] = "Overbought"
                    elif rsi < 30:
                        result_data["rsi_signal"] = "Oversold"
                    else:
                        result_data["rsi_signal"] = "Neutral"
            
            # ──────────────────────────────────────────────────────────
            # MACD
            # ──────────────────────────────────────────────────────────
            if has_indicator('MACD', indicators):
                macd_data = self._calculate_macd(df)
                result_data.update(macd_data)
            
            # ──────────────────────────────────────────────────────────
            # 🔧 FIX 2: SMA - Now properly detects SMA_20, SMA_50, etc.
            # ──────────────────────────────────────────────────────────
            if has_indicator('SMA', indicators):
                sma_data = self._calculate_sma(df)
                result_data.update(sma_data)
                
                self.logger.info(
                    f"[{symbol}] ✅ SMA calculated: "
                    f"SMA-20={sma_data.get('sma_20')}, "
                    f"SMA-50={sma_data.get('sma_50')}, "
                    f"SMA-200={sma_data.get('sma_200')}"
                )
            
            # ──────────────────────────────────────────────────────────
            # 🔧 FIX 3: EMA - Now properly detects EMA_12, EMA_26, etc.
            # ──────────────────────────────────────────────────────────
            if has_indicator('EMA', indicators):
                ema_data = self._calculate_ema(df)
                result_data.update(ema_data)
                
                self.logger.info(
                    f"[{symbol}] ✅ EMA calculated: "
                    f"EMA-12={ema_data.get('ema_12')}, "
                    f"EMA-26={ema_data.get('ema_26')}"
                )
            
            # ════════════════════════════════════════════════════════════
            # STEP 7: BOLLINGER BANDS WITH ADVANCED PATTERN DETECTION
            # ════════════════════════════════════════════════════════════
            if has_indicator('BB', indicators) or has_indicator('BOLLINGER', indicators):
                # Calculate basic Bollinger Bands
                bb_data = self._calculate_bollinger_bands(df)
                result_data.update(bb_data)
                
                self.logger.info(
                    f"[{symbol}] Bollinger Bands calculated: "
                    f"Upper={bb_data['bb_upper']}, Middle={bb_data['bb_middle']}, "
                    f"Lower={bb_data['bb_lower']}"
                )
                
                # ════════════════════════════════════════════════════════
                # NEW: Advanced Bollinger Pattern Detection
                # ════════════════════════════════════════════════════════
                # Only attempt if we have sufficient data (75+ days recommended)
                if len(df) >= 75:
                    try:
                        self.logger.info(
                            f"[{symbol}] Detecting Bollinger patterns "
                            f"(W-Bottom, M-Top, Squeeze)..."
                        )
                        
                        # Call pattern detection from TechnicalAnalysis
                        bb_patterns = TechnicalAnalysis.detect_bollinger_patterns(
                            df=df,
                            lookback=75  # Use 75 days for pattern detection
                        )
                        
                        if bb_patterns is None:
                            self.logger.warning(
                                f"[{symbol}] detect_bollinger_patterns() returned None"
                            )
                            bb_patterns = {}

                        def safe_get_pattern(pattern_dict, key):
                            """Safely get pattern that might be None"""
                            val = pattern_dict.get(key)
                            return val if isinstance(val, dict) else {}
                        
                        # Safely get patterns
                        w_bottom = safe_get_pattern(bb_patterns, "w_bottom")
                        m_top = safe_get_pattern(bb_patterns, "m_top")
                        squeeze = safe_get_pattern(bb_patterns, "squeeze_breakout")
                        
                        # Check if ANY pattern detected
                        has_patterns = any([
                            w_bottom.get("detected", False),
                            m_top.get("detected", False),
                            squeeze.get("detected", False)
                        ])
                        
                        if has_patterns:
                            # Initialize patterns container
                            result_data["bollinger_patterns"] = {}
                            
                            # ════════════════════════════════════════════════
                            # W-Bottom Pattern (Bullish Reversal)
                            # ════════════════════════════════════════════════
                            if w_bottom.get("detected"):
                                result_data["bollinger_patterns"]["w_bottom"] = {
                                    "detected": True,
                                    "confidence": w_bottom.get("confidence", 0.0),
                                    "signal": w_bottom.get("signal", "bullish"),
                                    "description": w_bottom.get("description", "W-Bottom pattern detected"),
                                    "first_bottom": w_bottom.get("first_bottom"),
                                    "second_bottom": w_bottom.get("second_bottom"),
                                    "breakout_confirmed": w_bottom.get("breakout_confirmed", False)
                                }
                                self.logger.info(
                                    f"[{symbol}] ✅ W-Bottom detected! "
                                    f"Confidence: {w_bottom.get('confidence', 0):.2f}"
                                )
                            else:
                                result_data["bollinger_patterns"]["w_bottom"] = {
                                    "detected": False,
                                    "confidence": 0.0
                                }
                            
                            # ════════════════════════════════════════════════
                            # M-Top Pattern (Bearish Reversal)
                            # ════════════════════════════════════════════════
                            if m_top.get("detected"):
                                result_data["bollinger_patterns"]["m_top"] = {
                                    "detected": True,
                                    "confidence": m_top.get("confidence", 0.0),
                                    "signal": m_top.get("signal", "bearish"),
                                    "description": m_top.get("description", "M-Top pattern detected"),
                                    "first_top": m_top.get("first_top"),
                                    "second_top": m_top.get("second_top"),
                                    "breakdown_confirmed": m_top.get("breakdown_confirmed", False)
                                }
                                
                                self.logger.info(
                                    f"[{symbol}] ⚠️ M-Top detected! "
                                    f"Confidence: {m_top.get('confidence', 0):.2f}, "
                                    f"Breakdown: {m_top.get('breakdown_confirmed', False)}"
                                )
                            else:
                                result_data["bollinger_patterns"]["m_top"] = {
                                    "detected": False,
                                    "confidence": 0.0
                                }
                            
                            # ════════════════════════════════════════════════
                            # Squeeze Breakout Pattern
                            # ════════════════════════════════════════════════
                            if squeeze.get("detected"):
                                result_data["bollinger_patterns"]["squeeze_breakout"] = {
                                    "detected": True,
                                    "confidence": squeeze.get("confidence", 0.0),
                                    "squeeze_active": squeeze.get("squeeze_active", False),
                                    "breakout_confirmed": squeeze.get("breakout_confirmed", False),
                                    "direction": squeeze.get("direction"),
                                    "signal": squeeze.get("signal"),
                                    "volume_confirmed": squeeze.get("volume_confirmed", False),
                                    "current_bandwidth": squeeze.get("current_bandwidth"),
                                    "description": squeeze.get("description", "Bollinger squeeze detected")
                                }
                                
                                if squeeze.get("breakout_confirmed"):
                                    self.logger.info(
                                        f"[{symbol}] 🚀 Squeeze BREAKOUT! "
                                        f"Direction: {squeeze.get('direction', 'unknown').upper()}, "
                                        f"Volume confirmed: {squeeze.get('volume_confirmed', False)}"
                                    )
                                else:
                                    self.logger.info(
                                        f"[{symbol}] ⏳ Squeeze ACTIVE - awaiting breakout"
                                    )
                            else:
                                result_data["bollinger_patterns"]["squeeze_breakout"] = {
                                    "detected": False,
                                    "confidence": 0.0
                                }
                            
                            self.logger.info(
                                f"[{symbol}] Pattern detection complete: "
                                f"W-Bottom={w_bottom.get('detected', False)}, "
                                f"M-Top={m_top.get('detected', False)}, "
                                f"Squeeze={squeeze.get('detected', False)}"
                            )
                        
                        else:
                            # No patterns detected
                            result_data["bollinger_patterns"] = {
                                "w_bottom": {"detected": False, "confidence": 0.0},
                                "m_top": {"detected": False, "confidence": 0.0},
                                "squeeze_breakout": {"detected": False, "confidence": 0.0}
                            }
                            
                            self.logger.info(
                                f"[{symbol}] No Bollinger patterns detected in current period"
                            )
                        
                        # ════════════════════════════════════════════════
                        # Add additional Bollinger metrics
                        # ════════════════════════════════════════════════
                        
                        # Bandwidth
                        if "bb_bandwidth" in latest and pd.notna(latest["bb_bandwidth"]):
                            result_data["bb_bandwidth"] = round(float(latest["bb_bandwidth"]), 4)
                        
                        # Squeeze status
                        if "bb_squeeze" in latest and pd.notna(latest["bb_squeeze"]):
                            result_data["bb_squeeze"] = bool(latest["bb_squeeze"])
                        
                        # Position relative to bands
                        try:
                            bb_position = TechnicalAnalysis._get_bb_position(latest)
                            result_data["bb_position"] = bb_position
                        except Exception as e:
                            self.logger.warning(f"[{symbol}] Could not calculate BB position: {e}")
                    
                    except Exception as e:
                        self.logger.error(
                            f"[{symbol}] Error detecting Bollinger patterns: {e}",
                            exc_info=True
                        )
                        
                        # Don't fail the entire tool - just skip patterns
                        result_data["bollinger_patterns"] = {
                            "error": "Pattern detection failed",
                            "details": str(e),
                            "w_bottom": {"detected": False, "confidence": 0.0},
                            "m_top": {"detected": False, "confidence": 0.0},
                            "squeeze_breakout": {"detected": False, "confidence": 0.0}
                        }
                
                else:
                    # Insufficient data for pattern detection
                    self.logger.warning(
                        f"[{symbol}] Insufficient data for pattern detection "
                        f"({len(df)} days, need 75+)"
                    )
                    
                    result_data["bollinger_patterns"] = {
                        "error": "Insufficient data for pattern detection",
                        "required_days": 75,
                        "available_days": len(df),
                        "w_bottom": {"detected": False, "confidence": 0.0},
                        "m_top": {"detected": False, "confidence": 0.0},
                        "squeeze_breakout": {"detected": False, "confidence": 0.0}
                    }
            
            # ════════════════════════════════════════════════════════════
            # STEP 8: ATR (Volatility)
            # ════════════════════════════════════════════════════════════
            if has_indicator('ATR', indicators):
                atr_data = self._calculate_atr(df)
                result_data.update(atr_data)
            
            # ════════════════════════════════════════════════════════════
            # STEP 9: Calculate execution time and return
            # ════════════════════════════════════════════════════════════
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # ──────────────────────────────────────────────────────────
            # 🔧 FIX 4: Enhanced logging for debugging
            # ──────────────────────────────────────────────────────────
            indicators_actually_calculated = []
            if 'rsi_14' in result_data:
                indicators_actually_calculated.append('RSI')
            if 'macd_line' in result_data:
                indicators_actually_calculated.append('MACD')
            if 'sma_20' in result_data:
                indicators_actually_calculated.append('SMA')
            if 'ema_12' in result_data:
                indicators_actually_calculated.append('EMA')
            if 'bb_upper' in result_data:
                indicators_actually_calculated.append('BB')
            if 'atr_14' in result_data:
                indicators_actually_calculated.append('ATR')
            
            self.logger.info(
                f"[getTechnicalIndicators] ✅ SUCCESS ({int(execution_time)}ms)"
            )
            self.logger.info(
                f"[{symbol}] Indicators calculated: {indicators_actually_calculated}"
            )
            self.logger.info(
                f"[{symbol}] Output keys: {list(result_data.keys())}"
            )
            
            return create_success_output(
                tool_name="getTechnicalIndicators",
                data=result_data,
                metadata={
                    "source": "atomic_tools",
                    "tool_name": "getTechnicalIndicators",
                    "symbols": [symbol],
                    "execution_time_ms": int(execution_time),
                    "lookback_days": lookback_days,
                    "indicators_calculated": indicators_actually_calculated,  # ✅ Actual indicators
                    "data_points": len(df),
                    "bollinger_patterns_enabled": has_indicator('BB', indicators),
                    "pattern_detection_status": "success" if "bollinger_patterns" in result_data else "skipped"
                }
            )
            
        except Exception as e:
            self.logger.error(
                f"[getTechnicalIndicators] Error for {symbol}: {e}",
                exc_info=True
            )
            
            return create_error_output(
                tool_name="getTechnicalIndicators",
                error=str(e),
                metadata={
                    "symbol": symbol,
                    "lookback_days": lookback_days,
                    "indicators_requested": indicators if indicators else []
                }
            )
    
    async def _fetch_historical_data(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch historical data from FMP"""
        url = f"{self.FMP_BASE_URL}/v3/historical-price-full/{symbol}"
        
        params = {
            "apikey": self.api_key,
            "timeseries": lookback_days
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if "historical" not in data:
                return []
            
            return data["historical"]
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD"""
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        current_histogram = float(histogram.iloc[-1])
        
        if current_histogram > 0:
            signal_text = "Bullish"
        else:
            signal_text = "Bearish"
        
        return {
            "macd_line": round(current_macd, 4),
            "macd_signal": round(current_signal, 4),
            "macd_histogram": round(current_histogram, 4),
            "macd_signal_text": signal_text
        }
    
    def _calculate_sma(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate SMAs"""
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        sma_200 = df['close'].rolling(window=200).mean()
        
        return {
            "sma_20": round(float(sma_20.iloc[-1]), 2),
            "sma_50": round(float(sma_50.iloc[-1]), 2),
            "sma_200": round(float(sma_200.iloc[-1]), 2)
        }
    
    def _calculate_ema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate EMAs"""
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        
        return {
            "ema_12": round(float(ema_12.iloc[-1]), 2),
            "ema_26": round(float(ema_26.iloc[-1]), 2)
        }
    
    def _calculate_bollinger_bands(
        self, 
        df: pd.DataFrame, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> Dict[str, Any]:
        """Calculate Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            "bb_upper": round(float(upper_band.iloc[-1]), 2),
            "bb_middle": round(float(sma.iloc[-1]), 2),
            "bb_lower": round(float(lower_band.iloc[-1]), 2)
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """Calculate ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return {
            "atr_14": round(float(atr.iloc[-1]), 2)
        }
    

    def _detect_bollinger_patterns(
        self,
        df: Any,  # pandas DataFrame
        bb_upper: np.ndarray,
        bb_middle: np.ndarray,
        bb_lower: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect Bollinger Bands patterns: W-Bottom, M-Top, Squeeze Breakout
        
        Args:
            df: DataFrame with OHLCV data
            bb_upper: Upper Bollinger Band
            bb_middle: Middle Bollinger Band (SMA 20)
            bb_lower: Lower Bollinger Band
            
        Returns:
            Dict with pattern detection results
        """
        patterns = {
            "w_bottom": self._detect_w_bottom(df, bb_middle, bb_lower),
            "m_top": self._detect_m_top(df, bb_middle, bb_upper),
            "squeeze_breakout": self._detect_squeeze_breakout(
                df, bb_upper, bb_middle, bb_lower
            )
        }
        
        return patterns
    
    def _detect_w_bottom(
        self,
        df: Any,
        bb_middle: np.ndarray,
        bb_lower: np.ndarray,
        lookback: int = 30
    ) -> Dict[str, Any]:
        """
        Detect W-Bottom pattern
        
        Pattern: 2 bottoms near lower band, with price breaking above middle band
        
        Bullish reversal signal
        """
        closes = df['close'].values
        lows = df['low'].values
        
        # Look for 2 lows in recent data
        recent_lows = []
        
        for i in range(len(closes) - lookback, len(closes) - 1):
            if i < 1:
                continue
            
            # Check if price is near lower band
            if lows[i] <= bb_lower[i] * 1.02:  # Within 2% of lower band
                # Check if it's a local minimum
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    recent_lows.append({
                        "index": i,
                        "price": lows[i],
                        "date": df.index[i].strftime("%Y-%m-%d") if hasattr(df.index[i], 'strftime') else str(df.index[i])
                    })
        
        # Need at least 2 lows
        if len(recent_lows) < 2:
            return {
                "detected": False,
                "confidence": 0.0
            }
        
        # Get last 2 lows
        first_low = recent_lows[-2]
        second_low = recent_lows[-1]
        
        # Check if price broke above middle band after second low
        broke_middle = False
        for i in range(second_low['index'] + 1, len(closes)):
            if closes[i] > bb_middle[i]:
                broke_middle = True
                break
        
        if broke_middle:
            # Calculate confidence based on pattern quality
            price_similarity = 1 - abs(first_low['price'] - second_low['price']) / first_low['price']
            confidence = min(0.95, price_similarity * 0.8)
            
            return {
                "detected": True,
                "confidence": round(confidence, 2),
                "first_bottom": {
                    "price": round(first_low['price'], 2),
                    "date": first_low['date']
                },
                "second_bottom": {
                    "price": round(second_low['price'], 2),
                    "date": second_low['date']
                },
                "breakout_confirmed": True,
                "signal": "bullish",
                "description": "W-Bottom pattern confirmed with middle band breakout"
            }
        
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": "No middle band breakout after second bottom"
        }
    
    def _detect_m_top(
        self,
        df: Any,
        bb_middle: np.ndarray,
        bb_upper: np.ndarray,
        lookback: int = 30
    ) -> Dict[str, Any]:
        """
        Detect M-Top pattern
        
        Pattern: 2 tops near upper band, with price breaking below middle band
        
        Bearish reversal signal
        """
        closes = df['close'].values
        highs = df['high'].values
        
        # Look for 2 highs in recent data
        recent_highs = []
        
        for i in range(len(closes) - lookback, len(closes) - 1):
            if i < 1:
                continue
            
            # Check if price is near upper band
            if highs[i] >= bb_upper[i] * 0.98:  # Within 2% of upper band
                # Check if it's a local maximum
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    recent_highs.append({
                        "index": i,
                        "price": highs[i],
                        "date": df.index[i].strftime("%Y-%m-%d") if hasattr(df.index[i], 'strftime') else str(df.index[i])
                    })
        
        # Need at least 2 highs
        if len(recent_highs) < 2:
            return {
                "detected": False,
                "confidence": 0.0
            }
        
        # Get last 2 highs
        first_high = recent_highs[-2]
        second_high = recent_highs[-1]
        
        # Check if price broke below middle band after second high
        broke_middle = False
        for i in range(second_high['index'] + 1, len(closes)):
            if closes[i] < bb_middle[i]:
                broke_middle = True
                break
        
        if broke_middle:
            # Calculate confidence based on pattern quality
            price_similarity = 1 - abs(first_high['price'] - second_high['price']) / first_high['price']
            confidence = min(0.95, price_similarity * 0.8)
            
            return {
                "detected": True,
                "confidence": round(confidence, 2),
                "first_top": {
                    "price": round(first_high['price'], 2),
                    "date": first_high['date']
                },
                "second_top": {
                    "price": round(second_high['price'], 2),
                    "date": second_high['date']
                },
                "breakdown_confirmed": True,
                "signal": "bearish",
                "description": "M-Top pattern confirmed with middle band breakdown"
            }
        
        return {
            "detected": False,
            "confidence": 0.0,
            "reason": "No middle band breakdown after second top"
        }
    
    def _detect_squeeze_breakout(
        self,
        df: Any,
        bb_upper: np.ndarray,
        bb_middle: np.ndarray,
        bb_lower: np.ndarray,
        squeeze_threshold: float = 0.05,
        squeeze_periods: int = 10
    ) -> Dict[str, Any]:
        """
        Detect Bollinger Bands Squeeze and Breakout
        
        Squeeze: Band width contracts below threshold for X periods
        Breakout: Price breaks out with volume increase
        
        Signal: High volatility expansion expected
        """
        closes = df['close'].values
        volumes = df['volume'].values if 'volume' in df.columns else None
        
        # Calculate bandwidth
        bandwidth = (bb_upper - bb_lower) / bb_middle
        
        # Check for squeeze (narrow bandwidth)
        recent_bandwidth = bandwidth[-squeeze_periods:]
        avg_bandwidth = np.mean(bandwidth)
        
        is_squeezed = np.mean(recent_bandwidth) < squeeze_threshold
        
        if not is_squeezed:
            return {
                "detected": False,
                "confidence": 0.0,
                "current_bandwidth": round(bandwidth[-1], 4),
                "reason": "No squeeze detected - bands not narrow enough"
            }
        
        # Check for breakout (price moved outside bands recently)
        breakout_direction = None
        breakout_index = None
        
        for i in range(len(closes) - 5, len(closes)):
            if closes[i] > bb_upper[i]:
                breakout_direction = "up"
                breakout_index = i
                break
            elif closes[i] < bb_lower[i]:
                breakout_direction = "down"
                breakout_index = i
                break
        
        if breakout_direction is None:
            return {
                "detected": True,
                "confidence": 0.5,
                "squeeze_active": True,
                "breakout_pending": True,
                "current_bandwidth": round(bandwidth[-1], 4),
                "description": "Squeeze active - awaiting breakout direction"
            }
        
        # Confirm with volume (if available)
        volume_confirmed = False
        if volumes is not None and breakout_index is not None:
            avg_volume = np.mean(volumes[-20:])
            breakout_volume = volumes[breakout_index]
            volume_confirmed = breakout_volume > avg_volume * 1.2
        
        confidence = 0.7 if volume_confirmed else 0.5
        
        return {
            "detected": True,
            "confidence": round(confidence, 2),
            "squeeze_active": False,
            "breakout_confirmed": True,
            "direction": breakout_direction,
            "volume_confirmed": volume_confirmed,
            "signal": "bullish" if breakout_direction == "up" else "bearish",
            "current_bandwidth": round(bandwidth[-1], 4),
            "description": f"Squeeze breakout {breakout_direction} with {'strong' if volume_confirmed else 'moderate'} volume"
        }