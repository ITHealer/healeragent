import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

class TechnicalAnalysis:
    """Technical analysis toolkit with improved performance and readability."""

    @staticmethod
    def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add a core set of technical indicators."""
        df = df.rename(columns=str.lower)

        try:
            # Adding trend indicators
            df["sma_20"] = ta.sma(df["close"], length=20)
            df["sma_50"] = ta.sma(df["close"], length=50)
            df["sma_200"] = ta.sma(df["close"], length=200)

            # Adding volatility indicators and volume
            daily_range = df["high"].sub(df["low"])
            adr = daily_range.rolling(window=20).mean()
            df["adrp"] = adr.div(df["close"]).mul(100)
            df["avg_20d_vol"] = df["volume"].rolling(window=20).mean()

            # Adding momentum indicators
            df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
            df["rsi"] = ta.rsi(df["close"], length=14)
            macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
            if macd is not None:
                df = pd.concat([df, macd], axis=1)

            # Adding Bollinger Bands - NEW
            bbands = ta.bbands(df["close"], length=20, std=2.0)
            if bbands is not None:
                df = pd.concat([df, bbands], axis=1)
                # Rename columns for clarity
                df.rename(columns={
                    'BBL_20_2.0': 'bb_lower',
                    'BBM_20_2.0': 'bb_middle', 
                    'BBU_20_2.0': 'bb_upper',
                    'BBB_20_2.0': 'bb_bandwidth',
                    'BBP_20_2.0': 'bb_percent'
                }, inplace=True)
                
                # Add squeeze indicator
                df['bb_squeeze'] = TechnicalAnalysis._detect_bollinger_squeeze(df)

            return df
        
        except KeyError as e:
            raise KeyError(f"Missing column in input DataFrame: {str(e)}")
        except Exception as e:
            raise Exception(f"Error calculating indicators: {str(e)}")

    @staticmethod
    def _detect_bollinger_squeeze(df: pd.DataFrame) -> pd.Series:
        """Detect Bollinger Bands squeeze conditions"""
        if 'bb_bandwidth' not in df.columns:
            return pd.Series([False] * len(df), index=df.index)
            
        # Average bandwidth over last 50 periods
        avg_bandwidth = df['bb_bandwidth'].rolling(50).mean()
        
        # Squeeze when current bandwidth < 50% of average
        squeeze = df['bb_bandwidth'] < (avg_bandwidth * 0.5)
        
        return squeeze

    @staticmethod
    def check_trend_status(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the current trend status including Bollinger Bands."""
        if df.empty:
            raise ValueError("DataFrame is empty. Ensure it contains valid data.")
        
        df = df.rename(columns=str.lower)
        latest = df.iloc[-1]
        
        trend_status = {
            "above_20sma": bool(latest["close"] > latest["sma_20"]) if pd.notna(latest["sma_20"]) else None,
            "above_50sma": bool(latest["close"] > latest["sma_50"]) if pd.notna(latest["sma_50"]) else None,
            "above_200sma": bool(latest["close"] > latest["sma_200"]) if pd.notna(latest["sma_200"]) else None,
            "20_50_bullish": bool(latest["sma_20"] > latest["sma_50"]) if pd.notna(latest["sma_20"]) and pd.notna(latest["sma_50"]) else None,
            "50_200_bullish": bool(latest["sma_50"] > latest["sma_200"]) if pd.notna(latest["sma_50"]) and pd.notna(latest["sma_200"]) else None,
            "rsi": float(latest["rsi"]) if pd.notna(latest["rsi"]) else None,
            "macd_bullish": bool(latest.get("MACD_12_26_9", 0) > latest.get("MACDs_12_26_9", 0)) if pd.notna(latest.get("MACD_12_26_9")) and pd.notna(latest.get("MACDs_12_26_9")) else None,
        }
        
        # Add Bollinger Bands status - NEW
        if 'bb_upper' in latest and pd.notna(latest['bb_upper']):
            trend_status.update({
                "bb_upper": float(latest["bb_upper"]),
                "bb_middle": float(latest["bb_middle"]),
                "bb_lower": float(latest["bb_lower"]),
                "bb_bandwidth": float(latest["bb_bandwidth"]) if pd.notna(latest["bb_bandwidth"]) else None,
                "bb_percent": float(latest["bb_percent"]) if pd.notna(latest["bb_percent"]) else None,
                "bb_squeeze": bool(latest["bb_squeeze"]) if pd.notna(latest["bb_squeeze"]) else False,
                "bb_position": TechnicalAnalysis._get_bb_position(latest)
            })
        
        return trend_status

    @staticmethod
    def _get_bb_position(latest: pd.Series) -> str:
        """Determine price position relative to Bollinger Bands"""
        if pd.isna(latest["close"]) or pd.isna(latest["bb_upper"]):
            return "unknown"
            
        if latest["close"] > latest["bb_upper"]:
            return "above_upper"
        elif latest["close"] < latest["bb_lower"]:
            return "below_lower"
        else:
            return "within_bands"

    @staticmethod
    def detect_bollinger_patterns(df: pd.DataFrame, lookback: int = 75) -> Dict[str, Any]:
        """Detect Bollinger Bands patterns including W bottom and M top"""
        patterns = {
            "w_bottom": None,
            "m_top": None,
            "squeeze_breakout": None
        }
        
        if len(df) < lookback:
            return patterns
            
        # Detect W bottom pattern
        w_pattern = TechnicalAnalysis._detect_w_bottom(df, lookback)
        if w_pattern:
            patterns["w_bottom"] = w_pattern
            
        # Detect M top pattern
        m_pattern = TechnicalAnalysis._detect_m_top(df, lookback)
        if m_pattern:
            patterns["m_top"] = m_pattern
            
        # Detect squeeze breakout
        squeeze_breakout = TechnicalAnalysis._detect_squeeze_breakout(df)
        if squeeze_breakout:
            patterns["squeeze_breakout"] = squeeze_breakout
            
        return patterns

    @staticmethod
    def _detect_w_bottom(df: pd.DataFrame, lookback: int) -> Optional[Dict[str, Any]]:
        """Detect W bottom pattern based on the algorithm in your code"""
        alpha = 0.0001  # Price tolerance
        
        i = len(df) - 1  # Current position
        
        # Condition 4: Price above upper band
        if df.iloc[i]['close'] > df.iloc[i]['bb_upper']:
            
            # Search for middle node (j) - Condition 2
            for j in range(i, max(i - lookback, 0), -1):
                if abs(df.iloc[j]['bb_middle'] - df.iloc[j]['close']) < alpha:
                    
                    # Search for first bottom (k) - Condition 1
                    for k in range(j, max(i - lookback, 0), -1):
                        if abs(df.iloc[k]['bb_lower'] - df.iloc[k]['close']) < alpha:
                            threshold = df.iloc[k]['close']
                            
                            # Search for second bottom (m) - Condition 3
                            for m in range(i, j, -1):
                                if (df.iloc[m]['close'] - df.iloc[m]['bb_lower'] < alpha) and \
                                   (df.iloc[m]['close'] > df.iloc[m]['bb_lower']) and \
                                   (df.iloc[m]['close'] < threshold):
                                    
                                    return {
                                        "detected": True,
                                        "confidence": 0.85,
                                        "coordinates": [k, j, m, i],
                                        "signal": "BUY",
                                        "description": "W bottom pattern detected - bullish reversal signal"
                                    }
        
        return None

    @staticmethod
    def _detect_m_top(df: pd.DataFrame, lookback: int) -> Optional[Dict[str, Any]]:
        """Detect M top pattern (inverse of W bottom)"""
        alpha = 0.0001
        
        i = len(df) - 1
        
        # Price below lower band
        if df.iloc[i]['close'] < df.iloc[i]['bb_lower']:
            
            # Similar logic but inverted
            for j in range(i, max(i - lookback, 0), -1):
                if abs(df.iloc[j]['bb_middle'] - df.iloc[j]['close']) < alpha:
                    
                    for k in range(j, max(i - lookback, 0), -1):
                        if abs(df.iloc[k]['bb_upper'] - df.iloc[k]['close']) < alpha:
                            threshold = df.iloc[k]['close']
                            
                            for m in range(i, j, -1):
                                if (df.iloc[m]['bb_upper'] - df.iloc[m]['close'] < alpha) and \
                                   (df.iloc[m]['close'] < df.iloc[m]['bb_upper']) and \
                                   (df.iloc[m]['close'] > threshold):
                                    
                                    return {
                                        "detected": True,
                                        "confidence": 0.80,
                                        "coordinates": [k, j, m, i],
                                        "signal": "SELL",
                                        "description": "M top pattern detected - bearish reversal signal"
                                    }
        
        return None

    @staticmethod
    def _detect_squeeze_breakout(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Detect squeeze breakout conditions"""
        if len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        prev_5 = df.iloc[-5:-1]
        
        # Check if we were in squeeze and now breaking out
        if prev_5['bb_squeeze'].any() and not latest['bb_squeeze']:
            # Determine direction
            if latest['close'] > latest['bb_middle']:
                direction = "bullish"
                signal = "BUY"
            else:
                direction = "bearish" 
                signal = "SELL"
                
            return {
                "detected": True,
                "direction": direction,
                "signal": signal,
                "bandwidth_expansion": float(latest['bb_bandwidth'] / prev_5['bb_bandwidth'].mean()),
                "description": f"Bollinger squeeze breakout - {direction} signal"
            }
            
        return None



# import pandas as pd
# import pandas_ta as ta
# from typing import Dict, Any

# class TechnicalAnalysis:
#     """Technical analysis toolkit with improved performance and readability."""

#     @staticmethod
#     def add_core_indicators(df: pd.DataFrame) -> pd.DataFrame:
#         """Add a core set of technical indicators."""

#         df = df.rename(columns=str.lower)

#         try:
#             # Adding trend indicators
#             df["sma_20"] = ta.sma(df["close"], length=20)
#             df["sma_50"] = ta.sma(df["close"], length=50)
#             df["sma_200"] = ta.sma(df["close"], length=200)

#             # Adding volatility indicators and volume
#             daily_range = df["high"].sub(df["low"])
#             adr = daily_range.rolling(window=20).mean()
#             df["adrp"] = adr.div(df["close"]).mul(100)
#             df["avg_20d_vol"] = df["volume"].rolling(window=20).mean()

#             # Adding momentum indicators
#             df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
#             df["rsi"] = ta.rsi(df["close"], length=14)
#             macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
#             if macd is not None:
#                 df = pd.concat([df, macd], axis=1)

#             return df
        
#         except KeyError as e:
#             raise KeyError(f"Missing column in input DataFrame: {str(e)}")
#         except Exception as e:
#             raise Exception(f"Error calculating indicators: {str(e)}")

#     @staticmethod
#     def check_trend_status(df: pd.DataFrame) -> Dict[str, Any]:
#         """Analyze the current trend status."""
#         if df.empty:
#             raise ValueError("DataFrame is empty. Ensure it contains valid data.")
        
#         df = df.rename(columns=str.lower)
#         latest = df.iloc[-1]
        
#         return {
#             "above_20sma": bool(latest["close"] > latest["sma_20"]) if pd.notna(latest["sma_20"]) else None,
#             "above_50sma": bool(latest["close"] > latest["sma_50"]) if pd.notna(latest["sma_50"]) else None,
#             "above_200sma": bool(latest["close"] > latest["sma_200"]) if pd.notna(latest["sma_200"]) else None,
#             "20_50_bullish": bool(latest["sma_20"] > latest["sma_50"]) if pd.notna(latest["sma_20"]) and pd.notna(latest["sma_50"]) else None,
#             "50_200_bullish": bool(latest["sma_50"] > latest["sma_200"]) if pd.notna(latest["sma_50"]) and pd.notna(latest["sma_200"]) else None,
#             "rsi": float(latest["rsi"]) if pd.notna(latest["rsi"]) else None,
#             "macd_bullish": bool(latest.get("MACD_12_26_9", 0) > latest.get("MACDs_12_26_9", 0)) if pd.notna(latest.get("MACD_12_26_9")) and pd.notna(latest.get("MACDs_12_26_9")) else None,
#         }