import pandas as pd
from typing import Dict, Any

class PatternRecognition:
    """Tools for detecting common chart patterns."""

    @staticmethod
    def detect_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect common chart patterns in price data.

        Args:
            df (pd.DataFrame): Historical price data

        Returns:
            Dict[str, Any]: Detected patterns and their properties
        """
        try:
            if len(df) < 60:  # Need enough data for pattern detection
                return {
                    "patterns": [],
                    "message": "Not enough data for pattern detection",
                }

            patterns = []

            # We'll use a window of the most recent data for our analysis
            recent_df = df.tail(60).copy()

            # Find local minima and maxima
            recent_df["is_min"] = (
                recent_df["low"].rolling(window=5, center=True).min()
                == recent_df["low"]
            )
            recent_df["is_max"] = (
                recent_df["high"].rolling(window=5, center=True).max()
                == recent_df["high"]
            )

            # Get the indices, prices, and dates of local minima and maxima
            minima = recent_df[recent_df["is_min"]].copy()
            maxima = recent_df[recent_df["is_max"]].copy()

            # Double Bottom Detection
            if len(minima) >= 2:
                for i in range(len(minima) - 1):
                    for j in range(i + 1, len(minima)):
                        price1 = minima.iloc[i]["low"]
                        price2 = minima.iloc[j]["low"]
                        date1 = minima.iloc[i].name
                        date2 = minima.iloc[j].name

                        # Check if the two bottoms are at similar price levels (within 3%)
                        if abs(price1 - price2) / price1 < 0.03:
                            # Check if they're at least 10 days apart
                            days_apart = (date2 - date1).days
                            if days_apart >= 10 and days_apart <= 60:
                                # Check if there's a peak in between that's at least 5% higher
                                mask = (recent_df.index > date1) & (
                                    recent_df.index < date2
                                )
                                if mask.any():
                                    max_between = recent_df.loc[mask, "high"].max()
                                    if max_between > price1 * 1.05:
                                        patterns.append(
                                            {
                                                "type": "Double Bottom",
                                                "start_date": date1.strftime(
                                                    "%Y-%m-%d"
                                                ),
                                                "end_date": date2.strftime("%Y-%m-%d"),
                                                "price_level": round(
                                                    (price1 + price2) / 2, 2
                                                ),
                                                "confidence": "Medium",
                                            }
                                        )

            # Double Top Detection (similar logic, but for maxima)
            if len(maxima) >= 2:
                for i in range(len(maxima) - 1):
                    for j in range(i + 1, len(maxima)):
                        price1 = maxima.iloc[i]["high"]
                        price2 = maxima.iloc[j]["high"]
                        date1 = maxima.iloc[i].name
                        date2 = maxima.iloc[j].name

                        if abs(price1 - price2) / price1 < 0.03:
                            days_apart = (date2 - date1).days
                            if days_apart >= 10 and days_apart <= 60:
                                mask = (recent_df.index > date1) & (
                                    recent_df.index < date2
                                )
                                if mask.any():
                                    min_between = recent_df.loc[mask, "low"].min()
                                    if min_between < price1 * 0.95:
                                        patterns.append(
                                            {
                                                "type": "Double Top",
                                                "start_date": date1.strftime(
                                                    "%Y-%m-%d"
                                                ),
                                                "end_date": date2.strftime("%Y-%m-%d"),
                                                "price_level": round(
                                                    (price1 + price2) / 2, 2
                                                ),
                                                "confidence": "Medium",
                                            }
                                        )

            # Check for potential breakouts
            close = df["close"].iloc[-1]
            recent_high = df["high"].iloc[-20:].max()
            recent_low = df["low"].iloc[-20:].min()

            # Resistance breakout
            if close > recent_high * 0.99 and close < recent_high * 1.02:
                patterns.append(
                    {
                        "type": "Resistance Breakout",
                        "price_level": round(recent_high, 2),
                        "confidence": "Medium",
                    }
                )

            # Support breakout (breakdown)
            if close < recent_low * 1.01 and close > recent_low * 0.98:
                patterns.append(
                    {
                        "type": "Support Breakdown",
                        "price_level": round(recent_low, 2),
                        "confidence": "Medium",
                    }
                )

            return {"patterns": patterns}

        except Exception as e:
            raise Exception(f"Error detecting patterns: {str(e)}")