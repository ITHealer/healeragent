import pandas as pd
from typing import Dict, Any

class RiskAnalysis:
    """Tools for risk management and position sizing."""

    @staticmethod
    def calculate_position_size(
        price: float,
        stop_price: float,
        risk_amount: float,
        account_size: float,
        max_risk_percent: float = 2.0,
    ) -> Dict[str, Any]:
        """
        Calculate appropriate position size based on risk parameters.

        Args:
            price (float): Current stock price
            stop_price (float): Stop loss price
            risk_amount (float): Amount willing to risk in dollars
            account_size (float): Total trading account size
            max_risk_percent (float): Maximum percentage of account to risk

        Returns:
            Dict[str, Any]: Position sizing recommendations
        """
        try:
            # Validate inputs
            if price <= 0 or account_size <= 0:
                raise ValueError("Price and account size must be positive")

            if price <= stop_price and stop_price != 0:
                raise ValueError(
                    "For long positions, stop price must be below entry price"
                )

            # Calculate risk per share
            risk_per_share = abs(price - stop_price)

            if risk_per_share == 0:
                raise ValueError(
                    "Risk per share cannot be zero. Entry and stop prices must differ."
                )

            # Calculate position size based on dollar risk
            shares_based_on_risk = int(risk_amount / risk_per_share)

            # Calculate maximum position size based on account risk percentage
            max_risk_dollars = account_size * (max_risk_percent / 100)
            max_shares = int(max_risk_dollars / risk_per_share)

            # Take the smaller of the two
            recommended_shares = min(shares_based_on_risk, max_shares)
            actual_dollar_risk = recommended_shares * risk_per_share

            # Calculate position cost
            position_cost = recommended_shares * price

            # Calculate R-Multiples (potential reward to risk ratios)
            r1_target = price + risk_per_share
            r2_target = price + 2 * risk_per_share
            r3_target = price + 3 * risk_per_share

            return {
                "recommended_shares": recommended_shares,
                "dollar_risk": round(actual_dollar_risk, 2),
                "risk_per_share": round(risk_per_share, 2),
                "position_cost": round(position_cost, 2),
                "account_percent_risked": round(
                    (actual_dollar_risk / account_size) * 100, 2
                ),
                "r_multiples": {
                    "r1": round(r1_target, 2),
                    "r2": round(r2_target, 2),
                    "r3": round(r3_target, 2),
                },
            }

        except Exception as e:
            raise Exception(f"Error calculating position size: {str(e)}")

    @staticmethod
    def suggest_stop_levels(df: pd.DataFrame) -> Dict[str, float]:
        """
        Suggest appropriate stop-loss levels based on technical indicators.

        Args:
            df (pd.DataFrame): Historical price data with technical indicators

        Returns:
            Dict[str, float]: Suggested stop levels
        """
        try:
            if len(df) < 20:
                raise ValueError("Not enough data for stop level analysis")

            latest = df.iloc[-1]
            close = latest["close"]

            # Calculate ATR-based stops
            atr = latest.get("atr", df["high"].iloc[-20:] - df["low"].iloc[-20:]).mean()

            # Different stop strategies
            stops = {
                "atr_1x": round(close - 1 * atr, 2),
                "atr_2x": round(close - 2 * atr, 2),
                "atr_3x": round(close - 3 * atr, 2),
                "percent_2": round(close * 0.98, 2),
                "percent_5": round(close * 0.95, 2),
                "percent_8": round(close * 0.92, 2),
            }

            # Add SMA-based stops if available
            for sma in ["sma_20", "sma_50", "sma_200"]:
                if sma in latest and not pd.isna(latest[sma]):
                    stops[sma] = round(latest[sma], 2)

            # Check for recent swing low
            recent_lows = df["low"].iloc[-20:].sort_values()
            if not recent_lows.empty:
                stops["recent_swing"] = round(recent_lows.iloc[0], 2)

            return stops

        except Exception as e:
            raise Exception(f"Error suggesting stop levels: {str(e)}")