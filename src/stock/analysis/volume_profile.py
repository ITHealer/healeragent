import pandas as pd
from typing import Dict, Any

class VolumeProfile:
    """Tools for analyzing volume distribution by price."""

    @staticmethod
    def analyze_volume_profile(df: pd.DataFrame, num_bins: int = 10) -> Dict[str, Any]:
        """
        Create a volume profile analysis by price level.

        Args:
            df (pd.DataFrame): Historical price and volume data
            num_bins (int): Number of price bins to create (default: 10)

        Returns:
            Dict[str, Any]: Volume profile analysis
        """
        try:
            if len(df) < 20:
                raise ValueError("Not enough data for volume profile analysis")

            # Find the price range for the period
            price_min = df["low"].min()
            price_max = df["high"].max()

            # Create price bins
            bin_width = (price_max - price_min) / num_bins

            # Initialize the profile
            profile = {
                "price_min": price_min,
                "price_max": price_max,
                "bin_width": bin_width,
                "bins": [],
            }

            # Calculate volume by price bin
            for i in range(num_bins):
                bin_low = price_min + i * bin_width
                bin_high = bin_low + bin_width
                bin_mid = (bin_low + bin_high) / 2

                # Filter data in this price range
                mask = (df["low"] <= bin_high) & (df["high"] >= bin_low)
                volume_in_bin = df.loc[mask, "volume"].sum()

                # Calculate percentage of total volume
                volume_percent = (
                    (volume_in_bin / df["volume"].sum()) * 100
                    if df["volume"].sum() > 0
                    else 0
                )

                profile["bins"].append(
                    {
                        "price_low": round(bin_low, 2),
                        "price_high": round(bin_high, 2),
                        "price_mid": round(bin_mid, 2),
                        "volume": int(volume_in_bin),
                        "volume_percent": round(volume_percent, 2),
                    }
                )

            # Find the Point of Control (POC) - the price level with the highest volume
            poc_bin = max(profile["bins"], key=lambda x: x["volume"])
            profile["point_of_control"] = round(poc_bin["price_mid"], 2)

            # Find Value Area (70% of volume)
            sorted_bins = sorted(
                profile["bins"], key=lambda x: x["volume"], reverse=True
            )
            cumulative_volume = 0
            value_area_bins = []

            for bin_data in sorted_bins:
                value_area_bins.append(bin_data)
                cumulative_volume += bin_data["volume_percent"]
                if cumulative_volume >= 70:
                    break

            if value_area_bins:
                profile["value_area_low"] = round(
                    min([b["price_low"] for b in value_area_bins]), 2
                )
                profile["value_area_high"] = round(
                    max([b["price_high"] for b in value_area_bins]), 2
                )

            return profile

        except Exception as e:
            raise Exception(f"Error analyzing volume profile: {str(e)}")