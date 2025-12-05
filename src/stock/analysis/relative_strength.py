from typing import List, Dict

class RelativeStrength:
    """Tools for calculating relative strength metrics."""

    @staticmethod
    async def calculate_rs(
        market_data,
        symbol: str,
        benchmark: str = "SPY",
        lookback_periods: List[int] = [21, 63, 126, 252],
    ) -> Dict[str, float]:
        """
        Calculate relative strength compared to a benchmark across multiple timeframes.

        Args:
            market_data: Our market data fetcher instance
            symbol (str): The stock symbol to analyze
            benchmark (str): The benchmark symbol (default: SPY for S&P 500 ETF)
            lookback_periods (List[int]): Periods in trading days to calculate RS (default: [21, 63, 126, 252])

        Returns:
            Dict[str, float]: Relative strength scores for each timeframe
        """
        try:
            if market_data is None:
                raise ValueError("market_data is required for relative strength calculation")
    
            # Get data for both the stock and benchmark
            stock_df = await market_data.get_historical_data_lookback_ver2(
                symbol, max(lookback_periods) + 10
            )
            benchmark_df = await market_data.get_historical_data_lookback_ver2(
                benchmark, max(lookback_periods) + 10
            )
            print(stock_df.columns)
            print(benchmark_df.columns)
            # common_columns = set(stock_df.columns).intersection(set(benchmark_df.columns))
            # print(f"Common columns: {common_columns}")

            # Calculate returns for different periods
            rs_scores = {}

            if stock_df.empty and benchmark_df.empty:
                rs_text = f"No historical data found for both {symbol} and {benchmark}."
                return rs_scores, rs_text
            elif stock_df.empty:
                rs_text = f"No historical data found for {symbol}."
                return rs_scores, rs_text
            elif benchmark_df.empty:
                rs_text = f"No historical data found for {benchmark}."
                return rs_scores, rs_text

            for period in lookback_periods:
                # Check if we have enough data for this period
                if len(stock_df) <= period or len(benchmark_df) <= period:
                    # Skip this period if we don't have enough data
                    continue

                # Calculate the percent change for both
                stock_return = (
                    stock_df["Close"].iloc[-1] / stock_df["Close"].iloc[-period] - 1
                ) * 100
                benchmark_return = (
                    benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-period]
                    - 1
                ) * 100

                # Calculate relative strength (stock return minus benchmark return)
                relative_performance = stock_return - benchmark_return
            
                # Convert to a 1-100 score (this is simplified; in practice you might use a more
                # sophisticated distribution model based on historical data)
                rs_score = min(max(50 + relative_performance, 1), 99)

                rs_scores[f"RS_{period}d"] = round(rs_score, 2)
                rs_scores[f"Return_{period}d"] = round(stock_return, 2)
                rs_scores[f"Benchmark_{period}d"] = round(benchmark_return, 2)
                rs_scores[f"Excess_{period}d"] = round(relative_performance, 2)


            if not rs_scores:
                rs_text = f"Insufficient historical data to calculate relative strength metrics."
            else:
                rs_text = f"""
                #### 📈 Relative Strength Analysis vs {benchmark}:

                """
                for period, score in rs_scores.items():
                    if period.startswith("RS_"):
                        days = period.split("_")[1]
                        rs_text += f"- {days} Relative Strength: {score}"

                        if score >= 80:
                            rs_text += " (Strong Outperformance) ⭐⭐⭐"
                        elif score >= 65:
                            rs_text += " (Moderate Outperformance) ⭐⭐"
                        elif score >= 50:
                            rs_text += " (Slight Outperformance) ⭐"
                        elif score >= 35:
                            rs_text += " (Slight Underperformance) ⚠️"
                        elif score >= 20:
                            rs_text += " (Moderate Underperformance) ⚠️⚠️"
                        else:
                            rs_text += " (Strong Underperformance) ⚠️⚠️⚠️"

                        rs_text += "\n"

                rs_text += "\nPerformance Details:\n"
                for period in ["21d", "63d", "126d", "252d"]:
                    if (
                        f"Return_{period}" not in rs_scores
                        or f"Benchmark_{period}" not in rs_scores
                        or f"Excess_{period}" not in rs_scores
                    ):
                        continue

                    stock_return = rs_scores.get(f"Return_{period}")
                    benchmark_return = rs_scores.get(f"Benchmark_{period}")
                    excess = rs_scores.get(f"Excess_{period}")

                    if stock_return is not None and benchmark_return is not None and excess is not None:
                        rs_text += f"- {period}: {symbol} {stock_return:+.2f}% vs {benchmark} {benchmark_return:+.2f}% = {excess:+.2f}%\n"

                if "\nPerformance Details:\n" == rs_text.split("\n")[-2] + "\n":
                    rs_text += "No performance details available due to insufficient historical data.\n"
            
            return rs_scores, rs_text

        except Exception as e:
            raise Exception(f"Error calculating relative strength: {str(e)}")