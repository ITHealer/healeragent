import pandas as pd
from typing import List, Dict, Any, Optional
from src.stock.analysis.technical_analysis import TechnicalAnalysis
from src.utils.logger.custom_logging import LoggerMixin

from src.stock.crawlers.market_data_provider import MarketData
from src.helpers.technical_analysis_llm_helper import TechnicalAnalysisLLMHelper
from src.providers.provider_factory import ModelProviderFactory, ProviderType

def convert_numpy_types(obj):
    """Convert numpy types to Python native types."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # (numpy.bool_, numpy.float64,...)
        return obj.item()
    else:
        return obj


def get_technical_analysis(symbol: str, data: List[dict]) -> Dict[str, Any]:
    df = pd.DataFrame(data)

    rename_cols = {
        "Date": "timestamp",
        "Open": "open",
        "High": "high", 
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    }
    
    # Rename columns in the data
    df = df.rename(columns={k: v for k, v in rename_cols.items() if k in df.columns})

    # Make sure the timestamp column is in datetime format
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif "Date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Date"], utc=True)

    df = TechnicalAnalysis.add_core_indicators(df)
    trend = TechnicalAnalysis.check_trend_status(df)

    # Detect Bollinger Bands patterns - NEW
    bb_patterns = TechnicalAnalysis.detect_bollinger_patterns(df)

    latest = df.iloc[-1]

    # Bollinger Bands analysis text - NEW
    bb_analysis = ""
    if 'bb_upper' in trend and trend['bb_upper'] is not None:
        bb_position = trend['bb_position']
        bb_analysis = f"""
    #### 📈 Bollinger Bands Analysis:
    - **Current Position**: {
        'Price above upper band - Overbought condition' if bb_position == 'above_upper' else
        'Price below lower band - Oversold condition' if bb_position == 'below_lower' else
        f'Price within bands - {trend["bb_percent"]:.1%} from lower band'
    }
    - **Bands**: Upper: ${trend['bb_upper']:.2f}, Middle: ${trend['bb_middle']:.2f}, Lower: ${trend['bb_lower']:.2f}
    - **Bandwidth**: {trend['bb_bandwidth']:.4f} - {
        'Very tight squeeze - Major move expected' if trend['bb_squeeze'] else
        'Wide bands - High volatility' if trend['bb_bandwidth'] > 0.1 else
        'Normal volatility'
    }
    - **Patterns Detected**: {
        f"W Bottom Pattern (Buy Signal) - {bb_patterns['w_bottom']['confidence']*100:.0f}% confidence" if bb_patterns.get('w_bottom') else
        f"M Top Pattern (Sell Signal) - {bb_patterns['m_top']['confidence']*100:.0f}% confidence" if bb_patterns.get('m_top') else
        f"Squeeze Breakout - {bb_patterns['squeeze_breakout']['direction'].title()} signal" if bb_patterns.get('squeeze_breakout') else
        "No significant patterns"
    }
        """
        
    analysis = f"""
    ### Technical Analysis for {symbol}

    #### 📊 Trend Analysis:
    - **Above 20 SMA**: {'✅' if trend['above_20sma'] else '❌'} - Price is {'above' if trend['above_20sma'] else 'below'} the 20-day simple moving average, indicating a short-term {'uptrend' if trend['above_20sma'] else 'downtrend'}.
    - **Above 50 SMA**: {'✅' if trend['above_50sma'] else '❌'} - Price is {'above' if trend['above_50sma'] else 'below'} the 50-day SMA, reflecting a medium-term {'bullish' if trend['above_50sma'] else 'bearish'} trend.
    - **Above 200 SMA**: {'✅' if trend['above_200sma'] else '❌'} - Price is {'above' if trend['above_200sma'] else 'below'} the 200-day SMA, suggesting a long-term {'uptrend' if trend['above_200sma'] else 'downtrend'}.
    - **20/50 SMA Bullish Cross**: {'✅' if trend['20_50_bullish'] else '❌'} - The 20-day SMA has {'crossed above' if trend['20_50_bullish'] else 'not yet crossed above'} the 50-day SMA, {('indicating a potential mid-term bullish signal' if trend['20_50_bullish'] else 'no crossover signal at this time')}.
    - **50/200 SMA Bullish Cross**: {'✅' if trend['50_200_bullish'] else '❌'} - The 50-day SMA has {'crossed above' if trend['50_200_bullish'] else 'not yet crossed above'} the 200-day SMA, {('Golden Cross – a strong bullish trend signal' if trend['50_200_bullish'] else 'Golden Cross not yet formed')}.

    #### ⚡ Momentum Indicators:
    - **RSI (14)**: {trend['rsi']:.2f} - {
        'Overbought (>70) - Potential pullback expected' if trend['rsi'] > 70 else 
        'Oversold (<30) - Possible short-term rebound' if trend['rsi'] < 30 else 
        'Neutral zone - Neither overbought nor oversold'}

    - **MACD Bullish**: {'✅' if trend['macd_bullish'] else '❌'} - MACD is {'above' if trend['macd_bullish'] else 'below'} the signal line, indicating {'positive' if trend['macd_bullish'] else 'negative'} momentum.

    {bb_analysis}

    #### 💰 Price & Volatility:
    - **Latest Price**: ${latest['close']:.2f}

    - **Average True Range (14)**: {latest['atr']:.2f} - {
        'High volatility - Greater trading opportunity but higher risk' if latest['atr'] > 5 else 
        'Moderate volatility - Balanced risk and opportunity' if 2 <= latest['atr'] <= 5 else
        'Low volatility - Consolidation phase, potential breakout ahead'}

    - **Average Daily Range %**: {latest['adrp']:.2f}% - {
        'Large daily range (>2%) - High intraday price movements' if latest['adrp'] > 2 else 
        'Moderate daily range (1-2%) - Typical market behavior' if 1 <= latest['adrp'] <= 2 else
        'Small daily range (<1%) - Sideways or range-bound market'}

    - **Average Volume (20D)**: {int(latest['avg_20d_vol']):,} shares/day - {
        'High liquidity - Suitable for large trades' if latest['avg_20d_vol'] > 5000000 else
        'Moderate liquidity - Adequate for most retail investors' if 1000000 <= latest['avg_20d_vol'] <= 5000000 else
        'Low liquidity - May face slippage for large orders'}
    """


    result = {
        "symbol": symbol,
        "trend": {
            "above_20sma": trend["above_20sma"],
            "above_50sma": trend["above_50sma"],
            "above_200sma": trend["above_200sma"],
            "20_50_bullish": trend["20_50_bullish"],
            "50_200_bullish": trend["50_200_bullish"]
        },
        "momentum": {
            "rsi": round(trend["rsi"], 2),
            "macd_bullish": trend["macd_bullish"]
        },
        "bollinger_bands": { 
            "upper": round(trend.get("bb_upper", 0), 2),
            "middle": round(trend.get("bb_middle", 0), 2),
            "lower": round(trend.get("bb_lower", 0), 2),
            "bandwidth": round(trend.get("bb_bandwidth", 0), 4),
            "percent_b": round(trend.get("bb_percent", 0), 4),
            "position": trend.get("bb_position", "unknown"),
            "squeeze": trend.get("bb_squeeze", False),
            "patterns": bb_patterns
        },
        "latest_price": round(latest["close"], 2),
        "atr": round(latest["atr"], 2),
        "adrp": round(latest["adrp"], 2),
        "avg_volume_20d": int(latest["avg_20d_vol"]),
        "analysis_summary": analysis
    }

    return convert_numpy_types(result)


async def get_technical_analysis_with_llm(
    symbol: str,
    model_name: str = "gpt-4.1-nano", 
    provider_type: str = "openai",
    user_question: Optional[str] = None,
    system_language: Optional[str] = None,
    lookback_days: int = 252,
    chat_history: Optional[str] = None 
) -> Dict[str, Any]:
    """
    Get technical analysis and interpretation using LLM

    Args:
        symbol: Stock code
        model_name: LLM model used
        provider_type: Provider type
        user_question: User question
        lookback_days: Number of days to get data

    Returns:
        Dict containing analysis and interpretation from LLM
    """
    try:
        # Get historical data
        market_data = MarketData()
        df = await market_data.get_historical_data_lookback_ver2(
            ticker=symbol,
            lookback_days=lookback_days
        )
        
        # Convert DataFrame to dict
        data_dict = df.reset_index().to_dict(orient="records")
        
        # Calculate technical analysis
        analysis_data = get_technical_analysis(symbol, data_dict)
        
        # Call LLM for analysis
        llm_helper = TechnicalAnalysisLLMHelper()
        
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(provider_type)
        
        llm_interpretation = await llm_helper.generate_analysis_with_llm(
            symbol=symbol,
            analysis_data=analysis_data,
            user_question=user_question,
            target_language=system_language,
            model_name=model_name,
            provider_type=provider_type,
            api_key=api_key,
            chat_history=chat_history
        )
        
        return {
            "symbol": symbol,
            "technical_data": analysis_data,
            "llm_interpretation": llm_interpretation,
            "lookback_days": lookback_days
        }
        
    except Exception as e:
        raise Exception(f"Error in technical analysis with LLM: {str(e)}")