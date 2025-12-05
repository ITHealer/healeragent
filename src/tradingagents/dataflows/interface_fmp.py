import os
import asyncio
from stockstats import wrap
from tqdm import tqdm
from typing import Annotated
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from openai import OpenAI
from src.tradingagents.dataflows.fmp_data import fmp_data
from src.tradingagents.dataflows.config import get_config, set_config, DATA_DIR
from src.stock.crawlers.market_data_provider import MarketData
from src.tradingagents.dataflows.stockstats_utils import StockstatsUtils

from src.tradingagents.dataflows.reddit_utils import fetch_top_from_category
from src.tradingagents.dataflows.googlenews_utils import *

# ================= NEWS TOOLS =================

def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


def get_global_news_openai(curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search global or macroeconomics news from 7 days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_fmp_news(
    ticker: Annotated[str, "Search query of a company, e.g. 'AAPL, TSM, etc.'"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"]
):
    """
    Retrieve news using FMP instead of Finnhub
    """
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before_str = before.strftime("%Y-%m-%d")
    
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        data = loop.run_until_complete(
            fmp_data.get_company_news(ticker, before_str, curr_date)
        )
        print(f"News {data}")
    finally:
        loop.close()
    
    if not data:
        return f"No news found for {ticker} from {before_str} to {curr_date}"
    
    # Format output similar to original
    result_str = f"## {ticker} finnhub news from {before_str} to {curr_date}:\n\n"
    
    for date, news_list in sorted(data.items(), reverse=True):
        for news in news_list:
            result_str += f"### Date: {date}\n"
            result_str += f"**Headline:** {news['headline']}\n"
            result_str += f"**Summary:** {news['summary']}\n"
            result_str += f"**Source:** {news['source']}\n"
            result_str += f"**URL:** {news['url']}\n\n"
    
    print(f"News final {result_str}")
    return result_str


# ================= SOCIAL MEDIA TOOLS =================
# def get_reddit_company_news(
#     ticker: Annotated[str, "ticker symbol of the company"],
#     start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
#     look_back_days: Annotated[int, "how many days to look back"],
#     max_limit_per_day: Annotated[int, "Maximum number of news per day"],
# ) -> str:
#     """
#     Retrieve the latest top reddit news
#     Args:
#         ticker: ticker symbol of the company
#         start_date: Start date in yyyy-mm-dd format
#         end_date: End date in yyyy-mm-dd format
#     Returns:
#         str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
#     """

#     start_date = datetime.strptime(start_date, "%Y-%m-%d")
#     before = start_date - relativedelta(days=look_back_days)
#     before = before.strftime("%Y-%m-%d")

#     posts = []
#     # iterate from start_date to end_date
#     curr_date = datetime.strptime(before, "%Y-%m-%d")

#     total_iterations = (start_date - curr_date).days + 1
#     pbar = tqdm(
#         desc=f"Getting Company News for {ticker} on {start_date}",
#         total=total_iterations,
#     )

#     while curr_date <= start_date:
#         curr_date_str = curr_date.strftime("%Y-%m-%d")
#         fetch_result = fetch_top_from_category(
#             "company_news",
#             curr_date_str,
#             max_limit_per_day,
#             ticker,
#             data_path=os.path.join(DATA_DIR, "reddit_data"),
#         )
#         posts.extend(fetch_result)
#         curr_date += relativedelta(days=1)

#         pbar.update(1)

#     pbar.close()

#     if len(posts) == 0:
#         return ""

#     news_str = ""
#     for post in posts:
#         if post["content"] == "":
#             news_str += f"### {post['title']}\n\n"
#         else:
#             news_str += f"### {post['title']}\n\n{post['content']}\n\n"

#     return f"##{ticker} News Reddit, from {before} to {curr_date}:\n\n{news_str}"


## Using search tool of openai (not sp 4.1-nano))
def get_stock_news_openai(ticker, curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Social Media for {ticker} from 7 days before {curr_date} to {curr_date}? Make sure you only get the data posted during that period.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


# ================= FUNDAMENTAL & SENTIMENT TOOLS =================
def get_finnhub_company_insider_sentiment(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"]
):
    """
    Retrieve insider sentiment using FMP data
    """
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before_str = before.strftime("%Y-%m-%d")

    print(f"[DEBUG] Range: {before_str} → {curr_date}  (look_back_days={look_back_days})")
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        data = loop.run_until_complete(
            fmp_data.get_insider_sentiment(ticker, before_str, curr_date)
        )
    finally:
        loop.close()
    
    if not data:
        return f"No insider sentiment data found for {ticker}"
    
    # Format output
    result_str = f"## {ticker} insider sentiment from {before_str} to {curr_date}:\n\n"
    
    for date, sentiment_list in sorted(data.items()):
        for sentiment in sentiment_list:
            result_str += f"### Month: {sentiment['year']}-{sentiment['month']:02d}\n"
            result_str += f"**MSPR (Monthly Share Purchase Ratio):** {sentiment['mspr']:.4f}\n"
            result_str += f"**Net Change:** {sentiment['change']}\n"
            result_str += f"**Buy Count:** {sentiment['buy_count']}\n"
            result_str += f"**Sell Count:** {sentiment['sell_count']}\n"
            result_str += f"**Total Buy Shares:** {sentiment['total_buy_shares']:,.0f}\n"
            result_str += f"**Total Sell Shares:** {sentiment['total_sell_shares']:,.0f}\n\n"
    
    result_str += "\nThe MSPR field refers to monthly share purchase ratio. "
    result_str += "Values > 0.5 indicate net buying, < 0.5 indicate net selling."

    return result_str


def get_finnhub_company_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"]
):
    """
    Retrieve insider transactions using FMP
    """
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before_str = before.strftime("%Y-%m-%d")
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        data = loop.run_until_complete(
            fmp_data.get_insider_transactions(ticker, before_str, curr_date)
        )
    finally:
        loop.close()
    
    if not data:
        return f"No insider transactions found for {ticker}"
    
    # Format output
    result_str = f"## {ticker} insider transactions from {before_str} to {curr_date}:\n\n"
    
    seen_transactions = set()
    
    for date, trans_list in sorted(data.items(), reverse=True):
        for trans in trans_list:
            # Create unique key to avoid duplicates
            trans_key = f"{trans['name']}_{trans['transactionDate']}_{trans['share']}"
            if trans_key in seen_transactions:
                continue
            seen_transactions.add(trans_key)
            
            result_str += f"### Filing Date: {trans['filingDate']}, {trans['name']}:\n"
            result_str += f"**Change:** {trans['change']:,.0f}\n"
            result_str += f"**Shares:** {trans['share']:,.0f}\n"
            result_str += f"**Transaction Price:** ${trans['transactionPrice']:.2f}\n"
            result_str += f"**Transaction Code:** {trans['transactionCode']}\n"
            result_str += f"**Transaction Date:** {trans['transactionDate']}\n\n"
    
    result_str += "\nTransaction codes: S = Sale, P = Purchase, M = Other\n"
    result_str += "The change field reflects the variation in share count (negative = sale, positive = purchase)"
    
    print(f"get_finnhub_company_insider_transactions {result_str}")
    
    return result_str


# ================= MARKET TOOLS =================
def get_fmp_data(
    symbol: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Get stock data using FMP provider
    """
    # Run async function in sync context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        df = loop.run_until_complete(
            fmp_data.get_historical_data(symbol, start_date, end_date)
        )
    finally:
        loop.close()
    
    return df

def get_fmp_data_online(
    symbol: str,
    start_date: str,
    end_date: str
) -> str:
    """
    Get stock data online using FMP
    """
    # Get FMP data
    df = get_fmp_data(symbol, start_date, end_date)
    
    if df.empty:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
    
    # Convert to CSV string với header
    csv_string = df.to_csv(index=False)
    
    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data source: Financial Modeling Prep (FMP)\n"
    header += f"# Retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    return header + csv_string


def get_fmp_data_window(
    symbol: str,
    curr_date: str,
    look_back_days: int
) -> str:
    """
    Get stock data với lookback window using FMP
    """
    # Calculate dates
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (date_obj - timedelta(days=look_back_days)).strftime("%Y-%m-%d")
    
    # Get data
    df = get_fmp_data(symbol, start_date, curr_date)
    
    if df.empty:
        return f"No data found for {symbol}"
    
    # Format as string table
    with pd.option_context(
        "display.max_rows", None, 
        "display.max_columns", None, 
        "display.width", None
    ):
        df_string = df.to_string()
    
    return (
        f"## Raw Market Data for {symbol} from {start_date} to {curr_date}:\n\n"
        + df_string
    )


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:
    """Get technical indicator values for a window of time using FMP data"""
    
    print(f"\n=== get_stock_stats_indicators_window CALLED ===")
    print(f"Processing: {symbol} - {indicator}")
    print(f"Date range: {look_back_days} days back from {curr_date}")
    print(f"Mode: {'ONLINE' if online else 'OFFLINE'}")
    
    # Technical indicators descriptions
    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }
    
    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )
    
    end_date = curr_date
    curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_obj - relativedelta(days=look_back_days)
    
    # Initialize utils
    utils = StockstatsUtils()
    market_data = MarketData()
    
    if not online:
        # Offline mode - use cached data
        config = get_config()
        data_dir = os.path.join(config["project_dir"], "market_data", "price_data")
        
        # Try to read cached FMP or YFin data
        fmp_file = os.path.join(data_dir, f"{symbol}-FMP-data-cached.csv")
        yfin_file = os.path.join(data_dir, f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv")
        
        data = None
        if os.path.exists(fmp_file):
            data = pd.read_csv(fmp_file)
        elif os.path.exists(yfin_file):
            data = pd.read_csv(yfin_file)
        else:
            raise Exception(f"No cached data found for {symbol}")
        
        # Get trading dates from data
        data["Date"] = pd.to_datetime(data["Date"])
        dates_in_df = data["Date"].dt.strftime("%Y-%m-%d")
        
        # Calculate indicators for each date
        ind_string = ""
        current = curr_date_obj
        
        while current >= before:
            date_str = current.strftime("%Y-%m-%d")
            
            # Only process trading days
            if date_str in dates_in_df.values:
                try:
                    indicator_value = get_stockstats_indicator(
                        symbol, indicator, date_str, online
                    )

                    print(f"    -> Indicator value: {indicator_value}")
                    ind_string += f"{date_str}: {indicator_value}\n"
                except Exception as e:
                    utils.logger.warning(f"Error getting {indicator} for {date_str}: {e}")
                    ind_string += f"{date_str}: Error - {str(e)}\n"
            
            current = current - relativedelta(days=1)
            
    else:
        # Online mode - fetch fresh data from FMP
        utils.logger.info(f"Fetching online FMP data for {symbol} indicators")
        
        # Get data for the entire period plus buffer for indicator calculation
        lookback_total = look_back_days + 200  # Extra days for indicator warm-up
        
        # Fetch FMP data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            fmp_data = loop.run_until_complete(
                market_data.get_historical_data_lookback_ver2(
                    ticker=symbol,
                    lookback_days=lookback_total
                )
            )
        finally:
            loop.close()
        
        if fmp_data.empty:
            return f"No FMP data available for {symbol}"
        
        # Format data for stockstats
        if fmp_data.index.name == 'Date':
            fmp_data = fmp_data.reset_index()
        
        fmp_data['Date'] = pd.to_datetime(fmp_data['Date'])
        
        # Calculate all indicators at once using stockstats
        df = wrap(fmp_data)
        df[indicator]  # Calculate the indicator
        
        # Extract values for date range
        ind_string = ""
        current = curr_date_obj
        
        while current >= before:
            date_str = current.strftime("%Y-%m-%d")
            
            # Find matching date
            matching = df[df['Date'].dt.strftime("%Y-%m-%d") == date_str]
            
            if not matching.empty:
                indicator_value = matching[indicator].values[0]
                ind_string += f"{date_str}: {indicator_value}\n"
            else:
                ind_string += f"{date_str}: N/A - Not a trading day\n"
            
            current = current - relativedelta(days=1)
    
    # Format result
    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )
    
    return result_str

def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:
    """Get single indicator value for a specific date"""
    
    try:
        config = get_config()
        data_dir = os.path.join(config["project_dir"], "market_data", "price_data")
        
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
            data_dir,
            online=online,
        )

        print(f"Indicator value: {indicator_value}")

        return str(indicator_value)
        
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return f"Error: {str(e)}"
# ================= END MARKET TOOLS =================


# ================= FUNDAMENTAL ONLINE TOOLS =================
def get_fundamentals_openai(ticker, curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_simfin_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 365
):
    """
    Retrieve balance sheet data using FMP instead of SimFin
    """
    ticker = str(ticker).upper()
    
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        data = loop.run_until_complete(
            fmp_data.get_balance_sheet(ticker, curr_date, look_back_days)
        )
    finally:
        loop.close()
    
    if not data:
        return f"No balance sheet data found for {ticker}"
    
    # Format output
    result_str = f"## {ticker} Balance Sheet Data:\n\n"
    
    for date, sheets in sorted(data.items(), reverse=True):
        for sheet in sheets:
            result_str += f"### Date: {date}\n"
            result_str += f"**Period:** {sheet['period']} {sheet['calendarYear']}\n"
            result_str += f"**Currency:** {sheet['reportedCurrency']}\n\n"
            
            result_str += "#### Assets\n"
            result_str += f"- Total Assets: ${sheet['totalAssets']:,.0f}\n"
            result_str += f"- Current Assets: ${sheet['totalCurrentAssets']:,.0f}\n"
            result_str += f"- Cash & Equivalents: ${sheet['cashAndCashEquivalents']:,.0f}\n"
            result_str += f"- Net Receivables: ${sheet['currentNetReceivables']:,.0f}\n"
            result_str += f"- Inventory: ${sheet['inventory']:,.0f}\n"
            result_str += f"- PP&E (Net): ${sheet['propertyPlantEquipmentNet']:,.0f}\n"
            result_str += f"- Intangible Assets: ${sheet['intangibleAssets']:,.0f}\n"
            result_str += f"- Goodwill: ${sheet['goodwill']:,.0f}\n\n"
            
            result_str += "#### Liabilities\n"
            result_str += f"- Total Liabilities: ${sheet['totalLiabilities']:,.0f}\n"
            result_str += f"- Current Liabilities: ${sheet['totalCurrentLiabilities']:,.0f}\n"
            result_str += f"- Accounts Payable: ${sheet['currentAccountsPayable']:,.0f}\n"
            result_str += f"- Short-term Debt: ${sheet['currentDebt']:,.0f}\n"
            result_str += f"- Long-term Debt: ${sheet['longTermDebt']:,.0f}\n\n"
            
            result_str += "#### Equity\n"
            result_str += f"- Total Equity: ${sheet['totalShareholderEquity']:,.0f}\n"
            result_str += f"- Retained Earnings: ${sheet['retainedEarnings']:,.0f}\n"
            result_str += f"- Common Stock: ${sheet['commonStock']:,.0f}\n"
            result_str += f"- Treasury Stock: ${sheet['treasuryStock']:,.0f}\n\n"
    
    return result_str


def get_simfin_income_statements(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 365
):
    """
    Retrieve income statement data using FMP instead of SimFin
    """
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        data = loop.run_until_complete(
            fmp_data.get_income_statement(ticker, curr_date, look_back_days)
        )
    finally:
        loop.close()
    
    if not data:
        return f"No income statement data found for {ticker}"
    
    # Format output
    result_str = f"## {ticker} Income Statement Data:\n\n"
    
    for date, statements in sorted(data.items(), reverse=True):
        for stmt in statements:
            result_str += f"### Date: {date}\n"
            result_str += f"**Period:** {stmt['period']} {stmt['calendarYear']}\n"
            result_str += f"**Currency:** {stmt['reportedCurrency']}\n\n"
            
            result_str += "#### Revenue & Profitability\n"
            result_str += f"- Revenue: ${stmt['revenue']:,.0f}\n"
            result_str += f"- Cost of Revenue: ${stmt['costOfRevenue']:,.0f}\n"
            result_str += f"- Gross Profit: ${stmt['grossProfit']:,.0f} ({stmt['grossProfitRatio']:.1%})\n"
            result_str += f"- Operating Income: ${stmt['operatingIncome']:,.0f} ({stmt['operatingIncomeRatio']:.1%})\n"
            result_str += f"- Net Income: ${stmt['netIncome']:,.0f} ({stmt['netIncomeRatio']:.1%})\n\n"
            
            result_str += "#### Operating Expenses\n"
            result_str += f"- R&D Expenses: ${stmt['researchAndDevelopmentExpenses']:,.0f}\n"
            result_str += f"- SG&A Expenses: ${stmt['sellingGeneralAndAdministrativeExpenses']:,.0f}\n"
            result_str += f"- Total Operating Expenses: ${stmt['operatingExpenses']:,.0f}\n\n"
            
            result_str += "#### Other Items\n"
            result_str += f"- Interest Expense: ${stmt['interestExpense']:,.0f}\n"
            result_str += f"- Interest Income: ${stmt['interestIncome']:,.0f}\n"
            result_str += f"- Income Tax: ${stmt['incomeTaxExpense']:,.0f}\n\n"
            
            result_str += "#### Per Share Data\n"
            result_str += f"- EPS: ${stmt['eps']:.2f}\n"
            result_str += f"- Diluted EPS: ${stmt['epsdiluted']:.2f}\n"
            result_str += f"- Shares Outstanding: {stmt['weightedAverageShsOut']:,.0f}\n"
            result_str += f"- Shares Outstanding (Diluted): {stmt['weightedAverageShsOutDil']:,.0f}\n\n"
    
    return result_str


def get_simfin_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
            str,
            "reporting frequency of the company's financial history: annual/quarterly",
        ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 365
):
    """
    Retrieve cashflow statement data using FMP instead of SimFin
    """
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        data = loop.run_until_complete(
            fmp_data.get_cashflow_statement(ticker, curr_date, look_back_days)
        )
    finally:
        loop.close()
    
    if not data:
        return f"No cashflow data found for {ticker}"
    
    # Format output
    result_str = f"## {ticker} Cash Flow Statement Data:\n\n"
    
    for date, cashflows in sorted(data.items(), reverse=True):
        for cf in cashflows:
            result_str += f"### Date: {date}\n"
            result_str += f"**Period:** {cf['period']} {cf['calendarYear']}\n"
            result_str += f"**Currency:** {cf['reportedCurrency']}\n\n"
            
            result_str += "#### Operating Activities\n"
            result_str += f"- Net Income: ${cf['netIncome']:,.0f}\n"
            result_str += f"- Depreciation & Amortization: ${cf['depreciationAndAmortization']:,.0f}\n"
            result_str += f"- Stock-Based Compensation: ${cf['stockBasedCompensation']:,.0f}\n"
            result_str += f"- Change in Working Capital: ${cf['changeInWorkingCapital']:,.0f}\n"
            result_str += f"- **Operating Cash Flow: ${cf['operatingCashFlow']:,.0f}**\n\n"
            
            result_str += "#### Investing Activities\n"
            result_str += f"- CapEx: ${cf['investmentsInPropertyPlantAndEquipment']:,.0f}\n"
            result_str += f"- Acquisitions: ${cf['acquisitionsNet']:,.0f}\n"
            result_str += f"- Investment Purchases: ${cf['purchasesOfInvestments']:,.0f}\n"
            result_str += f"- Investment Sales: ${cf['salesMaturitiesOfInvestments']:,.0f}\n"
            result_str += f"- **Investing Cash Flow: ${cf['netCashUsedForInvestingActivites']:,.0f}**\n\n"
            
            result_str += "#### Financing Activities\n"
            result_str += f"- Debt Repayment: ${cf['debtRepayment']:,.0f}\n"
            result_str += f"- Stock Issued: ${cf['commonStockIssued']:,.0f}\n"
            result_str += f"- Stock Repurchased: ${cf['commonStockRepurchased']:,.0f}\n"
            result_str += f"- Dividends Paid: ${cf['dividendsPaid']:,.0f}\n"
            result_str += f"- **Financing Cash Flow: ${cf['netCashUsedProvidedByFinancingActivities']:,.0f}**\n\n"
            
            result_str += "#### Summary\n"
            result_str += f"- Net Change in Cash: ${cf['netChangeInCash']:,.0f}\n"
            result_str += f"- Free Cash Flow: ${cf['freeCashFlow']:,.0f}\n"
            result_str += f"- Cash at End of Period: ${cf['cashAtEndOfPeriod']:,.0f}\n\n"
    
    return result_str


def get_crypto_market_analysis(
    symbol: Annotated[str, "Cryptocurrency symbol like BTC, ETH, ADA"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Get crypto market data using FMP
    """
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        crypto_data = loop.run_until_complete(
            fmp_data.get_crypto_quote(symbol)
        )
    finally:
        loop.close()
    
    if not crypto_data:
        return f"No market data available for {symbol}"

    result_str = f"## {symbol.upper()} Current Market Data (from FMP):\n\n"
    result_str += f"**Name:** {crypto_data.get('name', 'N/A')}\n"
    result_str += f"**Current Price:** ${crypto_data.get('price', 0):,.6f}\n"
    result_str += f"**Market Cap:** ${crypto_data.get('marketCap', 0):,.0f}\n"
    result_str += f"**24h Volume:** ${crypto_data.get('volume', 0):,.0f}\n"
    result_str += f"**24h Change:** {crypto_data.get('changesPercentage', 0):.2f}%\n"
    
    # FMP specific data
    result_str += f"**Day Low:** ${crypto_data.get('dayLow', 0):,.6f}\n"
    result_str += f"**Day High:** ${crypto_data.get('dayHigh', 0):,.6f}\n"
    result_str += f"**Year High:** ${crypto_data.get('yearHigh', 0):,.6f}\n"
    result_str += f"**Year Low:** ${crypto_data.get('yearLow', 0):,.6f}\n"
    result_str += f"**50-Day MA:** ${crypto_data.get('priceAvg50', 0):,.6f}\n"
    result_str += f"**200-Day MA:** ${crypto_data.get('priceAvg200', 0):,.6f}\n"
    result_str += f"**Avg Volume:** ${crypto_data.get('avgVolume', 0):,.0f}\n"
    result_str += f"**Circulating Supply:** {crypto_data.get('sharesOutstanding', 0):,.0f}\n"
    

    print(f"get_crypto_market_analysis: {result_str}")
    return result_str


def get_crypto_fundamentals_analysis(
    symbol: Annotated[str, "Cryptocurrency symbol like BTC, ETH, ADA"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Get crypto fundamentals using FMP
    Note: FMP có ít fundamental data cho crypto hơn CoinGecko
    """
   
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        crypto_data = loop.run_until_complete(
            fmp_data.get_crypto_quote(symbol)
        )
    finally:
        loop.close()

    if not crypto_data:
        return f"Error: Could not find data for symbol {symbol}"
    
    # Calculate additional metrics
    price = crypto_data.get('price', 0)
    market_cap = crypto_data.get('marketCap', 0)
    volume = crypto_data.get('volume', 0)
    avg_volume = crypto_data.get('avgVolume', 1)  # Avoid division by zero
    
    # Volume ratio
    volume_ratio = volume / avg_volume if avg_volume > 0 else 0
    
    # Price position in year range
    year_high = crypto_data.get('yearHigh', price)
    year_low = crypto_data.get('yearLow', price)
    price_position = ((price - year_low) / (year_high - year_low) * 100) if year_high > year_low else 50
    
    result_str = f"## {symbol.upper()} Fundamental Analysis (from FMP):\n\n"
    result_str += f"**Market Metrics:**\n"
    result_str += f"- Market Cap: ${market_cap:,.0f}\n"
    result_str += f"- Current Price: ${price:,.6f}\n"
    result_str += f"- Circulating Supply: {crypto_data.get('sharesOutstanding', 0):,.0f}\n\n"
    
    result_str += f"**Trading Metrics:**\n"
    result_str += f"- 24h Volume: ${volume:,.0f}\n"
    result_str += f"- Average Volume: ${avg_volume:,.0f}\n"
    result_str += f"- Volume Ratio: {volume_ratio:.2f}x avg\n"
    result_str += f"- 24h Change: {crypto_data.get('changesPercentage', 0):.2f}%\n\n"
    
    result_str += f"**Technical Levels:**\n"
    result_str += f"- 50-Day MA: ${crypto_data.get('priceAvg50', 0):,.6f}\n"
    result_str += f"- 200-Day MA: ${crypto_data.get('priceAvg200', 0):,.6f}\n"
    result_str += f"- Year High: ${year_high:,.6f}\n"
    result_str += f"- Year Low: ${year_low:,.6f}\n"
    result_str += f"- Price Position: {price_position:.1f}% of year range\n\n"
    
    # Add context
    result_str += f"**Analysis Context:**\n"
    if volume_ratio > 2:
        result_str += "- High trading volume indicates increased market interest\n"
    elif volume_ratio < 0.5:
        result_str += "- Low trading volume may indicate reduced liquidity\n"
        
    if price > crypto_data.get('priceAvg200', price):
        result_str += "- Price above 200-day MA suggests long-term uptrend\n"
    else:
        result_str += "- Price below 200-day MA suggests long-term downtrend\n"
        
    return result_str
    
    