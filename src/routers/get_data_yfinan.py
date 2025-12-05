import re
import traceback
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, timezone
from fastapi import Query
from fastapi import APIRouter
from pandas.api.types import is_datetime64tz_dtype
from typing import List, Optional

from src.helpers.get_data_yf_helpers import get_history_from_yfinance
from src.helpers.get_day_losers_helpers import GetDate
from src.utils.news_utils import get_asset_type, get_news

router = APIRouter(prefix="/yfinance")

@router.get("/ticker_tape")
async def get_market_overview(
    symbols: str = Query("^GSPC", description="Comma-separated list of stock symbols, e.g., '^GSPC,^AAPL' for multiple symbols")
):
    try:

        symbols_list = symbols.split(',')

        market_data_list = []

        for symbol in symbols_list:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            historical_data = ticker.history(period="1d", interval="1m")
            
            market_data = {
                "symbol": symbol,
                "symbol_name": info.get("longName", ""),
                "price": historical_data['Close'].iloc[-1] if not historical_data.empty else None,
                "change": info.get("regularMarketChange", None),
                "change_percent": info.get("regularMarketChangePercent", None)
            }
            market_data_list.append(market_data)

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "data": market_data_list
            }
        }

    except Exception as e:
        return {
            "message": f"Internal server error: {str(e)}",
            "status": "500",
            "data": {}
        }

@router.get("/equity/discovery/losers")
def get_top_losers(limit: int = 10):
    try:
        df = GetDate.get_day_losers(limit)

        losers = []
        for index, row in df.iterrows():
            price_string = row.get("Price")
            match = re.match(r"([0-9.]+)", price_string)
            price = float(match.group(1)) if match else None
            losers.append({
                "symbol": row.get("Symbol"),
                "name": row.get("Name"),
                "price": price,
                "change": row.get("Change"),
                "percent_change": row.get("Change %"),
                "volume": row.get("Volume"),
            })

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "data": losers
            }
        }

    except Exception as e:
        return {
            "message": f"Internal server error: {str(e)}",
            "status": "500",
            "data": {}
        }
    
@router.get("/equity/discovery/panel-losers")
def get_top_panel_losers():
    """
    Lấy danh sách 5 cổ phiếu giảm giá nhiều nhất trong ngày
    và lịch sử giá trong ngày (5 phút interval) cho mỗi cổ phiếu.
    """
    try:
        df_all_losers = GetDate.get_day_losers()

        if df_all_losers.empty:
             return {
                "message": "OK",
                "status": "200",
                "data": {
                    "data": []
                }
            }

        df_top_5 = df_all_losers.head(5)

        losers_result = []

        for index, row in df_top_5.iterrows():
            symbol = row.get("Symbol")
            name = row.get("Name")
            change = row.get("Change")
            percent_change = row.get("Change %")
            volume = row.get("Volume")
            price_string = row.get("Price")
            match = re.match(r"([0-9.]+)", price_string) if price_string else None
            current_price = float(match.group(1)) if match else None

            chart_data_formatted = []
            latest_price = None
            try:
                historical_data = get_history_from_yfinance(symbol, interval='1h', period='1d')
                for timestamp, row_data in historical_data.iterrows():
                    time_obj = timestamp
                    price_val = row_data.get("Close")

                    if price_val is not None:

                        chart_data_formatted.append({
                            "time": time_obj,
                            "price": price_val
                        })
                    else:
                         print(f"Warning: Invalid data point for {symbol}:  price={price_val}")

                if chart_data_formatted:
                    latest_price = chart_data_formatted[-1]['price']
                else: 
                   latest_price = current_price

            except Exception as history_error:
                print(f"Error fetching or processing history for {symbol}: {str(history_error)}")

                chart_data_formatted = []
                latest_price = None 

            losers_result.append({
                "symbol": symbol,
                "name": name,
                "price": latest_price, 
                "change": change,
                "percent_change": percent_change,
                "volume": volume,
                "chartData": chart_data_formatted 
            })

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "data": losers_result
            }
        }

    except Exception as e:
        print(f"General error in get_top_panel_losers: {str(e)}")
        return {
            "message": f"Internal server error: {str(e)}",
            "status": "500",
            "data": {}
        }

@router.get("/equity/discovery/active")
def get_most_active(limit: int = 10):
    try:
        df = GetDate.get_day_most_active(limit)

        losers = []
        for index, row in df.iterrows():
            price_string = row.get("Price")
            match = re.match(r"([0-9.]+)", price_string)
            price = float(match.group(1)) if match else None
            losers.append({
                "symbol": row.get("Symbol"),
                "name": row.get("Name"),
                "price": price,
                "change": row.get("Change"),
                "percent_change": row.get("Change %"),
                "volume": row.get("Volume"),
            })

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "data": losers
            }
        }

    except Exception as e:
        return {
            "message": f"Internal server error: {str(e)}",
            "status": "500",
            "data": {}
        }

@router.get("/equity/discovery/panel-active")
def get_most_panel_active():
    try:
        df_all_active = GetDate.get_day_most_active()

        if df_all_active.empty:
             return {
                "message": "OK",
                "status": "200",
                "data": {
                    "data": []
                }
            }

        df_top_5 = df_all_active.head(5)

        losers_result = []

        for index, row in df_top_5.iterrows():
            symbol = row.get("Symbol")
            name = row.get("Name")
            change = row.get("Change")
            percent_change = row.get("Change %")
            volume = row.get("Volume")
            price_string = row.get("Price")
            match = re.match(r"([0-9.]+)", price_string) if price_string else None
            current_price = float(match.group(1)) if match else None

            chart_data_formatted = []
            latest_price = None
            try:
                historical_data = get_history_from_yfinance(symbol, interval='1h', period='1d')
                for timestamp, row_data in historical_data.iterrows():
                    time_obj = timestamp
                    price_val = row_data.get("Close")

                    if price_val is not None:

                        chart_data_formatted.append({
                            "time": time_obj,
                            "price": price_val
                        })
                    else:
                         print(f"Warning: Invalid data point for {symbol}:  price={price_val}")

                if chart_data_formatted:
                    latest_price = chart_data_formatted[-1]['price']
                else: 
                   latest_price = current_price

            except Exception as history_error:
                print(f"Error fetching or processing history for {symbol}: {str(history_error)}")

                chart_data_formatted = []
                latest_price = None 

            losers_result.append({
                "symbol": symbol,
                "name": name,
                "price": latest_price, 
                "change": change,
                "percent_change": percent_change,
                "volume": volume,
                "chartData": chart_data_formatted 
            })

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "data": losers_result
            }
        }

    except Exception as e:
        print(f"General error in get_top_panel_losers: {str(e)}")
        return {
            "message": f"Internal server error: {str(e)}",
            "status": "500",
            "data": {}
        }

@router.get("/equity/discovery/gainers")
async def get_top_gainers(limit: int = 10):
    try:
        df = GetDate.get_day_gainers(limit)
        losers = []
        for index, row in df.iterrows():
            price_string = row.get("Price")
            match = re.match(r"([0-9.]+)", price_string)
            price = float(match.group(1)) if match else None
            losers.append({
                "symbol": row.get("Symbol"),
                "name": row.get("Name"),
                "price": price,
                "change": row.get("Change"),
                "percent_change": row.get("Change %"),
                "volume": row.get("Volume"),
            })

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "data": losers
            }
        }

    except Exception as e:
        return {
            "message": f"Internal server error: {str(e)}",
            "status": "500",
            "data": {}
        }

@router.get("/equity/discovery/panel-gainers")
async def get_top_panel_gainers():
    try:
        df_all_gainers = GetDate.get_day_gainers()

        if df_all_gainers.empty:
             return {
                "message": "OK",
                "status": "200",
                "data": {
                    "data": []
                }
            }

        df_top_5 = df_all_gainers.head(5)

        losers_result = []

        for index, row in df_top_5.iterrows():
            symbol = row.get("Symbol")
            name = row.get("Name")
            change = row.get("Change")
            percent_change = row.get("Change %")
            volume = row.get("Volume")
            price_string = row.get("Price")
            match = re.match(r"([0-9.]+)", price_string) if price_string else None
            current_price = float(match.group(1)) if match else None

            chart_data_formatted = []
            latest_price = None
            try:
                historical_data = get_history_from_yfinance(symbol, interval='1h', period='1d')
                for timestamp, row_data in historical_data.iterrows():
                    time_obj = timestamp
                    price_val = row_data.get("Close")

                    if price_val is not None:

                        chart_data_formatted.append({
                            "time": time_obj,
                            "price": price_val
                        })
                    else:
                         print(f"Warning: Invalid data point for {symbol}:  price={price_val}")

                if chart_data_formatted:
                    latest_price = chart_data_formatted[-1]['price']
                else: 
                   latest_price = current_price

            except Exception as history_error:
                print(f"Error fetching or processing history for {symbol}: {str(history_error)}")

                chart_data_formatted = []
                latest_price = None 

            losers_result.append({
                "symbol": symbol,
                "name": name,
                "price": latest_price, 
                "change": change,
                "percent_change": percent_change,
                "volume": volume,
                "chartData": chart_data_formatted 
            })

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "data": losers_result
            }
        }

    except Exception as e:
        print(f"General error in get_top_panel_losers: {str(e)}")
        return {
            "message": f"Internal server error: {str(e)}",
            "status": "500",
            "data": {}
        }

@router.get("/equity/overview_market")
async def get_market_overview(
    symbols: str = Query("^GSPC", description="Comma-separated list of stock symbols, e.g., '^GSPC,^AAPL' for multiple symbols")
):
    try:
        symbols_list = symbols.split(',')
        market_data_list = []

        for symbol in symbols_list:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            historical_data = ticker.history(period="1d", interval="1m")
            asset_type = get_asset_type(symbol)

            market_data = {
                "symbol": symbol,
                "asset_type": info.get("assetType", asset_type),
                "name": info.get("longName", ""),
                "exchange": info.get("exchange", None),
                "bid": info.get("bid", None),
                "bid_size": info.get("bidSize", None),
                "ask": info.get("ask", None),
                "ask_size": info.get("askSize", None),
                "last_price": historical_data['Close'].iloc[-1] if not historical_data.empty else None,
                "open": info.get("regularMarketOpen", None),
                "high": info.get("regularMarketDayHigh", None),
                "low": info.get("regularMarketDayLow", None),
                "volume": info.get("regularMarketVolume", None),
                "prev_close": info.get("regularMarketPreviousClose", None),
                "change": info.get("regularMarketChange", None),
                "change_percent": info.get("regularMarketChangePercent", None),
                "year_high": info.get("fiftyTwoWeekHigh", None),
                "year_low": info.get("fiftyTwoWeekLow", None),
                "short_name": info.get("shortName", "N/A"),
                "ma_50d": info.get("fiftyDayAverage", None),
                "ma_200d": info.get("twoHundredDayAverage", None),
                "volume_average": info.get("averageVolume", None),
                "volume_average_10d": info.get("averageVolume10days", None),
                "currency": info.get("currency", "USD")
            }

            market_data_list.append(market_data)

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "data": market_data_list
            }
        }

    except Exception as e:
        return {
            "message": f"Internal server error: {str(e)}",
            "status": "500",
            "data": {}
        }

@router.get("/equity/profile")
async def get_company_financials_revised(
    symbol: str = Query(..., description="Stock symbol, e.g., 'AAPL'")
):
    try:
        ticker = yf.Ticker(symbol)

        info = ticker.info
        if not info or info.get('quoteType') is None:
             return {"message": f"Data not found for symbol: {symbol}", "status": "404", "data": None}

        financials = ticker.financials
        balance_sheet = ticker.balance_sheet
        cash_flow = ticker.cashflow
        q_financials = ticker.quarterly_financials
        q_balance_sheet = ticker.quarterly_balance_sheet

        # --- Xác định các kỳ báo cáo chính ---
        latest_fy_date = max(financials.columns) if financials is not None and not financials.empty else None
        previous_fy_date = sorted(financials.columns, reverse=True)[1] if financials is not None and len(financials.columns) > 1 else None
        latest_quarter_date = max(q_financials.columns) if q_financials is not None and not q_financials.empty else None
        last_4_quarters_dates = sorted(q_financials.columns, reverse=True)[:4] if q_financials is not None and len(q_financials.columns) >= 4 else []

        # --- Lấy dữ liệu từ info (chủ yếu TTM) ---
        market_cap = info.get("marketCap")
        shares_outstanding = info.get("sharesOutstanding")
        employees = info.get("fullTimeEmployees")
        pe_ratio_ttm = info.get("trailingPE")
        price_to_sales_ttm = info.get("priceToSalesTrailing12Months")
        roa_ttm = info.get("returnOnAssets")
        roe_ttm = info.get("returnOnEquity")
        gross_margin_ttm = info.get("grossMargins")
        operating_margin_ttm = info.get("operatingMargins")
        profit_margin_ttm = info.get("profitMargins")
        ebitda_ttm = info.get("ebitda")
        beta = info.get("beta")
        year_high = info.get("fiftyTwoWeekHigh")
        year_low = info.get("fiftyTwoWeekLow")
        volume_average_10d = info.get("averageVolume10days")
        free_cash_flow_ttm = info.get("freeCashflow")
        basic_eps_ttm = info.get("trailingEps")
        diluted_eps_ttm = info.get("trailingAnnualisedDilutedEps", info.get("dilutedEps"))
        forward_annual_dividend_rate = info.get("dividendRate")
        current_price = info.get("currentPrice", info.get("regularMarketPrice"))

        # --- Lấy/Tính toán dữ liệu theo kỳ báo cáo  ---

        # == Dữ liệu Năm tài chính gần nhất (FY) ==
        revenue_fy = None
        if financials is not None and not financials.empty and latest_fy_date in financials.columns:
            for name in ["Total Revenue", "Revenues"]:
                if name in financials.index:
                    value = financials.loc[name].get(latest_fy_date)
                    if value is not None and not pd.isna(value):
                        revenue_fy = value.item() if hasattr(value, 'item') else value
                        break

        gross_profit_fy = None
        if financials is not None and not financials.empty and latest_fy_date in financials.columns:
            if "Gross Profit" in financials.index:
                 value = financials.loc["Gross Profit"].get(latest_fy_date)
                 if value is not None and not pd.isna(value):
                     gross_profit_fy = value.item() if hasattr(value, 'item') else value

        net_income_fy = None
        if financials is not None and not financials.empty and latest_fy_date in financials.columns:
            for name in ["Net Income", "Net Income Common Stockholders"]:
                if name in financials.index:
                    value = financials.loc[name].get(latest_fy_date)
                    if value is not None and not pd.isna(value):
                        net_income_fy = value.item() if hasattr(value, 'item') else value
                        break

        basic_eps_fy = None
        if financials is not None and not financials.empty and latest_fy_date in financials.columns:
             if "Basic EPS" in financials.index:
                 value = financials.loc["Basic EPS"].get(latest_fy_date)
                 if value is not None and not pd.isna(value):
                     basic_eps_fy = value.item() if hasattr(value, 'item') else value

        diluted_eps_fy = None
        if financials is not None and not financials.empty and latest_fy_date in financials.columns:
             if "Diluted EPS" in financials.index:
                 value = financials.loc["Diluted EPS"].get(latest_fy_date)
                 if value is not None and not pd.isna(value):
                     diluted_eps_fy = value.item() if hasattr(value, 'item') else value

        ebit_fy = None
        if financials is not None and not financials.empty and latest_fy_date in financials.columns:
             for name in ["EBIT", "Operating Income"]:
                 if name in financials.index:
                     value = financials.loc[name].get(latest_fy_date)
                     if value is not None and not pd.isna(value):
                         ebit_fy = value.item() if hasattr(value, 'item') else value
                         break

        tax_provision_fy = None
        if financials is not None and not financials.empty and latest_fy_date in financials.columns:
             for name in ["Tax Provision", "Income Tax Expense"]:
                 if name in financials.index:
                     value = financials.loc[name].get(latest_fy_date)
                     if value is not None and not pd.isna(value):
                         tax_provision_fy = value.item() if hasattr(value, 'item') else value
                         break

        total_equity_fy = None
        if balance_sheet is not None and not balance_sheet.empty and latest_fy_date in balance_sheet.columns:
             for name in ["Total Stockholder Equity", "Stockholders Equity"]:
                 if name in balance_sheet.index:
                     value = balance_sheet.loc[name].get(latest_fy_date)
                     if value is not None and not pd.isna(value):
                         total_equity_fy = value.item() if hasattr(value, 'item') else value
                         break

        last_year_revenue_fy = None
        if financials is not None and not financials.empty and previous_fy_date is not None and previous_fy_date in financials.columns:
             for name in ["Total Revenue", "Revenues"]:
                 if name in financials.index:
                     value = financials.loc[name].get(previous_fy_date)
                     if value is not None and not pd.isna(value):
                         last_year_revenue_fy = value.item() if hasattr(value, 'item') else value
                         break

        dividends_paid_cash_fy = None
        if cash_flow is not None and not cash_flow.empty and latest_fy_date in cash_flow.columns:
            for name in ["Dividends Paid", "Cash Dividends Paid"]:
                 if name in cash_flow.index:
                     value = cash_flow.loc[name].get(latest_fy_date)
                     if value is not None and not pd.isna(value):
                         dividends_paid_cash_fy = value.item() if hasattr(value, 'item') else value
                         if dividends_paid_cash_fy > 0 and name == "Dividends Paid":
                             dividends_paid_cash_fy = -dividends_paid_cash_fy
                         break

        dps_fy = None
        if financials is not None and not financials.empty and latest_fy_date in financials.columns:
            if "Dividends Per Share" in financials.index:
                 value = financials.loc["Dividends Per Share"].get(latest_fy_date)
                 if value is not None and not pd.isna(value):
                     dps_fy = value.item() if hasattr(value, 'item') else value
        if dps_fy is None:
            dps_fy = forward_annual_dividend_rate 

        # == Dữ liệu Quý gần nhất (MRQ) ==
        total_assets_mrq = None
        if q_balance_sheet is not None and not q_balance_sheet.empty and latest_quarter_date in q_balance_sheet.columns:
             if "Total Assets" in q_balance_sheet.index:
                  value = q_balance_sheet.loc["Total Assets"].get(latest_quarter_date)
                  if value is not None and not pd.isna(value):
                      total_assets_mrq = value.item() if hasattr(value, 'item') else value

        total_current_assets_mrq = None
        if q_balance_sheet is not None and not q_balance_sheet.empty and latest_quarter_date in q_balance_sheet.columns:
             if "Total Current Assets" in q_balance_sheet.index:
                  value = q_balance_sheet.loc["Total Current Assets"].get(latest_quarter_date)
                  if value is not None and not pd.isna(value):
                      total_current_assets_mrq = value.item() if hasattr(value, 'item') else value

        inventory_mrq = None
        if q_balance_sheet is not None and not q_balance_sheet.empty and latest_quarter_date in q_balance_sheet.columns:
             if "Inventory" in q_balance_sheet.index:
                  value = q_balance_sheet.loc["Inventory"].get(latest_quarter_date)
                  if value is not None and not pd.isna(value):
                      inventory_mrq = value.item() if hasattr(value, 'item') else value

        total_current_liabilities_mrq = None
        if q_balance_sheet is not None and not q_balance_sheet.empty and latest_quarter_date in q_balance_sheet.columns:
             if "Total Current Liabilities" in q_balance_sheet.index:
                  value = q_balance_sheet.loc["Total Current Liabilities"].get(latest_quarter_date)
                  if value is not None and not pd.isna(value):
                      total_current_liabilities_mrq = value.item() if hasattr(value, 'item') else value

        total_debt_mrq = None
        if q_balance_sheet is not None and not q_balance_sheet.empty and latest_quarter_date in q_balance_sheet.columns:
            if "Total Debt" in q_balance_sheet.index:
                 value = q_balance_sheet.loc["Total Debt"].get(latest_quarter_date)
                 if value is not None and not pd.isna(value):
                     total_debt_mrq = value.item() if hasattr(value, 'item') else value
        if total_debt_mrq is None: 
             total_debt_mrq = info.get("totalDebt")

        cash_and_equivalents_mrq = None
        if q_balance_sheet is not None and not q_balance_sheet.empty and latest_quarter_date in q_balance_sheet.columns:
             for name in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"]:
                 if name in q_balance_sheet.index:
                     value = q_balance_sheet.loc[name].get(latest_quarter_date)
                     if value is not None and not pd.isna(value):
                         cash_and_equivalents_mrq = value.item() if hasattr(value, 'item') else value
                         break

        total_equity_mrq = None
        if q_balance_sheet is not None and not q_balance_sheet.empty and latest_quarter_date in q_balance_sheet.columns:
             for name in ["Total Stockholder Equity", "Stockholders Equity"]:
                 if name in q_balance_sheet.index:
                     value = q_balance_sheet.loc[name].get(latest_quarter_date)
                     if value is not None and not pd.isna(value):
                         total_equity_mrq = value.item() if hasattr(value, 'item') else value
                         break

        gross_profit_mrq = None
        if q_financials is not None and not q_financials.empty and latest_quarter_date in q_financials.columns:
            if "Gross Profit" in q_financials.index:
                 value = q_financials.loc["Gross Profit"].get(latest_quarter_date)
                 if value is not None and not pd.isna(value):
                     gross_profit_mrq = value.item() if hasattr(value, 'item') else value

        minority_interest_mrq = 0 
        if q_balance_sheet is not None and not q_balance_sheet.empty and latest_quarter_date in q_balance_sheet.columns:
             for name in ["Minority Interest", "Noncontrolling Interest"]:
                 if name in q_balance_sheet.index:
                     value = q_balance_sheet.loc[name].get(latest_quarter_date)
                     if value is not None and not pd.isna(value):
                         minority_interest_mrq = value.item() if hasattr(value, 'item') else value
                         break

        preferred_stock_mrq = 0 
        if q_balance_sheet is not None and not q_balance_sheet.empty and latest_quarter_date in q_balance_sheet.columns:
             for name in ["Preferred Stock Equity", "Preferred Stock"]:
                 if name in q_balance_sheet.index:
                     value = q_balance_sheet.loc[name].get(latest_quarter_date)
                     if value is not None and not pd.isna(value):
                         preferred_stock_mrq = value.item() if hasattr(value, 'item') else value
                         break

        enterprise_value_mrq = None
        if market_cap is not None and total_debt_mrq is not None and cash_and_equivalents_mrq is not None \
           and isinstance(minority_interest_mrq,(int, float)) and isinstance(preferred_stock_mrq,(int, float)):
             if all(isinstance(v, (int, float)) for v in [market_cap, total_debt_mrq, cash_and_equivalents_mrq]):
                 enterprise_value_mrq = market_cap + total_debt_mrq - cash_and_equivalents_mrq + minority_interest_mrq + preferred_stock_mrq


        price_to_book_fy = None
        if market_cap is not None and total_equity_fy is not None and total_equity_fy != 0 \
            and isinstance(market_cap, (int, float)) and isinstance(total_equity_fy, (int, float)):
            price_to_book_fy = market_cap / total_equity_fy


        price_to_sales_fy = None
        if market_cap is not None and revenue_fy is not None and revenue_fy != 0 \
           and isinstance(market_cap, (int, float)) and isinstance(revenue_fy, (int, float)):
            price_to_sales_fy = market_cap / revenue_fy


        quick_ratio_mrq = None
        if total_current_assets_mrq is not None and inventory_mrq is not None and total_current_liabilities_mrq is not None and total_current_liabilities_mrq != 0 \
            and isinstance(total_current_assets_mrq, (int, float)) and isinstance(inventory_mrq, (int, float)) and isinstance(total_current_liabilities_mrq, (int, float)):
            quick_ratio_mrq = (total_current_assets_mrq - inventory_mrq) / total_current_liabilities_mrq
        elif info.get("quickRatio") is not None:
             quick_ratio_mrq = info.get("quickRatio")


        current_ratio_mrq = None
        if total_current_assets_mrq is not None and total_current_liabilities_mrq is not None and total_current_liabilities_mrq != 0 \
            and isinstance(total_current_assets_mrq, (int, float)) and isinstance(total_current_liabilities_mrq, (int, float)):
            current_ratio_mrq = total_current_assets_mrq / total_current_liabilities_mrq
        elif info.get("currentRatio") is not None: 
             current_ratio_mrq = info.get("currentRatio")

        debt_to_equity_mrq = None
        if total_debt_mrq is not None and total_equity_mrq is not None and total_equity_mrq != 0 \
            and isinstance(total_debt_mrq, (int, float)) and isinstance(total_equity_mrq, (int, float)):
            debt_to_equity_mrq = total_debt_mrq / total_equity_mrq


        net_debt_mrq = None
        if total_debt_mrq is not None and cash_and_equivalents_mrq is not None \
            and isinstance(total_debt_mrq, (int, float)) and isinstance(cash_and_equivalents_mrq, (int, float)):
             net_debt_mrq = total_debt_mrq - cash_and_equivalents_mrq


        return_invested_capital_annual = None
        if total_debt_mrq is not None and total_equity_fy is not None \
           and isinstance(total_debt_mrq, (int, float)) and isinstance(total_equity_fy, (int, float)) \
           and total_equity_fy > 0:
            invested_capital = total_debt_mrq + total_equity_fy 
            if invested_capital != 0 and ebit_fy is not None and tax_provision_fy is not None \
               and isinstance(ebit_fy, (int, float)) and isinstance(tax_provision_fy, (int, float)):
                if ebit_fy > 0:
                     effective_tax_rate = tax_provision_fy / ebit_fy
                     effective_tax_rate = max(0.0, min(effective_tax_rate, 1.0))
                     nopat = ebit_fy * (1 - effective_tax_rate)
                     return_invested_capital_annual = nopat / invested_capital

        revenue_per_employee_fy = None
        if revenue_fy is not None and employees is not None and employees > 0 \
           and isinstance(revenue_fy, (int, float)) and isinstance(employees, (int, float)):
            revenue_per_employee_fy = revenue_fy / employees


        forward_dividend_yield = None
        if forward_annual_dividend_rate is not None and current_price is not None and current_price != 0 \
           and isinstance(forward_annual_dividend_rate, (int, float)) and isinstance(current_price, (int, float)):
            forward_dividend_yield = forward_annual_dividend_rate / current_price

        pretax_margin_ttm = None
        if len(last_4_quarters_dates) == 4 and q_financials is not None and not q_financials.empty:
            ttm_revenue = 0
            ttm_pretax_income = 0
            valid_quarters = 0
            for q_date in last_4_quarters_dates:
                 if q_date not in q_financials.columns: continue 

                 q_revenue_val = None
                 for name in ["Total Revenue", "Revenues"]:
                     if name in q_financials.index:
                         val = q_financials.loc[name].get(q_date)
                         if val is not None and not pd.isna(val):
                             q_revenue_val = val.item() if hasattr(val, 'item') else val
                             break

                 q_pretax_val = None
                 for name in ["Pretax Income", "Earnings Before Tax"]:
                     if name in q_financials.index:
                          val = q_financials.loc[name].get(q_date)
                          if val is not None and not pd.isna(val):
                              q_pretax_val = val.item() if hasattr(val, 'item') else val
                              break

                 if q_revenue_val is not None and q_pretax_val is not None \
                    and isinstance(q_revenue_val, (int, float)) and isinstance(q_pretax_val, (int, float)):
                     ttm_revenue += q_revenue_val
                     ttm_pretax_income += q_pretax_val
                     valid_quarters += 1

            if valid_quarters == 4 and ttm_revenue != 0:
                pretax_margin_ttm = ttm_pretax_income / ttm_revenue


        # --- Tạo Dictionary kết quả ---
        financial_data = {
            "market_cap": market_cap,
            "enterprise_value": enterprise_value_mrq,
            "enterprise_to_ebitda": info.get("enterpriseToEbitda"),
            "shares_outstanding": shares_outstanding,
            "employees": employees,
            "shareholders": None,
            "pe_ratio": pe_ratio_ttm,
            "price_revenue_ratio": price_to_sales_ttm,
            "price_to_book": price_to_book_fy,
            "price_to_sales": price_to_sales_fy,
            "quick_ratio": quick_ratio_mrq,
            "current_ratio": current_ratio_mrq,
            "debt_to_equity": debt_to_equity_mrq,
            "net_debt": net_debt_mrq,
            "total_debt": total_debt_mrq,
            "total_assets": total_assets_mrq,
            "roa": roa_ttm,
            "roe": roe_ttm,
            "return_invested_capital": return_invested_capital_annual,
            "revenue_per_employee": revenue_per_employee_fy,
            "volume_average_10d": volume_average_10d,
            "beta": beta,
            "year_high": year_high,
            "year_low": year_low,
            "dividends_paid": dividends_paid_cash_fy,
            "dividend_yield": forward_dividend_yield,
            "dividends_per_share": dps_fy,
            "net_margin": profit_margin_ttm,
            "gross_margin": gross_margin_ttm,
            "operating_margin": operating_margin_ttm,
            "pretax_margin": pretax_margin_ttm,
            "basic_eps_fy": basic_eps_fy,
            "eps": basic_eps_ttm,
            "eps_diluted": diluted_eps_fy, 
            "net_income": net_income_fy,
            "ebitda": ebitda_ttm,
            "gross_profit_mrq": gross_profit_mrq,
            "gross_profit": gross_profit_fy,
            "last_year_revenue": last_year_revenue_fy,
            "revenue": revenue_fy,
            "free_cash_flow": free_cash_flow_ttm,
        }

        for key, value in financial_data.items():
            if pd.isna(value):
                financial_data[key] = None
            elif hasattr(value, 'item'):
                financial_data[key] = value.item()

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "symbol": symbol.upper(),
                "symbol_name": info.get("longName", symbol.upper()),
                "financials_data": [financial_data]
            }
        }

    except Exception as e:
        error_message = f"Error fetching or processing data for {symbol}: {str(e)}"
        return {
            "message": error_message,
            "status": "500",
            "data": None
        }
    
@router.get("/equity/price/historical")
def get_price_yf(
    provider: str = Query("yfinance"),
    interval: str = Query("1d"),
    timezone: str = Query("America/New_York"),
    source: str = Query("realtime"),
    adjustment: str = Query("splits_only"),
    extended_hours: bool = Query(False),
    sort: str = Query("asc"),
    limit: int = Query(49999),
    include_actions: bool = Query(True),
    symbol: str = Query(...),
    start_date: Optional[str] = Query(None),
    end_date: str = Query(...)
):
    
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    end_date += timedelta(days=1)

    ticker = yf.Ticker(symbol)
    ticker_info = ticker.info

    last_price = ticker_info.get('regularMarketPrice', None)
    change_realtime = ticker_info.get('regularMarketChange', None)
    change_percent_realtime = ticker_info.get('regularMarketChangePercent', None)

    try:

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        else:
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        df = yf.download(
            tickers=symbol,
            interval=interval,
            start=start_date,
            auto_adjust = True,
            end=end_date,
            prepost=extended_hours,
            actions=include_actions,
            progress=False
        )

        if df.empty:
            return {"message": "No data found", "status": "204", "data": {}}
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        datetime_col = "Datetime" if "Datetime" in df.columns else "Date"

        df[datetime_col] = pd.to_datetime(df[datetime_col])
        if not is_datetime64tz_dtype(df[datetime_col]):
            df[datetime_col] = df[datetime_col].dt.tz_localize("UTC")

        df[datetime_col] = df[datetime_col] - pd.Timedelta(hours=4)

        df = df.head(limit)

        result_data = []
        for _, row in df.iterrows():
            date_str = (
                row[datetime_col].strftime("%Y-%m-%dT%H:%M:%S")
                if isinstance(row[datetime_col], pd.Timestamp)
                else str(row[datetime_col])
            )

            open_val = row["Open"]
            close_val = row["Close"]

            change = close_val - open_val if pd.notna(open_val) and pd.notna(close_val) else 0
            change_percent = (change / open_val * 100) if pd.notna(open_val) and open_val != 0 else 0


            result_data.append({
                "date": date_str,
                "change": change,
                "change_percent": change_percent,
                "close": row["Close"],
                "close_previous": row["Open"],
                "dividend": 0, 
                "high": row["High"],
                "low": row["Low"],
                "open": row["Open"],
                "split_ratio": 0,  
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
            })

        return {
            "message": "OK",
            "status": "200",
            "data": {
                "symbol": symbol.upper(),
                "symbol_name": yf.Ticker(symbol).info.get("longName", symbol.upper()),
                "last_price": last_price,
                "change": change_realtime,
                "change_percent": change_percent_realtime,
                "data": result_data
            }
        }

    except Exception as e:
        return {"message": f"Error: {str(e)}", "status": "500", "data": {}}
    
@router.get("/news/company")
async def get_news_yf(symbol: str = Query(..., description="Stock symbol, e.g., 'NVDA'"), limit: int = Query(5, description="Limit the number of news articles")):
    try:
            news_items = await get_news(symbol, limit)
            if not news_items:
                return {
                    "message": "No news found",
                    "status": "204",
                    "data": {}
                }
            return {
                "message": "OK",
                "status": "200",
                "data": {
                    "data": news_items
                }
            }

    except Exception as e:
        return {
            "message": f"Internal server error: {str(e)}",
            "status": "500",
            "data": {}
        }