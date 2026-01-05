

from src.models.equity import (
    CryptoSpotlightItem, FMPCompanyOutlookProfile,CompanyProfileData, MarketOverviewData, TickerTapeData
)
from typing import Dict, List, Optional, Any

class EquityMapper:

    @staticmethod
    def map_fmp_quote_to_ticker_tape_item(fmp_quote_item: Dict[str, Any]) -> Optional[TickerTapeData]:
        """
        Ánh xạ một item từ API /v3/quote/ của FMP sang TickerTapeData.
        """
        if not fmp_quote_item or not isinstance(fmp_quote_item, dict):
            return None

        symbol = fmp_quote_item.get("symbol")
        if not symbol:
            return None

        def safe_float(value: Any) -> Optional[float]:
            if value is None: return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        return TickerTapeData(
            symbol=str(symbol),
            name=fmp_quote_item.get("name"),
            price=safe_float(fmp_quote_item.get("price")),
            change=safe_float(fmp_quote_item.get("change")),
            percent_change=safe_float(fmp_quote_item.get("changesPercentage"))
        )

    @staticmethod
    def map_fmp_crypto_quote_to_spotlight_item(
        fmp_crypto_item_data: Dict[str, Any],
        logo_url: Optional[str] 
    ) -> Optional[CryptoSpotlightItem]:
        if not fmp_crypto_item_data or not isinstance(fmp_crypto_item_data, dict):
            return None

        symbol = fmp_crypto_item_data.get("symbol")
        if not symbol:
            return None

        change_val = fmp_crypto_item_data.get("change")
        trend = None
        if change_val is not None:
            try:
                change_float = float(change_val)
                if change_float > 0:
                    trend = "up"
                elif change_float < 0:
                    trend = "down"
                else:
                    trend = "neutral"
            except (ValueError, TypeError):
                pass 

        def safe_float(value: Any) -> Optional[float]:
            if value is None: return None
            try: return float(value)
            except (ValueError, TypeError): return None

        return CryptoSpotlightItem(
            trend=trend,
            symbol=str(symbol),
            name=fmp_crypto_item_data.get("name"),
            logo_url=logo_url,
            price=safe_float(fmp_crypto_item_data.get("price")),
            change=safe_float(change_val),
            percent_change=safe_float(fmp_crypto_item_data.get("changesPercentage")),
            volume=safe_float(fmp_crypto_item_data.get("volume")),
            market_cap=safe_float(fmp_crypto_item_data.get("marketCap"))
        )

    @staticmethod
    def map_all_fmp_to_detailed_profile(
        symbol_upper: str, 
        outlook_data: Dict[str, Any], 
        quote_data_supplement: Optional[Dict[str, Any]],
        key_metrics_ttm_data: Optional[Dict[str, Any]]
    ) -> Optional[FMPCompanyOutlookProfile]:

        if not outlook_data:
            return None

        profile_sec = outlook_data.get("profile", {}) or {}
        metrics_sec_outlook = outlook_data.get("metrics", {}) or {}
        ratios_ttm_list = outlook_data.get("ratios", [])
        ratios_ttm = ratios_ttm_list[0] if ratios_ttm_list and isinstance(ratios_ttm_list, list) else {}
        
        fin_annual = outlook_data.get("financialsAnnual", {}) or {}
        income_annual_fy_list = fin_annual.get("income", [])
        income_annual_fy = income_annual_fy_list[0] if income_annual_fy_list else {}
        
        balance_quarterly_mrq_list = (outlook_data.get("financialsQuarter", {}) or {}).get("balance", [])
        balance_quarterly_mrq = balance_quarterly_mrq_list[0] if balance_quarterly_mrq_list else {}

        income_quarterly_mrq_list = (outlook_data.get("financialsQuarter", {}) or {}).get("income", [])
        income_quarterly_mrq = income_quarterly_mrq_list[0] if income_quarterly_mrq_list else {}

        data = {} 

        # --- Xử lý employees ---
        raw_employees_val = profile_sec.get("fullTimeEmployees")
        employees_int_val: Optional[int] = None
        if raw_employees_val is not None:
            try:
                if isinstance(raw_employees_val, str):
                    employees_int_val = int(float(str(raw_employees_val).replace(',', '')))
                else:
                    employees_int_val = int(float(raw_employees_val))
            except (ValueError, TypeError):
                print(f"EquityMapper Warning: Không thể chuyển đổi 'fullTimeEmployees' ('{raw_employees_val}') thành int cho {symbol_upper}")
        data["employees"] = employees_int_val

        # --- Xử lý revenue_fy_val ---
        raw_revenue_fy_val = income_annual_fy.get("revenue")
        revenue_float_val: Optional[float] = None
        if raw_revenue_fy_val is not None:
            try:
                revenue_float_val = float(raw_revenue_fy_val)
            except (ValueError, TypeError):
                 print(f"EquityMapper Warning: Không thể chuyển đổi 'revenue' ('{raw_revenue_fy_val}') thành float cho {symbol_upper}")
        
        data["market_cap"] = profile_sec.get("mktCap")
        q_supp = quote_data_supplement or {}
        km_ttm = key_metrics_ttm_data or {}

        data["enterprise_value"] = km_ttm.get("enterpriseValueTTM")
        data["enterprise_to_ebitda"] = ratios_ttm.get("enterpriseValueMultipleTTM")
        data["shares_outstanding"] = q_supp.get("sharesOutstanding", profile_sec.get("sharesOutstanding"))
        
        data["pe_ratio"] = ratios_ttm.get("priceEarningsRatioTTM", q_supp.get("pe"))
        data["price_to_book"] = ratios_ttm.get("priceToBookRatioTTM")
        data["price_to_sales"] = ratios_ttm.get("priceToSalesRatioTTM")

        data["roa"] = ratios_ttm.get("returnOnAssetsTTM")
        data["roe"] = ratios_ttm.get("returnOnEquityTTM")
        data["return_invested_capital"] = ratios_ttm.get("returnOnCapitalEmployedTTM")
        
        if revenue_float_val is not None and employees_int_val is not None and employees_int_val > 0:
            data["revenue_per_employee"] = revenue_float_val / employees_int_val
        else:
            data["revenue_per_employee"] = None 

        data["quick_ratio"] = ratios_ttm.get("quickRatioTTM")
        data["current_ratio"] = ratios_ttm.get("currentRatioTTM")
        data["debt_to_equity"] = ratios_ttm.get("debtEquityRatioTTM")
        
        data["total_debt"] = balance_quarterly_mrq.get("totalDebt")
        data["total_assets"] = balance_quarterly_mrq.get("totalAssets")
        cash_mrq = balance_quarterly_mrq.get("cashAndCashEquivalents")
        total_debt_float = None
        cash_mrq_float = None
        if data.get("total_debt") is not None:
            try: total_debt_float = float(data.get("total_debt"))
            except (ValueError, TypeError): pass
        if cash_mrq is not None:
            try: cash_mrq_float = float(cash_mrq)
            except (ValueError, TypeError): pass

        if total_debt_float is not None and cash_mrq_float is not None:
            data["net_debt"] = total_debt_float - cash_mrq_float
        else:
            data["net_debt"] = None
        
        data["volume_average_10d"] = km_ttm.get("averageTradingVolume10Day")
        data["beta"] = profile_sec.get("beta", q_supp.get("beta"))
        
        data["year_high"] = metrics_sec_outlook.get("yearHigh", q_supp.get("yearHigh"))
        data["year_low"] = metrics_sec_outlook.get("yearLow", q_supp.get("yearLow"))
        if data["year_high"] is None and profile_sec.get("range"):
            try:
                low_str, high_str = profile_sec.get("range").split('-')
                data["year_low"] = float(low_str)
                data["year_high"] = float(high_str)
            except (ValueError, TypeError, AttributeError): pass

        data["dividend_yield"] = ratios_ttm.get("dividendYielTTM", metrics_sec_outlook.get("dividendYielTTM"))
        data["dividends_per_share"] = ratios_ttm.get("dividendPerShareTTM")
        
        dps_ttm_val_str = data.get("dividends_per_share")
        shares_out_val_str = data.get("shares_outstanding")
        dps_ttm_float: Optional[float] = None
        shares_out_float: Optional[float] = None

        if dps_ttm_val_str is not None:
            try: dps_ttm_float = float(dps_ttm_val_str)
            except (ValueError, TypeError): pass
        if shares_out_val_str is not None:
            try: shares_out_float = float(shares_out_val_str)
            except (ValueError, TypeError): pass
            
        if dps_ttm_float is not None and shares_out_float is not None and shares_out_float > 0:
            data["dividends_paid"] = dps_ttm_float * shares_out_float
        else:
            data["dividends_paid"] = None

        data["net_margin"] = ratios_ttm.get("netProfitMarginTTM")
        data["gross_margin"] = ratios_ttm.get("grossProfitMarginTTM")
        data["operating_margin"] = ratios_ttm.get("operatingProfitMarginTTM")
        data["pretax_margin"] = ratios_ttm.get("pretaxProfitMarginTTM")

        data["basic_eps_fy"] = income_annual_fy.get("eps")
        data["eps"] = ratios_ttm.get("earningsPerShareTTM", q_supp.get("eps"))
        data["eps_diluted"] = income_annual_fy.get("epsdiluted")
        
        data["net_income"] = income_annual_fy.get("netIncome")
        data["ebitda"] = km_ttm.get("ebitdaTTM", income_annual_fy.get("ebitda"))
        
        data["gross_profit_mrq"] = income_quarterly_mrq.get("grossProfit")
        data["gross_profit"] = income_annual_fy.get("grossProfit")
        
        data["revenue"] = km_ttm.get("revenueTTM", revenue_float_val) 
        data["last_year_revenue"] = revenue_float_val 

        fcf_per_share_ttm_str = ratios_ttm.get("freeCashFlowPerShareTTM")
        fcf_per_share_ttm_float: Optional[float] = None
        if fcf_per_share_ttm_str is not None:
            try: fcf_per_share_ttm_float = float(fcf_per_share_ttm_str)
            except (ValueError, TypeError): pass

        if fcf_per_share_ttm_float is not None and shares_out_float is not None and shares_out_float > 0:
            data["free_cash_flow"] = fcf_per_share_ttm_float * shares_out_float
        else:
            data["free_cash_flow"] = None
            
        try:
            return FMPCompanyOutlookProfile(**data)
        except Exception as e:
            print(f"EquityMapper Error: Lỗi khi tạo FMPCompanyOutlookProfile cho {symbol_upper}: {e}")
            return None

    @staticmethod
    def map_fmp_to_market_overview(
        symbol: str,
        fmp_quote_data: Optional[Dict[str, Any]],
        fmp_profile_data: Optional[Dict[str, Any]],
        fmp_pre_post_data: Optional[Dict[str, Any]],
        latest_sma_50: Optional[float],
        latest_sma_200: Optional[float]
    ) -> Optional[MarketOverviewData]:

        overview = MarketOverviewData(symbol=symbol)

        # 1. Ánh xạ từ FMP Quote Data (fmp_quote_data)
        if fmp_quote_data:
            overview.last_price = fmp_quote_data.get('price')
            overview.open_price = fmp_quote_data.get('open')
            overview.day_high = fmp_quote_data.get('dayHigh')
            overview.day_low = fmp_quote_data.get('dayLow')
            overview.previous_close_price = fmp_quote_data.get('previousClose')
            overview.change = fmp_quote_data.get('change')
            overview.change_percent = fmp_quote_data.get('changesPercentage')
            overview.volume = fmp_quote_data.get('volume')
            overview.year_high = fmp_quote_data.get('yearHigh')
            overview.year_low = fmp_quote_data.get('yearLow')
            
            overview.volume_average = fmp_quote_data.get('avgVolume') 
            
            if not overview.name:
                overview.name = fmp_quote_data.get('name')
            if not overview.exchange:
                overview.exchange = fmp_quote_data.get('exchange')
                
        # 2. Ánh xạ từ FMP Profile Data (fmp_profile_data)
        if fmp_profile_data:
            overview.name = fmp_profile_data.get('companyName', overview.name)
            overview.short_name = fmp_profile_data.get('companyName', overview.name) 
            overview.currency = fmp_profile_data.get('currency')
            overview.exchange = fmp_profile_data.get('exchangeShortName', overview.exchange) 

            fmp_asset_type = fmp_profile_data.get('type', "").lower()
            if "stock" in fmp_asset_type or "equity" in fmp_asset_type:
                overview.asset_type = "STOCK"
            elif "etf" == fmp_asset_type:
                overview.asset_type = "ETF"
            elif "index" == fmp_asset_type:
                overview.asset_type = "INDEX"
            elif "crypto" == fmp_asset_type:
                overview.asset_type = "CRYPTO"
            elif "fund" in fmp_asset_type or "mutual_fund" in fmp_asset_type: 
                overview.asset_type = "FUND"
            elif fmp_asset_type: 
                overview.asset_type = fmp_asset_type.upper()
        
        overview.volume_average_10d = None 

        # 4. Ánh xạ từ FMP Pre/Post Market Data (fmp_pre_post_data) - MỚI
        if fmp_pre_post_data:
            overview.bid = fmp_pre_post_data.get('bid')
            overview.ask = fmp_pre_post_data.get('ask')
            overview.bid_size = fmp_pre_post_data.get('bsize') 
            overview.ask_size = fmp_pre_post_data.get('asize') 
        
        # 5. Gán giá trị MA50 và MA200 (đã được trích xuất từ quote data trước đó)
        if latest_sma_50 is not None:
            try:
                overview.ma_50d = float(latest_sma_50)
            except (ValueError, TypeError):
                print(f"Mapper Warning: Không thể chuyển đổi sma_50 '{latest_sma_50}' thành float cho {symbol}")
        if latest_sma_200 is not None:
            try:
                overview.ma_200d = float(latest_sma_200)
            except (ValueError, TypeError):
                print(f"Mapper Warning: Không thể chuyển đổi sma_200 '{latest_sma_200}' thành float cho {symbol}")
                
        if overview.last_price is None and overview.name is None:
            return None

        return overview