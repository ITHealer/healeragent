
from datetime import date, timedelta
import httpx
from typing import List, Optional, Dict, Any
import logging
import asyncio
import aioredis

from src.services.history_chart_service import HistoryChartService
from src.models.equity import ChartDataItem, DiscoveryItemOutput, TickerTapeData 
from src.utils.logger.set_up_log_dataFMP import setup_logger
from src.utils.config import settings

logger = setup_logger(__name__, log_level=logging.INFO)

FMP_API_KEY = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL

class MarketIndicesService:
    def __init__(self):
        self.market_indices_map: Dict[str, List[Dict[str, str]]] = {
            "US": [
                # Chỉ số Futures
                {"symbol": "ES=F", "name": "S&P Futures"},
                {"symbol": "YM=F", "name": "Dow Futures"},
                {"symbol": "NQ=F", "name": "Nasdaq Futures"},
                {"symbol": "RTY=F", "name": "Russell 2000 Futures"},
                # Chỉ số chính
                {"symbol": "^GSPC", "name": "S&P 500"},
                {"symbol": "^DJI", "name": "Dow 30"},
                {"symbol": "^IXIC", "name": "Nasdaq Composite"},
                {"symbol": "^VIX", "name": "CBOE Volatility Index"}, 
            ],
            "AMERICA": [
                {"symbol": "^GSPTSE", "name": "S&P/TSX (Canada)"},
                {"symbol": "^MXX", "name": "IPC (Mexico)"}, 
                {"symbol": "^BVSP", "name": "IBOVESPA (Brazil)"},  
                {"symbol": "^MERV", "name": "S&P MERVAL (Argentina)"}, 
                {"symbol": "^IPSA", "name": "S&P/CLX IPSA (Chile)"},    
                {"symbol": "^COLCAP", "name": "MSCI COLCAP (Colombia)"},#
            ],
            "EUROPE": [
                {"symbol": "^FTSE", "name": "FTSE 100 (UK)"},     
                {"symbol": "^GDAXI", "name": "DAX (Germany)"},    
                {"symbol": "^FCHI", "name": "CAC 40 (France)"},  
                # {"symbol": "STOXX50E.SW", "name": "EURO STOXX 50"},
                {"symbol": "^AEX", "name": "AEX (Netherlands)"},   
                {"symbol": "^BFX", "name": "BEL 20 (Belgium)"},   
                # {"symbol": "^PSI20", "name": "PSI 20 (Portugal)"}, 
                {"symbol": "^ISEQ", "name": "ISEQ 20 (Ireland)"},   
                {"symbol": "^SSMI", "name": "Swiss Market Index"}, 
                {"symbol": "^IBEX", "name": "IBEX 35 (Spain)"},   
                {"symbol": "FTSEMIB.MI", "name": "FTSE MIB (Italy)"},
                # {"symbol": "^OMXSPI", "name": "OMX Stockholm PI"},#
                {"symbol": "^OMXH25", "name": "OMX Helsinki 25"}, 
                # {"symbol": "^OMXC25", "name": "OMX Copenhagen 25"},
                # {"symbol": "OBX.OL", "name": "OBX Index (Norway)"},
                {"symbol": "^ATX", "name": "Austrian Traded Index"},
                {"symbol": "WIG20.WA", "name": "WIG20 (Poland)"},     
                # {"symbol": "^ATG", "name": "Athens General Comp."},
                {"symbol": "XU100.IS", "name": "BIST 100 (Turkey)"}, 
                # {"symbol": "^BUX", "name": "BUX (Hungary)"},      
                # {"symbol": "^PX", "name": "PX Index (Prague)"},
            ],
            "ASIA": [
                {"symbol": "^N225", "name": "Nikkei 225 (Japan)"},
                {"symbol": "^HSI", "name": "Hang Seng (Hong Kong)"},
                {"symbol": "000001.SS", "name": "SSE Composite (China)"},
                {"symbol": "399001.SZ", "name": "SZSE Component (China)"},
                {"symbol": "^BSESN", "name": "S&P BSE SENSEX (India)"},
                {"symbol": "^NSEI", "name": "NIFTY 50 (India)"},  
                {"symbol": "^AXJO", "name": "S&P/ASX 200 (Australia)"},
                {"symbol": "^NZ50", "name": "S&P/NZX 50 (New Zealand)"},
                {"symbol": "^STI", "name": "Straits Times (Singapore)"},
                {"symbol": "^KLSE", "name": "FTSE Bursa (Malaysia)"},
                {"symbol": "^JKSE", "name": "IDX Composite (Indonesia)"},
                {"symbol": "^SET", "name": "SET Index (Thailand)"},
                {"symbol": "^TWII", "name": "TSEC Weighted (Taiwan)"}, 
                {"symbol": "^KS11", "name": "KOSPI Composite (S. Korea)"},
                {"symbol": "^VNINDEX", "name": "VN-Index (Vietnam)"}, 
            ],
            "AFRICA": [
                {"symbol": "^JALSH", "name": "FTSE/JSE All Share (S. Africa)"},
                {"symbol": "^CASE30", "name": "EGX 30 (Egypt)"}, 
                {"symbol": "^TASI.SR", "name": "Tadawul All Share (Saudi)"},
                {"symbol": "^DFMGI", "name": "DFM General (Dubai)"},
                {"symbol": "^TA125.TA", "name": "TA-125 (Israel)"},
                {"symbol": "^QSI", "name": "QE Index (Qatar)"}, 
                {"symbol": "^KWSE", "name": "Boursa Kuwait Main 50"},
            ],
            "COMMODITIES": [
                {"symbol": "CL=F", "name": "Crude Oil"},
                {"symbol": "GC=F", "name": "Gold"},
                {"symbol": "SI=F", "name": "Silver"},
                {"symbol": "HG=F", "name": "Copper"},
                {"symbol": "NG=F", "name": "Natural Gas"},
                {"symbol": "BZ=F", "name": "Brent Crude"}
            ],
            "CRYPTOCURRENCIES": [
                {"symbol": "BTCUSD", "name": "Bitcoin USD"},
                {"symbol": "ETHUSD", "name": "Ethereum USD"},
                {"symbol": "SOLUSD", "name": "Solana USD"},
                {"symbol": "XRPUSD", "name": "XRP USD"},
            ]
        }

    def _get_hardcoded_list_for_region(self, region: str) -> List[Dict[str, str]]:
        """Lấy danh sách mã và tên đã định nghĩa sẵn cho một khu vực."""
        return self.market_indices_map.get(region.upper(), [])

    def _get_all_hardcoded_lists(self) -> List[Dict[str, str]]:
        """Lấy tất cả các mã và tên đã định nghĩa sẵn từ tất cả các khu vực."""
        all_indices = []
        for region_list in self.market_indices_map.values():
            all_indices.extend(region_list)
        return all_indices

    async def fetch_data_for_indices_list(
        self, 
        indices_to_fetch: List[Dict[str, str]],
        redis_client: Optional[aioredis.Redis] = None
    ) -> Optional[List[DiscoveryItemOutput]]:
        """
        Hàm chung: Nhận vào một danh sách các dict {'symbol': ..., 'name': ...},
        lấy dữ liệu quote và chart từ FMP, và trả về List[DiscoveryItemOutput].
        """
        if not indices_to_fetch:
            return []

        symbols = [index['symbol'] for index in indices_to_fetch]
        symbol_name_map = {item['symbol']: item['name'] for item in indices_to_fetch}

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Task 1: Lấy dữ liệu quote cho tất cả symbols trong một lần gọi batch
            symbols_str = ",".join(symbols)
            quote_url = f"{BASE_FMP_URL}/v3/quote/{symbols_str}?apikey={FMP_API_KEY}"
            quote_task = client.get(quote_url)

            # Task 2: Tạo nhiều task nhỏ để lấy historical chart đồng thời
            to_date = date.today()
            from_date = to_date - timedelta(days=5)
            chart_tasks = [
                HistoryChartService.get_historical_chart_from_fmp(
                    symbol=symbol,
                    interval="4hour",
                    start_date_str=from_date.isoformat(),
                    end_date_str=to_date.isoformat(),
                    client=client,
                    redis_client=redis_client 
                ) for symbol in symbols
            ]
        
            try:
                results = await asyncio.gather(quote_task, *chart_tasks, return_exceptions=True)
            except Exception as e:
                logger.exception("Lỗi nghiêm trọng trong quá trình asyncio.gather.")
                return None

            quotes_map: Dict[str, Dict[str, Any]] = {}
            quote_response = results[0]

            if isinstance(quote_response, httpx.Response):
                try:
                    quote_response.raise_for_status()
                    quote_data_list = quote_response.json()
                    if isinstance(quote_data_list, list):
                        quotes_map = {item["symbol"].upper(): item for item in quote_data_list if isinstance(item, dict) and item.get("symbol")}
                except Exception as e:
                    logger.error(f"Lỗi xử lý phản hồi từ API quote: {e}")
            elif isinstance(quote_response, Exception):
                logger.error(f"Không thể lấy dữ liệu quote: {quote_response}")

            chart_data_map: Dict[str, List[ChartDataItem]] = {}
            chart_results = results[1:]

            for i, chart_result in enumerate(chart_results):
                symbol_upper = symbols[i].upper()

                if isinstance(chart_result, dict) and 'chart_data' in chart_result:
                    actual_chart_data = chart_result['chart_data']
                    if isinstance(actual_chart_data, list):
                        chart_data_map[symbol_upper] = actual_chart_data
                    else:
                        logger.warning(f"Dữ liệu trong 'chart_data' không phải là list cho {symbol_upper}.")

                elif isinstance(chart_result, list):
                    chart_data_map[symbol_upper] = chart_result
                    # logger.info(f"Xử lý thành công {len(chart_result)} điểm dữ liệu chart cho {symbol_upper} (dạng list trực tiếp).")
                
                elif isinstance(chart_result, Exception):
                    logger.warning(f"Không thể lấy chart data cho {symbol_upper}: {chart_result}")

            final_results: List[DiscoveryItemOutput] = []
            
            for index_item in indices_to_fetch:
                symbol_original = index_item['symbol']
                custom_name = index_item['name']
                
                symbol_upper = symbol_original.upper()
                
                quote_data = quotes_map.get(symbol_upper)
                if not quote_data:
                    # logger.warning(f"Không có dữ liệu quote cho {symbol_upper}, bỏ qua item này.")
                    continue
                    
                chart_data = chart_data_map.get(symbol_upper, [])

                item = DiscoveryItemOutput(
                    symbol=symbol_original, 
                    name=custom_name,
                    url_logo=None,
                    price=quote_data.get('price'),
                    change=quote_data.get('change'),
                    percent_change=quote_data.get('changesPercentage'),
                    volume=quote_data.get('volume'),
                    chartData=chart_data
                )
                final_results.append(item)

            return final_results

    async def get_and_fetch_indices_by_region(self, region: str, redis_client: Optional[aioredis.Redis] = None) -> Optional[List[DiscoveryItemOutput]]:
        """Lấy danh sách mã cho một region và fetch dữ liệu đầy đủ cho chúng."""
        indices_list = self._get_hardcoded_list_for_region(region)
        return await self.fetch_data_for_indices_list(indices_list, redis_client=redis_client)

    async def get_and_fetch_all_indices(self, redis_client: Optional[aioredis.Redis] = None) -> Optional[List[DiscoveryItemOutput]]:
        """Lấy TẤT CẢ các mã và fetch dữ liệu đầy đủ cho chúng."""
        all_indices_list = self._get_all_hardcoded_lists()
        return await self.fetch_data_for_indices_list(all_indices_list, redis_client=redis_client)