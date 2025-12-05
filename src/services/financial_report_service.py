import httpx
import json
from typing import List, Optional, Dict, Any, Tuple, Type, Union
import logging

from src.models.equity import ( 
    IncomeStatement, BalanceSheetStatement, CashFlowStatement, FinancialStatementsData
)
from src.utils.logger.set_up_log_dataFMP import setup_logger
from src.utils.config import settings 

logger = setup_logger(__name__, log_level=logging.INFO)

FMP_API_KEY = settings.FMP_API_KEY
BASE_FMP_URL = settings.BASE_FMP_URL

class FinancialStatementsService:

    async def _fetch_single_statement_data(
        self,
        client: httpx.AsyncClient,
        symbol: str,
        statement_type: str, 
        period: str, 
        limit: int = 5 
    ) -> Optional[List[Dict[str, Any]]]:
        """Hàm helper để lấy dữ liệu cho một loại báo cáo cụ thể."""
        if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_PLACEHOLDER":
            logger.error(f"FMP API Key không được cấu hình cho {statement_type} của {symbol}.")
            return None

        endpoint = f"/v3/{statement_type}/{symbol.upper()}"
        params = {"period": period, "limit": limit, "apikey": FMP_API_KEY}
        log_params_display = {k:v for k,v in params.items() if k != "apikey"}
        url = f"{BASE_FMP_URL}{endpoint}"
        logger.debug(f"Gọi FMP API: {url} với params: {log_params_display}")

        try:
            response = await client.get(url, params=params, timeout=20.0)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                logger.info(f"Lấy thành công {len(data)} bản ghi {statement_type} ({period}) cho {symbol}.")
                return data
            logger.warning(f"Dữ liệu FMP {statement_type} ({period}) cho {symbol} không phải list: {type(data)}")
            return None
        except httpx.HTTPStatusError as e:
            logger.error(f"Lỗi HTTPStatusError khi lấy {statement_type} ({period}) cho {symbol}: {e.response.status_code} - {e.response.text[:200]}", exc_info=False)
        except httpx.RequestError as e:
            logger.error(f"Lỗi RequestError khi lấy {statement_type} ({period}) cho {symbol}: {e}", exc_info=True)
        except json.JSONDecodeError as e:
            logger.error(f"Lỗi JSONDecodeError khi lấy {statement_type} ({period}) cho {symbol}: {e}. Response: {response.text[:200] if 'response' in locals() else 'N/A'}", exc_info=False)
        except Exception as e:
            logger.exception(f"Lỗi không xác định khi lấy {statement_type} ({period}) cho {symbol}")
        return None

    def _parse_statement_data(
        self,
        raw_data_list: Optional[List[Dict[str, Any]]],
        model_type: Type[Union[IncomeStatement, BalanceSheetStatement, CashFlowStatement]]
    ) -> List[Union[IncomeStatement, BalanceSheetStatement, CashFlowStatement]]:
        """Parse danh sách dict thô thành list các Pydantic model."""
        if not raw_data_list:
            return []
        
        parsed_list = []
        for item_raw in raw_data_list:
            try:
                parsed_item = model_type(**item_raw)
                parsed_list.append(parsed_item)
            except Exception as e_parse:
                logger.warning(f"Lỗi parse item cho {model_type.__name__}: {item_raw}. Lỗi: {e_parse}")
        return parsed_list


    async def get_all_financial_statements(
        self,
        symbol: str,
        annual_limit: int = 5,
        quarterly_limit: int = 4 
    ) -> Optional[FinancialStatementsData]:
        logger.info(f"Bắt đầu lấy tất cả báo cáo tài chính cho symbol: {symbol}")
        output_data = FinancialStatementsData(symbol=symbol.upper())

        async with httpx.AsyncClient() as client:
            income_annual_raw = await self._fetch_single_statement_data(client, symbol, "income-statement", "annual", annual_limit)
            balance_annual_raw = await self._fetch_single_statement_data(client, symbol, "balance-sheet-statement", "annual", annual_limit)
            cashflow_annual_raw = await self._fetch_single_statement_data(client, symbol, "cash-flow-statement", "annual", annual_limit)

            income_quarterly_raw = await self._fetch_single_statement_data(client, symbol, "income-statement", "quarter", quarterly_limit)
            balance_quarterly_raw = await self._fetch_single_statement_data(client, symbol, "balance-sheet-statement", "quarter", quarterly_limit)
            cashflow_quarterly_raw = await self._fetch_single_statement_data(client, symbol, "cash-flow-statement", "quarter", quarterly_limit)

        output_data.income_statements_annual = self._parse_statement_data(income_annual_raw, IncomeStatement)
        output_data.balance_sheets_annual = self._parse_statement_data(balance_annual_raw, BalanceSheetStatement)
        output_data.cash_flow_statements_annual = self._parse_statement_data(cashflow_annual_raw, CashFlowStatement)

        output_data.income_statements_quarterly = self._parse_statement_data(income_quarterly_raw, IncomeStatement)
        output_data.balance_sheets_quarterly = self._parse_statement_data(balance_quarterly_raw, BalanceSheetStatement)
        output_data.cash_flow_statements_quarterly = self._parse_statement_data(cashflow_quarterly_raw, CashFlowStatement)

        if not (output_data.income_statements_annual or output_data.income_statements_quarterly or
                output_data.balance_sheets_annual or output_data.balance_sheets_quarterly or
                output_data.cash_flow_statements_annual or output_data.cash_flow_statements_quarterly):
            logger.warning(f"Không lấy được bất kỳ dữ liệu báo cáo tài chính nào cho {symbol}.")
        
        logger.info(f"Hoàn thành lấy báo cáo tài chính cho {symbol}.")
        return output_data