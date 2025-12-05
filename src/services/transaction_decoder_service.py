# File: src/services/transaction_decoder_service.py

import json
import httpx
from web3 import Web3
from typing import Dict, Any, Optional, Tuple
from src.utils.logger.set_up_log_dataFMP import setup_logger
from src.utils.config import settings
import logging

logger = setup_logger(__name__, log_level=logging.ERROR) 

ETHERSCAN_API_URL = "https://api.etherscan.io/api"
ETHERSCAN_API_KEY = settings.ETHERSCAN_API_KEY

class TransactionDecoderService:
    def __init__(self):
        self.w3 = Web3()
        # Dùng một dictionary đơn giản làm cache trong bộ nhớ cho ABI
        self.abi_cache: Dict[str, Any] = {}
        # Dùng để cache các đối tượng contract đã được khởi tạo
        self.contract_cache: Dict[str, Any] = {}

    async def _get_abi_from_etherscan(self, contract_address: str) -> Optional[Any]:
        """Hàm tự động lấy ABI của một hợp đồng từ API của Etherscan."""
        params = {
            "module": "contract",
            "action": "getabi",
            "address": contract_address,
            "apikey": ETHERSCAN_API_KEY,
        }
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(ETHERSCAN_API_URL, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "1":
                    # Etherscan trả về ABI dưới dạng một chuỗi JSON, cần parse lại
                    abi = json.loads(data["result"])
                    logger.info(f"Successfully fetched ABI for {contract_address}")
                    return abi
                else:
                    logger.warning(f"Etherscan API returned an error for {contract_address}: {data.get('result')}")
                    return None
        except Exception as e:
            logger.error(f"Failed to fetch ABI for {contract_address}: {e}", exc_info=True)
            return None

    async def decode_transaction_input(self, to_address: str, input_data: str) -> Optional[Tuple[Any, Any]]:
        """
        Giải mã dữ liệu input. Tự động tìm và cache ABI nếu cần.
        """
        checksum_address = Web3.to_checksum_address(to_address)
        
        # Kiểm tra xem đã có đối tượng contract trong cache chưa
        if checksum_address in self.contract_cache:
            contract = self.contract_cache[checksum_address]
        else:
            # Nếu chưa, kiểm tra xem đã có ABI trong cache chưa
            abi = self.abi_cache.get(checksum_address)
            if not abi:
                # Nếu chưa có ABI, gọi Etherscan để lấy
                abi = await self._get_abi_from_etherscan(checksum_address)
            
            if abi:
                # Lưu ABI vào cache và tạo đối tượng contract
                self.abi_cache[checksum_address] = abi
                contract = self.w3.eth.contract(address=checksum_address, abi=abi)
                self.contract_cache[checksum_address] = contract
            else:
                # Nếu không thể lấy ABI, không thể giải mã
                return None
        
        # Tiến hành giải mã
        try:
            func_obj, func_params = contract.decode_function_input(input_data)
            return func_obj, func_params
        except Exception:
            # Có thể input không phải là một lời gọi hàm (ví dụ: giao dịch tạo contract)
            return None