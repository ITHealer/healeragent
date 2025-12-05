# File: src/services/alchemy_service.py

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, List, Optional

import websockets
from websockets.client import WebSocketClientProtocol

# Giả sử bạn đã có logger
logger = logging.getLogger(__name__)

class AlchemyWsService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws_url = f"wss://eth-mainnet.g.alchemy.com/v2/{self.api_key}"
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.subscription_id: Optional[str] = None

    async def connect(self):
        """Thiết lập kết nối đến WebSocket của Alchemy."""
        try:
            logger.info(f"Đang kết nối đến WebSocket của Alchemy...")
            self.websocket = await websockets.connect(self.ws_url)
            logger.info("Kết nối WebSocket đến Alchemy thành công.")
        except Exception as e:
            logger.error(f"Không thể kết nối đến WebSocket của Alchemy: {e}", exc_info=True)
            self.websocket = None

    async def subscribe_to_mined_transactions(
        self, 
        addresses: List[Dict[str, str]], 
        hashes_only: bool = False
    ):
        """Gửi lệnh đăng ký nhận các giao dịch mới."""
        if not self.websocket:
            logger.error("Chưa kết nối WebSocket. Hãy gọi connect() trước.")
            return

        subscription_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_subscribe",
            "params": [
                "alchemy_minedTransactions",
                {
                    # "addresses": addresses,
                    "includeRemoved": True,
                    "hashesOnly": hashes_only
                }
            ]
        }
        
        try:
            await self.websocket.send(json.dumps(subscription_payload))
            
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if "result" in response_data:
                self.subscription_id = response_data["result"]
            else:
                logger.error(f"Đăng ký thất bại: {response_data.get('error')}")

        except Exception as e:
            logger.error(f"Lỗi trong quá trình đăng ký: {e}", exc_info=True)

    async def listen_for_transactions(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Lắng nghe và trả về (yield) các giao dịch mới."""
        if not self.websocket or not self.subscription_id:
            logger.error("Chưa đăng ký nhận tin. Hãy gọi subscribe_to_mined_transactions() trước.")
            return
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get("method") == "eth_subscription":
                    transaction_data = data.get("params", {}).get("result", {})
                    yield transaction_data
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Kết nối WebSocket đến Alchemy đã bị đóng: {e}")
        except Exception as e:
            logger.error(f"Lỗi khi đang lắng nghe giao dịch: {e}", exc_info=True)

    async def close(self):
        """Đóng kết nối WebSocket."""
        if self.websocket:
            logger.info("Đang đóng kết nối WebSocket...")
            await self.websocket.close()
            logger.info("Kết nối đã đóng.")