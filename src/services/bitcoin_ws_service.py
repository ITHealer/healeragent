# File: src/services/bitcoin_ws_service.py (tạo file mới)

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)
BLOCKCHAIN_INFO_WS_URL = "wss://ws.blockchain.info/inv"

class BitcoinWsService:
    def __init__(self):
        self.websocket: Optional[WebSocketClientProtocol] = None

    async def connect(self):
        try:
            logger.info("Đang kết nối đến WebSocket của Blockchain.com...")
            self.websocket = await websockets.connect(BLOCKCHAIN_INFO_WS_URL)
            logger.info("Kết nối WebSocket đến Blockchain.com thành công.")
        except Exception as e:
            logger.error(f"Không thể kết nối đến WebSocket của Blockchain.com: {e}", exc_info=True)
            self.websocket = None

    async def subscribe_to_new_transactions(self):
        """Gửi lệnh đăng ký nhận tất cả các giao dịch Bitcoin mới."""
        if not self.websocket:
            logger.error("Chưa kết nối WebSocket. Hãy gọi connect() trước.")
            return

        subscription_payload = {"op": "unconfirmed_sub"}
        try:
            logger.info(f"Đang gửi lệnh đăng ký: {subscription_payload}")
            await self.websocket.send(json.dumps(subscription_payload))
            logger.info("Đăng ký nhận giao dịch Bitcoin mới thành công.")
        except Exception as e:
            logger.error(f"Lỗi trong quá trình đăng ký Bitcoin: {e}", exc_info=True)

    async def listen_for_transactions(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Lắng nghe và trả về các giao dịch Bitcoin mới."""
        if not self.websocket: return
        
        logger.info("Bắt đầu lắng nghe các giao dịch Bitcoin mới...")
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data.get("op") == "utx":
                    yield data.get("x", {})
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Kết nối WebSocket đến Blockchain.com đã bị đóng: {e}")
        except Exception as e:
            logger.error(f"Lỗi khi đang lắng nghe giao dịch Bitcoin: {e}", exc_info=True)

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            logger.info("Kết nối WebSocket Bitcoin đã đóng.")