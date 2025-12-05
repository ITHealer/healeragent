import asyncio
import logging
from typing import List, Dict, Set, Any, Coroutine
from fastapi import WebSocket
from starlette.websockets import WebSocketState
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK


logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.topic_subscriptions: Dict[str, Set[WebSocket]] = {}
        self.client_topics: Dict[WebSocket, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, websocket: WebSocket, topic: str):
        async with self._lock:
            if topic not in self.topic_subscriptions:
                self.topic_subscriptions[topic] = set()
            self.topic_subscriptions[topic].add(websocket)

            if websocket not in self.client_topics:
                self.client_topics[websocket] = set()
            self.client_topics[websocket].add(topic)
        logger.info(f"Client {websocket.client} subscribed to topic: {topic}")

    async def unsubscribe(self, websocket: WebSocket, topic: str) -> bool:
        topic_became_empty = False
        async with self._lock:
            if topic in self.topic_subscriptions:
                self.topic_subscriptions[topic].discard(websocket)
                if not self.topic_subscriptions[topic]:
                    del self.topic_subscriptions[topic]
                    topic_became_empty = True
            if websocket in self.client_topics:
                self.client_topics[websocket].discard(topic)
                if not self.client_topics[websocket]:
                    del self.client_topics[websocket]
        
        if topic_became_empty:
            logger.info(f"Topic {topic} is now empty after client unsubscribed.")
        return topic_became_empty

    async def disconnect(self, websocket: WebSocket) -> List[str]:
        topics_that_might_become_empty = []
        async with self._lock:
            client_subscribed_topics = self.client_topics.pop(websocket, set())
            for topic in client_subscribed_topics:
                if topic in self.topic_subscriptions:
                    self.topic_subscriptions[topic].discard(websocket)
                    if not self.topic_subscriptions[topic]:
                        del self.topic_subscriptions[topic]
                        topics_that_might_become_empty.append(topic)
        logger.info(f"Client {websocket.client} disconnected. Removed from topics: {client_subscribed_topics}")
        return topics_that_might_become_empty

    async def broadcast_to_topic(self, topic: str, message_data: dict):
        subscribers_to_notify: List[WebSocket] = []
        async with self._lock:
            if topic in self.topic_subscriptions:
                subscribers_to_notify = list(self.topic_subscriptions[topic])

        if not subscribers_to_notify:
            return

        # Danh sách các client cần remove sau khi gửi thất bại
        disconnected_clients = []

        for ws_client in subscribers_to_notify:
            try:
                # Kiểm tra state của websocket trước khi gửi
                if hasattr(ws_client, 'client_state') and ws_client.client_state == WebSocketState.DISCONNECTED:
                    disconnected_clients.append(ws_client)
                    continue
                # if ws_client.client_state == WebSocketState.DISCONNECTED:
                #     disconnected_clients.append(ws_client)
                #     continue

                await asyncio.wait_for(
                    ws_client.send_json(message_data),
                    timeout=5.0
                )
                
                # await ws_client.send_json(message_data)
            except ConnectionClosedError:
                logger.warning(f"Connection closed for client on topic {topic}")
                disconnected_clients.append(ws_client)
            except ConnectionClosedOK:
                logger.debug(f"Connection closed normally for client on topic {topic}")
                disconnected_clients.append(ws_client)
            except Exception as e:
                logger.warning(f"Error sending to client on topic {topic}: {e}")
                disconnected_clients.append(ws_client)

        # Cleanup disconnected clients
        if disconnected_clients:
            for ws_client in disconnected_clients:
                topics_to_check = await self.disconnect(ws_client)
                # Trigger cleanup cho các topic rỗng
                for empty_topic in topics_to_check:
                    await stop_fetcher_task_if_unneeded(empty_topic)
                    
global_connection_manager = ConnectionManager()
active_data_fetcher_tasks: Dict[str, asyncio.Task] = {}
fetcher_management_lock = asyncio.Lock()

async def ensure_fetcher_task_running(topic: str, task_creator_coro: Coroutine):
    async with fetcher_management_lock:
        if topic not in active_data_fetcher_tasks or active_data_fetcher_tasks[topic].done():
            # logger.info(f"Fetcher for topic '{topic}' not running. Starting new one.")
            task = asyncio.create_task(task_creator_coro)
            active_data_fetcher_tasks[topic] = task
        else:
            logger.debug(f"Fetcher for topic '{topic}' is already running.")

async def stop_fetcher_task_if_unneeded(topic: str):
    async with fetcher_management_lock:
        should_stop = False
        async with global_connection_manager._lock:
            if topic not in global_connection_manager.topic_subscriptions:
                should_stop = True
        
        if should_stop:
            task_to_cancel = active_data_fetcher_tasks.pop(topic, None)
            if task_to_cancel and not task_to_cancel.done():
                # logger.info(f"No subscribers for topic '{topic}'. Cancelling data fetcher task.")
                task_to_cancel.cancel()
                try:
                    await task_to_cancel
                except asyncio.CancelledError:
                    logger.debug(f"Task for '{topic}' cancellation processed.")