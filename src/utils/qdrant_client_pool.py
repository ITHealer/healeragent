"""
Singleton Qdrant Client Pool

This module provides a centralized Qdrant client to prevent
creating multiple connections and potential blocking issues.
"""
import asyncio
from typing import Optional
from qdrant_client import QdrantClient

from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin


# =============================================================================
# CONFIGURATION
# =============================================================================

# Reduced timeout to fail fast instead of blocking
QDRANT_TIMEOUT = 60  # seconds (was 600)
QDRANT_GRPC_PORT = 6334
QDRANT_PREFER_GRPC = True  # gRPC is faster than HTTP


# =============================================================================
# SINGLETON QDRANT CLIENT
# =============================================================================

class QdrantClientPool(LoggerMixin):
    """
    Singleton Qdrant client pool.

    Usage:
        client = await QdrantClientPool.get_client()
        result = client.search(...)
    """

    _instance: Optional['QdrantClientPool'] = None
    _lock: asyncio.Lock = asyncio.Lock()
    _client: Optional[QdrantClient] = None

    def __init__(self):
        super().__init__()
        self._init_lock = asyncio.Lock()
        self.logger.info("[QDRANT_POOL] QdrantClientPool initialized")

    @classmethod
    async def get_instance(cls) -> 'QdrantClientPool':
        """Get or create singleton instance"""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    async def get_client(self) -> QdrantClient:
        """Get the shared Qdrant client"""
        if self._client is None:
            async with self._init_lock:
                if self._client is None:
                    await self._create_client()
        return self._client

    async def _create_client(self):
        """Create Qdrant client with optimized settings"""
        try:
            qdrant_url = settings.QDRANT_ENDPOINT

            # Parse URL to extract host/port
            if qdrant_url.startswith("http://"):
                host = qdrant_url.replace("http://", "").split(":")[0]
                port = int(qdrant_url.split(":")[-1]) if ":" in qdrant_url.split("//")[-1] else 6333
            else:
                host = qdrant_url.split(":")[0]
                port = 6333

            self._client = QdrantClient(
                host=host,
                port=port,
                grpc_port=QDRANT_GRPC_PORT,
                prefer_grpc=QDRANT_PREFER_GRPC,
                timeout=QDRANT_TIMEOUT,
            )

            self.logger.info(
                f"[QDRANT_POOL] Created Qdrant client: {host}:{port} "
                f"(grpc={QDRANT_PREFER_GRPC}, timeout={QDRANT_TIMEOUT}s)"
            )

        except Exception as e:
            self.logger.error(f"[QDRANT_POOL] Failed to create client: {e}")
            # Fallback to URL-based connection
            self._client = QdrantClient(
                url=settings.QDRANT_ENDPOINT,
                timeout=QDRANT_TIMEOUT,
            )

    async def health_check(self) -> bool:
        """Check if Qdrant connection is healthy"""
        try:
            client = await self.get_client()
            # Simple health check - get collections list
            collections = client.get_collections()
            return True
        except Exception as e:
            self.logger.error(f"[QDRANT_POOL] Health check failed: {e}")
            return False

    async def reconnect(self):
        """Force reconnection to Qdrant"""
        async with self._init_lock:
            try:
                if self._client:
                    self._client.close()
            except Exception:
                pass
            self._client = None
            await self._create_client()

    def close(self):
        """Close the Qdrant client"""
        try:
            if self._client:
                self._client.close()
                self._client = None
                self.logger.info("[QDRANT_POOL] Client closed")
        except Exception as e:
            self.logger.error(f"[QDRANT_POOL] Error closing client: {e}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def get_qdrant_client() -> QdrantClient:
    """Get the shared Qdrant client"""
    pool = await QdrantClientPool.get_instance()
    return await pool.get_client()


async def qdrant_health_check() -> bool:
    """Check Qdrant health"""
    pool = await QdrantClientPool.get_instance()
    return await pool.health_check()
