# src/handlers/content_processor_manager.py
from typing import Dict, Optional
import logging
from threading import Lock

from src.handlers.content_processor import ContentProcessor
from src.utils.logger.custom_logging import LoggerMixin

logger = LoggerMixin().logger


class ContentProcessorManager:
    """
    Singleton manager for ContentProcessor instances.
    Caches and reuses processors based on (model_name, provider_type) key.
    """
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._processors: Dict[str, ContentProcessor] = {}
        self._lock = Lock()
        self._initialized = True
        logger.info("[ContentProcessorManager] Initialized successfully")
    
    def get_processor(
        self,
        model_name: str,
        provider_type: str,
        api_key: Optional[str]
    ) -> ContentProcessor:
        """
        Get or create a ContentProcessor instance.
        Uses caching to avoid re-initialization.
        
        Args:
            model_name: Model name (e.g., 'gpt-4.1-nano', 'gpt-oss:20b')
            provider_type: Provider type (openai, ollama, gemini)
            api_key: API key for the provider
            
        Returns:
            ContentProcessor instance (cached or new)
        """
        # Create cache key
        cache_key = f"{provider_type}:{model_name}"
        
        # Check if processor exists in cache
        if cache_key in self._processors:
            logger.debug(f"Reusing cached processor: {cache_key}")
            return self._processors[cache_key]
        
        # Create new processor with lock (thread-safe)
        with self._lock:
            # Double-check after acquiring lock
            if cache_key in self._processors:
                return self._processors[cache_key]
            
            logger.info(f"Creating new processor: {cache_key}")
            
            try:
                processor = ContentProcessor(
                    api_key=api_key,
                    model_name=model_name,
                    provider_type=provider_type
                )
                
                # Cache the processor
                self._processors[cache_key] = processor
                logger.info(f"Processor cached: {cache_key}")
                
                return processor
                
            except Exception as e:
                logger.error(f"Failed to create processor {cache_key}: {e}")
                raise
    
    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Clear processor cache.
        
        Args:
            cache_key: Specific key to clear, or None to clear all
        """
        with self._lock:
            if cache_key:
                if cache_key in self._processors:
                    processor = self._processors.pop(cache_key)
                    processor.cleanup()
                    logger.info(f"Cleared processor cache: {cache_key}")
            else:
                for processor in self._processors.values():
                    processor.cleanup()
                self._processors.clear()
                logger.info("Cleared all processor caches")
    
    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics"""
        return {
            "cached_processors": len(self._processors),
            "cache_keys": list(self._processors.keys())
        }


# Global instance
processor_manager = ContentProcessorManager()