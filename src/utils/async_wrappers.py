"""
Async wrappers for synchronous blocking operations.

This module provides utilities to safely run blocking operations
(like yfinance, heavy CPU work) without blocking the async event loop.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from typing import TypeVar, Callable, Any

from src.utils.logger.custom_logging import LoggerMixin

# Type variable for generic return type
T = TypeVar('T')

# =============================================================================
# THREAD POOL EXECUTOR - Singleton
# =============================================================================

# Global thread pool for blocking I/O operations
# Limited to prevent thread exhaustion
_BLOCKING_IO_EXECUTOR = ThreadPoolExecutor(
    max_workers=10,  # Limit concurrent blocking operations
    thread_name_prefix="blocking_io_"
)

# Separate pool for CPU-intensive operations
_CPU_EXECUTOR = ThreadPoolExecutor(
    max_workers=4,  # Limit to prevent CPU contention
    thread_name_prefix="cpu_intensive_"
)

logger_mixin = LoggerMixin()
logger = logger_mixin.logger


# =============================================================================
# ASYNC WRAPPER FUNCTIONS
# =============================================================================

async def run_in_thread(
    func: Callable[..., T],
    *args,
    executor: ThreadPoolExecutor = None,
    **kwargs
) -> T:
    """
    Run a blocking function in a thread pool to avoid blocking the event loop.

    Usage:
        # Run blocking yfinance call
        result = await run_in_thread(yf.Ticker, "AAPL")

        # Or with method call
        result = await run_in_thread(stock.history, start="2024-01-01")

    Args:
        func: The blocking function to run
        *args: Positional arguments for the function
        executor: Optional custom executor (defaults to blocking I/O pool)
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function call
    """
    if executor is None:
        executor = _BLOCKING_IO_EXECUTOR

    loop = asyncio.get_running_loop()

    # Create partial function with kwargs
    if kwargs:
        func_with_args = partial(func, *args, **kwargs)
    else:
        func_with_args = partial(func, *args) if args else func

    try:
        result = await loop.run_in_executor(executor, func_with_args)
        return result
    except Exception as e:
        logger.error(f"[ASYNC_WRAPPER] Error in thread execution: {e}")
        raise


async def run_cpu_intensive(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run CPU-intensive operations in a dedicated thread pool.

    Usage:
        result = await run_cpu_intensive(heavy_computation, data)

    Args:
        func: The CPU-intensive function to run
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        The result of the function call
    """
    return await run_in_thread(func, *args, executor=_CPU_EXECUTOR, **kwargs)


def async_wrap(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to wrap a synchronous function to run in thread pool.

    Usage:
        @async_wrap
        def blocking_function(x, y):
            # ... blocking operation
            return result

        # Now can be awaited
        result = await blocking_function(1, 2)
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await run_in_thread(func, *args, **kwargs)
    return wrapper


# =============================================================================
# YFINANCE ASYNC WRAPPERS
# =============================================================================

async def get_yfinance_history(
    ticker: str,
    start: str = None,
    end: str = None,
    period: str = None,
    interval: str = "1d"
) -> Any:
    """
    Async wrapper for yfinance history fetching.

    Usage:
        data = await get_yfinance_history("AAPL", period="1y")

    Args:
        ticker: Stock ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        period: Alternative to start/end (e.g., "1y", "6mo")
        interval: Data interval (1d, 1h, etc.)

    Returns:
        pandas DataFrame with historical data
    """
    import yfinance as yf

    def _fetch():
        stock = yf.Ticker(ticker)
        if period:
            return stock.history(period=period, interval=interval)
        else:
            return stock.history(start=start, end=end, interval=interval)

    logger.debug(f"[YFINANCE_ASYNC] Fetching {ticker} in thread pool")
    result = await run_in_thread(_fetch)
    logger.debug(f"[YFINANCE_ASYNC] Completed fetching {ticker}")
    return result


async def get_yfinance_info(ticker: str) -> dict:
    """
    Async wrapper for yfinance stock info.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Dict with stock information
    """
    import yfinance as yf

    def _fetch():
        stock = yf.Ticker(ticker)
        return stock.info

    return await run_in_thread(_fetch)


# =============================================================================
# TIMEOUT WRAPPER
# =============================================================================

async def with_timeout(
    coro,
    timeout_seconds: float,
    default: Any = None,
    on_timeout: Callable = None
):
    """
    Run a coroutine with timeout, returning default or calling handler on timeout.

    Usage:
        result = await with_timeout(
            fetch_data(),
            timeout_seconds=30,
            default=[],
            on_timeout=lambda: logger.warning("Timeout!")
        )

    Args:
        coro: The coroutine to run
        timeout_seconds: Maximum time to wait
        default: Default value to return on timeout
        on_timeout: Optional callback to run on timeout

    Returns:
        Result of coroutine or default value on timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.warning(f"[TIMEOUT] Operation timed out after {timeout_seconds}s")
        if on_timeout:
            on_timeout()
        return default


# =============================================================================
# CLEANUP
# =============================================================================

def shutdown_executors():
    """Shutdown thread pool executors gracefully"""
    try:
        _BLOCKING_IO_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        _CPU_EXECUTOR.shutdown(wait=True, cancel_futures=True)
        logger.info("[ASYNC_WRAPPER] Thread pools shutdown successfully")
    except Exception as e:
        logger.error(f"[ASYNC_WRAPPER] Error shutting down executors: {e}")
