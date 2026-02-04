import time
import threading
import logging
from enum import Enum
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"       # Normal - requests pass through
    OPEN = "open"           # Failing - requests rejected
    HALF_OPEN = "half_open" # Testing - limited requests allowed


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejecting requests"""
    def __init__(self, service_name: str, retry_after: float):
        self.service_name = service_name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit breaker open for '{service_name}'. "
            f"Retry after {retry_after:.1f} seconds"
        )

@dataclass
class CircuitStats:
    """Statistics for a single circuit"""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    last_state_change: float = field(default_factory=time.time)
    total_requests: int = 0
    total_rejections: int = 0
    consecutive_successes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/monitoring"""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_rejections": self.total_rejections,
            "consecutive_successes": self.consecutive_successes,
            "last_failure": datetime.fromtimestamp(self.last_failure_time).isoformat()
                if self.last_failure_time else None,
            "last_success": datetime.fromtimestamp(self.last_success_time).isoformat()
                if self.last_success_time else None,
        }
    
class CircuitBreaker:
    """
    Circuit Breaker implementation with per-service tracking
    Features:
    - Per-service circuit tracking
    - Configurable failure threshold
    - Automatic recovery testing
    - Thread-safe operations
    - Statistics tracking

    Args:

        failure_threshold: Number of failures before opening circuit
        reset_timeout: Seconds before trying to recover (OPEN -> HALF_OPEN)
        success_threshold: Successes needed in HALF_OPEN to close circuit
        half_open_max_requests: Max concurrent requests in HALF_OPEN state
    """
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 2,
        half_open_max_requests: int = 1
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.half_open_max_requests = half_open_max_requests
        self._circuits: Dict[str, CircuitStats] = {}
        self._lock = threading.RLock()
        self._half_open_requests: Dict[str, int] = {}
        self.logger = logging.getLogger(__name__)

    def _get_circuit(self, service_name: str) -> CircuitStats:
        """Get or create circuit for a service"""
        if service_name not in self._circuits:
            self._circuits[service_name] = CircuitStats()
        return self._circuits[service_name]
    def allow_request(self, service_name: str) -> bool:
        """
        Check if a request should be allowed
        Args:
            service_name: Name of the service/tool
        Returns:
            True if request should proceed, False if circuit is open
        """
        with self._lock:
            circuit = self._get_circuit(service_name)
            current_time = time.time()
            if circuit.state == CircuitState.CLOSED:
                circuit.total_requests += 1
                return True
            elif circuit.state == CircuitState.OPEN:
                # Check if timeout has passed
                time_since_open = current_time - circuit.last_state_change
                if time_since_open >= self.reset_timeout:
                    # Transition to HALF_OPEN
                    self._transition_to(service_name, CircuitState.HALF_OPEN)
                    circuit.total_requests += 1
                    self._half_open_requests[service_name] = 1
                    self.logger.info(
                        f"[CIRCUIT_BREAKER] {service_name}: OPEN -> HALF_OPEN "
                        f"(testing recovery after {time_since_open:.1f}s)"
                    )
                    return True
                else:
                    # Still in timeout period
                    circuit.total_rejections += 1
                    return False
            elif circuit.state == CircuitState.HALF_OPEN:
                # Allow limited requests for testing
                current_requests = self._half_open_requests.get(service_name, 0)
                if current_requests < self.half_open_max_requests:
                    self._half_open_requests[service_name] = current_requests + 1
                    circuit.total_requests += 1
                    return True
                else:
                    circuit.total_rejections += 1
                    return False
            return False
        
    def record_success(self, service_name: str) -> None:
        """
        Record a successful request
        Args:
            service_name: Name of the service/tool
        """
        with self._lock:
            circuit = self._get_circuit(service_name)
            current_time = time.time()
            circuit.success_count += 1
            circuit.last_success_time = current_time
            circuit.consecutive_successes += 1
            if circuit.state == CircuitState.HALF_OPEN:
                if circuit.consecutive_successes >= self.success_threshold:
                    # Service has recovered
                    self._transition_to(service_name, CircuitState.CLOSED)
                    circuit.failure_count = 0
                    circuit.consecutive_successes = 0
                    self._half_open_requests.pop(service_name, None)
                    self.logger.info(
                        f"[CIRCUIT_BREAKER] {service_name}: HALF_OPEN -> CLOSED "
                        f"(service recovered after {self.success_threshold} successes)"
                    )
            elif circuit.state == CircuitState.CLOSED:
                # Reset failure count on success
                if circuit.failure_count > 0:
                    circuit.failure_count = max(0, circuit.failure_count - 1)
    
    def record_failure(self, service_name: str, error: Optional[Exception] = None) -> None:
        """
        Record a failed request
        Args:
            service_name: Name of the service/tool
            error: Optional exception that caused the failure
        """
        with self._lock:
            circuit = self._get_circuit(service_name)
            current_time = time.time()
            circuit.failure_count += 1
            circuit.last_failure_time = current_time
            circuit.consecutive_successes = 0
            if circuit.state == CircuitState.CLOSED:
                if circuit.failure_count >= self.failure_threshold:
                    # Too many failures - open the circuit
                    self._transition_to(service_name, CircuitState.OPEN)
                    self.logger.warning(
                        f"[CIRCUIT_BREAKER] {service_name}: CLOSED -> OPEN "
                        f"(threshold {self.failure_threshold} failures reached). "
                        f"Error: {error}"
                    )
            elif circuit.state == CircuitState.HALF_OPEN:
                # Failed during recovery testing - back to OPEN
                self._transition_to(service_name, CircuitState.OPEN)
                self._half_open_requests.pop(service_name, None)
                self.logger.warning(
                    f"[CIRCUIT_BREAKER] {service_name}: HALF_OPEN -> OPEN "
                    f"(recovery test failed). Error: {error}"
                )
    
    def _transition_to(self, service_name: str, new_state: CircuitState) -> None:
        """Transition circuit to a new state"""
        circuit = self._get_circuit(service_name)
        old_state = circuit.state
        circuit.state = new_state
        circuit.last_state_change = time.time()
        self.logger.debug(
            f"[CIRCUIT_BREAKER] {service_name}: {old_state.value} -> {new_state.value}"
        )
    
    def get_state(self, service_name: str) -> CircuitState:
        """Get current state for a service"""
        with self._lock:
            return self._get_circuit(service_name).state
        
    def get_stats(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for service(s)
        Args:
            service_name: Specific service or None for all
        Returns:
            Statistics dictionary
        """
        with self._lock:
            if service_name:
                circuit = self._get_circuit(service_name)
                return {service_name: circuit.to_dict()}
            else:
                return {
                    name: circuit.to_dict()
                    for name, circuit in self._circuits.items()
                }
            
    def get_retry_after(self, service_name: str) -> float:
        """
        Get seconds until circuit might allow requests
        Args:
            service_name: Name of the service
        Returns:
            Seconds until retry, 0 if requests are allowed
        """
        with self._lock:
            circuit = self._get_circuit(service_name)
            if circuit.state != CircuitState.OPEN:
                return 0.0
            time_since_open = time.time() - circuit.last_state_change
            remaining = self.reset_timeout - time_since_open
            return max(0.0, remaining)

    def check_request(self, service_name: str) -> tuple[bool, float]:
        """
        Atomically check if request is allowed and get retry_after time.

        This method prevents race conditions by doing both checks in a single
        lock acquisition. Use this instead of separate allow_request() and
        get_retry_after() calls.

        Args:
            service_name: Name of the service/tool

        Returns:
            Tuple of (allowed: bool, retry_after: float)
            - If allowed=True, retry_after is always 0.0
            - If allowed=False, retry_after is seconds until circuit might allow
        """
        with self._lock:
            circuit = self._get_circuit(service_name)
            current_time = time.time()

            if circuit.state == CircuitState.CLOSED:
                circuit.total_requests += 1
                return (True, 0.0)

            elif circuit.state == CircuitState.OPEN:
                time_since_open = current_time - circuit.last_state_change

                if time_since_open >= self.reset_timeout:
                    # Timeout expired - transition to HALF_OPEN and allow
                    self._transition_to(service_name, CircuitState.HALF_OPEN)
                    circuit.total_requests += 1
                    self._half_open_requests[service_name] = 1
                    self.logger.info(
                        f"[CIRCUIT_BREAKER] {service_name}: OPEN -> HALF_OPEN "
                        f"(testing recovery after {time_since_open:.1f}s)"
                    )
                    return (True, 0.0)
                else:
                    # Still in timeout - reject and return remaining time
                    circuit.total_rejections += 1
                    remaining = self.reset_timeout - time_since_open
                    return (False, remaining)

            elif circuit.state == CircuitState.HALF_OPEN:
                current_requests = self._half_open_requests.get(service_name, 0)
                if current_requests < self.half_open_max_requests:
                    self._half_open_requests[service_name] = current_requests + 1
                    circuit.total_requests += 1
                    return (True, 0.0)
                else:
                    circuit.total_rejections += 1
                    # HALF_OPEN but max requests reached - retry soon
                    return (False, 1.0)

            return (False, self.reset_timeout)
    
    def reset(self, service_name: Optional[str] = None) -> None:
        """
        Reset circuit(s) to closed state
        Args:
            service_name: Specific service or None for all
        """
        with self._lock:
            if service_name:
                if service_name in self._circuits:
                    self._circuits[service_name] = CircuitStats()
                    self._half_open_requests.pop(service_name, None)
                    self.logger.info(f"[CIRCUIT_BREAKER] Reset circuit for {service_name}")
            else:
                self._circuits.clear()
                self._half_open_requests.clear()
                self.logger.info("[CIRCUIT_BREAKER] Reset all circuits")
    
    def force_open(self, service_name: str) -> None:
        """Force a circuit to open state (for maintenance)"""
        with self._lock:
            self._transition_to(service_name, CircuitState.OPEN)
            self.logger.info(f"[CIRCUIT_BREAKER] Force opened circuit for {service_name}")
    
    def force_close(self, service_name: str) -> None:
        """Force a circuit to closed state"""
        with self._lock:
            circuit = self._get_circuit(service_name)
            circuit.failure_count = 0
            circuit.consecutive_successes = 0
            self._transition_to(service_name, CircuitState.CLOSED)
            self._half_open_requests.pop(service_name, None)
            self.logger.info(f"[CIRCUIT_BREAKER] Force closed circuit for {service_name}")

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_circuit_breaker: Optional[CircuitBreaker] = None

def get_circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    success_threshold: int = 2
) -> CircuitBreaker:
    """
    Get or create global circuit breaker instance
    Args:
        failure_threshold: Failures before opening
        reset_timeout: Seconds before recovery test
        success_threshold: Successes to close
    Returns:
        CircuitBreaker instance
    """
    global _global_circuit_breaker
    if _global_circuit_breaker is None:
        _global_circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            reset_timeout=reset_timeout,
            success_threshold=success_threshold
        )
    return _global_circuit_breaker

def reset_circuit_breaker() -> None:
    """Reset global circuit breaker"""
    global _global_circuit_breaker
    if _global_circuit_breaker:
        _global_circuit_breaker.reset()


# ============================================================================
# ASYNC DECORATOR
# ============================================================================

def with_circuit_breaker(
    service_name: str,
    fallback_result: any = None,
    raise_on_open: bool = False,
):
    """
    Decorator to wrap async functions with circuit breaker.

    Usage:
        @with_circuit_breaker("openai_api")
        async def call_openai(prompt: str) -> str:
            ...

    Args:
        service_name: Name for the circuit
        fallback_result: Return value when circuit is open (if raise_on_open=False)
        raise_on_open: If True, raises CircuitBreakerOpenError when open
    """
    import functools
    import asyncio

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            circuit = get_circuit_breaker()

            # Use atomic check_request to avoid race condition
            allowed, retry_after = circuit.check_request(service_name)
            if not allowed:
                if raise_on_open:
                    raise CircuitBreakerOpenError(service_name, retry_after)
                return fallback_result

            try:
                result = await func(*args, **kwargs)
                circuit.record_success(service_name)
                return result
            except Exception as e:
                circuit.record_failure(service_name, e)
                raise

        return wrapper
    return decorator


# Alias for consistency with architecture document
CircuitOpenError = CircuitBreakerOpenError


# ============================================================================
# PREDEFINED CIRCUIT NAMES
# ============================================================================

CIRCUIT_LLM_OPENAI = "llm_openai"
CIRCUIT_LLM_ANTHROPIC = "llm_anthropic"
CIRCUIT_LLM_GOOGLE = "llm_google"
CIRCUIT_FMP_API = "fmp_api"
CIRCUIT_BINANCE_API = "binance_api"
CIRCUIT_NEWS_API = "news_api"
CIRCUIT_TAVILY_API = "tavily_api"