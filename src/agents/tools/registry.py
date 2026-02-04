import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.agents.tools.base import BaseTool, ToolSchema, ToolOutput
from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    get_circuit_breaker
)

class ToolRegistry:
    """
    Singleton Tool Registry
    
    Central manager for all atomic tools with:
    - Tool discovery & registration
    - Schema management
    - Execution routing
    - Performance monitoring
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        circuit_breaker_enabled: bool = True,
        circuit_failure_threshold: int = 5,
        circuit_reset_timeout: float = 60.0
    ):
        if self._initialized:
            return
        
        self.logger = logging.getLogger(__name__)
        self._tools: Dict[str, BaseTool] = {}
        self._schemas: Dict[str, ToolSchema] = {}
        self._categories: Dict[str, List[str]] = {}
        self._execution_stats: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breaker configuration
        self._circuit_breaker_enabled = circuit_breaker_enabled
        if circuit_breaker_enabled:
            self._circuit_breaker = get_circuit_breaker(
                failure_threshold=circuit_failure_threshold,
                reset_timeout=circuit_reset_timeout
            )
            self.logger.info(
                f"ToolRegistry initialized with circuit breaker "
                f"(threshold={circuit_failure_threshold}, timeout={circuit_reset_timeout}s)"
            )
        else:
            self._circuit_breaker = None
            self.logger.info("ToolRegistry initialized (circuit breaker disabled)")

        self._initialized = True
        self.logger.info("ToolRegistry initialized")
    
    # ========================================================================
    # REGISTRATION
    # ========================================================================
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a single tool"""
        schema = tool.get_schema()
        tool_name = schema.name
        category = schema.category
        
        if tool_name in self._tools:
            self.logger.debug(f"Tool {tool_name} already registered, skipping")
            return
        
        self._tools[tool_name] = tool
        self._schemas[tool_name] = schema
        
        if category not in self._categories:
            self._categories[category] = []
        
        if tool_name not in self._categories[category]:
            self._categories[category].append(tool_name)
        
        self._execution_stats[tool_name] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_execution_time_ms": 0,
            "avg_execution_time_ms": 0
        }
        
        self.logger.info(f"Registered tool: {tool_name} (category: {category})")
    
    def register_tools(self, tools: List[BaseTool]) -> int:
        """Register multiple tools, returns count of newly registered"""
        count = 0
        for tool in tools:
            if tool.get_schema().name not in self._tools:
                self.register_tool(tool)
                count += 1
        return count
    
    # ========================================================================
    # GETTERS
    # ========================================================================
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get tool instance by name"""
        return self._tools.get(tool_name)
    
    def get_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """Get tool schema by name"""
        return self._schemas.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools"""
        return self._tools.copy()
    
    def get_all_schemas(self) -> Dict[str, ToolSchema]:
        """Get all tool schemas"""
        return self._schemas.copy()
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all tool schemas in OpenAI function calling format"""
        return [schema.to_json_schema() for schema in self._schemas.values()]
    
    def get_tools_by_category(self, category: str) -> Dict[str, BaseTool]:
        """Get all tools in a specific category"""
        tool_names = self._categories.get(category, [])
        return {name: self._tools[name] for name in tool_names if name in self._tools}
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self._categories.keys())
    
    def list_tools(self) -> Dict[str, List[str]]:
        """List all tools organized by category"""
        return self._categories.copy()
    
    # ========================================================================
    # EXECUTION
    # ========================================================================
    
    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        bypass_circuit_breaker: bool = False
    ) -> ToolOutput:
        """Execute a tool by name with circuit breaker protection"""
        tool = self.get_tool(tool_name)

        if not tool:
            error_msg = f"Tool '{tool_name}' not found"
            return ToolOutput(
                tool_name=tool_name,
                status="error",
                error=error_msg,
                formatted_context=f"Error: {error_msg}. This tool does not exist in the system."
            )

        # Check circuit breaker using atomic check_request to avoid race condition
        if self._circuit_breaker_enabled and self._circuit_breaker and not bypass_circuit_breaker:
            allowed, retry_after = self._circuit_breaker.check_request(tool_name)
            if not allowed:
                self.logger.warning(
                    f"[CIRCUIT_BREAKER] Tool '{tool_name}' is unavailable. "
                    f"Circuit open, retry after {retry_after:.1f}s"
                )
                error_msg = (
                    f"Tool '{tool_name}' temporarily unavailable (circuit breaker open). "
                    f"Retry after {retry_after:.1f} seconds."
                )
                return ToolOutput(
                    tool_name=tool_name,
                    status="error",
                    error=error_msg,
                    formatted_context=f"Error: {error_msg}",
                    metadata={
                        "circuit_breaker": True,
                        "retry_after_seconds": retry_after,
                        "circuit_state": self._circuit_breaker.get_state(tool_name).value
                    }
                )

        try:
            result = await tool.safe_execute(**params)
            # Record success/failure with circuit breaker
            if self._circuit_breaker_enabled and self._circuit_breaker and not bypass_circuit_breaker:
                if result.status == "success" or result.status == "200":
                    self._circuit_breaker.record_success(tool_name)
                elif result.status == "error":
                    # Only count as failure if it's a system/network error, not validation error
                    error_msg = result.error or ""
                    is_transient_error = any(
                        keyword in error_msg.lower()
                        for keyword in ["timeout", "connection", "network", "unavailable", "503", "502", "500"]
                    )
                    if is_transient_error:
                        self._circuit_breaker.record_failure(tool_name, Exception(error_msg))
            self._update_stats(tool_name, result)
            return result
        except Exception as e:
            # Record failure for unexpected exceptions
            if self._circuit_breaker_enabled and self._circuit_breaker and not bypass_circuit_breaker:
                self._circuit_breaker.record_failure(tool_name, e)
            self.logger.error(f"[TOOL_EXECUTION] Unexpected error in {tool_name}: {e}")
            error_msg = str(e)
            return ToolOutput(
                tool_name=tool_name,
                status="error",
                error=error_msg,
                formatted_context=f"Error executing {tool_name}: {error_msg}",
                metadata={"exception_type": type(e).__name__}
            )

    
    async def execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolOutput]:
        """Execute multiple tools in parallel"""
        import asyncio
        
        async def execute_single(call: Dict) -> ToolOutput:
            tool_name = call.get("tool_name")
            params = call.get("params", {})
            return await self.execute_tool(tool_name, params)
        
        return await asyncio.gather(*[execute_single(c) for c in tool_calls])
    
    def _update_stats(self, tool_name: str, result: ToolOutput) -> None:
        """Update execution statistics"""
        if tool_name not in self._execution_stats:
            return
        
        stats = self._execution_stats[tool_name]
        stats["total_calls"] += 1
        
        if result.status == "success" or result.status == "200":
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
        
        if result.execution_time_ms:
            stats["total_execution_time_ms"] += result.execution_time_ms
            stats["avg_execution_time_ms"] = (
                stats["total_execution_time_ms"] / stats["total_calls"]
            )
    
    # ========================================================================
    # SUMMARY & STATS
    # ========================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Get registry summary"""
        return {
            "total_tools": len(self._tools),
            "categories": {cat: len(tools) for cat, tools in self._categories.items()},
            "tools_by_category": self._categories.copy()
        }
    
    def get_execution_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Get execution statistics"""
        if tool_name:
            return self._execution_stats.get(tool_name, {})
        return self._execution_stats.copy()
    
    def get_tool_catalog(
        self, 
        categories: Optional[List[str]] = None,
        detail_level: str = "description"
    ) -> List[Dict[str, Any]]:
        """
        Get tool catalog for LLM with progressive disclosure
        
        Args:
            categories: Filter by categories
            detail_level: "name" | "description" | "full"
        """
        filtered_tools = {}
        
        if categories:
            for cat in categories:
                filtered_tools.update(self.get_tools_by_category(cat))
        else:
            filtered_tools = self._tools
        
        catalog = []
        for tool_name, tool in filtered_tools.items():
            schema = tool.get_schema()
            
            if detail_level == "name":
                catalog.append({"name": tool_name, "category": schema.category})
            elif detail_level == "description":
                catalog.append({
                    "name": tool_name,
                    "category": schema.category,
                    "description": schema.description,
                    "usage_hints": schema.usage_hints[:3] if schema.usage_hints else [],
                    "requires_symbol": schema.requires_symbol
                })
            else:
                catalog.append(schema.to_json_schema())
        
        return catalog
    
    def reset_stats(self) -> None:
        """Reset all execution statistics"""
        for stats in self._execution_stats.values():
            stats.update({
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_execution_time_ms": 0,
                "avg_execution_time_ms": 0
            })

    # ========================================================================
    # CIRCUIT BREAKER MANAGEMENT
    # ========================================================================

    def get_circuit_breaker_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get circuit breaker statistics

        Args:
            tool_name: Specific tool or None for all

        Returns:
            Circuit breaker stats dictionary
        """
        if not self._circuit_breaker_enabled or not self._circuit_breaker:
            return {"enabled": False}

        stats = self._circuit_breaker.get_stats(tool_name)
        return {
            "enabled": True,
            "failure_threshold": self._circuit_breaker.failure_threshold,
            "reset_timeout": self._circuit_breaker.reset_timeout,
            "circuits": stats
        }

    def get_tool_health(self) -> Dict[str, Any]:
        """
        Get health status of all tools based on circuit breaker state

        Returns:
            Health status for each tool
        """
        health = {}

        for tool_name in self._tools.keys():
            if self._circuit_breaker_enabled and self._circuit_breaker:
                state = self._circuit_breaker.get_state(tool_name)
                retry_after = self._circuit_breaker.get_retry_after(tool_name)

                health[tool_name] = {
                    "status": "healthy" if state == CircuitState.CLOSED else
                            "degraded" if state == CircuitState.HALF_OPEN else
                            "unhealthy",
                    "circuit_state": state.value,
                    "retry_after_seconds": retry_after if state == CircuitState.OPEN else None
                }
            else:
                health[tool_name] = {
                    "status": "healthy",
                    "circuit_state": "disabled",
                    "retry_after_seconds": None
                }

        return health

    def reset_circuit_breaker(self, tool_name: Optional[str] = None) -> None:
        """
        Reset circuit breaker for tool(s)

        Args:
            tool_name: Specific tool or None for all
        """
        if self._circuit_breaker_enabled and self._circuit_breaker:
            self._circuit_breaker.reset(tool_name)
            self.logger.info(
                f"[CIRCUIT_BREAKER] Reset circuit for: {tool_name or 'all tools'}"
            )

    def force_open_circuit(self, tool_name: str) -> None:
        """Force open circuit for a tool (for maintenance)"""
        if self._circuit_breaker_enabled and self._circuit_breaker:
            self._circuit_breaker.force_open(tool_name)

    def force_close_circuit(self, tool_name: str) -> None:
        """Force close circuit for a tool"""
        if self._circuit_breaker_enabled and self._circuit_breaker:
            self._circuit_breaker.force_close(tool_name)

# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get global registry instance (singleton)"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def initialize_registry() -> ToolRegistry:
    """
    Initialize and populate registry with all available tools
    
    Call once at application startup.
    """
    from src.agents.tools.tool_loader import load_all_tools
    
    registry, failed = load_all_tools()
    return registry