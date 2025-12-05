"""
Tool Registry - Central Manager for Atomic Tools

Registry Pattern cho Tool Management:
- Automatic tool discovery
- Schema-based validation
- Performance monitoring
- Parallel execution support

Usage:
    # Initialize (singleton)
    registry = ToolRegistry()
    
    # Register tools
    registry.register_tool(GetStockPriceTool())
    registry.register_tool(GetTechnicalIndicatorsTool())
    
    # Get tool by name
    tool = registry.get_tool("getStockPrice")
    
    # Get all tool schemas (cho Planning Agent)
    schemas = registry.get_all_tool_schemas()
    
    # Execute tool
    result = await registry.execute_tool(
        tool_name="getStockPrice",
        params={"symbol": "AAPL"}
    )
"""

import logging
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import json

from src.agents.tools.base import BaseTool, ToolSchema, ToolOutput, execute_tools_parallel


class ToolRegistry:
    """
    Singleton Tool Registry
    
    Central manager cho tất cả atomic tools:
    - Tool discovery & registration
    - Schema management
    - Execution routing
    - Performance monitoring
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize registry (only once)"""
        if self._initialized:
            return
        
        self.logger = logging.getLogger(__name__)
        self._tools: Dict[str, BaseTool] = {}
        self._schemas: Dict[str, ToolSchema] = {}
        self._categories: Dict[str, List[str]] = {}
        self._execution_stats: Dict[str, Dict[str, Any]] = {}
        
        self._initialized = True
        self.logger.info("ToolRegistry initialized")
    
    def register_tool(self, tool: BaseTool) -> None:
        """
        Register một atomic tool
        
        Args:
            tool: Tool instance to register
            
        Raises:
            ValueError: If tool has no schema or duplicate name
        """
        schema = tool.get_schema()
        tool_name = schema.name
        
        # Check duplicate
        if tool_name in self._tools:
            self.logger.warning(f"Tool {tool_name} already registered, overwriting")
        
        # Register
        self._tools[tool_name] = tool
        self._schemas[tool_name] = schema
        
        # Update category index
        category = schema.category
        if category not in self._categories:
            self._categories[category] = []
        
        if tool_name not in self._categories[category]:
            self._categories[category].append(tool_name)
        
        # Initialize stats
        self._execution_stats[tool_name] = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_execution_time_ms": 0,
            "avg_execution_time_ms": 0
        }
        
        self.logger.info(
            f"✅ Registered tool: {tool_name} (category: {category})"
        )
    
    def register_tools(self, tools: List[BaseTool]) -> None:
        """Register multiple tools at once"""
        for tool in tools:
            self.register_tool(tool)
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get tool instance by name
        
        Args:
            tool_name: Name of tool
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def get_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """
        Get tool schema by name
        
        Args:
            tool_name: Name of tool
            
        Returns:
            Tool schema or None if not found
        """
        return self._schemas.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """Get all registered tools"""
        return self._tools.copy()
    
    def get_all_schemas(self) -> Dict[str, ToolSchema]:
        """Get all tool schemas"""
        return self._schemas.copy()
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Get all tool schemas in OpenAI function calling format
        
        This is used by Planning Agent để understand available tools
        
        Returns:
            List of tool schemas in OpenAI format
        """
        return [
            schema.to_json_schema()
            for schema in self._schemas.values()
        ]
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """
        Get tool names by category
        
        Args:
            category: Category name (e.g., "price", "technical", "risk")
            
        Returns:
            List of tool names in category
        """
        return self._categories.get(category, []).copy()
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self._categories.keys())
    
    def list_tools(self) -> Dict[str, List[str]]:
        """
        List all tools organized by category
        
        Returns:
            Dict mapping category to list of tool names
        """
        return self._categories.copy()
    
    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any]
    ) -> ToolOutput:
        """
        Execute a tool by name
        
        Args:
            tool_name: Name of tool to execute
            params: Tool parameters
            
        Returns:
            ToolOutput
            
        Raises:
            ValueError: If tool not found
        """
        tool = self.get_tool(tool_name)
        
        if not tool:
            available_tools = list(self._tools.keys())
            error_msg = f"Tool '{tool_name}' not found. Available: {available_tools}"
            self.logger.error(error_msg)
            
            return ToolOutput(
                tool_name=tool_name,
                status="error",
                error=error_msg
            )
        
        # Execute
        result = await tool.safe_execute(**params)
        
        # Update stats
        self._update_stats(tool_name, result)
        
        return result
    
    async def execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolOutput]:
        """
        Execute multiple tools in parallel
        
        Args:
            tool_calls: List of dicts with 'tool_name' and 'params'
            
        Returns:
            List of ToolOutputs in same order as input
            
        Example:
            results = await registry.execute_tools_parallel([
                {"tool_name": "getStockPrice", "params": {"symbol": "AAPL"}},
                {"tool_name": "getTechnicalIndicators", "params": {"symbol": "AAPL"}}
            ])
        """
        tools = []
        params_list = []
        
        for call in tool_calls:
            tool_name = call.get("tool_name")
            tool = self.get_tool(tool_name)
            
            if not tool:
                # Return error for missing tool
                self.logger.error(f"Tool not found: {tool_name}")
                tools.append(None)
            else:
                tools.append(tool)
            
            params_list.append(call.get("params", {}))
        
        # Execute in parallel (handle None tools)
        results = []
        for tool, params in zip(tools, params_list):
            if tool is None:
                results.append(ToolOutput(
                    tool_name=params.get("tool_name", "unknown"),
                    status="error",
                    error="Tool not found"
                ))
            else:
                result = await tool.safe_execute(**params)
                self._update_stats(tool.schema.name, result)
                results.append(result)
        
        return results
    
    def _update_stats(self, tool_name: str, result: ToolOutput) -> None:
        """Update execution statistics"""
        if tool_name not in self._execution_stats:
            return
        
        stats = self._execution_stats[tool_name]
        stats["total_calls"] += 1
        
        if result.status == "success":
            stats["successful_calls"] += 1
        else:
            stats["failed_calls"] += 1
        
        if result.execution_time_ms:
            stats["total_execution_time_ms"] += result.execution_time_ms
            stats["avg_execution_time_ms"] = (
                stats["total_execution_time_ms"] / stats["total_calls"]
            )
    
    def get_execution_stats(self, tool_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution statistics
        
        Args:
            tool_name: Specific tool name or None for all tools
            
        Returns:
            Execution statistics
        """
        if tool_name:
            return self._execution_stats.get(tool_name, {})
        
        return self._execution_stats.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get registry summary
        
        Returns:
            Summary with tool counts, categories, etc.
        """
        return {
            "total_tools": len(self._tools),
            "categories": {
                cat: len(tools)
                for cat, tools in self._categories.items()
            },
            "tools_by_category": self._categories.copy(),
            "execution_stats": {
                "total_calls": sum(
                    stats["total_calls"]
                    for stats in self._execution_stats.values()
                ),
                "total_successful": sum(
                    stats["successful_calls"]
                    for stats in self._execution_stats.values()
                ),
                "total_failed": sum(
                    stats["failed_calls"]
                    for stats in self._execution_stats.values()
                )
            }
        }
    
    def reset_stats(self) -> None:
        """Reset all execution statistics"""
        for stats in self._execution_stats.values():
            stats["total_calls"] = 0
            stats["successful_calls"] = 0
            stats["failed_calls"] = 0
            stats["total_execution_time_ms"] = 0
            stats["avg_execution_time_ms"] = 0
        
        self.logger.info("Execution statistics reset")
    
    def __repr__(self) -> str:
        return (
            f"ToolRegistry("
            f"tools={len(self._tools)}, "
            f"categories={list(self._categories.keys())}"
            f")"
        )
    

    def get_tools_by_category(self, category: str) -> Dict[str, BaseTool]:
        """
        Get all tools in a specific category
        
        Args:
            category: Category name (price, technical, risk, fundamentals, news, market, crypto)
            
        Returns:
            Dict of {tool_name: tool_instance}
        """
        return {
            name: tool 
            for name, tool in self._tools.items() 
            if tool.schema.category == category
        }
    
    def get_all_categories(self) -> Dict[str, int]:
        """
        Get all available categories with tool counts
        
        Returns:
            Dict of {category_name: tool_count}
        """
        categories = {}
        for tool in self._tools.values():
            cat = tool.schema.category
            categories[cat] = categories.get(cat, 0) + 1
        return categories
    
    def get_tool_index(self) -> Dict[str, Any]:
        """
        Get tool index (like /tools/index.json in Claude Agent SDK)
        
        Returns structured index for LLM navigation
        
        Returns:
            Dict with categories and tool lists
        """
        index = {
            "total_tools": len(self._tools),
            "categories": {},
            "tools_by_category": {}
        }
        
        # Build category index
        for tool_name, tool in self._tools.items():
            category = tool.schema.category
            
            if category not in index["categories"]:
                index["categories"][category] = []
            
            index["categories"][category].append(tool_name)
        
        # Add metadata
        index["tools_by_category"] = {
            cat: len(tools) 
            for cat, tools in index["categories"].items()
        }
        
        return index
    
    def get_tool_catalog(
        self, 
        categories: Optional[List[str]] = None,
        detail_level: str = "description"  # "name" | "description" | "full"
    ) -> List[Dict[str, Any]]:
        """
        Get tool catalog for LLM with progressive disclosure
        
        Args:
            categories: Optional list of categories to filter
            detail_level: Level of detail to return
                - "name": Just tool name and category
                - "description": Name, category, description, usage hints
                - "full": Complete tool schema
        
        Returns:
            List of tool information dicts
        """
        tools_to_return = []
        
        # Filter by categories if specified
        if categories:
            filtered_tools = {}
            for cat in categories:
                filtered_tools.update(self.get_tools_by_category(cat))
        else:
            filtered_tools = self._tools
        
        # Build catalog based on detail level
        for tool_name, tool in filtered_tools.items():
            schema = tool.schema
            
            if detail_level == "name":
                tools_to_return.append({
                    "name": tool_name,
                    "category": schema.category
                })
            
            elif detail_level == "description":
                tools_to_return.append({
                    "name": tool_name,
                    "category": schema.category,
                    "description": schema.description,
                    "usage_hints": schema.usage_hints[:3] if schema.usage_hints else [],
                    "requires_symbol": schema.requires_symbol
                })
            
            else:  # full
                tools_to_return.append(schema.to_dict())
        
        return tools_to_return
    
    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """
        Get a specific tool by name
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)


# ============================================================================
# Global Registry Instance
# ============================================================================

_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """
    Get global registry instance (singleton)
    
    Usage:
        from src.tools.registry import get_registry
        
        registry = get_registry()
        result = await registry.execute_tool("getStockPrice", {"symbol": "AAPL"})
    """
    global _global_registry
    
    if _global_registry is None:
        _global_registry = ToolRegistry()
    
    return _global_registry


def initialize_registry() -> ToolRegistry:
    """
    Initialize and populate registry with all available tools
    
    This should be called once at application startup
    
    Returns:
        ToolRegistry with all tools registered
        
    Usage:
        # In main.py or app startup
        from src.agents.tools.registry import initialize_registry
        
        registry = initialize_registry()
        logger.info(f"Initialized {len(registry.get_all_tools())} tools")
    """
    import logging
    
    logger = logging.getLogger(__name__)
    registry = get_registry()
    
    # Track registration stats
    registered_count = 0
    failed_imports = []
    
    # ========================================================================
    # BATCH 1: Core Tools (Already Implemented)
    # ========================================================================
    try:
        from src.agents.tools.price.get_stock_price import GetStockPriceTool
        registry.register_tool(GetStockPriceTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: GetStockPriceTool")
    except ImportError as e:
        failed_imports.append(f"GetStockPriceTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import GetStockPriceTool: {e}")
    
    try:
        from src.agents.tools.technical.get_technical_indicators import GetTechnicalIndicatorsTool
        registry.register_tool(GetTechnicalIndicatorsTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: GetTechnicalIndicatorsTool")
    except ImportError as e:
        failed_imports.append(f"GetTechnicalIndicatorsTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import GetTechnicalIndicatorsTool: {e}")
    
    try:
        from src.agents.tools.risk.assess_risk import AssessRiskTool
        registry.register_tool(AssessRiskTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: AssessRiskTool")
    except ImportError as e:
        failed_imports.append(f"AssessRiskTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import AssessRiskTool: {e}")
    
    # ========================================================================
    # BATCH 2: New Tools (Just Implemented - Wrap Existing Handlers)
    # ========================================================================
    
    # Volume Profile
    try:
        from src.agents.tools.risk.get_volume_profile import GetVolumeProfileTool
        registry.register_tool(GetVolumeProfileTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: GetVolumeProfileTool")
    except ImportError as e:
        failed_imports.append(f"GetVolumeProfileTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import GetVolumeProfileTool: {e}")
    
    # Chart Patterns
    try:
        from src.agents.tools.technical.detect_chart_patterns import DetectChartPatternsTool
        registry.register_tool(DetectChartPatternsTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: DetectChartPatternsTool")
    except ImportError as e:
        failed_imports.append(f"DetectChartPatternsTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import DetectChartPatternsTool: {e}")
    
    # Relative Strength
    try:
        from src.agents.tools.technical.get_relative_strength import GetRelativeStrengthTool
        registry.register_tool(GetRelativeStrengthTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: GetRelativeStrengthTool")
    except ImportError as e:
        failed_imports.append(f"GetRelativeStrengthTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import GetRelativeStrengthTool: {e}")
    
    # Sentiment Analysis
    try:
        from src.agents.tools.risk.get_sentiment import GetSentimentTool
        registry.register_tool(GetSentimentTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: GetSentimentTool")
    except ImportError as e:
        failed_imports.append(f"GetSentimentTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import GetSentimentTool: {e}")
    
    # Stop Loss Suggestions
    try:
        from src.agents.tools.risk.suggest_stop_loss import SuggestStopLossTool
        registry.register_tool(SuggestStopLossTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: SuggestStopLossTool")
    except ImportError as e:
        failed_imports.append(f"SuggestStopLossTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import SuggestStopLossTool: {e}")

    
    # ========================================================================
    # BATCH 2: Price & Performance
    # ========================================================================

    # Stock Performance
    try:
        from src.agents.tools.price.get_stock_performance import GetStockPerformanceTool
        registry.register_tool(GetStockPerformanceTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: GetStockPerformanceTool")
    except ImportError as e:
        failed_imports.append(f"GetStockPerformanceTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import GetStockPerformanceTool: {e}")

    # Price Targets
    try:
        from src.agents.tools.price.get_price_targets import GetPriceTargetsTool
        registry.register_tool(GetPriceTargetsTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: GetPriceTargetsTool")
    except ImportError as e:
        failed_imports.append(f"GetPriceTargetsTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import GetPriceTargetsTool: {e}")

    # ========================================================================
    # BATCH 3: Technical Analysis Advanced
    # ========================================================================

    # Support/Resistance
    try:
        from src.agents.tools.technical.get_support_resistance import GetSupportResistanceTool
        registry.register_tool(GetSupportResistanceTool())
        registered_count += 1
        logger.info("[REGISTRY] ✅ Registered: GetSupportResistanceTool")
    except ImportError as e:
        failed_imports.append(f"GetSupportResistanceTool: {e}")
        logger.warning(f"[REGISTRY] ⚠️ Could not import GetSupportResistanceTool: {e}")


    # ═══════════════════════════════════════════════════════════════
    # BATCH 4: FUNDAMENTALS (5 tools)
    # ═══════════════════════════════════════════════════════════════
    from src.agents.tools.fundamentals import (
        GetIncomeStatementTool,
        GetBalanceSheetTool,
        GetCashFlowTool,
        GetFinancialRatiosTool,
        GetGrowthMetricsTool
    )
    
    # Initialize with API key
    import os
    fmp_api_key = os.environ.get("FMP_API_KEY")
    
    try:
        registry.register_tool(GetIncomeStatementTool(api_key=fmp_api_key))
        registered_count += 1
        registry.register_tool(GetBalanceSheetTool(api_key=fmp_api_key))
        registered_count += 1
        registry.register_tool(GetCashFlowTool(api_key=fmp_api_key))
        registered_count += 1
        registry.register_tool(GetFinancialRatiosTool(api_key=fmp_api_key))
        registered_count += 1
        registry.register_tool(GetGrowthMetricsTool(api_key=fmp_api_key))
        registered_count += 1
    except Exception as e:
        logger.error(f"❌ Failed to register Fundamentals tools: {e}")


    try:
        from src.agents.tools.news import (
            GetStockNewsTool,
            GetEarningsCalendarTool,
            GetCompanyEventsTool
        )
        
        # Require FMP API key for news tools
        if not fmp_api_key:
            logger.warning("⚠️  Skipping News tools - FMP_API_KEY required")
        else:
            registry.register_tool(GetStockNewsTool(api_key=fmp_api_key))
            registry.register_tool(GetEarningsCalendarTool(api_key=fmp_api_key))
            registry.register_tool(GetCompanyEventsTool(api_key=fmp_api_key))
            
            logger.info("✅ News & Events tools registered (3 tools)")
    
    except ImportError as e:
        logger.error(f"❌ Failed to register News & Events tools: {e}")
    except Exception as e:
        logger.error(f"❌ Error initializing News tools: {e}")

    # ========================================================================
    # BATCH 6: MARKET OVERVIEW TOOLS (5 tools)
    # ========================================================================
    try:
        from src.agents.tools.market import (
            GetMarketIndicesTool,
            GetSectorPerformanceTool,
            GetMarketMoversTool,
            GetMarketBreadthTool,
            GetStockHeatmapTool,
            GetMarketNewsTool
        )
        
        if not fmp_api_key:
            logger.warning("⚠️  Skipping Market tools - FMP_API_KEY required")
        else:
            registry.register_tool(GetMarketIndicesTool(api_key=fmp_api_key))
            registry.register_tool(GetSectorPerformanceTool(api_key=fmp_api_key))
            registry.register_tool(GetMarketMoversTool(api_key=fmp_api_key))
            registry.register_tool(GetMarketBreadthTool(api_key=fmp_api_key))
            registry.register_tool(GetStockHeatmapTool(api_key=fmp_api_key))
            registry.register_tool(GetMarketNewsTool(api_key=fmp_api_key))
            
            logger.info("✅ Market Overview tools registered (6 tools)")

    except ImportError as e:
        logger.error(f"❌ Failed to register Market tools: {e}")
    except Exception as e:
        logger.error(f"❌ Error initializing Market tools: {e}")
            
    # ========================================================================
    # BATCH 7: CRYPTO TOOLS (2 tools)
    # ========================================================================
    try:
        from src.agents.tools.crypto import (
            GetCryptoPriceTool,
            GetCryptoTechnicalsTool
        )
        
        if not fmp_api_key:
            logger.warning("⚠️  Skipping Crypto tools - FMP_API_KEY required")
        else:
            registry.register_tool(GetCryptoPriceTool(api_key=fmp_api_key))
            registry.register_tool(GetCryptoTechnicalsTool(api_key=fmp_api_key))
            
            logger.info("✅ Crypto tools registered (2 tools)")

    except ImportError as e:
        logger.error(f"❌ Failed to register Crypto tools: {e}")
    except Exception as e:
        logger.error(f"❌ Error initializing Crypto tools: {e}")


    # ========================================================================
    # BATCH 8: DISCOVERY TOOLS (Stock Screening)
    # ========================================================================
    try:
        from src.agents.tools.discovery import StockScreenerTool
        
        if not fmp_api_key:
            logger.warning("⚠️  Skipping Discovery tools - FMP_API_KEY required")
        else:
            registry.register_tool(StockScreenerTool())
            registered_count += 1
            logger.info("✅ Discovery tools registered (1 tool)")

    except ImportError as e:
        failed_imports.append(f"StockScreenerTool: {e}")
        logger.error(f"❌ Failed to register Discovery tools: {e}")
    except Exception as e:
        logger.error(f"❌ Error initializing Discovery tools: {e}")
    
    # ========================================================================
    # Registration Summary
    # ========================================================================

    summary = registry.get_summary()

    logger.info("=" * 70)
    logger.info(f"[REGISTRY] Registry initialized successfully")
    # logger.info(f"[REGISTRY] Total tools registered: {registered_count}")
    logger.info(f"[REGISTRY] Total tools registered: {summary['total_tools']}")
    logger.info(f"[REGISTRY] Tools by category:")
    
    # Count by category
    category_counts = {}
    for tool in registry.get_all_tools().values():
        category = tool.get_schema().category
        category_counts[category] = category_counts.get(category, 0) + 1

    for category, count in sorted(category_counts.items()):
        logger.info(f"[REGISTRY]   - {category}: {count} tools")
    
    if failed_imports:
        logger.warning(f"[REGISTRY] Failed to import {len(failed_imports)} tools:")
        for failure in failed_imports:
            logger.warning(f"[REGISTRY]   - {failure}")
    
    logger.info("=" * 70)
    
    return registry
    

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    # Test registry
    async def test_registry():
        registry = ToolRegistry()
        
        # Create mock tool
        from src.agents.tools.base import BaseTool, ToolSchema, ToolParameter, create_success_output
        
        class MockTool(BaseTool):
            def __init__(self):
                super().__init__()
                self.schema = ToolSchema(
                    name="mockTool",
                    category="test",
                    description="Mock tool for testing",
                    parameters=[
                        ToolParameter(
                            name="symbol",
                            type="string",
                            required=True
                        )
                    ],
                    returns={"message": "string"}
                )
            
            async def execute(self, symbol: str) -> ToolOutput:
                return create_success_output(
                    tool_name=self.schema.name,
                    data={"message": f"Processed {symbol}"}
                )
        
        # Register
        registry.register_tool(MockTool())
        
        # Test
        print("Registry Summary:")
        print(json.dumps(registry.get_summary(), indent=2))
        
        print("\nExecuting tool...")
        result = await registry.execute_tool(
            tool_name="mockTool",
            params={"symbol": "AAPL"}
        )
        print(json.dumps(result.model_dump(), indent=2))
        
        print("\nExecution Stats:")
        print(json.dumps(registry.get_execution_stats(), indent=2))
    
    asyncio.run(test_registry())