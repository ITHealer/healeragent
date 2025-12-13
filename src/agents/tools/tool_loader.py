# File: src/agents/tools/tool_loader.py
"""
Tool Loader - Centralized Tool Registration

Loads and registers all 31 tools across 9 categories:
- price: 3 tools
- technical: 4 tools
- risk: 4 tools
- fundamentals: 5 tools
- news: 4 tools
- market: 5 tools
- crypto: 2 tools
- discovery: 1 tool
- memory: 3 tools
- reasoning: 1 tool (Think Tool - optional)

Usage:
    from src.agents.tools.tool_loader import load_all_tools, get_registry
    
    registry, failed = load_all_tools()
"""

import os
import logging
from typing import List, Tuple, Optional

from src.agents.tools.base import BaseTool


logger = logging.getLogger(__name__)


# ============================================================================
# TOOL DEFINITIONS BY CATEGORY
# ============================================================================

TOOL_DEFINITIONS = {
    # ========================================================================
    # PRICE TOOLS (3)
    # ========================================================================
    "price": [
        ("src.agents.tools.price.get_stock_price", "GetStockPriceTool", None),
        ("src.agents.tools.price.get_stock_performance", "GetStockPerformanceTool", None),
        ("src.agents.tools.price.get_price_targets", "GetPriceTargetsTool", None),
    ],
    
    # ========================================================================
    # TECHNICAL TOOLS (4)
    # ========================================================================
    "technical": [
        ("src.agents.tools.technical.get_technical_indicators", "GetTechnicalIndicatorsTool", None),
        ("src.agents.tools.technical.detect_chart_patterns", "DetectChartPatternsTool", None),
        ("src.agents.tools.technical.get_relative_strength", "GetRelativeStrengthTool", None),
        ("src.agents.tools.technical.get_support_resistance", "GetSupportResistanceTool", None),
    ],
    
    # ========================================================================
    # RISK TOOLS (4)
    # ========================================================================
    "risk": [
        ("src.agents.tools.risk.assess_risk", "AssessRiskTool", None),
        ("src.agents.tools.risk.get_volume_profile", "GetVolumeProfileTool", None),
        ("src.agents.tools.risk.get_sentiment", "GetSentimentTool", None),
        ("src.agents.tools.risk.suggest_stop_loss", "SuggestStopLossTool", None),
    ],
    
    # ========================================================================
    # FUNDAMENTALS TOOLS (5) - Require API key
    # ========================================================================
    "fundamentals": [
        ("src.agents.tools.fundamentals", "GetIncomeStatementTool", "api_key"),
        ("src.agents.tools.fundamentals", "GetBalanceSheetTool", "api_key"),
        ("src.agents.tools.fundamentals", "GetCashFlowTool", "api_key"),
        ("src.agents.tools.fundamentals", "GetFinancialRatiosTool", "api_key"),
        ("src.agents.tools.fundamentals", "GetGrowthMetricsTool", "api_key"),
    ],
    
    # ========================================================================
    # NEWS TOOLS (4) - Require API key
    # ========================================================================
    "news": [
        ("src.agents.tools.news", "GetStockNewsTool", "api_key"),
        ("src.agents.tools.news", "GetEarningsCalendarTool", "api_key"),
        ("src.agents.tools.news", "GetCompanyEventsTool", "api_key"),
    ],
    
    # ========================================================================
    # MARKET TOOLS (5) - Require API key
    # ========================================================================
    "market": [
        ("src.agents.tools.market", "GetMarketIndicesTool", "api_key"),
        ("src.agents.tools.market", "GetSectorPerformanceTool", "api_key"),
        ("src.agents.tools.market", "GetMarketMoversTool", "api_key"),
        ("src.agents.tools.market", "GetMarketBreadthTool", "api_key"),
        ("src.agents.tools.market", "GetStockHeatmapTool", "api_key"),
        ("src.agents.tools.market", "GetMarketNewsTool", "api_key"),
    ],
    
    # ========================================================================
    # CRYPTO TOOLS (2) - Require API key
    # ========================================================================
    "crypto": [
        ("src.agents.tools.crypto", "GetCryptoPriceTool", "api_key"),
        ("src.agents.tools.crypto", "GetCryptoTechnicalsTool", "api_key"),
    ],
    
    # ========================================================================
    # DISCOVERY TOOLS (1)
    # ========================================================================
    "discovery": [
        ("src.agents.tools.discovery", "StockScreenerTool", None),
    ],
    
    # ========================================================================
    # MEMORY TOOLS (3)
    # ========================================================================
    "memory": [
        ("src.agents.tools.memory.search_recall_memory", "SearchRecallMemoryTool", None),
        ("src.agents.tools.memory.search_archival_memory", "SearchArchivalMemoryTool", None),
        ("src.agents.tools.memory.search_procedural_memory", "SearchProceduralMemoryTool", None),
    ],
    
    # ========================================================================
    # REASONING TOOLS (1) - Think Tool
    # ========================================================================
    "reasoning": [
        ("src.agents.tools.reasoning.think_tool", "ThinkTool", None),
    ],
}


def _import_tool_class(module_path: str, class_name: str):
    """Dynamically import a tool class"""
    import importlib
    
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _create_tool_instance(
    module_path: str,
    class_name: str,
    init_arg: Optional[str],
    api_key: Optional[str]
):
    """Create tool instance with optional init argument"""
    tool_class = _import_tool_class(module_path, class_name)
    
    if init_arg == "api_key" and api_key:
        return tool_class(api_key=api_key)
    else:
        return tool_class()


def load_all_tools() -> Tuple["ToolRegistry", List[str]]:
    """
    Load and register all available tools
    
    Returns:
        Tuple[ToolRegistry, List[str]]: (registry, failed_imports)
    """
    # Import here to avoid circular imports
    from src.agents.tools.registry import ToolRegistry
    
    registry = ToolRegistry()
    failed_imports = []
    registered_count = 0
    
    # Get FMP API key
    fmp_api_key = os.environ.get("FMP_API_KEY")
    
    if not fmp_api_key:
        logger.warning("⚠️ FMP_API_KEY not set - some tools may not work")
    
    logger.info("=" * 70)
    logger.info("[TOOL LOADER] Starting tool registration...")
    logger.info("=" * 70)
    
    # Register tools by category
    for category, tools in TOOL_DEFINITIONS.items():
        category_count = 0
        
        for module_path, class_name, init_arg in tools:
            try:
                tool = _create_tool_instance(
                    module_path=module_path,
                    class_name=class_name,
                    init_arg=init_arg,
                    api_key=fmp_api_key
                )
                
                registry.register_tool(tool)
                registered_count += 1
                category_count += 1
                
            except ImportError as e:
                error_msg = f"{class_name}: {e}"
                failed_imports.append(error_msg)
                logger.warning(f"⚠️ Could not import {class_name}: {e}")
                
            except Exception as e:
                error_msg = f"{class_name}: {e}"
                failed_imports.append(error_msg)
                logger.error(f"❌ Error creating {class_name}: {e}")
        
        if category_count > 0:
            logger.info(f"✅ {category}: {category_count} tools registered")
    
    # Summary
    summary = registry.get_summary()
    
    logger.info("=" * 70)
    logger.info("[TOOL LOADER] Registration complete")
    logger.info(f"[TOOL LOADER] Total tools: {summary['total_tools']}")
    logger.info(f"[TOOL LOADER] Categories: {list(summary['categories'].keys())}")
    
    for cat, count in sorted(summary['categories'].items()):
        logger.info(f"[TOOL LOADER]   - {cat}: {count} tools")
    
    if failed_imports:
        logger.warning(f"[TOOL LOADER] Failed: {len(failed_imports)} tools")
        for failure in failed_imports[:5]:
            logger.warning(f"[TOOL LOADER]   - {failure}")
    
    logger.info("=" * 70)
    
    return registry, failed_imports


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_registry_instance = None


def get_registry() -> "ToolRegistry":
    """
    Get singleton registry instance
    
    Loads all tools on first call, returns cached instance afterward.
    """
    global _registry_instance
    
    if _registry_instance is None:
        _registry_instance, _ = load_all_tools()
    
    return _registry_instance


def reload_registry() -> "ToolRegistry":
    """Force reload of registry"""
    global _registry_instance
    
    # Import here to avoid circular
    from src.agents.tools.registry import ToolRegistry
    
    # Reset singleton
    ToolRegistry._instance = None
    _registry_instance = None
    
    return get_registry()


def get_tool_count() -> int:
    """Get total number of registered tools"""
    registry = get_registry()
    return len(registry.get_all_tools())