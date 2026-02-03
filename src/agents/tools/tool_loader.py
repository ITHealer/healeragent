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
    # FUNDAMENTALS TOOLS (6) - Require API key
    # ========================================================================
    "fundamentals": [
        ("src.agents.tools.fundamentals", "GetIncomeStatementTool", "api_key"),
        ("src.agents.tools.fundamentals", "GetBalanceSheetTool", "api_key"),
        ("src.agents.tools.fundamentals", "GetCashFlowTool", "api_key"),
        ("src.agents.tools.fundamentals", "GetFinancialRatiosTool", "api_key"),
        ("src.agents.tools.fundamentals", "GetGrowthMetricsTool", "api_key"),
        ("src.agents.tools.fundamentals", "GetAnalystRatingsTool", "api_key"),
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
    # MARKET TOOLS (10) - Require API key
    # ========================================================================
    "market": [
        ("src.agents.tools.market", "GetMarketIndicesTool", "api_key"),
        ("src.agents.tools.market", "GetSectorPerformanceTool", "api_key"),
        ("src.agents.tools.market", "GetMarketMoversTool", "api_key"),
        ("src.agents.tools.market", "GetMarketBreadthTool", "api_key"),
        ("src.agents.tools.market", "GetStockHeatmapTool", "api_key"),
        ("src.agents.tools.market", "GetMarketNewsTool", "api_key"),
        ("src.agents.tools.market", "GetEconomicDataTool", "api_key"),  # NEW: Macro data
        ("src.agents.tools.market", "GetTopGainersTool", "api_key"),
        ("src.agents.tools.market", "GetTopLosersTool", "api_key"),
        ("src.agents.tools.market", "GetMostActivesTool", "api_key"),
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
        ("src.agents.tools.memory.search_conversation_history", "SearchConversationHistoryTool", None),
        ("src.agents.tools.memory.get_recent_conversations", "GetRecentConversationsTool", None),
        ("src.agents.tools.memory.memory_user_edits", "MemoryUserEditsTool", None),
        ("src.agents.tools.memory.search_recall_memory", "SearchRecallMemoryTool", None),
        ("src.agents.tools.memory.search_archival_memory", "SearchArchivalMemoryTool", None),
    ],
        
    # ========================================================================
    # REASONING TOOLS (1) - Think Tool
    # ========================================================================
    "reasoning": [
        ("src.agents.tools.reasoning.think_tool", "ThinkTool", None),
    ],

    # ========================================================================
    # WEB TOOLS (1) - Web Search
    # - WebSearchTool: PRIMARY OpenAI + FALLBACK Tavily (merged)
    # ========================================================================
    "web": [
        ("src.agents.tools.web", "WebSearchTool", "tavily_api_key"),  # OpenAI primary, Tavily fallback
    ],

    # ========================================================================
    # FINANCE GURU TOOLS - Quantitative Analysis
    # These are COMPUTATION tools that work with data from existing tools.
    # See: docs/ARCHITECTURE_CHAT_V2.md for integration details.
    #
    # Phase 1: Valuation (calculateDCF, calculateGraham, calculateDDM) ✅ IMPLEMENTED
    # Phase 2: Enhanced Technical (Ichimoku, Fibonacci, etc.) ✅ IMPLEMENTED
    # Phase 3: Enhanced Risk (VaR, Sharpe, Sortino, etc.) ✅ IMPLEMENTED
    # Phase 4: Portfolio (analyzePortfolio, calculateCorrelation) ✅ IMPLEMENTED
    # Phase 5: Backtest (runBacktest, compareStrategies) ✅ IMPLEMENTED
    # ========================================================================
    "finance_guru": [
        # Phase 1: Valuation Tools
        ("src.agents.tools.finance_guru.tools.valuation", "CalculateDCFTool", None),
        ("src.agents.tools.finance_guru.tools.valuation", "CalculateGrahamTool", None),
        ("src.agents.tools.finance_guru.tools.valuation", "CalculateDDMTool", None),
        ("src.agents.tools.finance_guru.tools.valuation", "GetValuationSummaryTool", None),
        ("src.agents.tools.finance_guru.tools.valuation", "CalculateComparablesTool", None),
        # Phase 2: Enhanced Technical Indicators
        ("src.agents.tools.finance_guru.tools.technical_enhanced", "GetIchimokuCloudTool", None),
        ("src.agents.tools.finance_guru.tools.technical_enhanced", "GetFibonacciLevelsTool", None),
        ("src.agents.tools.finance_guru.tools.technical_enhanced", "GetWilliamsRTool", None),
        ("src.agents.tools.finance_guru.tools.technical_enhanced", "GetCCITool", None),
        ("src.agents.tools.finance_guru.tools.technical_enhanced", "GetParabolicSARTool", None),
        ("src.agents.tools.finance_guru.tools.technical_enhanced", "GetEnhancedTechnicalsTool", None),
        # Phase 3: Enhanced Risk Metrics
        ("src.agents.tools.finance_guru.tools.risk_metrics", "GetRiskMetricsTool", None),
        ("src.agents.tools.finance_guru.tools.risk_metrics", "GetVaRTool", None),
        ("src.agents.tools.finance_guru.tools.risk_metrics", "GetSharpeRatioTool", None),
        ("src.agents.tools.finance_guru.tools.risk_metrics", "GetMaxDrawdownTool", None),
        ("src.agents.tools.finance_guru.tools.risk_metrics", "GetBetaAlphaTool", None),
        # Phase 4: Portfolio Analysis
        ("src.agents.tools.finance_guru.tools.portfolio", "OptimizePortfolioTool", None),
        ("src.agents.tools.finance_guru.tools.portfolio", "GetCorrelationMatrixTool", None),
        ("src.agents.tools.finance_guru.tools.portfolio", "GetEfficientFrontierTool", None),
        ("src.agents.tools.finance_guru.tools.portfolio", "AnalyzePortfolioDiversificationTool", None),
        ("src.agents.tools.finance_guru.tools.portfolio", "SuggestRebalancingTool", None),
        # Phase 5: Backtest Engine
        ("src.agents.tools.finance_guru.tools.backtest", "RunBacktestTool", None),
        ("src.agents.tools.finance_guru.tools.backtest", "CompareStrategiesTool", None),
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
    api_key: Optional[str],
    tavily_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
):
    """Create tool instance with optional init argument"""
    tool_class = _import_tool_class(module_path, class_name)

    if init_arg == "api_key" and api_key:
        return tool_class(api_key=api_key)
    elif init_arg == "tavily_api_key":
        return tool_class(api_key=tavily_api_key)
    elif init_arg == "openai_api_key":
        return tool_class(api_key=openai_api_key)
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
    
    # Get API keys
    fmp_api_key = os.environ.get("FMP_API_KEY")
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not fmp_api_key:
        logger.warning("FMP_API_KEY not set - some tools may not work")

    if not openai_api_key and not tavily_api_key:
        logger.warning("Neither OPENAI_API_KEY nor TAVILY_API_KEY set - web search disabled")

    # Register tools by category
    for category, tools in TOOL_DEFINITIONS.items():
        category_count = 0

        for module_path, class_name, init_arg in tools:
            try:
                tool = _create_tool_instance(
                    module_path=module_path,
                    class_name=class_name,
                    init_arg=init_arg,
                    api_key=fmp_api_key,
                    tavily_api_key=tavily_api_key,
                    openai_api_key=openai_api_key,
                )
                
                registry.register_tool(tool)
                registered_count += 1
                category_count += 1
                
            except ImportError as e:
                error_msg = f"{class_name}: {e}"
                failed_imports.append(error_msg)
                logger.warning(f"Could not import {class_name}: {e}")
                
            except Exception as e:
                import traceback
                error_msg = f"{class_name}: {e}"
                failed_imports.append(error_msg)
                logger.error(f"Error creating {class_name}: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # if category_count > 0:
        #     logger.info(f"{category}: {category_count} tools registered")
    
    # Summary
    summary = registry.get_summary()
    
    # logger.info("=" * 70)
    # logger.info("[TOOL LOADER] Registration complete")
    # logger.info(f"[TOOL LOADER] Total tools: {summary['total_tools']}")
    # logger.info(f"[TOOL LOADER] Categories: {list(summary['categories'].keys())}")
    
    # for cat, count in sorted(summary['categories'].items()):
    #     logger.info(f"[TOOL LOADER]   - {cat}: {count} tools")
    
    # if failed_imports:
    #     logger.warning(f"[TOOL LOADER] Failed: {len(failed_imports)} tools")
    #     for failure in failed_imports[:5]:
    #         logger.warning(f"[TOOL LOADER]   - {failure}")
    
    # logger.info("=" * 70)
    
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