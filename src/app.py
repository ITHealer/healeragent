class LoggerInstance(object):
    def __new__(cls):
        from src.utils.logger.custom_logging import LogHandler
        return LogHandler()

class IncludeAPIRouter(object):
    def __new__(cls):
        from fastapi.routing import APIRouter

        # =============================================================================
        # IMPORT ALL ROUTERS
        # =============================================================================
        
        # Core system routers
        from src.routers.health_check import router as router_health_check
        from src.routers.security import router as router_security
        
        # RAG system routers
        from src.routers.documents import router as router_document_management
        from src.routers.vectorstore import router as router_collection_management
        from src.routers.retriever import router as router_retriever
        from src.routers.rerank import router as router_rerank

        # Session and chat service routers
        from src.routers.session_management import router as router_session_management
        from src.routers.llm_chat import router as router_llm_chat

        # Title & question sugesstion
        from src.routers.chat_title import router as router_chat_title
        from src.routers.question_suggestions import router as router_qustion_suggestions
        
        # Financial analysis tool routers
        from src.routers.technical_analysis import router as router_technical_analysis
        from src.routers.relative_strength import router as router_relative_strength
        from src.routers.volume_profile import router as router_volume_profile
        from src.routers.pattern_recognition import router as router_pattern_recognition
        from src.routers.risk_analysis import router as router_risk_analysis
        from src.routers.sentiment_analysis import router as router_sentiment_analysis
        from src.routers.fundamental_analysis import router as router_fundamental_analysis
        from src.routers.news_analysis import router as router_news_analysis 
        from src.routers.market_analysis import router as router_market_analysis
        # from src.routers.options_strategy import router as router_options_strategy
        from src.routers.comprehensive_analysis import router as router_comprehensive_analysis

        # Trading service routers
        from src.routers.trading_agents import router as router_trading_agents
        from src.routers.hedge_fund_multi_agent import router as router_multi_agent
        from src.routers.content_processor import router as router_content_processor
        from src.routers.news_agent import router as router_news_agent
        from src.routers.news_aggregator import router as router_news_aggregator

        # Data provider routers
        from src.routers.equity import router as get_data_provider_fmp

        # =============================================================================
        # API UPGRADE VERSION
        # =============================================================================
        # from src.routers.v2.chat import router as router_chat
        from src.routers.v2.chat_thinking import router as router_chat_thinking
        from src.routers.v2.chat_assistant import router as router_chat_assistant
        from src.routers.v2.tool_call import router as router_tool_call
        from src.routers.v2.cookies_router import router as router_cookies
        from src.routers.v2.text_translator import router as router_text_translator
        from src.routers.v2.deep_research import router as router_deep_research

        # Live Analysis
        from src.routers.v2.equity_forecast import router as router_equity_forecast
        from src.routers.smc_analysis import router as router_smc_analysis
        
        # Data provider routers
        from src.routers.v2.data_providers.fmp.company_search import router as router_company_search
        from src.routers.v2.data_providers.fmp.symbol_directory import router as router_symbol_directory

        # Task system routers (API v1)
        from src.routers.tasks import router as router_tasks

        # =============================================================================
        # CONFIGURE MAIN ROUTER AND INCLUDE ALL SUB-ROUTERS  
        # =============================================================================
        
        # API version 2
        router = APIRouter(prefix='/api/v2')

        # Core system routers
        router.include_router(router_health_check, tags=['Health Check'])
        router.include_router(router_security, tags=['Authentication'])

        # RAG system routers
        router.include_router(router_document_management, tags=['RAG - Document Management'])
        router.include_router(router_collection_management, tags=['RAG - Vector Store'])
        router.include_router(router_retriever, tags=['RAG - Retrieval'])
        router.include_router(router_rerank, tags=['RAG - Reranking'])

        # Session and chat service routers
        router.include_router(router_session_management, tags=['Chat - Session Management'])
        router.include_router(router_llm_chat, tags=['Chat - LLM Conversation'])
        
        # Title & question sugesstion
        router.include_router(router_chat_title, tags=['Chat - Utilities'])
        router.include_router(router_qustion_suggestions, tags=['Chat - Utilities'])

        # Financial analysis tool routers
        router.include_router(router_technical_analysis, tags=['Tool - Technical Indicators Analysis'])
        router.include_router(router_relative_strength, tags=['Tool - Technical Indicators Analysis'])
        router.include_router(router_volume_profile, tags=['Tool - Technical Indicators Analysis'])
        router.include_router(router_pattern_recognition, tags=['Tool - Technical Indicators Analysis'])
        router.include_router(router_risk_analysis, tags=['Tool - Risk Analysis'])
        router.include_router(router_sentiment_analysis, tags=['Tool - Sentiment Analysis'])
        router.include_router(router_fundamental_analysis, tags=['Tool - Fundamental Analysis'])
        router.include_router(router_news_analysis, tags=['Tool - News Analysis'])
        router.include_router(router_market_analysis, tags=['Tool - Market Analysis'])
        # router.include_router(router_options_strategy, tags=['Tool - Strategy Trading'])
        router.include_router(router_comprehensive_analysis, tags=['Tool - Comprehensive Analysis'])

        # Trading service routers
        router.include_router(router_trading_agents, tags=['Tool - Trading Agent'])
        router.include_router(router_multi_agent, tags=['Chat - AI Character Multi-Agent'])
        
        # Data provider routers
        router.include_router(get_data_provider_fmp, tags=["Data Provider - FMP"])


        # =============================================================================
        # API UPGRADE VERSION
        # =============================================================================
        router.include_router(router_tool_call, tags=["TOL Router - Tool Call"])
        # router.include_router(router_chat, tags=["TOL Chat - Dev Chat"])
        router.include_router(router_chat_thinking, tags=["TOL Chat - Assistant"])
        router.include_router(router_chat_assistant, tags=["TOL Chat - Assistant - V2"])
        router.include_router(router_cookies, tags=["TOL Cookies - Cookie Management"])
        router.include_router(router_content_processor, tags=['TOL - Content Summarizer'])
        router.include_router(router_news_agent, tags=['TOL News Aggregator - Tavily News Agent'])
        router.include_router(router_news_aggregator, tags=['TOL News Aggregator - News Aggregator'])
        router.include_router(router_text_translator, tags=['TOL Translator - Text Translation'])
        router.include_router(router_equity_forecast, tags=['TOL Tool - Equity Forecast'])
        router.include_router(router_smc_analysis, tags=['TOL Tool - Crypto Live Analysis'])
        router.include_router(router_deep_research, tags=['Deep Research - Multi-Agent Research'])

        # Data provider routers
        router.include_router(router_company_search, tags=["TOL Data Provider - FMP"])
        router.include_router(router_symbol_directory, tags=["TOL Data Provider - FMP"])

        # =============================================================================
        # API V1 - TASK SYSTEM
        # =============================================================================
        router_v1 = APIRouter(prefix='/api/v1')
        router_v1.include_router(router_tasks, tags=["Task System - News Analysis"])

        # Return combined routers
        main_router = APIRouter()
        main_router.include_router(router)  # v2 endpoints
        main_router.include_router(router_v1)  # v1 endpoints

        return main_router
        

# Instance creation
logger_instance = LoggerInstance()