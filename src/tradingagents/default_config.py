import os

DEFAULT_CONFIG = {
    # API Keys from environment
    "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
    "fmp_api_key": os.getenv("FMP_API_KEY", ""),
    
    # LLM settings
    "llm_provider": "openai",
    "backend_url": os.getenv("LLM_BACKEND_URL", "https://api.openai.com/v1"),
    "deep_think_llm": "gpt-4o-mini",
    "quick_think_llm": "gpt-4o-mini", 
    
    # Agent settings
    "max_debate_rounds": 1,
    "online_tools": False,
    
    # Paths
    "project_dir": os.path.join(os.getcwd(), "src/tradingagents"),
    "cache_dir": os.path.join(os.getcwd(), "src/tradingagents/dataflows/data_cache"),
    "data_dir": os.path.join(os.getcwd(), "src", "tradingagents", "dataflows", "data"),
    
    # FMP settings
    "fmp_base_url": "https://financialmodelingprep.com/api",
}