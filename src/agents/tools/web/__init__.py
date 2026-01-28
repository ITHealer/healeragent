from src.agents.tools.web.web_search import WebSearchTool
from src.agents.tools.web.serp_search import SerpSearchTool
from src.agents.tools.web.smart_fetch import SmartFetchTool
from src.agents.tools.web.enhanced_web_search import EnhancedWebSearchTool

__all__ = [
    "WebSearchTool",         # PRIMARY: OpenAI + FALLBACK: Tavily (merged)
    "SerpSearchTool",        # Alternative: SerpAPI/Google
    "SmartFetchTool",        # Direct URL fetch with timestamp validation
    "EnhancedWebSearchTool", # Search + Fetch + Validate (for real-time data)
]