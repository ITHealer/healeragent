import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import aioredis
from langchain_core.messages import HumanMessage

from src.helpers.news_agent_cache_helper import (
    CACHE_TTL_DAILY_NEWS,
    CACHE_TTL_PERSONALIZED_NEWS,
    CACHE_TTL_SECTOR_ANALYSIS,
    CACHE_TTL_SYMBOL_NEWS,
    generate_daily_news_cache_key,
    generate_personalized_news_cache_key,
    generate_sector_analysis_cache_key,
    generate_symbol_news_cache_key,
    get_news_cache,
    set_news_cache,
)
from src.news_agent.news_agent import (
    NewsIntelligenceState,
    agent_logger,
    default_news_topics,
    error_logger,
    news_intelligence_agent_compiled,
)

import logging
app_logger = logging.getLogger("main_app")


def _build_sector_focused_topics(sectors: List[str], base_topics: Optional[str]) -> str:
    parts = [
        "SECTOR ANALYSIS:",
        f"Analyze trends in: {', '.join(sectors)}",
        "",
        "For each sector, find:",
        "- Industry trends and technological developments",
        "- Leading companies and market leaders performance",
        "- Regulatory changes and policy impacts",
        "- Investment opportunities and emerging risks",
        "- Analyst sector outlooks and forecasts",
        "- M&A activity and corporate developments",
        "",
    ]
    if base_topics:
        parts += ["ADDITIONAL CONTEXT:", base_topics]
    return "\n".join(parts)


async def execute_news_intelligence_standard(
    topics: Optional[str],
    max_searches: int,
    target_language: str = "en",
    include_videos: bool = True,
    include_embedded_links: bool = True,
    max_results_per_search: int = 5,
    processing_method: str = "tavily",
    model: str = "gpt-4.1-nano",
    provider_type: str = "openai",
    redis_client: Optional[aioredis.Redis] = None
) -> Dict[str, Any]:
    """Execute standard news agent."""
    
    cache_key = generate_daily_news_cache_key(
        topics=topics,
        max_searches=max_searches,
        target_language=target_language,
        max_results_per_search=max_results_per_search,
        processing_method=processing_method
    )
    
    cached_result = await get_news_cache(redis_client, cache_key)
    if cached_result:
        app_logger.info(f"[DAILY] Cached result")
        return cached_result
    
    start_time = datetime.now()
    thread_id = f"news_{uuid.uuid4().hex[:8]}"
    
    app_logger.info(f"[DAILY] thread={thread_id}, model={model}, provider={provider_type}")
    
    config = {"configurable": {"thread_id": thread_id}}
    topics_to_use = topics or default_news_topics
    
    query_parts = [
        "Gather the latest, most relevant news for today based on the following topics.",
        f"Target Language for ALL summaries: {target_language}",
        f"IMPORTANT: Use max_results={max_results_per_search} when calling tavily_search."
    ]
    
    if include_videos:
        query_parts.append("Include relevant YouTube videos when available.")
    if include_embedded_links:
        query_parts.append("Extract and process embedded links within articles.")
    
    query_parts.append(f"\n**Topics:**\n---\n{topics_to_use}\n---")
    query_parts.append("\nSearch for RECENT news (last 24-48 hours).")
    
    initial_state = NewsIntelligenceState(
        messages=[HumanMessage(content="\n".join(query_parts))],
        search_count=0,
        max_searches=max_searches,
        processed_urls=[],
        content_processor_results=[],
        target_language=target_language,
        tavily_search_results=[],
        processing_method=processing_method,
        max_results_per_search=max_results_per_search,
        model=model,
        provider_type=provider_type
    )
    
    try:
        final_state = await news_intelligence_agent_compiled.ainvoke(initial_state, config)
        processing_time = (datetime.now() - start_time).total_seconds()
        summaries = final_state.get("final_summaries", [])
        
        processing_stats = {
            "total_articles": len(summaries),
            "content_processor": sum(1 for s in summaries if s.get("processing_method") == "content_processor"),
            "tavily": sum(1 for s in summaries if s.get("processing_method") == "tavily"),
            "videos": sum(1 for s in summaries if s.get("content_type") == "video"),
            "articles": sum(1 for s in summaries if s.get("content_type") == "article"),
            "successful": sum(1 for s in summaries if not s.get("error")),
            "failed": sum(1 for s in summaries if s.get("error"))
        }
        
        result = {
            "status": "success",
            "summaries": summaries,
            "processing_stats": processing_stats,
            "metadata": {
                "thread_id": thread_id,
                "searches_performed": final_state.get("search_count", 0),
                "articles_found": len(summaries),
                "processing_time_seconds": round(processing_time, 2),
                "target_language": target_language,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "processing_method": processing_method,
                "model": model,
                "provider_type": provider_type
            }
        }
        
        if len(summaries) > 0:
            await set_news_cache(redis_client, cache_key, result, CACHE_TTL_DAILY_NEWS)
        
        return result
        
    except Exception as e:
        error_logger.error(f"[DAILY] Error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "summaries": [],
            "processing_stats": {},
            "metadata": {"error": str(e), "thread_id": thread_id}
        }


async def execute_news_intelligence_personalized(
    topics: str,
    max_searches: int,
    target_language: str = "en",
    include_videos: bool = True,
    include_embedded_links: bool = True,
    max_results_per_search: int = 5,
    processing_method: str = "tavily",
    model: str = "gpt-4.1-nano",
    provider_type: str = "openai",
    redis_client: Optional[aioredis.Redis] = None
) -> Dict[str, Any]:
    """Execute personalized news - FE provides full topics."""
    
    cache_key = generate_personalized_news_cache_key(
        user_id=0,  # Not used for caching anymore
        topics=topics,
        additional_topics=None,
        max_searches=max_searches,
        target_language=target_language,
        max_results_per_search=max_results_per_search,
        processing_method=processing_method
    )
    
    cached_result = await get_news_cache(redis_client, cache_key)
    if cached_result:
        app_logger.info(f"[PERSONALIZED] Cached result")
        return cached_result
    
    start_time = datetime.now()
    thread_id = f"news_personalized_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    
    app_logger.info(f"[PERSONALIZED] model={model}, provider={provider_type}")
    
    query_parts = [
        f"Gather personalized news based on user interests.",
        f"Target Language: {target_language}",
        f"Use max_results={max_results_per_search}"
    ]
    
    if include_videos:
        query_parts.append("Include video content.")
    
    query_parts.append(f"\n**Topics:**\n---\n{topics}\n---")
    query_parts.append("\nSearch for RECENT news (24-48 hours).")
    
    initial_state = NewsIntelligenceState(
        messages=[HumanMessage(content="\n".join(query_parts))],
        search_count=0,
        max_searches=max_searches,
        processed_urls=[],
        content_processor_results=[],
        target_language=target_language,
        tavily_search_results=[],
        processing_method=processing_method,
        max_results_per_search=max_results_per_search,
        model=model,
        provider_type=provider_type
    )
    
    try:
        final_state = await news_intelligence_agent_compiled.ainvoke(initial_state, config)
        processing_time = (datetime.now() - start_time).total_seconds()
        summaries = final_state.get("final_summaries", [])
        
        processing_stats = {
            "total_articles": len(summaries),
            "content_processor": sum(1 for s in summaries if s.get("processing_method") == "content_processor"),
            "tavily": sum(1 for s in summaries if s.get("processing_method") == "tavily"),
            "videos": sum(1 for s in summaries if s.get("content_type") == "video"),
            "articles": sum(1 for s in summaries if s.get("content_type") == "article"),
            "successful": sum(1 for s in summaries if not s.get("error")),
            "failed": sum(1 for s in summaries if s.get("error"))
        }
        
        result = {
            "status": "success",
            "summaries": summaries,
            "processing_stats": processing_stats,
            "metadata": {
                "thread_id": thread_id,
                "searches_performed": final_state.get("search_count", 0),
                "processing_time_seconds": round(processing_time, 2),
                "target_language": target_language,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "type": "personalized_news",
                "processing_method": processing_method,
                "model": model,
                "provider_type": provider_type
            }
        }
        
        if len(summaries) > 0:
            await set_news_cache(redis_client, cache_key, result, CACHE_TTL_PERSONALIZED_NEWS)
        
        return result
        
    except Exception as e:
        error_logger.error(f"[PERSONALIZED] Error: {str(e)}", exc_info=True)
        return {"status": "error", "summaries": [], "processing_stats": {}, "metadata": {"error": str(e), "thread_id": thread_id}}


async def execute_symbol_news(
    symbols: List[str],
    topics: Optional[str],
    max_searches: int,
    target_language: str = "en",
    include_videos: bool = True,
    include_embedded_links: bool = True,
    max_results_per_search: int = 10,
    processing_method: str = "tavily",
    model: str = "gpt-4.1-nano",
    provider_type: str = "openai",
    redis_client: Optional[aioredis.Redis] = None
) -> Dict[str, Any]:
    """Execute symbol-specific news - FE provides symbols."""
    
    cache_key = generate_symbol_news_cache_key(
        user_id=0,
        symbols=symbols,
        additional_symbols=None,
        topics=topics,
        max_searches=max_searches,
        target_language=target_language,
        max_results_per_search=max_results_per_search,
        processing_method=processing_method
    )
    
    cached_result = await get_news_cache(redis_client, cache_key)
    if cached_result:
        app_logger.info(f"[SYMBOLS] Cached result")
        return cached_result
    
    start_time = datetime.now()
    thread_id = f"news_symbols_{uuid.uuid4().hex[:8]}"
    
    app_logger.info(f"[SYMBOLS] {len(symbols)} symbols, model={model}, provider={provider_type}")
    
    config = {"configurable": {"thread_id": thread_id}}
    topics_to_use = topics or f"Latest news, earnings, analyst ratings for: {', '.join(symbols)}"
    
    query_parts = [
        f"Track symbol-specific news",
        f"Symbols: {', '.join(symbols)}",
        f"Target Language: {target_language}",
        f"Use max_results={max_results_per_search}"
    ]
    
    if include_videos:
        query_parts.append("Include video content.")
    
    query_parts.append(f"\n**Focus Topics:**\n---\n{topics_to_use}\n---")
    query_parts.append("\nSearch for RECENT symbol news (24-48 hours).")
    
    initial_state = NewsIntelligenceState(
        messages=[HumanMessage(content="\n".join(query_parts))],
        search_count=0,
        max_searches=max_searches,
        processed_urls=[],
        content_processor_results=[],
        target_language=target_language,
        tavily_search_results=[],
        processing_method=processing_method,
        symbols_to_track=symbols,
        max_results_per_search=max_results_per_search,
        model=model,
        provider_type=provider_type
    )
    
    try:
        final_state = await news_intelligence_agent_compiled.ainvoke(initial_state, config)
        processing_time = (datetime.now() - start_time).total_seconds()
        summaries = final_state.get("final_summaries", [])
        
        processing_stats = {
            "total_articles": len(summaries),
            "content_processor": sum(1 for s in summaries if s.get("processing_method") == "content_processor"),
            "tavily": sum(1 for s in summaries if s.get("processing_method") == "tavily"),
            "videos": sum(1 for s in summaries if s.get("content_type") == "video"),
            "articles": sum(1 for s in summaries if s.get("content_type") == "article"),
            "successful": sum(1 for s in summaries if not s.get("error")),
            "failed": sum(1 for s in summaries if s.get("error"))
        }
        
        result = {
            "status": "success",
            "summaries": summaries,
            "processing_stats": processing_stats,
            "metadata": {
                "thread_id": thread_id,
                "symbols": symbols,
                "searches_performed": final_state.get("search_count", 0),
                "processing_time_seconds": round(processing_time, 2),
                "target_language": target_language,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "type": "symbol_specific_news",
                "processing_method": processing_method,
                "model": model,
                "provider_type": provider_type
            }
        }
        
        if len(summaries) > 0:
            await set_news_cache(redis_client, cache_key, result, CACHE_TTL_SYMBOL_NEWS)
        
        return result
        
    except Exception as e:
        error_logger.error(f"[SYMBOLS] Error: {str(e)}", exc_info=True)
        return {"status": "error", "summaries": [], "processing_stats": {}, "metadata": {"error": str(e), "thread_id": thread_id}}


async def execute_sector_analysis(
    sectors: List[str],
    analysis_depth: str,
    topics: Optional[str],
    target_language: str = "en",
    include_videos: bool = True,
    max_results_per_search: int = 5,
    processing_method: str = "tavily",
    model: str = "gpt-4.1-nano",
    provider_type: str = "openai",
    redis_client: Optional[aioredis.Redis] = None
) -> Dict[str, Any]:
    """Execute sector analysis."""
    
    cache_key = generate_sector_analysis_cache_key(
        sectors=sectors,
        analysis_depth=analysis_depth,
        topics=topics,
        target_language=target_language,
        user_id=None,
        max_results_per_search=max_results_per_search,
        processing_method=processing_method
    )
    
    cached_result = await get_news_cache(redis_client, cache_key)
    if cached_result:
        app_logger.info(f"[SECTOR] Cached result")
        return cached_result
    
    start_time = datetime.now()
    thread_id = f"news_sector_{uuid.uuid4().hex[:8]}"
    
    app_logger.info(f"[SECTOR] {len(sectors)} sectors, model={model}, provider={provider_type}")
    
    if not sectors or len(sectors) == 0:
        return {
            "status": "error",
            "summaries": [],
            "processing_stats": {},
            "metadata": {"error": "At least one sector required", "thread_id": thread_id}
        }
    
    depth_mapping = {"quick": 1, "standard": 2, "deep": 3}
    
    if analysis_depth not in depth_mapping:
        analysis_depth = "standard"
    
    max_searches = depth_mapping[analysis_depth]
    config = {"configurable": {"thread_id": thread_id}}
    topics_to_use = _build_sector_focused_topics(sectors, topics)
    
    query_parts = [
        f"Conduct {analysis_depth} analysis of {len(sectors)} sector(s): {', '.join(sectors)}",
        "",
        "ANALYSIS OBJECTIVES:",
        "1. Identify key trends and market movements",
        "2. Find leading companies and competitive landscape",
        "3. Discover investment opportunities and risks",
        "4. Track regulatory changes and policy impacts",
        "5. Analyze sector performance and outlook",
        "",
        f"Target Language: {target_language}",
        f"Search depth: {analysis_depth} ({max_searches} searches)",
        f"Use max_results={max_results_per_search}"
    ]
    
    if include_videos:
        query_parts.append("Include sector analysis videos.")
    
    if analysis_depth == "quick":
        query_parts.append("\nFocus on: Latest headlines and major developments")
    elif analysis_depth == "deep":
        query_parts.append("\nFocus on: In-depth analysis, expert opinions, financial metrics")
    
    query_parts.append(f"\n**SECTOR ANALYSIS TOPICS:**\n---\n{topics_to_use}\n---")
    query_parts.append("\nSearch for RECENT sector news (last 7 days)")
    
    initial_state = NewsIntelligenceState(
        messages=[HumanMessage(content="\n".join(query_parts))],
        search_count=0,
        max_searches=max_searches,
        processed_urls=[],
        content_processor_results=[],
        target_language=target_language,
        tavily_search_results=[],
        processing_method=processing_method,
        sectors_to_analyze=sectors,
        additional_topics=topics or "",
        max_results_per_search=max_results_per_search,
        model=model,
        provider_type=provider_type
    )
    
    try:
        final_state = await news_intelligence_agent_compiled.ainvoke(initial_state, config)
        processing_time = (datetime.now() - start_time).total_seconds()
        summaries = final_state.get("final_summaries", [])
        
        processing_stats = {
            "total_articles": len(summaries),
            "content_processor": sum(1 for s in summaries if s.get("processing_method") == "content_processor"),
            "tavily": sum(1 for s in summaries if s.get("processing_method") == "tavily"),
            "videos": sum(1 for s in summaries if s.get("content_type") == "video"),
            "articles": sum(1 for s in summaries if s.get("content_type") == "article"),
            "successful": sum(1 for s in summaries if not s.get("error")),
            "failed": sum(1 for s in summaries if s.get("error"))
        }
        
        sector_insights = {}
        for sector in sectors:
            sector_summaries = [s for s in summaries if sector.lower() in s.get("title", "").lower() or 
                              sector.lower() in s.get("summary", "").lower()]
            sector_insights[sector] = {
                "articles_found": len(sector_summaries),
                "coverage": "high" if len(sector_summaries) >= 3 else "medium" if len(sector_summaries) >= 1 else "low"
            }
        
        result = {
            "status": "success",
            "summaries": summaries,
            "processing_stats": processing_stats,
            "metadata": {
                "thread_id": thread_id,
                "sectors": sectors,
                "analysis_depth": analysis_depth,
                "searches_performed": final_state.get("search_count", 0),
                "max_searches": max_searches,
                "processing_time_seconds": round(processing_time, 2),
                "target_language": target_language,
                "timestamp_utc": datetime.utcnow().isoformat(),
                "type": "sector_analysis",
                "sector_insights": sector_insights,
                "processing_method": processing_method,
                "model": model,
                "provider_type": provider_type
            }
        }
        
        if len(summaries) > 0:
            await set_news_cache(redis_client, cache_key, result, CACHE_TTL_SECTOR_ANALYSIS)
        
        return result
        
    except Exception as e:
        error_logger.error(f"[SECTOR] Error: {str(e)}", exc_info=True)
        return {"status": "error", "summaries": [], "processing_stats": {}, "metadata": {"error": str(e), "thread_id": thread_id}}