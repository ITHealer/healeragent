import asyncio
import logging
from typing import Optional

import aioredis
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from src.helpers.redis_cache import get_cache, set_cache, get_redis_client
from src.models.equity_forecast import (
    APIResponse,
    APIResponseData,
    PriceForecastBand,
    PriceForecastContext,
    ForecastAIAnalysisResponse,
)
from src.services.v2.forecast_service import ForecastService
from src.services.v2.forecast_ai_service import ForecastAIService
from src.utils.config import settings
from src.utils.logger.set_up_log_dataFMP import setup_logger

logger = setup_logger(__name__, log_level=logging.INFO)

router = APIRouter(prefix="/equity-forecast")

forecast_service = ForecastService()
forecast_ai_service = ForecastAIService()


@router.get("/band", response_model=APIResponse[PriceForecastBand])
async def get_forecast_band(
    symbol: str = Query(..., min_length=1),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client),
):
    """
    Get price forecast band for chart rendering.
    
    FMP API Called:
    - GET /v4/price-target-consensus?symbol={symbol}
    
    Frontend Usage:
    - Draw shaded zone from target_low to target_high
    - Draw dashed line at target_mean
    - Show tooltip: "Target: $X-$Y (Avg: $Z) - N analysts"
    """
    
    symbol = symbol.strip().upper()
    cache_key = f"forecast_band_{symbol}"

    cached = await get_cache(redis_client, cache_key, APIResponse[PriceForecastBand])
    if cached and cached.data:
        logger.info(f"[Band] Cache HIT: {cache_key}")
        return cached

    logger.info(f"[Band] Fetching from FMP: {symbol}")

    consensus = await forecast_service.get_price_target_consensus(symbol)
    if not consensus:
        raise HTTPException(404, f"No forecast data for {symbol}")

    band = PriceForecastBand(
        symbol=symbol,
        as_of=(
            consensus.lastUpdated.isoformat() if consensus.lastUpdated
            else (consensus.date.isoformat() if consensus.date else None)
        ),
        target_low=consensus.targetLow,
        target_high=consensus.targetHigh,
        target_mean=consensus.targetConsensus or consensus.targetAvg or consensus.consensus,
        target_median=consensus.targetMedian,
        number_of_analysts=consensus.numberOfAnalysts or consensus.analystCount or consensus.numAnalysts,
        currency=consensus.currency or consensus.targetCurrency,
    )

    response = APIResponse[PriceForecastBand](
        message="OK",
        status="200",
        provider_used="fmp",
        data=APIResponseData[PriceForecastBand](data=band),
    )

    ttl = getattr(settings, "CACHE_TTL_ANALYST_ESTIMATES", settings.CACHE_TTL_LISTS)
    await set_cache(redis_client, cache_key, response, expiry=ttl)
    
    return response


@router.get("/context", response_model=APIResponse[PriceForecastContext])
async def get_forecast_context(
    symbol: str = Query(..., min_length=1),
    period: str = Query("annual", pattern="^(annual|quarter)$"),
    limit_estimates: int = Query(4, ge=1, le=20),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client),
):
    """
    Get complete forecast context for detailed tooltip/panel.
    
    FMP APIs Called (parallel):
    1. GET /v4/price-target-consensus?symbol={symbol}
    2. GET /v3/analyst-estimates?symbol={symbol}&period={period}&limit={limit}
    3. GET /v3/rating?symbol={symbol}
    4. GET /v4/price-target-summary?symbol={symbol}
    5. GET /v4/score?symbol={symbol}
    6. GET /v3/discounted-cash-flow/{symbol}
    
    Frontend Usage:
    - Display in tooltip/side panel with sections:
      * Price Target (low, high, consensus, analysts)
      * Growth Projections (revenue, EPS with CAGR)
      * Financial Health (Piotroski, rating score)
      * Valuation (DCF vs market price)
    - "Ask AI to Explain" button triggers /explain
    """
    
    symbol = symbol.strip().upper()
    cache_key = f"forecast_context_{symbol}_{period}_{limit_estimates}"

    cached = await get_cache(redis_client, cache_key, APIResponse[PriceForecastContext])
    if cached and cached.data:
        logger.info(f"[Context] Cache HIT: {cache_key}")
        return cached

    logger.info(f"[Context] Fetching from FMP (6 endpoints): {symbol}")

    results = await asyncio.gather(
        forecast_service.get_price_target_consensus(symbol),
        forecast_service.get_analyst_estimates(symbol, period, limit_estimates),
        forecast_service.get_ratings_snapshot(symbol),
        forecast_service.get_price_target_summary(symbol),
        forecast_service.get_financial_score(symbol),
        forecast_service.get_discounted_cash_flow(symbol),
        return_exceptions=True,
    )

    def unwrap(val):
        if isinstance(val, Exception):
            logger.warning(f"[Context] Partial failure: {val}")
        return None if isinstance(val, Exception) else val

    consensus = unwrap(results[0])

    band = None
    if consensus:
        band = PriceForecastBand(
            symbol=symbol,
            as_of=(
                consensus.lastUpdated.isoformat() if consensus.lastUpdated
                else (consensus.date.isoformat() if consensus.date else None)
            ),
            target_low=consensus.targetLow,
            target_high=consensus.targetHigh,
            target_mean=consensus.targetConsensus or consensus.targetAvg or consensus.consensus,
            target_median=consensus.targetMedian,
            number_of_analysts=consensus.numberOfAnalysts or consensus.analystCount or consensus.numAnalysts,
            currency=consensus.currency or consensus.targetCurrency,
        )

    context = PriceForecastContext(
        symbol=symbol,
        price_target_band=band,
        price_target_consensus=consensus,
        price_target_summary=unwrap(results[3]),
        analyst_estimates=unwrap(results[1]),
        ratings_snapshot=unwrap(results[2]),
        financial_scores=unwrap(results[4]),
        dcf_items=unwrap(results[5]),
    )

    response = APIResponse[PriceForecastContext](
        message="OK",
        status="200",
        provider_used="fmp",
        data=APIResponseData[PriceForecastContext](data=context),
    )

    ttl = getattr(settings, "CACHE_TTL_ANALYST_ESTIMATES", settings.CACHE_TTL_LISTS)
    await set_cache(redis_client, cache_key, response, expiry=ttl)
    
    return response


@router.post("/explain", response_model=APIResponse[ForecastAIAnalysisResponse])
async def explain_forecast(
    symbol: str = Query(..., min_length=1),
    current_price: Optional[float] = Query(None),
    model_name: str = Query("gpt-4.1-nano-2025-04-14"),
    provider_type: str = Query("openai", pattern="^(openai|ollama|anthropic)$"),
    language: str = Query("vi", pattern="^(vi|en|zh)$"),
    period: str = Query("annual"),
    limit_estimates: int = Query(4),
    max_tokens: int = Query(4000, ge=1000, le=8000, description="Max tokens for analysis"),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client),
):
    """
    Generate AI analysis (production-grade with fallback).
    
    Returns markdown-based analysis that's always usable.
    """
    
    symbol = symbol.strip().upper()
    context_key = f"forecast_context_{symbol}_{period}_{limit_estimates}"
    
    # Get context
    cached_ctx = await get_cache(redis_client, context_key, APIResponse[PriceForecastContext])
    
    if cached_ctx and cached_ctx.data:
        logger.info(f"[Explain] Cached context: {symbol}")
        context = cached_ctx.data.data
    else:
        logger.info(f"[Explain] Fetching context: {symbol}")
        
        results = await asyncio.gather(
            forecast_service.get_price_target_consensus(symbol),
            forecast_service.get_analyst_estimates(symbol, period, limit_estimates),
            forecast_service.get_ratings_snapshot(symbol),
            forecast_service.get_price_target_summary(symbol),
            forecast_service.get_financial_score(symbol),
            forecast_service.get_discounted_cash_flow(symbol),
            return_exceptions=True,
        )
        
        def u(v):
            return None if isinstance(v, Exception) else v
        
        consensus = u(results[0])
        
        band = None
        if consensus:
            band = PriceForecastBand(
                symbol=symbol,
                as_of=(
                    consensus.lastUpdated.isoformat() if consensus.lastUpdated
                    else (consensus.date.isoformat() if consensus.date else None)
                ),
                target_low=consensus.targetLow,
                target_high=consensus.targetHigh,
                target_mean=consensus.targetConsensus or consensus.targetAvg,
                target_median=consensus.targetMedian,
                number_of_analysts=consensus.numberOfAnalysts or consensus.analystCount,
                currency=consensus.currency,
            )
        
        context = PriceForecastContext(
            symbol=symbol,
            price_target_band=band,
            price_target_consensus=consensus,
            price_target_summary=u(results[3]),
            analyst_estimates=u(results[1]),
            ratings_snapshot=u(results[2]),
            financial_scores=u(results[4]),
            dcf_items=u(results[5]),
        )
    
    # Generate analysis (with fallback)
    analysis = await forecast_ai_service.generate_forecast_analysis(
        symbol=symbol,
        forecast_context=context,
        current_price=current_price,
        model_name=model_name,
        provider_type=provider_type,
        language=language,
        max_tokens=max_tokens,
    )
    
    response = APIResponse[ForecastAIAnalysisResponse](
        message="OK",
        status="200",
        provider_used=f"ai_{provider_type}",
        data=APIResponseData[ForecastAIAnalysisResponse](data=analysis),
    )
    
    return response

@router.post("/explain/stream")
async def explain_forecast_stream(
    symbol: str = Query(..., min_length=1),
    current_price: Optional[float] = Query(None),
    model_name: str = Query("gpt-4.1-nano-2025-04-14"),
    provider_type: str = Query("openai"),
    language: str = Query("vi"),
    period: str = Query("annual"),
    limit_estimates: int = Query(4),
    redis_client: Optional[aioredis.Redis] = Depends(get_redis_client),
):
    """
    Stream markdown analysis for chat interfaces.
    
    Frontend Usage:
    - Display with typing effect
    - Use Server-Sent Events or fetch with streaming
    """
    
    symbol = symbol.strip().upper()
    context_key = f"forecast_context_{symbol}_{period}_{limit_estimates}"
    
    cached_ctx = await get_cache(redis_client, context_key, APIResponse[PriceForecastContext])
    
    if cached_ctx and cached_ctx.data:
        context = cached_ctx.data.data
    else:
        results = await asyncio.gather(
            forecast_service.get_price_target_consensus(symbol),
            forecast_service.get_analyst_estimates(symbol, period, limit_estimates),
            forecast_service.get_ratings_snapshot(symbol),
            forecast_service.get_price_target_summary(symbol),
            forecast_service.get_financial_score(symbol),
            forecast_service.get_discounted_cash_flow(symbol),
            return_exceptions=True,
        )
        
        def u(v):
            return None if isinstance(v, Exception) else v
        
        consensus = u(results[0])
        
        band = None
        if consensus:
            band = PriceForecastBand(
                symbol=symbol,
                as_of=(
                    consensus.lastUpdated.isoformat() if consensus.lastUpdated
                    else (consensus.date.isoformat() if consensus.date else None)
                ),
                target_low=consensus.targetLow,
                target_high=consensus.targetHigh,
                target_mean=consensus.targetConsensus or consensus.targetAvg,
                target_median=consensus.targetMedian,
                number_of_analysts=consensus.numberOfAnalysts or consensus.analystCount,
                currency=consensus.currency,
            )
        
        context = PriceForecastContext(
            symbol=symbol,
            price_target_band=band,
            price_target_consensus=consensus,
            price_target_summary=u(results[3]),
            analyst_estimates=u(results[1]),
            ratings_snapshot=u(results[2]),
            financial_scores=u(results[4]),
            dcf_items=u(results[5]),
        )
    
    async def stream():
        try:
            system = forecast_ai_service._build_system_prompt(language)
            user = forecast_ai_service._build_user_prompt(symbol, context, current_price)

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

            # Get API key based on provider type
            api_key = None
            if provider_type == "openai":
                api_key = settings.OPENAI_API_KEY
            elif provider_type == "anthropic":
                api_key = getattr(settings, "ANTHROPIC_API_KEY", None)

            async for chunk in forecast_ai_service.llm_provider.stream_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                clean_thinking=True,
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"[Stream] Error: {e}", exc_info=True)
            yield f"\n\n[Error: {str(e)}]"
    
    return StreamingResponse(stream(), media_type="text/plain")