"""
Deep Research API Router

Provides endpoints for the Deep Research Mode:
- POST /v2/deep-research/start: Start a new deep research session
- POST /v2/deep-research/confirm: Confirm a research plan
- POST /v2/deep-research/clarify: Submit clarification answers
- GET /v2/deep-research/status/{research_id}: Get research status

Deep Research Flow:
1. Client calls /start with query
2. Server may respond with clarification questions (if needed)
3. Client submits answers via /clarify
4. Server creates and returns research plan
5. Client confirms plan via /confirm
6. Server executes research and streams results
"""

import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.utils.config import settings
from src.agents.deep_research import (
    DeepResearchOrchestrator,
    DeepResearchConfig,
    ClarificationResponse,
    create_orchestrator,
)
from src.agents.deep_research.streaming.artifact_emitter import format_sse_done


# ============================================================================
# ROUTER SETUP
# ============================================================================

router = APIRouter(
    prefix="/v2/deep-research",
    tags=["Deep Research"],
)

logger = LoggerMixin()
api_key_auth = APIKeyAuth()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class DeepResearchStartRequest(BaseModel):
    """Request to start a deep research session."""
    query: str = Field(..., description="Research query")
    symbols: Optional[List[str]] = Field(None, description="Symbols to analyze")
    user_id: Optional[int] = Field(None, description="User ID for personalization")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    model_tier: str = Field("standard", description="Model tier: budget, standard, premium")
    skip_clarification: bool = Field(False, description="Skip clarification phase")
    auto_confirm_plan: bool = Field(False, description="Auto-confirm research plan")
    language: str = Field("en", description="Response language")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Analyze NVDA for long-term investment",
                "symbols": ["NVDA"],
                "model_tier": "standard",
                "language": "en"
            }
        }


class ClarificationAnswersRequest(BaseModel):
    """Request to submit clarification answers."""
    research_id: str = Field(..., description="Research session ID")
    answers: Dict[str, str] = Field(..., description="Answers to clarification questions")

    class Config:
        json_schema_extra = {
            "example": {
                "research_id": "dr_abc123",
                "answers": {
                    "time_horizon": "Long-term (> 2 years)",
                    "investment_goal": "Capital growth"
                }
            }
        }


class ConfirmPlanRequest(BaseModel):
    """Request to confirm a research plan."""
    research_id: str = Field(..., description="Research session ID")
    plan_id: str = Field(..., description="Plan ID to confirm")
    modifications: Optional[Dict[str, Any]] = Field(None, description="Optional plan modifications")


class DeepResearchStatusResponse(BaseModel):
    """Response with research status."""
    research_id: str
    status: str
    progress: float = 0.0
    current_phase: Optional[str] = None
    sections_completed: int = 0
    total_sections: int = 0
    error: Optional[str] = None


# ============================================================================
# IN-MEMORY SESSION STORAGE (for demo - use Redis in production)
# ============================================================================

_research_sessions: Dict[str, Dict[str, Any]] = {}


def get_session(research_id: str) -> Optional[Dict[str, Any]]:
    """Get a research session by ID."""
    return _research_sessions.get(research_id)


def save_session(research_id: str, data: Dict[str, Any]):
    """Save a research session."""
    _research_sessions[research_id] = {
        **data,
        "updated_at": datetime.utcnow().isoformat(),
    }


def cleanup_old_sessions(max_age_hours: int = 24):
    """Clean up old sessions (call periodically)."""
    cutoff = datetime.utcnow()
    for rid in list(_research_sessions.keys()):
        session = _research_sessions.get(rid)
        if session:
            created = session.get("created_at", "")
            # Simple cleanup - in production use proper TTL
            pass


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/start")
async def start_deep_research(
    request: DeepResearchStartRequest,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """
    Start a new deep research session.

    This endpoint initiates the research process:
    1. If clarification is needed, returns clarification questions
    2. If query is clear, proceeds to create research plan
    3. If auto_confirm_plan is True, immediately starts execution

    Response is SSE stream with events:
    - research_init: Research started
    - clarification_request: Questions for user (if needed)
    - plan_created: Research plan ready for confirmation
    - worker_spawned/progress/artifact/completed: Worker events
    - synthesis_*: Synthesis events
    - research_completed: All done

    Returns:
        StreamingResponse with SSE events
    """
    logger.logger.info(
        f"[DeepResearch] Starting research | query={request.query[:50]}... | "
        f"tier={request.model_tier}"
    )

    # Create config
    config = DeepResearchConfig(
        model_tier=request.model_tier,
        skip_clarification=request.skip_clarification,
        auto_confirm_plan=request.auto_confirm_plan,
    )

    # Create orchestrator
    orchestrator = DeepResearchOrchestrator(
        config=config,
        api_key=api_key,
    )

    # Save session
    save_session(orchestrator.research_id or "temp", {
        "research_id": orchestrator.research_id,
        "query": request.query,
        "symbols": request.symbols,
        "user_id": request.user_id,
        "session_id": request.session_id,
        "config": config.to_dict(),
        "status": "started",
        "created_at": datetime.utcnow().isoformat(),
    })

    async def generate_events():
        """Generate SSE events from orchestrator."""
        try:
            async for event in orchestrator.run_research(
                query=request.query,
                symbols=request.symbols,
                user_id=request.user_id,
                session_id=request.session_id,
                skip_clarification=request.skip_clarification,
            ):
                # Update session status based on event type
                event_type = event.get("type", "progress")
                current_session = get_session(orchestrator.research_id) or {}

                if event_type == "clarification_request":
                    updated_data = {
                        **current_session,
                        "status": "awaiting_clarification",
                        "clarification_questions": event.get("data", {}).get("questions"),
                    }
                    save_session(orchestrator.research_id, updated_data)

                elif event_type == "plan_created":
                    updated_data = {
                        **current_session,
                        "status": "awaiting_confirmation",
                        "plan": event.get("data", {}).get("plan"),
                    }
                    save_session(orchestrator.research_id, updated_data)

                elif event_type == "research_completed":
                    updated_data = {
                        **current_session,
                        "status": "completed",
                    }
                    save_session(orchestrator.research_id, updated_data)

                # Format as SSE
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

            yield format_sse_done()

        except Exception as e:
            logger.logger.error(f"[DeepResearch] Error: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "error": str(e),
                "research_id": orchestrator.research_id,
            }
            yield f"event: error\n"
            yield f"data: {json.dumps(error_event)}\n\n"
            yield format_sse_done()

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/clarify")
async def submit_clarification(
    request: ClarificationAnswersRequest,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """
    Submit answers to clarification questions.

    After receiving clarification_request event, client should
    collect user answers and submit them here.

    This resumes the research process with the provided context.
    """
    logger.logger.info(
        f"[DeepResearch] Clarification received | research_id={request.research_id}"
    )

    # Get session
    session = get_session(request.research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    if session.get("status") != "awaiting_clarification":
        raise HTTPException(
            status_code=400,
            detail=f"Research is not awaiting clarification. Status: {session.get('status')}"
        )

    # Create config from session
    config_data = session.get("config", {})
    config = DeepResearchConfig(
        model_tier=config_data.get("model_tier", "standard"),
        skip_clarification=True,  # Already have clarification
        auto_confirm_plan=config_data.get("auto_confirm_plan", False),
    )

    # Create new orchestrator with clarification response
    orchestrator = DeepResearchOrchestrator(
        config=config,
        api_key=api_key,
    )
    orchestrator.research_id = request.research_id

    clarification_response = ClarificationResponse(
        research_id=request.research_id,
        answers=request.answers,
    )

    async def generate_events():
        try:
            async for event in orchestrator.run_research(
                query=session.get("query", ""),
                symbols=session.get("symbols"),
                user_id=session.get("user_id"),
                session_id=session.get("session_id"),
                skip_clarification=True,
                clarification_response=clarification_response,
            ):
                event_type = event.get("type", "progress")
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

            yield format_sse_done()

        except Exception as e:
            logger.logger.error(f"[DeepResearch] Clarification error: {e}")
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield format_sse_done()

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
    )


@router.post("/confirm")
async def confirm_plan(
    request: ConfirmPlanRequest,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """
    Confirm a research plan and start execution.

    After receiving plan_created event, client should display
    the plan and allow user to confirm or modify.

    This endpoint confirms the plan and starts worker execution.
    """
    logger.logger.info(
        f"[DeepResearch] Plan confirmed | research_id={request.research_id}"
    )

    # Get session
    session = get_session(request.research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    if session.get("status") != "awaiting_confirmation":
        raise HTTPException(
            status_code=400,
            detail=f"Research is not awaiting plan confirmation. Status: {session.get('status')}"
        )

    # Create config from session
    config_data = session.get("config", {})
    config = DeepResearchConfig(
        model_tier=config_data.get("model_tier", "standard"),
        skip_clarification=True,
        auto_confirm_plan=True,  # Plan is confirmed
    )

    # Create orchestrator with confirmed plan
    orchestrator = DeepResearchOrchestrator(
        config=config,
        api_key=api_key,
    )
    orchestrator.research_id = request.research_id

    # Load and confirm plan
    plan_data = session.get("plan")
    if plan_data:
        orchestrator.clarification_answers = session.get("clarification_answers", {})
        # Plan will be recreated but auto-confirmed

    async def generate_events():
        try:
            async for event in orchestrator.run_research(
                query=session.get("query", ""),
                symbols=session.get("symbols"),
                user_id=session.get("user_id"),
                session_id=session.get("session_id"),
                skip_clarification=True,
                confirmed_plan_id=request.plan_id,
            ):
                event_type = event.get("type", "progress")
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

            yield format_sse_done()

        except Exception as e:
            logger.logger.error(f"[DeepResearch] Execution error: {e}")
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield format_sse_done()

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
    )


@router.get("/status/{research_id}")
async def get_research_status(
    research_id: str,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """
    Get the status of a research session.

    Useful for checking research progress or resuming
    after connection loss.
    """
    session = get_session(research_id)

    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    return DeepResearchStatusResponse(
        research_id=research_id,
        status=session.get("status", "unknown"),
        progress=session.get("progress", 0.0),
        current_phase=session.get("current_phase"),
        sections_completed=session.get("sections_completed", 0),
        total_sections=session.get("total_sections", 0),
        error=session.get("error"),
    )


@router.delete("/cancel/{research_id}")
async def cancel_research(
    research_id: str,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """
    Cancel an ongoing research session.
    """
    session = get_session(research_id)

    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    # Update session status
    save_session(research_id, {
        **session,
        "status": "cancelled",
        "cancelled_at": datetime.utcnow().isoformat(),
    })

    return {"message": "Research cancelled", "research_id": research_id}
