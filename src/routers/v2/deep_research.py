"""
Deep Research API Router (Refactored)

Clear flow design:
1. POST /start   → Returns clarification questions OR research plan (JSON)
2. POST /clarify → Submit answers, returns research plan (JSON)
3. POST /execute → Confirm plan & execute, returns SSE stream
4. GET  /status  → Get session status (JSON)
5. DELETE /cancel → Cancel research (JSON)

Flow Example:
  Client                                Server
    |                                     |
    |--- POST /start ------------------>  |
    |<-- {type: "plan", plan: {...}} ---  |  (or "clarify" with questions)
    |                                     |
    |--- POST /execute ---------------->  |
    |<-- SSE: worker events ----------->  |
    |<-- SSE: synthesis events -------->  |
    |<-- SSE: [DONE] ------------------>  |
"""

import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.utils.config import settings
from src.agents.deep_research import (
    DeepResearchOrchestrator,
    DeepResearchConfig,
    ClarificationResponse,
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
    user_id: Optional[int] = Field(None, description="User ID")
    session_id: Optional[str] = Field(None, description="Session ID")
    model_tier: str = Field("standard", description="Model tier: budget, standard, premium")
    language: str = Field("en", description="Response language")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Phân tích cổ phiếu NVDA cho đầu tư dài hạn",
                "symbols": ["NVDA"],
                "model_tier": "standard",
                "language": "vi"
            }
        }


class ClarificationAnswersRequest(BaseModel):
    """Submit answers to clarification questions."""
    research_id: str = Field(..., description="Research session ID")
    answers: Dict[str, str] = Field(..., description="Answers keyed by question_id")


class ExecuteResearchRequest(BaseModel):
    """Confirm plan and start execution."""
    research_id: str = Field(..., description="Research session ID")
    modifications: Optional[Dict[str, Any]] = Field(None, description="Optional plan modifications")


class DeepResearchStartResponse(BaseModel):
    """Response from /start endpoint."""
    type: str = Field(..., description="Response type: 'clarify' or 'plan'")
    research_id: str = Field(..., description="Research session ID")
    status: str = Field(..., description="Current status")
    # For clarify type
    questions: Optional[List[Dict[str, Any]]] = Field(None, description="Clarification questions")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Query analysis")
    # For plan type
    plan: Optional[Dict[str, Any]] = Field(None, description="Research plan")


# ============================================================================
# SESSION STORAGE (In-memory for demo - use Redis in production)
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


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/start", response_model=DeepResearchStartResponse)
async def start_deep_research(
    request: DeepResearchStartRequest,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """
    Start a new deep research session.

    Returns JSON with either:
    - type="clarify": Questions need to be answered via /clarify
    - type="plan": Research plan ready, call /execute to start

    Example Response (clarify):
    ```json
    {
        "type": "clarify",
        "research_id": "dr_abc123",
        "status": "awaiting_clarification",
        "questions": [
            {
                "question_id": "q1",
                "question": "Bạn muốn phân tích theo hướng nào?",
                "options": ["Kỹ thuật", "Cơ bản", "Cả hai"]
            }
        ]
    }
    ```

    Example Response (plan):
    ```json
    {
        "type": "plan",
        "research_id": "dr_abc123",
        "status": "awaiting_confirmation",
        "plan": {
            "title": "Phân tích NVDA",
            "sections": [...]
        }
    }
    ```
    """
    logger.logger.info(f"[DeepResearch] Starting | query={request.query[:50]}...")

    # Create config
    config = DeepResearchConfig(
        model_tier=request.model_tier,
        skip_clarification=False,
        auto_confirm_plan=False,
    )

    # Create orchestrator
    orchestrator = DeepResearchOrchestrator(config=config)
    research_id = orchestrator.research_id

    # Save initial session
    save_session(research_id, {
        "research_id": research_id,
        "query": request.query,
        "symbols": request.symbols or [],
        "user_id": request.user_id,
        "session_id": request.session_id,
        "config": config.to_dict(),
        "status": "analyzing",
        "created_at": datetime.utcnow().isoformat(),
    })

    try:
        # Run analysis to get clarification or plan
        result = await orchestrator.analyze_and_plan(
            query=request.query,
            symbols=request.symbols,
        )

        if result.get("needs_clarification"):
            # Update session
            save_session(research_id, {
                **get_session(research_id),
                "status": "awaiting_clarification",
                "clarification_questions": result.get("questions"),
            })

            return DeepResearchStartResponse(
                type="clarify",
                research_id=research_id,
                status="awaiting_clarification",
                questions=result.get("questions"),
                analysis=result.get("analysis"),
            )
        else:
            # Have plan ready
            plan = result.get("plan")
            save_session(research_id, {
                **get_session(research_id),
                "status": "awaiting_confirmation",
                "plan": plan,
            })

            return DeepResearchStartResponse(
                type="plan",
                research_id=research_id,
                status="awaiting_confirmation",
                plan=plan,
            )

    except Exception as e:
        logger.logger.error(f"[DeepResearch] Start error: {e}", exc_info=True)
        save_session(research_id, {
            **get_session(research_id),
            "status": "error",
            "error": str(e),
        })
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clarify", response_model=DeepResearchStartResponse)
async def submit_clarification(
    request: ClarificationAnswersRequest,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """
    Submit answers to clarification questions.

    After receiving type="clarify" from /start, submit answers here.
    Returns the research plan.

    Example Request:
    ```json
    {
        "research_id": "dr_abc123",
        "answers": {
            "q1": "Cả hai",
            "q2": "Dài hạn"
        }
    }
    ```
    """
    logger.logger.info(f"[DeepResearch] Clarify | id={request.research_id}")

    session = get_session(request.research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    if session.get("status") != "awaiting_clarification":
        raise HTTPException(
            status_code=400,
            detail=f"Research is not awaiting clarification. Status: {session.get('status')}"
        )

    # Create orchestrator with saved context
    config_data = session.get("config", {})
    config = DeepResearchConfig(
        model_tier=config_data.get("model_tier", "standard"),
    )

    orchestrator = DeepResearchOrchestrator(config=config)
    orchestrator.research_id = request.research_id
    orchestrator.clarification_answers = request.answers

    try:
        # Generate plan with clarification answers
        result = await orchestrator.generate_plan_with_context(
            query=session.get("query"),
            symbols=session.get("symbols"),
            clarification_answers=request.answers,
        )

        plan = result.get("plan")
        save_session(request.research_id, {
            **session,
            "status": "awaiting_confirmation",
            "plan": plan,
            "clarification_answers": request.answers,
        })

        return DeepResearchStartResponse(
            type="plan",
            research_id=request.research_id,
            status="awaiting_confirmation",
            plan=plan,
        )

    except Exception as e:
        logger.logger.error(f"[DeepResearch] Clarify error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute")
async def execute_research(
    request: ExecuteResearchRequest,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """
    Confirm plan and execute research.

    Returns SSE stream with research progress and results.

    SSE Events:
    - research_init: Research started
    - worker_spawned: Worker agent started
    - worker_progress: Worker progress update
    - worker_artifact: Worker produced finding
    - worker_completed: Worker finished
    - synthesis_started: Report synthesis started
    - synthesis_artifact: Report section generated
    - research_completed: All done

    Example Request:
    ```json
    {
        "research_id": "dr_abc123"
    }
    ```
    """
    logger.logger.info(f"[DeepResearch] Execute | id={request.research_id}")

    session = get_session(request.research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    if session.get("status") != "awaiting_confirmation":
        raise HTTPException(
            status_code=400,
            detail=f"Research is not ready to execute. Status: {session.get('status')}"
        )

    # Update status
    save_session(request.research_id, {
        **session,
        "status": "executing",
    })

    # Create orchestrator
    config_data = session.get("config", {})
    config = DeepResearchConfig(
        model_tier=config_data.get("model_tier", "standard"),
        auto_confirm_plan=True,
    )

    orchestrator = DeepResearchOrchestrator(config=config)
    orchestrator.research_id = request.research_id
    orchestrator.clarification_answers = session.get("clarification_answers", {})

    async def generate_events():
        """Generate SSE events during execution."""
        try:
            async for event in orchestrator.execute_research(
                query=session.get("query"),
                symbols=session.get("symbols"),
                plan_data=session.get("plan"),
            ):
                event_type = event.get("type", "progress")

                # Update session on completion
                if event_type == "research_completed":
                    save_session(request.research_id, {
                        **get_session(request.research_id),
                        "status": "completed",
                        "result": event,
                    })

                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"

            yield format_sse_done()

        except Exception as e:
            logger.logger.error(f"[DeepResearch] Execute error: {e}", exc_info=True)
            save_session(request.research_id, {
                **get_session(request.research_id),
                "status": "error",
                "error": str(e),
            })
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
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


@router.get("/status/{research_id}")
async def get_research_status(
    research_id: str,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """
    Get the status of a research session.

    Statuses:
    - analyzing: Initial analysis in progress
    - awaiting_clarification: Need to call /clarify
    - awaiting_confirmation: Need to call /execute
    - executing: Research in progress
    - completed: Research done
    - error: An error occurred
    - cancelled: User cancelled
    """
    session = get_session(research_id)

    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    return {
        "research_id": research_id,
        "status": session.get("status"),
        "query": session.get("query"),
        "symbols": session.get("symbols"),
        "created_at": session.get("created_at"),
        "updated_at": session.get("updated_at"),
        "plan": session.get("plan") if session.get("status") in ["awaiting_confirmation", "executing", "completed"] else None,
        "error": session.get("error"),
    }


@router.delete("/cancel/{research_id}")
async def cancel_research(
    research_id: str,
    api_key: str = Depends(api_key_auth.author_with_api_key),
):
    """Cancel an ongoing research session."""
    session = get_session(research_id)

    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    save_session(research_id, {
        **session,
        "status": "cancelled",
        "cancelled_at": datetime.utcnow().isoformat(),
    })

    return {"message": "Research cancelled", "research_id": research_id}
