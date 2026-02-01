"""
FastAPI router for the invest_agent module.

Why: Provides a clean, isolated API endpoint (POST /invest/chat/stream) that
does not touch or interfere with existing routes in src/routers/. This is the
single entry point for the V3 Mode System.

How: Receives an InvestChatRequest, wires it into the Orchestrator, and
streams SSE events back to the client via StreamingResponse.
"""

import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from src.invest_agent.core.config import InvestChatRequest
from src.invest_agent.core.events import (
    SSEEvent,
    event_session_start,
    event_error,
    event_done,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/invest", tags=["invest-agent-v3"])


async def _get_orchestrator():
    """Dependency injection for the Orchestrator singleton.

    Why lazy import: Avoids circular imports and allows the orchestrator
    to be initialized with all its dependencies at startup time via main.py.
    """
    from src.invest_agent.main import get_orchestrator
    return get_orchestrator()


@router.post("/chat/stream")
async def invest_chat_stream(
    request: InvestChatRequest,
    orchestrator=Depends(_get_orchestrator),
):
    """Stream an AI-powered investment chat response.

    Accepts a user query with optional mode selection and returns an SSE stream.
    The orchestrator handles mode resolution, tool execution, evaluation,
    and response synthesis internally.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def _generate() -> AsyncGenerator[str, None]:
        try:
            # Emit session start
            yield event_session_start(session_id).to_sse()

            # Delegate to orchestrator - it yields SSEEvent objects
            async for event in orchestrator.run(
                query=request.query,
                session_id=session_id,
                response_mode=request.response_mode,
                enable_thinking=request.enable_thinking,
                model_name_override=request.model_name,
                provider_type=request.provider_type,
                ui_context=request.ui_context,
                conversation_history=request.conversation_history,
                user_id=request.user_id,
            ):
                if isinstance(event, SSEEvent):
                    yield event.to_sse()
                elif isinstance(event, str):
                    # Raw content chunk passthrough
                    yield f"data: {event}\n\n"

        except Exception as exc:
            logger.exception(f"[InvestAgent] Stream error for session {session_id}")
            yield event_error(
                message=f"Internal error: {str(exc)[:200]}",
                code="stream_error",
            ).to_sse()

        # Signal end of stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
