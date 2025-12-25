from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, Request, HTTPException

from src.utils.constants import LocalModelName
from src.providers.provider_factory import ProviderType
from src.schemas.response import PromptRequest
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.chat_management_helper import ChatService
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.handlers.v2.tool_call_handler import tool_call


router = APIRouter(prefix="/router")

# Initialize Instance
api_key_auth = APIKeyAuth()
chat_service = ChatService()
logger_mixin = LoggerMixin()
logger = logger_mixin.logger

class ToolCallRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    query: str = Field(..., description="User query in any language")
    model_name: str = Field(LocalModelName.GPTOSS, description="LLM model name")
    provider_type: str = Field(ProviderType.OLLAMA, description="Provider type")


@router.post("/tool_calls")
async def analyze_tool_call(
    request: Request,
    tool_request: ToolCallRequest,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    try:
        # Get conversation history
        conversation_history = None
        # if tool_request.session_id:
        #     conversation_history = chat_service.get_chat_history(
        #         session_id=tool_request.session_id,
        #         limit=4
        #     )
        
        result = await tool_call(
            prompt=tool_request.query,
            model_name=tool_request.model_name,
            provider_type=tool_request.provider_type,
            conversation_history=conversation_history
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Tool analysis error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Tool analysis failed: {str(e)}"
        )
