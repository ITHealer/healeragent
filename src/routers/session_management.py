from typing import Dict, Any
from fastapi import APIRouter, Response, Query, status, Depends, Request

from src.schemas.response import BasicResponse
from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.handlers.llm_chat_handler import ChatHandler, ChatMessageHistory


api_key_auth = APIKeyAuth()
router = APIRouter()

logger_mixin = LoggerMixin()
logger = logger_mixin.logger


# ======================== CRUD: CHAT SESSIONS =======================
@router.post("/sessions/create-session", response_description="Create session")
async def create_session(
    request: Request,
    response: Response,
    user_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    organization_id = getattr(request.state, "organization_id", None)
    
    request_user_id = getattr(request.state, "user_id", None)
    if request_user_id != user_id:
        user_role = getattr(request.state, "role", None)
        if user_role != "ADMIN":
            response.status_code = status.HTTP_403_FORBIDDEN
            return BasicResponse(
                status="Failed",
                message="You can only create sessions for yourself",
                data=None
            )
    
    resp = ChatHandler().create_session_id(
        user_id=user_id,
        organization_id=organization_id
    )
    
    if resp.data:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp


@router.delete("/sessions/{session_id}", response_description="Delete chat session completely")
async def delete_chat_session(
    request: Request,
    response: Response,
    session_id: str,
    delete_documents: bool = Query(False, description="Delete related documents"),
    delete_collections: bool = Query(False, description="Delete related collections"),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Delete a chat session completely, with options to delete related documents and collections
    
    Args:
        request: Request object with user authentication info
        session_id: The ID of the chat session to delete
        delete_documents: Whether to delete documents referenced in the session
        delete_collections: Whether to delete collections containing the documents
        
    Returns:
        BasicResponse: Response indicating success or failure
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    resp = await ChatHandler().delete_session_completely(
        session_id=session_id,
        user_id=user_id,
        organization_id=organization_id,
        delete_documents=delete_documents,
        delete_collections=delete_collections
    )
    
    if resp.Status == "Success":
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    return resp


@router.post("/sessions/{session_id}/delete-history", response_description="Delete history of session id")
async def delete_chat_history(
    request: Request,
    response: Response,
    session_id: str,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Delete the chat history for a session
    
    Args:
        request: Request object with user authentication info
        session_id: The ID of the chat session
        
    Returns:
        JSON response indicating success or failure
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    resp = ChatMessageHistory().delete_message_history(
        session_id=session_id,
        user_id=user_id,
        organization_id=organization_id
    )
    
    if resp.status == "Success":
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp


@router.post("/sessions/{session_id}/get-chat-history", response_description="Chat history of session id")
async def get_chat_history(
    request: Request,
    response: Response,
    session_id: str,
    limit: int = 10,
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Get the chat history for a session
    
    Args:
        request: Request object with user authentication info
        session_id: The ID of the chat session
        limit: Maximum number of messages to retrieve (default: 10)
        
    Returns:
        JSON response with the chat history
    """
    user_id = getattr(request.state, "user_id", None)
    organization_id = getattr(request.state, "organization_id", None)
    
    resp = ChatMessageHistory().get_list_message_history(
        session_id=session_id,
        limit=limit,
        user_id=user_id,
        organization_id=organization_id
    )
    
    if resp.status == "Success":
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    return resp