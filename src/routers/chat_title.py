from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session
from datetime import datetime

from src.database.models.schemas import ChatSessions, Messages
from src.services.llm_service import LLMService
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.providers.provider_factory import ModelProviderFactory


router = APIRouter()

api_key_auth = APIKeyAuth()

# Request/Response Models
class GenerateTitleRequest(BaseModel):
    session_id: str
    model_name: str = Field(default="gpt-4.1-nano", description="LLM model name")
    provider_type: str = Field(default="openai", description="LLM provider type")
    max_messages: int = Field(default=5, description="Maximum messages to consider")
    
class GenerateTitleResponse(BaseModel):
    title: str
    session_id: str
    generated_at: datetime

class UpdateTitleRequest(BaseModel):
    session_id: str
    title: str

class GetTitleResponse(BaseModel):
    session_id: str
    title: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


# Prompt template to generate title
TITLE_GENERATION_PROMPT = """
Based on the following conversation, generate a concise title (up to 50 characters).
Match the language of the conversation.

Conversation:
{conversation}

Instructions:
- Focus on the main topic or question
- Be specific and descriptive
- Return ONLY the title without quotes or explanation
"""

async def get_db():
    """Get database session"""
    from src.database import PostgreSQLConnection
    
    db_connection = PostgreSQLConnection()
    session = db_connection.get_session()
    try:
        yield session
    finally:
        session.close()

@router.post("/sessions/{session_id}/generate-title", response_model=GenerateTitleResponse)
async def generate_chat_title(
    request: GenerateTitleRequest,
    db: Session = Depends(get_db),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Generate title cho chat session dựa trên nội dung messages
    """
    # Check if session exists
    session = db.query(ChatSessions).filter(ChatSessions.id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    # Get the first message of the session
    messages = db.query(Messages).filter(
        Messages.session_id == request.session_id
    ).order_by(Messages.created_at).limit(request.max_messages).all()
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages found in session")
    
    # Prepare conversation text
    conversation_text = "\n".join([
        f"{msg.sender_role}: {msg.content}" for msg in messages
    ])
    
    # Generate title using LLM
    try:
        llm_service = LLMService()
        # Get API Key
        api_key = ModelProviderFactory._get_api_key(request.provider_type)

        prompt = TITLE_GENERATION_PROMPT.format(conversation=conversation_text)
        generated_title = await llm_service.generate_text(prompt, provider_type=request.provider_type, model_name=request.model_name, api_key=api_key)
        
        # Clean and validate title
        generated_title = generated_title.strip()
        if len(generated_title) > 50:
            generated_title = generated_title[:47] + "..."
            
        # Update title trong database
        session.title = generated_title
        db.commit()
        
        return GenerateTitleResponse(
            title=generated_title,
            session_id=request.session_id,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error generating title: {str(e)}")


@router.put("/sessions/{session_id}/title")
async def update_chat_title(
    request: UpdateTitleRequest,
    db: Session = Depends(get_db),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    session = db.query(ChatSessions).filter(ChatSessions.id == request.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    session.title = request.title
    db.commit()
    
    return {"message": "Title updated successfully", "title": request.title}


@router.get("/sessions/{session_id}/title", response_model=GetTitleResponse)
async def get_chat_title(
    session_id: str,
    db: Session = Depends(get_db),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Get title of a chat session
    
    Args:
        session_id: The ID of the chat session
        
    Returns:
        GetTitleResponse: Session title information
    """
    # Query session from database
    session = db.query(ChatSessions).filter(ChatSessions.id == session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    return GetTitleResponse(
        session_id=str(session.id),
        title=session.title if session.title else "Untitled",
        created_at=session.start_date,
        updated_at=session.end_date
    )


# @router.get("/sessions/user/{user_id}/titles")
# async def get_user_session_titles(
#     user_id: str,
#     limit: int = 20,
#     offset: int = 0,
#     db: Session = Depends(get_db),
#     api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
# ):
#     """
#     Get all session titles for a specific user
    
#     Args:
#         user_id: The user ID
#         limit: Maximum number of results
#         offset: Number of results to skip
        
#     Returns:
#         List of sessions with titles
#     """
#     # Query sessions for user
#     sessions = db.query(
#         ChatSessions.id,
#         ChatSessions.title,
#         ChatSessions.start_date,
#         ChatSessions.end_date
#     ).filter(
#         ChatSessions.user_id == user_id
#     ).order_by(
#         ChatSessions.start_date.desc()
#     ).limit(limit).offset(offset).all()
    
#     # Format results
#     results = []
#     for session in sessions:
#         results.append({
#             "session_id": str(session.id),
#             "title": session.title if session.title else "Untitled",
#             "start_date": session.start_date,
#             "end_date": session.end_date
#         })
    
#     return {
#         "user_id": user_id,
#         "sessions": results,
#         "total": len(results),
#         "limit": limit,
#         "offset": offset
#     }