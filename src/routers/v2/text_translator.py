from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field, validator
from typing import Optional, List

from src.translation_agent.handlers.translation_service import TranslationService
from src.translation_agent.models.translation_models import TranslationRequest
from src.providers.provider_factory import ProviderType

router = APIRouter()

class TranslateAPIRequest(BaseModel):
    # Text & Languages
    source_text: str = Field(..., min_length=1)
    source_lang: str = Field(..., min_length=2)
    target_lang: str = Field(..., min_length=2)
    
    # Provider & Model
    provider_type: str = Field(default="openai", description="Provider: openai, gemini, ollama")
    model_name: str = Field(default="gpt-4.1-nano", description="Model name")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    
    # Context
    context: Optional[str] = None
    country: Optional[str] = None
    emotions: Optional[List[str]] = None
    
    # Options
    enable_reflection: bool = True
    max_tokens_per_chunk: int = Field(1000, ge=100, le=4000)
    

@router.post("/translate")
async def translate_text(request: TranslateAPIRequest = Body(...)):
    try:
        service = TranslationService()
        
        translation_request = TranslationRequest(
            source_text=request.source_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            context=request.context,
            country=request.country,
            emotions=request.emotions,
            enable_reflection=request.enable_reflection,
            max_tokens_per_chunk=request.max_tokens_per_chunk
        )
        
        response = await service.translate(
            request=translation_request,
            provider_type=request.provider_type,
            model_name=request.model_name,
            temperature=request.temperature
        )
        
        result = response.to_dict()
        result["provider_info"] = {
            "provider": request.provider_type,
            "model": request.model_name,
            "temperature": request.temperature
        }
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
