from fastapi.routing import APIRouter
from fastapi import status, Response, Depends, Request, HTTPException, Query
from typing import Dict, Any, List, Optional

from src.agents.memory.memory_manager import get_vector_store
from src.handlers.api_key_authenticator_handler import APIKeyAuth
from src.schemas.response import BasicResponse
from src.database.services.collection_management import CollectionManagementService
from src.database.repository.document import FileProcessingRepository
from src.utils.constants import TypeDatabase


router = APIRouter(prefix="/collection")

# API key authentication instance
api_key_auth = APIKeyAuth()
file_management_dal = FileProcessingRepository()


@router.post('/create_collection', response_description='Create collection in Qdrant')
async def create_collection(
    request: Request,
    response: Response,
    collection_name: str,
    is_personal: bool = Query(False, description="Whether this is a personal collection"),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    try:
        # Get user_id and organization_id information from request state
        user_id = getattr(request.state, "user_id", None)
        organization_id = getattr(request.state, "organization_id", None)
        
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return BasicResponse(
                status="Failed",
                message="User authentication required",
                data=None
            )
        
        # If this is an organizational collection, check ADMIN rights
        if not is_personal and organization_id:
            user_role = getattr(request.state, "role", None)
            if user_role != "ADMIN":
                response.status_code = status.HTTP_403_FORBIDDEN
                return BasicResponse(
                    status="Failed",
                    message="Only administrators can create organizational collections",
                    data=None
                )
        
        # Create user object from authenticated information
        user = {
            "id": user_id,
            "role": getattr(request.state, "role", None)
        }
        
        vector_store = get_vector_store()
        resp = await vector_store.create_qdrant_collection(
            collection_name=collection_name,
            user=user,
            organization_id=organization_id if not is_personal else None,
            is_personal=is_personal
        )
        
        if resp.data:
            response.status_code = status.HTTP_200_OK
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
        
        return resp
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="Failed",
            message=f"Error creating collection: {str(e)}",
            data=None
        )


@router.delete("/{collection_name}", response_description="Delete entire collection and all related documents")
async def delete_collection_with_documents(
    collection_name: str,
    response: Response,
    request: Request,
    type_db: str = Query(
        default=TypeDatabase.Qdrant.value,
        enum=TypeDatabase.list(),
        description="Select vector database type"
    ),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    """
    Delete a collection along with all its documents in both PostgreSQL and Qdrant.
    This is useful for when a chat session is deleted and all its related documents should be removed.
    """
    organization_id = getattr(request.state, "organization_id", None)
    user_role = getattr(request.state, "role", None)
    user_id = getattr(request.state, "user_id", None)
    
    try:
        # Check access/delete collection
        collection_service = CollectionManagementService()
        has_permission = collection_service.check_collection_permission(
            user_id=user_id,
            collection_name=collection_name,
            organization_id=organization_id,
            required_permission="delete"
        )
        
        if not has_permission and user_role != "ADMIN":
            response.status_code = status.HTTP_403_FORBIDDEN
            return BasicResponse(
                status="failed",
                message="You don't have permission to delete this collection",
                data=None
            )
        
        # 1. Get a list of all documents in a collection
        documents = file_management_dal.get_files_by_collection(collection_name, organization_id)
        document_count = len(documents)
        
        # 2. Delete all documents in PostgreSQL
        file_management_dal.delete_record_by_collection(collection_name, organization_id)
        
        # 3. Delete collection in Qdrant (using singleton and async)
        vector_store = get_vector_store()
        result = await vector_store.delete_qdrant_collection(
            collection_name=collection_name,
            user={"id": user_id, "role": user_role},
            organization_id=organization_id,
            is_personal=(user_role != "ADMIN")  # Assume collection is personal if user is not admin
        )
        
        response.status_code = status.HTTP_200_OK
        return BasicResponse(
            status="success",
            message=f"Successfully deleted collection '{collection_name}' with {document_count} documents",
            data={"collection_name": collection_name, "document_count": document_count}
        )
        
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BasicResponse(
            status="error",
            message=f"Error deleting collection: {str(e)}",
            data=None
        )


@router.get('/list_collections', response_description='List all collections in Qdrant')
async def list_collections(
    request: Request,
    response: Response,
    include_personal: bool = Query(True, description="Include personal collections"),
    include_organizational: bool = Query(True, description="Include organizational collections"),
    api_key_data: Dict[str, Any] = Depends(api_key_auth.author_with_api_key)
):
    try:
        # Get user_id and organization_id information from request state
        user_id = getattr(request.state, "user_id", None)
        organization_id = getattr(request.state, "organization_id", None)
        user_role = getattr(request.state, "role", None)
        
        if not user_id:
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"status": "Failed", "message": "User authentication required", "data": None}
        
        # Create user object from authenticated information
        user = {
            "id": user_id,
            "role": user_role,
            "is_admin": user_role == "ADMIN"
        }
        
        # Get collection list with filtering by organization_id (using singleton)
        try:
            vector_store = get_vector_store()
            collections = await vector_store.list_qdrant_collections(
                user=user,
                organization_id=organization_id,
                include_personal=include_personal,
                include_organizational=include_organizational
            )
            
            personal_collections = []
            org_collections = []
            
            for collection in collections:
                if collection.get("is_personal"):
                    personal_collections.append(collection)
                else:
                    org_collections.append(collection)
            
            result = {
                "personal_collections": personal_collections if include_personal else [],
                "organizational_collections": org_collections if include_organizational else []
            }
            
            response.status_code = status.HTTP_200_OK
            return {"status": "Success", "message": "List collections success", "data": result}
        except Exception as e:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"status": "Failed", "message": str(e), "data": None}
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "Failed", "message": f"Error listing collections: {str(e)}", "data": None}