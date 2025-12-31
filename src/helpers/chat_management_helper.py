"""
Chat Management Helper

PRODUCTION NOTES:
- Use get_chat_service() singleton to avoid memory leaks
- Use async methods (get_chat_history_async, etc.) for non-blocking operations
- Sync methods kept for backward compatibility but should be migrated
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from src.utils.logger.custom_logging import LoggerMixin
from src.database.repository.chat import ChatRepository
from src.database.models.schemas import ChatSessions
from src.database import get_postgres_db


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================
_chat_service_instance: Optional['ChatService'] = None

# Thread pool for running sync DB operations without blocking event loop
_chat_db_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="chat_service_")


def get_chat_service() -> 'ChatService':
    """
    Get singleton instance of ChatService.

    Use this instead of ChatService() to prevent memory leaks
    when handling thousands of concurrent requests.

    Returns:
        ChatService singleton instance
    """
    global _chat_service_instance

    if _chat_service_instance is None:
        _chat_service_instance = ChatService()

    return _chat_service_instance


class ChatService(LoggerMixin):
    """
    Chat management service for handling chat sessions and messages.

    IMPORTANT: Use get_chat_service() singleton instead of direct instantiation.

    For async contexts, use the async methods:
    - get_chat_history_async()
    - create_chat_session_async()
    - save_user_question_async()
    - save_assistant_response_async()
    """

    def __init__(self):
        super().__init__()
        self.chat_repo = ChatRepository()
        self.db = get_postgres_db()
        self._executor = _chat_db_executor  # Shared thread pool

    def create_chat_session(self, user_id: str, organization_id: Optional[str] = None) -> str:
        """
        Create a new chat session for a user
        
        Args:
            user_id: The ID of the user creating the chat session
            organization_id: The ID of the organization (optional)
            
        Returns:
            str: The generated session ID
        """
        try:
            
            with self.db.session_scope() as session:
                # Generate a unique session ID
                session_id = str(uuid.uuid4())
                
                # Create new session
                new_chat_session = ChatSessions(
                    id=session_id,
                    user_id=user_id,
                    organization_id=organization_id,  # save organization_id
                    start_date=datetime.now(),
                    title="New Chat",
                    state=1  # Active state
                )
                
                session.add(new_chat_session)
                
            self.logger.info(f"Created new chat session with ID: {session_id} for user: {user_id}, organization: {organization_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create chat session for user: {user_id}. Error: {str(e)}")
            raise
   
            
    def save_user_question(self, session_id: str, created_at: datetime, created_by: str, content: str) -> str:
        """
        Save a user's question in the database
        
        Args:
            session_id: The ID of the chat session
            created_at: When the question was created
            created_by: Who created the question
            content: The question text
            
        Returns:
            str: The ID of the saved question
        """
        try:
            from src.database.models.schemas import Messages
            from src.utils.constants import MessageType
            
            if not self.is_session_exist(session_id):
                self.logger.error("Chat session does not exist!")
                raise ValueError("Chat session does not exist")
                
            with self.db.session_scope() as session:
                question_id = str(uuid.uuid4())
                
                message = Messages(
                    id=question_id,
                    created_at=created_at,
                    created_by=created_by,
                    content=content,
                    type=MessageType.QUESTION,
                    session_id=session_id,
                    sender_role='user'
                )
                
                session.add(message)
                
            self.logger.info(f"Saved user question in session {session_id}")
            return question_id
        except Exception as e:
            self.logger.error(f"Failed to save user question in session {session_id}. Error: {str(e)}")
            raise

    def save_assistant_response(self, session_id: str, created_at: datetime, question_id: str, 
                            content: str, response_time: float) -> str:
        """
        Save the assistant's response in the database
        
        Args:
            session_id: The ID of the chat session
            created_at: When the response was created
            question_id: The ID of the question being answered
            content: The response text
            response_time: How long it took to generate the response
            
        Returns:
            str: The ID of the saved response
        """
        try:
            from src.database.models.schemas import Messages
            from src.utils.constants import MessageType
            
            if not self.is_session_exist(session_id):
                self.logger.error("Chat session does not exist!")
                raise ValueError("Chat session does not exist")
                
            with self.db.session_scope() as session:
                message_id = str(uuid.uuid4())
                
                message = Messages(
                    id=message_id,
                    created_at=created_at,
                    content=content,
                    type=MessageType.ANSWER,
                    question_id=question_id,
                    session_id=session_id,
                    sender_role='assistant',
                    response_time=response_time
                )
                
                session.add(message)
                
            self.logger.info(f"Saved assistant response in session {session_id}")
            return message_id
        except Exception as e:
            self.logger.error(f"Failed to save assistant response in session {session_id}. Error: {str(e)}")
            raise

    def is_session_exist(self, session_id: str) -> bool:
        """
        Check if a chat session exists
        
        Args:
            session_id: The ID of the session to check
            
        Returns:
            bool: True if the session exists, False otherwise
        """
        try:
            from src.database.models.schemas import ChatSessions
            
            with self.db.session_scope() as session:
                exists = session.query(session.query(ChatSessions).filter(
                    ChatSessions.id == session_id
                ).exists()).scalar()
                return exists
        except Exception as e:
            self.logger.error(f"Error checking if session exists: {str(e)}")
            return False
        
    def update_assistant_response(self, updated_at: datetime, message_id: str, 
                                content: str, response_time: float) -> None:
        """
        Update an assistant's response in the database
        
        Args:
            updated_at: When the response was updated
            message_id: The ID of the message being updated
            content: The updated response text
            response_time: The updated response time
        """
        try:
            self.chat_repo.update_assistant_response(
                updated_at=updated_at,
                message_id=message_id,
                content=content,
                response_time=response_time
            )
            self.logger.info(f"Updated assistant response with ID {message_id}")
        except Exception as e:
            self.logger.error(f"Failed to update assistant response with ID {message_id}. Error: {str(e)}")
            raise
    
    def get_chat_history(self, session_id: str, limit: int = 5) -> List[Tuple[str, str]]:
        """
        Get the chat history for a session
        
        Args:
            session_id: The ID of the chat session
            limit: Maximum number of messages to retrieve
            
        Returns:
            List[Tuple[str, str]]: List of tuples containing (content, sender_role)
        """
        try:
            history = self.chat_repo.get_chat_message_history_by_session_id(
                session_id=session_id,
                limit=limit
            )
            self.logger.info(f"Retrieved chat history for session {session_id}")
            return history
        except Exception as e:
            self.logger.error(f"Failed to retrieve chat history for session {session_id}. Error: {str(e)}")
            raise

    def delete_chat_history(self, session_id: str) -> None:
        """
        Delete the chat history for a session
        
        Args:
            session_id: The ID of the chat session to delete
        """
        try:
            # Delete chat history directly using SQLAlchemy
            from src.database.models.schemas import Messages, ChatSessions, ReferenceDocs
            
            with self.db.session_scope() as session:
                # 1. First, get all message IDs in this session
                message_ids = session.query(Messages.id).filter(
                    Messages.session_id == session_id
                ).all()
                
                message_ids = [str(mid[0]) for mid in message_ids]
                
                # 2. Delete all references from the reference_docs table that point to these messages
                if message_ids:
                    self.logger.info(f"Deleting references for {len(message_ids)} messages in session {session_id}")
                    session.query(ReferenceDocs).filter(
                        ReferenceDocs.message_id.in_(message_ids)
                    ).delete(synchronize_session=False)
                
                # 3. Then delete the messages
                message_count = session.query(Messages).filter(
                    Messages.session_id == session_id
                ).delete()
                
                # 4. Finally delete the session
                session.query(ChatSessions).filter(
                    ChatSessions.id == session_id
                ).delete()
                    
                self.logger.info(f"Deleted chat history for session {session_id} ({message_count} messages)")
        except Exception as e:
            self.logger.error(f"Failed to delete chat history for session {session_id}. Error: {str(e)}")
            raise
    
    
    async def delete_chat_session_completely(
        self,
        session_id: str,
        delete_documents: bool = False,
        delete_collections: bool = False,
        organization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Delete a chat session completely, optionally deleting related documents and collections
        
        Args:
            session_id: The ID of the chat session to delete
            delete_documents: Whether to delete documents referenced in the session
            delete_collections: Whether to delete collections containing the documents
            organization_id: Optional organization ID for permission check
            
        Returns:
            Dict[str, Any]: Information about what was deleted
        """
        try:
            # Check if session exists
            if not self.is_session_exist(session_id):
                self.logger.error(f"Chat session {session_id} does not exist")
                return {
                    "status": "failed",
                    "message": "Chat session does not exist",
                    "deleted_items": None
                }
            
            deleted_items = {
                "session": session_id,
                "documents": [],
                "collections": [],
                "memory_collections": [] 
            }
            
            # Find related documents and collections
            collections_docs = {}
            memory_collections_to_delete = [] 
            
            if delete_documents or delete_collections:
                with self.db.session_scope() as session:
                    from src.database.models.schemas import Messages, ReferenceDocs, Documents, ChatSessions
                    
                    # Get session info to retrieve user_id (needed for memory collections)
                    session_info = session.query(ChatSessions).filter(
                        ChatSessions.id == session_id
                    ).first()
                    
                    user_id = session_info.user_id if session_info else None
                    
                    # Get all message IDs in this session
                    message_ids = session.query(Messages.id).filter(
                        Messages.session_id == session_id
                    ).all()
                    
                    if message_ids:
                        message_ids = [str(mid[0]) for mid in message_ids]
                        
                        # Get related document IDs and collection names
                        document_query = session.query(
                            ReferenceDocs.document_id, 
                            Documents.collection_name
                        ).join(
                            Documents,
                            ReferenceDocs.document_id == Documents.id
                        ).filter(
                            ReferenceDocs.message_id.in_(message_ids)
                        ).distinct()
                        
                        results = document_query.all()
                        
                        # Group documents by collection
                        for doc_id, coll_name in results:
                            if coll_name not in collections_docs:
                                collections_docs[coll_name] = []
                            collections_docs[coll_name].append(str(doc_id))
                            
                            # If we have a collection name, also mark related memory collections for deletion
                            if delete_collections and coll_name:
                                # Add short and long memory collections based on this collection
                                memory_collections_to_delete.append(f"short_{coll_name}")
                                memory_collections_to_delete.append(f"long_{coll_name}")
                    
                    # Also try to find memory collections using hash-based naming (fallback)
                    if delete_collections and user_id:
                        try:
                            import hashlib
                            # Generate fallback memory collection names
                            for memory_type in ["short", "long"]:
                                hash_input = f"{user_id}_{session_id}_{memory_type}"
                                hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
                                fallback_name = f"memory_{memory_type}_{hash_suffix}"
                                memory_collections_to_delete.append(fallback_name)
                        except Exception as e:
                            self.logger.warning(f"Could not generate fallback memory collection names: {e}")
            
            # Delete chat history (messages, references, session)
            self.delete_chat_history(session_id)
            
            # Delete memory collections if needed
            if delete_collections and memory_collections_to_delete:
                try:
                    from src.helpers.qdrant_connection_helper import get_qdrant_connection
                    qdrant_conn = get_qdrant_connection()

                    # Get list of existing collections (async to avoid blocking)
                    existing_collections = await qdrant_conn.get_collections_async()
                    existing_collection_names = [col.name for col in existing_collections.collections]

                    # Delete each memory collection if it exists
                    for memory_collection in set(memory_collections_to_delete):  # Use set to avoid duplicates
                        if memory_collection in existing_collection_names:
                            try:
                                await qdrant_conn.delete_collection_async(collection_name=memory_collection)
                                deleted_items["memory_collections"].append(memory_collection)
                                self.logger.info(f"Deleted memory collection: {memory_collection}")
                            except Exception as e:
                                self.logger.error(f"Failed to delete memory collection {memory_collection}: {e}")
                except Exception as e:
                    self.logger.error(f"Error deleting memory collections: {e}")
            
            return {
                "status": "success",
                "message": "Chat session deleted successfully",
                "deleted_items": deleted_items,
                "collections_docs": collections_docs
            }
        except Exception as e:
            self.logger.error(f"Failed to delete chat session {session_id} completely: {str(e)}")
            return {
                "status": "failed",
                "message": f"Failed to delete chat session: {str(e)}",
                "deleted_items": None
            }
    
    # def delete_chat_session_completely(self, 
    #                                    session_id: str, 
    #                                    delete_documents: bool = False, 
    #                                    delete_collections: bool = False, 
    #                                    organization_id: Optional[str] = None
    #                                    ) -> Dict[str, Any]:
    #     """
    #     Delete a chat session completely, optionally deleting related documents and collections
        
    #     Args:
    #         session_id: The ID of the chat session to delete
    #         delete_documents: Whether to delete documents referenced in the session
    #         delete_collections: Whether to delete collections containing the documents
    #         organization_id: Optional organization ID for permission check
            
    #     Returns:
    #         Dict[str, Any]: Information about what was deleted
    #     """
    #     try:
    #         # Check if session exists
    #         if not self.is_session_exist(session_id):
    #             self.logger.error(f"Chat session {session_id} does not exist")
    #             return {
    #                 "status": "failed",
    #                 "message": "Chat session does not exist",
    #                 "deleted_items": None
    #             }
            
    #         deleted_items = {
    #             "session": session_id,
    #             "documents": [],
    #             "collections": []
    #         }
            
    #         # Find related documents and collections
    #         collections_docs = {}
            
    #         if delete_documents or delete_collections:
    #             with self.db.session_scope() as session:
    #                 from src.database.models.schemas import Messages, ReferenceDocs, Documents
                    
    #                 # Get all message IDs in this session
    #                 message_ids = session.query(Messages.id).filter(
    #                     Messages.session_id == session_id
    #                 ).all()
                    
    #                 if message_ids:
    #                     message_ids = [str(mid[0]) for mid in message_ids]
                        
    #                     # Get related document IDs and collection names
    #                     document_query = session.query(
    #                         ReferenceDocs.document_id, 
    #                         Documents.collection_name
    #                     ).join(
    #                         Documents,
    #                         ReferenceDocs.document_id == Documents.id
    #                     ).filter(
    #                         ReferenceDocs.message_id.in_(message_ids)
    #                     ).distinct()
                        
    #                     results = document_query.all()
                        
    #                     # Group documents by collection
    #                     for doc_id, coll_name in results:
    #                         if coll_name not in collections_docs:
    #                             collections_docs[coll_name] = []
    #                         collections_docs[coll_name].append(str(doc_id))
            
    #         # Delete chat history (messages, references, session)
    #         self.delete_chat_history(session_id)
            
    #         return {
    #             "status": "success",
    #             "message": "Chat session deleted successfully",
    #             "deleted_items": deleted_items,
    #             "collections_docs": collections_docs
    #         }
    #     except Exception as e:
    #         self.logger.error(f"Failed to delete chat session {session_id} completely: {str(e)}")
    #         return {
    #             "status": "failed",
    #             "message": f"Failed to delete chat session: {str(e)}",
    #             "deleted_items": None
    #         }
        

    def get_pageable_chat_history(self, session_id: str, page: int = 1, 
                             size: int = 10, sort: str = 'DESC') -> List[Dict[str, Any]]:
        """
        Get paginated chat history for a session
        
        Args:
            session_id: The ID of the chat session
            page: Page number (1-based)
            size: Number of items per page
            sort: Sort order ('ASC' or 'DESC')
            
        Returns:
            List[Dict[str, Any]]: List of message dictionaries
        """
        try:
            from src.database.models.schemas import Messages
            
            with self.db.session_scope() as session:
                # Calculate offset
                offset = (page - 1) * size
                
                # Build query
                query = session.query(
                    Messages.id,
                    Messages.session_id,
                    Messages.content,
                    Messages.sender_role,
                    Messages.created_at
                ).filter(
                    Messages.session_id == session_id
                )
                
                # Apply sorting
                if sort.upper() == 'DESC':
                    query = query.order_by(Messages.created_at.desc())
                else:
                    query = query.order_by(Messages.created_at.asc())
                
                # Apply pagination
                query = query.offset(offset).limit(size)
                
                # Execute and format results
                results = query.all()
                return [dict(zip(('id', 'session_id', 'content', 'sender_role', 'created_at'), r)) for r in results]
        except Exception as e:
            self.logger.error(f"Error getting pageable chat history: {str(e)}")
            raise ValueError(str(e))

    def save_reference_docs(self, message_id: str, document_id: str, page: int) -> None:
        """
        Save reference document for a message with duplicate check
        
        Args:
            message_id: The ID of the message
            document_id: The ID of the document
            page: The page number
        """
        try:
            from src.database.models.schemas import ReferenceDocs, Documents

            # Use a single transaction to check and add
            with self.db.session_scope() as session:
                # First check if reference already exists
                reference_exists = session.query(session.query(ReferenceDocs).filter(
                    ReferenceDocs.message_id == message_id,
                    ReferenceDocs.document_id == document_id
                ).exists()).scalar()
                
                if reference_exists:
                    self.logger.debug(f"Reference for document {document_id} and message {message_id} already exists")
                    return None  # Return early, reference already exists
                
                # Check if document exists
                document_exists = session.query(session.query(Documents).filter(
                    Documents.id == document_id
                ).exists()).scalar()
                
                if not document_exists:
                    self.logger.warning(f"Document {document_id} not found in database, skipping reference")
                    return None
                
                # Add the reference only if both checks pass
                ref_doc = ReferenceDocs(
                    message_id=message_id,
                    document_id=document_id,
                    page=page
                )
                
                session.add(ref_doc)
                self.logger.info(f"Saved reference document {document_id} for message {message_id}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to save reference document {document_id} for message {message_id}. Error: {str(e)}")
            return None
    
   
    def get_sources_by_message(self, message_id: str) -> List[Dict[str, Any]]:
        """
        Get the sources referenced by a message
        
        Args:
            message_id: The ID of the message
            
        Returns:
            List[Dict[str, Any]]: List of source dictionaries
        """
        try:
            from src.database.models.schemas import ReferenceDocs, Documents
            from src.utils.utils import extension_mapping
            
            with self.db.session_scope() as session:
                # Join query to get reference documents and their document info
                results = session.query(
                    ReferenceDocs.message_id,
                    ReferenceDocs.document_id,
                    Documents.file_name,
                    Documents.miniourl,
                    ReferenceDocs.page
                ).join(
                    Documents,
                    ReferenceDocs.document_id == Documents.id
                ).filter(
                    ReferenceDocs.message_id == message_id
                ).order_by(
                    ReferenceDocs.document_id
                ).all()
                
                # Convert to dictionaries
                result_dict = [dict(zip(('message_id', 'document_id', 'file_name', 'miniourl', 'page'), r)) for r in results]
                
                # Add extension info
                for doc in result_dict:
                    file_extension = doc.get("file_name", "").split(".")[-1].lower()
                    doc["extension"] = extension_mapping.get(file_extension, file_extension)
                
                return result_dict
        except Exception as e:
            self.logger.error(f"Error getting sources by message: {str(e)}")
            return []

    # =========================================================================
    # ASYNC WRAPPER METHODS (NON-BLOCKING)
    # Use these in async contexts to avoid blocking the event loop
    # =========================================================================

    async def get_chat_history_async(
        self,
        session_id: str,
        limit: int = 5
    ) -> List[Tuple[str, str]]:
        """
        Async version of get_chat_history.

        Runs the sync database operation in a thread pool to avoid
        blocking the event loop. Safe for production with high concurrency.

        Args:
            session_id: The ID of the chat session
            limit: Maximum number of messages to retrieve

        Returns:
            List of (content, sender_role) tuples
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.get_chat_history(session_id, limit)
        )

    async def create_chat_session_async(
        self,
        user_id: str,
        organization_id: Optional[str] = None
    ) -> str:
        """
        Async version of create_chat_session.

        Args:
            user_id: The ID of the user
            organization_id: Optional organization ID

        Returns:
            The generated session ID
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.create_chat_session(user_id, organization_id)
        )

    async def save_user_question_async(
        self,
        session_id: str,
        created_at: datetime,
        created_by: str,
        content: str
    ) -> str:
        """
        Async version of save_user_question.

        Args:
            session_id: The ID of the chat session
            created_at: When the question was created
            created_by: Who created the question
            content: The question text

        Returns:
            The ID of the saved question
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.save_user_question(session_id, created_at, created_by, content)
        )

    async def save_assistant_response_async(
        self,
        session_id: str,
        created_at: datetime,
        question_id: str,
        content: str,
        response_time: float
    ) -> str:
        """
        Async version of save_assistant_response.

        Args:
            session_id: The ID of the chat session
            created_at: When the response was created
            question_id: The ID of the question being answered
            content: The response text
            response_time: How long it took to generate

        Returns:
            The ID of the saved response
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.save_assistant_response(
                session_id, created_at, question_id, content, response_time
            )
        )

    async def is_session_exist_async(self, session_id: str) -> bool:
        """
        Async version of is_session_exist.

        Args:
            session_id: The ID of the session to check

        Returns:
            True if the session exists
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.is_session_exist,
            session_id
        )

    async def delete_chat_history_async(self, session_id: str) -> None:
        """
        Async version of delete_chat_history.

        Args:
            session_id: The ID of the chat session to delete
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.delete_chat_history,
            session_id
        )

    async def get_pageable_chat_history_async(
        self,
        session_id: str,
        page: int = 1,
        size: int = 10,
        sort: str = 'DESC'
    ) -> List[Dict[str, Any]]:
        """
        Async version of get_pageable_chat_history.

        Args:
            session_id: The ID of the chat session
            page: Page number (1-based)
            size: Number of items per page
            sort: Sort order ('ASC' or 'DESC')

        Returns:
            List of message dictionaries
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self.get_pageable_chat_history(session_id, page, size, sort)
        )