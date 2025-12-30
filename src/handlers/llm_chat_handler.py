import time
import json
import datetime
from operator import itemgetter
from collections.abc import AsyncGenerator
from typing import Optional, Dict, List, Any, Tuple
from pydantic import Field, BaseModel

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

from src.utils.logger.custom_logging import LoggerMixin
from src.schemas.response import BasicResponse, BasicResponseDelete

from src.database.models.schemas import ChatSessions
from src.database.repository.document import FileProcessingRepository

from src.handlers.multi_collection_retrieval_handler import multi_collection_retriever

from src.helpers.llm_helper import LLMGenerator, LLMGeneratorProvider
from src.helpers.chat_management_helper import ChatService
from src.helpers.qdrant_connection_helper import get_qdrant_connection
from src.helpers.prompt_template_helper import ContextualizeQuestionHistoryTemplate, QuestionAnswerTemplate
from src.agents.memory.memory_manager import get_memory_manager, get_retrieval, get_vector_store
from src.providers.provider_factory import ProviderType, ModelProviderFactory
from src.utils.config import settings
from src.helpers.language_detector import language_detector, DetectionMethod


# Initialize the chat service
chat_service = ChatService()

class ConversationAnalysis(BaseModel):
    importance_score: float = Field(ge=0.0, le=1.0, description="Importance score from 0.0 to 1.0")

class ChatHandler(LoggerMixin):
    def __init__(self) -> None:
        super().__init__()
        self.search_retrieval = get_retrieval()
        self.llm_generator = LLMGenerator()
        self.llm_generator_provider = LLMGeneratorProvider()
        self.memory_manager = get_memory_manager()
    
    def create_session_id(self, user_id: str, organization_id: Optional[str] = None) -> BasicResponse:
        try:
            session_id = chat_service.create_chat_session(
                user_id=user_id,
                organization_id=organization_id
            )
            self.logger.info(f"Created new chat session with ID: {session_id}")
            
            return BasicResponse(
                status="Success",
                message="Session created successfully",
                data=session_id
            )
        except Exception as e:
            self.logger.error(f"Failed to create session: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Failed to create session: {str(e)}",
                data=None
            )
        
    async def delete_session_completely(
        self, 
        session_id: str, 
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        delete_documents: bool = False,
        delete_collections: bool = False
    ) -> BasicResponseDelete:
        """
        Delete a chat session completely with options to delete related documents and collections
        
        Args:
            session_id: The ID of the chat session to delete
            user_id: The ID of the user (for permissions)
            organization_id: The ID of the organization (for permissions)
            delete_documents: Whether to delete related documents
            delete_collections: Whether to delete related collections (including memory collections)
            
        Returns:
            BasicResponseDelete: Response indicating success or failure
        """
        try:
            # Delete session and get info about related resources (async to avoid blocking)
            result = await chat_service.delete_chat_session_completely(
                session_id=session_id,
                delete_documents=delete_documents,
                delete_collections=delete_collections,
                organization_id=organization_id
            )
            
            if result["status"] != "success":
                return BasicResponseDelete(
                    Status="Failed",
                    Message=result["message"],
                    Data=None
                )
            
            deleted_items = result["deleted_items"]
            collections_docs = result.get("collections_docs", {})
            
            # Handle document deletion if needed
            if delete_documents and collections_docs:
                file_management = FileProcessingRepository()
                
                for collection_name, doc_ids in collections_docs.items():
                    # Delete documents from PostgreSQL
                    for doc_id in doc_ids:
                        file_management.delete_file_record(doc_id, organization_id)
                        deleted_items["documents"].append(doc_id)
                    
                    # Delete documents from vector store (use singleton to avoid blocking)
                    qdrant_client = get_qdrant_connection()

                    await qdrant_client.delete_document_by_batch_ids(
                        document_ids=doc_ids,
                        collection_name=collection_name,
                        organization_id=organization_id
                    )
            
            # Handle collection deletion if needed (including memory collections)
            if delete_collections:
                vector_store = get_vector_store()
                
                # Delete document collections
                if collections_docs:
                    for collection_name in collections_docs.keys():
                        result = await vector_store.delete_qdrant_collection(
                            collection_name=collection_name,
                            user={"id": user_id},
                            organization_id=organization_id,
                            is_personal=(organization_id is None)
                        )
                        if result.status == "Success":
                            deleted_items["collections"].append(collection_name)
                
                # Memory collections are already handled in delete_chat_session_completely
                # Just log the results
                if deleted_items.get("memory_collections"):
                    self.logger.info(f"Deleted {len(deleted_items['memory_collections'])} memory collections")
            
            # Prepare response data
            response_data = {
                "session_id": deleted_items["session"],
                "documents_deleted": len(deleted_items.get("documents", [])),
                "collections_deleted": deleted_items.get("collections", []),
                "memory_collections_deleted": deleted_items.get("memory_collections", [])
            }
            
            return BasicResponseDelete(
                Status="Success",
                Message=f"Chat session deleted successfully. Deleted {len(deleted_items.get('memory_collections', []))} memory collections.",
                Data=response_data
            )
                
        except Exception as e:
            self.logger.error(f"Failed to delete chat session completely: {str(e)}")
            return BasicResponseDelete(
                Status="Failed",
                Message=f"Failed to delete chat session: {str(e)}",
                Data=None
            )

    async def _get_chat_flow(self, model_name: str, collection_name: str, user_id: str = None, organization_id: str = None, use_multi_collection: bool = False) -> Tuple[Runnable, Runnable]:
        """
        Create the chat flow for retrieving context and generating responses
        
        Args:
            model_name: The name of the LLM model to use
            collection_name: The name of the vector collection to query
            user_id: User ID for multi-collection access (optional)
            organization_id: Organization ID for multi-collection access (optional)
            use_multi_collection: Whether to use both personal and organizational collections
            
        Returns:
            Tuple[Runnable, Runnable]: The conversation chain and rewrite chain
        """
        # Get the language model
        llm = await self.llm_generator.get_llm(model=model_name)
        
        # Chain for rewriting the question based on conversation history
        rewrite_prompt = ContextualizeQuestionHistoryTemplate
        rewrite_chain = (rewrite_prompt | llm | StrOutputParser()).with_config(run_name='rewrite_chain')

        # Define the retrieval function
        async def retriever_function(query):
            if use_multi_collection and user_id:
                # Use multi-collection retriever if required
                return await multi_collection_retriever.retrieve_from_collections(
                    query=query, 
                    user_id=user_id,
                    organization_id=organization_id,
                    top_k=5
                )
            else:
                # Use a regular retriever
                return await self.search_retrieval.qdrant_retrieval(
                    query=query, 
                    collection_name=collection_name
                )
        
        # Format documents function
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Main conversation chain that combines the rewritten query, context, and generates a response
        chain = (
            {
                "context": itemgetter("rewrite_input") | RunnableLambda(retriever_function).with_config(run_name='stage_retrieval') | format_docs,
                "input": itemgetter("input")
            }
            | QuestionAnswerTemplate
            | llm
            | StrOutputParser()
        ).with_config(run_name='conversational_rag')

        return chain, rewrite_chain


    def _get_api_key(self, provider_type: str) -> Optional[str]:
        """
        Get API key for a provider from environment variables.
        
        Args:
            provider_type: Provider type (ollama, openai, gemini)
            
        Returns:
            Optional[str]: API key for the provider
        """
        if provider_type == ProviderType.OPENAI:
            return settings.OPENAI_API_KEY
        elif provider_type == ProviderType.GEMINI:
            return settings.GEMINI_API_KEY
        elif provider_type == ProviderType.OLLAMA:
            return settings.OLLAMA_ENDPOINT
        
        return None

    def _convert_history_to_messages(self, chat_history: str) -> List[Dict[str, str]]:
        """
        Convert chat history to messages format.
        
        Args:
            chat_history: Chat history in string format
            
        Returns:
            List[Dict[str, str]]: List of messages in OpenAI format
        """
        messages = []
        
        if not chat_history:
            return messages
        
        lines = chat_history.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue
                
            if " - user: " in line:
                content = line.split(" - user: ", 1)[1]
                messages.append({"role": "user", "content": content})
            elif " - assistant: " in line:
                content = line.split(" - assistant: ", 1)[1]
                messages.append({"role": "assistant", "content": content})
        
        return messages

    
    async def analyze_conversation_importance(self,
        query: str,
        response: str,
        llm_provider: LLMGeneratorProvider,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Use LLM to analyze conversation importance and extract metadata for financial chatbot
        Supports all languages naturally
        """
        try:
            analysis_prompt = f"""Analyze this financial conversation and provide importance score (0.0-1.0):

    User Query: {query}
    Assistant Response: {response[:1000]}...

    Score based on:
    - Financial relevance and complexity
    - Actionable insights provided
    - Educational/strategic value
    - Whether it's follow-up or references context

    Focus on: trading, investment, market analysis, portfolio, risk management."""

            messages = [
                {
                    "role": "system", 
                    "content": "You are a financial conversation analyst."
                },
                {
                    "role": "user", 
                    "content": analysis_prompt
                }
            ]
            
            # Get API key
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            # Prepare structured output format for OpenAI
            structured_format = None
            if provider_type == ProviderType.OPENAI:
                structured_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "importance_analysis",
                        "schema": ConversationAnalysis.model_json_schema(),
                        "strict": False 
                    }
                }
            
            # Call LLM with structured output
            llm_response = await llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                temperature=0.1,
                response_format=structured_format
            )
            
            # Parse and validate response using Pydantic
            content = llm_response.get("content", "{}")
            parsed_data = json.loads(content)
            analysis = ConversationAnalysis.model_validate(parsed_data)
            
            return analysis.importance_score
            
        except Exception as e:
            self.logger.error(f"Error analyzing conversation importance with LLM: {e}")
            return 0.0

    # API /chat/provider/reasoning/stream
    def _format_reply_as_user_message(self, reply_to_text: str, question_input: str) -> str:
        """
        Format quoted text and question as a single contextual message
        
        This combines the quoted reference with the follow-up question in a way
        that makes the relationship explicit to the LLM.
        
        Args:
            reply_to_text: Text that user quoted to reply
            question_input: User's follow-up question
            
        Returns:
            Formatted message that shows question is about quoted content
        """
        return f"""[Referring to this content]
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    {reply_to_text}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    [User's follow-up question about the above content]
    {question_input}"""

    async def handle_chat_provider_reasoning_reply_text_stream(
        self,
        session_id: str,
        question_input: str,
        system_language: str,
        model_name: str,
        collection_name: str,
        provider_type: str,
        user_id: str = None,
        organization_id: str = None,
        use_multi_collection: bool = False,
        clean_thinking: bool = True,
        enable_thinking: bool = True,
        use_memory_context: bool = True,
        reply_to_text: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Handle a streaming chat request using ReAct+CoT via SSE
        """
        try:
            # Get API Key
            api_key = self._get_api_key(provider_type)

            # ============= Handle Reply Context =============
            # Determine if this is a reply conversation
            is_reply_conversation = bool(reply_to_text and reply_to_text.strip())
            
            # Prepare the query for LLM inference
            # - If reply: merge quoted text with question
            # - If not reply: use original question
            llm_query = (
                self._format_reply_as_user_message(reply_to_text, question_input)
                if is_reply_conversation
                else question_input
            )
            
            # Prepare query for retrieval (memory + document search)
            # - For reply conversations: use merged query for better context matching
            # - For normal conversations: use original question
            retrieval_query = llm_query if is_reply_conversation else question_input
            
            if is_reply_conversation:
                self.logger.info(f"Reply conversation detected. Quoted text length: {len(reply_to_text)}")
            # ============= End Reply Context Handling =============

            # Step 1: Retrieve memory context before inference
            memory_context = ""
            memory_stats = {}
            document_references = []
            
            if session_id and user_id and use_memory_context:
                try:
                    memory_context, memory_stats, document_references = await self.memory_manager.get_relevant_context(
                        session_id=session_id,
                        user_id=user_id,
                        current_query=retrieval_query,  # Use appropriate query based on reply status
                        llm_provider=self.llm_generator_provider,
                        max_short_term=5,
                        max_long_term=3,
                        base_collection=collection_name
                    )
                    
                    self.logger.info(f"Retrieved memory context for session {session_id}: {memory_stats}")
                except Exception as e:
                    self.logger.error(f"Error getting memory context: {e}")

            # Step 2: Get document context
            document_context = ""
            context_docs = []
            # Use singleton to avoid creating new connections that block event loop
            qdrant_client = get_qdrant_connection()

            if collection_name:
                try:
                    # Use async wrapper to avoid blocking event loop
                    if await qdrant_client.collection_exists_async(collection_name):
                        context_docs = await self.search_retrieval.qdrant_retrieval(
                            query=retrieval_query,  # Use appropriate query based on reply status
                            collection_name=collection_name,
                            top_k=5
                        )
                        
                        if context_docs:
                            document_context = "\n\n".join(doc.page_content for doc in context_docs)
                            self.logger.info(f"Found {len(context_docs)} documents from {collection_name}")
                    else:
                        self.logger.info(f"Collection {collection_name} does not exist, skip document search")
                        
                except Exception as e:
                    self.logger.error(f"Error searching documents: {e}")
            
            # Step 3: Combine contexts
            combined_context = ""
            
            if memory_context and document_context:
                combined_context = (
                    "=== CONVERSATION HISTORY ===\n"
                    f"{memory_context}\n\n"
                    "=== DOCUMENT KNOWLEDGE ===\n"
                    f"{document_context}"
                )
            elif memory_context:
                combined_context = f"=== CONVERSATION HISTORY ===\n{memory_context}"
            elif document_context:
                combined_context = f"=== DOCUMENT KNOWLEDGE ===\n{document_context}"
        
            # Step 4: Save the user's question to the database
            # ALWAYS save the ORIGINAL question_input for UI display
            question_id = chat_service.save_user_question(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                created_by=user_id if user_id else "user",
                content=question_input  # Original question for readability
            )
            
            # Create a placeholder for the assistant's response
            message_id = chat_service.save_assistant_response(
                session_id=session_id,
                created_at=datetime.datetime.now(),
                question_id=question_id,
                content="",
                response_time=0.0001
            )

            # Start timing the response
            start_time = time.time()
            
            # Step 5: Get recent exchange 
            chat_history = ChatMessageHistory.string_message_chat_history(session_id, 10)
            history_messages = self._convert_history_to_messages(chat_history)
            
            # Step 6: Language detection
            if len(question_input.split()) < 2:
                detection_method = DetectionMethod.LLM
            else:
                detection_method = DetectionMethod.LIBRARY

            language_info = await language_detector.detect(
                text=question_input,  # Original question only
                method=detection_method,
                system_language=system_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )

            detected_language = language_info["detected_language"]

            # Store full response to save in database
            full_response = []
            
            # Step 7: Stream response using prepared query
            async for chunk in self.llm_generator_provider.stream_react_cot_response(
                model_name=model_name,
                query=llm_query,  # Use merged query if reply, original if not
                target_language=detected_language,
                context=combined_context,
                history_messages=history_messages,
                provider_type=provider_type,
                api_key=api_key,
                clean_thinking=clean_thinking,
                enable_thinking=enable_thinking
            ):
                full_response.append(chunk)
                yield chunk
            
            # Calculate response time
            response_time = round(time.time() - start_time, 3)
            
            # Process full response to clean thinking if needed
            complete_response = "".join(full_response)
            if clean_thinking:
                complete_response = self.llm_generator_provider.clean_thinking(complete_response)
            
            # Update the assistant's response in the database
            chat_service.update_assistant_response(
                updated_at=datetime.datetime.now(),
                message_id=message_id,
                content=complete_response,
                response_time=response_time
            )
            
            # Save document references if available
            if document_references:
                await self._save_document_references_from_memory(message_id, document_references)

            # Step 8: Analyze conversation importance
            importance_score = 0.5  # Default score
            
            if session_id and user_id and use_memory_context:
                try:
                    analysis_model = "gpt-4.1-nano"
                    
                    # Use llm_query for importance analysis to capture full context
                    importance_score = await self.analyze_conversation_importance(
                        query=llm_query,  # Use merged query if reply, original if not
                        response=complete_response,
                        llm_provider=self.llm_generator_provider,
                        model_name=analysis_model,
                        provider_type=provider_type
                    )
                    
                    # Boost importance for certain types of conversations
                    if enable_thinking:
                        importance_score = min(1.0, importance_score + 0.1)
                    
                    if len(complete_response) > 2000:
                        importance_score = min(1.0, importance_score + 0.1)
                    
                    # Boost for reply conversations (contextual follow-ups are valuable)
                    if is_reply_conversation:
                        importance_score = min(1.0, importance_score + 0.15)
                    
                    self.logger.info(f"Conversation importance score: {importance_score}")
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing importance: {e}")
            
            # Step 9: Store conversation in memory system
            if session_id and user_id and use_memory_context:
                try:
                    # Prepare metadata
                    metadata = {
                        "type": "chat_reasoning_stream",
                        "model": model_name,
                        "provider": provider_type,
                        "collection": collection_name,
                        "enable_thinking": enable_thinking,
                        "response_time": response_time,
                        "has_document_context": len(document_references) > 0,
                        "memory_stats": memory_stats,
                        "is_reply": is_reply_conversation,
                    }
                    
                    await self.memory_manager.store_conversation_turn(
                        session_id=session_id,
                        user_id=user_id,
                        query=llm_query,  # Preserve full context in memory
                        response=complete_response,
                        metadata=metadata,
                        importance_score=importance_score
                    )
                    
                    self.logger.info(f"Stored conversation in memory for session {session_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error storing conversation in memory: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to handle streaming ReAct+CoT chat request: {str(e)}")
            yield f"An error occurred: {str(e)}"

    async def _save_document_references(self, message_id: str, context_docs: List) -> None:
        """
        Safely save document references, checking for duplicates and existence in database
        
        Args:
            message_id: Message ID to associate references with
            context_docs: Context documents from vector search
        """
        try:
            # Track document IDs that have been processed for this message
            processed_doc_ids = set()
            
            # Process each document, ensuring we only process each once
            for doc in context_docs:
                if 'document_id' in doc.metadata:
                    document_id = doc.metadata['document_id']
                    
                    # Skip if already processed for this message
                    if document_id in processed_doc_ids:
                        continue
                    
                    # Mark as processed
                    processed_doc_ids.add(document_id)
                    
                    # Get page number from metadata
                    page = doc.metadata.get('index', 0)
                    
                    # Try to save, ChatService.save_reference_docs will handle the checks
                    result = chat_service.save_reference_docs(
                        message_id=message_id,
                        document_id=document_id,
                        page=page
                    )
                    
                    # Use result to avoid logging for each failure
                    if result is None:
                        # Reference wasn't saved (already exists or document not found)
                        pass
                    
        except Exception as e:
            self.logger.error(f"Error saving document references: {str(e)}")


    async def _save_document_references_from_memory(self, message_id: str, document_references: List[Dict]) -> None:
        """
        Save document references retrieved from memory
        
        Args:
            message_id: Message ID to associate references with
            document_references: Document references from memory search
        """
        try:
            # Track document IDs that have been processed for this message
            processed_doc_ids = set()
            
            # Process each document reference
            for doc_ref in document_references:
                document_id = doc_ref.get('document_id')
                
                if document_id and document_id not in processed_doc_ids:
                    # Mark as processed
                    processed_doc_ids.add(document_id)
                    
                    # Get page number from metadata
                    page = doc_ref.get('page', 0)
                    
                    # Save reference
                    result = chat_service.save_reference_docs(
                        message_id=message_id,
                        document_id=document_id,
                        page=page
                    )
                    
                    if result:
                        self.logger.info(f"Saved reference for document {doc_ref.get('document_name', document_id)}")
                        
        except Exception as e:
            self.logger.error(f"Error saving document references from memory: {str(e)}")


class ChatMessageHistory(LoggerMixin):
    """
    Utility class for working with chat message history
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def messages_from_items(items: list) -> List[BaseMessage]:
        """
        Convert raw message items to BaseMessage objects
        
        Args:
            items: List of (content, type) tuples
            
        Returns:
            List[BaseMessage]: List of message objects
        """
        def _message_from_item(message: tuple) -> BaseMessage:
            _type = message[1]
            if _type == "human" or _type == "user":
                return HumanMessage(content=message[0])
            elif _type == "ai" or _type == "assistant":
                return AIMessage(content=message[0])
            elif _type == "system":
                return SystemMessage(content=message[0])
            else:
                raise ValueError(f"Got unexpected message type: {_type}")

        messages = [_message_from_item(msg) for msg in items]
        return messages

    @staticmethod
    def concat_message(messages: List[BaseMessage]) -> str:
        """
        Concatenate messages into a single string
        
        Args:
            messages: List of BaseMessage objects
            
        Returns:
            str: Concatenated message history
        """
        concat_chat = ""
        for mes in messages:
            if isinstance(mes, HumanMessage):
                concat_chat += " - user: " + mes.content + "\n"
            else:
                concat_chat += " - assistant: " + mes.content + "\n"
        return concat_chat
    
    @staticmethod
    def string_message_chat_history(session_id: str, limit: int = 10) -> str:
        """
        Get the chat history as a string
        
        Args:
            session_id: The ID of the chat session
            
        Returns:
            str: The chat history as a string
        """
        items = chat_service.get_chat_history(session_id=session_id, limit=10)
        messages = ChatMessageHistory.messages_from_items(items)
        
        # Reverse the order and skip the current message being processed
        history_str = ChatMessageHistory.concat_message(messages[::-1][:-2])
        return history_str
    
    @staticmethod
    def string_message_chat_history_update(session_id: str, limit: int = 10, exclude_last_n: int = 0) -> str:
        """
        Get the chat history as a string
        
        Args:
            session_id: The ID of the chat session
            limit: Maximum number of messages
            exclude_last_n: Number of recent messages to exclude (default: 0)
                
        Returns:
            str: The chat history as a string
        """
        items = chat_service.get_chat_history(session_id=session_id, limit=limit)
        messages = ChatMessageHistory.messages_from_items(items)
        
        # Reverse the order
        messages_reversed = messages[::-1]
        
        # Exclude last N messages if specified
        if exclude_last_n > 0:
            messages_to_use = messages_reversed[:-exclude_last_n]
        else:
            messages_to_use = messages_reversed
        
        history_str = ChatMessageHistory.concat_message(messages_to_use)
        return history_str
    

    @staticmethod
    def get_full_chat_history_for_context(session_id: str, limit: int = 20) -> str:
        """
        Get COMPLETE chat history for context assembly
        
        This method returns ALL messages without exclusion.
        Use this when you need to assemble context BEFORE saving new messages.
        
        Args:
            session_id: The ID of the chat session
            limit: Maximum number of messages to retrieve (default: 20)
                
        Returns:
            str: Complete chat history in chronological order
        """
        items = chat_service.get_chat_history(session_id=session_id, limit=limit)
        
        if not items:
            return ""
        
        messages = ChatMessageHistory.messages_from_items(items)
        
        # Reverse to chronological order (oldest first -> newest last)
        messages_chronological = messages[::-1]
        
        # Return ALL messages
        history_str = ChatMessageHistory.concat_message(messages_chronological)
        return history_str

    def get_list_message_history(
        self, 
        session_id: str, 
        limit: int = 10, 
        user_id: Optional[str] = None, 
        organization_id: Optional[str] = None
    ) -> BasicResponse:
        """
        Get the list of messages in the chat history
        
        Args:
            session_id: The ID of the chat session
            limit: Maximum number of messages to retrieve
            user_id: The ID of the requesting user (for authorization)
            organization_id: The ID of the organization (for filtering)
            
        Returns:
            BasicResponse: Response with message history as data
        """
        try:
            # Check access if user_id is provided
            if user_id:
                session_info = self.get_session_info(session_id)
                if session_info:
                    # Check if user_id matches session owner
                    if session_info.get("user_id") != user_id:
                        # Check if the user belongs to the organization that owns the session
                        if organization_id and session_info.get("organization_id") == organization_id:
                            # Organizational users, allowing access
                            pass
                        else:
                            return BasicResponse(
                                status="Failed",
                                message="You don't have permission to view this chat history",
                                data=None
                            )
            
            # Get history chat from repository
            items = chat_service.get_chat_history(session_id=session_id, limit=limit)
            
            # Format "{role} : {content}"
            formatted_items = [f"{item[1]} : {item[0]}" for item in items]
            
            return BasicResponse(
                status="Success",
                message="Retrieved message history successfully",
                data=formatted_items
            )
        except Exception as e:
            self.logger.error(f"Failed to get message history: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Failed to get message history: {str(e)}",
                data=None
            )
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get info a chat session
        
        Args:
            session_id: ID of chat session
            
        Returns:
            Optional[Dict[str, Any]]: Info session or None if not found
        """
        try:
            from src.database import get_postgres_db
            db = get_postgres_db()
            
            with db.session_scope() as session:
                chat_session = session.query(ChatSessions).filter(
                    ChatSessions.id == session_id
                ).first()
                
                if not chat_session:
                    return None
                    
                return {
                    "id": str(chat_session.id),
                    "user_id": chat_session.user_id,
                    "organization_id": chat_session.organization_id,
                    "title": chat_session.title,
                    "start_date": chat_session.start_date
                }
        except Exception as e:
            self.logger.error(f"Failed to get session info: {str(e)}")
            return None

    def format_history_for_llm(self, raw_history: str) -> str:
        """Convert raw history to LLM-friendly format"""
        if not raw_history:
            return ""
        
        lines = raw_history.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Fix the format from "- role:" to "Role:"
            if line.startswith('- assistant:') or line.startswith('- ai:'):
                content = line.replace('- assistant:', '').replace('- ai:', '').strip()
                formatted_lines.append(f"Assistant: {content}")
            elif line.startswith('- user:') or line.startswith('- human:'):
                content = line.replace('- user:', '').replace('- human:', '').strip()
                formatted_lines.append(f"Human: {content}")
            elif line.startswith('assistant:') or line.startswith('ai:'):
                formatted_lines.append(f"Assistant: {line.split(':', 1)[1].strip()}")
            elif line.startswith('user:') or line.startswith('human:'):
                formatted_lines.append(f"Human: {line.split(':', 1)[1].strip()}")
            else:
                # Keep as is if format unknown
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
                          
    def delete_message_history(
        self, 
        session_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> BasicResponse:
        """
        Delete the chat history for a session
        
        Args:
            session_id: The ID of the chat session to delete
            user_id: The ID of the requesting user (for authorization)
            organization_id: The ID of the organization (for filtering)
            
        Returns:
            BasicResponse: Response indicating success or failure
        """
        try:
            # Check delete permission if user_id is provided
            if user_id:
                session_info = self.get_session_info(session_id)
                if session_info:
                    # Check if user_id matches session owner
                    if session_info.get("user_id") != user_id:
                        # Check if user has admin rights in the organization
                        if organization_id and session_info.get("organization_id") == organization_id:
                            # Need to check admin role here if possible
                            from src.handlers.user_authorization_handler import UserRoleService
                            user_role_service = UserRoleService()
                            is_admin = user_role_service.is_admin(user_id, organization_id)
                            if not is_admin:
                                return BasicResponse(
                                    status="Failed",
                                    message="You don't have permission to delete this chat history",
                                    data=None
                                )
                        else:
                            return BasicResponse(
                                status="Failed",
                                message="You don't have permission to delete this chat history",
                                data=None
                            )
            
            if chat_service.is_session_exist(session_id):
                chat_service.delete_chat_history(session_id=session_id)
                return BasicResponse(
                    status="Success",
                    message="Chat history deleted successfully",
                    data=None
                )
            else:
                return BasicResponse(
                    status="Failed",
                    message="Chat session does not exist",
                    data=None
                )
        except Exception as e:
            self.logger.error(f"Failed to delete message history: {str(e)}")
            return BasicResponse(
                status="Failed",
                message=f"Failed to delete message history: {str(e)}",
                data=None
            )