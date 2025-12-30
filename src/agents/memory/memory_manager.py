import hashlib
from langchain_core.documents import Document
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any

from src.utils.logger.custom_logging import LoggerMixin
from src.handlers.retrieval_handler import SearchRetrieval
from src.handlers.vector_store_handler import VectorStoreQdrant
from src.helpers.qdrant_connection_helper import QdrantConnection
# from src.agents.memory.context_compressor import ContextCompressor
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.handlers.v2.tool_call_handler import tool_call
from src.database.repository.sessions import SessionRepository


# =============================================================================
# SINGLETON INSTANCES - Prevent multiple heavy objects
# =============================================================================

_memory_manager_instance: Optional['MemoryManager'] = None
_vector_store_instance: Optional[VectorStoreQdrant] = None
_retrieval_instance: Optional[SearchRetrieval] = None
_qdrant_conn_instance: Optional[QdrantConnection] = None


def get_memory_manager() -> 'MemoryManager':
    """Get or create singleton MemoryManager instance"""
    global _memory_manager_instance
    if _memory_manager_instance is None:
        _memory_manager_instance = MemoryManager()
    return _memory_manager_instance


def get_vector_store() -> VectorStoreQdrant:
    """Get or create singleton VectorStoreQdrant instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreQdrant()
    return _vector_store_instance


def get_retrieval() -> SearchRetrieval:
    """Get or create singleton SearchRetrieval instance"""
    global _retrieval_instance
    if _retrieval_instance is None:
        _retrieval_instance = SearchRetrieval()
    return _retrieval_instance


def get_qdrant_connection() -> QdrantConnection:
    """Get or create singleton QdrantConnection instance"""
    global _qdrant_conn_instance
    if _qdrant_conn_instance is None:
        _qdrant_conn_instance = QdrantConnection()
    return _qdrant_conn_instance


class MemoryManager(LoggerMixin):
    """
    Manage dual memory system: short-term (session) and long-term (knowledge)

    NOTE: Use get_memory_manager() to get singleton instance instead of
    creating new MemoryManager() to prevent resource waste.
    """

    def __init__(self):
        super().__init__()
        # Use singleton helpers to share connections
        self.vector_store = get_vector_store()
        self.qdrant_conn = get_qdrant_connection()
        self.retrieval = get_retrieval()
        self.session_repo = SessionRepository()

        # Memory configuration
        self.short_term_window = 10  # Keep last 10 conversation turns
        self.long_term_threshold = 0.7  # Importance score threshold

        self.logger.debug("[MEMORY_MANAGER] Initialized with shared connections")
        

    def get_memory_collection_name(self, 
                                   session_id: str, 
                                   user_id: str, 
                                   memory_type: str = "short",
                                   base_collection: Optional[str] = None) -> str:
        """
        Generate collection name for user's conversation memory
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            memory_type: "short" or "long" term memory
            base_collection: Base collection name from API input
            
        Returns:
            str: Collection name with appropriate prefix
        """
        if base_collection:
            return f"{memory_type}_{base_collection}"
        else:
            hash_input = f"{user_id}_{session_id}_{memory_type}"
            hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            return f"memory_{memory_type}_{hash_suffix}"
        

    async def ensure_session_memory_collections(self, session_id: str, user_id: str, base_collection: Optional[str] = None):
        """
        Ensure both short-term and long-term memory collections exist for a session/user.

        Args:
            session_id: Session identifier
            user_id: User identifier
            base_collection: Base collection name from API input (optional)
        """
        
        for memory_type in ["short", "long"]:
            collection_name = self.get_memory_collection_name(session_id, user_id, memory_type, base_collection)
            await self.ensure_memory_collection(collection_name)
    

    async def ensure_memory_collection(self, collection_name: str):
        """Ensure a single memory collection exists with the expected configuration"""
        try:
            # Use async wrapper to avoid blocking event loop
            if not await self.qdrant_conn.collection_exists_async(collection_name):
                is_created = await self.qdrant_conn._create_collection_async(collection_name=collection_name)
                if is_created:
                    self.logger.info(f"Created memory collection with hybrid search support: {collection_name}")
                else:
                    self.logger.error(f"Failed to create memory collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Error ensuring memory collection: {e}")
    

    async def store_conversation_turn(
        self,
        session_id: str,
        user_id: str,
        query: str,
        response: str,
        metadata: Optional[Dict] = None,
        importance_score: float = 0.5,
        base_collection: Optional[str] = None
    ):
        """
        Store a conversation turn in appropriate memory
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            query: User query
            response: AI response
            metadata: Additional metadata
            importance_score: Score to determine if should go to long-term memory
            base_collection: Base collection name from API input
        """
        try:
            # Step 1: Ensure collections exist
            await self.ensure_session_memory_collections(session_id, user_id)
            
            # Step 2: Create document with proper metadata structure
            content = f"User: {query}\nAssistant: {response}"
            
            if metadata is None:
                metadata = {}

            # Add required fields for memory documents
            metadata.update({
                "timestamp": datetime.now().isoformat(),
                "type": "conversation_memory",
                "query_length": len(query),
                "response_length": len(response),
                "importance_score": importance_score,
                "user_id": user_id,
                "session_id": session_id,
                # Add dummy fields to prevent query_headers errors if accidentally used
                "document_name": f"memory_{session_id}",
                "headers": "conversation_memory",
                "document_id": f"mem_{datetime.now().timestamp()}"
            })

            document = Document(
                page_content=content,
                metadata=metadata
            )
            
            # Always store in short-term memory using add_data method
            short_collection = self.get_memory_collection_name(session_id, user_id, "short", base_collection)
            success = await self.qdrant_conn.add_data(
                documents=[document],
                collection_name=short_collection
            )
            
            if not success:
                self.logger.error("Failed to store in short-term memory")
                return False
            
            self.logger.info(f"Stored conversation in short-term memory: {short_collection}")
            
            # Store in long-term memory if important enough
            if importance_score >= self.long_term_threshold:
                long_collection = self.get_memory_collection_name(session_id, user_id, "long")
                await self.qdrant_conn.add_data(
                    documents=[document],
                    collection_name=long_collection
                )
                self.logger.info("Stored important conversation in long-term memory")
                
            return True
        
        except Exception as e:
            self.logger.error(f"Error storing conversation: {e}")
            return False

    
    async def get_relevant_context(
        self,
        session_id: str,
        user_id: str,
        current_query: str,
        llm_provider: Any,
        max_short_term: int = 5,
        max_long_term: int = 3,
        base_collection: Optional[str] = None
    ) -> Tuple[str, Dict, List[Dict]]:
        """
        Get relevant context from both short and long-term memory, including document references
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            current_query: Current user query
            llm_provider: LLM provider for compression if needed
            max_short_term: Maximum number of short-term memories
            max_long_term: Maximum number of long-term memories (including documents)
            base_collection: Base collection name from API input (optional)
            
        Returns:
            Tuple[str, Dict, List[Dict]]: (combined_context, stats, document_references)
        """
        context_parts = []
        total_memories = 0
        document_references = []  # Track document references found
        
        try:
            # 1. Get short-term memory (recent conversations)
            short_collection = self.get_memory_collection_name(
                session_id, user_id, "short", base_collection
            )
            short_memories = await self.search_relevant_memory(
                query=current_query,
                collection_name=short_collection,
                top_k=max_short_term
            )
            
            if short_memories:
                short_context = "\n[Recent Conversations]\n"
                for i, mem in enumerate(short_memories, 1):
                    short_context += f"\n--- Recent {i} ---\n{mem['content']}\n"
                context_parts.append(short_context)
                total_memories += len(short_memories)
                self.logger.info(f"Added {len(short_memories)} short-term memories to context")
            
            # 2. Get long-term memory (important conversations AND documents)
            long_collection = self.get_memory_collection_name(
                session_id, user_id, "long", base_collection
            )
            long_memories = await self.search_relevant_memory(
                query=current_query,
                collection_name=long_collection,
                top_k=max_long_term
            )
            
            if long_memories:
                document_context = []
                conversation_context = []
                
                for mem in long_memories:
                    # Check if this is a document or conversation
                    if mem.get('metadata', {}).get('type') == 'document':
                        # This is a document - add to document context
                        document_context.append(mem['content'])
                        
                        # Extract document reference info
                        doc_ref = {
                            'document_id': mem['metadata'].get('document_id'),
                            'document_name': mem['metadata'].get('document_name'),
                            'page': mem['metadata'].get('page', 0),
                            'metadata': mem['metadata']
                        }
                        document_references.append(doc_ref)
                    else:
                        # This is a conversation - add to conversation context
                        conversation_context.append(mem['content'])
                
                # Add document context if found
                if document_context:
                    doc_context_str = "\n[Document Context]\n"
                    for i, doc_content in enumerate(document_context, 1):
                        doc_context_str += f"\n--- Document {i} ---\n{doc_content}\n"
                    context_parts.append(doc_context_str)
                    self.logger.info(f"Added {len(document_context)} documents to context")
                
                # Add conversation context if found
                if conversation_context:
                    conv_context_str = "\n[Important Past Conversations]\n"
                    for i, conv_content in enumerate(conversation_context, 1):
                        conv_context_str += f"\n--- Important Memory {i} ---\n{conv_content}\n"
                    context_parts.append(conv_context_str)
                    self.logger.info(f"Added {len(conversation_context)} important conversations to context")
                
                total_memories += len(long_memories)
            
            # 3. Combine contexts
            full_context = "\n".join(context_parts) if context_parts else ""
            
            stats = {
                "total_memories_found": total_memories,
                "short_term_count": len(short_memories) if short_memories else 0,
                "long_term_count": len(long_memories) if long_memories else 0,
                "documents_found": len(document_references)
            }
            
            return full_context, stats, document_references
            
        except Exception as e:
            self.logger.error(f"Error getting relevant context: {e}")
            return "", {"error": str(e)}, []
    

    async def search_relevant_memory(
        self,
        query: str,
        collection_name: str,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Search for relevant memories in a collection
        
        Args:
            query: Search query
            collection_name: Collection to search in
            top_k: Number of results to return
            
        Returns:
            List[Dict]: List of relevant memories with content and metadata
        """
        try:
            # Check and create collection if not exists (using async wrapper)
            if not await self.qdrant_conn.collection_exists_async(collection_name):
                self.logger.info(f"Collection {collection_name} does not exist, creating it...")
                # Create collection
                await self.ensure_memory_collection(collection_name)
                return []

            # Use qdrant retrieval to search
            results = await self.retrieval.qdrant_retrieval(
                query=query,
                collection_name=collection_name,
                top_k=top_k
            )
            
            # Convert results to memory format
            memories = []
            for doc in results:
                memory = {
                    'content': doc.page_content,
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                }
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            self.logger.error(f"Error searching memory in {collection_name}: {e}")
            return []
        

    async def get_memory_stats(self, session_id: str, user_id: str) -> Dict:
        """
        Get statistics about both memory collections
        
        Returns:
            Dict contains information about both short and long term memory
        """
        stats = {
            "short_term": {},
            "long_term": {}
        }
        
        for memory_type in ["short", "long"]:
            collection_name = self.get_memory_collection_name(session_id, user_id, memory_type)
            try:
                # Use async wrappers to avoid blocking event loop
                if await self.qdrant_conn.collection_exists_async(collection_name):
                    collection_info = await self.qdrant_conn.get_collection_async(collection_name)

                    stats[f"{memory_type}_term"] = {
                        "collection_name": collection_name,
                        "vectors_count": collection_info.vectors_count,
                        "points_count": collection_info.points_count,
                        "indexed_vectors": collection_info.indexed_vectors_count,
                        "status": "active"
                    }
                else:
                    stats[f"{memory_type}_term"] = {
                        "collection_name": collection_name,
                        "vectors_count": 0,
                        "status": "not_found"
                    }
            except Exception as e:
                stats[f"{memory_type}_term"] = {
                    "collection_name": collection_name,
                    "vectors_count": 0,
                    "status": "error",
                    "error": str(e)
                }

        return stats
    

    async def cleanup_old_memories(
        self,
        session_id: str,
        user_id: str,
        days_to_keep: int = 7
    ):
        """
        Clean up old memories from short-term memory
        Move important ones to long-term before deletion
        """
        try:
            short_collection = self.get_memory_collection_name(session_id, user_id, "short")

            # Check if collection exists (using async wrapper)
            if not await self.qdrant_conn.collection_exists_async(short_collection):
                self.logger.info(f"Collection {short_collection} does not exist, nothing to cleanup")
                return

            # Get all points from collection (using async wrapper)
            scroll_result = await self.qdrant_conn.scroll_async(
                collection_name=short_collection,
                limit=1000
            )
            points = scroll_result[0]

            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            points_to_delete = []

            for point in points:
                timestamp_str = point.payload.get("metadata", {}).get("timestamp", "")
                if timestamp_str:
                    timestamp = datetime.fromisoformat(timestamp_str)

                    if timestamp < cutoff_date:
                        # Check importance before deletion
                        importance = point.payload.get("metadata", {}).get("importance_score", 0)
                        if importance >= self.long_term_threshold:
                            # Move to long-term memory
                            long_collection = self.get_memory_collection_name(session_id, user_id, "long")

                            # Create document from point
                            doc = Document(
                                page_content=point.payload.get("page_content", ""),
                                metadata=point.payload.get("metadata", {})
                            )

                            # Add to long-term
                            await self.qdrant_conn.add_data(
                                documents=[doc],
                                collection_name=long_collection
                            )

                        # Mark for deletion from short-term
                        points_to_delete.append(point.id)

            # Delete old points (using async wrapper)
            if points_to_delete:
                await self.qdrant_conn.delete_async(
                    collection_name=short_collection,
                    points_selector=points_to_delete
                )
                self.logger.info(f"Cleaned up {len(points_to_delete)} old memories from {short_collection}")

        except Exception as e:
            self.logger.error(f"Error cleaning up memories: {e}")


    async def search_recall_memory(
        self,
        session_id: str,
        strategy: str,
        params: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search conversation history with different strategies
        
        Args:
            session_id: Session ID
            strategy: temporal/symbol/topic/hybrid
            params: Search parameters from Inner Thoughts
            user_id: User ID
            
        Returns:
            List of relevant messages
        """
        try:
            results = []
            
            if strategy == "temporal":
                results = await self._temporal_search(
                    session_id=session_id,
                    params=params,
                    user_id=user_id
                )
            
            elif strategy == "symbol":
                results = await self._symbol_search(
                    session_id=session_id,
                    symbols=params.get('symbols', []),
                    limit=params.get('limit', 10)
                )
            
            elif strategy == "topic":
                results = await self._topic_search(
                    session_id=session_id,
                    topic=params.get('topic', ''),
                    limit=params.get('limit', 10)
                )
            
            elif strategy == "hybrid":
                # Combine multiple strategies
                temporal_results = await self._temporal_search(
                    session_id=session_id,
                    params=params,
                    user_id=user_id
                )
                
                symbol_results = await self._symbol_search(
                    session_id=session_id,
                    symbols=params.get('symbols', []),
                    limit=5
                )
                
                # Merge and deduplicate
                seen_ids = set()
                for msg in temporal_results + symbol_results:
                    msg_id = msg.get('id')
                    if msg_id and msg_id not in seen_ids:
                        results.append(msg)
                        seen_ids.add(msg_id)
            
            self.logger.info(
                f"[RECALL SEARCH] Strategy: {strategy}, Found: {len(results)} messages"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in recall search: {e}")
            return []
    
    
    async def search_archival_memory(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search knowledge base (vector DB)
        
        Args:
            query: Search query
            user_id: User ID for personalized search
            limit: Number of results
            
        Returns:
            List of relevant documents
        """
        try:
            # Determine collection
            collection_name = f"archival_{user_id}" if user_id else "global_knowledge"

            # Use semantic search (this is MemoryManager, so use self)
            memories = await self.search_relevant_memory(
                query=query,
                collection_name=collection_name,
                top_k=limit
            )
            
            # Convert memory format to document format
            results = []
            for mem in memories:
                results.append({
                    'content': mem.get('content', ''),
                    'metadata': mem.get('metadata', {}),
                    'source': mem.get('metadata', {}).get('source', 'knowledge_base'),
                    'score': mem.get('metadata', {}).get('score', 0.0)
                })
            
            self.logger.info(
                f"[ARCHIVAL SEARCH] Query: '{query[:50]}...', Found: {len(results)} docs"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in archival search: {e}")
            return []
    
    
    async def execute_tool_with_context(
        self,
        tool_name: str,
        tool_params: Dict[str, Any],
        query: str,
        conversation_history: Optional[List] = None,
        provider_type: str = ProviderType.OPENAI,
        model_name: str = "gpt-4.1-nano"
    ) -> Dict[str, Any]:
        """
        Execute tool using existing tool_call handler
        Implements Deep Agents pattern
        
        Args:
            tool_name: Name of tool to execute
            tool_params: Parameters for tool
            query: Original user query
            conversation_history: Context for tool execution
            provider_type: LLM provider
            model_name: Model to use
            
        Returns:
            Tool execution result
        """
        try:
            # Use existing tool_call with context
            result = await tool_call(
                prompt=query,
                model_name=model_name,
                provider_type=provider_type,
                conversation_history=conversation_history
            )
            
            # Enhance result with metadata
            enhanced_result = {
                "tool_executed": tool_name,
                "execution_time": datetime.now().isoformat(),
                "original_result": result,
                "status": result.get('status', '200')
            }
            
            self.logger.info(
                f"[TOOL EXECUTION] Tool: {tool_name}, Status: {enhanced_result['status']}"
            )
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {e}")
            return {
                "tool_executed": tool_name,
                "status": "error",
                "error": str(e)
            }
    
    
    # Private helper methods
    
    async def _temporal_search(
        self,
        session_id: str,
        params: Dict,
        user_id: str
    ) -> List[Dict]:
        """Search by time period"""
        try:
            period = params.get('period', 'today')
            now = datetime.now()
            
            # Calculate time range
            if period == 'today':
                start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == 'yesterday':
                start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            elif period == 'last_week':
                start_date = now - timedelta(days=7)
            elif period == 'last_month':
                start_date = now - timedelta(days=30)
            else:
                start_date = now - timedelta(days=7)  # Default to last week
            
            # Get messages in time range
            messages = await self.session_repo.get_session_messages_by_date_range(
                session_id=session_id,
                start_date=start_date,
                end_date=now,
                limit=params.get('limit', 20)
            )
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Temporal search error: {e}")
            return []
    
    
    async def _symbol_search(
        self,
        session_id: str,
        symbols: List[str],
        limit: int
    ) -> List[Dict]:
        """Search by stock/crypto symbols"""
        try:
            if not symbols:
                return []
            
            # Get all session messages
            all_messages = await self.session_repo.get_session_messages(
                session_id=session_id,
                limit=100  # Get more to filter
            )
            
            # Filter messages containing symbols
            filtered = []
            for msg in all_messages:
                content = msg.get('content', '').upper()
                if any(symbol.upper() in content for symbol in symbols):
                    filtered.append(msg)
                    if len(filtered) >= limit:
                        break
            
            return filtered
            
        except Exception as e:
            self.logger.error(f"Symbol search error: {e}")
            return []
    
    
    async def _topic_search(
        self,
        session_id: str,
        topic: str,
        limit: int
    ) -> List[Dict]:
        """Search by topic/keywords using semantic search"""
        try:
            # Get collection name for this session's short-term memory
            # Use a simple hash of session_id for collection naming
            collection_name = f"memory_short_{session_id[:8]}"

            # Use semantic search (this is MemoryManager, so use self)
            memories = await self.search_relevant_memory(
                query=topic,
                collection_name=collection_name,
                top_k=limit
            )
            
            # Convert memory format to message format
            messages = []
            for mem in memories:
                messages.append({
                    'content': mem.get('content', ''),
                    'metadata': mem.get('metadata', {}),
                    'role': mem.get('metadata', {}).get('role', 'assistant'),
                    'created_at': mem.get('metadata', {}).get('timestamp', '')
                })
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Topic search error: {e}")
            return []
    
    
    async def format_search_results_for_context(
        self,
        recall_results: List[Dict],
        archival_results: List[Dict]
    ) -> str:
        """
        Format search results for inclusion in LLM context
        
        Args:
            recall_results: Results from conversation history
            archival_results: Results from knowledge base
            
        Returns:
            Formatted string for context
        """
        context_parts = []
        
        if recall_results:
            context_parts.append("ðŸ“ RELEVANT CONVERSATION HISTORY:")
            for i, msg in enumerate(recall_results[:5], 1):
                role = msg.get('role', 'user')
                content = msg.get('content', '')[:200]
                timestamp = msg.get('created_at', '')
                context_parts.append(
                    f"{i}. [{role.upper()}] {timestamp}: {content}..."
                )
            context_parts.append("")
        
        if archival_results:
            context_parts.append("ðŸ“š RELEVANT KNOWLEDGE:")
            for i, doc in enumerate(archival_results[:3], 1):
                content = doc.get('content', '')[:300]
                source = doc.get('source', 'knowledge_base')
                context_parts.append(
                    f"{i}. [{source}]: {content}..."
                )
            context_parts.append("")
        
        return "\n".join(context_parts) if context_parts else ""