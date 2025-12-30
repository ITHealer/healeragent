import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)


class SearchConversationHistoryTool(BaseTool):
    """
    Tool for searching conversation history ACROSS ALL sessions - Claude-style
    
    This is an EXPLICIT tool that the Planning Agent can select
    when the user asks about past conversations or needs historical context.
    
    Key Features:
    - CROSS-SESSION search (all user's conversations, not just current)
    - EXPLICIT invocation (LLM decides when to call)
    - Returns conversation snippets with session references
    - Hybrid search: semantic + keyword
    
    Differences from SearchRecallMemoryTool:
    - searchRecallMemory: Current session only, implicit
    - searchConversationHistory: All sessions, explicit, Claude-style
    
    Usage:
        tool = SearchConversationHistoryTool()
        result = await tool.safe_execute(
            query="AAPL analysis",
            user_id="user-123",
            max_results=5
        )
    """
    
    def __init__(self):
        """Initialize SearchConversationHistoryTool"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Lazy load to avoid circular imports
        self._memory_manager = None
        self._session_repo = None
        
        # Define schema following BaseTool pattern
        self.schema = ToolSchema(
            name="searchConversationHistory",
            category="memory",
            description=(
                "Search past conversations by topic or keyword across ALL sessions. "
                "Use when user asks about previous discussions, references past context, "
                "or says things like 'what did we discuss', 'as I mentioned before', "
                "'do you remember when we talked about'. "
                "Returns relevant conversation snippets with session references."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    required=True,
                    description="Topic or keywords to search for in past conversations"
                ),
                ToolParameter(
                    name="user_id",
                    type="string",
                    required=True,
                    description="User ID for cross-session search"
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    required=False,
                    description="Maximum number of results to return (1-10)",
                    default=5,
                    min_value=1,
                    max_value=10
                ),
                ToolParameter(
                    name="date_range",
                    type="object",
                    required=False,
                    description="Optional date range filter with 'after' and/or 'before' ISO dates"
                )
            ],
            returns={
                "query": "Original search query",
                "results": "List of conversation snippets with session info",
                "total_found": "Number of results found"
            },
            capabilities=[
                "Search ALL past conversations (cross-session)",
                "Find topic-specific discussions",
                "Retrieve context from conversation history",
                "Filter by date range"
            ],
            limitations=[
                "Only searches user's own conversations",
                "Quality depends on what was discussed",
                "May not find very old conversations if not indexed"
            ],
            usage_hints=[
                "Use when user says 'we discussed', 'remember when', 'as I mentioned'",
                "Use to find context about topics user has asked about before",
                "Use for personalization based on conversation history"
            ],
            requires_symbol=False
        )
    
    async def _get_memory_manager(self):
        """Lazy load memory manager using singleton"""
        if self._memory_manager is None:
            try:
                from src.agents.memory.memory_manager import get_memory_manager
                self._memory_manager = get_memory_manager()
            except ImportError as e:
                self.logger.warning(f"[CONVERSATION_SEARCH] Could not import MemoryManager: {e}")
        return self._memory_manager
    
    async def _get_session_repo(self):
        """Lazy load session repository"""
        if self._session_repo is None:
            try:
                from src.helpers.chat_management_helper import ChatService
                self._session_repo = ChatService()
            except ImportError as e:
                self.logger.warning(f"[CONVERSATION_SEARCH] Could not import ChatService: {e}")
        return self._session_repo
    
    async def execute(
        self,
        query: str,
        user_id: str,
        max_results: int = 5,
        date_range: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ToolOutput:
        """
        Execute cross-session conversation history search
        
        Args:
            query: Topic or keywords to search
            user_id: User ID for cross-session search (REQUIRED)
            max_results: Maximum results (1-10)
            date_range: Optional date filter {after: str, before: str}
            
        Returns:
            ToolOutput with search results
        """
        tool_name = self.schema.name
        start_time = datetime.now()
        
        # Validate inputs
        if not query or len(query.strip()) < 2:
            return create_error_output(
                tool_name=tool_name,
                error_message="Query must be at least 2 characters",
                error_type="validation_error"
            )
        
        if not user_id:
            return create_error_output(
                tool_name=tool_name,
                error_message="user_id is required for cross-session search",
                error_type="validation_error"
            )
        
        # Clamp max_results to valid range
        max_results = min(max(max_results, 1), 10)
        
        try:
            self.logger.info(
                f"[{tool_name}] Query: '{query}', user: {user_id}, max: {max_results}"
            )
            
            # Search using hybrid strategy
            results = await self._search_conversations(
                query=query,
                user_id=user_id,
                max_results=max_results,
                date_range=date_range
            )
            
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Format for LLM context
            formatted_context = self._format_for_context(results, query)
            
            self.logger.info(
                f"[{tool_name}] Found {len(results)} results in {execution_time_ms}ms"
            )
            
            return create_success_output(
                tool_name=tool_name,
                data={
                    "query": query,
                    "results": results,
                    "total_found": len(results),
                    "has_more": len(results) == max_results
                },
                formatted_context=formatted_context,
                symbols=[]
            )
            
        except Exception as e:
            self.logger.error(f"[{tool_name}] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name=tool_name,
                error_message=str(e),
                error_type="execution_error"
            )
    
    async def _search_conversations(
        self,
        query: str,
        user_id: str,
        max_results: int,
        date_range: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search conversations using hybrid strategy:
        1. Vector/semantic search for topic relevance
        2. Keyword matching for specific terms
        3. Combine and rank results
        """
        results = []
        
        try:
            memory_manager = await self._get_memory_manager()
            session_repo = await self._get_session_repo()
            
            # Strategy 1: Semantic search across all user's conversations
            semantic_results = await self._semantic_search(
                memory_manager=memory_manager,
                query=query,
                user_id=user_id,
                limit=max_results * 2  # Get more for ranking
            )
            
            # Strategy 2: Keyword search in session messages
            keyword_results = await self._keyword_search(
                session_repo=session_repo,
                query=query,
                user_id=user_id,
                limit=max_results * 2,
                date_range=date_range
            )
            
            # Merge and deduplicate
            seen_ids = set()
            combined = []
            
            # Add semantic results first (usually more relevant)
            for result in semantic_results:
                session_id = result.get("session_id")
                if session_id and session_id not in seen_ids:
                    seen_ids.add(session_id)
                    result["source"] = "semantic"
                    combined.append(result)
            
            # Add keyword results
            for result in keyword_results:
                session_id = result.get("session_id")
                if session_id and session_id not in seen_ids:
                    seen_ids.add(session_id)
                    result["source"] = "keyword"
                    combined.append(result)
            
            # Sort by relevance score (descending)
            combined.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Take top results and add URLs
            results = combined[:max_results]
            for result in results:
                result["url"] = f"/chat/{result.get('session_id', '')}"
            
        except Exception as e:
            self.logger.error(f"[CONVERSATION_SEARCH] Search error: {e}")
        
        return results
    
    async def _semantic_search(
        self,
        memory_manager,
        query: str,
        user_id: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Semantic search using vector embeddings"""
        results = []
        
        if not memory_manager:
            return results
        
        try:
            # Search in user's archival memory (cross-session)
            collection_name = f"conversations_{user_id}"
            
            memories = await memory_manager.search_relevant_memory(
                query=query,
                collection_name=collection_name,
                top_k=limit
            )
            
            for mem in memories:
                metadata = mem.get("metadata", {})
                results.append({
                    "session_id": metadata.get("session_id", ""),
                    "conversation_title": metadata.get("title", "Untitled"),
                    "snippet": mem.get("content", "")[:300],
                    "timestamp": metadata.get("timestamp", ""),
                    "relevance_score": mem.get("score", 0.5)
                })
                
        except Exception as e:
            self.logger.warning(f"[SEMANTIC_SEARCH] Error: {e}")
        
        return results
    
    async def _keyword_search(
        self,
        session_repo,
        query: str,
        user_id: str,
        limit: int,
        date_range: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Keyword-based search in session messages using ChatService"""
        results = []
        
        if not session_repo:
            return results
        
        try:
            # Get all user's sessions using ChatService
            from src.database import get_postgres_db
            from src.database.models.schemas import ChatSessions, Messages
            
            db = get_postgres_db()
            
            with db.session_scope() as session:
                # Query sessions for this user
                query_builder = session.query(ChatSessions).filter(
                    ChatSessions.user_id == user_id
                )
                
                # Apply date filters if provided
                if date_range:
                    if date_range.get("after"):
                        query_builder = query_builder.filter(
                            ChatSessions.start_date >= date_range["after"]
                        )
                    if date_range.get("before"):
                        query_builder = query_builder.filter(
                            ChatSessions.start_date <= date_range["before"]
                        )
                
                # Order by most recent first
                sessions = query_builder.order_by(
                    ChatSessions.start_date.desc()
                ).limit(50).all()
                
                # Search in each session's messages
                keywords = query.lower().split()
                
                for chat_session in sessions:
                    session_id = str(chat_session.id)
                    
                    # Get messages for this session
                    messages = session.query(Messages).filter(
                        Messages.session_id == session_id
                    ).order_by(Messages.created_at.desc()).limit(30).all()
                    
                    # Check for keyword matches
                    for msg in messages:
                        content = (msg.content or "").lower()
                        if any(kw in content for kw in keywords):
                            # Calculate simple relevance score
                            match_count = sum(1 for kw in keywords if kw in content)
                            relevance = match_count / len(keywords) if keywords else 0
                            
                            results.append({
                                "session_id": session_id,
                                "conversation_title": chat_session.title or "Untitled",
                                "snippet": (msg.content or "")[:300],
                                "timestamp": msg.created_at.isoformat() if msg.created_at else "",
                                "relevance_score": relevance
                            })
                            break  # One match per session is enough
                    
                    if len(results) >= limit:
                        break
                        
        except Exception as e:
            self.logger.warning(f"[KEYWORD_SEARCH] Error: {e}")
        
        return results
    
    def _format_for_context(
        self,
        results: List[Dict],
        query: str
    ) -> str:
        """Format search results for LLM context"""
        if not results:
            return f"No past conversations found about '{query}'."
        
        lines = [f"📝 PAST CONVERSATIONS about '{query}':"]
        
        for i, result in enumerate(results[:5], 1):
            title = result.get("conversation_title", "Untitled")
            snippet = result.get("snippet", "")[:200]
            timestamp = result.get("timestamp", "")
            session_url = result.get("url", "")
            
            lines.append(f"\n{i}. [{title}] ({timestamp})")
            lines.append(f"   {snippet}...")
            if session_url:
                lines.append(f"   → {session_url}")
        
        lines.append(f"\n(Found {len(results)} relevant conversations)")
        
        return '\n'.join(lines)