import logging
from typing import Dict, Any, Optional, List

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)


class SearchArchivalMemoryTool(BaseTool):
    """
    Tool for searching knowledge base (Archival Memory)
    
    This tool allows the agent to:
    - Search stored documents and knowledge
    - Find user-specific facts and preferences
    - Retrieve domain knowledge
    
    Usage:
        result = await tool.execute(
            query="user risk tolerance",
            limit=3,
            user_id="user-456"
        )
    """
    
    def __init__(self):
        """Initialize SearchArchivalMemoryTool"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.schema = ToolSchema(
            name="searchArchivalMemory",
            category="memory",
            description=(
                "Search knowledge base for stored facts, documents, and user information. "
                "Use when you need factual information about user, domain knowledge, "
                "or previously stored documents and notes."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    required=True,
                    description="Semantic search query - describe what you're looking for"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    required=False,
                    description="Maximum number of results (1-10)",
                    default=3,
                    min_value=1,
                    max_value=10
                ),
                ToolParameter(
                    name="user_id",
                    type="string",
                    required=True,
                    description="User ID for personalized search"
                ),
                ToolParameter(
                    name="include_global",
                    type="boolean",
                    required=False,
                    description="Include global knowledge base (not just user-specific)",
                    default=True
                )
            ],
            returns={
                "status": "success or error",
                "results": "List of relevant knowledge entries",
                "total_found": "Number of results",
                "sources": "Where the knowledge came from"
            },
            capabilities=[
                "Semantic search across knowledge base",
                "Find user preferences and facts",
                "Retrieve stored documents",
                "Access domain knowledge"
            ],
            limitations=[
                "Depends on what has been stored",
                "May not have real-time information",
                "Quality depends on storage"
            ],
            usage_hints=[
                "Use to find user's stated preferences",
                "Use for domain-specific knowledge",
                "Use when Core Memory doesn't have enough detail"
            ],
            requires_symbol=False
        )
    
    async def execute(
        self,
        query: str,
        user_id: str,
        limit: int = 3,
        include_global: bool = True
    ) -> ToolOutput:
        """
        Execute archival memory search
        
        Args:
            query: Semantic search query
            user_id: User ID
            limit: Max results
            include_global: Include global knowledge
            
        Returns:
            ToolOutput with search results
        """
        tool_name = self.schema.name
        
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
                error_message="user_id is required",
                error_type="validation_error"
            )
        
        limit = int(min(max(limit, 1), 10))  # Clamp to 1-10, ensure int
        
        try:
            # Import here to avoid circular imports
            from src.services.memory_search_service import MemorySearchService
            
            memory_search = MemorySearchService()
            
            # Execute search
            results = await memory_search.search_archival_memory(
                query=query,
                user_id=user_id if not include_global else None,
                limit=limit
            )
            
            # If no results with global, try user-specific only
            if not results and include_global:
                results = await memory_search.search_archival_memory(
                    query=query,
                    user_id=user_id,
                    limit=limit
                )
            
            # Format results
            formatted_results = []
            sources = set()
            
            for doc in results[:limit]:
                formatted_results.append({
                    'content': doc.get('content', '')[:500],
                    'source': doc.get('source', 'knowledge_base'),
                    'score': doc.get('score', 0.0),
                    'metadata': doc.get('metadata', {})
                })
                sources.add(doc.get('source', 'knowledge_base'))
            
            # Build formatted context for LLM
            formatted_context = self._format_for_context(formatted_results, query)
            
            return create_success_output(
                tool_name=tool_name,
                data={
                    'query': query,
                    'results': formatted_results,
                    'total_found': len(formatted_results),
                    'sources': list(sources),
                    'include_global': include_global
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
    
    def _format_for_context(
        self,
        results: List[Dict],
        query: str
    ) -> str:
        """Format search results for LLM context"""
        if not results:
            return f"No knowledge found for '{query}'."
        
        lines = [f"📚 KNOWLEDGE BASE results for '{query}':"]
        
        for i, result in enumerate(results[:5], 1):
            content = result.get('content', '')[:300]
            source = result.get('source', 'unknown')
            score = result.get('score', 0.0)
            
            lines.append(f"\n{i}. [Source: {source}] (relevance: {score:.2f})")
            lines.append(f"   {content}...")
        
        lines.append(f"\n(Found {len(results)} relevant entries)")
        
        return '\n'.join(lines)