# File: src/agents/tools/memory/search_recall_memory.py
"""
SearchRecallMemoryTool - Search Conversation History

Allows agent to dynamically search recall memory (conversation history)
during task execution. This implements the "memory-as-a-tool" pattern.

Category: memory
Purpose: Search past conversations for relevant context
Use when: User references past discussions, needs context from history

Based on:
- Google ADK: Dynamic memory retrieval
- Letta/MemGPT: Recall memory pattern
"""

import logging
from typing import Dict, Any, Optional, List, Literal
from datetime import datetime, timedelta

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)


class SearchRecallMemoryTool(BaseTool):
    """
    Tool for searching conversation history (Recall Memory)
    
    This tool allows the agent to:
    - Search past conversations by topic/keyword
    - Filter by timeframe
    - Find relevant context from history
    
    Usage:
        result = await tool.execute(
            query="AAPL analysis",
            timeframe="last_week",
            limit=5,
            session_id="session-123",
            user_id="user-456"
        )
    """
    
    def __init__(self):
        """Initialize SearchRecallMemoryTool"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        self.schema = ToolSchema(
            name="searchRecallMemory",
            category="memory",
            description=(
                "Search conversation history for relevant past discussions. "
                "Use when user references past conversations, needs context from history, "
                "or when you need to find what was discussed previously about a topic."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    required=True,
                    description="Search terms - topic, keyword, or symbol to find in history"
                ),
                ToolParameter(
                    name="timeframe",
                    type="string",
                    required=False,
                    description="Time filter: last_hour, last_day, last_week, last_month, all",
                    enum=["last_hour", "last_day", "last_week", "last_month", "all"],
                    default="last_week"
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    required=False,
                    description="Maximum number of results (1-20)",
                    default=5,
                    min_value=1,
                    max_value=20
                ),
                ToolParameter(
                    name="session_id",
                    type="string",
                    required=True,
                    description="Current session ID for context"
                ),
                ToolParameter(
                    name="user_id",
                    type="string",
                    required=True,
                    description="User ID for personalized search"
                )
            ],
            returns={
                "status": "success or error",
                "results": "List of relevant conversation excerpts",
                "total_found": "Number of matching results",
                "timeframe_applied": "Timeframe filter used"
            },
            capabilities=[
                "Search past conversations",
                "Filter by time",
                "Find topic-specific discussions",
                "Retrieve user context"
            ],
            limitations=[
                "Only searches current user's history",
                "Limited to conversation content",
                "May not find very old conversations"
            ],
            usage_hints=[
                "Use when user says 'we discussed', 'remember when', 'as I mentioned'",
                "Use to find context about symbols user has asked about before",
                "Use for personalization based on conversation history"
            ],
            requires_symbol=False
        )
    
    async def execute(
        self,
        query: str,
        session_id: str,
        user_id: str,
        timeframe: str = "last_week",
        limit: int = 5
    ) -> ToolOutput:
        """
        Execute recall memory search
        
        Args:
            query: Search terms
            session_id: Session ID
            user_id: User ID
            timeframe: Time filter
            limit: Max results
            
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
        
        if not session_id or not user_id:
            return create_error_output(
                tool_name=tool_name,
                error_message="session_id and user_id are required",
                error_type="validation_error"
            )
        
        limit = min(max(limit, 1), 20)  # Clamp to 1-20
        
        try:
            # Import here to avoid circular imports
            from src.services.memory_search_service import MemorySearchService
            
            memory_search = MemorySearchService()
            
            # Build search params based on timeframe
            date_filter = self._get_date_filter(timeframe)
            
            search_params = {
                'topic': query,
                'date_filter': date_filter,
                'limit': limit
            }
            
            # Execute search
            results = await memory_search.search_recall_memory(
                session_id=session_id,
                strategy="topic",
                params=search_params,
                user_id=user_id
            )
            
            # Format results
            formatted_results = []
            for msg in results[:limit]:
                formatted_results.append({
                    'role': msg.get('role', 'unknown'),
                    'content': msg.get('content', '')[:500],  # Truncate long content
                    'timestamp': msg.get('created_at', ''),
                    'relevance': msg.get('_relevance', 0.0)
                })
            
            # Build formatted context for LLM
            formatted_context = self._format_for_context(formatted_results, query)
            
            return create_success_output(
                tool_name=tool_name,
                data={
                    'query': query,
                    'results': formatted_results,
                    'total_found': len(formatted_results),
                    'timeframe_applied': timeframe,
                    'search_params': search_params
                },
                formatted_context=formatted_context,
                symbols=[]  # Memory search doesn't return symbols
            )
            
        except Exception as e:
            self.logger.error(f"[{tool_name}] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name=tool_name,
                error_message=str(e),
                error_type="execution_error"
            )
    
    def _get_date_filter(self, timeframe: str) -> Optional[str]:
        """Convert timeframe to date filter"""
        now = datetime.now()
        
        if timeframe == "last_hour":
            cutoff = now - timedelta(hours=1)
        elif timeframe == "last_day":
            cutoff = now - timedelta(days=1)
        elif timeframe == "last_week":
            cutoff = now - timedelta(weeks=1)
        elif timeframe == "last_month":
            cutoff = now - timedelta(days=30)
        elif timeframe == "all":
            return None
        else:
            cutoff = now - timedelta(weeks=1)  # Default
        
        return cutoff.isoformat()
    
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
            role = result.get('role', 'unknown')
            content = result.get('content', '')[:200]
            timestamp = result.get('timestamp', '')
            
            lines.append(f"\n{i}. [{role}] {timestamp}:")
            lines.append(f"   {content}...")
        
        lines.append(f"\n(Found {len(results)} relevant messages)")
        
        return '\n'.join(lines)