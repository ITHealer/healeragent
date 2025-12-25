import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)


class GetRecentConversationsTool(BaseTool):
    """
    Get recent conversations with time-based filtering - Claude-style
    
    This is an EXPLICIT tool that the Planning Agent can select
    when the user asks about recent conversations or timeline-based queries.
    
    Features:
    - Time-based retrieval (before/after datetime)
    - Pagination (n parameter)
    - Sort order (asc/desc)
    - Cross-session (all user's conversations)
    
    Usage:
        tool = GetRecentConversationsTool()
        result = await tool.safe_execute(
            user_id="user-123",
            n=5,
            sort_order="desc"
        )
    """
    
    def __init__(self):
        """Initialize GetRecentConversationsTool"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Lazy imports
        self._chat_service = None
        
        # Define schema following BaseTool pattern
        self.schema = ToolSchema(
            name="getRecentConversations",
            category="memory",
            description=(
                "Retrieve recent conversations based on time. "
                "Use when user asks about recent chats, yesterday's discussion, "
                "last week's conversations, or wants to continue a previous chat. "
                "Returns conversation summaries sorted by time."
            ),
            parameters=[
                ToolParameter(
                    name="user_id",
                    type="string",
                    required=True,
                    description="User ID for retrieving conversations"
                ),
                ToolParameter(
                    name="n",
                    type="integer",
                    required=False,
                    description="Number of recent conversations to retrieve (1-20)",
                    default=5,
                    min_value=1,
                    max_value=20
                ),
                ToolParameter(
                    name="sort_order",
                    type="string",
                    required=False,
                    description="Sort order: 'desc' for newest first (default), 'asc' for oldest first",
                    enum=["asc", "desc"],
                    default="desc"
                ),
                ToolParameter(
                    name="before",
                    type="string",
                    required=False,
                    description="Return conversations updated before this datetime (ISO format or relative)"
                ),
                ToolParameter(
                    name="after",
                    type="string",
                    required=False,
                    description="Return conversations updated after this datetime (ISO format or relative: 'yesterday', 'last_week')"
                )
            ],
            returns={
                "conversations": "List of recent conversations with metadata",
                "total_returned": "Number of conversations returned",
                "has_more": "Whether more conversations are available"
            },
            capabilities=[
                "Retrieve recent conversations by count",
                "Filter by date range",
                "Sort chronologically or reverse",
                "Get conversation summaries"
            ],
            limitations=[
                "Only retrieves user's own conversations",
                "Maximum 20 conversations per request",
                "Summaries may be truncated"
            ],
            usage_hints=[
                "Use when user says 'yesterday', 'last week', 'recent chats'",
                "Use for 'continue our last conversation'",
                "Use for 'what did we talk about recently'"
            ],
            requires_symbol=False
        )
    
    async def _get_chat_service(self):
        """Lazy load ChatService"""
        if self._chat_service is None:
            try:
                from src.helpers.chat_management_helper import ChatService
                self._chat_service = ChatService()
            except ImportError as e:
                self.logger.warning(f"[RECENT_CHATS] Could not import ChatService: {e}")
        return self._chat_service
    
    async def execute(
        self,
        user_id: str,
        n: int = 5,
        sort_order: str = "desc",
        before: Optional[str] = None,
        after: Optional[str] = None,
        **kwargs
    ) -> ToolOutput:
        """
        Execute recent conversations retrieval
        
        Args:
            user_id: User ID (REQUIRED)
            n: Number of conversations (1-20)
            sort_order: 'asc' or 'desc'
            before: Filter conversations before this datetime
            after: Filter conversations after this datetime
            
        Returns:
            ToolOutput with conversation list
        """
        tool_name = self.schema.name
        start_time = datetime.now()
        
        # Validate inputs
        if not user_id:
            return create_error_output(
                tool_name=tool_name,
                error_message="user_id is required for cross-session retrieval",
                error_type="validation_error"
            )
        
        # Validate and clamp parameters
        n = max(1, min(20, n))
        sort_order = sort_order if sort_order in ["asc", "desc"] else "desc"
        
        try:
            self.logger.info(
                f"[{tool_name}] n={n}, sort={sort_order}, "
                f"before={before}, after={after}"
            )
            
            # Parse relative time references
            after_parsed = self._parse_relative_time(after) if after else None
            before_parsed = self._parse_relative_time(before) if before else None
            
            # Retrieve conversations
            conversations = await self._get_recent_conversations(
                user_id=user_id,
                n=n,
                sort_order=sort_order,
                before=before_parsed,
                after=after_parsed
            )
            
            execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            # Format for LLM context
            formatted_context = self._format_for_context(conversations)
            
            self.logger.info(
                f"[{tool_name}] Retrieved {len(conversations)} conversations in {execution_time_ms}ms"
            )
            
            return create_success_output(
                tool_name=tool_name,
                data={
                    "conversations": conversations,
                    "total_returned": len(conversations),
                    "has_more": len(conversations) == n,  # Might have more
                    "sort_order": sort_order
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
    
    async def _get_recent_conversations(
        self,
        user_id: str,
        n: int,
        sort_order: str,
        before: Optional[str] = None,
        after: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve recent conversations from database"""
        results = []
        
        try:
            from src.database import get_postgres_db
            from src.database.models.schemas import ChatSessions, Messages
            from sqlalchemy import func, desc, asc
            
            db = get_postgres_db()
            
            with db.session_scope() as session:
                # Build query
                query = session.query(ChatSessions).filter(
                    ChatSessions.user_id == user_id
                )
                
                # Apply datetime filters
                if before:
                    try:
                        before_dt = datetime.fromisoformat(before.replace('Z', '+00:00'))
                        query = query.filter(ChatSessions.start_date < before_dt)
                    except ValueError:
                        pass
                
                if after:
                    try:
                        after_dt = datetime.fromisoformat(after.replace('Z', '+00:00'))
                        query = query.filter(ChatSessions.start_date > after_dt)
                    except ValueError:
                        pass
                
                # Sort by updated_at
                if sort_order == "asc":
                    query = query.order_by(asc(ChatSessions.start_date))
                else:
                    query = query.order_by(desc(ChatSessions.start_date))
                
                # Get sessions
                sessions = query.limit(n).all()
                
                # Format results with summaries
                for chat_session in sessions:
                    session_id = str(chat_session.id)
                    
                    # Get message count
                    msg_count = session.query(func.count(Messages.id)).filter(
                        Messages.session_id == session_id
                    ).scalar() or 0
                    
                    # Get first user message for summary
                    first_msg = session.query(Messages).filter(
                        Messages.session_id == session_id,
                        Messages.message_type == "human"
                    ).order_by(asc(Messages.created_at)).first()
                    
                    summary = ""
                    if first_msg and first_msg.content:
                        summary = f"Started with: {first_msg.content[:100]}..."
                    
                    results.append({
                        "session_id": session_id,
                        "title": chat_session.title or "Untitled Conversation",
                        "summary": summary,
                        "updated_at": chat_session.start_date.isoformat() if chat_session.start_date else "",
                        "created_at": chat_session.start_date.isoformat() if chat_session.start_date else "",
                        "message_count": msg_count,
                        "url": f"/chat/{session_id}"
                    })
                    
        except Exception as e:
            self.logger.error(f"[RECENT_CHATS] Database error: {e}")
        
        return results
    
    def _parse_relative_time(self, time_ref: str) -> Optional[str]:
        """
        Parse relative time references to ISO datetime
        
        Examples:
        - "yesterday" → datetime of yesterday
        - "last week" → datetime of 7 days ago
        - "this month" → datetime of start of current month
        """
        if not time_ref:
            return None
        
        now = datetime.now()
        time_ref_lower = time_ref.lower().strip()
        
        # Map of relative references
        if time_ref_lower == "yesterday":
            target = now - timedelta(days=1)
            return target.replace(hour=0, minute=0, second=0).isoformat()
        
        if time_ref_lower in ["last_week", "last week"]:
            target = now - timedelta(days=7)
            return target.isoformat()
        
        if time_ref_lower in ["this_week", "this week"]:
            # Start of current week (Monday)
            days_since_monday = now.weekday()
            target = now - timedelta(days=days_since_monday)
            return target.replace(hour=0, minute=0, second=0).isoformat()
        
        if time_ref_lower in ["last_month", "last month"]:
            target = now - timedelta(days=30)
            return target.isoformat()
        
        if time_ref_lower in ["this_month", "this month"]:
            target = now.replace(day=1, hour=0, minute=0, second=0)
            return target.isoformat()
        
        if time_ref_lower == "today":
            target = now.replace(hour=0, minute=0, second=0)
            return target.isoformat()
        
        # Return as-is if already ISO format
        return time_ref
    
    def _format_for_context(self, conversations: List[Dict]) -> str:
        """Format conversations for LLM context"""
        if not conversations:
            return "No recent conversations found."
        
        lines = [f"📋 RECENT CONVERSATIONS ({len(conversations)}):"]
        
        for i, conv in enumerate(conversations[:10], 1):
            title = conv.get("title", "Untitled")
            updated = conv.get("updated_at", "")[:10]  # Just date
            msg_count = conv.get("message_count", 0)
            summary = conv.get("summary", "")
            url = conv.get("url", "")
            
            lines.append(f"\n{i}. [{title}] ({updated})")
            lines.append(f"   Messages: {msg_count}")
            if summary:
                lines.append(f"   {summary}")
            lines.append(f"   → {url}")
        
        return '\n'.join(lines)


# Factory function for tool registry
def create_tool() -> GetRecentConversationsTool:
    return GetRecentConversationsTool()