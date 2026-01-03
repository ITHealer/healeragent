
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

from src.agents.tools.base import (
    BaseTool,
    ToolSchema,
    ToolParameter,
    ToolOutput,
    create_success_output,
    create_error_output
)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_EDITS = 30
MAX_CHARS_PER_EDIT = 200
CONSOLIDATION_THRESHOLD = 5  # Trigger consolidation after N edits in session

# Categories for memory edits
MEMORY_CATEGORIES = [
    "preference",   # User preferences (language, format, style)
    "fact",         # Personal facts (name, job, location)
    "goal",         # Current goals/projects
    "skill",        # Technical skills/expertise
    "context",      # Contextual info (timezone, platform)
    "other"         # Uncategorized
]

# Patterns for auto-detecting categories
CATEGORY_PATTERNS = {
    "preference": [
        r"prefer", r"like", r"want", r"style", r"format",
        r"language", r"tone", r"always", r"never"
    ],
    "fact": [
        r"name is", r"work at", r"live in", r"born", r"age",
        r"job", r"role", r"position", r"company", r"from"
    ],
    "goal": [
        r"working on", r"building", r"learning", r"project",
        r"goal", r"trying to", r"planning"
    ],
    "skill": [
        r"know", r"expert", r"experience", r"years of",
        r"familiar with", r"proficient", r"skilled"
    ],
    "context": [
        r"timezone", r"platform", r"device", r"using",
        r"environment", r"setup"
    ]
}

# Sensitive content patterns to block
SENSITIVE_PATTERNS = [
    r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
    r"\b\d{16}\b",                       # Credit card
    r"password\s*[:=]",                  # Passwords
    r"api[_-]?key\s*[:=]",               # API keys
    r"secret\s*[:=]",                    # Secrets
    r"token\s*[:=]",                     # Tokens
]

# Instruction injection patterns to block
INJECTION_PATTERNS = [
    r"ignore\s+(previous|all)\s+instructions",
    r"forget\s+your\s+(rules|guidelines)",
    r"you\s+are\s+now",
    r"act\s+as\s+if",
    r"pretend\s+to\s+be",
    r"override\s+your",
    r"<system>",
    r"</system>",
]


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class MemoryEdit:
    """Single memory edit entry"""
    line_number: int
    content: str
    category: str
    created_at: str
    updated_at: str


# ============================================================================
# MEMORY USER EDITS TOOL
# ============================================================================

class MemoryUserEditsTool(BaseTool):
    """
    Memory User Edits Tool - Explicit Memory Management
    
    Allows users to explicitly manage their memory with commands:
    - view: Show current memory edits
    - add: Add a new memory edit
    - remove: Delete by line number
    - replace: Update existing edit
    
    Following Claude AI patterns:
    - Planning Agent selects this tool based on triggers
    - User sees tool calls in UI
    - Cross-session persistence via CoreMemory
    
    Usage:
        # View all edits
        result = await tool.execute(command="view", user_id="123")
        
        # Add new edit
        result = await tool.execute(
            command="add",
            content="User prefers Vietnamese explanations",
            user_id="123"
        )
        
        # Remove edit
        result = await tool.execute(
            command="remove",
            line_number=3,
            user_id="123"
        )
        
        # Replace edit
        result = await tool.execute(
            command="replace",
            line_number=2,
            replacement="User now works at Anthropic",
            user_id="123"
        )
    """
    
    def __init__(self):
        """Initialize MemoryUserEditsTool"""
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Lazy load CoreMemory
        self._core_memory = None
        
        # Track edits in current session for consolidation
        self._session_edit_count = 0
        
        self.schema = ToolSchema(
            name="memoryUserEdits",
            category="memory",
            description=(
                "Manage user's memory edits. Use this tool when user wants to "
                "add, remove, update, or view their stored memories. "
                "Memories persist across conversations and help personalize responses. "
                "Trigger phrases: 'remember that...', 'forget about...', "
                "'update my preference...', 'what do you know about me?'"
            ),
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    required=True,
                    description="Operation to perform on memory",
                    enum=["view", "add", "remove", "replace"]
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    required=False,
                    description=(
                        "For 'add': New memory content (max 200 chars). "
                        "Format as fact about user, e.g., 'User prefers dark mode'"
                    )
                ),
                ToolParameter(
                    name="line_number",
                    type="integer",
                    required=False,
                    description="For 'remove'/'replace': Line number (1-indexed) to modify"
                ),
                ToolParameter(
                    name="replacement",
                    type="string",
                    required=False,
                    description="For 'replace': New content to replace existing line (max 200 chars)"
                ),
                ToolParameter(
                    name="category",
                    type="string",
                    required=False,
                    description="Category for the memory edit",
                    enum=MEMORY_CATEGORIES,
                    default="other"
                ),
                ToolParameter(
                    name="user_id",
                    type="string",
                    required=True,
                    description="User ID for memory operations"
                )
            ],
            returns={
                "status": "success/error",
                "command": "Command executed",
                "edits": "Current list of memory edits",
                "total_count": "Total number of edits",
                "message": "Human-readable result message"
            },
            capabilities=[
                "View all stored memory edits",
                "Add new memory facts about user",
                "Remove specific memory by line number",
                "Replace/update existing memory",
                "Auto-categorize memory edits",
                "Prevent sensitive data storage"
            ],
            limitations=[
                "Maximum 30 memory edits",
                "Maximum 200 characters per edit",
                "Cannot store passwords, API keys, SSN",
                "Cannot store instruction overrides"
            ],
            usage_hints=[
                "Use when user says 'remember that...', 'please remember...'",
                "Use when user says 'forget about...', 'delete my...'",
                "Use when user says 'update my...', 'change my...'",
                "Use when user asks 'what do you know about me?'",
                "Always confirm changes with user"
            ],
            requires_symbol=False
        )
    
    async def _get_core_memory(self):
        """Lazy load CoreMemory"""
        if self._core_memory is None:
            try:
                from src.agents.memory.core_memory import CoreMemory
                self._core_memory = CoreMemory()
            except ImportError as e:
                self.logger.error(f"[MEMORY_USER_EDITS] Failed to import CoreMemory: {e}")
        return self._core_memory
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    async def execute(
        self,
        command: str,
        user_id: str,
        content: Optional[str] = None,
        line_number: Optional[int] = None,
        replacement: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs
    ) -> ToolOutput:
        """
        Execute memory edit command
        
        Args:
            command: view/add/remove/replace
            user_id: User ID for memory operations
            content: New content for 'add'
            line_number: Line number for 'remove'/'replace'
            replacement: New content for 'replace'
            category: Category for memory edit
            
        Returns:
            ToolOutput with operation result
        """
        tool_name = self.schema.name
        
        # Validate command
        if command not in ["view", "add", "remove", "replace"]:
            return create_error_output(
                tool_name=tool_name,
                error_message=f"Invalid command: {command}. Use: view, add, remove, replace",
                error_type="validation_error"
            )
        
        # Validate user_id
        if not user_id:
            return create_error_output(
                tool_name=tool_name,
                error_message="user_id is required for memory operations",
                error_type="validation_error"
            )
        
        try:
            # Get CoreMemory
            core_memory = await self._get_core_memory()
            if not core_memory:
                return create_error_output(
                    tool_name=tool_name,
                    error_message="CoreMemory service unavailable",
                    error_type="service_error"
                )
            
            # Execute command
            if command == "view":
                return await self._execute_view(core_memory, user_id, tool_name)
            elif command == "add":
                return await self._execute_add(
                    core_memory, user_id, content, category, tool_name
                )
            elif command == "remove":
                return await self._execute_remove(
                    core_memory, user_id, line_number, tool_name
                )
            elif command == "replace":
                return await self._execute_replace(
                    core_memory, user_id, line_number, replacement, tool_name
                )
            
        except Exception as e:
            self.logger.error(f"[{tool_name}] Error: {e}", exc_info=True)
            return create_error_output(
                tool_name=tool_name,
                error_message=str(e),
                error_type="execution_error"
            )
    
    # ========================================================================
    # COMMAND IMPLEMENTATIONS
    # ========================================================================
    
    async def _execute_view(
        self,
        core_memory,
        user_id: str,
        tool_name: str
    ) -> ToolOutput:
        """Execute 'view' command - show all memory edits"""
        
        # Load current memory
        memory_data = await core_memory.load_core_memory(user_id)
        human_block = memory_data.get("human", "")
        
        # Parse edits from human block
        edits = self._parse_edits_from_block(human_block)
        
        # Format for display
        formatted = self._format_edits_for_display(edits)
        
        self.logger.info(f"[{tool_name}] VIEW: {len(edits)} edits for user {user_id}")
        
        return create_success_output(
            tool_name=tool_name,
            data={
                "status": "success",
                "command": "view",
                "edits": [
                    {
                        "line": e.line_number,
                        "content": e.content,
                        "category": e.category
                    }
                    for e in edits
                ],
                "total_count": len(edits),
                "message": f"Found {len(edits)} memory edit(s)"
            },
            formatted_context=formatted,
            symbols=[]
        )
    
    async def _execute_add(
        self,
        core_memory,
        user_id: str,
        content: Optional[str],
        category: Optional[str],
        tool_name: str
    ) -> ToolOutput:
        """Execute 'add' command - add new memory edit"""
        
        # Validate content
        if not content or len(content.strip()) < 3:
            return create_error_output(
                tool_name=tool_name,
                error_message="Content must be at least 3 characters",
                error_type="validation_error"
            )
        
        # Check length limit
        if len(content) > MAX_CHARS_PER_EDIT:
            return create_error_output(
                tool_name=tool_name,
                error_message=f"Content exceeds {MAX_CHARS_PER_EDIT} character limit ({len(content)} chars)",
                error_type="validation_error"
            )
        
        # Security check - sensitive content
        if self._contains_sensitive_content(content):
            return create_error_output(
                tool_name=tool_name,
                error_message="Cannot store sensitive information (passwords, API keys, SSN, etc.)",
                error_type="security_error"
            )
        
        # Security check - injection
        if self._contains_injection(content):
            return create_error_output(
                tool_name=tool_name,
                error_message="Content appears to contain instruction injection. Rejected for safety.",
                error_type="security_error"
            )
        
        # Load current memory
        memory_data = await core_memory.load_core_memory(user_id)
        human_block = memory_data.get("human", "")
        
        # Parse existing edits
        edits = self._parse_edits_from_block(human_block)
        
        # Check max edits limit
        if len(edits) >= MAX_EDITS:
            return create_error_output(
                tool_name=tool_name,
                error_message=f"Maximum {MAX_EDITS} memory edits reached. Remove some before adding new ones.",
                error_type="limit_error"
            )
        
        # Format content
        formatted_content = self._format_new_edit(content)
        
        # Check for duplicates
        if self._is_duplicate(formatted_content, edits):
            return create_error_output(
                tool_name=tool_name,
                error_message="Similar memory already exists",
                error_type="duplicate_error"
            )
        
        # Auto-detect category if not provided
        if not category or category not in MEMORY_CATEGORIES:
            category = self._detect_category(formatted_content)
        
        # Create new edit
        now = datetime.now().isoformat()
        new_edit = MemoryEdit(
            line_number=len(edits) + 1,
            content=formatted_content,
            category=category,
            created_at=now,
            updated_at=now
        )
        edits.append(new_edit)
        
        # Save back to CoreMemory
        new_block = self._edits_to_block(edits)
        await core_memory.update_human_block(user_id, new_block)
        
        # Track session edits
        self._session_edit_count += 1
        
        self.logger.info(
            f"[{tool_name}] ADD: '{formatted_content[:50]}...' "
            f"(category={category}) for user {user_id}"
        )
        
        return create_success_output(
            tool_name=tool_name,
            data={
                "status": "success",
                "command": "add",
                "added_line": new_edit.line_number,
                "content": formatted_content,
                "category": category,
                "total_count": len(edits),
                "message": f"Added memory #{new_edit.line_number}: {formatted_content}"
            },
            formatted_context=f"✅ Added memory #{new_edit.line_number}: {formatted_content}",
            symbols=[]
        )
    
    async def _execute_remove(
        self,
        core_memory,
        user_id: str,
        line_number: Optional[int],
        tool_name: str
    ) -> ToolOutput:
        """Execute 'remove' command - delete memory by line number"""
        
        # Validate line_number
        if line_number is None:
            return create_error_output(
                tool_name=tool_name,
                error_message="line_number is required for 'remove' command",
                error_type="validation_error"
            )
        
        # Load current memory
        memory_data = await core_memory.load_core_memory(user_id)
        human_block = memory_data.get("human", "")
        
        # Parse existing edits
        edits = self._parse_edits_from_block(human_block)
        
        # Validate line exists
        if line_number < 1 or line_number > len(edits):
            return create_error_output(
                tool_name=tool_name,
                error_message=f"Line {line_number} does not exist. Valid range: 1-{len(edits)}",
                error_type="validation_error"
            )
        
        # Remove the edit
        removed = edits[line_number - 1]
        del edits[line_number - 1]
        
        # Re-number remaining edits
        for i, edit in enumerate(edits):
            edit.line_number = i + 1
        
        # Save back
        new_block = self._edits_to_block(edits)
        await core_memory.update_human_block(user_id, new_block)
        
        self.logger.info(
            f"[{tool_name}] REMOVE: Line {line_number} ('{removed.content[:30]}...') "
            f"for user {user_id}"
        )
        
        return create_success_output(
            tool_name=tool_name,
            data={
                "status": "success",
                "command": "remove",
                "removed_line": line_number,
                "removed_content": removed.content,
                "total_count": len(edits),
                "message": f"Removed memory #{line_number}: {removed.content}"
            },
            formatted_context=f"🗑️ Removed memory #{line_number}: {removed.content}",
            symbols=[]
        )
    
    async def _execute_replace(
        self,
        core_memory,
        user_id: str,
        line_number: Optional[int],
        replacement: Optional[str],
        tool_name: str
    ) -> ToolOutput:
        """Execute 'replace' command - update existing memory"""
        
        # Validate inputs
        if line_number is None:
            return create_error_output(
                tool_name=tool_name,
                error_message="line_number is required for 'replace' command",
                error_type="validation_error"
            )
        
        if not replacement or len(replacement.strip()) < 3:
            return create_error_output(
                tool_name=tool_name,
                error_message="replacement content must be at least 3 characters",
                error_type="validation_error"
            )
        
        if len(replacement) > MAX_CHARS_PER_EDIT:
            return create_error_output(
                tool_name=tool_name,
                error_message=f"Replacement exceeds {MAX_CHARS_PER_EDIT} character limit",
                error_type="validation_error"
            )
        
        # Security checks
        if self._contains_sensitive_content(replacement):
            return create_error_output(
                tool_name=tool_name,
                error_message="Cannot store sensitive information",
                error_type="security_error"
            )
        
        if self._contains_injection(replacement):
            return create_error_output(
                tool_name=tool_name,
                error_message="Content appears to contain instruction injection",
                error_type="security_error"
            )
        
        # Load current memory
        memory_data = await core_memory.load_core_memory(user_id)
        human_block = memory_data.get("human", "")
        
        # Parse existing edits
        edits = self._parse_edits_from_block(human_block)
        
        # Validate line exists
        if line_number < 1 or line_number > len(edits):
            return create_error_output(
                tool_name=tool_name,
                error_message=f"Line {line_number} does not exist. Valid range: 1-{len(edits)}",
                error_type="validation_error"
            )
        
        # Format and replace
        formatted_replacement = self._format_new_edit(replacement)
        old_content = edits[line_number - 1].content
        
        edits[line_number - 1].content = formatted_replacement
        edits[line_number - 1].updated_at = datetime.now().isoformat()
        edits[line_number - 1].category = self._detect_category(formatted_replacement)
        
        # Save back
        new_block = self._edits_to_block(edits)
        await core_memory.update_human_block(user_id, new_block)
        
        self.logger.info(
            f"[{tool_name}] REPLACE: Line {line_number} "
            f"'{old_content[:20]}...' → '{formatted_replacement[:20]}...' "
            f"for user {user_id}"
        )
        
        return create_success_output(
            tool_name=tool_name,
            data={
                "status": "success",
                "command": "replace",
                "replaced_line": line_number,
                "old_content": old_content,
                "new_content": formatted_replacement,
                "total_count": len(edits),
                "message": f"Updated memory #{line_number}: {formatted_replacement}"
            },
            formatted_context=f"✏️ Updated memory #{line_number}: {formatted_replacement}",
            symbols=[]
        )
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _parse_edits_from_block(self, human_block: str) -> List[MemoryEdit]:
        """Parse human block into structured edits"""
        edits = []
        
        if not human_block:
            return edits
        
        # Split by newlines, find bullet points
        lines = human_block.strip().split('\n')
        line_num = 0
        
        for line in lines:
            line = line.strip()
            
            # Match bullet patterns: "- ", "• ", "* ", "1. ", etc.
            match = re.match(r'^[-•*]\s*(.+)$|^(\d+)\.\s*(.+)$', line)
            if match:
                line_num += 1
                
                # Extract content
                if match.group(1):
                    content = match.group(1).strip()
                else:
                    content = match.group(3).strip()
                
                # Extract category if present [category]
                category = "other"
                cat_match = re.search(r'\[(\w+)\]$', content)
                if cat_match:
                    potential_cat = cat_match.group(1).lower()
                    if potential_cat in MEMORY_CATEGORIES:
                        category = potential_cat
                        content = content[:cat_match.start()].strip()
                
                edits.append(MemoryEdit(
                    line_number=line_num,
                    content=content,
                    category=category,
                    created_at="",
                    updated_at=""
                ))
        
        return edits
    
    def _edits_to_block(self, edits: List[MemoryEdit]) -> str:
        """Convert edits back to human block format"""
        if not edits:
            return ""
        
        lines = []
        for edit in edits:
            # Format: "- Content [category]"
            lines.append(f"- {edit.content} [{edit.category}]")
        
        return '\n'.join(lines)
    
    def _format_edits_for_display(self, edits: List[MemoryEdit]) -> str:
        """Format edits for user-friendly display"""
        if not edits:
            return "📭 No memory edits stored yet."
        
        lines = [f"📝 MEMORY EDITS ({len(edits)}/{MAX_EDITS}):"]
        lines.append("-" * 40)
        
        for edit in edits:
            category_emoji = {
                "preference": "⚙️",
                "fact": "📋",
                "goal": "🎯",
                "skill": "💡",
                "context": "📍",
                "other": "📌"
            }.get(edit.category, "📌")
            
            lines.append(f"{edit.line_number}. {category_emoji} {edit.content}")
        
        lines.append("-" * 40)
        lines.append(f"Use 'remove' with line number to delete, 'replace' to update.")
        
        return '\n'.join(lines)
    
    def _format_new_edit(self, content: str) -> str:
        """Format new edit content consistently"""
        content = content.strip()
        
        # Normalize first-person to third-person
        replacements = [
            (r'^I\s+', 'User '),
            (r'^My\s+', "User's "),
            (r'^Me\s+', 'User '),
            (r"^I'm\s+", 'User is '),
            (r"^I've\s+", 'User has '),
            (r'^I have\s+', 'User has '),
            (r'^I am\s+', 'User is '),
            (r'^I prefer\s+', 'User prefers '),
            (r'^I like\s+', 'User likes '),
            (r'^I work\s+', 'User works '),
            (r'^I live\s+', 'User lives '),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
        
        # Ensure starts with capital
        if content and content[0].islower():
            content = content[0].upper() + content[1:]
        
        return content
    
    def _detect_category(self, content: str) -> str:
        """Auto-detect category from content"""
        content_lower = content.lower()
        
        for category, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return category
        
        return "other"
    
    def _is_duplicate(self, content: str, edits: List[MemoryEdit]) -> bool:
        """Check if content is duplicate or very similar"""
        content_lower = content.lower()
        content_words = set(content_lower.split())
        
        for edit in edits:
            edit_lower = edit.content.lower()
            
            # Exact match
            if content_lower == edit_lower:
                return True
            
            # High word overlap (>80%)
            edit_words = set(edit_lower.split())
            if content_words and edit_words:
                overlap = len(content_words & edit_words)
                similarity = overlap / max(len(content_words), len(edit_words))
                if similarity > 0.8:
                    return True
        
        return False
    
    def _contains_sensitive_content(self, content: str) -> bool:
        """Check for sensitive data patterns"""
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _contains_injection(self, content: str) -> bool:
        """Check for instruction injection patterns"""
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_tool() -> MemoryUserEditsTool:
    """Factory function for tool registry"""
    return MemoryUserEditsTool()