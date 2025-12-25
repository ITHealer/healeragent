from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"           # Not started yet
    IN_PROGRESS = "in_progress"   # Currently executing
    COMPLETED = "completed"       # Successfully completed
    FAILED = "failed"             # Failed after max retries
    SKIPPED = "skipped"           # Skipped due to dependencies


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"    # Must complete for answer
    HIGH = "high"            # Important but not critical
    MEDIUM = "medium"        # Helpful but optional
    LOW = "low"              # Nice to have


# ============================================================================
# Task Models
# ============================================================================

class ToolCall(BaseModel):
    """Tool call specification within a task"""
    model_config = ConfigDict(extra='forbid')
    
    tool_name: str = Field(description="Tool to execute")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Tool parameters"
    )
    purpose: str = Field(default="", description="Why this tool")


class Task(BaseModel):
    """
    Individual task in the plan
    """
    model_config = ConfigDict(extra='forbid')
    
    # Identity
    id: int = Field(description="Task ID (1-indexed)")
    description: str = Field(description="Human-readable task description")
    
    # Execution spec
    tools_needed: List[ToolCall] = Field(
        default_factory=list,
        description="Tools required for this task"
    )
    expected_data: List[str] = Field(
        default_factory=list,
        description="What data this task should produce"
    )
    
    # State
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current task status"
    )
    done: bool = Field(default=False, description="Task completion flag")
    
    # Results
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task execution results"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    
    # Metadata
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Task priority"
    )
    dependencies: List[int] = Field(
        default_factory=list,
        description="Task IDs this task depends on"
    )
    retry_count: int = Field(default=0, description="Number of retries")
    max_retries: int = Field(default=2, description="Max retry attempts")
    
    # Validation
    validation_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Validation confidence score"
    )
    validation_reasoning: str = Field(
        default="",
        description="Why task is/isn't complete"
    )


class TaskPlan(BaseModel):
    """
    Complete task plan for a query
    """
    model_config = ConfigDict(extra='forbid')
    
    # Tasks
    tasks: List[Task] = Field(
        default_factory=list,
        description="List of tasks to execute"
    )
    
    # Plan metadata
    query_intent: str = Field(description="What user wants")
    strategy: str = Field(
        default="sequential",
        description="Execution strategy: sequential/parallel/adaptive"
    )
    estimated_complexity: str = Field(
        default="simple",
        description="simple/moderate/complex"
    )
    
    # Symbols involved
    symbols: List[str] = Field(
        default_factory=list,
        description="Stock/crypto symbols in query"
    )
    
    # Language
    response_language: str = Field(
        default="auto",
        description="Language for response"
    )
    
    # Reasoning
    reasoning: str = Field(
        default="",
        description="Why this plan was chosen"
    )


class TaskExecutionResult(BaseModel):
    """Result of executing a single task"""
    model_config = ConfigDict(extra='forbid')
    
    task_id: int = Field(description="Task ID")
    status: TaskStatus = Field(description="Execution status")
    success: bool = Field(description="Whether task succeeded")
    
    # Data
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data produced by task"
    )

    tools_executed: List[str] = Field(
        default_factory=list,
        description="List of executed tool names in this task"
    )
    
    # Validation
    validation_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Validation result if validated"
    )
    
    # Timing
    execution_time: float = Field(default=0.0, description="Time in seconds")
    
    # Error handling
    error: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)
    
    # Next action suggestion
    suggested_next_action: str = Field(
        default="continue",
        description="continue/retry/replan/skip"
    )


class TaskPlanUpdate(BaseModel):
    """Update to task plan during execution (adaptive planning)"""
    model_config = ConfigDict(extra='forbid')
    
    action: str = Field(
        description="add_task/modify_task/remove_task/reorder"
    )
    
    # For add_task
    new_task: Optional[Task] = Field(default=None)
    insert_after: Optional[int] = Field(default=None)
    
    # For modify_task
    task_id: Optional[int] = Field(default=None)
    modifications: Dict[str, Any] = Field(default_factory=dict)
    
    # For remove_task / reorder
    task_ids: List[int] = Field(default_factory=list)
    
    reasoning: str = Field(default="", description="Why this update")


# ============================================================================
# Helper Functions
# ============================================================================

def create_simple_task(
    task_id: int,
    description: str,
    tool_name: str,
    params: Dict[str, Any],
    purpose: str = "",
    priority: TaskPriority = TaskPriority.MEDIUM
) -> Task:
    """Helper to create a simple single-tool task"""
    
    return Task(
        id=task_id,
        description=description,
        tools_needed=[
            ToolCall(
                tool_name=tool_name,
                params=params,
                purpose=purpose or description
            )
        ],
        priority=priority,
        expected_data=[f"{tool_name}_output"]
    )


def tasks_have_circular_dependencies(tasks: List[Task]) -> bool:
    """Check if tasks have circular dependencies"""
    
    def has_cycle(task_id: int, visited: set, rec_stack: set) -> bool:
        visited.add(task_id)
        rec_stack.add(task_id)
        
        # Get task
        task = next((t for t in tasks if t.id == task_id), None)
        if not task:
            return False
        
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in visited:
                if has_cycle(dep_id, visited, rec_stack):
                    return True
            elif dep_id in rec_stack:
                return True
        
        rec_stack.remove(task_id)
        return False
    
    visited = set()
    rec_stack = set()
    
    for task in tasks:
        if task.id not in visited:
            if has_cycle(task.id, visited, rec_stack):
                return True
    
    return False


def get_executable_tasks(tasks: List[Task]) -> List[Task]:
    """
    Get tasks that are ready to execute
    (dependencies satisfied, not done, not failed)
    """
    
    completed_ids = {t.id for t in tasks if t.done or t.status == TaskStatus.COMPLETED}
    
    executable = []
    for task in tasks:
        # Skip if already done or failed
        if task.done or task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED]:
            continue
        
        # Check dependencies
        deps_satisfied = all(dep_id in completed_ids for dep_id in task.dependencies)
        
        if deps_satisfied:
            executable.append(task)
    
    return executable


def estimate_plan_complexity(tasks: List[Task]) -> str:
    """Estimate overall plan complexity"""
    
    num_tasks = len(tasks)
    total_tools = sum(len(t.tools_needed) for t in tasks)
    has_deps = any(len(t.dependencies) > 0 for t in tasks)
    
    if num_tasks == 1 and total_tools == 1:
        return "simple"
    elif num_tasks <= 3 and total_tools <= 3 and not has_deps:
        return "moderate"
    else:
        return "complex"


def reorder_tasks_by_priority(tasks: List[Task]) -> List[Task]:
    """
    Reorder tasks by priority while respecting dependencies
    """
    
    # Priority order
    priority_order = {
        TaskPriority.CRITICAL: 0,
        TaskPriority.HIGH: 1,
        TaskPriority.MEDIUM: 2,
        TaskPriority.LOW: 3
    }
    
    # Group by dependencies
    no_deps = [t for t in tasks if len(t.dependencies) == 0]
    has_deps = [t for t in tasks if len(t.dependencies) > 0]
    
    # Sort each group by priority
    no_deps.sort(key=lambda t: priority_order.get(t.priority, 2))
    has_deps.sort(key=lambda t: (
        max(t.dependencies) if t.dependencies else 0,  # Respect deps
        priority_order.get(t.priority, 2)              # Then priority
    ))
    
    return no_deps + has_deps