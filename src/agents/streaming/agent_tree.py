import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# NODE TYPES
# ============================================================================

class NodeType(str, Enum):
    """Types of nodes in agent tree"""
    ROOT = "root"                    # Root node for entire query
    PHASE = "phase"                  # Major phase (planning, execution, etc.)
    CLASSIFICATION = "classification"  # Query classification
    PLANNING = "planning"            # Planning operations
    TOOL = "tool"                    # Tool execution
    VALIDATION = "validation"        # Validation step
    GENERATION = "generation"        # Response generation
    MEMORY = "memory"                # Memory operations
    DECISION = "decision"            # LLM decision point
    THOUGHT = "thought"              # LLM thought process


class NodeStatus(str, Enum):
    """Status of a node"""
    PENDING = "pending"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# TREE NODE
# ============================================================================

@dataclass
class TreeNode:
    """
    A single node in the agent tree
    
    Attributes:
        node_id: Unique identifier
        node_type: Type of operation
        name: Human-readable name
        parent_id: Parent node ID (None for root)
        children: List of child node IDs
        status: Current status
        start_time: When node started
        end_time: When node ended
        duration_ms: Execution duration
        metadata: Additional context
        error: Error message if failed
    """
    node_id: str
    node_type: NodeType
    name: str
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value if isinstance(self.node_type, NodeType) else self.node_type,
            "name": self.name,
            "parent_id": self.parent_id,
            "children": self.children,
            "status": self.status.value if isinstance(self.status, NodeStatus) else self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "error": self.error
        }


# ============================================================================
# AGENT TREE
# ============================================================================

class AgentTree(LoggerMixin):
    """
    Hierarchical tracking of agent execution
    
    Provides:
    - Node creation and management
    - Parent-child relationship tracking
    - Execution timing
    - Tree visualization for debugging
    
    Thread-safe for concurrent tool execution.
    """
    
    def __init__(self, flow_id: str):
        """
        Initialize agent tree
        
        Args:
            flow_id: Flow identifier for logging
        """
        super().__init__()
        
        self.flow_id = flow_id
        self.nodes: Dict[str, TreeNode] = {}
        self.root_id: Optional[str] = None
        self.current_node_id: Optional[str] = None
        
        # Create root node automatically
        self.root_id = self.start_node(
            node_type=NodeType.ROOT,
            name="query_processing",
            parent_id=None
        )
        
        self.logger.debug(f"[{flow_id}] AgentTree initialized with root: {self.root_id}")
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        return f"node_{uuid.uuid4().hex[:12]}"
    
    def start_node(
        self,
        node_type: NodeType,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Dict[str, Any] = None,
        node_id: str = None
    ) -> str:
        """
        Start a new node in the tree
        
        Args:
            node_type: Type of node
            name: Human-readable name
            parent_id: Parent node ID (uses current_node if None)
            metadata: Additional context
            node_id: Optional specific node ID
            
        Returns:
            Generated node ID
        """
        _node_id = node_id or self._generate_node_id()
        
        # Use current node as parent if not specified (except for root)
        if parent_id is None and node_type != NodeType.ROOT:
            parent_id = self.current_node_id or self.root_id
        
        # Create node
        node = TreeNode(
            node_id=_node_id,
            node_type=node_type,
            name=name,
            parent_id=parent_id,
            status=NodeStatus.STARTED,
            start_time=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.nodes[_node_id] = node
        
        # Add to parent's children
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].children.append(_node_id)
        
        # Update current node
        self.current_node_id = _node_id
        
        self.logger.debug(
            f"[{self.flow_id}] Tree node started: {name} ({node_type.value}) "
            f"[parent={parent_id}]"
        )
        
        return _node_id
    
    def update_node(
        self,
        node_id: str,
        status: NodeStatus = None,
        metadata: Dict[str, Any] = None,
        error: str = None
    ):
        """
        Update an existing node
        
        Args:
            node_id: Node to update
            status: New status
            metadata: Additional metadata to merge
            error: Error message
        """
        if node_id not in self.nodes:
            self.logger.warning(f"[{self.flow_id}] Node not found: {node_id}")
            return
        
        node = self.nodes[node_id]
        
        if status:
            node.status = status
        
        if metadata:
            node.metadata.update(metadata)
        
        if error:
            node.error = error
            node.status = NodeStatus.FAILED
    
    def end_node(
        self,
        node_id: str,
        success: bool = True,
        metadata: Dict[str, Any] = None,
        error: str = None
    ) -> int:
        """
        End a node and calculate duration
        
        Args:
            node_id: Node to end
            success: Whether operation succeeded
            metadata: Final metadata
            error: Error message if failed
            
        Returns:
            Duration in milliseconds
        """
        if node_id not in self.nodes:
            self.logger.warning(f"[{self.flow_id}] Node not found: {node_id}")
            return 0
        
        node = self.nodes[node_id]
        node.end_time = datetime.utcnow()
        
        # Calculate duration
        if node.start_time:
            node.duration_ms = int(
                (node.end_time - node.start_time).total_seconds() * 1000
            )
        
        # Update status
        node.status = NodeStatus.COMPLETED if success else NodeStatus.FAILED
        
        if error:
            node.error = error
        
        if metadata:
            node.metadata.update(metadata)
        
        # Move current_node back to parent
        if node.parent_id:
            self.current_node_id = node.parent_id
        
        self.logger.debug(
            f"[{self.flow_id}] Tree node ended: {node.name} "
            f"({node.status.value}, {node.duration_ms}ms)"
        )
        
        return node.duration_ms or 0
    
    def skip_node(self, node_id: str, reason: str = None):
        """Mark a node as skipped"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.status = NodeStatus.SKIPPED
            if reason:
                node.metadata["skip_reason"] = reason
    
    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[TreeNode]:
        """Get all children of a node"""
        if node_id not in self.nodes:
            return []
        
        return [
            self.nodes[child_id]
            for child_id in self.nodes[node_id].children
            if child_id in self.nodes
        ]
    
    def get_path_to_root(self, node_id: str) -> List[TreeNode]:
        """Get path from node to root"""
        path = []
        current_id = node_id
        
        while current_id and current_id in self.nodes:
            path.append(self.nodes[current_id])
            current_id = self.nodes[current_id].parent_id
        
        return path
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export entire tree as dictionary
        
        Returns:
            {
                "flow_id": str,
                "root_id": str,
                "total_nodes": int,
                "nodes": {node_id: node_dict, ...}
            }
        """
        return {
            "flow_id": self.flow_id,
            "root_id": self.root_id,
            "total_nodes": len(self.nodes),
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            }
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the tree
        
        Returns:
            {
                "total_nodes": int,
                "nodes_by_type": {type: count},
                "nodes_by_status": {status: count},
                "total_duration_ms": int,
                "failed_nodes": [node_names]
            }
        """
        type_counts = {}
        status_counts = {}
        total_duration = 0
        failed_nodes = []
        
        for node in self.nodes.values():
            # Count by type
            type_key = node.node_type.value if isinstance(node.node_type, NodeType) else str(node.node_type)
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
            
            # Count by status
            status_key = node.status.value if isinstance(node.status, NodeStatus) else str(node.status)
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            
            # Sum duration
            if node.duration_ms:
                total_duration += node.duration_ms
            
            # Track failures
            if node.status == NodeStatus.FAILED:
                failed_nodes.append(node.name)
        
        return {
            "total_nodes": len(self.nodes),
            "nodes_by_type": type_counts,
            "nodes_by_status": status_counts,
            "total_duration_ms": total_duration,
            "failed_nodes": failed_nodes
        }
    
    def visualize(self, include_metadata: bool = False) -> str:
        """
        Generate ASCII visualization of the tree
        
        Args:
            include_metadata: Whether to include metadata in output
            
        Returns:
            ASCII tree representation
        
        Example output:
            [ROOT] query_processing (completed, 1234ms)
            ├── [PHASE] planning (completed, 234ms)
            │   ├── [CLASSIFICATION] classify_query (completed, 100ms)
            │   └── [PLANNING] create_tasks (completed, 134ms)
            ├── [PHASE] execution (completed, 800ms)
            │   ├── [TOOL] getStockPrice (completed, 300ms)
            │   └── [TOOL] getTechnicalIndicators (completed, 500ms)
            └── [PHASE] generation (completed, 200ms)
        """
        if not self.root_id:
            return "(empty tree)"
        
        lines = []
        self._visualize_node(self.root_id, "", True, lines, include_metadata)
        return "\n".join(lines)
    
    def _visualize_node(
        self,
        node_id: str,
        prefix: str,
        is_last: bool,
        lines: List[str],
        include_metadata: bool
    ):
        """Recursive helper for tree visualization"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Build node line
        connector = "└── " if is_last else "├── "
        type_str = node.node_type.value.upper() if isinstance(node.node_type, NodeType) else str(node.node_type).upper()
        status_str = node.status.value if isinstance(node.status, NodeStatus) else str(node.status)
        duration_str = f"{node.duration_ms}ms" if node.duration_ms else "..."
        
        line = f"{prefix}{connector}[{type_str}] {node.name} ({status_str}, {duration_str})"
        
        if node.error:
            line += f" ❌ {node.error[:50]}"
        
        lines.append(line)
        
        # Add metadata if requested
        if include_metadata and node.metadata:
            meta_prefix = prefix + ("    " if is_last else "│   ")
            for key, value in node.metadata.items():
                lines.append(f"{meta_prefix}  └─ {key}: {value}")
        
        # Process children
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child_id in enumerate(node.children):
            is_last_child = i == len(node.children) - 1
            self._visualize_node(child_id, child_prefix, is_last_child, lines, include_metadata)


# ============================================================================
# TREE CONTEXT MANAGER
# ============================================================================

class TreeNodeContext:
    """
    Context manager for automatic node lifecycle management
    
    Usage:
        with TreeNodeContext(tree, "tool", "getStockPrice") as node_id:
            # Do work
            pass
        # Node automatically ended
    """
    
    def __init__(
        self,
        tree: AgentTree,
        node_type: NodeType,
        name: str,
        parent_id: str = None,
        metadata: Dict[str, Any] = None
    ):
        self.tree = tree
        self.node_type = node_type
        self.name = name
        self.parent_id = parent_id
        self.metadata = metadata
        self.node_id: Optional[str] = None
        self.success = True
        self.error: Optional[str] = None
    
    def __enter__(self) -> str:
        """Start node and return node_id"""
        self.node_id = self.tree.start_node(
            node_type=self.node_type,
            name=self.name,
            parent_id=self.parent_id,
            metadata=self.metadata
        )
        return self.node_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End node with appropriate status"""
        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)
        
        self.tree.end_node(
            node_id=self.node_id,
            success=self.success,
            error=self.error
        )
        
        # Don't suppress exceptions
        return False
    
    def fail(self, error: str):
        """Mark node as failed"""
        self.success = False
        self.error = error


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_tree_for_request(flow_id: str) -> AgentTree:
    """
    Create a new agent tree for a request
    
    Args:
        flow_id: Flow identifier
        
    Returns:
        Initialized AgentTree
    """
    return AgentTree(flow_id=flow_id)


def merge_trees(main_tree: AgentTree, sub_tree: AgentTree, parent_node_id: str):
    """
    Merge a sub-tree into the main tree
    
    Useful for parallel execution where each worker has its own tree
    """
    if not sub_tree.root_id:
        return
    
    # Get sub-tree root
    sub_root = sub_tree.nodes.get(sub_tree.root_id)
    if not sub_root:
        return
    
    # Re-parent sub-tree root to parent_node
    sub_root.parent_id = parent_node_id
    
    # Add all nodes from sub-tree
    for node_id, node in sub_tree.nodes.items():
        main_tree.nodes[node_id] = node
    
    # Add sub-tree root as child of parent
    if parent_node_id in main_tree.nodes:
        main_tree.nodes[parent_node_id].children.append(sub_tree.root_id)