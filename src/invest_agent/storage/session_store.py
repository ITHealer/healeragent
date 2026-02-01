"""
Session-based folder management for artifact storage.

Why: Each chat session gets its own folder under `data/sessions/{session_id}/`.
This isolation ensures artifacts from different sessions never collide, makes
cleanup straightforward (delete the folder), and supports auditing.

How: Creates session directories on-demand. Provides path resolution and
listing utilities for the ArtifactManager to use.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Default base directory for session data, relative to project root
DEFAULT_SESSIONS_BASE = "data/sessions"


class SessionStore:
    """Manages per-session directories on the filesystem.

    Why a dedicated class: Centralizes all path resolution logic. The
    ArtifactManager and ContextManager both need session paths; this
    class is the single source of truth for where session data lives.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self._base_dir = Path(base_dir or DEFAULT_SESSIONS_BASE)

    def get_session_dir(self, session_id: str) -> Path:
        """Get (and create if needed) the root directory for a session."""
        session_dir = self._base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def get_artifacts_dir(self, session_id: str) -> Path:
        """Get (and create if needed) the artifacts subdirectory for a session."""
        artifacts_dir = self.get_session_dir(session_id) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        return artifacts_dir

    def get_context_dir(self, session_id: str) -> Path:
        """Get (and create if needed) the context subdirectory for a session."""
        context_dir = self.get_session_dir(session_id) / "context"
        context_dir.mkdir(parents=True, exist_ok=True)
        return context_dir

    def list_artifacts(self, session_id: str) -> List[str]:
        """List all artifact filenames in a session's artifacts directory."""
        artifacts_dir = self._base_dir / session_id / "artifacts"
        if not artifacts_dir.exists():
            return []
        return sorted(f.name for f in artifacts_dir.iterdir() if f.is_file())

    def session_exists(self, session_id: str) -> bool:
        """Check if a session directory already exists."""
        return (self._base_dir / session_id).is_dir()

    def cleanup_session(self, session_id: str) -> bool:
        """Remove an entire session directory. Returns True if removed."""
        session_dir = self._base_dir / session_id
        if session_dir.exists():
            try:
                shutil.rmtree(session_dir)
                logger.info(f"[SessionStore] Cleaned up session: {session_id}")
                return True
            except OSError as e:
                logger.error(f"[SessionStore] Failed to clean session {session_id}: {e}")
                return False
        return False

    def get_total_size_bytes(self, session_id: str) -> int:
        """Calculate total size of all files in a session directory."""
        session_dir = self._base_dir / session_id
        if not session_dir.exists():
            return 0
        total = 0
        for f in session_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
