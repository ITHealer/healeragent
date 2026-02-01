"""
Artifact Manager - Context offloading to filesystem.

Why: When tools return large results (financial reports, multi-symbol data),
keeping the full payload in the LLM context wastes tokens and degrades quality.
The ArtifactManager saves large results to JSON files and returns a concise
summary + artifact_id for the LLM to reference. This keeps context lean while
preserving full data access when needed.

How: When a tool result exceeds `context_offload_threshold` characters:
1. Save full result to `data/sessions/{session_id}/artifacts/{artifact_id}.json`
2. Generate a brief summary of key metrics
3. Return the summary + artifact_id to the orchestrator
The LLM sees only the summary. If it needs the full data, it can request
a load via artifact_id.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.invest_agent.storage.session_store import SessionStore
from src.invest_agent.core.exceptions import ArtifactStorageError

logger = logging.getLogger(__name__)


class ArtifactRef(BaseModel):
    """Reference to a stored artifact, returned to the orchestrator."""
    artifact_id: str
    tool_name: str
    file_path: str
    summary: str = Field(description="Concise summary for LLM context")
    original_size_chars: int
    saved_at: float = Field(default_factory=time.time)


class ArtifactManager:
    """Manages saving and loading of tool result artifacts.

    Why: The orchestrator calls `maybe_offload()` after every tool execution.
    If the result is small, it passes through unchanged. If large, it's saved
    to disk and replaced with a summary. This is the core mechanism for
    preventing context window bloat.

    How it works with the Orchestrator:
    1. ToolExecutor returns raw ToolOutput
    2. Orchestrator calls `artifact_manager.maybe_offload(session_id, tool_name, result)`
    3. If offloaded: orchestrator puts summary in LLM messages, not full data
    4. If LLM needs full data later: orchestrator calls `load_artifact(artifact_id)`
    """

    def __init__(
        self,
        session_store: SessionStore,
        offload_threshold: int = 2000,
    ):
        self._session_store = session_store
        self._offload_threshold = offload_threshold
        # In-memory cache of recent artifacts for fast re-reads within a session
        self._cache: Dict[str, Any] = {}

    def maybe_offload(
        self,
        session_id: str,
        tool_name: str,
        result_data: Any,
        custom_summary: Optional[str] = None,
    ) -> tuple[bool, Any, Optional[ArtifactRef]]:
        """Decide whether to offload a tool result to disk.

        Returns:
            (was_offloaded, data_for_context, artifact_ref)
            - If not offloaded: (False, original_data, None)
            - If offloaded: (True, summary_string, ArtifactRef)
        """
        serialized = self._serialize(result_data)
        if len(serialized) <= self._offload_threshold:
            return False, result_data, None

        try:
            artifact_ref = self.save_artifact(
                session_id=session_id,
                tool_name=tool_name,
                data=result_data,
                custom_summary=custom_summary,
            )
            summary_for_context = (
                f"[Artifact saved: {artifact_ref.artifact_id}] "
                f"{artifact_ref.summary} "
                f"(Full data: {artifact_ref.original_size_chars} chars saved to disk)"
            )
            return True, summary_for_context, artifact_ref

        except ArtifactStorageError as e:
            # Non-fatal: fall back to keeping full data in context
            logger.warning(
                f"[ArtifactManager] Failed to offload {tool_name} result: {e}. "
                f"Keeping full data in context."
            )
            return False, result_data, None

    def save_artifact(
        self,
        session_id: str,
        tool_name: str,
        data: Any,
        custom_summary: Optional[str] = None,
    ) -> ArtifactRef:
        """Save tool result data to a JSON file on disk.

        Raises ArtifactStorageError on I/O failures.
        """
        artifact_id = f"{tool_name}_{uuid.uuid4().hex[:8]}"
        artifacts_dir = self._session_store.get_artifacts_dir(session_id)
        file_path = artifacts_dir / f"{artifact_id}.json"

        serialized = self._serialize(data)

        try:
            payload = {
                "artifact_id": artifact_id,
                "tool_name": tool_name,
                "timestamp": time.time(),
                "data": data if isinstance(data, (dict, list, str, int, float, bool)) else str(data),
            }
            file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, TypeError, ValueError) as e:
            raise ArtifactStorageError(f"Failed to save artifact {artifact_id}: {e}")

        summary = custom_summary or self._generate_summary(tool_name, data)

        ref = ArtifactRef(
            artifact_id=artifact_id,
            tool_name=tool_name,
            file_path=str(file_path),
            summary=summary,
            original_size_chars=len(serialized),
        )

        # Cache for fast re-reads
        self._cache[artifact_id] = data

        logger.info(
            f"[ArtifactManager] Saved artifact {artifact_id} "
            f"({len(serialized)} chars -> {len(summary)} char summary)"
        )
        return ref

    def load_artifact(self, artifact_id: str, session_id: Optional[str] = None) -> Any:
        """Load artifact data by ID. Checks cache first, then disk.

        Returns the raw data, or raises ArtifactStorageError if not found.
        """
        # Check cache first
        if artifact_id in self._cache:
            return self._cache[artifact_id]

        # Search on disk if session_id provided
        if session_id:
            artifacts_dir = self._session_store.get_artifacts_dir(session_id)
            file_path = artifacts_dir / f"{artifact_id}.json"
            if file_path.exists():
                try:
                    payload = json.loads(file_path.read_text(encoding="utf-8"))
                    data = payload.get("data")
                    self._cache[artifact_id] = data
                    return data
                except (json.JSONDecodeError, OSError) as e:
                    raise ArtifactStorageError(f"Failed to read artifact {artifact_id}: {e}")

        raise ArtifactStorageError(f"Artifact '{artifact_id}' not found")

    def _generate_summary(self, tool_name: str, data: Any) -> str:
        """Generate a concise summary of tool result data.

        Why per-tool summaries: Different tools return different key metrics.
        A stock price tool needs price/change/volume; a financial statement
        tool needs revenue/net_income/EPS. This ensures the LLM gets the
        most useful information in minimal tokens.
        """
        if not isinstance(data, dict):
            serialized = self._serialize(data)
            return f"{tool_name} returned {len(serialized)} chars of data"

        inner = data.get("data", data)

        # Price-related tools
        if "price" in tool_name.lower() or "quote" in tool_name.lower():
            return self._summarize_price(inner)

        # Financial statement tools
        if any(kw in tool_name.lower() for kw in ("income", "balance", "cashflow", "cash_flow")):
            return self._summarize_financial(inner)

        # Technical analysis tools
        if any(kw in tool_name.lower() for kw in ("technical", "indicator", "pattern", "support")):
            return self._summarize_technical(inner)

        # Risk tools
        if any(kw in tool_name.lower() for kw in ("risk", "sentiment", "volume")):
            return self._summarize_risk(inner)

        # Generic fallback: extract top-level keys and values
        return self._summarize_generic(tool_name, inner)

    @staticmethod
    def _summarize_price(data: dict) -> str:
        parts = []
        for key in ("symbol", "price", "change", "changesPercentage", "change_percent", "volume"):
            if key in data:
                parts.append(f"{key}: {data[key]}")
        return "Price data - " + ", ".join(parts) if parts else "Price data retrieved"

    @staticmethod
    def _summarize_financial(data: dict) -> str:
        parts = []
        for key in ("revenue", "netIncome", "eps", "totalAssets", "totalDebt"):
            if key in data:
                parts.append(f"{key}: {data[key]}")
        if isinstance(data, list) and data:
            return f"Financial data: {len(data)} periods retrieved"
        return "Financial data - " + ", ".join(parts) if parts else "Financial data retrieved"

    @staticmethod
    def _summarize_technical(data: dict) -> str:
        if isinstance(data, list):
            return f"Technical data: {len(data)} data points"
        parts = []
        for key in ("signal", "trend", "rsi", "macd", "sma"):
            if key in data:
                parts.append(f"{key}: {data[key]}")
        return "Technical analysis - " + ", ".join(parts) if parts else "Technical data retrieved"

    @staticmethod
    def _summarize_risk(data: dict) -> str:
        parts = []
        for key in ("risk_score", "sentiment", "risk_level", "beta"):
            if key in data:
                parts.append(f"{key}: {data[key]}")
        return "Risk data - " + ", ".join(parts) if parts else "Risk data retrieved"

    @staticmethod
    def _summarize_generic(tool_name: str, data: dict) -> str:
        if isinstance(data, list):
            return f"{tool_name}: {len(data)} items returned"
        keys = list(data.keys())[:5]
        return f"{tool_name}: keys={keys}"

    @staticmethod
    def _serialize(data: Any) -> str:
        """Serialize data to string for size measurement."""
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(data)
