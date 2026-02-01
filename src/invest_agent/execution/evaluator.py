"""
Data sufficiency evaluator for the Thinking mode's evaluation loop.

Why: In Thinking mode, the agent must decide whether gathered tool results
are sufficient to answer the user's query. Without this, the agent either
stops too early (incomplete answer) or loops endlessly (wasting tokens).
The evaluator makes an explicit "data sufficient?" decision after each
execution turn.

How: Uses a lightweight LLM call (nano model) to assess completeness.
The prompt includes the original query, gathered data summaries, and
asks for a structured yes/no + missing data list.
"""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.invest_agent.core.exceptions import EvaluationError

logger = logging.getLogger(__name__)


class EvaluationResult(BaseModel):
    """Result of a data-sufficiency evaluation."""
    is_sufficient: bool = Field(description="Whether gathered data can answer the query")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    missing_data: List[str] = Field(
        default_factory=list,
        description="List of data points still needed"
    )
    suggested_tools: List[str] = Field(
        default_factory=list,
        description="Tools to call next if data is insufficient"
    )
    reasoning: str = Field(default="", description="Why the evaluation reached this conclusion")


class DataEvaluator:
    """Evaluates whether gathered tool data is sufficient to answer a query.

    Why: The orchestrator calls this after each execution turn in Thinking mode.
    If sufficient, the loop breaks and moves to synthesis. If not, the evaluator
    suggests what's missing and which tools to try next.

    How: Two strategies:
    1. Heuristic evaluation (fast, no LLM call) - based on tool success rates
    2. LLM evaluation (slower, more accurate) - asks a nano model to assess

    The orchestrator uses heuristic first and only falls back to LLM if unsure.
    """

    def __init__(self, max_eval_iterations: int = 3):
        self._max_eval_iterations = max_eval_iterations

    def evaluate_heuristic(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        iteration: int,
    ) -> EvaluationResult:
        """Fast heuristic evaluation without LLM call.

        Rules:
        1. If all tools succeeded and at least one returned data -> sufficient
        2. If > 50% tools failed and iteration < max -> insufficient
        3. If iteration >= max -> force sufficient (bail-out)
        4. If no tools were called -> insufficient
        """
        if iteration >= self._max_eval_iterations:
            return EvaluationResult(
                is_sufficient=True,
                confidence=0.4,
                reasoning=f"Bail-out: reached max evaluation iterations ({self._max_eval_iterations})",
            )

        if not tool_results:
            return EvaluationResult(
                is_sufficient=False,
                confidence=0.8,
                missing_data=["No tool results gathered yet"],
                reasoning="No tools have been executed",
            )

        total = len(tool_results)
        succeeded = sum(1 for r in tool_results if r.get("success", False))
        failed = total - succeeded
        has_data = any(
            r.get("data") or r.get("formatted_context")
            for r in tool_results
            if r.get("success", False)
        )

        # If enough tools succeeded and we have data
        if succeeded > 0 and has_data and (failed / total) < 0.5:
            return EvaluationResult(
                is_sufficient=True,
                confidence=0.7 + (0.3 * succeeded / total),
                reasoning=f"{succeeded}/{total} tools succeeded with data",
            )

        # High failure rate
        if failed > 0 and (failed / total) >= 0.5:
            failed_tools = [
                r.get("tool_name", "unknown")
                for r in tool_results
                if not r.get("success", False)
            ]
            return EvaluationResult(
                is_sufficient=False,
                confidence=0.6,
                missing_data=[f"Tool {t} failed" for t in failed_tools],
                reasoning=f"High failure rate: {failed}/{total} tools failed",
            )

        # Partial data - let the next iteration try more
        return EvaluationResult(
            is_sufficient=False,
            confidence=0.5,
            missing_data=["Partial data gathered, may need additional tools"],
            reasoning="Some data gathered but may be incomplete",
        )

    async def evaluate_with_llm(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        iteration: int,
        llm_caller: Optional[Any] = None,
    ) -> EvaluationResult:
        """LLM-based evaluation for more accurate assessment.

        Falls back to heuristic if LLM call fails (defensive).

        Why LLM: The heuristic can't understand query semantics. A user asking
        "Compare AAPL and MSFT fundamentals" needs data for BOTH symbols;
        the heuristic only checks success rates, not completeness.
        """
        if llm_caller is None:
            return self.evaluate_heuristic(query, tool_results, iteration)

        try:
            prompt = self._build_evaluation_prompt(query, tool_results, iteration)

            response = await llm_caller(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            return self._parse_evaluation_response(response, iteration)

        except Exception as e:
            logger.warning(
                f"[Evaluator] LLM evaluation failed: {e}. Falling back to heuristic."
            )
            return self.evaluate_heuristic(query, tool_results, iteration)

    def _build_evaluation_prompt(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        iteration: int,
    ) -> str:
        """Build the evaluation prompt for the LLM."""
        results_summary = []
        for r in tool_results:
            status = "OK" if r.get("success") else "FAILED"
            name = r.get("tool_name", "unknown")
            context = r.get("formatted_context", "")[:200]
            results_summary.append(f"- {name}: {status} | {context}")

        results_text = "\n".join(results_summary) if results_summary else "No results yet"

        return f"""You are evaluating whether gathered data is sufficient to answer a user query.

User Query: {query}
Iteration: {iteration}/{self._max_eval_iterations}
Tool Results:
{results_text}

Respond in JSON:
{{
  "is_sufficient": true/false,
  "confidence": 0.0-1.0,
  "missing_data": ["list of what's still needed"],
  "suggested_tools": ["tool names to call next"],
  "reasoning": "brief explanation"
}}"""

    def _parse_evaluation_response(
        self,
        response: str,
        iteration: int,
    ) -> EvaluationResult:
        """Parse LLM evaluation response into structured result."""
        import json

        try:
            # Extract JSON from response (handle markdown code blocks)
            text = response.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            return EvaluationResult(
                is_sufficient=data.get("is_sufficient", False),
                confidence=data.get("confidence", 0.5),
                missing_data=data.get("missing_data", []),
                suggested_tools=data.get("suggested_tools", []),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"[Evaluator] Failed to parse LLM response: {e}")
            # Bail out to sufficient on parse failure
            return EvaluationResult(
                is_sufficient=(iteration >= self._max_eval_iterations - 1),
                confidence=0.4,
                reasoning=f"Parse error, defaulting based on iteration {iteration}",
            )
