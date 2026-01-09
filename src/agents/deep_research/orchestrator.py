"""
Deep Research Orchestrator (Lead Agent)

The orchestrator coordinates the entire deep research process:
1. Clarification: Ask user clarifying questions if needed
2. Planning: Create a research plan with sections
3. Execution: Spawn and coordinate worker agents
4. Synthesis: Combine worker results into final report

This is the main entry point for deep research.
"""

import uuid
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple

from src.agents.deep_research.base_agent import BaseDeepResearchAgent
from src.agents.deep_research.models import (
    AgentState,
    ResearchPlan,
    ResearchSection,
    WorkerTask,
    WorkerResult,
    WorkerRole,
    Artifact,
    ArtifactType,
    DeepResearchResult,
    DeepResearchConfig,
    ClarificationQuestion,
    ClarificationResponse,
)
from src.agents.deep_research.streaming.artifact_emitter import (
    ArtifactEmitter,
    DeepResearchStreamEvent,
)
from src.agents.deep_research.prompts.clarification import (
    CLARIFICATION_SYSTEM_PROMPT,
    generate_clarification_prompt,
)
from src.agents.deep_research.prompts.planning import (
    PLANNING_SYSTEM_PROMPT,
    generate_planning_prompt,
)
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.config import settings


class DeepResearchOrchestrator(BaseDeepResearchAgent):
    """
    Lead Agent for Deep Research.

    Coordinates the entire research process from clarification to final report.

    Usage:
        orchestrator = DeepResearchOrchestrator()

        async for event in orchestrator.run_research(
            query="Analyze NVDA for long-term investment",
            user_id=123,
        ):
            if event["type"] == "clarification_request":
                # Show questions to user
                pass
            elif event["type"] == "plan_created":
                # Show plan for confirmation
                pass
            elif event["type"] == "worker_artifact":
                # Display worker findings
                pass
            elif event["type"] == "research_completed":
                # Show final report
                pass
    """

    def __init__(
        self,
        config: Optional[DeepResearchConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            agent_id=f"orchestrator_{uuid.uuid4().hex[:8]}",
            config=config or DeepResearchConfig(),
        )

        self.api_key = api_key or settings.OPENAI_API_KEY
        self._provider = None
        self._emitter: Optional[ArtifactEmitter] = None

        # Research state - Generate research_id immediately
        self.research_id: str = f"dr_{uuid.uuid4().hex[:12]}"
        self.query: Optional[str] = None
        self.symbols: List[str] = []
        self.clarification_answers: Dict[str, str] = {}
        self.plan: Optional[ResearchPlan] = None
        self.worker_results: List[WorkerResult] = []

        # Internal state for think/act
        self._current_phase = "init"
        self._pending_action: Optional[Dict[str, Any]] = None

    @property
    def provider(self):
        """Lazy load LLM provider."""
        if self._provider is None:
            model = self.config.get_model_for_tier("lead")
            self._provider = ModelProviderFactory.create_provider(
                provider_type=ProviderType.OPENAI,
                model_name=model,
                api_key=self.api_key,
            )
        return self._provider

    async def _ensure_provider_initialized(self):
        """Ensure provider is initialized."""
        if self._provider is None:
            _ = self.provider
        if not hasattr(self._provider, '_client') or self._provider._client is None:
            await self._provider.initialize()

    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================

    async def run_research(
        self,
        query: str,
        symbols: Optional[List[str]] = None,
        user_id: Optional[int] = None,
        session_id: Optional[str] = None,
        user_context: Optional[str] = None,
        skip_clarification: bool = False,
        confirmed_plan_id: Optional[str] = None,
        clarification_response: Optional[ClarificationResponse] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the complete deep research process.

        Args:
            query: User's research query
            symbols: Pre-detected symbols (optional)
            user_id: User ID for personalization
            session_id: Session ID for tracking
            user_context: User profile/preferences context
            skip_clarification: Skip clarification phase
            confirmed_plan_id: ID of pre-confirmed plan (skip planning)
            clarification_response: User's answers to clarification questions

        Yields:
            SSE events for each phase of research
        """
        # Initialize research session (research_id already set in __init__)
        self.query = query
        self.symbols = symbols or []
        self._emitter = ArtifactEmitter(self.research_id)
        self._start_time = datetime.utcnow()

        self.logger.info(
            f"[{self.agent_id}] Starting deep research | "
            f"research_id={self.research_id} | query={query[:50]}..."
        )

        # Emit init event
        yield self._emitter.research_init(
            query=query,
            config=self.config.to_dict(),
        ).to_dict()

        try:
            await self._ensure_provider_initialized()

            # Handle clarification response if provided
            if clarification_response:
                self.clarification_answers = clarification_response.answers
                yield self._emitter.clarification_received(
                    answers=clarification_response.answers
                ).to_dict()

            # Phase 1: Clarification (if needed)
            if not skip_clarification and not self.clarification_answers and not confirmed_plan_id:
                self.state = AgentState.CLARIFYING
                async for event in self._run_clarification_phase(user_context):
                    yield event
                    # If clarification is needed, stop and wait for user response
                    if event.get("type") == "clarification_request":
                        return

            # Phase 2: Planning (if no confirmed plan)
            if not confirmed_plan_id:
                self.state = AgentState.PLANNING
                async for event in self._run_planning_phase(user_context):
                    yield event

            # Phase 3: Execution
            if self.plan and self.plan.confirmed:
                self.state = AgentState.EXECUTING
                async for event in self._run_execution_phase():
                    yield event

                # Phase 4: Synthesis
                self.state = AgentState.SYNTHESIZING
                async for event in self._run_synthesis_phase():
                    yield event

            # Complete
            self.state = AgentState.COMPLETED
            self._end_time = datetime.utcnow()

            yield self._emitter.research_completed(
                total_duration_ms=self.get_duration_ms(),
                sections_completed=len([r for r in self.worker_results if r.success]),
                sources_count=sum(len(r.sources) for r in self.worker_results),
                final_report_length=len(self.worker_results),
            ).to_dict()

        except Exception as e:
            self.state = AgentState.FAILED
            self._end_time = datetime.utcnow()
            self.logger.error(f"[{self.agent_id}] Research failed: {e}", exc_info=True)
            yield self._emitter.research_failed(
                error=str(e),
                phase=self._current_phase,
                duration_ms=self.get_duration_ms(),
            ).to_dict()

    # ========================================================================
    # PHASE 1: CLARIFICATION
    # ========================================================================

    async def _run_clarification_phase(
        self,
        user_context: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run clarification phase - ask user questions if needed.
        """
        self._current_phase = "clarification"
        self.logger.info(f"[{self.agent_id}] Starting clarification phase")

        yield self._emitter.progress(
            phase="clarification",
            progress=0.0,
            message="Analyzing query to determine if clarification is needed...",
        ).to_dict()

        # Call LLM to analyze if clarification is needed
        messages = [
            {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": generate_clarification_prompt(
                query=self.query,
                symbols=self.symbols,
                user_context=user_context,
            )},
        ]

        try:
            response = await self.provider.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
            )

            content = response.get("content", "")

            # Parse JSON response
            clarification_result = self._parse_json_response(content)

            if clarification_result.get("needs_clarification", False):
                questions = [
                    ClarificationQuestion(
                        question_id=q.get("question_id", f"q{i}"),
                        question=q.get("question", ""),
                        question_type=q.get("question_type", "single_choice"),
                        options=q.get("options", []),
                        default=q.get("default"),
                        required=q.get("required", True),
                    )
                    for i, q in enumerate(clarification_result.get("questions", []))
                ]

                if questions:
                    yield self._emitter.clarification_request(
                        questions=questions,
                        context=clarification_result.get("reasoning"),
                    ).to_dict()
                    return

            self.logger.info(f"[{self.agent_id}] No clarification needed")
            yield self._emitter.progress(
                phase="clarification",
                progress=1.0,
                message="Query is clear, proceeding to planning...",
            ).to_dict()

        except Exception as e:
            self.logger.error(f"[{self.agent_id}] Clarification error: {e}")
            # Continue without clarification on error
            yield self._emitter.progress(
                phase="clarification",
                progress=1.0,
                message="Proceeding to planning...",
            ).to_dict()

    # ========================================================================
    # PHASE 2: PLANNING
    # ========================================================================

    async def _run_planning_phase(
        self,
        user_context: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run planning phase - create research plan.
        """
        self._current_phase = "planning"
        self.logger.info(f"[{self.agent_id}] Starting planning phase")

        yield self._emitter.progress(
            phase="planning",
            progress=0.0,
            message="Creating research plan...",
        ).to_dict()

        # Call LLM to create plan
        messages = [
            {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
            {"role": "user", "content": generate_planning_prompt(
                query=self.query,
                symbols=self.symbols,
                clarification_answers=self.clarification_answers,
                user_context=user_context,
            )},
        ]

        try:
            response = await self.provider.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=2000,
            )

            content = response.get("content", "")
            plan_data = self._parse_json_response(content)

            # Create ResearchPlan
            sections = []
            for i, s in enumerate(plan_data.get("sections", [])):
                role_str = s.get("worker_role", "web_researcher")
                try:
                    role = WorkerRole(role_str)
                except ValueError:
                    role = WorkerRole.WEB_RESEARCHER

                sections.append(ResearchSection(
                    id=s.get("id", i + 1),
                    name=s.get("name", f"Section {i + 1}"),
                    description=s.get("description", ""),
                    worker_role=role,
                    tools_needed=s.get("tools_needed", []),
                    estimated_duration_sec=s.get("estimated_duration_sec", 60),
                    priority=s.get("priority", 1),
                    dependencies=s.get("dependencies", []),
                ))

            self.plan = ResearchPlan(
                research_id=self.research_id,
                title=plan_data.get("title", f"Research: {self.query[:50]}"),
                query=self.query,
                objective=plan_data.get("objective", ""),
                sections=sections,
                estimated_duration_min=plan_data.get("estimated_duration_min", 5),
                symbols=self.symbols,
            )

            yield self._emitter.progress(
                phase="planning",
                progress=0.5,
                message="Research plan created, awaiting confirmation...",
            ).to_dict()

            # Emit plan for user confirmation
            yield self._emitter.plan_created(plan=self.plan).to_dict()

            # Auto-confirm if configured
            if self.config.auto_confirm_plan:
                self.plan.confirmed = True
                self.plan.confirmed_at = datetime.utcnow()
                yield self._emitter.plan_confirmed(
                    plan_id=self.research_id,
                ).to_dict()

                yield self._emitter.progress(
                    phase="planning",
                    progress=1.0,
                    message="Plan auto-confirmed, starting execution...",
                ).to_dict()

        except Exception as e:
            self.logger.error(f"[{self.agent_id}] Planning error: {e}")
            raise

    # ========================================================================
    # PHASE 3: EXECUTION
    # ========================================================================

    async def _run_execution_phase(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run execution phase - spawn and coordinate workers.
        """
        self._current_phase = "execution"
        self.logger.info(f"[{self.agent_id}] Starting execution phase")

        if not self.plan or not self.plan.sections:
            self.logger.warning(f"[{self.agent_id}] No plan or sections to execute")
            return

        total_sections = len(self.plan.sections)
        yield self._emitter.progress(
            phase="execution",
            progress=0.0,
            message=f"Starting execution of {total_sections} research sections...",
        ).to_dict()

        # Group sections by priority for parallel execution
        priority_groups: Dict[int, List[ResearchSection]] = {}
        for section in self.plan.sections:
            priority = section.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(section)

        completed = 0
        for priority in sorted(priority_groups.keys()):
            sections = priority_groups[priority]
            self.logger.info(
                f"[{self.agent_id}] Executing priority {priority} sections: "
                f"{[s.name for s in sections]}"
            )

            # Execute sections in parallel within same priority
            tasks = []
            for section in sections:
                task = self._execute_section(section)
                tasks.append(task)

            # Run workers and yield events
            for coro in asyncio.as_completed(tasks):
                async for event in await coro:
                    yield event
                    completed += 1 / total_sections

                yield self._emitter.progress(
                    phase="execution",
                    progress=min(completed / total_sections, 0.99),
                    message=f"Completed {int(completed)} of {total_sections} sections",
                ).to_dict()

        yield self._emitter.progress(
            phase="execution",
            progress=1.0,
            message="All sections completed",
        ).to_dict()

    async def _execute_section(
        self,
        section: ResearchSection,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute a single research section using a worker.
        """
        worker_id = f"worker_{section.id}_{uuid.uuid4().hex[:6]}"

        self.logger.info(
            f"[{self.agent_id}] Spawning worker {worker_id} for section: {section.name}"
        )

        yield self._emitter.worker_spawned(
            worker_id=worker_id,
            role=section.worker_role.value if isinstance(section.worker_role, WorkerRole) else section.worker_role,
            section_id=section.id,
            section_name=section.name,
            task_description=section.description,
        ).to_dict()

        start_time = datetime.utcnow()

        try:
            # Import worker dynamically to avoid circular imports
            from src.agents.deep_research.workers.base_worker import ResearchWorker

            worker = ResearchWorker(
                worker_id=worker_id,
                role=section.worker_role,
                config=self.config,
                api_key=self.api_key,
            )

            task = WorkerTask(
                worker_id=worker_id,
                section_id=section.id,
                section_name=section.name,
                role=section.worker_role,
                prompt=section.description,
                symbols=self.symbols,
                tools=section.tools_needed,
            )

            # Run worker and collect events
            async for event in worker.execute(task):
                # Forward progress events
                if event.get("type") == "thinking":
                    yield self._emitter.worker_progress(
                        worker_id=worker_id,
                        progress=event.get("iteration", 0) / 5,
                        current_step=event.get("thought", "Processing...")[:100],
                        iteration=event.get("iteration", 0),
                    ).to_dict()

                elif event.get("type") == "artifact":
                    artifact = event.get("artifact")
                    if artifact:
                        yield self._emitter.worker_artifact(
                            worker_id=worker_id,
                            artifact=artifact,
                        ).to_dict()

            # Get worker result
            result = worker.get_result()
            result.duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self.worker_results.append(result)

            yield self._emitter.worker_completed(
                worker_id=worker_id,
                result=result,
            ).to_dict()

        except Exception as e:
            self.logger.error(f"[{self.agent_id}] Worker {worker_id} failed: {e}")
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            # Create failed result
            result = WorkerResult(
                worker_id=worker_id,
                section_id=section.id,
                section_name=section.name,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
            )
            self.worker_results.append(result)

            yield self._emitter.worker_failed(
                worker_id=worker_id,
                section_id=section.id,
                error=str(e),
                duration_ms=duration_ms,
            ).to_dict()

    # ========================================================================
    # PHASE 4: SYNTHESIS
    # ========================================================================

    async def _run_synthesis_phase(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run synthesis phase - combine worker results into final report.
        """
        self._current_phase = "synthesis"
        self.logger.info(f"[{self.agent_id}] Starting synthesis phase")

        successful_results = [r for r in self.worker_results if r.success]

        if not successful_results:
            self.logger.warning(f"[{self.agent_id}] No successful results to synthesize")
            yield self._emitter.error(
                error="No successful research results to synthesize",
                phase="synthesis",
                recoverable=False,
            ).to_dict()
            return

        yield self._emitter.synthesis_started(
            sections_to_combine=[r.section_id for r in successful_results],
        ).to_dict()

        # Import synthesis module
        from src.agents.deep_research.synthesis.report_generator import ReportGenerator

        generator = ReportGenerator(
            config=self.config,
            api_key=self.api_key,
        )

        async for event in generator.generate_report(
            query=self.query,
            plan=self.plan,
            worker_results=successful_results,
            clarification_answers=self.clarification_answers,
        ):
            if event.get("type") == "progress":
                yield self._emitter.synthesis_progress(
                    current_section=event.get("section", ""),
                    progress=event.get("progress", 0),
                ).to_dict()
            elif event.get("type") == "artifact":
                yield self._emitter.synthesis_artifact(
                    artifact=event.get("artifact"),
                ).to_dict()

    # ========================================================================
    # THINK/ACT IMPLEMENTATION (for base class compatibility)
    # ========================================================================

    async def think(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Think phase - decide what to do next.

        For orchestrator, this determines which phase to run.
        """
        if self.state == AgentState.IDLE:
            return True, "Starting research process", {"action": "start"}

        if self.state == AgentState.CLARIFYING:
            return True, "Need to ask clarification questions", {"action": "clarify"}

        if self.state == AgentState.PLANNING:
            return True, "Creating research plan", {"action": "plan"}

        if self.state == AgentState.EXECUTING:
            pending_sections = [s for s in (self.plan.sections if self.plan else [])
                                if s.status == "pending"]
            if pending_sections:
                return True, f"Executing {len(pending_sections)} sections", {
                    "action": "execute",
                    "sections": pending_sections,
                }
            return True, "All sections complete", {"action": "synthesize"}

        if self.state == AgentState.SYNTHESIZING:
            return True, "Synthesizing final report", {"action": "synthesize"}

        return False, "Research complete", None

    async def act(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Act phase - execute the planned action.

        For orchestrator, this delegates to the appropriate phase.
        """
        action = action_plan.get("action")

        if action == "clarify":
            # Run clarification (handled in run_research)
            return {"success": True, "result": "clarification_complete"}

        if action == "plan":
            # Run planning (handled in run_research)
            return {"success": True, "result": "plan_created"}

        if action == "execute":
            # Execute sections (handled in run_research)
            return {"success": True, "result": "execution_complete"}

        if action == "synthesize":
            # Run synthesis (handled in run_research)
            return {"success": True, "result": "synthesis_complete"}

        return {"success": False, "error": f"Unknown action: {action}"}

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown code blocks.
        """
        # Try to extract JSON from markdown code block
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.warning(f"[{self.agent_id}] JSON parse error: {e}")
            # Try to find JSON object in content
            brace_start = content.find("{")
            brace_end = content.rfind("}") + 1
            if brace_start >= 0 and brace_end > brace_start:
                try:
                    return json.loads(content[brace_start:brace_end])
                except json.JSONDecodeError:
                    pass
            return {}

    def confirm_plan(self, modifications: Optional[Dict[str, Any]] = None):
        """
        Confirm the research plan and apply any modifications.
        """
        if not self.plan:
            raise ValueError("No plan to confirm")

        if modifications:
            # Apply modifications to plan
            if "sections" in modifications:
                # Update sections based on modifications
                pass

        self.plan.confirmed = True
        self.plan.confirmed_at = datetime.utcnow()

    def cancel_research(self, reason: str = "User cancelled"):
        """
        Cancel the ongoing research.
        """
        self.state = AgentState.CANCELLED
        self.logger.info(f"[{self.agent_id}] Research cancelled: {reason}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_orchestrator(
    model_tier: str = "standard",
    api_key: Optional[str] = None,
) -> DeepResearchOrchestrator:
    """
    Create a configured orchestrator instance.

    Args:
        model_tier: Model tier (budget, standard, premium, enterprise)
        api_key: OpenAI API key

    Returns:
        Configured DeepResearchOrchestrator
    """
    config = DeepResearchConfig(model_tier=model_tier)
    return DeepResearchOrchestrator(config=config, api_key=api_key)
