"""
Base Research Worker

Base class for all research workers. Workers are specialized agents
that handle specific sections of the research plan.

Each worker:
1. Receives a task with section details
2. Uses tools to gather data
3. Synthesizes findings
4. Returns structured results

Workers use the think/act pattern from BaseDeepResearchAgent.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator, Tuple

from src.agents.deep_research.base_agent import ReActAgent
from src.agents.deep_research.models import (
    AgentState,
    WorkerTask,
    WorkerResult,
    WorkerRole,
    Artifact,
    ArtifactType,
    DeepResearchConfig,
)
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.config import settings


# Worker role to prompt mapping
WORKER_PROMPTS = {
    WorkerRole.MARKET_ANALYST: """You are a Market Analyst specialized in market position analysis.

Your role is to:
- Analyze market trends and sector performance
- Evaluate competitive positioning
- Identify market opportunities and threats
- Assess industry dynamics

Focus on providing actionable market insights based on the data gathered.""",

    WorkerRole.FINANCIAL_ANALYST: """You are a Financial Analyst specialized in fundamental analysis.

Your role is to:
- Analyze financial statements (income, balance sheet, cash flow)
- Calculate and interpret financial ratios
- Assess profitability, liquidity, and solvency
- Evaluate valuation metrics

Focus on providing data-driven financial insights.""",

    WorkerRole.TECHNICAL_ANALYST: """You are a Technical Analyst specialized in chart analysis.

Your role is to:
- Analyze price charts and patterns
- Interpret technical indicators (RSI, MACD, etc.)
- Identify support and resistance levels
- Assess trend strength and momentum

Focus on providing actionable technical trading signals.""",

    WorkerRole.WEB_RESEARCHER: """You are a Web Researcher specialized in news and information gathering.

Your role is to:
- Search for relevant news and articles
- Gather analyst opinions and ratings
- Find recent company announcements
- Compile relevant web sources

Focus on providing comprehensive, up-to-date information.""",

    WorkerRole.INDUSTRY_EXPERT: """You are an Industry Expert specialized in competitive analysis.

Your role is to:
- Analyze competitive landscape
- Evaluate industry trends
- Assess company positioning vs peers
- Identify competitive advantages

Focus on providing strategic competitive insights.""",

    WorkerRole.RISK_ANALYST: """You are a Risk Analyst specialized in risk assessment.

Your role is to:
- Identify potential risks and threats
- Assess volatility and downside scenarios
- Evaluate risk/reward profiles
- Analyze sentiment indicators

Focus on providing comprehensive risk analysis.""",
}


class ResearchWorker(ReActAgent):
    """
    Base worker agent for executing research sections.

    Workers gather data using tools and synthesize findings
    for their assigned section.
    """

    def __init__(
        self,
        worker_id: str,
        role: WorkerRole,
        config: Optional[DeepResearchConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            agent_id=worker_id,
            config=config or DeepResearchConfig(),
        )

        self.role = role
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._provider = None

        # Task and result tracking
        self.task: Optional[WorkerTask] = None
        self._result: Optional[WorkerResult] = None
        self._findings: List[str] = []
        self._data: Dict[str, Any] = {}
        self._sources: List[Dict[str, Any]] = []
        self._artifacts: List[Artifact] = []

        # Execution state
        self._tools_called: List[str] = []
        self.max_iterations = 5

    @property
    def provider(self):
        """Lazy load LLM provider."""
        if self._provider is None:
            model = self.config.get_model_for_tier("worker")
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
    # MAIN EXECUTION
    # ========================================================================

    async def execute(
        self,
        task: WorkerTask,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the assigned research task.

        Args:
            task: WorkerTask with section details

        Yields:
            Events during execution (thinking, artifacts, etc.)
        """
        self.task = task
        self._start_time = datetime.utcnow()
        self.state = AgentState.EXECUTING

        self.logger.info(
            f"[{self.agent_id}] Starting execution | "
            f"section={task.section_name} | role={self.role.value}"
        )

        try:
            await self._ensure_provider_initialized()

            # Run the think/act loop
            async for event in self.run():
                yield event

            # Create final result
            self._result = WorkerResult(
                worker_id=self.agent_id,
                section_id=task.section_id,
                section_name=task.section_name,
                success=self.state == AgentState.COMPLETED,
                findings=self._compile_findings(),
                data=self._data,
                sources=self._sources,
                artifacts=self._artifacts,
                duration_ms=self.get_duration_ms(),
                iterations_used=self.current_iteration,
                tools_called=self._tools_called,
            )

        except Exception as e:
            self.logger.error(f"[{self.agent_id}] Execution error: {e}")
            self.state = AgentState.FAILED
            self._result = WorkerResult(
                worker_id=self.agent_id,
                section_id=task.section_id,
                section_name=task.section_name,
                success=False,
                error=str(e),
                duration_ms=self.get_duration_ms(),
                iterations_used=self.current_iteration,
                tools_called=self._tools_called,
            )

    def get_result(self) -> WorkerResult:
        """Get the execution result."""
        if self._result is None:
            raise ValueError("No result available - execute() not called or still running")
        return self._result

    # ========================================================================
    # THINK/ACT IMPLEMENTATION
    # ========================================================================

    async def think(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Think phase - decide what tools to call or if done.
        """
        if not self.task:
            return False, "No task assigned", None

        # Check if we have enough data
        if self._has_sufficient_data():
            return False, "Sufficient data gathered, ready to compile findings", None

        # Determine what tools to call next
        tools_to_call = await self._determine_tools()

        if not tools_to_call:
            return False, "No more tools needed", None

        return True, f"Need to call tools: {[t['name'] for t in tools_to_call]}", {
            "action": "call_tools",
            "tools": tools_to_call,
        }

    async def act(self, action_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Act phase - execute tools and gather data.
        """
        action = action_plan.get("action")

        if action == "call_tools":
            tools = action_plan.get("tools", [])
            results = await self._execute_tools(tools)
            return {"success": True, "tool_results": results}

        return {"success": False, "error": f"Unknown action: {action}"}

    # ========================================================================
    # TOOL EXECUTION
    # ========================================================================

    async def _determine_tools(self) -> List[Dict[str, Any]]:
        """
        Determine which tools to call based on task and current data.
        """
        if not self.task:
            return []

        # Get available tools for this section
        available_tools = self.task.tools or self._get_default_tools()

        # Filter out already called tools (simple dedup)
        tools_to_call = []
        for tool_name in available_tools:
            if tool_name not in self._tools_called:
                tools_to_call.append({
                    "name": tool_name,
                    "params": self._get_tool_params(tool_name),
                })

        return tools_to_call[:3]  # Max 3 tools per iteration

    async def _execute_tools(
        self,
        tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Execute a list of tools and collect results.
        """
        results = []

        for tool_spec in tools:
            tool_name = tool_spec.get("name")
            params = tool_spec.get("params", {})

            self.logger.info(f"[{self.agent_id}] Calling tool: {tool_name}")
            self._tools_called.append(tool_name)

            try:
                result = await self.call_tool(tool_name, params)
                results.append(result)

                # Store data from successful calls
                if result.get("success") and result.get("data"):
                    self._data[tool_name] = result["data"]

                    # Create artifact for findings
                    artifact = Artifact.create(
                        artifact_type=ArtifactType.DATA,
                        title=f"Data from {tool_name}",
                        content=str(result["data"])[:1000],
                        worker_id=self.agent_id,
                        section_id=self.task.section_id if self.task else None,
                    )
                    self._artifacts.append(artifact)

            except Exception as e:
                self.logger.error(f"[{self.agent_id}] Tool {tool_name} failed: {e}")
                results.append({"name": tool_name, "success": False, "error": str(e)})

        return results

    def _get_tool_params(self, tool_name: str) -> Dict[str, Any]:
        """
        Get parameters for a tool based on task context.
        """
        params = {}

        # Add symbols if tool needs them
        if self.task and self.task.symbols:
            symbol_tools = [
                "get_stock_price", "get_technical_indicators", "get_financial_ratios",
                "get_stock_news", "get_support_resistance", "assess_risk",
                "get_income_statement", "get_balance_sheet", "get_cash_flow",
            ]
            if tool_name in symbol_tools:
                params["symbol"] = self.task.symbols[0]  # Primary symbol

        # Tool-specific params
        if tool_name == "web_search" and self.task:
            params["query"] = f"{' '.join(self.task.symbols)} {self.task.prompt}"

        return params

    def _get_default_tools(self) -> List[str]:
        """
        Get default tools for this worker role.
        """
        role_tools = {
            WorkerRole.MARKET_ANALYST: [
                "web_search", "get_market_news", "get_sector_performance"
            ],
            WorkerRole.FINANCIAL_ANALYST: [
                "get_financial_ratios", "get_income_statement", "get_balance_sheet"
            ],
            WorkerRole.TECHNICAL_ANALYST: [
                "get_stock_price", "get_technical_indicators", "get_support_resistance"
            ],
            WorkerRole.WEB_RESEARCHER: [
                "web_search", "get_stock_news"
            ],
            WorkerRole.INDUSTRY_EXPERT: [
                "web_search", "get_sector_performance"
            ],
            WorkerRole.RISK_ANALYST: [
                "assess_risk", "get_sentiment", "get_volume_profile"
            ],
        }
        return role_tools.get(self.role, ["web_search"])

    # ========================================================================
    # HELPERS
    # ========================================================================

    def _has_sufficient_data(self) -> bool:
        """
        Check if we have gathered sufficient data.
        """
        # Have we called at least some tools?
        if len(self._tools_called) < 1:
            return False

        # Have we exceeded max iterations?
        if self.current_iteration >= self.max_iterations:
            return True

        # Have we gathered data?
        if self._data:
            return True

        return False

    def _compile_findings(self) -> str:
        """
        Compile gathered data into findings markdown.
        """
        if not self._data:
            return "No data gathered."

        lines = [
            f"## {self.task.section_name if self.task else 'Research'} Findings",
            "",
        ]

        for tool_name, data in self._data.items():
            lines.append(f"### {tool_name.replace('_', ' ').title()}")

            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"- **{key}**: {value:,.2f}")
                    else:
                        lines.append(f"- **{key}**: {value}")
            else:
                lines.append(str(data)[:500])

            lines.append("")

        return "\n".join(lines)
