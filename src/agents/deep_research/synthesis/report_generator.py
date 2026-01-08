"""
Report Generator for Deep Research

Combines worker results into a coherent final report.
Uses LLM to synthesize findings and generate:
- Executive summary
- Detailed analysis sections
- Key findings
- Recommendations
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, AsyncGenerator

from src.agents.deep_research.models import (
    ResearchPlan,
    WorkerResult,
    Artifact,
    ArtifactType,
    DeepResearchConfig,
)
from src.agents.deep_research.prompts.synthesis import (
    SYNTHESIS_SYSTEM_PROMPT,
    generate_synthesis_prompt,
    generate_executive_summary_prompt,
    generate_key_findings_prompt,
)
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.config import settings


class ReportGenerator(LoggerMixin):
    """
    Generates comprehensive research reports from worker results.
    """

    def __init__(
        self,
        config: Optional[DeepResearchConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__()
        self.config = config or DeepResearchConfig()
        self.api_key = api_key or settings.OPENAI_API_KEY
        self._provider = None

    @property
    def provider(self):
        """Lazy load LLM provider."""
        if self._provider is None:
            model = self.config.get_model_for_tier("synthesis")
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

    async def generate_report(
        self,
        query: str,
        plan: ResearchPlan,
        worker_results: List[WorkerResult],
        clarification_answers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate the final research report.

        Args:
            query: Original research query
            plan: Research plan
            worker_results: Results from all workers
            clarification_answers: User's clarification responses

        Yields:
            Progress and artifact events
        """
        self.logger.info(f"[ReportGenerator] Starting synthesis for {len(worker_results)} results")

        await self._ensure_provider_initialized()

        # Prepare findings for synthesis
        findings = self._prepare_findings(worker_results)

        yield {
            "type": "progress",
            "section": "preparation",
            "progress": 0.1,
        }

        # Generate main report
        report_content = await self._generate_main_report(
            query=query,
            findings=findings,
            clarification_answers=clarification_answers,
        )

        yield {
            "type": "progress",
            "section": "main_report",
            "progress": 0.5,
        }

        # Create report artifact
        report_artifact = Artifact.create(
            artifact_type=ArtifactType.SECTION,
            title="Research Report",
            content=report_content,
            metadata={
                "query": query,
                "sections_count": len(worker_results),
            },
        )

        yield {
            "type": "artifact",
            "artifact": report_artifact,
        }

        yield {
            "type": "progress",
            "section": "executive_summary",
            "progress": 0.7,
        }

        # Generate executive summary
        summary = await self._generate_executive_summary(
            query=query,
            report=report_content,
            symbols=plan.symbols if plan else [],
        )

        summary_artifact = Artifact.create(
            artifact_type=ArtifactType.SUMMARY,
            title="Executive Summary",
            content=summary,
        )

        yield {
            "type": "artifact",
            "artifact": summary_artifact,
        }

        yield {
            "type": "progress",
            "section": "key_findings",
            "progress": 0.9,
        }

        # Generate key findings
        key_findings = await self._generate_key_findings(report_content)

        findings_artifact = Artifact.create(
            artifact_type=ArtifactType.FINDING,
            title="Key Findings",
            content="\n".join([f"- {f}" for f in key_findings]),
            metadata={"findings": key_findings},
        )

        yield {
            "type": "artifact",
            "artifact": findings_artifact,
        }

        yield {
            "type": "progress",
            "section": "complete",
            "progress": 1.0,
        }

    def _prepare_findings(
        self,
        worker_results: List[WorkerResult],
    ) -> List[Dict[str, Any]]:
        """
        Prepare worker findings for synthesis.
        """
        findings = []

        for result in worker_results:
            finding = {
                "section_name": result.section_name,
                "worker_role": result.worker_id.split("_")[1] if "_" in result.worker_id else "researcher",
                "findings": result.findings,
                "data": result.data,
                "success": result.success,
            }
            findings.append(finding)

        return findings

    async def _generate_main_report(
        self,
        query: str,
        findings: List[Dict[str, Any]],
        clarification_answers: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate the main report content.
        """
        messages = [
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": generate_synthesis_prompt(
                query=query,
                worker_findings=findings,
                clarification_answers=clarification_answers,
            )},
        ]

        try:
            response = await self.provider.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
            )
            return response.get("content", "Report generation failed.")

        except Exception as e:
            self.logger.error(f"[ReportGenerator] Main report error: {e}")
            return f"Error generating report: {e}"

    async def _generate_executive_summary(
        self,
        query: str,
        report: str,
        symbols: List[str],
    ) -> str:
        """
        Generate executive summary from full report.
        """
        messages = [
            {"role": "system", "content": "You are a financial report summarizer."},
            {"role": "user", "content": generate_executive_summary_prompt(
                query=query,
                full_report=report,
                symbols=symbols,
            )},
        ]

        try:
            response = await self.provider.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
            )
            return response.get("content", "Summary generation failed.")

        except Exception as e:
            self.logger.error(f"[ReportGenerator] Summary error: {e}")
            return f"Error generating summary: {e}"

    async def _generate_key_findings(
        self,
        report: str,
    ) -> List[str]:
        """
        Extract key findings from report.
        """
        messages = [
            {"role": "system", "content": "You extract key findings from reports. Respond with JSON array."},
            {"role": "user", "content": generate_key_findings_prompt(report)},
        ]

        try:
            response = await self.provider.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )

            content = response.get("content", "[]")

            # Parse JSON array
            try:
                # Handle markdown code blocks
                if "```" in content:
                    start = content.find("[")
                    end = content.rfind("]") + 1
                    if start >= 0 and end > start:
                        content = content[start:end]

                findings = json.loads(content)
                if isinstance(findings, list):
                    return findings[:7]  # Max 7 findings
            except json.JSONDecodeError:
                pass

            return ["Analysis complete - see detailed report for findings."]

        except Exception as e:
            self.logger.error(f"[ReportGenerator] Key findings error: {e}")
            return ["Error extracting key findings."]
