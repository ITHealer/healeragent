import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.agents.planning.task_models import (
    Task,
    TaskPlan,
    ToolCall,
    TaskPriority,
    TaskStatus,
)
from src.prompts.planning_prompts import (
    get_planning_prompt_builder,
    PlanningPromptBuilder
)

try:
    from src.agents.tools import get_registry
    ATOMIC_TOOLS_AVAILABLE = True
except ImportError:
    ATOMIC_TOOLS_AVAILABLE = False


# Valid tool categories for classification
VALID_CATEGORIES = [
    "price",        # Stock prices, quotes, performance
    "technical",    # Technical indicators, chart patterns
    "fundamentals", # P/E, ROE, financial ratios
    "news",         # News, events, announcements
    "market",       # Market overview, indices, trending
    "risk",         # Risk assessment, volatility
    "crypto",       # Cryptocurrency
    "discovery",    # Stock screening
    "memory",       # Cross-session memory search
]

# Categories indicating strong financial intent (not contextual)
STRONG_FINANCIAL_CATEGORIES = ["discovery", "risk", "technical", "fundamentals"]


class ModelCapability(Enum):
    """Model capability levels for adaptive prompting."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class PlanningAgent(LoggerMixin):
    """
    3-Stage Planning Agent with Working Memory Support.

    Flow:
    1. Classify - LLM determines query type and categories
    2. Load Tools - Registry lookup by categories (no LLM)
    3. Create Plan - LLM generates task plan with tool schemas
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider_type: str = None,
    ):
        super().__init__()

        self.model_name = model_name or "gpt-4.1-nano"
        self.provider_type = provider_type or ProviderType.OPENAI
        self.capability = self._detect_model_capability(self.model_name)

        self.logger.info(
            f"[PLAN:INIT] model={self.model_name} | capability={self.capability.value}"
        )

        # LLM and prompts
        self.llm_provider = LLMGeneratorProvider()
        self.api_key = ModelProviderFactory._get_api_key(self.provider_type)
        self.prompt_builder = get_planning_prompt_builder()

        # Tool registry
        self.tool_registry = None
        if ATOMIC_TOOLS_AVAILABLE:
            try:
                self.tool_registry = get_registry()
            except Exception as e:
                self.logger.error(f"[PLAN:INIT] Registry failed: {e}")
    
    def _detect_model_capability(self, model_name: str) -> ModelCapability:
        """Detect model capability level from model name."""
        model_lower = model_name.lower()

        if "gpt-5" in model_lower or "gpt5" in model_lower:
            return ModelCapability.ADVANCED
        elif "gpt-4o" in model_lower or "gpt4o" in model_lower:
            return ModelCapability.INTERMEDIATE
        else:
            return ModelCapability.BASIC

    async def think_and_plan(
        self,
        query: str,
        recent_chat: List[Dict[str, str]] = None,
        core_memory: Dict[str, Any] = None,
        summary: str = None,
        working_memory_context: str = None,
        pre_classification: Dict[str, Any] = None,
    ) -> TaskPlan:
        """
        Main planning method - 3 Stage Flow.

        Args:
            query: User query
            recent_chat: Conversation history
            core_memory: User memory/preferences
            summary: Conversation summary
            working_memory_context: Current session context
            pre_classification: Optional pre-computed classification from UnifiedClassifier
                               If provided, skips Stage 1 (classification) to avoid duplicate LLM calls.

        Returns TaskPlan with tasks to execute.
        """
        start_time = datetime.now()
        query_preview = query[:80] + '...' if len(query) > 80 else query

        try:
            self.logger.info(f"[PLAN:START] query=\"{query_preview}\"")

            if working_memory_context:
                self.logger.debug(f"[PLAN:START] Working memory: {len(working_memory_context)} chars")

            # Phase 0: Validate and format history
            formatted_history = self._validate_and_format_history(recent_chat)
            self.logger.debug(f"[PLAN:P0] History: {len(formatted_history)} messages")

            # Stage 1: Use pre-classification if provided, otherwise classify
            if pre_classification:
                # REUSE classification from UnifiedClassifier (avoid duplicate LLM call)
                self.logger.info(f"[PLAN:S1] Using pre-computed classification (skipping LLM)")
                classification = pre_classification
            else:
                # Fallback: Run classification (for backward compatibility)
                self.logger.info(f"[PLAN:S1] Classifying query...")
                classification = await self._stage1_classify(
                    query=query,
                    history=formatted_history,
                    working_memory_context=working_memory_context,
                    core_memory=core_memory
                )

            query_type = classification.get("query_type", "stock_specific")
            categories = classification.get("categories", classification.get("tool_categories", []))
            symbols = classification.get("symbols", [])
            response_language = classification.get("response_language", "auto")
            reasoning = classification.get("reasoning", "")

            self._log_stage1_result(query_type, categories, symbols, reasoning, response_language)

            # Safeguard: Override conversational if strong financial signals present
            should_override = False
            override_reason = ""

            if query_type == "conversational":
                if symbols:
                    should_override = True
                    override_reason = f"symbols={symbols}"
                    query_type = "stock_specific"
                elif any(cat in STRONG_FINANCIAL_CATEGORIES for cat in categories):
                    should_override = True
                    strong_cats = [c for c in categories if c in STRONG_FINANCIAL_CATEGORIES]
                    override_reason = f"strong_categories={strong_cats}"
                    query_type = "screener" if "discovery" in categories else "stock_specific"

            if should_override:
                self.logger.warning(
                    f"[PLAN:S1] OVERRIDE conversational -> {query_type} | {override_reason}"
                )

            # Early exit: Conversational query (no tools needed)
            if query_type == "conversational":
                elapsed = self._get_elapsed_ms(start_time)
                self.logger.info(f"[PLAN:S1] No tools needed | {elapsed}ms")

                return TaskPlan(
                    tasks=[],
                    query_intent=query_type,
                    strategy="direct_answer",
                    response_language=response_language,
                    reasoning="Conversational query - no tools needed"
                )

            # Early exit: No categories selected
            if not categories:
                elapsed = self._get_elapsed_ms(start_time)
                self.logger.info(f"[PLAN:S1] No categories | {elapsed}ms")

                return TaskPlan(
                    tasks=[],
                    query_intent=query_type,
                    strategy="direct_answer",
                    response_language=response_language,
                    reasoning="No relevant categories for this query"
                )
            
            # Thinking: Validate if tools are actually needed
            self.logger.debug(f"[PLAN:THINK] Validating necessity...")

            thinking = await self._thinking_validate_necessity(
                query, query_type, categories, symbols, reasoning
            )

            self.logger.debug(f"[PLAN:THINK] need_tools={thinking['need_tools']}")

            # Override think result only for genuine financial queries
            if not thinking["need_tools"]:
                has_symbols = bool(symbols)
                has_strong_financial_intent = query_type in ["stock_specific", "screener"]
                has_strong_categories = any(cat in STRONG_FINANCIAL_CATEGORIES for cat in categories)

                if has_symbols or (has_strong_financial_intent and has_strong_categories):
                    self.logger.warning(f"[PLAN:THINK] OVERRIDE | Financial query detected, using tools")
                else:
                    elapsed = self._get_elapsed_ms(start_time)
                    self.logger.info(f"[PLAN:THINK] No tools needed | {elapsed}ms")
                    return TaskPlan(
                        tasks=[],
                        query_intent=thinking["final_intent"],
                        strategy="direct_answer",
                        response_language=response_language,
                        reasoning=thinking["reason"]
                    )

            # Stage 2: Load tools from registry
            self.logger.info(f"[PLAN:S2] Loading tools for categories={categories}")

            available_tools = self._stage2_load_tools(categories)

            if not available_tools:
                self.logger.warning(f"[PLAN:S2] No tools found")
                elapsed = self._get_elapsed_ms(start_time)

                return TaskPlan(
                    tasks=[],
                    query_intent=thinking["final_intent"],
                    strategy="direct_answer",
                    response_language=response_language,
                    reasoning="No tools available for selected categories"
                )

            # Stage 3: Create task plan
            self.logger.info(f"[PLAN:S3] Creating plan with {len(available_tools)} tools...")

            context = self._build_context_string(
                formatted_history, core_memory, summary, working_memory_context
            )

            plan_dict = await self._stage3_create_plan(
                query=query,
                query_type=query_type,
                symbols=symbols,
                response_language=response_language,
                available_tools=available_tools,
                context=context,
                validated_intent=thinking["final_intent"]
            )

            # Parse into TaskPlan
            task_plan = self._parse_task_plan_dict(plan_dict, symbols, response_language)
            task_plan.query_intent = thinking["final_intent"]

            # Log final result
            elapsed = self._get_elapsed_ms(start_time)
            self._log_plan_result(task_plan, elapsed)

            return task_plan

        except Exception as e:
            self.logger.error(f"[PLAN:ERROR] {e}", exc_info=True)

            return TaskPlan(
                tasks=[],
                query_intent="error",
                strategy="direct_answer",
                response_language="auto",
                reasoning=f"Planning error: {str(e)}"
            )
    
    # --- Stage 1: Classification ---

    async def _stage1_classify(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None,
        core_memory: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Classify query semantically using LLM.

        Uses Working Memory for context: current symbols, previous intent, task state.
        """
        # Select classifier based on model capability
        if self.capability == ModelCapability.ADVANCED:
            result = await self._classify_advanced(
                query, history, working_memory_context, core_memory
            )
        elif self.capability == ModelCapability.INTERMEDIATE:
            result = await self._classify_intermediate(
                query, history, working_memory_context, core_memory
            )
        else:
            result = await self._classify_basic(
                query, history, working_memory_context, core_memory
            )

        # Validate and filter categories
        result = self._validate_categories_output(result)

        # Auto-add price/news if symbols present
        if result.get("symbols"):
            if "price" not in result.get("categories", []):
                result["categories"].append("price")
                self.logger.debug(f"[PLAN:S1] Auto-added 'price' category")
            if "news" not in result.get("categories", []):
                result["categories"].append("news")

        return result

    async def _classify_basic(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None,
        core_memory: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Classification for basic models."""
        history_text = self._format_history_for_prompt(history, max_messages=4)

        prompt = self.prompt_builder.build_classify_prompt(
            query=query,
            history_text=history_text,
            capability="basic",
            working_memory_context=working_memory_context,
            core_memory=core_memory
        )

        return await self._call_llm_json(prompt, "Stage1")

    async def _classify_intermediate(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None,
        core_memory: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Classification for intermediate models."""
        history_text = self._format_history_for_prompt(history, max_messages=6)

        prompt = self.prompt_builder.build_classify_prompt(
            query=query,
            history_text=history_text,
            capability="intermediate",
            working_memory_context=working_memory_context,
            core_memory=core_memory
        )

        return await self._call_llm_json(prompt, "Stage1")

    async def _classify_advanced(
        self,
        query: str,
        history: List[Dict[str, str]],
        working_memory_context: str = None,
        core_memory: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Classification for advanced models with CoT."""
        history_text = self._format_history_for_prompt(history, max_messages=6)

        prompt = self.prompt_builder.build_classify_prompt(
            query=query,
            history_text=history_text,
            capability="advanced",
            working_memory_context=working_memory_context,
            core_memory=core_memory
        )

        return await self._call_llm_json(prompt, "Stage1")

    # --- Stage 2: Load Tools ---

    def _stage2_load_tools(self, categories: List[str]) -> List[Dict[str, Any]]:
        """
        Load tools from registry by categories (no LLM call).

        Progressive disclosure: only load tools for selected categories.
        """
        if not self.tool_registry:
            self.logger.error(f"[PLAN:S2] Tool registry not available")
            return []

        available_tools = []
        tools_per_category = {}

        for category in categories:
            tools_dict = self.tool_registry.get_tools_by_category(category)

            if not tools_dict:
                self.logger.debug(f"[PLAN:S2] Category '{category}' has no tools")
                continue

            category_tools = []

            for tool_name in tools_dict:
                schema = self.tool_registry.get_schema(tool_name)
                if schema:
                    tool_schema = schema.to_json_schema()
                    available_tools.append(tool_schema)
                    category_tools.append(tool_name)

            tools_per_category[category] = category_tools

        # Log tools per category
        for cat, tools in tools_per_category.items():
            self.logger.debug(f"[PLAN:S2] {cat}: {tools}")

        if self.tool_registry:
            total = len(self.tool_registry.get_all_tools())
            self.logger.info(f"[PLAN:S2] Loaded {len(available_tools)}/{total} tools")

        return available_tools

    # --- Thinking Layer ---

    async def _thinking_validate_necessity(
        self,
        query: str,
        query_type: str,
        categories: List[str],
        symbols: List[str],
        reasoning: str
    ) -> Dict[str, Any]:
        """
        Validate if tools are actually needed.

        Prevents false negatives where LLM incorrectly says "no tools" for financial queries.
        """
        prompt = self.prompt_builder.build_thinking_prompt(
            query=query,
            query_type=query_type,
            categories=categories,
            symbols=symbols,
            reasoning=reasoning
        )

        try:
            result = await self._call_llm_json(prompt, "Thinking")

            if "need_tools" not in result:
                result["need_tools"] = query_type not in ["conversational", "memory_recall"]
            if "final_intent" not in result:
                result["final_intent"] = query
            if "reason" not in result:
                result["reason"] = "Validated"

            return result

        except Exception as e:
            self.logger.error(f"[PLAN:THINK] Error: {e}")
            return {
                "need_tools": query_type not in ["conversational", "memory_recall"],
                "final_intent": query,
                "reason": "Fallback validation"
            }

    # --- Stage 3: Create Plan ---

    async def _stage3_create_plan(
        self,
        query: str,
        query_type: str,
        symbols: List[str],
        response_language: str,
        available_tools: List[Dict[str, Any]],
        context: str,
        validated_intent: str
    ) -> Dict[str, Any]:
        """
        Create detailed task plan with LLM.

        Shows complete tool schemas for accurate parameter planning.
        """
        tools_text = self._format_tools_for_prompt(available_tools)
        self.logger.debug(f"[PLAN:S3] Tools for LLM: {len(available_tools)}")

        prompt = self.prompt_builder.build_plan_prompt(
            query=query,
            query_type=query_type,
            symbols=symbols,
            response_language=response_language,
            validated_intent=validated_intent,
            tools_text=tools_text
        )

        return await self._call_llm_json(prompt, "Stage3")
    
    def _format_tools_for_prompt(self, tools: List[Dict[str, Any]], max_tools: int = 20) -> str:
        """Format tools with complete info for LLM planning."""
        lines = []

        for tool in tools[:max_tools]:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")[:100]

            metadata = tool.get("metadata", {})
            requires_symbol = metadata.get("requires_symbol", True)

            params = tool.get("parameters", {}).get("properties", {})
            required_params = tool.get("parameters", {}).get("required", [])

            param_info = []
            for param_name, param_def in params.items():
                is_required = param_name in required_params
                param_type = param_def.get("type", "string")
                req_marker = "*" if is_required else ""
                param_info.append(f"{param_name}{req_marker}:{param_type}")

            params_str = ", ".join(param_info) if param_info else "none"
            symbol_str = "yes" if requires_symbol else "no"

            lines.append(
                f"- {name}\n"
                f"  desc: {description}\n"
                f"  symbol_required: {symbol_str}\n"
                f"  params: {params_str}"
            )

        return "\n\n".join(lines)

    # --- Helper Methods ---

    def _get_current_date_context(self) -> str:
        """Get current date context for prompts."""
        current_date = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M UTC")

        return f"""<current_context>
Date: {current_date} {current_time}
Data Status: Real-time market data available through tools
Note: Tools fetch LIVE data from financial APIs
</current_context>"""

    def _validate_and_format_history(
        self,
        recent_chat: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Validate and format chat history."""
        if not recent_chat:
            return []

        validated = []
        for msg in recent_chat:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "").strip().lower()
            content = msg.get("content", "").strip()

            if not role or not content:
                continue

            if role not in ["user", "assistant"]:
                continue

            validated.append({
                "role": role,
                "content": content,
                "timestamp": msg.get("created_at", "")
            })

        if all(msg.get("timestamp") for msg in validated):
            validated = sorted(validated, key=lambda x: x["timestamp"])

        return validated

    def _format_history_for_prompt(self, history: List[Dict[str, str]], max_messages: int = 6) -> str:
        """Format history for LLM prompt."""
        if not history:
            return "No previous conversation"

        recent = history[-max_messages:]

        lines = []
        for i, msg in enumerate(recent):
            role = msg["role"].upper()
            content = msg["content"][:200]
            lines.append(f"Turn {i+1} [{role}]: {content}")

        return "\n".join(lines)

    def _validate_categories_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Filter invalid categories from LLM output."""
        original = result.get("categories", [])

        if not original:
            return result

        valid = [c for c in original if c in VALID_CATEGORIES]
        invalid = [c for c in original if c not in VALID_CATEGORIES]

        if invalid:
            self.logger.debug(f"[PLAN:S1] Filtered invalid categories: {invalid}")

        result["categories"] = valid
        return result

    def _build_context_string(
        self,
        history: List[Dict[str, str]],
        core_memory: Dict[str, Any],
        summary: str,
        working_memory_context: str = None
    ) -> str:
        """Build context string from all sources."""
        parts = []

        if core_memory:
            lines = [f"  - {k}: {v}" for k, v in core_memory.items()]
            parts.append(f"User Info:\n" + "\n".join(lines))

        if summary:
            parts.append(f"Summary: {summary}")

        if working_memory_context:
            parts.append(f"Current Task State:\n{working_memory_context}")

        return "\n\n".join(parts) if parts else ""
    
    async def _call_llm_json(
        self,
        prompt: str,
        stage: str,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Call LLM and parse JSON response with retry logic.

        Handles markdown cleanup, truncation fixes, and fallback recovery.
        """
        content = ""

        for attempt in range(max_retries + 1):
            try:
                max_tokens = 2000 if stage == "Stage3" else 1000

                params = {
                    "model_name": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a financial assistant. Return ONLY valid JSON, no markdown, no explanation."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "provider_type": self.provider_type,
                    "api_key": self.api_key,
                    "enable_thinking": False,
                    "max_tokens": max_tokens
                }

                if self.capability != ModelCapability.ADVANCED:
                    params["temperature"] = 0.1

                response = await self.llm_provider.generate_response(**params)

                content = response.get("content", "") if isinstance(response, dict) else str(response)
                content = content.strip()

                # Remove thought blocks
                if "[/THOUGHT]" in content:
                    _, content = content.split("[/THOUGHT]", 1)

                # Remove markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    parts = content.split("```")
                    if len(parts) >= 2:
                        content = parts[1]

                content = content.strip()

                # Fix JSON issues
                content = self._fix_json_braces(content)
                content = self._fix_truncated_json(content, stage)

                return json.loads(content)

            except json.JSONDecodeError as e:
                self.logger.warning(f"[PLAN:{stage}] JSON parse error (attempt {attempt + 1}): {e}")

                if attempt < max_retries:
                    # Try to recover
                    recovered = self._try_recover_json(content, stage)
                    if recovered:
                        self.logger.info(f"[PLAN:{stage}] Recovered from truncated response")
                        return recovered

                    self.logger.debug(f"[PLAN:{stage}] Retrying with simplified prompt...")
                    prompt = self._simplify_prompt_for_retry(prompt, stage)
                    continue
                else:
                    self.logger.error(f"[PLAN:{stage}] JSON parse failed after {max_retries + 1} attempts")
                    self.logger.debug(f"[PLAN:{stage}] Content: {content[:500]}")
                    return self._get_fallback_response(stage)

            except Exception as e:
                self.logger.error(f"[PLAN:{stage}] LLM error: {e}", exc_info=True)
                raise

        return self._get_fallback_response(stage)


    def _fix_truncated_json(self, content: str, stage: str) -> str:
        """Fix truncated JSON by closing open structures."""
        if not content:
            return '{}'

        if content.endswith('}'):
            return content

        # Stage3 specific: close tasks array properly
        if stage == "Stage3":
            last_task_end = content.rfind('}]')
            if last_task_end > 0:
                content = content[:last_task_end + 2] + '}'
                return content

            if '"tasks"' in content and '[' in content:
                tasks_start = content.find('"tasks"')
                bracket_start = content.find('[', tasks_start)

                if bracket_start > 0:
                    last_open_brace = content.rfind('{')
                    if last_open_brace > bracket_start:
                        content = content + '}]}'
                    else:
                        content = content + ']}'
                    return content

        # Generic fix: close open braces/brackets
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')

        if open_brackets > 0:
            content = content + (']' * open_brackets)
        if open_braces > 0:
            content = content + ('}' * open_braces)

        return content


    def _try_recover_json(self, content: str, stage: str) -> Optional[Dict[str, Any]]:
        """Try to recover partial JSON using regex extraction."""
        import re

        if not content:
            return None

        # Try parsing with fixes first
        try:
            fixed = self._fix_truncated_json(content, stage)
            return json.loads(fixed)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Stage3: Extract metadata and tasks
        if stage == "Stage3":
            try:
                intent_match = re.search(r'"query_intent"\s*:\s*"([^"]*)"', content)
                strategy_match = re.search(r'"strategy"\s*:\s*"([^"]*)"', content)
                symbols_match = re.search(r'"symbols"\s*:\s*\[([^\]]*)\]', content)
                language_match = re.search(r'"response_language"\s*:\s*"([^"]*)"', content)

                if intent_match:
                    symbols = []
                    if symbols_match:
                        symbols_str = symbols_match.group(1)
                        symbols = [s.strip().strip('"') for s in symbols_str.split(',') if s.strip()]

                    # Extract complete tasks
                    tasks = []
                    tasks_match = re.search(r'"tasks"\s*:\s*\[(.*)', content, re.DOTALL)
                    if tasks_match:
                        tasks_content = tasks_match.group(1)
                        task_pattern = r'\{[^{}]*"tool_name"[^{}]*\}'
                        task_matches = re.findall(task_pattern, tasks_content)

                        for i, task_str in enumerate(task_matches[:3]):
                            try:
                                tool_name_match = re.search(r'"tool_name"\s*:\s*"([^"]*)"', task_str)
                                if tool_name_match:
                                    tasks.append({
                                        "id": i + 1,
                                        "description": f"Execute {tool_name_match.group(1)}",
                                        "tools_needed": [{
                                            "tool_name": tool_name_match.group(1),
                                            "params": {"symbol": symbols[0] if symbols else ""}
                                        }],
                                        "expected_data": [],
                                        "priority": "medium",
                                        "dependencies": []
                                    })
                            except (AttributeError, IndexError, KeyError):
                                continue

                    return {
                        "query_intent": intent_match.group(1),
                        "strategy": strategy_match.group(1) if strategy_match else "parallel",
                        "estimated_complexity": "moderate",
                        "symbols": symbols,
                        "response_language": language_match.group(1) if language_match else "vi",
                        "reasoning": "Recovered from truncated response",
                        "tasks": tasks
                    }
            except (AttributeError, IndexError, KeyError, re.error):
                pass

        # Stage1: Extract classification
        if stage == "Stage1":
            try:
                query_type_match = re.search(r'"query_type"\s*:\s*"([^"]*)"', content)
                categories_match = re.search(r'"categories"\s*:\s*\[([^\]]*)\]', content)
                symbols_match = re.search(r'"symbols"\s*:\s*\[([^\]]*)\]', content)

                if query_type_match:
                    categories = []
                    if categories_match:
                        cat_str = categories_match.group(1)
                        categories = [c.strip().strip('"') for c in cat_str.split(',') if c.strip()]

                    symbols = []
                    if symbols_match:
                        sym_str = symbols_match.group(1)
                        symbols = [s.strip().strip('"') for s in sym_str.split(',') if s.strip()]

                    return {
                        "query_type": query_type_match.group(1),
                        "categories": categories,
                        "symbols": symbols,
                        "reasoning": "Recovered from truncated response",
                        "response_language": "vi"
                    }
            except (AttributeError, IndexError, KeyError, re.error):
                pass

        return None

    def _simplify_prompt_for_retry(self, prompt: str, stage: str) -> str:
        """Add simplification hint for retry attempt."""
        if stage == "Stage3":
            return prompt + """

IMPORTANT FOR RETRY: Keep response SHORT.
- Maximum 2-3 tasks only
- Brief descriptions
- Essential tools only
- Return valid JSON immediately
"""
        return prompt

    def _get_fallback_response(self, stage: str) -> Dict[str, Any]:
        """Get fallback response when JSON parse fails."""
        if stage == "Stage1":
            return {
                "query_type": "stock_specific",
                "categories": ["price"],
                "symbols": [],
                "reasoning": "Fallback: classification failed",
                "response_language": "vi"
            }
        elif stage == "Stage3":
            return {
                "query_intent": "Analysis request",
                "strategy": "parallel",
                "estimated_complexity": "simple",
                "symbols": [],
                "response_language": "vi",
                "reasoning": "Fallback: plan creation failed",
                "tasks": []
            }
        elif stage == "Thinking":
            return {
                "need_tools": True,
                "final_intent": "Financial analysis",
                "reason": "Fallback: validation failed"
            }
        return {}

    def _fix_json_braces(self, content: str) -> str:
        """Fix mismatched braces in JSON (extra/missing braces, unclosed strings)."""
        if not content:
            return '{}'

        # Fix unclosed strings
        quote_count = content.count('"')
        if quote_count % 2 == 1:
            last_quote = content.rfind('"')
            if last_quote > 0:
                before_quote = content[:last_quote]
                if before_quote.endswith(':') or before_quote.endswith(': '):
                    content = content + '"'

        # Fix brace count
        open_count = content.count('{')
        close_count = content.count('}')

        if open_count == close_count:
            return content

        if close_count > open_count:
            excess = close_count - open_count
            while excess > 0 and content.rstrip().endswith('}'):
                content = content.rstrip()[:-1]
                excess -= 1
            return content

        if open_count > close_count:
            missing = open_count - close_count
            content = content.rstrip() + ('}' * missing)
            return content

        return content
    
    def _parse_task_plan_dict(
        self,
        plan_dict: Dict[str, Any],
        symbols: List[str],
        response_language: str
    ) -> TaskPlan:
        """Parse JSON dict into TaskPlan object."""
        tasks = []

        for task_data in plan_dict.get('tasks', []):
            tools_needed = []

            for tool_data in task_data.get('tools_needed', []):
                if "symbol" in tool_data and "params" not in tool_data:
                    tool_data["params"] = {"symbol": tool_data.pop("symbol")}
                tools_needed.append(ToolCall(**tool_data))

            task = Task(
                id=task_data.get('id', len(tasks) + 1),
                description=task_data.get('description', ''),
                tools_needed=tools_needed,
                expected_data=task_data.get('expected_data', []),
                priority=TaskPriority(task_data.get('priority', 'medium')),
                dependencies=task_data.get('dependencies', []),
                status=TaskStatus.PENDING,
                done=False
            )
            tasks.append(task)

        return TaskPlan(
            tasks=tasks,
            query_intent=plan_dict.get('query_intent', ''),
            strategy=plan_dict.get('strategy', 'parallel'),
            estimated_complexity=plan_dict.get('estimated_complexity', 'simple'),
            symbols=symbols or plan_dict.get('symbols', []),
            response_language=response_language,
            reasoning=plan_dict.get('reasoning', '')
        )

    # --- Logging Helpers ---

    def _log_stage1_result(
        self,
        query_type: str,
        categories: List[str],
        symbols: List[str],
        reasoning: str,
        language: str
    ):
        """Log Stage 1 classification results."""
        self.logger.info(
            f"[PLAN:S1] type={query_type} | categories={categories} | "
            f"symbols={symbols} | lang={language}"
        )
        if reasoning:
            self.logger.debug(f"[PLAN:S1] reasoning: {reasoning[:100]}")

    def _log_plan_result(self, task_plan: TaskPlan, elapsed_ms: int):
        """Log final plan result with task details."""
        self.logger.info(
            f"[PLAN:DONE] {elapsed_ms}ms | intent={task_plan.query_intent} | "
            f"tasks={len(task_plan.tasks)} | strategy={task_plan.strategy} | "
            f"symbols={task_plan.symbols}"
        )

        # Log each task for terminal visibility
        if task_plan.tasks:
            self.logger.info(f"[PLAN:TASKS] -------- Planned Tasks ({len(task_plan.tasks)}) --------")
            for idx, task in enumerate(task_plan.tasks, 1):
                tools = [t.tool_name for t in task.tools_needed]
                deps = f" (deps: {task.dependencies})" if task.dependencies else ""
                self.logger.info(f"[PLAN:TASKS]   [{idx}] {task.description}")
                self.logger.info(f"[PLAN:TASKS]       tools={tools}{deps}")

                # Log params for each tool
                for tool_call in task.tools_needed:
                    if tool_call.params:
                        params_str = ', '.join(f"{k}={v}" for k, v in tool_call.params.items())
                        self.logger.debug(f"[PLAN:TASKS]       -> {tool_call.tool_name}({params_str})")
            self.logger.info(f"[PLAN:TASKS] ------------------------------------------")

    def _get_elapsed_ms(self, start_time: datetime) -> int:
        """Get elapsed time in milliseconds."""
        return int((datetime.now() - start_time).total_seconds() * 1000)