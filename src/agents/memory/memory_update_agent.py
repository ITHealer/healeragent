import json
from typing import Dict, Optional, List, TypedDict, Any
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.memory.core_memory import CoreMemory
from src.agents.memory.consolidation_agent import (
    MemoryConsolidationAgent,
    ConsolidationAction,
    ConsolidationDecision
)
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.config import settings


class MemoryUpdateResult(TypedDict, total=False):
    """Result type for memory update operations"""
    updated: bool
    action: str  # ADD, UPDATE, DELETE, MERGE, NOOP
    reason: str
    categories: List[str]
    extracted_info: Dict[str, Any]
    error: str


class MemoryUpdateAgent(LoggerMixin):
    """
    Flow:
    1. Extract Info
    2. Load Existing Memory
    3. Consolidate
    4. Apply Action (ADD/UPDATE/DELETE/MERGE/NOOP)
    """
    
    def __init__(self, use_consolidation: bool = True):
        """
        Initialize Memory Update Agent
        
        Args:
            use_consolidation: Whether to use ConsolidationAgent for smart merging
        """
        super().__init__()
        self.core_memory = CoreMemory()
        self.llm_provider = LLMGeneratorProvider()
        self.use_consolidation = use_consolidation
        
        # Initialize consolidation agent if enabled
        if use_consolidation:
            self.consolidation_agent = MemoryConsolidationAgent()
        else:
            self.consolidation_agent = None
        
        self.logger.info(
            f"[MEMORY_UPDATE_AGENT] Initialized successfully "
            f"(consolidation={'ON' if use_consolidation else 'OFF'})"
        )
    
    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================
    
    async def analyze_for_updates(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        tool_results: Optional[Dict[str, Any]] = None,
        working_memory_context: Optional[str] = None,
        model_name: str = None,
        provider_type: str = None
    ) -> MemoryUpdateResult:
        """
        Analyze conversation turn for information to update Core Memory
        
        Args:
            user_id: User identifier
            user_message: User's message
            assistant_message: Assistant's response
            tool_results: Optional tool execution results (for action inference)
            model_name: LLM model
            provider_type: LLM provider
            
        Returns:
            MemoryUpdateResult with update status and details
        """
        self.logger.info(f"[MEMORY_UPDATE] Starting analysis for user {user_id}")
        self.logger.debug(f"[MEMORY_UPDATE] User message: {user_message[:100]}...")
        
        # Log working memory context if provided
        if working_memory_context:
            self.logger.info(f"[MEMORY_UPDATE] Working Memory context: {working_memory_context[:100]}...")
        
        # Quick input validation
        if not user_id:
            self.logger.warning("[MEMORY_UPDATE] Missing user_id")
            return {"updated": False, "error": "user_id is required"}
        
        if not user_message:
            self.logger.debug("[MEMORY_UPDATE] Empty user message, skipping")
            return {"updated": False, "reason": "Empty user message"}
        
        try:
            # Use defaults if not specified
            model = model_name or settings.MODEL_DEFAULT
            provider = provider_type or settings.PROVIDER_DEFAULT
            
            self.logger.info(f"[MEMORY_UPDATE] Using model: {model}, provider: {provider}")
            
            # ================================================================
            # Step 1: Enhanced Extraction (1 LLM call)
            # ================================================================
            self.logger.info("[MEMORY_UPDATE] Step 1: Extracting user info...")
            
            extraction_result = await self._extract_user_info(
                user_message=user_message,
                assistant_message=assistant_message,
                tool_results=tool_results,
                working_memory_context=working_memory_context,
                model_name=model,
                provider_type=provider
            )
            
            self.logger.info(
                f"[MEMORY_UPDATE] Extraction: has_updates={extraction_result.get('has_updates')}, "
                f"confidence={extraction_result.get('confidence', 0):.2f}"
            )
            self.logger.debug(f"[MEMORY_UPDATE] Extraction details: {extraction_result}")
            
            if not extraction_result.get('has_updates', False):
                self.logger.info(f"[MEMORY_UPDATE] No updates needed for user {user_id}")
                return {
                    'updated': False,
                    'action': 'NOOP',
                    'reason': extraction_result.get('reasoning', 'No relevant information found')
                }
            
            # ================================================================
            # Step 2: Load Current Core Memory
            # ================================================================
            self.logger.info("[MEMORY_UPDATE] Step 2: Loading current memory...")
            
            current_memory = await self.core_memory.load_core_memory(user_id)
            current_human = current_memory.get('human', '')
            
            self.logger.debug(f"[MEMORY_UPDATE] Current human block: {len(current_human)} chars")
            
            # ================================================================
            # Step 3: Consolidation (if enabled)
            # ================================================================
            extracted_info = extraction_result.get('extracted_info', {})
            categories = extraction_result.get('categories', [])
            
            if self.use_consolidation and self.consolidation_agent:
                self.logger.info("[MEMORY_UPDATE] Step 3: Consolidating with existing memory...")
                
                result = await self._consolidate_and_apply(
                    user_id=user_id,
                    current_human=current_human,
                    extracted_info=extracted_info,
                    categories=categories
                )
                
                return result
            else:
                # Simple append mode (backward compatible)
                self.logger.info("[MEMORY_UPDATE] Step 3: Simple append (no consolidation)...")
                
                # Check duplicates first
                if self._is_duplicate_info(current_human, extracted_info):
                    self.logger.info("[MEMORY_UPDATE] Information already exists, skipping")
                    return {
                        'updated': False,
                        'action': 'NOOP',
                        'reason': 'Information already exists in memory'
                    }
                
                # Append
                updated = await self._append_to_human_block(
                    user_id=user_id,
                    current_human=current_human,
                    new_info=extracted_info,
                    categories=categories
                )
                
                if updated:
                    return {
                        'updated': True,
                        'action': 'ADD',
                        'categories': categories,
                        'extracted_info': extracted_info
                    }
                else:
                    return {
                        'updated': False,
                        'action': 'NOOP',
                        'reason': 'Append failed or no changes needed'
                    }
                
        except Exception as e:
            self.logger.error(f"[MEMORY_UPDATE] ERROR: {e}", exc_info=True)
            return {
                'updated': False,
                'error': str(e)
            }
    
    # ========================================================================
    # ENHANCED EXTRACTION (Understands Actions + Declarations)
    # ========================================================================
    
    async def _extract_user_info(
        self,
        user_message: str,
        assistant_message: str,
        tool_results: Optional[Dict[str, Any]],
        working_memory_context: Optional[str],
        model_name: str,
        provider_type: str
    ) -> Dict:
        """
        Enhanced extraction that understands:
        1. Declarative statements ("I own NVDA")
        2. Action requests ("Add NVDA to my watchlist")
        3. Implicit preferences (tool results showing user actions)
        
        Returns:
            Dict with extraction results
        """
        # Prepare context
        assistant_snippet = assistant_message[:300] if assistant_message else ''
        
        wm_context = ""
        if working_memory_context:
            wm_context = f"""
<working_memory_context>
{working_memory_context}
</working_memory_context>

IMPORTANT: The Working Memory shows the current symbols in context.
When user says "this symbol", "these stocks", "symbol này", etc., 
refer to the symbols in Working Memory to understand what they mean.
"""
            
        # Format tool results if available
        tool_context = ""
        if tool_results:
            tool_context = f"""
<tool_execution_results>
{json.dumps(tool_results, ensure_ascii=False, indent=2)[:500]}
</tool_execution_results>
"""
        
        extraction_prompt = f"""<task>
Analyze the conversation and extract user profile information worth remembering.
</task>

<user_message>
{user_message}
</user_message>

<assistant_response>
{assistant_snippet}
</assistant_response>
{tool_context}
{wm_context}
<analysis_framework>
Think step-by-step:

1. **Declaration Detection**: Does the user make statements about themselves?
   - Ownership: "I own/have/hold X"
   - Preferences: "I like/prefer/want X"
   - Status: "I am a X trader", "My risk tolerance is X"

2. **Action Inference**: Does the user's ACTION reveal profile information?
   - "Add X to my watchlist" → User interested in X
   - "Buy/Sell X" → Portfolio action
   - "Set stop loss at X%" → Risk tolerance indicator
   - Tool results showing successful actions

3. **Temporal Stability**: Is this stable information worth remembering?
   - Stable: Portfolio holdings, watchlist, preferences, goals
   - Transient: One-time price checks, general questions

4. **Extraction Decision**: What should be remembered?
</analysis_framework>

<categories>
Available categories:
- Portfolio: Assets owned, positions held, entry prices
- Watchlist: Assets of interest, being monitored
- Trading_Style: Day trader, swing trader, long-term investor
- Risk_Tolerance: Conservative, moderate, aggressive
- Investment_Goals: Growth, income, retirement, speculation
- Preferences: Preferred sectors, strategies, analysis types
- Experience_Level: Beginner, intermediate, advanced
- Market_Focus: Geographic or asset class focus
</categories>

<extraction_rules>
EXTRACT when user:
- Makes declarative statements about themselves (ownership, preferences)
- Takes actions that reveal profile info (add to watchlist, buy/sell)
- Expresses ongoing interest in specific assets
- States goals, risk comfort, or trading approach

DO NOT EXTRACT:
- One-time data queries without self-reference
- General knowledge questions
- Ambiguous statements
</extraction_rules>

<output_format>
CRITICAL: Use CATEGORY NAME as the key in extracted_info.

Example for "Add TSLA and NVDA to my watchlist":
{{
  "has_updates": true,
  "confidence": 0.9,
  "reasoning": "User explicitly requested adding symbols to watchlist",
  "categories": ["Watchlist"],
  "extracted_info": {{
    "Watchlist": "Interested in TSLA, NVDA"
  }}
}}

Example for "I own 100 shares of AAPL":
{{
  "has_updates": true,
  "confidence": 0.95,
  "reasoning": "User declares ownership of AAPL shares",
  "categories": ["Portfolio"],
  "extracted_info": {{
    "Portfolio": "Owns 100 shares AAPL"
  }}
}}

If nothing to extract:
{{
  "has_updates": false,
  "confidence": 1.0,
  "reasoning": "No profile information disclosed or inferred",
  "categories": [],
  "extracted_info": {{}}
}}
</output_format>"""

        try:
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a user profile extraction agent for a financial AI assistant. "
                        "Extract both DECLARED information and information IMPLIED by user actions. "
                        "Use category names as keys in extracted_info. "
                        "Return ONLY valid JSON."
                    )
                },
                {"role": "user", "content": extraction_prompt}
            ]
            
            self.logger.debug(f"[EXTRACT] Calling LLM: {model_name}")
            
            response = await self.llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key,
                max_tokens=500
            )
            
            # Extract content from response
            content = self._extract_content_from_response(response)
            
            if not content:
                self.logger.warning("[EXTRACT] Empty content from LLM")
                return {'has_updates': False, 'categories': [], 'extracted_info': {}}
            
            # Clean and parse JSON
            content = self._clean_json_response(content)
            result = json.loads(content)
            
            # Normalize extracted_info format
            result = self._normalize_extraction_result(result)
            
            # Log result
            confidence = result.get('confidence', 0)
            has_updates = result.get('has_updates', False)
            reasoning = result.get('reasoning', '')
            
            self.logger.info(
                f"[EXTRACT] Result: has_updates={has_updates}, "
                f"confidence={confidence:.2f}, reason={reasoning[:80]}..."
            )
            
            # Reject low-confidence extractions
            if has_updates and confidence < 0.6:
                self.logger.info(f"[EXTRACT] Low confidence ({confidence:.2f}), skipping")
                return {'has_updates': False, 'categories': [], 'extracted_info': {}}
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"[EXTRACT] JSON parse error: {e}")
            return {'has_updates': False, 'categories': [], 'extracted_info': {}}
            
        except Exception as e:
            self.logger.error(f"[EXTRACT] Error: {e}", exc_info=True)
            return {'has_updates': False, 'categories': [], 'extracted_info': {}}
    
    def _normalize_extraction_result(self, result: Dict) -> Dict:
        """
        Normalize LLM output to ensure correct format
        
        Handles cases like:
        - {"category": "value"} → {"Portfolio": "value"}
        - Missing fields
        """
        extracted_info = result.get('extracted_info', {})
        categories = result.get('categories', [])
        
        # Fix wrong format: {"category": "value"} → {"CategoryName": "value"}
        if extracted_info:
            if "category" in extracted_info and len(extracted_info) == 1:
                if categories:
                    category_name = categories[0]
                    value = extracted_info["category"]
                    result['extracted_info'] = {category_name: value}
                    self.logger.debug(f"[EXTRACT] Normalized format: {result['extracted_info']}")
        
        return result
    
    # ========================================================================
    # CONSOLIDATION INTEGRATION
    # ========================================================================
    
    async def _consolidate_and_apply(
        self,
        user_id: str,
        current_human: str,
        extracted_info: Dict[str, Any],
        categories: List[str]
    ) -> MemoryUpdateResult:
        """
        Use ConsolidationAgent to intelligently manage memory
        
        Args:
            user_id: User ID
            current_human: Current human block content
            extracted_info: Newly extracted information
            categories: Categories of extracted info
            
        Returns:
            MemoryUpdateResult
        """
        # Parse current memory into structured entries
        existing_memories = self._parse_human_block_to_entries(current_human)
        
        results = []
        
        for category, info in extracted_info.items():
            if not info:
                continue
            
            # Get consolidation decision
            decision = await self.consolidation_agent.consolidate(
                new_info=f"{category}: {info}",
                existing_memories=existing_memories,
                category=category.lower()
            )
            
            # Apply decision
            success = await self._apply_consolidation_decision(
                user_id=user_id,
                current_human=current_human,
                decision=decision,
                category=category,
                new_info=info
            )
            
            results.append({
                'category': category,
                'action': decision.action.value,
                'success': success,
                'reasoning': decision.reasoning
            })
            
            # Update current_human for next iteration if modified
            if success and decision.action != ConsolidationAction.NOOP:
                current_memory = await self.core_memory.load_core_memory(user_id)
                current_human = current_memory.get('human', '')
        
        # Aggregate results
        any_updated = any(r['success'] and r['action'] != 'noop' for r in results)
        actions_taken = [r['action'] for r in results if r['success']]
        
        return {
            'updated': any_updated,
            'action': actions_taken[0] if actions_taken else 'NOOP',
            'categories': categories,
            'extracted_info': extracted_info,
            'details': results
        }
    
    async def _apply_consolidation_decision(
        self,
        user_id: str,
        current_human: str,
        decision: ConsolidationDecision,
        category: str,
        new_info: str
    ) -> bool:
        """
        Apply consolidation decision to Core Memory
        
        Args:
            user_id: User ID
            current_human: Current human block
            decision: Consolidation decision
            category: Category
            new_info: New information
            
        Returns:
            bool: Success
        """
        try:
            if decision.action == ConsolidationAction.ADD:
                # Append new entry
                timestamp = datetime.now().strftime('%Y-%m-%d')
                new_line = f"- {category}: {new_info} (added {timestamp})"
                
                updated_human = f"{current_human.strip()}\n{new_line}"
                
                success = await self.core_memory.update_human(user_id, updated_human)
                
                if success:
                    self.logger.info(f"[CONSOLIDATE] ADD: {category} for user {user_id}")
                return success
            
            elif decision.action == ConsolidationAction.UPDATE:
                # Update existing entry
                lines = current_human.split('\n')
                updated_lines = []
                target_found = False
                
                for line in lines:
                    # Check if this line should be updated
                    if decision.target_id and decision.target_id in line:
                        # Replace with new content
                        timestamp = datetime.now().strftime('%Y-%m-%d')
                        new_line = f"- {category}: {new_info} (updated {timestamp})"
                        updated_lines.append(new_line)
                        target_found = True
                        self.logger.info(f"[CONSOLIDATE] UPDATE: {category}")
                    elif category.lower() in line.lower() and not target_found:
                        # Fallback: update line matching category
                        timestamp = datetime.now().strftime('%Y-%m-%d')
                        new_line = f"- {category}: {new_info} (updated {timestamp})"
                        updated_lines.append(new_line)
                        target_found = True
                        self.logger.info(f"[CONSOLIDATE] UPDATE (fallback): {category}")
                    else:
                        updated_lines.append(line)
                
                if not target_found:
                    # If target not found, add as new
                    timestamp = datetime.now().strftime('%Y-%m-%d')
                    new_line = f"- {category}: {new_info} (added {timestamp})"
                    updated_lines.append(new_line)
                
                updated_human = '\n'.join(updated_lines)
                return await self.core_memory.update_human(user_id, updated_human)
            
            elif decision.action == ConsolidationAction.DELETE:
                # Remove obsolete entry
                lines = current_human.split('\n')
                updated_lines = [
                    line for line in lines
                    if decision.target_id not in line
                ]
                
                updated_human = '\n'.join(updated_lines)
                success = await self.core_memory.update_human(user_id, updated_human)
                
                if success:
                    self.logger.info(f"[CONSOLIDATE] DELETE: {decision.target_id}")
                return success
            
            elif decision.action == ConsolidationAction.MERGE:
                # Merge multiple entries
                lines = current_human.split('\n')
                updated_lines = []
                merged = False
                
                for line in lines:
                    should_remove = any(mid in line for mid in decision.merge_ids)
                    if should_remove and not merged:
                        # Add merged content once
                        timestamp = datetime.now().strftime('%Y-%m-%d')
                        merged_content = decision.new_content or new_info
                        new_line = f"- {category}: {merged_content} (merged {timestamp})"
                        updated_lines.append(new_line)
                        merged = True
                        self.logger.info(f"[CONSOLIDATE] MERGE: {len(decision.merge_ids)} entries")
                    elif not should_remove:
                        updated_lines.append(line)
                
                updated_human = '\n'.join(updated_lines)
                return await self.core_memory.update_human(user_id, updated_human)
            
            elif decision.action == ConsolidationAction.NOOP:
                self.logger.info(f"[CONSOLIDATE] NOOP: {decision.reasoning}")
                return True  # No action needed = success
            
            return False
            
        except Exception as e:
            self.logger.error(f"[CONSOLIDATE] Error applying decision: {e}")
            return False
    
    def _parse_human_block_to_entries(self, human_block: str) -> List[Dict[str, Any]]:
        """
        Parse human block into structured entries for consolidation
        
        Args:
            human_block: Raw human block content
            
        Returns:
            List of memory entry dicts
        """
        entries = []
        
        if not human_block:
            return entries
        
        lines = human_block.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('['):
                continue
            
            # Try to parse structured format: "- Category: Content"
            if line.startswith('-'):
                content = line[1:].strip()
                
                # Extract category if present
                category = 'general'
                if ':' in content:
                    parts = content.split(':', 1)
                    potential_category = parts[0].strip().lower()
                    if potential_category in [
                        'portfolio', 'watchlist', 'trading_style', 'risk_tolerance',
                        'investment_goals', 'preferences', 'experience_level', 'market_focus'
                    ]:
                        category = potential_category
                        content = parts[1].strip()
                
                # Generate ID from content hash
                import hashlib
                entry_id = hashlib.md5(content.encode()).hexdigest()[:8]
                
                entries.append({
                    'id': entry_id,
                    'content': content,
                    'category': category,
                    'line_number': i
                })
        
        return entries
    
    # ========================================================================
    # SIMPLE APPEND MODE (Backward Compatible)
    # ========================================================================
    
    def _is_duplicate_info(
        self,
        current_human: str,
        new_info: Dict[str, Any]
    ) -> bool:
        """Check if information already exists in memory"""
        if not current_human or not new_info:
            return False
        
        current_lower = current_human.lower()
        
        for category, info in new_info.items():
            if info and isinstance(info, str):
                # Check if key info already present
                info_lower = info.lower()
                
                # Extract key terms
                key_terms = [
                    term for term in info_lower.split()
                    if len(term) > 3 and term.upper() == term  # Symbols like NVDA
                ]
                
                # Check if key terms exist
                for term in key_terms:
                    if term in current_lower:
                        return True
        
        return False
    
    async def _append_to_human_block(
        self,
        user_id: str,
        current_human: str,
        new_info: Dict[str, Any],
        categories: List[str]
    ) -> bool:
        """
        Simple append to human block (backward compatible)
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            update_lines = [f"\n[Profile Update - {timestamp}]"]
            
            for category in categories:
                if category in new_info and new_info[category]:
                    info_text = new_info[category]
                    if not isinstance(info_text, str):
                        info_text = str(info_text)
                    update_lines.append(f"- {category}: {info_text}")
            
            if len(update_lines) <= 1:
                self.logger.debug("[APPEND] No content to append")
                return False
            
            update_text = '\n'.join(update_lines)
            
            self.logger.debug(f"[APPEND] Appending {len(update_text)} chars")
            
            success = await self.core_memory.append_to_human(
                user_id=user_id,
                new_info=update_text,
                section=None
            )
            
            if success:
                self.logger.info(f"[APPEND] Successfully appended for user {user_id}")
            else:
                self.logger.warning(f"[APPEND] Failed for user {user_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"[APPEND] Error: {e}", exc_info=True)
            return False
    
    # ========================================================================
    # MANUAL UPDATE API
    # ========================================================================
    
    async def manual_update(
        self,
        user_id: str,
        category: str,
        information: str
    ) -> bool:
        """
        Manually add information to user's memory profile
        
        Args:
            user_id: User ID
            category: Category name
            information: Information to add
            
        Returns:
            bool: Success
        """
        try:
            current_memory = await self.core_memory.load_core_memory(user_id)
            current_human = current_memory.get('human', '')
            
            if self.use_consolidation and self.consolidation_agent:
                # Use consolidation for smart merge
                existing_memories = self._parse_human_block_to_entries(current_human)
                
                decision = await self.consolidation_agent.consolidate(
                    new_info=f"{category}: {information}",
                    existing_memories=existing_memories,
                    category=category.lower()
                )
                
                return await self._apply_consolidation_decision(
                    user_id=user_id,
                    current_human=current_human,
                    decision=decision,
                    category=category,
                    new_info=information
                )
            else:
                # Simple append
                return await self._append_to_human_block(
                    user_id=user_id,
                    current_human=current_human,
                    new_info={category: information},
                    categories=[category]
                )
                
        except Exception as e:
            self.logger.error(f"[MANUAL_UPDATE] Error: {e}")
            return False
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _extract_content_from_response(self, response: Any) -> str:
        """Extract content string from LLM response"""
        if isinstance(response, dict):
            return response.get('content', '')
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def _clean_json_response(self, content: str) -> str:
        """Clean LLM response to extract valid JSON"""
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        return content.strip()


# ============================================================================
# Factory Function
# ============================================================================

def create_memory_update_agent(use_consolidation: bool = True) -> MemoryUpdateAgent:
    """
    Create a configured MemoryUpdateAgent instance
    
    Args:
        use_consolidation: Whether to enable smart consolidation
        
    Returns:
        Configured agent instance
    """
    return MemoryUpdateAgent(use_consolidation=use_consolidation)