# File: src/agents/memory/memory_update_agent.py
"""
Memory Update Agent - Automatically updates Core Memory based on conversation

Simplified version (Claude/ChatGPT pattern):
- 1 LLM call for extraction
- Simple duplicate check (no LLM)
- Append to memory (no consolidation)
"""

import json
from typing import Dict, Optional, List, TypedDict, Any
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.memory.core_memory import CoreMemory
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.config import settings


class MemoryUpdateResult(TypedDict, total=False):
    """Result type for memory update operations"""
    updated: bool
    reason: str
    categories: List[str]
    extracted_info: Dict[str, Any]
    error: str


class MemoryUpdateAgent(LoggerMixin):
    """
    Intelligent agent that analyzes conversations and updates Core Memory
    when user shares personal information
    
    Architecture (Simplified - 1 LLM call only):
    ┌─────────────────────────────────────────────────────┐
    │  1. Extract Info (1 LLM call)                       │
    │  2. Check Duplicates (no LLM - string matching)     │
    │  3. Append to Memory (no LLM - direct write)        │
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(self):
        """Initialize Memory Update Agent"""
        super().__init__()
        self.core_memory = CoreMemory()
        self.llm_provider = LLMGeneratorProvider()
        self.logger.info("✅ MemoryUpdateAgent initialized (simplified mode)")
    
    # ========================================================================
    # MAIN ENTRY POINT
    # ========================================================================
    
    async def analyze_for_updates(
        self,
        user_id: str,
        user_message: str,
        assistant_message: str,
        model_name: str = None,
        provider_type: str = None
    ) -> MemoryUpdateResult:
        """
        Analyze conversation turn for information to update Core Memory
        
        LLM Calls: 1 (extraction only)
        
        Args:
            user_id: User identifier
            user_message: User's message
            assistant_message: Assistant's response
            model_name: LLM model
            provider_type: LLM provider
            
        Returns:
            MemoryUpdateResult with update status and details
        """
        self.logger.info(f"[MEMORY_UPDATE] Starting analysis for user {user_id}")
        self.logger.debug(f"[MEMORY_UPDATE] User message: {user_message[:100]}...")
        
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
            # Step 1: Extract information from conversation (1 LLM call)
            # ================================================================
            self.logger.info("[MEMORY_UPDATE] Step 1: Extracting user info...")
            
            extraction_result = await self._extract_user_info(
                user_message=user_message,
                assistant_message=assistant_message,
                model_name=model,
                provider_type=provider
            )
            
            self.logger.info(f"[MEMORY_UPDATE] Extraction result: has_updates={extraction_result.get('has_updates')}")
            self.logger.debug(f"[MEMORY_UPDATE] Extraction details: {extraction_result}")
            
            if not extraction_result.get('has_updates', False):
                self.logger.info(f"[MEMORY_UPDATE] No updates needed for user {user_id}")
                return {
                    'updated': False,
                    'reason': 'No relevant information found'
                }
            
            # ================================================================
            # Step 2: Load current Core Memory
            # ================================================================
            self.logger.info("[MEMORY_UPDATE] Step 2: Loading current memory...")
            
            current_memory = await self.core_memory.load_core_memory(user_id)
            current_human = current_memory.get('human', '')
            
            self.logger.debug(f"[MEMORY_UPDATE] Current human block: {len(current_human)} chars")
            
            # ================================================================
            # Step 3: Check duplicates (no LLM - string matching)
            # ================================================================
            self.logger.info("[MEMORY_UPDATE] Step 3: Checking duplicates...")
            
            extracted_info = extraction_result.get('extracted_info', {})
            categories = extraction_result.get('categories', [])
            
            if self._is_duplicate_info(current_human, extracted_info):
                self.logger.info("[MEMORY_UPDATE] Information already exists, skipping")
                return {
                    'updated': False,
                    'reason': 'Information already exists in memory'
                }
            
            # ================================================================
            # Step 4: Append to memory (no LLM - direct write)
            # ================================================================
            self.logger.info("[MEMORY_UPDATE] Step 4: Appending to memory...")
            
            updated = await self._append_to_human_block(
                user_id=user_id,
                current_human=current_human,
                new_info=extracted_info,
                categories=categories
            )
            
            if updated:
                self.logger.info(
                    f"[MEMORY_UPDATE] ✅ SUCCESS - Updated memory for user {user_id}: "
                    f"categories={categories}"
                )
                return {
                    'updated': True,
                    'categories': categories,
                    'extracted_info': extracted_info
                }
            else:
                self.logger.warning("[MEMORY_UPDATE] Update failed or no changes needed")
                return {
                    'updated': False,
                    'reason': 'Update failed or no changes needed'
                }
                
        except Exception as e:
            self.logger.error(f"[MEMORY_UPDATE] ❌ ERROR: {e}", exc_info=True)
            return {
                'updated': False,
                'error': str(e)
            }
    
    # ========================================================================
    # EXTRACTION (1 LLM Call)
    # ========================================================================
    
    async def _extract_user_info(
        self,
        user_message: str,
        assistant_message: str,
        model_name: str,
        provider_type: str
    ) -> Dict:
        """
        Use LLM to extract relevant user information from conversation
        
        LLM Calls: 1
        
        Returns:
            Dict with extraction results:
            {
                "has_updates": bool,
                "categories": ["category1", ...],
                "extracted_info": {"category1": "info", ...}
            }
        """
        extraction_prompt = f"""Analyze this conversation and extract any personal/profile information about the USER that should be saved to their memory profile.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{assistant_message[:500] if assistant_message else 'N/A'}

EXTRACT INFORMATION IN THESE CATEGORIES (if mentioned):
1. Portfolio: Stocks/crypto they own, position sizes, entry prices
2. Watchlist: Assets they're interested in or monitoring
3. Trading_Style: Day trader, swing trader, long-term investor
4. Risk_Tolerance: Conservative, moderate, aggressive
5. Investment_Goals: Retirement, income, growth, speculation
6. Preferences: Preferred sectors, asset classes, strategies
7. Experience_Level: Beginner, intermediate, advanced
8. Language_Preference: Primary language for communication
9. Market_Focus: US stocks, crypto, international, specific sectors

IMPORTANT RULES:
- ONLY extract information explicitly stated by the user
- DO NOT infer or assume information not directly mentioned
- DO NOT extract general questions or market data queries
- Focus on factual, profile-relevant information

Return ONLY valid JSON in this exact format:
{{"has_updates": true, "categories": ["Portfolio"], "extracted_info": {{"Portfolio": "Owns 100 shares of AAPL"}}}}

If NO relevant profile information found, return:
{{"has_updates": false, "categories": [], "extracted_info": {{}}}}"""

        try:
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            messages = [
                {
                    "role": "system", 
                    "content": "You are a precise information extraction agent. Extract user profile information from conversations. Return ONLY valid JSON, no markdown, no explanation."
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
            
            self.logger.debug(f"[EXTRACT] Raw response type: {type(response)}")
            self.logger.debug(f"[EXTRACT] Raw response: {response}")
            
            # ================================================================
            # Handle different response formats
            # ================================================================
            content = self._extract_content_from_response(response)
            
            self.logger.debug(f"[EXTRACT] Extracted content: {content[:200] if content else 'empty'}...")
            
            if not content:
                self.logger.warning("[EXTRACT] Empty content from LLM")
                return {'has_updates': False, 'categories': [], 'extracted_info': {}}
            
            # Clean markdown code blocks if present
            content = self._clean_json_response(content)
            
            self.logger.debug(f"[EXTRACT] Cleaned content: {content[:200]}...")
            
            # Parse JSON
            result = json.loads(content)
            
            self.logger.info(f"[EXTRACT] ✅ Parsed successfully: has_updates={result.get('has_updates')}")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"[EXTRACT] JSON parse error: {e}")
            self.logger.debug(f"[EXTRACT] Failed content: {content if 'content' in dir() else 'N/A'}")
            return {'has_updates': False, 'categories': [], 'extracted_info': {}}
            
        except Exception as e:
            self.logger.error(f"[EXTRACT] Error: {e}", exc_info=True)
            return {'has_updates': False, 'categories': [], 'extracted_info': {}}
    
    def _extract_content_from_response(self, response: Any) -> str:
        """
        Extract string content from LLM response
        
        Handles multiple response formats:
        1. String directly
        2. Dict with 'content' key (string)
        3. Dict with 'content' key (dict with 'text')
        4. Dict with 'choices' key (OpenAI format)
        """
        # Case 1: Already a string
        if isinstance(response, str):
            return response
        
        # Case 2: Dict response
        if isinstance(response, dict):
            # Try 'content' key
            content = response.get('content')
            
            if content is None:
                # Try OpenAI format
                choices = response.get('choices', [])
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
            
            # Handle content being a dict
            if isinstance(content, dict):
                # Try 'text' key
                if 'text' in content:
                    return str(content['text'])
                # Try to serialize
                return json.dumps(content)
            
            # Handle content being a string
            if isinstance(content, str):
                return content
            
            # Last resort: serialize the whole response
            return json.dumps(response)
        
        # Case 3: Unknown type
        return str(response) if response else ''
    
    # def _clean_json_response(self, content: str) -> str:
    #     """Clean markdown code blocks and whitespace from JSON response"""
    #     if not content:
    #         return '{}'
        
    #     # Remove markdown code blocks
    #     if '```json' in content:
    #         content = content.split('```json')[1].split('```')[0]
    #     elif '```' in content:
    #         parts = content.split('```')
    #         if len(parts) >= 2:
    #             content = parts[1]
        
    #     # Strip whitespace
    #     content = content.strip()
        
    #     # Ensure valid JSON structure
    #     if not content.startswith('{'):
    #         # Try to find JSON object
    #         start = content.find('{')
    #         end = content.rfind('}')
    #         if start != -1 and end != -1:
    #             content = content[start:end+1]
        
    #     return content if content else '{}'

    def _clean_json_response(self, content: str) -> str:
        """
        Clean markdown code blocks and fix common JSON errors from LLM
        
        Handles:
        - Markdown code blocks (```json ... ```)
        - Extra trailing braces (common LLM error)
        - Missing/extra whitespace
        - Truncated JSON
        """
        if not content:
            return '{}'
        
        # Remove markdown code blocks
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            parts = content.split('```')
            if len(parts) >= 2:
                content = parts[1]
        
        # Strip whitespace
        content = content.strip()
        
        # Ensure valid JSON structure
        if not content.startswith('{'):
            # Try to find JSON object
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                content = content[start:end+1]
        
        # ================================================================
        # FIX: Handle extra trailing braces (common LLM error)
        # Example: {"has_updates": false, "categories": [], "extracted_info": {}}}
        # ================================================================
        content = self._fix_json_braces(content)
        
        return content if content else '{}'
    
    def _fix_json_braces(self, content: str) -> str:
        """
        Fix mismatched braces in JSON response
        
        Common LLM errors:
        - Extra trailing braces: {}}} → {}
        - Missing closing braces: {"a": {"b": 1} → {"a": {"b": 1}}
        """
        if not content:
            return '{}'
        
        # Count opening and closing braces
        open_count = content.count('{')
        close_count = content.count('}')
        
        if open_count == close_count:
            # Braces are balanced
            return content
        
        if close_count > open_count:
            # Too many closing braces - remove extras from end
            excess = close_count - open_count
            self.logger.debug(f"[JSON_FIX] Removing {excess} extra closing brace(s)")
            
            # Remove trailing braces
            while excess > 0 and content.endswith('}'):
                content = content[:-1]
                excess -= 1
            
            return content
        
        if open_count > close_count:
            # Missing closing braces - add at end
            missing = open_count - close_count
            self.logger.debug(f"[JSON_FIX] Adding {missing} missing closing brace(s)")
            content = content + ('}' * missing)
            return content
        
        return content
    
    # ========================================================================
    # DUPLICATE CHECK (No LLM)
    # ========================================================================
    
    def _is_duplicate_info(self, current_human: str, new_info: Dict) -> bool:
        """
        Check if extracted information already exists in memory
        
        Uses simple string matching - no LLM call
        """
        if not current_human or not new_info:
            return False
        
        current_lower = current_human.lower()
        
        for category, info_text in new_info.items():
            if not info_text or not isinstance(info_text, str):
                continue
                
            if len(info_text) < 10:
                # Skip very short info
                continue
            
            info_lower = info_text.lower()
            words = info_lower.split()
            
            if len(words) > 3:
                # Check if 70%+ of words already exist
                matches = sum(1 for word in words if len(word) > 3 and word in current_lower)
                match_ratio = matches / len(words) if words else 0
                
                if match_ratio > 0.7:
                    self.logger.debug(f"[DUPLICATE] Category '{category}' is duplicate (ratio: {match_ratio:.2f})")
                    return True
        
        return False
    
    # ========================================================================
    # MEMORY UPDATE (No LLM - Direct Append)
    # ========================================================================
    
    async def _append_to_human_block(
        self,
        user_id: str,
        current_human: str,
        new_info: Dict[str, Any],
        categories: List[str]
    ) -> bool:
        """
        Append extracted information to HUMAN block
        
        Simple append strategy (Claude/ChatGPT pattern):
        - No LLM consolidation
        - Just append with timestamp
        - Context compaction handles cleanup later
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            update_lines = [f"\n[Profile Update - {timestamp}]"]
            
            for category in categories:
                if category in new_info and new_info[category]:
                    info_text = new_info[category]
                    # Ensure info_text is string
                    if not isinstance(info_text, str):
                        info_text = str(info_text)
                    update_lines.append(f"- {category}: {info_text}")
            
            if len(update_lines) <= 1:
                self.logger.debug("[APPEND] No content to append")
                return False
            
            update_text = '\n'.join(update_lines)
            
            self.logger.debug(f"[APPEND] Appending {len(update_text)} chars to memory")
            
            success = await self.core_memory.append_to_human(
                user_id=user_id,
                new_info=update_text,
                section=None
            )
            
            if success:
                self.logger.info(f"[APPEND] ✅ Successfully appended to user {user_id}")
            else:
                self.logger.warning(f"[APPEND] Failed to append to user {user_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"[APPEND] Error: {e}", exc_info=True)
            return False