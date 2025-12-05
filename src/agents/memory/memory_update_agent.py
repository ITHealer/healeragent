"""
Memory Update Agent - Automatically updates Core Memory based on conversation
Extracts and updates user profile information from natural conversations
"""

import json
from typing import Dict, Optional, List, TypedDict, Any
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.memory.core_memory import CoreMemory
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.utils.constants import LocalModelName


class MemoryUpdateResult(TypedDict, total=False):
    updated: bool
    reason: str
    categories: list[str]
    extracted_info: Dict[str, Any]
    error: str

class MemoryUpdateAgent(LoggerMixin):
    """
    Intelligent agent that analyzes conversations and updates Core Memory
    when user shares personal information
    """
    
    def __init__(
        self
    ):
        """
        Initialize Memory Update Agent
        
        Args:
            model_name: LLM model
            provider_type: Provider type (Ollama, OpenAI, Gemini)
        """
        super().__init__()

        self.core_memory = CoreMemory()
        self.llm_provider = LLMGeneratorProvider()
        
    
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
        
        Args:
            user_id: User identifier
            user_message: User's message
            assistant_message: Assistant's response
            model_name: LLM model
            provider_type: LLM provider
            
        Returns:
            MemoryUpdateResult: {updated, reason?, categories?, extracted_info?, error?}
        """
        # Quick input validation
        if not user_id:
            return {"updated": False, "error": "user_id is required"}
        if not (user_message or assistant_message):
            return {"updated": False, "reason": "Empty content"}
        
        try:
            extraction_model = model_name or self.default_model_name

            # Step 1: Extract information from conversation
            extraction_result = await self._extract_user_info(
                user_message=user_message,
                assistant_message=assistant_message,
                model_name=extraction_model,
                provider_type=provider_type
            )
            
            if not extraction_result.get('has_updates', False):
                self.logger.info(f"No memory updates needed for user {user_id}")
                return {
                    'updated': False,
                    'reason': 'No relevant information found'
                }
            
            # Step 2: Load current Core Memory
            current_memory = await self.core_memory.load_core_memory(user_id)
            
            # Step 3: Update HUMAN block with new information
            updated = await self._update_human_block(
                user_id=user_id,
                current_human=current_memory['human'],
                new_info=extraction_result['extracted_info'],
                categories=extraction_result['categories']
            )
            
            if updated:
                self.logger.info(
                    f"Updated Core Memory for user {user_id}: "
                    f"categories={extraction_result['categories']}"
                )
                
                return {
                    'updated': True,
                    'categories': extraction_result['categories'],
                    'extracted_info': extraction_result['extracted_info']
                }
            else:
                return {
                    'updated': False,
                    'reason': 'Update failed or no changes needed'
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing for memory updates: {e}")
            return {
                'updated': False,
                'error': str(e)
            }
    
    
    async def _extract_user_info(
        self,
        user_message: str,
        assistant_message: str,
        model_name: str,
        provider_type: str
    ) -> Dict:
        """
        Use LLM to extract relevant user information from conversation
        
        Returns:
            Dict with extraction results
        """
        extraction_prompt = f"""Analyze this conversation and extract any personal/profile information about the USER that should be saved to their memory profile.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{assistant_message}

EXTRACT INFORMATION IN THESE CATEGORIES (if mentioned):
1. Portfolio: Stocks/crypto they own, position sizes, entry prices
2. Watchlist: Assets they're interested in or monitoring
3. Trading Style: Day trader, swing trader, long-term investor, etc.
4. Risk Tolerance: Conservative, moderate, aggressive
5. Investment Goals: Retirement, income, growth, speculation
6. Preferences: Preferred sectors, asset classes, strategies
7. Experience Level: Beginner, intermediate, advanced
8. Language Preference: Primary language for communication
9. Market Focus: US stocks, crypto, international, specific sectors
10. Financial Situation: Account size, investment capital (if explicitly shared)

IMPORTANT RULES:
- ONLY extract information explicitly stated by the user
- DO NOT infer or assume information not directly mentioned
- DO NOT extract general questions or market opinions
- DO NOT extract information from the assistant's response
- Focus on factual, profile-relevant information

Return ONLY valid JSON in this exact format:
{{
    "has_updates": true/false,
    "categories": ["category1", "category2"],
    "extracted_info": {{
        "category1": "specific information",
        "category2": "specific information"
    }}
}}

If NO relevant profile information found, return:
{{"has_updates": false, "categories": [], "extracted_info": {{}}}}"""

        try:
            api_key = ModelProviderFactory._get_api_key(provider_type)
            
            messages = [
                {"role": "system", "content": "You are a precise information extraction agent. Return only valid JSON."},
                {"role": "user", "content": extraction_prompt}
            ]
            
            response = await self.llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            )

            # print(f"Log _extract_user_info response {response}")
            
            # Parse JSON response
            content = response.get('content', '{}')
            
            # Clean markdown code blocks if present
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # print(f"Log _extract_user_info content {content}")
            result = json.loads(content)
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse extraction JSON: {e}")
            return {'has_updates': False, 'categories': [], 'extracted_info': {}}
        except Exception as e:
            self.logger.error(f"Error in extraction: {e}")
            return {'has_updates': False, 'categories': [], 'extracted_info': {}}
    
    
    async def _update_human_block(
        self,
        user_id: str,
        current_human: str,
        new_info: Dict,
        categories: List[str]
    ) -> bool:
        """
        Update HUMAN block with extracted information
        
        Args:
            user_id: User identifier
            current_human: Current HUMAN block content
            new_info: Dictionary of new information by category
            categories: List of categories to update
            
        Returns:
            bool: Success status
        """
        try:
            # Build update text
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            update_lines = [f"\n[Profile Update - {timestamp}]"]
            
            for category in categories:
                if category in new_info and new_info[category]:
                    info_text = new_info[category]
                    update_lines.append(f"\n{category}: {info_text}")
            
            update_text = '\n'.join(update_lines)
            
            # Check if this info already exists to avoid duplicates
            if self._is_duplicate_info(current_human, new_info):
                self.logger.info("Information already exists in memory, skipping update")
                return False
            
            # Append to HUMAN block
            success = await self.core_memory.append_to_human(
                user_id=user_id,
                new_info=update_text,
                section=None  # Will be auto-timestamped
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating HUMAN block: {e}")
            return False
    
    
    def _is_duplicate_info(self, current_human: str, new_info: Dict) -> bool:
        """
        Check if extracted information is already in memory
        Simple duplicate detection based on text similarity
        
        Args:
            current_human: Current HUMAN block content
            new_info: New information to check
            
        Returns:
            bool: True if likely duplicate
        """
        # Simple heuristic: check if key phrases already exist
        for category, info_text in new_info.items():
            if info_text and len(info_text) > 10:
                # Check if substantial portion of text already exists
                info_lower = info_text.lower()
                current_lower = current_human.lower()
                
                # If more than 70% of words in new info exist in current
                words = info_lower.split()
                if len(words) > 3:
                    matches = sum(1 for word in words if word in current_lower)
                    if matches / len(words) > 0.7:
                        return True
        
        return False
    
    
    async def manual_update(
        self,
        user_id: str,
        category: str,
        information: str
    ) -> bool:
        """
        Manually add information to Core Memory (for API calls)
        
        Args:
            user_id: User identifier
            category: Category of information (Portfolio, Watchlist, etc.)
            information: Information to add
            
        Returns:
            bool: Success status
        """
        try:
            success = await self.core_memory.append_to_human(
                user_id=user_id,
                new_info=information,
                section=category
            )
            
            if success:
                self.logger.info(
                    f"Manual update successful for user {user_id}, category: {category}"
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in manual update: {e}")
            return False