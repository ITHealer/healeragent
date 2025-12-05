"""
Core Memory System - Always-loaded memory blocks (Tier 1)

Memory Hierarchy:
- PERSONA Block: Agent identity, expertise, language, style
- HUMAN Block: User profile, portfolio, watchlist, preferences
"""

import yaml
import os
from typing import Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path

from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.token_counter import TokenCounter


class CoreMemory(LoggerMixin):
    """
    Core Memory Manager - Always loaded into context window
    
    Implements Tier 1 memory from MemGPT architecture:
    - PERSONA: Agent's identity and capabilities
    - HUMAN: User's profile and preferences
    """
    
    # Token limits
    MAX_PERSONA_TOKENS = 500
    MAX_HUMAN_TOKENS = 1500
    MAX_TOTAL_TOKENS = 2000
    
    def __init__(self, config_dir: str = "src/config"):
        """
        Initialize Core Memory Manager
        
        Args:
            config_dir: Directory containing YAML config files
        """
        super().__init__()
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.token_counter = TokenCounter()
        
        # Default paths
        self.default_persona_path = self.config_dir / "default_persona.yaml"
        
    
    def _get_user_config_path(self, user_id: str) -> Path:
        """Get path to user's core memory config file"""
        return self.config_dir / f"user_{user_id}_core_memory.yaml"
    
    
    async def load_core_memory(self, user_id: str) -> Dict[str, str]:
        """
        Load core memory for a user
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with 'persona' and 'human' blocks
        """
        try:
            user_config_path = self._get_user_config_path(user_id)
            
            # If user config exists, load it
            if user_config_path.exists():
                self.logger.info(f"Loading core memory for user {user_id}")
                with open(user_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    return {
                        'persona': config.get('persona', ''),
                        'human': config.get('human', '')
                    }
            
            # Otherwise, create default core memory
            self.logger.info(f"Creating default core memory for user {user_id}")
            return await self._create_default_core_memory(user_id)
            
        except Exception as e:
            self.logger.error(f"Error loading core memory for user {user_id}: {e}")
            # Fallback to default
            return await self._create_default_core_memory(user_id)
    
    
    async def _create_default_core_memory(self, user_id: str) -> Dict[str, str]:
        """
        Create default core memory from template
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with default 'persona' and 'human' blocks
        """
        try:
            # Load default persona template
            if self.default_persona_path.exists():
                with open(self.default_persona_path, 'r', encoding='utf-8') as f:
                    default_config = yaml.safe_load(f)
            else:
                # Hardcoded fallback if file doesn't exist
                default_config = self._get_hardcoded_default()
            
            # Create default human block
            default_human = f"""User ID: {user_id}
Joined: {datetime.now().strftime('%Y-%m-%d')}
Language Preference: Auto-detect
Portfolio: Not configured yet
Watchlist: Empty
Risk Tolerance: Not specified
Trading Style: Not specified
Interests: General stock/crypto analysis
"""
            
            core_memory = {
                'persona': default_config.get('persona', ''),
                'human': default_human
            }
            
            # Save to user's config file
            await self.save_core_memory(user_id, core_memory['persona'], core_memory['human'])
            
            return core_memory
            
        except Exception as e:
            self.logger.error(f"Error creating default core memory: {e}")
            return self._get_hardcoded_default()
    
    
    def _get_hardcoded_default(self) -> Dict[str, str]:
        """Hardcoded fallback default memory"""
        return {
            'persona': """I am a specialized AI assistant for stock and cryptocurrency market analysis.

My Expertise:
- Technical Analysis: Chart patterns, indicators, trend analysis
- Fundamental Analysis: Financial statements, valuation metrics
- Market Sentiment: News analysis, social sentiment
- Risk Management: Position sizing, stop-loss strategies
- Multi-Asset Coverage: Stocks, ETFs, Crypto, Options

My Approach:
- Evidence-based: Always cite data sources
- Multi-perspective: Consider technical, fundamental, and sentiment
- Risk-aware: Emphasize risk management
- Adaptive: Match user's language and expertise level
- Continuous learning: Update with latest market data

Communication Style:
- Clear and concise explanations
- Use charts/data when helpful
- Multilingual support (EN/VI primary)
- Professional yet approachable tone""",
            'human': "New user - Profile not configured yet"
        }
    
    
    async def save_core_memory(
        self, 
        user_id: str, 
        persona: str, 
        human: str
    ) -> bool:
        """
        Save core memory to YAML file
        
        Args:
            user_id: User identifier
            persona: Persona block content
            human: Human block content
            
        Returns:
            bool: Success status
        """
        try:
            # Validate token counts
            persona_tokens = self.token_counter.count_tokens(persona)
            human_tokens = self.token_counter.count_tokens(human)
            total_tokens = persona_tokens + human_tokens
            
            if persona_tokens > self.MAX_PERSONA_TOKENS:
                raise ValueError(
                    f"Persona block exceeds limit: {persona_tokens} > {self.MAX_PERSONA_TOKENS}"
                )
            
            if human_tokens > self.MAX_HUMAN_TOKENS:
                raise ValueError(
                    f"Human block exceeds limit: {human_tokens} > {self.MAX_HUMAN_TOKENS}"
                )
            
            if total_tokens > self.MAX_TOTAL_TOKENS:
                raise ValueError(
                    f"Total core memory exceeds limit: {total_tokens} > {self.MAX_TOTAL_TOKENS}"
                )
            
            # Save to YAML
            user_config_path = self._get_user_config_path(user_id)
            config = {
                'persona': persona,
                'human': human,
                'updated_at': datetime.now().isoformat(),
                'token_counts': {
                    'persona': persona_tokens,
                    'human': human_tokens,
                    'total': total_tokens
                }
            }
            
            with open(user_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            self.logger.info(
                f"Saved core memory for user {user_id} "
                f"(tokens: {total_tokens}/{self.MAX_TOTAL_TOKENS})"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving core memory for user {user_id}: {e}")
            return False
    
    
    async def update_persona(self, user_id: str, new_persona: str) -> bool:
        """Update only the persona block"""
        try:
            current = await self.load_core_memory(user_id)
            return await self.save_core_memory(
                user_id, 
                new_persona, 
                current['human']
            )
        except Exception as e:
            self.logger.error(f"Error updating persona for user {user_id}: {e}")
            return False
    
    
    async def update_human(self, user_id: str, new_human: str) -> bool:
        """Update only the human block"""
        try:
            current = await self.load_core_memory(user_id)
            return await self.save_core_memory(
                user_id, 
                current['persona'], 
                new_human
            )
        except Exception as e:
            self.logger.error(f"Error updating human profile for user {user_id}: {e}")
            return False
    
    
    async def append_to_human(
        self, 
        user_id: str, 
        new_info: str, 
        section: Optional[str] = None
    ) -> bool:
        """Append information to human block"""
        try:
            current = await self.load_core_memory(user_id)
            current_human = current['human']
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            if section:
                addition = f"\n\n[{section} - Updated {timestamp}]\n{new_info}"
            else:
                addition = f"\n\n[Updated {timestamp}]\n{new_info}"
            
            new_human = current_human + addition
            
            if self.token_counter.count_tokens(new_human) > self.MAX_HUMAN_TOKENS:
                self.logger.warning(
                    f"Cannot append - would exceed token limit for user {user_id}"
                )
                return False
            
            return await self.save_core_memory(
                user_id,
                current['persona'],
                new_human
            )
            
        except Exception as e:
            self.logger.error(f"Error appending to human block for user {user_id}: {e}")
            return False
    
    
    def format_for_context(self, core_memory: Dict[str, str]) -> str:
        """Format core memory for inclusion in context window"""
        persona = core_memory.get('persona', '')
        human = core_memory.get('human', '')
        
        formatted = f"""### CORE MEMORY (Always Active)

## PERSONA - Who I Am
{persona}

## HUMAN - Who You Are
{human}

---
"""
        return formatted
    
    
    async def get_memory_stats(self, user_id: str) -> Dict:
        """Get statistics about core memory"""
        try:
            core_memory = await self.load_core_memory(user_id)
            
            persona_tokens = self.token_counter.count_tokens(core_memory['persona'])
            human_tokens = self.token_counter.count_tokens(core_memory['human'])
            total_tokens = persona_tokens + human_tokens
            
            return {
                'user_id': user_id,
                'persona_tokens': persona_tokens,
                'persona_limit': self.MAX_PERSONA_TOKENS,
                'persona_usage_pct': round(persona_tokens / self.MAX_PERSONA_TOKENS * 100, 1),
                'human_tokens': human_tokens,
                'human_limit': self.MAX_HUMAN_TOKENS,
                'human_usage_pct': round(human_tokens / self.MAX_HUMAN_TOKENS * 100, 1),
                'total_tokens': total_tokens,
                'total_limit': self.MAX_TOTAL_TOKENS,
                'total_usage_pct': round(total_tokens / self.MAX_TOTAL_TOKENS * 100, 1),
                'has_custom_config': self._get_user_config_path(user_id).exists()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting memory stats for user {user_id}: {e}")
            return {'error': str(e)}