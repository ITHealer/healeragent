"""
Skills Module - Domain-Specific Expert System

This module implements the SKILL pattern from Claude AI Architecture:
- Domain-specific prompts and analysis frameworks
- Auto-invoked based on classification market_type
- Reduces token usage by loading only relevant domain context

Architecture:
    Classification → market_type → SkillSelector → Domain Skill → Enhanced Prompt

Available Skills:
    - STOCK_SKILL: Equity analysis, fundamentals, technicals
    - CRYPTO_SKILL: Cryptocurrency, DeFi, on-chain analysis
    - MIXED_SKILL: Cross-asset comparison, portfolio analysis

Usage:
    from src.agents.skills import SkillRegistry

    registry = SkillRegistry.get_instance()
    skill = registry.select_skill(market_type="stock")
    prompt = skill.get_system_prompt()
"""

from src.agents.skills.skill_base import (
    BaseSkill,
    SkillConfig,
    SkillContext,
)
from src.agents.skills.skill_registry import SkillRegistry
from src.agents.skills.stock_skill import StockSkill
from src.agents.skills.crypto_skill import CryptoSkill
from src.agents.skills.mixed_skill import MixedSkill

__all__ = [
    "BaseSkill",
    "SkillConfig",
    "SkillContext",
    "SkillRegistry",
    "StockSkill",
    "CryptoSkill",
    "MixedSkill",
]
