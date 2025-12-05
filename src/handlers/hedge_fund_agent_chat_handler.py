"""
Enhanced Agent Chat Handler
src/handlers/agent_chat_handler.py
"""

import asyncio
import json
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime
import time

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.hedge_fund.agents.warren_buffett import WarrenBuffettChatAgent
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ProviderType
from src.hedge_fund.llm.models import get_model, ModelProvider

class AgentChatHandler(LoggerMixin):
    """Handler for managing chat interactions with trading agent personalities"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize Warren Buffett agent (others can be added later)
        self.agents = {
            "warren_buffett": WarrenBuffettChatAgent()
        }
        
        # Analysis prompts for importance scoring
        self.importance_prompt = ChatPromptTemplate.from_template("""
        Analyze this investment conversation and rate its importance from 0 to 1.
        Consider factors like:
        - Specific investment decisions or recommendations
        - Important market insights or analysis
        - Personal investment strategy discussions
        - Risk assessments or portfolio changes
        
        Query: {query}
        Response: {response}
        Agent: {agent_name}
        
        Return ONLY a number between 0 and 1.
        """)
    
    async def process_message(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str],
        chat_history: str,
        memory_context: str,
        personality: str,
        model_name: str,
        provider_type: ProviderType,
        include_market_data: bool = True,
        identified_tickers: List[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a single message with the agent"""
        start_time = time.time()
        
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Build enhanced context with personality
        full_context = self._build_context(
            memory_context=memory_context,
            chat_history=chat_history,
            personality=personality,
            agent_id=agent_id
        )
        
        # Determine response mode based on tickers
        if identified_tickers and include_market_data:
            # Has tickers - provide investment analysis
            response = await agent.chat_with_analysis(
                message=message,
                context=full_context,
                personality=personality,
                tickers=identified_tickers,
                model_name=model_name,
                provider_type=provider_type.value  # Convert to string value
            )
        else:
            # No tickers - general conversation with personality
            response = await agent.chat_general(
                message=message,
                context=full_context,
                personality=personality,
                model_name=model_name,
                provider_type=provider_type.value  # Convert to string value
            )
        
        response_time = time.time() - start_time
        
        return {
            "content": response["content"],
            "tickers_analyzed": response.get("tickers_analyzed", []),
            "sources": response.get("sources", []),
            "response_time": response_time
        }
    
    async def stream_message(
        self,
        agent_id: str,
        message: str,
        session_id: Optional[str],
        enhanced_history: str,
        personality: str,
        model_name: str,
        provider_type: ProviderType,
        include_market_data: bool = True,
        identified_tickers: List[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response from agent"""
        
        agent = self.agents.get(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Fetch market data if tickers identified
        market_data = {}
        if identified_tickers and include_market_data:
            yield {
                "type": "thinking",
                "content": f"Analyzing {', '.join(identified_tickers)}..."
            }
            
            for ticker in identified_tickers[:3]:  # Limit to 3
                try:
                    data = await agent.analyze_stock(ticker)
                    market_data[ticker] = data
                except Exception as e:
                    self.logger.error(f"Error analyzing {ticker}: {e}")
            
            if market_data:
                yield {
                    "type": "metadata",
                    "tickers_analyzed": list(market_data.keys())
                }
        
        # Stream response with personality and context
        async for chunk in agent.stream_chat(
            message=message,
            history=enhanced_history,
            personality=personality,
            market_data=market_data,
            model_name=model_name,
            provider_type=provider_type.value  # Convert to string value
        ):
            yield chunk
    
    async def analyze_importance(
        self,
        query: str,
        response: str,
        agent_id: str,
        llm_provider,
        model_name: str,
        provider_type: ProviderType
    ) -> float:
        """Analyze conversation importance for memory storage"""
        try:
            agent_name = self.agents[agent_id].name
            
            # FIX: Convert ProviderType enum to ModelProvider string properly
            from src.hedge_fund.llm.models import ModelProvider, normalize_provider_name
            
            # Convert ProviderType to the correct ModelProvider value
            if provider_type == ProviderType.OPENAI:
                model_provider = ModelProvider.OPENAI
            else:
                # For other providers, normalize the string
                model_provider = normalize_provider_name(provider_type.value)
            
            llm = get_model(model_name, model_provider)
            chain = self.importance_prompt | llm
            
            result = await chain.ainvoke({
                "query": query,
                "response": response,
                "agent_name": agent_name
            })
            
            # Parse score
            score_text = result.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            self.logger.error(f"Error analyzing importance: {e}")
            return 0.5  # Default medium importance

    
    def _build_context(
        self,
        memory_context: str,
        chat_history: str,
        personality: str,
        agent_id: str
    ) -> str:
        """Build enhanced context with personality for agent"""
        context_parts = []
        
        # Always include personality first
        context_parts.append(f"[Your Personality and Style]\n{personality}")
        
        if memory_context:
            context_parts.append(f"[Relevant Past Discussions]\n{memory_context}")
        
        if chat_history:
            context_parts.append(f"[Recent Conversation]\n{chat_history}")
        
        return "\n\n".join(context_parts)