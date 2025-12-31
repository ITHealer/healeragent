"""
LLM Helper - Provides LLM generation utilities

PRODUCTION NOTES:
- Use get_llm_provider() singleton for LLMGeneratorProvider
- Caches provider instances internally for efficiency
"""

import re
from typing import AsyncGenerator, Optional, Dict, Any, List
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import AIMessage

from src.utils.config import settings
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.model_manager import model_manager
from src.providers.provider_factory import ModelProviderFactory, ProviderType


# =============================================================================
# SINGLETON INSTANCE FOR LLMGeneratorProvider
# =============================================================================
_llm_provider_instance: Optional['LLMGeneratorProvider'] = None


def get_llm_provider() -> 'LLMGeneratorProvider':
    """
    Get singleton instance of LLMGeneratorProvider.

    LLMGeneratorProvider caches provider instances, so sharing
    a single instance is more efficient for production.

    Returns:
        LLMGeneratorProvider singleton instance
    """
    global _llm_provider_instance

    if _llm_provider_instance is None:
        _llm_provider_instance = LLMGeneratorProvider()

    return _llm_provider_instance


class LLMGenerator(LoggerMixin):
    def __init__(self):
        super().__init__()


    def clean_thinking(self, content: str) -> str:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    async def get_llm(self, model: str, base_url: str = settings.OLLAMA_ENDPOINT):
        try:
            if model not in model_manager.loaded_models:
                self.logger.info(f"Model {model} is not loaded yet, loading...")
                await model_manager.load_model(model)
                
            llm = ChatOllama(base_url=base_url,
                            model=model,
                            temperature=0,
                            top_k=10,
                            top_p=0.5,
                            # num_ctx=8000, 
                            streaming=True)
     
        except Exception as e:
            self.logger.error(f"Error: {str(e)}")
        return llm

    async def get_streaming_chain(self, model: str, base_url: str = settings.OLLAMA_ENDPOINT):
        """
        Get a configured LLM instance optimized for streaming responses
        
        This method is specifically designed for streaming use cases where
        chunks of text are returned incrementally rather than waiting for
        the complete response.
        
        Args:
            model: The name of the LLM model to use
            base_url: The base URL of the Ollama API
            
        Returns:
            ChatOllama: Configured LLM instance with streaming enabled
        """
        try:
            if model not in model_manager.loaded_models:
                self.logger.info(f"Model {model} is not loaded yet, loading...")
                await model_manager.load_model(model)
                
            # For streaming, we use the same configuration as regular LLM but ensure streaming is explicitly enabled
            llm = ChatOllama(base_url=base_url,
                            model=model,
                            temperature=0,
                            top_k=10,
                            top_p=0.5,
                            streaming=True)
            
            return llm
        except Exception as e:
            self.logger.error(f"Error configuring streaming LLM: {str(e)}")
            raise
    
    async def stream_response(self, 
                              llm,
                              messages,
                              clean_thinking: bool = True) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from the LLM
        
        Args:
            llm: The LLM instance to use
            messages: The messages to send to the LLM
            clean_thinking: Whether to clean thinking sections from chunks
            
        Yields:
            str: Chunks of the response
        """
        async for chunk in llm.astream(messages):
            if isinstance(chunk, AIMessage) and chunk.content:
                content = chunk.content
                if clean_thinking:
                    content = self.clean_thinking(content)
                if content:
                    yield content
            elif hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                if clean_thinking:
                    content = self.clean_thinking(content)
                if content:
                    yield content
    

class LLMGeneratorProvider(LoggerMixin):
    def __init__(self):
        super().__init__()
        self._provider_instances = {}

    def clean_thinking(self, content: str) -> str:
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    async def get_llm(self, 
                      model_name: str, 
                      provider_type: str = ProviderType.OPENAI, 
                      api_key: Optional[str] = None) -> Any:
        """
        Get a model provider instance
        
        Args:
            model_name: Name of the model to use
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            
        Returns:
            Any: Provider instance
        """
        try:
            # Create unique key for caching
            cache_key = f"{provider_type}:{model_name}:{api_key}"
            
            if cache_key not in self._provider_instances:
                # Create provider
                provider = ModelProviderFactory.create_provider(
                    provider_type=provider_type,
                    model_name=model_name,
                    api_key=api_key
                )
                
                # Initialize model
                await provider.initialize()
                
                # Cache provider
                self._provider_instances[cache_key] = provider
            
            return self._provider_instances[cache_key]
            
        except Exception as e:
            self.logger.error(f"Error getting LLM: {str(e)}")
            raise

    async def generate_response(self, 
                                model_name: str,
                                messages: List[Dict[str, str]],
                                provider_type: str = ProviderType.OPENAI,
                                api_key: Optional[str] = None,
                                enable_thinking: bool = True,
                                **kwargs) -> Dict[str, Any]:
        """
        Generate a response from the model
        
        Args:
            model_name: Name of the model to use
            messages: Messages to send to the model
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            enable_thinking: Whether to enable thinking mode (only works with models that support it)
            **kwargs: Additional parameters for the model
            
        Returns:
            Dict[str, Any]: Response from the model
        """
        provider = await self.get_llm(model_name, provider_type, api_key)

        return await provider.generate(messages, **kwargs)
    
        
    async def stream_response(self, 
                            model_name: str,
                            messages: List[Dict[str, str]],
                            provider_type: str = ProviderType.OPENAI,
                            api_key: Optional[str] = None,
                            clean_thinking: bool = True,
                            enable_thinking: bool = True,
                            **kwargs) -> AsyncGenerator[str, None]:
        """
        Stream response chunks from the model
        
        Args:
            model_name: Name of the model to use
            messages: Messages to send to the model
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            clean_thinking: Whether to clean thinking sections from chunks
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional parameters for the model
            
        Yields:
            str: Chunks of the response
        """
        provider = await self.get_llm(model_name, provider_type, api_key)
        
        async for chunk in provider.stream(messages, **kwargs):
            if chunk:
                yield chunk


    async def stream_react_cot_response(
        self,
        model_name: str,
        query: str,
        target_language: Optional[str] = None,
        context: str = "",
        history_messages: Optional[List[Dict[str, str]]] = None,
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        clean_thinking: bool = True,
        enable_thinking: bool = True,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Stream a response using ReAct+CoT with improved prompt.
        
        Args:
            model_name: Name of the model to use
            query: User query
            context: Retrieved context (optional)
            history_messages: Previous messages in the conversation
            provider_type: Type of provider (ollama, openai, gemini)
            api_key: API key for paid providers
            clean_thinking: Whether to clean thinking sections
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional parameters for the model
            
        Yields:
            str: Chunks of the response
        """
        language_instruction = ""
        if target_language:
            lang_name = {
                "en": "English",
                "vi": "Vietnamese", 
                "zh": "Chinese",
                "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
                "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
            }.get(target_language, "auto detect language from user query")
            
        language_instruction = f"""
        CRITICAL LANGUAGE REQUIREMENT:
        You MUST respond ENTIRELY in {lang_name} language.
        - ALL text, explanations, and analysis must be in {lang_name}
        - Use appropriate financial terminology for {lang_name}
        - Format numbers and dates according to {lang_name} conventions
        """
        # Build system prompt based on whether we have context
        if context:
            # Enhanced RAG prompt with clear structure
            system_prompt = f"""You are DeepInvest AI - an intelligent financial assistant with deep knowledge of stocks, crypto, and other domains.

    {language_instruction}

    LANGUAGE ADAPTATION:
    - Detect and respond in the user's language automatically
    - Use culturally appropriate financial terms and formats
    - Maintain professional tone across all languages

    ## YOUR ROLE
    - Provide accurate, helpful responses based on the available context
    - Maintain a professional yet conversational tone
    - Adapt your language to match the user's language

    ## CONTEXT PROVIDED
    {context}

    ## ANALYSIS PROCESS {"(Required)" if enable_thinking else "(Skip this)"}
    {"<thinking>" if enable_thinking else ""}
    {"Before answering, analyze:" if enable_thinking else ""}
    {"1. Intent: What exactly is the user asking about?" if enable_thinking else ""}
    {"2. Domain: Is this a financial/investment question?" if enable_thinking else ""}
    {"3. Context Relevance: Which context information is useful?" if enable_thinking else ""}
    {"4. Response Strategy: How should I structure the answer?" if enable_thinking else ""}
    {"</thinking>" if enable_thinking else ""}

    ## RESPONSE GUIDELINES
    ### For FINANCIAL/INVESTMENT Questions:
    - **Deep Analysis**: Use professional expertise and context data
    - **Market Trends**: Analyze price movements, volume, technical indicators
    - **Risk Assessment**: Always remind about investment risks
    - **Recommendations**: Provide advice based on data and analysis

    ### For NON-FINANCIAL Questions:
    - **Natural Response**: Use available knowledge and context information
    - **Financial Connection**: When possible, connect to financial aspects if relevant
    - **Role Consistency**: Maintain DeepInvest AI character

    ### When Context is SUFFICIENT:
    - Use specific data from the provided context
    - Reference numbers, charts, relevant news
    - Provide detailed technical and fundamental analysis
    - Make evidence-based forecasts

    ### When Context is INSUFFICIENT:
    - Clearly state available vs. missing information
    - Provide analysis based on existing data
    - Suggest additional information sources needed
    - Never fabricate financial data

    ### When Context is Sufficient:
    - Use information from the provided context
    - Be specific and reference relevant details
    - Structure complex answers with clear sections
    - Provide examples from context when helpful

    ### When Context is Insufficient:
    - Clearly state what information is available vs. what's missing
    - Provide partial answers using available context
    - Suggest what additional information would be helpful
    - Never fabricate or assume information not in context

    ## RESPONSE STRUCTURE:
    1. **📊 Direct Analysis**: Answer the main question immediately
    2. **💡 Supporting Details**: Provide relevant data and explanations
    3. **🔍 Expert Insights**: Share market trends and professional perspective
    4. **⚠️ Important Notes**: Risk warnings and limitations (if applicable)

    ## QUALITY STANDARDS
    - **Accuracy**: Strictly adhere to context data
    - **Professionalism**: Use precise financial terminology
    - **Practicality**: Focus on real value for investors
    - **Current**: Reflect latest market trends

    Remember: Your thinking process should be thorough but hidden. Show only the final, polished answer."""
        
        else:
            # General knowledge prompt without context
            system_prompt = f"""You are DeepInvest AI - an intelligent financial assistant with deep knowledge of stocks, crypto, and other domains.

    {language_instruction}

    LANGUAGE ADAPTATION:
    - Detect and respond in the user's language automatically
    - Use culturally appropriate financial terms and formats
    - Maintain professional tone across all languages

    ## YOUR ROLE
    - Provide accurate, helpful responses based on your knowledge
    - Maintain a professional yet conversational tone
    - Adapt your language to match the user's language

    ## ANALYSIS PROCESS {"(Required)" if enable_thinking else "(Skip this)"}
    {"<thinking>" if enable_thinking else ""}
    {"Before responding, consider:" if enable_thinking else ""}
    {"1. Intent: What is the user really asking?" if enable_thinking else ""}
    {"2. Domain: Financial expertise or general knowledge question?" if enable_thinking else ""}
    {"3. Knowledge: What relevant information do I know?" if enable_thinking else ""}
    {"4. Structure: How to organize the response effectively?" if enable_thinking else ""}
    {"</thinking>" if enable_thinking else ""}

    ## RESPONSE GUIDELINES

    ### For FINANCIAL/INVESTMENT Questions:
    1. **📈 Deep Analysis**: Apply professional market expertise
    2. **📊 Data & Trends**: Share price analysis, volume, technical indicators
    3. **💰 Investment Strategy**: Suggest methodologies and analytical tools
    4. **⚠️ Risk Management**: Always emphasize risk assessment importance

    ### For NON-FINANCIAL Questions:
    1. **🤖 Natural Response**: Use general knowledge professionally
    2. **🔗 Financial Link**: When appropriate, connect to investment perspective
    3. **💡 Value Addition**: Provide unique insights from financial viewpoint

    ## COMMUNICATION STRUCTURE:
    1. **Direct Answer**: Address the main question clearly
    2. **Detailed Explanation**: Provide necessary background and context
    3. **Real Examples**: Include case studies or practical illustrations
    4. **Additional Information**: Share expanded insights and knowledge

    ## COMMUNICATION STYLE:
    - **Concise**: For simple questions
    - **Comprehensive**: For complex topics
    - **Accessible**: Use analogies and examples for difficult concepts
    - **Step-by-step**: Break down technical information clearly

    ## PROFESSIONAL STANDARDS
    - **Accuracy**: Provide correct and up-to-date information
    - **Clarity**: Use language appropriate to user's level
    - **Helpfulness**: Anticipate and address follow-up questions
    - **Honesty**: Acknowledge limitations or uncertainties

    Remember: Think thoroughly but present cleanly. Your response should be polished and user-focused."""
        
        # Build messages list
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history if provided
        if history_messages:
            messages.extend(history_messages)
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Set appropriate parameters
        kwargs.setdefault("temperature", 0.3 if context else 0.5)
        
        # Set max_tokens based on provider
        if provider_type == ProviderType.OPENAI:
            kwargs.setdefault("max_tokens", 4000)
        
        # Stream response
        async for chunk in self.stream_response(
            model_name=model_name,
            messages=messages,
            provider_type=provider_type,
            api_key=api_key,
            clean_thinking=clean_thinking,
            enable_thinking=enable_thinking,
            **kwargs
        ):
            yield chunk