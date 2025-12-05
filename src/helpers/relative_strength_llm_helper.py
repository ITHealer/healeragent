from typing import Dict, Any, Optional, List, AsyncGenerator
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod


class RelativeStrengthLLMHelper(LoggerMixin):
    def __init__(self):
        super().__init__()
        
    def create_relative_strength_prompt(
        self, 
        symbol: str, 
        benchmark: str,
        rs_data: Dict[str, Any],
        user_question: Optional[str] = None,
        chat_history: Optional[str] = None, 
        language_instruction: Optional[str] = None
    ) -> str:
        """Create prompt for relative strength analysis"""
        
        rs_text = rs_data.get('relative_strength_summary', '')
        rs_scores = rs_data.get('relative_strength', {})
        
        base_prompt = f"""You are an expert in relative strength analysis.

{language_instruction}

Analyze the performance of {symbol} compared to {benchmark}.

{rs_text}

Key Metrics:
"""
        # Add period-specific data
        for period in ["21d", "63d", "126d", "252d"]:
            if f"RS_{period}" in rs_scores:
                base_prompt += f"\n{period} Performance:"
                base_prompt += f"\n- Relative Strength Score: {rs_scores.get(f'RS_{period}', 'N/A')}"
                base_prompt += f"\n- Stock Return: {rs_scores.get(f'Return_{period}', 'N/A')}%"
                base_prompt += f"\n- Benchmark Return: {rs_scores.get(f'Benchmark_{period}', 'N/A')}%"
                base_prompt += f"\n- Excess Return: {rs_scores.get(f'Excess_{period}', 'N/A')}%\n"

        if chat_history and chat_history.strip():
            base_prompt += f"""

CONVERSATION CONTEXT:
The following is our previous conversation history. Use this context to provide more personalized and coherent responses:

{chat_history}

---
Based on the conversation history above, tailor your analysis to be consistent with previous discussions and address any follow-up questions naturally.
"""
            
        base_prompt += """
Please provide:
1. Overall relative strength assessment
2. Which timeframes show the strongest/weakest performance
3. Trading implications based on relative strength
4. Sectors or stocks that might show similar patterns
5. Risk considerations when using relative strength
"""

        if user_question:
            base_prompt += f"\n\nUser's question: {user_question}"
            
        return base_prompt
    
    async def generate_rs_analysis_with_llm(
        self,
        symbol: str,
        benchmark: str,
        rs_data: Dict[str, Any],
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        chat_history: Optional[str] = None
    ) -> str:
        """Generate RS analysis using LLM"""
        try:
            detection_method = ""
            if len(user_question.split()) < 2:
                detection_method = DetectionMethod.LLM
            else:
                detection_method = DetectionMethod.LIBRARY

            # Language detection
            language_info = await language_detector.detect(
                text=user_question,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )

            detected_language = language_info["detected_language"]

            if detected_language:
                lang_name = {
                    "en": "English",
                    "vi": "Vietnamese", 
                    "zh": "Chinese",
                    "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
                    "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
                }.get(detected_language, "the detected language")
                
            language_instruction = f"""
            CRITICAL LANGUAGE REQUIREMENT:
            You MUST respond ENTIRELY in {lang_name} language.
            - ALL text, explanations, and analysis must be in {lang_name}
            - Use appropriate financial terminology for {lang_name}
            - Format numbers and dates according to {lang_name} conventions
            """

            prompt = self.create_relative_strength_prompt(
                symbol, benchmark, rs_data, user_question, chat_history, language_instruction
            )
            
            provider = ModelProviderFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            await provider.initialize()
            
            messages = [
                {"role": "system", "content": "You are a relative strength analysis expert."},
                {"role": "user", "content": prompt}
            ]
            
            response = await provider.generate(messages, temperature=0.7)
            return response["content"]
            
        except Exception as e:
            self.logger.error(f"Error in RS LLM analysis: {str(e)}")
            raise

    
    async def stream_rs_analysis_with_llm(
        self,
        symbol: str,
        benchmark: str,
        rs_data: Dict[str, Any],
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        chat_history: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream RS analysis using LLM"""
        try:
            detection_method = ""
            if len(user_question.split()) < 2:
                detection_method = DetectionMethod.LLM
            else:
                detection_method = DetectionMethod.LIBRARY

            # Language detection
            language_info = await language_detector.detect(
                text=user_question,
                method=detection_method,
                system_language=target_language,
                model_name=model_name,
                provider_type=provider_type,
                api_key=api_key
            )

            detected_language = language_info["detected_language"]

            if detected_language:
                lang_name = {
                    "en": "English",
                    "vi": "Vietnamese", 
                    "zh": "Chinese",
                    "zh-cn": "CHINESE (SIMPLIFIED, CHINA)",
                    "zh-tw": "CHINESE (TRADITIONAL, TAIWAN)",
                }.get(detected_language, "the detected language")
                
            language_instruction = f"""
            CRITICAL LANGUAGE REQUIREMENT:
            You MUST respond ENTIRELY in {lang_name} language.
            - ALL text, explanations, and analysis must be in {lang_name}
            - Use appropriate financial terminology for {lang_name}
            - Format numbers and dates according to {lang_name} conventions
            """

            prompt = self.create_relative_strength_prompt(
                symbol, benchmark, rs_data, user_question, chat_history, language_instruction
            )
            
            provider = ModelProviderFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            await provider.initialize()
            
            messages = [
                {"role": "system", "content": "You are a relative strength analysis expert."},
                {"role": "user", "content": prompt}
            ]
            
            # Stream response
            async for chunk in provider.stream(messages, temperature=0.7):
                yield chunk
                
        except Exception as e:
            self.logger.error(f"Error in RS LLM streaming analysis: {str(e)}")
            yield f"Error in analysis: {str(e)}"