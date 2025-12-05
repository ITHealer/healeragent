from typing import Dict, Any, Optional, AsyncGenerator
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod


class ChartAnalysisLLMHelper(LoggerMixin):
    
    def create_chart_analysis_prompt(
        self,
        symbol: str,
        pattern_data: Dict[str, Any],
        user_question: Optional[str] = None,
        chat_history: Optional[str] = None,
        language_instruction: Optional[str] = None
    ) -> str:
        """Create prompt for chart pattern analysis"""
        
        patterns = pattern_data.get('patterns', [])
        
        base_prompt = f"""You are a professional technical analyst specializing in chart pattern recognition.

{language_instruction}        

PREVIOUS CONTEXT (Reference if relevant)
{chat_history}

Analyze the following chart patterns for {symbol}:

Chart Pattern Analysis Results:
"""
        
        if patterns:
            base_prompt += "\nDetected Patterns:\n"
            for pattern in patterns:
                base_prompt += f"\n- {pattern['type']}"
                if 'start_date' in pattern:
                    base_prompt += f" (from {pattern['start_date']} to {pattern['end_date']})"
                base_prompt += f"\n  Price Level: ${pattern['price_level']}"
                base_prompt += f"\n  Confidence: {pattern.get('confidence', 'N/A')}\n"
        else:
            base_prompt += "\nNo significant chart patterns detected in recent data.\n"
            
        base_prompt += """
Please provide:
1. Interpretation of detected patterns (if any)
2. Price targets based on patterns
3. Support and resistance levels
4. Trading strategy recommendations
5. Risk factors and pattern reliability
"""
        
        if user_question:
            base_prompt += f"\n\nUser's question: {user_question}"
            
        return base_prompt

    
    async def generate_chart_analysis_with_llm(
        self,
        symbol: str,
        pattern_data: Dict[str, Any],
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        chat_history: Optional[str] = None 
    ) -> str:
        """Generate chart analysis using LLM"""
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

            prompt = self.create_chart_analysis_prompt(
                symbol, pattern_data, user_question, chat_history, language_instruction
            )
            
            provider = ModelProviderFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            await provider.initialize()
            
            messages = [
                {"role": "system", "content": "You are an expert chart pattern analyst specializing in technical analysis."},
                {"role": "user", "content": prompt}
            ]
            
            response = await provider.generate(messages, temperature=0.7)
            return response["content"]
            
        except Exception as e:
            self.logger.error(f"Error in chart analysis LLM: {str(e)}")
            raise


    async def stream_generate_chart_analysis_with_llm(
        self,
        symbol: str,
        pattern_data: Dict[str, Any],
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        chat_history: Optional[str] = None 
    ) -> AsyncGenerator[str, None]:
        """Generate chart analysis using LLM"""
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

            prompt = self.create_chart_analysis_prompt(
                symbol, pattern_data, user_question, chat_history, language_instruction
            )
            
            provider = ModelProviderFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            await provider.initialize()
            
            messages = [
                {"role": "system", "content": "You are an expert chart pattern analyst specializing in technical analysis."},
                {"role": "user", "content": prompt}
            ]
            
            # Stream response
            async for chunk in provider.stream(messages, temperature=0.7):
                yield chunk
            
        except Exception as e:
            self.logger.error(f"Error in chart analysis LLM streaming: {str(e)}")
            raise