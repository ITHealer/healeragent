from typing import Dict, Any, Optional, AsyncGenerator
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod


class StatsAnalysisLLMHelper(LoggerMixin):
    def __init__(self):
        super().__init__()
        
    def create_stats_analysis_prompt(
        self,
        symbol: str,
        volume_data: Dict[str, Any],
        user_question: Optional[str] = None,
        chat_history: Optional[str] = None,
        language_instruction: Optional[str] = None
    ) -> str:
        """Create prompt for volume profile stats"""
        
        base_prompt = f"""You are a professional financial analyst specializing in volume profile analysis.

{language_instruction}

PREVIOUS CONTEXT (Reference if relevant)
{chat_history}

Analyze the following volume profile data for {symbol}:

Volume Profile Analysis:
- Point of Control (POC): ${volume_data.get('point_of_control', 0)} (Price level with highest volume)
- Value Area Low: ${volume_data.get('value_area_low', 0)}
- Value Area High: ${volume_data.get('value_area_high', 0)}
- Price Range: ${volume_data.get('price_min', 0):.2f} - ${volume_data.get('price_max', 0):.2f}

Volume Distribution by Price Level:
"""
        
        # Add top volume areas
        bins = volume_data.get('bins', [])
        sorted_bins = sorted(bins, key=lambda x: x['volume'], reverse=True)[:5]
        
        for i, bin_data in enumerate(sorted_bins):
            base_prompt += f"\n{i+1}. ${bin_data['price_low']:.2f}-${bin_data['price_high']:.2f}: {bin_data['volume_percent']:.1f}% of total volume"
            
        base_prompt += f"""

Analysis Period: {volume_data.get('lookback_days', 'N/A')} days
Total Data Points: {volume_data.get('data_points', 'N/A')}

Please analyze:
1. Key support/resistance levels based on volume concentration
2. Trading implications of the POC and value area (70% of volume)
3. Volume accumulation/distribution patterns
4. Potential breakout/breakdown levels based on volume gaps
5. Risk management strategies using volume profile
6. Entry/exit points based on volume nodes
"""
        
        if user_question:
            base_prompt += f"\n\nUser's specific question: {user_question}\n"
            base_prompt += "Please address the user's question while providing the volume profile analysis."
            
        return base_prompt
    
    async def generate_stats_analysis_with_llm(
        self,
        symbol: str,
        volume_data: Dict[str, Any],
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        chat_history: Optional[str] = None 
    ) -> str:
        """Generate volume stats analysis using LLM"""
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

            prompt = self.create_stats_analysis_prompt(
                symbol, volume_data, user_question, chat_history, language_instruction
            )
            
            provider = ModelProviderFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            await provider.initialize()
            
            messages = [
                {"role": "system", "content": "You are a professional market statistics analyst specializing in volume profile analysis."},
                {"role": "user", "content": prompt}
            ]
            
            response = await provider.generate(messages, temperature=0.7)
            return response["content"]
            
        except Exception as e:
            self.logger.error(f"Error in stats analysis LLM: {str(e)}")
            raise


    async def stream_generate_stats_analysis_with_llm(
        self,
        symbol: str,
        volume_data: Dict[str, Any],
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        chat_history: Optional[str] = None 
    ) -> AsyncGenerator[str, None]:
        """Generate volume stats analysis using LLM"""
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

            prompt = self.create_stats_analysis_prompt(
                symbol, volume_data, user_question, chat_history, language_instruction
            )
            
            provider = ModelProviderFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            await provider.initialize()
            
            messages = [
                {"role": "system", "content": "You are a professional market statistics analyst specializing in volume profile analysis."},
                {"role": "user", "content": prompt}
            ]
            
            # Stream response
            async for chunk in provider.stream(messages, temperature=0.7):
                yield chunk
            
        except Exception as e:
            self.logger.error(f"Error in stats analysis LLM streaming: {str(e)}")
            raise