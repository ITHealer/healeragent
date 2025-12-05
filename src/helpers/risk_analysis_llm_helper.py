from typing import Dict, Any, Optional, AsyncGenerator
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod


class RiskAnalysisLLMHelper(LoggerMixin):
    def __init__(self):
        super().__init__()
        
    def create_risk_analysis_prompt(
        self, 
        symbol: str, 
        stop_levels_data: Dict[str, Any],
        position_sizing_data: Optional[Dict[str, Any]] = None,
        user_question: Optional[str] = None,
        memory_context: Optional[str] = "",
        language_instruction: Optional[str] = None
    ) -> str:

        base_prompt = f"""You are an expert risk management specialist analyzing {symbol}. 

{language_instruction}

Provide comprehensive risk analysis based on the following data:

## Stop Loss Analysis:
Current Price: ${stop_levels_data.get('current_price', 0):.2f}

### ATR-Based Stop Levels:
- Conservative (1x ATR): ${stop_levels_data.get('stop_levels', {}).get('atr_1x', 0):.2f}
- Moderate (2x ATR): ${stop_levels_data.get('stop_levels', {}).get('atr_2x', 0):.2f}
- Aggressive (3x ATR): ${stop_levels_data.get('stop_levels', {}).get('atr_3x', 0):.2f}

### Percentage-Based Stop Levels:
- Tight (2%): ${stop_levels_data.get('stop_levels', {}).get('percent_2', 0):.2f}
- Medium (5%): ${stop_levels_data.get('stop_levels', {}).get('percent_5', 0):.2f}
- Wide (8%): ${stop_levels_data.get('stop_levels', {}).get('percent_8', 0):.2f}

### Technical Support Levels:
- 20-day SMA: ${stop_levels_data.get('stop_levels', {}).get('sma_20', 0):.2f}
- 50-day SMA: ${stop_levels_data.get('stop_levels', {}).get('sma_50', 0):.2f}
- Recent Swing Low: ${stop_levels_data.get('stop_levels', {}).get('recent_swing', 0):.2f}
"""

        if position_sizing_data:
            base_prompt += f"""
## Position Sizing Analysis:
- Recommended Shares: {position_sizing_data.get('recommended_shares', 0)}
- Position Cost: ${position_sizing_data.get('position_cost', 0):,.2f}
- Dollar Risk: ${position_sizing_data.get('dollar_risk', 0):.2f}
- Account Risk %: {position_sizing_data.get('account_percent_risked', 0):.2f}%

### R-Multiple Targets:
- R1 (1:1): ${position_sizing_data.get('r_multiples', {}).get('r1', 0):.2f}
- R2 (2:1): ${position_sizing_data.get('r_multiples', {}).get('r2', 0):.2f}
- R3 (3:1): ${position_sizing_data.get('r_multiples', {}).get('r3', 0):.2f}
"""

        base_prompt += """
Please provide:
1. Recommended stop loss strategy based on the trader's risk profile
2. Position sizing recommendations (if data available)
3. Risk/Reward analysis
4. Key risk factors to monitor
5. Overall risk management strategy for this position
"""

        if user_question:
            base_prompt += f"Previous risk analyses and conversations:\n{memory_context}\n\nUser's specific question: {user_question}\n"
            base_prompt += "Please address the user's question while providing the risk analysis."
        
        return base_prompt
    
    async def generate_risk_analysis_with_llm(
        self,
        symbol: str,
        stop_levels_data: Dict[str, Any],
        position_sizing_data: Optional[Dict[str, Any]] = None,
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        memory_context: Optional[str] = ""
    ) -> str:
        
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

            prompt = self.create_risk_analysis_prompt(
                symbol, 
                stop_levels_data, 
                position_sizing_data, 
                user_question,
                memory_context,
                language_instruction
            )
            
            provider = ModelProviderFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            # Initialize provider
            await provider.initialize()
            
            messages = [
                {"role": "system", "content": "You are a professional risk management analyst specializing in position sizing and stop loss strategies."},
                {"role": "user", "content": prompt}
            ]
            
            response = await provider.generate(messages, temperature=0.7)
            
            return response["content"]
            
        except Exception as e:
            self.logger.error(f"Error generating risk analysis with LLM: {str(e)}")
            raise


    async def stream_generate_risk_analysis_with_llm(
        self,
        symbol: str,
        stop_levels_data: Dict[str, Any],
        position_sizing_data: Optional[Dict[str, Any]] = None,
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        memory_context: Optional[str] = ""
    ) -> AsyncGenerator[str, None]:
        
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

            prompt = self.create_risk_analysis_prompt(
                symbol, 
                stop_levels_data, 
                position_sizing_data, 
                user_question,
                memory_context,
                language_instruction
            )
            
            provider = ModelProviderFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            # Initialize provider
            await provider.initialize()
            
            messages = [
                {"role": "system", "content": "You are a professional risk management analyst specializing in position sizing and stop loss strategies."},
                {"role": "user", "content": prompt}
            ]
            
            # Stream response
            async for chunk in provider.stream(messages, temperature=0.7):
                yield chunk
            
        except Exception as e:
            self.logger.error(f"Error generating risk analysis with LLM streaming: {str(e)}")
            raise