from typing import Dict, Any, Optional
from src.utils.logger.custom_logging import LoggerMixin
from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.helpers.language_detector import language_detector, DetectionMethod


class TechnicalAnalysisLLMHelper(LoggerMixin):
    def __init__(self):
        super().__init__()
        
    def create_technical_analysis_prompt(
        self,
        symbol: str,
        analysis_data: Dict[str, Any],
        user_question: Optional[str] = None,
        chat_history: Optional[str] = None,
        language_instruction: Optional[str] = None
    ) -> str:
        """
        Create a prompt for technical analysis with optional previous context.

        Args:
            symbol: Stock symbol to analyze
            analysis_data: Dict containing keys like 'analysis_summary', 'latest_price',
                'momentum', 'avg_volume_20d', 'trend'
            user_question: A specific question from the user to address
            chat_history: Previous conversation or analysis context to include
        """
        # 1. Build context section if chat history exists
        context_section = ""
        if chat_history:
            context_section = f"""
    Previous conversations context:\n
    {chat_history}

    Please reference and build upon any previous technical analyses when relevant.
    """

        # 2. Core system instructions and data
        prompt = f"""You are an expert financial analyst specializing in technical analysis.  
    
    {language_instruction}

    PREVIOUS CONTEXT (Reference if relevant)
    {context_section}

    Analyze the following technical indicators for {symbol} and provide insights.

    Technical Analysis Data:
    {analysis_data.get('analysis_summary', '')}

    Key Metrics:
    - Current Price: ${analysis_data.get('latest_price', 0)}
    - RSI: {analysis_data.get('momentum', {}).get('rsi', 0)}
    - MACD Bullish: {analysis_data.get('momentum', {}).get('macd_bullish', False)}
    - Average Volume (20D): {analysis_data.get('avg_volume_20d', 0):,}

    Trend Status:
    - Above 20 SMA: {analysis_data.get('trend', {}).get('above_20sma', 'N/A')}
    - Above 50 SMA: {analysis_data.get('trend', {}).get('above_50sma', 'N/A')}
    - Above 200 SMA: {analysis_data.get('trend', {}).get('above_200sma', 'N/A')}

    Please provide:
    1. A concise summary of the current technical setup
    2. Key support and resistance levels based on the moving averages
    3. Trading recommendation (Buy/Hold/Sell) with reasoning
    4. Risk factors to consider
    """

        # 3. Append user's specific question if provided
        if user_question:
            prompt += f"""

    User's specific question: {user_question}
    Please address the user's question while providing the technical analysis.
    """
        else:
            prompt += "\nPlease provide a comprehensive technical analysis."

        return prompt

    
    async def generate_analysis_with_llm(
        self,
        symbol: str,
        analysis_data: Dict[str, Any],
        user_question: Optional[str] = None,
        target_language: Optional[str] = None,
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None,
        chat_history: Optional[str] = None 
    ) -> str:
        '""Generate technical analysis using LLM"""'
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
            prompt = self.create_technical_analysis_prompt(symbol, analysis_data, user_question, chat_history, language_instruction)
            
            provider = ModelProviderFactory.create_provider(
                provider_type=provider_type,
                model_name=model_name,
                api_key=api_key
            )
            
            # Initialize provider
            await provider.initialize()
            
            messages = [
                {"role": "system", "content": "You are a professional financial analyst providing technical analysis insights."},
                {"role": "user", "content": prompt}
            ]
            response = await provider.generate(messages, temperature=0.7)
            
            return response["content"]
            
        except Exception as e:
            self.logger.error(f"Error generating analysis with LLM: {str(e)}")
            raise