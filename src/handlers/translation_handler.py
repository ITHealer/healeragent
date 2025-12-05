from openai import AsyncOpenAI
from src.utils.logger.custom_logging import LoggerMixin
from src.utils.config import settings
from src.helpers.model_manager import model_manager
from src.handlers.llm_chat_handler import chat_service


# # COUNT TOKENS
# from src.helpers.llm_helper import count_tokens

class TranslationHandler(LoggerMixin):
    DEFAULT_TRANSLATION_MODEL = "qwen3:14b"
    
    def __init__(self):
        super().__init__()
   

    async def translate_text(self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "English",
        model: str = None
    ) -> str:
        # # COUNT TOKENS
        # output_tokens = ""
        try:
            model_to_use = model if model else self.DEFAULT_TRANSLATION_MODEL
            
            if model_to_use not in model_manager.loaded_models:
                self.logger.info(f"Model {model_to_use} not loaded yet, loading...")
                await model_manager.load_model(model_to_use)

            self.logger.info(f"Translate text with model {model_to_use}, translation from {source_lang} to {target_lang}")
            
            client = AsyncOpenAI(
                api_key="EMPTY",
                base_url=f"{settings.OLLAMA_ENDPOINT}/v1"
            )
            
            # # COUNT TOKENS
            # system_message = f"You are a professional translator. Translate the text from {source_lang} to {target_lang} accurately, maintaining the original meaning and tone."
            # full_prompt = f"{system_message}\n\n{text}"
            # input_tokens_with_context = count_tokens(full_prompt)
            # self.logger.info(f"Translation input tokens (with system message): {input_tokens_with_context}")
            system_message = f"""
You are a translation engine.

Instruction:
- Translate ONLY the following input text from {source_lang} to {target_lang}.
- Keep the original meaning and tone.
- Do NOT add explanation.
- Output MUST be only the translated sentence.

"""
            messages = [
                {
                    "role": "system", 
                    "content": system_message #f"You are a professional translator. Translate the text from {source_lang} to {target_lang} accurately, maintaining the original meaning and tone."
                },
                {
                    "role": "user",
                    "content": f"Input text:\n{text}" # text
                }
            ]
            
            print(f"Messages translate: {messages}")
            completion = await client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                extra_body={
                    "translation_options": {
                        "source_lang": source_lang,
                        "target_lang": target_lang
                    }
                }
            )
            
            translated_text = completion.choices[0].message.content
            self.logger.info(f"Translation completed successfully")

            # # COUNT TOKENS
            # output_tokens = count_tokens(translated_text)
            # self.logger.info(f"Translation output tokens: {output_tokens}")
            # total_tokens = input_tokens_with_context + output_tokens
            # self.logger.info(f"Translation total tokens: {total_tokens} (input: {input_tokens_with_context}, output: {output_tokens})")
            
            print(f"Translated text: {translated_text}")
            return translated_text
        except Exception as e:
            self.logger.error(f"Error when translating text: {str(e)}")
            raise


    async def translate_text_with_session(self,
        text: str,
        session_id: str,
        source_lang: str = "auto",
        target_lang: str = "English",
        max_history_messages: int = 5,
        model: str = None
    ) -> str:
        try:
            model_to_use = model if model else self.DEFAULT_TRANSLATION_MODEL
            
            if model_to_use not in model_manager.loaded_models:
                self.logger.info(f"Model {model_to_use} not loaded yet, loading...")
                await model_manager.load_model(model_to_use)
            try:
                chat_history_tuples = chat_service.get_chat_history(
                    session_id=session_id, 
                    limit=max_history_messages
                )
                self.logger.info(f"Retrieved {len(chat_history_tuples)} messages from session {session_id}")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve chat history: {str(e)}. Using basic translation.")
                return await self.translate_text(text, source_lang, target_lang, model)
               
            
            # hardcoded_chat_history = [
            #     ("Hi, I need to see a dermatologist.", "user"),
            #     ("Sure. Do you prefer an online consultation or visiting the clinic?", "assistant"),
            #     ("Online would be great.", "user")
            # ]
            
            client = AsyncOpenAI(
                api_key="EMPTY",
                base_url=f"{settings.OLLAMA_ENDPOINT}/v1"
            )
            
            system_message = {
                "role": "system",
                "content": f"""You are a professional translator. Your task is to translate from {source_lang} to {target_lang} with a two-phase approach:

Phase 1: Translate the input text using the conversation context for accurate meaning.
Phase 2: Verify and refine your translation to ensure it sounds natural and matches the context.

Return ONLY the final translated text without explanations or additional content.
"""
            }
            
            user_message = {
                "role": "user",
                "content": f"""
Conversation history:
{chat_history_tuples}

Original text: "{text}"

Please translate this to {target_lang}, considering the context. First translate it, then refine your translation if needed.
"""
            }
            
            messages = [system_message, user_message]
            self.logger.info(f"Messages translate: {messages}")
            
            completion = await client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.1,
                extra_body={
                    "translation_options": {
                        "source_lang": source_lang,
                        "target_lang": target_lang
                    }
                }
            )
            
            translation = completion.choices[0].message.content.strip().strip('"\'')
            return translation
            
        except Exception as e:
            self.logger.error(f"Error in combined translation: {str(e)}")
            return await self.translate_text(text, source_lang, target_lang, model)