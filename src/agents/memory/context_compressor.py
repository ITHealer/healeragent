# from typing import Dict, List, Tuple
# import tiktoken, functools
# from src.utils.logger.custom_logging import LoggerMixin
# from src.providers.provider_factory import ProviderType

# @functools.lru_cache
# def safe_encoding(model_name: str):
#     try:
#         return tiktoken.encoding_for_model(model_name)
#     except KeyError:
#         return tiktoken.get_encoding("cl100k_base")


# class ContextCompressor(LoggerMixin):
#     """Advanced context compression with summarization"""
    
#     def __init__(self, model_name: str = "gpt-4.1-nano"):
#         super().__init__()

#         self.model_name = model_name
#         self.encoding = safe_encoding(self.model_name)
#         self.max_tokens = 8192             # Model context limit
#         self.compression_threshold = 0.95  # 95% threshold
#         self.target_compression = 0.7      # Target 70% after compression


#     def _count_tokens(self, text: str) -> int:
#         """Count tokens in text"""
#         if not text:
#             return 0
#         return len(self.encoding.encode(text))


#     # get_relevant_context
#     async def should_compress(
#         self, 
#         chat_history: str,
#         current_query: str,
#         additional_context: str = ""
#     ) -> Tuple[bool, Dict]:
#         """
#         Check if compression is needed
        
#         Returns:
#             (need_compress, stats)
#         """
#         # Count tokens
#         history_tokens = self._count_tokens(chat_history)
#         query_tokens = self._count_tokens(current_query)
#         context_tokens = self._count_tokens(additional_context)
        
#         total_tokens = history_tokens + query_tokens + context_tokens
        
#         # Calculate usage percentage
#         usage_percent = (total_tokens / self.max_tokens) * 100
        
#         stats = {
#             "history_tokens": history_tokens,
#             "query_tokens": query_tokens,
#             "context_tokens": context_tokens,
#             "total_tokens": total_tokens,
#             "max_tokens": self.max_tokens,
#             "usage_percent": round(usage_percent, 2),
#             "threshold_percent": self.compression_threshold * 100
#         }
        
#         # Need compress if over threshold
#         need_compress = total_tokens > (self.max_tokens * self.compression_threshold)
        
#         if need_compress:
#             self.logger.warning(f"Token usage: {usage_percent:.1f}% - Need compress!")
#         else:
#             self.logger.info(f"Token usage: {usage_percent:.1f}% - OK")
        
#         return need_compress, stats


#     async def compress_simple(self, chat_history: str) -> str:
#         """
#         Simple compression: Keep the most recent messages
#         """
#         if not chat_history:
#             return ""
        
#         # Split into messages
#         lines = chat_history.strip().split('\n')
        
#         # Keep up to 10 most recent messages
#         max_messages = 10
        
#         # Find USER/ASSISTANT pairs
#         messages = []
#         current_msg = []
        
#         for line in lines:
#             if line.startswith(('USER:', 'ASSISTANT:', 'Human:', 'Assistant:')):
#                 if current_msg:
#                     messages.append('\n'.join(current_msg))
#                 current_msg = [line]
#             else:
#                 current_msg.append(line)
        
#         if current_msg:
#             messages.append('\n'.join(current_msg))
        
#         # Keep the most recent messages
#         compressed = messages[-max_messages:] if len(messages) > max_messages else messages
        
#         result = '\n'.join(compressed)
        
#         self.logger.info(f"Compressed: {len(messages)} → {len(compressed)} messages")
        
#         return result


#     # get_relevant_context
#     async def compress_smart(self, chat_history: str, llm_provider, current_query: str = "", api_key: str = None) -> str:
#         """
#         Smart compression: Use LLM to summarize chat history
        
#         Args:
#             chat_history: Chat history to compress
#             llm_provider: LLMGeneratorProvider instance
#             current_query: Current query for context
#             api_key: API key for the provider
#         """
#         if not chat_history:
#             return ""
        
#         # Skip compression if history is too short
#         if self._count_tokens(chat_history) < 500:
#             return chat_history
        
#         # Compression prompt
#         compress_prompt = f"""Please summarize the following conversation, keeping important information:
#     - Main topics discussed
#     - Numbers, stock codes mentioned
#     - Keep timeline of events
#     - Important decisions or conclusions
#     - Context related to current query: "{current_query}"
#     - Remove redundant greetings and confirmations

#     Conversation:
#     {chat_history[:3000]}  # Limit input length

#     Brief summary (max 500 words):"""

#         try:
#             # Call LLM for compression
#             messages = [
#                 {"role": "system", "content": "You are a conversation summarization assistant. Please summarize concisely while keeping important information."},
#                 {"role": "user", "content": compress_prompt}
#             ]
            
#             # Use the provider's generate_response method
#             response = await llm_provider.generate_response(
#                 model_name=self.model_name, 
#                 messages=messages,
#                 provider_type=ProviderType.OPENAI,
#                 api_key=api_key,
#                 temperature=1, # TODO: changed from 0 to 1
#                 max_tokens=500,
#                 enable_thinking=False
#             )
            
#             compressed = response.get("content", "")
            
#             if compressed:
#                 self.logger.info("LLM compression successful")
#                 return f"[Previous conversation summary]\n{compressed}\n[Recent conversation]\n" + self._get_recent_messages(chat_history, 3)
#             else:
#                 # Fallback if LLM fails
#                 return await self.compress_simple(chat_history)
                
#         except Exception as e:
#             self.logger.error(f"Error during LLM compression: {e}")
#             # Fallback to simple compression
#             return await self.compress_simple(chat_history)
        

#     def _get_recent_messages(self, chat_history: str, num_messages: int = 3) -> str:
#         """Get N most recent messages"""
#         lines = chat_history.strip().split('\n')
#         messages = []
#         current_msg = []
        
#         for line in lines:
#             if line.startswith(('USER:', 'ASSISTANT:', 'Human:', 'Assistant:')):
#                 if current_msg:
#                     messages.append('\n'.join(current_msg))
#                 current_msg = [line]
#             else:
#                 current_msg.append(line)
        
#         if current_msg:
#             messages.append('\n'.join(current_msg))
        
#         # Get most recent messages
#         recent = messages[-num_messages:] if len(messages) > num_messages else messages
#         return '\n'.join(recent)