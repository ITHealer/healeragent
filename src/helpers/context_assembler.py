"""
Context Assembler for MemGPT-style Memory System
Assembles context window with proper priority ordering
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src.utils.logger.custom_logging import LoggerMixin
from src.agents.memory.core_memory import get_core_memory
from src.helpers.token_counter import TokenCounter
from src.handlers.llm_chat_handler import ChatMessageHistory
from src.helpers.chat_management_helper import ChatService


class ContextAssembler(LoggerMixin):
    """
    Assemble context window following MemGPT architecture priority:
    1. System Prompt
    2. Core Memory (Persona + Human)
    3. Recursive Summary (future)
    4. Recent Chat History (FIFO)
    5. Current Query
    """
    
    # Context budget allocation (tokens)
    MAX_CONTEXT_TOKENS = 180000  # Reserve 20K for response
    SYSTEM_PROMPT_BUDGET = 800
    CORE_MEMORY_BUDGET = 2000
    SUMMARY_BUDGET = 1000  # 
    HISTORY_BUDGET = 4000
    QUERY_BUDGET = 500
    
    def __init__(self):
        """Initialize Context Assembler"""
        super().__init__()
        self.core_memory = get_core_memory()
        self.token_counter = TokenCounter()
        self.chat_service = ChatService()
    
    
    async def assemble_context(
        self,
        user_id: str,
        session_id: str,
        current_query: str,
        system_prompt: str,
        max_history_messages: int = 10
    ) -> Dict:
        """
        Assemble complete context window with proper priority
        
        Args:
            user_id: User identifier
            session_id: Session identifier  
            current_query: Current user query
            system_prompt: System prompt for domain
            max_history_messages: Maximum history messages to include
            
        Returns:
            Dict with assembled context and metadata
        """
        try:
            assembly_start = datetime.now()
            
            # Step 1: Load Core Memory (always included)
            core_memory = await self.core_memory.load_core_memory(user_id)
            core_memory_formatted = self.core_memory.format_for_context(core_memory)
            core_memory_tokens = self.token_counter.count_tokens(core_memory_formatted)
            
            self.logger.info(
                f"Loaded core memory for user {user_id}: {core_memory_tokens} tokens"
            )
            
            # Step 2: Get Recent Chat History (FIFO - last N messages)
            chat_history_str = ""
            history_tokens = 0
            
            if session_id:
                try:
                    chat_history_str = ChatMessageHistory.get_full_chat_history_for_context(
                        session_id=session_id,
                        limit=max_history_messages
                    )
                    history_tokens = self.token_counter.count_tokens(chat_history_str)
                    
                    # Truncate if exceeds budget
                    if history_tokens > self.HISTORY_BUDGET:
                        chat_history_str = self.token_counter.truncate_to_token_limit(
                            chat_history_str,
                            self.HISTORY_BUDGET,
                            suffix="\n\n[...earlier messages truncated...]"
                        )
                        history_tokens = self.token_counter.count_tokens(chat_history_str)
                    
                    self.logger.info(
                        f"Loaded chat history for session {session_id}: "
                        f"{history_tokens} tokens ({max_history_messages} messages)"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error loading chat history: {e}")
            
            # Step 3: Count all components
            system_tokens = self.token_counter.count_tokens(system_prompt)
            query_tokens = self.token_counter.count_tokens(current_query)
            
            total_tokens = (
                system_tokens + 
                core_memory_tokens + 
                history_tokens + 
                query_tokens
            )
            
            # Step 4: Assemble context with priority order
            context_parts = []
            
            # Priority 1: System Prompt
            context_parts.append(("system_prompt", system_prompt, system_tokens))
            
            # Priority 2: Core Memory (always included)
            context_parts.append(("core_memory", core_memory_formatted, core_memory_tokens))
            
            # Priority 3: Recent Chat History
            if chat_history_str:
                formatted_history = f"\n### RECENT CONVERSATION HISTORY\n{chat_history_str}\n---\n"
                context_parts.append(("chat_history", formatted_history, history_tokens))
            
            # Priority 4: Current Query
            context_parts.append(("current_query", current_query, query_tokens))
            
            # Step 5: Build final context string
            final_context = ""
            for part_name, part_content, part_tokens in context_parts:
                final_context += part_content + "\n"
            
            # Step 6: Token budget analysis
            usage_percent = (total_tokens / self.MAX_CONTEXT_TOKENS) * 100
            
            assembly_time = (datetime.now() - assembly_start).total_seconds()
            
            result = {
                "context": final_context,
                "metadata": {
                    "total_tokens": total_tokens,
                    "max_tokens": self.MAX_CONTEXT_TOKENS,
                    "usage_percent": round(usage_percent, 2),
                    "token_breakdown": {
                        "system_prompt": system_tokens,
                        "core_memory": core_memory_tokens,
                        "chat_history": history_tokens,
                        "current_query": query_tokens
                    },
                    "core_memory_stats": await self.core_memory.get_memory_stats(user_id),
                    "history_message_count": max_history_messages,
                    "assembly_time_seconds": round(assembly_time, 3)
                },
                "components": {
                    "system_prompt": system_prompt,
                    "core_memory": core_memory,
                    "core_memory_formatted": core_memory_formatted,
                    "chat_history": chat_history_str
                }
            }
            
            self.logger.info(
                f"Context assembled: {total_tokens} tokens "
                f"({usage_percent:.1f}% of limit) in {assembly_time:.3f}s"
            )
            return result
            
        except Exception as e:
            self.logger.error(f"Error assembling context: {e}")
            raise
    
    
    def format_chat_history_for_llm(
        self,
        chat_history_str: str,
        current_query: str
    ) -> str:
        """
        Format chat history and current query for LLM
        
        Args:
            chat_history_str: Chat history string
            current_query: Current user query
            
        Returns:
            str: Formatted prompt for LLM
        """
        if chat_history_str and len(chat_history_str.strip()) > 0:
            formatted = f"""=== CURRENT QUESTION (Please respond in this language) ===
{current_query}

=== CONVERSATION HISTORY (For context only) ===
{chat_history_str}

Instructions: Answer the CURRENT QUESTION above in the SAME LANGUAGE it was asked, using relevant information from the history if needed."""
        else:
            formatted = current_query
        
        return formatted
    
    
    async def prepare_messages_for_llm(
        self,
        user_id: str,
        session_id: str,
        current_query: str,
        system_prompt: str = None,
        enable_thinking: bool = True,
        model_name: str = "gpt-4.1-nano",
        max_history_messages: int = 10
    ) -> Tuple[List[Dict], Dict]:
        """
        Prepare complete message list for LLM API call
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            current_query: Current user query
            system_prompt: System prompt for domain
            enable_thinking: Enable thinking mode
            model_name: Model name for token counting
            max_history_messages: Maximum history messages
            
        Returns:
            Tuple of (messages list, metadata dict)
        """
        try:
            # Assemble context
            context_result = await self.assemble_context(
                user_id=user_id,
                session_id=session_id,
                current_query=current_query,
                system_prompt=system_prompt,
                max_history_messages=max_history_messages
            )
            
            components = context_result["components"]
            
            # Build system message with Core Memory
            full_system_message = f"""{components['system_prompt']}

{components['core_memory_formatted']}"""
            
            # Format user message with history
            user_message = self.format_chat_history_for_llm(
                components['chat_history'],
                current_query
            )
            # Construct messages array
            messages = [
                {"role": "system", "content": full_system_message},
                {"role": "user", "content": user_message}
            ]
            
            # Count total message tokens
            total_message_tokens = self.token_counter.count_messages_tokens(
                messages, 
                model=model_name
            )
            
            metadata = context_result["metadata"]
            metadata["llm_messages_tokens"] = total_message_tokens
            
            self.logger.info(
                f"Prepared {len(messages)} messages for LLM: "
                f"{total_message_tokens} tokens"
            )
            
            return messages, metadata
            
        except Exception as e:
            self.logger.error(f"Error preparing messages for LLM: {e}")
            raise