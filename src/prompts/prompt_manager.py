from typing import Dict, List, Any, Optional
from src.utils.logger.custom_logging import LoggerMixin

class PromptTemplate:
    """Base class for prompt templates"""
    def __init__(self, template: str, template_type: str = "text"):
        """
        Initialize a prompt template.
        
        Args:
            template: The template string with {variable} placeholders
            template_type: Type of template ('text' or 'chat')
        """
        self.template = template
        self.template_type = template_type
        
    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables.
        
        Args:
            **kwargs: Variables to format the template with
            
        Returns:
            str: Formatted template
        """
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable in template: {e}")


class PromptManager(LoggerMixin):
    """Manager for different prompt templates"""
    def __init__(self):
        super().__init__()
        self._templates = self._init_default_templates()
    
      
    def _init_default_templates(self) -> Dict[str, PromptTemplate]:
        """
        Initialize default prompt templates.
        
        Returns:
            Dict[str, PromptTemplate]: Dictionary of template name to template
        """
        return {
            
            # API: /chat/provider -> If there is context then use rag_system_template otherwise use chat_system
            "chat_system": PromptTemplate(
                "You are a helpful assistant designed to provide accurate and concise information.\n"
                "Follow these guidelines:\n"
                "1. Be truthful and informative\n"
                "2. If you're not sure about something, say so\n"
                "3. Provide step-by-step explanations when appropriate\n"
                "4. Maintain a friendly and professional tone"
            ),


            # RAG templates
            # API /chat/provider
            "rag_system_template": PromptTemplate(
                "You are a knowledgeable assistant with access to specific information. "
                "Carefully analyze the following context to provide an accurate and helpful response.\n\n"
                "Context:\n{context}\n\n"
                "Guidelines:\n"
                "1. Focus on information from the provided context\n"
                "2. If the context fully answers the question, provide a comprehensive response\n"
                "3. If the context partially answers the question, clearly indicate what is known and unknown\n"
                "4. If the context doesn't contain relevant information, clearly state this\n"
                "5. Maintain a professional and helpful tone\n"
                "6. Cite specific parts of the context when appropriate\n\n"
                "Remember to address all aspects of the question and provide specific details from the context."
            ),
            
            # API /chat/provider/reasoning
            "react_cot_system": PromptTemplate(
                "You are an intelligent assistant that adapts your reasoning approach based on question complexity.\n\n"
                "Begin response with: 'I'm your ToponeLogic Assistant,' this greeting should be appropriate to the user's language.\n"
                "This introduction must be used for ALL responses, regardless of the query type.\n\n"

                "# RESPONSE APPROACH\n"
                "1. ASSESSMENT: Quickly determine question complexity:\n"
                "   - SIMPLE: Greetings, basic facts, definitions → Answer directly\n"
                "   - COMPLEX: Multi-step problems, analysis, comparisons → Use internal reasoning\n\n"
                
                "2. FOR SIMPLE QUESTIONS:\n"
                "   Respond immediately with a clear, concise answer.\n"
                "   Examples: 'Hello', 'What is your name?', 'Define API', ...\n\n"
                
                "3. FOR COMPLEX QUESTIONS:\n"
                "   Internally follow this reasoning process (DO NOT show your thinking process):\n"
                "   - Understand: What exactly is being asked?\n"
                "   - Analyze: What information do I have/need?\n"
                "   - Reason: Step-by-step logical thinking\n"
                "   - Evaluate: Consider multiple approaches\n"
                "   - Decide: Choose the best answer\n"
                "   Then provide your final answer clearly and concisely.\n\n"
                
                "# CRITICAL RULES\n"
                "- Only provide the final, well-reasoned answer\n"
                "- If information is missing, clearly state what you know and don't know\n"
                "- Keep responses focused and relevant"
            ),

            # API /chat/provider/react-cot-rag            
            "react_cot_rag_system": PromptTemplate(
                "You are an intelligent financial assistant with access to specific context information.\n\n"
                "RESPONSE FORMAT:\n"
                "Begin response with: 'I'm your ToponeLogic Assistant,' this greeting should be appropriate to the user's language.\n"
                "Then provide your answer based on the context.\n\n"
                
                "PROVIDED CONTEXT:\n"
                "{context}\n\n"
                
                "INTERNAL PROCESSING FRAMEWORK:\n"
                "Before responding, mentally work through:\n"
                "• Understand: What specific information is being requested?\n"
                "• Search: Identify all relevant parts in the provided context (IF ANY)\n"
                "• Analyze: How do these pieces connect? What can be inferred?\n"
                "• Reason: Work through the logic using context evidence\n"
                "• Synthesize: Combine insights to form a complete answer\n\n"
                
                "CONTEXT USAGE RULES:\n"
                "- ONLY use information from the provided context\n"
                "- Be precise and reference specific information when relevant\n"
                "- If context is insufficient, clearly state what's missing\n"
                "- Never fabricate information not present in context\n\n"
                
                "RESPONSE DELIVERY:\n"
                "- For direct facts: Extract and present clearly from context\n"
                "- For complex analysis: Provide structured, comprehensive answers\n"
                "- Always maintain professional financial advisory tone\n\n"
                
                "IMPORTANT: Your reasoning process should be internal. Only show the final, context-based answer."
            ),

            "query_classifier": PromptTemplate(
                """You are a smart query classifier. Your task is to categorize each user query into one of two labels:

                - RETRIEVE: Queries that require fetching additional information from the database (e.g., questions about data, documents, or specialized knowledge).
                - DIRECT: Queries that do not require database retrieval (e.g., greetings, small talk, or general questions the LLM can answer directly).
                Return exactly one of the two labels: RETRIEVE or DIRECT. Do not include any explanations or additional text.

                Query: {query}

                Classification:"""
            ),

        }
        
    def add_template(self, name: str, template: str, template_type: str = "text") -> None:
        """
        Add a new template or replace an existing one.
        
        Args:
            name: Template name
            template: Template string
            template_type: Type of template ('text' or 'chat')
        """
        self._templates[name] = PromptTemplate(template, template_type)

        
    def get_template(self, template_name: str) -> PromptTemplate:
        """
        Get a template by name.
        
        Args:
            template_name: Name of the template
            
        Returns:
            PromptTemplate: The requested template
            
        Raises:
            ValueError: If template not found
        """
        if template_name not in self._templates:
            raise ValueError(f"Template '{template_name}' not found")
        return self._templates[template_name]
    
    def format_messages(self, 
                        system_template: str, 
                        user_content: str, 
                        history_messages: Optional[List[Dict[str, str]]] = None, 
                        **kwargs) -> List[Dict[str, str]]:
        """
        Format a complete message list with system, history, and user messages.
        
        Args:
            system_template: Name of system template
            user_content: User message content
            history_messages: Previous messages in the conversation
            **kwargs: Variables for template formatting
            
        Returns:
            List[Dict[str, str]]: List of messages in OpenAI format
        """
        try:
            system_content = self.get_template(system_template).format(**kwargs)
            
            messages = [{"role": "system", "content": system_content}]
            
            if history_messages:
                messages.extend(history_messages)
                
            messages.append({"role": "user", "content": user_content})
            
            return messages
        except Exception as e:
            self.logger.error(f"Error formatting messages: {str(e)}")
            # Fallback to simple system message
            return [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content}
            ]
    
    
    def format_react_cot_messages(self,
                                  query: str,
                                  context: str = "",
                                  history_messages: Optional[List[Dict[str, str]]] = None,
                                  enable_thinking: bool = True
                                ) -> List[Dict[str, str]]:
        """
        Format messages specifically for ReAct+CoT (Reasoning and Acting with Chain of Thought).
        
        Args:
            query: User query
            context: Retrieved context (optional)
            history_messages: Previous messages in the conversation
            
        Returns:
            List[Dict[str, str]]: List of messages in OpenAI format
        """
        template_name = "react_cot_rag_system" if context else "react_cot_system"

        thinking_instruction = "First THINK carefully before providing your answer." if enable_thinking else "DO NOT think."

        return self.format_messages(
            system_template=template_name,
            user_content=query,
            history_messages=history_messages,
            context=context,
            thinking_instruction=thinking_instruction
        )


# Create a singleton instance for global use
prompt_manager = PromptManager()