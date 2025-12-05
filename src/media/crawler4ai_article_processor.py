# import asyncio
# import logging
# from typing import Optional, Dict, Any
# from dataclasses import dataclass
# from crawl4ai import AsyncWebCrawler
# from bs4 import BeautifulSoup
# import cloudscraper
# import random
# import time
# import requests
# from requests.adapters import HTTPAdapter
# from urllib3.util.retry import Retry
# import re
# import os
# from dotenv import load_dotenv

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import ChatOpenAI

# from langchain_core.prompts import ChatPromptTemplate

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document

# logger = logging.getLogger(__name__)

# @dataclass
# class URLResult:
#     """Dataclass for URL processing result"""
#     success: bool
#     url: str
#     title: Optional[str] = None
#     summary: Optional[str] = None
#     error: Optional[str] = None
#     original_length: int = 0
#     summary_length: int = 0
#     tokens_used: Optional[int] = None
#     method: Optional[str] = None

# @dataclass
# class CrawlResult:
#     """Dataclass for crawl result"""
#     url: str
#     title: str
#     content: str
#     success: bool
#     error: Optional[str] = None
#     metadata: Optional[Dict[str, Any]] = None

# class LangChainSummarizer:
#     """
#     Enhanced summarizer using map-reduce chunking for unlimited content length.
#     No truncation — processes entire content through chunking.
#     """
#     def __init__(self, api_key: str, model: str = "gpt-4.1-nano", max_tokens: int = 4000, temperature: float = 0.3):
#         self.api_key = api_key
#         self.llm = ChatOpenAI(
#             openai_api_key=api_key,
#             model=model,
#             temperature=temperature,
#             max_tokens=max_tokens
#         )

#         # chunking config
#         self.chunk_size = 2000
#         self.chunk_overlap = 200
#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap,
#             length_function=len,
#             separators=["\n\n", "\n", ". ", " ", ""]
#         )

#         self.language_names = {
#             "en": "English",
#             "vi": "Vietnamese (Tiếng Việt)",
#             "zh": "Chinese (中文（简体）)",
#             "ja": "Japanese (日本語)",
#             "ko": "Korean (한국어)",
#             "fr": "French (Français)",
#             "de": "German (Deutsch)",
#             "es": "Spanish (Español)",
#             "it": "Italian (Italiano)",
#             "pt": "Portuguese (Português)",
#             "ru": "Russian (Русский)",
#             "ar": "Arabic (العربية)"
#         }

#         self._init_prompts()
#         logger.info("LangChainSummarizer initialized — no truncation, unlimited content length")

#     def _init_prompts(self):
#         map_template = """
# You are a professional content summarizer. Summarize this text chunk in {language_name} using clear markdown formatting.

# FORMATTING RULES:
# - Use **bold** for key terms and important concepts
# - Use bullet points (•) for listing ideas
# - Keep paragraphs short (2-3 sentences max)
# - Focus on extracting main points and key information
# - Preserve important details, data, and examples

# LANGUAGE REQUIREMENT:
# - Write ONLY in {language_name}
# - Ensure natural, fluent expression in the target language

# Title: {title}
# Chunk Content:
# {text}

# Markdown Summary in {language_name}:
# """
#         combine_template = """
# You are a professional content editor. Create a comprehensive, well-formatted summary in {language_name} from these chunk summaries.

# MARKDOWN FORMATTING REQUIREMENTS:
# 1. Start with a **📌 Overview** section (2-3 sentences capturing the essence)
# 2. Use ## headers for main sections if multiple topics exist
# 3. Use **bold** for key terms, names, numbers, and important facts
# 4. Use bullet points (•) for listing related ideas
# 5. Use > blockquotes for important quotes or critical findings
# 6. End with a **🎯 Key Takeaways** section with 3-5 main points
# 7. Keep paragraphs short and scannable (2-3 sentences)
# 8. Use horizontal rules (---) to separate major sections if needed

# CONTENT REQUIREMENTS:
# - Write ENTIRELY in {language_name}
# - Maintain logical flow between sections
# - Remove redundancy while preserving all unique information
# - Prioritize most important information
# - Be comprehensive — include all significant details from chunks
# - Ensure readability and engagement

# Title: **{title}**

# Chunk Summaries to Combine:
# {text}

# Professional Markdown Summary in {language_name}:
# """

#         self.map_prompt = ChatPromptTemplate.from_template(map_template)
#         self.reduce_prompt = ChatPromptTemplate.from_template(combine_template)

#         self.map_chain = self.map_prompt | self.llm | StrOutputParser()
#         self.reduce_chain = self.reduce_prompt | self.llm | StrOutputParser()

#     async def summarize(self, transcript: str, target_language: str = "en", video_title: Optional[str] = None) -> str:
#         title = video_title or "Content"
#         language_name = self.language_names.get(target_language, target_language)
#         content_length = len(transcript)
#         logger.info(f"Processing {content_length} characters in {language_name}")

#         if content_length < self.chunk_size:
#             logger.info("Short content — using direct summarization")
#             return await self._direct_summary(transcript, title, language_name)
#         else:
#             num_chunks = (content_length // (self.chunk_size - self.chunk_overlap)) + 1
#             logger.info(f"Long content detected — ~{num_chunks} chunks")
#             return await self._chunked_summary(transcript, title, language_name)


#     async def summarize_enhanced(self, transcript: str, target_language: str = "en",
#                                 video_title: Optional[str] = None, 
#                                 custom_prompt: Optional[str] = None, **kwargs) -> str:
#         """
#         Enhanced summarize method with custom prompt support
#         Processes ENTIRE content without truncation
#         """
#         if custom_prompt:
#             # Custom prompt - still process entire content through chunks if needed
#             if len(transcript) > 5000:
#                 # Chunk and process with custom prompt
#                 logger.info("Using custom prompt with chunking for long content")
#                 chunks = self.text_splitter.split_text(transcript)
#                 chunk_summaries = []
                
#                 for i, chunk in enumerate(chunks):
#                     prompt = f"{custom_prompt}\n\n[Part {i+1}/{len(chunks)}]\nContent:\n{chunk}"
#                     response = await self.llm.ainvoke(prompt)
#                     chunk_summaries.append(response.content.strip())
                
#                 # Combine chunk summaries
#                 combine_prompt = f"Combine these summaries into one comprehensive summary:\n\n"
#                 combine_prompt += "\n\n---\n\n".join(chunk_summaries)
#                 final_response = await self.llm.ainvoke(combine_prompt)
#                 return final_response.content.strip()
#             else:
#                 # Short content with custom prompt
#                 response = await self.llm.ainvoke(custom_prompt + f"\n\nContent:\n{transcript}")
#                 return response.content.strip()
        
#         # Standard enhanced summarization (no truncation)
#         return await self.summarize(transcript, target_language, video_title)
    
#     async def _direct_summary(self, transcript: str, title: str, language_name: str) -> str:
#         prompt = f"""
# You are a professional content summarizer. Create a comprehensive summary in {language_name} with excellent markdown formatting.

# ## 📌 Overview
# Brief overview of the main topic (2-3 sentences)

# ## 📊 Main Points
# • Key points with **bold** emphasis
# • Important details and findings
# • Relevant examples or data

# ## 💡 Important Details
# Critical information, conclusions, or insights
# > Include any significant quotes if present

# ## 🎯 Key Takeaways
# • Main takeaway 1
# • Main takeaway 2
# • Main takeaway 3

# REQUIREMENTS:
# - Write ENTIRELY in {language_name}
# - Use **bold** for emphasis on key terms and numbers
# - Keep sections concise but comprehensive
# - Include all significant information

# Title: **{title}**

# Content to summarize:
# {transcript}

# Enhanced Markdown Summary in {language_name}:
# """
#         resp = await self.llm.ainvoke(prompt)
#         summary = resp.content.strip()
#         summary = self._post_process_markdown(summary)
#         logger.info(f"Direct summary generated: {len(summary)} chars from {len(transcript)} chars input")
#         return summary

#     async def _chunked_summary(self, transcript: str, title: str, language_name: str) -> str:
#         docs = self.text_splitter.create_documents([transcript])
#         num_chunks = len(docs)
#         logger.info(f"Processing {num_chunks} chunks")

#         tasks = [
#             self.map_chain.ainvoke({
#                 "text": doc.page_content,
#                 "language_name": language_name,
#                 "title": title
#             })
#             for doc in docs
#         ]
#         partials = await asyncio.gather(*tasks)

#         combined = "\n\n---\n\n".join(partials)
#         reduce_resp = await self.reduce_chain.ainvoke({
#             "text": combined,
#             "language_name": language_name,
#             "title": title
#         })
#         summary = reduce_resp.strip()
#         summary = self._post_process_markdown(summary)

#         total_input = sum(len(d.page_content) for d in docs)
#         logger.info(f"Chunked summary complete: {num_chunks} chunks, {total_input} chars → {len(summary)} chars")
#         return summary

#     def _post_process_markdown(self, summary: str) -> str:
#         summary = re.sub(r'(#{1,3}[^#\n]+)\n([^\n])', r'\1\n\n\2', summary)
#         summary = re.sub(r'\n•', r'\n• ', summary)
#         summary = re.sub(r'\n-([^\s])', r'\n- \1', summary)
#         summary = re.sub(r'\n\*([^\s*])', r'\n* \1', summary)
#         summary = re.sub(r'\n>', r'\n\n>', summary)
#         summary = re.sub(r'>\n([^>\n])', r'>\n\n\1', summary)
#         summary = re.sub(r'\n{4,}', r'\n\n\n', summary)
#         summary = re.sub(r'(\*\*[📌🎯📊💡🔍].+?\*\*)', r'\n\1\n', summary)
#         return summary.strip()

#     async def _emergency_fallback(self, transcript: str, title: str, target_language: str) -> str:
#         language_name = self.language_names.get(target_language, target_language)
#         prompt = f"""Create a brief summary in {language_name}:

# **{title}**

# {transcript}

# Brief summary in {language_name}:"""
#         try:
#             resp = await self.llm.ainvoke(prompt)
#             return resp.content.strip()
#         except Exception as e:
#             logger.error(f"Emergency fallback failed: {str(e)}")
#             return f"Summary generation failed: {str(e)}"

# class URLProcessor:
#     """
#     URL processor with enhanced summarization
#     """
#     def __init__(self, api_key: Optional[str] = None):
#         load_dotenv()
#         self.api_key = api_key or os.getenv("OPENAI_API_KEY")
#         if not self.api_key:
#             raise ValueError("OPENAI_API_KEY not found in environment")

#         self.summarizer = LangChainSummarizer(
#             api_key=self.api_key,
#             model=os.getenv("OPENAI_MODEL", "gpt-4.1-nano"),
#             max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
#             temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3"))
#         )
#         self.timeout = int(os.getenv("CRAWL_TIMEOUT", "50"))
#         self.language = os.getenv("SUMMARY_LANGUAGE", "english")

#     async def process_url(self, url: str, language: Optional[str] = None) -> URLResult:
#         try:
#             crawl_result = await crawl_url_async(url, timeout=self.timeout)

#             if not crawl_result.success:
#                 return URLResult(
#                     success=False,
#                     url=url,
#                     error=f"Error accessing page: {crawl_result.error}"
#                 )
#             summary_language = language or self.language
#             lang_code_map = {
#                 "english": "en", "vietnamese": "vi", "chinese": "zh",
#                 "japanese": "ja", "korean": "ko", "french": "fr",
#                 "german": "de", "spanish": "es", "italian": "it",
#                 "portuguese": "pt", "russian": "ru", "arabic": "ar"
#             }
#             lang_code = lang_code_map.get(summary_language.lower(), "en")
#             logger.info(f"Summarizing {len(crawl_result.content)} chars in {summary_language}")

#             summary = await self.summarizer.summarize(
#                 transcript=crawl_result.content,
#                 target_language=lang_code,
#                 video_title=crawl_result.title
#             )

#             return URLResult(
#                 success=True,
#                 url=url,
#                 title=crawl_result.title,
#                 summary=summary,
#                 original_length=len(crawl_result.content),
#                 summary_length=len(summary),
#                 tokens_used=len(summary) // 4,
#                 method=crawl_result.metadata.get("method") if crawl_result.metadata else "unknown"
#             )

#         except Exception as e:
#             logger.error(f"Unexpected error processing {url}: {str(e)}")
#             return URLResult(
#                 success=False,
#                 url=url,
#                 error=f"Error: {str(e)}"
#             )

import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import cloudscraper
import random
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from src.providers.provider_factory import ModelProviderFactory, ProviderType
from src.providers.base_provider import ModelProvider

logger = logging.getLogger(__name__)

@dataclass
class URLResult:
    """Dataclass for URL processing result"""
    success: bool
    url: str
    title: Optional[str] = None
    summary: Optional[str] = None
    error: Optional[str] = None
    original_length: int = 0
    summary_length: int = 0
    tokens_used: Optional[int] = None
    method: Optional[str] = None

@dataclass
class CrawlResult:
    """Dataclass for crawl result"""
    url: str
    title: str
    content: str
    success: bool
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class LangChainSummarizer:
    """
    Summarizer using provider factory for unlimited content length. 
    No truncation — processes entire content through chunking.
    """
    def __init__(
        self, 
        api_key: str, 
        model: str = "gpt-4.1-nano", 
        provider_type: str = "openai", 
        max_tokens: int = 4000, 
        temperature: float = 0.7
    ):
        """
        Initialize summarizer with support for multiple providers using factory
        
        Args:
            api_key: API key (not used for Ollama)
            model: Model name
            provider_type: "openai", "ollama", or "gemini"
            max_tokens: Max tokens for response
            temperature: Temperature for generation
        """
        self.api_key = api_key
        self.model = model
        self.provider_type = provider_type.lower()
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Create provider using factory
        try:
            self.provider: ModelProvider = ModelProviderFactory.create_provider(
                provider_type=self.provider_type,
                model_name=model,
                api_key=api_key if self.provider_type != ProviderType.OLLAMA else None
            )
            logger.info(f"LangChainSummarizer initialized with {self.provider_type} provider, model: {model}")
        except Exception as e:
            logger.error(f"Failed to create provider: {e}")
            raise

        # Chunking config
        self.chunk_size = 2000
        self.chunk_overlap = 200
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Language mapping
        self.language_names = {
            "en": "English",
            "vi": "Vietnamese (Tiếng Việt)",
            "zh": "Chinese (中文（简体）)",
            "ja": "Japanese (日本語)",
            "ko": "Korean (한국어)",
            "fr": "French (Français)",
            "de": "German (Deutsch)",
            "es": "Spanish (Español)",
            "it": "Italian (Italiano)",
            "pt": "Portuguese (Português)",
            "ru": "Russian (Русский)",
            "ar": "Arabic (العربية)"
        }

        logger.info("LangChainSummarizer initialized — no truncation, unlimited content length")

    async def summarize(
        self, 
        transcript: str, 
        target_language: str = "en", 
        video_title: Optional[str] = None
    ) -> str:
        """
        Summarize with NO TRUNCATION - handles unlimited content length
        
        Args:
            transcript: Full content to summarize
            target_language: Target language code
            video_title: Optional title for context
            
        Returns:
            Comprehensive summary
        """
        # Initialize provider
        await self.provider.initialize()
        
        title = video_title or "Content"
        language_name = self.language_names.get(target_language, target_language)
        content_length = len(transcript)
        
        logger.info(f"Processing {content_length} characters in {language_name}")

        # Short content - direct summarization
        if content_length < self.chunk_size:
            logger.info("Short content — using direct summarization")
            return await self._direct_summary(transcript, title, language_name)
        
        # Long content - chunk and process
        num_chunks = (content_length // (self.chunk_size - self.chunk_overlap)) + 1
        logger.info(f"Long content detected — ~{num_chunks} chunks")
        return await self._chunked_summary(transcript, title, language_name)

    async def summarize_enhanced(
        self, 
        transcript: str, 
        target_language: str = "en",
        video_title: Optional[str] = None, 
        custom_prompt: Optional[str] = None,
        content_type: Optional[str] = None,
        content_focus: Optional[str] = None,
        content_structure: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Enhanced summarization with content-type optimization
        Processes ENTIRE content without truncation
        """
        # Initialize provider
        await self.provider.initialize()
        
        title = video_title or "Content"
        language_name = self.language_names.get(target_language, target_language)
        
        # Build enhanced prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_enhanced_prompt(
                language_name=language_name,
                title=title,
                content_type=content_type,
                content_focus=content_focus,
                content_structure=content_structure
            )
        
        # Handle long content with chunking
        if len(transcript) > self.chunk_size:
            logger.info("Using chunking for long content with custom prompt")
            return await self._chunked_summary_with_prompt(
                transcript=transcript,
                prompt=prompt,
                language_name=language_name,
                title=title
            )
        
        # Short content - direct generation
        messages = [
            {"role": "system", "content": "You are an expert content summarizer."},
            {"role": "user", "content": f"{prompt}\n\nContent:\n{transcript}"}
        ]
        
        response = await self.provider.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.get("content", "").strip()

    async def summarize_streaming(
        self,
        transcript: str,
        target_language: str = "en",
        video_title: Optional[str] = None,
        custom_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream summary generation for real-time output
        """
        # Initialize provider
        await self.provider.initialize()
        
        title = video_title or "Content"
        language_name = self.language_names.get(target_language, target_language)
        
        # Build prompt
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = self._build_enhanced_prompt(
                language_name=language_name,
                title=title
            )
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are an expert content summarizer."},
            {"role": "user", "content": f"{prompt}\n\nContent:\n{transcript}"}
        ]
        
        try:
            # Stream using provider
            async for chunk in self.provider.stream(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            ):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming summarization: {e}")
            raise

    def _build_enhanced_prompt(
        self,
        language_name: str,
        title: str,
        content_type: Optional[str] = None,
        content_focus: Optional[str] = None,
        content_structure: Optional[str] = None
    ) -> str:
        """Build enhanced prompt for summarization"""
        
        prompt = f"""
Create a comprehensive summary in {language_name}.

**FORMATTING REQUIREMENTS:**
- Use clear markdown formatting for better readability
- Start with a brief overview (2-3 sentences)
- Use **bold** for key points and important terms
- Use bullet points (•) for listing main ideas
- Include section headers with ## if the content has multiple topics
- Add a "Key Takeaways" section at the end with 3-5 main points
- Keep paragraphs short and scannable
- Use > blockquotes for important quotes or findings if relevant

**CONTENT REQUIREMENTS:**
- Write entirely in {language_name}
- Focus on main arguments, key facts, and conclusions
- Maintain logical flow between sections
- Be concise but comprehensive

Title: **{title}**
"""
        
        if content_type:
            prompt += f"\nContent Type: {content_type}"
        
        if content_focus:
            prompt += f"\nFocus on: {content_focus}"
        
        if content_structure:
            prompt += f"\nStructure: {content_structure}"
        
        return prompt

    async def _direct_summary(
        self, 
        transcript: str, 
        title: str, 
        language_name: str
    ) -> str:
        """Direct summary for short content"""
        
        prompt = f"""
You are a professional content summarizer. Create a comprehensive summary in {language_name} with excellent markdown formatting.

**MARKDOWN FORMAT STRUCTURE:**

## 📌 Overview
Brief overview of the main topic (2-3 sentences)

## 📊 Main Points
- Key points with **bold** emphasis
- Important details and findings
- Relevant examples or data

## 💡 Important Details
Critical information, conclusions, or insights
> Include any significant quotes if present

## 🎯 Key Takeaways
- Main takeaway 1
- Main takeaway 2
- Main takeaway 3

REQUIREMENTS:
- Write ENTIRELY in {language_name}
- Use **bold** for emphasis on key terms and numbers
- Keep sections concise but comprehensive
- Include all significant information

Title: **{title}**

Content to summarize:
{transcript}

Enhanced Markdown Summary in {language_name}:
"""
        
        messages = [
            {"role": "system", "content": "You are an expert content summarizer."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.provider.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        summary = response.get("content", "").strip()
        summary = self._post_process_markdown(summary)
        
        logger.info(f"Direct summary: {len(summary)} chars from {len(transcript)} chars")
        return summary

    async def _chunked_summary(
        self, 
        transcript: str, 
        title: str, 
        language_name: str
    ) -> str:
        """Process content through chunking for long text"""
        
        # Split into chunks
        chunks = self.text_splitter.split_text(transcript)
        num_chunks = len(chunks)
        
        logger.info(f"Processing {num_chunks} chunks")

        # Map phase: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            map_prompt = f"""
Summarize this text chunk ({i+1}/{num_chunks}) in {language_name} using clear markdown formatting.

FORMATTING RULES:
- Use **bold** for key terms
- Use bullet points (•) for main ideas
- Keep paragraphs short (2-3 sentences)
- Preserve important details, data, and examples

Title: {title}
Chunk Content:
{chunk}

Summary in {language_name}:
"""
            
            messages = [
                {"role": "system", "content": "You are an expert content summarizer."},
                {"role": "user", "content": map_prompt}
            ]
            
            response = await self.provider.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000
            )
            
            chunk_summaries.append(response.get("content", "").strip())

        # Reduce phase: Combine all summaries
        combined_text = "\n\n---\n\n".join(chunk_summaries)
        
        reduce_prompt = f"""
Create a comprehensive, well-formatted summary in {language_name} from these chunk summaries.

MARKDOWN FORMATTING:
1. Start with **📌 Overview** (2-3 sentences)
2. Use ## headers for main sections
3. Use **bold** for key terms
4. Use bullet points (•) for related ideas
5. End with **🎯 Key Takeaways** (3-5 points)

CONTENT REQUIREMENTS:
- Write ENTIRELY in {language_name}
- Remove redundancy while preserving unique information
- Maintain logical flow
- Be comprehensive

Title: **{title}**

Chunk Summaries:
{combined_text}

Final Summary in {language_name}:
"""
        
        messages = [
            {"role": "system", "content": "You are an expert content editor."},
            {"role": "user", "content": reduce_prompt}
        ]
        
        response = await self.provider.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        summary = response.get("content", "").strip()
        summary = self._post_process_markdown(summary)

        logger.info(f"Chunked summary: {num_chunks} chunks → {len(summary)} chars")
        return summary

    async def _chunked_summary_with_prompt(
        self,
        transcript: str,
        prompt: str,
        language_name: str,
        title: str
    ) -> str:
        """Chunked summary with custom prompt"""
        
        chunks = self.text_splitter.split_text(transcript)
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            chunk_prompt = f"{prompt}\n\n[Part {i+1}/{len(chunks)}]\nContent:\n{chunk}"
            
            messages = [
                {"role": "system", "content": "You are an expert content summarizer."},
                {"role": "user", "content": chunk_prompt}
            ]
            
            response = await self.provider.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000
            )
            
            chunk_summaries.append(response.get("content", "").strip())
        
        # Combine summaries
        combine_prompt = f"Combine these summaries into one comprehensive summary in {language_name}:\n\n"
        combine_prompt += "\n\n---\n\n".join(chunk_summaries)
        
        messages = [
            {"role": "system", "content": "You are an expert content editor."},
            {"role": "user", "content": combine_prompt}
        ]
        
        final_response = await self.provider.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return final_response.get("content", "").strip()

    def _post_process_markdown(self, summary: str) -> str:
        """Post-process to ensure proper markdown formatting"""
        
        # Fix header spacing
        summary = re.sub(r'(#{1,3}[^#\n]+)\n([^\n])', r'\1\n\n\2', summary)
        
        # Fix bullet points
        summary = re.sub(r'\n•', r'\n• ', summary)
        summary = re.sub(r'\n-([^\s])', r'\n- \1', summary)
        summary = re.sub(r'\n\*([^\s*])', r'\n* \1', summary)
        
        # Fix blockquotes
        summary = re.sub(r'\n>', r'\n\n>', summary)
        summary = re.sub(r'>\n([^>\n])', r'>\n\n\1', summary)
        
        # Remove excessive blank lines
        summary = re.sub(r'\n{4,}', r'\n\n\n', summary)
        
        # Fix emoji headers
        summary = re.sub(r'(\*\*[📌🎯📊💡🔍].+?\*\*)', r'\n\1\n', summary)
        
        return summary.strip()


class URLProcessor:
    """URL processor with enhanced summarization"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-nano",
        provider_type: str = "openai"
    ):
        load_dotenv()
        
        # Get API key based on provider
        if provider_type.lower() == ProviderType.OLLAMA:
            self.api_key = ""  # Ollama doesn't need API key
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(f"API key not found for {provider_type}")

        # Initialize summarizer with provider support
        self.summarizer = LangChainSummarizer(
            api_key=self.api_key,
            model=model,
            provider_type=provider_type,
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        )
        
        self.timeout = int(os.getenv("CRAWL_TIMEOUT", "50"))
        self.language = os.getenv("SUMMARY_LANGUAGE", "english")

    async def process_url(self, url: str, language: Optional[str] = None) -> URLResult:
        """Process URL: crawl content then summarize WITHOUT TRUNCATION"""
        
        try:
            # Crawl content
            crawl_result = await crawl_url_async(url, timeout=self.timeout)

            if not crawl_result.success:
                return URLResult(
                    success=False,
                    url=url,
                    error=f"Error accessing page: {crawl_result.error}"
                )
            
            summary_language = language or self.language
            
            # Language code mapping
            lang_code_map = {
                "english": "en", "vietnamese": "vi", "chinese": "zh",
                "japanese": "ja", "korean": "ko", "french": "fr",
                "german": "de", "spanish": "es", "italian": "it",
                "portuguese": "pt", "russian": "ru", "arabic": "ar"
            }
            lang_code = lang_code_map.get(summary_language.lower(), "en")
            
            logger.info(f"Summarizing {len(crawl_result.content)} chars in {summary_language}")

            # Summarize ENTIRE content
            summary = await self.summarizer.summarize(
                transcript=crawl_result.content,
                target_language=lang_code,
                video_title=crawl_result.title
            )

            return URLResult(
                success=True,
                url=url,
                title=crawl_result.title,
                summary=summary,
                original_length=len(crawl_result.content),
                summary_length=len(summary),
                tokens_used=len(summary) // 4,
                method=crawl_result.metadata.get("method") if crawl_result.metadata else "unknown"
            )

        except Exception as e:
            logger.error(f"Unexpected error processing {url}: {str(e)}")
            return URLResult(
                success=False,
                url=url,
                error=f"Error: {str(e)}"
            )

class WebCrawler:
    """Web crawler with anti-detection and fallback"""
    def __init__(self, timeout: int = 30, user_agent: Optional[str] = None, proxy_list: Optional[list] = None):
        self.timeout = timeout
        self.proxy_list = proxy_list or []
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15"
        ]
        self.user_agent = user_agent or random.choice(self.user_agents)

    async def crawl_url(self, url: str, extract_main_content: bool = True) -> CrawlResult:
        try:
            browser_config = {
                "headless": True,
                "verbose": False,
                "browser_type": "chromium",
                "user_agent": self.user_agent,
                "viewport_width": 1920,
                "viewport_height": 1080,
                "headers": {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                },
                "browser_args": [
                    "--no-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-web-security",
                    "--disable-features=site-per-process",
                ]
            }

            async with AsyncWebCrawler(**browser_config) as crawler:
                stealth_js = """
                    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                    window.chrome = { runtime: {} };
                    const originalQuery = window.navigator.permissions.query;
                    window.navigator.permissions.query = (parameters) => (
                        parameters.name === 'notifications' ?
                            Promise.resolve({ state: Notification.permission }) :
                            originalQuery(parameters)
                    );
                    window.scrollTo(0, document.body.scrollHeight);
                    await new Promise(resolve => setTimeout(resolve, 3000));
                    await new Promise(resolve => setTimeout(resolve, Math.random() * 2000 + 1000));
                """
                
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=10,
                    # extraction_strategy="NoExtractionStrategy",
                    # chunking_strategy="RegexChunking",
                    bypass_cache=True,
                    wait_for="body",
                    timeout=self.timeout,
                    delay_before_return_html=2.0,
                    js_code=stealth_js,
                    magic=True
                )

                if not result.success:
                    return CrawlResult(
                        url=url,
                        title="",
                        content="",
                        success=False,
                        error=f"Crawl failed: {result.error_message}"
                    )

                title = self._extract_title(result.html)
                content = self._extract_content(result.html) if extract_main_content else result.cleaned_html or result.html
                cleaned_content = self._clean_content(content)

                if not cleaned_content or len(cleaned_content.strip()) < 250:
                    cleaned_content = ""

                return CrawlResult(
                    url=url,
                    title=title,
                    content=cleaned_content,
                    success=True,
                    metadata={
                        "method": "crawl4ai",
                        "word_count": len(cleaned_content.split()) if cleaned_content else 0,
                        "links_count": len(result.links) if result.links else 0,
                        "images_count": len(result.media) if result.media else 0
                    }
                )
        except Exception as e:
            logger.warning(f"crawl4ai failed: {str(e)}, falling back...")
            return await self._fallback_crawl(url)

    async def _fallback_crawl(self, url: str) -> CrawlResult:
        # fallback logic as you have it...
        try:
            scraper = cloudscraper.create_scraper(browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False})
            time.sleep(random.uniform(0.5, 2.0))
            response = scraper.get(url, timeout=self.timeout)
            response.raise_for_status()
            html_content = response.text
            title = self._extract_title(html_content)
            content = self._extract_content(html_content)
            cleaned_content = self._clean_content(content)
            return CrawlResult(
                url=url,
                title=title,
                content=cleaned_content,
                success=True,
                metadata={
                    "method": "cloudscraper",
                    "status_code": response.status_code,
                    "word_count": len(cleaned_content.split())
                }
            )
        except Exception as e:
            logger.warning(f"Cloudscraper failed: {str(e)}, trying requests...")

        try:
            session = requests.Session()
            retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0',
            }
            time.sleep(random.uniform(0.5, 2.0))
            response = session.get(url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            html_content = response.text
            title = self._extract_title(html_content)
            content = self._extract_content(html_content)
            cleaned_content = self._clean_content(content)
            return CrawlResult(
                url=url,
                title=title,
                content=cleaned_content,
                success=True,
                metadata={
                    "method": "requests_fallback",
                    "status_code": response.status_code,
                    "word_count": len(cleaned_content.split())
                }
            )
        except Exception as e:
            logger.error(f"Fallback crawl failed for {url}: {str(e)}")
            return CrawlResult(
                url=url,
                title="",
                content="",
                success=False,
                error=f"All crawl methods failed: {str(e)}"
            )
    
    def _extract_title(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title_el = soup.find('title')
            if title_el:
                return title_el.get_text().strip()
            h1 = soup.find('h1')
            if h1:
                return h1.get_text().strip()
            return "No title"
        except:
            return "No title"

    def _extract_content(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "meta", "link", "noscript"]):
                tag.decompose()
            content_selectors = [
                '.article__content', '.articleBody', '.story__content', '.mw-story-text',
                'div[class*="article"]', 'div[class*="story"]', 'div[class*="content"]',
                'article', '.article-content', '.post-content', '.content', 'main',
                '.main-content', '[role="main"]', '.story-body', '.article-body',
                '.entry-content', '.post-body', '.story-content', 'div[data-module="ArticleBody"]',
                'div p, section p, article p'
            ]
            best = ""
            best_len = 0
            for sel in content_selectors:
                elems = soup.select(sel)
                if elems:
                    if sel.endswith(' p'):
                        text = ' '.join([e.get_text(separator=' ', strip=True) for e in elems])
                    else:
                        text = elems[0].get_text(separator=' ', strip=True)
                    if len(text) > best_len and len(text) > 50:
                        best = text
                        best_len = len(text)
            if best and len(best) > 100:
                return best
            paras = soup.find_all('p')
            if paras:
                t = ' '.join([p.get_text(separator=' ', strip=True) for p in paras])
                if len(t) > 200:
                    return t
            body = soup.find('body')
            if body:
                for noise in body.find_all(['button','input','form','iframe','advertisement','ad']):
                    noise.decompose()
                return body.get_text(separator=' ', strip=True)
            return soup.get_text(separator=' ', strip=True)
        except Exception as e:
            logger.warning(f"Content extraction failed: {str(e)}")
            return ""

    def _clean_content(self, content: str) -> str:
        content = re.sub(r'\s+', ' ', content)
        return content.strip()

# Helper functions
async def crawl_url_async(url: str, timeout: int = 60) -> CrawlResult:
    crawler = WebCrawler(timeout=timeout)
    return await crawler.crawl_url(url)

def crawl_url_sync(url: str, timeout: int = 60) -> CrawlResult:
    return asyncio.run(crawl_url_async(url, timeout))
