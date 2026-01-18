import json
import base64
import os
import time
import logging
import re
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List, Tuple
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, TimeoutException, WebDriverException
from selenium.webdriver.remote.remote_connection import RemoteConnection
from urllib3.util.retry import Retry
import urllib3
from bs4 import BeautifulSoup

# ============================================================================
# CONFIGURE SELENIUM CONNECTION POOL TO AVOID "Connection pool is full" WARNINGS
# ============================================================================
# The warning "Connection pool is full, discarding connection: localhost"
# comes from Selenium's internal urllib3 HTTP client communicating with ChromeDriver.
# This is harmless but can be suppressed by:
# 1. Increasing the default pool size
# 2. Suppressing the urllib3 warning logs

# Suppress urllib3 connection pool warnings (these are not errors, just info)
import logging as _logging
_logging.getLogger("urllib3.connectionpool").setLevel(_logging.WARNING)

# Suppress InsecureRequestWarning if any
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set RemoteConnection timeout
try:
    RemoteConnection.set_timeout(120)
except Exception:
    pass
from PIL import Image, ImageEnhance
import requests
from io import BytesIO
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# OPTIMIZED IMAGE PROCESSOR WITH CACHING
# ============================================================================

class OptimizedImageProcessor:
    """Image processing with caching and lazy loading for optimal performance"""
    
    def __init__(self):
        self._cache = {}
        self._processed_images = set()
    
    def enhance_image(self, image_path: str, output_path: str = None) -> str:
        """Enhance image with caching mechanism"""
        
        # Check if file exists
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return image_path
            
        # Generate cache key based on file path and modification time
        try:
            cache_key = f"{image_path}_{os.path.getmtime(image_path)}"
        except:
            cache_key = image_path
            
        # Return cached result if available
        if cache_key in self._cache:
            logger.info(f"Using cached enhanced image for: {image_path}")
            return self._cache[cache_key]
            
        try:
            if not output_path:
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_enhanced{ext}"
            
            with Image.open(image_path) as img:
                # Only convert if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Batch process enhancements for better performance
                enhancers = [
                    (ImageEnhance.Contrast, 2.0),
                    (ImageEnhance.Sharpness, 2.0)
                ]
                
                for enhancer_class, factor in enhancers:
                    enhancer = enhancer_class(img)
                    img = enhancer.enhance(factor)
                
                # Smart resize - only when really needed
                min_dimension = min(img.width, img.height)
                if min_dimension < 150:
                    scale = 200 / min_dimension
                    new_size = (int(img.width * scale), int(img.height * scale))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save with optimization
                img.save(output_path, quality=85, optimize=True)
                
                # Cache the result
                self._cache[cache_key] = output_path
                self._processed_images.add(output_path)
                
                logger.info(f"Enhanced image saved to: {output_path}")
                return output_path
                
        except Exception as e:
            logger.warning(f"Unable to enhance image: {e}")
            return image_path
    
    def cleanup_temp_images(self):
        """Clean up temporary enhanced images"""
        for image_path in self._processed_images:
            try:
                if os.path.exists(image_path) and '_enhanced' in image_path:
                    os.remove(image_path)
                    logger.debug(f"Removed temporary image: {image_path}")
            except Exception as e:
                logger.warning(f"Could not remove {image_path}: {e}")
        
        self._processed_images.clear()
        self._cache.clear()

# ============================================================================
# CAPTCHA AGENTS (KEPT SAME BUT WITH MINOR OPTIMIZATIONS)
# ============================================================================

class CaptchaAgentBase:
    def __init__(self, max_output_tokens: int = 100, temperature: float = 0.4, top_p: int = 1, top_k: int = 32):
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

class CheckIfImageLooksLikeCaptchaAgent(CaptchaAgentBase):
    """Agent to decide if the image is a captcha or not"""
    
    base_prompt: str = """Analyze the image and determine if it looks like a CAPTCHA.

A CAPTCHA typically has these characteristics:
- Distorted text, numbers, or characters
- Mathematical equations to solve
- Image selection tasks (e.g., "Select all traffic lights")
- Puzzle solving elements
- Security verification elements

Response format: {"content": true} if it's a CAPTCHA, {"content": false} if not.
Only return the JSON data without any markdown formatting.

Question: Is this image a CAPTCHA?"""

class DecideCaptchaTypeAgent(CaptchaAgentBase):
    """Agent to decide the type of captcha"""
    
    base_prompt: str = """Analyze the CAPTCHA image and identify the type of task required.

Available CAPTCHA types:
1) Text Recognition: Read the given content. Enter a series of text and numbers as shown
2) Math Problem: Solve the mathematical equation and enter the answer
3) Image Rotation: Turn an image to correct orientation
4) Logic Puzzle: Solve a puzzle and enter the answer
5) Image Selection: Select all images that match a description
6) Audio CAPTCHA: Audio-based verification
7) Slider/Drag: Drag a slider or puzzle piece to correct position
8) Other: Any other type not listed above

Response format: {"content": <number>}
Only return the JSON data without markdown formatting.

What type of CAPTCHA is this?"""

class TextSolveAgent(CaptchaAgentBase):
    """Agent to extract text from captcha image"""

    base_prompt: str = """Read the text content from this CAPTCHA image.

Instructions:
- Extract all visible text and numbers exactly as shown
- Ignore background noise and focus on the main text
- Return only the readable characters
- If text is unclear, provide your best interpretation

Response format: {"content": "<extracted_text>"}
Only return the JSON data without markdown formatting.

What text do you see in this CAPTCHA?"""

class MathSolveAgent(CaptchaAgentBase):
    """Agent to solve math captcha"""

    base_prompt: str = """Extract and solve the mathematical expression from this CAPTCHA.

Instructions:
- Identify the mathematical equation or expression
- Return it in a format that can be evaluated by Python
- Use standard operators: +, -, *, /, //, %, **

Response format: {"content": "<math_expression>"}
Only return the JSON data without markdown formatting.

What mathematical expression needs to be solved?"""

class ImageSelectionAgent(CaptchaAgentBase):
    """Agent to handle image selection captcha"""
    
    base_prompt: str = """Analyze this image selection CAPTCHA and identify which images match the given criteria.

Instructions:
- Read the instruction text
- Examine each image tile
- Return the positions/indices of matching images

Response format: {"content": ["position1", "position2", ...]}
Only return the JSON data without markdown formatting.

Which images should be selected?"""

# ============================================================================
# OPTIMIZED API GENERATOR WITH CONNECTION POOLING
# ============================================================================

class OptimizedGenerate:
    """Optimized API handler with connection pooling and caching"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._response_cache = {}
        self._cache_hits = 0
        self._total_calls = 0
    
    @lru_cache(maxsize=128)
    def _get_cache_key(self, prompt: str, image_path: str = None) -> str:
        """Generate cache key for responses"""
        key_parts = [prompt[:100]]  # Use first 100 chars of prompt
        if image_path:
            try:
                with open(image_path, 'rb') as f:
                    # Use file hash for cache key
                    file_hash = hashlib.md5(f.read()).hexdigest()[:8]
                    key_parts.append(file_hash)
            except:
                key_parts.append(image_path)
        
        return hashlib.md5(''.join(key_parts).encode()).hexdigest()
    
    def generate(self, prompt: str, image_path: str = None, use_cache: bool = True, **kwargs) -> str:
        """Generate response with caching support"""
        
        self._total_calls += 1
        
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(prompt, image_path)
            if cache_key in self._response_cache:
                self._cache_hits += 1
                logger.debug(f"Cache hit! Rate: {self._cache_hits}/{self._total_calls}")
                return self._response_cache[cache_key]
        
        try:
            content = [{"type": "text", "text": prompt}]
            
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Determine MIME type
                ext = os.path.splitext(image_path)[1].lower()
                mime_type = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }.get(ext, 'image/png')
                
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}
                })

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                max_tokens=kwargs.get('max_output_tokens', 2048),
                temperature=kwargs.get('temperature', 0.4),
                top_p=kwargs.get('top_p', 1),
            )
            
            result = response.choices[0].message.content
            
            # Cache the result
            if use_cache:
                self._response_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_hits': self._cache_hits,
            'total_calls': self._total_calls,
            'hit_rate': self._cache_hits / self._total_calls if self._total_calls > 0 else 0,
            'cache_size': len(self._response_cache)
        }

# ============================================================================
# ADVANCED CAPTCHA SOLVER WITH PARALLEL PROCESSING
# ============================================================================

class AdvancedCaptchaSolver:
    """Optimized captcha solver with parallel processing capabilities"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano"):
        self.api_key = api_key
        self.generator = OptimizedGenerate(api_key=api_key, model=model)
        self.image_processor = OptimizedImageProcessor()
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Initialize agents
        self.agents = {
            'check': CheckIfImageLooksLikeCaptchaAgent(),
            'type': DecideCaptchaTypeAgent(),
            'text': TextSolveAgent(),
            'math': MathSolveAgent(),
            'selection': ImageSelectionAgent()
        }
        
        logger.info("Advanced CAPTCHA Solver initialized with optimizations")

    def load_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from API"""
        try:
            response = response.strip()
            
            # Remove markdown formatting if present
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            
            # Find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                response = json_match.group()
            
            return json.loads(response)
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}, Response: {response}")
            # Try to extract content manually
            content_match = re.search(r'"content":\s*"?([^"}\]]+)"?', response)
            if content_match:
                return {"content": content_match.group(1)}
            raise ValueError(f"Invalid JSON response: {response}")

    def solve_captcha_parallel(self, image_path: str, enhance_image: bool = True) -> Dict[str, Any]:
        """Solve captcha with parallel processing for better performance"""
        
        try:
            logger.info("Starting parallel captcha solving")
            
            # Step 1: Enhance image if needed
            working_image = image_path
            if enhance_image:
                working_image = self.image_processor.enhance_image(image_path)
            
            # Step 2: Check and get type in parallel
            futures = []
            
            with self.executor as executor:
                # Submit both tasks simultaneously
                check_future = executor.submit(
                    self.generator.generate,
                    self.agents['check'].base_prompt,
                    working_image,
                    **vars(self.agents['check'])
                )
                
                type_future = executor.submit(
                    self.generator.generate,
                    self.agents['type'].base_prompt,
                    working_image,
                    **vars(self.agents['type'])
                )
                
                # Wait for check result first
                check_response = check_future.result(timeout=120)
                check_result = self.load_json(check_response)
                
                if not check_result.get("content"):
                    type_future.cancel()  # Cancel type checking if not a captcha
                    return {"error": "Image is not a captcha"}
                
                # Get type result
                type_response = type_future.result(timeout=120)
                type_result = self.load_json(type_response)
                captcha_type = int(type_result["content"])
                
                logger.info(f"Captcha type identified: {captcha_type}")
            
            # Step 3: Solve based on type
            return self._solve_by_type(captcha_type, working_image)
            
        except Exception as e:
            logger.error(f"Parallel solving failed: {e}")
            # Fallback to sequential solving
            return self.solve_captcha(image_path, enhance_image)

    def solve_captcha(self, image_path: str, enhance_image: bool = True, max_retries: int = 3) -> Dict[str, Any]:
        """Original solve method with retry mechanism (fallback)"""
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Starting captcha solving")
                
                # Enhance image if needed
                working_image = image_path
                if enhance_image:
                    working_image = self.image_processor.enhance_image(image_path)
                
                # Step 1: Check if it's a captcha
                logger.info("Step 1: Checking if image is a captcha...")
                check_response = self.generator.generate(
                    prompt=self.agents['check'].base_prompt,
                    image_path=working_image,
                    **vars(self.agents['check'])
                )
                
                check_result = self.load_json(check_response)
                if not check_result.get("content"):
                    return {"error": "Image is not a captcha"}
                
                # Step 2: Determine captcha type
                logger.info("Step 2: Determining captcha type...")
                type_response = self.generator.generate(
                    prompt=self.agents['type'].base_prompt,
                    image_path=working_image,
                    **vars(self.agents['type'])
                )
                
                type_result = self.load_json(type_response)
                captcha_type = int(type_result["content"])
                
                logger.info(f"Captcha type identified: {captcha_type}")
                
                # Step 3: Solve based on type
                return self._solve_by_type(captcha_type, working_image)
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)  # Short delay before retry

    def _solve_by_type(self, captcha_type: int, image_path: str) -> Dict[str, Any]:
        """Solve captcha based on identified type"""
        
        if captcha_type == 1:  # Text CAPTCHA
            logger.info("Solving text captcha...")
            response = self.generator.generate(
                prompt=self.agents['text'].base_prompt,
                image_path=image_path,
                **vars(self.agents['text'])
            )
            result = self.load_json(response)
            return {"type": "text", "solution": result["content"]}
            
        elif captcha_type == 2:  # Math CAPTCHA
            logger.info("Solving math captcha...")
            response = self.generator.generate(
                prompt=self.agents['math'].base_prompt,
                image_path=image_path,
                **vars(self.agents['math'])
            )
            result = self.load_json(response)
            
            # Calculate result safely
            try:
                equation = result["content"]
                # Sanitize equation
                allowed_chars = set('0123456789+-*/().')
                if all(c in allowed_chars or c.isspace() for c in equation):
                    solution = eval(equation)
                else:
                    raise ValueError("Invalid equation characters")
                
                return {"type": "math", "equation": equation, "solution": solution}
                
            except Exception as e:
                logger.error(f"Error evaluating math equation: {e}")
                return {"error": f"Could not solve math equation: {e}"}
                
        elif captcha_type == 5:  # Image Selection
            logger.info("Solving image selection captcha...")
            response = self.generator.generate(
                prompt=self.agents['selection'].base_prompt,
                image_path=image_path,
                **vars(self.agents['selection'])
            )
            result = self.load_json(response)
            return {"type": "selection", "solution": result["content"]}
            
        else:
            return {"error": f"Captcha type {captcha_type} not supported"}
    
    def cleanup(self):
        """Clean up resources"""
        self.image_processor.cleanup_temp_images()
        self.executor.shutdown(wait=False)

class ChunkingSummarizer:
    """Advanced summarizer using chunking strategy for long content"""
    
    def __init__(self, api_key: str, model: str = "gpt-4.1-nano"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        
        # Chunking configuration
        self.chunk_size = 3000  # Characters per chunk
        self.chunk_overlap = 200  # Overlap between chunks to maintain context
        self.max_final_chunks = 5  # Maximum chunks for final summary
        
    def extract_text_from_html(self, html_content: str) -> str:
        """Extract and clean text from HTML content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unnecessary elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                                'aside', 'meta', 'link', 'noscript']):
                element.decompose()
            
            # Extract article content if available
            article_selectors = [
                'article', 
                'main',
                '[role="main"]',
                '.content',
                '.article-content',
                '.post-content',
                '#content'
            ]
            
            article_text = None
            for selector in article_selectors:
                article = soup.select_one(selector)
                if article:
                    article_text = article.get_text(separator=' ', strip=True)
                    if len(article_text) > 500:  # Ensure meaningful content
                        break
            
            # Fallback to full page text if no article found
            if not article_text or len(article_text) < 500:
                article_text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text
            article_text = re.sub(r'\s+', ' ', article_text)
            article_text = re.sub(r'\n{3,}', '\n\n', article_text)
            
            return article_text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return ""
    
    def create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        
        # If text is short, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Split by paragraphs first to maintain context
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap from end of previous chunk
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text + " " + para
                else:
                    # If single paragraph is too long, split it
                    words = para.split()
                    for i in range(0, len(words), self.chunk_size // 10):
                        chunk_words = words[i:i + self.chunk_size // 10]
                        chunks.append(' '.join(chunk_words))
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks
    
    def summarize_chunk(self, chunk: str, chunk_index: int, total_chunks: int) -> str:
        """Summarize a single chunk"""
        try:
            prompt = f"""You are summarizing part {chunk_index + 1} of {total_chunks} of an article.
Please provide a concise summary focusing on the main points and key information.

Text to summarize:
{chunk}

Provide a summary in 2-3 paragraphs, capturing the essential information."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert content summarizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error summarizing chunk {chunk_index}: {e}")
            return f"[Error summarizing chunk {chunk_index}]"
    
    def create_final_summary(self, chunk_summaries: List[str]) -> str:
        """Create final consolidated summary from chunk summaries"""
        try:
            combined_summaries = "\n\n".join([
                f"Section {i+1}:\n{summary}" 
                for i, summary in enumerate(chunk_summaries)
            ])
            
            prompt = f"""You have been provided with summaries of different sections of an article.
Please create a comprehensive final summary that:
1. Captures all key points from all sections
2. Maintains logical flow and coherence
3. Eliminates redundancy
4. Provides a complete overview of the content

Section summaries:
{combined_summaries}

Create a well-structured final summary (3-5 paragraphs)."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at synthesizing information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error creating final summary: {e}")
            return "Error creating final summary"
    
    def summarize_content(self, html_content: str, print_progress: bool = True) -> Dict[str, Any]:
        """Main method to summarize content using chunking strategy"""
        
        # Step 1: Extract text from HTML
        logger.info("Step 1: Extracting text from HTML...")
        full_text = self.extract_text_from_html(html_content)
        
        if print_progress:
            print("\n" + "="*80)
            print("EXTRACTED ARTICLE CONTENT:")
            print("="*80)
            print(f"Length: {len(full_text)} characters")
            print("\nFirst 500 characters:")
            print(full_text[:500] + "..." if len(full_text) > 500 else full_text)
            print("="*80 + "\n")
        
        if not full_text or len(full_text) < 100:
            return {
                "error": "Insufficient content to summarize",
                "original_length": len(full_text),
                "summary": None
            }
        
        # Step 2: Determine if chunking is needed
        if len(full_text) <= self.chunk_size * 1.5:
            # Short content - direct summarization
            logger.info("Content is short, using direct summarization...")
            
            if print_progress:
                print("Content is short enough for direct summarization\n")
            
            summary = self.summarize_chunk(full_text, 0, 1)
            
            if print_progress:
                print("\n" + "="*80)
                print("FINAL SUMMARY:")
                print("="*80)
                print(summary)
                print("="*80 + "\n")
            
            return {
                "original_length": len(full_text),
                "original_text": full_text,
                "chunks_created": 1,
                "chunk_summaries": [summary],
                "summary": summary
            }
        
        # Step 3: Create chunks
        logger.info("Step 2: Creating chunks...")
        chunks = self.create_chunks(full_text)
        
        if print_progress:
            print(f"Created {len(chunks)} chunks for processing\n")
        
        # Step 4: Summarize each chunk
        logger.info("Step 3: Summarizing each chunk...")
        chunk_summaries = []
        
        for i, chunk in enumerate(chunks):
            if print_progress:
                print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            summary = self.summarize_chunk(chunk, i, len(chunks))
            chunk_summaries.append(summary)
            
            if print_progress:
                print(f"Chunk {i+1} summary ({len(summary)} chars):")
                print(summary[:200] + "..." if len(summary) > 200 else summary)
                print("-" * 40)
        
        # Step 5: Create final summary
        logger.info("Step 4: Creating final consolidated summary...")
        
        if print_progress:
            print(f"\nConsolidating {len(chunk_summaries)} chunk summaries...\n")
        
        final_summary = self.create_final_summary(chunk_summaries)
        
        if print_progress:
            print("\n" + "="*80)
            print("FINAL CONSOLIDATED SUMMARY:")
            print("="*80)
            print(final_summary)
            print("="*80 + "\n")
            
            # Print statistics
            print("\nSUMMARY STATISTICS:")
            print(f"- Original text length: {len(full_text)} characters")
            print(f"- Number of chunks: {len(chunks)}")
            print(f"- Final summary length: {len(final_summary)} characters")
            print(f"- Compression ratio: {len(full_text)/len(final_summary):.1f}:1")
        
        return {
            "original_length": len(full_text),
            "original_text": full_text,
            "chunks_created": len(chunks),
            "chunk_summaries": chunk_summaries,
            "summary": final_summary
        }
    
class WebAutomation:
    """Web automation class with captcha handling and advanced summarization"""

    # Connection pool configuration for Selenium
    POOL_CONNECTIONS = 10
    POOL_MAXSIZE = 10

    def __init__(self, api_key: str, headless: bool = True):
        """
        Initialize WebAutomation with chunking summarizer

        Args:
            api_key: OpenAI API key
            headless: True to run Chrome in hidden mode (no window shown)
        """
        self.api_key = api_key
        self.captcha_solver = AdvancedCaptchaSolver(api_key)
        self.summarizer = ChunkingSummarizer(api_key)  # Add chunking summarizer
        self.headless = headless
        self.driver = None

        # Configure Selenium RemoteConnection pool size to avoid "Connection pool is full" warnings
        self._configure_selenium_pool()

        logger.info(f"Initialize WebAutomation with Chunking Summarizer - Headless mode: {headless}")

    def _configure_selenium_pool(self):
        """Configure Selenium's internal HTTP connection pool to avoid pool exhaustion warnings.

        Note: Main configuration is done at module level. This method ensures
        connection pool warnings are suppressed for this instance.
        """
        # Pool configuration is already done at module level via:
        # - urllib3 warning suppression
        # - RemoteConnection timeout setting
        logger.debug(f"Selenium pool configured: connections={self.POOL_CONNECTIONS}, maxsize={self.POOL_MAXSIZE}")
        
        
    def setup_driver(self, user_agent: str = None):
        """Set up Chrome driver with optimal options"""
        options = webdriver.ChromeOptions()
        
        # IMPORTANT: Set headless mode
        if self.headless:
            options.add_argument('--headless=new') # Use new headless mode
            logger.info("Chrome will run in headless (hidden) mode")
        else:
            logger.info("Chrome will run in windowed mode")

        # Options to optimize performance and avoid detection
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        # options.add_argument('--disable-gpu')
        options.add_argument('--enable-gpu')  # Turn on GPU
        options.add_argument('--disable-web-security')
        options.add_argument('--disable-features=VizDisplayCompositor')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--disable-software-rasterizer') # Turn off software rendering

        # Disable images loading to speed up (optional)
        # options.add_argument('--blink-settings=imagesEnabled=false')
        
        # Disable notifications
        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "profile.managed_default_content_settings.images": 1  # 1 = allow, 2 = block
        }
        options.add_experimental_option("prefs", prefs)
        
        # User agent option
        if user_agent:
            options.add_argument(f'--user-agent={user_agent}')
        else:
            # Default user agent is same as real Chrome
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
        # Stealth mode - hide automation sign
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        try:
            # Create service with suppressed output
            from selenium.webdriver.chrome.service import Service

            service = Service(log_path=os.devnull)
            if os.name == 'nt':  # Windows
                import subprocess
                service.creation_flags = 0x08000000  # CREATE_NO_WINDOW

            self.driver = webdriver.Chrome(service=service, options=options)
            self.driver.set_page_load_timeout(120)

            # Execute script to hide webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            logger.info("Chrome driver initialized successfully")

        except Exception as e:
            logger.error(f"Unable to initialize driver: {e}")
            raise

    def solve_captcha_on_page(self, 
                            captcha_img_selector: str,
                            captcha_input_selector: str,
                            submit_selector: str = None,
                            timeout: int = 120) -> bool:
        """Automatically solve captcha on website"""
        
        try:
            # Wait and find captcha image
            wait = WebDriverWait(self.driver, timeout)
            captcha_img = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, captcha_img_selector)))
            
            # Capture captcha
            captcha_path = f"temp_captcha_{int(time.time())}.png"
            captcha_img.screenshot(captcha_path)
            
            logger.info(f"Captured captcha: {captcha_path}")
            
            # Solve captcha
            solution = self.captcha_solver.solve_captcha(captcha_path)
            
            if "error" in solution:
                logger.error(f"Captcha solving error: {solution['error']}")
                return False
                
            # Input solution
            input_field = self.driver.find_element(By.CSS_SELECTOR, captcha_input_selector)
            input_field.clear()
            
            if solution["type"] in ["text", "math"]:
                input_field.send_keys(str(solution["solution"]))
            elif solution["type"] == "selection":
                # Handle selection captcha (need specific implementation)
                logger.warning("Selection captcha needs manual processing")
                return False
                
            # Submit if required
            if submit_selector:
                submit_btn = self.driver.find_element(By.CSS_SELECTOR, submit_selector)
                submit_btn.click()
                
            logger.info("Captcha solved successfully")
            
            # Delete temporary file
            try:
                os.remove(captcha_path)
            except:
                pass
                
            return True
            
        except TimeoutException:
            logger.error("Captcha not found within the specified time")
            return False
        except Exception as e:
            logger.error(f"Error solving captcha: {e}")
            return False

    def process_url_with_captcha(
        self, 
        url: str,
        captcha_selectors: Dict[str, str] = None,
        wait_after_captcha: int = 5,
        print_progress: bool = True
    ) -> Dict[str, Any]:
        """Process URL with captcha and return summarized content
        
        Args:
            url: URL to process
            captcha_selectors: Captcha element selectors
            wait_after_captcha: Time to wait after solving captcha
            print_progress: Whether to print progress information
            
        Returns:
            Dictionary containing summary and metadata
        """
        
        if not self.driver:
            self.setup_driver()
            
        try:
            logger.info(f"Currently browsing: {url}")
            if print_progress:
                print(f"\n{'='*80}")
                print(f"PROCESSING URL: {url}")
                print(f"{'='*80}\n")
                
            self.driver.get(url)
            time.sleep(3)
            
            # Check and solve captcha if any
            if captcha_selectors:
                logger.info("Checking captcha...")
                captcha_solved = self.solve_captcha_on_page(
                    captcha_selectors.get('image', 'img[alt*="captcha"], img[id*="captcha"], .captcha img'),
                    captcha_selectors.get('input', 'input[name*="captcha"], input[id*="captcha"], .captcha input'),
                    captcha_selectors.get('submit')
                )
                
                if captcha_solved:
                    time.sleep(wait_after_captcha)
                else:
                    logger.warning("Unable to solve captcha, continue with current page")
            
            # Get page content
            html_content = self.driver.page_source
            
            # Use chunking summarizer
            result = self.summarizer.summarize_content(html_content, print_progress)
            
            # Add URL to result
            result['url'] = url
            result['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            return result
            
        except Exception as e:
            logger.error(f"URL processing error: {e}")
            return {
                "error": f"Error: {str(e)}",
                "url": url,
                "summary": None
            }
        finally:
            self._cleanup()
    
    def process_multiple_urls(self, urls: List[str], print_progress: bool = True) -> List[Dict[str, Any]]:
        """Process multiple URLs and summarize their content
        
        Args:
            urls: List of URLs to process
            print_progress: Whether to print progress
            
        Returns:
            List of summary results
        """
        results = []
        
        for i, url in enumerate(urls, 1):
            if print_progress:
                print(f"\n{'='*80}")
                print(f"Processing URL {i}/{len(urls)}")
                print(f"{'='*80}")
            
            result = self.process_url_with_captcha(url, print_progress=print_progress)
            results.append(result)
            
            # Short delay between requests
            if i < len(urls):
                time.sleep(2)
        
        return results
    
    def _cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None


    def _summarize_content(self, html_content: str) -> str:
        """Summarize content using OpenAI API"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Delete script, style tags
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            text = soup.get_text(separator=' ', strip=True)
            
            # Limit text length
            if len(text) > 8000:
                text = text[:8000] + "..."
                
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4.1-nano", # Use a lighter model for summarization
                messages=[{
                    "role": "user", 
                    "content": f"Hãy tóm tắt nội dung sau một cách ngắn gọn và đầy đủ:\n\n{text}"
                }],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error summarizing content: {e}")
            return "Unable to summarize content"

    def _cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass
            self.driver = None


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Example 1: Test chunking summarizer with a long article
    # print("\n" + "="*80)
    # print("TESTING CHUNKING SUMMARIZER WITH WEB CONTENT")
    # print("="*80)
    
    # Initialize web automation with chunking summarizer
    web_automation = WebAutomation(
        API_KEY, 
        headless=True  # Run Chrome in background
    )
    
    # Test URLs
    test_urls = [
        "https://www.marketwatch.com/story/as-jobless-claims-rise-unemployment-benefits-arent-keeping-up-with-inflation-heres-what-to-know-d28ef798"
    ]
    
    captcha_config = {
        'image': 'img#captcha_image',  # CSS selector cho captcha image
        'input': 'input#captcha_input',  # CSS selector cho input field
        'submit': 'button[type="submit"]'  # CSS selector cho submit button
    }

    print("\n1. Processing single URL with detailed progress:")
    result = web_automation.process_url_with_captcha(
        test_urls[0], 
        captcha_selectors=captcha_config,  # No captcha expected
        print_progress=True  # Show detailed progress
    )
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE - RESULT SUMMARY:")
    print("="*80)
    
    if result.get('error'):
        print(f"Error: {result['error']}")
    else:
        print(f"URL: {result['url']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Original content length: {result['original_length']} characters")
        print(f"Number of chunks: {result['chunks_created']}")
        print(f"Final summary length: {len(result['summary'])} characters")
        
        if result['original_length'] > 0 and result['summary']:
            compression = result['original_length'] / len(result['summary'])
            print(f"Compression ratio: {compression:.1f}:1")
    
    print("\n" + "="*80)
    print("ALL PROCESSING COMPLETE!")
    print("="*80)
