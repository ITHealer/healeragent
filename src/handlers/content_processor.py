import os
import re
import logging
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Any
from urllib.parse import urlparse
from pathlib import Path
import time

# Video processing imports from AI-Video-Transcriber
from src.media.video_processor import VideoProcessor
from src.media.transcriber import Transcriber
from src.media.summarizer import Summarizer
from src.media.translator import Translator

# Article processing
from src.media.article_content_processor import WebAutomation
from src.media.crawler4ai_article_processor import LangChainSummarizer, WebCrawler
from src.helpers.language_detector import language_detector, DetectionMethod
from src.providers.provider_factory import ProviderType
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ContentTypeDetector:
    """Detects whether a URL contains video or article content"""
    
    # Video platform patterns
    VIDEO_PLATFORMS = {
        'youtube': [r'youtube\.com/watch', r'youtu\.be/', r'youtube\.com/embed/'],
        'tiktok': [r'tiktok\.com/', r'vm\.tiktok\.com/'],
        'bilibili': [r'bilibili\.com/video/', r'b23\.tv/'],
        'vimeo': [r'vimeo\.com/\d+'],
        'dailymotion': [r'dailymotion\.com/video/'],
        'twitter': [r'twitter\.com/.*/status/', r'x\.com/.*/status/'],
        'facebook': [r'facebook\.com/.*/videos/', r'fb\.watch/'],
        'instagram': [r'instagram\.com/p/', r'instagram\.com/reel/'],
    }
    
    # Video file extensions
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
    
    @classmethod
    def is_video_url(cls, url: str) -> bool:
        """Check if URL is a video"""
        url_lower = url.lower()
        
        # Check video platforms
        for platform, patterns in cls.VIDEO_PLATFORMS.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    logger.info(f"Detected video platform: {platform}")
                    return True
        
        # Check video file extensions
        parsed = urlparse(url)
        path = parsed.path.lower()
        for ext in cls.VIDEO_EXTENSIONS:
            if path.endswith(ext):
                logger.info(f"Detected video file extension: {ext}")
                return True
        
        return False


class ContentProcessor:
    """Main processor that handles both video and article content"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4.1-nano", provider_type: str = "openai"):
        """
        Initialize ContentProcessor with correct API key
        
        Args:
            api_key: The actual OpenAI API key (sk-...)
            model_name: Name of the model to use
            provider_type: Provider type (openai, anthropic, etc.)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.provider_type = provider_type
        
        # Initialize translator
        self.translator = Translator()

        # Initialize video processors
        try:
            self.video_processor = VideoProcessor()
            self.transcriber = Transcriber()
            self.video_summarizer = Summarizer()
            self.video_support = True
            logger.info("Video processing support enabled")
        except Exception as e:
            logger.warning(f"AI-Video-Transcriber not available: {e}")
            self.video_support = False
        
        try:
            self.summarizer = LangChainSummarizer(api_key=api_key, model=model_name, provider_type=provider_type)
            logger.info("LangChainSummarizer initialized with chunking support")
        except Exception as e:
            logger.warning(f"Summarizer initialization failed: {e}")
            self.summarizer = None

        # Initialize article processor
        # WebAutomation already has all the logic for article processing
        self.web_automation = WebAutomation(api_key, headless=True)

        self.language_map = {
            "en": "English",
            "vi": "Vietnamese",
            "zh": "中文（简体）",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ja": "日本語",
            "ko": "한국어",
            "ar": "العربية"
        }

        logger.info("Universal Content Processor initialized successfully")
 
    def _extract_detected_language_from_transcript(self, transcript: str) -> Optional[str]:
        """
        Extract detected language from Whisper transcript format
        AI-Video-Transcriber adds language info in transcript like:
        **Detected Language:** en
        """
        if "**Detected Language:**" in transcript:
            lines = transcript.split('\n')
            for line in lines:
                if "**Detected Language:**" in line:
                    # Extract language code, e.g.: "**Detected Language:** en"
                    lang_code = line.split(":")[-1].strip()
                    lang_code = re.sub(r'[^\w-]', '', lang_code)
                    return lang_code
        
        # Alternative format check
        if "**检测语言:**" in transcript:  # Chinese format
            lines = transcript.split('\n')
            for line in lines:
                if "**检测语言:**" in line:
                    lang_code = line.split(":")[-1].strip()
                    lang_code = re.sub(r'[^\w-]', '', lang_code)
                    return lang_code
        
        return None
    
    async def process_url(self, url: str, target_language: Optional[str] = None, print_progress: bool = True) -> Dict[str, Any]:
        """
        Process any URL - automatically detect and handle video or article
        
        Args:
            url: URL to process
            target_language: Target language for translation (e.g., 'en', 'vi', 'zh', 'ja')
                             If None, summary will be in the source language
            print_progress: Whether to print processing progress
            
        Returns:
            Dictionary containing processed results
        """
        try:
            logger.info(f"Starting to process URL: {url}")
            logger.info(f"Target language: {target_language if target_language else 'Source language (no translation)'}")
            
            # Detect content type
            is_video = ContentTypeDetector.is_video_url(url)
            
            if is_video:
                logger.info("URL identified as VIDEO content")
                if not self.video_support:
                    return {
                        "url": url,
                        "type": "video",
                        "error": "Video processing not available. Please ensure AI-Video-Transcriber is properly installed.",
                        "status": "error"
                    }
                return await self.process_video(url, target_language, print_progress)
            else:
                logger.info("URL identified as ARTICLE/BLOG content")
                return await self.process_article(url, target_language, print_progress)
                
        except Exception as e:
            logger.error(f"Error processing URL: {e}", exc_info=True)
            return {
                "url": url,
                "error": str(e),
                "status": "error"
            }
    
    async def process_video(self, url: str, target_language: Optional[str] = None, print_progress: bool = True) -> Dict[str, Any]:
        """
        Enhanced process video URL with better markdown summaries
        """
        try:
            start_time = time.time()
            logger.info(f"Starting enhanced video processing for: {url}")
            
            # Step 1: Download and extract audio
            if print_progress:
                self._print_video_header(url, target_language)
                print("\nStep 1/4: Downloading video and extracting audio...")
            
            audio_path, video_title = await self.video_processor.download_and_convert(url, Path.cwd() / "temp")
            logger.info(f"Audio extracted: {audio_path}")
            
            if print_progress:
                print(f"✓ Video Title: {video_title}")
                print(f"✓ Audio file: {os.path.basename(audio_path)}")
            
            # Step 2: Transcribe with language detection
            if print_progress:
                print("\nStep 2/4: Transcribing audio with AI...")
            
            transcript_markdown = await self.transcriber.transcribe(audio_path)
            detected_language = self._extract_detected_language_from_transcript(transcript_markdown)
            
            if not detected_language:
                detected_language = 'en'
                logger.warning("Language detection failed, defaulting to English")
            
            logger.info(f"Transcription complete. Language: {detected_language}")
            
            # Extract clean transcript text
            transcript = self._extract_clean_transcript(transcript_markdown)
            
            if print_progress:
                lang_name = self.language_map.get(detected_language, detected_language)
                print(f"✓ Detected Language: {lang_name} ({detected_language})")
                print(f"✓ Transcript length: {len(transcript)} characters")
                self._print_preview("Transcript", transcript)
            
            # Step 3: Detect video content type for optimized summary
            content_type = self._detect_video_content_type(transcript, video_title)
            logger.info(f"Detected video type: {content_type}")
            
            # Step 4: Generate enhanced summary
            summary_language = target_language if target_language else detected_language
            
            if print_progress:
                print(f"\nStep 3/4: Generating enhanced AI summary in {summary_language}...")
                print(f"✓ Video type detected: {content_type}")
            
            # Use enhanced summarizer with content-type awareness
            summary = await self._generate_enhanced_video_summary(
                transcript=transcript,
                video_title=video_title,
                target_language=summary_language,
                content_type=content_type
            )
            
            # Step 5: Handle translation if needed
            translation = None
            # if target_language and detected_language != target_language:
            #     if print_progress:
            #         print(f"\nStep 4/4: Translating key points to {target_language}...")
                
            #     # Translate key sections only for efficiency
            #     key_content = self._extract_key_content_for_translation(transcript)
            #     translation = await self.translator.translate_text(
            #         key_content,
            #         target_language,
            #         detected_language
            #     )
                
            #     if print_progress:
            #         print(f"✓ Translation completed")
            #         self._print_preview("Translation", translation)
            # else:
            #     if print_progress:
            #         print(f"\nStep 4/4: No translation needed")
            
            # Clean up temp files
            self._cleanup_temp_files(audio_path)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            
            # Print results
            if print_progress:
                self._print_video_complete(processing_time, video_title, summary_language, summary)
            
            # Return enhanced result
            return self._create_video_result(
                url=url,
                video_title=video_title,
                detected_language=detected_language,
                target_language=summary_language,
                transcript=transcript,
                transcript_markdown=transcript_markdown,
                summary=summary,
                translation=translation,
                content_type=content_type,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Video processing error: {e}", exc_info=True)
            return {
                "type": "video",
                "url": url,
                "error": str(e),
                "status": "error"
            }
    
    async def _generate_paywall_message(self, url: str, target_language: Optional[str] = None) -> str:
        """
        Generate appropriate paywall message using LLM in target language
        """
        try:
            # Default language mapping
            lang_names = {
                "vi": "Vietnamese (Tiếng Việt)",
                "en": "English",
                "zh": "Chinese (中文)",
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
            
            language = target_language if target_language else "en"
            language_name = lang_names.get(language, "English")
            
            prompt = f"""Generate a professional and helpful message in {language_name} explaining that this article requires a subscription or login to access the full content.

    The message should:
    1. Politely explain that the article is behind a paywall/subscription wall
    2. Suggest that the user may need to subscribe or log in to read the full content
    3. Be respectful and understanding
    4. Keep it concise but informative
    5. Write ONLY in {language_name}

    URL: {url}

    Message in {language_name}:"""
            
            api_key = os.getenv("OPENAI_API_KEY")
            llm = ChatOpenAI(
                openai_api_key=api_key,
                model_name="gpt-4.1-nano",
            )
            
            response = await llm.ainvoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate paywall message: {e}")
            # Fallback messages
            fallback_messages = {
                "vi": "Bài viết này yêu cầu đăng ký hoặc đăng nhập để xem nội dung đầy đủ. Vui lòng truy cập trang web và đăng ký/đăng nhập để đọc bài viết hoàn chỉnh.",
                "en": "This article requires a subscription or login to access the full content. Please visit the website and subscribe or log in to read the complete article.",
                "zh": "此文章需要订阅或登录才能查看完整内容。请访问网站并订阅或登录以阅读完整文章。",
                "ja": "この記事を全て読むには、サブスクリプションまたはログインが必要です。ウェブサイトにアクセスして、購読またはログインしてください。",
                "ko": "이 기사를 전체적으로 보려면 구독 또는 로그인이 필요합니다. 웹사이트를 방문하여 구독하거나 로그인해주세요.",
                "fr": "Cet article nécessite un abonnement ou une connexion pour accéder au contenu complet. Veuillez visiter le site Web et vous abonner ou vous connecter.",
                "de": "Dieser Artikel erfordert ein Abonnement oder eine Anmeldung, um den vollständigen Inhalt zu sehen. Bitte besuchen Sie die Website und abonnieren Sie oder melden Sie sich an.",
                "es": "Este artículo requiere una suscripción o inicio de sesión para acceder al contenido completo. Por favor visite el sitio web y suscríbase o inicie sesión."
            }
            
            return fallback_messages.get(target_language or "en", fallback_messages["en"])
        


    async def process_article(self, url: str, target_language: Optional[str] = None, print_progress: bool = True) -> Dict[str, Any]:
        """
        Process article/blog URL with optional translation
        Optimized version with cleaner structure and better error handling
        """
        start_time = time.time()
        logger.info(f"Starting article processing for: {url}")
        
        if print_progress:
            self._print_article_header(url, target_language)
        
        try:
            # Step 1: Extract content
            extraction_result = await self._extract_article_content(url, print_progress)
            
            # Handle paywall/subscription cases
            if extraction_result.get("paywall_detected"):
                return await self._handle_paywall_article(url, target_language, start_time)
            
            # Check if extraction failed completely
            if not extraction_result.get("success"):
                return self._create_error_result(url, extraction_result.get("error"), start_time)
            
            # Step 2: Detect language
            detected_language = await self._detect_article_language(
                extraction_result["content"], 
                target_language, 
                print_progress
            )
            print(f"Detected language =====================: {detected_language}")
            # Step 3: Generate summary
            summary = await self._generate_article_summary(
                extraction_result["content"],
                extraction_result["title"],
                detected_language,
                target_language,
                print_progress
            )
            
            # Step 4: Handle translation if needed
            translation = None
            # translation = await self._handle_article_translation(
            #     extraction_result["content"],
            #     detected_language,
            #     target_language,
            #     print_progress
            # )
            
            # Finalize results
            return self._create_success_result(
                url=url,
                extraction_result=extraction_result,
                detected_language=detected_language,
                target_language=target_language or detected_language,
                summary=summary,
                translation=translation,
                start_time=start_time,
                print_progress=print_progress
            )
            
        except Exception as e:
            logger.error(f"Error processing article: {e}", exc_info=True)
            return self._create_error_result(url, str(e), start_time)

    async def _extract_article_content(self, url: str, print_progress: bool) -> Dict[str, Any]:
        """
        Extract article content using crawl4ai first, fallback to web_automation
        Returns dict with: success, content, title, chunks_created, method, paywall_detected
        """
        if print_progress:
            print("\nStep 1/4: Fetching and extracting content...")
        
        # Try crawl4ai first
        crawl_result = await self._try_crawl4ai_extraction(url)
        if crawl_result["success"]:
            return crawl_result
        
        # Fallback to web_automation
        web_result = await self._try_web_automation_extraction(url)
        if web_result["success"]:
            return web_result
        
        # Both failed - likely paywall
        return {
            "success": False,
            "paywall_detected": True,
            "error": "Content requires subscription or login"
        }

    async def _try_crawl4ai_extraction(self, url: str) -> Dict[str, Any]:
        """Try extracting content using crawl4ai"""
        try:
            logger.info("Trying crawl4ai extraction...")
            crawler = WebCrawler(timeout=60)
            crawl_result = await crawler.crawl_url(url)
            
            if not crawl_result.success or not crawl_result.content:
                raise ValueError(f"crawl4ai failed: {crawl_result.error or 'No content'}")
            
            # Check for paywall indicators
            if self._is_paywall_content(crawl_result.content):
                raise ValueError("Paywall detected")
            
            return {
                "success": True,
                "content": crawl_result.content,
                "title": crawl_result.title or "Article",
                "chunks_created": 1,
                "method": "crawl4ai",
                "paywall_detected": False
            }
            
        except Exception as e:
            logger.warning(f"crawl4ai failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def _try_web_automation_extraction(self, url: str) -> Dict[str, Any]:
        """Try extracting content using web_automation"""
        try:
            logger.info("Trying web_automation extraction...")
            captcha_config = {
                'image': 'img#captcha_image',
                'input': 'input#captcha_input',
                'submit': 'button[type="submit"]'
            }
            
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self.web_automation.process_url_with_captcha,
                url,
                captcha_config,
                5,
                False
            )
            
            if result.get("error"):
                raise Exception(result['error'])
            
            content = result.get("original_text", "")
            
            # Check for paywall
            if self._is_paywall_content(content):
                raise Exception("Paywall detected")
            
            return {
                "success": True,
                "content": content,
                "title": result.get("title", "Article"),
                "chunks_created": result.get("chunks_created", 1),
                "method": "web_automation",
                "paywall_detected": False
            }
            
        except Exception as e:
            logger.warning(f"web_automation failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _is_paywall_content(self, content: str) -> bool:
        """Check if content indicates paywall/subscription requirement"""
        if not content or len(content.strip()) < 100:
            return True
        
        paywall_indicators = [
            "subscribe", "subscription", "sign in", "log in", "login",
            "premium", "member", "account", "paywall", "unlock",
            "continue reading", "view full article", "subscriber"
        ]
        
        content_lower = content.lower()[:500]  # Check first 500 chars
        return any(indicator in content_lower for indicator in paywall_indicators)

    async def _detect_article_language(self, content: str, target_language: Optional[str], print_progress: bool) -> str:
        """Detect the language of article content"""
        if print_progress:
            print("\nStep 2/4: Detecting source language...")
        
        detection_method = DetectionMethod.LLM if len(content.split()) < 2 else DetectionMethod.LIBRARY
        language_info = await language_detector.detect(
            text=content[:500],
            method=detection_method,
            system_language=target_language or "en",
            model_name=self.model_name,
            provider_type=self.provider_type,
            api_key=self.api_key
        )
        
        detected = language_info["detected_language"]

        # Normalize Chinese variants
        detected = {"zh-cn": "zh", "zh-tw": "zh"}.get(detected, detected)
        
        if print_progress:
            lang_name = self.language_map.get(detected, detected)
            print(f"✓ Detected Language: {lang_name} ({detected})")
        
        logger.info(f"Detected language: {detected}")
        return detected

    async def _generate_article_summary(
        self,
        content: str,
        title: str,
        detected_language: str,
        target_language: Optional[str],
        print_progress: bool
    ) -> str:
        """Generate summary with enhanced markdown formatting"""
        summary_language = target_language or detected_language
        
        language_map = {
            "en": "English",
            "vi": "Vietnamese",
            "zh": "中文（简体）",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ja": "日本語",
            "ko": "한국어",
            "ar": "العربية"
        }
        lang_name = language_map.get(summary_language, detected_language)
        print(f"🌐 Summary language: {lang_name}")
        if print_progress:
            print(f"\nStep 3/4: Generating AI summary in {summary_language}...")
        
        # Enhanced prompt for better markdown formatting
        enhanced_prompt = f"""
        Create a comprehensive summary of this article in {lang_name}.
        
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
        - Write entirely in {lang_name}
        - Focus on main arguments, key facts, and conclusions
        - Maintain logical flow between sections
        - Be concise but comprehensive
        
        Title: {title}
        """
        
        # Use enhanced summarizer
        summary = await self.summarizer.summarize_enhanced(
            transcript=content,
            target_language=summary_language,
            video_title=title,
            custom_prompt=enhanced_prompt
        )
        
        return summary

    async def _handle_article_translation(
        self,
        content: str,
        detected_language: str,
        target_language: Optional[str],
        print_progress: bool
    ) -> Optional[str]:
        """Handle translation if needed"""
        if not target_language or detected_language == target_language:
            if print_progress:
                print("\nStep 4/4: No translation needed")
            return None
        
        if print_progress:
            print(f"\nStep 4/4: Translating from {detected_language} to {target_language}...")
        
        translation = await self.translator.translate_text(
            content[:5000],  # Limit translation length
            target_language,
            detected_language
        )
        
        if print_progress:
            print(f"✓ Translation completed")
            self._print_preview("Translation", translation)
        
        return translation

    async def _handle_paywall_article(
        self,
        url: str,
        target_language: Optional[str],
        start_time: float
    ) -> Dict[str, Any]:
        """Handle paywall/subscription-required articles"""
        logger.warning(f"Paywall detected for URL: {url}")
        
        paywall_message = await self._generate_paywall_message(url, target_language)
        
        return {
            "type": "article",
            "url": url,
            "source_language": None,
            "target_language": target_language,
            "summary": paywall_message,
            "original_content": None,
            "translation": None,
            "translation_needed": False,
            "metadata": {
                "original_length": 0,
                "chunks_created": 0,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "paywall_detected": True
            },
            "error": "Content requires subscription or login",
            "status": "paywall_detected",
            "processing_time": time.time() - start_time
        }

    def _create_success_result(
        self,
        url: str,
        extraction_result: Dict,
        detected_language: str,
        target_language: str,
        summary: str,
        translation: Optional[str],
        start_time: float,
        print_progress: bool
    ) -> Dict[str, Any]:
        """Create successful processing result"""
        processing_time = time.time() - start_time
        
        if print_progress:
            self._print_article_complete(processing_time, len(extraction_result["content"]), target_language, summary)
        
        return {
            "type": "article",
            "url": url,
            "title": extraction_result["title"],
            "source_language": detected_language,
            "target_language": target_language,
            "original_text": extraction_result["content"],
            "original_length": len(extraction_result["content"]),
            "chunks_created": extraction_result["chunks_created"],
            "summary": summary,
            "translation": translation,
            "translation_needed": translation is not None,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "processing_time": processing_time,
            "status": "success",
            "method_used": extraction_result["method"]
        }

    def _create_error_result(self, url: str, error: str, start_time: float) -> Dict[str, Any]:
        """Create error result"""
        return {
            "type": "article",
            "url": url,
            "error": error,
            "status": "error",
            "processing_time": time.time() - start_time
        }

    # Helper print methods
    def _print_article_header(self, url: str, target_language: Optional[str]):
        """Print article processing header"""
        print("\n" + "="*80)
        print("PROCESSING ARTICLE/BLOG")
        print("="*80)
        print(f"URL: {url}")
        if target_language:
            print(f"Target Language: {target_language}")

    def _print_article_complete(self, processing_time: float, content_length: int, language: str, summary: str):
        """Print article completion info"""
        print("\n" + "="*80)
        print("ARTICLE PROCESSING COMPLETE")
        print("="*80)
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Original length: {content_length} characters")
        print(f"Summary language: {language}")
        print(f"\nFINAL SUMMARY:")
        print("-" * 40)
        print(summary)
        print("="*80 + "\n")

    def _print_preview(self, label: str, text: str, max_chars: int = 500):
        """Print text preview"""
        print(f"{label} preview (first {max_chars} chars):")
        print("-" * 40)
        print(text[:max_chars] + "..." if len(text) > max_chars else text)
        print("-" * 40)


    def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up WebAutomation resources (Chrome driver, etc.)
            if hasattr(self, 'web_automation'):
                self.web_automation._cleanup()
                logger.info("WebAutomation resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")



    async def _generate_enhanced_video_summary(
        self,
        transcript: str,
        video_title: str,
        target_language: str,
        content_type: str
    ) -> str:
        """
        Generate enhanced video summary with content-type optimization
        """
        # Get content-type specific enhancements
        type_prompts = {
            "tutorial": {
                "focus": "step-by-step instructions, prerequisites, tools needed, common mistakes",
                "structure": "overview → requirements → steps → tips → troubleshooting"
            },
            "interview": {
                "focus": "key questions, insights, quotes, personal stories, main themes",
                "structure": "guest intro → main topics → key insights → memorable quotes"
            },
            "presentation": {
                "focus": "main thesis, arguments, evidence, conclusions, action items",
                "structure": "premise → arguments → evidence → implications → conclusions"
            },
            "educational": {
                "focus": "concepts, definitions, examples, applications, key learnings",
                "structure": "objectives → concepts → examples → practice → summary"
            },
            "general": {
                "focus": "main topics, key points, insights, conclusions",
                "structure": "overview → main content → insights → takeaways"
            }
        }
        
        content_config = type_prompts.get(content_type, type_prompts["general"])
        
        # Call enhanced summarizer with content-aware prompting
        summary = await self.video_summarizer.summarize_enhanced(
            transcript=transcript,
            target_language=target_language,
            video_title=video_title,
            content_type=content_type,
            content_focus=content_config["focus"],
            content_structure=content_config["structure"]
        )
        
        return summary

    def _detect_video_content_type(self, transcript: str, title: Optional[str] = None) -> str:
        """
        Intelligently detect video content type
        """
        text_sample = (title or "").lower() + " " + transcript[:1000].lower()
        
        patterns = {
            "tutorial": ["how to", "tutorial", "guide", "step", "learn", "teach", "show you"],
            "interview": ["interview", "conversation", "guest", "talk with", "discuss", "q&a"],
            "presentation": ["presentation", "conference", "keynote", "speech", "talk", "audience"],
            "educational": ["lesson", "course", "lecture", "class", "education", "explain", "understand"],
        }
        
        scores = {}
        for content_type, keywords in patterns.items():
            score = sum(1 for kw in keywords if kw in text_sample)
            scores[content_type] = score
        
        # Return type with highest score, default to general
        best_type = max(scores.items(), key=lambda x: x[1])
        return best_type[0] if best_type[1] > 0 else "general"

    def _extract_clean_transcript(self, transcript_markdown: str) -> str:
        """
        Extract clean text from markdown transcript
        """
        lines = transcript_markdown.split('\n')
        clean_lines = []
        
        for line in lines:
            # Skip markdown headers and metadata
            if line.strip() and not line.startswith('#') and not line.startswith('**'):
                clean_lines.append(line.strip())
        
        return '\n'.join(clean_lines)

    def _extract_key_content_for_translation(self, transcript: str, max_length: int = 3000) -> str:
        """
        Extract key content for translation (optimize for cost/speed)
        """
        if len(transcript) <= max_length:
            return transcript
        
        # Extract beginning, middle, and end sections
        section_size = max_length // 3
        beginning = transcript[:section_size]
        middle_start = (len(transcript) - section_size) // 2
        middle = transcript[middle_start:middle_start + section_size]
        end = transcript[-section_size:]
        
        return f"{beginning}\n\n[...]\n\n{middle}\n\n[...]\n\n{end}"

    def _create_video_result(
        self,
        url: str,
        video_title: str,
        detected_language: str,
        target_language: str,
        transcript: str,
        transcript_markdown: str,
        summary: str,
        translation: Optional[str],
        content_type: str,
        processing_time: float
    ) -> Dict[str, Any]:
        """
        Create comprehensive video processing result
        """
        result = {
            "type": "video",
            "url": url,
            "video_info": {
                "title": video_title,
                "content_type": content_type,
                "duration_estimate": f"~{len(transcript) // 150} minutes"  # Rough estimate
            },
            "source_language": detected_language,
            "target_language": target_language,
            "transcript": transcript,
            "transcript_markdown": transcript_markdown,
            "summary": summary,
            "translation_needed": translation is not None,
            "processing_time": processing_time,
            "status": "success",
            "metadata": {
                "content_type": content_type,
                "transcript_length": len(transcript),
                "summary_length": len(summary),
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        if translation:
            result["translation"] = translation
            result["translation_preview"] = translation[:500] + "..." if len(translation) > 500 else translation
        
        return result

    def _cleanup_temp_files(self, audio_path: str):
        """
        Clean up temporary files
        """
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info(f"Cleaned up: {audio_path}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    # Helper print methods for better UX
    def _print_video_header(self, url: str, target_language: Optional[str]):
        """Print video processing header"""
        print("\n" + "="*80)
        print("🎬 PROCESSING VIDEO CONTENT")
        print("="*80)
        print(f"📍 URL: {url}")
        if target_language:
            lang_name = self.language_map.get(target_language, target_language)
            print(f"🌐 Target Language: {lang_name}")

    def _print_video_complete(self, processing_time: float, title: str, language: str, summary: str):
        """Print video completion with summary"""
        print("\n" + "="*80)
        print("✅ VIDEO PROCESSING COMPLETE")
        print("="*80)
        print(f"⏱️  Processing time: {processing_time:.2f} seconds")
        print(f"🎬 Video: {title}")
        print(f"🌐 Summary language: {self.language_map.get(language, language)}")
        print(f"\n📝 ENHANCED SUMMARY:")
        print("-" * 40)
        print(summary)
        print("="*80 + "\n")

    def _print_preview(self, label: str, text: str, max_chars: int = 500):
        """Print text preview with truncation"""
        preview = text[:max_chars] + "..." if len(text) > max_chars else text
        print(f"\n{label} preview:")
        print("-" * 40)
        print(preview)
        print("-" * 40)

# Example usage for testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not API_KEY:
        print("Error: OPENAI_API_KEY not found in environment variables")
        exit(1)
    
    # Initialize processor
    processor = ContentProcessor(API_KEY)
    
    # Test URLs
    test_urls = [
        # Video examples
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        # Article examples  
        "https://www.marketwatch.com/story/as-jobless-claims-rise-unemployment-benefits-arent-keeping-up-with-inflation-heres-what-to-know-d28ef798",
        "https://www.bbc.com/news/technology-68520346"
    ]
    
    async def test():
        """Test processing different types of URLs"""
        for i, url in enumerate(test_urls, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}/{len(test_urls)}")
            print(f"{'='*80}")
            
            result = await processor.process_url(url, print_progress=True)
            
            # Print result summary
            print(f"\n{'='*40}")
            print("RESULT SUMMARY:")
            print(f"{'='*40}")
            print(f"Status: {result.get('status')}")
            print(f"Content Type: {result.get('type')}")
            
            if result.get('status') == 'success':
                print(f"Summary Length: {len(result.get('summary', ''))} characters")
                print(f"Processing Time: {result.get('processing_time', 0):.2f} seconds")
            else:
                print(f"Error: {result.get('error')}")
            
            # Small delay between tests
            if i < len(test_urls):
                await asyncio.sleep(2)
        
        # Cleanup
        processor.cleanup()
        print(f"\n{'='*80}")
        print("ALL TESTS COMPLETED")
        print(f"{'='*80}")
    
    # Run tests
    asyncio.run(test())