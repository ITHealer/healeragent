import re
from urllib.parse import urlparse, unquote
from typing import Optional, Dict, Any
from enum import Enum
from lingua import Language, LanguageDetectorBuilder
from src.utils.logger.custom_logging import LoggerMixin
from src.helpers.llm_helper import LLMGeneratorProvider
from src.providers.provider_factory import ProviderType


class DetectionMethod(Enum):
    """Language detection methods"""
    LIBRARY = "library"  # Using lingua library (default)
    LLM = "llm"         # Using LLM for detection


# Helper functions
def extract_text_from_url(url: str) -> str:
    """
    Trích xuất văn bản từ URL để phân tích ngôn ngữ
    """
    # Giải mã URL encoding
    decoded_url = unquote(url)
    
    # Lấy các thành phần từ URL
    parsed = urlparse(decoded_url)
    
    # Tách các từ từ path, query, fragment
    text_parts = []
    
    # Từ domain (bỏ www và TLD)
    domain = parsed.netloc.replace('www.', '').split('.')[0]
    text_parts.append(domain)
    
    # Từ path (bỏ slashes, dashes, underscores)
    if parsed.path:
        path_words = re.sub(r'[/_-]', ' ', parsed.path)
        text_parts.append(path_words)
    
    # Từ query parameters
    if parsed.query:
        query_words = re.sub(r'[=&_-]', ' ', parsed.query)
        text_parts.append(query_words)
    
    return ' '.join(text_parts).strip()


def is_url(text: str) -> bool:
    """
    Kiểm tra xem input có phải là URL không
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'|^www\.'     # www.
        r'|\.[a-z]{2,}/'  # .com/ .vn/ etc
    )
    return bool(url_pattern.search(text.lower()))


class LanguageDetector(LoggerMixin):
    """
    Unified language detection with fallback mechanism
    """
    
    def __init__(self):
        super().__init__()
        self._language_detector = None
        self.llm_provider = LLMGeneratorProvider()
        self._init_language_detector()
    
    def _init_language_detector(self):
        """Initialize lingua language detector"""
        try:
            languages = [
                Language.ENGLISH,
                Language.VIETNAMESE,
                Language.CHINESE,
            ]
            self._language_detector = LanguageDetectorBuilder.from_languages(*languages).build()
            self.logger.info("Language detector initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing language detector: {str(e)}")
            self._language_detector = None
    
    def _is_url(self, text: str) -> bool:
        """Check if text is a URL"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(text.strip()))
    
    def _extract_words_from_url(self, url: str) -> str:
        """
        Extract meaningful words from URL for language detection
        
        Examples:
            https://vnexpress.net/tin-tuc-covid-19 -> vnexpress tin tuc covid 19
            https://example.com/news-today -> example news today
            https://baidu.com/搜索/新闻 -> baidu 搜索 新闻
        """
        try:
            parsed = urlparse(url)
            
            # Extract domain name (without TLD)
            domain = parsed.netloc
            domain_parts = domain.split('.')
            domain_name = domain_parts[0] if len(domain_parts) > 1 else domain
            
            # Extract path and split by common separators
            path = parsed.path
            
            # Replace common separators with spaces
            text = path.replace('/', ' ').replace('-', ' ').replace('_', ' ')
            
            # Remove numbers and special characters but keep letters (including Unicode)
            text = re.sub(r'[0-9]+', '', text)
            
            # Combine domain name with path words
            combined = f"{domain_name} {text}"
            
            # Clean up multiple spaces
            combined = re.sub(r'\s+', ' ', combined).strip()
            
            self.logger.debug(f"Extracted from URL: '{combined}'")
            return combined
            
        except Exception as e:
            self.logger.error(f"Error extracting words from URL: {str(e)}")
            return url
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text before language detection
        If URL, extract meaningful words
        """
        text = text.strip()
        
        if self._is_url(text):
            self.logger.info(f"Detected URL input: {text}")
            return self._extract_words_from_url(text)
        
        return text
    
    async def detect_language_with_llm(
        self, 
        text: str,
        model_name: str = "gpt-4.1-nano",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Detect language using LLM with URL support
        Returns ISO 639-1 code (2 letters) or None
        """
        try:
            # Preprocess text (extract words from URL if needed)
            processed_text = self._preprocess_text(text)
            
            lang_detection_prompt = '''
<system>
You are an expert in detecting languages. From the input provided identify the language and share the 2 letter ISO 639-1 code.

Instructions:
1. Identify the language from the input given below
2. The input might be extracted words from a URL - focus on the language of these words
3. Respond ONLY with the ISO 639-1 code (2 letters)
4. Common codes:
   - en: English
   - vi: Vietnamese  
   - zh: Chinese (Simplified)
   - zh-TW: Chinese (Traditional)
   - ja: Japanese
   - ko: Korean

Examples:
Input: The secret of getting ahead is getting started!
Output: en

Input: vnexpress tin tuc covid
Output: vi

Input: baidu 搜索 新闻
Output: zh

Input: yahoo ニュース 速報
Output: ja

Input: naver 뉴스 경제
Output: ko

Input: {input}
Output:
</system>'''
            
            messages = [
                {"role": "system", "content": "Language detection expert. Return ONLY the 2-letter ISO code."},
                {"role": "user", "content": lang_detection_prompt.format(input=processed_text[:500])}
            ]
            
            response = await self.llm_provider.generate_response(
                model_name=model_name,
                messages=messages,
                provider_type=provider_type,
                api_key=api_key
            )
            
            # Extract and validate ISO code
            iso_code = response.get("content", "").strip().lower()
            
            # Handle special cases
            if iso_code == "zh-tw":
                iso_code = "zh"  # Normalize to standard code
            
            # Validate it's a 2-letter code
            if len(iso_code) == 2 and iso_code.isalpha():
                self.logger.info(f"LLM detected language: {iso_code} from input: '{processed_text[:50]}...'")
                return iso_code
            else:
                self.logger.warning(f"Invalid ISO code from LLM: {iso_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in LLM language detection: {str(e)}")
            return None
    
    def detect_language_with_library(self, text: str) -> Optional[str]:
        """
        Detect language using lingua library
        Returns ISO 639-1 code (2 letters) or None
        """
        if not text or text.strip() == "":
            return None
            
        try:
            if not self._language_detector:
                self.logger.warning("Language detector not initialized, returning None")
                return None
                
            # Clean text for better detection
            clean_text = self._clean_text_for_detection(text)
            
            # Detect language
            detected_lang = self._language_detector.detect_language_of(clean_text)
            
            if detected_lang:
                # Get ISO 639-1 code (2 letters)
                iso_code = detected_lang.iso_code_639_1.name.lower()
                
                # Special mappings for common cases
                mapping = {
                    "zh": "zh",      # Chinese (simplified)
                    "cmn": "zh",     # Mandarin Chinese  
                    "yue": "zh",     # Cantonese
                    "vie": "vi",     # Vietnamese alternate code
                }
                
                result = mapping.get(iso_code, iso_code)
                self.logger.info(f"Library detected language: {result}")
                return result
            
            self.logger.warning("Could not detect language with library")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in library language detection: {str(e)}")
            return None
    
    def _clean_text_for_detection(self, text: str) -> str:
        """
        Clean text for better language detection
        Remove code, numbers, special chars but keep meaningful text
        """
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove stock tickers (all caps 1-5 letters)
        text = re.sub(r'\b[A-Z]{1,5}\b', '', text)
        
        # Remove excessive numbers and special characters
        text = re.sub(r'\d{4,}', '', text)  # Long numbers
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\uAC00-\uD7AF]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    async def detect(
        self,
        text: str,
        method: DetectionMethod = DetectionMethod.LIBRARY,
        system_language: str = "en",
        model_name: str = "gpt-5-nano-2025-08-07",
        provider_type: str = ProviderType.OPENAI,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main detection method with fallback mechanism
        
        Args:
            text: Text to detect language from
            method: Detection method to use (library or llm)
            system_language: Fallback language if detection fails
            model_name: LLM model name for LLM detection
            provider_type: Provider type for LLM detection
            api_key: API key for LLM provider
            
        Returns:
            Dict with detected language info:
            {
                "detected_language": "vi",  # ISO 639-1 code
                "language_name": "Vietnamese",
                "confidence": "high|medium|low", 
                "detection_method": "library|llm|fallback",
                "needs_translation": true/false  # If different from system language
            }
        """
        result = {
            "detected_language": system_language,
            "language_name": self._get_language_name(system_language),
            "confidence": "low",
            "detection_method": "fallback",
            "needs_translation": False
        }
        
        if not text or text.strip() == "":
            self.logger.warning("Empty text provided, using system language")
            return result
        
        detected_lang = None
        
        # Try primary detection method
        if method == DetectionMethod.LIBRARY:
            detected_lang = self.detect_language_with_library(text)
            if detected_lang:
                result["detection_method"] = "library"
                result["confidence"] = "high"
            else:
                # Fallback to LLM if library fails
                self.logger.info("Library detection failed, trying LLM")
                detected_lang = await self.detect_language_with_llm(
                    text, model_name, provider_type, api_key
                )
                if detected_lang:
                    result["detection_method"] = "llm"
                    result["confidence"] = "medium"
        else:  # LLM method
            detected_lang = await self.detect_language_with_llm(
                text, model_name, provider_type, api_key
            )
            if detected_lang:
                result["detection_method"] = "llm"
                result["confidence"] = "high"
            else:
                # Fallback to library if LLM fails
                self.logger.info("LLM detection failed, trying library")
                detected_lang = self.detect_language_with_library(text)
                if detected_lang:
                    result["detection_method"] = "library"
                    result["confidence"] = "medium"
        
        # Update result if detection successful
        if detected_lang:
            result["detected_language"] = detected_lang
            result["language_name"] = self._get_language_name(detected_lang)
            result["needs_translation"] = (detected_lang != system_language)
        else:
            self.logger.warning(f"All detection methods failed, using system language: {system_language}")
        
        self.logger.info(f"Language detection result: {result}")
        return result
    
    def _get_language_name(self, iso_code: str) -> str:
        language_names = {
            "en": "English",
            "vi": "Vietnamese",
            "zh": "Chinese",
            "ja": "Japanese", 
            "ko": "Korean",
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "ru": "Russian",
            "th": "Thai",
            "id": "Indonesian",
            "ms": "Malay",
            "hi": "Hindi",
            "ar": "Arabic",
            "pt": "Portuguese",
            "it": "Italian"
        }
        return language_names.get(iso_code, f"Language ({iso_code})")


# Singleton instance
language_detector = LanguageDetector()