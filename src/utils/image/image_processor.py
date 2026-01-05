"""
Image Processing Utilities for Multimodal Support.

Production-ready image processing like ChatGPT/Claude:
- Supports URL, base64, file upload
- Uses requests with retry mechanism
- Proper error handling and logging
"""

import base64
import asyncio
import ssl
from pathlib import Path
from enum import Enum
from typing import Optional, Union, Tuple, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logger.custom_logging import LoggerMixin


# ============================================================================
# Constants
# ============================================================================

# Supported image formats
SUPPORTED_FORMATS = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/webp": "webp",
    "image/gif": "gif",
}

# Extension to MIME type mapping
EXTENSION_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

# Maximum image size (20MB)
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024

# HTTP client settings
HTTP_TIMEOUT = 30
HTTP_MAX_RETRIES = 3
HTTP_BACKOFF_FACTOR = 1.0

# Thread pool for blocking requests
_request_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="image_fetch")


# ============================================================================
# Enums and Data Classes
# ============================================================================

class ImageSource(str, Enum):
    """Source type of the image."""
    FILE = "file"           # Local file path
    URL = "url"             # Remote URL
    BASE64 = "base64"       # Already base64 encoded
    BYTES = "bytes"         # Raw bytes


@dataclass
class ImageContent:
    """
    Raw image content before processing.

    This is the input format from API requests.
    """
    source: ImageSource
    data: str  # Path, URL, or base64 string
    media_type: Optional[str] = None  # MIME type if known

    @classmethod
    def from_file(cls, path: str) -> "ImageContent":
        """Create from file path."""
        return cls(source=ImageSource.FILE, data=path)

    @classmethod
    def from_url(cls, url: str) -> "ImageContent":
        """Create from URL."""
        return cls(source=ImageSource.URL, data=url)

    @classmethod
    def from_base64(cls, data: str, media_type: str = "image/png") -> "ImageContent":
        """Create from base64 string."""
        return cls(source=ImageSource.BASE64, data=data, media_type=media_type)


@dataclass
class ProcessedImage:
    """
    Processed image ready for LLM consumption.

    Contains base64 data and metadata needed for provider formatting.
    """
    base64_data: str
    media_type: str  # MIME type (image/png, image/jpeg, etc.)
    original_source: ImageSource
    size_bytes: int

    # Optional metadata
    width: Optional[int] = None
    height: Optional[int] = None
    filename: Optional[str] = None

    def to_data_url(self) -> str:
        """Convert to data URL format (data:image/png;base64,...)."""
        return f"data:{self.media_type};base64,{self.base64_data}"


# ============================================================================
# Validation Functions
# ============================================================================

def validate_image_format(
    media_type: Optional[str] = None,
    file_path: Optional[str] = None,
    data: Optional[bytes] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate image format.

    Args:
        media_type: MIME type if known
        file_path: File path to check extension
        data: Raw bytes to detect format from magic bytes

    Returns:
        Tuple of (is_valid, message, detected_media_type)
    """
    # Check from media_type
    if media_type:
        if media_type in SUPPORTED_FORMATS:
            return True, "Valid format", media_type
        return False, f"Unsupported format: {media_type}", None

    # Check from file extension
    if file_path:
        ext = Path(file_path).suffix.lower()
        if ext in EXTENSION_TO_MIME:
            detected_type = EXTENSION_TO_MIME[ext]
            return True, "Valid format", detected_type
        return False, f"Unsupported extension: {ext}", None

    # Check from magic bytes
    if data and len(data) >= 8:
        # PNG: 89 50 4E 47 0D 0A 1A 0A
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return True, "Valid format (PNG)", "image/png"

        # JPEG: FF D8 FF
        if data[:3] == b'\xff\xd8\xff':
            return True, "Valid format (JPEG)", "image/jpeg"

        # GIF: GIF87a or GIF89a
        if data[:6] in (b'GIF87a', b'GIF89a'):
            return True, "Valid format (GIF)", "image/gif"

        # WebP: RIFF....WEBP
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return True, "Valid format (WebP)", "image/webp"

    return False, "Could not determine image format", None


def validate_image_size(
    size_bytes: int,
    max_size: int = MAX_IMAGE_SIZE_BYTES,
) -> Tuple[bool, str]:
    """
    Validate image size.

    Args:
        size_bytes: Size in bytes
        max_size: Maximum allowed size (default 20MB)

    Returns:
        Tuple of (is_valid, message)
    """
    if size_bytes <= 0:
        return False, "Invalid image size (0 bytes)"

    if size_bytes > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        return False, f"Image too large: {actual_mb:.1f}MB (max {max_mb:.0f}MB)"

    return True, "Valid size"


# ============================================================================
# HTTP Session with Retry
# ============================================================================

def _create_retry_session() -> requests.Session:
    """Create a requests session with retry mechanism."""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=HTTP_MAX_RETRIES,
        backoff_factor=HTTP_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def _sync_fetch_url(url: str, timeout: int = HTTP_TIMEOUT) -> Tuple[bytes, str]:
    """
    Synchronous URL fetch with retry.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Tuple of (data, content_type)

    Raises:
        ValueError: If fetch fails
    """
    session = _create_retry_session()

    try:
        response = session.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "HealerAgent/1.0 (Image Processor)",
                "Accept": "image/*,*/*",
            },
            verify=True,  # Verify SSL by default
            allow_redirects=True,
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").split(";")[0].strip()
        return response.content, content_type

    except requests.exceptions.Timeout:
        raise ValueError(f"Timeout fetching image from URL (>{timeout}s)")
    except requests.exceptions.ConnectionError as e:
        raise ValueError(f"Connection error fetching image: {e}")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error fetching image: {e.response.status_code}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to fetch image: {e}")
    finally:
        session.close()


async def fetch_url_async(
    url: str,
    timeout: int = HTTP_TIMEOUT,
) -> Tuple[bytes, str]:
    """
    Async wrapper for URL fetch.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Tuple of (data, content_type)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _request_executor,
        _sync_fetch_url,
        url,
        timeout,
    )


# ============================================================================
# Conversion Functions
# ============================================================================

def image_to_base64(file_path: str) -> Tuple[str, str, int]:
    """
    Convert local file to base64.

    Args:
        file_path: Path to image file

    Returns:
        Tuple of (base64_data, media_type, size_bytes)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format is invalid or file too large
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    # Read file
    data = path.read_bytes()
    size_bytes = len(data)

    # Validate size
    valid, msg = validate_image_size(size_bytes)
    if not valid:
        raise ValueError(msg)

    # Validate and detect format
    valid, msg, media_type = validate_image_format(file_path=file_path, data=data)
    if not valid:
        raise ValueError(msg)

    # Encode to base64
    base64_data = base64.b64encode(data).decode("utf-8")

    return base64_data, media_type, size_bytes


async def url_to_base64(
    url: str,
    timeout: int = HTTP_TIMEOUT,
) -> Tuple[str, str, int]:
    """
    Fetch image from URL and convert to base64.

    Args:
        url: Image URL
        timeout: Request timeout in seconds

    Returns:
        Tuple of (base64_data, media_type, size_bytes)

    Raises:
        ValueError: If request fails, format is invalid, or image too large
    """
    # Fetch with retry
    data, content_type = await fetch_url_async(url, timeout)

    size_bytes = len(data)

    # Validate size
    valid, msg = validate_image_size(size_bytes)
    if not valid:
        raise ValueError(msg)

    # Validate and detect format
    valid, msg, media_type = validate_image_format(
        media_type=content_type if content_type in SUPPORTED_FORMATS else None,
        data=data,
    )
    if not valid:
        raise ValueError(msg)

    # Encode to base64
    base64_data = base64.b64encode(data).decode("utf-8")

    return base64_data, media_type, size_bytes


def bytes_to_base64(
    data: bytes,
    media_type: Optional[str] = None,
) -> Tuple[str, str, int]:
    """
    Convert raw bytes to base64.

    Args:
        data: Raw image bytes
        media_type: MIME type if known

    Returns:
        Tuple of (base64_data, media_type, size_bytes)

    Raises:
        ValueError: If format is invalid or image too large
    """
    size_bytes = len(data)

    # Validate size
    valid, msg = validate_image_size(size_bytes)
    if not valid:
        raise ValueError(msg)

    # Validate and detect format
    valid, msg, detected_type = validate_image_format(
        media_type=media_type,
        data=data,
    )
    if not valid:
        raise ValueError(msg)

    # Encode to base64
    base64_data = base64.b64encode(data).decode("utf-8")

    return base64_data, detected_type, size_bytes


# ============================================================================
# Main Processing Function
# ============================================================================

async def process_image_input(
    image: Union[ImageContent, str, dict],
) -> ProcessedImage:
    """
    Process any image input into a standardized ProcessedImage.

    Accepts:
    - ImageContent object
    - String (file path or URL, auto-detected)
    - Dict with 'source' and 'data' keys

    Args:
        image: Image input in various formats

    Returns:
        ProcessedImage ready for LLM consumption

    Raises:
        ValueError: If image is invalid or cannot be processed
    """
    # Convert to ImageContent if needed
    if isinstance(image, str):
        # Auto-detect: URL or file path
        if image.startswith(("http://", "https://")):
            image = ImageContent.from_url(image)
        elif image.startswith("data:"):
            # Data URL format: data:image/png;base64,iVBORw0...
            parts = image.split(",", 1)
            if len(parts) == 2:
                media_info = parts[0].replace("data:", "").replace(";base64", "")
                image = ImageContent.from_base64(parts[1], media_type=media_info)
            else:
                raise ValueError("Invalid data URL format")
        else:
            image = ImageContent.from_file(image)

    elif isinstance(image, dict):
        source = ImageSource(image.get("source", "file"))
        data = image.get("data", "")
        media_type = image.get("media_type")
        image = ImageContent(source=source, data=data, media_type=media_type)

    # Process based on source type
    if image.source == ImageSource.FILE:
        base64_data, media_type, size_bytes = image_to_base64(image.data)
        return ProcessedImage(
            base64_data=base64_data,
            media_type=media_type,
            original_source=ImageSource.FILE,
            size_bytes=size_bytes,
            filename=Path(image.data).name,
        )

    elif image.source == ImageSource.URL:
        base64_data, media_type, size_bytes = await url_to_base64(image.data)
        return ProcessedImage(
            base64_data=base64_data,
            media_type=media_type,
            original_source=ImageSource.URL,
            size_bytes=size_bytes,
        )

    elif image.source == ImageSource.BASE64:
        # Already base64, just validate
        try:
            data = base64.b64decode(image.data)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")

        size_bytes = len(data)

        # Validate size
        valid, msg = validate_image_size(size_bytes)
        if not valid:
            raise ValueError(msg)

        # Validate format
        valid, msg, media_type = validate_image_format(
            media_type=image.media_type,
            data=data,
        )
        if not valid:
            raise ValueError(msg)

        return ProcessedImage(
            base64_data=image.data,
            media_type=media_type or image.media_type or "image/png",
            original_source=ImageSource.BASE64,
            size_bytes=size_bytes,
        )

    elif image.source == ImageSource.BYTES:
        data = image.data if isinstance(image.data, bytes) else image.data.encode()
        base64_data, media_type, size_bytes = bytes_to_base64(data, image.media_type)
        return ProcessedImage(
            base64_data=base64_data,
            media_type=media_type,
            original_source=ImageSource.BYTES,
            size_bytes=size_bytes,
        )

    raise ValueError(f"Unknown image source type: {image.source}")


# ============================================================================
# Image Processor Class
# ============================================================================

class ImageProcessor(LoggerMixin):
    """
    High-level image processor with logging and batch processing.

    Production-ready features:
    - Retry mechanism for URL fetching
    - Proper error handling with detailed logs
    - Support for URL, base64, and file inputs
    """

    def __init__(self, max_size_bytes: int = MAX_IMAGE_SIZE_BYTES):
        super().__init__()
        self.max_size_bytes = max_size_bytes

    async def process(
        self,
        image: Union[ImageContent, str, dict],
    ) -> ProcessedImage:
        """
        Process a single image.

        Args:
            image: Image input (URL, base64, file path, or dict)

        Returns:
            ProcessedImage ready for LLM

        Raises:
            ValueError: If image cannot be processed
        """
        # Extract URL/source for logging
        source_info = self._get_source_info(image)

        try:
            result = await process_image_input(image)
            self.logger.debug(
                f"[IMAGE] Processed | source={result.original_source.value} | "
                f"type={result.media_type} | size={result.size_bytes / 1024:.1f}KB"
            )
            return result
        except ValueError as e:
            self.logger.error(f"[IMAGE] Processing failed for {source_info}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"[IMAGE] Unexpected error for {source_info}: {e}")
            raise ValueError(f"Failed to process image: {e}")

    async def process_batch(
        self,
        images: List[Union[ImageContent, str, dict]],
    ) -> List[ProcessedImage]:
        """
        Process multiple images.

        Args:
            images: List of image inputs

        Returns:
            List of ProcessedImage (failed images are skipped)
        """
        results = []
        for i, img in enumerate(images):
            try:
                result = await self.process(img)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"[IMAGE] Skipping image {i}: {e}")
                # Continue processing other images

        self.logger.info(f"[IMAGE] Batch processed: {len(results)}/{len(images)} images")
        return results

    def _get_source_info(self, image: Union[ImageContent, str, dict]) -> str:
        """Extract source info for logging."""
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                # Truncate URL for logging
                return f"URL={image[:100]}..." if len(image) > 100 else f"URL={image}"
            elif image.startswith("data:"):
                return "data_url"
            return f"file={image}"
        elif isinstance(image, dict):
            source = image.get("source", "unknown")
            data = image.get("data", "")
            if source == "url" and data:
                return f"URL={data[:100]}..." if len(data) > 100 else f"URL={data}"
            return f"source={source}"
        elif isinstance(image, ImageContent):
            if image.source == ImageSource.URL:
                return f"URL={image.data[:100]}..." if len(image.data) > 100 else f"URL={image.data}"
            return f"source={image.source.value}"
        return "unknown"

    def validate(
        self,
        image: Union[ImageContent, str, dict],
    ) -> Tuple[bool, str]:
        """
        Validate image without processing.

        Args:
            image: Image input

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            if isinstance(image, str):
                if image.startswith(("http://", "https://")):
                    # Can't fully validate URL without fetching
                    return True, "URL format valid (content not checked)"
                elif image.startswith("data:"):
                    # Validate data URL format
                    if "," not in image or "base64" not in image:
                        return False, "Invalid data URL format"
                    return True, "Data URL format valid"
                else:
                    # File path
                    return validate_image_format(file_path=image)[:2]

            elif isinstance(image, dict):
                source = image.get("source")
                if not source:
                    return False, "Missing 'source' field"
                if not image.get("data"):
                    return False, "Missing 'data' field"
                return True, "Dict format valid"

            elif isinstance(image, ImageContent):
                return True, "ImageContent valid"

            return False, f"Unknown image type: {type(image)}"

        except Exception as e:
            return False, str(e)
