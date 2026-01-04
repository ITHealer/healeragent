"""
Image processing utilities for multimodal support.

This module provides:
- Image validation (format, size)
- Image conversion (file/URL to base64)
- Provider-specific image formatting
"""

from src.utils.image.image_processor import (
    ImageProcessor,
    ImageContent,
    ImageSource,
    ProcessedImage,
    validate_image_format,
    validate_image_size,
    image_to_base64,
    url_to_base64,
    process_image_input,
)

from src.utils.image.multimodal_message import (
    MultimodalContent,
    TextContent,
    ImageContentBlock,
    build_multimodal_message,
    format_messages_for_provider,
)

__all__ = [
    # Image processor
    "ImageProcessor",
    "ImageContent",
    "ImageSource",
    "ProcessedImage",
    "validate_image_format",
    "validate_image_size",
    "image_to_base64",
    "url_to_base64",
    "process_image_input",
    # Multimodal message
    "MultimodalContent",
    "TextContent",
    "ImageContentBlock",
    "build_multimodal_message",
    "format_messages_for_provider",
]
