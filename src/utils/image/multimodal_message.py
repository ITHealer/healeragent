"""
Multimodal Message Builder for LLM Providers.

Provides:
- Provider-agnostic content structures
- Provider-specific message formatting
- Support for OpenAI, Anthropic, Google Gemini, Ollama
"""

from enum import Enum
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field

from src.utils.image.image_processor import ProcessedImage


# ============================================================================
# Content Types
# ============================================================================

class ContentType(str, Enum):
    """Type of content block."""
    TEXT = "text"
    IMAGE = "image"


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"  # Uses OpenAI format


@dataclass
class TextContent:
    """Text content block."""
    text: str
    type: ContentType = ContentType.TEXT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "text",
            "text": self.text,
        }


@dataclass
class ImageContentBlock:
    """
    Image content block.

    Stores processed image data ready for provider formatting.
    """
    image: ProcessedImage
    type: ContentType = ContentType.IMAGE
    alt_text: Optional[str] = None  # Description for accessibility

    def to_dict(self) -> Dict[str, Any]:
        """Generic dict representation."""
        return {
            "type": "image",
            "media_type": self.image.media_type,
            "data": self.image.base64_data,
            "alt_text": self.alt_text,
        }


@dataclass
class MultimodalContent:
    """
    Container for mixed text and image content.

    Represents the content of a single message that may contain
    multiple text and image blocks.
    """
    blocks: List[Union[TextContent, ImageContentBlock]] = field(default_factory=list)

    def add_text(self, text: str) -> "MultimodalContent":
        """Add text block."""
        self.blocks.append(TextContent(text=text))
        return self

    def add_image(
        self,
        image: ProcessedImage,
        alt_text: Optional[str] = None,
    ) -> "MultimodalContent":
        """Add image block."""
        self.blocks.append(ImageContentBlock(image=image, alt_text=alt_text))
        return self

    def has_images(self) -> bool:
        """Check if content contains any images."""
        return any(b.type == ContentType.IMAGE for b in self.blocks)

    def get_text_only(self) -> str:
        """Get concatenated text content only."""
        texts = [b.text for b in self.blocks if isinstance(b, TextContent)]
        return "\n".join(texts)

    def get_images(self) -> List[ProcessedImage]:
        """Get all images."""
        return [b.image for b in self.blocks if isinstance(b, ImageContentBlock)]


# ============================================================================
# Provider-Specific Formatters
# ============================================================================

def _format_for_openai(content: MultimodalContent) -> List[Dict[str, Any]]:
    """
    Format content for OpenAI Vision API.

    OpenAI format:
    [
        {"type": "text", "text": "..."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
    """
    result = []

    for block in content.blocks:
        if isinstance(block, TextContent):
            result.append({
                "type": "text",
                "text": block.text,
            })
        elif isinstance(block, ImageContentBlock):
            result.append({
                "type": "image_url",
                "image_url": {
                    "url": block.image.to_data_url(),
                    "detail": "auto",  # or "low", "high"
                },
            })

    return result


def _format_for_anthropic(content: MultimodalContent) -> List[Dict[str, Any]]:
    """
    Format content for Anthropic Claude Vision API.

    Anthropic format:
    [
        {"type": "text", "text": "..."},
        {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
    ]
    """
    result = []

    for block in content.blocks:
        if isinstance(block, TextContent):
            result.append({
                "type": "text",
                "text": block.text,
            })
        elif isinstance(block, ImageContentBlock):
            result.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": block.image.media_type,
                    "data": block.image.base64_data,
                },
            })

    return result


def _format_for_google(content: MultimodalContent) -> List[Dict[str, Any]]:
    """
    Format content for Google Gemini Vision API.

    Google format:
    [
        {"text": "..."},
        {"inline_data": {"mime_type": "...", "data": "..."}}
    ]
    """
    result = []

    for block in content.blocks:
        if isinstance(block, TextContent):
            result.append({
                "text": block.text,
            })
        elif isinstance(block, ImageContentBlock):
            result.append({
                "inline_data": {
                    "mime_type": block.image.media_type,
                    "data": block.image.base64_data,
                },
            })

    return result


def _format_for_ollama(content: MultimodalContent) -> Dict[str, Any]:
    """
    Format content for Ollama Vision models (llava, bakllava, etc.).

    Ollama format:
    {
        "content": "text content",
        "images": ["base64_data1", "base64_data2"]
    }

    Note: Ollama puts images in a separate 'images' array, not inline.
    """
    texts = []
    images = []

    for block in content.blocks:
        if isinstance(block, TextContent):
            texts.append(block.text)
        elif isinstance(block, ImageContentBlock):
            images.append(block.image.base64_data)

    result = {
        "content": "\n".join(texts),
    }

    if images:
        result["images"] = images

    return result


# ============================================================================
# Main Formatting Functions
# ============================================================================

def format_content_for_provider(
    content: MultimodalContent,
    provider: Union[Provider, str],
) -> Union[List[Dict[str, Any]], Dict[str, Any], str]:
    """
    Format multimodal content for a specific provider.

    Args:
        content: MultimodalContent object
        provider: Target provider

    Returns:
        Provider-specific content format
    """
    if isinstance(provider, str):
        provider = Provider(provider.lower())

    if provider == Provider.OPENAI or provider == Provider.OPENROUTER:
        return _format_for_openai(content)
    elif provider == Provider.ANTHROPIC:
        return _format_for_anthropic(content)
    elif provider == Provider.GOOGLE:
        return _format_for_google(content)
    elif provider == Provider.OLLAMA:
        return _format_for_ollama(content)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def build_multimodal_message(
    role: str,
    text: Optional[str] = None,
    images: Optional[List[ProcessedImage]] = None,
    provider: Union[Provider, str] = Provider.OPENAI,
) -> Dict[str, Any]:
    """
    Build a complete message with text and optional images.

    Args:
        role: Message role (user, assistant, system)
        text: Text content
        images: List of processed images
        provider: Target provider

    Returns:
        Complete message dict formatted for provider
    """
    # Build content
    content = MultimodalContent()

    if text:
        content.add_text(text)

    if images:
        for img in images:
            content.add_image(img)

    # Format for provider
    if isinstance(provider, str):
        provider = Provider(provider.lower())

    # Handle different provider message structures
    if provider == Provider.OLLAMA:
        # Ollama has a different structure
        formatted = _format_for_ollama(content)
        return {
            "role": role,
            **formatted,
        }
    else:
        # OpenAI, Anthropic, Google use content array
        formatted_content = format_content_for_provider(content, provider)

        # If no images, use simple string content
        if not content.has_images():
            return {
                "role": role,
                "content": text or "",
            }

        return {
            "role": role,
            "content": formatted_content,
        }


def format_messages_for_provider(
    messages: List[Dict[str, Any]],
    provider: Union[Provider, str],
) -> List[Dict[str, Any]]:
    """
    Format a list of messages for a specific provider.

    Handles messages that may contain:
    - Simple string content
    - Multimodal content with images
    - Already formatted content

    Args:
        messages: List of message dicts
        provider: Target provider

    Returns:
        List of provider-formatted messages
    """
    if isinstance(provider, str):
        provider = Provider(provider.lower())

    result = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        images = msg.get("images", [])

        # If content is already a list (multimodal format), convert it
        if isinstance(content, list):
            # Already in array format, might need conversion
            multimodal = MultimodalContent()

            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "text")

                    if block_type == "text":
                        multimodal.add_text(block.get("text", ""))

                    elif block_type == "image_url":
                        # OpenAI format - extract base64 from data URL
                        url = block.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            parts = url.split(",", 1)
                            if len(parts) == 2:
                                media_info = parts[0].replace("data:", "").replace(";base64", "")
                                base64_data = parts[1]
                                img = ProcessedImage(
                                    base64_data=base64_data,
                                    media_type=media_info,
                                    original_source="base64",
                                    size_bytes=len(base64_data),
                                )
                                multimodal.add_image(img)

                    elif block_type == "image":
                        # Anthropic format
                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            img = ProcessedImage(
                                base64_data=source.get("data", ""),
                                media_type=source.get("media_type", "image/png"),
                                original_source="base64",
                                size_bytes=len(source.get("data", "")),
                            )
                            multimodal.add_image(img)

            # Re-format for target provider
            formatted_content = format_content_for_provider(multimodal, provider)

            if provider == Provider.OLLAMA:
                result.append({
                    "role": role,
                    **formatted_content,
                })
            else:
                result.append({
                    "role": role,
                    "content": formatted_content,
                })

        elif isinstance(content, str) and images:
            # String content with separate images array
            formatted_msg = build_multimodal_message(
                role=role,
                text=content,
                images=images,
                provider=provider,
            )
            result.append(formatted_msg)

        else:
            # Simple string content, no conversion needed
            if provider == Provider.OLLAMA:
                result.append({
                    "role": role,
                    "content": content or "",
                })
            else:
                result.append({
                    "role": role,
                    "content": content or "",
                })

    return result


# ============================================================================
# Helper Functions
# ============================================================================

def create_user_message_with_image(
    text: str,
    image: ProcessedImage,
    provider: Union[Provider, str] = Provider.OPENAI,
) -> Dict[str, Any]:
    """
    Convenience function to create a user message with one image.

    Args:
        text: User's text prompt
        image: Single processed image
        provider: Target provider

    Returns:
        Formatted message dict
    """
    return build_multimodal_message(
        role="user",
        text=text,
        images=[image],
        provider=provider,
    )


def create_user_message_with_images(
    text: str,
    images: List[ProcessedImage],
    provider: Union[Provider, str] = Provider.OPENAI,
) -> Dict[str, Any]:
    """
    Convenience function to create a user message with multiple images.

    Args:
        text: User's text prompt
        images: List of processed images
        provider: Target provider

    Returns:
        Formatted message dict
    """
    return build_multimodal_message(
        role="user",
        text=text,
        images=images,
        provider=provider,
    )
