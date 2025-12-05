"""
Cookie Management API Router
Provides endpoints for uploading, managing and checking cookies
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging

from src.media.cookie_manager import get_cookie_manager
from src.schemas.response import BasicResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cookies")


class CookieUploadRequest(BaseModel):
    cookies_text: str = Field(
        ..., 
        description="Content of cookies.txt file in Netscape format"
    )


class CookieStatusResponse(BaseModel):
    """Response model for cookie status"""
    enabled: bool
    file_exists: bool
    cookie_count: int
    domains: list
    last_modified: Optional[str]
    file_size: int
    file_path: str


@router.get("/status", response_model=BasicResponse)
async def get_cookie_status():
    """
    Get current cookie status and information
    
    Returns information about:
    - Whether cookies are enabled
    - Cookie file location
    - Number of cookies
    - Domains covered
    - Last modification time
    """
    try:
        cookie_manager = get_cookie_manager()
        info = cookie_manager.get_cookies_info()
        
        return BasicResponse(
            status="success",
            message="Cookie status retrieved",
            data=info
        )
    
    except Exception as e:
        logger.error(f"Failed to get cookie status: {e}")
        return BasicResponse(
            status="error",
            message=f"Failed to get cookie status: {str(e)}",
            data={}
        )


# @router.post("/upload", response_model=BasicResponse)
# async def upload_cookies_text(request: CookieUploadRequest):
#     """
#     Upload cookies from text content
    
#     Accepts cookies in Netscape format (cookies.txt)
    
#     ## How to get cookies.txt:
    
#     1. Install browser extension:
#        - Chrome/Edge: "Get cookies.txt LOCALLY"
#        - Firefox: "cookies.txt"
    
#     2. Go to YouTube (or target site)
#     3. Export cookies using the extension
#     4. Copy the content and paste here
    
#     ## Format:
#     ```
#     # Netscape HTTP Cookie File
#     .youtube.com    TRUE    /    FALSE    0    CONSENT    YES+cb
#     .youtube.com    TRUE    /    TRUE    1234567890    SID    abc123...
#     ```
#     """
#     try:
#         cookie_manager = get_cookie_manager()
        
#         # Validate and save cookies
#         success = cookie_manager.save_cookies_from_text(request.cookies_text)
        
#         if success:
#             info = cookie_manager.get_cookies_info()
#             return BasicResponse(
#                 status="success",
#                 message=f"Cookies uploaded successfully. {info['cookie_count']} cookies loaded.",
#                 data=info
#             )
#         else:
#             return BasicResponse(
#                 status="error",
#                 message="Failed to save cookies. Please check the format.",
#                 data={}
#             )
    
#     except Exception as e:
#         logger.error(f"Failed to upload cookies: {e}")
#         return BasicResponse(
#             status="error",
#             message=f"Failed to upload cookies: {str(e)}",
#             data={}
#         )


@router.post("/upload-file", response_model=BasicResponse)
async def upload_cookies_file(file: UploadFile = File(...)):
    """
    Upload cookies.txt file directly
    
    Accepts a cookies.txt file in Netscape format
    
    ## Steps:
    1. Export cookies.txt from browser (using extension)
    2. Upload the file here
    3. Cookies will be validated and saved
    """
    try:
        # Read file content
        content = await file.read()
        cookies_text = content.decode('utf-8')
        
        # Save cookies
        cookie_manager = get_cookie_manager()
        success = cookie_manager.save_cookies_from_text(cookies_text)
        
        if success:
            info = cookie_manager.get_cookies_info()
            return BasicResponse(
                status="success",
                message=f"Cookies file uploaded successfully. {info['cookie_count']} cookies loaded.",
                data=info
            )
        else:
            return BasicResponse(
                status="error",
                message="Failed to save cookies file. Please check the format.",
                data={}
            )
    
    except Exception as e:
        logger.error(f"Failed to upload cookies file: {e}")
        return BasicResponse(
            status="error",
            message=f"Failed to upload cookies file: {str(e)}",
            data={}
        )


@router.post("/enable", response_model=BasicResponse)
async def enable_cookies():
    """
    Enable cookie support
    
    Cookies file must exist before enabling.
    Upload cookies first if not already uploaded.
    """
    try:
        cookie_manager = get_cookie_manager()
        success = cookie_manager.enable_cookies()
        
        if success:
            info = cookie_manager.get_cookies_info()
            return BasicResponse(
                status="success",
                message="Cookies enabled successfully",
                data=info
            )
        else:
            return BasicResponse(
                status="error",
                message="Cannot enable cookies. Upload cookies.txt file first.",
                data={"enabled": False}
            )
    
    except Exception as e:
        logger.error(f"Failed to enable cookies: {e}")
        return BasicResponse(
            status="error",
            message=f"Failed to enable cookies: {str(e)}",
            data={}
        )


@router.post("/disable", response_model=BasicResponse)
async def disable_cookies():
    """
    Disable cookie support
    
    Cookies file will remain but won't be used for downloads.
    """
    try:
        cookie_manager = get_cookie_manager()
        cookie_manager.disable_cookies()
        
        return BasicResponse(
            status="success",
            message="Cookies disabled successfully",
            data={"enabled": False}
        )
    
    except Exception as e:
        logger.error(f"Failed to disable cookies: {e}")
        return BasicResponse(
            status="error",
            message=f"Failed to disable cookies: {str(e)}",
            data={}
        )


@router.delete("/delete", response_model=BasicResponse)
async def delete_cookies():
    """
    Delete cookies file
    
    Warning: This will permanently delete the cookies.txt file.
    You will need to upload cookies again to use cookie support.
    """
    try:
        cookie_manager = get_cookie_manager()
        success = cookie_manager.delete_cookies()
        
        if success:
            return BasicResponse(
                status="success",
                message="Cookies deleted successfully",
                data={"enabled": False, "file_exists": False}
            )
        else:
            return BasicResponse(
                status="warning",
                message="No cookies file to delete",
                data={"enabled": False, "file_exists": False}
            )
    
    except Exception as e:
        logger.error(f"Failed to delete cookies: {e}")
        return BasicResponse(
            status="error",
            message=f"Failed to delete cookies: {str(e)}",
            data={}
        )


@router.get("/domains/{domain}", response_model=BasicResponse)
async def get_cookies_for_domain(domain: str):
    """
    Get cookies for a specific domain
    
    Args:
        domain: Domain name (e.g., youtube.com, twitter.com)
    
    Returns:
        List of cookies for the specified domain
    """
    try:
        cookie_manager = get_cookie_manager()
        
        if not cookie_manager.is_enabled():
            return BasicResponse(
                status="error",
                message="Cookies not enabled. Upload and enable cookies first.",
                data={"cookies": []}
            )
        
        cookies = cookie_manager.get_cookies_for_domain(domain)
        
        return BasicResponse(
            status="success",
            message=f"Found {len(cookies)} cookies for domain: {domain}",
            data={
                "domain": domain,
                "cookie_count": len(cookies),
                "cookies": cookies
            }
        )
    
    except Exception as e:
        logger.error(f"Failed to get cookies for domain {domain}: {e}")
        return BasicResponse(
            status="error",
            message=f"Failed to get cookies for domain: {str(e)}",
            data={"cookies": []}
        )


# @router.get("/help", response_model=BasicResponse)
# async def get_cookie_help():
#     """
#     Get help on how to use cookies
    
#     Returns comprehensive guide on:
#     - Why cookies are needed
#     - How to export cookies from browser
#     - How to upload cookies
#     - Troubleshooting
#     """
#     help_text = """
# # Cookie Support Guide

# ## Why do I need cookies?

# YouTube and other platforms are increasingly blocking automated downloads by requiring sign-in.
# By providing cookies from your browser, yt-dlp can authenticate as if you're signed in.

# ## Benefits of using cookies:

# ✅ Bypass "Sign in to confirm you're not a bot" errors
# ✅ Download age-restricted videos
# ✅ Access private/unlisted videos (if you have access)
# ✅ Download member-only content (with appropriate membership)
# ✅ Avoid rate limiting and IP blocks

# ## How to export cookies:

# ### Method 1: Browser Extension (Recommended)

# **For Chrome/Edge:**
# 1. Install "Get cookies.txt LOCALLY" extension
# 2. Go to YouTube.com (or target site)
# 3. Click the extension icon
# 4. Click "Export" → Copy to clipboard
# 5. Use the `/cookies/upload` endpoint to upload

# **For Firefox:**
# 1. Install "cookies.txt" extension  
# 2. Go to YouTube.com
# 3. Click extension → "Current Site"
# 4. Copy the content
# 5. Upload via API

# ### Method 2: Manual Cookie File

# 1. Export cookies.txt from browser
# 2. Upload file via `/cookies/upload-file` endpoint

# ## Cookie Format (Netscape):

# ```
# # Netscape HTTP Cookie File
# .youtube.com    TRUE    /    FALSE    0    CONSENT    YES+cb
# .youtube.com    TRUE    /    TRUE    1234567890    SID    abc123...
# ```

# ## API Workflow:

# 1. **Upload cookies**: POST `/cookies/upload` or `/cookies/upload-file`
# 2. **Check status**: GET `/cookies/status`
# 3. **Enable cookies**: POST `/cookies/enable` (auto-enabled after upload)
# 4. **Download videos**: Use video processor as normal
# 5. **Disable when not needed**: POST `/cookies/disable`

# ## Important Notes:

# ⚠️ **Security**: Cookies contain authentication tokens. Only upload to trusted services.
# ⚠️ **Privacy**: Don't share your cookies.txt file publicly.
# ⚠️ **Expiration**: Cookies expire. If downloads fail, refresh your cookies.
# ⚠️ **Updates**: Re-export cookies if you log out/in on YouTube.

# ## Troubleshooting:

# **Still getting "Sign in" error after uploading cookies?**
# - Cookies may have expired → Re-export from browser
# - Wrong format → Check cookie file format
# - Platform changed auth → Wait for yt-dlp update

# **Cookies not loading?**
# - Check file format (must be Netscape format)
# - Ensure cookies are for the correct domain
# - Check cookie expiration dates

# ## For Privacy-Conscious Users:

# You can also use the PO Token fix without cookies for many public videos.
# The system automatically uses tv_embedded client which doesn't require cookies for most content.

# ## Support:

# For more help, refer to:
# - yt-dlp documentation: https://github.com/yt-dlp/yt-dlp
# - Cookie export guides: https://github.com/yt-dlp/yt-dlp/wiki/Extractors
# """
    
#     return BasicResponse(
#         status="success",
#         message="Cookie help guide",
#         data={
#             "guide": help_text,
#             "supported_browsers": ["Chrome", "Edge", "Firefox", "Safari", "Brave"],
#             "recommended_extensions": {
#                 "chrome": "Get cookies.txt LOCALLY",
#                 "firefox": "cookies.txt"
#             }
#         }
#     )


# Add router to main app
def get_router():
    """Return router for inclusion in main app"""
    return router