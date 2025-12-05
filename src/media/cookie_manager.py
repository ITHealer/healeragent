"""
Cookie Manager for yt-dlp
Handles loading, validating and managing browser cookies for video downloads
Based on Varia's cookie implementation
"""

import os
import http.cookiejar
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class CookieManager:
    """Manages cookies for yt-dlp video downloads"""
    
    def __init__(self, cookies_dir: Optional[Path] = None):
        """
        Initialize Cookie Manager
        
        Args:
            cookies_dir: Directory to store cookies.txt file
                        Default: ./cookies (will be created if not exists)
        """
        if cookies_dir is None:
            cookies_dir = Path.cwd() / "cookies"
        
        self.cookies_dir = Path(cookies_dir)
        self.cookies_dir.mkdir(exist_ok=True, parents=True)
        
        self.cookies_file = self.cookies_dir / "cookies.txt"
        self.enabled = False
        
        logger.info(f"Cookie Manager initialized. Cookies directory: {self.cookies_dir}")
        
        # Check if cookies file exists
        if self.cookies_file.exists():
            self.enabled = True
            logger.info(f"Cookies file found: {self.cookies_file}")
        else:
            logger.info("No cookies file found. Cookie support disabled.")
    
    def is_enabled(self) -> bool:
        """Check if cookies are enabled"""
        return self.enabled and self.cookies_file.exists()
    
    def enable_cookies(self) -> bool:
        """
        Enable cookie support
        
        Returns:
            True if cookies file exists and can be enabled
        """
        if self.cookies_file.exists():
            self.enabled = True
            logger.info("Cookies enabled")
            return True
        else:
            logger.warning("Cannot enable cookies: cookies.txt file not found")
            return False
    
    def disable_cookies(self):
        """Disable cookie support"""
        self.enabled = False
        logger.info("Cookies disabled")
    
    def get_cookies_path(self) -> Optional[str]:
        """
        Get path to cookies file if enabled
        
        Returns:
            Path to cookies.txt if enabled, None otherwise
        """
        if self.is_enabled():
            return str(self.cookies_file)
        return None
    
    def load_cookies(self) -> Optional[http.cookiejar.MozillaCookieJar]:
        """
        Load cookies from cookies.txt file
        
        Returns:
            CookieJar object if successful, None otherwise
        """
        if not self.is_enabled():
            logger.warning("Cookies not enabled")
            return None
        
        try:
            cookie_jar = http.cookiejar.MozillaCookieJar(str(self.cookies_file))
            cookie_jar.load(ignore_discard=True, ignore_expires=True)
            logger.info(f"Loaded {len(cookie_jar)} cookies from {self.cookies_file}")
            return cookie_jar
        except Exception as e:
            logger.error(f"Failed to load cookies: {e}")
            return None
    
    def get_cookie_header_string(self) -> str:
        """
        Get cookies as header string format
        Used for aria2c and other download managers
        
        Returns:
            Cookie header string like "Cookie: name1=value1; name2=value2"
        """
        if not self.is_enabled():
            return ""
        
        try:
            cookie_jar = self.load_cookies()
            if cookie_jar:
                all_cookies = "; ".join([f"{item.name}={item.value}" for item in cookie_jar])
                return f"Cookie: {all_cookies}"
        except Exception as e:
            logger.error(f"Failed to generate cookie header: {e}")
        
        return ""
    
    def save_cookies_from_text(self, cookies_text: str) -> bool:
        """
        Save cookies from text content
        
        Args:
            cookies_text: Content of cookies.txt file (Netscape format)
        
        Returns:
            True if saved successfully
        """
        try:
            with open(self.cookies_file, 'w', encoding='utf-8') as f:
                f.write(cookies_text)
            
            # Validate the saved file
            if self.validate_cookies_file():
                self.enabled = True
                logger.info(f"Cookies saved successfully to {self.cookies_file}")
                return True
            else:
                logger.error("Saved cookies file is invalid")
                return False
        
        except Exception as e:
            logger.error(f"Failed to save cookies: {e}")
            return False
    
    def validate_cookies_file(self) -> bool:
        """
        Validate cookies.txt file format
        
        Returns:
            True if file is valid Netscape cookies format
        """
        if not self.cookies_file.exists():
            return False
        
        try:
            # Try to load the file
            cookie_jar = http.cookiejar.MozillaCookieJar(str(self.cookies_file))
            cookie_jar.load(ignore_discard=True, ignore_expires=True)
            
            # Check if we got at least one cookie
            if len(cookie_jar) > 0:
                logger.info(f"Cookies file validated: {len(cookie_jar)} cookies found")
                return True
            else:
                logger.warning("Cookies file is empty")
                return False
        
        except Exception as e:
            logger.error(f"Cookies validation failed: {e}")
            return False
    
    def delete_cookies(self) -> bool:
        """
        Delete cookies file
        
        Returns:
            True if deleted successfully
        """
        try:
            if self.cookies_file.exists():
                os.remove(self.cookies_file)
                self.enabled = False
                logger.info("Cookies file deleted")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete cookies: {e}")
            return False
    
    def get_cookies_info(self) -> Dict:
        """
        Get information about current cookies
        
        Returns:
            Dictionary with cookies info
        """
        info = {
            "enabled": self.enabled,
            "file_exists": self.cookies_file.exists(),
            "file_path": str(self.cookies_file),
            "cookies_dir": str(self.cookies_dir),
            "cookie_count": 0,
            "last_modified": None,
            "file_size": 0,
            "domains": []
        }
        
        if self.cookies_file.exists():
            try:
                # Get file stats
                stat = os.stat(self.cookies_file)
                info["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                info["file_size"] = stat.st_size
                
                # Load and count cookies
                cookie_jar = self.load_cookies()
                if cookie_jar:
                    info["cookie_count"] = len(cookie_jar)
                    
                    # Get unique domains
                    domains = set()
                    for cookie in cookie_jar:
                        domains.add(cookie.domain)
                    info["domains"] = sorted(list(domains))
            
            except Exception as e:
                logger.error(f"Failed to get cookies info: {e}")
        
        return info
    
    def get_cookies_for_domain(self, domain: str) -> List[Dict]:
        """
        Get cookies for a specific domain
        
        Args:
            domain: Domain name (e.g., 'youtube.com')
        
        Returns:
            List of cookie dictionaries
        """
        if not self.is_enabled():
            return []
        
        try:
            cookie_jar = self.load_cookies()
            if not cookie_jar:
                return []
            
            cookies = []
            for cookie in cookie_jar:
                if domain.lower() in cookie.domain.lower():
                    cookies.append({
                        "name": cookie.name,
                        "value": cookie.value,
                        "domain": cookie.domain,
                        "path": cookie.path,
                        "expires": cookie.expires,
                        "secure": cookie.secure
                    })
            
            return cookies
        
        except Exception as e:
            logger.error(f"Failed to get cookies for domain {domain}: {e}")
            return []


# Singleton instance
_cookie_manager_instance = None


def get_cookie_manager(cookies_dir: Optional[Path] = None) -> CookieManager:
    """
    Get singleton CookieManager instance
    
    Args:
        cookies_dir: Directory for cookies (only used on first call)
    
    Returns:
        CookieManager instance
    """
    global _cookie_manager_instance
    
    if _cookie_manager_instance is None:
        _cookie_manager_instance = CookieManager(cookies_dir)
    
    return _cookie_manager_instance