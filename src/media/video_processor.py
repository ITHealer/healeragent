import os
import yt_dlp
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict
import re
import shlex

from src.media.cookie_manager import get_cookie_manager

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processor that uses yt-dlp to download and convert videos"""
    
    def __init__(self, cookies_dir: Optional[Path] = None):

        # Initialize cookie manager
        self.cookie_manager = get_cookie_manager(cookies_dir)

        self.ydl_opts = {
            # More flexible format selection with multiple fallbacks
            # Priority: audio-only formats first, then combined formats, then any best
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio[ext=mp4]/bestaudio/best[acodec!=none]/best',
            'outtmpl': '%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                # Convert to mono 16k directly during extraction (smaller and more stable)
                'preferredcodec': 'm4a',
                'preferredquality': '192'
            }],
            # Global FFmpeg parameters: mono + 16k sample rate + faststart
            'postprocessor_args': ['-ac', '1', '-ar', '16000', '-movflags', '+faststart'],
            'prefer_ffmpeg': True,
            'quiet': True,
            'outtmpl': '%(title).100s.%(ext)s',  # Limit title length
            'no_warnings': True,
            'noplaylist': True,  # Force downloading single video only, not playlists
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-us,en;q=0.5',
                'Sec-Fetch-Mode': 'navigate',
            },
            # Use multiple YouTube player clients for better format availability
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios', 'web'],  # iOS client often has more formats available
                }
            },
        }

        logger.info("VideoProcessor initialized with cookie support")
        if self.cookie_manager.is_enabled():
            logger.info("✅ Cookies ENABLED - Using cookies for authentication")
        else:
            logger.info("⚠️  Cookies DISABLED - Operating without cookies (may fail for some videos)")
    
    # async def download_and_convert(self, url: str, output_dir: Path) -> tuple[str, str]:
    #     """
    #     Download video and convert to m4a format
        
    #     Args:
    #         url: Video URL
    #         output_dir: Output directory
            
    #     Returns:
    #         Path to converted audio file
    #     """
    #     try:
    #         # Create output directory
    #         output_dir.mkdir(exist_ok=True)
            
    #         # Generate unique filename
    #         import uuid
    #         unique_id = str(uuid.uuid4())[:8]
    #         output_template = str(output_dir / f"audio_{unique_id}.%(ext)s")
            
    #         # Update yt-dlp options
    #         ydl_opts = self.ydl_opts.copy()
    #         ydl_opts['outtmpl'] = output_template
            
    #         logger.info(f"Starting video download: {url}")
            
    #         # Execute synchronously without thread pool
    #         # In FastAPI, IO-intensive operations can be awaited directly
    #         import asyncio
    #         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #             # Get video info (put in thread pool to avoid blocking event loop)
    #             info = await asyncio.to_thread(ydl.extract_info, url, False)
    #             video_title = info.get('title', 'unknown')
    #             expected_duration = info.get('duration') or 0
    #             logger.info(f"Video title: {video_title}")
                
    #             # Download video (put in thread pool to avoid blocking event loop)
    #             await asyncio.to_thread(ydl.download, [url])
            
    #         # Find generated m4a file
    #         audio_file = str(output_dir / f"audio_{unique_id}.m4a")
            
    #         if not os.path.exists(audio_file):
    #             # If m4a file doesn't exist, look for other audio formats
    #             for ext in ['webm', 'mp4', 'mp3', 'wav']:
    #                 potential_file = str(output_dir / f"audio_{unique_id}.{ext}")
    #                 if os.path.exists(potential_file):
    #                     audio_file = potential_file
    #                     break
    #             else:
    #                 raise Exception("Downloaded audio file not found")
            
    #         # Validate duration, if significantly different from source video, try ffmpeg normalization re-encapsulation once
    #         try:
    #             import subprocess, shlex
    #             probe_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(audio_file)}"
    #             out = subprocess.check_output(probe_cmd, shell=True).decode().strip()
    #             actual_duration = float(out) if out else 0.0
    #         except Exception as _:
    #             actual_duration = 0.0
            
    #         if expected_duration and actual_duration and abs(actual_duration - expected_duration) / expected_duration > 0.1:
    #             logger.warning(
    #                 f"Audio duration abnormal, expected {expected_duration}s, actual {actual_duration}s, attempting re-encapsulation fix…"
    #             )
    #             try:
    #                 fixed_path = str(output_dir / f"audio_{unique_id}_fixed.m4a")
    #                 fix_cmd = f"ffmpeg -y -i {shlex.quote(audio_file)} -vn -c:a aac -b:a 160k -movflags +faststart {shlex.quote(fixed_path)}"
    #                 subprocess.check_call(fix_cmd, shell=True)
    #                 # Replace with fixed file
    #                 audio_file = fixed_path
    #                 # Re-probe
    #                 out2 = subprocess.check_output(probe_cmd.replace(shlex.quote(audio_file.rsplit('.',1)[0]+'.m4a'), shlex.quote(audio_file)), shell=True).decode().strip()
    #                 actual_duration2 = float(out2) if out2 else 0.0
    #                 logger.info(f"Re-encapsulation completed, new duration≈{actual_duration2:.2f}s")
    #             except Exception as e:
    #                 logger.error(f"Re-encapsulation failed: {e}")
            
    #         logger.info(f"Audio file saved: {audio_file}")
    #         return audio_file, video_title
            
    #     except Exception as e:
    #         logger.error(f"Video download failed: {str(e)}")
    #         raise Exception(f"Video download failed: {str(e)}")

    def _get_ydl_opts(self, custom_opts: Optional[Dict] = None) -> Dict:
        """
        Get yt-dlp options with cookie support
        
        Args:
            custom_opts: Custom options to merge
        
        Returns:
            Complete yt-dlp options dictionary
        """
        opts = self.ydl_opts.copy()
        
        # FIX 2: Add cookiefile if cookies are enabled
        if self.cookie_manager.is_enabled():
            cookies_path = self.cookie_manager.get_cookies_path()
            if cookies_path:
                opts['cookiefile'] = cookies_path
                logger.debug(f"Using cookies from: {cookies_path}")
        
        # Merge custom options
        if custom_opts:
            opts.update(custom_opts)
        
        return opts

    def sanitize_filename(self, filename: str) -> str:
        """
        Clean filename by removing Windows unsupported characters
        """
        # Remove or replace Windows unsupported characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Remove control characters
        filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
        
        # Limit filename length (Windows path limitation)
        if len(filename) > 100:
            filename = filename[:100]
        
        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')
        
        return filename
    
    def get_short_path(self, path: str) -> str:
        """
        Get short path name on Windows to avoid path length limitations
        """
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                from ctypes import wintypes
                
                _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
                _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
                _GetShortPathNameW.restype = wintypes.DWORD
                
                output_buf_size = 260
                output_buf = ctypes.create_unicode_buffer(output_buf_size)
                ret = _GetShortPathNameW(path, output_buf, output_buf_size)
                
                if ret:
                    return output_buf.value
        except Exception as e:
            logger.warning(f"Unable to get short path: {e}")
        
        return path

    # async def download_and_convert(self, url: str, output_dir: Path) -> tuple[str, str]:
    #     """
    #     Download video and convert to m4a format - Windows optimized version
        
    #     Args:
    #         url: Video URL
    #         output_dir: Output directory
            
    #     Returns:
    #         Path to converted audio file and video title
    #     """
    #     try:
    #         # Create output directory
    #         output_dir.mkdir(exist_ok=True)
            
    #         # Generate short unique ID
    #         import uuid
    #         unique_id = str(uuid.uuid4())[:8]
            
    #         # Use short filename to avoid path length issues
    #         temp_filename = f"temp_{unique_id}"
    #         output_template = str(output_dir / f"{temp_filename}.%(ext)s")
            
    #         # Use short path on Windows
    #         if os.name == 'nt':
    #             short_output_dir = self.get_short_path(str(output_dir))
    #             if short_output_dir != str(output_dir):
    #                 output_template = f"{short_output_dir}\\{temp_filename}.%(ext)s"
            
    #         # Update yt-dlp options
    #         ydl_opts = self.ydl_opts.copy()
    #         ydl_opts['outtmpl'] = output_template
            
    #         logger.info(f"Starting video download: {url}")
    #         logger.info(f"Output template: {output_template}")
            
    #         # Get video info and download
    #         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    #             # Get video info
    #             info = await asyncio.to_thread(ydl.extract_info, url, False)
    #             video_title = info.get('title', 'unknown')
    #             expected_duration = info.get('duration') or 0
                
    #             # Clean video title
    #             clean_title = self.sanitize_filename(video_title)
    #             logger.info(f"Video title: {clean_title}")
                
    #             # Download video
    #             await asyncio.to_thread(ydl.download, [url])
            
    #         # Find downloaded file
    #         audio_file = None
    #         possible_extensions = ['m4a', 'webm', 'mp4', 'mp3', 'wav', 'opus']
            
    #         for ext in possible_extensions:
    #             potential_file = output_dir / f"{temp_filename}.{ext}"
    #             if potential_file.exists():
    #                 audio_file = str(potential_file)
    #                 logger.info(f"Found downloaded file: {audio_file}")
    #                 break
            
    #         if not audio_file:
    #             # If not found, try to find any audio file starting with temp_
    #             for file_path in output_dir.glob(f"{temp_filename}.*"):
    #                 if file_path.suffix.lower() in ['.m4a', '.webm', '.mp4', '.mp3', '.wav', '.opus']:
    #                     audio_file = str(file_path)
    #                     logger.info(f"Found matching file: {audio_file}")
    #                     break
            
    #         if not audio_file:
    #             raise Exception("Downloaded audio file not found")
            
    #         # Rename to final filename (using cleaned title)
    #         final_filename = f"audio_{unique_id}.m4a"
    #         final_audio_path = output_dir / final_filename
            
    #         # If downloaded file is not m4a format, convert using ffmpeg
    #         if not audio_file.endswith('.m4a'):
    #             logger.info(f"Converting audio format: {audio_file} -> {final_audio_path}")
    #             await self.convert_to_m4a(audio_file, str(final_audio_path))
    #             # Delete original file
    #             try:
    #                 os.remove(audio_file)
    #             except Exception as e:
    #                 logger.warning(f"Failed to delete temp file: {e}")
    #             audio_file = str(final_audio_path)
    #         else:
    #             # If already m4a, just rename
    #             try:
    #                 os.rename(audio_file, str(final_audio_path))
    #                 audio_file = str(final_audio_path)
    #             except Exception as e:
    #                 logger.warning(f"Failed to rename file: {e}")
            
    #         # Validate file duration
    #         await self.validate_audio_duration(audio_file, expected_duration, output_dir, unique_id)
            
    #         logger.info(f"Audio file saved: {audio_file}")
    #         return audio_file, clean_title
            
    #     except Exception as e:
    #         logger.error(f"Video download failed: {str(e)}")
    #         raise Exception(f"Video download failed: {str(e)}")

    async def download_and_convert(self, url: str, output_dir: Path) -> tuple[str, str]:
        """
        Download video and convert to m4a format with cookie support
        
        Args:
            url: Video URL
            output_dir: Output directory
            
        Returns:
            Tuple of (audio_file_path, video_title)
        """
        try:
            output_dir.mkdir(exist_ok=True)
            
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            
            temp_filename = f"temp_{unique_id}"
            output_template = str(output_dir / f"{temp_filename}.%(ext)s")
            
            # Windows path optimization
            if os.name == 'nt':
                short_output_dir = self.get_short_path(str(output_dir))
                if short_output_dir != str(output_dir):
                    output_template = f"{short_output_dir}\\{temp_filename}.%(ext)s"
            
            # Get yt-dlp options with cookies
            ydl_opts = self._get_ydl_opts({'outtmpl': output_template})
            
            logger.info(f"Starting video download: {url}")
            logger.info(f"Using player clients: ios, web (flexible format selection)")

            if self.cookie_manager.is_enabled():
                logger.info("✅ Using cookies for authentication")
            else:
                logger.warning("⚠️  No cookies - may fail for restricted videos")

            logger.info(f"Output template: {output_template}")

            # Format fallback list - try progressively simpler formats if primary fails
            format_fallbacks = [
                'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio[ext=mp4]/bestaudio/best[acodec!=none]/best',
                'bestaudio/best',  # Simpler fallback
                'best',  # Last resort - any available format
            ]

            info = None
            video_title = 'unknown'
            expected_duration = 0
            download_success = False
            last_error = None
            clean_title = 'unknown'

            for format_idx, format_str in enumerate(format_fallbacks):
                try:
                    # Update format in options
                    ydl_opts['format'] = format_str

                    if format_idx > 0:
                        logger.info(f"Retrying with fallback format: {format_str}")

                    # Download with yt-dlp
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        # Get video info
                        info = await asyncio.to_thread(ydl.extract_info, url, False)
                        video_title = info.get('title', 'unknown')
                        expected_duration = info.get('duration') or 0

                        clean_title = self.sanitize_filename(video_title)
                        logger.info(f"Video title: {clean_title}")

                        # Download video
                        await asyncio.to_thread(ydl.download, [url])
                        download_success = True
                        break  # Success, exit loop

                except Exception as e:
                    last_error = e
                    error_str = str(e)
                    if "Requested format is not available" in error_str or "format" in error_str.lower():
                        logger.warning(f"Format '{format_str}' not available, trying next fallback...")
                        continue
                    else:
                        # Non-format related error, don't retry
                        raise

            if not download_success:
                if last_error:
                    raise last_error
                raise Exception("All format options failed")
            
            # Find downloaded file
            audio_file = None
            possible_extensions = ['m4a', 'webm', 'mp4', 'mp3', 'wav', 'opus']
            
            for ext in possible_extensions:
                potential_file = output_dir / f"{temp_filename}.{ext}"
                if potential_file.exists():
                    audio_file = str(potential_file)
                    logger.info(f"Found downloaded file: {audio_file}")
                    break
            
            if not audio_file:
                # Search for any file with temp_filename
                for file_path in output_dir.glob(f"{temp_filename}.*"):
                    if file_path.suffix.lower() in ['.m4a', '.webm', '.mp4', '.mp3', '.wav', '.opus']:
                        audio_file = str(file_path)
                        logger.info(f"Found matching file: {audio_file}")
                        break
            
            if not audio_file:
                raise Exception("Downloaded audio file not found")
            
            # Convert to m4a if needed
            final_filename = f"audio_{unique_id}.m4a"
            final_audio_path = output_dir / final_filename
            
            if not audio_file.endswith('.m4a'):
                logger.info(f"Converting audio format: {audio_file} -> {final_audio_path}")
                await self.convert_to_m4a(audio_file, str(final_audio_path))
                try:
                    os.remove(audio_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
                audio_file = str(final_audio_path)
            else:
                try:
                    os.rename(audio_file, str(final_audio_path))
                    audio_file = str(final_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to rename file: {e}")
            
            # Validate duration
            await self.validate_audio_duration(audio_file, expected_duration, output_dir, unique_id)
           
            logger.info(f"Audio file saved: {audio_file}")
            return audio_file, clean_title
            
        except Exception as e:
            logger.error(f"Video download failed: {str(e)}")
            
            # Provide helpful error message
            error_msg = str(e)
            if "Sign in to confirm" in error_msg or "bot" in error_msg.lower():
                if not self.cookie_manager.is_enabled():
                    error_msg += "\n\n💡 TIP: Enable cookies to bypass bot detection. Upload a cookies.txt file via API."
                else:
                    error_msg += "\n\n⚠️  Cookies are enabled but still failed. Try refreshing your cookies.txt file."
            
            raise Exception(f"Video download failed: {error_msg}")
    
    
    async def convert_to_m4a(self, input_file: str, output_file: str):
        """
        Convert audio to m4a format using ffmpeg
        """
        try:
            # Use appropriate quote handling on Windows
            if os.name == 'nt':
                cmd = [
                    'ffmpeg', '-y', '-i', input_file,
                    '-vn', '-c:a', 'aac', '-b:a', '160k',
                    '-movflags', '+faststart', output_file
                ]
            else:
                cmd = f"ffmpeg -y -i {shlex.quote(input_file)} -vn -c:a aac -b:a 160k -movflags +faststart {shlex.quote(output_file)}"
            
            process = await asyncio.create_subprocess_exec(
                *cmd if isinstance(cmd, list) else cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=not isinstance(cmd, list)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Conversion failed"
                raise Exception(f"FFmpeg conversion failed: {error_msg}")
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise
    
    async def validate_audio_duration(self, audio_file: str, expected_duration: float, output_dir: Path, unique_id: str):
        """
        Validate audio duration and fix if needed
        """
        try:
            # Get actual duration
            if os.name == 'nt':
                cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', audio_file
                ]
            else:
                cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {shlex.quote(audio_file)}"
            
            process = await asyncio.create_subprocess_exec(
                *cmd if isinstance(cmd, list) else cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=not isinstance(cmd, list)
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                duration_str = stdout.decode().strip()
                actual_duration = float(duration_str) if duration_str else 0.0
            else:
                actual_duration = 0.0
            
            # If duration differs significantly, try re-encapsulation
            if expected_duration and actual_duration and abs(actual_duration - expected_duration) / expected_duration > 0.1:
                logger.warning(
                    f"Audio duration abnormal, expected {expected_duration}s, actual {actual_duration}s, attempting re-encapsulation fix…"
                )
                
                fixed_path = str(output_dir / f"audio_{unique_id}_fixed.m4a")
                await self.convert_to_m4a(audio_file, fixed_path)
                
                # Replace original file with fixed file
                try:
                    os.remove(audio_file)
                    os.rename(fixed_path, audio_file)
                    logger.info("Re-encapsulation completed")
                except Exception as e:
                    logger.error(f"File replacement failed: {e}")
                    
        except Exception as e:
            logger.warning(f"Duration validation failed: {e}")
    
    def get_video_info(self, url: str) -> dict:
        """
        Get video information
        
        Args:
            url: Video URL
            
        Returns:
            Video information dictionary
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info.get('title', ''),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', ''),
                    'upload_date': info.get('upload_date', ''),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                }
        except Exception as e:
            logger.error(f"Failed to get video info: {str(e)}")
            raise Exception(f"Failed to get video info: {str(e)}")
        
    
    # Cookie management methods
    def enable_cookies(self) -> bool:
        """Enable cookie support"""
        return self.cookie_manager.enable_cookies()
    
    def disable_cookies(self):
        """Disable cookie support"""
        self.cookie_manager.disable_cookies()
    
    def is_cookies_enabled(self) -> bool:
        """Check if cookies are enabled"""
        return self.cookie_manager.is_enabled()
    
    def get_cookies_info(self) -> dict:
        """Get cookie information"""
        return self.cookie_manager.get_cookies_info()
    
    def upload_cookies(self, cookies_text: str) -> bool:
        """
        Upload cookies from text
        
        Args:
            cookies_text: Content of cookies.txt file
        
        Returns:
            True if uploaded successfully
        """
        return self.cookie_manager.save_cookies_from_text(cookies_text)
    
    def delete_cookies(self) -> bool:
        """Delete cookies file"""
        return self.cookie_manager.delete_cookies()