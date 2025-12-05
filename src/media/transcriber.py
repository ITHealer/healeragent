import os
from faster_whisper import WhisperModel
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Transcriber:
    """Audio transcriber that uses Faster-Whisper for speech-to-text conversion"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the transcriber
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_size = model_size
        self.model = None
        self.last_detected_language = None
        
    def _load_model(self):
        """Lazy load the model"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            try:
                self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error(f"Model loading failed: {str(e)}")
                raise Exception(f"Model loading failed: {str(e)}")
    
    async def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to the audio file
            language: Specify language (optional, auto-detect if not specified)
            
        Returns:
            Transcribed text (in Markdown format)
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                raise Exception(f"Audio file does not exist: {audio_path}")
            
            # Load model
            self._load_model()
            
            logger.info(f"Starting audio transcription: {audio_path}")
            
            # Direct call would block the event loop; run in thread to avoid blocking
            import asyncio
            def _do_transcribe():
                return self.model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=5,
                    best_of=5,
                    temperature=[0.0, 0.2, 0.4],  # Use temperature escalation strategy
                    # More robust: enable VAD with thresholds to reduce repetition from silence/noise
                    vad_filter=True,
                    vad_parameters={
                        "min_silence_duration_ms": 900,  # Silence detection duration
                        "speech_pad_ms": 300  # Speech padding
                    },
                    no_speech_threshold=0.7,  # No speech threshold
                    compression_ratio_threshold=2.3,  # Compression ratio threshold to detect repetition
                    log_prob_threshold=-1.0,  # Log probability threshold
                    # Avoid cascading repetition caused by error accumulation
                    condition_on_previous_text=False
                )
            segments, info = await asyncio.to_thread(_do_transcribe)
            
            detected_language = info.language
            self.last_detected_language = detected_language  # Save detected language
            logger.info(f"Detected language: {detected_language}")
            logger.info(f"Language detection probability: {info.language_probability:.2f}")
            
            # Assemble transcription results
            transcript_lines = []
            transcript_lines.append("# Video Transcription")
            transcript_lines.append("")
            transcript_lines.append(f"**Detected Language:** {detected_language}")
            transcript_lines.append(f"**Language Probability:** {info.language_probability:.2f}")
            transcript_lines.append("")
            transcript_lines.append("## Transcription Content")
            transcript_lines.append("")
            
            # Add timestamps and text
            for segment in segments:
                start_time = self._format_time(segment.start)
                end_time = self._format_time(segment.end)
                text = segment.text.strip()
                
                transcript_lines.append(f"**[{start_time} - {end_time}]**")
                transcript_lines.append("")
                transcript_lines.append(text)
                transcript_lines.append("")
            
            transcript_text = "\n".join(transcript_lines)
            logger.info("Transcription completed")
            
            return transcript_text
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise Exception(f"Transcription failed: {str(e)}")
    
    def _format_time(self, seconds: float) -> str:
        """
        Convert seconds to hours:minutes:seconds format
        
        Args:
            seconds: Number of seconds
            
        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def get_supported_languages(self) -> list:
        """
        Get list of supported languages
        """
        return [
            "zh", "en", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no"
        ]
    
    def get_detected_language(self, transcript_text: Optional[str] = None) -> Optional[str]:
        """
        Get the detected language
        
        Args:
            transcript_text: Transcribed text (optional, used to extract language info from text)
            
        Returns:
            Detected language code
        """
        # If there's a saved language, return it directly
        if self.last_detected_language:
            return self.last_detected_language
        
        # If transcript text is provided, try to extract language info from it
        if transcript_text and "**Detected Language:**" in transcript_text:
            lines = transcript_text.split('\n')
            for line in lines:
                if "**Detected Language:**" in line:
                    lang = line.split(":")[-1].strip()
                    return lang
        
        return None