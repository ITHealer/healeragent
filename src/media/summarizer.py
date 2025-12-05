import os
import openai
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class Summarizer:
    """Text Summarizer, generates multilingual summaries using the OpenAI API"""
    
    def __init__(self):
        """Initialize Summarizer"""
        # Get OpenAI API configuration from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        
        if not api_key:
            logger.warning("The OPENAI_API_KEY environment variable is not set, and the summary function will not be available")
        
        if api_key:
            if base_url:
                self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
                logger.info(f"The OpenAI client is initialized, using the custom endpoint: {base_url}")
            else:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized, using default endpoints")
        else:
            self.client = None
        
        # Supported language mappings
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
    
    async def optimize_transcript(self, raw_transcript: str) -> str:
        """
        Optimize transcripts: fix typos, segment by meaning
        Supports automatic segmentation of long text

        Args:
            raw_transcript: Raw transcript

        Returns:
            Optimized transcript (Markdown format)
        """
        try:
            if not self.client:
                logger.warning("OpenAI API unavailable, returning raw transcription")
                return raw_transcript

            # Preprocessing: remove only timestamps and meta information, keeping all spoken/repeated content
            preprocessed = self._remove_timestamps_and_meta(raw_transcript)
            # Use JS strategy: chunk by character length (closer to the tokens limit to avoid estimation errors)
            detected_lang_code = self._detect_transcript_language(preprocessed)
            max_chars_per_chunk = 4000  # Align JS: Maximum size per block is approximately 4000 characters

            if len(preprocessed) > max_chars_per_chunk:
                logger.info(f"The text is long ({len(preprocessed)} chars), enable chunking optimization")
                return await self._format_long_transcript_in_chunks(preprocessed, detected_lang_code, max_chars_per_chunk)
            else:
                return await self._format_single_chunk(preprocessed, detected_lang_code)

        except Exception as e:
            logger.error(f"Failed to optimize the transcript: {str(e)}")
            logger.info("Return to original transcript")
            return raw_transcript

    def _estimate_tokens(self, text: str) -> int:
        """
        Improved token count estimation algorithm
        More conservative estimation, taking into account system prompts and formatting overhead
        """
        # More conservative estimation: taking into account actual token inflation
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_words = len([word for word in text.split() if word.isascii() and word.isalpha()])
        
        # Calculate basic tokens
        base_tokens = chinese_chars * 1.5 + english_words * 1.3
        
        # Consider markdown format, timestamp and other overhead (about 30% additional overhead)
        format_overhead = len(text) * 0.15
        
        # Consider system prompt overhead (about 2000-3000 tokens)
        system_prompt_overhead = 2500
        
        total_estimated = int(base_tokens + format_overhead + system_prompt_overhead)
        
        return total_estimated

    async def _optimize_single_chunk(self, raw_transcript: str) -> str:
        """
        Optimize single text chunk
        """
        detected_lang = self._detect_transcript_language(raw_transcript)
        lang_instruction = self._get_language_instruction(detected_lang)
        
        system_prompt = f"""You are a professional text editing expert. Please optimize the provided video transcript text.

Special note: This may be an interview, dialogue, or speech. If it contains multiple speakers, you must maintain the original perspective of each speaker.

Requirements:
1. **Strictly maintain the original language ({lang_instruction}), absolutely do not translate to other languages**
2. **Completely remove all timestamp markers (like [00:00 - 00:05])**
3. **Intelligently identify and reorganize complete sentences split by timestamps**, grammatically incomplete sentence fragments need to be merged with context
4. Correct obvious typos and grammatical errors
5. Divide reorganized complete sentences into natural paragraphs based on semantic and logical meaning
6. Separate paragraphs with blank lines
7. **Strictly maintain original meaning unchanged, do not add or remove actual content**
8. **Absolutely do not change personal pronouns (like I/me, you, he/him, she/her, etc.)**
9. **Maintain the original perspective and context of each speaker**
10. **Identify dialogue structure: interviewers use "you", interviewees use "I/we", never confuse**
11. Ensure each sentence is grammatically complete, with fluent and natural language

Processing strategy:
- Prioritize identifying incomplete sentence fragments (like ending with prepositions, conjunctions, adjectives)
- Check adjacent text fragments, merge to form complete sentences
- Re-segment sentences to ensure each sentence is grammatically complete
- Re-divide into paragraphs based on topics and logic

Paragraph requirements:
- Divide by topics and logical meaning, each paragraph containing 1-8 related sentences
- Single paragraph length not exceeding 400 characters
- Avoid too many short paragraphs, merge related content
- Divide paragraphs when a complete thought or viewpoint is expressed

Output format:
- Plain text paragraphs, no timestamps or format markers
- Each sentence structurally complete
- Each paragraph discusses one main topic
- Separate paragraphs with blank lines

Important reminder: This is {lang_instruction} content, please optimize entirely in {lang_instruction}, focusing on solving incoherence problems caused by timestamp splitting of sentences! Be sure to perform reasonable paragraph division to avoid super long paragraphs!

**Key requirement: This may be an interview dialogue, absolutely do not change any personal pronouns or speaker perspectives! Interviewers say "you", interviewees say "I/we", must strictly maintain!**"""

        user_prompt = f"""Please optimize the following {lang_instruction} video transcript text into fluent paragraph text:

{raw_transcript}

Key tasks:
1. Remove all timestamp markers
2. Identify and reorganize complete sentences that were split
3. Ensure each sentence is grammatically complete and coherent in meaning
4. Re-divide into paragraphs by meaning, with blank lines between paragraphs
5. Keep {lang_instruction} language unchanged

Paragraph guidance:
- Divide by topics and logical meaning, each paragraph containing 1-8 related sentences
- Single paragraph length not exceeding 400 characters
- Avoid too many short paragraphs, merge related content
- Ensure clear blank lines between paragraphs

Please pay special attention to fixing sentence incompleteness issues caused by timestamp splitting and perform reasonable paragraph division!"""

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4000,  # Align JS: optimization/formatting stage max tokens ≈ 4000
            temperature=0.1
        )
        
        return response.choices[0].message.content

    async def _optimize_with_chunks(self, raw_transcript: str, max_tokens: int) -> str:
        """
        Optimize long text in chunks
        """
        detected_lang = self._detect_transcript_language(raw_transcript)
        lang_instruction = self._get_language_instruction(detected_lang)
        
        # Split original transcript by paragraphs (keep timestamps as splitting reference)
        chunks = self._split_into_chunks(raw_transcript, max_tokens)
        logger.info(f"Split into {len(chunks)} chunks for processing")
        
        optimized_chunks = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Optimizing chunk {i+1}/{len(chunks)}...")
            
            system_prompt = f"""You are a professional text editing expert. Please perform simple optimization on this transcript fragment.

This is part {i+1} of the complete transcript, total {len(chunks)} parts.

Simple optimization requirements:
1. **Strictly maintain original language ({lang_instruction})**, absolutely do not translate
2. **Only correct obvious typos and grammatical errors**
3. **Slightly adjust sentence fluency**, but do not extensively rewrite
4. **Maintain original text structure and length**, do not do complex paragraph reorganization
5. **Keep original meaning 100% unchanged**

Note: This is just preliminary cleanup, do not do complex rewriting or reorganization."""

            user_prompt = f"""Simply optimize the following {lang_instruction} text fragment (only correct typos and grammar):

{chunk}

Output cleaned text, maintaining original text structure."""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1200,  # Adapt to 4000 tokens total limit
                    temperature=0.1
                )
                
                optimized_chunk = response.choices[0].message.content
                optimized_chunks.append(optimized_chunk)
                
            except Exception as e:
                logger.error(f"Optimizing chunk {i+1} failed: {e}")
                # Use basic cleanup on failure
                cleaned_chunk = self._basic_transcript_cleanup(chunk)
                optimized_chunks.append(cleaned_chunk)
        
        # Merge all optimized chunks
        merged_text = "\n\n".join(optimized_chunks)
        
        # Perform secondary paragraph organization on merged text
        logger.info("Performing final paragraph organization...")
        final_result = await self._final_paragraph_organization(merged_text, lang_instruction)
        
        logger.info("Chunk optimization completed")
        return final_result

    # ===== JS openaiService.js port: chunking/context/deduplication/formatting =====

    def _ensure_markdown_paragraphs(self, text: str) -> str:
        """Ensure Markdown paragraph blank lines, blank lines after headers, compress excess blank lines."""
        
        if not text:
            return text
        formatted = text.replace("\r\n", "\n")
        import re
        # Add blank line after headers
        formatted = re.sub(r"(^#{1,6}\s+.*)\n([^\n#])", r"\1\n\n\2", formatted, flags=re.M)
        # Compress ≥3 line breaks to 2
        formatted = re.sub(r"\n{3,}", "\n\n", formatted)
        # Remove leading and trailing blank lines
        formatted = re.sub(r"^\n+", "", formatted)
        formatted = re.sub(r"\n+$", "", formatted)
        return formatted

    async def _format_single_chunk(self, chunk_text: str, transcript_language: str = 'zh') -> str:
        """Single chunk optimization (correction + formatting), following 4000 tokens limit."""
        # Build system/user prompts consistent with JS version
        if transcript_language == 'zh':
            # prompt = (
            #     "请对以下音频转录文本进行智能优化和格式化，要求：\n\n"
            #     "**内容优化（正确性优先）：**\n"
            #     "1. 错误修正（转录错误/错别字/同音字/专有名词）\n"
            #     "2. 适度改善语法，补全不完整句子，保持原意和语言不变\n"
            #     "3. 口语处理：保留自然口语与重复表达，不要删减内容，仅添加必要标点\n"
            #     "4. **绝对不要改变人称代词（I/我、you/你等）和说话者视角**\n\n"
            #     "**分段规则：**\n"
            #     "- 按主题和逻辑含义分段，每段包含1-8个相关句子\n"
            #     "- 单段长度不超过400字符\n"
            #     "- 避免过多的短段落，合并相关内容\n\n"
            #     "**格式要求：**Markdown 段落，段落间空行\n\n"
            #     f"原始转录文本：\n{chunk_text}"
            # )
            prompt = (
                "Please intelligently optimize and format the following audio transcript text, requirements:\n\n"
                "**Content Optimization (Accuracy Priority):**\n"
                "1. Error correction (transcription errors/typos/homophones/proper nouns)\n"
                "2. Moderate grammar improvement, complete incomplete sentences, maintain original meaning and language unchanged\n"
                "3. Speech processing: Keep natural speech and repetitive expressions, do not reduce content, only add necessary punctuation\n"
                "4. **Absolutely do not change personal pronouns (I/me, you, etc.) and speaker perspectives**\n\n"
                "**Paragraph Rules:**\n"
                "- Divide by topics and logical meaning, each paragraph containing 1-8 related sentences\n"
                "- Single paragraph length not exceeding 400 characters\n"
                "- Avoid too many short paragraphs, merge related content\n\n"
                "**Format Requirements:** Markdown paragraphs, blank lines between paragraphs\n\n"
                f"Original transcript text:\n{chunk_text}"
            )
            system_prompt = (
                "You are a professional audio transcript optimization assistant, correcting errors, improving fluency and formatting, "
                "must maintain original meaning, must not reduce speech/repetition/details; only remove timestamps or meta info."
                "Absolutely do not change personal pronouns or speaker perspectives. This may be an interview dialogue, interviewers use 'you', interviewees use 'I/we'."
            )
        else:
            prompt = (
                "Please intelligently optimize and format the following audio transcript text:\n\n"
                "Content Optimization (Accuracy First):\n"
                "1. Error Correction (typos, homophones, proper nouns)\n"
                "2. Moderate grammar improvement, complete incomplete sentences, keep original language/meaning\n"
                "3. Speech processing: keep natural fillers and repetitions, do NOT remove content; only add punctuation if needed\n"
                "4. **NEVER change pronouns (I, you, he, she, etc.) or speaker perspective**\n\n"
                "Segmentation Rules: Group 1-8 related sentences per paragraph by topic/logic; paragraph length NOT exceed 400 characters; avoid too many short paragraphs\n\n"
                "Format: Markdown paragraphs with blank lines between paragraphs\n\n"
                f"Original transcript text:\n{chunk_text}"
            )
            system_prompt = (
                "You are a professional transcript formatting assistant. Fix errors and improve fluency "
                "without changing meaning or removing any content; only timestamps/meta may be removed; keep Markdown paragraphs with blank lines. "
                "NEVER change pronouns or speaker perspective. This may be an interview: interviewer uses 'you', interviewee uses 'I/we'."
            )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,  # Align JS: optimization/formatting stage max tokens ≈ 4000
                temperature=0.1
            )
            optimized_text = response.choices[0].message.content or ""
            # Remove headings like "# Transcript" / "## Transcript"
            optimized_text = self._remove_transcript_heading(optimized_text)
            enforced = self._enforce_paragraph_max_chars(optimized_text.strip(), max_chars=400)
            return self._ensure_markdown_paragraphs(enforced)
        except Exception as e:
            logger.error(f"Single chunk text optimization failed: {e}")
            return self._apply_basic_formatting(chunk_text)

    def _smart_split_long_chunk(self, text: str, max_chars_per_chunk: int) -> list:
        """Safely split overlong text at sentence/space boundaries."""
        chunks = []
        pos = 0
        while pos < len(text):
            end = min(pos + max_chars_per_chunk, len(text))
            if end < len(text):
                # Prioritize sentence boundaries
                sentence_endings = ['。', '！', '？', '.', '!', '?']
                best = -1
                for ch in sentence_endings:
                    idx = text.rfind(ch, pos, end)
                    if idx > best:
                        best = idx
                if best > pos + int(max_chars_per_chunk * 0.7):
                    end = best + 1
                else:
                    # Second choice: space boundaries
                    space_idx = text.rfind(' ', pos, end)
                    if space_idx > pos + int(max_chars_per_chunk * 0.8):
                        end = space_idx
            chunks.append(text[pos:end].strip())
            pos = end
        return [c for c in chunks if c]

    def _find_safe_cut_point(self, text: str) -> int:
        """Find safe cut point (paragraph > sentence > phrase)."""
        import re
        # Paragraph
        p = text.rfind("\n\n")
        if p > 0:
            return p + 2
        # Sentence
        last_sentence_end = -1
        for m in re.finditer(r"[。！？\.!?]\s*", text):
            last_sentence_end = m.end()
        if last_sentence_end > 20:
            return last_sentence_end
        # Phrase
        last_phrase_end = -1
        for m in re.finditer(r"[，；,;]\s*", text):
            last_phrase_end = m.end()
        if last_phrase_end > 20:
            return last_phrase_end
        return len(text)

    def _find_overlap_between_texts(self, text1: str, text2: str) -> str:
        """Detect overlapping content between two adjacent texts for deduplication."""
        
        max_len = min(len(text1), len(text2))
        # Try step by step from long to short
        for length in range(max_len, 19, -1):
            suffix = text1[-length:]
            prefix = text2[:length]
            if suffix == prefix:
                cut = self._find_safe_cut_point(prefix)
                if cut > 20:
                    return prefix[:cut]
                return suffix
        return ""

    def _apply_basic_formatting(self, text: str) -> str:
        """Fallback when AI fails: join by sentences into paragraphs, paragraphs ≤250 characters, double line break separation."""
        
        if not text or not text.strip():
            return text
        import re
        parts = re.split(r"([。！？\.!?]+\s*)", text)
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                current += part
            else:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                    current = ""
        if current.strip():
            sentences.append(current.strip())
        paras = []
        cur = ""
        sentence_count = 0
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            sentence_count += 1
            # Improved paragraph logic: consider sentence count and length
            should_break = False
            if len(candidate) > 400 and cur:  # Paragraph too long
                should_break = True
            elif len(candidate) > 200 and sentence_count >= 3:  # Medium length and enough sentences
                should_break = True
            elif sentence_count >= 6:  # Too many sentences
                should_break = True
            
            if should_break:
                paras.append(cur.strip())
                cur = s
                sentence_count = 1
            else:
                cur = candidate
        if cur.strip():
            paras.append(cur.strip())
        return self._ensure_markdown_paragraphs("\n\n".join(paras))

    async def _format_long_transcript_in_chunks(self, raw_transcript: str, transcript_language: str, max_chars_per_chunk: int) -> str:
        """Smart chunking + context + deduplication to synthesize optimized text (JS strategy port)."""
        
        import re
        # First split by sentences, assemble chunks not exceeding max_chars_per_chunk
        parts = re.split(r"([。！？\.!?]+\s*)", raw_transcript)
        sentences = []
        buf = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                buf += part
            else:
                buf += part
                if buf.strip():
                    sentences.append(buf.strip())
                    buf = ""
        if buf.strip():
            sentences.append(buf.strip())

        chunks = []
        cur = ""
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = s
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        # Second-stage safe splitting for still overlong chunks
        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                final_chunks.extend(self._smart_split_long_chunk(c, max_chars_per_chunk))

        logger.info(f"Text divided into {len(final_chunks)} chunks for processing")

        optimized = []
        for i, c in enumerate(final_chunks):
            chunk_with_context = c
            if i > 0:
                prev_tail = final_chunks[i - 1][-100:]
                marker = f"[上文续：{prev_tail}]" if transcript_language == 'zh' else f"[Context continued: {prev_tail}]"
                chunk_with_context = marker + "\n\n" + c
            try:
                oc = await self._format_single_chunk(chunk_with_context, transcript_language)
                # Remove context markers
                oc = re.sub(r"^\[(上文续|Context continued)：?:?.*?\]\s*", "", oc, flags=re.S)
                optimized.append(oc)
            except Exception as e:
                logger.warning(f"Chunk {i+1} optimization failed, using basic formatting: {e}")
                optimized.append(self._apply_basic_formatting(c))

        # Adjacent chunk deduplication
        deduped = []
        for i, c in enumerate(optimized):
            cur_txt = c
            if i > 0 and deduped:
                prev = deduped[-1]
                overlap = self._find_overlap_between_texts(prev[-200:], cur_txt[:200])
                if overlap:
                    cur_txt = cur_txt[len(overlap):].lstrip()
                    if not cur_txt:
                        continue
            if cur_txt.strip():
                deduped.append(cur_txt)

        merged = "\n\n".join(deduped)
        merged = self._remove_transcript_heading(merged)
        enforced = self._enforce_paragraph_max_chars(merged, max_chars=400)
        return self._ensure_markdown_paragraphs(enforced)

    def _remove_timestamps_and_meta(self, text: str) -> str:
        """Only remove timestamp lines and obvious meta information (titles, detected language, etc.), preserve original speech/repetition."""
        lines = text.split('\n')
        kept = []
        for line in lines:
            s = line.strip()
            # Skip timestamps and meta information
            if (s.startswith('**[') and s.endswith(']**')):
                continue
            if s.startswith('# '):
                # Skip top-level titles (usually video titles, can be added back at the end)
                continue
            # if s.startswith('**检测语言:**') or s.startswith('**语言概率:**'):
            if s.startswith('**Detection language:**') or s.startswith('**Language probability:**'):
                continue
            kept.append(line)
        # Normalize blank lines
        cleaned = '\n'.join(kept)
        return cleaned

    def _enforce_paragraph_max_chars(self, text: str, max_chars: int = 400) -> str:
        """Split by paragraphs and ensure each paragraph doesn't exceed max_chars, split into multiple paragraphs at sentence boundaries if necessary."""
        if not text:
            return text
        import re
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p is not None]
        new_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) <= max_chars:
                new_paragraphs.append(para)
                continue

            # Sentence splitting
            parts = re.split(r"([。！？\.!?]+\s*)", para)
            sentences = []
            buf = ""
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    buf += part
                else:
                    buf += part
                    if buf.strip():
                        sentences.append(buf.strip())
                        buf = ""
            if buf.strip():
                sentences.append(buf.strip())
            cur = ""
            for s in sentences:
                candidate = (cur + (" " if cur else "") + s).strip()
                if len(candidate) > max_chars and cur:
                    new_paragraphs.append(cur)
                    cur = s
                else:
                    cur = candidate
            if cur:
                new_paragraphs.append(cur)
        return "\n\n".join([p.strip() for p in new_paragraphs if p is not None])

    def _remove_transcript_heading(self, text: str) -> str:
        """Remove lines with Transcript as title (any level #) at the beginning or in paragraphs, without changing main content."""
        if not text:
            return text
        import re
        # Remove title lines like '## Transcript', '# Transcript Text', '### transcript'
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^#{1,6}\s*transcript(\s+text)?\s*$", stripped, flags=re.I):
                continue
            filtered.append(line)
        return '\n'.join(filtered)

    def _split_into_chunks(self, text: str, max_tokens: int) -> list:
        """
        Intelligently split original transcript text into appropriately sized chunks
        Strategy: first extract pure text, then split naturally by sentences and paragraphs
        """
        import re
        
        # 1. First extract pure text content (remove timestamps, titles, etc.)
        pure_text = self._extract_pure_text(text)
        
        # 2. Split by sentences, maintaining sentence integrity
        sentences = self._split_into_sentences(pure_text)
        
        # 3. Assemble into chunks according to token limit
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            
            # Check if can add to current chunk
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Current chunk is full, save and start new chunk
                chunks.append(self._join_sentences(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                # Add to current chunk
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add last chunk
        if current_chunk:
            chunks.append(self._join_sentences(current_chunk))
        
        return chunks
    
    def _extract_pure_text(self, raw_transcript: str) -> str:
        """
        Extract pure text from raw transcript, remove timestamps and metadata
        """
        lines = raw_transcript.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip timestamps, titles, metadata
            if (line.startswith('**[') and line.endswith(']**') or
                line.startswith('#') or
                line.startswith('**检测语言:**') or
                line.startswith('**Detect language:**') or
                line.startswith('**语言概率:**') or
                line.startswith('**Language probability:**') or
                not line):
                continue
            text_lines.append(line)
        
        return ' '.join(text_lines)
    
    def _split_into_sentences(self, text: str) -> list:
        """
        Split text by sentences, considering Chinese and English differences
        """
        import re
        
        # Chinese and English sentence endings
        sentence_endings = r'[.!?。！？;；]+'
        
        # Split sentences, keep periods
        parts = re.split(f'({sentence_endings})', text)
        
        sentences = []
        current = ""
        
        for i, part in enumerate(parts):
            if re.match(sentence_endings, part):
                # This is a sentence ending, add to current sentence
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                # This is sentence content
                current += part
        
        # Handle last part without period
        if current.strip():
            sentences.append(current.strip())
        
        return [s for s in sentences if s.strip()]
    
    def _join_sentences(self, sentences: list) -> str:
        """
        Recombine sentences into paragraphs
        """
        return ' '.join(sentences)

    def _basic_transcript_cleanup(self, raw_transcript: str) -> str:
        """
        Basic transcript text cleanup: remove timestamps and title information
        Fallback when GPT optimization fails
        """
        lines = raw_transcript.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip timestamp lines
            if line.strip().startswith('**[') and line.strip().endswith(']**'):
                continue
            # Skip title lines
            if line.strip().startswith('# ') or line.strip().startswith('## '):
                continue
            # Skip detected language and other meta info lines
            if line.strip().startswith('**检测语言:**') or line.strip().startswith('**语言概率:**') or line.strip().startswith('**Detection language:**') or line.strip().startswith('**Language probability:**'):
                continue
            # Keep non-empty text lines
            if line.strip():
                cleaned_lines.append(line.strip())
        
        # Recombine sentences and intelligently segment paragraphs
        text = ' '.join(cleaned_lines)
        
        # Smarter sentence processing, considering Chinese-English differences
        import re
        
        # Split by periods, question marks, exclamation marks
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        paragraphs = []
        current_paragraph = []
        
        for i, sentence in enumerate(sentences):
            if sentence:
                current_paragraph.append(sentence)
                
                # Smart paragraph conditions:
                # 1. Every 3 sentences per paragraph (basic rule)
                # 2. Force paragraph break when encountering topic transition keywords
                # 3. Avoid super long paragraphs
                topic_change_keywords = [
                    '首先', '其次', '然后', '接下来', '另外', '此外', '最后', '总之',
                    'first', 'second', 'third', 'next', 'also', 'however', 'finally',
                    '现在', '那么', '所以', '因此', '但是', '然而',
                    'now', 'so', 'therefore', 'but', 'however'
                ]
                
                should_break = False
                
                # Check if paragraph break is needed
                if len(current_paragraph) >= 3:  # Basic length condition
                    should_break = True
                elif len(current_paragraph) >= 2:
                    for keyword in topic_change_keywords:
                        if sentence.lower().startswith(keyword.lower()):
                            should_break = True
                            break
                
                if should_break or len(current_paragraph) >= 4:  # Maximum length limit
                    # Combine current paragraph
                    paragraph_text = '. '.join(current_paragraph)
                    if not paragraph_text.endswith('.'):
                        paragraph_text += '.'
                    paragraphs.append(paragraph_text)
                    current_paragraph = []
        
        # Add remaining sentences
        if current_paragraph:
            paragraph_text = '. '.join(current_paragraph)
            if not paragraph_text.endswith('.'):
                paragraph_text += '.'
            paragraphs.append(paragraph_text)
        
        return '\n\n'.join(paragraphs)

    async def _final_paragraph_organization(self, text: str, lang_instruction: str) -> str:
        """
        Final paragraph organization for merged text
        Using improved prompts and engineering validation
        """
        try:
            # Estimate text length, if too long then process in chunks
            estimated_tokens = self._estimate_tokens(text)
            if estimated_tokens > 3000:  # For very long texts, process in chunks
                return await self._organize_long_text_paragraphs(text, lang_instruction)
            
            system_prompt = f"""You are a professional {lang_instruction} text paragraph organization expert. Your task is to reorganize paragraphs according to semantics and logic.

🎯 **Core Principles**:
1. **Strictly maintain original language ({lang_instruction})**, never translate
2. **Keep all content intact**, do not delete or add any information
3. **Segment by semantic logic**: each paragraph centers around one complete thought or topic
4. **Strictly control paragraph length**: each paragraph must not exceed 250 words
5. **Maintain natural flow**: paragraphs should have logical connections

📏 **Segmentation Standards**:
- **Semantic completeness**: each paragraph tells one complete concept or event
- **Moderate length**: 3-7 sentences, each paragraph must not exceed 250 words
- **Logical boundaries**: segment at topic transitions, time transitions, viewpoint transitions
- **Natural breakpoints**: follow speaker's natural pauses and logic

⚠️ **Strictly Forbidden**:
- Creating giant paragraphs exceeding 250 words
- Forcibly merging unrelated content
- Interrupting complete stories or discussions

Output format: separate paragraphs with blank lines."""

            user_prompt = f"""Please reorganize the paragraph structure of the following {lang_instruction} text. Strictly segment according to semantics and logic, ensuring each paragraph does not exceed 200 words:

{text}

Reorganized text:"""

            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,  # Align with JS: paragraph organization stage max tokens ≈ 4000
                temperature=0.05  # Lower temperature for better consistency
            )
            
            organized_text = response.choices[0].message.content
            
            # Engineering validation: check paragraph lengths
            validated_text = self._validate_paragraph_lengths(organized_text)
            
            return validated_text
            
        except Exception as e:
            logger.error(f"Final paragraph organization failed: {e}")
            # Use basic segmentation processing when failed
            return self._basic_paragraph_fallback(text)

    async def _organize_long_text_paragraphs(self, text: str, lang_instruction: str) -> str:
        """
        For very long texts, organize paragraphs in chunks
        """
        try:
            # Split by existing paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            organized_chunks = []
            
            current_chunk = []
            current_tokens = 0
            max_chunk_tokens = 2500  # Chunk size adapted to 4000 tokens limit
            
            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)
                
                if current_tokens + para_tokens > max_chunk_tokens and current_chunk:
                    # Process current chunk
                    chunk_text = '\n\n'.join(current_chunk)
                    organized_chunk = await self._organize_single_chunk(chunk_text, lang_instruction)
                    organized_chunks.append(organized_chunk)
                    
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens
            
            # Process the last chunk
            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                organized_chunk = await self._organize_single_chunk(chunk_text, lang_instruction)
                organized_chunks.append(organized_chunk)
            
            return '\n\n'.join(organized_chunks)
            
        except Exception as e:
            logger.error(f"Long text paragraph organization failed: {e}")
            return self._basic_paragraph_fallback(text)

    async def _organize_single_chunk(self, text: str, lang_instruction: str) -> str:
        """
        Organize paragraphs for a single text chunk
        """
        system_prompt = f"""You are a {lang_instruction} paragraph organization expert. Reorganize paragraphs by semantics, ensuring each paragraph does not exceed 200 words.

Core requirements:
1. Strictly maintain the original {lang_instruction} language
2. Organize by semantic logic, one theme per paragraph
3. Each paragraph must not exceed 250 words
4. Separate paragraphs with blank lines
5. Keep content complete, do not reduce information"""

        user_prompt = f"""Re-paragraph the following text in {lang_instruction}, ensuring each paragraph does not exceed 200 words:

{text}"""

        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1200,  # Adapted to 4000 tokens total limit
            temperature=0.05
        )
        
        return response.choices[0].message.content

    def _validate_paragraph_lengths(self, text: str) -> str:
        """
        Validate paragraph lengths, attempt to split if there are overly long paragraphs
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        validated_paragraphs = []
        
        for para in paragraphs:
            word_count = len(para.split())
            
            if word_count > 300:  # If paragraph exceeds 300 words
                logger.warning(f"Detected overly long paragraph ({word_count} words), attempting to split")
                # Attempt to split long paragraph by sentences
                split_paras = self._split_long_paragraph(para)
                validated_paragraphs.extend(split_paras)
            else:
                validated_paragraphs.append(para)
        
        return '\n\n'.join(validated_paragraphs)

    def _split_long_paragraph(self, paragraph: str) -> list:
        """
        Split overly long paragraphs
        """
        import re
        
        # Split by sentences
        sentences = re.split(r'[.!?。！？]\s+', paragraph)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        split_paragraphs = []
        current_para = []
        current_words = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_words + sentence_words > 200 and current_para:
                # Current paragraph reaches length limit
                split_paragraphs.append(' '.join(current_para))
                current_para = [sentence]
                current_words = sentence_words
            else:
                current_para.append(sentence)
                current_words += sentence_words
        
        # Add last paragraph
        if current_para:
            split_paragraphs.append(' '.join(current_para))
        
        return split_paragraphs

    def _basic_paragraph_fallback(self, text: str) -> str:
        """
        Basic segmentation fallback mechanism
        When GPT organization fails, use simple rule-based segmentation
        """
        import re
        
        # Remove extra blank lines
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        basic_paragraphs = []
        
        for para in paragraphs:
            word_count = len(para.split())
            
            if word_count > 250:
                # Split long paragraphs by sentences
                split_paras = self._split_long_paragraph(para)
                basic_paragraphs.extend(split_paras)
            elif word_count < 30 and basic_paragraphs:
                # Merge short paragraphs with previous one (if combined doesn't exceed 200 words)
                last_para = basic_paragraphs[-1]
                combined_words = len(last_para.split()) + word_count
                
                if combined_words <= 200:
                    basic_paragraphs[-1] = last_para + ' ' + para
                else:
                    basic_paragraphs.append(para)
            else:
                basic_paragraphs.append(para)
        
        return '\n\n'.join(basic_paragraphs)

    async def summarize(self, transcript: str, target_language: str = "zh", video_title: str = None) -> str:
        """
        Generate summary of video transcript
        
        Args:
            transcript: Transcript text
            target_language: Target language code
            
        Returns:
            Summary text (Markdown format)
        """
        try:
            if not self.client:
                logger.warning("OpenAI API unavailable, generating fallback summary")
                return self._generate_fallback_summary(transcript, target_language, video_title)
            
            # Estimate transcript text length, decide whether chunked summary is needed
            estimated_tokens = self._estimate_tokens(transcript)
            max_summarize_tokens = 4000  # Increase limit, prioritize single text processing for better summary quality
            
            if estimated_tokens <= max_summarize_tokens:
                # Summarize short text directly
                return await self._summarize_single_text(transcript, target_language, video_title)
            else:
                # Chunked summary for long text
                logger.info(f"Text is long ({estimated_tokens} tokens), enabling chunked summary")
                return await self._summarize_with_chunks(transcript, target_language, video_title, max_summarize_tokens)
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return self._generate_fallback_summary(transcript, target_language, video_title)

    async def _summarize_single_text(self, transcript: str, target_language: str, video_title: str = None) -> str:
        """
        Enhanced summarize for single video transcript with better markdown formatting
        """
        # Get target language name
        language_name = self.language_map.get(target_language, "English")
        
        # Enhanced prompts for video content with better structure
        system_prompt = f"""You are an expert video content analyst specializing in creating engaging, comprehensive summaries. 
    Your task is to transform video transcripts into well-structured, reader-friendly summaries in {language_name}.

    **YOUR ROLE:**
    - Extract and organize the key messages, insights, and valuable information from video content
    - Present information in a format that's easy to scan and understand
    - Anticipate viewer questions and provide clear answers
    - Make the content accessible and engaging

    **MARKDOWN FORMATTING REQUIREMENTS:**
    1. Use clear visual hierarchy with headers, bold text, and bullet points
    2. Start with a compelling overview that hooks the reader
    3. Use **bold** for emphasis on key terms, names, numbers, and important facts
    4. Use bullet points (•) for listing ideas, tips, or key points
    5. Use > blockquotes for memorable quotes or critical insights
    6. Include emoji icons sparingly for visual appeal (📌 🎯 💡 🔑 ⚡)
    7. Keep paragraphs short and scannable (2-3 sentences max)

    **CONTENT STRUCTURE:**
    1. Brief engaging introduction (what's this video about and why should I care?)
    2. Main content organized by topic/theme
    3. Key insights and practical takeaways
    4. Anticipated questions and clarifications
    5. Action items or next steps (if applicable)

    Write entirely in {language_name} with natural, fluent expression."""

        user_prompt = f"""Transform this video transcript into an engaging, comprehensive summary in {language_name}:

    **Video Title:** {video_title if video_title else "Video Content"}

    **Transcript:**
    {transcript}

    **REQUIREMENTS:**
    1. **📌 Overview** - Start with a compelling 2-3 sentence hook that captures the essence
    2. **Main Content** - Organize by themes, use natural paragraphs with proper spacing
    3. **💡 Key Insights** - Highlight the most valuable points and discoveries
    4. **❓ Anticipated Questions** - Address 2-3 questions viewers might have
    5. **🎯 Takeaways** - End with clear, actionable takeaways or conclusions

    Remember:
    - Write in a conversational yet informative tone
    - Focus on VALUE - what will the viewer gain from this?
    - Make it scannable with good visual hierarchy
    - Include specific examples, data, or quotes when relevant
    - Ensure smooth transitions between sections
    - Write entirely in {language_name}"""

        logger.info(f"Generating enhanced {language_name} video summary...")
        
        # Call OpenAI API with enhanced prompts
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3500,
            temperature=0.4  # Slightly higher for more engaging tone
        )
        
        summary = response.choices[0].message.content
        
        # Post-process for better formatting
        summary = self._enhance_markdown_formatting(summary)
        
        return self._format_video_summary(summary, target_language, video_title)


    def _format_video_summary(self, summary: str, target_language: str, video_title: str = None) -> str:
        """
        Format final video summary with appropriate metadata and structure
        """
        if video_title:
            # Add video title as main header
            formatted = f"# 🎬 {video_title}\n\n"
            formatted += summary
        else:
            formatted = summary
        
        # Add timestamp if not present
        if "---" not in formatted[-100:]:  # Check last 100 chars
            formatted += f"\n\n---\n\n"
        
        # Add language indicator if switching languages
        language_name = self.language_map.get(target_language, target_language)
        if target_language != "en":  # Add language note for non-English
            formatted += f"\n*Summary generated in {language_name}*"
        
        return formatted

    async def _summarize_with_chunks(self, transcript: str, target_language: str, video_title: str, max_tokens: int) -> str:
        """
        Enhanced chunked summary for long videos with better structure
        """
        language_name = self.language_map.get(target_language, "English")
        
        # Smart chunking by semantic boundaries
        chunks = self._smart_chunk_text(transcript, max_chars_per_chunk=4000)
        logger.info(f"Split into {len(chunks)} chunks for video summarization")
        
        chunk_summaries = []
        
        # Generate focused summary for each chunk
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing video chunk {i+1}/{len(chunks)}...")
            
            system_prompt = f"""You are a video content analyst creating a summary for part {i+1} of {len(chunks)} of a video.

    Focus on:
    - New information introduced in this segment
    - How it connects to the overall narrative
    - Key points that viewers need to remember
    - Any practical tips or insights

    Write in {language_name} with clear, engaging language."""

            user_prompt = f"""[Segment {i+1}/{len(chunks)}] Summarize this video segment in {language_name}:

    {chunk}

    Focus on:
    - What's being discussed in this part
    - Key points and insights
    - How this connects to the main topic
    - Any examples or demonstrations

    Keep it concise but comprehensive (150-250 words)."""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.4
                )
                
                chunk_summary = response.choices[0].message.content
                chunk_summaries.append(chunk_summary)
                
            except Exception as e:
                logger.error(f"Failed to summarize chunk {i+1}: {e}")
                chunk_summaries.append(f"[Segment {i+1} - Processing Error]")
        
        # Integrate all summaries into cohesive final summary
        final_summary = await self._integrate_video_summaries(chunk_summaries, target_language, video_title)
        
        return self._format_video_summary(final_summary, target_language, video_title)

    def _smart_chunk_text(self, text: str, max_chars_per_chunk: int = 3500) -> list:
        """Smart chunking (paragraph first then sentence), split by character limit."""
        chunks = []
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        cur = ""
        for p in paragraphs:
            candidate = (cur + "\n\n" + p).strip() if cur else p
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = p
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        # Second pass: split overly long chunks by sentences
        import re
        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                sentences = [s.strip() for s in re.split(r"[。！？\.!?]+", c) if s.strip()]
                scur = ""
                for s in sentences:
                    candidate = (scur + '。' + s).strip() if scur else s
                    if len(candidate) > max_chars_per_chunk and scur:
                        final_chunks.append(scur.strip())
                        scur = s
                    else:
                        scur = candidate
                if scur.strip():
                    final_chunks.append(scur.strip())
        return final_chunks

    async def _integrate_video_summaries(self, chunk_summaries: list, target_language: str, video_title: str) -> str:
        """
        Integrate multiple chunk summaries into a cohesive video summary
        """
        language_name = self.language_map.get(target_language, "English")
        combined = "\n\n---\n\n".join([f"**Segment {i+1}:**\n{s}" for i, s in enumerate(chunk_summaries)])
        
        try:
            system_prompt = f"""You are a video content editor creating a final, polished summary from multiple segments.

    Your task is to:
    1. Combine segment summaries into a cohesive narrative
    2. Remove redundancy while preserving all key information
    3. Create a smooth flow between topics
    4. Highlight the most important insights
    5. Add structure that makes the content easy to navigate

    Write in {language_name} with engaging, accessible language."""

            user_prompt = f"""Create a comprehensive video summary in {language_name} from these segments:

    **Video:** {video_title if video_title else "Video Content"}

    **Segment Summaries:**
    {combined}

    **FINAL SUMMARY STRUCTURE:**
    1. **📌 Overview** - Compelling introduction (what's this video about?)
    2. **📚 Main Content** - Organized by themes/topics from all segments
    3. **💡 Key Insights** - Most valuable discoveries and points
    4. **❓ Common Questions** - Address 3-4 questions viewers might have
    5. **🎯 Action Items** - Clear takeaways and next steps

    Requirements:
    - Maintain chronological flow where it makes sense
    - Use visual markers (bold, bullets, emojis) for scannability  
    - Include specific examples or data mentioned
    - Write entirely in {language_name}
    - Make it engaging and valuable for viewers"""

            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=3000,
                temperature=0.4
            )
            
            integrated = response.choices[0].message.content
            return self._enhance_markdown_formatting(integrated)
            
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            return "\n\n".join(chunk_summaries)


    def _format_summary_with_meta(self, summary: str, target_language: str, video_title: str = None) -> str:
        """
        Add title and meta information to summary
        """
        language_name = self.language_map.get(target_language, "中文（简体）")
        meta_labels = self._get_summary_labels(target_language)
        
        # Don't add any subheadings/disclaimers, can keep video title as primary heading
        if video_title:
            prefix = f"# {video_title}\n\n"
        else:
            prefix = ""
        return prefix + summary

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1200,
                temperature=0.2,
            )
            return resp.choices[0].message.content
        except Exception:
            return text
    

    def _enhance_markdown_formatting(self, text: str) -> str:
        """
        Post-process to ensure clean markdown formatting for video summaries
        """
        import re
        
        # Ensure proper spacing around headers
        text = re.sub(r'(#{1,3}[^#\n]+)\n([^\n])', r'\1\n\n\2', text)
        
        # Fix bullet points spacing
        text = re.sub(r'\n•', r'\n• ', text)
        text = re.sub(r'\n-([^\s])', r'\n- \1', text)
        
        # Ensure blockquotes have proper spacing
        text = re.sub(r'\n>', r'\n\n>', text)
        text = re.sub(r'>\n([^>\n])', r'>\n\n\1', text)
        
        # Remove excessive blank lines
        text = re.sub(r'\n{4,}', r'\n\n\n', text)
        
        # Ensure emoji headers have proper formatting
        text = re.sub(r'(\*\*[📌🎯📚💡❓🔑⚡🎬].+?\*\*)', r'\n\1\n', text)
        
        # Fix spacing around horizontal rules
        text = re.sub(r'\n---\n', r'\n\n---\n\n', text)
        
        return text.strip()

    def _get_video_summary_prompt_enhancement(self, content_type: str, target_language: str) -> str:
        """
        Get content-type specific prompt enhancements for video summaries
        """
        enhancements = {
            "tutorial": {
                "en": "Focus on step-by-step instructions, prerequisites, and common mistakes to avoid.",
                "vi": "Tập trung vào hướng dẫn từng bước, điều kiện tiên quyết và các lỗi thường gặp cần tránh.",
                "zh": "重点关注分步说明、先决条件和需要避免的常见错误。"
            },
            "presentation": {
                "en": "Highlight main arguments, supporting evidence, and key conclusions.",
                "vi": "Làm nổi bật các luận điểm chính, bằng chứng hỗ trợ và kết luận quan trọng.",
                "zh": "突出主要论点、支持证据和关键结论。"
            },
            "interview": {
                "en": "Focus on key questions asked, insights shared, and memorable quotes.",
                "vi": "Tập trung vào các câu hỏi chính, thông tin chi tiết được chia sẻ và trích dẫn đáng nhớ.",
                "zh": "关注提出的关键问题、分享的见解和难忘的引言。"
            },
            "educational": {
                "en": "Emphasize concepts explained, examples provided, and learning objectives.",
                "vi": "Nhấn mạnh các khái niệm được giải thích, ví dụ được cung cấp và mục tiêu học tập.",
                "zh": "强调解释的概念、提供的示例和学习目标。"
            }
        }
        
        content_prompts = enhancements.get(content_type, {})
        return content_prompts.get(target_language, "")

    # Add new helper method to detect video content type
    def _detect_video_content_type(self, transcript: str, title: str = None) -> str:
        """
        Detect the type of video content from transcript and title
        """
        transcript_lower = transcript.lower()
        title_lower = (title or "").lower()
        
        # Keywords for different content types
        tutorial_keywords = ["how to", "tutorial", "guide", "step by step", "learn", "teaching"]
        interview_keywords = ["interview", "conversation", "discuss", "talk with", "q&a", "question"]
        presentation_keywords = ["presentation", "keynote", "conference", "talk", "speech", "lecture"]
        educational_keywords = ["lesson", "course", "education", "training", "workshop", "class"]
        
        # Check title first (more reliable)
        combined_text = title_lower + " " + transcript_lower[:500]
        
        if any(kw in combined_text for kw in tutorial_keywords):
            return "tutorial"
        elif any(kw in combined_text for kw in interview_keywords):
            return "interview"
        elif any(kw in combined_text for kw in presentation_keywords):
            return "presentation"
        elif any(kw in combined_text for kw in educational_keywords):
            return "educational"
        else:
            return "general"
    
    def _generate_fallback_summary(self, transcript: str, target_language: str, video_title: str = None) -> str:
        """
        Generate fallback summary (when OpenAI API is unavailable)
        
        Args:
            transcript: Transcript text
            video_title: Video title
            target_language: Target language code
            
        Returns:
            Fallback summary text
        """
        language_name = self.language_map.get(target_language, "中文（简体）")
        
        # Simple text processing, extract key information
        lines = transcript.split('\n')
        content_lines = [line for line in lines if line.strip() and not line.startswith('#') and not line.startswith('**')]
        
        # Calculate approximate length
        total_chars = sum(len(line) for line in content_lines)
        
        # Use target language labels
        meta_labels = self._get_summary_labels(target_language)
        fallback_labels = self._get_fallback_labels(target_language)
        
        # Directly use video title as main heading  
        title = video_title if video_title else "Summary"
        
        summary = f"""# {title}

**{meta_labels['language_label']}:** {language_name}
**{fallback_labels['notice']}:** {fallback_labels['api_unavailable']}



## {fallback_labels['overview_title']}

**{fallback_labels['content_length']}:** {fallback_labels['about']} {total_chars} {fallback_labels['characters']}
**{fallback_labels['paragraph_count']}:** {len(content_lines)} {fallback_labels['paragraphs']}

## {fallback_labels['main_content']}

{fallback_labels['content_description']}

{fallback_labels['suggestions_intro']}

1. {fallback_labels['suggestion_1']}
2. {fallback_labels['suggestion_2']}
3. {fallback_labels['suggestion_3']}

## {fallback_labels['recommendations']}

- {fallback_labels['recommendation_1']}
- {fallback_labels['recommendation_2']}


<br/>

<p style="color: #888; font-style: italic; text-align: center; margin-top: 16px;"><em>{fallback_labels['fallback_disclaimer']}</em></p>"""
        
        return summary
    
    def _get_current_time(self) -> str:
        """Get current time string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_supported_languages(self) -> dict:
        """
        Get supported languages list
        
        Returns:
            Language code to language name mapping
        """
        return self.language_map.copy()
    
    def _detect_transcript_language(self, transcript: str) -> str:
        """
        Detect main language of transcript text
        
        Args:
            transcript: Transcript text
            
        Returns:
            Detected language code
        """
        # Simple language detection logic: look for language markers in transcript text
        if "**检测语言:**" in transcript or "**Detect language:**" in transcript:
            # Extract detected language from Whisper transcript
            lines = transcript.split('\n')
            for line in lines:
                if "**检测语言:**" in line or "**Detect language:**" in line:
                    # Extract language code, e.g.: "**检测语言:** en"
                    lang = line.split(":")[-1].strip()
                    return lang
        
        # If no language marker found, use simple character detection
        # Calculate ratios of English characters, Chinese characters, etc.
        total_chars = len(transcript)
        if total_chars == 0:
            return "en"  # Default English
            
        # Count Chinese characters
        chinese_chars = sum(1 for char in transcript if '\u4e00' <= char <= '\u9fff')
        chinese_ratio = chinese_chars / total_chars
        
        # Count English letters
        english_chars = sum(1 for char in transcript if char.isascii() and char.isalpha())
        english_ratio = english_chars / total_chars
        
        # Judge by ratio
        if chinese_ratio > 0.3:
            return "zh"
        elif english_ratio > 0.3:
            return "en"
        else:
            return "en"  # Default English
    
    def _get_language_instruction(self, lang_code: str) -> str:
        """
        Get language name used in optimization instructions based on language code
        
        Args:
            lang_code: Language code
            
        Returns:
            Language name
        """
        language_instructions = {
            "en": "English",
            "vi": "Vietnamese",
            "zh": "中文",
            "ja": "日本語",
            "ko": "한국어",
            "es": "Español",
            "fr": "Français",
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ar": "العربية"
        }
        return language_instructions.get(lang_code, "English")
    

    def _get_summary_labels(self, lang_code: str) -> dict:
        """
        Get multilingual labels for summary page
        
        Args:
            lang_code: Language code
            
        Returns:
            Labels dictionary
        """
        labels = {
            "en": {
                "language_label": "Summary Language",
                "disclaimer": "This summary is automatically generated by AI for reference only"
            },
            "vi": {
                "language_label": "Summary Language",
                "disclaimer": "Tóm tắt này được AI tự động tạo ra chỉ để tham khảo"
            },
            "zh": {
                "language_label": "摘要语言",
                "disclaimer": "本摘要由AI自动生成，仅供参考"
            },
            "ja": {
                "language_label": "要約言語",
                "disclaimer": "この要約はAIによって自動生成されており、参考用です"
            },
            "ko": {
                "language_label": "요약 언어",
                "disclaimer": "이 요약은 AI에 의해 자동 생성되었으며 참고용입니다"
            },
            "es": {
                "language_label": "Idioma del Resumen",
                "disclaimer": "Este resumen es generado automáticamente por IA, solo para referencia"
            },
            "fr": {
                "language_label": "Langue du Résumé",
                "disclaimer": "Ce résumé est généré automatiquement par IA, à titre de référence uniquement"
            },
            "de": {
                "language_label": "Zusammenfassungssprache",
                "disclaimer": "Diese Zusammenfassung wird automatisch von KI generiert, nur zur Referenz"
            },
            "it": {
                "language_label": "Lingua del Riassunto",
                "disclaimer": "Questo riassunto è generato automaticamente dall'IA, solo per riferimento"
            },
            "pt": {
                "language_label": "Idioma do Resumo",
                "disclaimer": "Este resumo é gerado automaticamente por IA, apenas para referência"
            },
            "ru": {
                "language_label": "Язык резюме",
                "disclaimer": "Это резюме автоматически генерируется ИИ, только для справки"
            },
            "ar": {
                "language_label": "لغة الملخص",
                "disclaimer": "هذا الملخص تم إنشاؤه تلقائياً بواسطة الذكاء الاصطناعي، للمرجع فقط"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def _get_fallback_labels(self, lang_code: str) -> dict:
        """
        Get multilingual labels for fallback summary
        
        Args:
            lang_code: Language code
            
        Returns:
            Labels dictionary
        """
        labels = {
            "en": {
                "notice": "Notice",
                "api_unavailable": "OpenAI API is unavailable, this is a simplified summary",
                "overview_title": "Transcript Overview",
                "content_length": "Content Length",
                "about": "About",
                "characters": "characters",
                "paragraph_count": "Paragraph Count",
                "paragraphs": "paragraphs",
                "main_content": "Main Content",
                "content_description": "The transcript contains complete video speech content. Since AI summary cannot be generated currently, we recommend:",
                "suggestions_intro": "For detailed information, we suggest you:",
                "suggestion_1": "Review the complete transcript text for detailed information",
                "suggestion_2": "Focus on important paragraphs marked with timestamps",
                "suggestion_3": "Manually extract key points and takeaways",
                "recommendations": "Recommendations",
                "recommendation_1": "Configure OpenAI API key for better summary functionality",
                "recommendation_2": "Or use other AI services for text summarization",
                "fallback_disclaimer": "This is an automatically generated fallback summary"
            },
            "zh": {
                "notice": "注意",
                "api_unavailable": "由于OpenAI API不可用，这是一个简化的摘要",
                "overview_title": "转录概览",
                "content_length": "内容长度",
                "about": "约",
                "characters": "字符",
                "paragraph_count": "段落数量",
                "paragraphs": "段",
                "main_content": "主要内容",
                "content_description": "转录文本包含了完整的视频语音内容。由于当前无法生成智能摘要，建议您：",
                "suggestions_intro": "为获取详细信息，建议您：",
                "suggestion_1": "查看完整的转录文本以获取详细信息",
                "suggestion_2": "关注时间戳标记的重要段落",
                "suggestion_3": "手动提取关键观点和要点",
                "recommendations": "建议",
                "recommendation_1": "配置OpenAI API密钥以获得更好的摘要功能",
                "recommendation_2": "或者使用其他AI服务进行文本总结",
                "fallback_disclaimer": "本摘要为自动生成的备用版本"
            },
            "vi": {
                "notice": "Lưu ý",
                "api_unavailable": "OpenAI API không khả dụng, đây là bản tóm tắt đơn giản",
                "overview_title": "Tổng quan bản ghi",
                "content_length": "Độ dài nội dung",
                "about": "Khoảng",
                "characters": "ký tự",
                "paragraph_count": "Số đoạn văn",
                "paragraphs": "đoạn",
                "main_content": "Nội dung chính",
                "content_description": "Bản ghi chứa toàn bộ nội dung lời nói của video. Do hiện tại không thể tạo tóm tắt thông minh, chúng tôi khuyến nghị:",
                "suggestions_intro": "Để có thông tin chi tiết, chúng tôi đề xuất bạn:",
                "suggestion_1": "Xem lại toàn bộ văn bản bản ghi để có thông tin chi tiết",
                "suggestion_2": "Tập trung vào các đoạn quan trọng được đánh dấu thời gian",
                "suggestion_3": "Tự tay trích xuất các điểm chính và kiến thức rút ra",
                "recommendations": "Khuyến nghị",
                "recommendation_1": "Cấu hình khóa API OpenAI để có chức năng tóm tắt tốt hơn",
                "recommendation_2": "Hoặc sử dụng các dịch vụ AI khác để tóm tắt văn bản",
                "fallback_disclaimer": "Đây là bản tóm tắt dự phòng được tạo tự động"
            }
        }
        return labels.get(lang_code, labels["en"])
    
    def is_available(self) -> bool:
        """
        Check if the summary service is available
        
        Returns:
            True if OpenAI API is configured, False otherwise
        """
        return self.client is not None

    async def summarize_enhanced(
        self,
        transcript: str,
        target_language: str = "en",
        video_title: Optional[str] = None,
        content_type: Optional[str] = None,
        content_focus: Optional[str] = None,
        content_structure: Optional[str] = None
    ) -> str:
        """
        Enhanced summarization method with content-aware formatting
        
        Args:
            transcript: Video transcript text
            target_language: Target language code
            video_title: Video title
            content_type: Type of video (tutorial, interview, presentation, etc.)
            content_focus: Specific elements to focus on
            content_structure: Suggested structure for the summary
            
        Returns:
            Enhanced markdown-formatted summary
        """
        try:
            if not self.client:
                logger.warning("OpenAI API unavailable, generating fallback summary")
                return self._generate_fallback_summary(transcript, target_language, video_title)
            
            # Estimate tokens to decide chunking strategy
            estimated_tokens = self._estimate_tokens(transcript)
            max_summarize_tokens = 4000
            
            if estimated_tokens <= max_summarize_tokens:
                # Single text summary with enhancement
                return await self._summarize_single_enhanced(
                    transcript, target_language, video_title, 
                    content_type, content_focus, content_structure
                )
            else:
                # Chunked summary with enhancement
                logger.info(f"Long video ({estimated_tokens} tokens), using chunked summary")
                return await self._summarize_chunks_enhanced(
                    transcript, target_language, video_title,
                    content_type, content_focus, content_structure,
                    max_summarize_tokens
                )
                
        except Exception as e:
            logger.error(f"Enhanced summarization failed: {str(e)}")
            return self._generate_fallback_summary(transcript, target_language, video_title)

    async def _summarize_single_enhanced(
        self,
        transcript: str,
        target_language: str,
        video_title: Optional[str],
        content_type: Optional[str],
        content_focus: Optional[str],
        content_structure: Optional[str]
    ) -> str:
        """
        Enhanced single text summarization with content awareness
        """
        language_name = self.language_map.get(target_language, "English")
        
        # Build content-aware system prompt
        system_prompt = f"""You are an expert video content analyst specializing in {content_type or 'general'} content.
Create an engaging, comprehensive summary in {language_name}.

**YOUR EXPERTISE:**
- Analyzing {content_type or 'video'} content for key insights
- Creating scannable, valuable summaries that viewers love
- Anticipating viewer questions and providing clear answers

**MARKDOWN FORMATTING REQUIREMENTS:**
1. Start with a compelling hook using emoji headers
2. Use **bold** for all key terms, names, numbers, and important facts
3. Use bullet points (•) for listing ideas, tips, or steps
4. Use > blockquotes for memorable quotes or critical insights
5. Include strategic emoji icons (📌 🎯 💡 🔑 ⚡ 📚 ❓)
6. Keep paragraphs short (2-3 sentences maximum)
7. Create clear visual hierarchy with spacing

**CONTENT FOCUS:**
{content_focus or 'main topics, key insights, practical takeaways'}

**SUGGESTED STRUCTURE:**
{content_structure or 'overview → main content → insights → questions → takeaways'}

Write entirely in {language_name} with natural, engaging expression."""

        # Build user prompt with specific requirements
        user_prompt = f"""Transform this {'[' + content_type.upper() + ']' if content_type else ''} video transcript into an outstanding summary in {language_name}:

**Video Title:** {video_title or "Video Content"}

**Transcript:**
{transcript}

**CREATE THIS STRUCTURE:**

## 📌 Overview
[2-3 compelling sentences that hook the reader and explain why this matters]

## 📚 Main Content
[Organized by logical themes, use subheadings if needed]
[Include specific examples, data, quotes]
[Use **bold** liberally for emphasis]

## 💡 Key Insights
[3-5 most valuable discoveries or points]
[What makes this content unique or important]

## ❓ Viewer Questions Answered
[2-3 questions viewers likely have]
[Provide clear, concise answers]

## 🎯 Action Items & Takeaways
[Clear next steps or main points to remember]
[Practical applications if relevant]

Remember:
- Write conversationally but informatively
- Focus on VALUE - what will viewers gain?
- Make it highly scannable with visual markers
- Include specific details that matter
- Ensure {language_name} fluency throughout"""

        logger.info(f"Generating enhanced {language_name} summary for {content_type or 'general'} video")
        
        # Generate with OpenAI
        response = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3500,
            temperature=0.4  # Balanced creativity
        )
        
        summary = response.choices[0].message.content
        
        # Post-process for perfect formatting
        summary = self._enhance_markdown_formatting(summary)
        
        return self._finalize_video_summary(summary, target_language, video_title)

    async def _summarize_chunks_enhanced(
        self,
        transcript: str,
        target_language: str,
        video_title: Optional[str],
        content_type: Optional[str],
        content_focus: Optional[str],
        content_structure: Optional[str],
        max_tokens: int
    ) -> str:
        """
        Enhanced chunked summarization for long videos
        """
        language_name = self.language_map.get(target_language, "English")
        
        # Smart chunk the transcript
        chunks = self._smart_chunk_text(transcript, max_chars_per_chunk=4000)
        logger.info(f"Processing {len(chunks)} chunks for {content_type or 'general'} video")
        
        chunk_summaries = []
        
        # Process each chunk with context awareness
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
            
            system_prompt = f"""You are analyzing segment {i+1} of {len(chunks)} from a {content_type or 'video'}.
Focus on extracting key information that connects to the overall narrative.
Write in {language_name} with clear, engaging language.

Content focus: {content_focus or 'key points and insights'}"""

            user_prompt = f"""[SEGMENT {i+1}/{len(chunks)}] 
Summarize this video segment in {language_name}:

{chunk}

Requirements:
- Extract main points and insights
- Note connections to overall topic
- Include specific examples or data
- Use **bold** for emphasis
- Keep it concise (150-250 words)"""

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.4
                )
                
                chunk_summary = response.choices[0].message.content
                chunk_summaries.append(chunk_summary)
                
            except Exception as e:
                logger.error(f"Chunk {i+1} summarization failed: {e}")
                chunk_summaries.append(f"[Segment {i+1} - Error processing]")
        
        # Integrate all chunks into final summary
        final_summary = await self._integrate_enhanced_summaries(
            chunk_summaries, target_language, video_title,
            content_type, content_structure
        )
        
        return self._finalize_video_summary(final_summary, target_language, video_title)

    async def _integrate_enhanced_summaries(
        self,
        chunk_summaries: list,
        target_language: str,
        video_title: Optional[str],
        content_type: Optional[str],
        content_structure: Optional[str]
    ) -> str:
        """
        Integrate chunk summaries into cohesive final summary
        """
        language_name = self.language_map.get(target_language, "English")
        combined = "\n\n---\n\n".join([f"**Segment {i+1}:**\n{s}" for i, s in enumerate(chunk_summaries)])
        
        try:
            system_prompt = f"""You are creating the final summary for a {content_type or 'video'}.
Combine segment summaries into a polished, cohesive narrative in {language_name}.

Your expertise: Creating engaging, valuable summaries that viewers love."""

            user_prompt = f"""Create the FINAL comprehensive video summary in {language_name}:

**Video:** {video_title or "Video Content"}
**Type:** {content_type or "General"}

**Segment Summaries to Integrate:**
{combined}

**CREATE THIS FINAL STRUCTURE:**

## 📌 Overview
[Compelling 2-3 sentence introduction covering the entire video]

## 📚 Main Content
[Synthesize all segments into coherent themes]
[Use subheadings for major topics]
[Include best examples and data from all segments]

## 💡 Key Insights
[3-5 most important discoveries across all segments]
[What makes this content valuable]

## ❓ Common Questions
[3-4 questions viewers might have after watching]
[Clear, helpful answers]

## 🎯 Action Items & Takeaways
[Synthesized next steps from entire video]
[Main points to remember]

Requirements:
- Remove redundancy while preserving all unique information
- Create smooth transitions between topics
- Use **bold** for emphasis throughout
- Write entirely in {language_name}
- Make it engaging and valuable"""

            response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=3000,
                temperature=0.4
            )
            
            integrated = response.choices[0].message.content
            return self._enhance_markdown_formatting(integrated)
            
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            return "\n\n".join(chunk_summaries)

    def _enhance_markdown_formatting(self, text: str) -> str:
        """
        Post-process to ensure perfect markdown formatting
        """
        import re
        
        # Fix header spacing
        text = re.sub(r'(#{1,3}[^#\n]+)\n([^\n])', r'\1\n\n\2', text)
        
        # Fix bullet points
        text = re.sub(r'\n•', r'\n• ', text)
        text = re.sub(r'\n-([^\s])', r'\n- \1', text)
        text = re.sub(r'\n\*([^\s*])', r'\n* \1', text)
        
        # Fix blockquotes
        text = re.sub(r'\n>', r'\n\n>', text)
        text = re.sub(r'>\n([^>\n])', r'>\n\n\1', text)
        
        # Remove excessive blank lines
        text = re.sub(r'\n{4,}', r'\n\n\n', text)
        
        # Ensure emoji headers have spacing
        text = re.sub(r'(\*\*[📌🎯📚💡❓🔑⚡🎬].+?\*\*)', r'\n\1\n', text)
        
        # Fix horizontal rules
        text = re.sub(r'\n---\n', r'\n\n---\n\n', text)
        
        return text.strip()

    def _finalize_video_summary(self, summary: str, target_language: str, video_title: Optional[str]) -> str:
        """
        Add final touches to video summary
        """
        if video_title:
            # Add video title as main header with emoji
            final = f"# 🎬 {video_title}\n\n"
            final += summary
        else:
            final = summary
        
        # Add footer with language note if not English
        if target_language != "en":
            language_name = self.language_map.get(target_language, target_language)
            final += f"\n\n---\n\n*Summary generated in {language_name}*"
        
        return final

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (roughly 4 chars per token)
        """
        return len(text) // 4

    def _smart_chunk_text(self, text: str, max_chars_per_chunk: int = 3500) -> list:
        """
        Intelligently chunk text by paragraph then sentence boundaries
        """
        chunks = []
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        current_chunk = ""
        
        for paragraph in paragraphs:
            candidate = (current_chunk + "\n\n" + paragraph).strip() if current_chunk else paragraph
            
            if len(candidate) > max_chars_per_chunk and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                current_chunk = candidate
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Second pass: split oversized chunks by sentences
        import re
        final_chunks = []
        
        for chunk in chunks:
            if len(chunk) <= max_chars_per_chunk:
                final_chunks.append(chunk)
            else:
                # Split by sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                sub_chunk = ""
                
                for sentence in sentences:
                    candidate = (sub_chunk + " " + sentence).strip() if sub_chunk else sentence
                    
                    if len(candidate) > max_chars_per_chunk and sub_chunk:
                        final_chunks.append(sub_chunk.strip())
                        sub_chunk = sentence
                    else:
                        sub_chunk = candidate
                
                if sub_chunk.strip():
                    final_chunks.append(sub_chunk.strip())
        
        return final_chunks

    # Keep existing summarize method for backward compatibility
    async def summarize(self, transcript: str, target_language: str = "en", video_title: str = None) -> str:
        """
        Standard summarize method (backward compatible)
        """
        return await self.summarize_enhanced(
            transcript=transcript,
            target_language=target_language,
            video_title=video_title
        )