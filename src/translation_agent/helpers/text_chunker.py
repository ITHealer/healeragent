import json
import tiktoken
from typing import List, Tuple, Dict
from src.translation_agent.helpers.json_utils import get_json_size

class SmartTextChunker:
    """
    Smart chunking kết hợp:
    1. Natural boundaries (paragraphs/sentences)
    2. Token-aware splitting
    3. Context preservation
    """
    
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def count_tokens(self, text: str) -> int:
        """Đếm tokens trong text"""
        return len(self.encoding.encode(text))
    
    def split_by_natural_boundaries(self, text: str) -> List[str]:
        """
        Split text theo ranh giới tự nhiên:
        1. Paragraphs (blank lines)
        2. Sentences (nếu paragraph quá dài)
        """
        import re
        
        # Split theo paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        segments = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Nếu paragraph < 500 tokens → giữ nguyên
            if self.count_tokens(para) < 500:
                segments.append(para)
            else:
                # Split theo sentences
                # Pattern: tách theo dấu câu + whitespace
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current = ""
                
                for sent in sentences:
                    if not sent.strip():
                        continue
                    
                    test_next = current + " " + sent if current else sent
                    
                    if self.count_tokens(test_next) < 500:
                        current = test_next
                    else:
                        if current:
                            segments.append(current.strip())
                        current = sent
                
                if current:
                    segments.append(current.strip())
        
        return segments if segments else [text]
    
    def chunk_with_context_preservation(
        self, 
        text: str, 
        max_tokens: int = 1000
    ) -> Tuple[List[str], int]:
        """
        Smart chunking với context preservation.
        
        Returns:
            - chunks: List text chunks
            - total_tokens: Tổng số tokens
        """
        # Count total tokens
        total_tokens = self.count_tokens(text)
        
        # Nếu text nhỏ → không cần chunk
        if total_tokens <= max_tokens:
            return [text], total_tokens
        
        # Step 1: Split theo natural boundaries
        segments = self.split_by_natural_boundaries(text)
        
        # Step 2: Group segments vào chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for segment in segments:
            seg_tokens = self.count_tokens(segment)
            
            # Nếu segment đơn lẻ đã quá max_tokens
            if seg_tokens > max_tokens:
                # Flush current chunk nếu có
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split segment theo lines
                lines = segment.split('\n')
                sub_chunk = []
                sub_tokens = 0
                
                for line in lines:
                    line_tokens = self.count_tokens(line)
                    
                    if sub_tokens + line_tokens > max_tokens and sub_chunk:
                        chunks.append('\n'.join(sub_chunk))
                        sub_chunk = [line]
                        sub_tokens = line_tokens
                    else:
                        sub_chunk.append(line)
                        sub_tokens += line_tokens
                
                if sub_chunk:
                    chunks.append('\n'.join(sub_chunk))
            
            # Nếu thêm segment vào chunk hiện tại vẫn OK
            elif current_tokens + seg_tokens <= max_tokens:
                current_chunk.append(segment)
                current_tokens += seg_tokens
            
            # Nếu thêm vào sẽ vượt quá → flush chunk hiện tại
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [segment]
                current_tokens = seg_tokens
        
        # Flush last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks, total_tokens