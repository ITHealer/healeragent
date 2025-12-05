import json
import re
from typing import List, Dict, Tuple

def get_json_size(js: dict) -> int:
    """Tính size của JSON object khi encode UTF-8"""
    return len(json.dumps(js, ensure_ascii=False).encode('utf-8'))

def fix_json_string(json_string: str) -> str:
    """Fix common JSON formatting issues"""
    def repl(m: re.Match):
        return f"""{'"' if m.group(1) else ""},\n"{m.group(2)}":{'"' if m.group(3) else ""}"""
    
    fixed_json = re.sub(
        r"""([“”"])?\s*[，,]\s*["“”]\s*(\d+)\s*["“”]\s*[：:]\s*(["“”])?""",
        repl,
        json_string,
        re.MULTILINE
    )
    return fixed_json

def segments2json_chunks(
    segments: List[str], 
    chunk_size_max: int
) -> Tuple[Dict[str, str], List[Dict[str, str]], List[Tuple[int, int]]]:
    """
    Chuyển đổi list segments thành JSON chunks với size limit.
    Logic từ TXTTranslator - smart chunking với context preservation.
    
    Args:
        segments: List các text segments
        chunk_size_max: Max size (bytes) cho mỗi JSON chunk
    
    Returns:
        - indexed_segments: Dict đầy đủ {index: segment}
        - json_chunks: List các JSON chunks
        - merged_indices: List các (start, end) indices của segments đã merge
    """
    # Step 1: Xử lý segments quá lớn
    processed_segments = []
    merged_indices_list = []
    
    for segment in segments:
        # Kiểm tra nếu segment đơn lẻ đã quá chunk_size_max
        long_key_estimate = str(len(segments) + len(processed_segments))
        test_obj = {long_key_estimate: segment}
        
        if get_json_size(test_obj) > chunk_size_max:
            # Segment quá lớn → split theo lines
            sub_segments = []
            lines = segment.splitlines(keepends=True)
            current_sub = ""
            
            for line in lines:
                next_sub = current_sub + line
                test_next = {long_key_estimate: next_sub}
                
                if get_json_size(test_next) > chunk_size_max:
                    if current_sub:
                        sub_segments.append(current_sub)
                    # Dù line đơn lẻ quá lớn vẫn phải add
                    sub_segments.append(line)
                    current_sub = ""
                else:
                    current_sub = next_sub
            
            if current_sub:
                sub_segments.append(current_sub)
            
            # Handle empty segment
            if not sub_segments and segment == "":
                sub_segments.append("")
            
            # Track merged indices
            start_idx = len(processed_segments)
            processed_segments.extend(sub_segments)
            end_idx = len(processed_segments)
            
            if end_idx - start_idx > 1:
                merged_indices_list.append((start_idx, end_idx))
        else:
            processed_segments.append(segment)
    
    # Step 2: Group segments vào JSON chunks
    json_chunks_list = []
    
    if not processed_segments:
        return {}, [], []
    
    current_chunk = {}
    for idx, segment in enumerate(processed_segments):
        prospective_chunk = current_chunk.copy()
        prospective_chunk[str(idx)] = segment
        
        # Kiểm tra size
        if get_json_size(prospective_chunk) > chunk_size_max and current_chunk:
            # Flush current chunk
            json_chunks_list.append(current_chunk)
            current_chunk = {str(idx): segment}
        else:
            current_chunk = prospective_chunk
    
    # Add last chunk
    if current_chunk:
        json_chunks_list.append(current_chunk)
    
    # Step 3: Tạo indexed_segments đầy đủ
    indexed_segments = {str(i): seg for i, seg in enumerate(processed_segments)}
    
    return indexed_segments, json_chunks_list, merged_indices_list