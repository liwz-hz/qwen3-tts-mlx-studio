"""
Subtitle file generation utilities (SRT and VTT formats).
"""

from datetime import timedelta


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:02d},{milliseconds:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to VTT timestamp format: HH:MM:SS.mmm"""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:02d}.{milliseconds:03d}"


def generate_srt_content(segments: list[dict]) -> str:
    """
    Generate SRT file content from segments.
    
    Args:
        segments: List of dicts with 'start', 'end', 'text' keys
    
    Returns:
        SRT file content as string
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_ts = format_timestamp_srt(seg["start"])
        end_ts = format_timestamp_srt(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def generate_vtt_content(segments: list[dict]) -> str:
    """
    Generate VTT file content from segments.
    
    Args:
        segments: List of dicts with 'start', 'end', 'text' keys
    
    Returns:
        VTT file content as string
    """
    lines = ["WEBVTT", ""]
    for seg in segments:
        start_ts = format_timestamp_vtt(seg["start"])
        end_ts = format_timestamp_vtt(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def save_subtitle_file(segments: list[dict], output_path: str, format: str = "srt") -> str:
    """
    Save subtitle file to disk.
    
    Args:
        segments: List of dicts with 'start', 'end', 'text' keys
        output_path: Output file path (without extension)
        format: "srt" or "vtt"
    
    Returns:
        Actual file path saved
    """
    if format == "srt":
        content = generate_srt_content(segments)
        actual_path = output_path if output_path.endswith(".srt") else f"{output_path}.srt"
    elif format == "vtt":
        content = generate_vtt_content(segments)
        actual_path = output_path if output_path.endswith(".vtt") else f"{output_path}.vtt"
    else:
        raise ValueError(f"Unsupported subtitle format: {format}")
    
    with open(actual_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return actual_path


def estimate_subtitle_duration(text: str, words_per_minute: int = 150) -> float:
    """
    Estimate reading duration for subtitle text.
    
    Args:
        text: Subtitle text
        words_per_minute: Average reading speed
    
    Returns:
        Estimated duration in seconds
    """
    words = len(text.split())
    duration = (words / words_per_minute) * 60
    return max(duration, 1.0)  # Minimum 1 second


def merge_short_segments(segments: list[dict], min_duration: float = 1.0) -> list[dict]:
    """
    Merge consecutive short segments into longer ones.
    
    Args:
        segments: Original segments
        min_duration: Minimum duration per segment
    
    Returns:
        Merged segments
    """
    if not segments:
        return segments
    
    merged = []
    current = None
    
    for seg in segments:
        if current is None:
            current = seg.copy()
        else:
            duration = seg["end"] - current["start"]
            if duration < min_duration:
                current["end"] = seg["end"]
                current["text"] += " " + seg["text"]
            else:
                merged.append(current)
                current = seg.copy()
    
    if current:
        merged.append(current)
    
    return merged


def split_long_segments(segments: list[dict], max_duration: float = 7.0, max_chars: int = 80) -> list[dict]:
    """
    Split segments that are too long for comfortable reading.
    
    Args:
        segments: Original segments
        max_duration: Maximum duration per segment
        max_chars: Maximum characters per segment
    
    Returns:
        Split segments
    """
    result = []
    
    for seg in segments:
        duration = seg["end"] - seg["start"]
        text = seg["text"].strip()
        
        if duration <= max_duration and len(text) <= max_chars:
            result.append(seg)
        else:
            # Split by words
            words = text.split()
            if len(words) <= 1:
                result.append(seg)
                continue
            
            # Calculate split point
            num_splits = max(1, int(len(text) / max_chars))
            words_per_split = len(words) // (num_splits + 1)
            
            split_duration = duration / (num_splits + 1)
            
            for i in range(num_splits + 1):
                start_time = seg["start"] + i * split_duration
                end_time = seg["start"] + (i + 1) * split_duration
                
                start_word = i * words_per_split
                end_word = (i + 1) * words_per_split if i < num_splits else len(words)
                
                split_text = " ".join(words[start_word:end_word])
                
                if split_text:
                    result.append({
                        "start": start_time,
                        "end": end_time,
                        "text": split_text
                    })
    
    return result