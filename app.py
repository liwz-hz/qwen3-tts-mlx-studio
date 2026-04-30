import os
import warnings

# Suppress known-harmless upstream warnings:
# - "model of type qwen3_tts to instantiate a model of type ." (unregistered model type)
# - "incorrect regex pattern ... fix_mistral_regex=True" (Qwen2Tokenizer regex issue)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# - "Trying to convert audio automatically from float32 to 16-bit int format" (Gradio)
warnings.filterwarnings("ignore", message="Trying to convert audio")

import argparse
import concurrent.futures
import shutil
import sys
import time
from datetime import datetime

import gradio as gr
import numpy as np
import soundfile as sf

from audio_utils import concatenate_audio, export_audio, split_text, unload_deepfilter
from config import (
    DEFAULT_AUTOSAVE,
    DEFAULT_BATCH_SPLIT_MODE,
    DEFAULT_DENOISE_REF,
    DEFAULT_EXPORT_FORMAT,
    DEFAULT_LOUDNORM,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_SIZE,
    DEFAULT_MP3_BITRATE,
    DEFAULT_QUANTIZATION,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SCRIPT_SILENCE_MS,
    DEFAULT_SILENCE_GAP_MS,
    DEFAULT_SPEAKERS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_TRIM_SILENCE,
    DEFAULT_WHISPER_MODEL,
    ENABLE_JIT_COMPILE,
    HISTORY_DIR,
    LANGUAGES,
    MAX_BATCH_SEGMENTS,
    MAX_SCRIPT_SPEAKERS,
    OUTPUT_DIR,
    SERVER_HOST,
    SERVER_PORT,
    VOICE_LIBRARY_DIR,
    WHISPER_MODEL_VARIANTS,
    WHISPER_MODEL_INFO,
    YT_CACHE_DIR,
)
from engine import TTSEngine
from error_handler import (
    UserInputError,
    ResourceError,
    SystemBusyError,
    GenerationTimeoutError,
    ERROR_MESSAGES,
    with_error_popup,
    with_error_popup_yield,
)
from history import GenerationHistory
from script_parser import parse_script, group_by_model_type
from subtitle_utils import (
    generate_srt_content,
    generate_vtt_content,
    save_subtitle_file,
    merge_short_segments,
    split_long_segments,
)
from theme import build_theme, custom_css
from voice_library import VoiceLibrary
from yt_voice import get_yt_extractor

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Qwen3-TTS MLX Studio")
parser.add_argument("--host", default=SERVER_HOST, help="Server host")
parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
parser.add_argument(
    "--model-size", choices=["0.6B", "1.7B"], default=DEFAULT_MODEL_SIZE
)
parser.add_argument(
    "--quant", choices=["4bit", "6bit", "8bit", "bf16"], default=DEFAULT_QUANTIZATION
)
parser.add_argument(
    "--share", action="store_true", help="Create public Gradio link"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def check_startup():
    warnings = []
    if sys.version_info < (3, 10):
        warnings.append("Python 3.10+ required")
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        warnings.append("mlx-audio not installed — run: pip install mlx-audio")
    if not shutil.which("ffmpeg"):
        warnings.append("ffmpeg not found — run: brew install ffmpeg")
    try:
        # Skip ffmpeg — already checked above
        yt_missing = [w for w in get_yt_extractor().check_dependencies()
                      if "ffmpeg" not in w]
        warnings.extend(yt_missing)
    except Exception as e:
        warnings.append(f"YT Voice Clone init error: {e}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    return warnings

startup_warnings = check_startup()

# ---------------------------------------------------------------------------
# Engine, library & history
# ---------------------------------------------------------------------------
engine = TTSEngine()
engine.model_size = args.model_size
engine.quantization = args.quant

library = VoiceLibrary()
history = GenerationHistory()
yt_extractor = get_yt_extractor()

# ---------------------------------------------------------------------------
# Logging setup - only warnings and errors to console, all levels to file
# ---------------------------------------------------------------------------
import logging
import sys

LOG_FILE = '/tmp/qwen3_tts_debug.log'
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
    ],
)
logger = logging.getLogger(__name__)
app_settings = {
    "temperature": DEFAULT_TEMPERATURE,
    "top_k": DEFAULT_TOP_K,
    "top_p": DEFAULT_TOP_P,
    "repetition_penalty": DEFAULT_REPETITION_PENALTY,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "timeout": DEFAULT_TIMEOUT,
    "output_dir": OUTPUT_DIR,
    "autosave": DEFAULT_AUTOSAVE,
    "export_format": DEFAULT_EXPORT_FORMAT,
    "mp3_bitrate": DEFAULT_MP3_BITRATE,
    "loudnorm": DEFAULT_LOUDNORM,
    "trim_silence": DEFAULT_TRIM_SILENCE,
    "denoise_ref": DEFAULT_DENOISE_REF,
    "default_language": "English",
}

# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------
def generate_with_timeout(func, *func_args, timeout_seconds=120, **func_kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *func_args, **func_kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise GenerationTimeoutError(ERROR_MESSAGES["timeout"])


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------
def _gen_kwargs():
    """Build kwargs dict for engine generate calls from current settings."""
    return {
        "temperature": app_settings["temperature"],
        "top_k": app_settings["top_k"],
        "top_p": app_settings["top_p"],
        "repetition_penalty": app_settings["repetition_penalty"],
        "max_tokens": app_settings["max_tokens"],
    }


def save_audio(audio_tuple, prefix="output"):
    """Save generated audio to the configured output directory."""
    if audio_tuple is None:
        return "No audio to save."
    sr, audio = audio_tuple
    out_dir = app_settings["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    fmt = app_settings["export_format"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{timestamp}.wav")
    final_path = export_audio(
        audio=audio,
        sr=sr,
        output_path=path,
        fmt=fmt,
        mp3_bitrate=app_settings["mp3_bitrate"],
        loudnorm=app_settings["loudnorm"],
        trim_silence=app_settings["trim_silence"],
    )
    if fmt != "wav" and final_path.endswith(".wav"):
        pass  # ffmpeg was not available or failed
    return f"Saved: {final_path}"


def _get_hf_cache_dir() -> str:
    """Return the ModelScope hub cache directory path."""
    from config import MODELSCOPE_CACHE_ROOT
    return MODELSCOPE_CACHE_ROOT


def _is_model_cached(repo_id: str) -> bool:
    """Check whether a ModelScope model is already in the local cache."""
    return os.path.isdir(repo_id) and bool(os.listdir(repo_id))


def _voice_choices():
    """Return list of saved voice names for dropdowns."""
    return ["None"] + [v["name"] for v in library.list_voices()]


def _voice_table():
    """Return voice data for the library dataframe."""
    voices = library.list_voices()
    if not voices:
        return [["(empty)", "", "", ""]]
    return [
        [v["name"], v.get("source", ""), v.get("language", ""), v.get("description", "")]
        for v in voices
    ]


# ---------------------------------------------------------------------------
# Generation handlers
# ---------------------------------------------------------------------------
@with_error_popup
def generate_custom_voice(text, speaker, language, instruct):
    if not text.strip():
        raise UserInputError(ERROR_MESSAGES["empty_text"])
    
    start = time.time()
    sr, audio = engine.generate_custom_voice(
        text, speaker, language, instruct,
        **_gen_kwargs(),
    )
    elapsed = time.time() - start
    result = (sr, audio)
    history.add(
        mode="custom_voice", text=text, language=language,
        audio=result, speaker=speaker,
        voice_params=instruct if instruct else "",
    )
    save_msg = ""
    if app_settings["autosave"]:
        save_msg = " | " + save_audio(result, "custom")
    status_text = f"✓ 完成 {elapsed:.1f}s | 模型: {engine.get_repo_id('custom_voice')}{save_msg}"
    return result, status_text


@with_error_popup
def generate_voice_design(text, language, instruct):
    if not text.strip():
        raise UserInputError(ERROR_MESSAGES["empty_text"])
    if not instruct.strip():
        raise UserInputError(ERROR_MESSAGES["empty_instruct"])
    
    start = time.time()
    sr, audio = engine.generate_voice_design(
        text, language, instruct,
        **_gen_kwargs(),
    )
    elapsed = time.time() - start
    result = (sr, audio)
    history.add(
        mode="voice_design", text=text, language=language,
        audio=result, voice_params=instruct,
    )
    save_msg = ""
    if app_settings["autosave"]:
        save_msg = " | " + save_audio(result, "design")
    status_text = f"✓ 完成 {elapsed:.1f}s | 模型: {engine.get_repo_id('voice_design')}{save_msg}"
    return result, status_text


@with_error_popup
def generate_voice_clone(text, ref_audio, ref_text, language, library_voice):
    if not text.strip():
        raise UserInputError(ERROR_MESSAGES["empty_text"])
    if library_voice and library_voice != "None":
        try:
            voice = library.load_voice(library_voice)
            ref_audio = library.get_ref_audio_path(library_voice)
            ref_text = voice["ref_text"]
        except FileNotFoundError:
            raise ResourceError(ERROR_MESSAGES["voice_not_found"].format(name=library_voice))
    if not ref_audio:
        raise UserInputError(ERROR_MESSAGES["empty_ref_audio"])
    if not ref_text or not ref_text.strip():
        raise UserInputError(ERROR_MESSAGES["empty_ref_text"])
    
    start = time.time()
    sr, audio = engine.generate_voice_clone(
        text, ref_audio, ref_text, language,
        denoise_ref=app_settings["denoise_ref"],
        **_gen_kwargs(),
    )
    elapsed = time.time() - start
    result = (sr, audio)
    voice_info = library_voice if library_voice and library_voice != "None" else "uploaded"
    history.add(
        mode="voice_clone", text=text, language=language,
        audio=result, voice_params=f"ref: {voice_info}",
    )
    save_msg = ""
    if app_settings["autosave"]:
        save_msg = " | " + save_audio(result, "clone")
    denoise_msg = " | 已降噪" if app_settings["denoise_ref"] else ""
    status_text = f"✓ 完成 {elapsed:.1f}s | 模型: {engine.get_repo_id('base')}{denoise_msg}{save_msg}"
    return result, status_text





# ---------------------------------------------------------------------------
# ASR transcription handlers
# ---------------------------------------------------------------------------
# ASR transcription handlers
# ---------------------------------------------------------------------------
@with_error_popup_yield
def transcribe_reference(ref_audio):
    """Transcribe reference audio and fill the transcript box."""
    if not ref_audio:
        raise UserInputError(ERROR_MESSAGES["empty_ref_audio"])
    yield gr.update(), "加载 ASR 模型中..."
    text = engine.transcribe(ref_audio, language="auto")
    if not text or not text.strip():
        raise UserInputError(ERROR_MESSAGES["asr_result_empty"])
    yield gr.update(value=text.strip()), f"✓ 已转录 ({len(text.split())} 词)"


@with_error_popup_yield
def transcribe_yt_clip(clip_audio):
    """Transcribe extracted YT clip audio."""
    if not clip_audio:
        raise UserInputError(ERROR_MESSAGES["asr_empty"])
    yield gr.update(), "加载 ASR 模型中..."
    text = engine.transcribe(clip_audio, language="auto")
    if not text or not text.strip():
        raise UserInputError(ERROR_MESSAGES["yt_transcribe_empty"])
    yield gr.update(value=text.strip()), f"✓ 已转录 ({len(text.split())} 词)"


@with_error_popup_yield
def transcribe_audio(audio_path, language):
    """Standalone transcription handler."""
    if not audio_path:
        raise UserInputError(ERROR_MESSAGES["asr_empty"])
    lang = "auto" if language == "Auto" else language
    yield gr.update(), "加载 ASR 模型中..."
    text = engine.transcribe(audio_path, language=lang)
    if not text or not text.strip():
        raise UserInputError(ERROR_MESSAGES["asr_result_empty"])
    yield gr.update(value=text.strip()), f"✓ 已转录 ({len(text.strip().split())} 词)"


def save_transcript(text):
    """Save transcription text to .txt file."""
    if not text or not text.strip():
        return "Nothing to save"
    out_dir = app_settings["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"transcript_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"Saved: {path}"


# ---------------------------------------------------------------------------
# Whisper subtitle generation handlers
# ---------------------------------------------------------------------------
@with_error_popup_yield
def generate_whisper_subtitles(audio_path, language, model_name, merge_short, split_long, progress=gr.Progress()):
    """Generate subtitles with timestamps using Whisper."""
    if not audio_path:
        raise UserInputError(ERROR_MESSAGES["asr_empty"])
    
    lang = "auto" if language == "Auto" else language.lower()
    
    progress(0.1, desc=f"加载 Whisper {model_name} 模型...")
    
    segments = engine.generate_subtitles(audio_path, language=lang, model_name=model_name)
    
    progress(0.8, desc="处理字幕片段...")
    
    if not segments:
        raise UserInputError("未生成字幕内容")
    
    if merge_short:
        segments = merge_short_segments(segments, min_duration=1.5)
    
    if split_long:
        segments = split_long_segments(segments, max_duration=7.0, max_chars=80)
    
    progress(1.0, desc="完成")
    
    text_only = "\n".join([seg["text"] for seg in segments])
    total_duration = max([seg["end"] for seg in segments]) if segments else 0
    
    summary = f"✓ 已生成 {len(segments)} 条字幕 | 总时长 {total_duration:.1f}s"
    
    yield (
        gr.update(value=text_only),
        segments,
        summary,
    )


@with_error_popup
def export_subtitles(segments, format_type, prefix):
    """Export subtitles to SRT or VTT file."""
    if not segments:
        raise UserInputError("没有字幕内容可导出")
    
    out_dir = app_settings["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}"
    path = os.path.join(out_dir, filename)
    
    actual_path = save_subtitle_file(segments, path, format=format_type)
    
    return f"✓ 已保存: {actual_path}"


@with_error_popup
def preview_subtitle_content(segments, format_type):
    """Generate preview of subtitle file content."""
    if not segments:
        return "(无字幕内容)"
    
    if format_type == "srt":
        return generate_srt_content(segments)
    elif format_type == "vtt":
        return generate_vtt_content(segments)
    else:
        return "(未知格式)"


@with_error_popup
def unload_whisper_model():
    """Unload Whisper model to free memory."""
    engine.unload_whisper()
    return "✓ Whisper 模型已卸载"


# ---------------------------------------------------------------------------
# Batch generation handlers
# ---------------------------------------------------------------------------
def _run_batch_custom_voice(text, speaker, language, instruct, split_mode, silence_ms, progress=gr.Progress()):
    segments = split_text(text, split_mode)
    if not segments:
        return None, [["(empty)", "", ""]], "No segments"
    if len(segments) > MAX_BATCH_SEGMENTS:
        return None, [["(error)", "", ""]], f"Too many segments (max {MAX_BATCH_SEGMENTS})"

    audio_parts = []
    table_rows = []
    succeeded, failed = 0, 0

    for i, seg in enumerate(progress.tqdm(segments, desc="Generating segments")):
        preview = seg[:50] + "..." if len(seg) > 50 else seg
        try:
            sr, audio = generate_with_timeout(
                engine.generate_custom_voice, seg, speaker, language, instruct,
                timeout_seconds=app_settings["timeout"],
                **_gen_kwargs(),
            )
            duration = len(audio) / sr
            audio_parts.append((sr, audio))
            table_rows.append([str(i + 1), preview, f"{duration:.1f}s"])
            succeeded += 1
        except Exception as e:
            table_rows.append([str(i + 1), preview, f"Failed: {e}"])
            failed += 1

    if not audio_parts:
        return None, table_rows, "All segments failed"

    combined = concatenate_audio(audio_parts, silence_ms=int(silence_ms))
    # Record combined to history
    history.add(
        mode="custom_voice", text=f"[Batch: {succeeded} segments]",
        language=language, audio=combined, speaker=speaker,
        voice_params=f"batch ({split_mode})",
    )
    status_msg = f"Generated {succeeded}/{len(segments)} segments"
    if failed:
        status_msg += f" ({failed} failed)"
    return gr.update(value=combined), table_rows, status_msg


def _run_batch_voice_design(text, language, instruct, split_mode, silence_ms, progress=gr.Progress()):
    if not instruct.strip():
        return None, [["(empty)", "", ""]], "Describe voice first"
    segments = split_text(text, split_mode)
    if not segments:
        return None, [["(empty)", "", ""]], "No segments"
    if len(segments) > MAX_BATCH_SEGMENTS:
        return None, [["(error)", "", ""]], f"Too many segments (max {MAX_BATCH_SEGMENTS})"

    audio_parts = []
    table_rows = []
    succeeded, failed = 0, 0

    for i, seg in enumerate(progress.tqdm(segments, desc="Generating segments")):
        preview = seg[:50] + "..." if len(seg) > 50 else seg
        try:
            sr, audio = generate_with_timeout(
                engine.generate_voice_design, seg, language, instruct,
                timeout_seconds=app_settings["timeout"],
                **_gen_kwargs(),
            )
            duration = len(audio) / sr
            audio_parts.append((sr, audio))
            table_rows.append([str(i + 1), preview, f"{duration:.1f}s"])
            succeeded += 1
        except Exception as e:
            table_rows.append([str(i + 1), preview, f"Failed: {e}"])
            failed += 1

    if not audio_parts:
        return None, table_rows, "All segments failed"

    combined = concatenate_audio(audio_parts, silence_ms=int(silence_ms))
    history.add(
        mode="voice_design", text=f"[Batch: {succeeded} segments]",
        language=language, audio=combined, voice_params=f"batch ({split_mode})",
    )
    status_msg = f"Generated {succeeded}/{len(segments)} segments"
    if failed:
        status_msg += f" ({failed} failed)"
    return gr.update(value=combined), table_rows, status_msg


def _run_batch_voice_clone(text, ref_audio, ref_text, language, library_voice, split_mode, silence_ms, progress=gr.Progress()):
    # Resolve library voice
    if library_voice and library_voice != "None":
        try:
            voice = library.load_voice(library_voice)
            ref_audio = library.get_ref_audio_path(library_voice)
            ref_text = voice["ref_text"]
        except FileNotFoundError:
            return None, [["(error)", "", ""]], "Voice not found"
    if not ref_audio:
        return None, [["(error)", "", ""]], "No reference audio"
    if not ref_text or not ref_text.strip():
        return None, [["(error)", "", ""]], "No transcript"

    segments = split_text(text, split_mode)
    if not segments:
        return None, [["(empty)", "", ""]], "No segments"
    if len(segments) > MAX_BATCH_SEGMENTS:
        return None, [["(error)", "", ""]], f"Too many segments (max {MAX_BATCH_SEGMENTS})"

    audio_parts = []
    table_rows = []
    succeeded, failed = 0, 0

    for i, seg in enumerate(progress.tqdm(segments, desc="Generating segments")):
        preview = seg[:50] + "..." if len(seg) > 50 else seg
        try:
            sr, audio = generate_with_timeout(
                engine.generate_voice_clone, seg, ref_audio, ref_text, language,
                denoise_ref=app_settings["denoise_ref"],
                timeout_seconds=app_settings["timeout"],
                **_gen_kwargs(),
            )
            duration = len(audio) / sr
            audio_parts.append((sr, audio))
            table_rows.append([str(i + 1), preview, f"{duration:.1f}s"])
            succeeded += 1
        except Exception as e:
            table_rows.append([str(i + 1), preview, f"Failed: {e}"])
            failed += 1

    if not audio_parts:
        return None, table_rows, "All segments failed"

    combined = concatenate_audio(audio_parts, silence_ms=int(silence_ms))
    voice_info = library_voice if library_voice and library_voice != "None" else "uploaded"
    history.add(
        mode="voice_clone", text=f"[Batch: {succeeded} segments]",
        language=language, audio=combined, voice_params=f"batch ref: {voice_info}",
    )
    status_msg = f"Generated {succeeded}/{len(segments)} segments"
    if failed:
        status_msg += f" ({failed} failed)"
    if app_settings["denoise_ref"]:
        status_msg += " | Noise reduction applied"
    return gr.update(value=combined), table_rows, status_msg


# ---------------------------------------------------------------------------
# Script mode handlers
# ---------------------------------------------------------------------------
def parse_script_handler(raw_text):
    """Parse script and return speaker info for voice assignment UI."""
    if not raw_text.strip():
        return gr.update(), "Enter a script first", {}

    parsed = parse_script(raw_text)

    if parsed.errors:
        return gr.update(), "; ".join(parsed.errors), {}

    speaker_info = {
        "speakers": parsed.speakers,
        "line_count": len(parsed.lines),
        "lines_per_speaker": {},
    }
    for line in parsed.lines:
        speaker_info["lines_per_speaker"].setdefault(line.speaker, 0)
        speaker_info["lines_per_speaker"][line.speaker] += 1

    summary_parts = [f"Found {len(parsed.speakers)} speakers, {len(parsed.lines)} lines:"]
    for spk in parsed.speakers:
        count = speaker_info["lines_per_speaker"].get(spk, 0)
        summary_parts.append(f"  {spk}: {count} lines")

    # Build visibility updates for speaker slots
    updates = []
    for i in range(MAX_SCRIPT_SPEAKERS):
        if i < len(parsed.speakers):
            updates.append(gr.update(visible=True, label=f"Speaker: {parsed.speakers[i]}"))
        else:
            updates.append(gr.update(visible=False))

    return updates, "\n".join(summary_parts), speaker_info


def generate_script_handler(raw_text, assignments_state, silence_ms, progress=gr.Progress()):
    """Generate audio for a parsed multi-speaker script.

    assignments_state is a dict mapping speaker name to voice config:
      {speaker: {"mode": ..., "speaker": ..., "language": ..., "instruct": ..., "library_voice": ...}}
    """
    if not raw_text.strip():
        return None, [["(empty)", "", "", ""]], "Enter script first"

    parsed = parse_script(raw_text)
    if parsed.errors:
        return None, [["(error)", "", "", ""]], parsed.errors[0]

    if not assignments_state:
        return None, [["(error)", "", "", ""]], "Parse script first"

    # Group lines by model type for efficient model swapping
    groups = group_by_model_type(parsed.lines, assignments_state)

    # Generate audio for each line, grouped by model type
    audio_by_line_number = {}  # line_number -> (sr, audio)
    table_rows = []
    succeeded, failed = 0, 0

    model_type_labels = {
        "custom_voice": "Custom Voice",
        "voice_design": "Voice Design",
        "base": "Voice Clone",
    }

    total_lines = len(parsed.lines)
    done = 0

    for model_type, lines in groups.items():
        label = model_type_labels.get(model_type, model_type)
        line_nums = [str(l.line_number) for l in lines]
        progress(done / total_lines, desc=f"Generating {label} lines ({', '.join(line_nums)})...")

        for line in lines:
            assignment = assignments_state.get(line.speaker, {})
            mode = assignment.get("mode", "custom_voice")
            lang = assignment.get("language", "English")

            try:
                if mode == "custom_voice":
                    sr, audio = generate_with_timeout(
                        engine.generate_custom_voice,
                        line.text,
                        assignment.get("speaker", DEFAULT_SPEAKERS[0]),
                        lang,
                        assignment.get("instruct", ""),
                        timeout_seconds=app_settings["timeout"],
                        **_gen_kwargs(),
                    )
                elif mode == "voice_design":
                    sr, audio = generate_with_timeout(
                        engine.generate_voice_design,
                        line.text,
                        lang,
                        assignment.get("instruct", ""),
                        timeout_seconds=app_settings["timeout"],
                        **_gen_kwargs(),
                    )
                elif mode == "voice_clone":
                    lib_voice = assignment.get("library_voice", "")
                    if not lib_voice or lib_voice == "None":
                        raise ValueError("No library voice selected for clone mode")
                    voice = library.load_voice(lib_voice)
                    ref_audio_path = library.get_ref_audio_path(lib_voice)
                    ref_text = voice["ref_text"]
                    sr, audio = generate_with_timeout(
                        engine.generate_voice_clone,
                        line.text, ref_audio_path, ref_text, lang,
                        denoise_ref=app_settings["denoise_ref"],
                        timeout_seconds=app_settings["timeout"],
                        **_gen_kwargs(),
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                audio_by_line_number[line.line_number] = (sr, audio)
                succeeded += 1
            except Exception as e:
                audio_by_line_number[line.line_number] = None
                failed += 1

            done += 1
            progress(done / total_lines)

    # Reassemble in script order and build results table
    audio_segments = []
    for line in parsed.lines:
        preview = line.text[:40] + "..." if len(line.text) > 40 else line.text
        result = audio_by_line_number.get(line.line_number)
        if result is not None:
            sr, audio = result
            duration = len(audio) / sr
            audio_segments.append((sr, audio))
            table_rows.append([str(line.line_number), line.speaker, preview, f"{duration:.1f}s"])
        else:
            table_rows.append([str(line.line_number), line.speaker, preview, "Failed"])

    if not audio_segments:
        return None, table_rows, "All lines failed"

    combined = concatenate_audio(audio_segments, silence_ms=int(silence_ms))
    # Record to history
    speakers_used = ", ".join(parsed.speakers[:4])
    if len(parsed.speakers) > 4:
        speakers_used += "..."
    history.add(
        mode="custom_voice",
        text=f"[Script: {succeeded} lines, speakers: {speakers_used}]",
        language="Multi", audio=combined,
        voice_params="script mode",
    )

    status_msg = f"Generated {succeeded}/{total_lines} lines"
    if failed:
        status_msg += f" ({failed} failed)"
    if app_settings["denoise_ref"]:
        status_msg += " | Noise reduction applied"
    return gr.update(value=combined), table_rows, status_msg


# ---------------------------------------------------------------------------
# History handlers
# ---------------------------------------------------------------------------
def history_preview(entry_id):
    """Load audio for a history entry."""
    if not entry_id or entry_id == "(empty)":
        return None
    audio = history.get_audio(entry_id)
    if audio is None:
        return None
    return audio


@with_error_popup
def history_delete(entry_id):
    """Delete a single history entry."""
    if not entry_id or entry_id == "(empty)":
        raise UserInputError(ERROR_MESSAGES["select_entry"])
    history.delete_entry(entry_id)
    return history.table_data(), f"✓ 已删除 {entry_id}"


def history_clear():
    """Clear all history."""
    history.clear()
    return history.table_data(), "✓ 历史记录已清空"


@with_error_popup
def history_save_audio(entry_id):
    """Save a history entry's audio to the output directory."""
    if not entry_id or entry_id == "(empty)":
        raise UserInputError(ERROR_MESSAGES["select_entry"])
    audio = history.get_audio(entry_id)
    if audio is None:
        raise ResourceError("音频未找到")
    return save_audio(audio, "history")


@with_error_popup
def history_regenerate(entry_id):
    """Get params from a history entry for regeneration."""
    if not entry_id or entry_id == "(empty)":
        raise UserInputError(ERROR_MESSAGES["select_entry"])
    entry = history.get_entry(entry_id)
    if entry is None:
        raise ResourceError("记录未找到")
    parts = [
        f"模式: {entry.mode.replace('_', ' ').title()}",
        f"语言: {entry.language}",
    ]
    if entry.speaker:
        parts.append(f"说话人: {entry.speaker}")
    if entry.voice_params:
        parts.append(f"参数: {entry.voice_params}")
    parts.append(f"文本: {entry.text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Save-to-library handlers
# ---------------------------------------------------------------------------
@with_error_popup
def save_design_to_library(audio_tuple, name, language, description, spoken_text):
    if audio_tuple is None:
        raise UserInputError(ERROR_MESSAGES["save_empty"])
    if not name.strip():
        raise UserInputError(ERROR_MESSAGES["import_empty_name"])
    sr, audio = audio_tuple
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_path = os.path.join(OUTPUT_DIR, f"_tmp_design_{int(time.time())}.wav")
    sf.write(tmp_path, audio, sr)
    library.save_voice(
        name=name,
        ref_audio_path=tmp_path,
        ref_text=spoken_text,
        language=language,
        description=description,
        source="design",
    )
    os.remove(tmp_path)
    return f"✓ 音色 '{name}' 已保存到库"


@with_error_popup
def save_clone_to_library(ref_audio, ref_text, name, language):
    if not ref_audio:
        raise UserInputError(ERROR_MESSAGES["import_empty_audio"])
    if not name.strip():
        raise UserInputError(ERROR_MESSAGES["import_empty_name"])
    if not ref_text or not ref_text.strip():
        raise UserInputError(ERROR_MESSAGES["import_empty_transcript"])
    library.save_voice(
        name=name,
        ref_audio_path=ref_audio,
        ref_text=ref_text,
        language=language,
        source="clone",
    )
    return f"✓ 音色 '{name}' 已保存到库"


@with_error_popup
def import_voice(audio_path, transcript, name, language):
    if not audio_path:
        raise UserInputError(ERROR_MESSAGES["import_empty_audio"])
    if not name.strip():
        raise UserInputError(ERROR_MESSAGES["import_empty_name"])
    if not transcript or not transcript.strip():
        raise UserInputError(ERROR_MESSAGES["import_empty_transcript"])
    library.save_voice(
        name=name.strip(),
        ref_audio_path=audio_path,
        ref_text=transcript.strip(),
        language=language,
        description="Imported voice",
        source="import",
    )
    return f"✓ 已导入 '{name.strip()}'"


# ---------------------------------------------------------------------------
# YT Voice Clone handlers
# ---------------------------------------------------------------------------
@with_error_popup
def fetch_yt_info(url):
    if not url or not url.strip():
        raise UserInputError(ERROR_MESSAGES["yt_url_empty"])
    info = yt_extractor.fetch_info(url.strip())
    dur = info.get("duration") or 0
    h, rem = divmod(int(dur), 3600)
    m, s = divmod(rem, 60)
    dur_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
    if info.get("has_manual_subs"):
        sub_note = "手动字幕"
    elif info.get("has_auto_subs"):
        sub_note = "自动字幕（可能需要编辑）"
    else:
        sub_note = "无字幕 — 需手动输入转录文本"
    md = (
        f"**{info['title']}**\n\n"
        f"时长: {dur_str}  |  {sub_note}"
        + (f"\n\n频道: {info['uploader']}" if info.get("uploader") else "")
    )
    state = {
        "id": info["id"],
        "title": info["title"],
        "duration": dur,
        "language": info.get("language"),
    }
    return (
        gr.update(value=info.get("thumbnail") or None),
        md,
        state,
        f"✓ 获取成功: {info['title'][:60]}",
    )


@with_error_popup
def extract_yt_clip(url, start_str, end_str, video_state, progress=gr.Progress()):
    if not url or not url.strip():
        raise UserInputError(ERROR_MESSAGES["yt_url_empty"])
    if not video_state or not video_state.get("id"):
        raise UserInputError(ERROR_MESSAGES["yt_fetch_first"])

    vid_dur = video_state.get("duration")

    start_sec = yt_extractor.parse_timestamp(start_str or "0")

    if not end_str or not end_str.strip():
        if not vid_dur:
            raise UserInputError("请输入结束时间")
        end_sec = float(vid_dur)
    else:
        end_sec = yt_extractor.parse_timestamp(end_str)

    if end_sec <= start_sec:
        raise UserInputError(ERROR_MESSAGES["yt_end_before_start"])

    clip_dur = end_sec - start_sec
    if clip_dur < 3.0:
        raise UserInputError(ERROR_MESSAGES["yt_clip_short"])
    if clip_dur > 60.0:
        end_sec = start_sec + 60.0
        clip_dur = 60.0

    if vid_dur and end_sec > vid_dur + 1:
        raise UserInputError(ERROR_MESSAGES["yt_time_exceeds"])

    video_id = video_state["id"]

    progress(0.0, desc="开始下载...")
    wav_path, subs_ok = yt_extractor.download_clip(
        url.strip(), video_id, start_sec, end_sec,
        progress_cb=lambda f, d: progress(f, desc=d),
    )
    progress(0.90, desc="提取转录文本...")
    transcript = yt_extractor.extract_transcript(video_id, start_sec, end_sec)
    progress(1.0, desc="完成")

    note = "转录文本已自动填充" if transcript else "无字幕 — 请手动输入转录文本"
    return (
        gr.update(value=wav_path),
        gr.update(value=transcript),
        f"✓ {clip_dur:.1f}s 片段已提取 — {note}",
    )


@with_error_popup
def clone_yt_voice(text, ref_audio, transcript, language, voice_name):
    errors = []
    if not text or not text.strip():
        errors.append("合成文本")
    if not ref_audio:
        errors.append("参考片段（先提取）")
    if not transcript or not transcript.strip():
        errors.append("参考转录文本")
    if not voice_name or not voice_name.strip():
        errors.append("音色名称")
    if errors:
        raise UserInputError(ERROR_MESSAGES["clone_missing_fields"].format(fields=", ".join(errors)))

    t0 = time.time()
    result = generate_with_timeout(
        engine.generate_voice_clone,
        text.strip(), ref_audio, transcript.strip(), language,
        denoise_ref=app_settings["denoise_ref"],
        timeout_seconds=app_settings["timeout"],
        **_gen_kwargs(),
    )
    elapsed = time.time() - t0
    lib_msg = save_clone_to_library(
        ref_audio, transcript.strip(), voice_name.strip(), language
    )
    history.add(
        mode="voice_clone",
        text=text.strip(),
        language=language,
        audio=result,
        voice_params=f"yt: {voice_name.strip()}",
    )
    extra = ""
    if app_settings["autosave"]:
        extra = " | " + save_audio(result, "yt_clone")
    denoise_msg = " | 已降噪" if app_settings["denoise_ref"] else ""
    status_msg = f"✓ 完成 {elapsed:.1f}s | {lib_msg}{denoise_msg}{extra}"
    return (
        gr.update(value=result),
        status_msg,
        f"✓ {lib_msg}",
        gr.update(choices=_voice_choices()),
        gr.update(value=_voice_table()),
    )


def clear_yt_cache():
    n = yt_extractor.clear_cache()
    return f"YT cache cleared — {n} entr{'y' if n == 1 else 'ies'} removed"


# ---------------------------------------------------------------------------
# Library management handlers
# ---------------------------------------------------------------------------
def preview_voice(voice_name):
    if not voice_name or voice_name == "(empty)":
        return None
    try:
        path = library.get_ref_audio_path(voice_name)
        if os.path.isfile(path):
            return path
    except Exception:
        pass
    return None


@with_error_popup
def delete_voice(voice_name):
    if not voice_name or voice_name == "(empty)":
        raise UserInputError(ERROR_MESSAGES["select_voice"])
    if not library.delete_voice(voice_name):
        raise ResourceError(ERROR_MESSAGES["voice_not_found"].format(name=voice_name))
    return _voice_table(), f"✓ 已删除 '{voice_name}'"


@with_error_popup
def rename_voice(old_name, new_name):
    if not old_name or old_name == "(empty)":
        raise UserInputError(ERROR_MESSAGES["select_voice"])
    if not new_name.strip():
        raise UserInputError(ERROR_MESSAGES["rename_empty"])
    ok = library.rename_voice(old_name, new_name.strip())
    if not ok:
        raise ResourceError("重命名失败（名称可能已存在）")
    return _voice_table(), f"✓ 已重命名 '{old_name}' → '{new_name.strip()}'"


# ---------------------------------------------------------------------------
# Settings handlers
# ---------------------------------------------------------------------------
def apply_settings(
    model_size, quantization,
    temperature, top_k, top_p, repetition_penalty, max_tokens, timeout,
    output_dir, autosave, jit_compile, default_language,
    export_format, mp3_bitrate, loudnorm, trim_silence, denoise_ref,
):
    model_changed = (
        model_size != engine.model_size
        or quantization != engine.quantization
        or jit_compile != engine.jit_compile
    )
    engine.model_size = model_size
    engine.quantization = quantization
    engine.jit_compile = jit_compile
    if model_changed:
        engine.unload_model()

    app_settings["temperature"] = temperature
    app_settings["top_k"] = int(top_k)
    app_settings["top_p"] = top_p
    app_settings["repetition_penalty"] = repetition_penalty
    app_settings["max_tokens"] = int(max_tokens)
    app_settings["timeout"] = int(timeout)
    app_settings["output_dir"] = output_dir.strip() or OUTPUT_DIR
    app_settings["autosave"] = autosave
    app_settings["export_format"] = export_format
    app_settings["mp3_bitrate"] = int(mp3_bitrate)
    app_settings["loudnorm"] = loudnorm
    app_settings["trim_silence"] = trim_silence
    app_settings["denoise_ref"] = denoise_ref
    if not denoise_ref:
        unload_deepfilter()
    app_settings["default_language"] = default_language

    os.makedirs(app_settings["output_dir"], exist_ok=True)

    parts = [f"size: {model_size}, quant: {quantization}"]
    if model_changed:
        parts.append("model unloaded")
    msg = f"Settings applied — {', '.join(parts)}."
    lang_update = gr.update(value=default_language)
    return msg, msg, lang_update, lang_update, lang_update, lang_update, lang_update, lang_update


GENERATION_PRESETS = {
    "Balanced": (0.9, 50, 1.0, 1.05, 4096, 120),
    "Creative": (1.2, 80, 0.95, 1.0, 4096, 120),
    "Precise": (0.3, 20, 0.9, 1.1, 4096, 120),
}


def apply_preset(preset_name):
    if preset_name == "Custom" or preset_name not in GENERATION_PRESETS:
        return [gr.update()] * 6
    temp, top_k, top_p, rep_pen, max_tok, timeout = GENERATION_PRESETS[preset_name]
    return (
        gr.update(value=temp),
        gr.update(value=top_k),
        gr.update(value=top_p),
        gr.update(value=rep_pen),
        gr.update(value=max_tok),
        gr.update(value=timeout),
    )


def reset_generation_defaults():
    return (
        gr.update(value=DEFAULT_TEMPERATURE),
        gr.update(value=DEFAULT_TOP_K),
        gr.update(value=DEFAULT_TOP_P),
        gr.update(value=DEFAULT_REPETITION_PENALTY),
        gr.update(value=DEFAULT_MAX_TOKENS),
        gr.update(value=DEFAULT_TIMEOUT),
        gr.update(value="Balanced"),
    )


def unload_model():
    engine.unload_model()
    return "Model unloaded. RAM freed.", "Not loaded (loads on demand)"


def unload_asr_setting():
    engine.unload_asr()
    return "ASR unloaded. RAM freed."


def delete_cached_models():
    """Delete all Qwen3-TTS model files from the ModelScope cache."""
    engine.unload_model()
    from config import MODELSCOPE_CACHE_ROOT
    ms_cache = MODELSCOPE_CACHE_ROOT
    if not os.path.isdir(ms_cache):
        return "ModelScope cache directory not found", "No model loaded"
    deleted = []
    failed = []
    for entry in os.listdir(ms_cache):
        if entry.startswith("Qwen3-TTS-12Hz-"):
            path = os.path.join(ms_cache, entry)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    deleted.append(entry)
                except OSError as e:
                    failed.append(f"{entry}: {e}")
    parts = []
    if deleted:
        parts.append(f"Deleted {len(deleted)} model(s): {', '.join(deleted)}")
    if failed:
        parts.append(f"Failed to delete {len(failed)}: {'; '.join(failed)}")
    if parts:
        return " | ".join(parts), "No model loaded"
    return "No Qwen3-TTS models found in cache", "No model loaded"


def get_model_status():
    if engine.current_model is None:
        return "No model loaded"
    repo = engine.get_repo_id(engine.current_model_type)
    return f"Loaded: {repo}"


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Qwen3-TTS MLX Studio") as app:

    gr.HTML(
        "<div class='app-header'>"
        "<h1>Qwen3-TTS MLX Studio</h1>"
        "<p class='subtitle'>Local AI Text-to-Speech &middot; MLX &middot; Apple Silicon</p>"
        "</div>"
    )

    with gr.Tabs():
        # =================================================================
        # Tab 1: Custom Voice
        # =================================================================
        with gr.Tab("Custom Voice"):
            with gr.Row():
                with gr.Column(scale=2):
                    cv_text = gr.Textbox(
                        label="Text",
                        lines=5,
                        placeholder="Enter text to speak...",
                    )
                    with gr.Row():
                        cv_speaker = gr.Dropdown(
                            choices=DEFAULT_SPEAKERS,
                            value=DEFAULT_SPEAKERS[0],
                            label="Speaker",
                        )
                        cv_language = gr.Dropdown(
                            choices=LANGUAGES,
                            value="English",
                            label="Language",
                        )
                    cv_instruct = gr.Textbox(
                        label="Style Instruction (optional)",
                        lines=1,
                        placeholder="e.g. Speak warmly, Sound excited...",
                    )
                    gr.Markdown(
                        "_Tip: 1–4 sentences work best. Very long text may hit the 120 s timeout._",
                        elem_classes=["text-hint"],
                    )
                    cv_generate = gr.Button("Generate", variant="primary")
                    # Batch Mode Accordion
                    with gr.Accordion("Batch Mode", open=False, elem_classes=["batch-accordion"]):
                        with gr.Row():
                            cv_batch_split = gr.Radio(
                                ["paragraph", "sentence", "line"],
                                value=DEFAULT_BATCH_SPLIT_MODE,
                                label="Split Mode",
                            )
                            cv_batch_silence = gr.Slider(
                                0, 2000, value=DEFAULT_SILENCE_GAP_MS, step=50,
                                label="Silence Gap (ms)",
                            )
                        cv_batch_generate = gr.Button("Generate Batch", variant="primary")
                        cv_batch_table = gr.Dataframe(
                            headers=["#", "Text", "Status"],
                            value=[["", "", ""]],
                            label="Batch Results",
                            interactive=False,
                        )
                        cv_batch_audio = gr.Audio(label="Combined Output", type="numpy", interactive=False)
                        with gr.Row():
                            cv_batch_save = gr.Button("Save Combined Audio")
                            cv_batch_status = gr.Textbox(label="Batch Status", interactive=False)
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    cv_audio = gr.Audio(label="Output", type="numpy", interactive=False)
                    cv_save = gr.Button("Save Audio")
                    cv_save_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Save path appears here…",
                        elem_classes=["save-status-text"],
                    )

        # =================================================================
        # Tab 2: Voice Design
        # =================================================================
        with gr.Tab("Voice Design"):
            with gr.Row():
                with gr.Column(scale=2):
                    vd_text = gr.Textbox(
                        label="Text",
                        lines=5,
                        placeholder="Enter text to speak...",
                    )
                    vd_instruct = gr.Textbox(
                        label="Voice Description",
                        lines=2,
                        placeholder="e.g. A deep, calm male narrator with a British accent",
                    )
                    vd_language = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="Language"
                    )
                    gr.Markdown(
                        "_Tip: 1–4 sentences work best. Very long text may hit the 120 s timeout._",
                        elem_classes=["text-hint"],
                    )
                    vd_generate = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Save to Voice Library", open=False, elem_classes=["lib-save-accordion"]):
                        with gr.Row():
                            vd_lib_name = gr.Textbox(
                                label="Voice Name", placeholder="my_narrator", scale=2
                            )
                            vd_lib_save = gr.Button("Save Voice to Library", scale=1)
                        vd_lib_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Library status…",
                            elem_classes=["save-status-text"],
                        )
                    # Batch Mode Accordion
                    with gr.Accordion("Batch Mode", open=False, elem_classes=["batch-accordion"]):
                        with gr.Row():
                            vd_batch_split = gr.Radio(
                                ["paragraph", "sentence", "line"],
                                value=DEFAULT_BATCH_SPLIT_MODE,
                                label="Split Mode",
                            )
                            vd_batch_silence = gr.Slider(
                                0, 2000, value=DEFAULT_SILENCE_GAP_MS, step=50,
                                label="Silence Gap (ms)",
                            )
                        vd_batch_generate = gr.Button("Generate Batch", variant="primary")
                        vd_batch_table = gr.Dataframe(
                            headers=["#", "Text", "Status"],
                            value=[["", "", ""]],
                            label="Batch Results",
                            interactive=False,
                        )
                        vd_batch_audio = gr.Audio(label="Combined Output", type="numpy", interactive=False)
                        with gr.Row():
                            vd_batch_save = gr.Button("Save Combined Audio")
                            vd_batch_status = gr.Textbox(label="Batch Status", interactive=False)
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    vd_audio = gr.Audio(label="Output", type="numpy", interactive=False)
                    vd_save = gr.Button("Save Audio")
                    vd_save_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Save path appears here…",
                        elem_classes=["save-status-text"],
                    )

        # =================================================================
        # Tab 3: Voice Cloning
        # =================================================================
        with gr.Tab("Voice Cloning"):
            gr.HTML(
                "<div class='info-notice'>"
                "<strong>Reference transcript must exactly match what is spoken in the audio.</strong> "
                "Use a clean 3–30 second clip for best results."
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        vc_language = gr.Dropdown(
                            choices=LANGUAGES, value="English", label="Language"
                        )
                        vc_library_voice = gr.Dropdown(
                            choices=_voice_choices(),
                            value="None",
                            label="Load from Library",
                        )
                    vc_ref_audio = gr.Audio(
                        label="Reference Audio",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    with gr.Row():
                        vc_transcribe_btn = gr.Button("Transcribe Reference", variant="secondary", scale=1)
                    gr.HTML("<div class='text-hint'>Auto-fills transcript using Qwen3-ASR-1.7B-8bit</div>")
                    vc_ref_text = gr.Textbox(
                        label="Reference Transcript (required)",
                        lines=2,
                        placeholder="Exact text spoken in reference audio",
                    )
                    vc_text = gr.Textbox(
                        label="Text to Speak",
                        lines=5,
                        placeholder="Enter text to speak in the cloned voice...",
                    )
                    vc_generate = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Save to Voice Library", open=False, elem_classes=["lib-save-accordion"]):
                        with gr.Row():
                            vc_lib_name = gr.Textbox(
                                label="Voice Name", placeholder="my_clone", scale=2
                            )
                            vc_lib_save = gr.Button("Save Voice to Library", scale=1)
                        vc_lib_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Library status…",
                            elem_classes=["save-status-text"],
                        )
                    # Batch Mode Accordion
                    with gr.Accordion("Batch Mode", open=False, elem_classes=["batch-accordion"]):
                        with gr.Row():
                            vc_batch_split = gr.Radio(
                                ["paragraph", "sentence", "line"],
                                value=DEFAULT_BATCH_SPLIT_MODE,
                                label="Split Mode",
                            )
                            vc_batch_silence = gr.Slider(
                                0, 2000, value=DEFAULT_SILENCE_GAP_MS, step=50,
                                label="Silence Gap (ms)",
                            )
                        vc_batch_generate = gr.Button("Generate Batch", variant="primary")
                        vc_batch_table = gr.Dataframe(
                            headers=["#", "Text", "Status"],
                            value=[["", "", ""]],
                            label="Batch Results",
                            interactive=False,
                        )
                        vc_batch_audio = gr.Audio(label="Combined Output", type="numpy", interactive=False)
                        with gr.Row():
                            vc_batch_save = gr.Button("Save Combined Audio")
                            vc_batch_status = gr.Textbox(label="Batch Status", interactive=False)
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    vc_audio = gr.Audio(label="Output", type="numpy", interactive=False)
                    vc_save = gr.Button("Save Audio")
                    vc_save_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Save path appears here…",
                        elem_classes=["save-status-text"],
                    )

        # =================================================================
        # Tab 4: YT Voice Clone
        # =================================================================
        with gr.Tab("YT Voice Clone"):
            gr.HTML(
                "<div class='info-notice'>"
                "Clone any voice from YouTube: fetch a video, pick a timestamp range, "
                "auto-extract the clip with aligned transcript, then generate."
                "</div>"
            )
            yt_video_state = gr.State({"id": None, "title": None, "duration": None})

            with gr.Row():
                # Left column — step-by-step controls
                with gr.Column(scale=3):
                    gr.Markdown("**Step 1 — Video URL**", elem_classes=["yt-step"])
                    yt_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        show_label=False,
                    )
                    yt_fetch_btn = gr.Button("Fetch Video Info", variant="secondary")

                    gr.Markdown("**Step 2 — Select Clip**", elem_classes=["yt-step"])
                    with gr.Row():
                        yt_start = gr.Textbox(
                            label="Start Time (mm:ss)", placeholder="blank = beginning", scale=1
                        )
                        yt_end = gr.Textbox(
                            label="End Time (mm:ss)", placeholder="blank = end of video", scale=1
                        )
                    yt_extract_btn = gr.Button("Extract Clip", variant="primary")

                    gr.Markdown("**Step 3 — Review Transcript**", elem_classes=["yt-step"])
                    yt_transcript = gr.Textbox(
                        label="Reference Transcript (auto-filled from subtitles — edit if needed)",
                        lines=3,
                        placeholder="Transcript appears here after extraction, or enter manually…",
                    )
                    yt_transcribe_btn = gr.Button("Transcribe Clip", variant="secondary")
                    gr.HTML("<div class='text-hint'>Use ASR when subtitles are unavailable or inaccurate</div>")

                    gr.Markdown("**Step 4 — Generate & Save**", elem_classes=["yt-step"])
                    yt_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter text to speak in the cloned voice…",
                    )
                    with gr.Row():
                        yt_language = gr.Dropdown(
                            choices=LANGUAGES, value="English", label="Language", scale=1
                        )
                        yt_voice_name = gr.Textbox(
                            label="Voice Name (saved to library)",
                            placeholder="yt_speaker",
                            scale=2,
                        )
                    yt_clone_btn = gr.Button("Clone & Save to Library", variant="primary")

                # Right column — preview + output panel
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    yt_thumbnail = gr.Image(
                        label="Video Thumbnail",
                        height=180,
                    )
                    yt_video_info = gr.Markdown(
                        "_Video info appears here after fetching._"
                    )
                    yt_clip_audio = gr.Audio(
                        label="Reference Clip Preview",
                        type="filepath",
                        interactive=False,
                    )
                    yt_audio_out = gr.Audio(
                        label="Generated Audio", type="numpy", interactive=False
                    )
                    yt_status = gr.Textbox(
                        label="Status", interactive=False, elem_classes=["save-status-text"]
                    )

        # =================================================================
        # Tab 5: Script Mode
        # =================================================================
        with gr.Tab("Script Mode") as script_tab:
            gr.HTML(
                "<div class='info-notice'>"
                "<strong>Multi-Speaker Script</strong> &nbsp;—&nbsp; "
                "Each line: <code>SPEAKER: Dialogue text</code> &nbsp; Lines without a label are narration."
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    sm_script = gr.Textbox(
                        label="Script",
                        lines=10,
                        placeholder=(
                            "NARRATOR: Once upon a time, there lived a curious inventor.\n"
                            "EMMA: Father, look what I found in the attic!\n"
                            "FATHER: That is something I built a long time ago."
                        ),
                        elem_classes=["script-editor"],
                    )
                    with gr.Row():
                        sm_parse_btn = gr.Button("Parse Script", variant="primary", scale=1)
                        sm_silence = gr.Slider(
                            0, 2000, value=DEFAULT_SCRIPT_SILENCE_MS, step=50,
                            label="Silence Between Lines (ms)", scale=2,
                        )
                    sm_parse_status = gr.Textbox(label="Parse Result", interactive=False, lines=2)

                    # Voice assignment state
                    sm_assignments = gr.State({})

                    gr.Markdown("### Voice Assignments")
                    # Pre-allocate speaker slots (show/hide based on parse)
                    sm_speaker_groups = []
                    sm_speaker_modes = []
                    sm_speaker_speakers = []
                    sm_speaker_instructs = []
                    sm_speaker_languages = []
                    sm_speaker_lib_voices = []

                    for i in range(MAX_SCRIPT_SPEAKERS):
                        with gr.Group(visible=False, elem_classes=[f"speaker-slot-{i}"]) as grp:
                            sm_speaker_groups.append(grp)
                            with gr.Row():
                                mode = gr.Radio(
                                    ["Custom Voice", "Voice Design", "Voice Clone"],
                                    value="Custom Voice",
                                    label=f"Speaker {i+1} Mode",
                                    scale=2,
                                )
                                sm_speaker_modes.append(mode)
                                lang = gr.Dropdown(
                                    choices=LANGUAGES, value="English",
                                    label="Language", scale=1,
                                )
                                sm_speaker_languages.append(lang)
                            with gr.Row():
                                spk = gr.Dropdown(
                                    choices=DEFAULT_SPEAKERS,
                                    value=DEFAULT_SPEAKERS[0],
                                    label="Speaker",
                                    scale=1,
                                )
                                sm_speaker_speakers.append(spk)
                                inst = gr.Textbox(
                                    label="Instruct / Description",
                                    placeholder="Style instruction or voice description",
                                    scale=2,
                                )
                                sm_speaker_instructs.append(inst)
                                lib_v = gr.Dropdown(
                                    choices=_voice_choices(),
                                    value="None",
                                    label="Library Voice",
                                    scale=1,
                                )
                                sm_speaker_lib_voices.append(lib_v)

                    with gr.Row():
                        sm_generate_btn = gr.Button("Generate Script", variant="primary")
                        sm_save_btn = gr.Button("Save Combined Audio")

                with gr.Column(scale=1):
                    sm_audio = gr.Audio(label="Combined Output", type="numpy", interactive=False)
                    sm_table = gr.Dataframe(
                        headers=["Line", "Speaker", "Text", "Status"],
                        value=[["", "", "", ""]],
                        label="Script Results",
                        interactive=False,
                    )
                    sm_status = gr.Textbox(label="Status", interactive=False)

        # =================================================================
        # Tab 6: Transcription
        # =================================================================
        with gr.Tab("Transcription"):
            gr.HTML(
                "<div class='info-notice'>"
                "Transcribe audio files locally using Qwen3-ASR. "
                "Supports up to ~20 minutes of audio."
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    asr_audio = gr.Audio(
                        label="Upload Audio",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    with gr.Row():
                        asr_language = gr.Dropdown(
                            choices=["Auto"] + LANGUAGES,
                            value="Auto",
                            label="Language",
                        )
                    asr_transcribe_btn = gr.Button("Transcribe", variant="primary")
                    asr_output = gr.Textbox(
                        label="Transcription",
                        lines=12,
                    )
                    with gr.Row():
                        asr_save_btn = gr.Button("Save as .txt")
                        asr_save_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Save path appears here...",
                            elem_classes=["save-status-text"],
                        )
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    asr_info = gr.Markdown(
                        "**Model:** Qwen3-ASR-1.7B-8bit\n\n"
                        "**Supported languages:** Auto-detect, English, Chinese, Japanese, Korean, "
                        "German, French, Russian, Portuguese, Spanish, Italian\n\n"
                        "**Max duration:** ~20 minutes per file"
                    )

        # =================================================================
        # Tab 7: Whisper Subtitles
        # =================================================================
        with gr.Tab("Whisper Subtitles"):
            gr.HTML(
                "<div class='info-notice'>"
                "使用 Whisper 模型生成带时间戳的字幕文件（SRT/VTT）。"
                "支持多种模型大小，自动语言检测。"
                "</div>"
            )
            
            whisper_segments_state = gr.State([])
            
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 输入音频")
                    whisper_audio = gr.Audio(
                        label="上传音频文件",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    
                    gr.Markdown("### Whisper 模型选择")
                    with gr.Row():
                        whisper_model = gr.Dropdown(
                            choices=list(WHISPER_MODEL_VARIANTS.keys()),
                            value=DEFAULT_WHISPER_MODEL,
                            label="模型",
                            scale=2,
                        )
                        whisper_language = gr.Dropdown(
                            choices=["Auto"] + LANGUAGES,
                            value="Auto",
                            label="语言",
                            scale=1,
                        )
                    
                    with gr.Accordion("模型信息", open=False):
                        whisper_model_info_text = gr.Markdown(
                            f"**当前模型:** {DEFAULT_WHISPER_MODEL}\n\n"
                            f"**大小:** {WHISPER_MODEL_INFO[DEFAULT_WHISPER_MODEL]['size']}\n\n"
                            f"**速度:** {WHISPER_MODEL_INFO[DEFAULT_WHISPER_MODEL]['speed']}\n\n"
                            f"**支持语言:** {WHISPER_MODEL_INFO[DEFAULT_WHISPER_MODEL]['languages']}+"
                        )
                    
                    gr.Markdown("### 字幕处理选项")
                    with gr.Row():
                        whisper_merge_short = gr.Checkbox(
                            value=True,
                            label="合并短片段",
                            info="将小于 1.5 秒的片段合并",
                        )
                        whisper_split_long = gr.Checkbox(
                            value=True,
                            label="拆分长片段",
                            info="将超过 7 秒或 80 字符的片段拆分",
                        )
                    
                    whisper_generate_btn = gr.Button("生成字幕", variant="primary")
                    whisper_unload_btn = gr.Button("卸载模型", variant="secondary")
                    
                    whisper_status = gr.Textbox(
                        label="状态",
                        interactive=False,
                        elem_classes=["save-status-text"],
                    )
                    
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    gr.Markdown("### 输出预览")
                    whisper_text_output = gr.Textbox(
                        label="纯文本预览",
                        lines=8,
                        interactive=False,
                    )
                    
                    gr.Markdown("### 字幕片段")
                    whisper_table = gr.Dataframe(
                        headers=["开始", "结束", "文本"],
                        value=[["", "", ""]],
                        label="字幕片段列表",
                        interactive=False,
                    )
            
            with gr.Accordion("导出字幕文件", open=True):
                with gr.Row():
                    whisper_format = gr.Radio(
                        choices=["srt", "vtt"],
                        value="srt",
                        label="格式",
                        scale=1,
                    )
                    whisper_prefix = gr.Textbox(
                        value="subtitles",
                        label="文件名前缀",
                        scale=2,
                    )
                whisper_preview_btn = gr.Button("预览字幕内容", variant="secondary")
                whisper_preview_content = gr.Textbox(
                    label="字幕文件预览",
                    lines=12,
                    interactive=False,
                )
                with gr.Row():
                    whisper_export_btn = gr.Button("导出字幕文件", variant="primary")
                    whisper_export_status = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        elem_classes=["save-status-text"],
                    )

        # =================================================================
        # Tab 8: Voice Library
        # =================================================================
        with gr.Tab("Voice Library"):
            with gr.Row():
                with gr.Column(scale=2):
                    lib_table = gr.Dataframe(
                        headers=["Name", "Source", "Language", "Description"],
                        value=_voice_table(),
                        label="Saved Voices",
                        interactive=False,
                    )
                    with gr.Row():
                        lib_selected = gr.Textbox(
                            label="Voice Name",
                            placeholder="Type or paste voice name",
                            scale=2,
                        )
                        lib_preview_btn = gr.Button("Preview", scale=1)
                        lib_delete_btn = gr.Button("Delete", scale=1)
                    with gr.Row():
                        lib_new_name = gr.Textbox(
                            label="Rename To", placeholder="new_name", scale=2
                        )
                        lib_rename_btn = gr.Button("Rename", scale=1)
                    lib_preview_audio = gr.Audio(label="Reference Audio Preview")
                    lib_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Status…",
                        elem_classes=["save-status-text"],
                    )
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    gr.Markdown("### Import Voice")
                    lib_import_audio = gr.Audio(
                        label="Audio File", type="filepath"
                    )
                    lib_import_transcript = gr.Textbox(
                        label="Transcript", lines=3
                    )
                    lib_import_name = gr.Textbox(
                        label="Name", placeholder="imported_voice"
                    )
                    lib_import_language = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="Language"
                    )
                    lib_import_btn = gr.Button("Import Voice", variant="primary")

        # =================================================================
        # Tab 9: History
        # =================================================================
        with gr.Tab("History"):
            hist_table = gr.Dataframe(
                headers=["ID", "Time", "Mode", "Text", "Duration"],
                value=history.table_data(),
                label="Generation History",
                interactive=False,
                elem_classes=["history-table"],
            )
            with gr.Row():
                hist_selected = gr.Textbox(
                    label="Entry ID",
                    placeholder="Paste entry ID to preview/manage",
                    scale=3,
                )
                hist_preview_btn = gr.Button("Preview", scale=1)
                hist_delete_btn = gr.Button("Delete Entry", scale=1)
            with gr.Row():
                hist_save_btn = gr.Button("Save Audio", scale=1)
                hist_regen_btn = gr.Button("Show Params", scale=1)
                hist_clear_btn = gr.Button("Clear All History", scale=1)
            hist_status = gr.Textbox(
                show_label=False, interactive=False,
                placeholder="Status…",
                elem_classes=["save-status-text"],
            )
            with gr.Row():
                hist_audio = gr.Audio(
                    label="Audio Preview", type="numpy", interactive=False, scale=1
                )
                hist_regen_info = gr.Textbox(
                    label="Regeneration Params", interactive=False, lines=3, scale=1
                )

        # =================================================================
        # Tab 10: Settings
        # =================================================================
        with gr.Tab("Settings"):
            with gr.Row():
                # --- Column 1: Model & Language ---
                with gr.Column(scale=1):
                    gr.Markdown("### Model")
                    set_size = gr.Radio(
                        ["0.6B", "1.7B"],
                        value=engine.model_size,
                        label="Model Size",
                    )
                    set_quant = gr.Radio(
                        ["4bit", "6bit", "8bit", "bf16"],
                        value=engine.quantization,
                        label="Quantization",
                    )
                    set_status = gr.Textbox(
                        label="Loaded Model",
                        value=get_model_status(),
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    set_unload = gr.Button("Unload Model / Free RAM")
                    gr.Markdown("### Reference Audio")
                    set_denoise_ref = gr.Checkbox(
                        value=DEFAULT_DENOISE_REF,
                        label="Denoise reference audio (DeepFilterNet, 8MB model)",
                        info="Pre-processes voice clone references to remove background noise",
                    )
                    gr.Markdown("### Language")
                    set_default_language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="English",
                        label="Default Language",
                    )
                    set_jit = gr.Checkbox(
                        value=ENABLE_JIT_COMPILE,
                        label="JIT compile model (faster after first run; unloads model on change)",
                        container=False,
                    )
                    with gr.Accordion("Model Cache & ASR", open=False, elem_classes=["settings-accordion"]):
                        gr.Markdown("### Model Cache")
                        gr.Textbox(
                            label="ModelScope Cache Directory",
                            value=os.path.abspath(_get_hf_cache_dir()),
                            interactive=False,
                            elem_classes=["model-status"],
                        )
                        set_delete_models = gr.Button(
                            "Delete Downloaded Models",
                            variant="stop",
                        )
                        set_delete_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Models will be re-downloaded on next use.",
                            elem_classes=["save-status-text"],
                        )
                        gr.Markdown("### Speech Recognition")
                        set_asr_status = gr.Textbox(
                            label="ASR Model",
                            value="Not loaded (loads on demand)",
                            interactive=False,
                            elem_classes=["model-status"],
                        )
                        set_asr_unload = gr.Button("Unload ASR Model")

                # --- Column 2: Generation ---
                with gr.Column(scale=1):
                    gr.Markdown("### Generation")
                    set_preset = gr.Radio(
                        ["Balanced", "Creative", "Precise", "Custom"],
                        value="Balanced",
                        label="Generation Presets",
                        info="Presets fill sliders below. Adjust freely afterward.",
                    )
                    set_temperature = gr.Slider(
                        0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.05,
                        label="Temperature",
                    )
                    set_top_k = gr.Slider(
                        0, 100, value=DEFAULT_TOP_K, step=1,
                        label="Top-K",
                    )
                    set_top_p = gr.Slider(
                        0.0, 1.0, value=DEFAULT_TOP_P, step=0.05,
                        label="Top-P",
                    )
                    set_rep_penalty = gr.Slider(
                        1.0, 2.0, value=DEFAULT_REPETITION_PENALTY, step=0.05,
                        label="Repetition Penalty",
                    )
                    set_max_tokens = gr.Slider(
                        512, 8192, value=DEFAULT_MAX_TOKENS, step=256,
                        label="Max Tokens",
                    )
                    set_timeout = gr.Slider(
                        30, 300, value=DEFAULT_TIMEOUT, step=10,
                        label="Generation Timeout (seconds)",
                    )
                    set_reset = gr.Button("Reset to Defaults")

                # --- Column 3: Output ---
                with gr.Column(scale=1):
                    gr.Markdown("### Output")
                    set_output_dir = gr.Textbox(
                        value=OUTPUT_DIR,
                        label="Output Directory",
                    )
                    set_autosave = gr.Checkbox(
                        value=DEFAULT_AUTOSAVE,
                        label="Auto-save generated audio",
                    )
                    gr.Markdown("### Export Format")
                    set_export_format = gr.Radio(
                        ["wav", "mp3", "ogg"],
                        value=DEFAULT_EXPORT_FORMAT,
                        label="Audio Format",
                    )
                    set_mp3_bitrate = gr.Slider(
                        64, 320, value=DEFAULT_MP3_BITRATE, step=32,
                        label="MP3 Bitrate (kbps)",
                        visible=False,
                    )
                    gr.Markdown("### Post-Processing")
                    set_loudnorm = gr.Checkbox(
                        value=DEFAULT_LOUDNORM,
                        label="EBU R128 loudness normalization",
                    )
                    set_trim_silence = gr.Checkbox(
                        value=DEFAULT_TRIM_SILENCE,
                        label="Trim leading/trailing silence",
                    )
                    with gr.Accordion("Storage & Cache", open=False, elem_classes=["settings-accordion"]):
                        gr.Markdown("### YT Cache")
                        set_yt_cache_btn = gr.Button("Clear YT Cache")
                        set_yt_cache_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder=f"Cache: {YT_CACHE_DIR}/",
                            elem_classes=["save-status-text"],
                        )
                        gr.Markdown("### Storage Paths")
                        gr.Textbox(
                            label="Voice Library",
                            value=os.path.abspath(VOICE_LIBRARY_DIR),
                            interactive=False,
                            elem_classes=["model-status"],
                        )
                        gr.Textbox(
                            label="History",
                            value=os.path.abspath(HISTORY_DIR),
                            interactive=False,
                            elem_classes=["model-status"],
                        )

            set_apply = gr.Button("Apply Settings", variant="primary")

    # Status bar
    status = gr.Textbox(
        show_label=False,
        interactive=False,
        elem_classes=["status-bar"],
        value="Ready" + (f" | Warnings: {'; '.join(startup_warnings)}" if startup_warnings else ""),
    )

    # ===================================================================
    # Event wiring
    # ===================================================================

    # --- Custom Voice ---
    cv_generate.click(
        fn=generate_custom_voice,
        inputs=[cv_text, cv_speaker, cv_language, cv_instruct],
        outputs=[cv_audio, status],
        show_progress="minimal",
    )
    cv_save.click(
        fn=lambda audio: save_audio(audio, "custom"),
        inputs=[cv_audio],
        outputs=[cv_save_status],
    )
    cv_batch_generate.click(
        fn=_run_batch_custom_voice,
        inputs=[cv_text, cv_speaker, cv_language, cv_instruct, cv_batch_split, cv_batch_silence],
        outputs=[cv_batch_audio, cv_batch_table, cv_batch_status],
        show_progress="full",
    )
    cv_batch_save.click(
        fn=lambda audio: save_audio(audio, "batch_custom"),
        inputs=[cv_batch_audio],
        outputs=[cv_batch_status],
    )

    # --- Voice Design ---
    vd_generate.click(
        fn=generate_voice_design,
        inputs=[vd_text, vd_language, vd_instruct],
        outputs=[vd_audio, status],
        show_progress="minimal",
    )
    vd_save.click(
        fn=lambda audio: save_audio(audio, "design"),
        inputs=[vd_audio],
        outputs=[vd_save_status],
    )
    vd_batch_generate.click(
        fn=_run_batch_voice_design,
        inputs=[vd_text, vd_language, vd_instruct, vd_batch_split, vd_batch_silence],
        outputs=[vd_batch_audio, vd_batch_table, vd_batch_status],
        show_progress="full",
    )
    vd_batch_save.click(
        fn=lambda audio: save_audio(audio, "batch_design"),
        inputs=[vd_batch_audio],
        outputs=[vd_batch_status],
    )

    def _save_design_and_refresh(*args):
        result = save_design_to_library(*args)
        return result, gr.update(choices=_voice_choices())

    vd_lib_save.click(
        fn=_save_design_and_refresh,
        inputs=[vd_audio, vd_lib_name, vd_language, vd_instruct, vd_text],
        outputs=[vd_lib_status, vc_library_voice],
    )

    # --- Voice Cloning ---
    vc_transcribe_btn.click(
        fn=transcribe_reference,
        inputs=[vc_ref_audio],
        outputs=[vc_ref_text, status],
        show_progress="minimal",
    )
    vc_generate.click(
        fn=generate_voice_clone,
        inputs=[vc_text, vc_ref_audio, vc_ref_text, vc_language, vc_library_voice],
        outputs=[vc_audio, status],
        show_progress="minimal",
    )
    vc_save.click(
        fn=lambda audio: save_audio(audio, "clone"),
        inputs=[vc_audio],
        outputs=[vc_save_status],
    )
    vc_batch_generate.click(
        fn=_run_batch_voice_clone,
        inputs=[vc_text, vc_ref_audio, vc_ref_text, vc_language, vc_library_voice,
                vc_batch_split, vc_batch_silence],
        outputs=[vc_batch_audio, vc_batch_table, vc_batch_status],
        show_progress="full",
    )
    vc_batch_save.click(
        fn=lambda audio: save_audio(audio, "batch_clone"),
        inputs=[vc_batch_audio],
        outputs=[vc_batch_status],
    )

    def _save_clone_and_refresh(*args):
        result = save_clone_to_library(*args)
        return result, gr.update(choices=_voice_choices())

    vc_lib_save.click(
        fn=_save_clone_and_refresh,
        inputs=[vc_ref_audio, vc_ref_text, vc_lib_name, vc_language],
        outputs=[vc_lib_status, vc_library_voice],
    )

    # Refresh library dropdown when clone tab is selected
    def refresh_clone_library():
        return gr.update(choices=_voice_choices(), value="None")

    vc_library_voice.focus(
        fn=refresh_clone_library,
        outputs=[vc_library_voice],
    )

    # --- YT Voice Clone ---
    yt_fetch_btn.click(
        fn=fetch_yt_info,
        inputs=[yt_url],
        outputs=[yt_thumbnail, yt_video_info, yt_video_state, yt_status],
        show_progress="minimal",
    )
    yt_extract_btn.click(
        fn=extract_yt_clip,
        inputs=[yt_url, yt_start, yt_end, yt_video_state],
        outputs=[yt_clip_audio, yt_transcript, yt_status],
        show_progress="full",
    )
    yt_transcribe_btn.click(
        fn=transcribe_yt_clip,
        inputs=[yt_clip_audio],
        outputs=[yt_transcript, yt_status],
        show_progress="minimal",
    )

    yt_clone_btn.click(
        fn=clone_yt_voice,
        inputs=[yt_text, yt_clip_audio, yt_transcript, yt_language, yt_voice_name],
        outputs=[yt_audio_out, status, yt_status, vc_library_voice, lib_table],
        show_progress="minimal",
    )

    # --- Transcription ---
    asr_transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[asr_audio, asr_language],
        outputs=[asr_output, status],
        show_progress="minimal",
    )
    asr_save_btn.click(
        fn=save_transcript,
        inputs=[asr_output],
        outputs=[asr_save_status],
    )

    # --- Whisper Subtitles ---
    def _update_whisper_table(segments):
        """Convert segments to table format."""
        if not segments:
            return [["", "", ""]]
        return [
            [f"{seg['start']:.2f}s", f"{seg['end']:.2f}s", seg['text']]
            for seg in segments
        ]
    
    def _update_whisper_model_info(model_name):
        """Update model info display."""
        info = WHISPER_MODEL_INFO.get(model_name, {})
        return f"**当前模型:** {model_name}\n\n" \
               f"**大小:** {info.get('size', 'N/A')}\n\n" \
               f"**速度:** {info.get('speed', 'N/A')}\n\n" \
               f"**支持语言:** {info.get('languages', 99)}+"
    
    whisper_model.change(
        fn=_update_whisper_model_info,
        inputs=[whisper_model],
        outputs=[whisper_model_info_text],
    )
    
    whisper_generate_btn.click(
        fn=generate_whisper_subtitles,
        inputs=[whisper_audio, whisper_language, whisper_model, 
                whisper_merge_short, whisper_split_long],
        outputs=[whisper_text_output, whisper_segments_state, whisper_status],
        show_progress="full",
    ).then(
        fn=_update_whisper_table,
        inputs=[whisper_segments_state],
        outputs=[whisper_table],
    )
    
    whisper_unload_btn.click(
        fn=unload_whisper_model,
        outputs=[whisper_status],
    )
    
    whisper_preview_btn.click(
        fn=preview_subtitle_content,
        inputs=[whisper_segments_state, whisper_format],
        outputs=[whisper_preview_content],
    )
    
    whisper_export_btn.click(
        fn=export_subtitles,
        inputs=[whisper_segments_state, whisper_format, whisper_prefix],
        outputs=[whisper_export_status],
    )

    # --- Script Mode ---
    def _parse_and_update_slots(raw_text):
        """Parse script and update speaker slot visibility."""
        if not raw_text.strip():
            updates = [gr.update(visible=False) for _ in range(MAX_SCRIPT_SPEAKERS)]
            return *updates, "Enter a script first", {}

        parsed = parse_script(raw_text)

        if parsed.errors:
            updates = [gr.update(visible=False) for _ in range(MAX_SCRIPT_SPEAKERS)]
            return *updates, "; ".join(parsed.errors), {}

        # Build summary
        summary_parts = [f"Found {len(parsed.speakers)} speakers, {len(parsed.lines)} lines:"]
        lines_per_speaker = {}
        for line in parsed.lines:
            lines_per_speaker.setdefault(line.speaker, 0)
            lines_per_speaker[line.speaker] += 1
        for spk in parsed.speakers:
            count = lines_per_speaker.get(spk, 0)
            summary_parts.append(f"  {spk}: {count} lines")

        # Build visibility updates for speaker slots
        updates = []
        for i in range(MAX_SCRIPT_SPEAKERS):
            if i < len(parsed.speakers):
                updates.append(gr.update(visible=True))
            else:
                updates.append(gr.update(visible=False))

        # Initial assignments state
        assignments = {}
        for spk in parsed.speakers:
            assignments[spk] = {
                "mode": "custom_voice",
                "speaker": DEFAULT_SPEAKERS[0],
                "language": "English",
                "instruct": "",
                "library_voice": "None",
            }

        return *updates, "\n".join(summary_parts), assignments

    sm_parse_btn.click(
        fn=_parse_and_update_slots,
        inputs=[sm_script],
        outputs=[*sm_speaker_groups, sm_parse_status, sm_assignments],
    )

    # Update assignments state when speaker slot controls change
    def _build_assignments_from_slots(current_assignments, script_text,
                                      *slot_values):
        """Rebuild assignments dict from all speaker slot values."""
        if not current_assignments or not script_text.strip():
            return current_assignments

        parsed = parse_script(script_text)
        if parsed.errors or not parsed.speakers:
            return current_assignments

        # slot_values: for each of MAX_SCRIPT_SPEAKERS slots:
        #   mode, speaker, instruct, language, library_voice
        values_per_slot = 5
        assignments = {}
        mode_map = {
            "Custom Voice": "custom_voice",
            "Voice Design": "voice_design",
            "Voice Clone": "voice_clone",
        }
        for i, spk in enumerate(parsed.speakers):
            if i >= MAX_SCRIPT_SPEAKERS:
                break
            base = i * values_per_slot
            mode_label = slot_values[base] if base < len(slot_values) else "Custom Voice"
            assignments[spk] = {
                "mode": mode_map.get(mode_label, "custom_voice"),
                "speaker": slot_values[base + 1] if base + 1 < len(slot_values) else DEFAULT_SPEAKERS[0],
                "instruct": slot_values[base + 2] if base + 2 < len(slot_values) else "",
                "language": slot_values[base + 3] if base + 3 < len(slot_values) else "English",
                "library_voice": slot_values[base + 4] if base + 4 < len(slot_values) else "None",
            }

        return assignments

    # Collect all slot control components in order
    all_slot_controls = []
    for i in range(MAX_SCRIPT_SPEAKERS):
        all_slot_controls.extend([
            sm_speaker_modes[i],
            sm_speaker_speakers[i],
            sm_speaker_instructs[i],
            sm_speaker_languages[i],
            sm_speaker_lib_voices[i],
        ])

    # Wire generate button
    def _generate_script_with_assignments(raw_text, assignments, silence_ms, *slot_values, progress=gr.Progress()):
        """Build fresh assignments from slot values, then generate."""
        # Rebuild assignments from current slot values
        fresh = _build_assignments_from_slots(assignments, raw_text, *slot_values)
        return generate_script_handler(raw_text, fresh, silence_ms, progress)

    sm_generate_btn.click(
        fn=_generate_script_with_assignments,
        inputs=[sm_script, sm_assignments, sm_silence, *all_slot_controls],
        outputs=[sm_audio, sm_table, sm_status],
        show_progress="full",
    )
    sm_save_btn.click(
        fn=lambda audio: save_audio(audio, "script"),
        inputs=[sm_audio],
        outputs=[sm_status],
    )

    # Refresh library voice dropdowns in script mode on focus
    def _refresh_single_lib_voice():
        return gr.update(choices=_voice_choices())

    for lib_voice_dropdown in sm_speaker_lib_voices:
        lib_voice_dropdown.focus(
            fn=_refresh_single_lib_voice,
            outputs=[lib_voice_dropdown],
        )

    # --- Voice Library ---
    lib_preview_btn.click(
        fn=preview_voice,
        inputs=[lib_selected],
        outputs=[lib_preview_audio],
    )
    lib_delete_btn.click(
        fn=delete_voice,
        inputs=[lib_selected],
        outputs=[lib_table, lib_status],
    )
    lib_rename_btn.click(
        fn=rename_voice,
        inputs=[lib_selected, lib_new_name],
        outputs=[lib_table, lib_status],
    )
    lib_import_btn.click(
        fn=import_voice,
        inputs=[lib_import_audio, lib_import_transcript, lib_import_name, lib_import_language],
        outputs=[lib_table, lib_status],
    )

    # --- History ---
    hist_preview_btn.click(
        fn=history_preview,
        inputs=[hist_selected],
        outputs=[hist_audio],
    )
    hist_delete_btn.click(
        fn=history_delete,
        inputs=[hist_selected],
        outputs=[hist_table, hist_status],
    )
    hist_clear_btn.click(
        fn=history_clear,
        outputs=[hist_table, hist_status],
    )
    hist_save_btn.click(
        fn=history_save_audio,
        inputs=[hist_selected],
        outputs=[hist_status],
    )
    hist_regen_btn.click(
        fn=history_regenerate,
        inputs=[hist_selected],
        outputs=[hist_regen_info],
    )

    # --- Settings ---
    set_export_format.change(
        fn=lambda fmt: gr.update(visible=(fmt == "mp3")),
        inputs=[set_export_format],
        outputs=[set_mp3_bitrate],
    )
    set_apply.click(
        fn=apply_settings,
        inputs=[
            set_size, set_quant,
            set_temperature, set_top_k, set_top_p, set_rep_penalty,
            set_max_tokens, set_timeout,
            set_output_dir, set_autosave, set_jit, set_default_language,
            set_export_format, set_mp3_bitrate, set_loudnorm, set_trim_silence,
            set_denoise_ref,
        ],
        outputs=[
            set_status, status,
            cv_language, vd_language, vc_language, yt_language, asr_language, lib_import_language,
        ],
    )
    set_preset.change(
        fn=apply_preset,
        inputs=[set_preset],
        outputs=[set_temperature, set_top_k, set_top_p, set_rep_penalty, set_max_tokens, set_timeout],
    )
    set_reset.click(
        fn=reset_generation_defaults,
        outputs=[set_temperature, set_top_k, set_top_p, set_rep_penalty, set_max_tokens, set_timeout, set_preset],
    )
    set_unload.click(
        fn=unload_model,
        outputs=[set_status, set_asr_status],
    )
    set_asr_unload.click(
        fn=unload_asr_setting,
        outputs=[set_asr_status],
    )
    set_delete_models.click(
        fn=delete_cached_models,
        outputs=[set_delete_status, set_status],
    )
    set_yt_cache_btn.click(fn=clear_yt_cache, outputs=[set_yt_cache_status])

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.queue(max_size=5).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        css=custom_css,
        theme=build_theme(),
    )
