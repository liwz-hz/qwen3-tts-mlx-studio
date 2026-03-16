"""Audio utility functions for batch and script generation."""

import gc
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from config import DEFAULT_SAMPLE_RATE, DEEPFILTER_REPO


_deepfilter_model = None  # lazy singleton


def _get_deepfilter_model():
    """Load DeepFilterNet on first use (8MB download, cached after)."""
    global _deepfilter_model
    if _deepfilter_model is None:
        from mlx_audio.sts.models.deepfilternet import DeepFilterNetModel
        _deepfilter_model = DeepFilterNetModel.from_pretrained(DEEPFILTER_REPO)
    return _deepfilter_model


def denoise_ref_audio(input_path: str) -> str:
    """Denoise a reference audio file via DeepFilterNet.

    Loads audio, resamples to 48kHz, enhances, resamples back,
    writes to a temp file. Returns path to denoised temp file.
    Caller is responsible for cleanup.
    """
    model = _get_deepfilter_model()
    audio, orig_sr = sf.read(input_path, dtype="float32")
    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Upsample to 48kHz (DeepFilterNet requirement)
    target_sr = 48000
    if orig_sr != target_sr:
        ratio = target_sr / orig_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    # Enhance
    enhanced = model.enhance_array(audio)
    # Downsample back
    if orig_sr != target_sr:
        ratio = orig_sr / target_sr
        new_len = int(len(enhanced) * ratio)
        indices = np.linspace(0, len(enhanced) - 1, new_len)
        enhanced = np.interp(indices, np.arange(len(enhanced)), enhanced).astype(np.float32)
    # Write temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp_path, enhanced, orig_sr)
    return tmp_path


def unload_deepfilter():
    """Free DeepFilterNet model from memory."""
    global _deepfilter_model
    if _deepfilter_model is not None:
        del _deepfilter_model
        _deepfilter_model = None
        gc.collect()


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize a numpy audio array to prevent clipping."""
    if audio.size == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    return audio * (target_peak / peak)


def concatenate_audio(
    segments: list[tuple[int, np.ndarray]],
    silence_ms: int = 300,
) -> tuple[int, np.ndarray]:
    """Join multiple (sample_rate, audio) tuples with silence gaps.

    All segments are resampled to the first segment's sample rate if they differ,
    and peak-normalized before joining.
    """
    if not segments:
        raise ValueError("No audio segments to concatenate")
    if len(segments) == 1:
        sr, audio = segments[0]
        return (sr, normalize_audio(audio))

    sr = segments[0][0]
    silence_samples = int(sr * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    parts = []
    for i, (seg_sr, seg_audio) in enumerate(segments):
        normalized = normalize_audio(seg_audio)
        if seg_sr != sr:
            # Simple linear resampling for mismatched sample rates
            ratio = sr / seg_sr
            new_len = int(len(normalized) * ratio)
            indices = np.linspace(0, len(normalized) - 1, new_len)
            normalized = np.interp(indices, np.arange(len(normalized)), normalized).astype(np.float32)
        parts.append(normalized)
        if i < len(segments) - 1:
            parts.append(silence)

    return (sr, np.concatenate(parts))


def split_text(text: str, mode: str = "paragraph") -> list[str]:
    """Split text into segments by paragraph, sentence, or line.

    Returns non-empty stripped segments.
    """
    if mode == "paragraph":
        segments = re.split(r"\n\s*\n", text)
    elif mode == "sentence":
        # Split on sentence-ending punctuation followed by space or end
        segments = re.split(r"(?<=[.!?])\s+", text)
    elif mode == "line":
        segments = text.split("\n")
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    return [s.strip() for s in segments if s.strip()]


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def export_audio(
    audio: np.ndarray,
    sr: int,
    output_path: str,
    fmt: str = "wav",
    mp3_bitrate: int = 192,
    loudnorm: bool = False,
    trim_silence: bool = False,
) -> str:
    """Export audio to the specified format with optional post-processing.

    Writes a temp WAV, runs ffmpeg for conversion/processing, cleans up temp.
    Returns the final output path.  Falls back to plain WAV if ffmpeg is
    unavailable and a non-WAV format was requested.
    """
    base, _ = os.path.splitext(output_path)
    ext_map = {"wav": ".wav", "mp3": ".mp3", "ogg": ".ogg"}
    final_path = base + ext_map.get(fmt, ".wav")

    needs_ffmpeg = (fmt != "wav") or loudnorm or trim_silence

    if not needs_ffmpeg:
        sf.write(final_path, audio, sr)
        return final_path

    if not check_ffmpeg():
        fallback_path = base + ".wav"
        sf.write(fallback_path, audio, sr)
        return fallback_path

    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        sf.write(tmp_wav, audio, sr)

        filters = []
        if trim_silence:
            filters.append(
                "silenceremove=start_periods=1:start_threshold=-50dB:start_duration=0.01,"
                "areverse,"
                "silenceremove=start_periods=1:start_threshold=-50dB:start_duration=0.01,"
                "areverse"
            )
        if loudnorm:
            filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

        cmd = ["ffmpeg", "-y", "-i", tmp_wav]
        if filters:
            cmd += ["-af", ",".join(filters)]

        if fmt == "mp3":
            cmd += ["-codec:a", "libmp3lame", "-b:a", f"{mp3_bitrate}k"]
        elif fmt == "ogg":
            cmd += ["-codec:a", "libvorbis", "-q:a", "6"]

        cmd += ["-ar", str(sr), "-ac", "1", final_path]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg error: {result.stderr[:300]}")

        return final_path
    finally:
        if os.path.isfile(tmp_wav):
            os.remove(tmp_wav)
