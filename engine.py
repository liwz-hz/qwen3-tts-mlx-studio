import gc
import logging
import os
import threading

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
from mlx_audio.stt.utils import load_model as load_stt_model

from config import (
    MODELSCOPE_CACHE_ROOT,
    REPO_TEMPLATE,
    MODEL_VARIANTS,
    DEFAULT_MODEL_SIZE,
    DEFAULT_QUANTIZATION,
    ENABLE_JIT_COMPILE,
    ASR_REPO_ID,
    WHISPER_MODEL_VARIANTS,
    DEFAULT_WHISPER_MODEL,
)

logger = logging.getLogger(__name__)
LOCK_TIMEOUT = 10


def _find_local_model_path(repo_id: str) -> str:
    """
    Find local model path in ModelScope cache.
    ModelScope replaces '.' with '___' in directory names.
    Returns absolute path if found, None otherwise.
    """
    if not repo_id:
        return None
    
    model_name = repo_id.split("/")[-1]
    cache_dir = MODELSCOPE_CACHE_ROOT
    
    # Ensure cache directory exists
    if not os.path.isdir(cache_dir):
        logger.warning(f"ModelScope cache directory not found: {cache_dir}")
        return None
    
    # Try exact name first (symlink)
    exact_path = os.path.join(cache_dir, model_name)
    if os.path.isdir(exact_path) and os.listdir(exact_path):
        logger.info(f"Found model at exact path: {exact_path}")
        return os.path.abspath(exact_path)
    
    # Try ModelScope's dot-to-underscore conversion
    converted_name = model_name.replace(".", "___")
    converted_path = os.path.join(cache_dir, converted_name)
    if os.path.isdir(converted_path) and os.listdir(converted_path):
        logger.info(f"Found model at converted path: {converted_path}")
        return os.path.abspath(converted_path)
    
    # Try alternative naming patterns
    alt_patterns = [
        model_name.replace("-", "_"),
        model_name.replace("-", "___"),
        model_name.replace(".", "-"),
    ]
    for pattern in alt_patterns:
        alt_path = os.path.join(cache_dir, pattern)
        if os.path.isdir(alt_path) and os.listdir(alt_path):
            logger.info(f"Found model at alternative path: {alt_path}")
            return os.path.abspath(alt_path)
    
    logger.warning(f"Model not found in cache: {model_name}")
    return None


def _ensure_model_available(repo_id: str, model_type: str) -> str:
    """
    Ensure model is available locally. Returns local path or raises error with download instructions.
    """
    local_path = _find_local_model_path(repo_id)
    
    if local_path:
        # Verify path contains actual model files
        required_files = ["config.json", "model.safetensors"]
        has_model = any(
            f.endswith(".safetensors") for f in os.listdir(local_path)
        )
        if has_model:
            return local_path
        logger.warning(f"Model directory exists but may be incomplete: {local_path}")
    
    model_name = repo_id.split("/")[-1]
    raise FileNotFoundError(
        f"{'模型' if 'TTS' in model_name else 'ASR/Whisper 模型'}未找到: {model_name}\n\n"
        f"请先下载模型:\n"
        f"  modelscope download {repo_id}\n\n"
        f"或访问: https://modelscope.cn/models/{repo_id}\n\n"
        f"模型缓存目录: {MODELSCOPE_CACHE_ROOT}"
    )


class TTSEngine:
    """Manages model loading, unloading, and inference."""

    def __init__(self):
        self.current_model = None
        self.current_model_type = None
        self.model_size = DEFAULT_MODEL_SIZE
        self.quantization = DEFAULT_QUANTIZATION
        self.jit_compile = ENABLE_JIT_COMPILE
        self._model_loaded_with_jit = False
        self.asr_model = None
        self.whisper_model = None
        self.whisper_model_name = None
        self._lock = threading.Lock()

    def _acquire_lock(self):
        if not self._lock.acquire(timeout=LOCK_TIMEOUT):
            raise RuntimeError(
                "Engine is busy with a previous generation. Please wait and try again."
            )

    def get_repo_id(self, model_type: str) -> str:
        """Build local ModelScope model path from model_type + size + quant."""
        variant = MODEL_VARIANTS[model_type]
        repo_name = REPO_TEMPLATE.format(
            size=self.model_size, variant=variant, quant=self.quantization
        )
        repo_id = f"mlx-community/{repo_name}"
        
        return _ensure_model_available(repo_id, model_type)

    def _load_model(self, model_type: str):
        """Load model, unloading previous if different. Caller must hold lock."""
        self._unload_asr_unlocked()
        
        if self.current_model_type != model_type:
            self._unload_model_unlocked()
            repo_id = self.get_repo_id(model_type)
            logger.info(f"Loading TTS model: {repo_id}")
            self.current_model = load_model(repo_id)
            self.current_model_type = model_type
            self._model_loaded_with_jit = False
            mx.eval(self.current_model.parameters())

    def _apply_jit_compile(self):
        """Apply JIT compilation to model. Only compiles once per model load."""
        if self._model_loaded_with_jit or not self.jit_compile:
            return
        
        talker = getattr(self.current_model, "talker", None)
        if talker is None:
            logger.warning("No talker found in model, skipping JIT")
            return
        
        try:
            logger.info("Applying JIT compilation...")
            talker.__call__ = mx.compile(talker.__call__, shapeless=False)
            
            code_pred = getattr(talker, "code_predictor", None)
            if code_pred is not None:
                code_pred.__call__ = mx.compile(code_pred.__call__, shapeless=False)
            
            self._model_loaded_with_jit = True
            logger.info("JIT compilation applied successfully")
        except Exception as e:
            logger.warning(f"JIT compilation failed, continuing without JIT: {e}")
            self.jit_compile = False

    def is_model_loaded(self, model_type: str) -> bool:
        return self.current_model_type == model_type

    def unload_model(self):
        self._acquire_lock()
        try:
            self._unload_model_unlocked()
        finally:
            self._lock.release()

    def _unload_model_unlocked(self):
        self._unload_asr_unlocked()
        if self.current_model is not None:
            logger.info(f"Unloading model: {self.current_model_type}")
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            self._model_loaded_with_jit = False
            gc.collect()

    def generate_custom_voice(self, text, speaker, language, instruct="", **kwargs) -> tuple:
        self._acquire_lock()
        try:
            self._load_model("custom_voice")
            self._apply_jit_compile()
            
            results = list(
                self.current_model.generate_custom_voice(
                    text=text, speaker=speaker, language=language, instruct=instruct,
                    **kwargs,
                )
            )
            return self._to_numpy(results[0])
        finally:
            self._lock.release()

    def generate_voice_design(self, text, language, instruct, **kwargs) -> tuple:
        self._acquire_lock()
        try:
            self._load_model("voice_design")
            self._apply_jit_compile()
            
            results = list(
                self.current_model.generate_voice_design(
                    text=text, language=language, instruct=instruct,
                    **kwargs,
                )
            )
            return self._to_numpy(results[0])
        finally:
            self._lock.release()

    def generate_voice_clone(self, text, ref_audio_path, ref_text, language="English",
                             denoise_ref=False, **kwargs) -> tuple:
        self._acquire_lock()
        tmp_denoised = None
        try:
            logger.info(f"generate_voice_clone called: text='{text[:50]}...', ref_audio={ref_audio_path}, ref_text='{ref_text[:30]}...', language={language}, denoise={denoise_ref}")
            
            # Validate inputs
            if not text or not text.strip():
                logger.error("Empty text")
                raise ValueError("生成文本不能为空")
            if not ref_audio_path:
                logger.error("Empty ref_audio_path")
                raise ValueError("参考音频路径为空")
            if not isinstance(ref_audio_path, str):
                logger.error(f"ref_audio_path wrong type: {type(ref_audio_path)}")
                raise ValueError(f"参考音频路径格式错误: 期望字符串，实际 {type(ref_audio_path).__name__}")
            if not os.path.isfile(ref_audio_path):
                logger.error(f"Audio file not found: {ref_audio_path}")
                raise ValueError(f"参考音频文件不存在: {ref_audio_path}")
            if not ref_text or not ref_text.strip():
                logger.error("Empty ref_text")
                raise ValueError("参考转录文本不能为空")
            
            # Check audio file validity
            try:
                import soundfile as sf
                info = sf.info(ref_audio_path)
                logger.info(f"Audio file info: {info.duration:.1f}s, {info.samplerate}Hz, {info.channels} channels, {info.format}")
                if info.duration < 1.0:
                    logger.warning(f"Audio too short: {info.duration}s (recommended: 3-30s)")
                elif info.duration > 60.0:
                    logger.warning(f"Audio too long: {info.duration}s (recommended: 3-30s)")
            except Exception as e:
                logger.error(f"Failed to read audio file: {e}")
                raise ValueError(f"无法读取音频文件: {e}")
            
            if denoise_ref:
                logger.info("Applying denoising...")
                from audio_utils import denoise_ref_audio
                tmp_denoised = denoise_ref_audio(ref_audio_path)
                ref_audio_path = tmp_denoised
                logger.info(f"Denoised audio saved to: {tmp_denoised}")
            
            logger.info("Loading base model...")
            self._load_model("base")
            self._apply_jit_compile()
            
            logger.info("Starting generation...")
            results = list(
                self.current_model.generate(
                    text=text, ref_audio=ref_audio_path, ref_text=ref_text,
                    language=language,
                    **kwargs,
                )
            )
            logger.info(f"Generation completed, result type: {type(results[0])}")
            return self._to_numpy(results[0])
        except Exception as e:
            logger.exception(f"Voice clone generation failed: {type(e).__name__}: {e}")
            raise
        finally:
            if tmp_denoised and os.path.isfile(tmp_denoised):
                logger.info(f"Cleaning up denoised temp file: {tmp_denoised}")
                os.remove(tmp_denoised)
            self._lock.release()

    def _load_asr(self):
        """Load ASR model from local cache. Caller must hold lock."""
        if self.asr_model is not None:
            return
        self._unload_whisper_unlocked()
        self._unload_model_unlocked()
        
        local_path = _ensure_model_available(ASR_REPO_ID, "asr")
        logger.info(f"Loading ASR model: {local_path}")
        self.asr_model = load_stt_model(local_path)
        mx.eval(self.asr_model.parameters())

    def _unload_asr_unlocked(self):
        if self.asr_model is not None:
            logger.info("Unloading ASR model")
            del self.asr_model
            self.asr_model = None
            gc.collect()

    def unload_asr(self):
        self._acquire_lock()
        try:
            self._unload_asr_unlocked()
        finally:
            self._lock.release()

    def is_asr_loaded(self) -> bool:
        return self.asr_model is not None

    def transcribe(self, audio_path, language="auto") -> str:
        self._acquire_lock()
        try:
            # Validate input
            if not audio_path or not os.path.isfile(audio_path):
                raise ValueError(f"音频文件不存在: {audio_path}")
            
            self._load_asr()
            result = self.asr_model.generate(audio_path, language=language)
            return result.text
        finally:
            self._unload_asr_unlocked()
            self._lock.release()

    def _load_whisper(self, model_name: str = None):
        """Load Whisper model from local cache. Caller must hold lock."""
        model_name = model_name or DEFAULT_WHISPER_MODEL
        if self.whisper_model is not None and self.whisper_model_name == model_name:
            return
        
        self._unload_asr_unlocked()
        self._unload_model_unlocked()
        
        repo_id = WHISPER_MODEL_VARIANTS.get(model_name)
        if not repo_id:
            raise ValueError(f"Unknown Whisper model: {model_name}")
        
        local_path = _ensure_model_available(repo_id, "whisper")
        logger.info(f"Loading Whisper model: {local_path}")
        self.whisper_model = load_stt_model(local_path)
        self.whisper_model_name = model_name
        mx.eval(self.whisper_model.parameters())

    def _unload_whisper_unlocked(self):
        if self.whisper_model is not None:
            logger.info("Unloading Whisper model")
            del self.whisper_model
            self.whisper_model = None
            self.whisper_model_name = None
            gc.collect()

    def unload_whisper(self):
        self._acquire_lock()
        try:
            self._unload_whisper_unlocked()
        finally:
            self._lock.release()

    def is_whisper_loaded(self) -> bool:
        return self.whisper_model is not None

    def generate_subtitles(self, audio_path, language="auto", model_name=None) -> list[dict]:
        self._acquire_lock()
        try:
            if not audio_path or not os.path.isfile(audio_path):
                raise ValueError(f"音频文件不存在: {audio_path}")
            
            self._load_whisper(model_name)
            result = self.whisper_model.generate(audio_path, language=language)
            
            segments = []
            if hasattr(result, "segments"):
                for seg in result.segments:
                    segments.append({
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "text": seg.get("text", "").strip(),
                    })
            elif hasattr(result, "text"):
                segments.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": result.text.strip(),
                })
            
            return segments
        finally:
            self._unload_whisper_unlocked()
            self._lock.release()

    def _to_numpy(self, result) -> tuple:
        audio = np.array(result.audio, dtype=np.float32)
        sr = getattr(result, "sample_rate", 24000)
        return (sr, audio)