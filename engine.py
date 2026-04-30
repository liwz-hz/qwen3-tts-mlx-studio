import gc
import os
import threading

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
from mlx_audio.stt.utils import load_model as load_stt_model

from config import MODELSCOPE_CACHE_ROOT, REPO_TEMPLATE, MODEL_VARIANTS, DEFAULT_MODEL_SIZE, DEFAULT_QUANTIZATION, ENABLE_JIT_COMPILE, ASR_REPO_ID

LOCK_TIMEOUT = 10  # seconds to wait for lock before raising


class TTSEngine:
    """Manages model loading, unloading, and inference."""

    def __init__(self):
        self.current_model = None
        self.current_model_type = None  # "custom_voice" | "voice_design" | "base"
        self.model_size = DEFAULT_MODEL_SIZE
        self.quantization = DEFAULT_QUANTIZATION
        self.jit_compile = ENABLE_JIT_COMPILE
        self.asr_model = None
        self.last_max_tokens = None  # Track last max_tokens to detect changes
        self._lock = threading.Lock()

    def _acquire_lock(self):
        """Acquire lock with timeout so callers don't hang forever."""
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
        return os.path.join(MODELSCOPE_CACHE_ROOT, repo_name)

    def _load_model(self, model_type: str, max_tokens=None):
        """Load model, unloading previous if different. Caller must hold lock.
        
        If max_tokens changes and JIT is enabled, force reload to clear compile cache.
        """
        self._unload_asr_unlocked()
        
        need_reload = False
        
        if self.current_model_type != model_type:
            need_reload = True
        elif self.jit_compile and max_tokens is not None and self.last_max_tokens != max_tokens:
            need_reload = True
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[ENGINE] max_tokens changed from {self.last_max_tokens} to {max_tokens}, reloading model")
        
        if need_reload:
            self._unload_model_unlocked()
            repo_id = self.get_repo_id(model_type)
            self.current_model = load_model(repo_id)
            self.current_model_type = model_type
            if self.jit_compile:
                self._apply_compile()
            mx.eval(self.current_model.parameters())
        
        if max_tokens is not None:
            self.last_max_tokens = max_tokens

    def _apply_compile(self):
        """Wrap talker forward passes with mx.compile for faster inference."""
        talker = getattr(self.current_model, "talker", None)
        if talker is None:
            return
        talker.__call__ = mx.compile(talker.__call__, shapeless=True)
        code_pred = getattr(talker, "code_predictor", None)
        if code_pred is not None:
            code_pred.__call__ = mx.compile(code_pred.__call__, shapeless=True)

    def is_model_loaded(self, model_type: str) -> bool:
        """Return True if this model type is already in memory (no swap needed)."""
        return self.current_model_type == model_type

    def unload_model(self):
        """Free memory from current model (thread-safe)."""
        self._acquire_lock()
        try:
            self._unload_model_unlocked()
        finally:
            self._lock.release()

    def _unload_model_unlocked(self):
        """Free memory from current model (and ASR if loaded). Caller must hold lock."""
        self._unload_asr_unlocked()
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            gc.collect()

    def generate_custom_voice(self, text, speaker, language, instruct="", **kwargs) -> tuple:
        """Returns (sample_rate, numpy_audio_array)."""
        self._acquire_lock()
        try:
            max_tokens = kwargs.get("max_tokens")
            self._load_model("custom_voice", max_tokens=max_tokens)
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
        """Returns (sample_rate, numpy_audio_array)."""
        self._acquire_lock()
        try:
            max_tokens = kwargs.get("max_tokens")
            self._load_model("voice_design", max_tokens=max_tokens)
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
        """Returns (sample_rate, numpy_audio_array)."""
        self._acquire_lock()
        tmp_denoised = None
        try:
            if denoise_ref:
                from audio_utils import denoise_ref_audio
                tmp_denoised = denoise_ref_audio(ref_audio_path)
                ref_audio_path = tmp_denoised
            max_tokens = kwargs.get("max_tokens")
            self._load_model("base", max_tokens=max_tokens)
            results = list(
                self.current_model.generate(
                    text=text, ref_audio=ref_audio_path, ref_text=ref_text,
                    language=language,
                    **kwargs,
                )
            )
            return self._to_numpy(results[0])
        finally:
            if tmp_denoised and os.path.isfile(tmp_denoised):
                os.remove(tmp_denoised)
            self._lock.release()

    # ----- ASR -----

    def _load_asr(self):
        """Load ASR model, unloading TTS first. Caller must hold lock."""
        if self.asr_model is not None:
            return  # already loaded
        self._unload_model_unlocked()  # free TTS
        self.asr_model = load_stt_model(ASR_REPO_ID)
        mx.eval(self.asr_model.parameters())

    def _unload_asr_unlocked(self):
        """Free ASR model. Caller must hold lock."""
        if self.asr_model is not None:
            del self.asr_model
            self.asr_model = None
            gc.collect()

    def unload_asr(self):
        """Free ASR model (thread-safe)."""
        self._acquire_lock()
        try:
            self._unload_asr_unlocked()
        finally:
            self._lock.release()

    def is_asr_loaded(self) -> bool:
        return self.asr_model is not None

    def transcribe(self, audio_path, language="auto") -> str:
        """Transcribe audio file, returns text. Loads/unloads ASR automatically."""
        self._acquire_lock()
        try:
            self._load_asr()
            result = self.asr_model.generate(audio_path, language=language)
            return result.text
        finally:
            self._unload_asr_unlocked()
            self._lock.release()

    def _to_numpy(self, result) -> tuple:
        """Convert mlx array result to numpy for Gradio Audio component."""
        audio = np.array(result.audio, dtype=np.float32)
        sr = getattr(result, "sample_rate", 24000)
        return (sr, audio)
