import os

# Model defaults
DEFAULT_MODEL_SIZE = "1.7B"
DEFAULT_QUANTIZATION = "8bit"

# ModelScope local cache path
MODELSCOPE_CACHE_ROOT = os.path.join(
    os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", "mlx-community"
)

# Model repo template — used for display and cache checking
REPO_TEMPLATE = "Qwen3-TTS-12Hz-{size}-{variant}-{quant}"

# Model variant mapping
MODEL_VARIANTS = {
    "custom_voice": "CustomVoice",
    "voice_design": "VoiceDesign",
    "base": "Base",
}

# ASR model (Qwen3-ASR for transcription, no timestamps)
ASR_REPO_ID = "mlx-community/Qwen3-ASR-1.7B-8bit"

# Whisper model variants for subtitle generation (with timestamps)
WHISPER_MODEL_VARIANTS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}
DEFAULT_WHISPER_MODEL = "large-v3-turbo"

# Whisper model sizes (approximate, for display)
WHISPER_MODEL_INFO = {
    "tiny": {"size": "39M", "speed": "~32x", "languages": 99},
    "base": {"size": "74M", "speed": "~16x", "languages": 99},
    "small": {"size": "244M", "speed": "~6x", "languages": 99},
    "medium": {"size": "769M", "speed": "~2x", "languages": 99},
    "large-v3-turbo": {"size": "1.5GB", "speed": "~8x", "languages": 100},
    "large-v3": {"size": "1.5GB", "speed": "~1x", "languages": 100},
}

# Supported languages
LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian",
]

# Default speakers — must match model's actual speaker IDs exactly (case-sensitive)
DEFAULT_SPEAKERS = [
    "serena", "vivian", "uncle_fu", "ryan", "aiden",
    "ono_anna", "sohee", "eric", "dylan",
]

# Audio settings
DEFAULT_SAMPLE_RATE = 24000
OUTPUT_DIR = "./outputs"
VOICE_LIBRARY_DIR = "./voices"
HISTORY_DIR = "./outputs/history"
YT_CACHE_DIR = ".yt_cache"
RECORDINGS_DIR = "./outputs/recordings"

# History settings
MAX_HISTORY_ENTRIES = 50
MAX_HISTORY_AUDIO_CACHE = 10

# Batch settings
DEFAULT_BATCH_SPLIT_MODE = "paragraph"
DEFAULT_SILENCE_GAP_MS = 300
MAX_BATCH_SEGMENTS = 50

# Script settings
MAX_SCRIPT_SPEAKERS = 8
DEFAULT_SCRIPT_SILENCE_MS = 500

# Generation defaults
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120

# Output defaults
DEFAULT_AUTOSAVE = False
DEFAULT_EXPORT_FORMAT = "wav"    # "wav", "mp3", "ogg"
DEFAULT_MP3_BITRATE = 192       # kbps
DEFAULT_LOUDNORM = False        # EBU R128 loudness normalization
DEFAULT_TRIM_SILENCE = False    # Strip leading/trailing silence
DEFAULT_DENOISE_REF = False     # DeepFilterNet ref audio denoising

# DeepFilterNet model (8MB, cached after first download)
DEEPFILTER_REPO = "mlx-community/DeepFilterNet-mlx"

# JIT compilation — set False to disable if you hit issues
ENABLE_JIT_COMPILE = True

# Gradio settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860
