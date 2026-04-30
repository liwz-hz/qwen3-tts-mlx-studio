#!/usr/bin/env bash
set -e

# ── Change to script directory ────────────────────────────────────────────────
cd "$(dirname "$0")"

# ── Tokyonight color theme (ANSI 256) ────────────────────────────────────────
C_BLUE="122"     # #7aa2f7 — primary accent
C_CYAN="117"     # #7dcfff — cyan accent
C_GREEN="149"    # #9ece6a — success
C_ORANGE="209"   # #ff9e64 — warning
C_RED="204"      # #f7768e — error
C_TEXT="189"     # #c0caf5 — primary text
C_MUTED="60"     # #565f89 — muted text
C_BORDER="60"    # #3b4261 — borders

# ── Bootstrap Gum (skip if Homebrew broken) ──────────────────────────────────
GUM_OK=0
if command -v gum &>/dev/null; then
    GUM_OK=1
elif [[ -x ".gum_bin/gum" ]]; then
    export PATH="$PWD/.gum_bin:$PATH"
    GUM_OK=1
elif command -v brew &>/dev/null; then
    export HOMEBREW_API_DOMAIN="https://mirrors.aliyun.com/homebrew/brew-api"
    export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.aliyun.com/homebrew/homebrew-bottles"
    export HOMEBREW_NO_AUTO_UPDATE=1
    export HOMEBREW_NO_INSTALL_UPGRADE=1
    if timeout 15 brew install gum --quiet 2>/dev/null; then
        GUM_OK=1
    fi
fi

if [[ "$GUM_OK" -ne 1 ]]; then
    gum() {
        local last=""
        for arg in "$@"; do
            case "$arg" in
                --*) ;;
                *) last="$arg" ;;
            esac
        done
        echo "$last"
    }
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
ok()   { gum style --foreground "$C_GREEN"  "  [OK]  $1"; }
warn() { gum style --foreground "$C_ORANGE" "  [!!]  $1"; }
fail() { gum style --foreground "$C_RED"    "  [ERR] $1"; exit 1; }
info() { gum style --foreground "$C_BLUE" --bold "  >>>   $1"; }

# ── Header banner ────────────────────────────────────────────────────────────
echo ""
gum style --border double --border-foreground "$C_BLUE" \
    --foreground "$C_TEXT" --bold --padding "1 4" --align center \
    "Qwen3-TTS MLX Studio" "Installer"
echo ""

# ── Preflight: macOS + Apple Silicon ──────────────────────────────────────────
info "Checking system requirements..."

if [[ "$(uname)" != "Darwin" ]]; then
    fail "This app requires macOS. Detected: $(uname)"
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    fail "This app requires Apple Silicon (M1/M2/M3/M4). Detected: $ARCH"
fi
ok "macOS on Apple Silicon ($ARCH)"

# ── Preflight: Conda ─────────────────────────────────────────────────────────
info "Checking conda installation..."

if ! command -v conda &>/dev/null; then
    echo ""
    gum style --foreground "$C_TEXT" "   Conda (Miniconda/Anaconda) is required to manage the Python environment."
    echo ""
    gum style --foreground "$C_TEXT" "   Install Miniconda from: https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    fail "Conda not found on PATH."
fi
ok "Conda found: $(conda --version)"

# ── Preflight: ffmpeg ─────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found"
else
    warn "ffmpeg not found — required for audio processing."
    if command -v brew &>/dev/null; then
        if gum confirm "Install ffmpeg via Homebrew?"; then
            info "Installing ffmpeg via Homebrew..."
            gum spin --spinner dot --title "Installing ffmpeg..." \
                --spinner.foreground "$C_BLUE" -- brew install ffmpeg
            ok "ffmpeg installed"
        else
            warn "Skipping ffmpeg — audio processing may not work."
        fi
    else
        gum style --foreground "$C_TEXT" "   Install with Homebrew:  brew install ffmpeg"
        gum style --foreground "$C_TEXT" "   (If you don't have Homebrew: https://brew.sh)"
        echo ""
        if ! gum confirm --default=no "Continue without ffmpeg?"; then
            exit 1
        fi
    fi
fi

# ── Create/activate conda environment ─────────────────────────────────────────
info "Setting up conda environment 'audio'..."

if conda env list 2>/dev/null | grep -q "^audio"; then
    ok "Conda environment 'audio' already exists"
else
    gum spin --spinner dot --title "Creating conda environment 'audio' (Python 3.12)..." \
        --spinner.foreground "$C_BLUE" -- conda create -n audio python=3.12 -y
    ok "Created conda environment 'audio'"
fi

eval "$(conda shell.bash hook)"
conda activate audio

# ── Install dependencies ──────────────────────────────────────────────────────
info "Installing Python packages (this may take a few minutes)..."

gum spin --spinner dot --title "Installing Python packages..." \
    --spinner.foreground "$C_BLUE" -- pip install -U -r requirements.txt -q

ok "All packages installed"

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p outputs/history voices
ok "Created output directories"

# ── Model download (DISABLED — use ModelScope local cache) ────────────────────
#
# 模型预下载已禁用。请使用 ModelScope 下载模型：
#
#   方法 1: pip 安装 modelscope 后使用 CLI
#     pip install modelscope
#     modelscope download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
#
#   方法 2: Python 代码下载
#     from modelscope.hub.snapshot_download import snapshot_download
#     snapshot_download("mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit")
#
#   模型列表:
#     - mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-{bf16|8bit|6bit|4bit}
#     - mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-{bf16|8bit|6bit|4bit}
#     - mlx-community/Qwen3-TTS-12Hz-1.7B-Base-{bf16|8bit|6bit|4bit}
#
#   模型将下载到: ~/.cache/modelscope/hub/models/mlx-community/
#
# 以下为原 HuggingFace 预下载代码（已注释）:
#
# echo ""
# info "Model download (optional)"
# gum style --foreground "$C_TEXT" "   The TTS models can be downloaded now, or they'll"
# gum style --foreground "$C_TEXT" "   download automatically when you first use each voice mode."
# echo ""
#
# if gum confirm --default=no "Download models now?"; then
#     info "Select quantization (bf16 is default, 4bit is smallest):"
#     QUANT=$(gum choose --selected="bf16" "bf16" "8bit" "6bit" "4bit")
#     info "Downloading ${QUANT} models..."
#     pip install -q huggingface_hub
#
#     gum spin --spinner dot --title "[1/3] Downloading CustomVoice model (${QUANT})..." \
#         --spinner.foreground "$C_BLUE" -- \
#         huggingface-cli download "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-${QUANT}" --quiet
#     ok "CustomVoice model downloaded"
#
#     gum spin --spinner dot --title "[2/3] Downloading VoiceDesign model (${QUANT})..." \
#         --spinner.foreground "$C_BLUE" -- \
#         huggingface-cli download "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-${QUANT}" --quiet
#     ok "VoiceDesign model downloaded"
#
#     gum spin --spinner dot --title "[3/3] Downloading Base model (${QUANT})..." \
#         --spinner.foreground "$C_BLUE" -- \
#         huggingface-cli download "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-${QUANT}" --quiet
#     ok "Base model downloaded"
# else
#     ok "Skipped — models will download on first use"
# fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
gum style --border double --border-foreground "$C_GREEN" \
    --foreground "$C_GREEN" --bold --padding "1 4" --align center \
    "Installation complete!"
echo ""
echo "  To start the app, run:"
echo ""
gum style --foreground "$C_BLUE" --bold "    ./run.sh"
echo ""
