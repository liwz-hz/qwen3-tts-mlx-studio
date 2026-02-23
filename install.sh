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

# ── Bootstrap Gum ────────────────────────────────────────────────────────────
if ! command -v gum &>/dev/null; then
    if [[ -x ".gum_bin/gum" ]]; then
        export PATH="$PWD/.gum_bin:$PATH"
    elif command -v brew &>/dev/null; then
        echo "Installing gum via Homebrew..."
        brew install gum --quiet
    else
        echo "Downloading gum binary..."
        mkdir -p .gum_bin
        curl -fsSL "https://github.com/charmbracelet/gum/releases/download/v0.16.0/gum_0.16.0_Darwin_arm64.tar.gz" \
            | tar xz -C .gum_bin gum
        export PATH="$PWD/.gum_bin:$PATH"
    fi
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
ok()   { gum style --foreground "$C_GREEN"  "  [OK]  $1"; }
warn() { gum style --foreground "$C_ORANGE" "  [!!]  $1"; }
fail() { gum style --foreground "$C_RED"    "  [ERR] $1"; exit 1; }
info() { gum style --foreground "$C_BLUE" --bold "  >>>   $1"; }

# Returns 0 and prints "MAJOR.MINOR" if the given python binary is 3.10–3.13
check_python_compat() {
    local py_bin="$1"
    if [[ ! -x "$py_bin" ]] && ! command -v "$py_bin" &>/dev/null; then
        return 1
    fi
    local ver major minor
    ver=$("$py_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || return 1
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -le 13 ]]; then
        echo "$ver"
        return 0
    fi
    return 1
}

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

# ── Preflight: Resolve compatible Python (3.10–3.13) ────────────────────────
info "Checking for compatible Python (3.10–3.13)..."

PYTHON_CMD=""
PY_VERSION=""

# Step 1: Try system python3
if ver=$(check_python_compat python3); then
    PYTHON_CMD="python3"
    PY_VERSION="$ver"
fi

# Step 2: Try versioned binaries on PATH (prefer 3.12 for best wheel support)
if [[ -z "$PYTHON_CMD" ]]; then
    for minor in 12 13 11 10; do
        if ver=$(check_python_compat "python3.${minor}"); then
            PYTHON_CMD="python3.${minor}"
            PY_VERSION="$ver"
            break
        fi
    done
fi

# Step 3: Try Homebrew prefix explicitly (handles PATH not including brew bin)
if [[ -z "$PYTHON_CMD" ]] && command -v brew &>/dev/null; then
    BREW_PREFIX="$(brew --prefix)"
    for minor in 12 13 11 10; do
        local_bin="${BREW_PREFIX}/bin/python3.${minor}"
        if ver=$(check_python_compat "$local_bin"); then
            PYTHON_CMD="$local_bin"
            PY_VERSION="$ver"
            break
        fi
    done
fi

# Step 4: Offer to brew install python@3.12
if [[ -z "$PYTHON_CMD" ]]; then
    if command -v python3 &>/dev/null; then
        sys_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
        warn "System python3 is $sys_ver — outside compatible range (3.10–3.13)."
    else
        warn "No python3 found on PATH."
    fi

    if command -v brew &>/dev/null; then
        echo ""
        gum style --foreground "$C_TEXT" "   Python 3.12 is recommended (best compatibility with MLX ecosystem)."
        if gum confirm "Install python@3.12 via Homebrew?"; then
            info "Installing python@3.12 via Homebrew..."
            gum spin --spinner dot --title "Installing python@3.12..." \
                --spinner.foreground "$C_BLUE" -- brew install python@3.12
            BREW_PREFIX="$(brew --prefix)"
            PYTHON_CMD="${BREW_PREFIX}/bin/python3.12"
            if ver=$(check_python_compat "$PYTHON_CMD"); then
                PY_VERSION="$ver"
            else
                fail "Homebrew installed python@3.12 but it failed validation."
            fi
        else
            fail "No compatible Python available. Install manually: brew install python@3.12"
        fi
    else
        echo ""
        gum style --foreground "$C_TEXT" "   No compatible Python found and Homebrew is not installed."
        echo ""
        gum style --foreground "$C_TEXT" "   Option 1: Install Homebrew (https://brew.sh), then:"
        gum style --foreground "$C_TEXT" "             brew install python@3.12"
        echo ""
        gum style --foreground "$C_TEXT" "   Option 2: Download Python 3.12 from:"
        gum style --foreground "$C_TEXT" "             https://www.python.org/downloads/"
        echo ""
        fail "Cannot continue without Python 3.10–3.13."
    fi
fi

ok "Using Python $PY_VERSION ($PYTHON_CMD)"

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

# ── Create virtual environment ────────────────────────────────────────────────
info "Setting up Python virtual environment..."

if [[ -d ".venv" ]]; then
    # Verify existing venv uses a compatible Python
    if ver=$(check_python_compat .venv/bin/python3); then
        ok "Virtual environment already exists (.venv/) — Python $ver"
    else
        VENV_PY_VER=$(.venv/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
        warn "Existing .venv uses Python $VENV_PY_VER (incompatible). Recreating..."
        rm -rf .venv
        "$PYTHON_CMD" -m venv .venv
        ok "Recreated virtual environment with Python $PY_VERSION"
    fi
else
    "$PYTHON_CMD" -m venv .venv
    ok "Created virtual environment (.venv/)"
fi

source .venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
info "Installing Python packages (this may take a few minutes)..."

gum spin --spinner dot --title "Upgrading pip..." \
    --spinner.foreground "$C_BLUE" -- pip install --upgrade pip -q
gum spin --spinner dot --title "Installing Python packages..." \
    --spinner.foreground "$C_BLUE" -- pip install -U -r requirements.txt -q

ok "All packages installed"

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p outputs/history voices
ok "Created output directories"

# ── Optional: Pre-download models ─────────────────────────────────────────────
echo ""
info "Model download (optional)"
gum style --foreground "$C_TEXT" "   The TTS models (~6 GB total) can be downloaded now, or they'll"
gum style --foreground "$C_TEXT" "   download automatically when you first use each voice mode."
echo ""

if gum confirm --default=no "Download models now?"; then
    info "Downloading models (this will take a while)..."
    pip install -q huggingface_hub

    gum spin --spinner dot --title "[1/3] Downloading CustomVoice model..." \
        --spinner.foreground "$C_BLUE" -- \
        huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit --quiet
    ok "CustomVoice model downloaded"

    gum spin --spinner dot --title "[2/3] Downloading VoiceDesign model..." \
        --spinner.foreground "$C_BLUE" -- \
        huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit --quiet
    ok "VoiceDesign model downloaded"

    gum spin --spinner dot --title "[3/3] Downloading Base model (Voice Cloning)..." \
        --spinner.foreground "$C_BLUE" -- \
        huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit --quiet
    ok "Base model downloaded"
else
    ok "Skipped — models will download on first use"
fi

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
