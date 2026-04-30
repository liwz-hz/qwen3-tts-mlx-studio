#!/usr/bin/env bash
set -e

# ── Change to script directory ────────────────────────────────────────────────
cd "$(dirname "$0")"

# ── Tokyonight color theme (ANSI 256) ────────────────────────────────────────
C_BLUE="122"     # #7aa2f7 — primary accent
C_CYAN="117"     # #7dcfff — cyan accent
C_GREEN="149"    # #9ece6a — success
C_RED="204"      # #f7768e — error
C_TEXT="189"     # #c0caf5 — primary text
C_MUTED="60"     # #565f89 — muted text

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

# ── Check for conda and audio environment ─────────────────────────────────────
if ! command -v conda &>/dev/null; then
    gum style --foreground "$C_RED" --bold "Conda not found. Please install Miniconda first."
    exit 1
fi

if ! conda env list 2>/dev/null | grep -q "^audio"; then
    gum style --foreground "$C_RED" --bold "Conda 'audio' environment not found. Run ./install.sh first."
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate audio

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
gum style --border rounded --border-foreground "$C_BLUE" \
    --foreground "$C_TEXT" --bold --padding "1 4" --align center \
    "Qwen3-TTS MLX Studio"
echo ""
gum style --foreground "$C_MUTED" "  The UI will open in your browser at: $(gum style --foreground "$C_CYAN" --bold "http://localhost:7860")"
gum style --foreground "$C_MUTED" "  Press Ctrl+C to stop"
echo ""

python app.py "$@"
