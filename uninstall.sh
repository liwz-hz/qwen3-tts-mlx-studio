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

# ── Header banner ────────────────────────────────────────────────────────────
echo ""
gum style --border double --border-foreground "$C_RED" \
    --foreground "$C_TEXT" --bold --padding "1 4" --align center \
    "Qwen3-TTS MLX Studio" "Uninstaller"
echo ""

# ── Resolve HuggingFace cache directory ──────────────────────────────────────
HF_CACHE_DIR="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface/hub}}"

# ── Scan for artifacts ───────────────────────────────────────────────────────
info "Scanning for installed artifacts..."
echo ""

FOUND_ITEMS=()
FOUND_LABELS=()

scan_local() {
    local dir="$1" label="$2"
    if [[ -d "$dir" ]]; then
        local size
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        FOUND_ITEMS+=("$dir")
        FOUND_LABELS+=("$label ($size)")
        gum style --foreground "$C_TEXT" "  Found  $label  $(gum style --foreground "$C_MUTED" "$size")"
    fi
}

scan_hf_model() {
    local pattern="$1" label="$2"
    for dir in "$HF_CACHE_DIR"/$pattern; do
        if [[ -d "$dir" ]]; then
            local size
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            FOUND_ITEMS+=("$dir")
            FOUND_LABELS+=("$label — $(basename "$dir") ($size)")
            gum style --foreground "$C_TEXT" "  Found  $label  $(gum style --foreground "$C_MUTED" "$(basename "$dir")  $size")"
        fi
    done
}

# Local directories
scan_local ".venv"        "Virtual environment"
scan_local "outputs"      "Generated audio"
scan_local "voices"       "Voice library"
scan_local ".yt_cache"    "YouTube cache"
scan_local "__pycache__"  "Python cache"
scan_local ".gum_bin"     "Gum binary"

# HuggingFace cached models
scan_hf_model "models--mlx-community--Qwen3-TTS-12Hz-*"          "HF model"
scan_hf_model "models--mlx-community--Qwen3-ASR-1.7B-8bit"       "HF model"
scan_hf_model "models--mlx-community--DeepFilterNet-mlx"          "HF model"

# ── Nothing to remove? ──────────────────────────────────────────────────────
if [[ ${#FOUND_ITEMS[@]} -eq 0 ]]; then
    echo ""
    ok "Nothing to remove — already clean."
    echo ""
    exit 0
fi

# ── Confirm ──────────────────────────────────────────────────────────────────
echo ""
gum style --foreground "$C_ORANGE" --bold "  ${#FOUND_ITEMS[@]} item(s) will be permanently deleted."
echo ""

if ! gum confirm --default=no "Continue with uninstall?"; then
    echo ""
    warn "Cancelled — nothing was deleted."
    echo ""
    exit 0
fi

echo ""

# ── Stop set -e so one failure doesn't abort the rest ────────────────────────
set +e

# ── Delete local app data ────────────────────────────────────────────────────
remove_dir() {
    local dir="$1" label="$2"
    if [[ -d "$dir" ]]; then
        rm -rf "$dir"
        if [[ $? -eq 0 ]]; then
            ok "Removed $label"
        else
            warn "Failed to remove $label ($dir)"
        fi
    fi
}

remove_dir "outputs"      "generated audio (outputs/)"
remove_dir "voices"       "voice library (voices/)"
remove_dir ".yt_cache"    "YouTube cache (.yt_cache/)"
remove_dir "__pycache__"  "Python cache (__pycache__/)"

# ── Delete cached models ────────────────────────────────────────────────────
for dir in "$HF_CACHE_DIR"/models--mlx-community--Qwen3-TTS-12Hz-*; do
    if [[ -d "$dir" ]]; then
        rm -rf "$dir"
        if [[ $? -eq 0 ]]; then
            ok "Removed HF model: $(basename "$dir")"
        else
            warn "Failed to remove HF model: $(basename "$dir")"
        fi
    fi
done

for model in "models--mlx-community--Qwen3-ASR-1.7B-8bit" "models--mlx-community--DeepFilterNet-mlx"; do
    dir="$HF_CACHE_DIR/$model"
    if [[ -d "$dir" ]]; then
        rm -rf "$dir"
        if [[ $? -eq 0 ]]; then
            ok "Removed HF model: $model"
        else
            warn "Failed to remove HF model: $model"
        fi
    fi
done

# ── Delete venv & gum ───────────────────────────────────────────────────────
remove_dir ".venv"     "virtual environment (.venv/)"
remove_dir ".gum_bin"  "gum binary (.gum_bin/)"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
gum style --border double --border-foreground "$C_GREEN" \
    --foreground "$C_GREEN" --bold --padding "1 4" --align center \
    "Uninstall complete!"
echo ""
gum style --foreground "$C_MUTED" "  Project source files remain in: $(pwd)"
gum style --foreground "$C_MUTED" "  To remove them:  rm -rf $(pwd)"
gum style --foreground "$C_MUTED" "  To reinstall:    ./install.sh"
echo ""
