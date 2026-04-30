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
    echo "Installing gum via Homebrew..."
    export HOMEBREW_API_DOMAIN="https://mirrors.aliyun.com/homebrew/brew-api"
    export HOMEBREW_BOTTLE_DOMAIN="https://mirrors.aliyun.com/homebrew/homebrew-bottles"
    export HOMEBREW_NO_AUTO_UPDATE=1
    export HOMEBREW_NO_INSTALL_UPGRADE=1
    if brew install gum --quiet 2>/dev/null; then
        GUM_OK=1
    fi
fi

if [[ "$GUM_OK" -ne 1 ]]; then
    echo "  gum not available, using plain text output."
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
gum style --border double --border-foreground "$C_RED" \
    --foreground "$C_TEXT" --bold --padding "1 4" --align center \
    "Qwen3-TTS MLX Studio" "Uninstaller"
echo ""

# ── Resolve ModelScope cache directory ──────────────────────────────────────
MS_CACHE_DIR="$HOME/.cache/modelscope/hub/models/mlx-community"

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

# Local directories
scan_local ".venv"        "Virtual environment"
scan_local "outputs"      "Generated audio"
scan_local "voices"       "Voice library"
scan_local ".yt_cache"    "YouTube cache"
scan_local "__pycache__"  "Python cache"
scan_local ".gum_bin"     "Gum binary"

# ModelScope cached models
if [[ -d "$MS_CACHE_DIR" ]]; then
    for dir in "$MS_CACHE_DIR"/Qwen3-TTS-12Hz-*; do
        if [[ -d "$dir" ]]; then
            local size
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            FOUND_ITEMS+=("$dir")
            FOUND_LABELS+=("MS model — $(basename "$dir") ($size)")
            gum style --foreground "$C_TEXT" "  Found  MS model  $(gum style --foreground "$C_MUTED" "$(basename "$dir")  $size")"
        fi
    done
    for model in "Qwen3-ASR-1.7B-8bit" "DeepFilterNet-mlx"; do
        dir="$MS_CACHE_DIR/$model"
        if [[ -d "$dir" ]]; then
            size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            FOUND_ITEMS+=("$dir")
            FOUND_LABELS+=("MS model — $model ($size)")
            gum style --foreground "$C_TEXT" "  Found  MS model  $(gum style --foreground "$C_MUTED" "$model  $size")"
        fi
    done
fi

# Conda environment
if conda env list 2>/dev/null | grep -q "^audio"; then
    FOUND_ITEMS+=("conda:audio")
    FOUND_LABELS+=("Conda environment 'audio'")
    gum style --foreground "$C_TEXT" "  Found  Conda environment 'audio'"
fi

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

# ── Delete cached models from ModelScope ────────────────────────────────────
if [[ -d "$MS_CACHE_DIR" ]]; then
    for dir in "$MS_CACHE_DIR"/Qwen3-TTS-12Hz-*; do
        if [[ -d "$dir" ]]; then
            rm -rf "$dir"
            if [[ $? -eq 0 ]]; then
                ok "Removed MS model: $(basename "$dir")"
            else
                warn "Failed to remove MS model: $(basename "$dir")"
            fi
        fi
    done

    for model in "Qwen3-ASR-1.7B-8bit" "DeepFilterNet-mlx"; do
        dir="$MS_CACHE_DIR/$model"
        if [[ -d "$dir" ]]; then
            rm -rf "$dir"
            if [[ $? -eq 0 ]]; then
                ok "Removed MS model: $model"
            else
                warn "Failed to remove MS model: $model"
            fi
        fi
    done
fi

# ── Delete conda environment ───────────────────────────────────────────────
if conda env list 2>/dev/null | grep -q "^audio"; then
    conda env remove -n audio
    if [[ $? -eq 0 ]]; then
        ok "Removed conda environment 'audio'"
    else
        warn "Failed to remove conda environment 'audio'"
    fi
fi

# ── Delete venv & gum (legacy) ─────────────────────────────────────────────
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
