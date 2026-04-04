#!/bin/sh
# ─── Ollama Model Init ─────────────────────────────────────────
# Config-driven auto-downloader for required Ollama models.
#
# Reads config/ollama_models.yaml and pulls any model not already
# present in the Ollama instance. Retries on failure.
#
# Software Factory: add a model to config/ollama_models.yaml →
# it's auto-downloaded on next container start. Zero code changes.
# ────────────────────────────────────────────────────────────────

set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://sutra-ai-ollama:11434}"
CONFIG_FILE="${CONFIG_FILE:-/app/config/ollama_models.yaml}"
MAX_RETRIES=3
RETRY_DELAY=10
HEALTH_TIMEOUT=120   # Max seconds to wait for Ollama to be ready

# ─── Colors ─────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo "${GREEN}[ollama-init]${NC} $1"; }
warn()  { echo "${YELLOW}[ollama-init]${NC} $1"; }
error() { echo "${RED}[ollama-init]${NC} $1"; }

# ─── Wait for Ollama to be healthy ──────────────────────────────
wait_for_ollama() {
    log "Waiting for Ollama at ${OLLAMA_HOST}..."
    elapsed=0
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        if ollama list >/dev/null 2>&1; then
            log "Ollama is ready!"
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    error "Ollama not ready after ${HEALTH_TIMEOUT}s. Aborting."
    exit 1
}

# ─── Parse models from YAML (no Python needed) ─────────────────
# Extracts uncommented `- model:tag` lines from the config file.
parse_models() {
    if [ ! -f "$CONFIG_FILE" ]; then
        error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    # Extract lines matching "  - model:tag" (indented list items)
    # Skip commented lines (starting with #)
    grep -E '^\s+-\s+' "$CONFIG_FILE" | grep -v '^\s*#' | sed 's/^\s*-\s*//' | sed 's/\s*#.*//' | tr -d ' '
}

# ─── Get currently installed models ─────────────────────────────
get_installed() {
    ollama list 2>/dev/null | tail -n +2 | awk '{print $1}'
}

# ─── Pull a single model with retries ──────────────────────────
pull_model() {
    model="$1"
    attempt=1
    while [ $attempt -le $MAX_RETRIES ]; do
        log "Pulling ${model} (attempt ${attempt}/${MAX_RETRIES})..."
        if ollama pull "$model"; then
            log "✅ ${model} pulled successfully!"
            return 0
        fi
        warn "Pull failed for ${model}. Retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
        attempt=$((attempt + 1))
    done
    error "❌ Failed to pull ${model} after ${MAX_RETRIES} attempts."
    return 1
}

# ─── Main ───────────────────────────────────────────────────────
main() {
    log "═══════════════════════════════════════════"
    log "  Ollama Model Auto-Provisioner"
    log "  Config: ${CONFIG_FILE}"
    log "═══════════════════════════════════════════"

    wait_for_ollama

    # Parse required models from config
    required=$(parse_models)
    if [ -z "$required" ]; then
        warn "No models found in config. Nothing to do."
        exit 0
    fi

    # Get already installed models
    installed=$(get_installed)

    log "Required models:"
    echo "$required" | while read -r m; do log "  → $m"; done

    log "Installed models:"
    if [ -z "$installed" ]; then
        log "  (none)"
    else
        echo "$installed" | while read -r m; do log "  ✓ $m"; done
    fi

    # Pull only missing models
    pulled=0
    skipped=0
    failed=0

    echo "$required" | while read -r model; do
        [ -z "$model" ] && continue

        if echo "$installed" | grep -qx "$model"; then
            log "⏭️  ${model} already installed. Skipping."
            skipped=$((skipped + 1))
        else
            if pull_model "$model"; then
                pulled=$((pulled + 1))
            else
                failed=$((failed + 1))
            fi
        fi
    done

    log "═══════════════════════════════════════════"
    log "  Done! Models ready."
    log "═══════════════════════════════════════════"

    # Final inventory
    log "Final model inventory:"
    ollama list
}

main "$@"
