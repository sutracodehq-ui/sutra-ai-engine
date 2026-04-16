#!/bin/sh
# ─── Ollama Model Init ─────────────────────────────────────────
# Config-driven auto-downloader for required Ollama models.
#
# Reads config/ollama_models.yaml (all list items under core/domain/embeddings)
# plus optional OLLAMA_MODEL from the environment (same as API .env).
#
# Uses `ollama show` to detect presence (handles :latest and partial names).
# Exits non-zero if any required model fails after retries.
# ────────────────────────────────────────────────────────────────

OLLAMA_HOST="${OLLAMA_HOST:-http://sutra-ai-ollama:11434}"
export OLLAMA_HOST
CONFIG_FILE="${CONFIG_FILE:-/app/config/ollama_models.yaml}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-10}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-300}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo "${GREEN}[ollama-init]${NC} $1"; }
warn()  { echo "${YELLOW}[ollama-init]${NC} $1"; }
error() { echo "${RED}[ollama-init]${NC} $1"; }

wait_for_ollama() {
    log "Waiting for Ollama at ${OLLAMA_HOST}..."
    elapsed=0
    while [ "$elapsed" -lt "$HEALTH_TIMEOUT" ]; do
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

# All uncommented "  - model" lines from core, domain, embeddings sections
parse_models() {
    if [ ! -f "$CONFIG_FILE" ]; then
        error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    awk '
      /^core:|^domain:|^embeddings:/ { sec=1; next }
      /^[a-zA-Z0-9_]+:/ && !/^core:|^domain:|^embeddings:/ { sec=0; next }
      sec && /^[[:space:]]*-[[:space:]]+/ && $0 !~ /^[[:space:]]*#/ {
        gsub(/^[[:space:]]*-[[:space:]]*/, "", $0)
        sub(/#.*/, "", $0)
        gsub(/[[:space:]]+$/, "", $0)
        if (length($0) > 0) print $0
      }
    ' "$CONFIG_FILE"
}

# True if Ollama already has this model (name or tag resolution)
model_present() {
    m="$1"
    [ -z "$m" ] && return 1
    if ollama show "$m" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

pull_model() {
    model="$1"
    attempt=1
    while [ "$attempt" -le "$MAX_RETRIES" ]; do
        log "Pulling ${model} (attempt ${attempt}/${MAX_RETRIES})..."
        if ollama pull "$model"; then
            log "✅ ${model} pulled successfully!"
            return 0
        fi
        warn "Pull failed for ${model}. Retrying in ${RETRY_DELAY}s..."
        sleep "$RETRY_DELAY"
        attempt=$((attempt + 1))
    done
    error "❌ Failed to pull ${model} after ${MAX_RETRIES} attempts."
    return 1
}

# Build unique required model list (config + OLLAMA_MODEL)
build_required_list() {
    tmpf="$(mktemp)"
    parse_models >>"$tmpf"
    if [ -n "${OLLAMA_MODEL:-}" ]; then
        echo "$OLLAMA_MODEL" >>"$tmpf"
    fi
    sort -u "$tmpf"
    rm -f "$tmpf"
}

main() {
    log "═══════════════════════════════════════════"
    log "  Ollama Model Auto-Provisioner"
    log "  Config: ${CONFIG_FILE}"
    log "  Extra from env: ${OLLAMA_MODEL:-<none>}"
    log "═══════════════════════════════════════════"

    wait_for_ollama

    required_list="$(build_required_list)"
    if [ -z "$required_list" ]; then
        warn "No models found in config. Nothing to do."
        exit 0
    fi

    log "Required models (unique):"
    echo "$required_list" | while IFS= read -r m; do [ -n "$m" ] && log "  → $m"; done

    failed=0
    # No pipe → no subshell: variables persist
    while IFS= read -r model; do
        [ -z "$model" ] && continue
        if model_present "$model"; then
            log "⏭️  ${model} already present. Skipping."
            continue
        fi
        if ! pull_model "$model"; then
            failed=$((failed + 1))
        fi
    done <<EOF
$required_list
EOF

    log "═══════════════════════════════════════════"
    log "  Verifying all required models are available..."
    verify_failed=0
    while IFS= read -r model; do
        [ -z "$model" ] && continue
        if model_present "$model"; then
            log "  ✓ ${model}"
        else
            error "  ✗ missing after pull: ${model}"
            verify_failed=$((verify_failed + 1))
        fi
    done <<EOF
$required_list
EOF

    if [ "$failed" -gt 0 ] || [ "$verify_failed" -gt 0 ]; then
        error "Init finished with failures (pull_failed=$failed verify_failed=$verify_failed)."
        ollama list || true
        exit 1
    fi

    log "Done — all required models are available."
    ollama list
}

main "$@"
