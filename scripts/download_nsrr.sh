#!/usr/bin/env bash
# Download a full NSRR dataset polysomnography archive using the nsrr gem.
# Usage:  bash scripts/download_nsrr.sh --dataset mesa|shhs|mros
#
# The nsrr gem handles resume, checksum verification, and directory creation
# automatically.  Files land in the directory the gem creates under the repo
# root (e.g. mesa/polysomnography/), which mirrors the NSRR remote path.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NSRR_BIN="/scratch/project_2019517/miniconda3/share/rubygems/bin/nsrr"
export GEM_HOME="/scratch/project_2019517/miniconda3/share/rubygems"
export GEM_PATH="/scratch/project_2019517/miniconda3/share/rubygems"

# ── Argument parsing ──────────────────────────────────────────────────────────
DATASET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "Usage: bash scripts/download_nsrr.sh --dataset mesa|shhs|mros"
    exit 1
fi

case "$DATASET" in
    mesa|shhs|mros) ;;
    *) echo "[ERROR] --dataset must be mesa, shhs, or mros (got: ${DATASET})"; exit 1 ;;
esac

# ── Load NSRR token from .env ─────────────────────────────────────────────────
ENV_FILE="${REPO_ROOT}/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[ERROR] .env file not found at ${ENV_FILE}"
    echo "        Create it with:  NSRR_TOKEN=your_token_here"
    exit 1
fi

NSRR_TOKEN=""
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line//[$'\r\n']}"
    if [[ "$line" == NSRR_TOKEN=* ]]; then
        NSRR_TOKEN="${line#NSRR_TOKEN=}"
        NSRR_TOKEN="${NSRR_TOKEN//\"/}"
        NSRR_TOKEN="${NSRR_TOKEN//\'/}"
    fi
done < "$ENV_FILE"

if [[ -z "$NSRR_TOKEN" || "$NSRR_TOKEN" == "your_token_here" ]]; then
    echo "[ERROR] NSRR_TOKEN not set in ${ENV_FILE}"
    exit 1
fi

# ── Check nsrr binary ─────────────────────────────────────────────────────────
if [[ ! -x "$NSRR_BIN" ]]; then
    echo "[ERROR] nsrr binary not found or not executable: ${NSRR_BIN}"
    echo "        Install with: gem install nsrr --no-document"
    exit 1
fi

# ── Login and download ────────────────────────────────────────────────────────
cd "$REPO_ROOT"

echo "[INFO] Logging in with NSRR token..."
"$NSRR_BIN" login --token "$NSRR_TOKEN"

REMOTE_PATH="${DATASET}/polysomnography"
echo "[INFO] Starting download: ${REMOTE_PATH}"
echo "[INFO] Files will land in ${REPO_ROOT}/${REMOTE_PATH}/"
echo "[INFO] The nsrr gem resumes incomplete downloads automatically."
echo ""

"$NSRR_BIN" download "$REMOTE_PATH"

echo ""
echo "[INFO] Download complete: ${DATASET}"
