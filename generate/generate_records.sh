#!/usr/bin/env bash
# Load YAML config and run parallel env-centric record generation (GPU production).
#
# Usage:
#   ./generate/generate_records.sh
#   ./generate/generate_records.sh generate/configs/record_gen.yaml
#   CONFIG=generate/configs/my_run.yaml ./generate/generate_records.sh
#
# Edit generate/configs/record_gen.yaml (default device=gpu, schedule=task).

set -euo pipefail

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-${CONFIG:-generate/configs/record_gen.yaml}}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

echo "[generate_records] config=$CONFIG"
exec "$PYTHON_BIN" -m generate.parallel_records --config "$CONFIG"
