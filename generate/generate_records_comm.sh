#!/usr/bin/env bash
# GPU production env-centric dump including attack_target (comm schema).
#
# Usage:
#   ./generate/generate_records_comm.sh
#   ./generate/generate_records_comm.sh generate/configs/record_gen.yaml
#
# Same config as generate_records.sh; forces dump_attack_target=true.

set -euo pipefail

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-python}"
CONFIG="${1:-${CONFIG:-generate/configs/record_gen.yaml}}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

echo "[generate_records_comm] config=$CONFIG dump_attack_target=true"
exec "$PYTHON_BIN" -m generate.env_centric_comm --config "$CONFIG"
