#!/usr/bin/env bash
# Compatibility entrypoint. The maintained script lives at: scripts/gpu/run_gpu_pp_migration.sh
set -euo pipefail
SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"
exec bash \"${SCRIPT_DIR}/scripts/gpu/run_gpu_pp_migration.sh\" \"$@\"
