#!/usr/bin/env bash
# Compatibility entrypoint. The maintained script lives at: scripts/semantic/run_semantic_gpu_tp_static_f32.sh
set -euo pipefail
SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"
exec bash \"${SCRIPT_DIR}/scripts/semantic/run_semantic_gpu_tp_static_f32.sh\" \"$@\"
