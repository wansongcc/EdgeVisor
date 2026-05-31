#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../config/env.sh
source "${SCRIPT_DIR}/../config/env.sh"
cd "${EDGEVISOR_ENGINE_DIR}"
make DLLAMA_VULKAN="${DLLAMA_VULKAN:-1}" dllama
