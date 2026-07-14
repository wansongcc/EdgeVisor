#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "EdgeVisor" / "scripts" / "gpu_compute_disturbance.sh"
GPU_DOC = ROOT / "EdgeVisor" / "docs" / "HOW_TO_RUN_GPU.md"
MIGRATION_DOC = ROOT / "EdgeVisor" / "docs" / "HOW_TO_ONLINE_MIGRATION.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    require(SCRIPT.exists(), f"missing script: {SCRIPT}")
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)

    text = SCRIPT.read_text(encoding="utf-8")
    require("qwen3_14b_q40" in text, "script should include the 14B disturbance workload")
    require("qwen3_8b_q40" in text, "script should include the 8B disturbance workload")
    require("setsid bash -lc" in text, "script should launch disturbance loops in process groups")
    require('kill -- "-${pgid}"' in text, "script should stop only recorded disturbance process groups")
    require("pkill" not in text, "script must not globally kill dllama processes")

    docs = GPU_DOC.read_text(encoding="utf-8") + "\n" + MIGRATION_DOC.read_text(encoding="utf-8")
    require("gpu_compute_disturbance.sh" in docs, "docs should mention the disturbance script")
    require("1xQwen3-14B + 1xQwen3-8B" in docs, "docs should document the recommended workload")
    require("Do not use `pkill -x dllama` after workers are running" in docs,
            "docs should warn against killing worker processes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
