#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "EdgeVisor" / "scripts" / "jetson_freq_inject.sh"
README = ROOT / "EdgeVisor" / "README.md"

GPU_FREQS = "306000000 408000000 612000000 816000000 1020000000"
CPU_FREQS = "729600 1190400 1516800 1894400"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeSysfs:
    """Builds a temporary fake sysfs tree with GPU, CPU, and EMC subsystems."""

    def __init__(
        self,
        root: Path,
        *,
        gpu: bool = True,
        cpu: bool = True,
        emc: bool = True,
        cpu_count: int = 3,
        gpu_layout: str = "r36",
    ) -> None:
        self.root = root / "sys"
        self.lock_file = root / "injector.lock"
        self.write_log = root / "writes.log"

        if gpu:
            self._make_gpu(gpu_layout)
        if cpu:
            self._make_cpu(cpu_count)
        if emc:
            self._make_emc()

    def _make_gpu(self, layout: str) -> None:
        if layout == "r36":
            self.gpu_platform = self.root / "devices" / "platform" / "17000000.gpu"
            gpu_name = "17000000.gpu"
        else:
            self.gpu_platform = self.root / "devices" / "17000000.ga10b"
            gpu_name = "17000000.ga10b"
        self.gpu_dir = self.gpu_platform / "devfreq" / gpu_name
        self.gpu_dir.mkdir(parents=True)
        self._write(self.gpu_dir / "available_frequencies", GPU_FREQS)
        self._write(self.gpu_dir / "min_freq", "306000000")
        self._write(self.gpu_dir / "max_freq", "1020000000")
        self._write(self.gpu_dir / "cur_freq", "612000000")
        (self.gpu_platform / "enable_3d_scaling").write_text("1\n", encoding="utf-8")

    def _make_cpu(self, count: int) -> None:
        self.cpu_dirs = []
        for i in range(count):
            cpu_dir = self.root / "devices" / "system" / "cpu" / f"cpu{i}" / "cpufreq"
            cpu_dir.mkdir(parents=True)
            self._write(cpu_dir / "scaling_available_frequencies", CPU_FREQS)
            self._write(cpu_dir / "scaling_governor", "schedutil")
            self._write(cpu_dir / "scaling_min_freq", "729600")
            self._write(cpu_dir / "scaling_max_freq", "1894400")
            self._write(cpu_dir / "cpuinfo_min_freq", "729600")
            self._write(cpu_dir / "cpuinfo_max_freq", "1894400")
            self._write(cpu_dir / "scaling_available_governors", "performance schedutil userspace")
            self.cpu_dirs.append(cpu_dir)

    def _make_emc(self) -> None:
        self.emc_dir = self.root / "kernel" / "debug" / "bpmp" / "debug" / "clk" / "emc"
        self.emc_dir.mkdir(parents=True)
        self._write(self.emc_dir / "rate", "2133000000")
        self._write(self.emc_dir / "min_rate", "204000000")
        self._write(self.emc_dir / "max_rate", "2133000000")

    @staticmethod
    def _write(path: Path, value: str) -> None:
        path.write_text(value.rstrip("\n") + "\n", encoding="utf-8")

    def gpu_read(self, name: str) -> str:
        return (self.gpu_dir / name).read_text(encoding="utf-8").strip()

    @property
    def gpu_scaling(self) -> str:
        return (self.gpu_platform / "enable_3d_scaling").read_text(encoding="utf-8").strip()

    def cpu_read(self, core: int, name: str) -> str:
        return (self.cpu_dirs[core] / name).read_text(encoding="utf-8").strip()

    def emc_read(self, name: str) -> str:
        return (self.emc_dir / name).read_text(encoding="utf-8").strip()

    def env(self, *, euid: int = 0, no_sleep: bool = True) -> dict[str, str]:
        env = os.environ.copy()
        env["JETSON_FREQ_TEST_EUID"] = str(euid)
        env["JETSON_FREQ_LOCK_FILE"] = str(self.lock_file)
        env["JETSON_FREQ_WRITE_LOG"] = str(self.write_log)
        if no_sleep:
            env["JETSON_FREQ_TEST_NO_SLEEP"] = "1"
        else:
            env.pop("JETSON_FREQ_TEST_NO_SLEEP", None)
        return env


def run(fake: FakeSysfs, *args: str, euid: int = 0, no_sleep: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args, "--sysfs-root", str(fake.root)],
        text=True,
        capture_output=True,
        env=fake.env(euid=euid, no_sleep=no_sleep),
        timeout=5,
        check=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_help_and_list() -> None:
    help_result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], text=True, capture_output=True, check=False
    )
    require(help_result.returncode == 0, help_result.stderr)
    require("default: 20" in help_result.stdout, "help must document the 20-second default")
    require("--target" in help_result.stdout, "help must document --target")

    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--list", euid=1000)
        require(result.returncode == 0, result.stderr)
        require("gpu_available_frequencies_hz=" in result.stdout, "list must show GPU frequencies")
        require("cpu_available_frequencies_khz=" in result.stdout, "list must show CPU frequencies")
        require("emc_mode=" in result.stdout, "list must show EMC mode")


def test_default_lock_uses_root_owned_directory() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    require("JETSON_FREQ_LOCK_FILE:-/run/lock/jetson_freq_inject.lock" in text,
            "the root process must not create its predictable lock in world-writable /tmp")


def test_gpu_only_inject_and_restore() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--level", "20", "--target", "gpu")
        require(result.returncode == 0, result.stderr)
        require("gpu_target_freq_hz=408000000" in result.stdout, "level 20 must select 408 MHz GPU")
        require("duration_sec=20" in result.stdout, "omitted duration must default to 20 seconds")
        require(fake.gpu_read("min_freq") == "306000000", "GPU min not restored")
        require(fake.gpu_read("max_freq") == "1020000000", "GPU max not restored")
        require(fake.gpu_scaling == "1", "GPU 3D scaling not restored")
        # CPU and EMC should be untouched
        require(fake.cpu_read(0, "scaling_governor") == "schedutil", "CPU governor must not change with --target gpu")


def test_cpu_only_inject_and_restore() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--level", "0", "--target", "cpu")
        require(result.returncode == 0, result.stderr)
        require("cpu_target_freq_khz=729600" in result.stdout, "level 0 must select lowest CPU freq")
        # All cores restored
        for i in range(3):
            require(fake.cpu_read(i, "scaling_governor") == "schedutil",
                    f"cpu{i} governor not restored")
            require(fake.cpu_read(i, "scaling_min_freq") == "729600",
                    f"cpu{i} min not restored")
            require(fake.cpu_read(i, "scaling_max_freq") == "1894400",
                    f"cpu{i} max not restored")
        # GPU should be untouched
        require(fake.gpu_read("min_freq") == "306000000", "GPU min must not change with --target cpu")


def test_emc_only_inject_and_restore() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--level", "0", "--target", "emc")
        require(result.returncode == 0, result.stderr)
        require("emc_target_freq=204000000" in result.stdout, "level 0 must select lowest EMC freq")
        require(fake.emc_read("min_rate") == "204000000", "EMC min not restored")
        require(fake.emc_read("max_rate") == "2133000000", "EMC max not restored")


def test_bpmp_emc_uses_rate_not_read_only_bounds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--level", "0", "--target", "emc")
        require(result.returncode == 0, result.stderr)
        writes = fake.write_log.read_text(encoding="utf-8").splitlines()
        require(writes == ["rate=204000000", "rate=2133000000"],
                f"BPMP EMC must write rate only, got: {writes}")


def test_all_combined_inject_and_restore() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--level", "50")
        require(result.returncode == 0, result.stderr)
        require("gpu_target_freq_hz=" in result.stdout, "combined must report GPU target")
        require("cpu_target_freq_khz=" in result.stdout, "combined must report CPU target")
        require("emc_target_freq=" in result.stdout, "combined must report EMC target")
        # All restored
        require(fake.gpu_read("min_freq") == "306000000", "GPU min not restored after combined")
        require(fake.gpu_read("max_freq") == "1020000000", "GPU max not restored after combined")
        for i in range(3):
            require(fake.cpu_read(i, "scaling_max_freq") == "1894400",
                    f"cpu{i} max not restored after combined")
        require(fake.emc_read("max_rate") == "2133000000", "EMC max not restored after combined")


def test_level_mapping_per_subsystem() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        # GPU: level 20 -> 408 MHz (nearest to 20% of 306M..1020M range)
        result = run(fake, "--level", "20", "--target", "gpu")
        require("gpu_target_freq_hz=408000000" in result.stdout, "GPU level 20 mapping")

        # CPU: level 100 -> 1894400 KHz (max)
        result = run(fake, "--level", "100", "--target", "cpu")
        require("cpu_target_freq_khz=1894400" in result.stdout, "CPU level 100 mapping")

        # EMC: level 0 -> 204000000 (min)
        result = run(fake, "--level", "0", "--target", "emc")
        require("emc_target_freq=204000000" in result.stdout, "EMC level 0 mapping")


def test_emc_unavailable_graceful_degradation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp), emc=False)
        result = run(fake, "--level", "20")
        require(result.returncode == 0, f"--target all must succeed without EMC: {result.stderr}")
        require("skipping EMC" in result.stderr, "must warn about missing EMC")
        require("gpu_target_freq_hz=" in result.stdout, "GPU must still work")
        require("cpu_target_freq_khz=" in result.stdout, "CPU must still work")


def test_cpu_multi_core_consistency() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp), cpu_count=6)
        result = run(fake, "--level", "50", "--target", "cpu")
        require(result.returncode == 0, result.stderr)
        require("cpu_count=6" in result.stdout, "must report all 6 cores")
        # During injection all cores should have been set to the same freq
        # After restore, all should be back
        for i in range(6):
            require(fake.cpu_read(i, "scaling_min_freq") == "729600",
                    f"cpu{i} min not restored")
            require(fake.cpu_read(i, "scaling_max_freq") == "1894400",
                    f"cpu{i} max not restored")


def test_safe_write_order_gpu() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        # Start GPU at 816/816
        (fake.gpu_dir / "min_freq").write_text("816000000\n")
        (fake.gpu_dir / "max_freq").write_text("816000000\n")
        result = run(fake, "--level", "0", "--duration", "0.01", "--target", "gpu")
        require(result.returncode == 0, result.stderr)
        writes = fake.write_log.read_text(encoding="utf-8").splitlines()
        gpu_writes = [w for w in writes if "min_freq" in w or "max_freq" in w]
        # Find injection writes (first occurrence)
        inject_min = gpu_writes.index("min_freq=306000000")
        inject_max = gpu_writes.index("max_freq=306000000")
        require(inject_min < inject_max, "lowering GPU bounds must write min before max")


def test_sigterm_restores_all_subsystems() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        proc = subprocess.Popen(
            ["bash", str(SCRIPT), "--level", "0", "--duration", "5",
             "--sysfs-root", str(fake.root)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=fake.env(no_sleep=False),
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and fake.gpu_read("max_freq") != "306000000":
            require(proc.poll() is None, "injector exited before applying the lock")
            time.sleep(0.01)
        require(fake.gpu_read("max_freq") == "306000000", "injector did not lock GPU before timeout")
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=3)
        require(proc.returncode != 0, "SIGTERM should produce a nonzero exit")
        # All subsystems restored
        require(fake.gpu_read("min_freq") == "306000000", f"GPU min not restored: {stdout} {stderr}")
        require(fake.gpu_read("max_freq") == "1020000000", f"GPU max not restored: {stdout} {stderr}")
        require(fake.gpu_scaling == "1", "GPU scaling not restored")
        for i in range(3):
            require(fake.cpu_read(i, "scaling_max_freq") == "1894400",
                    f"cpu{i} max not restored after SIGTERM")
            require(fake.cpu_read(i, "scaling_governor") == "schedutil",
                    f"cpu{i} governor not restored after SIGTERM")
        require(fake.emc_read("max_rate") == "2133000000", "EMC max not restored after SIGTERM")


def test_invalid_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        for args in (("--level", "-1"), ("--level", "101"), ("--level", "abc"),
                     ("--level", "20", "--duration", "0"),
                     ("--level", "20", "--duration", "abc"),
                     ("--level", "20", "--target", "invalid")):
            result = run(fake, *args)
            require(result.returncode != 0, f"invalid input unexpectedly succeeded: {args}")


def test_root_check_and_lock_contention() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--level", "20", euid=1000)
        require(result.returncode != 0, "non-root injection must fail")
        require("root" in result.stderr.lower(), "root failure should explain the requirement")

        with fake.lock_file.open("w", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run(fake, "--level", "20")
            require(result.returncode != 0, "lock contention must fail")
            require("already running" in result.stderr, "lock failure should identify contention")


def test_restore_failures_print_manual_recovery() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        (fake.gpu_dir / "min_freq").chmod(0o444)
        result = run(fake, "--level", "20", "--target", "gpu")
        require(result.returncode != 0, "a sysfs write failure must fail the command")
        require("restoration failed" in result.stderr.lower(),
                "a restoration write failure must be reported")
        require("manual recovery" in result.stderr.lower(),
                "a restoration write failure must print manual recovery commands")


def test_partial_target_gpu_cpu() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--level", "20", "--target", "gpu,cpu")
        require(result.returncode == 0, result.stderr)
        require("gpu_target_freq_hz=" in result.stdout, "must report GPU target")
        require("cpu_target_freq_khz=" in result.stdout, "must report CPU target")
        require("emc_target_freq=" not in result.stdout, "must NOT report EMC target with gpu,cpu")
        # EMC untouched
        require(fake.emc_read("max_rate") == "2133000000", "EMC must not be modified")


def test_jetpack_5_gpu_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp), gpu_layout="r35")
        result = run(fake, "--level", "20", "--target", "gpu")
        require(result.returncode == 0, result.stderr)
        require("gpu_target_freq_hz=408000000" in result.stdout,
                "JetPack 5 ga10b layout must use the same level mapping")
        require(fake.gpu_read("min_freq") == "306000000", "r35 min not restored")
        require(fake.gpu_read("max_freq") == "1020000000", "r35 max not restored")


def test_readme_documents_usage() -> None:
    text = README.read_text(encoding="utf-8")
    for token in ("jetson_freq_inject.sh", "--target"):
        require(token in text, f"README must document {token!r}")


def main() -> int:
    require(SCRIPT.exists(), f"missing script: {SCRIPT}")
    test_help_and_list()
    test_default_lock_uses_root_owned_directory()
    test_gpu_only_inject_and_restore()
    test_cpu_only_inject_and_restore()
    test_emc_only_inject_and_restore()
    test_bpmp_emc_uses_rate_not_read_only_bounds()
    test_all_combined_inject_and_restore()
    test_level_mapping_per_subsystem()
    test_emc_unavailable_graceful_degradation()
    test_cpu_multi_core_consistency()
    test_safe_write_order_gpu()
    test_sigterm_restores_all_subsystems()
    test_invalid_inputs()
    test_root_check_and_lock_contention()
    test_restore_failures_print_manual_recovery()
    test_partial_target_gpu_cpu()
    test_jetpack_5_gpu_layout()
    test_readme_documents_usage()
    print("PASS: Jetson combined frequency injector semantic tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
