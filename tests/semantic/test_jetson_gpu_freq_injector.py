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
SCRIPT = ROOT / "EdgeVisor" / "scripts" / "jetson_gpu_freq_inject.sh"
README = ROOT / "EdgeVisor" / "README.md"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeSysfs:
    def __init__(
        self,
        root: Path,
        *,
        minimum: int = 306_000_000,
        maximum: int = 1_020_000_000,
        current: int = 612_000_000,
        layout: str = "r36",
    ) -> None:
        self.root = root / "sys"
        if layout == "r36":
            self.platform = self.root / "devices" / "platform" / "17000000.gpu"
            gpu_name = "17000000.gpu"
        elif layout == "r35":
            self.platform = self.root / "devices" / "17000000.ga10b"
            gpu_name = "17000000.ga10b"
        else:
            raise ValueError(f"unsupported fake layout: {layout}")
        self.gpu = self.platform / "devfreq" / gpu_name
        self.gpu.mkdir(parents=True)
        self.write("available_frequencies", "306000000 408000000 612000000 816000000 1020000000")
        self.write("min_freq", str(minimum))
        self.write("max_freq", str(maximum))
        self.write("cur_freq", str(current))
        (self.platform / "enable_3d_scaling").write_text("1\n", encoding="utf-8")
        self.lock_file = root / "injector.lock"
        self.write_log = root / "writes.log"

    def write(self, name: str, value: str) -> None:
        (self.gpu / name).write_text(value.rstrip("\n") + "\n", encoding="utf-8")

    def read(self, name: str) -> str:
        return (self.gpu / name).read_text(encoding="utf-8").strip()

    @property
    def scaling(self) -> str:
        return (self.platform / "enable_3d_scaling").read_text(encoding="utf-8").strip()

    def env(self, *, euid: int = 0, no_sleep: bool = True) -> dict[str, str]:
        env = os.environ.copy()
        env["JETSON_GPU_TEST_EUID"] = str(euid)
        env["JETSON_GPU_LOCK_FILE"] = str(self.lock_file)
        env["JETSON_GPU_WRITE_LOG"] = str(self.write_log)
        if no_sleep:
            env["JETSON_GPU_TEST_NO_SLEEP"] = "1"
        else:
            env.pop("JETSON_GPU_TEST_NO_SLEEP", None)
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


def test_help_and_list() -> None:
    help_result = subprocess.run(
        ["bash", str(SCRIPT), "--help"], text=True, capture_output=True, check=False
    )
    require(help_result.returncode == 0, help_result.stderr)
    require("default: 20" in help_result.stdout, "help must document the 20-second default")

    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--list", euid=1000)
        require(result.returncode == 0, result.stderr)
        require("306000000" in result.stdout and "1020000000" in result.stdout,
                "list must print available frequencies")
        require("current_freq_hz=612000000" in result.stdout, "list must print current frequency")


def test_level_mapping_and_normal_restore() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        result = run(fake, "--level", "20")
        require(result.returncode == 0, result.stderr)
        require("target_freq_hz=408000000" in result.stdout, "level 20 must select 408 MHz")
        require("duration_sec=20" in result.stdout, "omitted duration must default to 20 seconds")
        require(fake.read("min_freq") == "306000000", "minimum frequency was not restored")
        require(fake.read("max_freq") == "1020000000", "maximum frequency was not restored")
        require(fake.scaling == "1", "3D scaling state was not restored")

        result = run(fake, "--level", "100", "--duration", "0.01")
        require(result.returncode == 0, result.stderr)
        require("target_freq_hz=1020000000" in result.stdout, "level 100 must select current max")


def test_jetpack_5_ga10b_layout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp), layout="r35")
        result = run(fake, "--level", "20")
        require(result.returncode == 0, result.stderr)
        require("target_freq_hz=408000000" in result.stdout,
                "JetPack 5 ga10b layout must use the same level mapping")
        require(fake.read("min_freq") == "306000000", "r35 minimum was not restored")
        require(fake.read("max_freq") == "1020000000", "r35 maximum was not restored")
        require(fake.scaling == "1", "r35 3D scaling state was not restored")


def test_safe_write_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp), minimum=816_000_000, maximum=816_000_000, current=816_000_000)
        result = run(fake, "--level", "0", "--duration", "0.01")
        require(result.returncode == 0, result.stderr)
        writes = fake.write_log.read_text(encoding="utf-8").splitlines()
        inject_min = writes.index("min_freq=306000000")
        inject_max = writes.index("max_freq=306000000")
        restore_max = writes.index("max_freq=816000000", inject_max + 1)
        restore_min = writes.index("min_freq=816000000", restore_max + 1)
        require(inject_min < inject_max, "lowering bounds must write min before max")
        require(restore_max < restore_min, "raising bounds must write max before min")


def test_sigterm_restores_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        proc = subprocess.Popen(
            ["bash", str(SCRIPT), "--level", "0", "--duration", "5", "--sysfs-root", str(fake.root)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=fake.env(no_sleep=False),
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and fake.read("max_freq") != "306000000":
            require(proc.poll() is None, "injector exited before applying the lock")
            time.sleep(0.01)
        require(fake.read("max_freq") == "306000000", "injector did not lock before timeout")
        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=3)
        require(proc.returncode != 0, "SIGTERM should produce a nonzero exit")
        require(fake.read("min_freq") == "306000000", f"minimum not restored after SIGTERM: {stdout} {stderr}")
        require(fake.read("max_freq") == "1020000000", f"maximum not restored after SIGTERM: {stdout} {stderr}")
        require(fake.scaling == "1", "scaling not restored after SIGTERM")


def test_invalid_inputs_and_missing_sysfs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        for args in (("--level", "-1"), ("--level", "101"), ("--level", "abc"),
                     ("--level", "20", "--duration", "0"),
                     ("--level", "20", "--duration", "abc")):
            result = run(fake, *args)
            require(result.returncode != 0, f"invalid input unexpectedly succeeded: {args}")

        (fake.gpu / "available_frequencies").unlink()
        result = run(fake, "--level", "20")
        require(result.returncode != 0, "missing available_frequencies must fail")


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
        (fake.gpu / "min_freq").chmod(0o444)
        result = run(fake, "--level", "20")
        require(result.returncode != 0, "a sysfs write failure must fail the command")
        require("automatic restoration failed" in result.stderr,
                "a restoration write failure must not be masked by later writes")
        require("Manual recovery:" in result.stderr,
                "a restoration write failure must print manual recovery commands")

    with tempfile.TemporaryDirectory() as tmp:
        fake = FakeSysfs(Path(tmp))
        proc = subprocess.Popen(
            ["bash", str(SCRIPT), "--level", "0", "--duration", "5", "--sysfs-root", str(fake.root)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=fake.env(no_sleep=False),
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and fake.read("max_freq") != "306000000":
            require(proc.poll() is None, "injector exited before applying the lock")
            time.sleep(0.01)
        require(fake.read("max_freq") == "306000000", "injector did not lock before timeout")
        (fake.gpu / "min_freq").unlink()
        proc.send_signal(signal.SIGTERM)
        _, stderr = proc.communicate(timeout=3)
        require("automatic restoration failed" in stderr,
                "a missing sysfs node during restoration must be reported")
        require("Manual recovery:" in stderr,
                "a restoration read failure must print manual recovery commands")


def test_readme_documents_usage() -> None:
    text = README.read_text(encoding="utf-8")
    for token in ("jetson_gpu_freq_inject.sh", "--level 20", "--duration 8", "thermal"):
        require(token in text, f"README must document {token!r}")


def main() -> int:
    require(SCRIPT.exists(), f"missing script: {SCRIPT}")
    test_help_and_list()
    test_level_mapping_and_normal_restore()
    test_jetpack_5_ga10b_layout()
    test_safe_write_order()
    test_sigterm_restores_state()
    test_invalid_inputs_and_missing_sysfs()
    test_root_check_and_lock_contention()
    test_restore_failures_print_manual_recovery()
    test_readme_documents_usage()
    print("PASS: Jetson GPU frequency injector semantic tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
