#!/usr/bin/env python3
"""EdgeVisor PP 层迁移 / 算力扰动实验数据记录脚本。

从 dllama root 进程的 stdout 日志（如 root_gpu0.log）中解析一次完整实验，
产出两个文件：

  * <name>_<时间戳>_tokens.csv —— 逐 token 时序表（e2e wall、各 node/stage exec、
    bubble-shadow-kv 指标、阶段/事件标注），供后续画时序图。
  * <name>_<时间戳>.json       —— 结构化实验数据：记录窗口决策（起点/回落/终点）、
    全部迁移事件、逐 token 数组、日志尾部总数据（Evaluation/Prediction、
    Stage/Node Profile Summary、Migration TPOT Summary）。

记录窗口语义
------------
  起点(onset)：推理速度明显劣化的第一个 token。判定优先级：
      1. --onset-pos 手动指定；
      2. 日志中 [Migration TPOT] 行在 onset_detected=yes 时给出的 before=[X,..) 的 X；
      3. 自动检测：对 [token-e2e] wall 序列做 3-token 滑动窗口跳变检测
         （近窗均值 > 基窗均值 × (1 + jump%) 且持续，取最早的持续跳变点，
         与 C++ 侧 Solution A 一致）。
  基线段：起点之前额外记录 --baseline-tokens 个稳定 token（默认 20）。
  迁移事件：全部记录（route、层数、层列表、armed pos、anchor、recover stall 等），
      记录窗口按第一次迁移计算。
  终点：第一次迁移 anchor 之后，当 wall 较劣化段峰值回落 ≥ --recovery-drop-pct%
      （默认 15%）并连续 --recovery-confirm 个 token（默认 2）保持，即认定迁移生效；
      首个生效 token 计为第 1 个，记录到第 --tokens-after 个（默认 16）为止。
      若到日志结尾仍未回落，则记录到日志结尾并告警。
  无迁移（纯扰动对照实验）：从基线段记录到日志结尾，JSON 中 migrations 为空。

用法
----
  python3 record_migration_experiment.py path/to/root_gpu0.log
  python3 record_migration_experiment.py - < root_gpu0.log          # stdin
  dllama inference ... 2>&1 | python3 record_migration_experiment.py - --name disturb_run1
  python3 record_migration_experiment.py run.log --onset-pos 233 --meta disturbance=gpu_node2
"""

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# 日志行正则
# ---------------------------------------------------------------------------

RE_E2E = re.compile(
    r"\[token-e2e\] pos=(\d+) token=(-?\d+) wall=([\d.]+)\s*ms")

RE_PROF = re.compile(
    r"\[token-prof\] pos=(\d+) node=(\d+) stage=(\S+) "
    r"total=([\d.]+)ms exec=([\d.]+)ms sync=([\d.]+)ms bubble=([\d.]+)ms")

RE_BUBBLE = re.compile(
    r"\[bubble-shadow-kv\] who=(\S+) node=(\d+) pos=(\d+) batch=(\d+) mode=(\S+) "
    r"segments=(\d+) attn=(\d+) ffn=(\d+) other=(\d+) layers=(\d+) ops=(\d+) "
    r"skipped_sync=(\d+) budget_hit=(\d+) completed=(\d+) "
    r"drain_us=(\d+) elapsed_us=(\d+)")

RE_APPLY = re.compile(
    r"\[kv-migrate\] apply pp command cacheSeq=(\d+) layerCount=(\d+) "
    r"route=(\d+)->(\d+) layers=\[([^\]]*)\]")

RE_REJECT = re.compile(
    r"\[kv-migrate\] reject pp command route=(\d+)->(\d+) reason=(.*)$")

RE_ARMED = re.compile(
    r"\[kv-migrate\] auto collect armed layer=(-?\d+) pos=(-?\d+) mode=(\d+) "
    r"route=(\d+)->(\d+) layers=(\d+)")

RE_RECOVER = re.compile(
    r"\[kv-migrate\] recover mode=(\S+) status=(\S+) stallMs=([\d.]+) "
    r"stateBytes=(\d+) recomputeUnits=(\d+) layers=(\d+) posRange=\[(\d+),(\d+)\]")

RE_ANCHOR = re.compile(
    r"\[migrate-prof\] anchor layer=(\d+) pos=(\d+) window=(\d+)-before/(\d+)-after")

RE_TPOT_INLINE = re.compile(
    r"\[Migration TPOT\] anchor\(layer=(\d+) pos=(\d+)\) "
    r"before=\[(\d+),(\d+)\)\((\d+) tokens\) after=\[(\d+),(\d+)\)\((\d+) tokens\) "
    r"onset_detected=(\S+)")

RE_TPOT_MAIN = re.compile(
    r"TPOT\(stage-sum\):\s*before=([\d.]+)\s*ms after=([\d.]+)\s*ms "
    r"delta=([+-]?[\d.]+)\s*ms improve=([+-]?[\d.]+)%")

RE_TPOT_COMPUTE_INLINE = re.compile(
    r"compute-only before=([\d.]+)\s*ms after=([\d.]+)\s*ms "
    r"delta=([+-]?[\d.]+)\s*ms improve=([+-]?[\d.]+)%\s*\|\s*samples=(\d+)/(\d+)")

RE_TPOT_NODE = re.compile(
    r"Node (\d+): before=([\d.]+) ms \(exec=\s*([\d.]+) sync=\s*([\d.]+) "
    r"bubble=\s*([\d.]+)\) "
    r"\| after=([\d.]+) ms \(exec=\s*([\d.]+) sync=\s*([\d.]+) bubble=\s*([\d.]+)\) "
    r"\| Δ=\s*([+-]?[\d.]+) ms")

RE_STAGE_PROFILE = re.compile(
    r"Stage (\d+) Node (\d+): per-fwd total=([\d.]+) ms "
    r"\(exec=\s*([\d.]+) sync=\s*([\d.]+) bubble=\s*([\d.]+)\) \| "
    r"per-tok total=([\d.]+) ms "
    r"\(exec=\s*([\d.]+) sync=\s*([\d.]+) bubble=\s*([\d.]+)\) \| "
    r"bubbleSeg=(\d+) bubbleOps=(\d+) fwd=(\d+) tok=(\d+)")

RE_STAGE_PROFILE_SYNC = re.compile(
    r"sync/fwd: ppSend=\s*([\d.]+) ppRecv=\s*([\d.]+) rootWait=\s*([\d.]+) "
    r"logits=\s*([\d.]+) other=\s*([\d.]+) ms \| bubbleDrain/fwd=\s*([\d.]+) ms "
    r"complete=(\d+)/(\d+) skippedSyncs=(\d+)")

RE_MIG_SUMMARY_HEAD = re.compile(
    r"\[Migration TPOT Summary\]")

RE_MIG_SUMMARY_ANCHOR = re.compile(
    r"anchor layer=(\d+) pos=(\d+) window=(\d+) tokens "
    r"before=\[(\d+),(\d+)\) after=\[(\d+),(\d+)\) samples=(\d+)/(\d+)")

RE_MIG_SUMMARY_COMPUTE = re.compile(
    r"compute-only:\s*before=([\d.]+)\s*ms after=([\d.]+)\s*ms "
    r"delta=([+-]?[\d.]+)\s*ms improve=([+-]?[\d.]+)%")

RE_STAGE_SUMMARY_HEADER = re.compile(r"\[Stage/Node Profile Summary\]")

SECTION_HEADERS = (
    "Evaluation",
    "Evaluation (root wall-clock)",
    "Prediction",
    "Prediction (root wall-clock)",
)
SECTION_KEYS = {
    "Evaluation": "evaluation",
    "Evaluation (root wall-clock)": "evaluation_root_wall",
    "Prediction": "prediction",
    "Prediction (root wall-clock)": "prediction_root_wall",
}

RE_SECTION_INT = re.compile(r"^\s*(nBatches|nTokens):\s*(\d+)\s*$")
RE_SECTION_TPS = re.compile(r"^\s*tokens/s:\s*([\d.]+)\s*\(([\d.]+)\s*ms/tok\)\s*$")


def _f(value):
    return float(value)


def _i(value):
    return int(value)


# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------

class ExperimentRecorder:
    """逐行吃入 root 日志，最终产出结构化实验记录。"""

    def __init__(self, args):
        self.args = args
        self.warnings = []

        # 逐 token 数据，按 pos 归并（日志里同一 pos 的多条行顺序不固定）
        self.tokens = {}          # pos -> record dict
        self.migrations = []      # 迁移事件（按发生顺序）
        self.rejected_commands = []

        # 尾部总数据
        self.sections = {}        # evaluation / prediction / ...
        self.stage_profile = []   # Stage/Node Profile Summary 条目
        self.migration_tpot_summary = None

        # 状态机
        self._current_section = None
        self._current_section_key = None
        self._current_stage_entry = None
        self._in_stage_summary = False
        self._in_mig_summary = False
        self._inline_tpot = None  # 当前 [Migration TPOT] 内联块
        self._system_onset = None  # 来自 [Migration TPOT] 的 onset

    # -- 数据归并 -----------------------------------------------------------

    def _token(self, pos):
        rec = self.tokens.get(pos)
        if rec is None:
            rec = {"pos": pos, "token": None, "e2e_wall_ms": None,
                   "stages": {}, "bubble_shadow": None}
            self.tokens[pos] = rec
        return rec

    def _last_migration(self, what):
        if not self.migrations:
            # armed/recover/anchor 可能先于 apply（如 env 触发），补一个合成事件
            self.warnings.append(
                "migration lifecycle line (%s) appeared before any "
                "'apply pp command'; created synthetic migration entry" % what)
            self.migrations.append({"index": 0, "apply": None})
        return self.migrations[-1]

    # -- 逐行解析 -----------------------------------------------------------

    def feed(self, line):
        line = line.rstrip("\n")

        m = RE_E2E.search(line)
        if m:
            pos, tok, wall = _i(m.group(1)), _i(m.group(2)), _f(m.group(3))
            rec = self._token(pos)
            rec["token"] = tok
            rec["e2e_wall_ms"] = wall
            return

        m = RE_PROF.search(line)
        if m:
            pos = _i(m.group(1))
            node = _i(m.group(2))
            stage_raw = m.group(3)
            stage = None if stage_raw == "unknown" else _i(stage_raw)
            rec = self._token(pos)
            rec["stages"][node] = {
                "node": node,
                "stage": stage,
                "exec_ms": _f(m.group(5)),
            }
            return

        m = RE_BUBBLE.search(line)
        if m:
            pos = _i(m.group(3))
            rec = self._token(pos)
            rec["bubble_shadow"] = {
                "node": _i(m.group(2)),
                "mode": m.group(5),
                "skipped_sync": _i(m.group(12)),
                "budget_hit": _i(m.group(13)),
                "completed": _i(m.group(14)),
                "drain_us": _i(m.group(15)),
                "elapsed_us": _i(m.group(16)),
            }
            return

        m = RE_APPLY.search(line)
        if m:
            layers = [int(x) for x in m.group(5).split(",") if x.strip()] \
                if m.group(5).strip() else []
            self.migrations.append({
                "index": len(self.migrations),
                "apply": {
                    "cache_seq": _i(m.group(1)),
                    "layer_count": _i(m.group(2)),
                    "route_from": _i(m.group(3)),
                    "route_to": _i(m.group(4)),
                    "layers": layers,
                },
            })
            return

        m = RE_REJECT.search(line)
        if m:
            self.rejected_commands.append({
                "route_from": _i(m.group(1)),
                "route_to": _i(m.group(2)),
                "reason": m.group(3).strip(),
            })
            return

        m = RE_ARMED.search(line)
        if m:
            mig = self._last_migration("auto collect armed")
            mig["armed"] = {
                "layer": _i(m.group(1)),
                "pos": _i(m.group(2)),
                "mode": _i(m.group(3)),
                "route_from": _i(m.group(4)),
                "route_to": _i(m.group(5)),
                "layers": _i(m.group(6)),
            }
            return

        m = RE_RECOVER.search(line)
        if m:
            mig = self._last_migration("recover")
            mig["recover"] = {
                "mode": m.group(1),
                "status": m.group(2),
                "stall_ms": _f(m.group(3)),
                "state_bytes": _i(m.group(4)),
                "recompute_units": _i(m.group(5)),
                "layers": _i(m.group(6)),
                "pos_range": [_i(m.group(7)), _i(m.group(8))],
            }
            return

        m = RE_ANCHOR.search(line)
        if m:
            mig = self._last_migration("migrate-prof anchor")
            mig["anchor"] = {
                "layer": _i(m.group(1)),
                "pos": _i(m.group(2)),
                "window_before": _i(m.group(3)),
                "window_after": _i(m.group(4)),
            }
            return

        m = RE_TPOT_INLINE.search(line)
        if m:
            self._inline_tpot = {
                "anchor_layer": _i(m.group(1)),
                "anchor_pos": _i(m.group(2)),
                "before_range": [_i(m.group(3)), _i(m.group(4))],
                "before_samples": _i(m.group(5)),
                "after_range": [_i(m.group(6)), _i(m.group(7))],
                "after_samples": _i(m.group(8)),
                "onset_detected": m.group(9),
                "nodes": [],
            }
            onset_flag = m.group(9)
            if onset_flag == "yes":
                self._system_onset = _i(m.group(3))
            if self.migrations:
                self.migrations[-1]["inline_tpot"] = self._inline_tpot
            return

        if self._inline_tpot is not None:
            m = RE_TPOT_MAIN.search(line)
            if m and "samples=" in line:
                self._inline_tpot["tpot_stage_sum"] = {
                    "before_ms": _f(m.group(1)),
                    "after_ms": _f(m.group(2)),
                    "delta_ms": _f(m.group(3)),
                    "improve_pct": _f(m.group(4)),
                }
                c = RE_TPOT_COMPUTE_INLINE.search(line)
                if c:
                    self._inline_tpot["tpot_compute_only"] = {
                        "before_ms": _f(c.group(1)),
                        "after_ms": _f(c.group(2)),
                        "delta_ms": _f(c.group(3)),
                        "improve_pct": _f(c.group(4)),
                        "samples": [_i(c.group(5)), _i(c.group(6))],
                    }
                return
            m = RE_TPOT_NODE.search(line)
            if m:
                self._inline_tpot["nodes"].append({
                    "node": _i(m.group(1)),
                    "before_total_ms": _f(m.group(2)),
                    "before_exec_ms": _f(m.group(3)),
                    "before_sync_ms": _f(m.group(4)),
                    "before_bubble_ms": _f(m.group(5)),
                    "after_total_ms": _f(m.group(6)),
                    "after_exec_ms": _f(m.group(7)),
                    "after_sync_ms": _f(m.group(8)),
                    "after_bubble_ms": _f(m.group(9)),
                    "delta_ms": _f(m.group(10)),
                })
                return
            if line.strip() and not line.startswith(" "):
                self._inline_tpot = None  # 内联块结束

        # ---- 尾部总数据 ----
        stripped = line.strip()

        if RE_MIG_SUMMARY_HEAD.search(line):
            self._in_mig_summary = True
            self._in_stage_summary = False
            self._current_stage_entry = None
            self.migration_tpot_summary = {}
            return

        if self._in_mig_summary:
            m = RE_MIG_SUMMARY_ANCHOR.search(line)
            if m:
                self.migration_tpot_summary.update({
                    "anchor_layer": _i(m.group(1)),
                    "anchor_pos": _i(m.group(2)),
                    "window_tokens": _i(m.group(3)),
                    "before_range": [_i(m.group(4)), _i(m.group(5))],
                    "after_range": [_i(m.group(6)), _i(m.group(7))],
                    "samples": [_i(m.group(8)), _i(m.group(9))],
                })
                return
            m = RE_TPOT_MAIN.search(line)
            if m:
                self.migration_tpot_summary["tpot_stage_sum"] = {
                    "before_ms": _f(m.group(1)),
                    "after_ms": _f(m.group(2)),
                    "delta_ms": _f(m.group(3)),
                    "improve_pct": _f(m.group(4)),
                }
                return
            m = RE_MIG_SUMMARY_COMPUTE.search(line)
            if m:
                self.migration_tpot_summary["tpot_compute_only"] = {
                    "before_ms": _f(m.group(1)),
                    "after_ms": _f(m.group(2)),
                    "delta_ms": _f(m.group(3)),
                    "improve_pct": _f(m.group(4)),
                }
                self._in_mig_summary = False
                return

        if RE_STAGE_SUMMARY_HEADER.search(line):
            self._in_stage_summary = True
            self._current_stage_entry = None
            return

        if self._in_stage_summary:
            m = RE_STAGE_PROFILE.search(line)
            if m:
                entry = {
                    "stage": _i(m.group(1)),
                    "node": _i(m.group(2)),
                    "per_fwd": {"total_ms": _f(m.group(3)), "exec_ms": _f(m.group(4)),
                                "sync_ms": _f(m.group(5)), "bubble_ms": _f(m.group(6))},
                    "per_tok": {"total_ms": _f(m.group(7)), "exec_ms": _f(m.group(8)),
                                "sync_ms": _f(m.group(9)), "bubble_ms": _f(m.group(10))},
                    "bubble_seg": _i(m.group(11)),
                    "bubble_ops": _i(m.group(12)),
                    "fwd": _i(m.group(13)),
                    "tok": _i(m.group(14)),
                    "sync": None,
                }
                self.stage_profile.append(entry)
                self._current_stage_entry = entry
                return
            m = RE_STAGE_PROFILE_SYNC.search(line)
            if m and self._current_stage_entry is not None:
                self._current_stage_entry["sync"] = {
                    "pp_send_ms": _f(m.group(1)),
                    "pp_recv_ms": _f(m.group(2)),
                    "root_wait_ms": _f(m.group(3)),
                    "logits_ms": _f(m.group(4)),
                    "other_ms": _f(m.group(5)),
                    "bubble_drain_ms": _f(m.group(6)),
                    "complete": [_i(m.group(7)), _i(m.group(8))],
                    "skipped_syncs": _i(m.group(9)),
                }
                self._current_stage_entry = None
                return
            if stripped.startswith("Hint:"):
                self._in_stage_summary = False
                return

        if stripped in SECTION_HEADERS:
            self._current_section = stripped
            self._current_section_key = SECTION_KEYS[stripped]
            self.sections.setdefault(self._current_section_key, {})
            return

        if self._current_section is not None:
            sec = self.sections[self._current_section_key]
            m = RE_SECTION_INT.search(line)
            if m:
                sec[m.group(1)] = _i(m.group(2))
                return
            m = RE_SECTION_TPS.search(line)
            if m:
                sec["tokens_per_s"] = _f(m.group(1))
                sec["ms_per_tok"] = _f(m.group(2))
                self._current_section = None  # tokens/s 是每个小节的最后一行
                return
            if stripped == "":
                return
            # 其它内容说明小节已结束
            self._current_section = None

    # -- 起点/回落判定 -------------------------------------------------------

    def _detect_onset_auto(self, walls):
        """walls: [(pos, wall_ms)] 按 pos 升序。返回 onset pos 或 None。

        与 C++ 侧 Solution A 相同：3-token 滑动窗口从后向前扫描，
        近窗均值 > 基窗均值 × (1+jump%) 视为跳变，取最早持续跳变点。
        """
        k = 3
        jump = 1.0 + self.args.jump_pct / 100.0
        n = len(walls)
        if n < 2 * k:
            return None
        onset_idx = None
        for i in range(n - k, k - 1, -1):
            recent = sum(w[1] for w in walls[i:i + k]) / k
            baseline = sum(w[1] for w in walls[i - k:i]) / k
            if baseline > 0.0 and recent > baseline * jump:
                onset_idx = i  # 继续向前找最早持续跳变
            elif onset_idx is not None:
                break
        return walls[onset_idx][0] if onset_idx is not None else None

    def _detect_recovery(self, onset_pos, anchor_pos):
        """从第一次迁移 anchor 之后扫描 wall，较劣化段峰值回落 >= drop% 且连续
        confirm 个 token 保持 -> (effective_pos, peak_ms, threshold_ms)。
        未确认返回 (None, peak_ms, threshold_ms)。"""
        ratio = 1.0 - self.args.recovery_drop_pct / 100.0
        walls = sorted((p, r["e2e_wall_ms"]) for p, r in self.tokens.items()
                       if r["e2e_wall_ms"] is not None and p >= onset_pos)
        peak = 0.0
        run = []  # 当前连续满足回落条件的 pos 列表（容忍中间缺 token）
        threshold = None
        for pos, wall in walls:
            peak = max(peak, wall)
            threshold = peak * ratio
            if pos <= anchor_pos:
                run = []
                continue
            if wall <= threshold:
                run.append(pos)
                if len(run) >= self.args.recovery_confirm:
                    return run[0], peak, threshold
            else:
                run = []
        return None, peak, threshold

    # -- 收尾 -----------------------------------------------------------------

    def finalize(self):
        args = self.args
        walls = sorted((p, r["e2e_wall_ms"]) for p, r in self.tokens.items()
                       if r["e2e_wall_ms"] is not None)

        # ---- 起点 ----
        onset_pos = None
        onset_source = None
        if args.onset_pos is not None:
            onset_pos = args.onset_pos
            onset_source = "manual(--onset-pos)"
            if onset_pos not in self.tokens:
                self.warnings.append(
                    "manual onset pos=%d has no [token-e2e] record; "
                    "window still computed relative to it" % onset_pos)
        elif self._system_onset is not None:
            onset_pos = self._system_onset
            onset_source = "system_tpot(onset_detected=yes)"
        else:
            onset_pos = self._detect_onset_auto(walls)
            if onset_pos is not None:
                onset_source = "auto(wall-jump %.0f%%)" % args.jump_pct
            elif walls:
                onset_pos = walls[0][0]
                onset_source = "fallback(first-token)"
                self.warnings.append(
                    "no degradation onset detected (no [Migration TPOT] with "
                    "onset_detected=yes, wall-jump scan found nothing); "
                    "using first token pos=%d as onset" % onset_pos)

        if onset_pos is None:
            self.warnings.append("no [token-e2e] data at all; nothing to record")
            return None

        # ---- 迁移事件相对位置 ----
        for mig in self.migrations:
            anchor = mig.get("anchor")
            armed = mig.get("armed")
            ref = (anchor or {}).get("pos", (armed or {}).get("pos"))
            mig["ref_pos"] = ref
            mig["rel_pos"] = (ref - onset_pos) if ref is not None else None

        first = self.migrations[0] if self.migrations else None
        anchor_pos = first["ref_pos"] if first else None

        # ---- 回落确认与终点 ----
        recovery = {"confirmed": False, "effective_pos": None, "peak_wall_ms": None,
                    "threshold_ms": None, "drop_pct": args.recovery_drop_pct,
                    "confirm_tokens": args.recovery_confirm}
        end_pos = None
        end_source = None
        if anchor_pos is not None:
            effective, peak, threshold = self._detect_recovery(onset_pos, anchor_pos)
            recovery["peak_wall_ms"] = round(peak, 2) if peak else None
            recovery["threshold_ms"] = round(threshold, 2) if threshold else None
            if effective is not None:
                recovery["confirmed"] = True
                recovery["effective_pos"] = effective
                end_pos = effective + args.tokens_after - 1
                end_source = ("recovery+%d(first-effective-counted-as-1)"
                              % args.tokens_after)
            else:
                self.warnings.append(
                    "migration recovery not confirmed (wall never dropped >=%.0f%% "
                    "from peak for %d consecutive tokens); recording to log end"
                    % (args.recovery_drop_pct, args.recovery_confirm))
        else:
            if not self.migrations:
                self.warnings.append(
                    "no migration events found (pure disturbance run?); "
                    "recording from baseline to log end")
            else:
                self.warnings.append(
                    "migration events found but no anchor/armed pos; "
                    "recording to log end")

        last_pos = walls[-1][0] if walls else onset_pos
        if end_pos is None or end_pos > last_pos:
            if end_pos is not None:
                eff = recovery["effective_pos"]
                eff = eff if eff is not None else end_pos
                self.warnings.append(
                    "end_pos=%d exceeds last parsed token pos=%d; truncated to "
                    "log end (only %d of requested %d tokens after effective)"
                    % (end_pos, last_pos,
                       max(last_pos - eff + 1, 0), args.tokens_after))
            end_pos = last_pos
            end_source = "log-end"

        baseline_start = onset_pos - args.baseline_tokens
        first_pos = walls[0][0] if walls else onset_pos
        if baseline_start < first_pos:
            self.warnings.append(
                "baseline truncated by log start: wanted %d tokens before onset "
                "(pos %d), log begins at pos %d"
                % (args.baseline_tokens, baseline_start, first_pos))

        # ---- 逐 token 输出 ----
        effective_pos = recovery["effective_pos"]
        token_rows = []
        for pos in sorted(p for p in self.tokens
                          if baseline_start <= p <= end_pos):
            rec = self.tokens[pos]
            if rec["e2e_wall_ms"] is None and not rec["stages"]:
                continue  # 仅有 bubble 等外围行的 pos（如窗口外的前瞻行）
            if pos < onset_pos:
                phase = "baseline"
            elif effective_pos is not None and pos >= effective_pos:
                phase = "recovered"
            else:
                phase = "degraded"
            events = []
            if pos == onset_pos:
                events.append("onset")
            for mig in self.migrations:
                if mig["ref_pos"] == pos:
                    apply = mig.get("apply") or {}
                    events.append("migration#%d:%s->%sxL%d" % (
                        mig["index"],
                        apply.get("route_from", "?"),
                        apply.get("route_to", "?"),
                        apply.get("layer_count",
                                  (mig.get("armed") or {}).get("layers", 0))))
            if pos == effective_pos:
                events.append("recovery")
            token_rows.append({
                "pos": pos,
                "rel_pos": pos - onset_pos,
                "phase": phase,
                "e2e_wall_ms": rec["e2e_wall_ms"],
                "stages": [rec["stages"][n] for n in sorted(rec["stages"])],
                "bubble_shadow": rec["bubble_shadow"],
                "events": events,
            })

        # ---- node->stage 映射（CSV 列注释用）----
        node_stage = {}
        for pos in sorted(self.tokens):
            for node, s in self.tokens[pos]["stages"].items():
                node_stage[node] = s["stage"]
        all_nodes = sorted(node_stage)

        window = {
            "onset_pos": onset_pos,
            "onset_source": onset_source,
            "system_tpot_onset": self._system_onset,
            "baseline_start": baseline_start,
            "baseline_tokens": args.baseline_tokens,
            "baseline_tokens_recorded": sum(
                1 for t in token_rows if t["phase"] == "baseline"),
            "end_pos": end_pos,
            "end_source": end_source,
            "recovery": recovery,
            "tokens_recorded": len(token_rows),
        }

        summary = {
            "sections": self.sections,
            "stage_node_profile": self.stage_profile,
            "migration_tpot_summary": self.migration_tpot_summary,
        }

        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "experiment": args.name,
            "meta": args.meta,
            "source": args.source_label,
            "params": {
                "baseline_tokens": args.baseline_tokens,
                "jump_pct": args.jump_pct,
                "recovery_drop_pct": args.recovery_drop_pct,
                "recovery_confirm": args.recovery_confirm,
                "tokens_after": args.tokens_after,
            },
            "warnings": self.warnings,
            "window": window,
            "migrations": self.migrations,
            "rejected_commands": self.rejected_commands,
            "columns": {
                "nodes": all_nodes,
                "node_stage": {str(k): v for k, v in node_stage.items()},
            },
            "tokens": token_rows,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_outputs(result, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "%s_%s" % (name, ts)
    json_path = os.path.join(outdir, base + ".json")
    csv_path = os.path.join(outdir, base + "_tokens.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    nodes = result["columns"]["nodes"]
    node_stage = result["columns"]["node_stage"]
    header = ["pos", "rel_pos", "phase", "e2e_wall_ms"]
    for n in nodes:
        s = node_stage.get(str(n))
        header.append("node%d_exec_ms%s" % (
            n, "" if s is None else ("_stage%d" % s)))
    header += ["bubble_drain_us", "bubble_elapsed_us", "bubble_completed", "events"]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in result["tokens"]:
            exec_by_node = {s["node"]: s["exec_ms"] for s in t["stages"]}
            row = [t["pos"], t["rel_pos"], t["phase"],
                   "" if t["e2e_wall_ms"] is None else ("%.2f" % t["e2e_wall_ms"])]
            for n in nodes:
                v = exec_by_node.get(n)
                row.append("" if v is None else ("%.2f" % v))
            b = t["bubble_shadow"]
            row += ["" if b is None else b["drain_us"],
                    "" if b is None else b["elapsed_us"],
                    "" if b is None else b["completed"],
                    ";".join(t["events"])]
            writer.writerow(row)

    return json_path, csv_path


def print_report(result, json_path, csv_path, quiet=False):
    if quiet:
        return
    w = result["window"]
    out = sys.stderr
    print("[recorder] experiment=%s source=%s"
          % (result["experiment"], result["source"]), file=out)
    print("[recorder] onset pos=%d (source: %s)"
          % (w["onset_pos"], w["onset_source"]), file=out)
    print("[recorder] baseline window [%d, %d), %d baseline tokens recorded"
          % (w["baseline_start"], w["onset_pos"],
             w["baseline_tokens_recorded"]), file=out)
    for mig in result["migrations"]:
        apply = mig.get("apply") or {}
        anchor = mig.get("anchor") or {}
        recover = mig.get("recover") or {}
        print("[recorder] migration#%d route=%s->%s layers=%s anchor_pos=%s "
              "rel_pos=%s recover=%s stall=%s"
              % (mig["index"],
                 apply.get("route_from", "?"), apply.get("route_to", "?"),
                 apply.get("layers", "?"),
                 anchor.get("pos", (mig.get("armed") or {}).get("pos", "?")),
                 mig.get("rel_pos", "?"),
                 recover.get("status", "?"),
                 recover.get("stall_ms", "?")), file=out)
    rec = w["recovery"]
    if rec["confirmed"]:
        print("[recorder] recovery confirmed @pos=%d (peak=%.2fms, "
              "threshold=%.2fms, drop>=%.0f%%), end_pos=%d"
              % (rec["effective_pos"], rec["peak_wall_ms"], rec["threshold_ms"],
                 rec["drop_pct"], w["end_pos"]), file=out)
    else:
        print("[recorder] recovery not confirmed; end_pos=%d (%s)"
              % (w["end_pos"], w["end_source"]), file=out)
    print("[recorder] tokens recorded: %d (pos range of output rows)"
          % w["tokens_recorded"], file=out)
    for warning in result["warnings"]:
        print("[recorder] WARN: %s" % warning, file=out)
    print("[recorder] csv  -> %s" % csv_path, file=out)
    print("[recorder] json -> %s" % json_path, file=out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_meta(values):
    meta = {}
    for item in values or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "--meta expects KEY=VALUE, got %r" % item)
        key, _, value = item.partition("=")
        meta[key.strip()] = value
    return meta


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="EdgeVisor PP 层迁移/算力扰动实验数据记录脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("log", help="root 进程日志文件路径，'-' 表示从 stdin 读取")
    p.add_argument("--name", default="pp_migration",
                   help="实验名，用于输出文件命名（默认 pp_migration）")
    p.add_argument("--outdir", default=".",
                   help="输出目录（默认当前目录）")
    p.add_argument("--onset-pos", type=int, default=None,
                   help="手动指定劣化起点 pos（优先级最高）")
    p.add_argument("--baseline-tokens", type=int, default=20,
                   help="起点前记录的稳定基线 token 数（默认 20）")
    p.add_argument("--jump-pct", type=float, default=15.0,
                   help="自动检测劣化跳变的滑动窗口涨幅阈值%%（默认 15）")
    p.add_argument("--recovery-drop-pct", type=float, default=15.0,
                   help="较劣化峰值回落多少%%认定迁移生效（默认 15）")
    p.add_argument("--recovery-confirm", type=int, default=2,
                   help="回落需连续保持的 token 数（默认 2）")
    p.add_argument("--tokens-after", type=int, default=16,
                   help="首个生效 token 计为第 1 个，记录到第 N 个为止（默认 16）")
    p.add_argument("--meta", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="实验元数据（扰动类型/强度/机器等），可重复")
    p.add_argument("--quiet", action="store_true", help="不打印人类可读报告")
    return p


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.meta = parse_meta(args.meta)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    recorder = ExperimentRecorder(args)

    if args.log == "-":
        args.source_label = "stdin"
        stream = sys.stdin
        for line in stream:
            recorder.feed(line)
    else:
        args.source_label = os.path.abspath(args.log)
        if not os.path.isfile(args.log):
            print("[recorder] ERROR: log file not found: %s" % args.log,
                  file=sys.stderr)
            return 2
        with open(args.log, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                recorder.feed(line)

    result = recorder.finalize()
    if result is None:
        print("[recorder] ERROR: %s" % "; ".join(recorder.warnings),
              file=sys.stderr)
        return 1

    json_path, csv_path = write_outputs(result, args.outdir, args.name)
    print_report(result, json_path, csv_path, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
