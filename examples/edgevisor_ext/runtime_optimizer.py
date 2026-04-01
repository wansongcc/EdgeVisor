#!/usr/bin/env python3
import argparse
import json
import os
import socket
import time
from dataclasses import dataclass

from schemas import (
    validate_layer_prof_response,
    validate_perf_response,
    validate_plan_snapshot_response,
    validate_status_response,
)


class UdsSocketClient:
    def __init__(self, socket_path, timeout_s=2.0):
        self.socket_path = socket_path
        self.timeout_s = timeout_s

    def request(self, payload):
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(self.timeout_s)
            s.connect(self.socket_path)
            s.sendall(data)
            buf = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    buf = buf.split(b"\n", 1)[0]
                    break
        if not buf:
            raise RuntimeError("empty UDS response")
        return json.loads(buf.decode("utf-8"))


@dataclass
class OptimizerConfig:
    min_improve_ratio: float = 0.05
    layer_no_improve_epochs: int = 3
    cooldown_steps: int = 2
    timeout_steps: int = 8
    backoff_base_steps: int = 2
    max_backoff_steps: int = 20
    seq_start: int = 1


@dataclass
class InFlight:
    kind: str
    seq: int
    baseline_metric: float
    age_steps: int = 0


class RuntimeOptimizer:
    MODE_LAYER = "LAYER_MODE"
    MODE_TOKEN = "TOKEN_MODE"

    def __init__(self, client, cfg):
        self.client = client
        self.cfg = cfg
        self.mode = self.MODE_LAYER
        self.seq = max(1, int(cfg.seq_start))
        self.in_flight = None
        self.no_improve_count = 0
        self.backoff_steps = 0
        self.consecutive_failures = 0
        self._warned_embedded_dyn = False
        self._step = 0
        self._last_apply_ts = 0

    def _log(self, event, **kwargs):
        payload = {"ts": time.time(), "event": event, "mode": self.mode, "step": self._step}
        payload.update(kwargs)
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _is_truthy_env(name):
        v = os.getenv(name, "")
        if not v:
            return False
        return v.lower() not in ("0", "false", "no", "off")

    @staticmethod
    def _stage_times_from_perf(perf_items):
        stage_times = {}
        for item in perf_items:
            try:
                st = int(item.get("stageIndex", 0))
                score = float(item.get("execUs", 0)) + float(item.get("syncUs", 0))
            except Exception:
                continue
            stage_times[st] = max(stage_times.get(st, 0.0), score)
        return stage_times

    @staticmethod
    def _find_bottleneck_stage(stage_times):
        if not stage_times:
            return None
        return max(stage_times, key=lambda s: stage_times[s])

    @staticmethod
    def _stage_by_index(plan_snapshot):
        stages = plan_snapshot.get("stages", [])
        return {int(s["stageIndex"]): s for s in stages if isinstance(s, dict) and "stageIndex" in s}

    @staticmethod
    def _split_field(snapshot, split_name, field):
        splits = snapshot.get("splits", {})
        split = splits.get(split_name, {})
        arr = split.get(field)
        return arr if isinstance(arr, list) else None

    @staticmethod
    def _pick_hot_layer(layer_prof):
        layers = layer_prof.get("layers", [])
        best_layer = None
        best_score = -1
        for li, row in enumerate(layers):
            if not isinstance(row, list):
                continue
            row_best = -1
            for cell in row:
                if not isinstance(cell, dict) or not cell.get("ok"):
                    continue
                score = int(cell.get("attnUs", 0)) + int(cell.get("ffnUs", 0))
                row_best = max(row_best, score)
            if row_best > best_score:
                best_score = row_best
                best_layer = li
        return best_layer

    @staticmethod
    def _rebalance_head_moves(devices):
        # devices ordered by stage rank:
        # {node, t_ms, cur_start, cur_len, kv_start, kv_len}
        if len(devices) < 2:
            return []

        src_idx = max(range(len(devices)), key=lambda i: devices[i]["t_ms"])
        src = devices[src_idx]
        current_heads = int(src["cur_len"])
        if current_heads <= 1:
            return []

        t_src_unit = src["t_ms"] / current_heads
        max_movable = current_heads - 1
        move_left = 0
        move_right = 0

        if src_idx > 0:
            left = devices[src_idx - 1]
            l_heads = int(left["cur_len"])
            t_left_unit = (left["t_ms"] / l_heads) if l_heads > 0 else t_src_unit
            if src["t_ms"] > left["t_ms"]:
                denom = t_src_unit + t_left_unit
                if denom > 0:
                    move_left = int((src["t_ms"] - left["t_ms"]) // denom)

        if src_idx < len(devices) - 1:
            right = devices[src_idx + 1]
            r_heads = int(right["cur_len"])
            t_right_unit = (right["t_ms"] / r_heads) if r_heads > 0 else t_src_unit
            if src["t_ms"] > right["t_ms"]:
                denom = t_src_unit + t_right_unit
                if denom > 0:
                    move_right = int((src["t_ms"] - right["t_ms"]) // denom)

        move_left = max(0, move_left)
        move_right = max(0, move_right)
        if move_left + move_right > max_movable:
            total_req = float(move_left + move_right)
            ratio = float(max_movable) / total_req if total_req > 0 else 0.0
            move_left = int(move_left * ratio)
            move_right = int(move_right * ratio)
            while move_left + move_right > max_movable:
                if move_left >= move_right and move_left > 0:
                    move_left -= 1
                elif move_right > 0:
                    move_right -= 1
                else:
                    break

        if move_left > 0 and src_idx > 0:
            left = devices[src_idx - 1]
            current_left_end = int(left["cur_start"]) + int(left["cur_len"]) - 1
            max_possible = int(left["kv_start"]) + int(left["kv_len"]) - 1 - current_left_end
            move_left = min(move_left, max(0, max_possible))

        if move_right > 0 and src_idx < len(devices) - 1:
            right = devices[src_idx + 1]
            current_right_start = int(right["cur_start"])
            max_possible = current_right_start - int(right["kv_start"])
            move_right = min(move_right, max(0, max_possible))

        moves = []
        if move_left > 0 and src_idx > 0:
            moves.append(
                {
                    "fromNodeIndex": int(src["node"]),
                    "toNodeIndex": int(devices[src_idx - 1]["node"]),
                    "cmdKind": 1,
                    "headMove": int(move_left),
                    "ffnMove": 0,
                }
            )
        if move_right > 0 and src_idx < len(devices) - 1:
            moves.append(
                {
                    "fromNodeIndex": int(src["node"]),
                    "toNodeIndex": int(devices[src_idx + 1]["node"]),
                    "cmdKind": 1,
                    "headMove": int(move_right),
                    "ffnMove": 0,
                }
            )
        return moves

    def _decide_layer_command(self, stage_times, plan_snapshot):
        stage_map = self._stage_by_index(plan_snapshot)
        bottleneck_stage = self._find_bottleneck_stage(stage_times)
        if bottleneck_stage is None or bottleneck_stage not in stage_map:
            return None

        candidates = []
        for delta in (-1, 1):
            sid = bottleneck_stage + delta
            if sid in stage_map and sid in stage_times:
                candidates.append(sid)
        if not candidates:
            return None

        target_stage = min(candidates, key=lambda sid: stage_times[sid])
        if stage_times[target_stage] >= stage_times[bottleneck_stage]:
            return None

        from_node = int(stage_map[bottleneck_stage]["rootNodeIndex"])
        to_node = int(stage_map[target_stage]["rootNodeIndex"])
        req = {
            "op": "set_pp_migration",
            "cmd": {
                "seq": int(self.seq),
                "mode": "next_barrier",
                "stageIndex": 0xFFFFFFFF,
                "fromNodeIndex": from_node,
                "toNodeIndex": to_node,
                "layerCount": 1,
            },
        }
        return {"kind": "layer", "request": req, "from": from_node, "to": to_node, "stage": bottleneck_stage}

    def _decide_token_command(self, stage_times, plan_snapshot):
        stage_map = self._stage_by_index(plan_snapshot)
        stage_candidates = [sid for sid in stage_times if sid in stage_map]
        if not stage_candidates:
            return None
        stage_candidates.sort(key=lambda sid: stage_times[sid], reverse=True)

        kv_starts = self._split_field(plan_snapshot, "kvHeadSplit", "starts")
        kv_lens = self._split_field(plan_snapshot, "kvHeadSplit", "lengths")
        kvc_starts = self._split_field(plan_snapshot, "kvHeadComputeSplit", "starts")
        kvc_lens = self._split_field(plan_snapshot, "kvHeadComputeSplit", "lengths")
        if not all(isinstance(x, list) for x in (kv_starts, kv_lens, kvc_starts, kvc_lens)):
            return None

        for candidate_stage in stage_candidates:
            st = stage_map[candidate_stage]
            stage_nodes = [int(x) for x in st.get("nodes", [])]
            if len(stage_nodes) < 2:
                continue

            try:
                lp_resp = self.client.request(
                    {
                        "op": "layer_prof",
                        "all": True,
                        "stageIndex": int(st["stageIndex"]),
                        "rootNodeIndex": int(st["rootNodeIndex"]),
                    }
                )
                validate_layer_prof_response(lp_resp)
            except Exception:
                continue

            lp = lp_resp["layer_prof"]
            hot_layer = self._pick_hot_layer(lp)
            if hot_layer is None:
                continue

            layers = lp.get("layers", [])
            row = layers[hot_layer] if hot_layer < len(layers) else []
            score_by_node = {}
            for cell in row:
                if isinstance(cell, dict) and cell.get("ok"):
                    node = int(cell.get("nodeIndex", -1))
                    score = int(cell.get("attnUs", 0)) + int(cell.get("ffnUs", 0))
                    score_by_node[node] = score

            devices = []
            for node in stage_nodes:
                if node >= len(kv_starts) or node >= len(kv_lens) or node >= len(kvc_starts) or node >= len(kvc_lens):
                    continue
                cur_len = int(kv_lens[node])
                kv_len = int(kvc_lens[node])
                if cur_len <= 0 or kv_len <= 0:
                    continue
                t_ms = float(score_by_node.get(node, 0)) / 1000.0
                devices.append(
                    {
                        "node": node,
                        "t_ms": t_ms,
                        "cur_start": int(kv_starts[node]),
                        "cur_len": cur_len,
                        "kv_start": int(kvc_starts[node]),
                        "kv_len": kv_len,
                    }
                )

            moves = self._rebalance_head_moves(devices)
            if not moves:
                continue

            req = {
                "op": "set_plan",
                "cmd": {
                    "seq": int(self.seq),
                    "mode": "next_barrier",
                    "stageIndex": int(st["stageIndex"]),
                    "moves": moves,
                },
            }
            return {"kind": "token", "request": req, "stage": candidate_stage, "layer": hot_layer, "moves": moves}
        return None

    def _send_decision(self, decision, baseline_metric):
        resp = self.client.request(decision["request"])
        if not resp.get("ok"):
            raise RuntimeError(f"command rejected: {resp}")
        self.in_flight = InFlight(kind=decision["kind"], seq=self.seq, baseline_metric=baseline_metric)
        self._log("command_sent", decision=decision, response=resp)
        self.seq += 1

    def _handle_in_flight(self, metric):
        self.in_flight.age_steps += 1
        if self.in_flight.age_steps >= self.cfg.timeout_steps:
            try:
                self.client.request({"op": "clear"})
            except Exception:
                pass
            self.consecutive_failures += 1
            self.backoff_steps = min(self.cfg.max_backoff_steps, self.cfg.backoff_base_steps * (2 ** (self.consecutive_failures - 1)))
            self._log("in_flight_timeout", seq=self.in_flight.seq, backoff_steps=self.backoff_steps)
            self.in_flight = None
            return

        if self.in_flight.age_steps < self.cfg.cooldown_steps:
            self._log("cooldown_wait", seq=self.in_flight.seq, age_steps=self.in_flight.age_steps)
            return

        baseline = max(self.in_flight.baseline_metric, 1.0)
        improve = (baseline - metric) / baseline
        success = improve >= self.cfg.min_improve_ratio
        self._log("in_flight_result", seq=self.in_flight.seq, improve=improve, success=success)

        if self.in_flight.kind == "layer":
            if success:
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= self.cfg.layer_no_improve_epochs and self.mode == self.MODE_LAYER:
                    self.mode = self.MODE_TOKEN
                    self._log("mode_switch", reason="layer_no_improve", no_improve_count=self.no_improve_count)

        # "No improvement" is not treated as transport/controller failure.
        self.consecutive_failures = 0
        self.in_flight = None

    def step_once(self):
        self._step += 1

        status = self.client.request({"op": "status"})
        validate_status_response(status)
        if not status.get("enablePlanBarrier", False):
            raise RuntimeError("enablePlanBarrier=false; runtime optimizer requires --enable-plan-barrier")

        # best-effort observability: recently applied/rejected command summary
        try:
            last_apply_resp = self.client.request({"op": "last_apply"})
            if last_apply_resp.get("ok"):
                la = last_apply_resp.get("last_apply", {})
                ts = int(la.get("tsMs", 0))
                if bool(la.get("valid")) and ts > self._last_apply_ts:
                    self._last_apply_ts = ts
                    self._log("last_apply", summary=la)
        except Exception:
            pass

        if self._is_truthy_env("DLLAMA_DYNAMIC_LAYER_ENABLE") and not self._warned_embedded_dyn:
            self._log("warn", message="DLLAMA_DYNAMIC_LAYER_ENABLE is set; disable embedded dynamic thread to avoid dual controllers")
            self._warned_embedded_dyn = True

        perf = self.client.request({"op": "perf"})
        validate_perf_response(perf)
        raw_stage_times = self._stage_times_from_perf(perf.get("perf", []))

        snapshot_resp = self.client.request({"op": "plan_snapshot"})
        validate_plan_snapshot_response(snapshot_resp)
        snapshot = snapshot_resp["plan_snapshot"]
        valid_stage_map = self._stage_by_index(snapshot)
        stage_times = {sid: v for sid, v in raw_stage_times.items() if sid in valid_stage_map}
        if not stage_times:
            self._log("skip", reason="no_stage_times")
            return
        metric = max(stage_times.values())

        if self.in_flight is not None:
            self._handle_in_flight(metric)
            return

        if self.backoff_steps > 0:
            self.backoff_steps -= 1
            self._log("backoff", remaining=self.backoff_steps)
            return

        if self.mode == self.MODE_LAYER:
            decision = self._decide_layer_command(stage_times, snapshot)
        else:
            decision = self._decide_token_command(stage_times, snapshot)

        if decision is None:
            self._log("no_action", stage_times=stage_times)
            return

        try:
            self._send_decision(decision, baseline_metric=metric)
        except Exception as e:
            self.consecutive_failures += 1
            self.backoff_steps = min(self.cfg.max_backoff_steps, self.cfg.backoff_base_steps * (2 ** (self.consecutive_failures - 1)))
            self._log("send_failed", error=str(e), backoff_steps=self.backoff_steps)


def main():
    ap = argparse.ArgumentParser(description="EdgeVisor runtime optimizer (Layer -> Token fallback).")
    ap.add_argument("socket", help="UDS path, e.g. /tmp/dllama_plan.sock")
    ap.add_argument("--poll-ms", type=int, default=500)
    ap.add_argument("--min-improve-ratio", type=float, default=0.05)
    ap.add_argument("--layer-no-improve-epochs", type=int, default=3)
    ap.add_argument("--cooldown-steps", type=int, default=2)
    ap.add_argument("--timeout-steps", type=int, default=8)
    ap.add_argument("--backoff-base-steps", type=int, default=2)
    ap.add_argument("--max-backoff-steps", type=int, default=20)
    ap.add_argument("--seq-start", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 means run forever")
    ap.add_argument("--uds-timeout-s", type=float, default=2.0)
    args = ap.parse_args()

    cfg = OptimizerConfig(
        min_improve_ratio=args.min_improve_ratio,
        layer_no_improve_epochs=args.layer_no_improve_epochs,
        cooldown_steps=args.cooldown_steps,
        timeout_steps=args.timeout_steps,
        backoff_base_steps=args.backoff_base_steps,
        max_backoff_steps=args.max_backoff_steps,
        seq_start=args.seq_start,
    )
    optimizer = RuntimeOptimizer(UdsSocketClient(args.socket, timeout_s=args.uds_timeout_s), cfg)

    count = 0
    try:
        while True:
            optimizer.step_once()
            count += 1
            if args.max_steps > 0 and count >= args.max_steps:
                break
            time.sleep(max(0.01, args.poll_ms / 1000.0))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
