#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  # 1) 推荐：source 进当前 shell，自动 export DLLAMA_SHARDING_UPDATE_FILE
  source examples/sharding-update.sh -f /tmp/dllama_shard_update.txt \
    -l 9 -p 0 -n 2,3,4 -H 8,12,12

  # 2) 不 source：只写文件并 touch 触发更新（export 只在脚本进程内生效）
  examples/sharding-update.sh -f /tmp/dllama_shard_update.txt \
    -l 9 -p 0 -n 2,3,4 -H 8,12,12

  # 3) 不 source 但想设置环境变量：打印 export 语句
  examples/sharding-update.sh --print-export -f /tmp/dllama_shard_update.txt

Options:
  -f <path>   请求文件路径（DLLAMA_SHARDING_UPDATE_FILE 指向它）
  -l <u32>    layerIndex
  -p <u32>    pos（可选，默认 0）
  -n <csv>    stage_nodes（例如 2,3,4）
  -H <csv>    head_lens（例如 8,12,12；必须与 stage_nodes 数量一致）
  --print-export  仅打印 export DLLAMA_SHARDING_UPDATE_FILE=...（不写文件）
  -h          显示帮助

Notes:
  - 该脚本写入的字段与当前实现兼容：layer/pos/stage_nodes/head_lens。
  - 触发机制依赖 root 轮询 mtime：脚本会在写完后 touch 文件。
EOF
}

is_sourced() {
  # bash-only; if not bash, fall back to not sourced
  [[ -n "${BASH_VERSION:-}" ]] && [[ "${BASH_SOURCE[0]}" != "$0" ]]
}

PRINT_EXPORT=0
FILE=""
LAYER=""
POS="0"
STAGE_NODES=""
HEAD_LENS=""

# Basic arg parsing (supports one long flag)
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f) FILE="${2:-}"; shift 2;;
    -l) LAYER="${2:-}"; shift 2;;
    -p) POS="${2:-}"; shift 2;;
    -n) STAGE_NODES="${2:-}"; shift 2;;
    -H) HEAD_LENS="${2:-}"; shift 2;;
    --print-export) PRINT_EXPORT=1; shift 1;;
    -h|--help) usage; return 0 2>/dev/null || exit 0;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$FILE" ]]; then
  echo "Missing -f <path>" >&2
  usage
  exit 2
fi

if [[ "$PRINT_EXPORT" == "1" ]]; then
  printf 'export DLLAMA_SHARDING_UPDATE_FILE=%q\n' "$FILE"
  exit 0
fi

if [[ -z "$LAYER" || -z "$STAGE_NODES" || -z "$HEAD_LENS" ]]; then
  echo "Missing required args: -l <layer> -n <stage_nodes> -H <head_lens>" >&2
  usage
  exit 2
fi

# Validate CSV lengths match (best-effort)
IFS=',' read -r -a _nodes <<<"$STAGE_NODES"
IFS=',' read -r -a _lens  <<<"$HEAD_LENS"
if [[ "${#_nodes[@]}" -ne "${#_lens[@]}" ]]; then
  echo "stage_nodes count (${#_nodes[@]}) != head_lens count (${#_lens[@]})" >&2
  exit 2
fi

# Export env var if sourced, otherwise only in this process.
export DLLAMA_SHARDING_UPDATE_FILE="$FILE"
if is_sourced; then
  : # exported into caller shell
else
  echo "[info] DLLAMA_SHARDING_UPDATE_FILE exported only for this script process." >&2
  echo "[info] If you want it in your current shell: source examples/sharding-update.sh ..." >&2
fi

# Write request file atomically then touch to bump mtime.
TMP_FILE="${FILE}.tmp.$$"
cat >"$TMP_FILE" <<EOF
layer=$LAYER
pos=$POS
stage_nodes=$STAGE_NODES
head_lens=$HEAD_LENS
EOF
mv -f "$TMP_FILE" "$FILE"
touch "$FILE"

echo "[ok] wrote update request to: $FILE" >&2
