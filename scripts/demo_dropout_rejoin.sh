#!/bin/bash
# 在控制机的终端里直接跑。当前屏幕会变成 rog 上的 EdgeVisor 面板。
# 主控是 rog。四台 Jetson 由 rog 经 192.168.137.x 的有线来开关进程，推理也走这段有线。
#   rog -> nx1 -> nx2 -> nano1 -> nano2 -> rog
# 解码一段之后关掉中间的 nx2，面板上它变为 off；停几秒再拉起来，回到原来的层。
#
#   bash /home/byh/B01/EdgeVisor-wansongcc/scripts/demo_dropout_rejoin.sh
#
# 二进制在各机器 /tmp/ev_demo1/EdgeVisor/edgevisor。没有就现场编 CPU 版。
# 加 --rebuild 强制重编。Ctrl-C 会停掉五台机器上的 edgevisor。

set -u

ROOT_HOST=rog
WORKER_HOSTS=(nx1 nx2 nano1 nano2)
# 推理和开关进程都用 137 有线。WiFi 地址探得通，但撑不住一整次加载。
INFER_ADDR=(
    192.168.137.13
    192.168.137.15
    192.168.137.18
    192.168.137.16
)
DROP_HOST=nx2
DROP_NODE=2
PORT=18091
STEPS=120
KILL_POS=40
OFFLINE_HOLD_SEC=8
REMOTE_DIR=/tmp/ev_demo1/EdgeVisor
TYPESCRIPT=/tmp/ev_demo1_ui.typescript
DRIVER_LOG=/tmp/ev_demo1_ui_driver.log
MODEL_JETSON=/home/jetson/cc/models/llama3.2_3b_instruct_q40/edgevisor_model_llama3.2-3b-instruct_q40.m
TOK_JETSON=/home/jetson/cc/models/llama3.1_instruct_q40/edgevisor_tokenizer_llama_3_1.t
MODEL_ROG=/home/cc/models/llama3.2_3b_instruct_q40/edgevisor_model_llama3.2-3b-instruct_q40.m
TOK_ROG=/home/cc/models/llama3.1_instruct_q40/edgevisor_tokenizer_llama_3_1.t
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 -o ConnectionAttempts=1)
LAB_SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_lab"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REBUILD=0
DRIVER_PID=""

if [ "${1:-}" = "--rebuild" ]; then
    REBUILD=1
fi

say() { printf '%s\n' "$*"; }

ip_of() {
    case "$1" in
        rog) printf '%s\n' 192.168.137.20 ;;
        nx1) printf '%s\n' 192.168.137.13 ;;
        nx2) printf '%s\n' 192.168.137.15 ;;
        nano1) printf '%s\n' 192.168.137.18 ;;
        nano2) printf '%s\n' 192.168.137.16 ;;
        *) return 1 ;;
    esac
}

# 短命令。Jetson 经 rog 跳到 137 网段。timeout 包住 ssh，避免横幅不回来时卡死。
remote() {
    local host="$1" cmd="$2" q
    if [ "$host" = "rog" ]; then
        timeout -k 2 12 "${SSH[@]}" -n rog "$cmd"
        return
    fi
    q=$(printf '%q' "$cmd")
    timeout -k 2 15 "${SSH[@]}" -n rog "$LAB_SSH jetson@$(ip_of "$host") $q"
}

stop_all() {
    local host
    for host in "$ROOT_HOST" "${WORKER_HOSTS[@]}"; do
        remote "$host" 'pkill -x edgevisor >/dev/null 2>&1 || true' >/dev/null 2>&1 &
    done
    wait || true
}

on_exit() {
    if [ -n "$DRIVER_PID" ]; then
        kill "$DRIVER_PID" >/dev/null 2>&1 || true
        wait "$DRIVER_PID" >/dev/null 2>&1 || true
        DRIVER_PID=""
    fi
    stop_all
}

on_interrupt() {
    trap - EXIT INT TERM
    printf '\n已中断。\n'
    if [ -n "$DRIVER_PID" ]; then
        kill -KILL "$DRIVER_PID" >/dev/null 2>&1 || true
        DRIVER_PID=""
    fi
    stop_all >/dev/null 2>&1 &
    exit 130
}
trap on_exit EXIT
trap on_interrupt INT TERM

model_for() {
    if [ "$1" = "rog" ]; then
        printf '%s\n' "$MODEL_ROG"
    else
        printf '%s\n' "$MODEL_JETSON"
    fi
}

# 经 rog 再 ssh 到 Jetson 时不能靠管道传文件：里面那条 ssh -n 会把标准输入丢掉。
write_starter() {
    cat > /tmp/ev_start_worker.sh <<'EOF'
#!/bin/bash
MODEL="$1"
NTHREADS="${2:-1}"
cd /tmp/ev_demo1/EdgeVisor || exit 1
python3 - "$MODEL" "$NTHREADS" <<'PY'
import os, sys
model, nthreads = sys.argv[1], sys.argv[2]
if os.fork() > 0:
    os._exit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)
log = os.open("/tmp/ev_demo1_worker.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log, 1)
os.dup2(log, 2)
null = os.open("/dev/null", os.O_RDONLY)
os.dup2(null, 0)
os.execv("./edgevisor", ["./edgevisor", "worker", "--port", "18091",
                      "--model", model, "--nthreads", nthreads])
PY
exit 0
EOF
    chmod +x /tmp/ev_start_worker.sh
}

install_worker_starter() {
    local host="$1"
    if [ "$host" = "rog" ]; then
        scp -o BatchMode=yes -o ConnectTimeout=5 -q /tmp/ev_start_worker.sh rog:/tmp/ev_start_worker.sh
        remote rog 'chmod +x /tmp/ev_start_worker.sh'
        return
    fi
    "${SSH[@]}" -n rog "scp -o BatchMode=yes -o ConnectTimeout=5 -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_lab /tmp/ev_start_worker.sh jetson@$(ip_of "$host"):/tmp/ev_start_worker.sh"
    remote "$host" 'chmod +x /tmp/ev_start_worker.sh'
}

start_worker() {
    local host="$1"
    remote "$host" "bash /tmp/ev_start_worker.sh $(model_for "$host") 1" >/dev/null
}

wait_listen() {
    local host="$1" i
    for i in 1 2 3 4 5 6 7 8 9 10; do
        if remote "$host" "ss -ltn | grep -q ':${PORT}'"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

ensure_binaries() {
    local host missing=0
    if [ "$REBUILD" -eq 1 ]; then
        missing=1
    else
        for host in "$ROOT_HOST" "${WORKER_HOSTS[@]}"; do
            if ! remote "$host" "test -x ${REMOTE_DIR}/edgevisor"; then
                missing=1
                break
            fi
        done
    fi
    if [ "$missing" -eq 0 ]; then
        say "使用各机器上已有的 /tmp/ev_demo1/EdgeVisor/edgevisor。"
        return 0
    fi
    say "同步源码到 rog，编译 rog 和 nx1，再把 aarch64 二进制拷到其余三台..."
    "${SSH[@]}" rog 'mkdir -p /tmp/ev_demo1'
    rsync -az \
        --exclude '*.o' --exclude '*.d' --exclude 'edgevisor' --exclude 'edgevisor-api' \
        --exclude '*-test' --exclude '.git' \
        "${REPO}/EdgeVisor/" rog:${REMOTE_DIR}/
    "${SSH[@]}" rog 'bash -s' << EOF
set -e
cd ${REMOTE_DIR} && export EDGEVISOR_VULKAN=0 && make edgevisor -j4 EDGEVISOR_CUDA=0
RSH="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i \$HOME/.ssh/id_ed25519_lab"
rsync -az -e "\$RSH" \
    --exclude '*.o' --exclude '*.d' --exclude 'edgevisor' --exclude 'edgevisor-api' \
    --exclude '*-test' --exclude '.git' \
    ${REMOTE_DIR}/ jetson@192.168.137.13:${REMOTE_DIR}/
\$RSH jetson@192.168.137.13 'cd ${REMOTE_DIR} && export EDGEVISOR_VULKAN=0 && make edgevisor -j4 EDGEVISOR_CUDA=0'
rsync -az -e "\$RSH" jetson@192.168.137.13:${REMOTE_DIR}/edgevisor /tmp/ev_aarch64_edgevisor
for ip in 192.168.137.15 192.168.137.18 192.168.137.16; do
    \$RSH jetson@\$ip 'mkdir -p ${REMOTE_DIR}'
    rsync -az -e "\$RSH" /tmp/ev_aarch64_edgevisor jetson@\$ip:${REMOTE_DIR}/edgevisor
    \$RSH jetson@\$ip 'chmod +x ${REMOTE_DIR}/edgevisor'
done
rm -f /tmp/ev_aarch64_edgevisor
EOF
}

write_driver() {
    cat > /tmp/ev_demo1_ui_driver.sh <<EOF
#!/bin/bash
# 跟着 rog 的 typescript 走。输出只进驱动日志，不占用面板那个终端。
set -u
SSH=(timeout -k 2 12 ssh -o BatchMode=yes -o ConnectTimeout=5 -o ConnectionAttempts=1 -n rog)
LAB='ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_lab'
TYPESCRIPT=${TYPESCRIPT}
KILL_POS=${KILL_POS}
HOLD=${OFFLINE_HOLD_SEC}
DROP_IP=$(ip_of "$DROP_HOST")
log() { printf '%s %s\n' "\$(date '+%H:%M:%S')" "\$*" ; }

max_pos() {
    "\${SSH[@]}" "tr -d '\\r' < \$TYPESCRIPT 2>/dev/null | grep -a -oE 'pos=[0-9]+' | awk -F= 'BEGIN{m=0} {if (\\\$2+0>m) m=\\\$2+0} END{print m}'"
}

seen() {
    "\${SSH[@]}" "test -f \$TYPESCRIPT && tr -d '\\r' < \$TYPESCRIPT | grep -aF -q -- \"\$1\""
}

log "driver start, kill at pos>=\$KILL_POS"
pos=0
for _ in \$(seq 1 240); do
    if seen 'PP delivery is ambiguous'; then
        log "stop: ambiguous before the planned kill"
        exit 2
    fi
    if seen '[e1-end]'; then
        log "stop: inference ended before kill"
        exit 2
    fi
    pos=\$(max_pos 2>/dev/null || echo 0)
    case "\$pos" in ''|*[!0-9]*) pos=0 ;; esac
    if [ "\$pos" -ge "\$KILL_POS" ]; then
        break
    fi
    sleep 1
done
if [ "\$pos" -lt "\$KILL_POS" ]; then
    log "stop: decode never reached pos=\$KILL_POS (last=\$pos)"
    exit 2
fi
log "decode pos=\$pos, killing ${DROP_HOST}"
"\${SSH[@]}" "\$LAB jetson@\$DROP_IP 'pkill -x edgevisor >/dev/null 2>&1 || true'"
log "${DROP_HOST} killed"

took=0
for _ in \$(seq 1 90); do
    if seen 'takeover deadNode=${DROP_NODE}'; then
        took=1
        break
    fi
    if seen 'PP delivery is ambiguous' || seen 'inference stopped'; then
        log "stop: inference halted without takeover"
        exit 2
    fi
    sleep 1
done
if [ "\$took" -ne 1 ]; then
    log "stop: no takeover within 90s"
    exit 2
fi
log "takeover seen, holding offline for \${HOLD}s"
sleep "\$HOLD"
log "starting ${DROP_HOST} worker"
"\${SSH[@]}" "\$LAB jetson@\$DROP_IP 'bash /tmp/ev_start_worker.sh ${MODEL_JETSON} 1'" >/dev/null 2>&1 || true
for _ in \$(seq 1 180); do
    if seen 'rejoined'; then
        log "rejoined"
        exit 0
    fi
    if seen '[e1-end]' || seen 'inference stopped' || seen 'Critical error'; then
        log "stop: run ended before rejoin"
        exit 2
    fi
    sleep 1
done
log "stop: rejoin not seen within 180s"
exit 2
EOF
    chmod +x /tmp/ev_demo1_ui_driver.sh
}

if [ ! -t 1 ]; then
    say "请在一个真实终端里跑这个脚本，面板才能画在当前屏幕上。"
    exit 1
fi

cols="$(tput cols 2>/dev/null || echo 0)"
say "EdgeVisor 掉线 / 上线演示（主控 rog）"
say "终端宽度 ${cols} 列。100 列以上时五段流水线能排在一行里。"
say "五跳都在 192.168.137.x：rog → nx1 → nx2 → nano1 → nano2 → rog。"
say "顺序：五台一起推理 → 关掉中间的 nx2（面板 off）→ 再启动 nx2（回到原来的层，面板 live）。"
say "这一轮大约两三分钟。Ctrl-C 会停掉五台机器上的 edgevisor。"
say

say "检查五台机器（经 rog 的 192.168.137.x）..."
for host in "$ROOT_HOST" "${WORKER_HOSTS[@]}"; do
    printf '  %s ... ' "$host"
    if remote "$host" 'echo ok' >/dev/null; then
        say "通"
    else
        say "连不上。面板开在 rog 上，四台 Jetson 要能从 rog 经 137 网段登入。"
        exit 1
    fi
done
if ! remote rog 'command -v script >/dev/null'; then
    say "rog 上没有 script，面板需要它来分配一个终端。"
    exit 1
fi

ensure_binaries
write_starter
for host in "$ROOT_HOST" "${WORKER_HOSTS[@]}"; do
    install_worker_starter "$host"
done

say "停掉残留的 edgevisor..."
stop_all
sleep 1

say "启动 nx1、nx2、nano1、nano2 的 worker..."
for host in "${WORKER_HOSTS[@]}"; do
    start_worker "$host"
done
for host in "${WORKER_HOSTS[@]}"; do
    if ! wait_listen "$host"; then
        say "$host 没有在端口 ${PORT} 上监听。"
        remote "$host" "echo starter_bytes=\$(wc -c < /tmp/ev_start_worker.sh 2>/dev/null || echo missing); tail -n 30 /tmp/ev_demo1_worker.log 2>/dev/null || true" || true
        exit 1
    fi
    say "  $host 已监听 ${PORT}"
done

write_driver
: > "$DRIVER_LOG"
remote rog "rm -f ${TYPESCRIPT}"
setsid /tmp/ev_demo1_ui_driver.sh >>"$DRIVER_LOG" 2>&1 < /dev/null &
DRIVER_PID=$!

say
say "面板现在切到这个终端。nx2 会在大约第 ${KILL_POS} 步被关掉，停 ${OFFLINE_HOLD_SEC} 秒后再回来。"
sleep 2

WORKERS="${INFER_ADDR[0]}:${PORT} ${INFER_ADDR[1]}:${PORT} ${INFER_ADDR[2]}:${PORT} ${INFER_ADDR[3]}:${PORT}"
"${SSH[@]}" -t rog "rm -f ${TYPESCRIPT}; cd ${REMOTE_DIR} && script -q -f -c 'EDGEVISOR_IGNORE_EOS=1 EDGEVISOR_BUBBLE_SHADOW_KV=1 stdbuf -oL -eL ./edgevisor inference --model ${MODEL_ROG} --tokenizer ${TOK_ROG} --prompt Hi --steps ${STEPS} --temperature 0 --seed 1 --buffer-float-type q80 --max-seq-len 256 --nthreads 1 --ui --enable-pp-migration --workers ${WORKERS}' ${TYPESCRIPT}"
ui_rc=$?

say
say "======== 这一轮 ========"
if [ -f "$DRIVER_LOG" ]; then
    cat "$DRIVER_LOG"
fi
say "---- rog 日志里的接管 / 上线 ----"
remote rog "tr -d '\\r' < ${TYPESCRIPT} 2>/dev/null | grep -aE 'takeover deadNode|dropout node=|rejoining |rejoined |ambiguous|inference stopped|\\[e1-end\\]|tokens/s' | grep -av redundant-infer" || true
say "界面进程退出码 ${ui_rc}。五台 edgevisor 正在停掉。"
exit "$ui_rc"
