#!/bin/bash
# 在控制机的真实终端里跑。这个脚本只负责启动这一次推理：它不会去杀 nx2，也不会自动把 nx2 拉回来。
# 人手动给 nx2 断电再上电，等它起来之后，在第二个终端里跑一条预装好的脚本让它回到流水线。
#
#   bash /home/byh/B01/EdgeVisor-wansongcc/scripts/demo_power_cycle.sh
#   bash /home/byh/B01/EdgeVisor-wansongcc/scripts/demo_power_cycle.sh --rebuild
#
# 二进制放在掉电也不会丢的目录：rog 用 /home/cc/ev_demo2，四台 Jetson 用 /home/jetson/ev_demo2。
# Ctrl-C 会停掉五台机器上的 dllama。

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
DROP_ADDR=192.168.137.15
PORT=18091
STEPS=512
ROG_DIR=/home/cc/ev_demo2
JETSON_DIR=/home/jetson/ev_demo2
BUILD_DIR=/tmp/ev_demo2_build
LOCAL_STARTER=/tmp/ev_demo2_start_worker.sh
TYPESCRIPT=/tmp/ev_demo2_ui.typescript
NEXT_FILE=/tmp/ev_demo2_next.txt
MODEL_JETSON=/home/jetson/cc/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m
TOK_JETSON=/home/jetson/cc/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t
MODEL_ROG=/home/cc/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m
TOK_ROG=/home/cc/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=5 -o ConnectionAttempts=1)
LAB_SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=5 -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_lab"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REBUILD=0

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

persist_dir() {
    if [ "$1" = "$ROOT_HOST" ]; then
        printf '%s\n' "$ROG_DIR"
    else
        printf '%s\n' "$JETSON_DIR"
    fi
}

# 短命令。Jetson 经 rog 跳到 137 网段。timeout 包住 ssh，避免横幅不回来时卡死。
remote() {
    local host="$1" cmd="$2" q
    if [ "$host" = "$ROOT_HOST" ]; then
        timeout -k 2 12 "${SSH[@]}" -n rog "$cmd"
        return
    fi
    q=$(printf '%q' "$cmd")
    timeout -k 2 15 "${SSH[@]}" -n rog "$LAB_SSH jetson@$(ip_of "$host") $q"
}

stop_all() {
    local host
    for host in "$ROOT_HOST" "${WORKER_HOSTS[@]}"; do
        remote "$host" 'pkill -x dllama >/dev/null 2>&1 || true' >/dev/null 2>&1 &
    done
    wait || true
}

on_exit() {
    stop_all
}

on_interrupt() {
    trap - EXIT INT TERM
    printf '\n已中断。\n'
    # 本地没有后台驱动要收，只把五台机器上的 dllama 停掉；放到后台跑，不在前台再开 ssh。
    stop_all >/dev/null 2>&1 &
    exit 130
}
trap on_exit EXIT
trap on_interrupt INT TERM

rejoin_command() {
    printf '%s\n' "ssh rog \"ssh -o BatchMode=yes -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_lab jetson@${DROP_ADDR} 'bash ${JETSON_DIR}/start_worker.sh'\""
}

write_next_file() {
    rejoin_command > "$NEXT_FILE"
}

write_starter() {
    cat > "$LOCAL_STARTER" <<EOF
#!/bin/bash
cd ${JETSON_DIR} || exit 1
if ss -ltn 2>/dev/null | grep -q ':${PORT}'; then
    echo "already listening on ${PORT}"
    exit 0
fi
export DLLAMA_TCP_KEEPIDLE_S=2 DLLAMA_TCP_KEEPINTVL_S=1 DLLAMA_TCP_KEEPCNT=3 DLLAMA_TCP_USER_TIMEOUT_MS=5000
python3 <<'PY'
import os
if os.fork() > 0:
    os._exit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)
log = os.open("${JETSON_DIR}/worker.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log, 1)
os.dup2(log, 2)
null = os.open("/dev/null", os.O_RDONLY)
os.dup2(null, 0)
os.execv("./dllama", ["./dllama", "worker", "--port", "${PORT}",
                      "--model", "${MODEL_JETSON}",
                      "--nthreads", "1"])
PY
exit 0
EOF
    # 模型路径和端口都写死在脚本里，跑的时候不再传参数。
    chmod +x "$LOCAL_STARTER"
}

# 从 rog 上那份同名脚本（/tmp/ev_demo2_start_worker.sh）再 scp 到 Jetson。
# 脚本文本不能走 ssh -n 的管道，那条 ssh 会把标准输入吃掉。
install_worker_starter() {
    local host="$1"
    remote "$host" "mkdir -p ${JETSON_DIR}" >/dev/null
    "${SSH[@]}" -n rog "scp -o BatchMode=yes -o ConnectTimeout=5 -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519_lab ${LOCAL_STARTER} jetson@$(ip_of "$host"):${JETSON_DIR}/start_worker.sh"
    remote "$host" "chmod +x ${JETSON_DIR}/start_worker.sh" >/dev/null
}

start_worker() {
    local host="$1"
    remote "$host" "bash ${JETSON_DIR}/start_worker.sh" >/dev/null
}

wait_listen() {
    local host="$1" start
    start=$SECONDS
    while [ $((SECONDS - start)) -lt 10 ]; do
        if remote "$host" "ss -ltn | grep -q ':${PORT}'"; then
            return 0
        fi
        sleep 1
    done
    return 1
}

# 需要重编时返回 0，可以直接用持久目录里现有二进制时返回 1。
need_build() {
    local host dir head rev
    if [ "$REBUILD" -eq 1 ]; then
        say "--rebuild，强制重编。"
        return 0
    fi
    if [ -n "$(git -C "$REPO" status --porcelain -- EdgeVisor/src EdgeVisor/Makefile)" ]; then
        say "EdgeVisor/src 或 EdgeVisor/Makefile 有未提交改动，重新编译。"
        return 0
    fi
    head="$(git -C "$REPO" rev-parse HEAD)"
    for host in "$ROOT_HOST" "${WORKER_HOSTS[@]}"; do
        dir="$(persist_dir "$host")"
        if ! remote "$host" "test -x ${dir}/dllama"; then
            say "$host 上没有可执行的 ${dir}/dllama，重新编译。"
            return 0
        fi
        rev="$(remote "$host" "cat ${dir}/source_rev 2>/dev/null" | tr -d '\r')"
        if [ "$rev" != "$head" ]; then
            say "$host 的 ${dir}/source_rev 和当前 HEAD 不一致，重新编译。"
            return 0
        fi
    done
    return 1
}

ensure_binaries() {
    local head rc host dir
    if ! need_build; then
        say "使用五台机器持久目录里已有的 dllama（对应 HEAD $(git -C "$REPO" rev-parse --short HEAD)）。"
        return 0
    fi
    head="$(git -C "$REPO" rev-parse HEAD)"
    say "同步源码到 rog:${BUILD_DIR}，先编 rog，再编 nx1，最后把 aarch64 二进制拷到其余三台..."
    "${SSH[@]}" rog "mkdir -p ${BUILD_DIR}"
    rsync -az \
        --exclude '*.o' --exclude '*.d' --exclude 'dllama' --exclude 'dllama-api' \
        --exclude '*-test' --exclude '.git' \
        "${REPO}/EdgeVisor/" rog:${BUILD_DIR}/
    # 这一整段是长时间编译，不能用 remote 里那 12/15 秒的 timeout 包住。
    "${SSH[@]}" rog 'bash -s' << EOF
set -e
cd ${BUILD_DIR}
DLLAMA_VULKAN=0 make dllama -j4 DLLAMA_CUDA=0
mkdir -p ${ROG_DIR}
cp ${BUILD_DIR}/dllama ${ROG_DIR}/dllama
chmod +x ${ROG_DIR}/dllama
printf '%s\n' '${head}' > ${ROG_DIR}/source_rev
RSH="ssh -o BatchMode=yes -o ConnectTimeout=5 -o IdentitiesOnly=yes -i \$HOME/.ssh/id_ed25519_lab"
rsync -az -e "\$RSH" \
    --exclude '*.o' --exclude '*.d' --exclude 'dllama' --exclude 'dllama-api' \
    --exclude '*-test' --exclude '.git' \
    ${BUILD_DIR}/ jetson@192.168.137.13:${BUILD_DIR}/
\$RSH -n jetson@192.168.137.13 'cd ${BUILD_DIR} && DLLAMA_VULKAN=0 make dllama -j4 DLLAMA_CUDA=0' </dev/null
\$RSH -n jetson@192.168.137.13 'mkdir -p ${JETSON_DIR}' </dev/null
rsync -az -e "\$RSH" jetson@192.168.137.13:${BUILD_DIR}/dllama /tmp/ev_demo2_aarch64_dllama
rsync -az -e "\$RSH" /tmp/ev_demo2_aarch64_dllama jetson@192.168.137.13:${JETSON_DIR}/dllama
\$RSH -n jetson@192.168.137.13 "chmod +x ${JETSON_DIR}/dllama && printf '%s\n' '${head}' > ${JETSON_DIR}/source_rev" </dev/null
for ip in 192.168.137.15 192.168.137.18 192.168.137.16; do
    \$RSH -n jetson@\$ip 'mkdir -p ${JETSON_DIR}' </dev/null
    rsync -az -e "\$RSH" /tmp/ev_demo2_aarch64_dllama jetson@\$ip:${JETSON_DIR}/dllama
    \$RSH -n jetson@\$ip "chmod +x ${JETSON_DIR}/dllama && printf '%s\n' '${head}' > ${JETSON_DIR}/source_rev" </dev/null
done
rm -f /tmp/ev_demo2_aarch64_dllama
EOF
    rc=$?
    if [ "$rc" -ne 0 ]; then
        say "编译失败（ssh 退出码 ${rc}）。"
        return "$rc"
    fi
    # ssh 退出 0 只说明 rog 上那段脚本跑完了；heredoc 里任何一条命令读走标准输入都会让它提前结束。
    # 所以再回读五台机器的持久目录，确认 aarch64 二进制真的落到了 ${JETSON_DIR}/dllama。
    for host in "$ROOT_HOST" "${WORKER_HOSTS[@]}"; do
        dir="$(persist_dir "$host")"
        if ! remote "$host" "test -x ${dir}/dllama"; then
            say "$host 的持久目录里没有可执行的 ${dir}/dllama，编译产物没拷到位。"
            return 1
        fi
    done
    say "编译完成，五台机器上的持久目录都已更新到 HEAD ${head}。"
    return 0
}

if [ ! -t 1 ]; then
    say "请在一个真实终端里跑这个脚本，面板才能画在当前屏幕上。"
    exit 1
fi

write_next_file

cols="$(tput cols 2>/dev/null || echo 0)"
say "EdgeVisor 断电 / 重新上线演示（主控 rog）"
say "这个脚本要在真实终端里跑，面板会直接画在当前屏幕上。"
say "终端宽度 ${cols} 列。面板需要 100 列以上，五段流水线才排得开。"
say "五跳都在 192.168.137.x 的有线上：rog → nx1 → nx2 → nano1 → nano2 → rog。"
say "出 tokens 之后，由人手动拔掉 ${DROP_HOST}（${DROP_ADDR}）的电源。"
say "不要拔 rog，也不要拔 nano2 —— 它们在流水线两头，拔了这次就演不下去。"
say "${DROP_HOST} 重新上电以后，打开第二个终端跑下面这条命令（${DROP_ADDR} 能 ping 通再跑）："
say
say "$(rejoin_command)"
say
say "同样的内容也写在 ${NEXT_FILE}。这一轮大约 6 到 10 分钟。Ctrl-C 会停掉五台机器上的 dllama。"
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

ensure_binaries || exit 1

say "安装重新上线要用的脚本到四台 Jetson..."
write_starter
scp -o BatchMode=yes -o ConnectTimeout=5 -q "$LOCAL_STARTER" rog:"$LOCAL_STARTER"
for host in "${WORKER_HOSTS[@]}"; do
    install_worker_starter "$host"
done

say "停掉残留的 dllama..."
stop_all
sleep 1

say "启动 nx1、${DROP_HOST}、nano1、nano2 的 worker..."
for host in "${WORKER_HOSTS[@]}"; do
    start_worker "$host"
done
for host in "${WORKER_HOSTS[@]}"; do
    if ! wait_listen "$host"; then
        say "$host 没有在端口 ${PORT} 上监听。"
        remote "$host" "echo starter_bytes=\$(wc -c < ${JETSON_DIR}/start_worker.sh 2>/dev/null || echo missing); tail -n 30 ${JETSON_DIR}/worker.log 2>/dev/null || true" || true
        exit 1
    fi
    say "  $host 已监听 ${PORT}"
done

WORKERS="${INFER_ADDR[0]}:${PORT} ${INFER_ADDR[1]}:${PORT} ${INFER_ADDR[2]}:${PORT} ${INFER_ADDR[3]}:${PORT}"

say
say "面板现在切到这个终端。等 tokens 开始出来，就去拔 ${DROP_HOST}（${DROP_ADDR}）。"
say "${DROP_HOST} 回来后，在第二个终端里跑（同样的内容也在 ${NEXT_FILE}）："
say "$(rejoin_command)"
sleep 2

"${SSH[@]}" -t rog "rm -f ${TYPESCRIPT}; cd ${ROG_DIR} && script -q -f -c 'DLLAMA_IGNORE_EOS=1 DLLAMA_BUBBLE_SHADOW_KV=1 DLLAMA_TCP_KEEPIDLE_S=2 DLLAMA_TCP_KEEPINTVL_S=1 DLLAMA_TCP_KEEPCNT=3 DLLAMA_TCP_USER_TIMEOUT_MS=5000 stdbuf -oL -eL ./dllama inference --model ${MODEL_ROG} --tokenizer ${TOK_ROG} --prompt Hi --steps ${STEPS} --temperature 0 --seed 1 --buffer-float-type q80 --max-seq-len 512 --nthreads 1 --ui --enable-pp-migration --workers ${WORKERS}' ${TYPESCRIPT}"
ui_rc=$?

say
say "======== 这一轮 ========"
say "${DROP_HOST} 还没回来的话，在第二个终端里跑（同样的内容也在 ${NEXT_FILE}）："
say "$(rejoin_command)"
say "---- rog 日志里的接管 / 重连 ----"
remote rog "tr -d '\\r' < ${TYPESCRIPT} 2>/dev/null | grep -aE 'takeover deadNode|dropout node=|rejoining|rejoined|ambiguous|inference stopped|\\[e1-end\\]|tokens/s' | grep -av redundant-infer" || true
say "界面进程退出码 ${ui_rc}。五台 dllama 正在停掉。"
exit "$ui_rc"
