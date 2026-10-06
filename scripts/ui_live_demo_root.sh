#!/bin/bash
# 在 nx1 上跑。先 ssh 登上 nx1，再执行：
#   bash /home/jetson/cc/edgevisor_fresh/scripts/ui_live_demo_root.sh
# 当前终端就是面板。这一次运行会拉起其余机器，并在面板上出现
# 预留设备上线、设备劣化、层迁移、设备下线。

cd /home/jetson/cc/edgevisor_fresh/EdgeVisor || exit 1

if [ ! -t 1 ]; then
    echo "这个脚本要在 ssh 登上 nx1 之后的终端里跑，面板才能画在当前屏幕上。"
    exit 1
fi

NX2=jetson@10.47.235.49
NANO1=jetson@10.47.107.34
NANO2=jetson@10.47.215.50
LAPTOP=cc@10.47.72.162
MODEL_JETSON=/home/jetson/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m
MODEL_LAPTOP=/home/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m
ROOT_LOG=/tmp/ui_demo_root.log
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new"
TIMELINE_PID=""

remote() {
    # shellcheck disable=SC2086
    $SSH -n "$1" "$2"
}

stop_demo() {
    if [ -n "$TIMELINE_PID" ]; then
        kill "$TIMELINE_PID" >/dev/null 2>&1 || true
        wait "$TIMELINE_PID" >/dev/null 2>&1 || true
        TIMELINE_PID=""
    fi
    remote "$NX2" "pkill -x gpu_hog >/dev/null 2>&1 || true; pkill -x dllama >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
    for target in "$NANO1" "$NANO2" "$LAPTOP"; do
        remote "$target" "pkill -x dllama >/dev/null 2>&1 || true" >/dev/null 2>&1 || true
    done
    pkill -x dllama >/dev/null 2>&1 || true
    tmux kill-session -t evdemo >/dev/null 2>&1 || true
}

install_worker_script() {
    # shellcheck disable=SC2086
    $SSH "$1" "cat > /tmp/start_ui_worker.sh && chmod +x /tmp/start_ui_worker.sh" <<'EOF'
#!/bin/bash
export WORKER_DIR="$1"
export WORKER_MODEL="$2"
python3 - <<'PY'
import os
if os.fork() > 0:
    os._exit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)
os.chdir(os.environ["WORKER_DIR"])
log = os.open("/tmp/ui_demo_worker.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log, 1)
os.dup2(log, 2)
null = os.open("/dev/null", os.O_RDONLY)
os.dup2(null, 0)
os.execv("./dllama", ["./dllama", "worker", "--port", "18091", "--model", os.environ["WORKER_MODEL"]])
PY
exit 0
EOF
}

install_hog_script() {
    # shellcheck disable=SC2086
    $SSH "$NX2" "cat > /tmp/start_ui_hog.sh && chmod +x /tmp/start_ui_hog.sh" <<'EOF'
#!/bin/bash
python3 - <<'PY'
import os
if os.fork() > 0:
    os._exit(0)
os.setsid()
if os.fork() > 0:
    os._exit(0)
log = os.open("/tmp/ui_demo_hog.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(log, 1)
os.dup2(log, 2)
null = os.open("/dev/null", os.O_RDONLY)
os.dup2(null, 0)
os.execv("/tmp/gpu_hog", ["/tmp/gpu_hog", "4096"])
PY
exit 0
EOF
}

start_worker() {
    remote "$1" "bash /tmp/start_ui_worker.sh $2 $3" >/dev/null 2>&1 || true
}

wait_port() {
    target="$1"
    seconds="$2"
    deadline=$((SECONDS + seconds))
    while [ "$SECONDS" -lt "$deadline" ]; do
        if remote "$target" "python3 -c 'import socket;s=socket.socket();s.settimeout(1);r=s.connect_ex((\"127.0.0.1\",18091));s.close();raise SystemExit(0 if r==0 else 1)'"; then
            return 0
        fi
        sleep 2
    done
    return 1
}

wait_root() {
    pattern="$1"
    seconds="$2"
    deadline=$((SECONDS + seconds))
    while [ "$SECONDS" -lt "$deadline" ]; do
        hit=$(grep -a -F -- "$pattern" "$ROOT_LOG" 2>/dev/null | tail -n 1 || true)
        if [ -n "$hit" ]; then
            return 0
        fi
        sleep 5
    done
    return 1
}

count_root() {
    pattern="$1"
    n=$(grep -a -o -F -- "$pattern" "$ROOT_LOG" 2>/dev/null | wc -l | tr -d '[:space:]')
    if [ -z "$n" ]; then
        n=0
    fi
    printf '%s' "$n"
}

run_timeline() {
    echo "timeline start $(date '+%H:%M:%S')" >> /tmp/ui_demo_timeline.log
    if ! wait_root "redundancy:" 180; then
        echo "FAIL shadow" >> /tmp/ui_demo_timeline.log
        return
    fi
    if ! wait_root "<think>" 240; then
        echo "FAIL tokens" >> /tmp/ui_demo_timeline.log
        return
    fi
    echo "tokens flowing $(date '+%H:%M:%S')" >> /tmp/ui_demo_timeline.log
    sleep 15
    ratios=$(grep -a -F -- "ratios=" "$ROOT_LOG" | tail -n 1 || true)
    offline=$(printf '%s' "$ratios" | grep -o '1@0' | wc -l | tr -d '[:space:]')
    if [ "$offline" = "1" ]; then
        start_worker "$NANO1" "/home/jetson/cc/edgevisor_fresh/EdgeVisor" "$MODEL_JETSON"
        if wait_root "joined" 240; then
            echo "joined $(date '+%H:%M:%S')" >> /tmp/ui_demo_timeline.log
        else
            echo "FAIL online" >> /tmp/ui_demo_timeline.log
        fi
        sleep 12
    fi
    slow_before=$(count_root "SLOW")
    move_before=$(count_root "moving layers")
    if remote "$NX2" "test -x /tmp/gpu_hog"; then
        remote "$NX2" "bash /tmp/start_ui_hog.sh" >/dev/null 2>&1 || true
        deadline=$((SECONDS + 180))
        while [ "$SECONDS" -lt "$deadline" ]; do
            slow_now=$(count_root "SLOW")
            move_now=$(count_root "moving layers")
            if [ "$slow_now" -gt "$slow_before" ] || [ "$move_now" -gt "$move_before" ]; then
                echo "degraded $(date '+%H:%M:%S')" >> /tmp/ui_demo_timeline.log
                break
            fi
            sleep 5
        done
    fi
    sleep 8
    if [ "$offline" = "1" ]; then
        remote "$NANO1" "pkill -x dllama; echo KILLED" >/dev/null 2>&1 || true
        if wait_root "offline" 90; then
            echo "offline $(date '+%H:%M:%S')" >> /tmp/ui_demo_timeline.log
        else
            echo "FAIL offline" >> /tmp/ui_demo_timeline.log
        fi
    fi
    echo "OK" >> /tmp/ui_demo_timeline.log
}

echo "stopping leftover demo processes"
stop_demo
trap 'stop_demo' EXIT INT TERM
sleep 2

echo "starting nx2, laptop, and nano2. nano1 stays reserved until the panel is up."
install_worker_script "$NX2"
install_worker_script "$LAPTOP"
install_worker_script "$NANO2"
install_worker_script "$NANO1"
install_hog_script
start_worker "$NX2" "/home/jetson/cc/edgevisor_fresh/EdgeVisor" "$MODEL_JETSON"
start_worker "$LAPTOP" "/home/cc/edgevisor_fresh/EdgeVisor" "$MODEL_LAPTOP"
start_worker "$NANO2" "/home/jetson/cc/edgevisor_fresh/EdgeVisor" "$MODEL_JETSON"

for target in "$NX2" "$LAPTOP" "$NANO2"; do
    if ! wait_port "$target" 90; then
        echo "worker did not accept port 18091: $target"
        stop_demo
        exit 1
    fi
done

rm -f "$ROOT_LOG" /tmp/ui_demo_timeline.log
cols=$(tput cols 2>/dev/null || echo 110)
lines=$(tput lines 2>/dev/null || echo 32)
tmux new-session -d -s evdemo -x "$cols" -y "$lines" "stdbuf -oL -eL ./dllama inference --prompt 'What is the capital of France?' --steps 600 --model $MODEL_JETSON --tokenizer /home/jetson/cc/models/qwen3_14b_q40/dllama_tokenizer_qwen3_14b_q40.t --workers 192.168.137.15:18091 192.168.137.31:18091 192.168.137.16:18091 192.168.137.18:18091 --auto --ui"
tmux set-option -t evdemo status off
tmux set-option -t evdemo remain-on-exit on
tmux pipe-pane -t evdemo -o "stdbuf -oL cat >> $ROOT_LOG"

run_timeline &
TIMELINE_PID=$!

echo "opening the panel in this terminal"
tmux attach -t evdemo
