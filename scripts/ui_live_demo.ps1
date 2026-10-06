# Five-device live demo.
# Canonical copy: /home/jetson/cc/edgevisor_fresh/scripts/ui_live_demo.ps1
# Run it on the Windows host that has SSH aliases nx1, nx2, nano1, nano2, rog:
#   powershell -ExecutionPolicy Bypass -File .\ui_live_demo.ps1
# nx1 cannot SSH to the other devices, so do not run this file on nx1.
# It logs into the five machines, runs a real 14B inference, and attaches
# the root terminal. Ctrl-C or leaving the view stops the run and clears workers.
# The "Listening on" wait below is known to miss a worker that is already
# bound to port 18091, because that line is fully buffered when stdout is a file.
param([switch]$SelfTest)

$ErrorActionPreference = "Continue"
$RootScript = "/home/jetson/cc/edgevisor_fresh/scripts/ui_live_demo_root.sh"
$ModelJetson = "/home/jetson/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m"
$ModelLaptop = "/home/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m"

function Remote([string]$Target, [string]$Command) {
    & ssh -n -o BatchMode=yes -o ConnectTimeout=20 $Target $Command
    return $LASTEXITCODE
}

function Stop-Demo {
    Remote nx2 "pkill -x gpu_hog >/dev/null 2>&1 || true; pkill -x dllama >/dev/null 2>&1 || true" | Out-Null
    foreach ($target in @("nx1", "nano1", "nano2", "rog")) {
        Remote $target "pkill -x dllama >/dev/null 2>&1 || true" | Out-Null
    }
    Remote nx1 "tmux kill-session -t evdemo >/dev/null 2>&1 || true" | Out-Null
}

function Start-Worker([string]$Target, [string]$Dir, [string]$Model) {
    Remote $Target "bash /tmp/start_ui_worker.sh $Dir $Model" | Out-Null
}

function Wait-Root([string]$Pattern, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        $hit = & ssh -n -o BatchMode=yes -o ConnectTimeout=20 nx1 "grep -a -F -- '$Pattern' /tmp/ui_demo_root.log | tail -n 1"
        if ($hit) { return $true }
        Start-Sleep -Seconds 5
    }
    return $false
}

function Capture([string]$Name) {
    Remote nx1 "tmux capture-pane -t evdemo -e -p > /tmp/ui_phase_$Name.ans" | Out-Null
}

function Wait-Listening([string]$Target, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        $hit = & ssh -n -o BatchMode=yes -o ConnectTimeout=20 $Target "grep -a -F -- 'Listening on' /tmp/ui_demo_worker.log | tail -n 1"
        if ($hit) { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

Write-Output "stopping leftover demo processes"
Stop-Demo
Start-Sleep -Seconds 2

Write-Output "starting nx2, laptop, nano2"
Start-Worker "nx2" "/home/jetson/cc/edgevisor_fresh/EdgeVisor" $ModelJetson
Start-Worker "rog" "/home/cc/edgevisor_fresh/EdgeVisor" $ModelLaptop
Start-Worker "nano2" "/home/jetson/cc/edgevisor_fresh/EdgeVisor" $ModelJetson

Write-Output "waiting until those three workers are listening"
foreach ($target in @("nx2", "rog", "nano2")) {
    if (-not (Wait-Listening $target 90)) {
        Write-Output "timeline: FAIL listen $target"
        Stop-Demo
        exit 1
    }
}

Write-Output "starting root panel"
Remote nx1 "rm -f /tmp/ui_demo_root.log /tmp/ui_phase_pool.ans /tmp/ui_phase_online.ans /tmp/ui_phase_move.ans /tmp/ui_phase_offline.ans; tmux kill-session -t evdemo >/dev/null 2>&1 || true; tmux new-session -d -s evdemo -x 110 -y 32 bash $RootScript; tmux set-option -t evdemo remain-on-exit on; tmux pipe-pane -t evdemo -o 'cat >> /tmp/ui_demo_root.log'" | Out-Null

$timeline = Start-Job -ScriptBlock {
    function Wait-Root([string]$Pattern, [int]$Seconds) {
        $deadline = (Get-Date).AddSeconds($Seconds)
        while ((Get-Date) -lt $deadline) {
            $hit = & ssh -n -o BatchMode=yes -o ConnectTimeout=20 nx1 "grep -a -F -- '$Pattern' /tmp/ui_demo_root.log | tail -n 1"
            if ($hit) { return $true }
            Start-Sleep -Seconds 5
        }
        return $false
    }
    function Capture([string]$Name) {
        & ssh -n -o BatchMode=yes -o ConnectTimeout=20 nx1 "tmux capture-pane -t evdemo -e -p > /tmp/ui_phase_$Name.ans" | Out-Null
    }
    function Count-Root([string]$Pattern) {
        $n = & ssh -n -o BatchMode=yes -o ConnectTimeout=20 nx1 "grep -a -o -F -- '$Pattern' /tmp/ui_demo_root.log | wc -l"
        return [int]($n | Select-Object -Last 1)
    }
    $model = "/home/jetson/cc/models/qwen3_14b_q40/dllama_model_qwen3_14b_q40.m"
    if (-not (Wait-Root "redundancy:" 180)) { return "FAIL shadow" }
    $ratios = & ssh -n -o BatchMode=yes -o ConnectTimeout=20 nx1 "grep -a -F -- 'ratios=' /tmp/ui_demo_root.log | tail -n 1"
    $offlineSlots = ([regex]::Matches([string]$ratios, '1@0')).Count
    if ($offlineSlots -ne 1) { return "FAIL split $ratios" }
    & ssh -n -o BatchMode=yes -o ConnectTimeout=20 nano1 "bash /tmp/start_ui_worker.sh /home/jetson/cc/edgevisor_fresh/EdgeVisor $model" | Out-Null
    if (-not (Wait-Root "reserved" 600)) { return "FAIL pool" }
    Capture "pool"
    if (-not (Wait-Root "joined" 240)) { return "FAIL online" }
    Capture "online"
    Start-Sleep -Seconds 12
    $slowBefore = Count-Root "SLOW"
    $moveBefore = Count-Root "moving layers"
    & ssh -n -o BatchMode=yes -o ConnectTimeout=20 nx2 "bash /tmp/start_ui_hog.sh" | Out-Null
    $deadline = (Get-Date).AddSeconds(180)
    $moved = $false
    while ((Get-Date) -lt $deadline) {
        if ((Count-Root "SLOW") -gt $slowBefore -or (Count-Root "moving layers") -gt $moveBefore) {
            $moved = $true
            break
        }
        Start-Sleep -Seconds 5
    }
    if (-not $moved) { return "FAIL migrate" }
    Capture "move"
    Start-Sleep -Seconds 8
    & ssh -n -o BatchMode=yes -o ConnectTimeout=20 nano2 "pkill -x dllama; echo KILLED" | Out-Null
    if (-not (Wait-Root "offline" 90)) { return "FAIL offline" }
    Capture "offline"
    return "OK"
}

if (-not $SelfTest) {
    Write-Output "attaching the root terminal. Leave this view when the run finishes."
    & ssh -t -o BatchMode=yes nx1 "tmux attach -t evdemo"
}

$result = Receive-Job -Job $timeline -Wait -AutoRemoveJob
Write-Output "timeline: $result"
Stop-Demo
if ($result -ne "OK") { exit 1 }
exit 0
