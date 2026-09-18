#!/usr/bin/env bash
# ============================================================================
# 【服务器端】训练脚本。不要直接双击/./运行，由本地 ./script/train.sh 通过 ssh 调起。
#
# 为什么必须用 `bash script/train-remote.sh` 而不是 `./script/train-remote.sh`：
# Mutagen 同步默认用 portable 权限模式（文件 0600 / 目录 0700），
# 到服务器上可执行位会丢，所以只能显式用解释器调。
#
# 子命令（本地脚本会转发过来）：
#   doctor            体检：系统 / shell / uv / tmux / python / GPU / 目录
#   setup             建虚拟环境（有就跳过）+ 按需装依赖，不训练
#   start <cfg> [参数] 起 tmux 训练会话
#   status | log | stop | kill
#   __run <cfg> <log> [参数]   实际跑训练（由 tmux 调用）
# ============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
VENV="$REPO/.venv"
PY="$VENV/bin/python"
REQS="$HERE/requirements.txt"
RUN_DIR="$REPO/logs"
STAMP="$VENV/.requirements.sha256"
PIDFILE="$RUN_DIR/train.pid"
RCFILE="$RUN_DIR/last_exit_code"

# 优先级：ssh 转发的环境变量（本地解析出来的）> 同步过来的 .env > 内置默认值。
# ssh 默认不继承本地环境，所以本地脚本会用 `NANOGPT_SESSION=... bash script/...` 的形式显式转发；
# 这里先把转发进来的值记下来，读完 .env 再盖回去，避免被 .env 覆盖。
_preset_session="${NANOGPT_SESSION-}"
_preset_config="${NANOGPT_CONFIG-}"
_preset_device="${NANOGPT_DEVICE-}"
_preset_compile="${NANOGPT_COMPILE-}"
_preset_python="${NANOGPT_PYTHON-}"

if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
fi

if [ -n "$_preset_session" ]; then export NANOGPT_SESSION="$_preset_session"; fi
if [ -n "$_preset_config" ]; then export NANOGPT_CONFIG="$_preset_config"; fi
if [ -n "$_preset_device" ]; then export NANOGPT_DEVICE="$_preset_device"; fi
if [ -n "$_preset_compile" ]; then export NANOGPT_COMPILE="$_preset_compile"; fi
if [ -n "$_preset_python" ]; then export NANOGPT_PYTHON="$_preset_python"; fi

SESSION="${NANOGPT_SESSION:-nanogpt-train}"
CONFIG_DEFAULT="${NANOGPT_CONFIG:-config/train_shakespeare_char.py}"
PYTHON_VERSION="${NANOGPT_PYTHON:-3.12}"

say() { printf '%s\n' "$*"; }
die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

# 远端 shell 用的单引号转义（POSIX 安全：tmux 内部可能用 dash 执行）
shq() {
    local s=$1
    s=${s//\'/\'\\\'\'}
    printf "'%s'" "$s"
}

ensure_uv() {
    command -v uv >/dev/null 2>&1 || die "服务器上没有 uv。安装：curl -LsSf https://astral.sh/uv/install.sh | sh"
}

ensure_tmux() {
    command -v tmux >/dev/null 2>&1 || die "服务器上没有 tmux。安装：apt install tmux / yum install tmux（WSL 里用 sudo apt install tmux）"
}

# 虚拟环境：有就跳过创建；依赖按 requirements.txt 的哈希判断，变了才重装
ensure_env() {
    ensure_uv
    if [ -x "$PY" ]; then
        say "虚拟环境已存在（跳过创建）：$VENV"
    else
        say "创建虚拟环境：uv venv --python $PYTHON_VERSION"
        (cd "$REPO" && uv venv --python "$PYTHON_VERSION" .venv)
    fi

    local want have
    want="$(sha256sum "$REQS" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$REQS" | awk '{print $1}')"
    have="$(cat "$STAMP" 2>/dev/null || true)"
    if [ "$want" = "$have" ] && "$PY" -c 'import torch, numpy' >/dev/null 2>&1; then
        say "依赖已就绪（跳过安装）"
    else
        say "安装依赖：uv pip install -r script/requirements.txt"
        (cd "$REPO" && uv pip install --python "$PY" -r "$REQS")
        printf '%s' "$want" >"$STAMP"
        say "完成：$("$PY" -c 'import torch; print("torch", torch.__version__)')"
    fi
}

# 设备检测：NVIDIA → cuda；Apple MPS → mps；否则 cpu
pick_device() {
    if [ -n "${NANOGPT_DEVICE:-}" ]; then
        printf '%s' "$NANOGPT_DEVICE"
        return
    fi
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        printf 'cuda'
        return
    fi
    if "$PY" -c 'import sys, torch; sys.exit(0 if torch.backends.mps.is_available() else 1)' >/dev/null 2>&1; then
        printf 'mps'
        return
    fi
    printf 'cpu'
}

# ---------------------------------------------------------------------------
cmd_doctor() {
    say "== 系统 =="
    say "  uname    : $(uname -srm 2>/dev/null || echo '不可用（原生 Windows？）')"
    say "  shell    : ${SHELL:-未知}     HOME: ${HOME:-未知}"
    say "  pwd      : $(pwd)"
    say "  内核/容器: $(cat /proc/version 2>/dev/null | head -c 120 || echo '无 /proc/version')"
    say ""
    say "== 工具 =="
    local c
    for c in bash sh uv tmux python3 git curl nvidia-smi; do
        printf '  %-10s: %s\n' "$c" "$(command -v "$c" 2>/dev/null || echo '缺')"
    done
    if command -v uv >/dev/null 2>&1; then say "  uv 版本  : $(uv --version)"; fi
    say ""
    say "== GPU =="
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | sed 's/^/  /'
    else
        say "  没有 nvidia-smi（CPU/MPS 训练，或 GPU 没挂进来）"
    fi
    say ""
    say "== 仓库 =="
    say "  仓库目录 : $REPO"
    say "  venv     : $([ -x "$PY" ] && echo "已存在 ($("$PY" -c 'import torch;print(torch.__version__)' 2>/dev/null || echo '依赖不全'))" || echo '还没有')"
    say "  数据文件 :"
    ls -lh "$REPO"/data/*/train.bin 2>/dev/null | sed 's/^/    /' || say "    一个都没有 —— 先跑 .venv/bin/python data/shakespeare_char/prepare.py"
    say "  磁盘剩余 : $(df -h "$REPO" | awk 'NR==2 {print $4"（可用）"}')"
}

# ---------------------------------------------------------------------------
cmd__run() {
    local config="$1" logfile="$2"
    shift 2
    cd "$REPO"
    mkdir -p "$RUN_DIR"

    {
        echo "================================================================"
        echo "开始时间 : $(date '+%F %T')"
        echo "主机     : $(hostname 2>/dev/null)  ($(uname -m))"
        echo "配置     : $config"
        echo "额外参数 : ${*:-（无）}"
        echo "设备     : ${NANOGPT_DEVICE:-?}    torch.compile: ${NANOGPT_COMPILE:-false}"
        echo "================================================================"
    } >>"$logfile"

    set +e
    # 进程替换而不是管道：$! 就是 python 自己的 PID，stop 能精确杀掉训练进程，
    # 同时 tee 让 attach 上去也能实时看到输出。
    "$PY" train.py "$config" "$@" > >(tee -a "$logfile") 2>&1 &
    local py_pid=$!
    echo "$py_pid" >"$PIDFILE"
    wait "$py_pid"
    local rc=$?
    rm -f "$PIDFILE"
    echo "$rc" >"$RCFILE"

    echo "----------------------------------------------------------------" | tee -a "$logfile"
    echo "结束时间 : $(date '+%F %T')   退出码: $rc" | tee -a "$logfile"
    echo "完整日志 : $logfile" | tee -a "$logfile"
    echo "----------------------------------------------------------------" | tee -a "$logfile"
    echo
    echo "训练进程已退出，会话留着方便翻看。Ctrl-b 然后 d 脱离；./script/train.sh kill 关掉。"
    exec "${SHELL:-/bin/bash}" -l
}

cmd_start() {
    ensure_tmux
    ensure_env

    local config="${1:-$CONFIG_DEFAULT}"
    if [ "$#" -gt 0 ]; then shift; fi
    [ -f "$REPO/$config" ] || die "找不到配置文件：$REPO/$config"

    if tmux has-session -t "$SESSION" 2>/dev/null; then
        die "tmux 会话 '$SESSION' 已存在。用 status/attach 看，或 stop/kill 处理"
    fi

    local device dtype_args compile_flag
    device="$(pick_device)"
    if [ "${NANOGPT_COMPILE:-false}" = "true" ]; then compile_flag="True"; else compile_flag="False"; fi
    if [ "$device" = "cuda" ]; then
        dtype_args="" # CUDA 上让 train.py 自己挑 bfloat16，最合适
    else
        dtype_args="--dtype=float32" # MPS/CPU 下不走 autocast，float32 免掉 GradScaler 警告
    fi

    mkdir -p "$RUN_DIR"
    local logfile="$RUN_DIR/train-$(date +%Y%m%d-%H%M%S).log"
    ln -sfn "$(basename "$logfile")" "$RUN_DIR/latest.log"

    export NANOGPT_DEVICE="$device" NANOGPT_COMPILE="$compile_flag"

    # tmux 内部可能用 dash 执行命令，所以显式用 bash 调（文件同步过来没有可执行位）
    local inner
    inner="bash $(shq "$SELF") __run $(shq "$config") $(shq "$logfile") $(shq "--device=$device") $(shq "--compile=$compile_flag")"
    if [ -n "$dtype_args" ]; then inner="$inner $(shq "$dtype_args")"; fi
    local a
    for a in "$@"; do inner="$inner $(shq "$a")"; done

    tmux new-session -d -s "$SESSION" -c "$REPO" "$inner"
    tmux set-option -t "$SESSION" history-limit 100000 >/dev/null 2>&1 || true
    sleep 1
    tmux has-session -t "$SESSION" 2>/dev/null || die "tmux 会话启动失败"

    say "训练已启动（远端 tmux 会话 '$SESSION'）"
    say "  主机     : $(hostname 2>/dev/null)"
    say "  配置     : $config"
    say "  设备     : $device    compile: $compile_flag    ${dtype_args:-dtype: 用 train.py 默认}"
    say "  额外参数 : ${*:-（无）}"
    say "  日志     : $logfile"
}

cmd_status() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            say "训练运行中：PID $(cat "$PIDFILE")（tmux 会话 '$SESSION'，主机 $(hostname 2>/dev/null)）"
            ps -o pid,etime,%cpu,%mem,command -p "$(cat "$PIDFILE")" | tail -n +2
        else
            say "tmux 会话 '$SESSION' 在，但训练进程已结束（会话留着给你看输出）"
        fi
    else
        say "没有在运行的训练会话（'$SESSION'）"
    fi
    if [ -f "$RCFILE" ]; then say "上次退出码：$(cat "$RCFILE")"; fi
    if [ -L "$RUN_DIR/latest.log" ]; then
        say "最近日志：$RUN_DIR/latest.log（$(wc -l <"$RUN_DIR/latest.log" | tr -d ' ') 行）"
    fi
}

cmd_log() {
    [ -e "$RUN_DIR/latest.log" ] || die "还没有日志"
    exec tail -n 200 -f "$RUN_DIR/latest.log"
}

cmd_stop() {
    if [ ! -f "$PIDFILE" ]; then
        say "没有记录到正在运行的训练进程"
        return 0
    fi
    local pid
    pid="$(cat "$PIDFILE")"
    if ! kill -0 "$pid" 2>/dev/null; then
        say "进程 $pid 已经不在了"
        rm -f "$PIDFILE"
        return 0
    fi
    say "发送 SIGINT 给训练进程 $pid"
    kill -INT "$pid"
    local i
    for i in $(seq 1 30); do
        kill -0 "$pid" 2>/dev/null || {
            say "已停止（SIGINT 不做收尾保存；续训用 --init_from=resume）"
            return 0
        }
        sleep 1
    done
    say "30 秒没退出，可以 kill -9 $pid 或 ./script/train.sh kill"
}

cmd_kill() {
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
        say "已关闭 tmux 会话 '$SESSION'"
    else
        say "没有 '$SESSION' 会话"
    fi
    rm -f "$PIDFILE"
}

cmd_prepare() {
    local dataset="${1:-shakespeare_char}"
    local script="data/$dataset/prepare.py"
    [ -f "$REPO/$script" ] || die "找不到 $REPO/$script"
    if [ -x "$PY" ]; then
        say "用虚拟环境跑 $script"
        (cd "$REPO" && "$PY" "$script")
    else
        say "还没有虚拟环境，用系统 python3 跑 $script（prepare.py 只需要 numpy/requests）"
        (cd "$REPO" && python3 "$script")
    fi
    say ""
    say "生成的数据文件："
    ls -lh "$REPO/data/$dataset"/*.bin "$REPO/data/$dataset"/*.pkl 2>/dev/null | sed 's/^/  /' || say "  （没看到产物，检查上面的报错）"
}

case "${1:-}" in
doctor) cmd_doctor ;;
setup) ensure_env ;;
prepare) shift; cmd_prepare "$@" ;;
start) shift; cmd_start "$@" ;;
status) cmd_status ;;
log) cmd_log ;;
stop) cmd_stop ;;
kill) cmd_kill ;;
__run) shift; cmd__run "$@" ;;
*) die "未知子命令 '$1'" ;;
esac
