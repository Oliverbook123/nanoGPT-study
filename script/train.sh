#!/usr/bin/env bash
# ============================================================================
# 【本地】训练启动器：本机执行，命令通过 ssh 送到服务器上跑。
#
#   本地只做三件事：解析服务器地址、拼远端命令、把输出/日志带回来。
#   真正的 uv 建环境、tmux、train.py 全部在服务器上执行。
#   训练跑在服务器的 tmux 会话里 → 本机断网 / 合盖 / 关终端都不影响它。
#
#   ./script/train.sh doctor   ★ 先跑这个：体检服务器（系统/shell/uv/tmux/GPU/数据）
#   ./script/train.sh setup     在服务器上建虚拟环境 + 装依赖（有就跳过）
#   ./script/train.sh prepare   在服务器上准备数据集（默认 shakespeare_char）
#   ./script/train.sh start [config] [train.py 参数...]
#   ./script/train.sh status    训练在不在跑 / PID / 最近日志
#   ./script/train.sh log       实时 tail 服务器上的日志（Ctrl-C 退出，不影响训练）
#   ./script/train.sh attach    进入服务器上的 tmux 看实时输出（Ctrl-b 然后 d 脱离）
#   ./script/train.sh stop      发 SIGINT 停训练（会话保留，可翻看输出）
#   ./script/train.sh kill      关掉服务器上的 tmux 会话
#   ./script/train.sh shell     直接进服务器的仓库目录开个 shell
#   ./script/train.sh setup-key ★ 装 SSH 公钥（输一次密码，之后永久免密）
#
#   # 例子
#   ./script/train.sh start                          # 默认 config/train_shakespeare_char.py
#   ./script/train.sh start --max_iters=200          # 额外参数原样传给 train.py
#   ./script/train.sh start --init_from=resume       # 从 ckpt.pt 续训
#
# 服务器地址来源（优先级从高到低）：
#   1. 环境变量 NANOGPT_REMOTE / NANOGPT_REMOTE_DIR
#   2. 本仓库 .env 里的 NANOGPT_BETA（和 Mutagen 共用同一份配置，形如 user@host:~/path）
#   3. 内置默认值
#   注意：.env 那种 user@host:path 的写法不支持带端口的 host；
#         要非默认端口就显式设 NANOGPT_REMOTE=user@host:端口 和 NANOGPT_REMOTE_DIR=/路径
#
# 免去手输密码（可选，二选一）：
#
#   A. 用 macOS 钥匙串（推荐，密码不落任何文件）——先你自己在终端执行一次：
#        security add-generic-password -a <你的用户名> -s nanogpt-ssh -w   # 会提示你输入密码
#      然后在 .env 里加两行：
#        NANOGPT_KEYCHAIN_SERVICE=nanogpt-ssh
#        NANOGPT_KEYCHAIN_ACCOUNT=<你的用户名>
#      之后本脚本会给 ssh 装上 SSH_ASKPASS 钩子，自动从钥匙串取密码。
#
#   B. 装 SSH 密钥（更彻底，一次之后永远不需要密码）：
#        ./script/train.sh setup-key      # 等价于 ssh-copy-id + 免密验证
#
#   也可用 NANOGPT_ASKPASS_CMD='你自己的取密码命令' 或 NANOGPT_ASKPASS_FILE=/path/0600文件。
#
#   注意：这只对**本脚本发起的 ssh** 生效。Mutagen 的同步会话不受影响——
#   它会把 SSH_ASKPASS 强制指向自己（实测），所以 Mutagen 要免密只能装密钥，
#   或用 MUTAGEN_SSH_PATH 指向一个包装过的 ssh。
#
# 可选环境变量：
#   NANOGPT_SESSION    tmux 会话名          默认 nanogpt-train
#   NANOGPT_DEVICE     强制设备             默认服务器上自动检测 cuda/mps/cpu
#   NANOGPT_COMPILE    true/false           默认 false
#   NANOGPT_PYTHON     uv 用的 Python 版本  默认 3.12
#   NANOGPT_DRY_RUN=1  只打印要执行的 ssh 命令，不真连（调试用）
#   NANOGPT_SSH_MUX=1  开启 SSH 连接复用，一次脚本调用里只问一次密码
#                      （默认关：这会在 ssh 命令行上加 ControlMaster 选项）
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- 解析服务器地址与设置 ---------------------------------------------------
# 注意顺序：必须先把 .env 读进来，再解析各项配置，否则 .env 里的值不生效
if [ -f "$REPO/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO/.env"
    set +a
fi
if [ -n "${NANOGPT_BETA:-}" ]; then
    : "${NANOGPT_REMOTE:=${NANOGPT_BETA%%:*}}"
    : "${NANOGPT_REMOTE_DIR:=${NANOGPT_BETA#*:}}"
fi
REMOTE="${NANOGPT_REMOTE:-}"
REMOTE_DIR="${NANOGPT_REMOTE_DIR:-~/pro/nanoGPT}"
SESSION="${NANOGPT_SESSION:-nanogpt-train}"

say() { printf '%s\n' "$*"; }
die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

[ -n "$REMOTE" ] || die "没解析出服务器地址：检查 .env 的 NANOGPT_BETA，或设 NANOGPT_REMOTE"
[ -n "$REMOTE_DIR" ] || die "没解析出服务器上的仓库目录：设 NANOGPT_REMOTE_DIR"
# 远端目录要拼进远端命令；~ 必须留给远端展开，所以只做字符白名单而不是加引号
case "$REMOTE_DIR" in
*[!A-Za-z0-9_~/.@+-]*) die "NANOGPT_REMOTE_DIR 含可疑字符（空格/引号等）：$REMOTE_DIR" ;;
esac

SSH_OPTS=(-o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=3)
if [ "${NANOGPT_SSH_MUX:-0}" = "1" ]; then
    SSH_OPTS+=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%r@%h-%p" -o ControlPersist=10m)
fi

# 远端 shell 用的单引号转义
shq() {
    local s=$1
    s=${s//\'/\'\\\'\'}
    printf "'%s'" "$s"
}

# ---- 可选：让 ssh 自动取密码 -------------------------------------------------
# 原理：OpenSSH 官方的 SSH_ASKPASS 钩子——没有终端时，ssh 会运行这个程序，
# 并从它的 stdout 读密码；OpenSSH 8.4+ 用 SSH_ASKPASS_REQUIRE=force 可强制启用。
# 密码不写进脚本/仓库：helper 只负责"去哪取"，来源由你决定（钥匙串/自定义命令/0600 文件）。
setup_askpass() {
    ASKPASS_DIR=""
    local svc="${NANOGPT_KEYCHAIN_SERVICE:-}" acct="${NANOGPT_KEYCHAIN_ACCOUNT:-}"
    local cmd="${NANOGPT_ASKPASS_CMD:-}" file="${NANOGPT_ASKPASS_FILE:-}"
    if [ -z "$svc$cmd$file" ]; then return 0; fi

    ASKPASS_DIR="$(mktemp -d)"
    chmod 700 "$ASKPASS_DIR"
    if [ -n "$file" ]; then
        case "$(stat -f '%Lp' "$file" 2>/dev/null)" in
        600 | 400) ;;
        *) die "$file 权限必须是 0600（当前 $(stat -f '%Lp' "$file" 2>/dev/null)）：chmod 600 $file" ;;
        esac
        printf '#!/bin/sh\nexec cat %s\n' "$(shq "$file")" >"$ASKPASS_DIR/askpass"
    elif [ -n "$cmd" ]; then
        printf '#!/bin/sh\nexec sh -c %s\n' "$(shq "$cmd")" >"$ASKPASS_DIR/askpass"
    else
        # 没条目时直接给出明确提示，而不是让 ssh 把 security 的报错文本当密码用
        security find-generic-password -s "$svc" -a "${acct:-$USER}" >/dev/null 2>&1 ||
            die "钥匙串里没有条目 $svc/${acct:-$USER}。先执行一次：
      security add-generic-password -a ${acct:-$USER} -s $svc -w   # 会提示你输入密码"
        printf '#!/bin/sh\nexec security find-generic-password -w -a %s -s %s\n' \
            "$(shq "${acct:-$USER}")" "$(shq "$svc")" >"$ASKPASS_DIR/askpass"
    fi
    chmod 700 "$ASKPASS_DIR/askpass"
    export SSH_ASKPASS="$ASKPASS_DIR/askpass" SSH_ASKPASS_REQUIRE=force
    export DISPLAY="${DISPLAY:-:0}"
    trap 'rm -rf "$ASKPASS_DIR"' EXIT
    say "已启用自动取密码（来源：${svc:+钥匙串 ${svc}}${cmd:+自定义命令}${file:+文件}）"
}

if [ "${NANOGPT_DRY_RUN:-0}" != "1" ]; then
    setup_askpass
fi

# 把本地解析好的设置转发到远端（ssh 不会继承本地环境；远端那份 .env 也会被同步过去，
# 但同步有延迟，所以本地解析出来的值优先由这里显式带过去）
remote_env_prefix() {
    local out=" NANOGPT_SESSION=$(shq "$SESSION")"
    local v val
    for v in NANOGPT_CONFIG NANOGPT_DEVICE NANOGPT_COMPILE NANOGPT_PYTHON; do
        val="${!v-}"
        if [ -n "$val" ]; then out="$out $v=$(shq "$val")"; fi
    done
    printf '%s' "$out"
}

rsh() { # 执行一条远端命令
    if [ "${NANOGPT_DRY_RUN:-0}" = "1" ]; then
        printf '[dry-run] ssh %s %s %s\n' "${SSH_OPTS[*]}" "$REMOTE" "$(printf '%q' "$1")"
        return 0
    fi
    ssh "${SSH_OPTS[@]}" "$REMOTE" "$@"
}

rrun() { # 在远端仓库目录里执行 script/train-remote.sh
    local sub="$1"
    shift
    local cmd="cd $REMOTE_DIR &&$(remote_env_prefix) bash script/train-remote.sh $(shq "$sub")"
    local a
    for a in "$@"; do cmd="$cmd $(shq "$a")"; done
    rsh "$cmd"
}

rrun_tty() { # 需要 TTY 的子命令（attach / log）
    local sub="$1"
    shift
    local cmd="cd $REMOTE_DIR &&$(remote_env_prefix) bash script/train-remote.sh $(shq "$sub")"
    local a
    for a in "$@"; do cmd="$cmd $(shq "$a")"; done
    if [ "${NANOGPT_DRY_RUN:-0}" = "1" ]; then
        printf '[dry-run] ssh -t %s %s %s\n' "${SSH_OPTS[*]}" "$REMOTE" "$(printf '%q' "$cmd")"
        return 0
    fi
    ssh -t "${SSH_OPTS[@]}" "$REMOTE" "$cmd"
}

# ---- 子命令 ---------------------------------------------------------------
banner() {
    say "服务器   : $REMOTE"
    say "仓库目录 : $REMOTE_DIR"
    say "会话名   : $SESSION"
    say ""
}

cmd_doctor() {
    banner
    say "===== 第一步：裸探测（不依赖远端脚本，万一是原生 Windows shell 也能看出来）====="
    rsh 'echo "uname : $(uname -srm 2>/dev/null || echo 不可用)"; \
         echo "HOME  : ${HOME:-未设置}"; \
         echo "SHELL : ${SHELL:-未设置}"; \
         for c in bash sh uv tmux python3 nvidia-smi; do printf "%-10s: " "$c"; command -v "$c" 2>/dev/null || echo 缺失; done' || true

    say ""
    say "===== 第二步：远端脚本体检（依赖 Mutagen 已把 script/ 同步过去）====="
    if ! rsh "[ -f $REMOTE_DIR/script/train-remote.sh ] && echo FOUND" | grep -q FOUND; then
        die "服务器上还没有 $REMOTE_DIR/script/train-remote.sh —— 先在本地跑 ./sync.sh create（或 ./sync.sh flush）把仓库同步过去"
    fi
    rrun doctor
}

cmd_start() {
    banner
    if [ "$#" -gt 0 ]; then
        rrun start "$@"
    else
        rrun start
    fi
}

cmd_setup_key() {
    banner
    say "把本机公钥装到服务器上（会让你输一次密码，之后永久免密）"
    say "公钥：$(ssh-keygen -lf "$HOME/.ssh/id_ed25519.pub" 2>/dev/null | awk '{print $2, $3}')"
    say ""
    if ! command -v ssh-copy-id >/dev/null 2>&1; then
        die "本机没有 ssh-copy-id"
    fi
    # ssh-copy-id 依赖远端能跑 sh；原生 Windows OpenSSH 会失败
    ssh-copy-id -o ConnectTimeout=10 "$REMOTE" || die "ssh-copy-id 失败：远端可能没有 sh（原生 Windows OpenSSH）。
     那种情况要手动把公钥追加到 C:\\Users\\<用户>\\.ssh\\authorized_keys（注意 ACL 只允许本人+SYSTEM）
     先跑 ./script/train.sh doctor 看清远端到底是什么环境"

    say ""
    say "验证：只用公钥认证能否登入（BatchMode 保证不会偷偷退回问密码）…"
    if ssh -o BatchMode=yes -o PreferredAuthentications=publickey "${SSH_OPTS[@]}" "$REMOTE" 'echo KEY-OK'; then
        say ""
        say "✅ 免密已生效。"
        say "   - script/train.sh 的 doctor/start/status 都不再问密码"
        say "   - Mutagen 不用重建会话：下次掉线/重启后它会自己用密钥重连"
        say "   - 想立刻验证自动重连：pkill -f 'mutagen-agent synchronizer'，几秒后看 ./sync.sh list 是否回到 Connected"
    else
        die "公钥似乎装上了但仍登不上：检查远端 authorized_keys 的内容与权限（Windows 上过宽的 ACL 会让 sshd 直接忽略该文件）"
    fi
}

cmd_shell() {
    if [ "${NANOGPT_DRY_RUN:-0}" = "1" ]; then
        printf '[dry-run] ssh -t %s %s %s\n' "${SSH_OPTS[*]}" "$REMOTE" "$(printf '%q' "cd $REMOTE_DIR && exec bash -l")"
        return 0
    fi
    exec ssh -t "${SSH_OPTS[@]}" "$REMOTE" "cd $REMOTE_DIR && exec bash -l"
}

usage() {
    # 打印文件头部的注释块（到第一个非注释行结束），比写死行号可靠
    awk 'NR > 1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

case "${1:-}" in
doctor) cmd_doctor ;;
setup)
    banner
    rrun setup
    ;;
prepare)
    banner
    if [ "$#" -gt 1 ]; then shift; rrun prepare "$@"; else rrun prepare; fi
    ;;
start) shift; cmd_start "$@" ;;
status)
    banner
    rrun status
    ;;
log) rrun_tty log ;;
attach) rrun_tty attach ;;
stop)
    banner
    rrun stop
    ;;
kill)
    banner
    rrun kill
    ;;
shell) cmd_shell ;;
setup-key) cmd_setup_key ;;
"" | -h | --help | help) usage ;;
*) die "未知命令 '$1'（doctor / setup / setup-key / prepare / start / status / log / attach / stop / kill / shell）" ;;
esac
