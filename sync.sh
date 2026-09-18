#!/usr/bin/env bash
# Mutagen 同步包装脚本（本地 nanoGPT ⇄ 远程机器）
#
#   ./sync.sh create   ★ 用密码登录时走这条：会调用 ssh，密码提示直接出现在你的终端
#   ./sync.sh resume     掉线/重启后续传，同样会重新问密码
#   ./sync.sh start      项目流程：一次管一组会话，但**必须先配好 SSH 免密**
#                        （守护进程在后台连，弹不出密码框，会报 prompter not found）
#   ./sync.sh list       状态 / 文件数 / 冲突
#   ./sync.sh flush      强制走一轮完整同步
#   ./sync.sh pause | resume | terminate | reset
#
# 服务器地址来源（优先级从高到低）：
#   1. 命令行：NANOGPT_BETA="user@your-server:~/nanoGPT" ./sync.sh create
#   2. .env 里的 NANOGPT_BETA
# 注意：值一定要加引号，否则 bash 会把 ~ 展开成本机家目录（脚本会警告）。

set -euo pipefail
cd "$(dirname "$0")"

SESSION="nanogpt"
CONFIG="mutagen.base.yml" # 模式 + 忽略规则，两条路共用
REPO="$(pwd)"

command="${1:-}"
if [ -z "$command" ]; then
    exec mutagen sync --help
fi

_preset_beta="${NANOGPT_BETA-}" # 命令行传入的值，优先级高于 .env
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi
[ -n "$_preset_beta" ] && export NANOGPT_BETA="$_preset_beta"

# 防呆：~ 没加引号时会被 bash 展开成本机家目录，远程路径就错了
case "${NANOGPT_BETA:-}" in
    *":/Users/$USER/"*)
        echo "警告：NANOGPT_BETA=$NANOGPT_BETA" >&2
        echo "      这里的 /Users/$USER/ 是本机家目录 —— ~ 被 bash 提前展开了。" >&2
        echo "      请给值加引号：NANOGPT_BETA=\"user@host:~/nanoGPT\"，或改用远程绝对路径。" >&2
        ;;
esac

need_beta() {
    if [ -z "${NANOGPT_BETA:-}" ] || [[ "$NANOGPT_BETA" == *gpu-host* ]]; then
        echo "错误：还没配置服务器地址。" >&2
        echo "  请编辑 .env，填 NANOGPT_BETA=\"用户名@主机:~/nanoGPT\"" >&2
        echo "  或临时指定：NANOGPT_BETA=\"user@your-server:~/nanoGPT\" $0 $command" >&2
        exit 1
    fi
}

case "$command" in
start)
    need_beta
    {
        cat "$CONFIG"
        cat <<EOF

  $SESSION:
    alpha: "."                                   # 本地仓库根目录
    beta: "$NANOGPT_BETA"   # 由 sync.sh 从 .env 注入
    flushOnCreate: true
EOF
    } >mutagen.yml
    echo "已生成 mutagen.yml  →  beta = $NANOGPT_BETA"
    # 上次没收拾干净时（会话卡在连接中、或手动删过 mutagen.yml），
    # 守护进程仍认为项目在运行，直接报 already running。自动清掉重试。
    if ! _out="$(mutagen project start 2>&1)"; then
        if printf '%s' "$_out" | grep -q 'already running'; then
            echo "检测到上次残留的项目锁，自动 terminate 后重启…"
            mutagen project terminate >/dev/null 2>&1 || true
            exec mutagen project start
        fi
        printf '%s\n' "$_out" >&2
        exit 1
    fi
    printf '%s\n' "$_out"
    ;;

create)
    # 直接建会话：ssh 的密码提示能回到你的终端（mutagen 只在 create/resume/reset 中转发提示）
    need_beta
    echo "接下来 OpenSSH 会向你索要密码（不是 mutagen 在问）。"
    exec mutagen sync create -c "$CONFIG" -n "$SESSION" "$REPO" "$NANOGPT_BETA"
    ;;

*)
    # list / flush / pause / resume / terminate / reset：
    # 项目在跑就操作项目，否则操作单会话
    if [ -f mutagen.yml ] && mutagen project list >/dev/null 2>&1; then
        exec mutagen project "$@"
    fi
    case "$command" in
    list) exec mutagen sync list ;;
    *) exec mutagen sync "$command" "$SESSION" "${@:2}" ;;
    esac
    ;;
esac
