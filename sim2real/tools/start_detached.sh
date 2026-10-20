#!/usr/bin/env bash
# 脱机运行：在 Jetson 上把手柄桥和策略程序放到后台启动，写日志到 Jetson 本机，之后可以拔掉 USB 数据线。
#
# 用法（在 Jetson 的 u2canfd 仓库目录，经 SSH 执行）：
#   DURATION_S=60 TAG=ground1 bash sim2real/tools/start_detached.sh bennett_1_free_deploy.py /path/to/policy.pt [额外参数...]
# 额外参数原样传给入口程序，例如 --output_scale 0.5。
#
# 流程：脚本启动手柄桥和策略程序，等策略程序打印 "[ROS2] waiting" 后提示可以拔线。
# 拔线后用手柄：摇杆回中、松开 L1、长按 START 1 秒启动；程序在运行 DURATION_S 秒后自动退出（电机失能）。
# 手柄 B 键只锁速度，不会停止程序；终端 q 在脱机时不可用，所以必须设置 DURATION_S。
# 电机看门狗（--motor_watchdog_ms，缺省 300）在程序崩溃或 USB 掉线时让电机自动失能。
#
# 输出目录 <总目录>/ground_logs/<时间>_<TAG>/：
#   deploy.csv  策略步日志（每关节一行：目标、实测位置速度力矩、IMU、指令）
#   policy.log  策略程序输出（含退出时的 [TIMING]）
#   teleop.log  手柄桥输出
#   meta.txt    提交号、参数、时间
# 测试结束后停手柄桥：kill $(cat <输出目录>/teleop.pid)。重新接线后把整个目录拷回电脑，放到 System_Identification/workstreams/07_actuator_identification/ground/data/raw/ 下。
set -u

if [ "$#" -lt 2 ]; then
    sed -n '2,20p' "${BASH_SOURCE[0]}"
    exit 2
fi
entry=$1
policy=$2
shift 2
duration=${DURATION_S:-60}
tag=${TAG:-run}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
onboard=$(cd "$repo/.." && pwd)
if [ ! -x "$onboard/policy_python.sh" ] && [ ! -f "$onboard/policy_python.sh" ]; then
    echo "[错误] 找不到 $onboard/policy_python.sh（应在 u2canfd 的上一级目录）"
    exit 3
fi
if [ ! -f "$repo/sim2real/$entry" ]; then
    echo "[错误] 找不到入口 sim2real/$entry"
    exit 3
fi
case "$duration" in
    ''|*[!0-9.]*) echo "[错误] DURATION_S 必须是正数（秒）"; exit 3 ;;
esac

stamp=$(date +%Y%m%d_%H%M%S)
out="$onboard/ground_logs/${stamp}_${tag}"
mkdir -p "$out"
{
    echo "time: $(date -Is)"
    echo "git: $(git -C "$repo" rev-parse HEAD)"
    git -C "$repo" status --short
    echo "entry: $entry"
    echo "policy: $policy"
    echo "duration_s: $duration"
    echo "extra_args: $*"
    uname -a
} > "$out/meta.txt" 2>&1

setsid nohup bash "$repo/sim2real/start_ros2_teleop.sh" > "$out/teleop.log" 2>&1 < /dev/null &
echo $! > "$out/teleop.pid"
sleep 2

cd "$repo" || exit 3
setsid nohup "$onboard/policy_python.sh" "sim2real/$entry" --policy "$policy" --ros2 \
    --duration_s "$duration" --log_csv "$out/deploy.csv" "$@" > "$out/policy.log" 2>&1 < /dev/null &
echo $! > "$out/policy.pid"

for _ in $(seq 1 90); do
    if grep -aq "\[ROS2\] waiting" "$out/policy.log" 2>/dev/null; then
        echo "[就绪] 策略程序在等待 START（电机未使能）。输出目录：$out"
        echo "       现在可以拔线。摇杆回中、松开 L1、长按 START 1 秒启动；约 ${duration} s 后自动退出。"
        exit 0
    fi
    if ! kill -0 "$(cat "$out/policy.pid")" 2>/dev/null; then
        echo "[失败] 策略程序已退出，日志末尾："
        tail -n 20 "$out/policy.log"
        exit 4
    fi
    sleep 1
done
echo "[超时] 90 s 内没有看到等待提示，日志末尾："
tail -n 20 "$out/policy.log"
exit 5
