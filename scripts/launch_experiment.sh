#!/bin/bash
# launch_experiment.sh — 训练实验标准启动器（2026-08-13 景修）
#
# 执行协议 v2 三原则（严谨/安全/可靠）的启动环节自动化：
#   1. PID 锁：已有实验在跑 → 拒绝启动（防并行 OOM 事故）
#   2. 内存预检：可用 < 3000MB → 拒绝启动
#   3. 启动：nice + nohup + setsid + python -u（防断连被杀、输出实时可见）
#   4. 写 PID 文件到 logs/<EXP_ID>.pid
#   5. 10 秒存活确认：进程活着 + 日志有 "start training"
#
# 用法:
#   ./scripts/launch_experiment.sh <EXP_ID> [--] <python 命令...>
#   ./scripts/launch_experiment.sh EXP-17-0a-r2 python3 run_longExp.py --is_training 1 ...
#
# 退出码: 0=成功启动, 1=PID锁拒绝, 2=内存不足, 3=启动失败, 4=存活确认失败

set -u

EXP_ID="$1"; shift
# 允许 "--" 分隔符
[ "${1:-}" = "--" ] && shift
[ $# -eq 0 ] && { echo "用法: $0 <EXP_ID> [--] <python命令...>"; exit 3; }

LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${EXP_ID}.log"
PID_FILE="${LOG_DIR}/${EXP_ID}.pid"
mkdir -p "$LOG_DIR"

# --- 1. PID 锁 ---
for pidf in "$LOG_DIR"/*.pid; do
    [ -e "$pidf" ] || continue
    old_pid=$(cat "$pidf" 2>/dev/null || echo 0)
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "[拒绝] 已有实验在跑: $(basename "$pidf") PID=$old_pid"
        ps -o pid,etime,args -p "$old_pid" | tail -1
        exit 1
    else
        echo "[提示] 清理失效 PID 文件: $(basename "$pidf") ($old_pid 已不存在)"
        rm -f "$pidf"
    fi
done

# --- 2. 内存预检 ---
avail_mb=$(free -m | awk '/^Mem:/ {print $7}')
if [ -z "$avail_mb" ] || [ "$avail_mb" -lt 3000 ]; then
    echo "[拒绝] 可用内存不足: ${avail_mb:-?}MB < 3000MB。请等当前任务结束或检查系统状态。"
    free -m
    exit 2
fi
echo "[内存] 可用 ${avail_mb}MB ≥ 3000MB ✓"

# --- 3. 启动 ---
export OMP_NUM_THREADS=4
export PYTHONPATH="$(pwd)/PatchTST_supervised"
echo "[启动] EXP_ID=$EXP_ID"
echo "[启动] OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "[启动] 命令: $*"
echo "[启动] 日志: $LOG_FILE"
echo "[启动] 时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"

nohup setsid nice -n 10 "$@" > "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
echo "[启动] PID=$NEW_PID (已写 $PID_FILE)"

# --- 4. 10 秒存活确认 ---
sleep 10
if ! kill -0 "$NEW_PID" 2>/dev/null; then
    echo "[失败] 进程 10 秒内退出。日志尾部："
    tail -20 "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 3
fi

if grep -q "start training" "$LOG_FILE" 2>/dev/null; then
    echo "[确认] 进程存活且已开始训练 ✓"
    exit 0
elif grep -qiE "error|traceback|exception" "$LOG_FILE" 2>/dev/null; then
    echo "[失败] 日志出现异常关键字。日志尾部："
    tail -20 "$LOG_FILE"
    echo "[失败] 请检查后重试。进程仍在运行，可 kill $NEW_PID 后重来。"
    exit 4
else
    echo "[确认] 进程存活 ✓（尚未输出 'start training'，可能仍在数据加载）"
    exit 0
fi
