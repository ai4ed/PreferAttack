#!/usr/bin/env bash
# 监听 PID 208683 (Multi_Agent_Framework.py Llama-3.2-1B alpaca_eval) 退出,
# 等待 GPU 释放后, 自动启动 run_pairwise_qwen3_all_code_judge.sh。
# 用法: nohup bash run_after_current.sh > /dev/null 2>&1 &

set -uo pipefail

WAIT_PID="${WAIT_PID:-208683}"
TARGET_SCRIPT="/root/PreferAttack/run_pairwise_qwen3_all_code_judge.sh"
LOG_DIR="/root/PreferAttack/logs"
mkdir -p "${LOG_DIR}"

TS=$(date +%Y%m%d_%H%M%S)
WATCH_LOG="${LOG_DIR}/watch_${TS}.log"
RUN_LOG="${LOG_DIR}/run_pairwise_qwen3_all_code_judge_${TS}.log"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${WATCH_LOG}"; }

log "watcher started, pid=$$"
log "waiting for PID ${WAIT_PID} to exit..."
log "  cmdline: $(cat /proc/${WAIT_PID}/cmdline 2>/dev/null | tr '\0' ' ')"

# 阶段 1: 等 PID 退出
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 60
done
log "PID ${WAIT_PID} exited."

# 阶段 2: 等 GPU 真正释放 (vLLM 子进程可能比父进程晚退出)
log "waiting for GPU memory to free up..."
for i in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
  if [ -z "${used}" ] || [ "${used}" -lt 1000 ]; then
    log "GPU free (used=${used} MiB)."
    break
  fi
  log "  still ${used} MiB used, sleep 30s ($i/60)"
  sleep 30
done

# 阶段 3: 启动目标脚本
log "launching target script: ${TARGET_SCRIPT}"
log "run log: ${RUN_LOG}"
log "========================================================"

cd /root/PreferAttack
# 不要用 set -e, 让脚本自己处理错误
bash "${TARGET_SCRIPT}" > "${RUN_LOG}" 2>&1
status=$?

log "========================================================"
log "target script finished, exit status=${status}"
log "run log: ${RUN_LOG}"
