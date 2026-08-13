#!/usr/bin/env bash
# Seed sweep on code_judge_bench / Llama-3.2-1B-Instruct (前 20 条样本)
# 用法: bash run_seed_sweep.sh
set -uo pipefail

export device=0
export VLLM_WORKER_MULTIPC_METHOD=spawn
export VLLM_SWAP_SPACE_GB=32
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES="${device}"
# 锁定 Python 哈希种子,确保 dict/set 遍历顺序确定
export PYTHONHASHSEED=0

MODEL="/root/autodl-tmp/Llama-3.2-1B-Instruct/"
SPLIT_INFO="data/split/code_judge_bench_split_info.json"
SPLIT="test"
NUM_SAMPLES=20
NUM_STEPS=100
BATCH_SIZE=64
START=0

mkdir -p results/seed_sweep

# 5 个不同 seed,覆盖常见取值
SEEDS=(0 1 2 42 123)

for SEED in "${SEEDS[@]}"; do
  TS=$(date +%Y%m%d_%H%M%S)
  SAVE="results/seed_sweep/seed_${SEED}_${TS}.json"
  LOG="results/seed_sweep/seed_${SEED}_${TS}.log"
  echo "============================================================"
  echo "[seed=${SEED}] start at $(date '+%Y-%m-%d %H:%M:%S') -> ${SAVE}"
  echo "============================================================"
  python Multi_Agent_Framework.py \
    --model "${MODEL}" \
    --qwen3_path "${MODEL}" \
    --split_info "${SPLIT_INFO}" \
    --split "${SPLIT}" \
    --num_samples "${NUM_SAMPLES}" \
    --start "${START}" \
    --save_path "${SAVE}" \
    --batch_size "${BATCH_SIZE}" \
    --num_steps "${NUM_STEPS}" \
    --dtype bf16 \
    --use_rl_controller \
    --rl_lr 0.1 --rl_gamma 0.9 --rl_epsilon 0.2 \
    --gpt_mutation_prob 0.05 \
    --seed "${SEED}" \
    > "${LOG}" 2>&1
  echo "[seed=${SEED}] finished at $(date '+%Y-%m-%d %H:%M:%S') (log: ${LOG})"
done

echo "============================================================"
echo "All seeds done. Summary:"
for SEED in "${SEEDS[@]}"; do
  LATEST=$(ls -t results/seed_sweep/seed_${SEED}_*.json 2>/dev/null | head -1)
  if [ -n "${LATEST}" ]; then
    python -c "
import json
with open('${LATEST}') as f: d=json.load(f)
ag=d['meta'].get('aggregate_stats',{})
print(f'  seed=${SEED}  rate={ag.get(\"success_rate\")}  n={len(d[\"records\"])}  file=${LATEST}')
"
  fi
done
