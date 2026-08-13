#!/usr/bin/env bash
# Chain3: 等 Phase 2 (PID 14627) 跑完 → 自动跑 K=5 × Qwen2.5-1.5B × alpaca_eval 全量(161)
set -uo pipefail
cd /root/PreferAttack

PHASE2_PID=14627

echo "[chain3] waiting for Phase 2 (PID ${PHASE2_PID}) at $(date '+%F %T')"
while kill -0 ${PHASE2_PID} 2>/dev/null; do sleep 60; done
echo "[chain3] Phase 2 finished at $(date '+%F %T'), starting K=5 × Qwen-1.5B × alpaca_eval"

export device=0
export VLLM_WORKER_MULTIPC_METHOD=spawn
export VLLM_SWAP_SPACE_GB=32
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES="${device}"
export PYTHONHASHSEED=0

MODEL="/root/autodl-tmp/Qwen2.5-1.5B-Instruct"
SPLIT_INFO="data/split/alpaca_eval_split_info.json"
SPLIT="test"
NUM_SAMPLES=161
NUM_STEPS=100
BATCH_SIZE=64
START=0

mkdir -p results/phase3_alpaca_qwen15b

SEEDS=(0 1 2 42 123)

for SEED in "${SEEDS[@]}"; do
  TS=$(date +%Y%m%d_%H%M%S)
  SAVE="results/phase3_alpaca_qwen15b/seed_${SEED}_${TS}.json"
  LOG="results/phase3_alpaca_qwen15b/seed_${SEED}_${TS}.log"
  echo "============================================================"
  echo "[chain3 seed=${SEED}] start at $(date '+%F %T') -> ${SAVE}"
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
  echo "[chain3 seed=${SEED}] finished at $(date '+%F %T') (log: ${LOG})"
done

echo "============================================================"
echo "[chain3] All 5 seeds done at $(date '+%F %T'). Summary:"
python3 << 'PY'
import json, glob, os, math
rates = []
for s in [0,1,2,42,123]:
    fs = sorted(glob.glob(f'results/phase3_alpaca_qwen15b/seed_{s}_*.json'))
    fs = [f for f in fs if '.checkpoint' not in f]
    if not fs:
        print(f'  seed={s}: MISSING')
        continue
    with open(fs[-1]) as fp: d = json.load(fp)
    rate = d['meta']['aggregate_stats']['success_rate']
    n = len(d['records'])
    rates.append((s, rate, n))
    print(f'  seed={s:3d}  ASR={rate:.4f}  n={n}')

if len(rates) == 5:
    xs = [r for _, r, _ in rates]
    K = len(xs)
    mean = sum(xs)/K
    var = sum((x-mean)**2 for x in xs)/(K-1)
    std = math.sqrt(var)
    sem = std/math.sqrt(K)
    t = 2.776
    print(f'\n  K=5  mean={mean:.4f}  std={std:.4f}  95% CI=[{mean-t*sem:.4f}, {mean+t*sem:.4f}]')
PY
echo "============================================================"
