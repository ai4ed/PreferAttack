#!/usr/bin/env bash
# Chain: 等 sweep 完 → 自动算 best seed → 启动 Phase 2 (best seed × 421)
set -uo pipefail
cd /root/PreferAttack

echo "[chain2] waiting for sweep (PID 4084) at $(date '+%F %T')"
while kill -0 4084 2>/dev/null; do sleep 30; done
echo "[chain2] sweep finished at $(date '+%F %T')"

# 算 best seed:从 5 个 sweep 结果 JSON 里挑 ASR 最高的 seed
BEST_SEED=$(python3 -c "
import json, glob, os
results = {}
for f in glob.glob('results/seed_sweep/seed_*_*.json'):
    if '.checkpoint' in f: continue
    s = int(os.path.basename(f).split('_')[1])
    with open(f) as fp: d = json.load(fp)
    rate = d['meta']['aggregate_stats']['success_rate']
    results[s] = rate
    print(f'  seed={s}  rate={rate:.4f}')
best = max(results, key=results.get)
print(f'BEST_SEED={best}  rate={results[best]:.4f}')
import sys
sys.stderr.write(str(best))
" 2>&1)
# 上面把 best seed 输出到 stderr,从 BEST_SEED= 行解析
BEST_SEED=$(echo "$BEST_SEED" | grep "BEST_SEED=" | cut -d= -f2 | awk '{print $1}')
echo "[chain2] picked best_seed=${BEST_SEED}"

# 用 best seed 跑 Phase 2
TS=$(date +%Y%m%d_%H%M%S)
SAVE="results/phase2_best_seed/best_seed_${BEST_SEED}_${TS}.json"
LOG="results/phase2_best_seed/best_seed_${BEST_SEED}_${TS}.log"
mkdir -p results/phase2_best_seed

echo "[chain2] launching Phase 2 at $(date '+%F %T') -> ${SAVE}"

export device=0
export VLLM_WORKER_MULTIPC_METHOD=spawn
export VLLM_SWAP_SPACE_GB=32
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES="${device}"
export PYTHONHASHSEED=0

python Multi_Agent_Framework.py \
  --model /root/autodl-tmp/Llama-3.2-1B-Instruct/ \
  --qwen3_path /root/autodl-tmp/Llama-3.2-1B-Instruct/ \
  --split_info data/split/code_judge_bench_split_info.json \
  --split test \
  --num_samples 421 \
  --start 0 \
  --save_path "${SAVE}" \
  --batch_size 64 \
  --num_steps 100 \
  --dtype bf16 \
  --use_rl_controller \
  --rl_lr 0.1 --rl_gamma 0.9 --rl_epsilon 0.2 \
  --gpt_mutation_prob 0.05 \
  --seed "${BEST_SEED}" \
  > "${LOG}" 2>&1

echo "[chain2] Phase 2 finished at $(date '+%F %T')"
python3 -c "
import json
with open('${SAVE}') as f: d=json.load(f)
ag=d['meta']['aggregate_stats']
print(f'  best_seed=${BEST_SEED}  rate={ag[\"success_rate\"]:.4f}  n={len(d[\"records\"])}')
print(f'  vs original (no seed) rate=0.5677')
print(f'  delta: {(ag[\"success_rate\"] - 0.5677)*100:+.2f}pp')
"
