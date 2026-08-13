#!/usr/bin/env bash
set -euo pipefail
# ========= 消融实验: AttackAgent 只使用 Module-Level Recombination =========
# 与 run_pairwise_qwen3.sh 同结构, 仅:
#   1) METHOD 改为 modulelevel_only 标识, 结果文件名据此生成
#   2) Python 入口换为 Multi_Agent_Framework_ModuleLevelOnly.py
#   3) RL Q-table 单独保存, 避免与主实验 / 另一组消融互相覆盖
# ========= 设备选择 =========
export device=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SWAP_SPACE_GB=32
export VLLM_GPU_MEMORY_UTILIZATION=0.85
device="${device:-0}"
export CUDA_VISIBLE_DEVICES="${device}"

# ========= 可自定义区域 =========
PYTHON_BIN="python"
QWEN3_PATH="/root/autodl-tmp/Qwen3-VL-8B-Instruct/"
SPLIT_INFO="data/split/code_judge_bench_split_info.json"
SPLIT="test"
DTYPE="bf16"
DEVICE="cuda"
START="0"
NUM_SAMPLES="500"
BATCH_SIZE="64"
NUM_STEPS="100"
METHOD="multi_agent_pairwise_eval_modulelevel_only"   # ← 消融标识
PYTHON_ENTRY="Multi_Agent_Framework_ModuleLevelOnly.py"   # ← 消融入口

split_base=$(basename "${SPLIT_INFO}")
DATASET=$(echo "${split_base}" | sed -E 's/(_split_info|_split)?\.json$//')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="results"
SAVE_FILENAME="${METHOD}_${DATASET}_${TIMESTAMP}.json"
SAVE_PATH="${SAVE_DIR}/${SAVE_FILENAME}"

USE_RL_CONTROLLER="true"
USE_ATTACK_COMBINER="false"
GENOME_LEN="1"
START_WITH_COMBINER="false"
FALLBACK_TO_COMBINER_AFTER="10"
ATTACK_EVOLVE_STEPS="5"
ATTACK_EVAL_SAMPLES="8"

RL_LR="0.1"
RL_GAMMA="0.9"
RL_EPSILON="0.2"
RL_SAVE_PATH="results/rl_qtable_modulelevel_only.json"   # ← 与主实验隔离
GPT_MUTATION_PROB="0.05"

PREVIOUS_RESULTS=""
RESUME="false"

# Parse command-line options to override defaults (supports long options)
SAVE_PATH_USER_PROVIDED="false"
TEMP=$(getopt -o h --long help,device:,python-bin:,qwen3-path:,split-info:,split:,num-samples:,batch-size:,num-steps:,method:,save-dir:,save-path:,previous-results:,resume,use-rl-controller,use-attack-combiner -n 'run_ablation_modulelevel.sh' -- "$@")
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true; do
  case "$1" in
    --device ) device="$2"; shift 2 ;;
    --python-bin ) PYTHON_BIN="$2"; shift 2 ;;
    --qwen3-path ) QWEN3_PATH="$2"; shift 2 ;;
    --split-info ) SPLIT_INFO="$2"; shift 2 ;;
    --split ) SPLIT="$2"; shift 2 ;;
    --num-samples ) NUM_SAMPLES="$2"; shift 2 ;;
    --batch-size ) BATCH_SIZE="$2"; shift 2 ;;
    --num-steps ) NUM_STEPS="$2"; shift 2 ;;
    --method ) METHOD="$2"; shift 2 ;;
    --save-dir ) SAVE_DIR="$2"; shift 2 ;;
    --save-path ) SAVE_PATH="$2"; SAVE_PATH_USER_PROVIDED="true"; shift 2 ;;
    --previous-results ) PREVIOUS_RESULTS="$2"; shift 2 ;;
    --resume ) RESUME="true"; shift ;;
    --use-rl-controller ) USE_RL_CONTROLLER="true"; shift ;;
    --use-attack-combiner ) USE_ATTACK_COMBINER="true"; shift ;;
    -h | --help )
      echo "Usage: $0 [--device N] [--python-bin PATH] [--qwen3-path PATH] [--split-info PATH]"
      echo "           [--split test|train] [--num-samples N] [--batch-size N] [--num-steps N]"
      echo "           [--method NAME] [--save-dir DIR] [--save-path PATH] [--previous-results PATH] --resume"
      exit 0 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

split_base=$(basename "${SPLIT_INFO}")
DATASET=$(echo "${split_base}" | sed -E 's/(_split_info|_split)?\.json$//')
if [ "${SAVE_PATH_USER_PROVIDED}" != "true" ]; then
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  SAVE_FILENAME="${METHOD}_${DATASET}_${TIMESTAMP}.json"
  SAVE_PATH="${SAVE_DIR}/${SAVE_FILENAME}"
fi
export CUDA_VISIBLE_DEVICES="${device}"

mkdir -p "$(dirname "${SAVE_PATH}")"

echo "== Ablation: Module-Level Recombination only =="
echo "device=${device}  ->  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Model path: ${QWEN3_PATH}"
echo "Split info: ${SPLIT_INFO}  [split=${SPLIT}]"
echo "Device (torch): ${DEVICE} | DTYPE: ${DTYPE}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Python entry: ${PYTHON_ENTRY}"
echo "Save path: ${SAVE_PATH}"
echo "==============================================="

set -- \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --qwen3_path "${QWEN3_PATH}" \
  --split_info "${SPLIT_INFO}" \
  --split "${SPLIT}" \
  --num_samples "${NUM_SAMPLES}" \
  --start "${START}" \
  --save_path "${SAVE_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --num_steps "${NUM_STEPS}" \

if [ "${USE_RL_CONTROLLER}" = "true" ]; then
  echo "Enabling RL controller: lr=${RL_LR}, gamma=${RL_GAMMA}, eps=${RL_EPSILON}"
  set -- "$@" --use_rl_controller
  set -- "$@" --rl_lr "${RL_LR}"
  set -- "$@" --rl_gamma "${RL_GAMMA}"
  set -- "$@" --rl_epsilon "${RL_EPSILON}"
  if [ -n "${RL_SAVE_PATH}" ]; then
    set -- "$@" --rl_save_path "${RL_SAVE_PATH}"
  fi
fi

if [ "${USE_ATTACK_COMBINER}" = "true" ]; then
  set -- "$@" --use_attack_combiner
  set -- "$@" --genome_len "${GENOME_LEN}"
  if [ "${START_WITH_COMBINER}" = "true" ]; then
    set -- "$@" --start_with_combiner
  fi
  if [ -n "${FALLBACK_TO_COMBINER_AFTER}" ]; then
    set -- "$@" --fallback_to_combiner_after "${FALLBACK_TO_COMBINER_AFTER}"
  fi
  if [ -n "${ATTACK_EVOLVE_STEPS}" ]; then
    set -- "$@" --attack_evolve_steps "${ATTACK_EVOLVE_STEPS}"
  fi
  if [ -n "${ATTACK_EVAL_SAMPLES}" ]; then
    set -- "$@" --attack_eval_samples "${ATTACK_EVAL_SAMPLES}"
  fi
fi

if [ -n "${GPT_MUTATION_PROB}" ]; then
  set -- "$@" --gpt_mutation_prob "${GPT_MUTATION_PROB}"
fi

if [ -n "${PREVIOUS_RESULTS:-}" ]; then
  set -- "$@" --previous_results "${PREVIOUS_RESULTS}"
fi
if [ "${RESUME:-false}" = "true" ]; then
  set -- "$@" --resume
fi

start_time=$(date +%s)
start_ts=$(date '+%Y-%m-%d %H:%M:%S')
echo "Started at: ${start_ts}"
cmd=(env CUDA_VISIBLE_DEVICES="${device}" "${PYTHON_BIN}" "${PYTHON_ENTRY}" "${@}")
echo "Running command: ${cmd[*]}"
"${cmd[@]}"
status=$?
end_time=$(date +%s)
end_ts=$(date '+%Y-%m-%d %H:%M:%S')
elapsed=$((end_time - start_time))
hours=$((elapsed/3600))
mins=$(((elapsed%3600)/60))
secs=$((elapsed%60))
printf "Finished at: %s\nTotal elapsed: %02d:%02d:%02d (%d seconds)\n" "${end_ts}" ${hours} ${mins} ${secs} ${elapsed}
run_dir="$(dirname "${SAVE_PATH}")"
mkdir -p "${run_dir}"
printf "%s\tstart=%s\tend=%s\telapsed_sec=%d\n" "${SAVE_PATH}" "${start_ts}" "${end_ts}" ${elapsed} >> "${run_dir}/run_time.txt"
exit ${status}
