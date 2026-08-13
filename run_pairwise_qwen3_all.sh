#!/usr/bin/env bash
set -euo pipefail
# ========= 在 run_pairwise_qwen3_datasets.sh 基础上加一层「外层串行多模型」循环 =========
# 与 run_pairwise_qwen3_datasets.sh 的差异:
#   1) 新增 MODELS 列表, 外层依次串行运行每个模型; 每个模型内部仍按 DATASETS 串行跑数据集
#   2) 每个 (model, dataset) 组合独立生成结果文件 (METHOD_<model>_<dataset>_<TIMESTAMP>.json)
#   3) 多模型 + 多数据集模式下 RL Q-table 自动追加 _<model>_<dataset> 后缀, 避免互相覆盖
#   4) 单个 (model, dataset) 失败不会中断整轮 sweep, 最终以聚合状态退出
#   5) 兼容老用法:
#        --qwen3-path PATH   → 退化为单模型 (跳过外层循环)
#        --split-info PATH   → 退化为单数据集 (跳过内层循环)
#   run_pairwise_qwen3.sh 和 run_pairwise_qwen3_datasets.sh 均保持不变。
# ========= 设备选择（和 run_pairwise_qwen3.sh 同风格） =========
export device=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SWAP_SPACE_GB=32
export VLLM_GPU_MEMORY_UTILIZATION=0.85
device="${device:-0}"
export CUDA_VISIBLE_DEVICES="${device}"

# ========= 可自定义区域 =========
PYTHON_BIN="python"

# 默认串行运行的模型列表 (按需增删, 空格分隔, 每项是模型目录的路径)
MODELS="/root/autodl-tmp/MiniCPM5-1B/"

# 默认串行运行的数据集 (每个对应 data/split/<ds>_split_info.json)
DATASETS="arena_hard"
DATA_SPLIT_DIR="data/split"

SPLIT="test"
DTYPE="bf16"
DEVICE="cuda"
START="0"
NUM_SAMPLES="500"
BATCH_SIZE="64"
NUM_STEPS="100"
METHOD="multi_agent_pairwise_eval_origin_opti"
PYTHON_ENTRY="Multi_Agent_Framework.py"

SAVE_DIR="results"

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
RL_SAVE_PATH="results/rl_qtable.json"   # 多模型/多数据集模式下自动追加 _<model>_<dataset> 后缀
GPT_MUTATION_PROB="0.05"

PREVIOUS_RESULTS=""
RESUME="false"

# 用户显式覆盖项 (CLI 解析后填充)
USER_QWEN3_PATH=""
USER_SPLIT_INFO=""
USER_SAVE_PATH=""
SAVE_PATH_USER_PROVIDED="false"

# Parse command-line options to override defaults (supports long options)
TEMP=$(getopt -o h --long help,device:,python-bin:,qwen3-path:,models:,split-info:,split:,num-samples:,batch-size:,num-steps:,method:,save-dir:,save-path:,datasets:,previous-results:,resume,use-rl-controller,use-attack-combiner -n 'run_pairwise_qwen3_models.sh' -- "$@")
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true; do
  case "$1" in
    --device ) device="$2"; shift 2 ;;
    --python-bin ) PYTHON_BIN="$2"; shift 2 ;;
    --qwen3-path ) USER_QWEN3_PATH="$2"; shift 2 ;;
    --models ) MODELS="$2"; shift 2 ;;
    --split-info ) USER_SPLIT_INFO="$2"; shift 2 ;;
    --split ) SPLIT="$2"; shift 2 ;;
    --num-samples ) NUM_SAMPLES="$2"; shift 2 ;;
    --batch-size ) BATCH_SIZE="$2"; shift 2 ;;
    --num-steps ) NUM_STEPS="$2"; shift 2 ;;
    --method ) METHOD="$2"; shift 2 ;;
    --save-dir ) SAVE_DIR="$2"; shift 2 ;;
    --save-path ) USER_SAVE_PATH="$2"; SAVE_PATH_USER_PROVIDED="true"; shift 2 ;;
    --datasets ) DATASETS="$2"; shift 2 ;;
    --previous-results ) PREVIOUS_RESULTS="$2"; shift 2 ;;
    --resume ) RESUME="true"; shift ;;
    --use-rl-controller ) USE_RL_CONTROLLER="true"; shift ;;
    --use-attack-combiner ) USE_ATTACK_COMBINER="true"; shift ;;
    -h | --help )
      echo "Usage: $0 [--device N] [--python-bin PATH]"
      echo "           [--models 'path1 path2 ...'] [--qwen3-path PATH]"
      echo "           [--datasets 'ds1 ds2'] [--split-info PATH] [--split test|train]"
      echo "           [--num-samples N] [--batch-size N] [--num-steps N]"
      echo "           [--method NAME] [--save-dir DIR] [--save-path PATH] [--previous-results PATH] --resume"
      echo "Default MODELS: ${MODELS}"
      echo "Default DATASETS: ${DATASETS} (run serially inside each model)"
      exit 0 ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${device}"
mkdir -p "${SAVE_DIR}"

# ========= 决定要跑的模型列表 =========
# 若显式给了 --qwen3-path, 退化为单模型模式 (兼容老用法); 否则按 MODELS 循环。
if [ -n "${USER_QWEN3_PATH}" ]; then
  RUN_MODELS="${USER_QWEN3_PATH}"
  MULTI_MODEL_MODE="false"
else
  RUN_MODELS="${MODELS}"
  MULTI_MODEL_MODE="true"
fi

# ========= 决定要跑的数据集列表 =========
# 若显式给了 --split-info, 退化为单数据集模式 (兼容老用法); 否则按 DATASETS 循环。
if [ -n "${USER_SPLIT_INFO}" ]; then
  RUN_SPLIT_INFO="${USER_SPLIT_INFO}"
  SINGLE_DS=$(basename "${USER_SPLIT_INFO}" | sed -E 's/(_split_info|_split)?\.json$//')
  RUN_DATASETS="${SINGLE_DS}"
  MULTI_DATASET_MODE="false"
else
  RUN_SPLIT_INFO=""
  RUN_DATASETS="${DATASETS}"
  MULTI_DATASET_MODE="true"
fi

echo "== Running multi-agent pairwise (serial models × serial datasets) =="
echo "device=${device}  ->  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Models (outer loop): ${RUN_MODELS}"
echo "Datasets (inner loop): ${RUN_DATASETS}"
echo "Split: ${SPLIT} | Num samples: ${NUM_SAMPLES} | Num steps: ${NUM_STEPS}"
echo "Device (torch): ${DEVICE} | DTYPE: ${DTYPE}"
echo "Python entry: ${PYTHON_ENTRY}"
echo "==============================================="

# ========= 单个 (model, dataset) 组合的运行逻辑 =========
# 用函数封装, 内部 set -- 只影响函数内的位置参数, 不会污染外部。
run_one() {
  local model_path="$1"
  local ds="$2"

  # 从路径推断 model 名 (去掉尾部斜杠后取 basename)
  local model_name dataset_name
  model_name=$(basename "${model_path%/}")
  dataset_name="${ds}"

  local split_info
  if [ -n "${RUN_SPLIT_INFO}" ]; then
    split_info="${RUN_SPLIT_INFO}"
  else
    split_info="${DATA_SPLIT_DIR}/${ds}_split_info.json"
  fi

  # per-(model,dataset) 结果文件路径
  local ts save_path
  ts=$(date +%Y%m%d_%H%M%S)
  if [ "${SAVE_PATH_USER_PROVIDED}" = "true" ]; then
    save_path="${USER_SAVE_PATH}"
  else
    save_path="${SAVE_DIR}/${METHOD}_${model_name}_${dataset_name}_${ts}.json"
  fi

  # per-(model,dataset) RL Q-table: 多模型/多数据集模式下追加后缀, 避免互相覆盖
  local rl_save_path="${RL_SAVE_PATH}"
  if [ "${MULTI_MODEL_MODE}" = "true" ] || [ "${MULTI_DATASET_MODE}" = "true" ]; then
    rl_save_path="${RL_SAVE_PATH%.json}_${model_name}_${dataset_name}.json"
  fi

  echo ""
  echo "########## Model: ${model_name} | Dataset: ${dataset_name} ##########"
  echo "Model path: ${model_path}"
  echo "Split info: ${split_info}  [split=${SPLIT}]"
  echo "Save path: ${save_path}"
  echo "RL save path: ${rl_save_path}"

  if [ ! -d "${model_path}" ]; then
    echo "[WARN] model dir not found: ${model_path} — skipping ${model_name}/${dataset_name}"
    return 3
  fi
  if [ ! -f "${split_info}" ]; then
    echo "[WARN] split_info not found: ${split_info} — skipping ${model_name}/${dataset_name}"
    return 3
  fi
  mkdir -p "$(dirname "${save_path}")"

  set -- \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --qwen3_path "${model_path}" \
    --split_info "${split_info}" \
    --split "${SPLIT}" \
    --num_samples "${NUM_SAMPLES}" \
    --start "${START}" \
    --save_path "${save_path}" \
    --batch_size "${BATCH_SIZE}" \
    --num_steps "${NUM_STEPS}" \

  if [ "${USE_RL_CONTROLLER}" = "true" ]; then
    echo "Enabling RL controller: lr=${RL_LR}, gamma=${RL_GAMMA}, eps=${RL_EPSILON}"
    set -- "$@" --use_rl_controller
    set -- "$@" --rl_lr "${RL_LR}"
    set -- "$@" --rl_gamma "${RL_GAMMA}"
    set -- "$@" --rl_epsilon "${RL_EPSILON}"
    if [ -n "${rl_save_path}" ]; then
      set -- "$@" --rl_save_path "${rl_save_path}"
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

  local start_time start_ts status
  start_time=$(date +%s)
  start_ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "Started at: ${start_ts}"
  echo "Running command: env CUDA_VISIBLE_DEVICES=${device} ${PYTHON_BIN} ${PYTHON_ENTRY} $*"

  # 关闭 errexit, 单个 (model,dataset) 失败不中断整个 sweep
  status=0
  set +e
  env CUDA_VISIBLE_DEVICES="${device}" "${PYTHON_BIN}" "${PYTHON_ENTRY}" "$@" || status=$?
  set -e

  local end_time end_ts elapsed hours mins secs
  end_time=$(date +%s)
  end_ts=$(date '+%Y-%m-%d %H:%M:%S')
  elapsed=$((end_time - start_time))
  hours=$((elapsed/3600))
  mins=$(((elapsed%3600)/60))
  secs=$((elapsed%60))
  printf "Finished at: %s\nTotal elapsed: %02d:%02d:%02d (%d seconds)\n" "${end_ts}" ${hours} ${mins} ${secs} ${elapsed}
  printf "%s\tmodel=%s\tstart=%s\tend=%s\telapsed_sec=%d\n" "${save_path}" "${model_name}" "${start_ts}" "${end_ts}" ${elapsed} >> "${SAVE_DIR}/run_time.txt"

  return ${status}
}

# ========= 外层串行模型, 内层串行数据集 =========
overall_status=0
total_fail=0
total_ok=0
for model_path in ${RUN_MODELS}; do
  model_name=$(basename "${model_path%/}")
  echo ""
  echo "============================================================"
  echo ">>> Entering MODEL: ${model_name} (${model_path})"
  echo "============================================================"
  for ds in ${RUN_DATASETS}; do
    if run_one "${model_path}" "${ds}"; then
      echo "[OK] ${model_name} / ${ds} completed"
      total_ok=$((total_ok+1))
    else
      s=$?
      echo "[FAIL] ${model_name} / ${ds} exited with status ${s} — continuing to next combination"
      overall_status=${s}
      total_fail=$((total_fail+1))
    fi
  done
done

echo ""
echo "All (model, dataset) combinations finished."
echo "Summary: OK=${total_ok}  FAIL=${total_fail}  Overall exit status: ${overall_status}"
exit ${overall_status}
