# PreferAttack：A collaborative multi-agent framework for preference-reversal attacks on LLM-as-a-Judge models

**PreferAttack** is a collaborative multi-agent framework for **preference-reversal attacks** on **LLM-as-a-Judge** models. It optimizes an *adversarial suffix* appended to the user instruction so that the judge flips its pairwise preference between two candidate responses (A vs. B), while the responses themselves are left untouched.

<p align="center">
  🤗 <a href="#project-structure"><b>Project Structure</b></a> &nbsp; | &nbsp;
  🖥️ <a href="#quick-start-main-attack"><b>Quickstart</b></a> &nbsp; | &nbsp;
  🧪 <a href="#defense-experiments"><b>Defense</b></a> &nbsp; | &nbsp;
  ©️ <a href="#citation"><b>Citation</b></a>
</p>

---

## Highlights

- **Collaborative multi-agent optimization** — a Scoring agent, a Decision agent, and an Attack agent cooperate to search for a preference-flipping suffix.
- **Adaptive search** — a DQN controller observes the search state (improvement, diversity, patience, average fitness) and steers the genetic-algorithm hyper-parameters online.
- **Multi-granularity operators** — word-level recombination, module-level attention recombination, and a hybrid of the two.
- **Stealth by design** — suffixes are evolved through semantic-preserving transformations rather than injected gibberish.
- **Three LLM-as-a-Judge benchmarks** — Alpaca Eval, Arena Hard, and Code Judge Bench.
- **Defense studies** — perplexity (PPL), learned classifiers (LR / MLP), and semantic-similarity detectors.

---

## Overview

Given a pairwise evaluation sample `(instruction, response_a, response_b)`, a judge model produces an initial preference. PreferAttack searches for a short suffix `s` such that

```
instruction' = instruction.rstrip() + " " + s
```

and, when judged on `(instruction', response_a, response_b)`, the judge flips its preference. The attack is **stealthy** by design: the suffix is evolved through *semantic-preserving* transformations rather than injected gibberish.

The framework is composed of three collaborating agents:

| Agent | Responsibility | Where |
|---|---|---|
| **Scoring Agent** | Builds attacked candidates, queries the judge, computes fitness (Eq. (6): `flipped → 1−conf`, `not flipped → conf`; lower is better) | `ScoringAgent` in `Multi_Agent_Framework.py` |
| **Decision Agent** | Selects search mode and GA hyper-parameters, optionally steered by a DQN controller | `DecisionAgent` + `utils/rl_controller.py` |
| **Attack Agent** | Evolves the suffix population via word-level / module-level / multi-granularity operators | `AttackAgent` + `utils/opt_utils.py` |

The judge is a **vLLM-served local model** (the default runners target Qwen3-VL-8B-Instruct), implemented in `utils/vllm_judge.py`.

---

## Project Structure

```
├── Multi_Agent_Framework.py           # main attack entry point
├── utils/
│   ├── vllm_judge.py                  # vLLM-backed judge + scoring
│   ├── vllm_judge_enhanced.py         # enhanced judge (v2)
│   ├── judge.py                       # judge base/abstractions
│   ├── qwen_judge.py                  # Qwen judge helpers
│   ├── rl_controller.py               # DQN controller (state → action)
│   ├── opt_utils.py                   # word-level / module-level / hybrid operators
│   ├── pairwise_loader.py             # data loading from JSON arrays
│   ├── string_utils.py                # suffix / template utilities
│   └── data_types.py                  # dataclasses & enums
├── src/defense/
│   ├── model_defense.py               # LR classifier + semantic-similarity defenses
│   ├── mlp_defense.py                 # MLP classifier defense
│   └── ppl_defense.py                 # perplexity (PPL) defense utilities
│   ├── tail_consistency_defense.py    # Tail-Consistency Defense (TCD) core
├── run_*.sh                           # shell runners (see below)
├── run_stealth_defenses.py            # LR + similarity defense runner
├── run_mlp_defense.py                 # MLP defense runner
├── run_tail_consistency_defense.py    # TCD standalone runner
├── pred_defense.py                    # PRED (PPL ensemble) defense
├── pred_tcd_cascade.py                # PRED + TCD two-stage cascade
├── check_asr_pairwise.py              # ASR / query statistics from a result file
├── data/split/                        # datasets + split-info files
├── assets/prompt_group.pth            # seed suffix population (loaded at startup)
```

---

## Data

Data lives under `data/split/`. Each benchmark has a **split-info** file mapping to its train/test JSON:

| Benchmark | Split info | Train / Test |
|---|---|---|
| Alpaca Eval | `data/split/alpaca_eval_split_info.json` | `alpaca_eval_train.json` / `alpaca_eval_test.json` |
| Arena Hard | `data/split/arena_hard_split_info.json` | `arena_hard_train.json` / `arena_hard_test.json` |
| Code Judge Bench | `data/split/code_judge_bench_split_info.json` | `code_judge_bench_train.json` / `code_judge_bench_test.json` |

### Format

Data files are **JSON arrays** (`[...]`), *not* JSONL — the loader calls `json.load`, not line-by-line parsing (`utils/pairwise_loader.py`). Each element has this shape:

```json
{
  "question_id": "alpaca_445",
  "instruction": "...",
  "response_a": "...",
  "response_b": "...",
  "model_a": "gpt-4o-2024-05-13",
  "model_b": "claude-3-5-sonnet-20240620",
  "metadata": { "...": "..." }
}
```

A split-info file looks like:

```json
{
  "benchmark": "alpaca_eval",
  "data_files": {
    "train": "data/split/alpaca_eval_train.json",
    "test":  "data/split/alpaca_eval_test.json"
  }
}
```

---

## Quick start (main attack)

### Setup environment

```bash
# clone and enter the repo
git clone https://github.com/ai4ed/PreferAttack.git
cd PreferAttack

# install deps (see Requirements)
pip install -r requirements.txt
```

### Run the attack

The main entry point is `Multi_Agent_Framework.py`, which reads samples one at a time and evolves a suffix population per sample.

```bash
python Multi_Agent_Framework.py \
  --qwen3_path /path/to/your/judge/model \
  --split_info data/split/alpaca_eval_split_info.json \
  --split test \
  --num_samples 20 \
  --batch_size 64 \
  --num_steps 100 \
  --use_rl_controller \
  --gpt_mutation_prob 0.05 \
  --save_path results/demo.json
```

Results are written to the path given by **`--save_path`** (there is **no** `result.dir` config file — output location is controlled directly by this argument and by `--save-dir` in the shell runners). Each result file is a JSON object `{"meta": {...}, "records": [...]}` with per-sample `baseline`, `attack`, `success`, and `queries` fields, plus an `aggregate_stats` block.

### Via the shell runners

The runners are written for a **Linux/bash + GNU `getopt`** environment and default to server-specific model paths. Override the model path with `--qwen3-path`.

| Script | Purpose |
|---|---|
| `run_pairwise_qwen3.sh` | serial multi-dataset run (default `alpaca_eval arena_hard`) |
| `run_pairwise_qwen3_all.sh` | outer multi-model × inner multi-dataset sweep |
| `run_pairwise_qwen3_all_code_judge.sh` | multi-model sweep on Code Judge Bench |
| `run_seed_sweep.sh` | 5-seed sweep on a 20-sample subset |
Example:

```bash
bash run_pairwise_qwen3.sh \
  --qwen3-path /path/to/your/judge/model \
  --datasets "alpaca_eval arena_hard" \
  --num-samples 20
```

### Key parameters

| Argument | Default | Meaning |
|---|---|---|
| `--qwen3_path` | *(required)* | local judge model path |
| `--split_info` / `--data_json` | — | dataset split-info file (or a single data JSON) |
| `--split` | `test` | which split to run (`train` / `test`) |
| `--num_samples` | 50 | number of samples to attack |
| `--start` | 0 | starting sample index |
| `--num_steps` | 100 | max search generations per sample |
| `--batch_size` | 256 | suffix population size |
| `--num_elites` | 0.05 | elite fraction for selection |
| `--crossover` | 0.5 | crossover probability |
| `--mutation` | 0.01 | mutation probability (see note below) |
| `--gpt_mutation_prob` | unset | GPT mutation prob — the runners set it to `0.05` |
| `--patience` | 10 | generations without improvement before early stop / fallback |
| `--batch_max_size` | 32 | max judge batch size per call |
| `--word_dict_topk` | 2000 | momentum-dictionary prune size |
| `--use_rl_controller` | off | enable the DQN controller |
| `--rl_lr` / `--rl_gamma` / `--rl_epsilon` | `0.1` / `0.9` / `0.2` | RL hyper-parameters |
| `--stop_on_first_success` | off | stop a sample immediately after the first flip |
| `--append_to` | `instruction` | where to append the suffix |
| `--seed` | unset | global RNG seed for reproducibility |

---

## Configuration

### vLLM environment variables

The shell runners set these; set them yourself if invoking the Python entry directly:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SWAP_SPACE_GB=32
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES=0
```

### Paths

- **Judge model** — `--qwen3_path` (required). The runners default to `/root/autodl-tmp/Qwen3-VL-8B-Instruct/`; override for your machine.
- **Data** — `--split_info data/split/<benchmark>_split_info.json`, or `--data_json` for a single data file.
- **Output** — `--save_path` (main entry) or `--save-dir` (runners); default `results/`.
- **Seed population** — the initial suffix population is loaded from `assets/prompt_group.pth` (a PyTorch tensor) at startup via `torch.load`.

---

## Evaluating results (ASR & query statistics)

```bash
python check_asr_pairwise.py --path results/<your_result_file>.json
```

This prints **ASR** (attack success rate) plus average **API calls**, **candidates evaluated**, and **token** counts (overall and for successful samples).
---

## Defense experiments

Two defense runners reproduce the metric protocol of the paper's PPL-defense tables (`ASR` / `ASR-W` / `ASR-Reduction` / `FNR` / `FPR`), applied to *learned* and *semantic-similarity* detectors:

```bash
python run_stealth_defenses.py   # LR classifier + similarity defenses → results/stealth_defenses_summary.json
python run_mlp_defense.py        # MLP classifier defense (prints LR-vs-MLP comparison) → results/mlp_defense_summary.json
```

Underlying utilities live in `src/defense/` (`model_defense.py`, `mlp_defense.py`, `ppl_defense.py`).


---

### Additional defenses (TCD / PRED / cascade)

| File | Role |
|---|---|
| `src/defense/tail_consistency_defense.py` | **Tail-Consistency Defense (TCD)** core. Instead of detecting "does this suffix look adversarial", it asks whether the suffix has *causal influence*: it re-queries the judge after truncating the last `K` tokens of the attacked instruction. Benign preferences are truncation-robust, whereas an append-only attack lives entirely in the tail, so truncation reverts it to baseline. |
| `run_tail_consistency_defense.py` | **TCD standalone runner.** Loads a vLLM judge, calibrates the TCD threshold on clean pairs, evaluates TCD on attack samples, and writes `ASR / ASR-W / ASR-Reduction / FNR / FPR` to a summary JSON. |
| `pred_defense.py` | **PRED (PREference-Robust Ensemble Defense).** Three PPL-derived gates — windowed PPL `S_w`, local anomaly factor `S_l = ppl_w/ppl`, and relative ratio `S_r = ppl_w(attacked)/ppl_w(clean)` — each calibrated on clean data to FPR ≤ α and combined under OR / AND / per-gate / majority modes. |
| `pred_tcd_cascade.py` | **PRED + TCD two-stage cascade.** Cheap offline PRED runs first (`filter` / `escalate` / `accept`); borderline samples escalate to the expensive online TCD truncation re-judgment. |

```bash
# TCD (v1 agreement-based; add --use_v2 for the position-bias-corrected variant)
python src/defense/run_tail_consistency_defense.py \
  --attack_json results/<attack>.json \
  --judge_model /path/to/judge_model \
  --use_v2
```
---

PRED reads the per-sample PPL signals produced by the `compute_fnr_fpr.py` step:

```bash
# PRED (PPL ensemble) — input: results_fnr_fpr_<dataset>_llama3b.json
python pred_defense.py \
  --inputs results_fnr_fpr_cjb_llama3b.json results_fnr_fpr_<dataset>_llama3b.json \
  --target_fpr 0.01 \
  --asr_w_dir /root/PreferAttack \
  --output_json results/pred_defense_summary.json
```

The cascade runs PRED + TCD end-to-end and loads a judge_model checkpoint as both the PPL detector and the judge:

```bash
# PRED + TCD cascade
python pred_tcd_cascade.py \
  --attack_results_json results/<attack>.json \
  --fnr_fpr_json results_fnr_fpr_cjb_llama3b.json \
  --ppl_judge_model /path/to/judge_model \
  --clean_calibration_json data/split/code_judge_bench_train.json \
  --output_json results/results_cascade_cjb.json
```
## Citation

If you use this code, please cite:

```bibtex
@article{2026zhoua,
  title={A Collaborative Multi-Agent Framework for Preference-Reversal Attacks on LLM-as-a-Judge Models},
  author={Zhou, Zheng and Li, Xueyi and Dou, Wenzhou and Liu, Zitao},
  journal={Expert Systems with Applications},
  volume={333},
  pages={133992},
  year={2026}
}
```

<!-- TODO: the manuscript PDF is password-protected, so author names could not be
     extracted automatically. Replace <authors> and confirm the venue/year. -->

---

## License

Apache-2.0
