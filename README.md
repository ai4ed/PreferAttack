# PreferAttack

**PreferAttack** 是一个针对 **LLM-as-a-Judge（大模型评判器）** 的**协作式多智能体偏好翻转攻击**框架。它优化一段追加在用户指令后的*对抗后缀*，使评判器在两条候选回答（A vs. B）之间翻转其成对偏好，而回答本身保持不变。

**PreferAttack** is a collaborative multi-agent framework for **preference-reversal attacks** on **LLM-as-a-Judge** models. It optimizes an *adversarial suffix* appended to the user instruction so that the judge flips its pairwise preference between two candidate responses (A vs. B), while the responses themselves are left untouched.

<p align="center">
  🖥️ <a href="#quick-start-main-attack"><b>快速开始 / Quickstart</b></a> &nbsp; | &nbsp;
  🧪 <a href="#defense-experiments"><b>防御实验 / Defense</b></a> &nbsp; | &nbsp;
  ⚠️ <a href="#reproducibility-status"><b>可复现性 / Reproducibility</b></a> &nbsp; | &nbsp;
  ©️ <a href="#citation"><b>引用 / Citation</b></a>
</p>

> 本仓库提供论文 *"A Collaborative Multi-Agent Framework for Preference Attacks on LLM-as-a-Judge Models"*（投稿 ESWA）的参考实现。完整的**代码 ↔ 论文一致性审计**见 [`PreferAttack代码与论文一致性审计报告.md`](PreferAttack代码与论文一致性审计报告.md)。
>
> This repository provides the reference implementation accompanying the manuscript *"A Collaborative Multi-Agent Framework for Preference Attacks on LLM-as-a-Judge Models"* (submitted to ESWA). A full **code ↔ paper consistency audit** is in [`PreferAttack代码与论文一致性审计报告.md`](PreferAttack代码与论文一致性审计报告.md) (in Chinese).

---

## 亮点 / Highlights

- **协作式多智能体优化** —— 评分智能体（Scoring）、决策智能体（Decision）与攻击智能体（Attack）协同搜索能翻转偏好的后缀。
- **自适应搜索** —— DQN 控制器观测搜索状态（改进量、多样性、patience、平均适应度），在线调节遗传算法的超参数。
- **多粒度算子** —— 词级重组、模块级注意力重组，以及二者的混合。
- **天然隐蔽性** —— 后缀通过*语义保持*变换演化，而非注入乱码。
- **三个 LLM-as-a-Judge 基准** —— Alpaca Eval、Arena Hard、Code Judge Bench。
- **防御研究** —— 困惑度（PPL）、学习式分类器（LR / MLP）、语义相似度检测器。

> ⚠️ 以上是论文声明的主要贡献。系统审计（链接见上）发现其中若干项在当前代码中**并未忠实实现**，请先阅读 [可复现性状态](#reproducibility-status) 再引用相关数字。
>
> ⚠️ These are the paper's stated contributions. A systematic audit (linked above) found that several of them are **not faithfully implemented** in this checkout. See [Reproducibility status](#reproducibility-status) before relying on the numbers.

---

## 概述 / Overview

给定一个成对评测样本 `(instruction, response_a, response_b)`，评判模型先给出初始偏好。PreferAttack 搜索一个短后缀 `s`，使得

```
instruction' = instruction.rstrip() + " " + s
```

当评判器对 `(instruction', response_a, response_b)` 进行评判时，其偏好被翻转。攻击的**隐蔽性**来自：后缀通过*语义保持*变换演化，而非注入乱码。

Given a pairwise evaluation sample `(instruction, response_a, response_b)`, a judge model produces an initial preference. PreferAttack searches for a short suffix `s` such that, when judged on `(instruction', response_a, response_b)`, the judge flips its preference. The attack is **stealthy** by design: the suffix is evolved through *semantic-preserving* transformations rather than injected gibberish.

框架由三个协同智能体组成 / The framework is composed of three collaborating agents:

| 智能体 / Agent | 职责 / Responsibility | 位置 / Where |
|---|---|---|
| **评分智能体 / Scoring** | 构造攻击候选、查询评判器、计算适应度（式(6)：`翻转 → 1−conf`，`未翻转 → conf`；越小越好） | `ScoringAgent` in `Multi_Agent_Framework.py` |
| **决策智能体 / Decision** | 选择搜索模式与 GA 超参数，可选由 DQN 控制器驱动 | `DecisionAgent` + `utils/rl_controller.py` |
| **攻击智能体 / Attack** | 通过词级 / 模块级 / 多粒度算子演化后缀种群 | `AttackAgent` + `utils/opt_utils.py` |

评判器是 **vLLM 服务的本地模型**（默认 runner 面向 Qwen3-VL-8B-Instruct），实现于 `utils/vllm_judge.py`。

The judge is a **vLLM-served local model** (the default runners target Qwen3-VL-8B-Instruct), implemented in `utils/vllm_judge.py`.

---

## 目录结构 / Repository layout

```
├── Multi_Agent_Framework.py           # 主攻击入口 / main attack entry point
├── Multi_Agent_Framework_v2.py        # 实验性 v2 入口（见审计 §7 — 接口不匹配）
├── utils/
│   ├── vllm_judge.py                  # vLLM 评判器 + 评分
│   ├── vllm_judge_enhanced.py         # 增强评判器（v2）
│   ├── judge.py                       # 评判器基类 / 抽象
│   ├── qwen_judge.py                  # Qwen 评判器辅助
│   ├── rl_controller.py               # DQN 控制器（状态 → 动作）
│   ├── opt_utils.py                   # 词级 / 模块级 / 混合算子
│   ├── pairwise_loader.py             # JSON 数组数据加载
│   ├── string_utils.py                # 后缀 / 模板工具
│   └── data_types.py                  # 数据类与枚举
├── src/defense/
│   ├── model_defense.py               # LR 分类器 + 语义相似度防御
│   ├── mlp_defense.py                 # MLP 分类器防御
│   └── ppl_defense.py                 # 困惑度（PPL）防御工具
├── run_*.sh                           # shell runner（见下）
├── run_stealth_defenses.py            # LR + 相似度防御 runner
├── run_mlp_defense.py                 # MLP 防御 runner
├── check_asr_pairwise.py              # 从结果文件统计 ASR / 查询次数
├── data/split/                        # 数据集 + split-info 文件
├── assets/prompt_group.pth            # 初始后缀种群（启动时加载）
└── *.tex                              # rebuttal 用 LaTeX 表格
```

---

## 依赖 / Requirements

固定版本依赖见 [`requirements.txt`](requirements.txt)：

The pinned dependencies are in [`requirements.txt`](requirements.txt):

```
fschat==0.2.20  nltk==3.8.1  numpy==1.26.0  openai==0.28.1
torch==2.0.1  tqdm==4.66.1  transformers==4.28.0  sentencepiece==0.1.99
protobuf==4.24.4  accelerate==0.23.0
```

**注意：** `requirements.txt` **不足以**跑完整流程。视所跑实验，还需额外安装：

**Important:** `requirements.txt` is *not* sufficient to run the full pipeline. Depending on which experiments you run, you will additionally need:

| 依赖 / Dependency | 用途 / Needed for |
|---|---|
| `vllm` | 主攻击（`utils/vllm_judge.py` 使用进程内 `vllm.LLM`） |
| `sentence-transformers` | 防御实验（MiniLM 嵌入） |
| `scikit-learn` | 防御实验（LR / 交叉验证 / 指标） |

主攻击需要 CUDA GPU 与本地评判模型（如 Qwen3 / Qwen2.5 / Llama 指令微调权重）。shell runner 硬编码了服务器路径（`/root/autodl-tmp/...`），请用 `--qwen3-path` 覆盖（见下）。

A CUDA-capable GPU and a local judge model (e.g. a Qwen3 / Qwen2.5 / Llama instruction-tuned checkpoint) are required for the main attack. The shell runners hard-code server-specific paths (`/root/autodl-tmp/...`) — override them with `--qwen3-path` (see below).

> **环境说明：** 论文实现细节写 Python 3.12 / PyTorch 2.8 / Transformers 4.57，但 `requirements.txt` 固定的是 `torch==2.0.1` / `transformers==4.28.0`。请以 pin 文件作为本 checkout 的权威依赖。
>
> **Note on the environment claim:** the paper's implementation details mention Python 3.12 / PyTorch 2.8 / Transformers 4.57, but `requirements.txt` pins `torch==2.0.1` / `transformers==4.28.0`. Treat the pin file as the authoritative dependency set for this checkout.

---

## 数据 / Data

数据位于 `data/split/`。每个基准有一个 **split-info** 文件映射到其 train/test JSON：

Data lives under `data/split/`. Each benchmark has a **split-info** file mapping to its train/test JSON:

| 基准 / Benchmark | Split info | Train / Test |
|---|---|---|
| Alpaca Eval | `data/split/alpaca_eval_split_info.json` | `alpaca_eval_train.json` / `alpaca_eval_test.json` |
| Arena Hard | `data/split/arena_hard_split_info.json` | `arena_hard_train.json` / `arena_hard_test.json` |
| Code Judge Bench | `data/split/code_judge_bench_split_info.json` | `code_judge_bench_train.json` / `code_judge_bench_test.json` |

### 格式 / Format

数据文件是 **JSON 数组**（`[...]`），**不是** JSONL —— loader 调用 `json.load`，而非逐行解析（`utils/pairwise_loader.py`）。每个元素结构如下：

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

split-info 文件形如 / A split-info file looks like:

```json
{
  "benchmark": "alpaca_eval",
  "data_files": {
    "train": "data/split/alpaca_eval_train.json",
    "test":  "data/split/alpaca_eval_test.json"
  }
}
```

> **已知问题（审计 §5.2）：** Code Judge Bench 按回答对（response pair）而非 `question_id` 切分 —— 约 87% 的测试行与训练集共享 `question_id`。在用于“未见题目泛化”结论前，请先查阅审计报告。
>
> **Known issue (audit §5.2):** Code Judge Bench is split at the response-pair level, not by `question_id` — ~87% of test rows share a `question_id` with the training split. See the audit report before relying on it for "unseen-question generalization" claims.

---

## 快速开始（主攻击）/ Quick start (main attack)

### 环境准备 / Setup environment

```bash
# 1) 克隆并进入仓库 / clone and enter the repo
git clone <repo-url> PreferAttack
cd PreferAttack

# 2) 安装依赖（见依赖说明 —— 需显式加 vllm）
# install deps (see Requirements — add vllm explicitly)
pip install -r requirements.txt vllm
```

### 运行攻击 / Run the attack

主入口为 `Multi_Agent_Framework.py`，逐样本读取数据，并为每个样本演化后缀种群。

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

结果写入 **`--save_path`** 指定路径（**没有** `result.dir` 配置文件 —— 输出位置由该参数、以及 shell runner 中的 `--save-dir` 直接控制）。每个结果文件是 JSON 对象 `{"meta": {...}, "records": [...]}`，含逐样本的 `baseline`、`attack`、`success`、`queries` 字段，以及 `aggregate_stats` 汇总块。

Results are written to the path given by **`--save_path`** (there is **no** `result.dir` config file — output location is controlled directly by this argument and by `--save-dir` in the shell runners). Each result file is a JSON object `{"meta": {...}, "records": [...]}` with per-sample `baseline`, `attack`, `success`, and `queries` fields, plus an `aggregate_stats` block.

### 通过 shell runner / Via the shell runners

runner 面向 **Linux/bash + GNU `getopt`** 环境，默认使用服务器路径。请用 `--qwen3-path` 覆盖模型路径。

The runners are written for a **Linux/bash + GNU `getopt`** environment and default to server-specific paths. Override the model path with `--qwen3-path`.

| 脚本 / Script | 用途 / Purpose |
|---|---|
| `run_pairwise_qwen3.sh` | 串行多数据集运行（默认 `alpaca_eval arena_hard`） |
| `run_pairwise_qwen3_all.sh` | 外层多模型 × 内层多数据集 sweep |
| `run_pairwise_qwen3_all_code_judge.sh` | Code Judge Bench 上的多模型 sweep |
| `run_seed_sweep.sh` | 20 条样本子集上的 5-seed sweep |
| `run_phase1_alpaca.sh` / `run_phase2_best_seed.sh` | 多 seed 阶段 1 / 阶段 2 运行 |
| `chain_sweep_to_phase2.sh` / `chain_phase2_to_alpaca_qwen15b.sh` | 链式 sweep → 阶段运行 |
| `run_ablation_wordlevel.sh` / `run_ablation_modulelevel.sh` | 词级 / 模块级消融 |

示例 / Example:

```bash
bash run_pairwise_qwen3.sh \
  --qwen3-path /path/to/your/judge/model \
  --datasets "alpaca_eval arena_hard" \
  --num-samples 20
```

### 关键参数 / Key parameters

| 参数 / Argument | 默认值 / Default | 含义 / Meaning |
|---|---|---|
| `--qwen3_path` | *(必填 / required)* | 本地评判模型路径 |
| `--split_info` / `--data_json` | — | 数据集 split-info 文件（或单个数据 JSON） |
| `--split` | `test` | 运行哪个 split（`train` / `test`） |
| `--num_samples` | 50 | 攻击的样本数 |
| `--start` | 0 | 起始样本下标 |
| `--num_steps` | 100 | 每样本最大搜索代数 |
| `--batch_size` | 256 | 后缀种群大小 |
| `--num_elites` | 0.05 | 精英比例（选择用） |
| `--crossover` | 0.5 | 交叉概率 |
| `--mutation` | 0.01 | 变异概率（见下注） |
| `--gpt_mutation_prob` | 未设 / unset | GPT 变异概率 —— runner 设为 `0.05` |
| `--patience` | 10 | 早停 / 回退前无改进代数 |
| `--batch_max_size` | 32 | 每次调用的最大评判批大小 |
| `--word_dict_topk` | 2000 | 动量词典裁剪大小 |
| `--use_rl_controller` | 关 / off | 启用 DQN 控制器 |
| `--rl_lr` / `--rl_gamma` / `--rl_epsilon` | `0.1` / `0.9` / `0.2` | RL 超参数 |
| `--use_cache` | 关 / off | 启用逐样本评判缓存 |
| `--stop_on_first_success` | 关 / off | 首次翻转后立即停止该样本 |
| `--append_to` | `instruction` | 后缀追加位置 |
| `--seed` | 未设 / unset | 全局随机种子（可复现） |
| `--verify_best_every` | 0 | 每 N 代验证一次最佳后缀 |

---

## 配置 / Configuration

### vLLM 环境变量 / vLLM environment variables

shell runner 已设置这些变量；若直接运行 Python 入口，请自行设置：

The shell runners set these; set them yourself if invoking the Python entry directly:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_SWAP_SPACE_GB=32
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export CUDA_VISIBLE_DEVICES=0
```

### 路径 / Paths

- **评判模型** —— `--qwen3_path`（必填）。runner 默认 `/root/autodl-tmp/Qwen3-VL-8B-Instruct/`，请按你的机器覆盖。
  **Judge model** — `--qwen3_path` (required). Runners default to `/root/autodl-tmp/Qwen3-VL-8B-Instruct/`; override for your machine.
- **数据** —— `--split_info data/split/<benchmark>_split_info.json`，或 `--data_json` 指定单个数据文件。
  **Data** — `--split_info data/split/<benchmark>_split_info.json`, or `--data_json` for a single data file.
- **输出** —— `--save_path`（主入口）或 `--save-dir`（runner）；默认 `results/`。
  **Output** — `--save_path` (main entry) or `--save-dir` (runners); default `results/`.
- **种子种群** —— 初始后缀种群在启动时通过 `torch.load` 从 `assets/prompt_group.pth`（PyTorch 张量）加载。
  **Seed population** — the initial suffix population is loaded from `assets/prompt_group.pth` (a PyTorch tensor) at startup via `torch.load`.

---

## 结果评测（ASR 与查询统计）/ Evaluating results (ASR & query statistics)

```bash
python check_asr_pairwise.py --path results/<your_result_file>.json
```

该脚本打印 **ASR**（攻击成功率）以及平均 **API 调用次数**、**候选评估数**、**token 数**（总体与成功样本分别统计）。

This prints **ASR** (attack success rate) plus average **API calls**, **candidates evaluated**, and **token** counts (overall and for successful samples).

> **注意（审计 §3.1）：** `api_calls` 字段按 `ceil(候选数 / batch_max_size)` 计算，而非实际 `llm.generate` 调用数或候选评估数。`candidates_evaluated` 是更忠实的逐样本查询计数。
>
> **Note (audit §3.1):** the `api_calls` field is computed as `ceil(candidates / batch_max_size)`, not as the actual number of `llm.generate` calls or candidate evaluations. `candidates_evaluated` is the more faithful per-sample query counter.

---

## 防御实验 / Defense experiments

两个防御 runner 复现论文 PPL 防御表格的指标口径（`ASR` / `ASR-W` / `ASR-Reduction` / `FNR` / `FPR`），应用于*学习式*与*语义相似度*检测器：

Two defense runners reproduce the metric protocol of the paper's PPL-defense tables (`ASR` / `ASR-W` / `ASR-Reduction` / `FNR` / `FPR`), applied to *learned* and *semantic-similarity* detectors:

```bash
python run_stealth_defenses.py   # LR 分类器 + 相似度防御 → results/stealth_defenses_summary.json
python run_mlp_defense.py        # MLP 分类器防御（结尾打印 LR-vs-MLP 对比）→ results/mlp_defense_summary.json
```

底层工具位于 `src/defense/`（`model_defense.py`、`mlp_defense.py`、`ppl_defense.py`）。

Underlying utilities live in `src/defense/` (`model_defense.py`, `mlp_defense.py`, `ppl_defense.py`).

> **注意：** 这些脚本读取**未提交到仓库**的攻击结果文件（`results/*.json`），以及 `/root/autodan/...`、`/root/AutoDAN-main/...` 下的外部 GCG/AutoDAN 基线。请先用主入口生成 PreferAttack 攻击结果，并提供基线文件，这些 runner 才能复现其表格。
>
> **Note:** these scripts read attack-result files that are **not committed** to the repo (`results/*.json`) and external GCG/AutoDAN baselines under `/root/autodan/...` and `/root/AutoDAN-main/...`. Generate the PreferAttack attack results with the main entry point first, and provide the baseline files before these runners can reproduce their tables.

---

## 可复现性状态 / Reproducibility status

本仓库实现了 PreferAttack 的*总体*结构（后缀追加的 prompt 构造、成对评判 prompt、A/B 解析与置信度层级、式(6) 适应度形式、三智能体分解，以及 experience-replay / policy-net / target-net / ε-greedy 的 RL 骨架）。这些与论文一致。

This repository implements the *overall* PreferAttack structure (suffix-appending prompt construction, the pairwise judge prompt, the A/B parsing + confidence hierarchy, the Eq. (6) fitness form, the three-agent decomposition, and the experience-replay / policy-net / target-net / ε-greedy RL skeleton). These are consistent with the paper.

但系统审计（[`PreferAttack代码与论文一致性审计报告.md`](PreferAttack代码与论文一致性审计报告.md)）发现实质性偏差，意味着**当前代码尚不能被视为论文结果的严格可复现实现**。主要问题：

However, a systematic audit ([`PreferAttack代码与论文一致性审计报告.md`](PreferAttack代码与论文一致性审计报告.md)) found material discrepancies that mean **the current code should not yet be treated as a strict, ready-to-reproduce implementation of the paper's reported results.** The main ones:

1. **DQN 在多数样本上实际不学习。** 每个样本新建一个 `RLController`；经验池需 ≥64 条 transition 才开始训练，而每样本最多 100 代，导致很多（往往是最简单、最早成功的）样本在零梯度更新下完成，target network 从不同步（审计 §2.1）。
   **DQN does not actually learn on most samples.** A fresh `RLController` is created *per sample*; with a replay buffer that requires ≥64 transitions before training and ≤100 generations/sample, many (often the easiest, earliest-success) samples complete with zero gradient updates, and the target network never syncs (§2.1).
2. **DQN 动作空间 ≠ 论文。** 论文定义联合 `(strategy, mutation, crossover, top-k, elite)` 动作；代码用 11 个独立参数微调 + 4 个仅选策略的 no-op，且 `top-k` 指动量词典大小而非候选保留数（§2.2）。
   **DQN action space ≠ paper.** The paper defines joint `(strategy, mutation, crossover, top-k, elite)` actions; the code uses 11 independent parameter tweaks + 4 strategy-only no-ops, with `top-k` meaning momentum-dictionary size rather than candidate retention (§2.2).
3. **模块级注意力方向反转**，偏向复用*高*（更差）适应度类别（§2.3）；混合算子交错两组独立种群，而非论文的*级联* word-then-module 后代（§2.4）；词级算子未实现式(12) 的指数同义词混合或 `μ`/`B` 预算（§2.5）。
   **Module-level attention sign is inverted**, biasing re-use toward *high* (worse) fitness categories (§2.3); the hybrid operator interleaves two independent populations instead of the paper's *cascaded* word-then-module offspring (§2.4); the word-level operator does not implement Eq. (12)'s exponential synonym mixture or `μ`/`B` budget (§2.5).
4. **搜索解码 ≠ greedy。** 候选批默认 `temperature=0.1` / `max_new_tokens=10`，而非论文统一的 `temperature=0, max_tokens=16`（§2.6）；runner 用 `GPT_MUTATION_PROB=0.05`，而非论文的 `mutation=0.01`（§2.7）；默认运行**不会**首次成功即停止（§2.8）。
   **Search decoding ≠ greedy.** Candidate batches default to `temperature=0.1` / `max_new_tokens=10`, not the paper's uniform `temperature=0, max_tokens=16` (§2.6); the runner uses `GPT_MUTATION_PROB=0.05`, not the paper's `mutation=0.01` (§2.7); and the default run does **not** stop on first success (§2.8).
5. **指标：** `AQSA` 使用上述批计数统计（§3.1）；**BRR**（基线反转率）未实现（§3.2）；无法解析的*干净*基线输出被静默赋予随机标签（§3.3）；逐样本缓存默认关闭（§3.4）。
   **Metrics:** `AQSA` uses the batch-count statistic above (§3.1); **BRR** (Baseline Reversal Rate) is not implemented (§3.2); unparseable *clean* baseline outputs are silently assigned a random label (§3.3); the per-sample cache is disabled by default (§3.4).
6. **防御与数据：** ASR-W / ASR-R 只用成功子集做分母（§4.1）；完整 prompt 迁移检测器训练与测试视图不一致（§4.2）；防御脚本针对的模型/数据集与论文 Table 17 不同（§4.3）；TCD / PRED / 级联防御无代码（§4.4）；Code Judge Bench 存在题目级 train/test 交叉（§5.2）；数据 loader 丢失原始 `question_id`（§5.3）。
   **Defenses & data:** ASR-W / ASR-R use only the succeeded subset as denominator (§4.1); the full-prompt transfer detector trains and tests on inconsistent views (§4.2); defense scripts target a different model/dataset than the paper's Table 17 (§4.3); TCD / PRED / cascade defenses have no code (§4.4); Code Judge Bench has question-level train/test overlap (§5.2); the data loader drops the original `question_id` (§5.3).
7. **工程：** v2 入口（`Multi_Agent_Framework_v2.py`）存在评判器接口签名不匹配，无法完成主流程（§7）；消融入口文件（`Multi_Agent_Framework_WordLevelOnly.py` / `_ModuleLevelOnly.py`）本地存在但被 **git-ignore**，全新 clone 无法运行 `run_ablation_*.sh`（§6.2）。
   **Engineering:** the v2 entry (`Multi_Agent_Framework_v2.py`) has a judge-interface signature mismatch and cannot complete the main flow (§7); ablation entry files (`Multi_Agent_Framework_WordLevelOnly.py` / `_ModuleLevelOnly.py`) are present locally but **git-ignored**, so a fresh clone cannot run `run_ablation_*.sh` (§6.2).

审计报告（§11）给出了按优先级排序的修复计划。在第一、二优先级修复并重新运行实验之前，请将论文的 ASR、消融增益、查询效率与防御结论视为*待重新验证*。

The audit report (§11) gives a priority-ordered repair plan. Until the first- and second-priority fixes are applied and experiments re-run, treat the paper's ASR, ablation-gain, query-efficiency, and defense claims as *pending re-validation* against this code.

---

## 引用 / Citation

若使用本代码，请引用 / If you use this code, please cite:

```bibtex
@article{preferattack,
  title   = {A Collaborative Multi-Agent Framework for Preference Attacks on {LLM-as-a-Judge} Models},
  author  = {<authors>},   % TODO: 请从定稿论文填入作者名 / fill in from the camera-ready manuscript
  journal = {Expert Systems with Applications},
  note    = {Under review},
  year    = {2026}
}
```

<!-- TODO: 论文 PDF 为密码保护，无法自动提取作者名；请替换 <authors> 并确认 venue/year。
     TODO: the manuscript PDF is password-protected, so author names could not be
     extracted automatically. Replace <authors> and confirm the venue/year. -->

---

## 许可 / License

见 [`LICENSE`](LICENSE) / See [`LICENSE`](LICENSE).
