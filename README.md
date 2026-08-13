# PreferAttack
# A Collaborative Multi-Agent Framework for Preference Attacks on LLM-as-a-Judge Models

## Abstract

Large language models (LLMs) are increasingly employed as evaluative judges in a wide range of decision-making tasks. However, these models remain vulnerable to preference-reversal attacks, where adversaries manipulate prompts to induce preference judgments opposite to the models’ originally aligned behavior. Existing work most relevant to this problem largely builds on attack mechanisms adapted from jailbreak methods. Unfortunately, when transferred to preference-reversal settings, these jailbreak-derived attack paradigms exhibit three key limitations: (1) \textbf{optimization–feedback mismatch}, as most existing jailbreak attacks are designed to optimize against continuous surrogate signals, whereas preference-reversal tasks are evaluated through discrete, noisy, and comparison-dependent feedback; (2) \textbf{insufficient strategic adaptability}, as a  fixed jailbreak attack paradigm cannot reliably handle input pairs with varying reversal difficulty and sensitivity; and (3) \textbf{limited stealthiness}, as gradient-based optimization often produces semantically unnatural prompts that distort the original pairwise comparison context, making attacks easier to detect. In light of these challenges, we ask: Can we develop a collaborative preference-reversal attack framework that adapts to discrete and noisy decision feedback, dynamically selects effective strategies for heterogeneous samples, and generates stealthy prompts that evade detection? In this paper, we propose PreferAttack, a collaborative multi-agent framework that addresses these challenges through coordinated scoring, decision, and attack agents. Across three benchmark datasets and ten judge models, PreferAttack outperforms competitive baselines in attack success rate and achieves a favorable trade-off between success rate and attack efficiency. 

## Installation

### Requirements

- Python 3.12
- CUDA-capable GPU (recommended)
- PyTorch 2.8.0
- Transformers 4.57.3

### Setup

```bash
git clone https://github.com/xjyp/PreferAttack.git
cd CamouflageAttack
pip install -r requirements.txt
```

## Quick Start

### Run Attack

```bash
bash run_pairwise.sh
```

The results will be saved in the directory specified by `result.dir` in the configuration file.

## Project Structure

```
Multi-Agent Framework/
├── assets/
│   ├── build_prompt_group_pairwise_target_append.py
│   ├── prompt_group.pth
│   ├── prompt_group_multi_strategy.pth
│   ├── prompt_group_pairwise.pth
│   └── prompt_group_pairwise_target_append_custom_1010...
│
├── data/
│   └── split/
│       ├── alpaca_eval_split_info.json
│       ├── alpaca_eval_test.json
│       ├── alpaca_eval_train.json
│       ├── arena_hard_split_info.json
│       ├── arena_hard_test.json
│       ├── arena_hard_train.json
│       ├── code_judge_bench_split_info.json
│       ├── code_judge_bench_test.json
│       └── code_judge_bench_train.json
│
├── src/
│   └── defense/
│       ├── __init__.py
│       └── ppl_defense.py
│
├── utils/
│   ├── data_types.py
│   ├── judge.py
│   ├── opt_utils.py
│   ├── pairwise_loader.py
│   ├── qwen_judge.py
│   ├── rl_controller.py
│   ├── string_utils.py
│   └── vllm_judge.py
│
├── LICENSE
├── Multi_Agent_Framework.py
├── README.md
├── check_asr_pairwise.py
├── requirements.txt
└── run_pairwise_qwen3.sh
```

## Configuration

### Data Configuration

- `name`: Dataset name (for logging purposes)
- `path`: Path to dataset file (JSONL format)

### Output Configuration

- `log.dir`: Directory for log files
- `result.dir`: Directory for result files

```

## How It Works

PreferAttack employs a three-agent framework:

1. **Decision Agent**: Improves strategic adaptability by dynamically selecting attack strategies and adaptively scheduling attack parameters according to task-specific feedback and intermediate search states.

2. **Scoring Agent**: Mitigates optimization–feedback mismatch by modeling the judge in a preference-aware manner and deriving a task-specific fitness signal from discrete preference-reversal outcomes.

3. **Action Agent**: Enhances stealthiness by applying semantics-preserving transformations to generate and iteratively refine adversarial prompts while preserving semantic coherence with the original comparison context.



## Citation

If you find this repository useful in your research, please consider citing our work:

```bibtex
@article{zhou2026collaborative,
  title={A Collaborative Multi-Agent Framework for Preference Attacks on LLM-as-a-Judge Models},
  author={Zhou,Zheng and Dou, Wenzhou and Li, Xueyi and  Liu, Zitao},
  journal={Expert Systems with Applications},
  volume={xxx},
  pages={xxx-xxx},
  year={2026}
}
```


## License

MIT License
