"""
v2 入口脚本: 用 EnhancedVLLMJudge 替换原 VLLMJudge 跑 Multi_Agent_Framework。

设计目标 (来自用户决策):
1. 不动原文件 (Multi_Agent_Framework.py / utils/vllm_judge.py)
2. 增强解析器: utils/vllm_judge_enhanced.py
3. baseline preference=None 时跳过该样本 (不抛 int(None) 错误, 不污染优化)
4. candidate preference=None 时, score_candidates 已有逻辑 (line 145-147) 会给 0.5 中性分

实现方式:
- 不复制 720 行 main 函数, 而是 inspect.getsource(maf.main) 拿源码 →
  字符串替换 judge 工厂和 baseline 保护 → exec 生成 main_v2
- 替换点用 assertion 保护, 原文件结构变化时会立刻报错而不是悄悄失效

CLI 用法 (与 Multi_Agent_Framework.py 完全相同, 多一个 --retry_max_tokens):
    python Multi_Agent_Framework_v2.py \\
        --device cuda --dtype bf16 \\
        --qwen3_path /root/autodl-tmp/MiniCPM5-1B \\
        --split_info data/split/alpaca_eval_split_info.json \\
        --split test --num_samples 50 --num_steps 50 \\
        --save_path results/v2_alpaca_minicpm5.json \\
        --retry_max_tokens 64
"""

import inspect
import os
import sys

# 把当前目录加入 path, 确保可以 import Multi_Agent_Framework
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Multi_Agent_Framework as maf
from utils.vllm_judge_enhanced import create_enhanced_vllm_judge


# ====== 1. monkey-patch: 把 create_vllm_judge 替换为 enhanced 版本 ======
# 注意: 原 main 里调用是 create_vllm_judge(model_path=..., dtype=..., seed=...)
# enhanced 工厂同签名 + retry_max_tokens (从环境变量读, 避免 argv 干扰)

def _patched_create_vllm_judge(*a, **kw):
    retry_max_tokens = int(os.environ.get("ENHANCED_JUDGE_RETRY_MAX_TOKENS", "64"))
    kw.setdefault("retry_max_tokens", retry_max_tokens)
    return create_enhanced_vllm_judge(*a, **kw)


maf.create_vllm_judge = _patched_create_vllm_judge


# ====== 2. 动态生成 main_v2: 改 judge 工厂调用 + 加 baseline None 保护 ======
def _build_main_v2():
    src = inspect.getsource(maf.main)

    # 替换 1: 把 main 改名为 main_v2 (避免与原 main 冲突)
    assert src.startswith("def main():"), "expected main() signature"
    src = src.replace("def main():", "def main_v2():", 1)

    # 替换 2: judge 工厂调用替换 (注意原文件用 create_vllm_judge)
    # monkey-patch 已经替换了符号, 但 main 源码内部仍是直接调用 create_vllm_judge
    # 用源码层的替换确保 enhanced 版本被使用
    old_judge_line = "judge = create_vllm_judge(model_path=args.qwen3_path or \"Qwen/Qwen3-8B\", dtype=args.dtype, seed=args.seed)"
    new_judge_line = (
        "judge = _patched_create_vllm_judge("
        "model_path=args.qwen3_path or \"Qwen/Qwen3-8B\", "
        "dtype=args.dtype, seed=args.seed)"
    )
    assert old_judge_line in src, f"judge line not found in source, original may have changed"
    src = src.replace(old_judge_line, new_judge_line)

    # 替换 3: 在 base_resp = attack_agent.judge_pairwise(pa) 后插入 None 保护
    old_baseline_block = (
        "        base_resp = attack_agent.judge_pairwise(pa)\n"
        "        base_choice = base_resp.preference\n"
    )
    new_baseline_block = (
        "        base_resp = attack_agent.judge_pairwise(pa)\n"
        "        # [v2 patch] baseline preference=None 表示 judge 输出无法解析\n"
        "        # (EnhancedVLLMJudge retry 后仍无效), 跳过该样本避免污染优化轨迹\n"
        "        if base_resp.preference is None:\n"
        "            print(f\"[v2-skip] sample id={ex.get('meta', {}).get('id', idx)} \"\n"
        "                  f\"baseline unparseable, raw={(base_resp.raw_response or '')[:80]!r}, skipping\")\n"
        "            skipped_baseline_invalid += 1\n"
        "            continue\n"
        "        base_choice = base_resp.preference\n"
    )
    assert old_baseline_block in src, "baseline block not found, original may have changed"
    src = src.replace(old_baseline_block, new_baseline_block)

    # 替换 4: 在主循环开始前初始化 skipped_baseline_invalid 计数器
    # 找 "for idx, ex in todo_items:" 这一行, 在它之前插入初始化
    old_loop_start = "    # 全局 wall-clock 起点，用于吞吐率 / 成本统计\n    overall_t0 = time.time()\n"
    new_loop_start = (
        "    # 全局 wall-clock 起点，用于吞吐率 / 成本统计\n"
        "    overall_t0 = time.time()\n"
        "    # [v2 patch] 记录因 baseline 无法解析而跳过的样本数\n"
        "    skipped_baseline_invalid = 0\n"
    )
    assert old_loop_start in src, "loop start marker not found"
    src = src.replace(old_loop_start, new_loop_start)

    # 替换 5: 在 main 函数末尾打印 skipped 统计
    # 找 main 函数末尾的 aggregate stats 打印部分, 在前面加 skipped 打印
    # 注意原 main 末尾是 try/except 包了 aggregate stats 计算
    src = src.rstrip()
    src += (
        "\n    # [v2 patch] 打印因 baseline 无法解析而跳过的样本统计\n"
        "    try:\n"
        "        print(f\"\\n=== v2 skipped samples (baseline unparseable) ===\\n\"\n"
        "              f\"skipped_baseline_invalid = {skipped_baseline_invalid} / {len(todo_items)}\")\n"
        "    except Exception:\n"
        "        pass\n"
    )

    # exec 进 maf 命名空间 (这样函数内引用的全局符号都能解析)
    # 注意: _patched_create_vllm_judge 必须能在 maf 命名空间里访问
    maf._patched_create_vllm_judge = _patched_create_vllm_judge
    exec(compile(src, "<main_v2>", "exec"), maf.__dict__)
    return maf.__dict__["main_v2"]


main_v2 = _build_main_v2()


def _consume_retry_arg_from_argv():
    """从 sys.argv 提取 --retry_max_tokens, 移除它避免原 argparse 报 unknown arg."""
    if "--retry_max_tokens" in sys.argv:
        i = sys.argv.index("--retry_max_tokens")
        if i + 1 < len(sys.argv):
            val = int(sys.argv[i + 1])
            os.environ["ENHANCED_JUDGE_RETRY_MAX_TOKENS"] = str(val)
            # 移除 --retry_max_tokens 和它的值
            sys.argv.pop(i)
            sys.argv.pop(i)
            print(f"[v2] retry_max_tokens = {val}")


if __name__ == "__main__":
    _consume_retry_arg_from_argv()
    main_v2()
