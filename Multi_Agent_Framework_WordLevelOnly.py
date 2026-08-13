"""
消融实验脚本: Attack Agent 只使用 Word-Level Recombination 动作。

背景
----
Multi_Agent_Framework.py 的 AttackAgent.run_update 在每一代会在两类动作里挑选:
  1. Word-Level Recombination  -> adaptive_word_level_recombination  (utils/opt_utils.py)
  2. Module-Level Recombination -> multi_granularity_hybrid           (utils/opt_utils.py)

主框架通过 strategy_mode / search_mode 在两者之间切换 (默认 hybrid, 即 Module-Level)。
本脚本通过 monkey-patch AttackAgent.run_update, 强制只调用 Word-Level 动作,
屏蔽 strategy_mode / search_mode 的影响, 用于消融对比两类动作各自的贡献。

注意: 不修改原 Multi_Agent_Framework.py, 仅在本文件中替换 run_update 行为。
其余流程 (ScoringAgent / DecisionAgent / RL 控制器 / 早停 / 统计) 保持一致,
确保消融结果与主框架结果直接可比。
"""
import sys
from pathlib import Path

# 保证当前目录在 sys.path 中, 使 `import Multi_Agent_Framework` 与原脚本里的
# `from utils.xxx import ...` 都能正确解析 (与 run_stealth_defenses.py 同样的做法)。
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import Multi_Agent_Framework as MAF
from utils.opt_utils import adaptive_word_level_recombination


def run_update_word_level_only(self, *, params, search_mode, population, score_list,
                               word_dict, num_elites_used, batch_size, crossover_used,
                               mutation_used, topk_used, reference):
    """强制只使用 Word-Level Recombination, 忽略 params['strategy_mode'] 与 search_mode。"""
    return adaptive_word_level_recombination(
        word_dict=word_dict,
        control_suffixs=population,
        score_list=score_list,
        num_elites=num_elites_used,
        batch_size=batch_size,
        crossover=crossover_used,
        mutation=mutation_used,
        API_key=None,
        reference=reference,
        topk=topk_used,
    )


# 替换 AttackAgent 的种群更新方法。main() 在实例化 AttackAgent 之后才会调用 run_update,
# 因此 monkey-patch 类方法即可影响后续所有样本的所有代。
MAF.AttackAgent.run_update = run_update_word_level_only


if __name__ == "__main__":
    print("[Ablation] AttackAgent restricted to Word-Level Recombination only "
          "(adaptive_word_level_recombination).")
    MAF.main()
