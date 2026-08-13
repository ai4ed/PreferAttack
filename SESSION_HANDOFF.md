# PreferAttack 防御实验 — 会话交接

> 用途:下次重开服务器后,把本文件发给 Claude(或让它读 `PreferAttack/SESSION_HANDOFF.md`),即可接着上次的进度继续工作,无需重新摸索。
> 上次会话日期:2026-06-30

---

## 0. 一句话现状

已经为 PreferAttack 论文补做了**两类防御实验**(轻量分类器 + 语义相似度),口径对齐论文 §5.11 的 PPL 防御(Table 9/10),并在 code_judge_bench / Qwen3-VL-8B 上跑完。结论**强支撑 stealthiness 论点**:完整 prompt 视角下,线性(LR)和非线性(MLP)分类器都检测不了,语义漂移防御也失效,而同样的防御能抓 GCG。

---

## 1. 项目背景

- **论文**:`PreferAttack` — A Collaborative Multi-Agent Framework for Preference-Reversal Attacks on LLM-as-a-Judge Models(投稿 ESWA)。
- **PDF**:`/root/PreferAttack/A_Collaborative_Multi_Agent_Framework_for_Preference_Attacks_on_LLM_as_a_Judge_Models_eswa.pdf`(57 页,文本已提取到 `/tmp/preferattack.txt`,服务器重启后需重新 `pdftotext -layout` 提取)。
- **方法核心**:三智能体(Scoring/Decision/Attack)协同优化一段追加在 instruction 后的对抗后缀 `s`,使 judge 的 pairwise 偏好翻转。Stealthiness 来自 Attack Agent 的"语义保持变换"。
- **本次任务动机**:审稿人/作者想用"基于模型的防御(轻量分类器)和基于语义相似度的防御"来加强 stealthiness 论证——论文 §5.11 只有 PPL 防御,缺这两类。本次就是补这个缺口。
- **攻击主框架入口**:`Multi_Agent_Framework.py`(后缀拼接规则在 `build_attacked_instruction`,`instruction.rstrip() + " " + suffix`)。

---

## 2. 已完成的实验

口径与论文 Table 9/10 完全一致:`ASR` / `ASR-W`(防御后仍成功的)/ `ASR-Reduction` / `FNR@1%FPR` / `FPR`(阈值在干净数据上取 99 分位保证 ≤1%)。

### 数据(均已就位,无需重新生成)
| 来源 | 数量 | 路径 |
|---|---|---|
| PreferAttack 攻击结果(被测对象) | 423 条(414 成功,ASR=97.9%) | `results/multi_agent_pairwise_eval_origin_opti_code_judge_bench_20260628_203045.json`(target=Qwen3-VL-8B) |
| PreferAttack alpaca(太小,仅 2 条) | 2 | `results/multi_agent_pairwise_eval_origin_opti_alpaca_eval_20260427_134843.json` |
| GCG 后缀(迁移检测器训练) | 150 | `/root/autodan/results/gcg_baseline_arena_hard_20251024_150918.json`(只有 `question_id`,靠 arena_hard 数据关联指令) |
| AutoDAN 后缀(迁移检测器训练) | 847 | `/root/AutoDAN-main/results/pairwise_ga_*.json`(排除 `tts`/`checkpoint`) |
| 干净指令(负类/FPR 标定) | 325 | 从上述文件的 `instruction` 字段去重派生 |
| arena_hard 指令(GCG 关联用) | — | `data/split/arena_hard_test.json` / `arena_hard_train.json` |

### 实验 A — 轻量分类器(MiniLM 嵌入 + LR/MLP)
两种威胁模型:**A1 同分布 5-fold CV**(最强防御者,见过 PreferAttack)、**A2 迁移**(在 GCG+AutoDAN 上训练,在 PreferAttack 上测,真实部署)。两种输入视角:**suffix**(oracle,假设能切出后缀)、**attacked_instr**(完整 prompt,judge 实际看到的)。

| 输入视角 | 分类器 | A1 CV AUC | A1 FNR@1%FPR | A2 迁移 FNR | A2 ASR-Reduction |
|---|---|---|---|---|---|
| suffix(oracle) | LR | 1.000 | 0.000 | 0.000 | 100% |
| suffix(oracle) | MLP | 1.000 | 0.000 | 0.000 | 100% |
| **attacked_instr(真实)** | LR | 0.173 | 0.988 | 0.974 | **2.66%** |
| **attacked_instr(真实)** | MLP | 0.303 | 1.000 | 0.983 | **1.69%** |

### 实验 B — 语义相似度防御
| 变体 | PreferAttack FNR@1%FPR | AUC(vs 良性追加) | GCG 对照 FNR | ASR-Reduction |
|---|---|---|---|---|
| **prompt_drift** `cos(e(instr+suffix), e(instr))` — 可部署 | 0.941 | 0.497(≈随机) | 0.320(抓 102/150) | **5.80%** |
| suffix_vs_instr `cos(e(suffix), e(instr))` — 需 oracle 切分 | 0.515 | 0.275 | 0.733 | 49.0% |

### 关键结论(可直接写进论文/rebuttal)
1. **真实分类器防御几乎无效**:完整 prompt 视角下 LR AUC=0.17、MLP AUC=0.30(都接近随机),迁移检测器只抓 11/423(LR)或 7/423(MLP),ASR 仅降 1.7–2.7%。
2. **换成非线性 MLP 也没用**:排除了"线性容量不足"的解释,强化 stealthiness 论证。
3. **prompt_drift 相似度防御也失效**(AUC=0.497),而同防御能抓 68% GCG —— PreferAttack 的语义保持变换不漂移整句嵌入,GCG 乱码会。
4. **诚实边界**:若防御者能精确切出后缀(oracle),分类器 AUC=1.0、suffix_vs_instr 能降 49% ASR。但这要求事先知道 instruction/suffix 边界,在 PreferAttack 威胁模型下不现实。建议在 rebuttal 主动说明:隐蔽性是**整句级别**的,非**后缀片段级别**。

---

## 3. 代码文件清单

### 本次会话新增(都是独立、可重跑的)
| 文件 | 作用 |
|---|---|
| `src/defense/model_defense.py` | LR 分类器 `ClassifierDefense` + 相似度 `SimilarityDefense` + 数据加载/指标工具(`load_records`/`load_gcg_records`/`Embedder`/`asr_under_defense`/`fnr_fpr`)。**这是基础工具模块。** |
| `src/defense/mlp_defense.py` | `MLPClassifierDefense`(2 层 MLP,128-64,ReLU,alpha=1e-3,early stopping)。独立于 LR,只 import `Embedder` 等公共工具。 |
| `run_stealth_defenses.py` | **LR + 相似度** runner(实验 A 的 LR + 实验 B 全部)。 |
| `run_mlp_defense.py` | **MLP** runner(实验 A 的 MLP),跑完自动读 `stealth_defenses_summary.json` 打印 LR vs MLP 对比表。 |
| `results/stealth_defenses_summary.json` | LR + 相似度 结果(机读)。 |
| `results/mlp_defense_summary.json` | MLP 结果(机读)。 |

### 原有、未改动(不要动)
- `src/defense/ppl_defense.py`、`src/defense/__init__.py`
- `Multi_Agent_Framework.py`、`utils/*`、`check_asr_pairwise.py`、`run_pairwise_qwen3.sh`
- 所有 `data/` 和 `results/` 下的攻击结果 JSON

---

## 4. 环境

- **GPU**:1× RTX 4080 SUPER(32GB)。MiniLM 推理用 CUDA。
- **已安装 pip 包**:`sentence-transformers==5.6.0`、`scikit-learn==1.9.0`(本次新装)。
- **已装系统包**:`poppler-utils`(`pdftotext`/`pdfinfo`/`pdftoppm`,本次新装,用于读 PDF)。
- **已下载模型**:`sentence-transformers/all-MiniLM-L6-v2`(384 维,在 HF 缓存,联网时已下载)。
- **本地大模型缓存**:`/root/autodl-tmp/` 下有 Qwen2.5/Qwen3-VL-8B/Gemma/Llama 等(攻击时用,本次防御实验不需要)。

---

## 5. 如何重跑(一键复现)

```bash
cd /root/PreferAttack

# 1) LR + 相似度防御(实验 A 的 LR + 实验 B)
python3 run_stealth_defenses.py

# 2) MLP 防御(实验 A 的 MLP),并在结尾打印 LR vs MLP 对比表
python3 run_mlp_defense.py
```

两个脚本各自独立、互不依赖对方运行(MLP runner 只**读** LR 的 JSON 做对比,不修改它)。

---

## 6. 待办 / 可继续的方向(按优先级)

1. **写成 LaTeX 表格塞进论文**:新增 §5.12 "Learned & Similarity-Based Defenses",与 §5.11 PPL 防御并列。表头用 `ASR / ASR-W / ASR-Reduction / FNR / FPR`。需要的话让 Claude 生成。
2. **加更强的检测器做第三组对比**:把 MiniLM 换成 `RoBERTa-base`/`DeBERTa-v3-base` 做特征提取(或端到端微调一个二分类器),看检测上限。预期仍难检测完整 prompt,但能进一步坐实结论。
3. **在 Arena Hard / Alpaca Eval 上补跑**:目前只有 code_judge_bench。需要先跑出这两个数据集的 PreferAttack 攻击结果(`Multi_Agent_Framework.py` + `run_pairwise_qwen3.sh`),再用本脚本评估。仓库里 alpaca 只有 2 条结果,不够。
4. **rebuttal 里主动写明边界**:suffix-only oracle 能检测(AUC=1.0),说明隐蔽性是整句级而非后缀片段级——主动交代反而显得严谨。
5. **MLP 调参**(可选):当前 MLP AUC=0.30,可尝试更大网络/更长训练,但预计提升有限(瓶颈在特征被长指令稀释,不在分类器容量)。

---

## 7. 给下次 Claude 的提示

- 用户在做 PreferAttack 论文的审稿回复实验,目标是**支撑 stealthiness 论点**。
- 所有防御实验的**口径必须与论文 §5.11 PPL 防御一致**(FPR=1% 标定、ASR/ASR-W/ASR-Reduction),否则不可比。
- 用户偏好:**保留原有脚本不动,新方法新建独立脚本**(本次 MLP 就是按这个要求做的独立文件)。
- 用户用中文交流,回复用中文。
- 防御实验**不需要重跑 judge 模型**——直接用攻击结果 JSON 里已解析的 `baseline.choice`/`attack.new_choice`,防御 flag 后回退到原始偏好即可算 ASR-W(见 `model_defense.py:asr_under_defense`)。
- 完整结果数字见上方第 2 节表格,或读 `results/stealth_defenses_summary.json` / `results/mlp_defense_summary.json`。
