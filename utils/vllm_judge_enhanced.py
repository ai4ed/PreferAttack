"""
增强版 VLLMJudge (v2): 不动 utils/vllm_judge.py, 在此文件独立扩展。

两处核心改造 (针对 MiniCPM5-1B / 小模型作为 judge 时的 ~31% 不规则输出):

1) _parse_response 分层规则扩展:
   - 阶段 0 (0.95): 完整独立的 "Output (a)" / "Output (b)" (与原版兼容)
   - 阶段 1 (0.92): 强信号短语, "output (a) is better" / "i prefer output (b)" /
                     "output (a) because ..." — 模型明确选边 + 解释
   - 阶段 2 (0.88): 标题式 "## output (a):" / "output (b):" — 模型当作续写答案标题
                     (首个标题通常是模型选择)
   - 阶段 3 (0.85): 多次出现的 "Output (X)", 用"最后一次非列举位置"作为答案
                     (排除 "output (a) or output (b)" 这种成对列举)
   - 阶段 4 (0.82): 被截断的 "output (a" / "output (b" (max_tokens 切右括号)
   - 阶段 5 (0.80/0.85): 单字符 "(a)" / "a" / "Response A" 等向后兼容
   - 阶段 6: 全不命中 → (preference=None, confidence=0.0) 标记为无效

2) judge_pairwise / judge_examples 内置 retry:
   - 检测到 preference=None 时, 用 max_tokens=64 (默认 16 的 4 倍) 再试一次
   - retry 后仍无效 → 返回 preference=None, confidence=0.0
   - 上层 (Multi_Agent_Framework) 现有逻辑 (line 145-147) 会自然给 score=0.5 中性分
     baseline 兜底则由 v2 入口脚本保护 (跳过该样本)
"""

import os
import re
from typing import List, Optional, Tuple

from vllm import SamplingParams

from utils.data_types import PairwiseExample, JudgeResponse
from utils.vllm_judge import VLLMJudge, create_vllm_judge, get_judge_prompt


# ====== 阶段化规则定义 ======
# 注意: 所有规则都在 lower-case 文本上匹配, 因此模式里都用小写

# 阶段 0: 标准格式 "Output (a)" / "Output (b)" (优先级最高, 与原版一致)
STAGE0_A = [r"^output\s*\(a\)\s*$", r"^output\s*\(a\)", r"output\s*\(a\)\s*$"]
STAGE0_B = [r"^output\s*\(b\)\s*$", r"^output\s*\(b\)", r"output\s*\(b\)\s*$"]

# 阶段 1: 强信号短语 (明确选边 + 解释), 例如:
#   "Output (a) is better because ..."
#   "I prefer Output (b) ..."
#   "Output (a) provides a more detailed ..."
#   "The better option is Output (a)"
STRONG_A = [
    r"output\s*\(a\)\s+is\s+better",
    r"output\s*\(a\)\s+(?:provides|follows|is|gives|has|presents|seems|appears|looks|feels|sounds)",
    r"output\s*\(a\)\s+because",
    r"prefer\s+output\s*\(a\)",
    r"choose\s+output\s*\(a\)",
    r"select\s+output\s*\(a\)",
    r"(?:better|best)\s+(?:option|choice|response|answer|one)?\s*(?:is)?\s*output\s*\(a\)",
    r"output\s*\(a\)\s+(?:more|better|clearly|more\s+\w+)",
]
STRONG_B = [p.replace(r"\(a\)", r"\(b\)") for p in STRONG_A]

# 阶段 2: 标题式 "## output (a):" / "output (a):" — 单独行首
# 通过 (?:^|\n) 锚定行首; : 后可以是换行或空格
TITLE_A = r"(?:^|\n)\s*##?\s*output\s*\(a\)\s*:"
TITLE_B = r"(?:^|\n)\s*##?\s*output\s*\(b\)\s*:"

# 阶段 3 / 4 用 finditer 在主函数里处理

# 阶段 5: 向后兼容 (单字符、Response A 旧格式)
LEGACY_RULES: List[Tuple[List[str], List[str], float]] = [
    ([r"^\(a\)\s*$", r"^\(a\)", r"\(a\)\s*$"],
     [r"^\(b\)\s*$", r"^\(b\)", r"\(b\)\s*$"], 0.85),
    ([r"^a\s*$", r"^a\s", r"\sa\s*$"],
     [r"^b\s*$", r"^b\s", r"\sb\s*$"], 0.80),
    ([r"response a", "a is better", "prefer a", "choose a"],
     [r"response b", "b is better", "prefer b", "choose b"], 0.70),
]

# 列举判定: "output (a)" or "output (b)" 之间的字符距离
# 经验值: prompt 模板里两个选项通常被 ' or ' 或引号分隔, 距离 15-30 字符;
# 设为 30 可以覆盖 'output (a)" or "output (b)' 这种带引号 + or 的列举。
LISTING_PROXIMITY = 30


def _classify(gen_text: str) -> Tuple[Optional[int], float]:
    """对 lower-case 生成文本执行分层匹配, 返回 (preference, confidence)。

    preference:
      - 0 → 偏好 A
      - 1 → 偏好 B
      - None → 所有规则未命中 (上层走中性 0.5 分支)
    """
    # ----- 阶段 0: 完美 "Output (a)/(b)" 标准 -----
    a0 = any(re.search(p, gen_text) for p in STAGE0_A)
    b0 = any(re.search(p, gen_text) for p in STAGE0_B)
    if a0 and not b0:
        return 0, 0.95
    if b0 and not a0:
        return 1, 0.95

    # ----- 阶段 1: 强信号短语 -----
    a1 = any(re.search(p, gen_text) for p in STRONG_A)
    b1 = any(re.search(p, gen_text) for p in STRONG_B)
    if a1 and not b1:
        return 0, 0.92
    if b1 and not a1:
        return 1, 0.92

    # ----- 阶段 2: 标题式 "## output (a):" -----
    a2 = re.search(TITLE_A, gen_text) is not None
    b2 = re.search(TITLE_B, gen_text) is not None
    if a2 and not b2:
        return 0, 0.88
    if b2 and not a2:
        return 1, 0.88

    # ----- 阶段 3: 多次完整出现, 取"最后一个非列举" -----
    all_a = [m.start() for m in re.finditer(r"output\s*\(a\)", gen_text)]
    all_b = [m.start() for m in re.finditer(r"output\s*\(b\)", gen_text)]

    # 标记成对列举的位置 (互相距离 < LISTING_PROXIMITY)
    paired_a, paired_b = set(), set()
    for pa in all_a:
        for pb in all_b:
            if abs(pa - pb) < LISTING_PROXIMITY:
                paired_a.add(pa)
                paired_b.add(pb)
    clean_a = [p for p in all_a if p not in paired_a]
    clean_b = [p for p in all_b if p not in paired_b]

    if clean_a and not clean_b:
        return 0, 0.85
    if clean_b and not clean_a:
        return 1, 0.85
    if clean_a and clean_b:
        # 双方都有非列举出现: 弱信号取最后一个, 但降低置信度避免污染优化
        # (典型场景: 模型列了 "Output (a) pros ... Output (b) cons ..." 没明确结论)
        if max(clean_a) > max(clean_b):
            return 0, 0.75
        return 1, 0.75

    # ----- 阶段 4: 被截断的 "output (a" / "output (b" -----
    # (右括号被 max_tokens 切掉)
    # negative lookahead (?!\s*\)) 排除 "output (a)" 这种完整形式 (阶段 3 已处理)
    trunc_a = re.search(r"\boutput\s*\(a(?!\s*\))", gen_text) is not None
    trunc_b = re.search(r"\boutput\s*\(b(?!\s*\))", gen_text) is not None
    if trunc_a and not trunc_b:
        return 0, 0.82
    if trunc_b and not trunc_a:
        return 1, 0.82

    # ----- 阶段 5: 单字符 / Response A 类 -----
    for a_pats, b_pats, conf in LEGACY_RULES:
        a_m = any(re.search(p, gen_text) for p in a_pats)
        b_m = any(re.search(p, gen_text) for p in b_pats)
        if a_m and not b_m:
            return 0, conf
        if b_m and not a_m:
            return 1, conf

    # ----- 阶段 6: 全不命中 -----
    return None, 0.0


class EnhancedVLLMJudge(VLLMJudge):
    """继承 VLLMJudge, 仅替换 _parse_response, 并在 judge_pairwise / judge_examples
    里加入 retry 逻辑。

    通过 create_enhanced_vllm_judge() 工厂函数构造, 不要直接 new。
    """

    def __init__(self, *args, retry_max_tokens: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_max_tokens = int(retry_max_tokens)

    def _parse_response(self, response: str, prompt: str) -> Tuple[Optional[int], float]:
        gen_text = response.replace(prompt, "").strip().lower()
        return _classify(gen_text)

    # ---------- 单样本推理 + retry ----------
    def judge_pairwise(self, example: PairwiseExample,
                       modified_instruction: Optional[str] = None) -> JudgeResponse:
        instr = modified_instruction if modified_instruction else example.instruction
        tmp = PairwiseExample(
            question_id=example.question_id,
            instruction=instr,
            response_a=example.response_a,
            response_b=example.response_b,
            model_a=example.model_a,
            model_b=example.model_b,
        )
        prompt = self.get_judge_prompt(tmp)

        # 第一次推理: 沿用主 sampling params (max_tokens=16)
        text, usage = self._generate_one(prompt, self.sparams)
        pref, conf = self._parse_response(text, prompt)

        # retry 1 次: 仅当 pref is None (无法解析)
        if pref is None and self.retry_max_tokens > self.sparams.max_tokens:
            retry_sparams = SamplingParams(
                max_tokens=self.retry_max_tokens,
                temperature=self.sparams.temperature,
                top_p=1.0,
            )
            text2, usage2 = self._generate_one(prompt, retry_sparams)
            pref2, conf2 = self._parse_response(text2, prompt)
            # 合并 usage (两次都计入)
            usage = self._merge_usage(usage, usage2)
            if pref2 is not None:
                return JudgeResponse(
                    preference=pref2, confidence=conf2,
                    raw_response=text2, usage=usage,
                )
            # retry 后仍无效 → 保留 retry 的 raw_response 用于 debug
            return JudgeResponse(
                preference=None, confidence=0.0,
                raw_response=text2, usage=usage,
            )

        return JudgeResponse(preference=pref, confidence=conf, raw_response=text, usage=usage)

    # ---------- 批量推理 + retry ----------
    def judge_examples(
        self,
        examples: List[PairwiseExample],
        modified_instructions: Optional[List[str]] = None,
        batch_size: int = 8,
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        do_sample: bool = False,
        truncation: bool = True,
    ) -> List[JudgeResponse]:
        prompts: List[str] = []
        for i, ex in enumerate(examples):
            instr = ex.instruction if modified_instructions is None else modified_instructions[i]
            tmp = PairwiseExample(
                question_id=ex.question_id,
                instruction=instr,
                response_a=ex.response_a,
                response_b=ex.response_b,
                model_a=ex.model_a,
                model_b=ex.model_b,
            )
            prompts.append(self.get_judge_prompt(tmp))

        sparams = SamplingParams(
            max_tokens=min(max_new_tokens or self.sparams.max_tokens, 64),
            temperature=temperature if temperature is not None else self.sparams.temperature,
            top_p=1.0,
        )

        # 第一次批量推理
        out: List[JudgeResponse] = []
        results = self.llm.generate(prompts, sparams)
        usages = []
        texts = []
        for r in results:
            t = r.outputs[0].text if r and r.outputs else ""
            from utils.vllm_judge import _extract_usage_from_request_output as _eu
            u = _eu(r) if r else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            texts.append(t)
            usages.append(u)

        # 找出 pref is None 的位置, 集中 retry
        retry_idxs = []
        for i, t in enumerate(texts):
            pref, _ = self._parse_response(t, prompts[i])
            if pref is None:
                retry_idxs.append(i)

        if retry_idxs and self.retry_max_tokens > sparams.max_tokens:
            retry_sparams = SamplingParams(
                max_tokens=self.retry_max_tokens,
                temperature=sparams.temperature,
                top_p=1.0,
            )
            retry_prompts = [prompts[i] for i in retry_idxs]
            retry_results = self.llm.generate(retry_prompts, retry_sparams)
            from utils.vllm_judge import _extract_usage_from_request_output as _eu
            for idx, r in zip(retry_idxs, retry_results):
                t2 = r.outputs[0].text if r and r.outputs else ""
                u2 = _eu(r) if r else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                texts[idx] = t2  # 用 retry 后的文本覆盖 (用于 debug)
                usages[idx] = self._merge_usage(usages[idx], u2)

        # 最终解析 + 组装响应
        for i, t in enumerate(texts):
            pref, conf = self._parse_response(t, prompts[i])
            out.append(JudgeResponse(
                preference=pref, confidence=conf,
                raw_response=t, usage=usages[i],
            ))
        return out

    # ---------- helpers ----------
    def _generate_one(self, prompt: str, sparams) -> Tuple[str, dict]:
        outputs = self.llm.generate([prompt], sparams)
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        from utils.vllm_judge import _extract_usage_from_request_output as _eu
        usage = _eu(outputs[0]) if outputs else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return text, usage

    @staticmethod
    def _merge_usage(u1: dict, u2: dict) -> dict:
        return {
            "prompt_tokens": int(u1.get("prompt_tokens", 0)) + int(u2.get("prompt_tokens", 0)),
            "completion_tokens": int(u1.get("completion_tokens", 0)) + int(u2.get("completion_tokens", 0)),
            "total_tokens": int(u1.get("total_tokens", 0)) + int(u2.get("total_tokens", 0)),
        }


def create_enhanced_vllm_judge(
    model_path: Optional[str] = None,
    tensor_parallel_size: int = 1,
    dtype: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 16,
    seed: Optional[int] = None,
    retry_max_tokens: int = 64,
) -> EnhancedVLLMJudge:
    """工厂函数: 与 create_vllm_judge 同签名, 多一个 retry_max_tokens。

    注意: 为了不重复 vllm 初始化逻辑, 这里先调用原工厂函数构造一个普通 VLLMJudge
    拿到 self.llm, 然后用 __new__ + 手动复制的方式把它"升级"为 EnhancedVLLMJudge。
    这避免了重复加载模型, 同时不修改原 VLLMJudge 类。
    """
    base = create_vllm_judge(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        dtype=dtype,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
    # 把 base 的属性"迁移"到 enhanced 实例
    enh = EnhancedVLLMJudge.__new__(EnhancedVLLMJudge)
    enh.__dict__.update(base.__dict__)
    enh.retry_max_tokens = int(retry_max_tokens)
    return enh
