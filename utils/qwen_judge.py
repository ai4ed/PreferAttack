"""
Qwen3-8B Judge实现
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional, List
import json
import re

from typing import Tuple
try:
    from utils.data_types import PairwiseExample, JudgeResponse
    from .judge import BaseJudge
except ImportError:
    from utils.data_types import PairwiseExample, JudgeResponse
from evaluation.judge import BaseJudge


def judge_optimized_impl(
    judger,
    examples: List["PairwiseExample"],
    modified_instructions: List[str] = None,
    batch_size: int = 8,
    max_new_tokens: int = 10,
    temperature: float = 0.1,
    do_sample: bool = False,
    truncation: bool = True,
):
    """优化版批量评估：长度排序 + inference_mode，减少 padding 浪费"""
    if modified_instructions is not None and len(modified_instructions) != len(examples):
        raise ValueError("modified_instructions 长度必须与 examples 一致，或为 None")

    # 构建 prompts
    prompts: List[str] = []
    for i, ex in enumerate(examples):
        instr = ex.instruction if modified_instructions is None else modified_instructions[i]
        tmp_ex = PairwiseExample(
            question_id=ex.question_id,
            instruction=instr,
            response_a=ex.response_a,
            response_b=ex.response_b,
            model_a=ex.model_a,
            model_b=ex.model_b,
        )
        prompts.append(judger.prompt_builder(tmp_ex))
    # pad_id 是用于填充的 token ID，确保输入序列长度一致，避免模型处理时出错
    pad_id = judger.tokenizer.pad_token_id
    if pad_id is None and judger.tokenizer.eos_token_id is not None:
        pad_id = judger.tokenizer.eos_token_id

    judger.model.eval()
    n = len(prompts)
    lengths = [len(p) for p in prompts]
    order = sorted(range(n), key=lambda i: lengths[i], reverse=True)

    decoded_all: List[Optional[str]] = [None] * n

    import torch as _torch
    with _torch.inference_mode():
        for start in range(0, n, batch_size):
            ids = order[start:start + batch_size]
            batch_prompts = [prompts[i] for i in ids]

            inputs = judger.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
            ).to(judger.device)

            gen = judger.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=pad_id,
                use_cache=True,
            )

            decoded = judger.tokenizer.batch_decode(gen, skip_special_tokens=True)
            for j, orig_idx in enumerate(ids):
                decoded_all[orig_idx] = decoded[j]

    results: List["JudgeResponse"] = []
    for i, resp in enumerate(decoded_all):
        pref, conf = judger.response_parser(resp, prompts[i])
        results.append(JudgeResponse(preference=pref, confidence=conf, raw_response=resp))
    return results


class BatchPairwiseJudger:
    """
    单卡批量 Pairwise 评估器：
    - 依赖外部提供的 tokenizer / model / device
    - 通过 prompt_builder 构造每个样本的 prompt
    - 通过 response_parser 解析每条生成结果
    """

    def __init__(self, tokenizer, model, device, prompt_builder, response_parser):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser

    # 为单条样本构造 prompt，用于指示大语言模型如何做“偏好判断”。
    def build_prompts(self, examples: List["PairwiseExample"], modified_instructions: List[str] = None) -> List[str]:
        prompts: List[str] = []
        for i, ex in enumerate(examples):
            instr = ex.instruction if modified_instructions is None else modified_instructions[i]
            tmp_ex = PairwiseExample(
                question_id=ex.question_id,
                instruction=instr,
                response_a=ex.response_a,
                response_b=ex.response_b,
                model_a=ex.model_a,
                model_b=ex.model_b,
            )
            prompts.append(self.prompt_builder(tmp_ex))
        return prompts
    # 批量与逐个样本都在这里进行模型评估，逐个样本的评估不在这里进行，这里是专门为批量评估设计的。
    def judge(
        self,
        examples: List["PairwiseExample"],
        modified_instructions: List[str] = None,
        batch_size: int = 8,
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        do_sample: bool = True,
        truncation: bool = True,
    ) -> List["JudgeResponse"]:
        """批量评估 examples，返回 List[JudgeResponse]"""
        if modified_instructions is not None and len(modified_instructions) != len(examples):
            raise ValueError("modified_instructions 长度必须与 examples 一致，或为 None")

        prompts = self.build_prompts(examples, modified_instructions)

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None and self.tokenizer.eos_token_id is not None:
            pad_id = self.tokenizer.eos_token_id

        self.model.eval()
        decoded_all: List[str] = []
        n = len(prompts)
        # 这里的examples体现在prompts的数组上，prompts是通过example构建出来的。
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_prompts = prompts[start:end]

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=truncation,
            ).to(self.device)

            with torch.no_grad():
                # 批量评估所生成的回答，然后解码输出
                gen = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=pad_id,
                )
            # 批量评估的解码输出
            decoded = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            decoded_all.extend(decoded)

        results: List[JudgeResponse] = []
        for i, resp in enumerate(decoded_all):
            pref, conf = self.response_parser(resp, prompts[i])
            results.append(JudgeResponse(preference=pref, confidence=conf, raw_response=resp))
        return results

# 当前类包含judge_examples与judge_pairwise
class QwenJudge(BaseJudge):
    """Qwen3-8B作为Judge的实现"""
    
    def __init__(self, model_path: str = "/root/autodl-tmp/Qwen3-8B", device: str = "cuda"):
        # 创建一个虚拟的JudgeConfig来满足BaseJudge的要求
        from .judge import JudgeConfig
        config = JudgeConfig(api_key="dummy")  # QwenJudge不需要API key
        super().__init__(config)
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()
        self._batch_judger = None
    
    # ...existing code...
    def _load_model(self):
        """加载Qwen3-8B模型（自动检测CUDA并在不可用时回退到CPU）"""
        print(f"Loading Qwen3-8B model from {self.model_path}...")

        import os

        # 判断是本地目录还是 hub id
        local_files_only = os.path.isdir(self.model_path)

        # 决定使用的设备：尊重 self.device，但在 self.device=="cuda" 且无 CUDA 时回退到 cpu
        requested_device = getattr(self, "device", None) or "cuda"
        # print(f"Requested device: {requested_device}")
        # print(torch.cuda.is_available())
        has_cuda = torch.cuda.is_available()
        if requested_device == "cuda" and not has_cuda:
            print("⚠️ 请求使用 CUDA，但未检测到可用 GPU，已回退到 CPU")
            target_device = "cpu"
        else:
            target_device = requested_device if requested_device in ("cpu", "cuda") else ("cuda" if has_cuda else "cpu")

        # 更新实例 device 字段，后续 inputs.to(self.device) 等能正常工作
        self.device = target_device
        # print(self.device)
        try:
            # 先加载 tokenizer（本地或 hub）
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=local_files_only,
            )
            # 强制使用 left padding，确保 decoder-only 模型生成行为正确
            try:
               self.tokenizer.padding_side = "left"
            except Exception:
                pass

            # 确保有 pad_token / pad_token_id（部分自定义 tokenizer 可能缺失）
            try:
                if getattr(self.tokenizer, "pad_token_id", None) is None:
                    if getattr(self.tokenizer, "eos_token", None) is not None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    else:
                        # 若没有 eos_token，也尝试添加一个 pad token
                        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            except Exception:
                pass
            
            # 根据目标设备选择加载参数
            if self.device == "cuda":
                # 先尝试 device_map="auto"，失败时降级到显式 GPU 0，再失败回退到 CPU
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        local_files_only=local_files_only,
                    )
                except Exception as e_auto:
                    print(f"⚠️ device_map='auto' 失败: {e_auto}. 尝试显式映射到 GPU 0...")
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            dtype=torch.float16,
                            device_map={"": 0},
                            trust_remote_code=True,
                            local_files_only=local_files_only,
                        ).to(self.device)
                    except Exception as e_gpu0:
                        print(f"⚠️ 显式映射到 GPU 0 失败: {e_gpu0}. 回退到 CPU 加载模型...")
                        # 回退到 CPU
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            dtype=torch.float32,
                            device_map={"": "cpu"},
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            local_files_only=local_files_only,
                        )
            else:
                # CPU 回退：使用 float32，明确把模型放到 CPU，尝试减内存占用的选项
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=local_files_only,
                )

            print("✅ Qwen3-8B model loaded successfully on", self.device)

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
# ...existing code...
    # 作用是构造并返回送给 Qwen 模型的评估提示（prompt），用于让模型在两个回答之间做出优劣判断。
#     def get_judge_prompt(self, example: PairwiseExample) -> str:
#         """获取Qwen3-8B的judge prompt"""
#         # todo 修改prompt尝试只有两个选项
#         return f"""Please evaluate the quality of two AI assistant responses and choose the better one.

# Question: {example.instruction}

# Response A: {example.response_a}

# Response B: {example.response_b}

# Please choose from the following options:
# A. Response A is better
# B. Response B is better
# C. Both responses are of similar quality

# Please respond with only one letter: A, B, or C."""
    def get_judge_prompt(self, example: PairwiseExample) -> str:
            """获取Qwen3-8B的judge prompt，来自ICLR25 paper judgebench的vanilla prompt template，10月21号后使用"""
            return f"""You are a helpful assistant in evaluating the quality of the outputs for a given instruction. Your goal is to select the best output for the given instruction.
    Select the Output (a) or Output (b) that is better for the given instruction. The two outputs are generated by two different AI chatbots respectively.
    Do NOT provide any explanation for your choice.
    Do NOT say both / neither are good.
    You should answer using ONLY “Output (a)” or “Output (b)”. Do NOT output any other words.
    # Instruction:
    {example.instruction} 
    # Output (a):
    {example.response_a} 
    # Output (b):
    {example.response_b}
    Which is better, Output (a) or Output (b)? Your response should be either “Output (a)” or “Output (b)”:"""
    
    def judge_pairwise(self, example: PairwiseExample, modified_instruction: str = None) -> JudgeResponse:
        """使用Qwen3-8B进行pairwise判断"""
        try:
            # 如果提供了修改后的instruction，使用它；否则使用原始的
            instruction = modified_instruction if modified_instruction else example.instruction
            
            # 创建临时的example用于judge
            temp_example = PairwiseExample(
                question_id=example.question_id,
                instruction=instruction,
                response_a=example.response_a,
                response_b=example.response_b,
                model_a=example.model_a,
                model_b=example.model_b
            )
            
            prompt = self.get_judge_prompt(temp_example)
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            # print(self.device)
            # 生成回答，攻击成功是让大模型改变选择偏好
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取判断结果
            preference, confidence = self._parse_response(response, prompt)
            
            return JudgeResponse(
                preference=preference,
                confidence=confidence,
                raw_response=response
            )
            
        except Exception as e:
            print(f"Error in QwenJudge: {e}")
            # 返回随机结果作为fallback
            import random
            return JudgeResponse(
                preference=random.randint(0, 1),
                confidence=0.5,
                raw_response=f"Error: {str(e)}"
            )
    
    # def _parse_response(self, response: str, prompt: str) -> Tuple[int, float]:
    #     """解析Qwen3-8B的响应"""
    #     # 移除prompt部分，只保留生成的内容
    #     generated_text = response[len(prompt):].strip()
        
    #     # 查找A、B、C选项
    #     if "A" in generated_text and "B" not in generated_text and "C" not in generated_text:
    #         return 0, 0.8  # 偏好A
    #     elif "B" in generated_text and "A" not in generated_text and "C" not in generated_text:
    #         return 1, 0.8  # 偏好B
    #     elif "C" in generated_text:
    #         # 如果选择C，随机选择一个
    #         import random
    #         return random.randint(0, 1), 0.5
    #     else:
    #         # 如果无法解析，尝试其他模式
    #         if "Response A" in generated_text or "A is better" in generated_text:
    #             return 0, 0.7
    #         elif "Response B" in generated_text or "B is better" in generated_text:
    #             return 1, 0.7
    #         else:
    #             # 默认随机选择
    #             import random
    #             return random.randint(0, 1), 0.3
# 
    def _parse_response(self, response: str, prompt: str) -> Tuple[int, float]:
        """解析Qwen3-8B响应：针对新的prompt模板优化，匹配"Output (a)"或"Output (b)"格式"""
        # 1. 提取纯生成内容（清除prompt和首尾空白，统一小写）
        gen_text = response.replace(prompt, "").strip().lower()
        
        # 2. 定义匹配规则：(A匹配模式列表, B匹配模式列表, 对应置信度)，按优先级排序
        match_rules: List[Tuple[List[str], List[str], float]] = [
            # 优先级1：精准匹配"Output (a)"或"Output (b)"格式，最高置信度
            (["^output\\s*\\(a\\)\\s*$", "^output\\s*\\(a\\)", "output\\s*\\(a\\)\\s*$"], 
             ["^output\\s*\\(b\\)\\s*$", "^output\\s*\\(b\\)", "output\\s*\\(b\\)\\s*$"], 0.95),
            # 优先级2：匹配带引号的格式"Output (a)"或"Output (b)"
            (["\"output\\s*\\(a\\)\"", "'output\\s*\\(a\\)'"], 
             ["\"output\\s*\\(b\\)\"", "'output\\s*\\(b\\)'"], 0.9),
            # 优先级3：匹配简化的(a)或(b)格式
            (["^\\(a\\)\\s*$", "^\\(a\\)", "\\(a\\)\\s*$"], 
             ["^\\(b\\)\\s*$", "^\\(b\\)", "\\(b\\)\\s*$"], 0.85),
            # 优先级4：匹配单独的a或b（作为兜底）
            (["^a\\s*$", "^a\\s", "\\sa\\s*$"], 
             ["^b\\s*$", "^b\\s", "\\sb\\s*$"], 0.8),
            # 优先级5：匹配旧的Response A/B格式（向后兼容）
            (["response a", "a is better", "prefer a", "choose a"], 
             ["response b", "b is better", "prefer b", "choose b"], 0.7)
        ]
        
        # 3. 按优先级遍历规则，匹配则返回结果
        for a_patterns, b_patterns, conf in match_rules:
            # 检查A是否匹配（且B不匹配，避免歧义）
            a_match = any(re.search(pat, gen_text) for pat in a_patterns)
            b_match = any(re.search(pat, gen_text) for pat in b_patterns)
            if a_match and not b_match:
                return 0, conf
            elif b_match and not a_match:
                return 1, conf
        
        # 4. 兜底：无法解析时返回随机结果（低置信度）
        import random
        return random.randint(0, 1), 0.3
    
    def get_batch_judger(self) -> "BatchPairwiseJudger":
        """惰性创建批量评估器，复用当前 QwenJudge 的资源"""
        # 不是在 QwenJudge 初始化时就立刻创建 BatchPairwiseJudger，而是第一次需要用到时（第一次调用 get_batch_judger()）才创建。
        if self._batch_judger is None:
            # 与BatchPairwiseJudger建立联系，后续通过self.get_batch_judger()进行judge
            self._batch_judger = BatchPairwiseJudger(
                tokenizer=self.tokenizer,
                model=self.model,
                device=self.device,
                prompt_builder=self.get_judge_prompt,
                response_parser=self._parse_response,
            )
        return self._batch_judger

    def judge_examples(
        self,
        examples: List["PairwiseExample"],
        modified_instructions: List[str] = None,
        batch_size: int = 8,
        max_new_tokens: int = 10,
        temperature: float = 0.1,
        do_sample: bool = True,
        truncation: bool = True,
    ) -> List["JudgeResponse"]:
        """单卡批量评估入口：返回 List[JudgeResponse]"""
        # 使用优化后的批量评估实现
        return judge_optimized_impl(
            self.get_batch_judger(),
            examples,
            modified_instructions=modified_instructions,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            truncation=truncation,
        )


# def create_qwen_judge(model_path: str = "/root/autodl-tmp/Qwen3-8B") -> QwenJudge:
#     """创建Qwen3-8B Judge实例"""
#     return QwenJudge(model_path)
# ...existing code...
def create_qwen_judge(model_path: str = "/root/autodl-tmp/Qwen3-8B", device: str = "cuda") -> QwenJudge:
    """
    创建并返回 QwenJudge 实例。
    - model_path: 本地模型目录或 HF repo id（默认为 /root/autodl-tmp/Qwen3-8B）。
    - device: "cuda" 或 "cpu"（会传递给 QwenJudge）。
    处理要点：
    - 若本地目录中包含自定义 tokenizer（如 Qwen2Tokenizer），尝试把具体包含该实现的子目录加入 sys.path，
      以便 transformers 在 trust_remote_code=True 时能找到自定义类。
    """
    import os
    import sys

    if model_path is None:
        model_path = os.environ.get("JUDGE_MODEL_PATH", model_path)

    # 如果是本地目录，尝试定位定义 Qwen2Tokenizer 的子目录并加入 sys.path（避免加入整个模型根目录）
    try:
        if os.path.isdir(model_path):
            tokenizer_class_name = "Qwen2Tokenizer"
            tokenizer_dir_to_add = None
            for root, _, files in os.walk(model_path):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                            if tokenizer_class_name in fh.read():
                                tokenizer_dir_to_add = root
                                break
                    except Exception:
                        continue
                if tokenizer_dir_to_add:
                    break
            if tokenizer_dir_to_add and tokenizer_dir_to_add not in sys.path:
                sys.path.insert(0, tokenizer_dir_to_add)
    except Exception:
        # 任何路径扫描错误都不应阻止后续加载，继续让 QwenJudge 自行尝试加载
        pass

    try:
        judge = QwenJudge(model_path=model_path, device=device)
        return judge
    except Exception as e:
        raise RuntimeError(f"无法初始化 QwenJudge (path={model_path}, device={device}): {e}") from e
# ...existing code...
# def create_qwen_judge(model_path: str = None, device: str = "cuda") -> QwenJudge:
#     """
#     创建并返回 QwenJudge 实例。
#     - model_path: 本地模型目录或 HuggingFace repo id；若为 None，则尝试从环境变量 JUDGE_MODEL_PATH 读取。
#     - 对本地目录会把目录加入 sys.path，确保自定义 tokenizer/class 可被 transformers 导入。
#     """
#     import os, sys

#     if model_path is None:
#         model_path = os.environ.get("JUDGE_MODEL_PATH", None)

#     if model_path is None:
#         raise RuntimeError("create_qwen_judge 需要 model_path（或设置环境变量 JUDGE_MODEL_PATH）")

#     # 如果是本地目录，确保其在 sys.path 中，便于 transformers 导入自定义类（如 Qwen2Tokenizer）
#     if os.path.exists(model_path) and os.path.isdir(model_path):
#         if model_path not in sys.path:
#             sys.path.insert(0, model_path)

#     try:
#         judge = QwenJudge(model_path=model_path, device=device)
#         return judge
#     except Exception as e:
#         raise RuntimeError(f"无法初始化 QwenJudge (path={model_path}): {e}") from e

# # ...existing code...
