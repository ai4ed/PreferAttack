import gc
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import openai
from tqdm import tqdm
import re
import nltk
# Do NOT attempt blocking downloads at import time. Instead, try to import
# the NLTK corpus objects and prepare safe fallbacks if resources are missing.
from nltk import data as nltk_data
try:
    from nltk.corpus import stopwords, wordnet
    try:
        STOP_WORDS = set(stopwords.words('english'))
    except Exception:
        STOP_WORDS = set()
    WORDNET_AVAILABLE = True
except Exception:
    # NLTK corpora not available in this environment; use conservative fallbacks.
    STOP_WORDS = set()
    wordnet = None
    WORDNET_AVAILABLE = False
from collections import defaultdict, OrderedDict
from utils.string_utils import SuffixManager

# Lazy-loaded local Gemma model (used to replace OpenAI API calls)
GEMMA_MODEL = None
GEMMA_TOKENIZER = None
GEMMA_DEVICE = None
GEMMA_MODEL_PATH = "/share/disk/llm_cache/gemma-3-4b-it/"

# forward函数是用于批量处理输入数据并获取模型输出的函数
def forward(*, model, input_ids, attention_mask, batch_size=512):
    logits = []
    for i in range(0, input_ids.shape[0], batch_size):

        batch_input_ids = input_ids[i:i + batch_size]
        if attention_mask is not None:
            batch_attention_mask = attention_mask[i:i + batch_size]
        else:
            batch_attention_mask = None
        # 这行代码是将当前批次的数据传入模型，获取输出的 logits（未归一化的概率分布）
        logits.append(model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits)

        gc.collect()

    del batch_input_ids, batch_attention_mask

    return torch.cat(logits, dim=0)


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0', **kwargs):
    # 这里需要下载transformer模块，才能找到AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        **kwargs
    ).to(device).eval()

    tokenizer_path = model_path if tokenizer_path is None else tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False
    )

    if 'oasst-sft-6-llama-30b' in tokenizer_path:
        tokenizer.bos_token_id = 1
        tokenizer.unk_token_id = 0
    if 'guanaco' in tokenizer_path:
        tokenizer.eos_token_id = 2
        tokenizer.unk_token_id = 0
    if 'llama-2' in tokenizer_path:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.padding_side = 'left'
    if 'falcon' in tokenizer_path:
        tokenizer.padding_side = 'left'
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# utils/opt_utils.py 末尾或合适位置新增

def load_qwen3_local(model_path: str, device: str = "cuda", dtype: str = "auto"):
    """
    直接从本地路径加载 Qwen3（例如: /root/models/Qwen3-8B-Instruct）。
    不依赖 download_models.py。
    """
    torch_dtype = None
    if dtype == "auto":
        torch_dtype = "auto"
    elif dtype == "bf16":
        import torch
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        import torch
        torch_dtype = torch.float16

    tok = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map=None  # 显式放到单卡
    )
    model = model.to(device).eval()

    # 对部分显卡/环境更稳的注意项（按需保留）：
    try:
        model.generation_config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    return model, tok

### GA ###
def roulette_wheel_selection(data_list, score_list, num_selected, if_softmax=True):
    if if_softmax:
        selection_probs = np.exp(score_list - np.max(score_list))
        selection_probs = selection_probs / selection_probs.sum()
    else:
        total_score = sum(score_list)
        selection_probs = [score / total_score for score in score_list]

    selected_indices = np.random.choice(len(data_list), size=num_selected, p=selection_probs, replace=True)

    selected_data = [data_list[i] for i in selected_indices]
    return selected_data


def apply_crossover_and_mutation(selected_data, crossover_probability=0.5, num_points=3, mutation_rate=0.01,
                                 API_key=None,
                                 reference=None, if_api=True):
    offspring = []

    for i in range(0, len(selected_data), 2):
        parent1 = selected_data[i]
        parent2 = selected_data[i + 1] if (i + 1) < len(selected_data) else selected_data[0]

        if random.random() < crossover_probability:
            child1, child2 = crossover(parent1, parent2, num_points)
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(parent1)
            offspring.append(parent2)

    mutated_offspring = apply_gpt_mutation(offspring, mutation_rate, API_key, reference, if_api)

    return mutated_offspring


def crossover(str1, str2, num_points):
    # Function to split text into paragraphs and then into sentences
    def split_into_paragraphs_and_sentences(text):
        paragraphs = text.split('\n\n')
        # use raw-string for regex to avoid invalid escape sequence warnings
        return [re.split(r'(?<=[,.!?])\s+', paragraph) for paragraph in paragraphs]

    paragraphs1 = split_into_paragraphs_and_sentences(str1)
    paragraphs2 = split_into_paragraphs_and_sentences(str2)

    new_paragraphs1, new_paragraphs2 = [], []

    for para1, para2 in zip(paragraphs1, paragraphs2):
        max_swaps = min(len(para1), len(para2)) - 1
        num_swaps = min(num_points, max_swaps)

        swap_indices = sorted(random.sample(range(1, max_swaps + 1), num_swaps))

        new_para1, new_para2 = [], []
        last_swap = 0
        for swap in swap_indices:
            if random.choice([True, False]):
                new_para1.extend(para1[last_swap:swap])
                new_para2.extend(para2[last_swap:swap])
            else:
                new_para1.extend(para2[last_swap:swap])
                new_para2.extend(para1[last_swap:swap])
            last_swap = swap

        if random.choice([True, False]):
            new_para1.extend(para1[last_swap:])
            new_para2.extend(para2[last_swap:])
        else:
            new_para1.extend(para2[last_swap:])
            new_para2.extend(para1[last_swap:])

        new_paragraphs1.append(' '.join(new_para1))
        new_paragraphs2.append(' '.join(new_para2))

    return '\n\n'.join(new_paragraphs1), '\n\n'.join(new_paragraphs2)

def gpt_mutate(sentence, API_key=None):
    """
    Use a local Gemma model to revise a single sentence. This replaces the previous
    OpenAI API call. The function lazily loads the model/tokenizer from
    `GEMMA_MODEL_PATH` and generates a revision for `sentence`.

    If generation fails for any reason, the original sentence is returned.
    """
    global GEMMA_MODEL, GEMMA_TOKENIZER, GEMMA_DEVICE, GEMMA_MODEL_PATH

    # Prepare a short instruction prompt similar to the original
    system_msg = 'You are a helpful and creative assistant who writes well.'
    user_message = (
        f'Please revise the following sentence with no changes to its length and '
        f'only output the revised version. Keep the original paragraph structure. '
        f'Do not change the words "[REPLACE]", "[PROMPT]", "[KEEPER]", and "[MODEL]" if present. '
        f'The sentence to revise: "{sentence}".'
    )

    # Lazy-load model and tokenizer
    try:
        if GEMMA_MODEL is None or GEMMA_TOKENIZER is None:
            GEMMA_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            GEMMA_TOKENIZER = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH, trust_remote_code=True, use_fast=False)
            # choose bf16/fp16 if available on GPU for speed/memory
            torch_dtype = None
            if GEMMA_DEVICE.type == 'cuda':
                try:
                    torch_dtype = torch.bfloat16
                except Exception:
                    torch_dtype = torch.float16
            GEMMA_MODEL = AutoModelForCausalLM.from_pretrained(
                GEMMA_MODEL_PATH,
                trust_remote_code=True,
                torch_dtype=(torch_dtype if torch_dtype is not None else None),
                device_map=None,
            )
            GEMMA_MODEL = GEMMA_MODEL.to(GEMMA_DEVICE).eval()
    except Exception as e:
        # if loading the local model fails, fall back to returning original sentence
        print(f"Failed to load local Gemma model at {GEMMA_MODEL_PATH}: {e}")
        return sentence

    prompt = system_msg + "\n" + user_message
    try:
        inputs = GEMMA_TOKENIZER(prompt, return_tensors='pt').to(GEMMA_DEVICE)
        # generate a short reply; constrain max_new_tokens to avoid very long outputs
        gen = GEMMA_MODEL.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=1.0,
            top_p=0.9,
            eos_token_id=GEMMA_TOKENIZER.eos_token_id,
            pad_token_id=GEMMA_TOKENIZER.eos_token_id,
        )
        out_ids = gen[0]
        # decode only the new tokens beyond the input length
        input_len = inputs['input_ids'].shape[1]
        new_ids = out_ids[input_len:]
        revised_sentence = GEMMA_TOKENIZER.decode(new_ids, skip_special_tokens=True).strip()
        # perform light post-processing to match earlier behavior
        revised_sentence = revised_sentence.replace('\n', ' ').strip()
        if revised_sentence.startswith("\'") or revised_sentence.startswith('"'):
            revised_sentence = revised_sentence[1:]
        if revised_sentence.endswith("\'") or revised_sentence.endswith('"'):
            revised_sentence = revised_sentence[:-1]
        if revised_sentence.endswith("'.") or revised_sentence.endswith('\".'):
            revised_sentence = revised_sentence[:-2]
        if not revised_sentence:
            return sentence
        print(f'revised: {revised_sentence}')
        return revised_sentence
    except Exception as e:
        print(f"Gemma generation error: {e}")
        return sentence

def apply_gpt_mutation(offspring, mutation_rate=0.01, API_key=None, reference=None, if_api=True):
    if if_api:
        # Prepare a safe pool to draw replacements from. Prefer reference tail
        # (to avoid selecting initial population) but fall back to any
        # available reference entries, and finally to an empty-string
        # placeholder to avoid IndexError when the pool is empty.
        pool = None
        try:
            if reference:
                if len(reference) > len(offspring):
                    pool = reference[len(offspring):]
                else:
                    pool = list(reference)
        except Exception:
            pool = None

        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                if API_key is None:
                    # Choose from the prepared pool when possible
                    if pool:
                        offspring[i] = random.choice(pool)
                    else:
                        # Fallback to any part of reference if available
                        if reference:
                            try:
                                offspring[i] = random.choice(reference)
                            except Exception:
                                offspring[i] = ""
                        else:
                            offspring[i] = ""
                else:
                    mutated = gpt_mutate(offspring[i], API_key)
                    # If the API returns None (e.g., invalid request), keep original
                    if mutated is not None:
                        offspring[i] = mutated
    else:       
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                try:
                    offspring[i] = mutate_structure_template(offspring[i])
                except Exception:
                    # Fallback to synonym replacement if structural mutation fails
                    offspring[i] = replace_with_synonyms(offspring[i])
    return offspring

def _classify_paragraph(para: str):
    p = para.strip()
    low = p.lower()
    # Role
    role_kw = ("you are", "act as", "persona:", "role:")
    # Constraints
    cons_kw = ("constraints", "rules", "do not", "must", "never", "limitation")
    # Steps / Procedure
    steps_kw = ("steps:", "procedure:", "step ", "first,", "then", "finally", "instructions:")
    # Output format
    out_kw = ("output:", "format:", "return", "provide", "json", "table", "list")
    # Verification
    ver_kw = ("verify", "check", "ensure", "validate", "audit", "self-evaluate")

    def any_in(keys):
        return any(k in low for k in keys)

    # Heuristics for numbered/bulleted steps
    if re.match(r"^\s*(?:\d+\.|[-*])\s+", p):
        return "steps"
    if any_in(role_kw):
        return "role"
    if any_in(cons_kw):
        return "constraint"
    if any_in(steps_kw):
        return "steps"
    if any_in(out_kw):
        return "output"
    if any_in(ver_kw):
        return "verification"
    return "misc"


def _extract_template_modules(text: str):
    # Split into paragraphs by blank lines, keep line-grouping simple
    paras = re.split(r"\n\s*\n", text.strip()) if text.strip() else []
    buckets = {"role": [], "constraint": [], "steps": [], "output": [], "verification": [], "misc": []}
    order = []
    for para in paras:
        cat = _classify_paragraph(para)
        buckets[cat].append(para)
        order.append((cat, para))
    return buckets, order


def _mutate_output_module(out_text: str):
    # Replace output format guidance among list/table/JSON
    formats = ["list", "table", "JSON"]
    choice = random.choice(formats)
    base = out_text.strip()
    # Normalize existing phrasing lightly
    base = re.sub(r"(?i)\bjson\b", "JSON", base)
    base = re.sub(r"(?i)\btable\b", "table", base)
    base = re.sub(r"(?i)\blist\b", "list", base)
    if choice == "JSON":
        # Provide a concise JSON schema hint
        replacement = "Output: Provide results in JSON with fields {\"summary\", \"steps\", \"verification\"}."
    elif choice == "table":
        replacement = "Output: Provide results in a table with columns [Step, Action, Note]."
    else:
        replacement = "Output: Provide results as a clear, numbered list."
    # If the original mentions format, replace the sentence, otherwise append guidance
    if re.search(r"(?i)(output|format|json|table|list)", base):
        return replacement
    return base + "\n\n" + replacement


def _mutate_steps_module(steps_text: str):
    s = steps_text
    # Perturb declared step count like "N steps"
    def repl(m):
        n = int(m.group(1))
        delta = random.choice([-1, 1, 2])
        new_n = max(2, n + delta)
        return f"{new_n} steps"
    s = re.sub(r"(?i)\b(\d+)\s*steps\b", repl, s)

    # If enumerated list, optionally add a reinforcing step
    lines = s.splitlines()
    if any(re.match(r"^\s*\d+\.\s+", ln) for ln in lines):
        insert_idx = min(len(lines), random.randint(1, max(1, len(lines)-1)))
        lines.insert(insert_idx, f"{insert_idx+1}. Add a brief verification summary.")
        # Renumber to keep monotonic ordering
        k = 1
        new_lines = []
        for ln in lines:
            if re.match(r"^\s*\d+\.\s+", ln):
                new_lines.append(re.sub(r"^\s*\d+\.", f"{k}.", ln))
                k += 1
            else:
                new_lines.append(ln)
        s = "\n".join(new_lines)
    return s


def mutate_structure_template(text: str) -> str:
    buckets, order = _extract_template_modules(text)
    # If nothing to mutate, return original
    total_len = sum(len(v) for v in buckets.values())
    if total_len == 0:
        return text

    # Copy buckets for mutation
    role = buckets["role"]
    cons = buckets["constraint"]
    steps = buckets["steps"]
    outp = buckets["output"]
    veri = buckets["verification"]
    misc = buckets["misc"]

    # 1) Module reordering: swap Steps and Constraint with some probability
    order_list = [("role", role), ("constraint", cons), ("steps", steps), ("output", outp), ("verification", veri), ("misc", misc)]
    if steps and cons and random.random() < 0.5:
        cons, steps = steps, cons

    # 2) Module replacement: change OutputFormat
    if outp and random.random() < 0.6:
        outp = [_mutate_output_module("\n".join(outp))]

    # 3) Module duplication: reinforce constraints or format
    if random.random() < 0.4:
        target = None
        if cons and random.random() < 0.6:
            target = "constraint"
        elif outp:
            target = "output"
        if target == "constraint":
            cons = cons + ["Additionally, adhere strictly to all constraints above."]
        elif target == "output":
            outp = outp + ["Also, restate the output in the specified format to reinforce."]

    # 4) Parameter perturbation for steps
    if steps:
        steps = [_mutate_steps_module("\n".join(steps))]

    # Recompose in canonical order: Role, Constraint, Steps, OutputFormat, Verification, Misc
    parts = []
    if role:
        parts.append("\n".join(role))
    if cons:
        parts.append("\n".join(cons))
    if steps:
        parts.append("\n".join(steps))
    if outp:
        parts.append("\n".join(outp))
    if veri:
        parts.append("\n".join(veri))
    if misc:
        parts.append("\n".join(misc))

    mutated = "\n\n".join([p for p in parts if p and p.strip()])
    return mutated if mutated.strip() else text

def apply_init_gpt_mutation(offspring, mutation_rate=0.01, API_key=None, if_api=True):
    for i in tqdm(range(len(offspring)), desc='initializing...'):
        if if_api:
            if random.random() < mutation_rate:
                offspring[i] = gpt_mutate(offspring[i], API_key)
        else:
            if random.random() < mutation_rate:
                offspring[i] = replace_with_synonyms(offspring[i])
    return offspring


def replace_with_synonyms(sentence, num=10):
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    stop_words = STOP_WORDS
    try:
        words = nltk.word_tokenize(sentence)
    except Exception:
        words = re.findall(r"\w+|[^\w\s]", sentence)
    uncommon_words = [word for word in words if word.lower() not in stop_words and word.lower() not in T]
    selected_words = random.sample(uncommon_words, min(num, len(uncommon_words)))
    for word in selected_words:
        synonyms = wordnet.synsets(word)
        if synonyms and synonyms[0].lemmas():
            synonym = synonyms[0].lemmas()[0].name()
            sentence = sentence.replace(word, synonym, 1)
    print(f'revised: {sentence}')
    return sentence

def adaptive_word_level_recombination(word_dict, control_suffixs, score_list, num_elites, batch_size, crossover=0.5,
                               mutation=0.01, API_key=None, reference=None, if_api=True, topk=2000):
    score_list = [-x for x in score_list]
    # Step 1: Sort the score_list and get corresponding control_suffixs
    sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
    sorted_control_suffixs = [control_suffixs[i] for i in sorted_indices]

    # Step 2: Select the elites
    elites = sorted_control_suffixs[:num_elites]
    parents_list = sorted_control_suffixs[num_elites:]

    # Step 3: Construct word list (with optional top-k pruning)
    # (函数签名在文件顶部已更新以接受 topk)
    try:
        word_dict = construct_momentum_word_dict(word_dict, control_suffixs, score_list, topk=topk)
    except TypeError:
        # 兼容旧的 construct_momentum_word_dict 调用（若未更新），回退到不剪枝的调用
        word_dict = construct_momentum_word_dict(word_dict, control_suffixs, score_list)
    if isinstance(topk, int) and topk > 0:
        print(f"Pruned/updated word_dict to top {topk} entries (current length: {len(word_dict)})")
    else:
        print(f"Length of current word dictionary: {len(word_dict)}")

    # check the length of parents
    parents_list = [x for x in parents_list if len(x) > 0]
    needed = batch_size - num_elites - len(parents_list)
    if needed > 0:
        print("Not enough parents, using reference instead.")
        # 这里补足父代数量，以保证总数够用
        # 优先使用 reference[batch_size:]（未被当作初始 population 的部分），若该 pool 为空则回退到整个 reference
        try:
            pool = reference[batch_size:]
        except Exception:
            pool = reference

        if not pool:
            # 最后退路：如果 reference 为空或不可用，则使用空字符串占位，避免 IndexError
            parents_list += ["" for _ in range(needed)]
        else:
            # 如果 pool 里元素少也没关系，random.choices 支持重复采样
            try:
                parents_list += random.choices(pool, k=needed)
            except Exception:
                # 兜底：若 random.choices 仍失败，改用循环选择
                for _ in range(needed):
                    try:
                        parents_list.append(random.choice(pool))
                    except Exception:
                        # 最后仍然失败时用空字符串占位
                        parents_list.append("")
        
    # Step 4: Apply word replacement with roulette wheel selection
    offspring = apply_word_replacement(word_dict, parents_list, crossover)
    offspring = apply_gpt_mutation(offspring, mutation, API_key, reference, if_api)

    # Combine elites with the mutated offspring
    next_generation = elites + offspring[:batch_size-num_elites]

    assert len(next_generation) == batch_size
    return next_generation, word_dict

### hybrid ###
def multi_granularity_hybrid(word_dict, control_suffixs, score_list, num_elites, batch_size, crossover=0.5,
                               mutation=0.01, API_key=None, reference=None, if_api=True, topk=2000):
    score_list = [-x for x in score_list]
    # Step 1: Sort the score_list and get corresponding control_suffixs
    sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k], reverse=True)
    sorted_control_suffixs = [control_suffixs[i] for i in sorted_indices]

    # Step 2: Select the elites
    elites = sorted_control_suffixs[:num_elites]
    parents_list = sorted_control_suffixs[num_elites:]

    # Step 3: Construct word list (with optional top-k pruning)
    # 这里是构建/更新动量词典的地方；默认会保留 top-K 高分词汇以限制增长
    # (函数签名在文件顶部已更新以接受 topk)
    try:
        # 这行代码是调用 construct_momentum_word_dict 函数来更新词典
        word_dict = construct_momentum_word_dict(word_dict, control_suffixs, score_list, topk=topk)
    except TypeError:
        # 兼容旧的 construct_momentum_word_dict 调用（若未更新），回退到不剪枝的调用
        word_dict = construct_momentum_word_dict(word_dict, control_suffixs, score_list)
    if isinstance(topk, int) and topk > 0:
        print(f"Pruned/updated word_dict to top {topk} entries (current length: {len(word_dict)})")
    else:
        print(f"Length of current word dictionary: {len(word_dict)}")

    # check the length of parents
    parents_list = [x for x in parents_list if len(x) > 0]
    needed = batch_size - num_elites - len(parents_list)
    if needed > 0:
        print("Not enough parents, using reference instead.")
        # 这里补足父代数量，以保证总数够用
        # 优先使用 reference[batch_size:]（未被当作初始 population 的部分），若该 pool 为空则回退到整个 reference
        try:
            pool = reference[batch_size:]
        except Exception:
            pool = reference

        if not pool:
            # 最后退路：如果 reference 为空或不可用，则使用空字符串占位，避免 IndexError
            parents_list += ["" for _ in range(needed)]
        else:
            # 如果 pool 里元素少也没关系，random.choices 支持重复采样
            try:
                # 这行代码是使用 random.choices 从 pool 中补足所需数量的父代
                parents_list += random.choices(pool, k=needed)
            except Exception:
                # 兜底：若 random.choices 仍失败，改用循环选择
                for _ in range(needed):
                    try:
                        parents_list.append(random.choice(pool))
                    except Exception:
                        # 最后仍然失败时用空字符串占位
                        parents_list.append("")
        
    # Step 4: Apply word replacement with roulette wheel selection
    # Branch A: word-level replacement offspring (existing path)
    offspring_word = apply_word_replacement(word_dict, parents_list, crossover)
    offspring_word = apply_gpt_mutation(offspring_word, mutation, API_key, reference, if_api)

    # Branch B: module-level recombination offspring
    offspring_mod = module_level_recombination(
        parents_list,
        control_suffixs,
        score_list,
        crossover_probability=crossover,
        max_modules_per_suffix=4,
    )
    offspring_mod = apply_gpt_mutation(offspring_mod, mutation, API_key, reference, if_api)

    # 下面这块代码是将两种后代交织在一起，形成最终的后代列表，是如何实现交织的
    offspring = []
    # for i in range(max(len(offspring_word), len(offspring_mod))):
    for i in range(len(offspring_mod)):
        # 交替添加 word-level 和 module-level 的后代
        if i < len(offspring_word):
            offspring.append(offspring_word[i])
        # i < len(offspring_mod)是用来控制添加模块级后代的条件，确保不会超出列表长度
        if i < len(offspring_mod):
            offspring.append(offspring_mod[i])
    # keep offspring size aligned with parents_list length
    offspring = offspring[:len(parents_list)]
    # Combine elites with the mutated offspring
    next_generation = elites + offspring[:batch_size-num_elites]

    assert len(next_generation) == batch_size
    return next_generation, word_dict

def build_strategy_library(suffix_list):
    """
    Build a reusable modular strategy library from a list of suffixes.
    Each suffix is segmented into coarse-grained modules such as:
    - role_play: persona or role instructions
    - override: instruction overrides, jailbreak cues
    - framing: context setup, scenario anchoring
    - eval_bias: hints affecting judge/score framing
    - misc: anything not categorized

    Returns a dict: {category: [module_text, ...]} with simple dedup.
    """
    lib = {"role_play": [], "override": [], "framing": [], "evaluation_bias": [], "misc": []}
    seen = {k: set() for k in lib}
    for s in suffix_list:
        modules = parse_suffix_modules(s or "")
        for cat, text in modules:
            text_norm = text.strip()
            if text_norm and text_norm not in seen[cat]:
                lib[cat].append(text_norm)
                seen[cat].add(text_norm)
    return lib

def module_level_recombination(parents_list, control_suffixs, score_list, crossover_probability=0.5,
                               max_modules_per_suffix=4):
    """
    Third action: Module-Level Recombination.
    Build a strategy library, compute module attention, and generate offspring
    with module replacement in one place.
    """
    strategy_lib = build_strategy_library(control_suffixs)
    attn = compute_module_attention(strategy_lib, control_suffixs, score_list)
    offspring_mod = apply_module_replacement(
        parents_list,
        strategy_lib,
        attn,
        max_modules_per_suffix=max_modules_per_suffix,
        crossover_probability=crossover_probability,
    )
    return offspring_mod

def parse_suffix_modules(suffix):
    """
    Heuristic segmentation of a suffix into labeled modules.
    The method relies on lightweight patterns and separators.
    Returns list of tuples: [(category, text), ...].
    """
    if suffix is None:
        return [("misc", "")]
    try:
        text = str(suffix)
    except Exception:
        text = ""

    # Split at common separators to form candidate blocks
    blocks = re.split(r"\n\n|\n|\r|\t|---|===|\*\*\*|\[\w+\]|\(\w+\)", text)
    modules = []
    for b in blocks:
        t = b.strip()
        if not t:
            continue
        lt = t.lower()
        if any(k in lt for k in ["you are", "as a", "role", "persona", "act as"]):
            modules.append(("role_play", t))
        elif any(k in lt for k in ["ignore", "override", "disregard", "bypass", "jailbreak", "do not follow"]):
            modules.append(("override", t))
        elif any(k in lt for k in ["scenario", "context", "frame", "consider", "assume", "imagine"]):
            modules.append(("framing", t))
        elif any(k in lt for k in ["evaluation", "judge", "score", "rating", "preference", "winrate", "bias"]):
            modules.append(("evaluation_bias", t))
        else:
            modules.append(("misc", t))
    # Fallback if nothing detected
    if not modules:
        modules = [("misc", text)]
    return modules

def compute_module_attention(strategy_lib, suffix_list, score_list):
    """
    Compute attention weights for modules via simple credit assignment.
    For each category, derive a weight from average scores of suffixes
    containing blocks from that category. Then normalize to probabilities.
    Returns dict {category: weight}.
    """
    # Map suffix → categories present
    cat_scores = {"role_play": [], "override": [], "framing": [], "evaluation_bias": [], "misc": []}
    for s, sc in zip(suffix_list, score_list):
        cats = set(cat for cat, _ in parse_suffix_modules(s or ""))
        for c in cats:
            cat_scores[c].append(sc)
    weights = {}
    for c, arr in cat_scores.items():
        if arr:
            # Higher fitness corresponds to lower loss; invert as above
            fitness_vals = [-x for x in arr]
            weights[c] = float(np.mean(fitness_vals))
        else:
            weights[c] = 0.0
    # Normalize
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 1e-8:
        # Even distribution if no signal
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: max(v, 0.0) / total for k, v in weights.items()}

def apply_module_replacement(parents_list, strategy_lib, attn_weights, max_modules_per_suffix=4,
                             crossover_probability=0.5):
    """
    Create new  offspring by replacing parts of each parent suffix with modules
    sampled from the strategy library, guided by attention weights.
    """
    cats = list(strategy_lib.keys())
    probs = np.array([attn_weights.get(c, 0.0) for c in cats], dtype=float)
    if probs.sum() <= 1e-8:
        probs = np.ones(len(cats), dtype=float) / max(1, len(cats))
    else:
        probs = probs / probs.sum()

    offspring = []
    for parent in parents_list:
        # Parse the parent into modules
        parent_modules = parse_suffix_modules(parent or "")
        new_modules = []
        for cat, text in parent_modules:
            if random.random() < crossover_probability:
                # Replace this block with a sampled module from weighted category
                # 1) pick a category by attention
                idx = np.random.choice(len(cats), p=probs)
                pick_cat = cats[idx]
                pool = strategy_lib.get(pick_cat, [])
                if pool:
                    new_text = random.choice(pool)
                else:
                    new_text = text
                new_modules.append((pick_cat, new_text))
            else:
                new_modules.append((cat, text))

        # Limit total modules to avoid explosion; sample if too many
        if len(new_modules) > max_modules_per_suffix:
            try:
                new_modules = random.sample(new_modules, max_modules_per_suffix)
            except Exception:
                new_modules = new_modules[:max_modules_per_suffix]

        # Recompose into text in a simple block-wise form
        recomposed = []
        for cat, text in new_modules:
            # keep a minimal header to separate blocks; avoid special tokens that may conflict
            recomposed.append(text.strip())
        offspring.append("\n\n".join(recomposed))
    return offspring

def construct_momentum_word_dict(word_dict, control_suffixs, score_list, topk=-1):
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    # Use project-level STOP_WORDS fallback (handles environments without NLTK corpora)
    stop_words = STOP_WORDS
    if len(control_suffixs) != len(score_list):
        raise ValueError("control_suffixs and score_list must have the same length.")

    word_scores = defaultdict(list)

    for prefix, score in zip(control_suffixs, score_list):
        # Ensure prefix is a string-like object; if not, coerce to string safely.
        if prefix is None:
            continue
        if not isinstance(prefix, str):
            try:
                prefix = str(prefix)
            except Exception:
                # cannot coerce, skip this item
                continue
        # Try NLTK tokenization first; if resources missing, fallback to regex split
        try:
            toks = nltk.word_tokenize(prefix)
        except Exception:
            try:
                toks = re.findall(r"\w+|[^\w\s]", prefix)
            except Exception:
                toks = []
        words = set([word for word in toks if word.lower() not in STOP_WORDS and word.lower() not in T])
        for word in words:
            word_scores[word].append(score)

    for word, scores in word_scores.items():
        avg_score = sum(scores) / len(scores)
        if word in word_dict:
            word_dict[word] = (word_dict[word] + avg_score) / 2
        else:
            word_dict[word] = avg_score

    sorted_word_dict = OrderedDict(sorted(word_dict.items(), key=lambda x: x[1], reverse=True))
    if topk == -1:
        topk_word_dict = dict(list(sorted_word_dict.items()))
    else:
        topk_word_dict = dict(list(sorted_word_dict.items())[:topk])
    return topk_word_dict


def get_synonyms(word):
    # If WordNet is unavailable, return empty synonym list
    if not WORDNET_AVAILABLE or wordnet is None:
        return []
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def word_roulette_wheel_selection(word, word_scores):
    if not word_scores:
        return word
    min_score = min(word_scores.values())
    adjusted_scores = {k: v - min_score for k, v in word_scores.items()}
    total_score = sum(adjusted_scores.values())
    pick = random.uniform(0, total_score)
    current_score = 0
    for synonym, score in adjusted_scores.items():
        current_score += score
        if current_score > pick:
            if word.istitle():
                return synonym.title()
            else:
                return synonym
            

def replace_with_best_synonym(sentence, word_dict, crossover_probability):
    # Guard: if sentence is empty/None, return as-is
    if sentence is None:
        return ""
    # If sentence is not a string (e.g., genome list), convert to a reasonable string
    if not isinstance(sentence, str):
        try:
            if isinstance(sentence, (list, tuple)):
                # join list/tuple items with commas for tokenization
                sentence = ",".join(map(str, sentence))
            else:
                sentence = str(sentence)
        except Exception:
            return ""

    stop_words = STOP_WORDS
    T = {"llama2", "meta", "vicuna", "lmsys", "guanaco", "theblokeai", "wizardlm", "mpt-chat",
         "mosaicml", "mpt-instruct", "falcon", "tii", "chatgpt", "modelkeeper", "prompt"}
    paragraphs = sentence.split('\n\n')
    modified_paragraphs = []
    # If word_dict is empty, use 0 as a neutral baseline to avoid ValueError
    if word_dict:
        min_value = min(word_dict.values())
    else:
        min_value = 0

    for paragraph in paragraphs:
        try:
            words = replace_quotes(nltk.word_tokenize(paragraph))
        except Exception:
            try:
                words = replace_quotes(re.findall(r"\w+|[^\w\s]", paragraph))
            except Exception:
                words = []

        # If tokenization yields no words (e.g., paragraph is empty), keep it unchanged
        if not words:
            modified_paragraphs.append("")
            continue

        count = 0
        for i, word in enumerate(words):
            if random.random() < crossover_probability:
                if word.lower() not in stop_words and word.lower() not in T:
                    synonyms = get_synonyms(word.lower())
                    word_scores = {syn: word_dict.get(syn, min_value) for syn in synonyms}
                    best_synonym = word_roulette_wheel_selection(word, word_scores)
                    if best_synonym:
                        words[i] = best_synonym
                        count += 1
                        if count >= 5:
                            break
            else:
                if word.lower() not in stop_words and word.lower() not in T:
                    synonyms = get_synonyms(word.lower())
                    word_scores = {syn: word_dict.get(syn, 0) for syn in synonyms}
                    best_synonym = word_roulette_wheel_selection(word, word_scores)
                    if best_synonym:
                        words[i] = best_synonym
                        count += 1
                        if count >= 5:
                            break
        modified_paragraphs.append(join_words_with_punctuation(words))
    return '\n\n'.join(modified_paragraphs)

def replace_quotes(words):
    new_words = []
    quote_flag = True

    for word in words:
        if word in ["``", "''"]:
            if quote_flag:
                new_words.append('“')
                quote_flag = False
            else:
                new_words.append('”')
                quote_flag = True
        else:
            new_words.append(word)
    return new_words

def apply_word_replacement(word_dict, parents_list, crossover=0.5):
    return [replace_with_best_synonym(sentence, word_dict, crossover) for sentence in parents_list]

def join_words_with_punctuation(words):
    # Guard against empty word lists
    if not words:
        return ""

    sentence = words[0]
    previous_word = words[0]
    flag = 1
    for word in words[1:]:
        if word in [",", ".", "!", "?", ":", ";", ")", "]", "}", '”']:
            sentence += word
        else:
            if previous_word in ["[", "(", "'", '"', '“']:
                if previous_word in ["'", '"'] and flag == 1:
                    sentence += " " + word
                else:
                    sentence += word
            else:
                if word in ["'", '"'] and flag == 1:
                    flag = 1 - flag
                    sentence += " " + word
                elif word in ["'", '"'] and flag == 0:
                    flag = 1 - flag
                    sentence += word
                else:
                    if "'" in word and re.search('[a-zA-Z]', word):
                        sentence += word
                    else:
                        sentence += " " + word
        previous_word = word
    return sentence