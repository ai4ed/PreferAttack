# utils/pairwise_loader.py
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

def load_pairwise_json(path: str) -> List[Dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 标准化字段名（如果你的字段不同，在这里改映射）
    norm = []
    for i, ex in enumerate(data):
        norm.append({
            "instruction": ex.get("instruction") or ex.get("prompt") or "",
            "response_a": ex.get("response_a") or ex.get("output_a") or ex.get("A") or "",
            "response_b": ex.get("response_b") or ex.get("output_b") or ex.get("B") or "",
            "meta": ex.get("meta") or {"id": i}
        })
    return norm

def load_from_split_info(split_info_path: str, split: str = "test") -> Tuple[List[Dict], str]:
    """
    读取你上传的 split 描述文件，拿到实际数据路径再加载。
    split: 'train' or 'test'
    """
    with open(split_info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    data_files = info["data_files"]  # 包含 train/test 的相对路径
    target = data_files[split]       # 例如 data/split/alpaca_eval_test.json
    return load_pairwise_json(target), target
