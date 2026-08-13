"""
数据类型定义
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum


class BenchmarkType(Enum):
    """支持的基准测试类型"""
    ALPACA_EVAL = "alpaca_eval"
    CODE_JUDGE_BENCH = "code_judge_bench"
    ARENA_HARD = "arena_hard"


class AttackType(Enum):
    """支持的攻击类型"""
    FLIP_ATTACK = "flip_attack"
    SUFFIX_INJECTION = "suffix_injection"
    UNCERTAINTY_CALIBRATION = "uncertainty_calibration"
    ORDER_MANIPULATION = "order_manipulation"
    TONE_MODIFICATION = "tone_modification"
    POSITION_MANIPULATION = "position_manipulation"
    DISTRACTOR_ATTACK = "distractor_attack"
    PROMPT_INJECTION = "prompt_injection"
    MARKER_INJECTION = "marker_injection"
    FORMATTING_ATTACK = "formatting_attack"
    AUTHORITY_ATTACK = "authority_attack"
    UNICODE_ATTACK = "unicode_attack"
    COT_POISONING = "cot_poisoning"


class JudgeType(Enum):
    """支持的Judge类型"""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    CLAUDE_3 = "claude-3"
    CLAUDE_3_5 = "claude-3-5-sonnet"


@dataclass
class PairwiseExample:
    """Pairwise评估样本"""
    question_id: str
    instruction: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AttackResult:
    """攻击结果"""
    question_id: str
    original_preference: int  # 0 for A, 1 for B
    attacked_preference: int  # 0 for A, 1 for B
    success: bool
    query_count: int
    attack_method: str
    modified_response_a: Optional[str] = None
    modified_response_b: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class JudgeResponse:
    """Judge响应"""
    preference: int  # 0 for A, 1 for B
    confidence: float
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


@dataclass
class RLEpisode:
    """强化学习回合"""
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    done: bool
    info: Optional[Dict[str, Any]] = None


@dataclass
class AttackConfig:
    """攻击配置"""
    benchmark: BenchmarkType
    judge: JudgeType
    attack_methods: List[AttackType]
    max_queries: int
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 30


@dataclass
class RLConfig:
    """强化学习配置"""
    algorithm: str = "PPO"
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    save_freq: int = 50000
    # DQN specific parameters
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    eval_episodes: int = 10
