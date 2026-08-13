"""
Judge接口实现
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import aiohttp
import json
import time
from dataclasses import dataclass

from utils.data_types import PairwiseExample, JudgeResponse, JudgeType


@dataclass
class JudgeConfig:
    """Judge配置"""
    api_key: str
    base_url: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 30


class BaseJudge(ABC):
    """基础Judge类"""
    
    def __init__(self, config: JudgeConfig):
        self.config = config
    
    @abstractmethod
    async def judge_pairwise(self, example: PairwiseExample) -> JudgeResponse:
        """进行pairwise判断"""
        pass
    
    @abstractmethod
    def get_judge_prompt(self, example: PairwiseExample) -> str:
        """获取judge prompt"""
        pass


class GPT4Judge(BaseJudge):
    """GPT-4 Judge实现"""
    
    def __init__(self, config: JudgeConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://www.dmxapi.com/v1/chat/completions"
    
    def get_judge_prompt(self, example: PairwiseExample) -> str:
        """获取GPT-4 judge prompt"""
        return f"""You are a helpful assistant, that ranks models by the quality of their answers.

I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

Here is the prompt:
{{
    "instruction": "{example.instruction}"
}}

Here are the outputs of the models:
[
    {{
        "model": "model_1",
        "answer": "{example.response_a}"
    }},
    {{
        "model": "model_2", 
        "answer": "{example.response_b}"
    }}
]

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:
[
    {{'model': <model-name>, 'rank': <model-rank>}},
    {{'model': <model-name>, 'rank': <model-rank>}}
]

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give."""
    
    async def judge_pairwise(self, example: PairwiseExample) -> JudgeResponse:
        """使用GPT-4进行pairwise判断"""
        prompt = self.get_judge_prompt(example)
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
            try:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["choices"][0]["message"]["content"]
                        
                        # 解析响应
                        preference, confidence = self._parse_response(content)
                        usage_data = result.get("usage") or {}
                        prompt_tokens = int(usage_data.get("prompt_tokens", 0) or 0)
                        completion_tokens = int(usage_data.get("completion_tokens", 0) or 0)
                        total_tokens = int(usage_data.get("total_tokens", prompt_tokens + completion_tokens) or 0)
                        usage = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        }
                        
                        return JudgeResponse(
                            preference=preference,
                            confidence=confidence,
                            raw_response=content,
                            usage=usage,
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {response.status} - {error_text}")
            except asyncio.TimeoutError:
                raise Exception("Request timeout")
            except Exception as e:
                raise Exception(f"Judge request failed: {str(e)}")
    
    def _parse_response(self, content: str) -> tuple[int, float]:
        """解析GPT-4的响应"""
        try:
            # 尝试解析JSON响应
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # 找到rank为1的模型
                for item in result:
                    if item.get('rank') == 1:
                        model_name = item.get('model', '')
                        if 'model_1' in model_name:
                            return 0, 0.8  # 偏好A
                        elif 'model_2' in model_name:
                            return 1, 0.8  # 偏好B
                
                # 如果没找到rank=1，尝试其他解析方式
                if len(result) == 2:
                    rank1 = result[0].get('rank', 0)
                    rank2 = result[1].get('rank', 0)
                    if rank1 < rank2:
                        return 0, 0.7
                    else:
                        return 1, 0.7
            
            # 如果JSON解析失败，尝试文本解析
            if "model_1" in content.lower() and "better" in content.lower():
                return 0, 0.6
            elif "model_2" in content.lower() and "better" in content.lower():
                return 1, 0.6
            
            # 默认返回随机偏好
            import random
            return random.randint(0, 1), 0.5
            
        except Exception:
            # 解析失败，返回随机偏好
            import random
            return random.randint(0, 1), 0.3


class ClaudeJudge(BaseJudge):
    """Claude Judge实现"""
    
    def __init__(self, config: JudgeConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com/v1/messages"
    
    def get_judge_prompt(self, example: PairwiseExample) -> str:
        """获取Claude judge prompt"""
        return f"""Please evaluate the quality of two AI assistant responses to the following question:

Question: {example.instruction}

Response A: {example.response_a}

Response B: {example.response_b}

Please determine which response is better in terms of helpfulness, accuracy, and overall quality. Respond with either "A" or "B" followed by a brief explanation."""
    
    async def judge_pairwise(self, example: PairwiseExample) -> JudgeResponse:
        """使用Claude进行pairwise判断"""
        prompt = self.get_judge_prompt(example)
        
        headers = {
            "x-api-key": self.config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.timeout)) as session:
            try:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result["content"][0]["text"]
                        
                        # 解析响应
                        preference, confidence = self._parse_response(content)
                        usage_data = result.get("usage") or {}
                        input_tokens = int(usage_data.get("input_tokens", 0) or 0)
                        output_tokens = int(usage_data.get("output_tokens", 0) or 0)
                        total_tokens = int(usage_data.get("total_tokens", input_tokens + output_tokens) or 0)
                        usage = {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": total_tokens,
                        }
                        
                        return JudgeResponse(
                            preference=preference,
                            confidence=confidence,
                            raw_response=content,
                            usage=usage,
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(f"API request failed: {response.status} - {error_text}")
            except asyncio.TimeoutError:
                raise Exception("Request timeout")
            except Exception as e:
                raise Exception(f"Judge request failed: {str(e)}")
    
    def _parse_response(self, content: str) -> tuple[int, float]:
        """解析Claude的响应"""
        content_lower = content.lower()
        
        if "response a" in content_lower or "a)" in content_lower:
            return 0, 0.8
        elif "response b" in content_lower or "b)" in content_lower:
            return 1, 0.8
        else:
            # 默认返回随机偏好
            import random
            return random.randint(0, 1), 0.5


def create_judge(judge_type: JudgeType, config: JudgeConfig) -> BaseJudge:
    """创建Judge实例"""
    if judge_type == JudgeType.GPT4:
        return GPT4Judge(config)
    elif judge_type == JudgeType.CLAUDE_3:
        return ClaudeJudge(config)
    else:
        raise ValueError(f"Unsupported judge type: {judge_type}")
