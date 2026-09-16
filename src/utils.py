"""LinearRAG 各模块共享的通用工具。

本模块提供内容哈希、LLM 调用封装、答案标准化、
日志初始化和分数归一化等跨模块能力。
"""

from hashlib import md5
from dataclasses import dataclass, field
from typing import List, Dict
import httpx
from openai import OpenAI
from collections import defaultdict
import multiprocessing as mp
import re
import string
import logging
import numpy as np
import os


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """计算内容的 MD5 哈希标识，并在需要时添加命名空间前缀。"""
    return prefix + md5(content.encode()).hexdigest()


# LLM 模型封装类
class LLM_Model:
    def __init__(self, llm_model):
        http_client = httpx.Client(timeout=60.0, trust_env=False)
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            http_client=http_client
        )

        self.llm_config = {
            "model": llm_model,
            "max_tokens": 2000,
            "temperature": 0,
        }

    def infer(self, messages):
        response = self.openai_client.chat.completions.create(**self.llm_config, messages=messages)
        return response.choices[0].message.content


def normalize_answer(s):
    """将答案转为适合包含比较的小写、无标点、无冠词形式。"""
    # 处理空值与非字符串类型
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s) 

    def remove_articles(text):
        """移除英文冠词。"""
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        """将连续空白合并为单个空格。"""
        return " ".join(text.split())

    def remove_punc(text):
        """移除文本中的标点符号。"""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        """将文本转换为小写。"""
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def setup_logging(log_file):
    """配置控制台和文件日志，并压低 HTTP 客户端的冗余日志。"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]  
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handlers.append(logging.FileHandler(log_file, mode='a', encoding='utf-8')) # type: ignore
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        force=True
    )

    # 屏蔽 httpx/openai 产生的冗余 HTTP 请求日志，例如 401 Unauthorized。
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def min_max_normalize(x):
    """对数值数组执行 min-max 归一化。"""
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val

    # 处理所有数值相同，即取值范围为零的情况。
    if range_val == 0:
        return np.ones_like(x)  # 返回与 x 形状相同的全 1 数组。

    return (x - min_val) / range_val
