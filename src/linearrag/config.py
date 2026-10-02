"""LinearRAG 的运行配置。

本模块集中定义数据集、嵌入与 LLM、存储、并发、图检索以及可选属性增强所需的参数。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from src.common.utils import LLM_Model
from src.paths import CACHE_DIR


@dataclass
class LinearRAGConfig:
    # == 数据集与模型 ==
    dataset_name: str
    embedding_model: str = "all-mpnet-base-v2"
    llm_model: LLM_Model = None

    # == 文本切分 ==
    chunk_token_size: int = 1000
    chunk_overlap_token_size: int = 100

    # == NER、存储与批处理 ==
    spacy_model: str = "en_core_web_trf"
    working_dir: Union[str, Path] = CACHE_DIR
    batch_size: int = 128
    max_workers: int = 16

    # == 检索与图传播 ==
    retrieval_top_k: int = 5
    max_iterations: int = 3
    top_k_sentence: int = 1
    passage_ratio: float = 1.5
    passage_node_weight: float = 0.05
    damping: float = 0.5
    iteration_threshold: float = 0.5
    use_vectorized_retrieval: bool = False  # True 表示向量化矩阵计算，False 表示 BFS 迭代

    # == 可选分支：属性增强 ==
    enable_hybrid_attribute_fallback: bool = False
    attribute_keyword_boost: float = 0.25
    attribute_query_keywords: list[str] = field(default_factory=lambda: [
        "born", "birth", "where", "when", "located", "location", "founded", "founder",
        "died", "death", "nationality", "capital", "date", "year"
    ])
