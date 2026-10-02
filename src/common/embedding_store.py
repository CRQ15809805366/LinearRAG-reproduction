"""文本嵌入的生成、索引与持久化存储。

本模块使用内容哈希标识文本，批量编码尚未存储的文本，
并通过 Parquet 文件维护文本、哈希 ID 和嵌入向量之间的映射。
"""

from copy import deepcopy
from src.common.utils import compute_mdhash_id
import numpy as np
import pandas as pd
import os


class EmbeddingStore:
    # == 初始化与已有数据加载 ==
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        # 初始化嵌入存储类
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace

        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        self.hash_id_to_text = {}
        self.hash_id_to_idx = {}
        self.text_to_hash_id = {}

        # 加载现有数据
        self._load_data()

    def _load_data(self):
        """如果 Parquet 存储文件存在，则加载其中的文本、哈希 ID 和嵌入。"""

        # 如果数据库文件存在，则从中加载数据
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)

            # 将数据加载到类的属性中
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["text"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
            self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}

            # 打印加载的记录数
            print(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")


    # == 写入：新文本编码、映射更新与持久化 ==
    def insert_text(self, text_list):
        """为新文本生成哈希 ID 和嵌入，并写入存储。"""

        # 初始化一个字典来存储文本的哈希 ID 和内容
        nodes_dict = {}

        # 计算文本的哈希 ID，并将其存储在字典中
        for text in text_list:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}

        # 获取所有哈希 ID，并找出哪些是新的（即不在现有存储中的）
        all_hash_ids = list(nodes_dict.keys())
        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]      
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        all_embeddings = self.embedding_model.encode(texts_to_encode, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)

        # 将缺失的文本及其嵌入插入到存储中
        self._upsert(missing_ids, texts_to_encode, all_embeddings)

    def _upsert(self, hash_ids, texts, embeddings):
        """将新的哈希 ID、文本和嵌入追加到内存映射并持久化。"""
        # 将新的哈希 ID、文本和嵌入添加到现有的存储中
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)

        # 创建哈希 ID 到索引、文本和嵌入的映射
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}

        # 保存数据到文件
        self._save_data()

    def _save_data(self):
        """将当前嵌入存储写入 Parquet 文件。"""
        # 将当前的哈希 ID、文本和嵌入保存到 Parquet 文件中
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "text": self.texts,
            "embedding": self.embeddings
        })
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        data_to_save.to_parquet(self.db_filename, index=False)


    # == 对外接口：读取映射、编码文本与取得向量 ==
    def get_hash_id_to_text(self):
        """返回哈希 ID 到文本映射的深拷贝。"""
        return deepcopy(self.hash_id_to_text)

    def encode_texts(self, texts):
        """使用当前嵌入模型批量编码文本。"""
        return self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)

    def get_embeddings(self, hash_ids):
        if not hash_ids:
            return np.array([])

        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings)[indices]

        return embeddings
