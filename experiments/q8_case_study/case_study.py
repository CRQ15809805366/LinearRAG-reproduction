"""复现 Q8 单案例，并观察原版 HippoRAG 的局部图和支持文本排名。"""

from __future__ import annotations

import json
import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.linearrag.config import LinearRAGConfig
from src.linearrag.LinearRAG import LinearRAG
from src.paths import CACHE_DIR, DATASETS_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.common.utils import LLM_Model


DATASET = "2wikimultihop"
QUESTION_ID = "3a3c2efe0bdc11eba7f7acde48001122"
EXPERIMENT_ID = "q8-20passages-case-20261001"

# 没有改算法, 只是周围加了一层简单记录，把每轮选中了什么句子、激活了什么实体保存下来。
class CaseStudyLinearRAG(LinearRAG):
    """记录原 BFS 选中的句子和激活实体，不改变传播计算。"""

    def __init__(self, global_config: LinearRAGConfig):
        """初始化案例轨迹和原 LinearRAG。"""
        self.trace: list[dict] = []
        super().__init__(global_config)

    def restore_index_read_only(self) -> None:
        """从已有缓存恢复内存图，不重建或写入缓存。"""
        cache_dir = Path(self.config.working_dir) / self.dataset_name
        ner_path = cache_dir / "ner_results.json"
        if not ner_path.exists():
            raise FileNotFoundError(f"Existing cache required: {ner_path}")
        if not all(
            store.hash_ids
            for store in (
                self.passage_embedding_store,
                self.entity_embedding_store,
                self.sentence_embedding_store,
            )
        ):
            raise RuntimeError("Existing embedding caches are required; refusing to rebuild")

        cached = json.loads(ner_path.read_text(encoding="utf-8"))
        (
            _entity_nodes,
            _sentence_nodes,
            passage_to_entities,
            self.entity_to_sentence,
            self.sentence_to_entity,
        ) = self.extract_nodes_and_edges(
            cached["passage_hash_id_to_entities"], cached["sentence_to_entities"]
        )
        self.entity_hash_id_to_sentence_hash_ids = {
            self.entity_embedding_store.text_to_hash_id[entity]: [
                self.sentence_embedding_store.text_to_hash_id[sentence]
                for sentence in sentences
            ]
            for entity, sentences in self.entity_to_sentence.items()
        }
        self.sentence_hash_id_to_entity_hash_ids = {
            self.sentence_embedding_store.text_to_hash_id[sentence]: [
                self.entity_embedding_store.text_to_hash_id[entity]
                for entity in entities
            ]
            for sentence, entities in self.sentence_to_entity.items()
        }
        from collections import defaultdict

        self.node_to_node_stats = defaultdict(dict)
        self.add_entity_to_passage_edges(passage_to_entities)
        self.add_adjacent_passage_edges()
        self.augment_graph()

    def calculate_entity_scores(
        self,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        """执行原 BFS 计算，并保留每轮选中的句子与激活实体。"""
        active_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for index, text, hash_id, score in zip(
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
        ):
            active_entities[hash_id] = (index, score, 1)
            entity_weights[self.node_name_to_vertex_idx[hash_id]] = score
            self.trace.append({
                "iteration": 0,
                "entity": text,
                "score": float(score),
                "kind": "seed",
            })

        used_sentences = set()
        current_entities = active_entities.copy()
        iteration = 1
        while current_entities and iteration < self.config.max_iterations:
            new_entities = {}
            for entity_hash_id, (_entity_id, entity_score, _tier) in current_entities.items():
                if entity_score < self.config.iteration_threshold:
                    continue
                sentence_hash_ids = [
                    sid
                    for sid in self.entity_hash_id_to_sentence_hash_ids[entity_hash_id]
                    if sid not in used_sentences
                ]
                if not sentence_hash_ids:
                    continue
                sentence_indices = [
                    self.sentence_embedding_store.hash_id_to_idx[sid]
                    for sid in sentence_hash_ids
                ]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_column = question_embedding.reshape(-1, 1)
                similarities = np.dot(sentence_embeddings, question_column).flatten()
                top_indices = np.argsort(similarities)[::-1][: self.config.top_k_sentence]
                for top_index in top_indices:
                    sentence_hash_id = sentence_hash_ids[top_index]
                    sentence_score = float(similarities[top_index])
                    used_sentences.add(sentence_hash_id)
                    activated = []
                    for next_hash_id in self.sentence_hash_id_to_entity_hash_ids[sentence_hash_id]:
                        next_score = float(entity_score * sentence_score)
                        if next_score < self.config.iteration_threshold:
                            continue
                        node_index = self.node_name_to_vertex_idx[next_hash_id]
                        entity_weights[node_index] += next_score
                        new_entities[next_hash_id] = (node_index, next_score, iteration + 1)
                        activated.append({
                            "entity": self.entity_embedding_store.hash_id_to_text[next_hash_id],
                            "score": next_score,
                        })
                    self.trace.append({
                        "iteration": iteration,
                        "from_entity": self.entity_embedding_store.hash_id_to_text[entity_hash_id],
                        "sentence": self.sentence_embedding_store.hash_id_to_text[sentence_hash_id],
                        "sentence_score": sentence_score,
                        "activated_entities": activated,
                    })
            active_entities.update(new_entities)
            current_entities = new_entities.copy()
            iteration += 1
        return entity_weights, active_entities


def prepare_case():
    """固定论文问题、完整语料和两条支持事实所在的段落。"""
    questions = json.loads(
        (DATASETS_DIR / DATASET / "questions.json").read_text(encoding="utf-8")
    )
    question = next(item for item in questions if item.get("id") == QUESTION_ID)
    chunks = json.loads(
        (DATASETS_DIR / DATASET / "chunks.json").read_text(encoding="utf-8")
    )
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]

    # 用事实原文定位支持段落，不把标准答案注入 HippoRAG 检索。
    support_fragments = [
        "holy roman empress by marriage to frederick barbarossa",
        "he was elected king of germany at frankfurt on 4 march 1152",
    ]
    support_indices = []
    for fragment in support_fragments:
        matches = [i for i, passage in enumerate(passages) if fragment in passage.lower()]
        if len(matches) != 1:
            raise ValueError(f"Expected one support passage for: {fragment}")
        support_indices.append(matches[0])

    # 只读文本列核对缓存语料；准备阶段不实例化模型或客户端。
    import pandas as pd
    cached_passages = pd.read_parquet(
        CACHE_DIR / DATASET / "passage_embedding.parquet", columns=["text"]
    )["text"].tolist()
    if len(cached_passages) != len(passages) or set(cached_passages) != set(passages):
        raise ValueError("LinearRAG cached corpus differs from dataset")

    # 保留两个支持片段，再固定抽取十八个干扰片段；双方使用同一份小语料。
    distractors = [i for i in range(len(passages)) if i not in support_indices]
    source_indices = sorted(support_indices + random.Random(20261001).sample(distractors, 18))
    passages = [passages[i] for i in source_indices]
    observation = dict( # 运行后，请检查这三个实体，以及这两条支持文本的状态
        entities=["Beatrice I", "Frederick Barbarossa", "Germany"],
        support_passage_indices=[source_indices.index(i) for i in support_indices],
        source_passage_indices=source_indices,
    )
    return question, passages, observation


def main() -> int:
    """先准备固定输入；显式运行时才执行双方检索和答案生成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--llm-model", default="qwen3.8-flash")
    args = parser.parse_args()
    question, passages, observation = prepare_case()

    output_dir = EXPERIMENT_RESULTS_DIR / "q8_case_study" / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = dict(
        status="prepared", dataset=DATASET, question=question,
        passage_count=len(passages),
        corpus_sha256=hashlib.sha256(
            json.dumps(passages, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        generation_model=args.llm_model, extraction_model="qwen3.8-flash",
        embedding_model="all-mpnet-base-v2", baseline="HippoRAG 2024 v1.0.0",
        case_observation=observation,
        observation_role="Read-only diagnostics; not used in ranking or generation",
        indexing_llm_calls_if_no_cache=2 * len(passages),
        query_ner_llm_calls_if_no_cache=1, generation_calls=2,
    )
    (output_dir / "prepared.json").write_text(
        json.dumps(prepared, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.prepare_only:
        print(json.dumps(prepared, ensure_ascii=False, indent=2))
        return 0

    # 先运行 LinearRAG，立即保存，避免后续 HippoRAG 失败丢失结果。
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(str(MODELS_DIR / "all-mpnet-base-v2"), device="cuda")
    llm_model = LLM_Model(args.llm_model)
    config = LinearRAGConfig(
        dataset_name=DATASET,
        embedding_model=embedding_model,
        llm_model=llm_model,
        spacy_model="en_core_web_trf",
        working_dir=output_dir / "linearrag_cache",
        max_workers=1,
        max_iterations=3,
        iteration_threshold=0.4,
        passage_ratio=0.05,
        top_k_sentence=1,
        use_vectorized_retrieval=False,
    )
    rag = CaseStudyLinearRAG(config)
    rag.index(passages)
    # 两种方法必须消费完全相同的 passage 文本；旧缓存只能读取。
    if set(rag.passage_embedding_store.texts) != set(passages):
        raise ValueError("LinearRAG cached corpus differs from HippoRAG input")
    prediction = rag.qa([question])[0]
    prediction["id"] = question["id"]

    linear_result = dict(
        prediction=prediction, trace=rag.trace,
        parameters=dict(max_iterations=3, iteration_threshold=0.4,
                        passage_ratio=0.05, top_k_sentence=1,
                        retrieval_top_k=5, retrieval_path="BFS"),
    )
    (output_dir / "linearrag.json").write_text(
        json.dumps(linear_result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    from src.baselines.hipporag import HippoRAG
    hippo = HippoRAG(
        llm_model=llm_model, extraction_model="qwen3.8-flash",
        retrieval_top_k=5, max_workers=4,
        working_dir=output_dir, experiment_id="hipporag",
        case_observation=observation,
    )
    hippo.index(passages)
    hippo_prediction = hippo.qa([question])[0]

    result = dict(
        experiment_id=args.experiment_id, dataset=DATASET, question=question,
        generation_model=args.llm_model, embedding_model="all-mpnet-base-v2",
        baseline="HippoRAG 2024 v1.0.0; not HippoRAG2",
        linearrag=linear_result, hipporag=hippo_prediction,
    )
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(dict(
        output=str(output_dir / "result.json"),
        linearrag_answer=prediction["pred_answer"],
        hipporag_answer=hippo_prediction["pred_answer"],
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
