"""Reproduce the single LinearRAG case shown in paper Table 7 (Q8)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from src.paths import CACHE_DIR, DATASETS_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.utils import LLM_Model


DATASET = "2wikimultihop"
QUESTION_ID = "3a3c2efe0bdc11eba7f7acde48001122"
EXPERIMENT_ID = "q8-paper-case-20260921"

# 没有改算法, 只是周围加了一层简单记录，把每轮选中了什么句子、激活了什么实体保存下来。
class CaseStudyLinearRAG(LinearRAG):
    """Add a small observation window to the original BFS propagation."""

    def __init__(self, global_config: LinearRAGConfig):
        self.trace: list[dict] = []
        super().__init__(global_config)

    def restore_index_read_only(self) -> None:
        """Reconstruct the in-memory graph from existing cache without writing it."""
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
        """Run the original BFS calculation while retaining its selected path."""
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


def main() -> int:
    questions = json.loads(
        (DATASETS_DIR / DATASET / "questions.json").read_text(encoding="utf-8")
    )
    question = next(item for item in questions if item.get("id") == QUESTION_ID)


    embedding_model = SentenceTransformer(str(MODELS_DIR / "all-mpnet-base-v2"), device="cuda")
    llm_model = LLM_Model("qwen3.8-flash")
    config = LinearRAGConfig(
        dataset_name=DATASET,
        embedding_model=embedding_model,
        llm_model=llm_model,
        spacy_model="en_core_web_trf",
        working_dir=CACHE_DIR,
        max_workers=1,
        max_iterations=3,
        iteration_threshold=0.4,
        passage_ratio=0.05,
        top_k_sentence=1,
        use_vectorized_retrieval=False,
    )
    rag = CaseStudyLinearRAG(config)
    rag.restore_index_read_only()
    prediction = rag.qa([question])[0]
    prediction["id"] = question["id"]

    output_dir = EXPERIMENT_RESULTS_DIR / "q8_case_study" / EXPERIMENT_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "dataset": DATASET,
        "model": "qwen3.8-flash",
        "cache_mode": "existing_cache_read_only",
        "parameters": {
            "max_iterations": 3,
            "iteration_threshold": 0.4,
            "passage_ratio": 0.05,
            "top_k_sentence": 1,
            "retrieval_top_k": 5,
            "retrieval_path": "BFS",
        },
        "question": question,
        "trace": rag.trace,
        "prediction": prediction,
        "contains_gold": question["answer"].lower() in prediction["pred_answer"].lower(),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output_dir / "result.json"),
        "prediction": prediction["pred_answer"],
        "contains_gold": result["contains_gold"],
        "trace_steps": len(rag.trace),
        "top5_count": len(prediction["sorted_passage"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
