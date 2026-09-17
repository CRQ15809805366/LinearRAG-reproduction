"""通过官方 LinearRAG 核心执行可观察且不依赖 OpenAI 的冒烟测试。

本运行器调用真实的索引和检索方法。BFS 重放只用于观测：程序会断言
重放权重与官方实现返回的权重相等，重放结果不会参与最终排序。向量化
分支直接调用官方稀疏矩阵实现，冒烟层只采集设备和结果。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import spacy
from sentence_transformers import SentenceTransformer

from src.LinearRAG import LinearRAG
from src.config import LinearRAGConfig
from src.paths import CACHE_DIR, EXAMPLES_DIR, MODELS_DIR, PROJECT_ROOT, SMOKE_OUTPUT_DIR


DEFAULT_INPUT = EXAMPLES_DIR / "smoke" / "input.json"
ORIGINAL_INPUT = EXAMPLES_DIR / "smoke" / "original_input.json"
DEFAULT_OUTPUT = SMOKE_OUTPUT_DIR / "smoke_result.json"


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return value


def _gpu_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        line = subprocess.check_output(command, text=True, timeout=10).strip().splitlines()[0]
        name, total, used, free, driver = [part.strip() for part in line.split(",")]
        return {
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "memory_free_mib": int(free),
            "driver": driver,
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        return {"available": False}


def _ner_snapshot(nlp: Any, payload: dict[str, Any]) -> dict[str, Any]:
    passages = []
    for index, passage in enumerate(payload["passages"]):
        indexed = f"{index}:{passage}"
        doc = nlp(indexed)
        passages.append(
            {
                "id": index,
                "text": passage,
                "indexed_text": indexed,
                "sentences": [sentence.text for sentence in doc.sents],
                "entities": [
                    {"text": entity.text, "label": entity.label_}
                    for entity in doc.ents
                    if entity.label_ not in {"ORDINAL", "CARDINAL"}
                ],
            }
        )
    question_doc = nlp(payload["question"])
    return {
        "passages": passages,
        "question": payload["question"],
        "question_entities": [
            {"text": entity.text, "label": entity.label_}
            for entity in question_doc.ents
            if entity.label_ not in {"ORDINAL", "CARDINAL"}
        ],
    }


class ObservableLinearRAG(LinearRAG):
    """在官方方法外围采集轨迹，不介入方法内部计算。"""

    def __init__(self, global_config: LinearRAGConfig):
        self.trace: dict[str, Any] = {}
        super().__init__(global_config)

    def get_seed_entities(self, question: str):
        result = super().get_seed_entities(question)
        indices, texts, hash_ids, scores = result
        query_entities = sorted(self.spacy_ner.question_ner(question))
        self.trace["query_entity_matching"] = [
            {
                "query_entity": query_entity,
                "graph_entity": graph_entity,
                "graph_entity_hash_id": hash_id,
                "embedding_index": index,
                "cosine_similarity": float(score),
            }
            for query_entity, graph_entity, hash_id, index, score in zip(
                query_entities, texts, hash_ids, indices, scores
            )
        ]
        return result

    def calculate_entity_scores(
        self,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        official_weights, official_active = super().calculate_entity_scores(
            question_embedding,
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
        )
        replay_weights, replay_active, rounds = self._replay_bfs_trace(
            question_embedding,
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
        )
        if not np.allclose(official_weights, replay_weights, rtol=1e-6, atol=1e-7):
            raise AssertionError("Observation replay diverged from official BFS entity weights")
        official_scores = {key: float(value[1]) for key, value in official_active.items()}
        replay_scores = {key: float(value[1]) for key, value in replay_active.items()}
        if official_scores.keys() != replay_scores.keys() or any(
            not np.isclose(official_scores[key], replay_scores[key], rtol=1e-6, atol=1e-7)
            for key in official_scores
        ):
            raise AssertionError("Observation replay diverged from official active entities")
        self.trace["bfs_rounds"] = rounds
        self.trace["active_entities"] = [
            {
                "entity": self.entity_embedding_store.hash_id_to_text[hash_id],
                "score": float(values[1]),
                "tier": int(values[2]),
            }
            for hash_id, values in official_active.items()
        ]
        self.trace["bfs_replay_matches_official"] = True
        return official_weights, official_active

    def calculate_entity_scores_vectorized(
        self,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        official_weights, official_active = super().calculate_entity_scores_vectorized(
            question_embedding,
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
        )
        self.trace["active_entities"] = [
            {
                "entity": self.entity_embedding_store.hash_id_to_text[hash_id],
                "score": float(values[1]),
                "tier": int(values[2]),
            }
            for hash_id, values in official_active.items()
        ]
        self.trace["vectorized_sparse_devices"] = {
            "entity_to_sentence": str(self.entity_to_sentence_sparse.device),
            "sentence_to_entity": str(self.sentence_to_entity_sparse.device),
        }
        return official_weights, official_active

    def _replay_bfs_trace(
        self,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        active = {}
        weights = np.zeros(len(self.graph.vs["name"]))
        for entity_index, entity_hash, score in zip(
            seed_entity_indices, seed_entity_hash_ids, seed_entity_scores
        ):
            active[entity_hash] = (entity_index, score, 1)
            weights[self.node_name_to_vertex_idx[entity_hash]] = score
        used_sentences = set()
        current = active.copy()
        iteration = 1
        rounds = []
        while current and iteration < self.config.max_iterations:
            round_trace = {"iteration": iteration, "processed_entities": []}
            new_entities = {}
            for entity_hash, (_, entity_score, tier) in current.items():
                entity_trace = {
                    "entity": self.entity_embedding_store.hash_id_to_text[entity_hash],
                    "incoming_score": float(entity_score),
                    "tier": int(tier),
                    "selected_sentences": [],
                }
                if entity_score < self.config.iteration_threshold:
                    entity_trace["stopped"] = "below_iteration_threshold"
                    round_trace["processed_entities"].append(entity_trace)
                    continue
                sentence_hashes = [
                    sentence_hash
                    for sentence_hash in self.entity_hash_id_to_sentence_hash_ids[entity_hash]
                    if sentence_hash not in used_sentences
                ]
                if not sentence_hashes:
                    entity_trace["stopped"] = "no_unused_sentences"
                    round_trace["processed_entities"].append(entity_trace)
                    continue
                sentence_indices = [
                    self.sentence_embedding_store.hash_id_to_idx[sentence_hash]
                    for sentence_hash in sentence_hashes
                ]
                sentence_embeddings = self.sentence_embeddings[sentence_indices]
                question_column = (
                    question_embedding.reshape(-1, 1)
                    if len(question_embedding.shape) == 1
                    else question_embedding
                )
                similarities = np.dot(sentence_embeddings, question_column).flatten()
                top_indices = np.argsort(similarities)[::-1][: self.config.top_k_sentence]
                for top_index in top_indices:
                    sentence_hash = sentence_hashes[top_index]
                    sentence_score = similarities[top_index]
                    used_sentences.add(sentence_hash)
                    discoveries = []
                    for next_hash in self.sentence_hash_id_to_entity_hash_ids[sentence_hash]:
                        next_score = entity_score * sentence_score
                        accepted = bool(next_score >= self.config.iteration_threshold)
                        discoveries.append(
                            {
                                "entity": self.entity_embedding_store.hash_id_to_text[next_hash],
                                "score": float(next_score),
                                "accepted": accepted,
                            }
                        )
                        if not accepted:
                            continue
                        node_index = self.node_name_to_vertex_idx[next_hash]
                        weights[node_index] += next_score
                        new_entities[next_hash] = (node_index, next_score, iteration + 1)
                    entity_trace["selected_sentences"].append(
                        {
                            "sentence": self.sentence_embedding_store.hash_id_to_text[sentence_hash],
                            "question_similarity": float(sentence_score),
                            "discovered_entities": discoveries,
                        }
                    )
                round_trace["processed_entities"].append(entity_trace)
            active.update(new_entities)
            round_trace["new_entities"] = [
                {
                    "entity": self.entity_embedding_store.hash_id_to_text[entity_hash],
                    "score": float(values[1]),
                }
                for entity_hash, values in new_entities.items()
            ]
            rounds.append(round_trace)
            current = new_entities.copy()
            iteration += 1
        return weights, active, rounds

    def calculate_passage_scores(self, question, question_embedding, active_entities):
        weights = super().calculate_passage_scores(question, question_embedding, active_entities)
        self.trace["initial_passage_scores"] = [
            {
                "passage": self.passage_embedding_store.hash_id_to_text[hash_id],
                "score_after_passage_node_weight": float(weights[self.node_name_to_vertex_idx[hash_id]]),
            }
            for hash_id in self.passage_embedding_store.hash_ids
        ]
        return weights

    def run_ppr(self, node_weights):
        hash_ids, scores = super().run_ppr(node_weights)
        self.trace["ppr_passage_scores"] = [
            {
                "rank": rank,
                "passage": self.passage_embedding_store.hash_id_to_text[hash_id],
                "score": float(score),
            }
            for rank, (hash_id, score) in enumerate(zip(hash_ids, scores), start=1)
        ]
        return hash_ids, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--spacy-model", default="en_core_web_trf")
    parser.add_argument("--use-vectorized-retrieval", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    process = psutil.Process()
    rss_before = process.memory_info().rss
    gpu_before = _gpu_snapshot()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    original_payload = json.loads(ORIGINAL_INPUT.read_text(encoding="utf-8"))

    nlp = spacy.load(args.spacy_model)
    original_ner = _ner_snapshot(nlp, original_payload)
    adjusted_ner = _ner_snapshot(nlp, payload)
    del nlp

    model = SentenceTransformer(str(args.embedding_model), device="cuda")
    config = LinearRAGConfig(
        dataset_name="smoke_gpu",
        embedding_model=model,
        llm_model=None,
        spacy_model=args.spacy_model,
        working_dir=str(CACHE_DIR),
        batch_size=2,
        max_workers=1,
        retrieval_top_k=3,
        max_iterations=3,
        top_k_sentence=1,
        passage_ratio=2,
        passage_node_weight=0.05,
        damping=0.5,
        # MPNet 在这个极小样本上的第二跳句子相似度约为 0.176；
        # 0.05 是本冒烟测试专用阈值，用于稳定展示预期的 Germany 传播。
        iteration_threshold=0.05,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
    )
    rag = ObservableLinearRAG(config)
    indexed_passages = [f"{index}:{text}" for index, text in enumerate(payload["passages"])]
    rag.index(indexed_passages)

    ner_results = json.loads((CACHE_DIR / "smoke_gpu" / "ner_results.json").read_text())
    retrieval = rag.retrieve(
        [{"question": payload["question"], "answer": payload["expected_answer"]}]
    )[0]

    passage_count = len(rag.passage_embedding_store.hash_ids)
    sentence_count = len(rag.sentence_embedding_store.hash_ids)
    entity_count = len(rag.entity_embedding_store.hash_ids)
    edge_type_counts = {"passage_entity": 0, "passage_passage": 0, "other": 0}
    for edge in rag.graph.es:
        source = rag.graph.vs[edge.source]["name"]
        target = rag.graph.vs[edge.target]["name"]
        prefixes = {source.split("-", 1)[0], target.split("-", 1)[0]}
        if prefixes == {"passage", "entity"}:
            edge_type_counts["passage_entity"] += 1
        elif prefixes == {"passage"}:
            edge_type_counts["passage_passage"] += 1
        else:
            edge_type_counts["other"] += 1
    graph_stats = {
        "passage_nodes": passage_count,
        "sentence_nodes_in_embedding_store": sentence_count,
        "entity_nodes": entity_count,
        "actual_igraph_vertices": rag.graph.vcount(),
        "actual_igraph_edges": rag.graph.ecount(),
        "actual_igraph_edge_types": edge_type_counts,
        "entity_sentence_links": sum(
            len(sentence_hashes)
            for sentence_hashes in rag.entity_hash_id_to_sentence_hash_ids.values()
        ),
        "note": "Official igraph contains passage and entity vertices; sentences are embedding-store nodes used by BFS, not igraph vertices.",
    }

    ranked_ids = [int(item.split(":", 1)[0]) for item in retrieval["sorted_passage"]]
    active_names = {item["entity"] for item in rag.trace["active_entities"]}
    seed_names = {
        item["graph_entity"] for item in rag.trace["query_entity_matching"]
    }
    expectation = {
        "passages_1_and_2_before_passage_3": ranked_ids.index(0) < ranked_ids.index(2)
        and ranked_ids.index(1) < ranked_ids.index(2),
        "query_seed_contains_beatrice": any("Beatrice" in name for name in seed_names),
        "propagation_contains_frederick_barbarossa": "Frederick Barbarossa" in active_names,
        "propagation_contains_germany": "Germany" in active_names,
    }
    expectation["all_passed"] = all(expectation.values())

    elapsed = time.perf_counter() - started
    rss_after = process.memory_info().rss
    memory_info = process.memory_info()
    report = {
        "status": "passed" if expectation["all_passed"] else "failed",
        "working_directory": str(PROJECT_ROOT),
        "execution": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "spacy_model": args.spacy_model,
            "embedding_model": str(args.embedding_model),
            "embedding_device": str(model.device),
            "retrieval_method": (
                "vectorized matrix-based" if args.use_vectorized_retrieval else "official BFS iteration"
            ),
            "openai_used": False,
            "elapsed_seconds": elapsed,
            "process_rss_before_mib": rss_before / (1024**2),
            "process_rss_after_mib": rss_after / (1024**2),
            "process_peak_working_set_mib": (
                getattr(memory_info, "peak_wset", 0) / (1024**2)
                if getattr(memory_info, "peak_wset", None) is not None
                else None
            ),
            "gpu_before": gpu_before,
            "gpu_after": _gpu_snapshot(),
        },
        "original_input": original_payload,
        "original_ner_observation": original_ner,
        "adjusted_input": payload,
        "adjusted_ner": adjusted_ner,
        "official_saved_ner_results": ner_results,
        "graph_statistics": graph_stats,
        "trace": rag.trace,
        "top_k": [
            {"rank": rank, "passage": passage, "score": float(score)}
            for rank, (passage, score) in enumerate(
                zip(retrieval["sorted_passage"], retrieval["sorted_passage_scores"]), start=1
            )
        ],
        "human_expectation": expectation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_value),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_value))
    return 0 if expectation["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
