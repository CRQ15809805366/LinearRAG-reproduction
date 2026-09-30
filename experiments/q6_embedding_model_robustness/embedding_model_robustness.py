"""Run the Q6 sentence-embedding robustness evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import igraph as ig
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.q3_ablation_study.ablation_study import (
    compare_retrievals,
    evaluate_predictions,
    write_json,
)
from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from src.ner import SpacyNER
from src.paths import CACHE_DIR, DATASETS_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.utils import LLM_Model, setup_logging


DATASET_NAME = "2wikimultihop"
MODEL_NAMES = (
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "bge-large-en-v1.5",
    "e5-large-v2",
)
BASELINE_MODEL = "all-mpnet-base-v2"

# 接收实验参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-stage LinearRAG Q6 robustness check")
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--llm-model", default="qwen3.8-flash")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=CACHE_DIR / "q6_embedding_model_robustness",
    )
    parser.add_argument(
        "--shared-ner-cache",
        type=Path,
        default=CACHE_DIR / DATASET_NAME / "ner_results.json",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_questions <= 0:
        parser.error("--max-questions must be greater than 0")
    return args

# 用固定随机种子抽取题，并加载完整语料
def load_inputs(max_questions: int, seed: int):
    """Select one immutable question set shared by all embedding models."""
    dataset_dir = DATASETS_DIR / DATASET_NAME
    with (dataset_dir / "questions.json").open(encoding="utf-8") as stream:
        all_questions = json.load(stream)
    with (dataset_dir / "chunks.json").open(encoding="utf-8") as stream:
        chunks = json.load(stream)
    if max_questions > len(all_questions):
        raise ValueError(f"Requested {max_questions} questions, but only {len(all_questions)} exist")
    rng = random.Random(f"{seed}:{DATASET_NAME}")
    indices = sorted(rng.sample(range(len(all_questions)), max_questions))
    questions = [all_questions[index] for index in indices]
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]
    sample_ids = [str(question.get("id", index)) for index, question in zip(indices, questions)]
    return questions, passages, sample_ids

# 为每个模型建立独立向量缓存
def prepare_model_cache(cache_root: Path, model_name: str, shared_ner_cache: Path) -> Path:
    """Give every model isolated embeddings while reusing model-independent NER output."""
    model_cache = cache_root / model_name
    dataset_cache = model_cache / DATASET_NAME
    dataset_cache.mkdir(parents=True, exist_ok=True)
    target = dataset_cache / "ner_results.json"
    if not target.exists():
        if not shared_ner_cache.exists():
            raise FileNotFoundError(f"Shared NER cache is missing: {shared_ner_cache}")
        shutil.copy2(shared_ner_cache, target)
    return model_cache

# 继承自 LinearRAG，增加观察信息收集功能
class ObservedLinearRAG(LinearRAG):
    """Run the real BFS path and expose only small mechanism-level diagnostics."""

    def __init__(self, global_config: LinearRAGConfig):
        self.diagnostics: list[dict] = []
        self.config = global_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = global_config.dataset_name
        self.llm_model = global_config.llm_model
        self.spacy_ner = SpacyNER(global_config.spacy_model)
        self.graph = ig.Graph(directed=False)
        self.load_embedding_store()

    def graph_search_with_seed_entities(
        self,
        question,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        started = time.perf_counter()
        entity_weights, active_entities = self.calculate_entity_scores(
            question_embedding,
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
        )
        passage_weights = self.calculate_passage_scores(
            question, question_embedding, active_entities
        )
        passage_ids, passage_scores = self.run_ppr(entity_weights + passage_weights)
        seed_ids = set(seed_entity_hash_ids)
        active_ids = set(active_entities)
        self.diagnostics.append({
            "question": question,
            "seed_entities": list(seed_entities),
            "seed_entity_count": len(seed_ids),
            "active_entity_count": len(active_ids),
            "propagated_entity_count": len(active_ids - seed_ids),
            "graph_search_seconds": time.perf_counter() - started,
        })
        return passage_ids, passage_scores

#  特意把“检索”和“生成”拆开
def generate_answers(
    retrieval_results: list[dict],
    questions: list[dict],
    llm_model: LLM_Model,
    max_workers: int,
):
    """Hold retrieval fixed and add answers without re-running the retriever."""
    system_prompt = (
        "As an advanced reading comprehension assistant, your task is to analyze text "
        "passages and corresponding questions meticulously. Your response start after "
        '"Thought: ", where you will methodically break down the reasoning process, '
        'illustrating how you arrive at conclusions. Conclude with "Answer: " to present '
        "a concise, definitive response, devoid of additional elaborations."
    )
    messages = []
    for result in retrieval_results:
        prompt_user = "".join(f"{passage}\n" for passage in result["sorted_passage"])
        prompt_user += f'Question: {result["question"]}\n Thought: '
        messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_user},
        ])

    responses = [None] * len(messages)
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(llm_model.infer, message): index
            for index, message in enumerate(messages)
        }
        with tqdm(total=len(futures), desc="QA Reading (Parallel)") as progress:
            for future in as_completed(futures):
                index = futures[future]
                try:
                    responses[index] = future.result()
                except Exception as exc:
                    responses[index] = "Answer: [generation rejected by model service]"
                    errors.append({
                        "index": index,
                        "id": questions[index].get("id"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                progress.update(1)

    for question, response, result in zip(questions, responses, retrieval_results):
        result["id"] = question.get("id")
        result["pred_answer"] = (
            response.split("Answer:", 1)[1].strip() if "Answer:" in response else response
        )
    return retrieval_results, errors

# 检查实验是否真正完成
def validate_run(predictions: list[dict], diagnostics: list[dict], expected_count: int) -> dict:
    if len(predictions) != expected_count:
        raise AssertionError(f"Expected {expected_count} predictions, got {len(predictions)}")
    if any(len(item["sorted_passage"]) != 5 for item in predictions):
        raise AssertionError("Every prediction must contain five retrieved passages")
    graph_queries = len(diagnostics)
    return {
        "status": "passed",
        "prediction_count": len(predictions),
        "passages_per_prediction": 5,
        "graph_query_count": graph_queries,
        "dense_fallback_query_count": expected_count - graph_queries,
        "mean_active_entity_count": float(np.mean([
            item["active_entity_count"] for item in diagnostics
        ])) if diagnostics else 0.0,
        "mean_propagated_entity_count": float(np.mean([
            item["propagated_entity_count"] for item in diagnostics
        ])) if diagnostics else 0.0,
        "mean_graph_search_seconds": float(np.mean([
            item["graph_search_seconds"] for item in diagnostics
        ])) if diagnostics else 0.0,
    }

# 断点续跑
def load_completed_model(model_dir: Path):
    required = (
        model_dir / "result.json",
        model_dir / "predictions.json",
    )
    if not all(path.exists() for path in required):
        return None
    with required[0].open(encoding="utf-8") as stream:
        row = json.load(stream)
    with required[1].open(encoding="utf-8") as stream:
        predictions = json.load(stream)
    return row, predictions

# 让一个模型完成整场考试
def run_model(
    model_name: str,
    questions: list[dict],
    passages: list[str],
    output_dir: Path,
    args: argparse.Namespace,
    llm_model: LLM_Model,
):
    model_dir = output_dir / "models" / model_name
    if args.resume:
        completed = load_completed_model(model_dir)
        if completed is not None:
            print(f"[resume] {model_name}", flush=True)
            return completed

    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Embedding model is missing: {model_path}")
    model_cache = prepare_model_cache(args.cache_root, model_name, args.shared_ner_cache)
    print(f"[start] {model_name}", flush=True)

    load_started = time.perf_counter()
    embedding_model = SentenceTransformer(str(model_path), device="cuda")
    load_seconds = time.perf_counter() - load_started
    embedding_dimension = embedding_model.get_sentence_embedding_dimension()

    config = LinearRAGConfig(
        dataset_name=DATASET_NAME,
        embedding_model=embedding_model,
        llm_model=llm_model,
        working_dir=model_cache,
        spacy_model="en_core_web_trf",
        max_workers=args.max_workers,
        retrieval_top_k=5,
        use_vectorized_retrieval=False,
        max_iterations=3,
        top_k_sentence=1,
        iteration_threshold=0.4,
        passage_ratio=0.05,
    )
    rag = ObservedLinearRAG(config)
    index_started = time.perf_counter()
    rag.index(passages)
    index_seconds = time.perf_counter() - index_started

    retrieval_started = time.perf_counter()
    retrieval_results = rag.retrieve(questions)
    retrieval_seconds = time.perf_counter() - retrieval_started
    predictions, generation_errors = generate_answers(
        retrieval_results, questions, llm_model, args.max_workers
    )
    write_json(model_dir / "generation_errors.json", generation_errors)
    write_json(model_dir / "retrieval_diagnostics.json", rag.diagnostics)
    validation = validate_run(predictions, rag.diagnostics, len(questions))
    write_json(model_dir / "validation.json", validation)
    metrics = evaluate_predictions(predictions, model_dir, llm_model, args.max_workers)

    row = {
        "model": model_name,
        "model_path": str(model_path.resolve()),
        "embedding_dimension": embedding_dimension,
        "load_seconds": load_seconds,
        "index_seconds": index_seconds,
        "retrieval_seconds_total": retrieval_seconds,
        "retrieval_seconds_mean": retrieval_seconds / len(questions),
        "metrics": metrics,
        "validation": validation,
        "generation_error_count": len(generation_errors),
    }
    write_json(model_dir / "result.json", row)
    print(f"[done] {model_name}: mean={metrics['average_accuracy']:.3f}", flush=True)

    del rag
    del embedding_model
    gc.collect()
    torch.cuda.empty_cache()
    return row, predictions


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q6_embedding_model_robustness" / experiment_id
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    setup_logging(str(output_dir / "experiment.log"))

    questions, passages, sample_ids = load_inputs(args.max_questions, args.seed)
    manifest = {
        "research_question": "Q6 Effectiveness of Different Sentence Embeddings",
        "scope": "One-tenth-paper-scale four-model evaluation",
        "experiment_id": experiment_id,
        "dataset": DATASET_NAME,
        "question_count": len(questions),
        "passage_count": len(passages),
        "sample_ids": sample_ids,
        "seed": args.seed,
        "embedding_models": MODEL_NAMES,
        "baseline_model": BASELINE_MODEL,
        "llm_model": args.llm_model,
        "retrieval_path": "official BFS",
        "parameters": {
            "retrieval_top_k": 5,
            "max_iterations": 3,
            "top_k_sentence": 1,
            "iteration_threshold": 0.4,
            "passage_ratio": 0.05,
        },
        "cache_root": str(args.cache_root.resolve()),
        "cache_policy": "isolated passage/entity/sentence embeddings per model; shared NER only",
    }
    if not args.resume:
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "samples.json", {"sample_ids": sample_ids, "questions": questions})

    llm_model = LLM_Model(args.llm_model)

    rows = []
    predictions_by_model = {}
    for model_name in MODEL_NAMES:
        row, predictions = run_model(
            model_name, questions, passages, output_dir, args, llm_model
        )
        rows.append(row)
        predictions_by_model[model_name] = predictions

    baseline_predictions = predictions_by_model[BASELINE_MODEL]
    for row in rows:
        row["retrieval_comparison_with_baseline"] = compare_retrievals(
            baseline_predictions, predictions_by_model[row["model"]]
        )
    summary = {
        "status": "passed",
        "primary_metric": "average_accuracy",
        "primary_metric_definition": "(llm_accuracy + contain_accuracy) / 2",
        "baseline_model": BASELINE_MODEL,
        "models": rows,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Evidence written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
