"""Run the first-stage Q3 ablation design check on fixed 2Wiki questions.

The experiment compares the original BFS retrieval path with two single-module
ablations: seed entities only (no semantic-bridging activation), and direct
passage-weight ranking (no personalized PageRank aggregation).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LinearRAGConfig
from src.evaluate import Evaluator
from src.LinearRAG import LinearRAG
from src.paths import CACHE_DIR, DATASETS_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.utils import LLM_Model, setup_logging


DATASET_NAME = "2wikimultihop"
DATASET_PARAMETERS = {
    "spacy_model": "en_core_web_trf",
    "max_iterations": 3,
    "passage_ratio": 0.05,
    "iteration_threshold": 0.4,
    "top_k_sentence": 1,
}
VARIANTS = (
    "full",
    "without_entity_activation",
    "without_global_importance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-stage LinearRAG Q3 ablation check")
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--llm-model", default="qwen3.8-flash")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--cache-root", type=Path, default=CACHE_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Validate the three retrieval paths without generation or LLM evaluation.",
    )
    args = parser.parse_args()
    if args.max_questions <= 0:
        parser.error("--max-questions must be greater than 0")
    return args


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")

# 固定实验输入
def load_inputs(max_questions: int, seed: int) -> tuple[list[dict], list[str], list[str]]:
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

# 实验适配层，允许我们选择不同的变体而不修改原始LinearRAG类
class AblationLinearRAG(LinearRAG):
    """Expose the two paper-defined Q3 ablations without changing src/LinearRAG.py."""

    def __init__(self, global_config: LinearRAGConfig, variant: str):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown Q3 variant: {variant}")
        self.variant = variant
        self.diagnostics: list[dict] = []
        self.ppr_calls = 0
        super().__init__(global_config)

    # 实现“去掉实体激活”, 关闭语义桥接
    def _seed_entities_only(
        self,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        """Keep query seed entities but suppress entity-to-sentence-to-entity propagation."""
        active_entities = {}
        entity_weights = np.zeros(len(self.graph.vs["name"]))
        for entity_index, entity, entity_hash_id, score in zip(
            seed_entity_indices,
            seed_entities,
            seed_entity_hash_ids,
            seed_entity_scores,
        ):
            active_entities[entity_hash_id] = (entity_index, score, 1)
            entity_weights[self.node_name_to_vertex_idx[entity_hash_id]] = score
        return entity_weights, active_entities

    # 去掉全局重要性聚合
    def _rank_initial_passage_weights(self, passage_weights):
        """Rank passage nodes directly, omitting personalized PageRank."""
        scores = np.array([passage_weights[index] for index in self.passage_node_indices])
        order = np.argsort(scores)[::-1]
        passage_hash_ids = [
            self.vertex_idx_to_node_name[self.passage_node_indices[index]] for index in order
        ]
        return passage_hash_ids, scores[order].tolist()

    def run_ppr(self, node_weights):
        self.ppr_calls += 1
        return super().run_ppr(node_weights)

    """它是三条实验路径的分岔点：
       判断当前运行哪个变体；
       选择是否执行实体传播；
       选择是否执行 PPR；
       记录种子实体、传播实体和 PPR 使用情况。"""
    def graph_search_with_seed_entities(
        self,
        question,
        question_embedding,
        seed_entity_indices,
        seed_entities,
        seed_entity_hash_ids,
        seed_entity_scores,
    ):
        if self.variant == "without_entity_activation":
            entity_weights, active_entities = self._seed_entities_only(
                seed_entity_indices,
                seed_entities,
                seed_entity_hash_ids,
                seed_entity_scores,
            )
        else:
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
        if self.variant == "without_global_importance":
            passage_ids, passage_scores = self._rank_initial_passage_weights(passage_weights)
            used_ppr = False
        else:
            passage_ids, passage_scores = self.run_ppr(entity_weights + passage_weights)
            used_ppr = True

        seed_hash_ids = set(seed_entity_hash_ids)
        active_hash_ids = set(active_entities)
        self.diagnostics.append({
            "question": question,
            "seed_entity_count": len(seed_hash_ids),
            "active_entity_count": len(active_hash_ids),
            "propagated_entity_count": len(active_hash_ids - seed_hash_ids),
            "used_ppr": used_ppr,
        })
        return passage_ids, passage_scores

# 检查实验是否按设计执行, 即"我们拆的零件对不对"
def validate_retrieval(
    variant: str,
    results: list[dict],
    diagnostics: list[dict],
    ppr_calls: int,
    expected_count: int,
) -> dict:
    if len(results) != expected_count:
        raise AssertionError(f"{variant}: expected {expected_count} results, got {len(results)}")
    for index, result in enumerate(results):
        if len(result["sorted_passage"]) != 5 or len(result["sorted_passage_scores"]) != 5:
            raise AssertionError(f"{variant}: question {index} did not return five passages/scores")

    graph_queries = len(diagnostics)
    dense_fallback_queries = expected_count - graph_queries
    if variant == "without_entity_activation":
        if any(item["propagated_entity_count"] != 0 for item in diagnostics):
            raise AssertionError("Entity-activation ablation produced propagated entities")
        if ppr_calls != graph_queries:
            raise AssertionError("Entity-activation ablation did not preserve PPR")
    elif variant == "without_global_importance":
        if ppr_calls != 0 or any(item["used_ppr"] for item in diagnostics):
            raise AssertionError("Global-importance ablation invoked PPR")
    elif ppr_calls != graph_queries:
        raise AssertionError("Full variant did not invoke PPR for every graph query")

    return {
        "status": "passed",
        "result_count": len(results),
        "passages_per_result": 5,
        "graph_query_count": graph_queries,
        "dense_fallback_query_count": dense_fallback_queries,
        "ppr_call_count": ppr_calls,
        "propagated_entity_counts": [
            item["propagated_entity_count"] for item in diagnostics
        ],
    }


def evaluate_predictions(
    predictions: list[dict],
    output_dir: Path,
    llm_model: LLM_Model,
    max_workers: int,
) -> dict:
    predictions_path = output_dir / "predictions.json"
    write_json(predictions_path, predictions)
    evaluator = Evaluator(llm_model=llm_model, predictions_path=str(predictions_path))
    llm_accuracy, contain_accuracy = evaluator.evaluate(max_workers=max_workers)
    return {
        "llm_accuracy": llm_accuracy,
        "contain_accuracy": contain_accuracy,
        "average_accuracy": (llm_accuracy + contain_accuracy) / 2,
        "sample_count": len(predictions),
    }

""" 两层观察：
1. 最终答案准确率有没有下降；
2. 即使答案没下降，底层 Top-5 检索是否已经变化。"""
def compare_retrievals(full_predictions: list[dict], ablated_predictions: list[dict]) -> dict:
    per_question = []
    for full, ablated in zip(full_predictions, ablated_predictions):
        full_passages = full["sorted_passage"]
        ablated_passages = ablated["sorted_passage"]
        overlap_count = len(set(full_passages) & set(ablated_passages))
        per_question.append({
            "id": full.get("id"),
            "top5_overlap_count": overlap_count,
            "top5_overlap_ratio": overlap_count / 5,
            "same_ranked_top5": full_passages == ablated_passages,
            "same_generated_answer": full.get("pred_answer") == ablated.get("pred_answer"),
        })
    return {
        "mean_top5_overlap_ratio": float(np.mean([
            item["top5_overlap_ratio"] for item in per_question
        ])),
        "questions_with_changed_top5_set": sum(
            item["top5_overlap_count"] < 5 for item in per_question
        ),
        "questions_with_changed_top5_ranking": sum(
            not item["same_ranked_top5"] for item in per_question
        ),
        "questions_with_changed_generated_answer": sum(
            not item["same_generated_answer"] for item in per_question
        ),
        "per_question": per_question,
    }

def build_summary(
    variant_metrics: dict[str, dict],
    variant_predictions: dict[str, list[dict]],
    dataset_name: str = DATASET_NAME,
) -> dict:
    full = variant_metrics["full"]
    primary_metric = "llm_accuracy" if dataset_name == "medical" else "average_accuracy"
    rows = []
    for variant in VARIANTS:
        metrics = variant_metrics[variant]
        rows.append({
            "variant": variant,
            **metrics,
            "primary_metric": primary_metric,
            "primary_accuracy": metrics[primary_metric],
            "full_minus_variant": {
                metric: full[metric] - metrics[metric]
                for metric in ("llm_accuracy", "contain_accuracy", "average_accuracy")
            },
        })
    retrieval_comparisons = {
        variant: compare_retrievals(
            variant_predictions["full"], variant_predictions[variant]
        )
        for variant in VARIANTS
        if variant != "full"
    }
    return {
        "dataset": dataset_name,
        "variants": rows,
        "retrieval_comparisons_against_full": retrieval_comparisons,
    }


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q3_ablation_study" / experiment_id
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    setup_logging(str(output_dir / "experiment.log"))

    questions, passages, sample_ids = load_inputs(args.max_questions, args.seed)
    manifest = {
        "research_question": "Q3 Ablation Study",
        "scope": "First-stage three-variant design check",
        "experiment_id": experiment_id,
        "dataset": DATASET_NAME,
        "passage_count": len(passages),
        "question_count": len(questions),
        "sample_ids": sample_ids,
        "seed": args.seed,
        "embedding_model": str(args.embedding_model.resolve()),
        "llm_model": None if args.retrieval_only else args.llm_model,
        "retrieval_top_k": 5,
        "retrieval_path": "official BFS",
        "cache_root": str(args.cache_root.resolve()),
        "dataset_parameters": DATASET_PARAMETERS,
        "variants": {
            "full": "Original entity propagation plus personalized PageRank",
            "without_entity_activation": "Seed entities only; personalized PageRank retained",
            "without_global_importance": "Entity propagation retained; passages ranked by initial passage weights without PPR",
        },
        "operational_interpretation": (
            "The paper does not publish ablation code. Direct ranking of the existing "
            "calculate_passage_scores output is used to isolate removal of PPR."
        ),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "samples.json", {"sample_ids": sample_ids, "questions": questions})

    embedding_model = SentenceTransformer(str(args.embedding_model), device="cuda")
    llm_model = None
    if not args.retrieval_only:
        llm_model = LLM_Model(args.llm_model)

    variant_metrics = {}
    variant_predictions = {}
    validations = {}
    for variant in VARIANTS:
        variant_dir = output_dir / variant
        validation_path = variant_dir / "validation.json"
        evaluation_path = variant_dir / "evaluation_results.json"
        retrieval_path = variant_dir / "retrieval_results.json"
        completed = validation_path.exists() and (
            retrieval_path.exists() if args.retrieval_only else evaluation_path.exists()
        )
        if args.resume and completed:
            print(f"[resume] completed variant skipped: {variant}", flush=True)
            with validation_path.open(encoding="utf-8") as stream:
                validations[variant] = json.load(stream)
            if not args.retrieval_only:
                with evaluation_path.open(encoding="utf-8") as stream:
                    metrics = json.load(stream)
                metrics["average_accuracy"] = (
                    metrics["llm_accuracy"] + metrics["contain_accuracy"]
                ) / 2
                metrics["sample_count"] = len(questions)
                variant_metrics[variant] = metrics
                with (variant_dir / "predictions.json").open(encoding="utf-8") as stream:
                    variant_predictions[variant] = json.load(stream)
            continue

        print(f"[start] {variant}", flush=True)
        config = LinearRAGConfig(
            dataset_name=DATASET_NAME,
            embedding_model=embedding_model,
            llm_model=llm_model,
            working_dir=args.cache_root,
            max_workers=args.max_workers,
            retrieval_top_k=5,
            use_vectorized_retrieval=False,
            **DATASET_PARAMETERS,
        )
        model = AblationLinearRAG(config, variant)
        model.index(passages)
        if args.retrieval_only:
            predictions = model.retrieve(questions)
        else:
            predictions = model.qa(questions)
        for question_info, prediction in zip(questions, predictions):
            prediction["id"] = question_info.get("id")

        write_json(variant_dir / "retrieval_diagnostics.json", model.diagnostics)
        validations[variant] = validate_retrieval(
            variant,
            predictions,
            model.diagnostics,
            model.ppr_calls,
            len(questions),
        )
        write_json(variant_dir / "validation.json", validations[variant])
        if args.retrieval_only:
            write_json(variant_dir / "retrieval_results.json", predictions)
        else:
            variant_metrics[variant] = evaluate_predictions(
                predictions, variant_dir, llm_model, args.max_workers
            )
            variant_predictions[variant] = predictions
        print(f"[done] {variant}", flush=True)
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    result = {"status": "passed", "validations": validations}
    if variant_metrics:
        result["evaluation"] = build_summary(variant_metrics, variant_predictions)
    write_json(output_dir / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"Evidence written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
