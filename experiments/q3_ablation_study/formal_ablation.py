"""Run the bounded formal Q3 ablation experiment on four datasets."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LinearRAGConfig
from src.evaluate import Evaluator
from src.LinearRAG import LinearRAG
from src.paths import CACHE_DIR, DATASETS_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.utils import LLM_Model, setup_logging


VARIANTS = (
    "full",
    "without_entity_activation",
    "without_global_importance",
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default)


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


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
    dataset_name: str,
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



DATASET_CONFIGS = {
    "hotpotqa": {
        "spacy_model": "en_core_web_trf",
        "max_iterations": 3,
        "passage_ratio": 0.05,
        "iteration_threshold": 0.4,
        "top_k_sentence": 1,
    },
    "2wikimultihop": {
        "spacy_model": "en_core_web_trf",
        "max_iterations": 3,
        "passage_ratio": 0.05,
        "iteration_threshold": 0.4,
        "top_k_sentence": 1,
    },
    "musique": {
        "spacy_model": "en_core_web_trf",
        "max_iterations": 5,
        "passage_ratio": 2.0,
        "iteration_threshold": 0.1,
        "top_k_sentence": 4,
    },
    "medical": {
        "spacy_model": "en_core_sci_scibert",
        "max_iterations": 3,
        "passage_ratio": 1.5,
        "iteration_threshold": 0.5,
        "top_k_sentence": 1,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded formal LinearRAG Q3 experiment")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_CONFIGS, default=list(DATASET_CONFIGS))
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--llm-model", default="qwen3.8-flash")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--cache-root", type=Path, default=CACHE_DIR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_questions <= 0:
        parser.error("--max-questions must be greater than 0")
    return args


def load_inputs(dataset_name: str, max_questions: int, seed: int):
    dataset_dir = DATASETS_DIR / dataset_name
    with (dataset_dir / "questions.json").open(encoding="utf-8") as stream:
        all_questions = json.load(stream)
    with (dataset_dir / "chunks.json").open(encoding="utf-8") as stream:
        chunks = json.load(stream)
    rng = random.Random(f"{seed}:{dataset_name}")
    indices = sorted(rng.sample(range(len(all_questions)), max_questions))
    questions = [all_questions[index] for index in indices]
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]
    sample_ids = [str(question.get("id", index)) for index, question in zip(indices, questions)]
    return questions, passages, sample_ids


def load_completed_variant(variant_dir: Path, question_count: int):
    required = (
        variant_dir / "validation.json",
        variant_dir / "evaluation_results.json",
        variant_dir / "predictions.json",
    )
    if not all(path.exists() for path in required):
        return None
    with required[0].open(encoding="utf-8") as stream:
        validation = json.load(stream)
    with required[1].open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    with required[2].open(encoding="utf-8") as stream:
        predictions = json.load(stream)
    metrics["average_accuracy"] = (metrics["llm_accuracy"] + metrics["contain_accuracy"]) / 2
    metrics["sample_count"] = question_count
    return validation, metrics, predictions


def qa_with_isolated_failures(model: AblationLinearRAG, questions: list[dict]):
    """Generate answers concurrently without discarding a whole variant on one rejected prompt."""
    retrieval_results = model.retrieve(questions)
    system_prompt = (
        'As an advanced reading comprehension assistant, your task is to analyze text '
        'passages and corresponding questions meticulously. Your response start after '
        '"Thought: ", where you will methodically break down the reasoning process, '
        'illustrating how you arrive at conclusions. Conclude with "Answer: " to present '
        'a concise, definitive response, devoid of additional elaborations.'
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
    with ThreadPoolExecutor(max_workers=model.config.max_workers) as executor:
        futures = {
            executor.submit(model.llm_model.infer, message): index
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

    for response, result in zip(responses, retrieval_results):
        result["pred_answer"] = (
            response.split("Answer:", 1)[1].strip() if "Answer:" in response else response
        )
    return retrieval_results, errors


def main() -> int:
    args = parse_args()
    output_dir = EXPERIMENT_RESULTS_DIR / "q3_ablation_study" / args.experiment_id
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    setup_logging(str(output_dir / "experiment.log"))

    samples = {}
    manifest_datasets = {}
    for dataset_name in args.datasets:
        questions, passages, sample_ids = load_inputs(dataset_name, args.max_questions, args.seed)
        samples[dataset_name] = (questions, passages)
        manifest_datasets[dataset_name] = {
            "question_count": len(questions),
            "passage_count": len(passages),
            "sample_ids": sample_ids,
            "parameters": DATASET_CONFIGS[dataset_name],
        }
    if not args.resume:
        write_json(output_dir / "manifest.json", {
            "research_question": "Q3 Ablation Study",
            "scope": "Bounded formal four-dataset three-variant experiment",
            "experiment_id": args.experiment_id,
            "seed": args.seed,
            "max_questions_per_dataset": args.max_questions,
            "embedding_model": str(args.embedding_model.resolve()),
            "llm_model": args.llm_model,
            "retrieval_top_k": 5,
            "retrieval_path": "official BFS",
            "cache_root": str(args.cache_root.resolve()),
            "variants": list(VARIANTS),
            "datasets": manifest_datasets,
        })

    embedding_model = SentenceTransformer(str(args.embedding_model), device="cuda")
    llm_model = LLM_Model(args.llm_model)

    all_summaries = []
    for dataset_name in args.datasets:
        questions, passages = samples[dataset_name]
        dataset_dir = output_dir / dataset_name
        validations = {}
        metrics = {}
        predictions = {}
        pending = []
        for variant in VARIANTS:
            completed = load_completed_variant(dataset_dir / variant, len(questions))
            if completed:
                validations[variant], metrics[variant], predictions[variant] = completed
                print(f"[resume] {dataset_name}/{variant}", flush=True)
            else:
                pending.append(variant)

        if pending:
            config = LinearRAGConfig(
                dataset_name=dataset_name,
                embedding_model=embedding_model,
                llm_model=llm_model,
                working_dir=args.cache_root,
                max_workers=args.max_workers,
                retrieval_top_k=5,
                use_vectorized_retrieval=False,
                **DATASET_CONFIGS[dataset_name],
            )
            model = AblationLinearRAG(config, pending[0])
            print(f"[index] {dataset_name}: one index load/build for {len(pending)} pending variants", flush=True)
            model.index(passages)

            for variant in pending:
                print(f"[start] {dataset_name}/{variant}", flush=True)
                model.variant = variant
                model.diagnostics = []
                model.ppr_calls = 0
                variant_predictions, generation_errors = qa_with_isolated_failures(model, questions)
                for question_info, prediction in zip(questions, variant_predictions):
                    prediction["id"] = question_info.get("id")
                variant_dir = dataset_dir / variant
                write_json(variant_dir / "generation_errors.json", generation_errors)
                write_json(variant_dir / "retrieval_diagnostics.json", model.diagnostics)
                validations[variant] = validate_retrieval(
                    variant,
                    variant_predictions,
                    model.diagnostics,
                    model.ppr_calls,
                    len(questions),
                )
                write_json(variant_dir / "validation.json", validations[variant])
                metrics[variant] = evaluate_predictions(
                    variant_predictions, variant_dir, llm_model, args.max_workers
                )
                predictions[variant] = variant_predictions
                print(f"[done] {dataset_name}/{variant}", flush=True)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        dataset_summary = {
            "validations": validations,
            "evaluation": build_summary(metrics, predictions, dataset_name),
        }
        write_json(dataset_dir / "summary.json", dataset_summary)
        all_summaries.append(dataset_summary["evaluation"])

    write_json(output_dir / "summary.json", {"status": "passed", "datasets": all_summaries})
    print(f"Q3 formal experiment completed: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
