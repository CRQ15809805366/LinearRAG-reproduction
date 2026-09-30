"""Run the bounded Q1 generation-accuracy reproduction.

This is deliberately a single experiment entry point: it fixes one question
sample, runs Vanilla RAG and the original LinearRAG on that sample, evaluates
both outputs, and writes the comparison under data/output/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LinearRAGConfig
from src.evaluate import Evaluator
from src.LinearRAG import LinearRAG
from src.paths import DATASETS_DIR, MODELS_DIR, EXPERIMENT_RESULTS_DIR
from src.utils import LLM_Model, setup_logging
from src.vanilla_rag import VanillaRAG


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
    parser = argparse.ArgumentParser(
        description="Bounded Q1 comparison: Vanilla RAG versus LinearRAG."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS),
        default=list(DATASET_CONFIGS),
    )
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write and validate the fixed sample manifest without loading models.",
    )
    args = parser.parse_args()
    if args.max_questions <= 0:
        parser.error("--max-questions must be greater than 0")
    return args


def load_dataset(dataset_name: str) -> tuple[list[dict], list[str]]:
    dataset_dir = DATASETS_DIR / dataset_name
    with (dataset_dir / "questions.json").open(encoding="utf-8") as stream:
        questions = json.load(stream)
    with (dataset_dir / "chunks.json").open(encoding="utf-8") as stream:
        chunks = json.load(stream)
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]
    return questions, passages


def fixed_sample(questions: list[dict], size: int, seed: int, dataset_name: str) -> list[dict]:
    if size > len(questions):
        raise ValueError(
            f"{dataset_name} has {len(questions)} questions, fewer than requested {size}."
        )
    # Dataset-specific text prevents otherwise identical RNG sequences across datasets.
    rng = random.Random(f"{seed}:{dataset_name}")
    selected_indices = sorted(rng.sample(range(len(questions)), size))
    return [questions[index] for index in selected_indices]


def question_identifier(question: dict, fallback_index: int) -> str:
    return str(question.get("id", fallback_index))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def run_vanilla_rag(
    dataset_name: str,
    questions: list[dict],
    passages: list[str],
    embedding_model: SentenceTransformer,
    llm_model: LLM_Model,
    max_workers: int,
) -> list[dict]:
    """Run the frozen dense-retrieval baseline with top-k fixed at five."""
    model = VanillaRAG(
        dataset_name=dataset_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
        max_workers=max_workers,
        retrieval_top_k=5,
    )
    model.index(passages)
    return model.qa(questions)


def run_linearrag(
    dataset_name: str,
    questions: list[dict],
    passages: list[str],
    embedding_model: SentenceTransformer,
    llm_model: LLM_Model,
    max_workers: int,
) -> list[dict]:
    params = DATASET_CONFIGS[dataset_name]
    config = LinearRAGConfig(0
        dataset_name=dataset_name,
        embedding_model=embedding_model,
        llm_model=llm_model,
        max_workers=max_workers,
        retrieval_top_k=5,
        use_vectorized_retrieval=False,
        **params,
    )
    model = LinearRAG(global_config=config)
    model.index(passages)
    results = model.qa(questions)
    for question_info, result in zip(questions, results):
        result["id"] = question_info.get("id")
    return results


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
        "sample_count": len(predictions),
    }


def build_comparison(dataset_name: str, vanilla: dict, linearrag: dict) -> dict:
    reported_metrics = ["llm_accuracy"] if dataset_name == "medical" else [
        "contain_accuracy",
        "llm_accuracy",
    ]
    deltas = {
        metric: linearrag[metric] - vanilla[metric]
        for metric in reported_metrics
    }
    return {
        "dataset": dataset_name,
        "sample_count": vanilla["sample_count"],
        "reported_metrics": reported_metrics,
        "vanilla_rag": vanilla,
        "linearrag": linearrag,
        "linearrag_minus_vanilla": deltas,
    }


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q1_generation_accuracy" / experiment_id
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)

    samples = {}
    manifest_datasets = {}
    for dataset_name in args.datasets:
        questions, passages = load_dataset(dataset_name)
        sampled_questions = fixed_sample(
            questions, args.max_questions, args.seed, dataset_name
        )
        samples[dataset_name] = (sampled_questions, passages)
        manifest_datasets[dataset_name] = {
            "available_questions": len(questions),
            "passage_count": len(passages),
            "sample_ids": [
                question_identifier(question, index)
                for index, question in enumerate(sampled_questions)
            ],
            "linearrag_parameters": DATASET_CONFIGS[dataset_name],
        }

    manifest = {
        "research_question": "Q1 Generation Accuracy",
        "scope": "Bounded Vanilla RAG versus original LinearRAG comparison",
        "experiment_id": experiment_id,
        "seed": args.seed,
        "max_questions_per_dataset": args.max_questions,
        "embedding_model": str(args.embedding_model.resolve()),
        "llm_model": args.llm_model,
        "enable_thinking": False if args.llm_model == "qwen3.8-flash" else None,
        "retrieval_top_k": 5,
        "linear_retrieval_path": "official BFS",
        "datasets": manifest_datasets,
    }
    if not args.resume:
        write_json(output_dir / "manifest.json", manifest)

    if args.prepare_only:
        print(f"Prepared Q1 sample manifest: {output_dir / 'manifest.json'}")
        return 0

    setup_logging(str(output_dir / "experiment.log"))
    embedding_model = SentenceTransformer(str(args.embedding_model), device="cuda")
    llm_model = LLM_Model(args.llm_model)

    comparisons = []
    for dataset_name in args.datasets:
        questions, passages = samples[dataset_name]
        dataset_output = output_dir / dataset_name
        comparison_path = dataset_output / "comparison.json"

        if args.resume and comparison_path.exists():
            print(f"[resume] Completed dataset skipped: {dataset_name}", flush=True)
            with comparison_path.open(encoding="utf-8") as stream:
                comparisons.append(json.load(stream))
            continue

        vanilla_dir = dataset_output / "vanilla_rag"
        vanilla_metrics_path = vanilla_dir / "evaluation_results.json"
        if args.resume and vanilla_metrics_path.exists():
            print(f"[resume] Completed Vanilla RAG skipped: {dataset_name}", flush=True)
            with vanilla_metrics_path.open(encoding="utf-8") as stream:
                vanilla_metrics = json.load(stream)
            vanilla_metrics["sample_count"] = len(questions)
        else:
            print(f"[start] Vanilla RAG: {dataset_name}", flush=True)
            vanilla_predictions = run_vanilla_rag(
                dataset_name,
                questions,
                passages,
                embedding_model,
                llm_model,
                args.max_workers,
            )
            vanilla_metrics = evaluate_predictions(
                vanilla_predictions, vanilla_dir, llm_model, args.max_workers
            )
            print(f"[done] Vanilla RAG: {dataset_name}", flush=True)

        linearrag_dir = dataset_output / "linearrag"
        linearrag_metrics_path = linearrag_dir / "evaluation_results.json"
        if args.resume and linearrag_metrics_path.exists():
            print(f"[resume] Completed LinearRAG skipped: {dataset_name}", flush=True)
            with linearrag_metrics_path.open(encoding="utf-8") as stream:
                linearrag_metrics = json.load(stream)
            linearrag_metrics["sample_count"] = len(questions)
        else:
            print(f"[start] LinearRAG: {dataset_name}", flush=True)
            linearrag_predictions = run_linearrag(
                dataset_name,
                questions,
                passages,
                embedding_model,
                llm_model,
                args.max_workers,
            )
            linearrag_metrics = evaluate_predictions(
                linearrag_predictions, linearrag_dir, llm_model, args.max_workers
            )
            print(f"[done] LinearRAG: {dataset_name}", flush=True)

        comparison = build_comparison(dataset_name, vanilla_metrics, linearrag_metrics)
        comparisons.append(comparison)
        write_json(comparison_path, comparison)

    write_json(output_dir / "summary.json", {"datasets": comparisons})
    print(f"Q1 experiment completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
