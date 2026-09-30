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
from experiments.q3_ablation_study.formal_ablation import (
    AblationLinearRAG,
    VARIANTS,
    build_summary,
    evaluate_predictions,
    validate_retrieval,
    write_json,
)

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
