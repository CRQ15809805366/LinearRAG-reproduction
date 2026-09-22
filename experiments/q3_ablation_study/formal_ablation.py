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
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.q3_ablation_study.ablation_study import (
    AblationLinearRAG,
    VARIANTS,
    build_summary,
    evaluate_predictions,
    validate_retrieval,
    write_json,
)
from src.config import LinearRAGConfig
from src.paths import CACHE_DIR, DATASETS_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.utils import LLM_Model, setup_logging


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
