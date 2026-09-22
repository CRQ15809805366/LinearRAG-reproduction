"""Run the first-stage Q4 retrieval-quality comparison.

The experiment holds the Medical corpus, sampled questions, embedding model,
Top-k, and evaluator fixed while comparing Vanilla RAG with LinearRAG.

The two evaluator prompts are adapted from GraphRAG-Benchmark's MIT-licensed
Evaluation/metrics implementation:
https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
Copyright (c) 2025 XMU-DeepLIT
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from src.paths import CACHE_DIR, DATASETS_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.utils import LLM_Model, setup_logging
from src.vanilla_rag import VanillaRAG


DATASET_NAME = "medical"
QUESTION_TYPES = (
    "Fact Retrieval",
    "Complex Reasoning",
    "Contextual Summarize",
    "Creative Generation",
)
MEDICAL_PARAMETERS = {
    "spacy_model": "en_core_sci_scibert",
    "max_iterations": 3,
    "passage_ratio": 1.5,
    "iteration_threshold": 0.5,
    "top_k_sentence": 1,
}

CONTEXT_RELEVANCE_PROMPT = """### Instructions
Evaluate whether the Context contains proper information to answer the Question.
Use only the Context and Question, not prior knowledge.

Score 0 if no relevant information is present.
Score 1 if relevant information is only partially present.
Score 2 if the information needed to answer is fully present.

Return only JSON in this form: {{"score": 0}}

Question: {question}
Context: {context}
/no_think
"""

EVIDENCE_RECALL_PROMPT = """### Task
For every item in Evidence, decide whether it can be attributed to the Context.
Return only a JSON object with a "classifications" list. Every item must contain
the exact evidence as "statement", a one-sentence "reason", and "attributed": 1
when supported by the Context or 0 otherwise.

Context: {context}
Evidence: {evidence}
Question: {question}

Return JSON only.
/no_think
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="First-stage LinearRAG Q4 retrieval-quality check")
    parser.add_argument("--questions-per-type", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--llm-model", default="qwen3.8-flash")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--cache-root", type=Path, default=CACHE_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Stop after validating and saving both methods' Top-5 retrievals.",
    )
    args = parser.parse_args()
    if args.questions_per_type <= 0:
        parser.error("--questions-per-type must be greater than 0")
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

# 它决定这次考试考哪些题
def load_inputs(questions_per_type: int, seed: int) -> tuple[list[dict], list[str], dict]:
    dataset_dir = DATASETS_DIR / DATASET_NAME
    with (dataset_dir / "questions.json").open(encoding="utf-8") as stream:
        all_questions = json.load(stream)
    with (dataset_dir / "chunks.json").open(encoding="utf-8") as stream:
        chunks = json.load(stream)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for question in all_questions:
        grouped[question["question_type"]].append(question)

    selected = []
    sample_ids_by_type = {}
    for question_type in QUESTION_TYPES:
        available = grouped[question_type]
        if questions_per_type > len(available):
            raise ValueError(
                f"{question_type} has {len(available)} questions, fewer than requested "
                f"{questions_per_type}"
            )
        rng = random.Random(f"{seed}:{DATASET_NAME}:{question_type}")
        indices = sorted(rng.sample(range(len(available)), questions_per_type))
        questions = [available[index] for index in indices]
        selected.extend(questions)
        sample_ids_by_type[question_type] = [question["id"] for question in questions]

    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]
    sample_info = {
        "available_counts": dict(Counter(q["question_type"] for q in all_questions)),
        "sample_ids_by_type": sample_ids_by_type,
    }
    return selected, passages, sample_info

# 把原问题中的这些字段重新装回检索结果, 以便benchmark可以评测
def attach_gold_fields(questions: list[dict], results: list[dict]) -> list[dict]:
    if len(questions) != len(results):
        raise AssertionError("Question and retrieval-result counts differ")
    enriched = []
    for question, result in zip(questions, results):
        enriched.append({
            "id": question["id"],
            "question": question["question"],
            "question_type": question["question_type"],
            "evidence": question["evidence"],
            "gold_answer": question["answer"],
            "context": result["sorted_passage"],
            "retrieval_scores": result["sorted_passage_scores"],
        })
    return enriched

# 我们现在比较的真的是同一场考试吗？
def validate_retrieval(results_by_method: dict[str, list[dict]], expected_count: int) -> dict:
    expected_ids = None
    validations = {}
    for method, results in results_by_method.items():
        if len(results) != expected_count:
            raise AssertionError(f"{method}: expected {expected_count} results, got {len(results)}")
        ids = [item["id"] for item in results]
        if len(ids) != len(set(ids)):
            raise AssertionError(f"{method}: duplicate question IDs")
        if any(len(item["context"]) != 5 for item in results):
            raise AssertionError(f"{method}: at least one question did not return Top-5 context")
        if any(not item["evidence"] for item in results):
            raise AssertionError(f"{method}: at least one question has no gold evidence")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise AssertionError("The two methods did not evaluate identical ordered questions")
        validations[method] = {
            "result_count": len(results),
            "passages_per_result": 5,
            "question_type_counts": dict(Counter(item["question_type"] for item in results)),
        }
    return {"status": "passed", "methods": validations, "paired_question_ids": expected_ids}


def parse_json_object(raw: str) -> dict:
    cleaned = re.sub(r"```(?:json)?|```", "", raw or "").strip()
    candidates = [cleaned]
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return {}


def infer_json(llm_model: LLM_Model, prompt: str, validator, max_retries: int = 2) -> tuple[Any, list[str]]:
    raw_responses = []
    messages = [{"role": "user", "content": prompt}]
    for _ in range(max_retries):
        raw = llm_model.infer(messages)
        raw_responses.append(raw)
        parsed = parse_json_object(raw)
        validated = validator(parsed)
        if validated is not None:
            return validated, raw_responses
    return None, raw_responses


def relevance_rating(value: dict) -> float | None:
    score = value.get("score", value.get("rating"))
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 2 else None


def evidence_classifications(value: dict, reference_evidence: list[str]) -> list[dict] | None:
    items = value.get("classifications")
    if not isinstance(items, list):
        return None
    valid = []
    seen = set()
    for item in items:
        if not isinstance(item, dict) or item.get("attributed") not in (0, 1):
            continue
        if "statement" not in item or "reason" not in item:
            continue
        statement = str(item["statement"])
        if statement not in reference_evidence or statement in seen:
            continue
        seen.add(statement)
        valid.append({
            "statement": statement,
            "reason": str(item["reason"]),
            "attributed": int(item["attributed"]),
        })
    return valid or None


def single_evidence_classification(value: dict, evidence: str) -> list[dict] | None:
    items = value.get("classifications")
    if not isinstance(items, list) or len(items) != 1:
        return None
    item = items[0]
    if not isinstance(item, dict) or item.get("attributed") not in (0, 1):
        return None
    if "reason" not in item:
        return None
    return [{
        "statement": evidence,
        "reason": str(item["reason"]),
        "attributed": int(item["attributed"]),
    }]

"""
这个上下文是否包含回答问题所需的信息？(相关性)
这条 evidence 能否从 Top-5 中得到支持？(召回率)
"""
def evaluate_one(item: dict, llm_model: LLM_Model) -> dict:
    context = "\n".join(item["context"])[:20000]
    relevance_prompt = CONTEXT_RELEVANCE_PROMPT.format(
        question=item["question"], context=context
    )
    ratings = []
    relevance_raw = []
    for _ in range(2):
        rating, raw = infer_json(llm_model, relevance_prompt, relevance_rating)
        relevance_raw.extend(raw)
        if rating is not None:
            ratings.append(rating)

    recall_prompt = EVIDENCE_RECALL_PROMPT.format(
        question=item["question"],
        context=context,
        evidence=json.dumps(item["evidence"], ensure_ascii=False),
    )
    classifications, recall_raw = infer_json(
        llm_model,
        recall_prompt,
        lambda value: evidence_classifications(value, item["evidence"]),
    )
    classifications = classifications or []
    classified_statements = {row["statement"] for row in classifications}
    for evidence in item["evidence"]:
        if evidence in classified_statements:
            continue
        single_prompt = EVIDENCE_RECALL_PROMPT.format(
            question=item["question"],
            context=context,
            evidence=json.dumps([evidence], ensure_ascii=False),
        )
        single, raw = infer_json(
            llm_model,
            single_prompt,
            lambda value, target=evidence: single_evidence_classification(value, target),
        )
        recall_raw.extend(raw)
        if single:
            classifications.extend(single)
            classified_statements.add(evidence)
    return {
        "id": item["id"],
        "question_type": item["question_type"],
        "context_relevance": (
            sum(rating / 2 for rating in ratings) / len(ratings)
            if ratings else math.nan
        ),
        "evidence_recall": (
            sum(row["attributed"] for row in classifications) / len(classifications)
            if classifications else math.nan
        ),
        "reference_evidence_count": len(item["evidence"]),
        "classified_evidence_count": len(classifications),
        "relevance_rating_count": len(ratings),
        "relevance_ratings": ratings,
        "evidence_classifications": classifications,
        "raw_judge_responses": {
            "context_relevance": relevance_raw,
            "evidence_recall": recall_raw,
        },
    }


def mean_valid(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid)) if valid else math.nan

# 扩展到全部问题, 计算四组宏观平均
def evaluate_method(
    results: list[dict],
    llm_model: LLM_Model,
    max_workers: int,
    checkpoint_path: Path,
) -> dict:
    completed_by_id = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as stream:
            completed_by_id = {
                item["id"]: item
                for item in json.load(stream)
                if item is not None
                and not math.isnan(item["context_relevance"])
                and not math.isnan(item["evidence_recall"])
                and item["relevance_rating_count"] == 2
                and item["classified_evidence_count"] == item["reference_evidence_count"]
            }
    detailed = [completed_by_id.get(item["id"]) for item in results]
    pending_indices = [index for index, item in enumerate(detailed) if item is None]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(evaluate_one, results[index], llm_model): index
            for index in pending_indices
        }
        with tqdm(
            total=len(results),
            initial=len(results) - len(pending_indices),
            desc="LLM retrieval evaluation",
        ) as progress:
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                detailed[index] = future.result()
                progress.update(1)
                if progress.n % 5 == 0:
                    write_json(checkpoint_path, [item for item in detailed if item is not None])
    write_json(checkpoint_path, detailed)

    grouped = defaultdict(list)
    for item in detailed:
        grouped[item["question_type"]].append(item)

    by_type = {}
    for question_type in QUESTION_TYPES:
        rows = grouped[question_type]
        by_type[question_type] = {
            "sample_count": len(rows),
            "context_relevance": mean_valid([row["context_relevance"] for row in rows]),
            "evidence_recall": mean_valid([row["evidence_recall"] for row in rows]),
            "invalid_context_relevance_count": sum(
                math.isnan(row["context_relevance"]) for row in rows
            ),
            "invalid_evidence_recall_count": sum(
                math.isnan(row["evidence_recall"]) for row in rows
            ),
            "incomplete_evidence_classification_count": sum(
                row["classified_evidence_count"] != row["reference_evidence_count"]
                for row in rows
            ),
        }

    return {
        "by_question_type": by_type,
        "macro_average": {
            "context_relevance": mean_valid([
                by_type[name]["context_relevance"] for name in QUESTION_TYPES
            ]),
            "evidence_recall": mean_valid([
                by_type[name]["evidence_recall"] for name in QUESTION_TYPES
            ]),
        },
        "detailed": detailed,
    }

# 比较两个总平均值，按照问题 ID 一一对齐
def build_paired_comparison(vanilla: dict, linearrag: dict) -> dict:
    vanilla_by_id = {item["id"]: item for item in vanilla["detailed"]}
    linearrag_by_id = {item["id"]: item for item in linearrag["detailed"]}
    per_question = []
    quadrants = Counter()
    for question_id, vanilla_row in vanilla_by_id.items():
        linear_row = linearrag_by_id[question_id]
        recall_delta = linear_row["evidence_recall"] - vanilla_row["evidence_recall"]
        relevance_delta = linear_row["context_relevance"] - vanilla_row["context_relevance"]
        if math.isnan(recall_delta) or math.isnan(relevance_delta):
            quadrant = "invalid_metric"
        elif recall_delta > 0 and relevance_delta > 0:
            quadrant = "both_improved"
        elif recall_delta > 0 and relevance_delta < 0:
            quadrant = "recall_up_relevance_down"
        elif recall_delta < 0 and relevance_delta > 0:
            quadrant = "recall_down_relevance_up"
        elif recall_delta < 0 and relevance_delta < 0:
            quadrant = "both_declined"
        else:
            quadrant = "at_least_one_tied"
        quadrants[quadrant] += 1
        per_question.append({
            "id": question_id,
            "question_type": vanilla_row["question_type"],
            "evidence_recall_delta": recall_delta,
            "context_relevance_delta": relevance_delta,
            "quadrant": quadrant,
        })

    by_type = {}
    for question_type in QUESTION_TYPES:
        by_type[question_type] = {
            metric: (
                linearrag["by_question_type"][question_type][metric]
                - vanilla["by_question_type"][question_type][metric]
            )
            for metric in ("context_relevance", "evidence_recall")
        }
    return {
        "linearrag_minus_vanilla_by_question_type": by_type,
        "quadrant_counts": dict(quadrants),
        "per_question": per_question,
    }


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q4_retrieval_quality" / experiment_id
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    setup_logging(str(output_dir / "experiment.log"))

    questions, passages, sample_info = load_inputs(args.questions_per_type, args.seed)
    if not args.resume:
        write_json(output_dir / "samples.json", questions)
    manifest = {
        "research_question": "Q4 Retrieval Quality Evaluation",
        "scope": "First-stage balanced four-task design check",
        "experiment_id": experiment_id,
        "dataset": DATASET_NAME,
        "question_count": len(questions),
        "questions_per_type": args.questions_per_type,
        "passage_count": len(passages),
        "seed": args.seed,
        "sample_info": sample_info,
        "embedding_model": str(args.embedding_model.resolve()),
        "evaluator_model": None if args.retrieval_only else args.llm_model,
        "retrieval_top_k": 5,
        "linear_retrieval_path": "official BFS",
        "cache_root": str(args.cache_root.resolve()),
        "medical_parameters": MEDICAL_PARAMETERS,
        "metrics": ["context_relevance", "evidence_recall"],
        "metric_source": "GraphRAG-Benchmark official retrieval evaluator, locally adapted",
    }
    if not args.resume:
        write_json(output_dir / "manifest.json", manifest)

    embedding_model = SentenceTransformer(str(args.embedding_model), device="cuda")

    vanilla_path = output_dir / "vanilla_rag" / "retrieval_results.json"
    if args.resume and vanilla_path.exists():
        with vanilla_path.open(encoding="utf-8") as stream:
            vanilla_results = json.load(stream)
        print("[resume] Vanilla RAG retrieval", flush=True)
    else:
        print("[start] Vanilla RAG retrieval", flush=True)
        vanilla = VanillaRAG(
            dataset_name=DATASET_NAME,
            embedding_model=embedding_model,
            llm_model=None,
            max_workers=args.max_workers,
            retrieval_top_k=5,
            working_dir=args.cache_root,
        )
        vanilla.index(passages)
        vanilla_results = attach_gold_fields(questions, vanilla.retrieve(questions))
        write_json(vanilla_path, vanilla_results)
        del vanilla
        gc.collect()
        print("[done] Vanilla RAG retrieval", flush=True)

    linearrag_path = output_dir / "linearrag" / "retrieval_results.json"
    if args.resume and linearrag_path.exists():
        with linearrag_path.open(encoding="utf-8") as stream:
            linearrag_results = json.load(stream)
        print("[resume] LinearRAG retrieval", flush=True)
    else:
        print("[start] LinearRAG retrieval", flush=True)
        config = LinearRAGConfig(
            dataset_name=DATASET_NAME,
            embedding_model=embedding_model,
            llm_model=None,
            working_dir=args.cache_root,
            max_workers=args.max_workers,
            retrieval_top_k=5,
            use_vectorized_retrieval=False,
            **MEDICAL_PARAMETERS,
        )
        linearrag = LinearRAG(global_config=config)
        linearrag.index(passages)
        linearrag_results = attach_gold_fields(questions, linearrag.retrieve(questions))
        write_json(linearrag_path, linearrag_results)
        del linearrag
        gc.collect()
        print("[done] LinearRAG retrieval", flush=True)

    results_by_method = {
        "vanilla_rag": vanilla_results,
        "linearrag": linearrag_results,
    }
    validation = validate_retrieval(results_by_method, len(questions))
    write_json(output_dir / "retrieval_validation.json", validation)

    if args.retrieval_only:
        summary = {"status": "retrieval_passed", "validation": validation}
        write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print(f"Evidence written to: {output_dir}", flush=True)
        return 0

    llm_model = LLM_Model(args.llm_model)

    evaluations = {}
    for method, results in results_by_method.items():
        print(f"[start] {method} retrieval evaluation", flush=True)
        evaluations[method] = evaluate_method(
            results,
            llm_model,
            args.max_workers,
            output_dir / method / "evaluation_checkpoint.json",
        )
        write_json(output_dir / method / "evaluation_results.json", evaluations[method])
        print(f"[done] {method} retrieval evaluation", flush=True)

    paired = build_paired_comparison(evaluations["vanilla_rag"], evaluations["linearrag"])
    write_json(output_dir / "paired_comparison.json", paired)
    summary = {
        "status": "passed",
        "validation": validation,
        "method_summaries": {
            method: {
                "by_question_type": result["by_question_type"],
                "macro_average": result["macro_average"],
            }
            for method, result in evaluations.items()
        },
        "paired_comparison": paired,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Evidence written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
