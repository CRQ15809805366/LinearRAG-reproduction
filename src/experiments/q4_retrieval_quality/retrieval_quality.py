"""运行 Q4 检索质量比较，支持原版 HippoRAG 与离线实验准备。

The experiment holds the Medical corpus, sampled questions, embedding model,
Top-k, and evaluator fixed while comparing selected retrieval methods.

The two evaluator prompts are adapted from GraphRAG-Benchmark's MIT-licensed
Evaluation/metrics implementation:
https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
Copyright (c) 2025 XMU-DeepLIT
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.linear.config import LinearRAGConfig
from src.methods.linear.LinearRAG import LinearRAG
from src.paths import DATASET_CACHE_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR, PROJECT_ROOT
from src.datasets.input_package import load_package, input_record
from src.common.utils import LLM_Model, setup_logging
from src.methods.vanilla.vanilla_rag import VanillaRAG
from src.methods.hippo.adapter import HippoRAG


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


from src.datasets.sampling import build_sample_inputs


def parse_args() -> argparse.Namespace:
    """解析完整检索质量实验的输入和方法参数。"""
    parser = argparse.ArgumentParser(description="Q4 Medical retrieval-quality comparison")
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--llm-model", default="qwen3.8-flash")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--cache-root", type=Path, default=DATASET_CACHE_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--methods", nargs="+", # 加入 HippoRAG 是显式选择的
                        choices=["vanilla_rag", "linearrag", "hipporag"],
                        default=["vanilla_rag", "linearrag"])
    parser.add_argument("--hipporag-extraction-model", default="qwen3.8-flash")
    parser.add_argument("--hipporag-workers", type=int, default=4)
    parser.add_argument("--corpus-dir", type=Path, help="Use a fixed Medical input package without resampling.")
    parser.add_argument("--datasets", nargs="+", choices=['medical'], default=['medical'])
    parser.add_argument("--questions-per-type", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--corpus-id")
    args = parser.parse_args()
    if not args.corpus_dir:
        if args.questions_per_type is None or args.questions_per_type < 1:
            parser.error("Provide --questions-per-type with a positive count, or use --corpus-dir")
        if len(set(args.datasets)) != len(args.datasets):
            parser.error("datasets must be unique")
        paths = build_sample_inputs(
            datasets=args.datasets,
            count=args.questions_per_type,
            seed=args.seed,
            corpus_prefix="q4",
            corpus_id=args.corpus_id,
            stratified=True,
        )
        args.corpus_dir = paths[0]
    if min(args.max_workers, args.hipporag_workers) < 1:
        parser.error("worker counts must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")

    for name in ("embedding_model", "cache_root", "corpus_dir"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, (PROJECT_ROOT / path).resolve())
    return args


def write_json(path: Path, value: Any) -> None:
    """保存实验输入、逐题证据或汇总结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    """将路径与 NumPy 数值转换为 JSON 可保存的类型。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")

def load_inputs(args) -> tuple[list[dict], list[str], dict]:
    """读取完整 Medical 原语料，并核对四类等量问题样本。"""
    dataset_dir = args.corpus_dir
    selected, passages, corpus_manifest = load_package(dataset_dir)
    if corpus_manifest.get("corpus_scope") != "full_source_chunks":
        raise ValueError("Q4 requires a full-source input package")
    if corpus_manifest["source_dataset"] != DATASET_NAME:
        raise ValueError("Q4 requires a Medical corpus")
    counts = Counter(q["question_type"] for q in selected)
    if set(counts) != set(QUESTION_TYPES) or len(set(counts.values())) != 1:
        raise ValueError("Q4 requires equal nonzero question counts for all four task types")
    if len(passages) < 5 or len({q["id"] for q in selected}) != len(selected):
        raise ValueError("Q4 requires at least five passages and unique question IDs")

    identity = hashlib.sha256(json.dumps(
        {"questions": selected, "passages": passages}, ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")).hexdigest()

    info = {
        "input_dir": str(dataset_dir),
        "input_identity": identity,
        "question_type_counts": dict(counts),
        "seed": corpus_manifest.get("seed"),
        "input_package": input_record(dataset_dir, corpus_manifest),
    }
    return selected, passages, info


# 把原问题中的这些字段重新装回检索结果, 以便benchmark可以评测
def attach_gold_fields(questions: list[dict], results: list[dict]) -> list[dict]:
    """核对检索结果顺序，并附上任务类型和参考证据。"""
    if len(questions) != len(results):
        raise AssertionError("Question and retrieval-result counts differ")
    enriched = []
    for question, result in zip(questions, results):
        # LinearRAG 的原接口保留问题文本但不返回 ID，按顺序核对后补回。
        if result["question"] != question["question"]:
            raise AssertionError("Question and retrieval-result texts differ")
        if "id" in result and result["id"] != question["id"]:
            raise AssertionError("Question and retrieval-result IDs differ")
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
    """核对方法间的问题顺序、证据与 Top-5 完整性。"""
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
            raise AssertionError("Methods did not evaluate identical ordered questions")
        validations[method] = {
            "result_count": len(results),
            "passages_per_result": 5,
            "question_type_counts": dict(Counter(item["question_type"] for item in results)),
        }
    return {"status": "passed", "methods": validations, "paired_question_ids": expected_ids}


def parse_json_object(raw: str) -> dict:
    """从裁判回复中提取 JSON 对象。"""
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
    """调用裁判并校验 JSON 输出，补试格式不完整的回复。"""
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
    """验证裁判给出的零到二分相关性评分。"""
    score = value.get("score", value.get("rating"))
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= 2 else None


def evidence_classifications(value: dict, reference_evidence: list[str]) -> list[dict] | None:
    """保留与参考证据逐字对应的有效支持判断。"""
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
    """验证单条补评证据的支持判断。"""
    items = value.get("classifications", [value])
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
    """评价单题上下文相关性与证据召回，并保存原始回复。"""
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
        single_prompt = (
            "Judge ONLY whether the reference statement is supported by the context. "
            "Do not extract or rewrite statements from the context. "
            "Return only JSON: {\"reason\": \"brief explanation\", \"attributed\": 0}. "
            "Use attributed=1 if supported, otherwise 0.\n"
            f"Context: {context}\nQuestion: {item['question']}\n"
            f"Reference statement: {evidence}\n/no_think"
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
    """计算有效指标的平均值。"""
    valid = [value for value in values if not math.isnan(value)]
    return float(np.mean(valid)) if valid else math.nan


def evaluation_is_complete(item: dict | None) -> bool:
    """只有两次相关性评分和全部证据判断完成时才复用该题评价。"""
    return (
        item is not None
        and not math.isnan(item["context_relevance"])
        and not math.isnan(item["evidence_recall"])
        and item["relevance_rating_count"] == 2
        and item["classified_evidence_count"] == item["reference_evidence_count"]
    )

# 扩展到全部问题, 计算四组宏观平均
def evaluate_method(
    results: list[dict],
    llm_model: LLM_Model,
    max_workers: int,
    checkpoint_path: Path,
) -> dict:
    """恢复已完成评价，补评缺失问题并按任务汇总。"""
    completed_by_id = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as stream:
            completed_by_id = {
                item["id"]: item
                for item in json.load(stream)
                if evaluation_is_complete(item)
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

def main() -> int:
    """在同一固定输入上执行所选方法，统一评价并列汇总。"""
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q4_retrieval_quality" / experiment_id
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {output_dir}")

    questions, passages, sample_info = load_inputs(args)
    manifest = {
        "research_question": "Q4 Retrieval Quality Evaluation",
        "scope": "Bounded balanced Medical retrieval comparison",
        "experiment_id": experiment_id,
        "dataset": DATASET_NAME,
        "question_count": len(questions),
        "passage_count": len(passages),
        "seed": sample_info["seed"],
        "sample_info": sample_info,
        "embedding_model": str(args.embedding_model),
        "evaluator_model": args.llm_model,
        "retrieval_top_k": 5,
        "linear_retrieval_path": "official BFS",
        "cache_root": str(args.cache_root),
        "medical_parameters": MEDICAL_PARAMETERS,
        "metrics": ["context_relevance", "evidence_recall"],
        "metric_source": "GraphRAG-Benchmark official retrieval evaluator, locally adapted",
        "methods": args.methods,
        "hipporag_extraction_model": args.hipporag_extraction_model,
    }
    if args.resume:
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Resume requires identical inputs, methods and model configuration")
    else:
        output_dir.mkdir(parents=True)
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "samples.json", questions)

    setup_logging(str(output_dir / "experiment.log"))

    # 只有尚缺本地检索结果时才加载嵌入模型，旧证据和 HippoRAG 不依赖此对象。
    needs_local = any(
        method != "hipporag" and not (output_dir / method / "retrieval_results.json").exists()
        for method in args.methods
    )
    embedding_model = SentenceTransformer(str(args.embedding_model), device="cuda") if needs_local else None
    runtime_questions = [
        {key: q[key] for key in ("id", "question", "answer")} for q in questions
    ]
    results_by_method = {}
    for method in args.methods:
        retrieval_path = output_dir / method / "retrieval_results.json"
        if retrieval_path.exists():
            results = json.loads(retrieval_path.read_text(encoding="utf-8"))
            print(f"[reuse] {method} retrieval", flush=True)
        else:
            print(f"[start] {method} retrieval", flush=True)
            if method == "vanilla_rag":
                retriever = VanillaRAG(
                    dataset_name=DATASET_NAME, embedding_model=embedding_model,
                    llm_model=None, max_workers=args.max_workers,
                    retrieval_top_k=5, working_dir=args.cache_root,
                )
            elif method == "linearrag":
                config = LinearRAGConfig(
                    dataset_name=DATASET_NAME, embedding_model=embedding_model,
                    llm_model=None, working_dir=args.cache_root,
                    max_workers=args.max_workers, retrieval_top_k=5,
                    use_vectorized_retrieval=False, **MEDICAL_PARAMETERS,
                )
                retriever = LinearRAG(global_config=config)
            else: # 创建 HippoRAG 的分支
                retriever = HippoRAG(
                    embedding_model_path=args.embedding_model,
                    extraction_model=args.hipporag_extraction_model,
                    max_workers=args.hipporag_workers, retrieval_top_k=5,
                    working_dir=output_dir, experiment_id="hipporag",
                )
            retriever.index(passages)
            results = attach_gold_fields(questions, retriever.retrieve(runtime_questions))
            write_json(retrieval_path, results)
            del retriever
            gc.collect()
        results_by_method[method] = results

    validation = validate_retrieval(results_by_method, len(questions))
    write_json(output_dir / "retrieval_validation.json", validation)

    llm_model = LLM_Model(args.llm_model)
    evaluations = {}
    for method, results in results_by_method.items():
        evaluations[method] = evaluate_method(
            results, llm_model, args.max_workers,
            output_dir / method / "evaluation_checkpoint.json",
        )
        write_json(output_dir / method / "evaluation_results.json", evaluations[method])

    complete = all(
        evaluation_is_complete(row)
        for result in evaluations.values() for row in result["detailed"]
    )
    summary = {
        "status": "passed" if complete else "incomplete_evaluation",
        "validation": validation,
        "method_summaries": {
            method: {key: result[key] for key in ("by_question_type", "macro_average")}
            for method, result in evaluations.items()
        },
    }
    write_json(output_dir / "summary.json", summary)

    # 直接并列已有均值，四类任务和宏平均采用相同的表格列。
    with (output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["task", "metric", *args.methods])
        for task in (*QUESTION_TYPES, "macro_average"):
            for metric in ("context_relevance", "evidence_recall"):
                values = []
                for method in args.methods:
                    result = summary["method_summaries"][method]
                    group = result["macro_average"] if task == "macro_average" else result["by_question_type"][task]
                    values.append(group[metric])
                writer.writerow([task, metric, *values])
    print(f"Evidence written to: {output_dir}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
