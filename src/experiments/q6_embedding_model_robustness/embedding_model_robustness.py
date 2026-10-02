"""Run the Q6 sentence-embedding robustness evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import os
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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.linear.config import LinearRAGConfig
from src.methods.linear.LinearRAG import LinearRAG
from src.methods.linear.ner import SpacyNER
from src.paths import EMBEDDING_CACHE_DIR, DATASET_CACHE_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.common.evaluate import Evaluator
from src.common.utils import LLM_Model, setup_logging
from src.datasets.input_package import load_package, input_record, runtime_questions
from src.common.cache_identity import linear_stages, write_manifest


DATASET_NAME = "2wikimultihop"
MODEL_NAMES = (
    "all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "bge-large-en-v1.5",
    "e5-large-v2",
)
BASELINE_MODEL = "all-mpnet-base-v2"

from src.datasets.sampling import build_sample_inputs


def json_default(value):
    """将路径及 NumPy 对象转换为可写入 JSON 的值。"""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")

def write_json(path: Path, value) -> None:
    """将实验结果写入 JSON 文件，保留原有序列化规则。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default)

def evaluate_predictions(
    predictions: list[dict],
    output_dir: Path,
    llm_model: LLM_Model,
    max_workers: int,
) -> dict:
    """评估当前 Q 的预测，并汇总两项准确率及其平均值。"""
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

def compare_retrievals(full_predictions: list[dict], ablated_predictions: list[dict]) -> dict:
    """比较基准和当前模型的 Top-5 检索及生成答案。"""
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


# 接收实验参数
def parse_args() -> argparse.Namespace:
    """解析按模型隔离缓存的 Q6 实验参数。"""
    parser = argparse.ArgumentParser(description="LinearRAG Q6 embedding robustness experiment")
    parser.add_argument("--llm-model", default="qwen3.8-flash")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=EMBEDDING_CACHE_DIR,
    )
    parser.add_argument(
        "--shared-ner-cache",
        type=Path,
        default=DATASET_CACHE_DIR / DATASET_NAME / "ner_results.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--corpus-dir", type=Path, help="实验输入包目录。")
    parser.add_argument("--datasets", nargs="+", choices=['2wikimultihop'], default=['2wikimultihop'])
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--corpus-id")
    args = parser.parse_args()
    if not args.corpus_dir:
        if args.max_questions is None or args.max_questions < 1:
            parser.error("Provide --max-questions with a positive count, or use --corpus-dir")
        if len(set(args.datasets)) != len(args.datasets):
            parser.error("datasets must be unique")
        paths = build_sample_inputs(
            datasets=args.datasets,
            count=args.max_questions,
            seed=args.seed,
            corpus_prefix="q6",
            corpus_id=args.corpus_id,
        )
        args.corpus_dir = paths[0]
    return args

# 读取已准备的问题集与完整语料
def load_inputs(corpus_dir):
    """读取原语料输入包，所有实验条件共享同一问题集与片段。"""
    questions, passages, package = load_package(corpus_dir)
    if package["source_dataset"] != DATASET_NAME or package.get("corpus_scope") != "full_source_chunks":
        raise ValueError("This experiment requires a prepared full-source 2Wiki input package")
    if len(passages) < 5:
        raise ValueError("At least five passages are required")
    return questions, passages, package["question_ids"], package

# 为每个模型建立独立向量缓存
def prepare_model_cache(cache_root: Path, model_name: str, shared_ner_cache: Path, passages) -> Path:
    """为各模型隔离向量，仅从匹配抽取身份的共享 NER 复制结果。"""
    model_cache = cache_root / model_name
    dataset_cache = model_cache / DATASET_NAME
    dataset_cache.mkdir(parents=True, exist_ok=True)
    target = dataset_cache / "ner_results.json"
    if not target.exists():
        if not shared_ner_cache.exists():
            raise FileNotFoundError(f"Shared NER cache is missing: {shared_ner_cache}")
        stages = linear_stages(passages, MODELS_DIR / model_name, "en_core_web_trf")
        source = json.loads((shared_ner_cache.parent / "manifest.json").read_text(encoding="utf-8"))
        if source["stages"]["extraction"]["id"] != stages["extraction"]["id"]:
            raise ValueError("Shared NER corpus/model/protocol differs from Q6")
        shutil.copy2(shared_ner_cache, target)
        write_manifest(dataset_cache, dict(schema=2, method="linearrag", stages=stages,
            reuse="Q6 changes embeddings only; NER was copied from a matching extraction identity."))
    return model_cache

# 继承自 LinearRAG，增加观察信息收集功能
class ObservedLinearRAG(LinearRAG):
    """Run the real BFS path and expose only small mechanism-level diagnostics."""

    def __init__(self, global_config: LinearRAGConfig):
        """初始化原算法的资源，并附加诊断列表。"""
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
    model_cache = prepare_model_cache(args.cache_root, model_name, args.shared_ner_cache, passages)
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

    model_questions = runtime_questions(questions)
    retrieval_started = time.perf_counter()
    retrieval_results = rag.retrieve(model_questions)
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

    questions, passages, sample_ids, package = load_inputs(args.corpus_dir)
    manifest = {
        "research_question": "Q6 Effectiveness of Different Sentence Embeddings",
        "scope": "One-tenth-paper-scale four-model evaluation",
        "experiment_id": experiment_id,
        "dataset": DATASET_NAME,
        "question_count": len(questions),
        "passage_count": len(passages),
        "sample_ids": sample_ids,
        "seed": package.get("seed"),
        "input_package": input_record(args.corpus_dir, package),
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
    if args.resume:
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("input_package") != manifest["input_package"]:
            raise ValueError("Resume input package differs; use a new experiment ID")
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
