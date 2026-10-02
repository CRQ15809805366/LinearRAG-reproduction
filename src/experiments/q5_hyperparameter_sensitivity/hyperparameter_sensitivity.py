"""运行 Q5 超参数敏感性实验。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.linear.config import LinearRAGConfig
from src.methods.linear.LinearRAG import LinearRAG
from src.paths import DATASET_CACHE_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.common.evaluate import Evaluator
from src.common.utils import LLM_Model, setup_logging
from src.datasets.input_package import load_package, input_record, runtime_questions


DATASET_NAME = "2wikimultihop"
BASE_DELTA = 0.4
BASE_LAMBDA = 0.05
DELTA_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
LAMBDA_VALUES = (0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0)

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

def qa_with_isolated_failures(model: LinearRAG, questions: list[dict]):
    """并发生成答案，记录单题失败并保留其他问题的结果。"""
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


# 接收输入包、模型和输出编号等运行参数
def parse_args() -> argparse.Namespace:
    """解析参数扫描及可复用的数据集缓存路径。"""
    parser = argparse.ArgumentParser(description="LinearRAG Q5 hyperparameter sensitivity experiment")
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--llm-model", default="qwen3.8-flash")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--cache-root", type=Path, default=DATASET_CACHE_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--corpus-dir", type=Path, help="实验输入包目录。")
    parser.add_argument("--datasets", nargs="+", choices=['2wikimultihop'], default=['2wikimultihop'])
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260920)
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
            corpus_prefix="q5",
            corpus_id=args.corpus_id,
        )
        args.corpus_dir = paths[0]
    return args

# 保证所有参数点面对的是完全相同的考试题
def load_inputs(corpus_dir):
    """读取原语料输入包，所有实验条件共享同一问题集与片段。"""
    questions, passages, package = load_package(corpus_dir)
    if package["source_dataset"] != DATASET_NAME or package.get("corpus_scope") != "full_source_chunks":
        raise ValueError("This experiment requires a prepared full-source 2Wiki input package")
    if len(passages) < 5:
        raise ValueError("At least five passages are required")
    return questions, passages, package["question_ids"], package


def value_slug(value: float) -> str:
    return str(value).replace(".", "p")

# 产生两组实验点：9 个 δ 点, 7 个 λ 点
def experiment_points():
    for value in DELTA_VALUES:
        yield "delta", value, value, BASE_LAMBDA
    for value in LAMBDA_VALUES:
        yield "lambda", value, BASE_DELTA, value

# 继承真实 LinearRAG, 给原来的黑箱加一个透明观察窗
class SensitivityLinearRAG(LinearRAG):
    """Record the size and cost of the real BFS retrieval path."""

    def __init__(self, global_config: LinearRAGConfig):
        self.diagnostics: list[dict] = []
        super().__init__(global_config)

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
            "delta": self.config.iteration_threshold,
            "lambda": self.config.passage_ratio,
            "seed_entity_count": len(seed_ids),
            "active_entity_count": len(active_ids),
            "propagated_entity_count": len(active_ids - seed_ids),
            "graph_search_seconds": time.perf_counter() - started,
        })
        return passage_ids, passage_scores

# 检查每个参数点是否真的跑完整
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

# 把每个参数点与基准参数比较: 参数有没有改变检索？检索变化有没有传递到最终答案？
def top5_overlap(baseline: list[dict], candidate: list[dict]) -> dict:
    overlaps = []
    changed_rankings = 0
    changed_answers = 0
    for base, current in zip(baseline, candidate):
        base_passages = base["sorted_passage"]
        current_passages = current["sorted_passage"]
        overlaps.append(len(set(base_passages) & set(current_passages)) / 5)
        changed_rankings += base_passages != current_passages
        changed_answers += base.get("pred_answer") != current.get("pred_answer")
    return {
        "mean_top5_overlap_with_baseline": float(np.mean(overlaps)),
        "questions_with_changed_top5_ranking": changed_rankings,
        "questions_with_changed_answer": changed_answers,
    }


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q5_hyperparameter_sensitivity" / experiment_id
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    setup_logging(str(output_dir / "experiment.log"))

    questions, passages, sample_ids, package = load_inputs(args.corpus_dir)
    manifest = {
        "research_question": "Q5 Hyper-parameter Sensitivity",
        "scope": "Fixed-sample one-factor-at-a-time sensitivity experiment",
        "experiment_id": experiment_id,
        "dataset": DATASET_NAME,
        "question_count": len(questions),
        "passage_count": len(passages),
        "sample_ids": sample_ids,
        "seed": package.get("seed"),
        "input_package": input_record(args.corpus_dir, package),
        "embedding_model": str(args.embedding_model.resolve()),
        "llm_model": args.llm_model,
        "retrieval_path": "official BFS",
        "retrieval_top_k": 5,
        "fixed_parameters": {"max_iterations": 3, "top_k_sentence": 1},
        "baseline": {"delta": BASE_DELTA, "lambda": BASE_LAMBDA},
        "delta_values": DELTA_VALUES,
        "lambda_values": LAMBDA_VALUES,
        "parameter_mapping": {
            "delta": "LinearRAGConfig.iteration_threshold",
            "lambda": "LinearRAGConfig.passage_ratio",
        },
    }
    if args.resume:
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("input_package") != manifest["input_package"]:
            raise ValueError("Resume input package differs; use a new experiment ID")
    if not args.resume:
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "samples.json", {"sample_ids": sample_ids, "questions": questions})

    embedding_model = SentenceTransformer(str(args.embedding_model), device="cuda")
    llm_model = LLM_Model(args.llm_model)
    config = LinearRAGConfig(
        dataset_name=DATASET_NAME,
        embedding_model=embedding_model,
        llm_model=llm_model,
        working_dir=args.cache_root,
        spacy_model="en_core_web_trf",
        max_workers=args.max_workers,
        retrieval_top_k=5,
        use_vectorized_retrieval=False,
        max_iterations=3,
        top_k_sentence=1,
        iteration_threshold=BASE_DELTA,
        passage_ratio=BASE_LAMBDA,
    )
    model = SensitivityLinearRAG(config)
    model.index(passages)

    rows = []
    predictions_by_point = {}
    for axis, value, delta, lambda_value in experiment_points():
        point_name = f"{axis}_{value_slug(value)}"
        point_dir = output_dir / axis / value_slug(value)
        result_path = point_dir / "result.json"
        if args.resume and result_path.exists():
            with result_path.open(encoding="utf-8") as stream:
                row = json.load(stream)
            with (point_dir / "predictions.json").open(encoding="utf-8") as stream:
                predictions_by_point[point_name] = json.load(stream)
            rows.append(row)
            print(f"[resume] {point_name}", flush=True)
            continue

        print(f"[start] {point_name}: delta={delta}, lambda={lambda_value}", flush=True)
        model.config.iteration_threshold = delta
        model.config.passage_ratio = lambda_value
        model.diagnostics = []
        predictions, generation_errors = qa_with_isolated_failures(model, runtime_questions(questions))
        for question_info, prediction in zip(questions, predictions):
            prediction["id"] = question_info.get("id")
        write_json(point_dir / "generation_errors.json", generation_errors)
        write_json(point_dir / "retrieval_diagnostics.json", model.diagnostics)
        validation = validate_run(predictions, model.diagnostics, len(questions))
        write_json(point_dir / "validation.json", validation)
        metrics = evaluate_predictions(predictions, point_dir, llm_model, args.max_workers)
        row = {
            "axis": axis,
            "value": value,
            "delta": delta,
            "lambda": lambda_value,
            "metrics": metrics,
            "validation": validation,
            "generation_error_count": len(generation_errors),
        }
        write_json(result_path, row)
        predictions_by_point[point_name] = predictions
        rows.append(row)
        print(f"[done] {point_name}: mean={metrics['average_accuracy']:.3f}", flush=True)

    baseline = predictions_by_point["delta_0p4"]
    for row in rows:
        point_name = f"{row['axis']}_{value_slug(row['value'])}"
        row["retrieval_comparison"] = top5_overlap(
            baseline, predictions_by_point[point_name]
        )
    summary = {
        "status": "passed",
        "primary_metric": "average_accuracy",
        "primary_metric_definition": "(llm_accuracy + contain_accuracy) / 2",
        "baseline": {"delta": BASE_DELTA, "lambda": BASE_LAMBDA},
        "points": rows,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Evidence written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
