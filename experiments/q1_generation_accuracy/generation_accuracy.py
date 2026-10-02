"""运行有明确边界的 Q1 问答准确率比较。

支持原数据集抽样与预先固定的小语料输入包，复用生成和评价流程。
HippoRAG 是显式选择的外部基线；prepare-only 只核验和记录输入。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from src.paths import PROJECT_ROOT, DATASETS_DIR, MODELS_DIR, EXPERIMENT_RESULTS_DIR, CACHE_DIR


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
    """解析原实验与自定义语料输入参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_CONFIGS),
                        default=list(DATASET_CONFIGS))
    parser.add_argument("--corpus-dir", type=Path) # 自定义语料目录
    parser.add_argument("--methods", nargs="+", # 选择要比较的方法
                        choices=["vanilla_rag", "linearrag", "hipporag"],
                        default=["vanilla_rag", "linearrag"])
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--embedding-model", type=Path,
                        default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--hipporag-extraction-model", default="qwen3.8-flash") # HippoRAG 抽取模型
    parser.add_argument("--hipporag-workers", type=int, default=4) # HippoRAG 工作线程数
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if min(args.max_questions, args.max_workers, args.hipporag_workers) < 1:
        parser.error("question count and worker counts must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    return args


def file_hash(path: Path) -> str:
    """记录输入文件哈希，防止恢复时混用不同输入。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(dataset_name: str, dataset_dir=None) -> tuple[list[dict], list[str]]:
    """读取已有问题与片段格式，为三种方法添加一致的片段编号。"""
    dataset_dir = dataset_dir or DATASETS_DIR / dataset_name
    questions = json.loads((dataset_dir / "questions.json").read_text(encoding="utf-8"))
    chunks = json.loads((dataset_dir / "chunks.json").read_text(encoding="utf-8"))
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]
    return questions, passages


def fixed_sample(questions: list[dict], size: int, seed: int, dataset_name: str) -> list[dict]:
    """按原实验的固定种子规则抽样，保持问题原顺序。"""
    if size > len(questions):
        raise ValueError(f"{dataset_name} has fewer than {size} questions")
    rng = random.Random(f"{seed}:{dataset_name}")
    selected_indices = sorted(rng.sample(range(len(questions)), size))
    return [questions[index] for index in selected_indices]


def question_identifier(question: dict, fallback_index: int) -> str:
    """取得可供结果配对的问题 ID。"""
    return str(question.get("id", fallback_index))


def write_json(path: Path, value) -> None:
    """保存配置、预测或比较结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_inputs(args) -> tuple[dict, dict]:
    """使实验既能继续使用原始数据集现场抽样，也能直接消费提前构造好的固定 corpus；
    无论走哪条路径，最终都统一成同样的 questions + passages，再交给各个 RAG 方法"""

    corpus_manifest = None
    if args.corpus_dir:
        args.corpus_dir = (
            args.corpus_dir if args.corpus_dir.is_absolute() else PROJECT_ROOT / args.corpus_dir
        ).resolve()
        corpus_manifest = json.loads(
            (args.corpus_dir / "manifest.json").read_text(encoding="utf-8")
        )
        dataset_names = [corpus_manifest["source_dataset"]]
    else:
        dataset_names = args.datasets

    samples = {}
    records = {}
    for dataset_name in dataset_names:
        dataset_dir = args.corpus_dir or DATASETS_DIR / dataset_name
        questions, passages = load_dataset(dataset_name, dataset_dir)
        hashes = {
            name: file_hash(dataset_dir / f"{name}.json") for name in ("questions", "chunks")
        }
        if corpus_manifest:
            for name, digest in hashes.items():
                if digest != corpus_manifest[f"{name}_sha256"]:
                    raise ValueError(f"Changed corpus file: {name}.json")
            selected = questions  # 构造阶段已抽题，本阶段不重新抽样。
            if [str(q["id"]) for q in selected] != corpus_manifest["question_ids"]:
                raise ValueError("Corpus question IDs differ from manifest")
            cache_identity = hashlib.sha256(
                (hashes["questions"] + hashes["chunks"] + str(args.embedding_model.resolve())).encode()
            ).hexdigest()[:20]
            working_dir = CACHE_DIR / "derived_corpora" / cache_identity
        else:
            selected = fixed_sample(questions, args.max_questions, args.seed, dataset_name)
            working_dir = CACHE_DIR

        if not selected or len(passages) < 5:
            raise ValueError("Q1 needs nonempty questions and at least five passages")
        ids = [question_identifier(q, i) for i, q in enumerate(selected)]
        if len(set(ids)) != len(ids):
            raise ValueError("Question IDs must be unique")
        # 候选上下文、答案和来源映射只用于审计，不传给检索与生成模型。
        runtime_questions = [
            dict(id=identifier, question=q["question"], answer=q["answer"])
            for identifier, q in zip(ids, selected)
        ]
        samples[dataset_name] = (runtime_questions, passages, working_dir)
        records[dataset_name] = dict(
            available_questions=len(questions), sample_count=len(selected),
            passage_count=len(passages), sample_ids=ids, input_sha256=hashes,
            input_dir=str(dataset_dir), cache_dir=str(working_dir),
            linearrag_parameters=DATASET_CONFIGS[dataset_name],
            corpus_manifest_sha256=(file_hash(dataset_dir / "manifest.json")
                                    if corpus_manifest else None),
            construction_seed=corpus_manifest["seed"] if corpus_manifest else None,
        )
    return samples, records


def run_vanilla_rag(dataset_name, questions, passages, embedding_model, llm_model,
                    max_workers, working_dir=CACHE_DIR) -> list[dict]:
    """在指定缓存中运行原 Top-5 向量检索和生成。"""
    from src.baselines.vanilla_rag import VanillaRAG

    model = VanillaRAG(
        dataset_name=dataset_name, embedding_model=embedding_model,
        llm_model=llm_model, max_workers=max_workers,
        retrieval_top_k=5, working_dir=working_dir,
    )
    model.index(passages)
    return model.qa(questions)


def run_linearrag(dataset_name, questions, passages, embedding_model, llm_model,
                  max_workers, working_dir=CACHE_DIR) -> list[dict]:
    """使用源数据集参数和独立语料缓存，运行原 BFS 检索。"""
    from src.linearrag.config import LinearRAGConfig
    from src.linearrag.LinearRAG import LinearRAG

    config = LinearRAGConfig(
        dataset_name=dataset_name, embedding_model=embedding_model,
        llm_model=llm_model, max_workers=max_workers, retrieval_top_k=5,
        working_dir=working_dir, use_vectorized_retrieval=False,
        **DATASET_CONFIGS[dataset_name],
    )
    model = LinearRAG(global_config=config)
    model.index(passages)
    results = model.qa(questions)
    for question_info, result in zip(questions, results):
        result["id"] = question_info["id"]
    return results


def run_hipporag(questions, passages, args, llm_model, output_dir) -> list[dict]:
    """通过已有适配器运行原版 HippoRAG，并在本次结果目录保留缓存。"""
    from src.baselines.hipporag import HippoRAG

    model = HippoRAG(
        llm_model=llm_model, embedding_model_path=args.embedding_model,
        extraction_model=args.hipporag_extraction_model,
        retrieval_top_k=5, max_workers=args.hipporag_workers,
        working_dir=output_dir.parent, experiment_id=output_dir.name,
    )
    model.index(passages)
    return model.qa(questions)


def evaluate_predictions(predictions, output_dir, llm_model, max_workers) -> dict:
    """复用现有评价器，保存逐题分数及总体指标。"""
    from src.common.evaluate import Evaluator

    predictions_path = output_dir / "predictions.json"
    write_json(predictions_path, predictions)
    evaluator = Evaluator(llm_model=llm_model, predictions_path=str(predictions_path))
    llm_accuracy, contain_accuracy = evaluator.evaluate(max_workers=max_workers)
    return dict(llm_accuracy=llm_accuracy, contain_accuracy=contain_accuracy,
                sample_count=len(predictions))


def build_comparison(dataset_name: str, vanilla: dict, linearrag: dict) -> dict:
    """保留原两方法比较格式，供旧结果使用。"""
    metrics = ["llm_accuracy"] if dataset_name == "medical" else [
        "contain_accuracy", "llm_accuracy",
    ]
    return dict(
        dataset=dataset_name, sample_count=vanilla["sample_count"],
        reported_metrics=metrics, vanilla_rag=vanilla, linearrag=linearrag,
        linearrag_minus_vanilla={m: linearrag[m] - vanilla[m] for m in metrics},
    )


def compare_methods(dataset_name: str, results: dict) -> dict:
    """汇总所选方法，并记录 LinearRAG 相对已运行对照的差值。"""
    if list(results) == ["vanilla_rag", "linearrag"]:
        return build_comparison(dataset_name, results["vanilla_rag"], results["linearrag"])

    metrics = ["llm_accuracy"] if dataset_name == "medical" else [
        "contain_accuracy", "llm_accuracy",
    ]
    comparison = dict(dataset=dataset_name, reported_metrics=metrics, **results)
    if "linearrag" in results:
        for method, scores in results.items():
            if method != "linearrag":
                comparison[f"linearrag_minus_{method}"] = {
                    metric: results["linearrag"][metric] - scores[metric] for metric in metrics
                }
    return comparison


def validate_predictions(predictions, questions, passages) -> None:
    """核验问题配对、Top-5 数量和返回片段归属，避免错误汇总。"""
    if len(predictions) != len(questions):
        raise ValueError("Prediction count differs from question count")
    passage_set = set(passages)
    for prediction, question in zip(predictions, questions):
        if str(prediction["id"]) != str(question["id"]):
            raise ValueError("Prediction IDs are not aligned")
        if prediction["question"] != question["question"]:
            raise ValueError("Prediction question text differs")
        if prediction["gold_answer"] != question["answer"] or "pred_answer" not in prediction:
            raise ValueError("Prediction answer fields differ or are incomplete")
        context = prediction["sorted_passage"]
        if len(context) != 5 or not set(context).issubset(passage_set):
            raise ValueError("Prediction context differs from shared corpus/Top-5")


def main() -> int:
    """准备输入清单，或运行所选方法并复用已有完整预测和评价。"""
    args = parse_args()
    args.embedding_model = (
        args.embedding_model if args.embedding_model.is_absolute()
        else PROJECT_ROOT / args.embedding_model
    ).resolve()
    samples, records = prepare_inputs(args)
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q1_generation_accuracy" / experiment_id
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {output_dir}")

    manifest = dict(
        research_question="Q1 Generation Accuracy", experiment_id=experiment_id,
        seed=args.seed if not args.corpus_dir else None,
        methods=args.methods, input_mode="fixed_corpus" if args.corpus_dir else "sampled",
        embedding_model=str(args.embedding_model), llm_model=args.llm_model,
        max_workers=args.max_workers, hipporag_workers=args.hipporag_workers,
        hipporag_extraction_model=args.hipporag_extraction_model,
        hipporag_retrieval_config=dict(damping=0.5, sim_threshold=0.8),
        enable_thinking=False if args.llm_model == "qwen3.8-flash" else None,
        retrieval_top_k=5, linear_retrieval_path="official BFS", datasets=records,
    )
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Resume inputs/configuration differ; use a new experiment ID")
    else:
        write_json(manifest_path, manifest)

    if args.prepare_only:
        print(f"Prepared Q1 sample manifest: {manifest_path}")
        return 0

    from sentence_transformers import SentenceTransformer
    from src.common.utils import LLM_Model, setup_logging

    setup_logging(str(output_dir / "experiment.log"))
    embedding_model = SentenceTransformer(str(args.embedding_model), device="cuda")
    llm_model = LLM_Model(args.llm_model)
    comparisons = []
    for dataset_name, (questions, passages, working_dir) in samples.items():
        dataset_output = output_dir / dataset_name
        scores = {}
        for method in args.methods:
            method_dir = dataset_output / method
            predictions_path = method_dir / "predictions.json"
            metrics_path = method_dir / "evaluation_results.json"
            if args.resume and predictions_path.exists():
                predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
                validate_predictions(predictions, questions, passages)
            else:
                print(f"[start] {method}: {dataset_name}", flush=True)
                if method == "hipporag":
                    predictions = run_hipporag(questions, passages, args, llm_model, method_dir)
                elif method == "vanilla_rag":
                    predictions = run_vanilla_rag(
                        dataset_name, questions, passages, embedding_model,
                        llm_model, args.max_workers, working_dir,
                    )
                else:
                    predictions = run_linearrag(
                        dataset_name, questions, passages, embedding_model,
                        llm_model, args.max_workers, working_dir,
                    )
                validate_predictions(predictions, questions, passages)

            if args.resume and metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                metrics["sample_count"] = len(questions)
            else:
                metrics = evaluate_predictions(predictions, method_dir, llm_model, args.max_workers)
            scores[method] = metrics

        comparison = compare_methods(dataset_name, scores)
        write_json(dataset_output / "comparison.json", comparison)
        comparisons.append(comparison)

    write_json(output_dir / "summary.json", dict(datasets=comparisons))
    print(f"Q1 experiment completed: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
