"""Run the bounded first-stage Q2 efficiency design check."""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from src.paths import DATASETS_DIR, MODELS_DIR, EXPERIMENT_RESULTS_DIR
from src.utils import setup_logging


DATASET_NAME = "2wikimultihop"
DATASET_PARAMETERS = {
    "spacy_model": "en_core_web_trf",
    "max_iterations": 3,
    "passage_ratio": 0.05,
    "iteration_threshold": 0.4,
    "top_k_sentence": 1,
}

# 假LLM, 被传入LinearRAGConfig中, 如果在索引或检索阶段意外调用LLM, 会立即失败
class NoLLM:
    """Fail immediately if indexing or retrieval unexpectedly invokes an LLM."""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, _messages: Any) -> str:
        self.calls += 1
        raise AssertionError("Q2 indexing and retrieval must not call an LLM")

# 定义测试规模
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded LinearRAG Q2 efficiency experiment")
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--retrieval-repeats", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument(
        "--resume-cache",
        type=Path,
        default=None,
        help="Reuse a completed Q2 cache instead of rebuilding the index from zero.",
    )
    args = parser.parse_args()
    if args.max_questions <= 0:
        parser.error("--max-questions must be greater than 0")
    if args.retrieval_repeats <= 0:
        parser.error("--retrieval-repeats must be greater than 0")
    return args


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default)

"""加载全部 2Wiki chunks;
给所有 passage 加上编号；
按固定种子抽取问题。"""
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

#   GPU 计算时, 需要在计时前后调用 torch.cuda.synchronize() 来确保所有 GPU 操作完成
def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def timed_call(function, *args):
    synchronize_cuda()
    started = time.perf_counter()
    result = function(*args)
    synchronize_cuda()
    return result, time.perf_counter() - started

# 保存本次环境
def environment_snapshot() -> dict:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "spacy": spacy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": gpu_name,
    }

""" 它验证：
- 输入 10 个问题，就必须得到 10 个结果；
- 每个问题必须返回 5 个 passage;
- 每个结果必须有 5 个对应分数。"""
def validate_retrieval(results: list[dict], expected_count: int) -> None:
    if len(results) != expected_count:
        raise AssertionError(f"Expected {expected_count} retrieval results, got {len(results)}")
    for index, result in enumerate(results):
        if len(result["sorted_passage"]) != 5:
            raise AssertionError(f"Question {index} did not return exactly five passages")
        if len(result["sorted_passage_scores"]) != 5:
            raise AssertionError(f"Question {index} did not return exactly five scores")

# 准备->索引->检索->记录
def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q2_efficiency_analysis" / experiment_id
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    setup_logging(str(output_dir / "experiment.log"))

    cache_root = args.resume_cache.resolve() if args.resume_cache else output_dir / "cache"
    if args.resume_cache:
        if not (cache_root / DATASET_NAME / "ner_results.json").exists():
            raise FileNotFoundError(f"Completed cache not found: {cache_root}")
    elif cache_root.exists():
        raise AssertionError(f"Cold-cache directory must not exist before setup: {cache_root}")

    questions, passages, sample_ids = load_inputs(args.max_questions, args.seed)
    manifest = {
        "research_question": "Q2 Efficiency Analysis",
        "scope": "Formal bounded LinearRAG-only efficiency experiment",
        "experiment_id": experiment_id,
        "dataset": DATASET_NAME,
        "passage_count": len(passages),
        "question_count": len(questions),
        "sample_ids": sample_ids,
        "seed": args.seed,
        "embedding_model": str(args.embedding_model.resolve()),
        "retrieval_top_k": 5,
        "retrieval_path": "official BFS",
        "warmup_runs": 1,
        "timed_retrieval_repeats": args.retrieval_repeats,
        "cold_cache": args.resume_cache is None,
        "resumed_from_completed_cache": args.resume_cache is not None,
        "cache_root": str(cache_root.resolve()),
        "dataset_parameters": DATASET_PARAMETERS,
        "timing_boundaries": {
            "setup_seconds": "SentenceTransformer and LinearRAG construction",
            "index_seconds": "LinearRAG.index(passages); cache restoration when --resume-cache is used",
            "retrieval_seconds": "LinearRAG.retrieve(questions)",
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "environment.json", environment_snapshot())
    write_json(output_dir / "samples.json", {"sample_ids": sample_ids, "questions": questions})

    no_llm = NoLLM()
    setup_started = time.perf_counter()
    embedding_model = SentenceTransformer(str(args.embedding_model), device="cuda")
    config = LinearRAGConfig(
        dataset_name=DATASET_NAME,
        embedding_model=embedding_model,
        llm_model=no_llm,
        working_dir=cache_root,
        max_workers=args.max_workers,
        retrieval_top_k=5,
        use_vectorized_retrieval=False,
        **DATASET_PARAMETERS,
    )
    model = LinearRAG(global_config=config)
    setup_seconds = time.perf_counter() - setup_started

    index_mode = "cache restoration" if args.resume_cache else "cold indexing"
    print(f"[index] {index_mode}: {len(passages)} passages", flush=True)
    _, index_seconds = timed_call(model.index, passages)

    print(f"[warmup] Retrieving {len(questions)} questions (untimed result)", flush=True)
    warmup_results = model.retrieve(questions)
    validate_retrieval(warmup_results, len(questions))

    retrieval_runs = []
    final_results = None
    for repeat in range(1, args.retrieval_repeats + 1):
        print(f"[retrieve] Timed repeat {repeat}/{args.retrieval_repeats}", flush=True)
        final_results, elapsed = timed_call(model.retrieve, questions)
        validate_retrieval(final_results, len(questions))
        retrieval_runs.append({
            "repeat": repeat,
            "total_seconds": elapsed,
            "seconds_per_question": elapsed / len(questions),
        })

    if no_llm.calls != 0:
        raise AssertionError(f"Unexpected LLM calls: {no_llm.calls}")

    per_question = [run["seconds_per_question"] for run in retrieval_runs]
    measurements = {
        "status": "passed",
        "setup_seconds": setup_seconds,
        "index_seconds": index_seconds,
        "index_mode": index_mode,
        "retrieval_runs": retrieval_runs,
        "retrieval_seconds_per_question_median": float(np.median(per_question)),
        "llm_calls_during_index_and_retrieval": no_llm.calls,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "validation": {
            "result_count": len(final_results),
            "passages_per_result": 5,
            "isolated_cache": args.resume_cache is None,
            "resumed_cache": args.resume_cache is not None,
            "default_bfs": True,
        },
    }
    write_json(output_dir / "retrieval_results.json", final_results)
    write_json(output_dir / "measurements.json", measurements)
    print(json.dumps(measurements, ensure_ascii=False, indent=2), flush=True)
    print(f"Evidence written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
