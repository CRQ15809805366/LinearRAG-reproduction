"""测量 LinearRAG 与原版 HippoRAG 的有界索引和检索效率。

支持原 2Wiki 抽样和共享固定 corpus；prepare-only 不加载模型、不调用 API。
回答生成和评价属于 Q1，不进入本实验计时或 token 统计。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORT_ROOT))

from src.paths import PROJECT_ROOT, DATASETS_DIR, MODELS_DIR, EXPERIMENT_RESULTS_DIR


DATASET_NAME = "2wikimultihop"
DATASET_PARAMETERS = {
    "spacy_model": "en_core_web_trf",
    "max_iterations": 3,
    "passage_ratio": 0.05,
    "iteration_threshold": 0.4,
    "top_k_sentence": 1,
}


class NoLLM:
    """阻止 LinearRAG 索引和检索阶段意外调用生成式 LLM。"""

    def __init__(self) -> None:
        """初始化动态调用计数。"""
        self.calls = 0

    def infer(self, _messages: Any) -> str:
        """记录调用并立即报错，避免把付费调用混入无 LLM 测量。"""
        self.calls += 1
        raise AssertionError("LinearRAG indexing and retrieval must not call an LLM")


def parse_args() -> argparse.Namespace:
    """解析共同输入、方法和计时参数，prepare-only 用于离线准备。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path)
    parser.add_argument("--methods", nargs="+", choices=["linearrag", "hipporag"], default=["linearrag"])
    parser.add_argument("--max-questions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--retrieval-repeats", type=int, default=2)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--hipporag-extraction-model", default="qwen3.8-flash")
    parser.add_argument("--hipporag-workers", type=int, default=4)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--resume-cache", type=Path, help="Reuse LinearRAG index; this is not a cold build.")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Continue an identical prepared or interrupted run.")
    args = parser.parse_args()
    if min(args.max_questions, args.retrieval_repeats, args.max_workers, args.hipporag_workers) < 1:
        parser.error("question, repetition and worker counts must be positive")
    if len(set(args.methods)) != len(args.methods):
        parser.error("methods must be unique")
    if args.corpus_dir and args.resume_cache:
        parser.error("Fixed-corpus measurements use isolated caches; use --resume for their own run")

    for name in ("corpus_dir", "embedding_model", "resume_cache"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, (PROJECT_ROOT / path).resolve())
    return args


def file_hash(path: Path) -> str:
    """计算输入与测量代码哈希，防止准备后换用其他版本。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_default(value: Any) -> Any:
    """转换路径及 NumPy 返回值，避免准备阶段导入模型依赖。"""
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    """保存输入、测量或准备记录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")


def load_inputs(args) -> tuple[list[dict], list[str], dict]:
    """固定包全量读取且不再抽题；旧模式仍在完整 2Wiki 语料上抽题。"""
    dataset_dir = args.corpus_dir or DATASETS_DIR / DATASET_NAME
    questions = json.loads((dataset_dir / "questions.json").read_text(encoding="utf-8"))
    chunks = json.loads((dataset_dir / "chunks.json").read_text(encoding="utf-8"))
    hashes = {name: file_hash(dataset_dir / f"{name}.json") for name in ("questions", "chunks")}

    if args.corpus_dir:
        package = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        dataset_name = package["source_dataset"]
        if dataset_name not in ("hotpotqa", "2wikimultihop"):
            raise ValueError("This Q2 supplement supports HotpotQA and 2Wiki configurations")
        if any(hashes[name] != package[f"{name}_sha256"] for name in hashes):
            raise ValueError("Corpus files differ from their construction manifest")
        if [str(q["id"]) for q in questions] != package["question_ids"]:
            raise ValueError("Corpus question IDs differ from manifest")
    else:
        package = None
        dataset_name = DATASET_NAME
        indices = sorted(random.Random(f"{args.seed}:{dataset_name}").sample(
            range(len(questions)), args.max_questions
        ))
        questions = [questions[index] for index in indices]

    if not questions or len(chunks) < 5:
        raise ValueError("Q2 requires nonempty questions and at least five passages")
    # 只传问题文本、ID 和答案；候选上下文及映射不能进入检索器。
    runtime_questions = [
        dict(id=str(q.get("id", index)), question=q["question"], answer=q["answer"])
        for index, q in enumerate(questions)
    ]
    if len({q["id"] for q in runtime_questions}) != len(runtime_questions):
        raise ValueError("Question IDs must be unique")
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(chunks)]
    inputs = dict(
        dataset=dataset_name, input_dir=str(dataset_dir), input_sha256=hashes,
        input_mode="fixed_corpus" if package else "sampled",
        corpus_manifest_sha256=file_hash(dataset_dir / "manifest.json") if package else None,
        construction_seed=package["seed"] if package else None,
        sample_seed=None if package else args.seed,
        sample_ids=[q["id"] for q in runtime_questions],
        question_count=len(questions), passage_count=len(passages),
    )
    return runtime_questions, passages, inputs


def synchronize_cuda() -> None:
    """等待 GPU 操作完成，保证墙钟计时包含实际计算。"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(function, *args):
    """在 GPU 同步边界内执行调用并返回结果及耗时。"""
    synchronize_cuda()
    started = time.perf_counter()
    result = function(*args)
    synchronize_cuda()
    return result, time.perf_counter() - started


def validate_retrieval(results, questions, passages) -> None:
    """核对共同问题顺序、Top-5 和文档归属。"""
    if len(results) != len(questions):
        raise AssertionError("Retrieval result count differs from questions")
    corpus = set(passages)
    for result, question in zip(results, questions):
        if str(result.get("id", question["id"])) != question["id"] or result["question"] != question["question"]:
            raise AssertionError("Retrieval question order/text differs from inputs")
        if len(result["sorted_passage"]) != 5 or len(result["sorted_passage_scores"]) != 5:
            raise AssertionError("Each question must return five passages and scores")
        if any(passage not in corpus for passage in result["sorted_passage"]):
            raise AssertionError("Retrieved passage is outside the shared corpus")


def environment_snapshot() -> dict:
    """记录实际运行环境，准备阶段不调用。"""
    import torch
    import spacy

    return dict(
        platform=platform.platform(), python=platform.python_version(),
        torch=torch.__version__, spacy=spacy.__version__,
        cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )


def run_linearrag(args, questions, passages, dataset_name, output_dir):
    """分别测量初始化、索引和预热后的真实 BFS 检索。"""
    import statistics
    from sentence_transformers import SentenceTransformer
    from src.linearrag.config import LinearRAGConfig
    from src.linearrag.LinearRAG import LinearRAG

    cache_root = args.resume_cache or output_dir / "cache"
    reused = (cache_root / dataset_name / "ner_results.json").exists()
    if args.resume_cache and not reused:
        raise FileNotFoundError(f"Completed cache not found: {cache_root}")
    # 中断建库也可能已有部分缓存，不能标记为完整冷索引。
    cache_present = cache_root.exists() and any(cache_root.iterdir())
    no_llm = NoLLM()
    synchronize_cuda()
    started = time.perf_counter()
    encoder = SentenceTransformer(str(args.embedding_model), device="cuda")
    model = LinearRAG(global_config=LinearRAGConfig(
        dataset_name=dataset_name, embedding_model=encoder, llm_model=no_llm,
        working_dir=cache_root, max_workers=args.max_workers, retrieval_top_k=5,
        use_vectorized_retrieval=False, **DATASET_PARAMETERS,
    ))
    synchronize_cuda()
    setup_seconds = time.perf_counter() - started

    _, index_seconds = timed_call(model.index, passages)
    first_results, first_seconds = timed_call(model.retrieve, questions)
    validate_retrieval(first_results, questions, passages)
    validate_retrieval(model.retrieve(questions), questions, passages)
    runs = []
    for repeat in range(1, args.retrieval_repeats + 1):
        results, seconds = timed_call(model.retrieve, questions)
        validate_retrieval(results, questions, passages)
        runs.append(dict(repeat=repeat, total_seconds=seconds, seconds_per_question=seconds / len(questions)))

    for result, question in zip(results, questions):
        result["id"] = question["id"]

    measurement = dict(
        status="passed", setup_seconds=setup_seconds, index_seconds=index_seconds,
        index_mode="cache reuse or recovery" if cache_present else "cold indexing",
        index_boundary="LinearRAG.index; model setup excluded",
        first_query_seconds_per_question=first_seconds / len(questions),
        retrieval_runs=runs,
        retrieval_seconds_per_question_median=statistics.median(row["seconds_per_question"] for row in runs),
        retrieval_boundary="warm index; actual BFS retrieval, one untimed batch warmup",
        llm_calls_during_index_and_retrieval=no_llm.calls, prompt_tokens=0, completion_tokens=0,
        validation=dict(result_count=len(results), passages_per_result=5, default_bfs=True,
                        resumed_cache=cache_present, isolated_cache=args.resume_cache is None),
    )
    return results, measurement


def run_hipporag(args, questions, passages, output_dir):
    """调用外部驱动测量；隔离索引缓存，并读取实际分阶段 token。"""
    from src.baselines.hipporag import HippoRAG

    model = HippoRAG(
        embedding_model_path=args.embedding_model, extraction_model=args.hipporag_extraction_model,
        max_workers=args.hipporag_workers, retrieval_top_k=5,
        working_dir=output_dir.parent, experiment_id=output_dir.name,
        cache_dir=output_dir / "cache", damping=0.5, sim_threshold=0.8,
    )
    model.index(passages)
    results, measurement = model.measure_efficiency(questions, args.retrieval_repeats)
    validate_retrieval(results, questions, passages)
    measurement["token_usage"] = json.loads((output_dir / "token_usage.json").read_text(encoding="utf-8"))
    measurement["stage_timing"] = json.loads((output_dir / "timing.json").read_text(encoding="utf-8"))
    measurement["adapter_timing"] = json.loads((output_dir / "adapter_timing.json").read_text(encoding="utf-8"))
    return results, measurement


def main() -> int:
    """固定输入与测量协议，按参数准备或运行各方法。"""
    args = parse_args()
    questions, passages, inputs = load_inputs(args)
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q2_efficiency_analysis" / experiment_id
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    if args.resume and not output_dir.exists():
        raise FileNotFoundError(f"Cannot resume missing run directory: {output_dir}")

    code_paths = [Path(__file__), PROJECT_ROOT / "src/baselines/hipporag.py",
                  PROJECT_ROOT / "experiments/hipporag/official_runner.py"]
    manifest = dict(
        research_question="Q2 Efficiency Analysis", scope="bounded local efficiency supplement",
        experiment_id=experiment_id, inputs=inputs, methods=args.methods,
        embedding_model=str(args.embedding_model), linearrag_parameters=DATASET_PARAMETERS,
        retrieval_top_k=5, retrieval_path="official BFS", warmup_runs=1,
        timed_retrieval_repeats=args.retrieval_repeats, max_workers=args.max_workers,
        hipporag_extraction_model=args.hipporag_extraction_model, hipporag_workers=args.hipporag_workers,
        hipporag_retrieval_config=dict(damping=0.5, sim_threshold=0.8),
        resume_cache=str(args.resume_cache) if args.resume_cache else None,
        code_sha256={str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in code_paths},
        generation_and_evaluation=False,
    )
    if args.resume:
        previous = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Resume requires identical inputs, methods, code and measurement configuration")
    else:
        write_json(output_dir / "manifest.json", manifest)
        write_json(output_dir / "samples.json", dict(questions=questions, passages=passages))

    if args.prepare_only:
        write_json(output_dir / "preparation.json", dict(
            status="prepared_not_executed", question_count=len(questions), passage_count=len(passages),
            logical_api_calls_if_fresh=dict(
                indexing=2 * len(passages) if "hipporag" in args.methods else 0,
                query_ner=len(questions) if "hipporag" in args.methods else 0,
                generation=0, evaluation=0,
            ),
            cost_scope="Successful extraction/query-NER calls only; no retries or currency estimate",
            timing_scope="Fresh index; first HippoRAG NER+ranking; warm real retrieval repeats",
        ))
        print(f"Prepared only: {output_dir}")
        return 0

    from src.common.utils import setup_logging
    setup_logging(str(output_dir / "experiment.log"))
    write_json(output_dir / "environment.json", environment_snapshot())
    measurements = {}
    for method in args.methods:
        method_dir = output_dir / method
        saved = method_dir / "measurements.json"
        if args.resume and saved.exists():
            results = json.loads((method_dir / "retrieval_results.json").read_text(encoding="utf-8"))
            validate_retrieval(results, questions, passages)
            measurements[method] = json.loads(saved.read_text(encoding="utf-8"))
            continue
        method_dir.mkdir(parents=True, exist_ok=True)
        print(f"Measuring {method}", flush=True)
        if method == "linearrag":
            results, measurement = run_linearrag(args, questions, passages, inputs["dataset"], method_dir)
        else:
            results, measurement = run_hipporag(args, questions, passages, method_dir)
        write_json(method_dir / "retrieval_results.json", results)
        write_json(saved, measurement)
        measurements[method] = measurement

    write_json(output_dir / "summary.json", dict(
        status="passed", inputs=inputs, methods=measurements,
        accuracy_source="Use Q1 results only when input hashes and method configurations match",
    ))
    print(f"Evidence written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
