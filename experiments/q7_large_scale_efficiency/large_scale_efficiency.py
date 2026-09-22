"""Run a bounded cold-index scalability measurement for LinearRAG Q7."""

from __future__ import annotations

import argparse
import json
import platform
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
from src.paths import DATASETS_DIR, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.utils import setup_logging


SOURCE_DATASET = "hotpotqa"
DATASET_PARAMETERS = {
    "spacy_model": "en_core_web_trf",
    "max_iterations": 3,
    "passage_ratio": 0.05,
    "iteration_threshold": 0.4,
    "top_k_sentence": 1,
}


class NoLLM:
    """Fail if cold indexing unexpectedly reaches a generative model."""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, _messages: Any) -> str:
        self.calls += 1
        raise AssertionError("Q7 indexing must not call a generative LLM")

# 收集参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded LinearRAG Q7 cold-index experiment")
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--source-dataset", action="append", dest="source_datasets")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--experiment-id", default=None)
    args = parser.parse_args()
    if not args.source_datasets:
        args.source_datasets = [SOURCE_DATASET]
    if args.target_tokens <= 0:
        parser.error("--target-tokens must be greater than 0")
    return args


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=json_default)

# 拿到原始语料
def load_chunks(dataset_names: list[str]) -> list[str]:
    combined: list[str] = []
    for dataset_name in dataset_names:
        chunks_path = DATASETS_DIR / dataset_name / "chunks.json"
        with chunks_path.open(encoding="utf-8") as stream:
            chunks = json.load(stream)
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise TypeError(f"Expected a JSON list of strings: {chunks_path}")
        combined.extend(chunks)
    return combined

# 从语料开头依次取 passage，直到达到目标 token 数
def build_prefix_subset(chunks: list[str], tokenizer: Any, target_tokens: int) -> tuple[list[str], int]:
    """Build a deterministic prefix, truncating only its final passage."""
    subset: list[str] = []
    token_count = 0

    for chunk in chunks:
        token_ids = tokenizer.encode(chunk, add_special_tokens=False)
        remaining = target_tokens - token_count
        if len(token_ids) <= remaining:
            subset.append(chunk)
            token_count += len(token_ids)
        else:
            if remaining:
                subset.append(tokenizer.decode(token_ids[:remaining], skip_special_tokens=True))
                token_count += remaining
            break

        if token_count == target_tokens:
            break

    if token_count != target_tokens:
        raise ValueError(
            f"Source corpus contains only {token_count} usable tokens; requested {target_tokens}"
        )
    return subset, token_count


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_call(function: Any, *args: Any) -> tuple[Any, float]:
    synchronize_cuda()
    started = time.perf_counter()
    result = function(*args)
    synchronize_cuda()
    return result, time.perf_counter() - started


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def environment_snapshot() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "spacy": spacy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> int:
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = EXPERIMENT_RESULTS_DIR / "q7_large_scale_efficiency" / experiment_id
    if output_dir.exists():
        raise FileExistsError(f"Run directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    setup_logging(str(output_dir / "experiment.log"))

    cache_root = output_dir / "cache"
    if cache_root.exists():
        raise AssertionError(f"Cold-cache directory must not exist before setup: {cache_root}")

    setup_started = time.perf_counter()
    embedding_model = SentenceTransformer(str(args.embedding_model), device="cuda")
    source_chunks = load_chunks(args.source_datasets)
    subset, measured_tokens = build_prefix_subset(
        source_chunks, embedding_model.tokenizer, args.target_tokens
    )
    passages = [f"{index}:{chunk}" for index, chunk in enumerate(subset)]

    no_llm = NoLLM()
    source_label = "_".join(args.source_datasets)
    dataset_name = f"q7_{source_label}_{args.target_tokens}t"
    config = LinearRAGConfig(
        dataset_name=dataset_name,
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

    manifest = {
        "research_question": "Q7 Large-scale Efficiency Analysis",
        "scope": "Bounded LinearRAG-only scale-proxy cold-index experiment",
        "experiment_id": experiment_id,
        "source_datasets": args.source_datasets,
        "paper_dataset": "ATLAS-Wiki (not used locally)",
        "subset_rule": "deterministic corpus prefix; only the final passage may be token-truncated",
        "tokenizer": str(args.embedding_model.resolve()),
        "target_tokens": args.target_tokens,
        "measured_tokens": measured_tokens,
        "passage_count": len(passages),
        "embedding_model": str(args.embedding_model.resolve()),
        "cold_cache": True,
        "resumed_from_completed_cache": False,
        "cache_root": str(cache_root.resolve()),
        "dataset_parameters": DATASET_PARAMETERS,
        "timing_boundary": "LinearRAG.index(passages), including embedding, NER, graph construction, and cache writes",
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "environment.json", environment_snapshot())

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    print(
        f"[index] cold indexing {len(passages)} passages / {measured_tokens} tokens",
        flush=True,
    )
    _, index_seconds = timed_call(model.index, passages)

    if no_llm.calls != 0:
        raise AssertionError(f"Unexpected LLM calls: {no_llm.calls}")

    cache_dataset_dir = cache_root / dataset_name
    measurements = {
        "status": "passed",
        "setup_seconds": setup_seconds,
        "index_seconds": index_seconds,
        "seconds_per_million_tokens": index_seconds / measured_tokens * 1_000_000,
        "llm_calls_during_index": no_llm.calls,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "passage_count": len(model.passage_embedding_store.texts),
        "entity_count": len(model.entity_embedding_store.texts),
        "sentence_count": len(model.sentence_embedding_store.texts),
        "graph_vertex_count": model.graph.vcount(),
        "graph_edge_count": model.graph.ecount(),
        "cache_bytes": directory_size(cache_dataset_dir),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        ),
        "validation": {
            "isolated_cache": True,
            "resumed_cache": False,
            "measured_tokens_equal_target": measured_tokens == args.target_tokens,
            "generative_llm_disabled": True,
        },
    }
    write_json(output_dir / "measurements.json", measurements)
    print(json.dumps(measurements, ensure_ascii=False, indent=2), flush=True)
    print(f"Evidence written to: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
