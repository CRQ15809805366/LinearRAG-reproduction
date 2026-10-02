"""读取准备好的规模输入包，测量 Q7 的运行初始化与原冷索引边界。"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.methods.linear.config import LinearRAGConfig
from src.methods.linear.LinearRAG import LinearRAG
from src.paths import PROJECT_ROOT, EXPERIMENT_RESULTS_DIR, MODELS_DIR
from src.common.utils import setup_logging
from src.datasets.input_package import load_package, input_record, resolve_path
from src.datasets.build_corpus import file_hash


DATASET_PARAMETERS = {
    "spacy_model": "en_core_web_trf",
    "max_iterations": 3,
    "passage_ratio": 0.05,
    "iteration_threshold": 0.4,
    "top_k_sentence": 1,
}


from src.paths import DATASETS_DIR, DERIVED_CORPORA_DIR
from src.datasets.input_package import save_package


def load_chunks(dataset_names: list[str]) -> list[str]:
    """按原数据集顺序合并原始片段，不改写文本。"""
    combined: list[str] = []
    for dataset_name in dataset_names:
        chunks_path = DATASETS_DIR / dataset_name / "chunks.json"
        with chunks_path.open(encoding="utf-8") as stream:
            chunks = json.load(stream)
        if not isinstance(chunks, list) or not all(isinstance(chunk, str) for chunk in chunks):
            raise TypeError(f"Expected a JSON list of strings: {chunks_path}")
        combined.extend(chunks)
    return combined


def build_prefix_subset(chunks: list[str], tokenizer: Any, target_tokens: int) -> tuple[list[str], int]:
    """沿用原前缀规则，只截断最后一段，按原 encode 预算记录 token 数。"""
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


def load_tokenizer(model_path):
    """按现有 SentenceTransformer 的 Transformer 配置加载同一分词器。"""
    from transformers import AutoTokenizer

    transformer_path = model_path
    modules_path = model_path / "modules.json"
    if modules_path.exists():
        modules = json.loads(modules_path.read_text(encoding="utf-8"))
        if modules[0]["type"] != "sentence_transformers.models.Transformer":
            raise ValueError("Q7 preparation requires the original Transformer tokenizer route")
        transformer_path = model_path / modules[0]["path"]
    config = {}
    for name in ("sentence_bert_config.json", "sentence_roberta_config.json",
                 "sentence_distilbert_config.json", "sentence_camembert_config.json",
                 "sentence_albert_config.json", "sentence_xlm-roberta_config.json",
                 "sentence_xlnet_config.json"):
        path = transformer_path / name
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8"))
            break
    tokenizer_path = config.get("tokenizer_name_or_path") or transformer_path
    tokenizer_path = resolve_path(tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=True, **config.get("tokenizer_args", {})
    )
    names = {"modules.json", "config.json", "tokenizer_config.json", "tokenizer.json",
             "special_tokens_map.json", "added_tokens.json", "vocab.txt", "vocab.json",
             "merges.txt", "tokenizer.model", "spiece.model", "sentencepiece.bpe.model"}
    files = {str(path.resolve()): file_hash(path)
             for directory in {model_path, transformer_path, tokenizer_path}
             for path in directory.iterdir()
             if path.is_file() and (path.name in names or path.name.startswith("sentence_"))}
    return tokenizer, files


def build_scale_inputs(args):
    """构造各目标 token 规模的输入包，供当前实验直接使用。"""
    paths = []
    started = time.perf_counter()
    model_path = resolve_path(args.embedding_model)
    tokenizer, tokenizer_files = load_tokenizer(model_path)
    chunks = load_chunks(args.source_datasets)
    sources = []
    for dataset in args.source_datasets:
        path = DATASETS_DIR / dataset / "chunks.json"
        sources.append(dict(dataset=dataset, chunks_path=str(path), chunks_sha256=file_hash(path),
                            passage_count=len(json.loads(path.read_text(encoding="utf-8")))))
    shared_seconds = time.perf_counter() - started

    for target in args.target_tokens:
        subset_started = time.perf_counter()
        subset, measured_tokens = build_prefix_subset(chunks, tokenizer, target)
        preparation_seconds = shared_seconds + time.perf_counter() - subset_started
        source_label = "_".join(args.source_datasets)
        corpus_id = args.corpus_id or f"q7-{source_label}-{target}t"
        if args.corpus_id and len(args.target_tokens) > 1:
            corpus_id += f"-{target}t"
        manifest = dict(
            prepared_for="q7", source_dataset=source_label, source_datasets=args.source_datasets,
            source_files=sources, dataset_name=f"q7_{source_label}_{target}t",
            target_tokens=target, measured_tokens=measured_tokens,
            tokenizer=str(model_path), tokenizer_files_sha256=tokenizer_files,
            subset_rule="deterministic corpus prefix; only the final passage may be token-truncated",
            token_count_rule="encode without special tokens; final passage counts retained token IDs before decode",
            source_mapping="concatenated source order; output passage i comes from combined source chunk i",
            input_preparation_seconds=preparation_seconds,
            shared_source_and_tokenizer_seconds=shared_seconds,
            input_preparation_boundary="source loading, tokenizer loading, provenance hashes and prefix truncation; package writes excluded; shared loading attributed to each scale",
            builder_sha256=file_hash(Path(__file__)),
        )
        output = DERIVED_CORPORA_DIR / corpus_id
        if output.exists():
            _, _, existing = load_package(output)
            for key in ("source_files", "target_tokens", "tokenizer_files_sha256", "subset_rule"):
                if existing.get(key) != manifest[key]:
                    raise ValueError(f"Existing scale package differs: {output} ({key})")
        else:
            output = save_package(corpus_id, subset, None, manifest)
        paths.append(output)
    return paths


class NoLLM:
    """Fail if cold indexing unexpectedly reaches a generative model."""

    def __init__(self) -> None:
        self.calls = 0

    def infer(self, _messages: Any) -> str:
        self.calls += 1
        raise AssertionError("Q7 indexing must not call a generative LLM")

# 收集参数
def parse_args() -> argparse.Namespace:
    """解析规模输入包和运行参数。"""
    parser = argparse.ArgumentParser(description="Bounded LinearRAG Q7 cold-index experiment")
    parser.add_argument("--corpus-dir", type=Path, nargs="+",
                        help="实验输入包目录。")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--embedding-model", type=Path, default=MODELS_DIR / "all-mpnet-base-v2")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--target-tokens", type=int, nargs="+")
    parser.add_argument("--source-dataset", action="append", dest="source_datasets")
    parser.add_argument("--corpus-id")
    args = parser.parse_args()
    if not args.corpus_dir:
        if not args.target_tokens or min(args.target_tokens) <= 0:
            parser.error("Provide positive --target-tokens, or use --corpus-dir")
        if len(set(args.target_tokens)) != len(args.target_tokens):
            parser.error("target token counts must be unique")
        args.source_datasets = args.source_datasets or ["hotpotqa"]
        args.corpus_dir = build_scale_inputs(args)
    args.embedding_model = resolve_path(args.embedding_model)
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


def run_scale(args, corpus_dir, experiment_id) -> int:
    """读取一个规模包，只对运行初始化和原 index 边界分别计时。"""
    _, passages, package = load_package(corpus_dir, require_questions=False)
    if package.get("prepared_for") != "q7" or package["target_tokens"] != package["measured_tokens"]:
        raise ValueError("Q7 requires a prepared scale package with the original exact token budget")
    if resolve_path(package["tokenizer"]) != args.embedding_model:
        raise ValueError("Q7 embedding model must match the preparation tokenizer source")
    for path, digest in package["tokenizer_files_sha256"].items():
        if file_hash(resolve_path(path)) != digest:
            raise ValueError("Q7 tokenizer files changed since preparation")
    measured_tokens = package["measured_tokens"]
    target_tokens = package["target_tokens"]
    if target_tokens <= 0:
        raise ValueError("Q7 token count must be positive")
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
    no_llm = NoLLM()
    dataset_name = package["dataset_name"]
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
        "source_datasets": package["source_datasets"],
        "paper_dataset": "ATLAS-Wiki (not used locally)",
        "subset_rule": "deterministic corpus prefix; only the final passage may be token-truncated",
        "tokenizer": package["tokenizer"],
        "input_package": input_record(corpus_dir, package),
        "input_preparation_seconds": package["input_preparation_seconds"],
        "input_preparation_boundary": package["input_preparation_boundary"],
        "setup_timing_boundary": "SentenceTransformer and LinearRAG initialization only; input reading/checking and preparation excluded",
        "setup_history_comparison": "Not directly comparable with historical setup_seconds that included source loading and truncation",
        "target_tokens": target_tokens,
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
        "input_preparation_seconds": package["input_preparation_seconds"],
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
            "measured_tokens_equal_target": measured_tokens == target_tokens,
            "generative_llm_disabled": True,
        },
    }
    write_json(output_dir / "measurements.json", measurements)
    print(json.dumps(measurements, ensure_ascii=False, indent=2), flush=True)
    print(f"Evidence written to: {output_dir}", flush=True)
    return 0


def main() -> int:
    """依输入包顺序测量各规模，分别保存冷索引结果，不重新构造语料。"""
    args = parse_args()
    experiment_id = args.experiment_id or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for index, corpus_dir in enumerate(args.corpus_dir):
        run_id = experiment_id if len(args.corpus_dir) == 1 else f"{experiment_id}-{index + 1}"
        run_scale(args, corpus_dir, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
