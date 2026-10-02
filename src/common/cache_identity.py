"""统一缓存的阶段身份、模型内容指纹和复用条件，不读取 API 密钥。"""

from functools import lru_cache
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil

from src.paths import PROJECT_ROOT, MODELS_DIR


# 仅在对应阶段的语义发生变化时升级协议，不随格式、日志或计时修改失效。
HIPPO_PIN = "b144c46df14cabe5f5822d8caded4bec5f709461"
PROTOCOLS = {
    "linear_extraction": "spacy-entities-sentences-v1",
    "linear_embedding": "sentence-transformer-normalized-float32-v1",
    "linear_graph": "entity-occurrence-adjacent-passages-v1",
    "linear_query": "seed-propagation-ppr-v1",
    "hippo_extraction": "official-ner-openie-json-or-literal-v1",
    "hippo_graph": "official-entities-only-synonyms-mean-v1",
    "hippo_query": "official-question-ner-ppr-v1",
    "hippo_question_ner": "official-question-ner-v1",
    "hippo_observation": "focus-one-hop-full-ranking-v1",
}


def content_identity(value):
    """对规范化 JSON 计算完整 SHA-256；不使用文件位置或代码排版。"""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def file_digest(path):
    """分块读取文件，避免模型权重整份进入内存。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name, site_packages=None):
    """记录实际环境中的结果相关依赖版本。"""
    if site_packages is not None:
        for distribution in metadata.Distribution.discover(name=name, path=[str(site_packages)]):
            return distribution.version
        return "unavailable"
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


@lru_cache(maxsize=16)
def _model_files_identity(files):
    """同一进程内复用未变更模型文件的内容摘要。"""
    return content_identity([(relative, file_digest(path))
                             for relative, path, size, modified in files])


def embedding_identity(model, backend="sentence_transformers"):
    """识别实际模型权重、tokenizer 和编码配置；迁移路径不改变身份。"""
    if isinstance(model, (str, Path)):
        model_path = Path(model)
        max_length = None
    else:
        first = model._first_module()
        model_path = Path(first.auto_model.config._name_or_path)
        max_length = model.max_seq_length
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.is_dir():
        local = MODELS_DIR / model_path.name
        if local.is_dir():
            model_path = local
        else:
            raise ValueError(f"Cache identity requires a local model snapshot: {model_path}")
    # 只纳入 PyTorch/SentenceTransformer 实际消费的文件，排除 .git、说明和导出格式。
    files = []
    tokenizer_files = {"config.json", "tokenizer_config.json", "tokenizer.json",
                       "special_tokens_map.json", "added_tokens.json", "vocab.txt",
                       "vocab.json", "merges.txt", "tokenizer.model", "spiece.model",
                       "sentencepiece.bpe.model", "model.safetensors.index.json",
                       "pytorch_model.bin.index.json"}
    has_safetensors = any(model_path.glob("*.safetensors"))
    for path in sorted(model_path.rglob("*")):
        relative = path.relative_to(model_path)
        if any(part in (".git", "onnx", "openvino", "__pycache__") for part in relative.parts):
            continue
        if path.suffix == ".bin" and has_safetensors:
            continue
        used = path.name in tokenizer_files or path.suffix in (".safetensors", ".bin")
        if backend == "huggingface_mean" and len(relative.parts) != 1:
            used = False
        if backend == "sentence_transformers":
            used = used or path.name in ("modules.json", "config_sentence_transformers.json", "sentence_bert_config.json")
        if path.is_file() and used:
            stat = path.stat()
            files.append((relative.as_posix(), str(path), stat.st_size, stat.st_mtime_ns))
    if max_length is None:
        config = model_path / "sentence_bert_config.json"
        if config.exists():
            max_length = json.loads(config.read_text(encoding="utf-8")).get("max_seq_length")
    identity = dict(snapshot_sha256=_model_files_identity(tuple(files)), backend=backend)
    if backend == "sentence_transformers":
        identity.update(max_seq_length=max_length, normalize_embeddings=True,
                        transformers=package_version("transformers"),
                        sentence_transformers=package_version("sentence-transformers"))
    return identity


def corpus_identity(passages):
    """仅用实际进入索引的有序文本定义语料，不纳入问题或答案。"""
    return content_identity(list(passages))[:20]


def stage_manifest(stage, inputs):
    """为阶段保存可解释的输入描述和稳定身份。"""
    identity = content_identity(dict(stage=stage, inputs=inputs))
    return dict(id=identity[:20], sha256=identity, inputs=inputs)


def linear_stages(passages, embedding_model, spacy_model):
    """分别定义 LinearRAG 抽取、向量和图身份；传播参数只属于查询。"""
    corpus = dict(id=corpus_identity(passages), count=len(passages))
    extraction = stage_manifest("linear_extraction", dict(
        corpus=corpus, protocol=PROTOCOLS["linear_extraction"],
        spacy_model=spacy_model, model_version=package_version(spacy_model),
        spacy_version=package_version("spacy"),
    ))
    embedding = stage_manifest("linear_embedding", dict(
        corpus=corpus, model=embedding_identity(embedding_model),
        protocol=PROTOCOLS["linear_embedding"],
    ))
    graph = stage_manifest("linear_graph", dict(
        extraction_id=extraction["id"], protocol=PROTOCOLS["linear_graph"],
    ))
    return dict(extraction=extraction, embedding=embedding, graph=graph)


def write_manifest(directory, value):
    """原子保存 manifest，不覆盖半写状态。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def prepare_linear_cache(directory, passages, model, spacy_model=None):
    """匹配阶段 manifest，隔离不兼容向量，并复用兼容的已有 NER。"""
    directory = Path(directory)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    desired = linear_stages(passages, model, spacy_model) if spacy_model else {
        "embedding": stage_manifest("linear_embedding", dict(
            corpus=dict(id=corpus_identity(passages), count=len(passages)),
            model=embedding_identity(model), protocol=PROTOCOLS["linear_embedding"],
        )),
    }
    candidates = [directory] + sorted((directory / "variants").glob("*/"))
    for candidate in candidates:
        path = candidate / "manifest.json"
        if not path.exists():
            continue
        existing = json.loads(path.read_text(encoding="utf-8"))
        if all(existing.get("stages", {}).get(name, {}).get("id") == stage["id"]
               for name, stage in desired.items()):
            return candidate
        if (spacy_model and set(existing.get("stages", {})) == {"embedding"}
                and existing["stages"]["embedding"]["id"] == desired["embedding"]["id"]
                and not (candidate / "ner_results.json").exists()):
            existing["stages"] = desired
            write_manifest(candidate, existing)
            return candidate

    occupied = directory.exists() and any(directory.iterdir())
    target = directory / "variants" / content_identity(desired)[:20] if occupied else directory
    if spacy_model:
        for candidate in candidates:
            path = candidate / "manifest.json"
            ner = candidate / "ner_results.json"
            if not path.exists() or not ner.exists():
                continue
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("stages", {}).get("extraction", {}).get("id") == desired["extraction"]["id"]:
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ner, target / ner.name)
                break
    write_manifest(target, dict(schema=2, method="linearrag", stages=desired,
                               reuse="Match each stage ID; query/answer/worker settings do not invalidate indexing."))
    return target


def linear_query_identity(stages, questions, config, variant="full"):
    """记录检索所需输入，明确 BFS 与 vectorized 路径，不缓存生成答案。"""
    names = ("retrieval_top_k", "max_iterations", "top_k_sentence", "passage_ratio",
             "passage_node_weight", "damping", "iteration_threshold",
             "use_vectorized_retrieval", "enable_hybrid_attribute_fallback")
    settings = {name: getattr(config, name) for name in names}
    if variant == "without_entity_activation":
        for name in ("max_iterations", "top_k_sentence", "iteration_threshold"):
            settings.pop(name)
    if variant == "without_global_importance":
        settings.pop("damping")
    if config.enable_hybrid_attribute_fallback:
        settings.update(attribute_keyword_boost=config.attribute_keyword_boost,
                        attribute_query_keywords=config.attribute_query_keywords)
    return stage_manifest("linear_query", dict(
        graph_id=stages["graph"]["id"], embedding_id=stages["embedding"]["id"],
        questions=[q["question"] for q in questions], config=settings, variant=variant,
        protocol=PROTOCOLS["linear_query"],
    ))


def hippo_stages(passages, config):
    """分离官方抽取和建图身份，不纳入 runner 文件哈希、路径或线程数。"""
    external_root = Path(config.get("external_root", PROJECT_ROOT.parent / "HippoRAG-2024"))
    external_packages = external_root / ".venv/Lib/site-packages"
    extraction = stage_manifest("hippo_extraction", dict(
        corpus=dict(id=corpus_identity(passages), count=len(passages)),
        model=config["extraction_model"], official_commit=HIPPO_PIN,
        provider_scope=config.get("provider_scope", provider_scope()),
        protocol=PROTOCOLS["hippo_extraction"],
        sampling=dict(temperature=0,
                      enable_thinking=False if config["extraction_model"] == "qwen3.8-flash" else None),
    ))
    graph = stage_manifest("hippo_graph", dict(
        extraction_id=extraction["id"], model=embedding_identity(config["embedding_model_path"], "huggingface_mean"),
        encoder_runtime={name: package_version(name, external_packages)
                         for name in ("transformers", "torch", "faiss-cpu")},
        sim_threshold=config["sim_threshold"], pool_method="mean",
        entities_only=True, official_commit=HIPPO_PIN,
        protocol=PROTOCOLS["hippo_graph"],
    ))
    return dict(extraction=extraction, graph=graph)


def hippo_query_manifest(index_id, questions, config, observation=None):
    """查询身份只含图、问题 ID/文本、排序配置和所需观察产物。"""
    inputs = dict(index_id=index_id,
                  question_ner_id=hippo_question_ner_manifest(config)["id"],
                  questions=[dict(id=q["id"], question=q["question"]) for q in questions],
                  top_k=config["top_k"], damping=config["damping"],
                  doc_ensemble=False, dpr_only=False, protocol=PROTOCOLS["hippo_query"])
    if observation is not None:
        inputs["case_observation"] = observation
        inputs["observation_protocol"] = PROTOCOLS["hippo_observation"]
    return dict(schema=2, stage="query", **stage_manifest("hippo_query", inputs),
                reuse="Exact query identity; answers, timing repeats and workers are excluded.")


def hippo_question_ner_manifest(config):
    """问题 NER 按文本逐条复用，不依赖图、embedding、Top-k 或语料。"""
    return dict(schema=2, stage="question_ner", **stage_manifest("hippo_question_ner", dict(
        model=config["extraction_model"], official_commit=HIPPO_PIN,
        provider_scope=config.get("provider_scope", provider_scope()),
        protocol=PROTOCOLS["hippo_question_ner"], temperature=0,
        enable_thinking=False if config["extraction_model"] == "qwen3.8-flash" else None,
    )), reuse="Checkpoint keys are exact question texts; match model and NER protocol.")


def provider_scope():
    """仅对端点标识生成摘要；不读取或记录 API key，不改变进程环境。"""
    path = PROJECT_ROOT / ".env.local"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "OPENAI_BASE_URL":
                return content_identity(value.strip().strip("\"'").rstrip("/"))
    return "unrecorded"
