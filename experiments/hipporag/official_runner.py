"""在共享索引目录运行官方 HippoRAG，将查询缓存和实验统计分开保存。"""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

# 实验声称使用 HippoRAG 2024 官方实现时，实际执行的 checkout 必须就是指定 commit
PIN = "b144c46df14cabe5f5822d8caded4bec5f709461"
BOOT_TIME = time.perf_counter()
BOOT_WALL_NS = time.time_ns()

from langchain_core.callbacks import BaseCallbackHandler


def save(path, value): # 写 tmp → replace, 避免 checkpoint 半写坏
    """原子写入检查点或统计文件，保留已有文件格式。"""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


class TimedLoader:
    """记录本次子进程的模型与 tokenizer 加载耗时。"""

    def __init__(self, loader, loading):
        """绑定原加载器和共享计时字典。"""
        self.loader = loader
        self.loading = loading

    def __call__(self, *args, **kwargs):
        """执行原加载器，并累计时间与调用次数。"""
        started = time.perf_counter()
        model = self.loader(*args, **kwargs)
        self.loading["seconds"] += time.perf_counter() - started
        self.loading["calls"] += 1
        return model


class UsageRecorder(BaseCallbackHandler):
    """当抽取 worker 完成时，安全地追加实际的 API 用量。"""

    raise_error = True

    def __init__(self, model_name, usage_path):
        """将实际 API usage 追加到当前阶段的缓存目录。"""
        self.model_name = model_name
        self.usage_path = usage_path
        self.lock = threading.Lock()

    def on_llm_end(self, response, **kwargs):
        """按原始响应记录成功调用的实际 token，线程间串行写入。"""
        tokens = (response.llm_output or {}).get("token_usage")
        if not tokens:
            raise RuntimeError("LLM endpoint returned no actual token usage")

        with self.lock, self.usage_path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(dict(model=self.model_name, usage=tokens)) + "\n"
            )


class ClientFactory:
    """为官方客户端钩子保持端点配置显式。"""

    def __init__(self, model_name, credentials, usage_recorder):
        """保存客户端配置和分阶段 usage 记录器。"""
        self.model_name = model_name
        self.credentials = credentials
        self.usage_recorder = usage_recorder

    def __call__(self, provider, name, **kwargs):
        """创建官方函数使用的兼容客户端，保持原提示和模型参数。"""
        import httpx
        from langchain_openai import ChatOpenAI

        if provider != "openai":
            raise ValueError(
                "This bounded bridge uses the project's OpenAI-compatible endpoint"
            )

        return ChatOpenAI(
            model=self.model_name,
            api_key=self.credentials["OPENAI_API_KEY"],
            base_url=self.credentials["OPENAI_BASE_URL"],
            temperature=0,
            max_retries=0,
            timeout=180,
            callbacks=[self.usage_recorder],
            http_client=httpx.Client(trust_env=True, timeout=180),
            model_kwargs=(
                {"extra_body": {"enable_thinking": False}}
                if self.model_name == "qwen3.8-flash"
                else {}
            ),
        )


def extract_passage(row, openie, model_name, np):
    """通过 LLM 对 passage 做 named entity recognition 和 OpenIE triple extraction。"""
    # 先进行命名实体识别
    entities, _ = openie.named_entity_recognition(row["passage"])
    entities = list(np.unique(entities))

    # 再进行 OpenIE 三元组抽取
    triples, _ = openie.openie_post_ner_extract(row["passage"], entities, model_name)
    try:
        parsed = json.loads(triples)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(triples)
    extracted = parsed["triples"]

    # 长度不为 3 的候选由官方 create_graph 记录并丢弃，不在适配层阻断整批。
    if not isinstance(extracted, list) or any(
        not isinstance(triple, (list, tuple))
        for triple in extracted
    ):
        raise ValueError(f"Malformed triples for passage {row['idx']}")

    return dict(row, extracted_entities=entities, extracted_triples=extracted)


def prepare_and_extract_passages(request, dataset, model_name, openie, np, timings):
    """转换输入，并仅抽取缓存中缺失的段落。"""
    start = time.perf_counter()
    passages = request["passages"]
    corpus = []

    # passage 在进入 HippoRAG 前被格式化
    for i, passage in enumerate(passages):
        title, separator, text = passage.partition("\n")
        if not separator:
            title, text = "", passage
        corpus.append(dict(idx=i, title=title, text=text, passage=passage)) # passage 保留原始字符串

    save(f"data/{dataset}_corpus.json", corpus)
    timings["input_conversion_seconds"] = time.perf_counter() - start

    # 实验恢复机制: 断点重跑
    checkpoint_path = Path("extraction_checkpoint.json")
    checkpoint = (
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_path.exists()
        else {}
    )

    # 只对缺少抽取结果的段落调用 LLM。
    missing = [row for row in corpus if str(row["idx"]) not in checkpoint]
    timings["extraction_reused_passages"] = len(corpus) - len(missing)
    start = time.perf_counter()
    failures = []
    with ThreadPoolExecutor(max_workers=request["config"]["max_workers"]) as pool:
        futures = {
            pool.submit(extract_passage, row, openie, model_name, np): str(row["idx"])
            for row in missing
        }
        for future in as_completed(futures):
            if future.cancelled():
                continue
            try:
                checkpoint[futures[future]] = future.result()
            except Exception as error:
                failures.append(error)
                for pending in futures:
                    pending.cancel()
                print(f"Passage {futures[future]} failed: {type(error).__name__}", flush=True)
                continue
            save(checkpoint_path, checkpoint)
            print(f"Extracted {len(checkpoint)}/{len(corpus)} passages", flush=True)
    if failures:
        raise failures[0]
    docs = [checkpoint[str(i)] for i in range(len(corpus))]
    save(
        f"output/openie_{dataset}_results_ner_{model_name}_{len(corpus)}.json",
        dict(docs=docs, ents_by_doc=[row["extracted_entities"] for row in docs]),
    )
    timings["passage_extraction_seconds"] = time.perf_counter() - start


def build_graph(dataset, model_name, retriever, cfg, loading, timings):
    """调用官方建图和实体近邻搜索；完整图存在时复用。"""
    import src.create_graph as graph
    from src.RetrievalModule import RetrievalModule

    graph.args = SimpleNamespace(dataset=dataset)
    graph_flag = Path("graph_complete.json") # 存在时不重新建图
    load_before_graph = loading["seconds"]
    synchronize_cuda()
    start = time.perf_counter()
    if not graph_flag.exists():
        graph.create_graph(
            dataset,
            "ner",
            model_name,
            retriever,
            retriever.replace("/", "_").replace(".", ""),
            cfg["sim_threshold"], # 重要相似度阈值
            False,
            True,
        )

        # 这条流程只从 kb_to_kb 获取“哪些东西是同义的”这种关系，不需要 query 和 relation 的 KNN 文件
        # 故出现双次建图
        RetrievalModule(retriever, "output/kb_to_kb.tsv", pool_method="mean")
        graph.create_graph(
            dataset,
            "ner",
            model_name,
            retriever,
            retriever.replace("/", "_").replace(".", ""),
            cfg["sim_threshold"], # 重要相似度阈值
            True,
            True,
        )
        save(graph_flag, dict(complete=True))
        timings["graph_reused"] = False
    else:
        timings["graph_reused"] = True
    synchronize_cuda()
    timings["graph_index_seconds_including_encoder_load"] = time.perf_counter() - start
    timings["graph_encoder_loading_seconds"] = loading["seconds"] - load_before_graph
    timings["graph_index_seconds_excluding_encoder_load"] = (
        timings["graph_index_seconds_including_encoder_load"]
        - timings["graph_encoder_loading_seconds"]
    )


def observe_case(hippo, observation, ranks, scores, logs):
    """只导出指定实体的一跳图和支持段落排名，不改变官方评分。"""
    import pickle
    from src.processing import processing_phrases

    # 关系字典是建图时的标签；同一实体对的标签可能被后续相似边覆盖。
    relation_path = next(Path("output").glob("*_graph_relation_dict_*.subset.p"))
    with relation_path.open("rb") as stream:
        relations = pickle.load(stream)

    nodes = []
    focus_ids = set()
    # 同时查看实际链接到的起始节点，避免人名变体使诊断漏掉真正的入口。
    focus_names = list(dict.fromkeys(
        observation["entities"] + [item[1] for item in logs.get("linked_node_scores", [])]
    ))
    for name in focus_names:
        normalized = processing_phrases(name)
        node_id = hippo.kb_node_phrase_to_id.get(normalized)
        nodes.append(dict(requested=name, normalized=normalized, node_id=node_id))
        if node_id is not None:
            focus_ids.add(node_id)

    edges = []
    for (source, target), weight in hippo.graph_plus.items():
        if source not in focus_ids and target not in focus_ids:
            continue
        source_name = str(hippo.phrases[source])
        target_name = str(hippo.phrases[target])
        relation = relations.get((source_name, target_name))
        edges.append(dict(
            source=source_name, target=target_name, weight=float(weight),
            relation_label=relation,
            edge_kind=("similarity" if relation == "equivalent" else
                       "relation" if relation is not None else "unlabelled"),
        ))

    # 完整排名只用于定位支持段落；不把所有文档的排名写到诊断文件。
    positions = {int(index): position for position, index in enumerate(ranks)}
    support = []
    for index in observation["support_passage_indices"]:
        position = positions[index]
        row = hippo.extracted_triples[index]
        support.append(dict(
            passage_index=index, rank=position + 1, score=float(scores[position]),
            in_top5=position < 5, passage=hippo.get_passage_by_idx(index),
            extracted_triples=row["extracted_triples"],
        ))

    return dict(
        graph_node_count=hippo.g.vcount(), graph_edge_count=hippo.g.ecount(),
        focus_nodes=nodes, one_hop_edges=edges, support_passages=support,
        relation_label_limit="Pair labels may be overwritten; consult extracted triples too.",
    )


def synchronize_cuda():
    """等待 GPU 操作结束，使阶段耗时包含实际计算。"""
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_queries(hippo, request, client, query_ner, np, timings):
    """记录首次 NER 加排名，并在预热后重复执行缓存 NER 的实际排名。"""
    passages = request["passages"]
    ner_path = Path("query_ner_checkpoint.json")
    ner_cache = json.loads(ner_path.read_text(encoding="utf-8")) if ner_path.exists() else {}
    results = []

    # 首轮保留逐题 NER 状态；中断恢复不能伪装成完整冷查询。
    for question in request["questions"]:
        text = question["question"]
        reused = text in ner_cache
        synchronize_cuda()
        started = time.perf_counter()
        if not reused:
            raw, _ = query_ner(client, text)
            parsed = ast.literal_eval(raw)
            if not isinstance(parsed.get("named_entities"), list):
                raise ValueError("Invalid question NER")
            ner_cache[text] = parsed
            save(ner_path, ner_cache)
        hippo.named_entity_cache[text] = ner_cache[text]
        ner_seconds = time.perf_counter() - started

        synchronize_cuda()
        started = time.perf_counter()
        ranks, scores, logs = hippo.rank_docs(text, top_k=request["config"]["top_k"])
        synchronize_cuda()
        rank_seconds = time.perf_counter() - started
        if len(ranks) != request["config"]["top_k"] or not all(np.isfinite(scores)):
            raise ValueError("Incomplete or nonfinite ranking")
        if any(hippo.get_passage_by_idx(i) != passages[i] for i in ranks):
            raise ValueError("Official corpus order/text differs from exported corpus")
        results.append(dict(
            id=question["id"], question=text,
            sorted_passage=[passages[i] for i in ranks], sorted_passage_scores=scores,
            ranked_indices=ranks, logs=logs,
            query_ner_seconds=ner_seconds, rank_seconds=rank_seconds,
            retrieval_seconds=ner_seconds + rank_seconds, query_ner_reused=reused,
        ))
        save(Path(request["query_dir"]) / "retrieval.json", results)

    # 排名文件仅保存输出，任何计时重复都必须重新进入官方 rank_docs。
    for question in request["questions"]:
        hippo.rank_docs(question["question"], top_k=request["config"]["top_k"])
    synchronize_cuda()

    repeats = []
    for repeat in range(1, request["efficiency_measurement"]["retrieval_repeats"] + 1):
        synchronize_cuda()
        started = time.perf_counter()
        for question in request["questions"]:
            ranks, scores, _ = hippo.rank_docs(
                question["question"], top_k=request["config"]["top_k"]
            )
            if len(ranks) != request["config"]["top_k"] or not all(np.isfinite(scores)):
                raise ValueError("Incomplete or nonfinite repeated ranking")
        synchronize_cuda()
        seconds = time.perf_counter() - started
        repeats.append(dict(
            repeat=repeat, total_seconds=seconds,
            seconds_per_question=seconds / len(results),
        ))

    cold_index = (
        not timings.get("index_artifacts_present_at_start", False)
        and timings["extraction_reused_passages"] == 0
        and not timings["graph_reused"]
    )
    measurement = dict(
        status="passed", index_mode="cold indexing" if cold_index else "cache reuse or recovery",
        index_seconds=(timings["input_conversion_seconds"]
                       + timings["passage_extraction_seconds"]
                       + timings["graph_index_seconds_excluding_encoder_load"]
                       + timings["retriever_initialization_seconds_including_node_vectors"]
                       - timings["retriever_encoder_loading_seconds"]),
        index_boundary="conversion + extraction + graph/KNN + runtime graph/node vectors; model/tokenizer loading excluded",
        cold_queries=not any(row["query_ner_reused"] for row in results),
        first_query_seconds_per_question=sum(row["retrieval_seconds"] for row in results) / len(results),
        first_query_records=results,
        retrieval_runs=repeats,
        retrieval_seconds_per_question_median=float(np.median([
            row["seconds_per_question"] for row in repeats
        ])),
        retrieval_boundary="warm graph and cached query NER; real rank_docs, one untimed batch warmup",
        retrieval_result_cache_used=False,
        extraction_reused_passages=timings["extraction_reused_passages"],
        graph_reused=timings["graph_reused"],
        index_artifacts_present_at_start=timings.get("index_artifacts_present_at_start", False),
    )
    save(Path(request["run_dir"]) / "efficiency.json", measurement)


def retrieve_questions(
    request, dataset, model_name, retriever, cfg, client, query_ner, np, timings,
    loading=None,
):
    """逐题恢复 NER 缓存并检索，保持输入原文和结果字段。"""
    passages = request["passages"]
    synchronize_cuda()
    from src.hipporag import HippoRAG

    load_before_retriever = loading["seconds"] if loading is not None else 0.0
    start = time.perf_counter()

    hippo = HippoRAG(
        dataset,
        "openai",
        model_name,
        retriever,

        # 重要实验参数
        sim_threshold=cfg["sim_threshold"],
        damping=cfg["damping"],
        doc_ensemble=False,
        dpr_only=False,
    )
    synchronize_cuda()
    timings["retriever_initialization_seconds_including_node_vectors"] = (
        time.perf_counter() - start
    )

    if request.get("efficiency_measurement") is not None:
        timings["retriever_encoder_loading_seconds"] = (
            loading["seconds"] - load_before_retriever
        )
        measure_queries(hippo, request, client, query_ner, np, timings)
        return

    query_dir = Path(request["query_dir"])
    results_path = query_dir / "retrieval.json"
    results = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else []
    )
    by_id = {r["id"]: r for r in results}
    timings["retrieval_reused_questions"] = len(by_id)

    # NER 只依赖问题文本，在同一索引下不同查询批次间也可以复用。
    ner_path = Path("query_ner_checkpoint.json")
    ner_cache = (
        json.loads(ner_path.read_text(encoding="utf-8"))
        if ner_path.exists()
        else {}
    )
    for q in request["questions"]:
        if q["id"] in by_id:
            continue
        start = time.perf_counter()
        ner_reused = q["question"] in ner_cache
        if not ner_reused:
            ner_json, _ = query_ner(client, q["question"]) # LLM NER
            parsed = ast.literal_eval(ner_json)
            if not isinstance(parsed.get("named_entities"), list):
                raise ValueError("Invalid question NER")
            ner_cache[q["question"]] = parsed
            save(ner_path, ner_cache)
        hippo.named_entity_cache[q["question"]] = ner_cache[q["question"]] # 塞进去 NER 结果
        ner_seconds = time.perf_counter() - start

        start = time.perf_counter()
        # HippoRAG 利用 query + graph，得到 document ranking
        observation = request.get("case_observation")
        ranking_size = len(passages) if observation is not None else cfg["top_k"] # 观察模式要拿完整排名
        ranks, scores, logs = hippo.rank_docs(q["question"], top_k=ranking_size)
        rank_seconds = time.perf_counter() - start
        case_trace = None
        if observation is not None:
            case_trace = observe_case(hippo, observation, ranks, scores, logs)
            ranks, scores = ranks[:cfg["top_k"]], scores[:cfg["top_k"]]
        if any(hippo.get_passage_by_idx(i) != passages[i] for i in ranks): # document mapping 没有错位
            raise ValueError("Official corpus order/text differs from exported corpus")
        if len(ranks) != min(cfg["top_k"], len(passages)) or not all(np.isfinite(scores)): # ranking 必须完整且数值正常
            raise ValueError("Incomplete or nonfinite ranking")

        # 将索引转换回原始passage，不添加标题
        by_id[q["id"]] = dict(
            id=q["id"],
            question=q["question"],
            sorted_passage=[passages[i] for i in ranks], # 排序后的文档
            sorted_passage_scores=scores,
            ranked_indices=ranks,
            logs=logs,
            query_ner_seconds=ner_seconds,
            rank_seconds=rank_seconds,
            retrieval_seconds=ner_seconds + rank_seconds,
            query_ner_reused=ner_reused,
        )
        if case_trace is not None:
            by_id[q["id"]]["case_trace"] = case_trace
        save(
            results_path,
            [by_id[item["id"]] for item in request["questions"] if item["id"] in by_id],
        )
        print(f"Retrieved {q['id']}", flush=True)


def read_credentials(project_root):
    """抽取 API key/base URL, 只在内存中交给客户端。"""
    credentials = {}
    credentials_path = Path(project_root) / ".env.local"
    for line in credentials_path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            if key.strip() in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
                credentials[key.strip()] = value.strip().strip("\"'")

    return credentials


def save_run_summary(timings, loading, request, usage_before):
    """分开保存共享累计 usage 与本次新增 usage，计时归属实验。"""
    run_dir = Path(request["run_dir"])
    index_dir = Path(request["index_dir"])
    query_dir = Path(request["query_dir"])
    timings["total_child_seconds"] = time.perf_counter() - BOOT_TIME
    timings["model_and_tokenizer_loading_seconds"] = loading["seconds"]
    timings["model_and_tokenizer_loading_calls"] = loading["calls"]
    save(run_dir / "timing.json", timings)
    with (run_dir / "timing_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(timings) + "\n")
    added_usage = {}
    for stage, directory in (("indexing", index_dir), ("query_ner", query_dir)):
        total = usage_total(directory / "llm_usage.jsonl")
        save(directory / "token_usage.json", total)
        added_usage[stage] = {
            key: total[key] - usage_before[stage][key] for key in total
        }
    usage_record = dict(
        added_this_invocation=added_usage,
        scope="Successful extraction/query-NER responses; excludes generation/evaluation",
    )
    save(run_dir / "token_usage.json", usage_record)
    with (run_dir / "token_usage_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(usage_record) + "\n")
    if (run_dir / "failure.json").exists():
        (run_dir / "failure.json").replace(run_dir / "recovered_failure.json")


def usage_total(path):
    """汇总当前阶段成功调用的 usage；没有调用时返回零。"""
    total = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            usage = json.loads(line)["usage"]
            for key in total:
                total[key] += usage.get(key, 0)
    return total


def main(request_path):
    """准备运行环境，然后依次抽取、建图、检索和保存统计。"""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    root = Path(request["external_root"])

    # 实际上游 checkout 必须是官方指定 commit，否则不允许继续执行
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if revision != PIN:
        raise RuntimeError(f"Expected official {PIN}, found {revision}")

    # 显式规定 src 为 external HippoRAG-2024/src
    sys.path.insert(0, str(root))
    sys.path.insert(1, str(root / "src"))

    run_dir = Path(request_path).parent.resolve()
    request["run_dir"] = str(run_dir)
    index_dir = Path(request["index_dir"])
    query_dir = Path(request["query_dir"])
    # 必须在创建目录或写入数据前观察初始状态，否则所有运行都会看似复用了缓存。
    index_artifacts_present = any(
        (index_dir / name).exists()
        for name in ("data", "output", "extraction_checkpoint.json", "llm_usage.jsonl", "graph_complete.json")
    )
    # 官方代码使用相对路径；仅将 cwd 指向共享索引目录，不改其算法。
    os.chdir(index_dir)
    for folder in ("data", "output", "data/lm_vectors"):
        Path(folder).mkdir(parents=True, exist_ok=True)
    cfg = request["config"]

    # adapter 技术: 人为创建一个 bridge dataset 传给上游 HippoRAG 的 file-based dataset API
    dataset = "linearrag_bridge"

    # 使用相对路径，可以避免 Windows 上的上游缓存名称中出现 :（冒号）。
    retriever = os.path.relpath(cfg["embedding_model_path"], index_dir).replace("\\", "/")
    model_name = cfg["extraction_model"]
    timings = dict(
        official_commit=PIN,
        extraction_model=model_name,
        embedding_model=cfg["embedding_model_path"],
        faiss_device="cpu",
        top_k=cfg["top_k"],
        damping=cfg["damping"],
        sim_threshold=cfg["sim_threshold"],
        index_artifacts_present_at_start=index_artifacts_present,
    )
    launch_wall_ns = int(os.environ.get("HIPPORAG_LAUNCH_WALL_NS", BOOT_WALL_NS))
    timings["process_startup_and_stdlib_seconds"] = (
        BOOT_WALL_NS - launch_wall_ns
    ) / 1e9
    timings["windows_openmp_workaround"] = os.environ.get("KMP_DUPLICATE_LIB_OK")
    import httpx
    from langchain_openai import ChatOpenAI
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("The pinned HuggingFace route requires CUDA for encoding")
    timings["cuda_device"] = torch.cuda.get_device_name()
    loading = {"seconds": 0.0, "calls": 0}
    for loader_class in (AutoModel, AutoTokenizer):
        loader_class.from_pretrained = staticmethod(
            TimedLoader(loader_class.from_pretrained, loading)
        )

    usage_before = {
        "indexing": usage_total(index_dir / "llm_usage.jsonl"),
        "query_ner": usage_total(query_dir / "llm_usage.jsonl"),
    }
    usage_recorder = UsageRecorder(model_name, index_dir / "llm_usage.jsonl")

    credentials = read_credentials(request["project_root"])
    init_client = ClientFactory(model_name, credentials, usage_recorder)

    # 为了让官方代码能接我们的 endpoint/model，只替换 client initialization；
    # prompt 和 extraction function 本身继续使用官方逻辑
    import src.langchain_util as langchain_util
    langchain_util.init_langchain_model = init_client
    import src.openie_with_retrieval_option_parallel as openie
    from src.named_entity_extraction_parallel import named_entity_recognition as query_ner
    client = init_client("openai", model_name)
    openie.client = client

    timings["imports_and_client_seconds"] = time.perf_counter() - BOOT_TIME
    prepare_and_extract_passages(
        request, dataset, model_name, openie, np, timings
    )

    build_graph(dataset, model_name, retriever, cfg, loading, timings)

    usage_recorder.usage_path = query_dir / "llm_usage.jsonl"
    retrieve_questions(
        request, dataset, model_name, retriever, cfg, client, query_ner, np, timings,
        loading=loading,
    )

    save_run_summary(timings, loading, request, usage_before)
    print("HippoRAG batch complete", flush=True)


if __name__ == "__main__":
    try:
        main(Path(sys.argv[1]).resolve())
    except Exception as exc:
        # Exceptions from HTTP libraries may contain endpoint details; keep credentials out of logs.
        save(
            Path(sys.argv[1]).parent / "failure.json", # 错误记录
            dict(stage="official_runner", error_type=type(exc).__name__),
        )
        raise
