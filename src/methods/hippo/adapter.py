"""连接原版 HippoRAG：共享抽取与检索缓存，单独保存实验答案。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.paths import PROJECT_ROOT, OUTPUT_DIR, MODELS_DIR, HIPPORAG_CACHE_DIR
from src.common.cache_identity import (
    content_identity, hippo_stages, hippo_query_manifest,
    hippo_question_ner_manifest, write_manifest,
)


class HippoRAG:
    """保持 index/retrieve/qa 接口，共享缓存由内容确定，答案归属实验。"""

    def __init__(
        self,
        llm_model=None,
        *,
        external_root=None,
        embedding_model_path=MODELS_DIR / "all-mpnet-base-v2",
        extraction_model="qwen3.8-flash",
        retrieval_top_k=5,
        max_workers=4,
        working_dir=OUTPUT_DIR / "hipporag",
        experiment_id="minimal",
        damping=0.5,
        sim_threshold=0.8,
        cache_dir=HIPPORAG_CACHE_DIR,
        case_observation=None, # 适配器新增的参数
    ):
        """设置官方运行环境、共享缓存根目录与本次实验输出位置。"""
        self.external_root = Path(
            external_root
            or os.environ.get("HIPPORAG_ROOT", str(PROJECT_ROOT.parent / "HippoRAG-2024"))
        ).resolve()

        # 启动另一个 Python 解释器来运行 HippoRAG 的官方索引/检索代码
        self.python = self.external_root / ".venv/Scripts/python.exe"
        self.runner = PROJECT_ROOT / "src/methods/hippo/official_runner.py"

        output_path = Path(working_dir)
        model_path = Path(embedding_model_path)
        self.run_dir = (
            output_path if output_path.is_absolute() else PROJECT_ROOT / output_path
        ).resolve() / experiment_id
        cache_path = Path(cache_dir)
        self.cache_root = (
            cache_path if cache_path.is_absolute() else PROJECT_ROOT / cache_path
        ).resolve()
        model_path = (
            model_path if model_path.is_absolute() else PROJECT_ROOT / model_path
        ).resolve()
        self.config = dict(
            external_root=str(self.external_root),
            embedding_model_path=str(model_path),
            extraction_model=extraction_model,
            top_k=retrieval_top_k,
            max_workers=max_workers,
            damping=damping,
            sim_threshold=sim_threshold,
        )
        if retrieval_top_k < 1 or max_workers < 1:
            raise ValueError("top-k and max-workers must be positive")
        self.llm_model = llm_model
        self.max_workers = max_workers
        self.passages = []
        self.case_observation = case_observation

    def index(self, passages):
        """保存语料；实际抽取和建图由 retrieve 启动的外部进程完成。"""
        self.passages = list(passages)
        if not self.passages:
            raise ValueError("HippoRAG requires a nonempty corpus")

    def retrieve(self, questions):
        """相同检索直接读取缓存；缺失结果才启动官方子进程。"""
        if not self.passages:
            raise RuntimeError("Call index(passages) before retrieve(questions)")

        request, payload = self._prepare_request(questions)
        write_seconds = self._write_request(request, payload)
        cached = self.query_dir / "retrieval.json"
        complete = cached.exists() and len(
            json.loads(cached.read_text(encoding="utf-8"))
        ) == len(self.questions)
        if complete:
            wall_seconds = 0.0
        else:
            env = self._subprocess_environment()
            wall_seconds = self._run_official_process(request, env)
        return self._read_retrieval(wall_seconds, write_seconds)

    def measure_efficiency(self, questions, retrieval_repeats):
        """启动真实阶段测量，不将已有排名文件作为检索计时结果。"""
        if not self.passages:
            raise RuntimeError("Call index(passages) before measuring efficiency")
        if retrieval_repeats < 1:
            raise ValueError("retrieval-repeats must be positive")

        request, payload = self._prepare_request( # 构造一种不同于普通检索的请求
            questions, measurement=dict(retrieval_repeats=retrieval_repeats)
        )
        write_seconds = self._write_request(request, payload)
        wall_seconds = self._run_official_process(request, self._subprocess_environment())
        results = self._read_retrieval(wall_seconds, write_seconds)
        measurement = json.loads((self.run_dir / "efficiency.json").read_text(encoding="utf-8"))
        return results, measurement

    def _prepare_request(self, questions, measurement=None):
        """固定问题 ID，并拒绝在同一实验目录中混用不同输入。"""
        questions = [dict(q, id=str(q.get("id", i))) for i, q in enumerate(questions)]
        if len({q["id"] for q in questions}) != len(questions):
            raise ValueError("Question IDs must be unique")
        if not questions:
            raise ValueError("HippoRAG requires nonempty questions")

        self.questions = questions
        stages = hippo_stages(self.passages, self.config)
        extraction_id = stages["extraction"]["id"]
        self.extraction_dir = self.cache_root / "extractions" / extraction_id
        index_id = stages["graph"]["id"]
        index_manifest = dict(
            schema=2, stages=stages, extraction_dir=str(self.extraction_dir),
            reuse="Extraction matches corpus/model/protocol; graph additionally matches encoder/threshold.",
        )
        self.index_dir = self.cache_root / index_id
        query_manifest = hippo_query_manifest(
            index_id, questions, self.config, self.case_observation,
        )
        query_id = query_manifest["id"]
        self.query_dir = self.index_dir / "queries" / query_id
        ner_manifest = hippo_question_ner_manifest(self.config)
        self.question_ner_dir = self.cache_root / "question_ner" / ner_manifest["id"]

        # request.json是两个世界之间的实验协议
        payload = dict(
            config=self.config,
            passages=self.passages,
            questions=questions,
            project_root=str(PROJECT_ROOT),
            external_root=str(self.external_root),
            index_dir=str(self.index_dir),
            query_dir=str(self.query_dir),
            extraction_dir=str(self.extraction_dir),
            question_ner_path=str(self.question_ner_dir / "checkpoint.json"),
            generation_config=getattr(self.llm_model, "llm_config", None),
        )
        if self.case_observation is not None:
            payload["case_observation"] = self.case_observation
        if measurement is not None:
            payload["efficiency_measurement"] = measurement

        # 相同 run ID 必须对应完全相同的一套实验输入(config + passages + questions + paths)
        # hash everything
        identity = content_identity(payload)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        request = self.run_dir / "request.json"
        if request.exists():
            previous = json.loads(request.read_text(encoding="utf-8"))
            if previous.get("identity") != identity:
                raise ValueError("Run ID already belongs to different inputs/configuration")
        payload["identity"] = identity

        # 阶段 manifest 解释身份与复用边界；历史 run 请求不自动覆盖。
        self.query_dir.mkdir(parents=True, exist_ok=True)
        for directory, manifest in (
            (self.extraction_dir, dict(schema=2, stage="extraction",
                **stages["extraction"], reuse="Independent of encoder, graph threshold, questions and workers.")),
            (self.index_dir, index_manifest), (self.query_dir, query_manifest),
            (self.question_ner_dir, ner_manifest),
        ):
            # 保留迁移来源，不能用一次请求抹掉历史证据。
            previous_path = directory / "manifest.json"
            if previous_path.exists():
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                for key in ("migration", "legacy_manifests"):
                    if key in previous:
                        manifest[key] = previous[key]
            write_manifest(directory, manifest)
        (self.run_dir / "cache_reference.json").write_text(
            json.dumps(dict(index_id=index_id, query_id=query_id,
                            extraction_id=extraction_id, extraction_dir=str(self.extraction_dir),
                            index_dir=str(self.index_dir), query_dir=str(self.query_dir)),
                       indent=2),
            encoding="utf-8",
        )

        return request, payload

    def _write_request(self, request, payload):
        """将本次实验请求写入结果目录，返回文件交换耗时。"""
        started = time.perf_counter()
        request.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return time.perf_counter() - started

    def _subprocess_environment(self):
        """继承环境，配置 Windows 兼容设置和系统代理。"""
        env = os.environ.copy()
        env.update(
            PYTHONUTF8="1",
            TOKENIZERS_PARALLELISM="false",
            KMP_DUPLICATE_LIB_OK="TRUE",
            OMP_NUM_THREADS="1",
        )

        self._inherit_windows_proxy(env)
        return env

    @staticmethod
    def _inherit_windows_proxy(env):
        """将已启用的 Windows 系统代理显式交给子进程。"""
        # 显式继承 Windows 系统代理设置，供下载操作和子进程使用
        if os.name == "nt":
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            ) as key:
                try:
                    enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
                    proxy = winreg.QueryValueEx(key, "ProxyServer")[0]
                    if enabled:
                        if ";" in proxy or "=" in proxy:
                            for part in proxy.split(";"):
                                scheme, address = part.split("=", 1)
                                if scheme in ("http", "https"):
                                    env[scheme.upper() + "_PROXY"] = "http://" + address
                        else:
                            proxy = proxy if "://" in proxy else "http://" + proxy
                            env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy)
                except FileNotFoundError:
                    pass

    def _run_official_process(self, request, env):
        """调用官方驱动，日志仍归属本次实验。"""
        started = time.perf_counter()
        env["HIPPORAG_LAUNCH_WALL_NS"] = str(time.time_ns())

        # 外部程序出错时, 直接抛出异常, 并在 process.log 中查看详细信息
        with (self.run_dir / "process.log").open("a", encoding="utf-8") as log:
            process = subprocess.run(
                [str(self.python), str(self.runner), str(request)],
                cwd=str(self.run_dir),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        wall_seconds = time.perf_counter() - started
        if process.returncode:
            raise RuntimeError(
                f"HippoRAG exited {process.returncode}; see {self.run_dir / 'process.log'}"
            )

        return wall_seconds

    def _read_retrieval(self, wall_seconds, exchange_write_seconds):
        """读取无答案的共享检索结果，再补上本次问题的标准答案。"""
        started = time.perf_counter()
        response = json.loads(
            (self.query_dir / "retrieval.json").read_text(encoding="utf-8")
        )
        by_id = {result["id"]: result for result in response}
        response = [dict(by_id[q["id"]], gold_answer=q.get("answer")) for q in self.questions]

        # 写入耗时, 实际运行时间, 读取耗时
        read_seconds = time.perf_counter() - started
        timings = dict(
            subprocess_wall_seconds=wall_seconds,
            exchange_write_seconds=exchange_write_seconds,
            exchange_read_seconds=read_seconds,
            retrieval_cache_hit=wall_seconds == 0.0,
        )
        (self.run_dir / "adapter_timing.json").write_text(
            json.dumps(timings, indent=2), encoding="utf-8"
        )
        if wall_seconds == 0.0:
            zero = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
            usage = dict(
                added_this_invocation=dict(indexing=zero, query_ner=zero),
                scope="Retrieval cache hit; no extraction/query-NER calls",
            )
            (self.run_dir / "token_usage.json").write_text(
                json.dumps(usage, indent=2), encoding="utf-8"
            )
            with (self.run_dir / "token_usage_history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(usage) + "\n")

        return response

    # HippoRAG 官方实现不负责最终回答生成, 复用VanillaRAG的提示词
    def qa(self, questions):
        """复用共享检索，生成和恢复本次实验自己的答案。"""
        if self.llm_model is None:
            raise RuntimeError("qa() requires the project's generation LLM")
        from src.methods.vanilla.vanilla_rag import QA_SYSTEM_PROMPT

        results = self.retrieve(questions)

        # resume 断点续跑
        predictions_path = self.run_dir / "predictions.json"
        self._restore_predictions(predictions_path, results)
        pending = [r for r in results if "pred_answer" not in r]

        # answer generation 可以并发
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [
                pool.submit(self._generate_answer, result, QA_SYSTEM_PROMPT)
                for result in pending
            ]
            for future in as_completed(futures):
                future.result()

                self._save_predictions(predictions_path, results)

        return results

    def _generate_answer(self, result, system_prompt):
        """用项目统一提示生成单题答案，不写入共享缓存。"""
        context = "".join(f"{p}\n" for p in result["sorted_passage"])
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{context}Question: {result['question']}\n Thought: ",
            },
        ]
        text = self.llm_model.infer(messages)
        result["pred_answer"] = text.split("Answer:", 1)[-1].strip()
        return result

    @staticmethod
    def _save_predictions(predictions_path, results):
        """原子保存本次实验中已完成的答案。"""
        # predictions.json.tmp 完整写好 replace predictions.json
        temp = predictions_path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(
                [result for result in results if "pred_answer" in result],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp.replace(predictions_path)

    @staticmethod
    def _restore_predictions(predictions_path, results):
        """只从本次实验目录恢复生成答案。"""
        if predictions_path.exists():
            completed = {
                result["id"]: result
                for result in json.loads(predictions_path.read_text(encoding="utf-8"))
            }
            for result in results:
                if result["id"] in completed:
                    result["pred_answer"] = completed[result["id"]]["pred_answer"]
