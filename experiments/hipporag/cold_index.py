"""用空缓存重测 HippoRAG 冷索引，采用流式抽取并单列初始化耗时。

沿用官方提示、NER/OpenIE、建图与检索器初始化；不执行查询或生成答案。
2048-token 输出预算仅在截断时提升至 4096，所有重试计入本轮建库时间。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def load_module(name, path):
    """按文件加载模块，避免项目和外部仓库的 src 名称冲突。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StreamingClient:
    """让官方 invoke 接口使用流式传输，并记录实际响应 token。"""

    def __init__(self, model, credentials, recorder):
        """绑定模型、凭据和线程安全的用量记录器。"""
        import httpx
        from openai import OpenAI

        self.model = model
        self.recorder = recorder
        self.client = OpenAI(
            api_key=credentials['OPENAI_API_KEY'],
            base_url=credentials['OPENAI_BASE_URL'],
            max_retries=0,
            http_client=httpx.Client(trust_env=True, timeout=60),
        )

    def invoke(self, messages, **kwargs):
        """收齐流式响应；拒绝截断结果，并将重试用量一起记录。"""
        messages = [
            dict(role={'system': 'system', 'human': 'user', 'ai': 'assistant'}[m.type], content=m.content)
            for m in messages
        ]
        from openai import BadRequestError

        for budget, json_mode in ((2048, True), (4096, False)):
            pieces = []
            usage = None
            finish = None
            try:
                stream = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0,
                    max_tokens=budget, **({'response_format': {'type': 'json_object'}} if json_mode else {}),
                    extra_body={'enable_thinking': False}, stream=True,
                    stream_options={'include_usage': True},
                )
                for event in stream:
                    if event.usage is not None:
                        usage = event.usage.model_dump()
                    if event.choices:
                        pieces.append(event.choices[0].delta.content or '')
                        finish = event.choices[0].finish_reason or finish
            except BadRequestError:
                if json_mode:
                    continue
                raise
            if usage is None:
                raise RuntimeError('Streaming endpoint returned no actual token usage')
            with self.recorder.lock, self.recorder.usage_path.open('a', encoding='utf-8') as log:
                log.write(json.dumps(dict(model=self.model, usage=usage, max_tokens=budget,
                                          finish_reason=finish, transport='stream', json_mode=json_mode)) + '\n')
            if finish == 'stop':
                content = ''.join(pieces)
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    continue
                if 'max_tokens' in kwargs and not isinstance(parsed.get('triples'), list):
                    continue
                return SimpleNamespace(content=content, response_metadata={'token_usage': usage})
            if finish != 'length':
                raise RuntimeError('Unexpected completion finish reason: ' + str(finish))

        raise RuntimeError('OpenIE/NER output remained truncated at 4096 tokens')


def streaming_factory(self, provider, name, **kwargs):
    """替换客户端传输方式，继续让官方函数组织全部提示。"""
    return StreamingClient(self.model_name, self.credentials, self.usage_recorder)


def initialize_index(request, dataset, model_name, retriever, cfg, client, query_ner, np, timings, loading=None):
    """准备运行时图和节点向量；不进行问题 NER 或排名。"""
    from src.hipporag import HippoRAG
    import torch

    torch.cuda.synchronize()
    before_loading = loading['seconds']
    started = time.perf_counter()
    HippoRAG(dataset, 'openai', model_name, retriever,
             sim_threshold=cfg['sim_threshold'], damping=cfg['damping'],
             doc_ensemble=False, dpr_only=False)
    torch.cuda.synchronize()
    timings['retriever_initialization_seconds_including_node_vectors'] = time.perf_counter() - started
    timings['retriever_encoder_loading_seconds'] = loading['seconds'] - before_loading

    seconds = (timings['input_conversion_seconds'] + timings['passage_extraction_seconds']
               + timings['graph_index_seconds_excluding_encoder_load']
               + timings['retriever_initialization_seconds_including_node_vectors']
               - timings['retriever_encoder_loading_seconds'])
    output = Path(request['run_dir']) / 'cold_index_measurements.json'
    output.write_text(json.dumps(dict(
        status='passed', index_mode='cold-build trajectory resumed', passage_count=len(request['passages']),
        index_seconds=seconds, extraction_reused_passages=timings['extraction_reused_passages'], graph_reused=timings['graph_reused'],
        index_artifacts_present_at_start=timings['index_artifacts_present_at_start'], query_calls=0,
        boundary='conversion + extraction + graph/KNN + runtime graph/node vectors; model/tokenizer loading excluded',
        extraction_protocol='Official prompts; streaming JSON; 2048 tokens, 4096 only after length finish',
    ), indent=2), encoding='utf-8')


def inherit_proxy():
    """显式继承已启用的 Windows 系统代理，不输出代理地址。"""
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                        r'Software\Microsoft\Windows\CurrentVersion\Internet Settings') as key:
        if winreg.QueryValueEx(key, 'ProxyEnable')[0]:
            proxy = winreg.QueryValueEx(key, 'ProxyServer')[0]
            if ';' in proxy or '=' in proxy:
                for part in proxy.split(';'):
                    scheme, address = part.split('=', 1)
                    if scheme in ('http', 'https'):
                        os.environ[scheme.upper() + '_PROXY'] = 'http://' + address
            else:
                proxy = proxy if '://' in proxy else 'http://' + proxy
                os.environ.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy)


def main():
    """复制已固定的输入到新测量目录，从空索引开始连续执行。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-request', type=Path, required=True)
    parser.add_argument('--experiment-id', required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    paths = load_module('project_paths', root / 'src/paths.py')
    output = paths.EXPERIMENT_RESULTS_DIR / 'q2_efficiency_analysis' / args.experiment_id
    output.mkdir(parents=True, exist_ok=args.resume)
    timing_path = output / 'session_timing.json'
    previous = json.loads(timing_path.read_text()) if args.resume and timing_path.exists() else {}
    cumulative_before = previous.get('cumulative_session_wall_seconds', previous.get('session_wall_seconds', 0.0))
    request = json.loads(args.source_request.read_text(encoding='utf-8'))
    request['index_dir'] = str(output / 'cache')
    request['query_dir'] = str(output / 'cache/queries/unused')
    request.pop('efficiency_measurement', None)
    request['cold_index_only'] = True
    request['streaming_extraction'] = dict(initial_max_tokens=2048, on_length_max_tokens=4096,
                                          temperature=0, read_timeout=60, max_retries=0,
                                          json_400_fallback='plain output with official JSON prompt')
    request['source_request'] = str(args.source_request.resolve())
    Path(request['query_dir']).mkdir(parents=True, exist_ok=args.resume)
    request_path = output / 'request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding='utf-8')

    os.environ.update(PYTHONUTF8='1', TOKENIZERS_PARALLELISM='false',
                      KMP_DUPLICATE_LIB_OK='TRUE', OMP_NUM_THREADS='1')
    inherit_proxy()
    runner = load_module('official_cold_runner', root / 'experiments/hipporag/official_runner.py')
    runner.ClientFactory.__call__ = streaming_factory
    runner.retrieve_questions = initialize_index
    started = time.perf_counter()
    try:
        runner.main(request_path)
    finally:
        (output / 'session_timing.json').write_text(json.dumps(dict(
            session_wall_seconds=time.perf_counter() - started,
            cumulative_session_wall_seconds=cumulative_before + time.perf_counter() - started,
            scope='Cumulative active cold-build trajectory; includes initialization, failures and recovery, excludes gaps between invocations',
        ), indent=2), encoding='utf-8')
    print('Cold indexing completed:', output, flush=True)


if __name__ == '__main__':
    main()
