# HippoRAG 2024 接入与共享缓存

官方实现位于 `D:\code\HippoRAG-2024`，使用独立 `.venv`，锁定 `v1.0.0` 提交 `b144c46df14cabe5f5822d8caded4bec5f709461`，叠加本目录 `compatibility.patch`。本项目通过 JSON 文件与子进程调用官方代码；没有安装 HippoRAG2。

## 用法

```python
from src.baselines.hipporag import HippoRAG

model = HippoRAG(experiment_id="my-run")
model.index(passages)
results = model.retrieve(questions)
```

传入项目的 `LLM_Model` 后可调用 `qa(questions)`。返回格式仍是问题 ID、问题文本、排序段落、分数与本次标准答案。`working_dir / experiment_id` 只控制实验结果；共享缓存默认位于 `src.paths.CACHE_DIR / "hipporag"`，无需手工选择缓存 ID。

## 两类目录

```text
data/output/cache/hipporag/<index-id>/
    manifest.json
    extraction_checkpoint.json
    query_ner_checkpoint.json
    官方 data/、output/（图与向量）
    llm_usage.jsonl、token_usage.json（索引累计用量）
    queries/<query-id>/
        manifest.json
        retrieval.json（无标准答案、无生成答案）
        llm_usage.jsonl、token_usage.json（本批次问题 NER 用量）

data/output/<实验结果目录>/
    request.json、cache_reference.json
    predictions.json、evaluation_results.json
    process.log、timing.json、adapter_timing.json
    token_usage.json、token_usage_history.jsonl（本次新增用量）
```

索引身份来自有序语料、抽取模型、嵌入路径、建图相似度阈值和驱动版本；换问题、Top-k、PPR damping、线程数或生成模型不会重做文档抽取。查询身份来自索引、问题 ID/文本和检索配置。Q1/Q4 共用同一输入与配置时，直接读取已有排序，不启动子进程；换问题批次时复用索引及重叠问题的 NER。

标准答案在返回时附加。生成答案与评价留在各自实验目录，不共享。缓存记录保留首次抽取成本；实验 usage 记录本次新增成功抽取/问题 NER token，不包含生成与评价。`token_usage.json` 表示最近一次调用，历史 delta 追加到 `token_usage_history.jsonl`。失败前的成功响应仍保存在缓存 usage 日志中。

相同实验 ID 拒绝不同输入或生成配置。旧的 `minimal-20260930` 等目录保留为历史证据，不自动迁移，不能用新协议恢复。共享缓存顺序使用，不支持并发写入。外部源码、依赖或同路径模型内容改变时，显式使用新的 `cache_dir`；没有全环境自动指纹。

## 最小调用与费用

```powershell
.venv\Scripts\python.exe -m experiments.hipporag.minimal
.venv\Scripts\python.exe -m experiments.hipporag.minimal --retrieval-only
```

默认新实验 ID 为 `minimal-cache-split-20261001`，12 个虚构段落、3 题、Top-5、4 线程。首次运行有付费文档实体/关系抽取、问题 NER；完整命令还有生成和评价。不要将此命令当作离线检查。已完成的抽取、检索与生成可恢复，最小例子的评价仍会重新执行。

本地配置是 `qwen3.8-flash` 抽取/生成/评价、`all-mpnet-base-v2` 编码、CUDA PyTorch、CPU FAISS，PPR damping=0.5、sim_threshold=0.8，保持官方抽取提示、图构建与排序。它不等同于论文的 GPT/Contriever/ColBERT 配置。

环境版本见 `requirements-lock.txt`；兼容补丁与历史安装恢复记录见 `project/agent/decisions/HIPPORAG_2024_INTEGRATION_2026-09-30.md`。适配器直接从 `.env.local` 读取凭据传给客户端，子进程继承系统代理；超时先诊断来源与代理再重试一次。

本次缓存拆分仅做离线模拟验证，没有正式 Q1/Q4 HippoRAG 比较结果。
