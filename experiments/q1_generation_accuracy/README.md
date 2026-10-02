# Q1: Generation Accuracy

## 2026-10-01：小语料三方法补充实验

新增 `--corpus-dir` 和 `--methods`，复用原生成提示、评价器、Top-5 和默认 BFS。输入包模式读取全部固定问题，`--max-questions`、`--seed`、`--datasets` 不改变包内样本。源数据集名称决定 LinearRAG 参数；输入哈希和嵌入路径决定独立缓存身份。

已准备 20 题、200 篇候选上下文、50 篇额外随机文档。这些不是金标准支持标注，构造边界见 `../corpus_construction/README.md`。

仅核验准备，不加载模型或调用 API：

```powershell
.venv\Scripts\python.exe -m experiments.q1_generation_accuracy.generation_accuracy --corpus-dir data/input/derived_corpora/hotpotqa-context20-noise50-s20261001 --methods vanilla_rag linearrag hipporag --llm-model qwen3.8-flash --experiment-id q1-hotpotqa-context20-noise50-ready-20261001 --prepare-only --resume
```

**付费正式运行命令，本轮未执行：**

```powershell
.venv\Scripts\python.exe -m experiments.q1_generation_accuracy.generation_accuracy --corpus-dir data/input/derived_corpora/hotpotqa-context20-noise50-s20261001 --methods vanilla_rag linearrag hipporag --llm-model qwen3.8-flash --experiment-id q1-hotpotqa-context20-noise50-ready-20261001 --resume
```

冷启动预期：250 次 passage NER + 250 次关系抽取 + 20 次问题 NER + 三个方法各 20 次生成和 20 次评价 = 640 次逻辑 API 调用，不含失败重试。语料共 139,509 字符、22,794 个空白分隔词，均非计费 token。重复 few-shot 提示和输出也计费，不能按字符缩减比例推算金额。提供商单价与真实输出长度尚未核实，没有人民币估价或硬预算承诺。

HippoRAG 使用原版 2024、本地 `all-mpnet-base-v2`、`qwen3.8-flash` 抽取、4 个抽取/生成线程；damping=0.5、sim_threshold=0.8。共享抽取/索引和查询缓存位于 `data/output/cache/hipporag/`，由内容与配置自动定位。本次 `hotpotqa/hipporag/` 只保存请求、缓存引用、生成答案、日志和本次新增 usage。各方法最终 `predictions.json` 保留检索片段，可供后续 Q4 使用。

恢复必须保持 manifest 输入和配置一致。完整预测已保存但评价缺失时复用预测；HippoRAG 内部还支持逐文档抽取和逐题生成恢复。Vanilla/LinearRAG 生成未提供逐题恢复；评价器中途失败可能重复此前部分评价。旧版 manifest 不满足新增恢复校验，保留旧结果，另用新 ID。更改代码或依赖版本也应另用新 ID，缓存没有自动对整个环境做哈希。

离线验证与实验设计记录：`project/agent/experiments/q1_generation_accuracy/PREPARED_2026-10-01_SMALL_CORPUS.md`。

## 原两方法实验

This bounded reproduction compares a frozen Vanilla RAG baseline with the
original LinearRAG implementation on the same sampled questions. One script
performs sampling, both runs, evaluation, and result aggregation.

Prepare and inspect a deterministic sample without loading models:

```powershell
.venv\Scripts\python.exe experiments\q1_generation_accuracy\generation_accuracy.py --prepare-only
```

Run the bounded four-dataset experiment (100 questions per dataset by default):

```powershell
.venv\Scripts\python.exe experiments\q1_generation_accuracy\generation_accuracy.py
```

Generation and evaluation credentials are read directly from the untracked
project-root `.env.local` by `src.common.utils`. The values do not need to be set in
the process environment; do not commit or display the credential file.

For a small end-to-end trial, select one dataset and a smaller sample:

```powershell
.venv\Scripts\python.exe experiments\q1_generation_accuracy\generation_accuracy.py --datasets 2wikimultihop --max-questions 10
```

Raw predictions, evaluation files, the fixed sample manifest, and the final
comparison are written to
`data/output/experiment_results/q1_generation_accuracy/<experiment-id>/`.
Medical reports GPT accuracy only; the other datasets report both contain-match
and GPT-evaluated accuracy. The default LinearRAG path is the official BFS path.
