# Q2 效率实验

比较索引耗时、检索耗时及索引/检索阶段的 LLM token 用量；不生成答案，不进行答案评价。

## 当前状态

旧实验已完成：2Wiki 全部 658 个片段、100 题检索和三次独立冷索引，仅测 LinearRAG。历史结果见 `project/agent/experiments/q2_efficiency_analysis/RUN_2026-09-20_100Q.md`。

新补充已准备，尚未运行：直接复用新 Q1 的 HotpotQA 固定输入包，全部 20 题、250 篇文档，不再抽样。这是本地小规模对照，不是论文 Table 2 的直接复现。

## 固定输入与测量边界

- `--corpus-dir` 读取构造包并核对哈希、问题 ID；所有方法收到相同文档编号和问题顺序。
- LinearRAG 保持默认 BFS；HippoRAG 使用原版 2024 外部实现、`all-mpnet-base-v2`、`qwen3.8-flash` 抽取、4 个抽取 worker。
- 每种方法在本次结果目录内使用独立索引缓存，不借用 Q1 的暖索引测冷建库。
- 初始化/加载单列。LinearRAG 测 `index()`；HippoRAG 测输入转换、NER/OpenIE、建图/KNN、运行时图和节点向量准备，扣除记录到的模型/tokenizer 加载时间。
- 两种方法都记录首次查询。HippoRAG 首次查询包含问题 NER 与排名，并记录 NER 是否已缓存。
- 首轮后做一次不计时批量预热，再做五次实际检索重复。HippoRAG 此时复用问题 NER，但每次重新执行官方 `rank_docs()`，不读取旧排名代替计算。
- 首次查询与缓存 NER 后的重复耗时分别报告；不能把后者当成首次查询速度。GPU 计算前后同步。
- HippoRAG 子进程启动、文件交换、模型加载及阶段时间分别保留。比较时同时检查这些开销，避免将不同运行边界混为一谈。
- 中断恢复保留完成的抽取，避免重复支付；有旧索引产物的恢复耗时标记为 `cache reuse or recovery`，不能作为完整冷索引结果。

## 已准备的运行

离线准备（不加载模型、不调用 API）：

```powershell
.venv\Scripts\python.exe experiments/q2_efficiency_analysis/efficiency_analysis.py --corpus-dir data/input/derived_corpora/hotpotqa-context20-noise50-s20261001 --methods linearrag hipporag --retrieval-repeats 5 --experiment-id q2-hotpotqa-context20-noise50-ready-20261001-r2 --prepare-only
```

正式启动命令（本次未执行；会产生付费调用）：

```powershell
.venv\Scripts\python.exe experiments/q2_efficiency_analysis/efficiency_analysis.py --corpus-dir data/input/derived_corpora/hotpotqa-context20-noise50-s20261001 --methods linearrag hipporag --retrieval-repeats 5 --experiment-id q2-hotpotqa-context20-noise50-ready-20261001-r2 --resume
```

`--resume` 要求输入、测量代码、方法及配置与已保存 manifest 完全一致。完成方法读取本次已保存结果；未完成方法继续执行。代码修改后使用新 experiment ID。固定 corpus 模式不接受外部 `--resume-cache`；原 2Wiki 模式仍支持该参数，但它测的是缓存恢复。

首次完整运行预计 250 次文档 NER + 250 次 OpenIE + 20 次问题 NER = **520 次逻辑 API 调用**，不含失败重试。实际 token 来自成功响应；恢复调用的增量与缓存累计用量分别保存，不等于服务商全部账单。没有币种成本估计。

## 输出与验证

新输出位于 `data/output/experiment_results/q2_efficiency_analysis/<experiment-id>/`：共同 `manifest.json`、`samples.json`、`environment.json`、`summary.json`，以及各方法自己的 `measurements.json`、`retrieval_results.json`。HippoRAG 还保存阶段时间、token、外部日志和隔离缓存。

准备目录目前只有输入和准备记录，没有模型输出或正式 summary。离线验证见 `data/output/experiment_results/q2_efficiency_analysis/preparation-20261001/offline_validation.json`，使用假模型验证输入一致、真实排名路由、恢复及请求校验；不证明实时端点、实际耗时或图质量。

准确率引用同一输入哈希、同一方法配置下的新 Q1 结果。旧 Q1/Q2 数值不能与本次小语料结果拼接。
