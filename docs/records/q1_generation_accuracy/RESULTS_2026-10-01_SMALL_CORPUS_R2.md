# Q1 小语料三方法结果：2026-10-01

原始证据：`data/output/q1_generation_accuracy/q1-hotpotqa-context20-noise50-20261001-r2/`。最终证据保存在摘要、各方法预测/评价和比较结果中；输入元数据保留语料哈希与替换来源。

## 输入与协议

20 道 HotpotQA 问题、250 篇文档；139402 个字符、22786 个空白分词词项。三种方法使用相同的修订后有序语料、问题 ID/文本/标准答案和 Top-5 上下文预算。嵌入模型：`all-mpnet-base-v2`；生成/评价：`qwen3.8-flash`；LinearRAG 使用默认 BFS。HippoRAG 为采用本地配置的原始 2024 版，没有使用论文模型设置。

LinearRAG 使用 `en_core_web_trf`、`max_iterations=3`、`passage_ratio=0.05`、`iteration_threshold=0.4` 和 `top_k_sentence=1`。HippoRAG 抽取使用 `qwen3.8-flash`、四个工作线程、`damping=0.5` 和 `sim_threshold=0.8`。

第 227 段的抽取被服务商输出审查拒绝（`InternalError.Algo.DataInspectionFailed`）。随机干扰文档 "Taichung International Airport" 被替换为 "Boltzmann constant"；替换使用种子 `20261001:replacement:227`，从排除已有文本后的剩余来源候选中抽取。问题及其候选上下文均未变化。这次拒绝不能证明机场类别触发了审查。所有方法均在修订后输入上运行；最终比较不混入原始语料分数。

## 结果

| 方法 | 包含匹配准确率 | LLM 准确率 |
|---|---:|---:|
| Vanilla RAG | 16/20 (80%) | 16/20 (80%) |
| LinearRAG | 15/20 (75%) | 16/20 (80%) |
| HippoRAG 2024 | 18/20 (90%) | 20/20 (100%) |

## 缓存与用量范围

HippoRAG 复用了 231 条已完成且文本完全相同的抽取记录，抽取剩余 19 条。未转移图；它为修订后语料构建图，并对全部 20 题执行检索、生成和评价。这属于部分抽取缓存复用，不是全新冷构建。复用抽取携带的历史用量代表获取成本，不是新增调用。固定驱动继承代理设置，HTTP 超时为 120 秒；提示词及抽取/生成模型未变化。

新增成功索引/问题 NER 用量：25567 + 2651 = 28218 tokens，不含答案生成和评价。未据此估算金额或完整账单总量。

## 结论边界

这些结果来自一次 20 题的有利设置样本。HippoRAG 在此处的观察准确率更高；不作显著性、普遍优势或完整论文复现声明。实际输入由最终清单定义，包括替换内容，不能仅依据原始构造输出。
