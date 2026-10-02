# 实验设计

Q1–Q8 明确研究问题、对照与测量方案。所有 LinearRAG 实验使用默认 BFS 检索。共同的本地条件为 `all-mpnet-base-v2` 嵌入模型；需要生成或 LLM 评价时使用 `qwen3.8-flash`。

HippoRAG 对照使用带本地兼容补丁的原始 2024 版 `v1.0.0`，采用 `qwen3.8-flash` 抽取、CPU FAISS 与 PPR（`damping=0.5`、`sim_threshold=0.8`），不使用 HippoRAG2。

## Q1：生成准确率

**研究问题。** 在输入与回答条件一致时，检索方法如何影响答案准确率？

**设计理由。** Vanilla RAG 提供向量检索对照。小规模共享语料上的 HippoRAG 对照限制抽取成本；四数据集对照考察不同数据集上的差异。

**设计与测量。**

- 双方法范围：Vanilla RAG 与 LinearRAG，在 HotpotQA、2WikiMultiHopQA、MuSiQue 和 Medical 上各使用 100 道固定种子抽样的问题。
- 三方法范围：Vanilla RAG、LinearRAG 与 HippoRAG，使用种子 `20261001` 随机选择 20 道 HotpotQA 问题，不按既往答案正确性筛选。候选文档保留原始标题和句子，按全文去重，加入 50 篇随机来源文档，再打乱共 250 篇文档的语料。候选上下文不等同于标准支持事实；随机加入的文档也不保证无关。
- 各方法使用相同的有序段落、问题、嵌入模型、生成与评价提示词以及 Top-5 上下文预算。固定各数据集的 LinearRAG 参数；HippoRAG 使用四个抽取工作线程。
- 测量包含匹配准确率和 LLM 判断准确率，即被判为正确的答案比例；Medical 只报告 LLM 准确率。在各自范围内按相同问题比较方法。

选择小规模 HotpotQA 设置的依据是既往 LinearRAG 对照结果较有利，因此不能据此确立普遍优势。重构文档会改变边界、大小写与检索空间，所有方法必须使用同一构造语料；原始语料分数不能与这些分数合并。来源映射用于审计，不作为检索器输入或标准支持标签。

## Q2：效率分析

**研究问题。** LinearRAG 与 HippoRAG 的索引成本、首次查询延迟和重复检索延迟有何差异？

**设计理由。** 区分冷索引、首次查询工作与热检索，揭示抽取和问题 NER 的成本。对照内部使用相同输入。

**设计与测量。**

- 使用固定的 20 道 HotpotQA 问题、250 个段落、相同嵌入模型和 Top-5。每种方法使用独立的空索引缓存。
- 固定 LinearRAG 的 `max_iterations=3`、`passage_ratio=0.05`、`iteration_threshold=0.4`、`top_k_sentence=1` 和 `en_core_web_trf`；HippoRAG 使用四个抽取工作线程。
- 索引计时一次，首次查询批次计时一次，进行一次不计时的预热批次，再重复五次实际检索。报告首次查询的每题秒数，以及五次热检索批次平均每题耗时的中位数。
- HippoRAG 首次查询包含问题 NER 和排序；热检索复用 NER，但重新执行排序。
- LinearRAG 索引计时不包含模型初始化。HippoRAG 索引计时包含转换、段落抽取、图/KNN 构建和运行时图/节点向量初始化，不包含编码器与分词器加载。启动、模型加载及文件/子进程开销单独报告。
- GPU 阶段计时边界同步 CUDA。随耗时声明 GPU 编码、CPU FAISS 和运行设置。
- 排除答案生成和评价。测量成功抽取及问题 NER 响应的输入与输出 tokens，不含生成和评价用量。区分缓存获取成本与新增调用增量；二者都不构成完整服务商账单。恢复索引产物不能算作冷构建。

这些属于本地测量边界。单次索引构建不能估计稳定的耗时分布。

## Q3：消融研究

**研究问题。** 实体激活和全局重要性聚合分别对 LinearRAG 有何贡献？

**设计理由。** 每次移除一个组件，保持输入和其余系统条件不变。

**设计与测量。**

- 完整系统：BFS 实体传播后执行个性化 PageRank（PPR）。
- `without_entity_activation`：仅使用问题种子实体，保留 PPR。
- `without_global_importance`：保留初始段落评分，移除 PPR。作者未公开消融代码，因此这是对论文描述的操作化解释。
- HotpotQA、2WikiMultiHopQA、MuSiQue 和 Medical 各使用相同的 100 题样本；同一数据集的各变体共享一个模型实例和图。
- 除共同模型条件外，固定 `en_core_web_trf`、Top-5 和 16 个工作线程。
- 比较准确率与检索上下文变化。HotpotQA、2Wiki 和 MuSiQue 使用 LLM 准确率与包含匹配准确率的均值；Medical 只使用 LLM 准确率。

## Q4：检索质量

**研究问题。** 不同检索方法在各类 Medical 任务中如何权衡上下文相关性与证据召回率？

**设计理由。** 均衡任务抽样用于区分任务特有效应。相关性衡量上下文的有用程度，召回率衡量参考证据的覆盖程度。

**设计与测量。**

- 双方法范围：Vanilla RAG 与 LinearRAG，使用完整的 225 段 Medical 语料，每类任务按固定种子抽取 50 题，共 200 题。
- 三方法范围：加入 HippoRAG，使用种子 `20261001` 选择每类三题、共 12 题，以及 30 个来源段落。任务类别为 Fact Retrieval、Complex Reasoning、Contextual Summarize 和 Creative Generation。
- 小语料由 TF-IDF 问题/参考关系 Top-2 候选、审查后加入的段落和五个随机段落组成；保留原文并打乱。
- 各方法使用相同段落、问题和 Top-5。参考证据不参与检索，仅用于评价。
- 固定 LinearRAG 的 Medical NER `en_core_sci_scibert`、`max_iterations=3`、`passage_ratio=1.5`、`iteration_threshold=0.5` 和 `top_k_sentence=1`。
- 使用从 GraphRAG-Benchmark 改编的 LLM 上下文相关性与证据召回评价。每题获取两次相关性评分，将每条参考证据判为被检索上下文支持或不支持。
- 比较每题指标、任务均值和跨任务宏平均；结合指标差异检查 Top-5 重合度。

小规模候选语料没有经过完整的标准支持证据认证。不被语料支持的参考细节和 LLM 评价波动限制了对召回率及方法排名的解释。

## Q5：超参数敏感性

**研究问题。** LinearRAG 对剪枝阈值 `delta` 和初始段落权衡系数 `lambda` 有多敏感？

**设计理由。** 以固定基线为中心进行单因素扫描（OFAT），分别考察各参数。两者分别映射到 `iteration_threshold` 和 `passage_ratio`。

**设计与测量。**

- 使用完整的 658 段 2Wiki 语料，每个参数点使用相同的 100 道问题。固定 Top-5、`max_iterations=3`、`top_k_sentence=1` 和 `PYTHONHASHSEED=0`。
- 基线：`delta=0.4`、`lambda=0.05`。`0.4` 依据图 5 和发布参数，而不是论文正文中可能为笔误的 `delta=4`。
- 固定 lambda，delta 从 `0.1` 到 `0.9`，步长 `0.1`。
- 固定 delta，lambda 扫描 `0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0`。中间扫描值根据图 5 推断。
- 测量 LLM 准确率与包含匹配准确率的均值、激活/传播实体数、图搜索耗时，以及相对基线的 Top-5 重合度、排序变化和答案变化。

OFAT 不估计参数交互；未经重复的参数点不能确立唯一最优值。

## Q6：嵌入模型稳健性

**研究问题。** LinearRAG 在不同嵌入模型下是否仍然有效？

**设计理由。** 替换问题、查询实体、段落、句子和图实体共用的嵌入模型，保持其他条件不变。

**设计与测量。**

- 使用完整的 658 段 2Wiki 语料，以及种子 `20260921` 抽取的 100 道问题。
- 比较 `all-mpnet-base-v2`、`all-MiniLM-L6-v2`、`bge-large-en-v1.5` 和 `e5-large-v2`。
- 固定 Top-5、`max_iterations=3`、`top_k_sentence=1`、`delta=0.4`、`lambda=0.05` 和 `PYTHONHASHSEED=0`。
- 各模型隔离嵌入缓存；仅复用相同输入下与模型无关的 NER。
- 比较 LLM 准确率、包含匹配准确率及二者均值；相对 `all-mpnet-base-v2` 检查 Top-5 重合度、排序、激活/传播实体和检索耗时。

## Q7：大规模效率

**研究问题。** LinearRAG 索引成本如何随输入 token 规模变化？

**设计理由。** 确定性的 HotpotQA 前缀和固定分词器提供受控的本地规模代理。

**设计与测量。**

- 在独立准备入口使用 `all-mpnet-base-v2` 分词器生成各规模输入包，取确定性语料前缀，仅在分词器边界截断最后一段；输入准备耗时单独记录。
- 各规模进行独立冷索引，包含嵌入、NER 和图构建。排除生成式 LLM 调用、检索和评价。
- `setup_seconds` 仅测运行初始化，排除输入准备及输入包读取、核验；`index_seconds` 仍测 `LinearRAG.index(passages)`，包含嵌入、NER、图构建和缓存写入。继续报告每百万 tokens 耗时、缓存字节数、CUDA 峰值及实体/句子/图规模。
- 历史结果保留；新旧 `setup_seconds` 口径不同，不可直接比较。

本设计仅测量 LinearRAG。各规模单次本地观察不能确立稳定的扩展性，也不能证明论文 ATLAS-Wiki 规模下的表现。

## Q8：案例研究

**研究问题。** LinearRAG 与 HippoRAG 如何为表 7 的问题连接和检索证据？

**设计理由。** 固定小语料便于比较传播、实体链接和证据排序。分别检查答案正确性与机制正确性。

**设计与测量。**

- 使用 2Wiki 问题 `3a3c2efe0bdc11eba7f7acde48001122`：`What nationality is Beatrice I, Countess Of Burgundy's husband?`。
- 保留婚姻与国籍支持段落，加入种子 `20261001` 抽取的 18 个干扰段落。两种方法索引相同的 20 段语料，检索 Top-5。
- 除共同模型条件外，固定 LinearRAG 的 `max_iterations=3`、`iteration_threshold=0.4`、`passage_ratio=0.05` 和 `top_k_sentence=1`。
- 追踪 LinearRAG 的种子、选中的传播句、激活实体及分数。检查 HippoRAG 的查询实体、链接节点/相似度，以及与案例有关的图边。
- 比较支持段落排名和生成答案；分别检查 Beatrice I–Frederick Barbarossa 桥接和国籍证据。

支持段落是刻意保留的。该单例用于解释机制，不代表总体准确率，也不复现论文的 HippoRAG2 对照。
