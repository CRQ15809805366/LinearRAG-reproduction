# LinearRAG 是如何变成代码的？

理解了算法，再打开代码，仍然可能迷路。论文中的一个矩阵，可能变成几个字典；图中的一类节点，可能根本没有放进图对象；一个实体的“位置”，还可能同时有几种编号。

读代码时，可以始终跟着两件东西：一份文本怎样成为可检索的材料，一个问题怎样使用这些材料。每遇到一个变量，先弄清它装着什么，再看它如何变化。

本章解释官方实现的核心做法，阅读位置使用当前仓库整理后的路径。首次引入的官方版本为 `bcc94e66c221f798801255efba09311d6fbcd8d6`，来源见[基线记录](../../foundation/LINEARRAG_IMPORT.md)。当前文件还包含本地的路径、缓存身份与结构整理，不能把整个文件都当成未经改动的上游原件；需要核对原貌时，可在首次引入提交 `a5a73cb0df38cc9d82d003f5159403152cded21c` 中查看 `src/LinearRAG.py`。

## 1. 先找到程序的主线

从 [src/run.py](../../../src/run.py) 的 `main()` 看起。把资源准备的细节暂时收起来，主线可以读成下面这段。它是删去日志和文件写入后的流程摘写：

```python
embedding_model = load_embedding_model(args.embedding_model)
questions, passages = load_dataset(args.dataset_name)
llm_model = LLM_Model(args.llm_model)

config = LinearRAGConfig(...)
rag_model = LinearRAG(global_config=config)

rag_model.index(passages)
questions = rag_model.qa(questions)

# 保存 questions 后，由 Evaluator 读取预测并评价。
```

`index()` 接收语料，准备以后检索需要的东西。`qa()` 接收问题，先调用 `retrieve()` 取段落，再组织提示调用回答模型。`Evaluator` 则读取预测，比较预测答案与标准答案。

这三个阶段有不同的工作量。索引需要处理一整批材料；检索对每个问题寻找材料；生成读取检索结果并写出答案。观察“运行了多久”时，必须知道时间落在了哪一段。下一章的效率实验正是从这里拆开测量。

初始化 `LinearRAG` 时，会绑定配置、NER、嵌入存储和 LLM 对象，并建立空的 `igraph` 图。LLM 对象在那里，不代表索引和检索会使用它。真正调用 `self.llm_model.infer` 的问答步骤在 `qa()` 中。

本地 `src.run` 是走通系统的入口。Q1–Q8 各有自己的实验入口和参数；沿主线读懂方法后，再看实验入口，就容易辨认额外的安排。

## 2. 论文中的对象，在内存里是什么？

继续使用[上一章](algorithm.md)的三段材料。运行输入通常带有编号前缀，例如：

```text
0:Alice founded Northbridge Institute.
1:Northbridge Institute is located in Bristol.
2:Alice visited Paris during a summer holiday.
```

这些字符串是实际进入索引的 passage。编号也属于字符串内容，会参与文本身份和编码；它还会被后面的相邻段落连接代码使用。

原文、向量和编号主要保存在 [EmbeddingStore](../../../src/common/embedding_store.py) 中。段落、句子、实体各有一份 store，分别使用 `passage`、`sentence`、`entity` 命名空间。文本通过内容哈希得到 ID，同一命名空间中相同文本会落到同一身份上。

可以把 store 理解成一本同时提供几种查法的小册子：

| 成员 | 用它回答什么 |
|---|---|
| `hash_id_to_text` | 这个 ID 对应哪段文字？ |
| `text_to_hash_id` | 这段文字的 ID 是什么？ |
| `hash_id_to_idx` | 这个 ID 的向量在第几行？ |
| `hash_ids`、`texts`、`embeddings` | 按对应位置取 ID、原文和向量 |

“第几行”只在当前向量存储中有意义。实体向量的第 3 行，与图对象的第 3 个节点不是同一个概念。因此核心实现还维护 `node_name_to_vertex_idx`，把哈希 ID 翻译成图节点位置。读传播和 PPR 时，要留意一个下标正在索引哪个容器。

句子与实体之间的联系由两份映射保存：

```text
entity_hash_id_to_sentence_hash_ids：从实体找到提到它的句子
sentence_hash_id_to_entity_hash_ids：从句子找到它提到的实体
```

例如，假设 NER 得到了上一章约定的结果，Alice 会关联创办句子和旅游句子；创办句子会关联 Alice 与 Northbridge Institute。这已经足以让默认检索沿“实体 → 句子 → 实体”前进。

最终 PPR 使用的 `self.graph` 另有职责：`add_nodes()` 加入实体和段落节点，句子没有加入这张 `igraph` 图。论文中的 Tri-Graph 因而在代码中分成了两种表示：句子联系用映射承载，实体与段落联系用图承载。读到这一步，再看到 `entity_to_sentence` 不在图里，就不会误以为句子丢失了。

## 3. 原始文本怎样一步步变成索引？

打开 [LinearRAG.py](../../../src/methods/linear/LinearRAG.py) 的 `index()`，它的顺序可以分成几个连贯的动作。

首先把段落交给 `passage_embedding_store.insert_text()`。store 找出尚未保存的文本，用嵌入模型编码，再把文本、哈希 ID 和向量保存到 Parquet。编码使用 `normalize_embeddings=True`，后续向量点积便可以作为归一化向量的余弦相似度。

接着读取已有 NER 结果，对尚未处理的段落执行 `SpacyNER.batch_ner()`。[ner.py](../../../src/methods/linear/ner.py) 利用 spaCy 得到实体和它所在的句子，跳过 `ORDINAL` 与 `CARDINAL`，保存段落到实体、句子到实体的联系。这里依赖模型的句子划分和实体识别结果，实际输出不一定等于人手标注。

`extract_nodes_and_edges()` 再把这些结果整理为实体集合、句子集合和双向映射。实体与句子分别编码保存，字符串联系转换为哈希 ID 联系。经过这一步，程序既能找到邻居，也能取得邻居的向量。

最后构建 PPR 图。`add_entity_to_passage_edges()` 统计某实体的字符串在段落中出现了几次，用它占该段已识别实体总出现次数的比例作为边权。这里使用原字符串的 `count()`，属于字符串计数，没有另做完整的词边界或别名判断。它与论文中二值的 contain matrix 表达不同，代码实际使用的是带权联系。

`add_adjacent_passage_edges()` 还会解析开头的数字，将按编号排序的相邻段落连接，边权为 1.0。例子中的三段都有前缀，所以 P0–P1、P1–P2 也会有边。这个“相邻”来自输入编号，不能自动理解为两段内容具有相同主题；编号列表有间隔时，代码仍连接排序列表中的邻居。因此，输入的顺序与编号是会影响图的条件。

`augment_graph()` 将收集的节点和边写入 `igraph`，随后把图保存为 `LinearRAG.graphml`。NER 结果、三份向量存储和图文件各自保留不同阶段的产物。之后再次调用 `index()`，已有文本和 NER 结果可以减少重新处理的工作，但这个方法仍会组织映射并执行构图相关步骤，不能直接把第二次 `index()` 的时间视为从零索引时间。

当前仓库在开始索引前增加了 `prepare_linear_cache()`，核对实际语料、嵌入模型与 NER 身份。它解决的是本地实验中“目录名字相同，里面却是另一套向量”的问题。这个安排属于本地缓存管理，理解官方算法时应与传播和排序逻辑分别辨认。

## 4. 一个问题怎样穿过检索代码？

先找到 `retrieve()`。它为问题编码，然后调用 `get_seed_entities()`。后者对问题做 NER，把问题实体编码，与所有语料实体计算点积，逐个选择相似度最高的候选。

这一步使用 `argmax`，没有额外的最低匹配分数门槛。已有问题实体时，最相似候选会成为种子；传播阈值随后决定它是否继续扩展。起点选择与扩展剪枝是两个不同动作。

如果问题 NER 没有得到实体，`retrieve()` 会走 `dense_passage_retrieval()`，按问题与段落向量相似度取 Top-k。该分支直接返回段落，不执行实体传播和 PPR。所以，即使结果由 `LinearRAG` 对象返回，也需要查看它实际走了哪条路。

有种子时进入 `graph_search_with_seed_entities()`。默认 BFS 路径的骨架如下，参数列表为便于阅读作了省略：

```python
entity_weights, actived_entities = self.calculate_entity_scores(...)
passage_weights = self.calculate_passage_scores(
    question, question_embedding, actived_entities
)
node_weights = entity_weights + passage_weights
passage_ids, passage_scores = self.run_ppr(node_weights)
```

先停在 `calculate_entity_scores()`。种子写入 `actived_entities`，BFS 的种子层级为 1；同时把种子分数写到按图节点位置排列的 `entity_weights` 中。随后反复做三件事：找到当前实体尚未使用的关联句子，按句子与整个问题的相似度取 `top_k_sentence`，把当前实体分数乘句子相似度，传给句子中的实体。

回到 Alice 的例子，假设程序选中了创办句子。它会处理句中的 Alice 和 Northbridge Institute。程序没有要求下一实体必须是从未见过的新名字，因此 Alice 自己也可能再次收到贡献。达到阈值的实体进入下一轮，使用过的句子则记录下来，限制后面重复使用。

循环条件是 `iteration < max_iterations`，计数从 1 开始。因而 `max_iterations=3` 在这条 BFS 中最多执行两轮扩展，种子相当于最初的一层。把参数名称直接读成“额外走三轮”，会多算一轮。

然后进入 `calculate_passage_scores()`。它先给所有段落计算向量相似度，并做 min-max 归一化；再逐段查看激活实体的小写字符串出现次数，结合实体分数、层级和对数项加分。`passage_ratio` 控制直接相似度的贡献，`passage_node_weight` 调整整项段落起始权重。

最后，`run_ppr()` 把实体权重和段落权重相加，清理负值与 NaN，传入 `personalized_pagerank(reset=...)`。这使问题相关信号成为反复重新出发的分布。PPR 在已有的实体、段落图上计算全部节点分数，然后只取段落节点，排序并还原为原文。

代码没有在这里把所有未激活实体从图上删除。理解“激活”时，应想到它改变了问题相关的起始权重，而整张 PPR 图仍保留已有联系。

选好的段落返回 `retrieve()`，组成 `sorted_passage` 和 `sorted_passage_scores`。接着 `qa()` 拼接段落和问题，调用 LLM，提取 `Answer:` 后的文本作为 `pred_answer`。`gold_answer` 会保存在结果中供评价使用，但没有拼入这份生成提示。

## 5. 论文与代码应当怎样对照着读？

论文给出了紧凑的计算表达，程序还要决定怎样访问邻居、限制工作量和保存状态。两者可以互相帮助理解，但有些差别需要明确记住。

| 论文中的表达 | 默认官方实现中的具体位置与做法 |
|---|---|
| 公式 (1)：段落包含实体 | `add_entity_to_passage_edges()`；按字符串出现次数比例设置边权 |
| 公式 (2)：句子提及实体 | `extract_nodes_and_edges()` 与两份哈希 ID 映射 |
| 公式 (3)：种子匹配 | `get_seed_entities()`；向量相似度最高候选 |
| 公式 (4)：问题与句子相似度 | BFS 中对当前实体可访问的句子执行向量点积 |
| 公式 (5)：矩阵聚合与 MAX 更新 | `calculate_entity_scores()`；逐实体、逐句子传播，带 Top-k、阈值和句子使用记录 |
| 公式 (6)：图上重要性分配 | `run_ppr()`；带边权、damping 与问题相关 reset 的库调用 |
| 公式 (7)：段落混合初始化 | `calculate_passage_scores()`；归一化段落相似度加实体贡献 |

公式 (5) 尤其需要慢一点看。BFS 向 `entity_weights` 累加接受到的贡献；`actived_entities` 保存的则是实体最近一次更新的分数和层级，后续段落加分使用这份记录。同一轮多个来源触发同一实体时，`new_entities` 的字典赋值还会覆盖前一次记录。于是“累计的图实体权重”和“用于继续传播、段落加分的实体记录”不能混为同一个量。

这些行为与论文公式逐项取 MAX 的更新不同。知道这点后，讨论某次实验是在验证什么就更准确：当前 Q1–Q8 使用的是所引入官方实现的默认 BFS 行为。

官方代码还提供 `calculate_entity_scores_vectorized()`，可通过 `use_vectorized_retrieval` 切换。它构造稀疏矩阵，把实体到句子的联系存成 E × S，使用时再转置来乘实体向量。论文 M 的形状为 S × E；辨认存储方向之后，这个转置就容易理解。

该分支也带有 Top-k、去重、阈值和累计分数逻辑，不能因为出现矩阵乘法，就认定它逐字实现了公式 (5)，或与 BFS 在所有输入上严格等价。先理解默认路径，再将可选分支作为另一份需要对照行为的实现阅读。

此外，`config.py`、`src.run` 命令行默认值和各实验传入的参数可能不同。研究一份输出时，应回看那次运行的配置；仅打开配置类读默认值，还不能得知那次运行实际用了什么。

## 6. 回到本地仓库，怎样找到这些实现？

重新接手时，可以先按下面几处定位，等有了具体疑问再继续深入。

| 想看的动作 | 当前文件 |
|---|---|
| 索引、检索、生成的主线 | [src/run.py](../../../src/run.py) |
| 图检索的主要计算 | [LinearRAG.py](../../../src/methods/linear/LinearRAG.py) |
| 问题实体、句子与实体识别 | [ner.py](../../../src/methods/linear/ner.py) |
| 文本与向量怎样保存 | [embedding_store.py](../../../src/common/embedding_store.py) |
| 参数成员 | [config.py](../../../src/methods/linear/config.py) |
| 本地缓存的阶段身份 | [cache_identity.py](../../../src/common/cache_identity.py) |
| 答案怎样调用模型、怎样评价 | [utils.py](../../../src/common/utils.py)、[evaluate.py](../../../src/common/evaluate.py) |

例如，想知道 P1 为什么没有进入 Top-5，可以先看问题有没有种子，再看是否激活了 Northbridge Institute，再看它给 P1 的起始分数，最后看 PPR 后的排名。想知道一次运行为什么很快，则回到索引和缓存阶段，确认它实际重做了什么。

这些问题让阅读有了落点，也让目录不再只是一串路径。带着“哪一步正在影响现象”去看函数，能逐渐把仓库读成一个完整过程。

下一章从这条过程出发，解释[本地实验怎样设计](experiments.md)：怎样安排输入和比较，才能知道一次变化说明了什么。
