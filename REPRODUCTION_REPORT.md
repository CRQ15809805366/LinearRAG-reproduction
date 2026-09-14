# LinearRAG 最小本地冒烟验证报告

## 1. 范围与结论

工作目录：`D:\code\LinearRAG-reproduction`。

官方 LinearRAG 的核心索引和检索流程，已经在这台 Windows 机器上通过可人工核验、仅使用 CPU 的极小冒烟验证。这不代表已经复现论文基准结果、答案生成、GPT 评测或扩展性结论。

## 2. 源码与目录结构

官方来源：

- 地址：https://github.com/DEEP-PolyU/LinearRAG
- 分支：`main`
- 提交：`bcc94e66c221f798801255efba09311d6fbcd8d6`
- 获取日期：2026-09-13（Asia/Shanghai）
- 导入方式：Git 传输连接重置后，改用 GitHub codeload 下载精确提交；删除临时检出目录前，对全部 15 个官方文件执行了 SHA-256 校验。
- 审计记录：`SOURCE.md`。

最终逻辑结构如下，其中生成的缓存会显示在目录中，但被 Git 忽略：

```text
LinearRAG-reproduction/
|-- 2510.10114v4.pdf
|-- 提示词.md
|-- readme.md                       # 官方 README 的中文版本
|-- requirements.txt               # 官方依赖文件
|-- run.py                         # 官方组合入口
|-- src/                           # 算法逻辑未改，说明注释已中文化
|-- scripts/                       # 官方脚本
|-- figure/                        # 官方图片
|-- requirements-windows.txt       # 本地兼容与观测依赖
|-- smoke_test.py                  # 无 OpenAI 的冒烟入口与轨迹观测器
|-- examples/smoke/                # 原始及调整后的极小输入
|-- tools/                         # NER 探测与模型恢复脚本
|-- artifacts/                     # 输出、环境及安装证据
|-- model/                         # 被忽略的本地嵌入模型
|-- import/smoke/                  # 被忽略的官方索引及缓存
`-- .venv/                         # 被忽略的独立 Python 环境
```

任务中填写的论文路径是 `D:\code\2510.10114v4.pdf`；实际提供并使用的文件位于 `D:\code\LinearRAG-reproduction\2510.10114v4.pdf`。

## 3. 机器与环境

以下信息来自实际检测，而非预设：

- Windows 11 专业版 10.0.26200（内部版本 26200）。
- Intel Core i7-13620H，16 个逻辑处理器。
- 物理内存 15.74 GiB；首次检测时可用 7.86 GiB。
- NVIDIA GeForce RTX 4050 Laptop GPU；显存 6141 MiB，首次检测时空闲 5877 MiB；驱动版本 572.83。
- 系统 Python 3.10.11；项目 Python 3.9.25。
- Git 2.47.0.windows.2；uv 0.11.21。
- 检测时剩余空间：C 盘 20.54 GiB，D 盘 267.09 GiB。

精确快照保存在 `artifacts/system-environment.json` 和 `artifacts/environment-freeze.txt`。关键安装版本为 NumPy 1.21.0、Pandas 1.3.0、spaCy 3.6.1、en-core-web-sm 3.6.0、SentenceTransformers 2.2.2、Transformers 4.30.2、PyTorch 2.8.0+cpu、igraph 0.11.8、PyArrow 12.0.1 和 sentencepiece 0.1.99。

环境恢复命令：

```powershell
uv python install 3.9
uv venv --python 3.9 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-windows.txt
uv pip install --python .venv\Scripts\python.exe "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.6.0/en_core_web_sm-3.6.0-py3-none-any.whl"
powershell -ExecutionPolicy Bypass -File tools\download_embedding_model.ps1
.venv\Scripts\python.exe smoke_test.py
```

官方固定依赖在 Python 3.10 上安装失败，原因是 NumPy 1.21.0 没有适用于 Windows/Python 3.10 的 wheel，源码构建又需要 MSVC。切换 Python 3.9 后，官方直接依赖可以解析。仍需固定一个传递依赖：sentencepiece 0.2.2 与官方 Transformers 4.30.2 组合时触发原生访问冲突，因此 `requirements-windows.txt` 将其固定为 sentencepiece 0.1.99。`psutil` 只用于内存观测。`requirements.txt` 中的原始约束保持不变，失败日志保存在 `artifacts/setup/`。

官方首选 spaCy 模型是 `en_core_web_trf`，README 还为 medical 数据集指定了 SciSpaCy 模型。本次低资源冒烟使用 `en_core_web_sm`。官方嵌入模型路径是 `model/all-mpnet-base-v2`，但其 438 MB 权重下载多次因 CDN 连接中断而失败，因此本次在 CPU 上使用 API 兼容的 `all-MiniLM-L6-v2`。模型固定在提交 `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`，模型文件 SHA-256 为 `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`。

## 4. 官方程序实际提供的功能

- 真实组合入口：`run.py:main()`。
- 索引入口：`LinearRAG.index(passages)`。
- 检索入口：`LinearRAG.retrieve(questions)`；默认使用 BFS，可选向量化检索。
- 答案生成：`LinearRAG.qa()` 通过 OpenAI 聊天补全调用 `LLM_Model.infer()`。
- GPT 评测：`Evaluator.evaluate()` 调用大语言模型正确性裁判，同时计算答案包含准确率。
- 数据集：`dataset/<name>/chunks.json` 是字符串 JSON 列表；`questions.json` 是对象列表，每个对象至少包含 `question` 和 `answer`。`run.py` 会给文本块加上 `index:` 前缀。
- 模型加载：SentenceTransformer 从本地或配置路径加载；官方 `run.py` 强制设置 `device="cuda"`；spaCy 按包名加载。
- Python：官方 README 建议 Python 3.9；`requirements.txt` 包含年代较早的精确版本约束。
- CUDA：官方 `run.py` 硬编码 `CUDA_VISIBLE_DEVICES=4`；向量化检索会在可用时选择 CUDA。冒烟入口不采用这两项设置，而是明确使用 CPU/BFS。
- 路径：官方缓存默认写入 `./import/<dataset>/`；日志与预测默认写入 `results/<dataset>/<timestamp>/`。从仓库根目录启动时，相对路径可在 Windows 上工作。
- OpenAI 边界：`run.py`、`qa()` 和大语言模型评测需要 API 配置。直接调用时，passage 读取、NER、嵌入、索引、BFS 或向量化检索、passage 初始化、PPR 和 Top-k 检索均不需要 OpenAI。

论文描述的 Tri-Graph 包含 passage、sentence 和 entity 三类节点。在所检查的提交中，sentence 节点保存在句子嵌入存储及 BFS 使用的句子—实体映射中，而 PPR 实际使用的 igraph 只包含 passage 和 entity 顶点。这里只记录实现差异，没有修改代码使二者强行统一。

## 5. 本地修改与算法边界

1. 环境适配：`requirements-windows.txt`、本地 Python 3.9 环境、本地 CPU 嵌入模型和小型 spaCy 模型。
2. 极小样例接入：`examples/smoke/*.json` 和 `smoke_test.py`。
3. 可观察性：`ObservableLinearRAG` 记录查询匹配、官方 passage 重置权重和 PPR 分数，并执行只用于观测的 BFS 重放。程序断言重放得到的实体权重和分数与官方方法输出一致；重放数据不会参与最终排序。环境与原始结构化输出保存在 `artifacts/`。
4. 算法修改：无。包括 `src/LinearRAG.py` 在内的算法表达式和可执行语句没有因中文本土化而改变；说明注释与 docstring 已翻译为中文。

运行配置刻意保持很小：CPU 嵌入、批大小 2、单 worker、BFS、每个实体选取 Top-1 句子、配置三轮迭代，阈值为 0.05。阈值含义及比较逻辑保持官方原样；之所以使用 0.05，是因为 MiniLM 的第二跳分数为 0.06655，最初尝试的 0.1 会按官方逻辑将其正确剪枝。

## 6. 极小输入与实体识别

原始要求中的输入保存在 `examples/smoke/original_input.json`。passage 带编号前缀时，en-core-web-sm 3.6.0 未能在 passage 1 中识别 `Beatrice I`，但问题中识别出了 `Beatrice`；继续使用原句会导致查询与图实体错误匹配。完整失败观测保存在 `artifacts/smoke_result.json` 的 `original_ner_failure_evidence` 字段中。

调整后的输入：

```text
P1: Beatrice, Duchess of Burgundy, was married to Frederick Barbarossa.
P2: Frederick Barbarossa was King of Germany.
P3: Paris is the capital of France.
Q:  What nationality was the husband of Beatrice, Duchess of Burgundy?
```

加入官方 `index:` 前缀后的实际实体识别结果：

- P1 sentence: `0:Beatrice, Duchess of Burgundy, was married to Frederick Barbarossa.` -> Beatrice (ORG), Frederick Barbarossa (PERSON).
- P2 sentence: `1:Frederick Barbarossa was King of Germany.` -> Frederick Barbarossa (PERSON), Germany (GPE).
- P3 sentence: `2:Paris is the capital of France.` -> Paris (GPE), France (GPE).
- Query -> Beatrice (ORG).

其中 ORG 标签并不准确，但官方代码只使用实体文本，并且仅过滤 CARDINAL/ORDINAL；本项目没有增加标签修正规则。

## 7. 图、匹配、传播与排序

图与索引统计：

- Passage 节点：3。
- 句子嵌入存储节点：3。
- Entity 节点：5。
- 实际 igraph 顶点：8（3 个 passage + 5 个 entity）。
- 实际 igraph 边：8（6 条 passage—entity 出现边 + 2 条相邻 passage 边）。
- BFS 使用的 entity—sentence 连接：6。

查询匹配：

- 查询实体 `beatrice` → 图实体 `Beatrice`，余弦相似度为 1.00000。

带一致性断言的观测器记录到的官方 BFS 传播过程：

- 第 1 轮以 1.00000 处理 Beatrice；选中 P1 句子，查询相似度为 0.75503；以 0.75503 激活 Frederick Barbarossa 和 Beatrice。
- 第 2 轮处理 Frederick Barbarossa；选中 P2 句子，相似度为 0.08814；以 0.06655 激活 Germany 和 Frederick Barbarossa。此时 Beatrice 没有未使用句子。
- 观测器一致性断言：通过。

配置 `passage_node_weight=0.05` 后，官方计算得到的 passage 节点初始权重：

- P1: 0.1122276154.
- P2: 0.0015143826.
- P3: 0.0066872624.

PPR 与最终 Top-k：

1. P1, 0.2661878878.
2. P2, 0.1333502280.
3. P3, 0.0267827737.

人工预期：通过。P1 和 P2 均排在 P3 前；Beatrice 是种子实体；Frederick Barbarossa 和 Germany 均出现在真实计算得到的传播轨迹中。

最终保存的运行从空的 `import/smoke` 缓存重新构建索引，耗时约 1.21 秒。进程 RSS 从约 331.3 MiB 增长到 520.3 MiB，峰值工作集约为 545.2 MiB。`nvidia-smi` 显示运行前后均占用 45 MiB；检索和模型负载均未放到 GPU。没有发生 MemoryError、系统错误 1455、CUDA 显存不足或进程终止。

## 8. 验收矩阵

| 验收项 | 结果 | 证据 |
|---|---|---|
| A. 隔离 | 通过 | 所有正式文件、缓存和结果均位于工作目录；未访问封档旧项目。 |
| B. 来源 | 通过 | `SOURCE.md` 记录了地址、分支、提交、导入方式、上游原始哈希及中文本土化边界；没有嵌套 Git 仓库。 |
| C. 环境 | 经记录的适配后通过 | 已检测软硬件；使用独立 `.venv`；保留 Python 3.10 的原始失败；Python 3.9 可恢复；无需 OpenAI。 |
| D. 程序 | 通过 | 官方索引和检索方法可以运行；极小索引及检索完成，无不可接受的内存错误或硬编码答案。 |
| E. 行为 | 调整措辞后通过 | 结构化轨迹包含匹配、两跳传播及 P1/P2/P3 排名；保留了原始措辞的失败结果。 |
| F. 可复现 | 通过 | 已保存输入、输出、环境冻结、安装失败、源码哈希、模型下载脚本和单命令冒烟入口。 |

## 9. 仍然存在的限制

- 冒烟使用 en-core-web-sm 和 all-MiniLM-L6-v2，而不是论文或官方默认的 en-core-web-trf 和 all-mpnet-base-v2，因为本阶段优先完成小规模本地运行，且当前网络无法完整下载较大模型。
- 原始 `Beatrice I` 措辞在小型 spaCy 模型上无法通过 passage 实体识别，因此必须使用报告中已记录的调整措辞。
- 官方 `run.py` 的运行逻辑仍然硬编码 GPU 选择，并将检索与 OpenAI 生成及评测绑定。本阶段应使用 `smoke_test.py`。
- 本次测试只有三个 passage 和一个问题，无法证明基准准确率、扩展能力、GPU 向量化行为、答案质量或与论文表格一致。

## 10. 本阶段可以和不可以得出的结论

可以得出：官方 LinearRAG 核心索引与检索流程，已经使用极小人工样例在这台 Windows 机器上完成本地冒烟验证；实体识别、嵌入、图构建、查询匹配、实体传播、passage 初始化、PPR 和 Top-k 输出均可观察。

不能得出：论文已经完整复现、官方基准分数已经达到、默认大模型一定能在本机运行，或生成、GPT 评测及扩展性结论已经得到验证。

唯一的下一阶段建议：网络条件允许后，只用官方 `en_core_web_trf` 和 `all-mpnet-base-v2` 重跑同一个样例，不改变语料、算法或评测范围，仅比较实体识别、匹配与传播轨迹。
