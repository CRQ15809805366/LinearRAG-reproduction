# LinearRAG 复现与学习项目

这是一个用于阅读、运行和分析 LinearRAG 的本地项目。LinearRAG 是一种基于实体关系传播的 GraphRAG 方法；本仓库保留其主要实现，并围绕论文中的研究问题编写了 Q1–Q8 实验脚本。

目前仓库里有四个数据集和本地嵌入模型。项目已经完成若干有边界的本地实验，包括生成效果、索引效率、模块消融、检索质量、参数敏感性、嵌入模型比较、规模测试和案例分析。它们使用的样本规模、模型和数据条件与论文实验不完全相同，所以这些结果用于理解实现和观察本地行为，不等同于完整复现论文结果。

## 如何进入项目

在 Windows PowerShell 中进入仓库目录：

```powershell
Set-Location D:\code\LinearRAG-reproduction
```

用 VS Code 等编辑器打开这个目录即可浏览项目。想先了解各项实验做到哪里、结果在哪里，可以阅读 [Q1–Q8 实验索引](project/agent/experiments/README.md)。

## 文件和目录分布

| 路径 | 用途 |
|---|---|
| `src/` | 保留 `run.py`、`smoke_test.py` 运行入口和统一路径定义 `paths.py`。 |
| `src/linearrag/` | LinearRAG 本体、运行配置和实体识别。理解算法时从这里看。 |
| `src/baselines/` | 对照方法：普通向量 RAG，以及调用外部官方 HippoRAG 2024 的薄适配层；[最小接入说明](experiments/hipporag/README.md)。 |
| `src/common/` | 共享的嵌入存储、评价和工具代码。 |
| `experiments/` | Q1–Q8 实验脚本。它们为不同研究问题准备数据、设置条件和保存测量结果，调用 `src/` 中的实现。 |
| `data/input/datasets/` | 本地数据集：HotpotQA、2WikiMultiHopQA、MuSiQue 和 Medical。 |
| `data/input/models/` | 实验使用的本地嵌入模型文件。 |
| `data/input/examples/smoke/` | smoke 检查使用的小型固定输入。 |
| `data/output/cache/` | 可重建的运行缓存。共享缓存可以被后续运行复用。 |
| `data/output/experiment_results/` | 实验生成的预测、检索记录、测量、汇总和日志。正式结果与小样本设计检查都按实验 ID 分目录保存。 |
| `data/output/smoke/` | smoke 检查的结果文件。 |
| `project/human/` | 个人学习笔记、计划和理解记录。 |
| `project/agent/` | 给 Codex 等 Agent 使用的项目背景、实验记录、决策和论文材料。 |
| `research_report/` | 面向项目外读者的报告和整理后的研究材料。 |

`data/README.md` 解释运行数据的细分用途；`project/README.md` 解释个人记录和 Agent 记录的边界；`experiments/README.md` 说明实验脚本的职责。

## 如何使用

### 运行最小检查

在仓库根目录运行：

```powershell
.venv\Scripts\python.exe -m src.smoke_test
```

这项检查使用小型固定输入，验证基本的索引和检索流程。它不是完整数据集实验，也不会替代下面的问答运行。

### 运行一个 bounded 问答示例

```powershell
.venv\Scripts\python.exe -m src.run --dataset_name 2wikimultihop --max_questions 10
```

这会从 2WikiMultiHopQA 取前 10 个问题，运行检索、回答生成和评估，并把输出写入 `data/output/experiment_results/`。这是一个小规模端到端示例，不是论文的完整实验。主入口默认的数据集名称不在当前仓库中，因此运行时请显式指定 `hotpotqa`、`2wikimultihop`、`musique` 或 `medical`。

默认检索方式是论文实现中的 BFS 路径。`--use_vectorized_retrieval` 会切换到另一条可选实现；若目的是观察默认行为，不要添加这个参数。

### 查看或运行某项研究实验

每个 Q 问题都有对应的脚本目录，例如 `experiments/q3_ablation_study/`。运行前先看该目录的 README，再看 `project/agent/experiments/README.md` 中对应的实验记录。设计检查通常是小样本流程或机制检查；正式 bounded 结果的样本更大，但仍需要结合记录中的模型、样本规模和限制来解读。

## 模型配置

回答生成和 LLM 评估使用 OpenAI 兼容服务。项目从根目录未纳入 Git 的 `.env.local` 读取 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`；不要提交或展示这个文件的内容。当前本地实验使用的模型是 `qwen3.8-flash`，它不是论文中的 `gpt-4o-mini`。

## 阅读顺序建议

1. 先读本页和 `project/agent/experiments/README.md`，了解项目结构和 Q1–Q8 的当前状态。
2. 想理解 LinearRAG 怎么运行，从 `src/run.py` 进入 `src/linearrag/LinearRAG.py`，查看索引、检索和问答流程；评价代码在 `src/common/evaluate.py`。
3. 想理解某项实验为什么这样设计，查看对应的 `experiments/q*/README.md` 和 `project/agent/experiments/q*/` 记录。
4. 想看实际运行结果，进入记录给出的 `data/output/experiment_results/` 路径；不要把缓存目录和实验结果目录混为一谈。
