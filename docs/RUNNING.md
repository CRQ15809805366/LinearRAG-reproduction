# 运行说明

本文只说明稳定的运行前提和入口。各实验的研究问题、参数设计和已完成结果分别以[实验设计](foundation/EXPERIMENT_DESIGN.md)和[实验状态索引](EXPERIMENT_STATE.md)为准，不在这里重复维护。

## 运行环境

建议使用以下环境：

- Windows PowerShell
- Python 3.9
- NVIDIA GPU 与可用的 CUDA 环境

在项目根目录创建并激活虚拟环境，然后安装依赖：

```powershell
py -3.9 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

LinearRAG 默认使用 `en_core_web_trf` 进行英文实体识别。该模型不包含在 `requirements.txt` 中，需要单独安装：

```powershell
.venv\Scripts\python.exe -m spacy download en_core_web_trf
```

Medical 实验使用的领域模型、HippoRAG 对照环境以及其他实验专用依赖，应按对应实验入口和实验记录准备；它们不是基础冒烟检查的共同前提。

## 本地资源

数据集、本地嵌入模型、缓存和生成结果不会随 GitHub 仓库一起提供。运行前需要自行准备相应资源。

默认 LinearRAG 入口需要：

- `data/input/datasets/<dataset_name>/questions.json`
- `data/input/datasets/<dataset_name>/chunks.json`
- `data/input/models/all-mpnet-base-v2/`

`questions.json` 是问题对象列表，`chunks.json` 是段落字符串列表。实验构造输入、缓存和输出的目录职责见[数据目录说明](../data/README.md)。

如果资源位于其他位置，可以通过相应入口的命令行参数显式指定；可用参数始终以入口的 `--help` 输出为准。

## LLM 凭据

需要答案生成、LLM 评价或 HippoRAG 抽取的运行，会从项目根目录未跟踪的 `.env.local` 读取：

```dotenv
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
```

程序直接读取该文件并把凭据传给客户端，不要求将其写入系统或进程环境。不要提交、展示或复制 `.env.local` 的内容。

不调用 LLM 的冒烟检查不需要这些凭据。

## 冒烟检查

冒烟检查使用少量本地样例调用实际 LinearRAG 索引和检索流程，不执行答案生成或 LLM 评价，也不代表正式实验：

```powershell
.venv\Scripts\python.exe -m src.smoke_test
```

查看可选参数：

```powershell
.venv\Scripts\python.exe -m src.smoke_test --help
```

默认结果写入 `data/output/smoke/`。只有当本地 embedding 模型、spaCy 模型和 CUDA 环境均已准备完成时，该检查才能直接运行。

## 运行 LinearRAG

下面的命令展示普通端到端入口。它会加载数据集、建立或复用索引、生成答案并执行评价，因此可能使用 GPU 和付费 LLM API：

```powershell
.venv\Scripts\python.exe -m src.run --dataset_name 2wikimultihop --max_questions 10
```

查看当前支持的全部参数：

```powershell
.venv\Scripts\python.exe -m src.run --help
```

默认检索方式是复现实现使用的 BFS 实体传播。只有显式传入 `--use_vectorized_retrieval` 时才会使用可选的稀疏矩阵实现。两种路径不应在结果记录中混写。

普通端到端结果写入 `data/output/<dataset_name>/<运行时间>/`。运行使用的模型、数据范围和关键参数应随结果一同记录。

## 运行 Q1–Q8 实验

各研究问题有独立入口：

```text
src/experiments/
├── q1_generation_accuracy/
├── q2_efficiency_analysis/
├── q3_ablation_study/
├── q4_retrieval_quality/
├── q5_hyperparameter_sensitivity/
├── q6_embedding_model_robustness/
├── q7_large_scale_efficiency/
└── q8_case_study/
```

不要把这里的普通运行示例直接当作正式复现实验。执行某个实验前，先查看其在[实验设计](foundation/EXPERIMENT_DESIGN.md)中的输入、对照和测量边界，再通过对应模块的 `--help` 核对当前参数。

已执行实验及其有效证据统一登记在[实验状态索引](EXPERIMENT_STATE.md)，详细解释保存在 `docs/records/`。新增一次运行不需要修改本文；只有安装方式、资源要求或入口结构发生变化时，才需要同步更新。

## 结果边界

- `src.smoke_test` 只验证小样例上的索引和检索流程，不是正式实验。
- `src.run` 是普通端到端入口，不自动满足 Q1–Q8 的受控实验协议。
- 缓存恢复结果不能作为冷索引耗时。
- 本地数据规模、模型、硬件和评价器与论文不同时，只能报告本地有界结果。
- 实际使用的模型名称必须记录；不得把本地模型结果替换成论文模型结果。
