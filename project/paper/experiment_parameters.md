# LinearRAG 论文实验参数

以下参数整理自官方仓库原有的 `scripts/run.sh`，用于通过 `run.py` 运行四个数据集实验。

所有实验均使用：

- 嵌入模型：`data/input/models/all-mpnet-base-v2`
- LLM：`gpt-4o-mini`
- `max_workers`：`16`

| 数据集 | spaCy 模型 | `max_iterations` | `passage_ratio` | `iteration_threshold` | `top_k_sentence` |
|---|---|---:|---:|---:|---:|
| `medical` | `en_core_sci_scibert` | 3 | 1.5 | 0.5 | 1 |
| `musique` | `en_core_web_trf` | 5 | 2.0 | 0.1 | 4 |
| `2wikimultihop` | `en_core_web_trf` | 3 | 0.05 | 0.4 | 1 |
| `hotpotqa` | `en_core_web_trf` | 3 | 0.05 | 0.4 | 1 |

这些参数不是 `run.py` 的全部默认值，而是作者针对不同数据集留下的实验配置。原脚本中的配置块默认处于注释状态，运行具体实验时需要显式传入对应参数。
