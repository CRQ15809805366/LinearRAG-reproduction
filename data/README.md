# 数据目录

按数据在实验中的用途分成三类：**输入、缓存、输出**。

## 目录索引图

```text
data/
├── input/                           # 实验使用什么
│   ├── datasets/                    # 原始数据集
│   │   ├── hotpotqa/
│   │   ├── 2wikimultihop/
│   │   ├── musique/
│   │   └── medical/
│   ├── derived_corpora/              # 从原始数据构造的实验语料及审查材料
│   ├── models/                       # 本地嵌入模型
│   └── smoke/                        # 冒烟检查的小样例
├── cache/                           # 哪些中间计算可以复用
│   ├── linearrag/
│   │   ├── datasets/                 # 原始数据集的索引缓存
│   │   ├── derived_corpora/           # 构造语料的索引缓存
│   │   └── embedding_models/         # 按嵌入模型区分的缓存
│   └── hipporag/                     # 抽取、问题 NER、索引及查询缓存
└── output/                          # 每次实验实际产生什么
    ├── q1_generation_accuracy/       # 生成准确率
    ├── q2_efficiency_analysis/       # 索引与检索效率
    ├── q3_ablation_study/            # 模块消融
    ├── q4_retrieval_quality/         # 检索质量
    ├── q5_hyperparameter_sensitivity/ # 超参数敏感性
    ├── q6_embedding_model_robustness/ # 嵌入模型鲁棒性
    ├── q7_large_scale_efficiency/    # 大规模索引效率
    └── q8_case_study/                # 单例机制与方法对照
```

索引图列出当前目录的主要层级，不展开模型文件、缓存 ID 和每次运行的文件。

## 缓存复用

各入口通过 `src/common/cache_identity.py` 按阶段计算身份，只纳入影响该阶段结果的输入与配置；具体复用条件见缓存中的 `manifest.json`。相同语料与抽取配置下，更换 embedding 或查询参数可复用抽取结果。Q2/Q7 冷索引使用各自输出目录内的独立空缓存，恢复已有产物不能计为冷构建。

## 分类依据

| 目录                       | 放什么                                                           | 为什么放这里                                                       |
| -------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------ |
| `input/datasets/`        | 数据集的问题、段落和参考答案                                     | 实验的原始输入                                                     |
| `input/derived_corpora/` | 抽样、加入干扰段落等操作得到的语料，以及构造清单、来源和审查材料 | 虽由程序生成，但会成为后续实验的输入                               |
| `input/models/`          | 下载到本地的嵌入模型文件                                         | 运行依赖的模型资源                                                 |
| `input/smoke/`           | `input.json` 和 `original_input.json`                        | 用少量样例检查程序是否能运行；不作为正式实验结果                   |
| `cache/linearrag/`       | 向量、NER、图等可复用的中间产物                                  | 避免重复计算；按语料或嵌入模型区分                                 |
| `cache/hipporag/`        | 段落抽取、问题 NER、索引及查询产物                               | 复用已完成的抽取和检索准备工作；目录中的 ID 用于区分不同输入与配置 |
| `output/q*/`             | 各次运行的配置、样本、预测、评价、测量和日志                     | 保存实际运行结果及其解释依据，先按实验问题，再按运行目录组织       |
