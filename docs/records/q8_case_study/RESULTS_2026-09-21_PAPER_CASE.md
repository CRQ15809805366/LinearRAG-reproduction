# Q8 论文案例结果：单个 2WikiMultiHopQA 样例

## 执行状态

论文单例于 2026-09-21 完成。运行以只读模式使用已有 2WikiMultiHopQA 缓存，未重建或改写索引。

原始证据：`data/output/q8_case_study/q8-paper-case-20260921/result.json`

模型输出为 `German`，标准答案为 `Germany`。原始精确包含匹配字段为 `false`，但在该国籍问题中二者语义等价。

## 范围与固定条件

- 数据集：`2wikimultihop`。
- 问题 ID：`3a3c2efe0bdc11eba7f7acde48001122`。
- 问题原文：`What nationality is Beatrice I, Countess Of Burgundy's husband?`。
- 标准答案：`Germany`。
- 嵌入模型：`all-mpnet-base-v2`。
- 生成模型：`qwen3.8-flash`。
- 检索：官方 BFS 路径，Top-5。
- 参数：`max_iterations=3`、`iteration_threshold=0.4`、`passage_ratio=0.05` 和 `top_k_sentence=1`。
- 索引状态：只读恢复已有缓存。

未执行基线、批量评价或其他 GraphRAG 对照。

## 实体传播轨迹

| 步骤 | 来源实体 | 选中句子（来源原文） | 句子分数 | 激活实体 |
|---:|---|---|---:|---|
| 0 | `beatrice i` | 种子实体 | 1.0000001 | 仅种子 |
| 1 | `beatrice i` | Beatrice I was Holy Roman Empress by marriage to Frederick Barbarossa | 0.7454857 | `holy roman`, `1148`, `frederick barbarossa`, `beatrice i`, 日期片段和 `burgundy`, 各自分数为 `0.7454858` |
| 2 | `holy roman` | Beatrice I was the daughter of Otto I and Adelaide of Italy | 0.4219636 | 无实体超过阈值 |
| 2 | `1148` | 关于 Urraca of Portugal 的句子 | 0.3766946 | 无实体超过阈值 |
| 2 | `frederick barbarossa` | Frederick Barbarossa was Holy Roman Emperor from 1155 | 0.2433424 | 无实体超过阈值 |
| 2 | `beatrice i` | Otto I was the fourth son of Frederick I and Beatrice I | 0.5916153 | 五个实体，分数为 `0.4410408` |
| 2 | `burgundy` | Beatrice of Navarre was Duchess of Burgundy by marriage to Hugh IV | 0.7726580 | 五个实体，分数为 `0.5760056` |

本地轨迹在第 1 轮激活 `frederick barbarossa`，但第 2 轮没有将 `germany` 激活为图实体。第 2 轮选中的 Frederick Barbarossa 句子传播分数为 `0.2433424`，低于 `0.4` 阈值。

## Top-5 检索

| 排名 | 来源段落索引 | 段落分数 | 证据作用 |
|---:|---:|---:|---|
| 1 | 28 | 0.0714092 | 包含 Beatrice I 与 Frederick Barbarossa 的婚姻证据 |
| 2 | 27 | 0.0684109 | 包含 Frederick Barbarossa 与 Germany 相关证据 |
| 3 | 590 | 0.0181357 | 无本案例关键证据 |
| 4 | 29 | 0.0130645 | 无本案例关键证据 |
| 5 | 26 | 0.0125601 | 无本案例关键证据 |

所需的两个事实出现在排名前两位的段落中。第一段包含 Beatrice I 到 Frederick Barbarossa 的桥接，第二段包含 Frederick Barbarossa 与 Germany 的联系。

## 答案与缓存完整性

| 字段 | 值 |
|---|---|
| 标准答案 | `Germany` |
| 预测答案 | `German` |
| 精确包含匹配 | `false` |
| 案例语义结果 | 国籍答案正确 |

已有实验记录说明，五个预先存在的缓存文件在运行前后 SHA-256 完全相同。案例运行未重建或修改缓存。

## 论文对照与边界

本次与论文总体案例结果一致：前两段检索到两个支持事实，并生成正确国籍答案；也复现了 Beatrice I 到 Frederick Barbarossa 的语义桥接。

本次没有精确复现论文展示的最终实体激活 `Germany`。相关 Frederick Barbarossa 句子的传播分数低于阈值，但本地运行仍通过段落排序找回第二个事实。这是单例机制展示，不是普遍准确率结论，也不复现论文的 HippoRAG2 对照。
