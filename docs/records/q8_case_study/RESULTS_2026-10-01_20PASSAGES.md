# Q8 20 段语料补充案例

在共享的 20 段语料上完成论文问题 `3a3c2efe0bdc11eba7f7acde48001122`：两个支持段落，加上种子 `20261001` 抽取的 18 个干扰段落。保留来源段落 ID/文本和来源索引。两种方法均使用 `all-mpnet-base-v2`，由 `qwen3.8-flash` 生成答案；HippoRAG 抽取也使用 `qwen3.8-flash`。

原始证据：`data/output/q8_case_study/q8-20passages-case-20261001/result.json`。最终调用复用了已保存的 LinearRAG 结果和 HippoRAG 抽取检查点，不属于冷索引测量。

- LinearRAG 默认 BFS：`German`；婚姻和 Germany 支持段落分别排名 1、2。
- 原始 HippoRAG 2024 v1.0.0：`German`；Germany 和婚姻支持段落分别排名 1、2。
- HippoRAG 图：1,939 个节点、6,884 条边；`case_trace` 保存了涉及案例/链接关注节点的 112 条边。
- HippoRAG 将查询实体 `beatrice i  countess of burgundy` 链接到 `countess joan ii of burgundy`（相似度 0.8324581981），尽管图中存在 `beatrice i`、`frederick barbarossa` 和 `germany` 节点。因此，检索和答案正确不能证明初始实体链接正确。

范围是刻意保留支持段落的小语料单例。两种答案相对标准答案 `Germany` 均语义正确。这不是论文的 HippoRAG2 对照，也不是总体准确率证据；没有复现论文中的基线失败。未改变算法或抽取提示词。
