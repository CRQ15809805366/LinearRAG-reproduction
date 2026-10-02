# 实验索引与 Agent 指南

状态审查日期：**2026-10-02**。本文是 Q1–Q8 的正式导航索引。每个实验范围链接到最终结果记录；影响解释的 Q1/Q2 中断例外已合并到补充结果中。

## 状态定义

- **已完成**：所述有界范围要求的全部输出均存在。这不意味着完整论文复现。
- **部分完成**：已有有用的实际证据，但缺少必要测量或目标规模。

状态适用于明确的实验范围，而不是整个论文问题。即使存在较新的补充实验，不同语料和协议仍作为独立证据。

## 缓存现状

共享缓存已迁移至 `data/cache/`，目录职责与复用规则见 [数据目录说明](../data/README.md)。历史实验记录保留原路径；旧缓存缺少完整环境信息时，manifest 标明按当前环境登记的复用假设，不代表已验证历史版本等价。

## 当前实验登记

| 实验范围 | 状态 | 有效结果记录 | 主要原始证据 | 边界 |
| ------------------------------------------------------------------ | ----------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Q1：四个数据集，各 100 题，Vanilla / LinearRAG | **已完成** | [有界结果](records/q1_generation_accuracy/RESULTS_2026-09-17_10PCT.md) | [摘要](../data/output/q1_generation_accuracy/q1-formal-10pct-20260917-r2/summary.json) | 本地 `qwen3.8-flash`；不是论文规模或全部基线比较。 |
| Q1：HotpotQA，20 题 / 250 段，三种方法 | **已完成** | [最终 r2 结果](records/q1_generation_accuracy/RESULTS_2026-10-01_SMALL_CORPUS_R2.md) | [摘要](../data/output/q1_generation_accuracy/q1-hotpotqa-context20-noise50-20261001-r2/summary.json) | 所有方法使用修订后语料；复用 231 条 HippoRAG 抽取。原始与修订语料结果不混合。 |
| Q2：既有 2Wiki 检索计时与三次冷索引 | **已完成** | [100 题计时与冷索引观测](records/q2_efficiency_analysis/RUN_2026-09-20_100Q.md) | [测量](../data/output/q2_efficiency_analysis/q2-formal-100q-20260920/measurements.json) | 仅 LinearRAG。正式检索运行恢复缓存；记录保留独立冷索引证据。 |
| Q2：共享 HotpotQA 的 LinearRAG / HippoRAG 效率 | **部分完成** | [恢复后的最终状态](records/q2_efficiency_analysis/RESULTS_2026-10-01_SHARED_CORPUS.md) | [摘要](../data/output/q2_efficiency_analysis/q2-hotpotqa-context20-noise50-run-20261001/summary.json) | 两种方法均完成首次查询和五次热检索批次。完整 HippoRAG 冷索引耗时不可得；检查点恢复不是冷索引。运行状态：`passed_with_recovery`。 |
| Q3：四个数据集，各 100 题，完整系统 / 两种消融 | **已完成** | [有界消融](records/q3_ablation_study/RESULTS_2026-09-20_10PCT.md) | [摘要](../data/output/q3_ablation_study/q3-formal-10pct-20260920/summary.json) | 操作化消融；效果随数据集变化。 |
| Q4：Medical，200 道均衡问题，Vanilla / LinearRAG | **已完成** | [200 题结果](records/q4_retrieval_quality/RUN_2026-09-20_200Q.md) | [摘要](../data/output/q4_retrieval_quality/q4-formal-200q-20260920/summary.json) | 完整本地 225 段语料；本地 LLM 评价；不是表 4 的复现。 |
| Q4：Medical，12 题 / 30 段，三种方法 | **已完成** | [小语料结果与补判](records/q4_retrieval_quality/RUN_2026-10-01_SMALL_CORPUS.md) | [并列摘要](../data/output/q4_retrieval_quality/q4-medical-small-ready-20261001/summary.csv) | 已补判缺失证据分类；候选筛选语料未完成标准支持证据认证。与 200 题结果分开。 |
| Q5：2Wiki，100 题 delta/lambda OFAT 扫描 | **已完成** | [扫描结果](records/q5_hyperparameter_sensitivity/RESULTS_2026-09-20_100Q.md) | [摘要](../data/output/q5_hyperparameter_sensitivity/q5-formal-100q-20260920/summary.json) | 一个热索引，每个参数点运行一次；未估计交互或通过重复确立最优值。 |
| Q6：2Wiki，100 题，四个嵌入模型 | **已完成** | [四模型结果](records/q6_embedding_model_robustness/RESULTS_2026-09-21_100Q.md) | [摘要](../data/output/q6_embedding_model_robustness/q6-formal-100q-20260921/summary.json) | 有界对照；不作普遍最佳模型声明。 |
| Q7：LinearRAG HotpotQA 规模代理 | **部分完成** | [250K–500K 规模对照](records/q7_large_scale_efficiency/RESULTS_2026-09-21_SCALE_SERIES.md) | [250K](../data/output/q7_large_scale_efficiency/q7-formal-250k-20260921/measurements.json), [500K](../data/output/q7_large_scale_efficiency/q7-formal-500k-r2-20260921/measurements.json) | 两个冷索引各自**已完成**。每个规模仅单次观测。1M 在 NER 阶段失败，未保留可用测量。没有 ATLAS-Wiki 5M/10M 或实际 HippoRAG 对照。 |
| Q8：已有 2Wiki 缓存上的原论文案例 | **已完成** | [原始案例](records/q8_case_study/RESULTS_2026-09-21_PAPER_CASE.md) | [结果](../data/output/q8_case_study/q8-paper-case-20260921/result.json) | 仅 LinearRAG 的机制案例；未精确复现最终 Germany 实体激活。 |
| Q8：共享 20 段的 LinearRAG / 原始 HippoRAG 案例 | **已完成** | [补充案例](records/q8_case_study/RESULTS_2026-10-01_20PASSAGES.md) | [结果](../data/output/q8_case_study/q8-20passages-case-20261001/result.json) | 两种答案均语义正确；未证明 HippoRAG 初始链接正确。HippoRAG 2024 不等于论文的 HippoRAG2。 |
