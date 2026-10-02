# 共享小语料构造

此目录只负责离线生成实验输入，不运行检索、生成或评价。当前支持 HotpotQA；构造过程不使用任何方法成绩。

## 数据与构造边界

本地 `questions.json` 的 `evidence` 是每题约十篇候选上下文，含干扰文档，没有 `supporting_facts`。原 `chunks.json` 是合并、标准化后的长片段，不能简单按标题定位金标准支持证据。

工具保留抽中问题的全部候选文档，采用原标题加换行及原句拼接，按完整文本去重，再从全体候选文档中抽取额外随机文档，最后打乱共同语料顺序。随机额外文档不保证无关；候选上下文不是金标准证据。

这改变了原实验的文档边界、大小写和检索空间。所有方法必须在新语料上运行，不能拼接旧 Q1 分数。Q4 可复用文档映射；金标准 Evidence Recall 仍需真实支持标注。

## 使用

从项目根目录执行，不加载模型、不调用 API：

```powershell
.venv\Scripts\python.exe -m experiments.corpus_construction.build_corpus --audit-only
.venv\Scripts\python.exe -m experiments.corpus_construction.build_corpus
```

默认 20 题、50 篇额外干扰、种子 `20261001`。题目随机抽取，不挑旧实验答对的题。改变规模时显式给出参数与新 ID：

```powershell
.venv\Scripts\python.exe -m experiments.corpus_construction.build_corpus --questions 30 --distractors 50 --seed 20261001 --corpus-id hotpotqa-context30-noise50-s20261001
```

输出到 `src.paths.DERIVED_CORPORA_DIR / <corpus-id>`，即 `data/input/derived_corpora/<corpus-id>/`，已被 Git 忽略。已有目录拒绝覆盖。

- `questions.json`：选中问题的原始字段。
- `chunks.json`：共享文档字符串列表，不预加序号。
- `manifest.json`：来源和构造代码哈希、输出哈希、种子、问题 ID、每题候选文档下标、文档原始位置与角色、规模统计。

下游按 chunks 顺序添加 `index:text` 序号；映射下标对应 index。映射只用于审计，不传给检索器。生成物不提交 Git。Q1、Q4 及内部参数/消融实验可读取同一输入包，各自需要声明实验边界。
