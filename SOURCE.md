# 官方源码来源记录

- 官方仓库：https://github.com/DEEP-PolyU/LinearRAG
- 官方分支：`main`
- 精确提交：`bcc94e66c221f798801255efba09311d6fbcd8d6`
- 提交时间：2026-07-05T00:55:44Z
- 获取日期：2026-09-13（Asia/Shanghai）
- 导入方式：首次通过 Git HTTPS 获取时，两次出现 `Recv failure: Connection was reset`。随后从 GitHub 官方 codeload 地址下载上述精确提交的归档，在本项目内的临时目录解压，将工作树复制到项目根目录，并逐文件完成 SHA-256 对比后删除临时目录和归档。
- 嵌套仓库：无。导入的归档不包含 `.git` 目录。

## 上游原始文件哈希基线

以下哈希记录的是首次导入时，与官方提交逐字节一致的原始文件。当前工作树已经对 README 和源码注释进行中文本土化，因此这些哈希用于追溯上游基线，不表示所有当前文件仍与上游逐字节一致。

```text
3817e5dd849dc7dc9f1131af4deded419129eb2ea192b5a4235322be47d7a202  .gitignore
c3b270c990e2e2c385d9e73f58469d8b662e50e8fe989906ee74ac92a200f403  LICENSE.txt
a966dd60745fc131e12c1ceb935514330e655f9c69889fc5b5b721ad68a6f0ff  readme.md
b1be61256aa900c47c069db2b208f49e659e4bee555c37522c97b2b5e542993d  requirements.txt
206b2d9eef7e1b860ca8733355bc4cdb08b2278a446fb125babd2e66e065c6ce  run.py
90166d5f4d7b1d30d1a80539476138f5f0d2ecb420c3bd7a442997b9b3b9a8ac  scripts/run.sh
bdf03409f6bf3c6bdde1ee5970b46dbf27c4955eb10b5b10c138402c415dddec  src/config.py
8c7349f994cb0a6f159c9cad8cbb596464662050863c30c1f29832267a8dcc06  src/embedding_store.py
4f8adc89b170216f3a2e97a6b1eecc2e053d876cc753177d12298a2cb1d4421a  src/evaluate.py
7cd48aa9805c04e9e32e988adba398a0edcb7e1142633ff21a8e2f290fe9542c  src/LinearRAG.py
8b4b6f1ccd931f7caa793402e0409e42c6bc1f4c47bd4d06d743ff9a4cdfc3f2  src/ner.py
06404d8928465596ff515fe54b0c4b5501b23803ec6c1b15e15540639c133d94  src/utils.py
78c3e5ed98407952fe23fc3dd082c7efb9eebea94147323e6aa9af1bd39c18f1  figure/efficiency_result.png
4e17a81aa5ba00d418c904c2766ab37041e86727aaec0b69cebd2a16abb3ea18  figure/generation_results.png
f6238d72e530336adecd74f3371392a647b81b7796327b6b66e3ba82fd69c005  figure/main_figure.png
```

## 中文本土化修改

当前 `readme.md` 已翻译为中文。翻译只改变说明文字，命令、路径、链接、参数和 BibTeX 均保持原意，不影响程序运行。需要对照上游原文时，可通过上方官方仓库和精确提交查看。

源码、脚本和依赖文件中的说明性注释与 docstring 也已翻译为中文。技术标识、变量名、CLI 参数、运行时日志、模型提示词、算法表达式和可执行语句保持原意。除 docstring 字符串常量外，本次注释本地化不改变 Python 的可执行语法结构或程序行为。

其余本地新增内容包括 `requirements-windows.txt`、`smoke_test.py`、`examples/`、`tools/`、`artifacts/` 和本来源记录。
