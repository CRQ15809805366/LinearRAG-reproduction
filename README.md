# LinearRAG Reproduction

本项目对论文 **LinearRAG: Linear Graph Retrieval-Augmented Generation** 进行本地复现，并围绕生成准确率、效率、消融、检索质量、超参数、嵌入模型、规模扩展和案例分析开展有界实验。

仓库保留了复现代码、实验设计、实际运行记录和结果边界。现有结果来自本地数据、硬件和模型条件，不等同于完整论文复现。

## 开始使用

- [在线阅读：LinearRAG 复现总结（GitBook）](https://mail-hfut-edu-1.gitbook.io/mail.hfut.edu-docs/)
- [仓库内书稿：结果报告、算法理解与接手读物](docs/book/README.md)
- [安装、资源准备与运行方式](docs/RUNNING.md)
- [当前实验状态与结果索引](docs/EXPERIMENT_STATE.md)
- [Q1–Q8 实验设计](docs/foundation/EXPERIMENT_DESIGN.md)

## 已有说明

- [LinearRAG 基线来源与引入边界](docs/foundation/LINEARRAG_IMPORT.md)
- [HippoRAG 基线来源与引入边界](docs/foundation/HIPPORAG_IMPORT.md)
- [分项实验记录](docs/records/)

## 来源与引用

- 论文：Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang, and Xiao Huang. *LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora*. arXiv:2510.10114, 2025. [arXiv](https://arxiv.org/abs/2510.10114)
- 官方实现：[DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)。本项目在其公开实现基础上开展复现；具体引入版本与本地边界见[基线来源记录](docs/foundation/LINEARRAG_IMPORT.md)。

```bibtex
@article{zhuang2025linearrag,
  title={LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora},
  author={Zhuang, Luyao and Chen, Shengyuan and Xiao, Yilin and Zhou, Huachi and Zhang, Yujing and Chen, Hao and Zhang, Qinggang and Huang, Xiao},
  journal={arXiv preprint arXiv:2510.10114},
  year={2025}
}
```

## 许可

本项目使用 [GNU General Public License v3.0](LICENSE.txt)。
