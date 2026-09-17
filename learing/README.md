# nanoGPT 学习导航

> 面向初学者的项目学习文档。nanoGPT 是 Karpathy 写的极简 GPT 实现：`model.py`(约300行) 定义 GPT 模型，`train.py`(约300行) 是训练循环，`sample.py` 是推理/采样。
>
> **学习顺序建议：** 数据 → 架构 → 训练 → 推理，先弄懂"数据长什么样"，再看"模型怎么算"，最后是"怎么练 + 怎么用"。

## 文档目录

| 章节 | 文件 | 内容 |
|------|------|------|
| 0 | [00_项目总览.md](./00_项目总览.md) | 项目结构、运行方式、前置概念 |
| 1 | [01_数据.md](./01_数据.md) | 文本如何变成模型能用的数据(token 化、train.bin、meta.pkl) |
| 2 | [02_架构.md](./02_架构.md) | GPT 模型架构：Embedding / Attention / MLP / Block / 顶层 |
| 3 | [03_训练.md](./03_训练.md) | 训练循环、损失、优化器、学习率调度、DDP |
| 4 | [04_推理.md](./04_推理.md) | 推理/采样原理、temperature、top_k |

## 核心一行流程

```
原始文本 →(prepare.py)→ 整数 id 序列 train.bin →(train.py)→ 模型权重 ckpt.pt →(sample.py)→ 生成文本
```

## 推荐参考

- Karpathy 官方视频解说：YouTube "Let's build GPT: from scratch"（B站有搬运）
- GPT-2 论文(Text Models 由语言模型 "Language Models are Unsupervised Multitask Learners")