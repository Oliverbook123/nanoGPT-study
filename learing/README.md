# nanoGPT 学习导航

> 面向初学者的项目学习文档。nanoGPT 是 Karpathy 写的极简 GPT 实现：`model.py`(330 行) 定义 GPT 模型，`train.py`(336 行) 是训练循环，`sample.py` 是推理/采样。
>
> **两条阅读线，建议交叉看：**
> - **概念线（00–04）**：按"数据 → 架构 → 训练 → 推理"讲**为什么**这么设计。
> - **源码线（05–10）**：按"文件 + 行号"讲**代码到底怎么写的**，每个论断都能回源码核对。
>
> 想快速上手：先读 [00_项目总览.md](./00_项目总览.md) 和 [02_架构.md](./02_架构.md)，再按需翻源码线的对应章节。

## 文档目录（概念线）

| 章节 | 文件 | 内容 |
|------|------|------|
| 0 | [00_项目总览.md](./00_项目总览.md) | 项目结构、运行方式、前置概念 |
| 1 | [01_数据.md](./01_数据.md) | 文本如何变成模型能用的数据(token 化、train.bin、meta.pkl) |
| 2 | [02_架构.md](./02_架构.md) | GPT 模型架构：Embedding / Attention / MLP / Block / 顶层 |
| 3 | [03_训练.md](./03_训练.md) | 训练循环、损失、优化器、学习率调度、DDP |
| 4 | [04_推理.md](./04_推理.md) | 推理/采样原理、temperature、top_k |

## 文档目录（源码线）

| 章节 | 文件 | 对应源码 | 讲什么 |
|------|------|----------|--------|
| 5 | [05_源码解读_model.md](./05_源码解读_model.md) | `model.py`（330 行） | 5 个 `nn.Module` + 1 个 `GPTConfig` 的逐块解读、张量形状表、权重绑定、参数量实测 |
| 6 | [06_源码解读_train.md](./06_源码解读_train.md) | `train.py`（336 行） | 配置区、`get_batch`、三种 `init_from`、训练循环五段拆解 |
| 7 | [07_源码解读_configurator.md](./07_源码解读_configurator.md) | `configurator.py`（47 行） | 配置覆盖机制的原理、边界与实测报错 |
| 8 | [08_源码解读_sample与bench.md](./08_源码解读_sample与bench.md) | `sample.py` / `bench.py` | 采样闭环、编码器选择陷阱、测速脚本的坑 |
| 9 | [09_源码解读_数据准备.md](./09_源码解读_数据准备.md) | `data/*/prepare.py` | 4 个数据集的差异、红楼梦抓取脚本的 6 处修补与实测产物 |
| 10 | [10_源码解读_配置与脚本.md](./10_源码解读_配置与脚本.md) | `config/`、`sync.sh`、`script/` | 各训练配置的差异、同步与远程训练的完整调用链 |

> 另有 `gpt_architecture.html` / `gpt_architecture_mermaid.html` 两份架构图（浏览器直接打开）。

## 核心一行流程

```
原始文本 →(prepare.py)→ 整数 id 序列 train.bin →(train.py)→ 模型权重 ckpt.pt →(sample.py)→ 生成文本
```

本仓库实际跑过的例子（红楼梦字符级）：

```
data/hongloumeng/raw/001.txt … 120.txt（120 个文件）→(prepare.py)→ 880,184 字符 / 词表 4339
  → train.bin 792,165 token + val.bin 88,019 token
  →(train.py config/train_hongloumeng_char.py)→ out-hongloumeng-char/ckpt.pt（12.29M 参数，`get_num_params()` 口径）
  →(sample.py --out_dir=out-hongloumeng-char)→ 生成中文（本机要采样得先有 ckpt.pt：同步规则忽略 `*.pt`）
```

## 实测数据速查

| 项 | 值 | 出处 |
|----|-----|------|
| 红楼梦语料 | 120 回，880,184 字符，词表 4339 | [`09`](./09_源码解读_数据准备.md) |
| 训练 / 验证 token | 792,165 / 88,019（90% / 10%） | [`09`](./09_源码解读_数据准备.md) |
| 红楼梦模型参数 | 12.29M（`get_num_params()` 口径） | [`05`](./05_源码解读_model.md)、`logs/train-20260918-131745.log` |
| 一次真实训练 | 5001 个 iter（编号 0…5000，`max_iters=5000`），train loss 0.73 / val loss 4.69（过拟合） | [`06`](./06_源码解读_train.md) |

## 推荐参考

- Karpathy 官方视频解说：YouTube "Let's build GPT: from scratch"（B站有搬运）
- GPT-2 论文：Language Models are Unsupervised Multitask Learners
- 源码本体：本仓库 `model.py` / `train.py`（所有讲解都以这两个文件的当前版本为准）
