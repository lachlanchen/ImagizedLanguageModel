# 从 MiniMind 的一条训练样本到视觉语言模型

研究对象是 MiniMind-3 的稠密模型，源码固定在 `858008eb`，对照权重是作者发布的原生 `full_sft_768.pth`。我们运行了它的数据加载器、前向损失和自由生成，并把同一批中英文阅读问题交给了视觉模型。项目分别保存了原始出处、本机测量以及两个模型写出的答案。[源码版本](https://github.com/jingyaogong/minimind/tree/858008eb98da4428f78053a1cc4d3a3bd5973629) · [权重版本](https://huggingface.co/jingyaogong/minimind-3-pytorch/tree/edba70ec15e06bc4280fbb96ac3383d73a7eab91)

## 两小时能训练出什么？

作者给出两种口径。简介的脚注只计算 SFT；详细成本表估算一张 3090 上的稠密 Zero 配方，预训练约 1.21 小时，指令微调约 1.10 小时。两阶段各跑一遍约 2.31 小时，按每小时 1.3 元约合 3 元。两个训练脚本的默认值都是两遍。复现这张表应显式设置 `--epochs 1`，使用约 1.2 GB 的迷你预训练集及 1.6 GB 的迷你指令集。数据准备、合成语料的教师成本和选做的强化学习不在这笔租卡费用内。本研究直接使用公开微调权重作对照。其完整训练成本不能从 Zero 的预算倒推。[简介脚注](https://github.com/jingyaogong/minimind/blob/858008eb98da4428f78053a1cc4d3a3bd5973629/README.md#L32-L42) · [成本表](https://github.com/jingyaogong/minimind/blob/858008eb98da4428f78053a1cc4d3a3bd5973629/README.md#L647-L678) · [数据目录](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/312afb4f76391145c6902f765bb51691c09a12f5)

## 一条样本怎样变成一次更新？

1. **预训练。** 加载器读取 JSONL 的 `text`，编码后加起止符，补齐至长度上限，把补位标签设为 −100。模型内部将预测和目标错开一位。一个批次中，每个有效位置都练习预测下一个符号；其后的文字对这个位置不可见。默认训练窗口是 340。
2. **指令微调。** 每条记录包含对话角色。模板决定用户、助手和停止符的边界。训练仍使用同一因果模型，只给助手回复及约定的结尾计损失。默认窗口是 768。随机种子固定为 42 后，我们将官方迷你集开头的真实对话送入原生加载器，再用发布的权重计算损失。这样的检查能发现标签错位、误学用户提问、停止符漏训等问题。实测预训练样本有 339 个有效预测，平均损失 3.0655 nats；指令样本有 549 个非补位位置，414 个有效预测，平均损失 1.0959 nats。两者均使用公开微调权重。
3. **生成。** 推理时只有用户的消息和助手开头。解码器根据刚写出的内容继续生成，缓存历史键和值，遇结束符或上限退出。本研究关闭采样，记录停止原因和完整回复。对有两个问题的同一页，只有两道都答对才记该页成功。这样能抓住重复页面中同一句话的模型。[训练数据加载](https://github.com/jingyaogong/minimind/blob/858008eb98da4428f78053a1cc4d3a3bd5973629/dataset/lm_dataset.py) · [模型与生成](https://github.com/jingyaogong/minimind/blob/858008eb98da4428f78053a1cc4d3a3bd5973629/model/model_minimind.py) · [本机样本复核](evidence/visual_story_writer_v1/minimind_trace.json)

实测共享参数去重后共 63,912,192 个。八层、宽 768、八个查询头共用四组键值，前馈层宽 2,432。输入和输出共用 6,400 项嵌入。模型对“言”使用一个词表项，对“𠮷”使用多个字节片段。后者仍能表示这个编码字符；真正要保留“言”的两幅不同甲骨拓片，还需要可区分两幅笔画图的输出空间。配置中的 32,768 个位置也不能推出模型训练过整本书。[模型配置与实现](https://github.com/jingyaogong/minimind/blob/858008eb98da4428f78053a1cc4d3a3bd5973629/model/model_minimind.py) · [预训练入口](https://github.com/jingyaogong/minimind/blob/858008eb98da4428f78053a1cc4d3a3bd5973629/trainer/train_pretrain.py) · [指令微调入口](https://github.com/jingyaogong/minimind/blob/858008eb98da4428f78053a1cc4d3a3bd5973629/trainer/train_full_sft.py)

## 对 ILM 的具体启发

视觉模型也需要明确的监督位置、能够随历史变化的预测，以及独立的停止事件。我们的训练目标把每个可写目标字形的两条概率路径相加：从已学习的位图集合写出它，或从当前页面复制完全相同的像素。输出的历史经循环层和带缓存的因果注意力进入下一步。页面中出现两个相同字形时，其复制概率应相加。把“写得出这个字形”和“在这个问题下应该写它”分开检验，可以直接解释一种常见失败。[训练与损失](../ilm/visual_lm/visual_story_writer.py) · [经 Lean 检查的可达性命题](../formal/span_list/VisibleWriterSupport.lean)

如果目标既不在可写集合，也不在被选中的证据条，增加训练时间不能让这条复制路径产生目标。加入整页后，训练中的 5,230 个古字形目标从全部不可达到全部可达。完整模型却只答对了一半跨语言、跨版式的字形指令。词表外输出的物理能力和根据指令选择输出之间，还有一个需要学习的关系。[实验协议](visual_story_writer_protocol.md) · [交叉测试](evidence/visual_story_writer_v1/forms.json)

公开模型的 C-Eval 和 CMMLU 报告值约为 25%，PIQA 约为 51%。这些评估通常比较给定选项的条件概率。这里的独立阅读比较删除选项，逐字生成答复，还单独评估意义相同的改写。两者回答的是不同的实验问题。衡量“快十倍”需要先选定相同的任务、质量和停止标准，再测端到端的显存、时间与能耗；64M 个参数或一次快训练都不足以得出这个倍数。[上游评测说明](https://github.com/jingyaogong/minimind/blob/858008eb98da4428f78053a1cc4d3a3bd5973629/README.md#L1602-L1655) · [本机自由阅读实验](visual_story_writer_result.md)
