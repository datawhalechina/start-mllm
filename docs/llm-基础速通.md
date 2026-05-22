# LLM 基础速通

> **前置章节**：[Chapter 0 环境准备](./chapter0-环境准备.md)　|　**下一章**：[第一章 多模态大模型概览](./chapter1/第一章%20多模态大模型概览.md)

如果你只有 Python 基础，对 Transformer、Embedding、自回归生成这些概念还不熟悉，花 30 分钟读完本章就够了。我们只讲**后续章节会直接用到**的概念，不展开数学推导。

> **想更系统地补课？** 本教程属于 [Datawhale](https://github.com/datawhalechina) 开源社区，社区内有更完整的前置课程：[聪明办法学 Python](https://github.com/datawhalechina/learn-python-the-smart-way-v2)、[thorough-pytorch](https://github.com/datawhalechina/thorough-pytorch)、[base-llm：从 NLP 到 LLM](https://github.com/datawhalechina/base-llm)、[self-llm：开源大模型食用指南](https://github.com/datawhalechina/self-llm)。本章已覆盖后续所需的最小知识量，上述课程供按需深入。

## 一、Transformer：LLM 的底层架构

当前几乎所有大语言模型都基于 **Transformer** 架构。你不需要记住它的全部细节，但需要理解它的两个核心思想。

### 1.1 Self-Attention：让每个词关注其他词

传统模型处理文本是从左到右逐个读，像流水线一样。Transformer 的不同之处在于：**每个 token 可以同时“看到”序列中所有其他 token**，然后决定该关注哪些。

这个机制叫 **Self-Attention（自注意力）**。用一句话概括：

> 每个 token 都在问：“序列里哪些 token 和我最相关？”然后把相关 token 的信息加权混合到自己身上。

举个例子，处理“大语言模型很强大”时：

- 当处理“模型”这个 token 时，Self-Attention 会计算它和“大”、“语言”、“很”、“强大”各自的关联度
- “模型”和“语言”关联度高 → 后者的信息权重更大
- 最终“模型”的表示就融合了上下文信息

这就是 Transformer 能理解上下文的核心原因。

### 1.2 Decoder-Only 架构

现代 LLM（GPT、Qwen、Llama 等）采用 **Decoder-Only** 架构：只有解码器，没有编码器。你可以把它理解为：

```
输入文本 → Transformer 层 × N → 预测下一个 token
```

- 输入和输出都是 token 序列
- 模型逐层提取更高层的语义特征
- 最终在词表上输出概率分布，预测“下一个最可能的 token”

“Decoder-Only”这个词你会在论文和模型卡片里反复见到，记住它表示“只有解码器、靠预测下一个 token 来工作”就够了。

### 1.3 为什么叫“大”模型

LLM 的“大”指的是参数量。一个 7B（70 亿参数）的模型，就是有 70 亿个可学习的权重值。这些参数大部分分布在 Transformer 的注意力层和前馈网络里。参数越多，模型的表达能力越强，但推理时需要的算力和显存也越大。

> **和 Python 类比**：你可以把模型想象成一个超大的字典（dict），里面有几十亿个键值对。输入一个 token 序列，模型通过这些参数计算出下一个 token 的概率分布。

## 二、Token 与 Embedding

### 2.1 Token：模型处理文本的基本单位

模型不能直接处理字符串。文本在进入模型之前，需要先被切成一个个 **Token**。

```text
输入文本：  "今天天气真好"
分词结果：  ["今天", "天气", "真", "好"]
Token ID：  [2345,   1876,   932, 428]
```

- 英文通常按子词切分：`"unhappiness"` → `["un", "happiness"]`
- 中文通常按字或词切分：`"大语言模型"` → `["大", "语言", "模型"]`
- 每个 token 对应一个整数 ID，在模型的词表（Vocabulary）中唯一

词表大小通常是几万到十几万（如 Qwen2.5 的词表约 15 万 token）。分词器（Tokenizer）负责文本和 token ID 之间的转换，不同模型有不同的分词器。

### 2.2 Embedding：把 ID 变成向量

Token ID 是整数，不能直接输入神经网络。**Embedding（嵌入）** 就是把每个 token ID 映射成一个高维向量的过程。

```text
Token ID：     2345
               ↓ 查表
Embedding：    [0.12, -0.87, 0.33, ..., 0.56]   （2048 维向量）
```

你可以把 Embedding 层理解为一张查找表（Lookup Table）：词表里有 15 万个 token，每个 token 对应一个 2048 维的向量。输入一个 token ID，查表得到对应的向量。

为什么需要 Embedding？因为向量可以表示语义。在训练好的模型里：

- “猫”和“狗”的向量距离很近（都是动物）
- “猫”和“汽车”的向量距离很远（语义无关）
- “国王” - “男人” + “女人” ≈ “女王”（这是经典的 Word2Vec 向量运算类比，用于帮助你直觉理解 Embedding 的语义性质。现代 LLM 的 Embedding 机制有所不同，但核心思想——相近语义在向量空间中靠近——是一致的）

> **和 Python 类比**：Embedding 层就像一个 `dict`，key 是 token ID（整数），value 是一个固定长度的 list（向量）。查表操作就是 `embedding[token_id]`。

### 2.3 完整流程

```text
文本 → 分词器 → Token ID 序列 → Embedding 查表 → 向量序列 → 送入 Transformer
```

经过 Transformer 逐层处理后，最后一层的输出会再映射回词表大小的向量，得到每个 token 作为“下一个词”的概率。

## 三、自回归生成：LLM 怎么“说话”

LLM 生成文本的方式叫 **自回归（Autoregressive）**，核心就一句话：

> 每一步只预测一个 token，然后把它拼到输入末尾，继续预测下一个。

```text
输入：    "今天天气"
Step 1：  预测 → "真"     输入变为："今天天气真"
Step 2：  预测 → "好"     输入变为："今天天气真好"
Step 3：  预测 → "，"     输入变为："今天天气真好，"
Step 4：  预测 → [EOS]    结束
```

每一步，模型都在做同一件事：看前面所有的 token，预测“下一个最可能的 token”。

### 3.1 生成参数

模型在每一步会输出一个概率分布（词表中每个 token 的得分）。如何从这个分布中选出最终的 token，取决于**生成参数**：

**Temperature（温度）**：控制随机性

```text
Temperature = 0.1：  分布很“尖”，几乎总是选概率最高的 token → 输出确定、重复
Temperature = 1.0：  分布不变，按原始概率采样
Temperature = 1.5：  分布变“平”，低概率 token 也有机会被选中 → 输出多样、随机
```

**Top-p（核采样）**：限制候选范围

```text
Top-p = 0.9：  只在累积概率达到 90% 的最小 token 集合中采样
               排除那 10% 的低概率 token，避免生成离谱内容
```

实际使用中，Temperature 和 Top-p 通常一起调：

| 场景 | Temperature | Top-p | 理由 |
|---|---|---|---|
| 精确提取（OCR、表格） | 0~0.3 | 0.1~0.3 | 需要确定性输出 |
| 日常对话 | 0.7~0.9 | 0.8~0.95 | 平衡准确和自然 |
| 创意写作 | 1.0~1.5 | 0.9~1.0 | 需要多样性和意外性 |

### 3.2 上下文长度

模型能处理的最大 token 数量叫 **上下文长度（Context Length）**，它限制了输入 + 输出的总 token 数。

```text
上下文长度 = 128K token
  ├── 输入：图像 token（约 1000~4000） + 文本 token（用户问题）
  └── 输出：模型回答的 token
  总和不能超过 128K
```

- 上下文越长，能处理的历史对话和参考资料越多
- 但上下文越长，推理速度越慢、显存占用越大
- 多图输入时，每张图可能占用 1000~4000 个 token，上下文消耗很快

> **和 Python 类比**：上下文长度就像一个固定大小的 list，你往里面放 token，放满了就不能再加。实际使用中需要控制输入长度，为输出留够空间。

## 3.5 Chat Template：模型怎么知道”谁在说话”

预训练阶段，模型只做续写——给它一段文字，它接着往后写。但对话场景里，模型需要区分“系统指令”“用户消息”“助手回复”三种角色。

**Chat Template（对话模板）** 就是一套固定格式，告诉模型这些角色边界在哪里。

### 为什么它很重要

不同模型有不同的模板。以 Qwen 为例：

```text
<|im_start|>system
你是一个有用的助手。<|im_end|>
<|im_start|>user
这张图里有什么？<|im_end|>
<|im_start|>assistant
图中是一只橘猫...<|im_end|>
```

而 Llama 的模板长这样：

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一个有用的助手。<|eot_id|>
<|start_header_id|>user<|end_header_id|>
这张图里有什么？<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

模板中那些 `<|im_start|>`、`<|eot_id|>` 等特殊标记，是模型在训练时学到的角色分隔符。如果推理时用了错误的模板，模型就不知道谁在说话，输出会乱掉。

### 关键规则只有一条

> **训练和推理必须用同一套 Chat Template。**

这条规则简单但极其重要。第七章里你会看到，很多“模型回答空白或很奇怪”的问题，根源都是模板不对。

### 实际使用中你不需要手动拼模板

Huggingface 的 `transformers` 库提供了 `apply_chat_template` 方法，会自动按模型的模板格式拼接消息：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "什么是多模态大模型？"},
]

# 自动拼接成模型需要的格式
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(text)
```

你只需要按 `{"role": ..., "content": ...}` 的标准格式构造消息列表，模板转换交给库处理。

> 如果你用 OpenAI 兼容接口（第七、八章的 API 方案），模板转换在服务端自动完成，你同样只需要关心消息列表的结构。

## 四、训练的核心概念

你不需要自己训练模型，但后续章节（第三、四章）会频繁提到这些概念，先建立基本认知会省很多力气。

### 4.1 Loss（损失）

**损失（Loss）** 是一个数字，衡量模型的预测和正确答案之间的差距。

```text
Loss 低 → 模型预测接近正确答案 → 学得好
Loss 高 → 模型预测偏离正确答案 → 学得差
```

在训练日志里你会看到：

```text
epoch 1/3  step 1200  loss 2.31
epoch 1/3  step 2400  loss 1.85
epoch 2/3  step 4800  loss 1.42
```

Loss 逐步下降说明模型在学习。但要注意：

- Loss 的绝对值因任务而异，重要的是**趋势**（是否在下降）
- 如果验证集的 Loss 开始上升，说明模型**过拟合**了——在训练数据上学得太好，泛化能力反而下降

### 4.2 训练过程的基本单位

```text
总训练数据（10000 条）
  └── 拆成多个 Batch（每批 32 条）
        └── 每个 Batch 跑一次前向 + 反向传播 → 更新一次参数（1 Step）
              └── 所有数据跑完一遍 = 1 Epoch
                    └── 通常训练 1~5 个 Epoch
```

| 概念 | 含义 | 例子 |
|---|---|---|
| **Epoch** | 模型完整看一遍所有训练数据 | 训练 3 个 Epoch = 所有数据看 3 遍 |
| **Batch Size** | 每次送入模型的样本数 | Batch Size = 32，每次用 32 条样本计算 |
| **Step** | 每次参数更新为 1 Step | 10000 条 ÷ 32 = 每 Epoch 约 312 Step |
| **Learning Rate（学习率）** | 参数更新的步幅 | 大 → 学得快但可能不稳定；小 → 学得稳但慢 |

训练日志里的 `lr 1e-4` 就是当前的学习率。常见策略是预训练阶段用较大 lr（如 1e-4），微调阶段用较小 lr（如 2e-5）。

### 4.3 预训练与微调

大模型的训练通常分两大阶段：

**预训练（Pre-training）**：从零开始，在海量文本上学习语言能力

```text
训练数据：  互联网文本、书籍、代码...（TB 级）
训练目标：  预测下一个 token
训练成本：  需要大量 GPU 和时间（通常数周到数月）
产出：      基座模型（Base Model，即只经预训练、尚未指令微调的模型，详见[第四节“基座模型”](#44-基座模型)）
```

**指令微调（SFT, Supervised Fine-Tuning）**：在基座模型上，用任务数据进一步训练

```text
训练数据：  "指令-回答"格式的标注数据（千~万条）
训练目标：  让模型学会执行具体任务
训练成本：  远低于预训练（单卡几小时到几天）
产出：      指令模型（Chat Model / Instruct Model）
```

这个范式叫 **“预训练 → 微调”**，是当前 LLM 和 MLLM 的标准流程：

```text
预训练 → 基座模型（会续写文本，但不会听指令）
  ↓ 指令微调
指令模型（会听指令、会对话、会完成任务）
```

### 4.4 基座模型

**基座模型（Base Model）** 就是只经过预训练、还没经过指令微调的模型。

- 它会续写文本，但不会“回答问题”
- 你给它“法国的首都是”，它会续写“法国的首都是巴黎，巴黎是...”，而不是干净地回答“巴黎”
- 常见基座模型：Qwen2.5-7B、Llama3-8B、DeepSeek-V2-Lite

后续章节里提到“冻结基座模型”，就是指保持预训练参数不变，只训练新加的模块（如 Connector、LoRA 适配器）。

## 五、PyTorch 五分钟速查

如果你走 **API 路线**（第七章 OpenAI 兼容接口、第八章 Gradio Demo、第九章 Agent），完全不需要 PyTorch，可以跳过本节。

如果你想走 **本地推理路线**（第七章 Transformers 方式）或动手做 **LoRA 微调**（第四章），下面 5 个模式够你用了：

### 5.1 检查 GPU 是否可用

```python
import torch
print(torch.cuda.is_available())        # True = 有可用 GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # 显卡型号
```

### 5.2 加载预训练模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
```

`from_pretrained` 是你最常用的方法——给一个模型名或本地路径，它自动下载/加载模型权重。

### 5.3 把模型放到 GPU 上

```python
model = model.to("cuda")          # 整个模型搬到 GPU
# 或者加载时直接指定：
# model = AutoModelForCausalLM.from_pretrained("...", device_map="auto")
```

`device_map="auto"` 会自动把模型分配到可用的 GPU 上，大模型推荐用这种方式。

### 5.4 推理模式

```python
with torch.no_grad():              # 推理时关闭梯度计算，省显存
    outputs = model.generate(**inputs, max_new_tokens=512)
```

`torch.no_grad()` 告诉 PyTorch 不需要记录计算图（那是训练用的），推理时务必加上。

### 5.5 Tensor 基础

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])  # 创建张量（类似 numpy array）
print(x.shape)                      # torch.Size([3])
print(x.device)                     # cpu 或 cuda:0
x = x.to("cuda")                    # 搬到 GPU
```

Tensor 就是 PyTorch 版的多维数组。你在后续章节里看到的 `input_ids`、`pixel_values`、`attention_mask` 都是 tensor。

> **一句话总结**：PyTorch 在本教程里主要就做两件事——加载模型（`from_pretrained`）和生成输出（`model.generate`）。更深入的训练用法（Dataset、DataLoader、Trainer）在第四章 LoRA 微调时会结合具体代码讲解。

## 六、章节跳转

- 上一篇：[Chapter 0 环境准备](./chapter0-环境准备.md)
- 下一篇：[第一章 多模态大模型概览](./chapter1/第一章%20多模态大模型概览.md)
