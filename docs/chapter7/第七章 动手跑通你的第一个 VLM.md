# 第七章 动手跑通你的第一个 VLM

> **实战篇**（第 5～10 章）：评测、部署、推理与 Demo、Agent、学习路线收束。

从这一章开始就不只是“看懂”，而是要亲手跑起来。目标很简单：打通一条最小多模态链路，并看清输入输出到底长什么样。

**与前后章的关系**：顺序可按 [前言](../前言.md) 任选——可先读 [第六章](../chapter6/第六章%20推理部署与%20Serving.md) 建立链路再跑本章脚本，也可先跑通本章再回读第六章对照「部署与观测」表述。若显存吃紧或细节全丢，回补 [第二章](../chapter2/第二章%20视觉编码器与跨模态对齐.md) 的分辨率与 patch。跑通后建议用 [第五章](../chapter5/第五章%20评测体系与工程选型.md) 的小评测集固定提问，养成记录失败 `id` 的习惯。

本章提供两种方式：

1. 直接用 `Transformers` 调本地模型
2. 用 OpenAI 兼容接口调用多模态服务

## 一、代码位置

本章配套代码位于：

- `docs/chapter7/code/transformers_chat.py`
- `docs/chapter7/code/openai_compatible_client.py`
- `docs/chapter7/code/requirements.txt`

## 二、方式一：用 Transformers 本地推理

### 1. 安装依赖

```bash
pip install -r docs/chapter7/code/requirements.txt
```

### 2. 准备模型和图片

你需要准备：

- 一个支持图文对话的视觉语言模型
- 一张待测试图片（仓库未默认附带示例图；可先在 `docs/chapter5/code` 运行 `python create_placeholder_images.py`，再用路径 `docs/chapter5/code/images/sample_ui.png`——以下命令默认你在**仓库根目录**执行）

脚本默认使用环境变量：

- `MODEL_ID`：模型名称或本地路径
- `IMAGE_PATH`：图片路径
- `QUESTION`：用户问题

推荐做法：把 `docs/chapter7/code/.env.example` 复制为同目录 `.env` 后直接改值，脚本会自动读取。若你暂时不想建 `.env`，也可以直接在终端里逐行写：

```powershell
$env:MODEL_ID="你的真实模型名"
$env:IMAGE_PATH="你的图片路径"
$env:QUESTION="你想问模型的问题"
```

### 3. 运行示例

```powershell
$env:MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
$env:IMAGE_PATH="docs/chapter5/code/images/sample_ui.png"
$env:QUESTION="请详细描述这张图，并指出其中最重要的视觉元素。"
python docs/chapter7/code/transformers_chat.py
```

Linux / macOS：

```bash
export MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct
export IMAGE_PATH=docs/chapter5/code/images/sample_ui.png
export QUESTION=请详细描述这张图，并指出其中最重要的视觉元素。
python docs/chapter7/code/transformers_chat.py
```

### 4. 这个脚本其实在做什么

脚本的关键步骤其实就四步：

1. 加载模型和 processor
2. 构造图文消息
3. 让 processor 把消息变成模型输入
4. 调 `generate` 输出答案

建议你别只盯“命令成功”，而是重点看这三个点：

- 图像消息在代码里是如何表示的
- chat template 是怎么把图文消息拼接成输入的
- 生成后怎样把 prompt 部分裁掉，只保留模型新增输出

## 三、方式二：调用 OpenAI 兼容接口

很多时候你的模型并不直接由业务代码加载，而是由某个服务提供 OpenAI 兼容接口。这时候应用层只需要做两件事：

1. 把图片转成 URL 或 base64 data URL
2. 用 `chat.completions` 方式提交图文消息

### 1. 运行方式

`openai_compatible_client.py` 与 `transformers_chat.py` 共用 `docs/learner_paths.py`：**不设置 `IMAGE_PATH` 时**，若已生成第五章占位图，会自动使用 `docs/chapter5/code/images/sample_ui.png`；也可显式指定任意路径（支持在仓库根目录或章节目录下运行）。

推荐先把 `docs/chapter7/code/.env.example` 复制为 `.env`，把接口地址、模型名和图片路径填进去，再直接运行脚本。若你想临时在终端里配置，最容易抄的写法是：

```powershell
$env:OPENAI_BASE_URL="你的真实接口"
$env:OPENAI_API_KEY="你的真实密钥"
$env:MODEL_ID="你的真实模型名"
$env:IMAGE_PATH="docs/chapter5/code/images/sample_ui.png"
$env:QUESTION="这张图传递了什么信息？"
python docs/chapter7/code/openai_compatible_client.py
```

Linux / macOS：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
export MODEL_ID=your-vlm
export IMAGE_PATH=docs/chapter5/code/images/sample_ui.png
export QUESTION=这张图传递了什么信息？
python docs/chapter7/code/openai_compatible_client.py
```

### 2. 为什么最好保留这一层

如果你后续要做：

- Web Demo
- API 服务
- Agent 工作流
- 多模型路由

那 OpenAI 兼容接口会比直接在业务层加载模型更灵活。

## 四、实践时最常见的报错

先按现象粗分一层，再进下面的细项（与第二章 Debug 顺序一致：先模型与模板，再协议，再资源）：

| 你看到的 | 优先怀疑 |
| --- | --- |
| 回答空白、极短、或明显复读模板 | `apply_chat_template`、消息结构、`processor` 是否吃到图 |
| HTTP 400/422、或服务端报缺图/格式错 | `image_url` / `data:` URL、字段名、多模态 schema |
| 显存溢出、进程被 kill | 分辨率、`max_new_tokens`、模型体量与量化 |

### 1. 模型能加载，但回答空白或很奇怪

优先检查：

- 是否用了正确的 chat template
- 图像是否真的传给了 processor
- 输入消息里的 `type` 字段是否符合模型要求

### 2. 接口调用成功，但服务端说图像格式不对

优先检查：

- 是不是把本地路径直接当 URL 发了
- data URL 是否包含 MIME 类型前缀
- 服务端是否真的支持多模态消息

### 3. 显存不够

可以从这些方向降低压力：

- 更换小参数模型
- 量化部署
- 降低输入图像分辨率
- 减少 `max_new_tokens`

## 五、建议你顺手做的三个小实验

### 实验 1：同一张图问不同问题

观察模型是否会围绕问题聚焦不同证据。

### 实验 2：换成截图或文档

感受自然图像能力和 OCR/文档能力之间的差异。

### 实验 3：对同一张图问“图中没有的东西”

例如“这张图里有没有红色汽车”，观察模型是否会幻觉。

## 六、跑通后你应该具备什么能力

如果你顺利跑通本章代码，你应该已经能：

- 读懂一个基础 VLM 推理脚本
- 知道图像消息如何进入模型
- 知道 OpenAI 兼容多模态接口怎么写
- 开始自己搭更复杂的 Demo

## 七、进阶实战：把“单次提问”扩成“实验任务”

如果你已经跑通脚本，建议不要停在“问一张图一个问题”。更好的做法是立刻把脚本变成实验工具。

### 可以继续加的三个方向

1. 固定 5 张图片，比较同一 prompt 下的稳定性。
2. 固定 1 张图片，比较不同 prompt 对结果的影响。
3. 固定任务，比较 `Transformers` 本地推理和 OpenAI 兼容接口的回答差异。

### 观察时建议记录

- 模型输出是否基于图像证据
- 是否出现模板化回答
- 对截图、小字、图表是否明显变差
- 同一问题多次请求是否稳定

做到这里，你就从“会跑命令”进到“会做实验”了。

## 八、章末练习

**动机**：本章是全书动手枢纽；练完你能独立改 `messages`、对照模板排查协议错误，后面 Demo、评测、Agent 都站在同一条链路上。

### 必做（约 15 分钟）

1. 使用本章脚本，对同一张图设计 3 个不同层次的问题。
2. 找一张自然图像和一张截图，比较模型回答差异。

### 进阶（约 30 分钟）

1. 修改脚本，加入 `SYSTEM_PROMPT` 或输出文件保存逻辑。
2. 解释 `apply_chat_template` 在多模态推理链路里的作用。

### 挑战（1 小时左右）

1. 写一个小循环：对固定文件夹下 5 张图调用同一脚本，把 `image_path / question / answer` 追加写入一个 JSONL，作为你个人评测集的雏形。

## 十、这一章最关键的收获

本章不是为了让你记住某个模型命令，而是为了把“多模态推理链路”变具体。只要你能读懂这里的代码，后面无论是做 Demo、做 Agent 还是做服务封装，难度都会明显下降。

你在本章摸清的 `messages` 与模板，会直接决定 **第八章** `Gradio` 发出的请求体是否与 **第五章** 评测脚本一致；下面把同一套调用嵌进第六章所说的「应用服务」层。

## 十一、章节跳转

- 上一篇：[第六章 推理部署与 Serving](../chapter6/第六章%20推理部署与%20Serving.md)
- 下一篇：[第八章 构建一个图像问答 Demo](../chapter8/第八章%20构建一个图像问答%20Demo.md)
- 配套代码：[code 目录](./code)（仓库路径 `docs/chapter7/code`）
