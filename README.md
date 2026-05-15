<div align="center">
  <h1>Start-MLLM（⚠️ Alpha内测版）</h1>
  <h3>从零开始学习多模态大模型</h3>
  <p><em>从概念、架构、训练到部署，系统搭建你的第一套 MLLM 知识体系</em></p>
</div>

> [!CAUTION]
> ⚠️ Alpha内测版本警告：此为早期构建版本，内容仍在持续补全与调整中，部分章节、示例或表述可能继续变化，欢迎通过 Issue 反馈问题或建议。

## 项目介绍

`Start-MLLM` 是一个面向中文学习者的多模态大模型教程项目，定位为“能读懂、能跑通、能继续扩展”的开源学习仓库。

这个项目想做的事情很简单：和大家一起学习，把多模态大模型里那些看起来很吓人的概念、架构和工程问题，一步一步拆成能理解、能上手、能继续延伸的内容。

这份教程不想只停留在“知道一些名词”，而是希望带大家走完一条更完整的学习链路：

1. 先理解什么是多模态大模型，以及它与传统 CV、NLP、LLM 的关系。
2. 再理解视觉编码器、跨模态对齐、投影层和生成式架构是怎么工作的。
3. 接着学习数据、训练、评测、部署与应用设计。
4. 最后亲手跑通一个视觉语言模型，并做一个简单的图像问答 Demo。

如果你有 Python 基础，希望系统入门 MLLM，这个项目就是为你准备的。

## 你将收获什么

- 系统理解多模态大模型的核心概念、能力边界与主流技术路线
- 理解视觉编码器、CLIP、Projector、Connector、Instruction Tuning 的工程角色
- 学会阅读常见 VLM/MLLM 架构图，并知道它们为什么这样设计
- 了解数据构建、SFT、LoRA、评测基准、部署选型等关键工程问题
- 能够基于 `Transformers` 或 OpenAI 兼容接口跑通图文问答
- 能够进一步把单模态 Agent 扩展成多模态 Agent
- 能够从教程内容过渡到自己的评测脚本、Demo 和小型开源项目

## 项目受众

- 想系统学习多模态大模型的中文学习者
- 已经了解一点 LLM，想进一步理解 VLM / MLLM 的开发者
- 想把图像、文档、截图能力接入自己项目或 Agent 的工程实践者

你可以从这个项目中获得：

- 一条相对完整的 MLLM 入门路径
- 一组可直接上手的代码脚手架
- 一套从理论到实战的章节导航

基础要求：

- 具备 Python 基础语法
- 能使用命令行安装依赖、运行脚本
- 对 Transformer / LLM 有初步认识会更顺手，但不是硬性要求

## 在线阅读

当前仓库以本地 Markdown 阅读为主，你可以从以下入口开始：

- [Docs 首页](./docs/README.md)
- [前言](./docs/前言.md)
- [Chapter 0 环境准备](./docs/chapter0-环境准备.md)

## 目录

### 理论篇（第 1～4 章）

| 章节名 | 简介 | 状态 |
| ---- | ---- | ---- |
| [前言](./docs/前言.md) | 全书导读、能力边界、章节依赖与学习建议 | ✅ |
| [第一章 多模态大模型概览](./docs/chapter1/第一章%20多模态大模型概览.md) | 建立任务地图，理解多模态能力边界与常见误区 | ✅ |
| [第二章 视觉编码器与跨模态对齐](./docs/chapter2/第二章%20视觉编码器与跨模态对齐.md) | 理解视觉 token、CLIP、对齐训练与输入表示 | ✅ |
| [第三章 多模态生成架构](./docs/chapter3/第三章%20多模态生成架构.md) | 理解 BLIP-2、LLaVA 与主流生成式架构路线 | ✅ |
| [第四章 数据、训练与微调](./docs/chapter4/第四章%20数据、训练与微调.md) | 了解数据格式、训练流程、LoRA/QLoRA 与最小微调闭环 | ✅ |

### 实战篇（第 5～10 章）

| 章节名 | 简介 | 状态 |
| ---- | ---- | ---- |
| [第五章 评测体系与工程选型](./docs/chapter5/第五章%20评测体系与工程选型.md) | 建立多模态评测意识，跑通最小评测脚本 | ✅ |
| [第六章 推理部署与 Serving](./docs/chapter6/第六章%20推理部署与%20Serving.md) | 理解部署链路、资源规划、性能观测与服务化思路 | ✅ |
| [第七章 动手跑通你的第一个 VLM](./docs/chapter7/第七章%20动手跑通你的第一个%20VLM.md) | 使用 `Transformers` 或 OpenAI 兼容接口跑通图文问答 | ✅ |
| [第八章 构建一个图像问答 Demo](./docs/chapter8/第八章%20构建一个图像问答%20Demo.md) | 基于 `Gradio` 构建最小可交互 Demo | ✅ |
| [第九章 从单模态 Agent 到多模态 Agent](./docs/chapter9/第九章%20从单模态%20Agent%20到多模态%20Agent.md) | 把图像感知、工具调用与工作流连接起来 | ✅ |
| [第十章 学习路线与开源项目实战建议](./docs/chapter10/第十章%20学习路线与开源项目实战建议.md) | 收束全书路线，帮助你继续扩展项目 | ✅ |

### Extra-Chapter

| 章节名 | 简介 | 状态 |
| ---- | ---- | ---- |
| [Extra01 OCR 与文档理解专题](./Extra-Chapter/Extra01-OCR与文档理解专题.md) | 面向票据、合同、截图、版面理解等场景 | ✅ |
| [Extra02 多图输入与比较专题](./Extra-Chapter/Extra02-多图输入与比较专题.md) | 面向商品多图、版本比较、跨图归纳等任务 | ✅ |
| [Extra03 长图处理与切块策略专题](./Extra-Chapter/Extra03-长图处理与切块策略专题.md) | 面向长截图、长网页、长文档输入策略 | ✅ |

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :---- | :---- | :---- |
| [ZXJC-niusile](https://github.com/ZXJC-niusile) | 项目负责人 / 当前唯一贡献者 | 和大家一起学习多模态大模型，把复杂知识拆成能读懂、能跑通的教程 |

## 参与贡献

- 如果你发现内容错误、表达不清或代码问题，欢迎提 Issue 反馈。
- 如果你想补充章节、修正示例、完善脚手架，欢迎提 Pull Request。
- 如果你正在学习 MLLM，也欢迎把你的阅读建议、踩坑记录或改进想法分享出来，一起把这个项目打磨得更顺手。

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
