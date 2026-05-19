"""
多模态 Agent 骨架 v2：基于 OpenAI Tool-Calling 的 JSON Schema 路由
===============================================================
与 agent_workflow_demo.py 的区别：
  - v1（demo 版）用关键字匹配（if "报错" in summary）路由工具，教学和原型可用
  - v2（本文件）用 OpenAI 标准的 tool-calling JSON schema，更适合生产环境

依赖：
    pip install openai python-dotenv

运行：
    $env:OPENAI_BASE_URL="你的真实接口"
    $env:OPENAI_API_KEY="你的真实密钥"
    $env:MODEL_ID="你的真实模型名"
    $env:IMAGE_PATH="docs/chapter5/code/images/sample_ui.png"
    $env:QUESTION="请分析这张截图并给出下一步建议。"
    python agent_workflow_v2.py
"""

import base64
import json
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_DOCS = Path(__file__).resolve().parents[2]
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))
from learner_paths import resolve_image_path


load_dotenv()


def to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


# ========== Tool Schema 定义 ==========
# 这是 OpenAI 兼容接口的标准函数调用格式，比关键字匹配更稳定、更可扩展
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "knowledge_lookup",
            "description": "当截图或描述中包含报错、异常、错误码时，调用此工具查询知识库或 FAQ",
            "parameters": {
                "type": "object",
                "properties": {
                    "error_keywords": {
                        "type": "string",
                        "description": "提取到的关键错误信息或错误码",
                    },
                },
                "required": ["error_keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "structured_extract",
            "description": "当需要提取结构化字段（如金额、日期、表格内容）时调用此工具",
            "parameters": {
                "type": "object",
                "properties": {
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "需要提取的字段列表，如 ['金额', '日期', '商户名']",
                    },
                },
                "required": ["fields"],
            },
        },
    },
]


class MultimodalAgentV2:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
            api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        )
        self.model_id = os.getenv("MODEL_ID", "your-vlm")

    def perceive_and_plan(self, image_path: str, question: str) -> dict:
        """
        第一步：感知 + 规划。
        把图片和问题发给模型，让模型自己决定是否需要调用工具。
        如果模型决定调用工具，会返回 tool_calls；否则直接返回文本回答。
        """
        data_url = to_data_url(image_path)

        # 系统提示：告诉模型它的角色和可用工具
        system_prompt = (
            "你是一个多模态助手。请先分析用户提供的图片，然后根据用户问题决定下一步：\n"
            "1. 如果图片包含报错、异常、错误码，请调用 knowledge_lookup 工具\n"
            "2. 如果需要提取金额、日期、表格等结构化信息，请调用 structured_extract 工具\n"
            "3. 否则直接回答用户问题"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 让模型自己决定是否调用工具
            max_tokens=512,
        )

        message = response.choices[0].message
        result = {
            "content": message.content,
            "tool_calls": [],
        }

        # 检查模型是否请求了工具调用
        if message.tool_calls:
            for tc in message.tool_calls:
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })

        return result

    def execute_tools(self, tool_calls: list[dict]) -> list[dict]:
        """
        第二步：执行工具。
        根据模型决定的 tool_calls，执行对应的本地函数。
        """
        outputs = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]
            tool_id = tc["id"]

            if name == "knowledge_lookup":
                output = self.knowledge_lookup(args.get("error_keywords", ""))
            elif name == "structured_extract":
                output = self.structured_extract(args.get("fields", []))
            else:
                output = f"未知工具: {name}"

            outputs.append({
                "tool_call_id": tool_id,
                "name": name,
                "output": output,
            })

        return outputs

    def knowledge_lookup(self, error_keywords: str) -> str:
        """占位实现：实际项目中应接入知识库或向量检索。"""
        return (
            f"[knowledge_lookup 结果] 已检索到与 '{error_keywords}' 相关的文档。\n"
            "建议：1) 检查配置文件路径是否正确；2) 确认依赖版本匹配；3) 查看官方 FAQ。"
        )

    def structured_extract(self, fields: list[str]) -> str:
        """占位实现：实际项目中应接入 OCR 或规则抽取。"""
        return (
            f"[structured_extract 结果] 已提取字段: {', '.join(fields)}。\n"
            "占位值示例：金额=1280.00, 日期=2024-01-15, 商户名=示例商店"
        )

    def summarize(self, original_question: str, perception: dict, tool_outputs: list[dict]) -> str:
        """
        第三步：汇总。
        把感知结果和工具执行结果发给模型，生成最终回答。
        """
        # 构建包含工具结果的消息链
        messages = [
            {"role": "system", "content": "你是一个多模态助手。请根据图片分析结果和工具返回的信息，给出简洁、准确的最终回答。"},
            {"role": "user", "content": f"用户问题：{original_question}"},
        ]

        # 添加感知结果
        if perception["content"]:
            messages.append({"role": "assistant", "content": f"图片分析：{perception['content']}"})

        # 添加工具结果（用 tool 角色）
        for out in tool_outputs:
            messages.append({
                "role": "tool",
                "content": out["output"],
                "tool_call_id": out["tool_call_id"],
            })

        # 请求最终总结
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=512,
        )

        return response.choices[0].message.content or ""

    def run(self, image_path: str, question: str) -> dict[str, object]:
        if not question.strip():
            return {"status": "error", "error": "Question is empty."}

        # Step 1: 感知 + 规划
        perception = self.perceive_and_plan(image_path, question)

        # Step 2: 执行工具（如果有）
        tool_outputs = []
        if perception["tool_calls"]:
            tool_outputs = self.execute_tools(perception["tool_calls"])

        # Step 3: 汇总生成最终回答
        final_answer = self.summarize(question, perception, tool_outputs)

        return {
            "status": "ok",
            "question": question,
            "perception": perception["content"],
            "tool_calls": perception["tool_calls"],
            "tool_outputs": tool_outputs,
            "final_answer": final_answer,
        }


def main() -> None:
    raw_path = os.getenv("IMAGE_PATH", "").strip()
    image_path = str(resolve_image_path(raw_path or None, script_file=__file__))
    question = os.getenv("QUESTION", "请分析这张图片，并给出下一步建议。")

    try:
        agent = MultimodalAgentV2()
        result = agent.run(image_path, question)
    except Exception as exc:
        result = {
            "status": "error",
            "error": str(exc),
            "question": question,
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
