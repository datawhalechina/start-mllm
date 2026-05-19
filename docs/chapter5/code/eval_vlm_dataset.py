import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path

from dotenv import load_dotenv

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


def _resolve_image_path(raw: str, *, dataset_path: Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = dataset_path.parent / p
    return p


def _ensure_dataset_images_exist(dataset_path: Path) -> None:
    """在发请求前检查 JSONL 引用的图片是否存在，避免跑到一半才 FileNotFound。"""
    base = dataset_path.parent
    with dataset_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            raw = sample.get("image", "")
            if not raw:
                continue
            path = _resolve_image_path(raw, dataset_path=dataset_path)
            if not path.exists():
                raise FileNotFoundError(
                    f"line {lineno}: image not found: {path} "
                    f"(run create_placeholder_images.py under {base}, or update JSONL paths)"
                )


def _resolve_path_arg(raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def parse_args() -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(
        description="对 JSONL 评测集批量调用 OpenAI 兼容多模态接口，写出 eval_results.jsonl。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "跑通后可用同目录 summarize_eval_results.py 汇总结果；"
            "未接 API 时可先用 sample_eval_results_output.jsonl 试汇总脚本。"
        ),
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=env.get("DATASET_PATH", "sample_eval_dataset.jsonl"),
        help="评测 JSONL 路径（默认环境变量 DATASET_PATH 或 sample_eval_dataset.jsonl）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=env.get("OUTPUT_PATH", "eval_results.jsonl"),
        help="输出结果路径（默认环境变量 OUTPUT_PATH 或 eval_results.jsonl）",
    )
    parser.add_argument(
        "--base-url",
        default=env.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
        help="OpenAI 兼容 API 根地址",
    )
    parser.add_argument(
        "--api-key",
        default=env.get("OPENAI_API_KEY", "EMPTY"),
        help="API Key（默认读环境变量 OPENAI_API_KEY）",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=env.get("MODEL_ID", "your-vlm"),
        help="模型名（默认环境变量 MODEL_ID）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(env.get("EVAL_MAX_TOKENS", "256")),
        help="每条请求 max_tokens（默认 256 或环境变量 EVAL_MAX_TOKENS）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = _resolve_path_arg(args.dataset)
    output_path = _resolve_path_arg(args.output)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    _ensure_dataset_images_exist(dataset_path)

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    results = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            start = time.perf_counter()
            result = {
                "id": sample.get("id", ""),
                "question": sample.get("question", ""),
                "reference": sample.get("reference", ""),
                "tags": sample.get("tags", []),
            }
            try:
                data_url = to_data_url(str(_resolve_image_path(sample["image"], dataset_path=dataset_path)))
                response = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": sample["question"]},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                    max_tokens=args.max_tokens,
                )
                result["prediction"] = response.choices[0].message.content
                result["status"] = "ok"
            except Exception as exc:
                result["prediction"] = ""
                result["status"] = "error"
                result["error"] = str(exc)
            result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
            results.append(result)

    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {output_path}")


if __name__ == "__main__":
    main()
