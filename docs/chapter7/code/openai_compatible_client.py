import base64
import mimetypes
import os
import sys
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

_DOCS = Path(__file__).resolve().parents[2]
if str(_DOCS) not in sys.path:
    sys.path.insert(0, str(_DOCS))
from learner_paths import resolve_image_path

load_dotenv()


BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL_ID = os.getenv("MODEL_ID", "your-vlm")
IMAGE_PATH = os.getenv("IMAGE_PATH", "").strip()
QUESTION = os.getenv("QUESTION", "请详细描述这张图，并说明你判断的依据。")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))


def to_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def main() -> None:
    try:
        client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        image_file = str(resolve_image_path(IMAGE_PATH or None, script_file=__file__))
        data_url = to_data_url(image_file)

        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": QUESTION},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=MAX_TOKENS,
        )

        print(response.choices[0].message.content)
    except Exception as exc:
        print(f"Request failed: {exc}")


if __name__ == "__main__":
    main()
