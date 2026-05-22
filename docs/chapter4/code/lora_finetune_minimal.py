"""
最小可跑的多模态 LoRA 微调脚本
================================
目标：在 Colab T4 (16GB) 或本地 24GB 卡上，对 Qwen2.5-VL-3B-Instruct
      做 100 条数据的 caption 微调，建立"数据 → 训练 → 验证"的闭环手感。

依赖安装（与 requirements-api.txt 不冲突，需要 torch）：
    pip install torch transformers accelerate peft datasets pillow

最低版本要求：transformers >= 4.45（Qwen2.5-VL 模型支持）

运行前准备：
    1. 准备自己的图文数据，格式同 sample_multimodal_sft.jsonl
    2. 修改下面 DATA_PATH 指向你的 JSONL
    3. 运行：python lora_finetune_minimal.py
"""

import json
import os
import warnings
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    __version__ as transformers_version,
)

_TRANSFORMERS_MIN = "4.45"
_TRANSFORMERS_VER_TUPLE = tuple(int(x) for x in _TRANSFORMERS_MIN.split("."))
_TRANSFORMERS_CUR_TUPLE = tuple(int(x) for x in transformers_version.split(".")[:2])
if _TRANSFORMERS_CUR_TUPLE < _TRANSFORMERS_VER_TUPLE:
    warnings.warn(
        f"当前 transformers 版本为 {transformers_version}，建议升级到 >= {_TRANSFORMERS_MIN} 以确保 Qwen2.5-VL 兼容。"
        f"运行：pip install --upgrade transformers",
        stacklevel=1,
    )


# ======================== 用户配置区 ========================
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DATA_PATH = Path(__file__).with_name("sample_multimodal_sft.jsonl")
OUTPUT_DIR = Path("./lora_output")
MAX_SAMPLES = 100          # 先用 100 条建立手感
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4             # 等效 batch_size = 1 * 4 = 4
LR = 2e-4
MAX_SEQ_LEN = 512
LORA_R = 16                # LoRA rank：越大表达能力越强，显存越高
LORA_ALPHA = 32            # 常见取值为 rank 或 2 * rank；越大对原模型影响越强
LORA_DROPOUT = 0.05
# Qwen2.5-VL 的微调目标模块（含注意力与 FFN 层），不同模型需查官方文档
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# ==========================================================


def load_jsonl(path: Path) -> list[dict]:
    """加载 JSONL，取前 MAX_SAMPLES 条。"""
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= MAX_SAMPLES:
                break
    return samples


def format_for_qwen25vl(sample: dict) -> dict:
    """
    把仓库标准 JSONL 格式转成 Qwen2.5-VL 的 conversations 格式。
    输入 sample 结构：
        {"image": "path/to/img.jpg", "messages": [{"role": "user", "content": [...]}, ...]}
    输出：
        {"image": PIL.Image, "conversations": [...]}
    """
    img_path = sample.get("image", "")
    if not Path(img_path).is_absolute():
        # 尝试相对 JSONL 文件目录解析
        base = DATA_PATH.parent
        img_path = base / img_path

    if not Path(img_path).exists():
        return None  # 标记为无效样本，由 SimpleMultimodalDataset 过滤
    image = Image.open(img_path).convert("RGB")

    conversations = []
    for msg in sample.get("messages", []):
        role = msg["role"]
        content_parts = msg.get("content", [])
        text_parts = []
        for part in content_parts:
            if part.get("type") == "text":
                text_parts.append(part["text"])
        conversations.append({
            "from": "human" if role == "user" else "gpt",
            "value": "\n".join(text_parts),
        })

    return {"image": image, "conversations": conversations}


def collate_fn(batch, processor, device):
    """
    把一批样本拼成模型输入（含 labels）。
    这里用最简单的方式：单条独立编码，后续可用更高效的 batch 策略。
    """
    texts = []
    images = []
    # 预计算每条样本中 assistant 回复的起始字符位置（用于构造 labels）
    assistant_start_chars = []
    for item in batch:
        conv = item["conversations"]
        text = ""
        asst_start = 0
        for c in conv:
            prefix = "User: " if c["from"] == "human" else "Assistant: "
            if c["from"] == "gpt" and asst_start == 0:
                asst_start = len(text)
            text += prefix + c["value"] + "\n"
        texts.append(text.strip())
        images.append(item["image"])
        assistant_start_chars.append(asst_start)

    # processor 处理图文输入
    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    ).to(device)

    # 构造 labels：复制 input_ids，将非 assistant 回复部分 mask 为 -100
    # 这样 Trainer 只在 assistant 回复上计算 loss，不会学习"复读"用户问题
    # 注意：此 labels 构造为简化示例。由于 processor（含图像占位 token）和
    # tokenizer（纯文本）的分词结果可能存在偏移，生产环境请使用 chat_template
    # + tokenizer 的 return_offsets_mapping 精确计算边界。
    labels = inputs["input_ids"].clone()
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    for i, asst_start in enumerate(assistant_start_chars):
        # 通过 tokenizer 找到 assistant 回复在 token 序列中的起始位置
        prompt_enc = processor.tokenizer(
            texts[i][:asst_start], return_tensors=None, truncation=True, max_length=MAX_SEQ_LEN
        )
        prompt_len = len(prompt_enc["input_ids"])
        # 将 prompt 部分（含 image tokens）mask 为 -100
        labels[i, :prompt_len] = -100
        # 将 padding 部分 mask 为 -100
        if pad_token_id is not None:
            labels[i, inputs["input_ids"][i] == pad_token_id] = -100

    inputs["labels"] = labels
    return inputs


class SimpleMultimodalDataset(torch.utils.data.Dataset):
    """极简 Dataset，返回格式化后的单条样本。"""
    def __init__(self, samples: list[dict]):
        formatted = [format_for_qwen25vl(s) for s in samples]
        # 过滤掉图片缺失的样本
        before_count = len(formatted)
        self.samples = [s for s in formatted if s is not None]
        skipped = before_count - len(self.samples)
        if skipped > 0:
            print(f"[警告] 跳过 {skipped} 条图片缺失的样本（共 {before_count} 条）")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载模型和 processor
    print(f"\nLoading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")

    # 2. 配置 LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 加载数据
    print(f"\nLoading data from: {DATA_PATH}")
    raw_samples = load_jsonl(DATA_PATH)
    print(f"Total samples: {len(raw_samples)}")
    dataset = SimpleMultimodalDataset(raw_samples)

    # 4. 训练参数
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to="none",  # 不连 wandb，简化首次体验
    )

    # 5. 自定义 data collator（简化版）
    def data_collator(batch):
        return collate_fn(batch, processor, device)

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # 7. 开始训练
    print("\n========== 开始训练 ==========")
    trainer.train()

    # 8. 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR / "lora_adapter")
    processor.save_pretrained(OUTPUT_DIR / "lora_adapter")
    print(f"\nLoRA adapter saved to: {OUTPUT_DIR / 'lora_adapter'}")

    # 9. 显存占用总结
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {max_mem:.2f} GB")
        print(f"\n若显存不足，可尝试：")
        print(f"  - 减小 LORA_R（当前 {LORA_R}）→ 8")
        print(f"  - 减小 MAX_SEQ_LEN（当前 {MAX_SEQ_LEN}）→ 256")
        print(f"  - 启用 QLoRA（把模型加载改为 4-bit）")

    print("\n========== 训练完成 ==========")
    print("下一步建议：")
    print("  1. 用 merge_and_unload() 合并 LoRA 权重，或直接加载 adapter 推理")
    print("  2. 在同样的 100 条数据上做推理，对比微调前后的输出差异")
    print("  3. 把评测 JSONL（第五章格式）跑一遍，量化收益")


if __name__ == "__main__":
    main()
