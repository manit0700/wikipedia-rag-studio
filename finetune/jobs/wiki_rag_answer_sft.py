#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=0.12.0",
#     "peft>=0.7.0",
#     "transformers>=4.45.0",
#     "accelerate>=0.24.0",
#     "huggingface_hub>=0.20.0",
#     "datasets",
#     "bitsandbytes",
#     "torch",
# ]
# ///
"""
SFT training for the Wikipedia RAG cited-answer adapter.

Self-contained script for HuggingFace Jobs:
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 2h jobs/wiki_rag_answer_sft.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from huggingface_hub import login


BASE_MODEL = "Qwen/Qwen3-1.7B"
OUTPUT_MODEL = "tobil/qmd-wiki-rag-answer-1.7B-sft"

_SOURCE_FILE = Path(__file__).resolve().parent.parent / "data" / "wiki_rag_answer_sources.json"


def load_examples() -> list[dict]:
    with _SOURCE_FILE.open("r", encoding="utf-8") as f:
        examples = json.load(f)
    if not isinstance(examples, list):
        raise ValueError(f"{_SOURCE_FILE} must contain a JSON list")
    return examples


def build_text(example: dict, tokenizer) -> str:
    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    sources = example.get("sources", [])
    if not question or not answer:
        raise ValueError(f"Invalid example: {example!r}")

    source_lines = []
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            sid = source.get("id")
            title = str(source.get("title", "")).strip()
            url = str(source.get("url", "")).strip()
            excerpt = str(source.get("excerpt", "")).strip()
            if title and url and excerpt:
                source_lines.append(f"[{sid}] {title} — {url}\nExcerpt: {excerpt}")

    prompt = [
        "/no_think Answer the question using only the provided sources.",
        "Cite every factual claim with bracketed citations like [1].",
        "If the sources do not support the answer, say so clearly.",
        "",
        f"Question: {question}",
    ]
    if source_lines:
        prompt.extend(["", "Sources:", *source_lines])

    messages = [
        {"role": "user", "content": "\n".join(prompt)},
        {"role": "assistant", "content": answer},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return text.replace("<think>\n\n</think>\n\n", "")


def main() -> None:
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    print(f"Loading source examples from {_SOURCE_FILE}...")
    raw_examples = load_examples()
    print(f"Loaded {len(raw_examples)} examples")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted = []
    for example in raw_examples:
        formatted.append(
            {
                "question": example["question"],
                "text": build_text(example, tokenizer),
            }
        )

    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    config = SFTConfig(
        output_dir="qmd-wiki-rag-answer-1.7B-sft",
        push_to_hub=True,
        hub_model_id=OUTPUT_MODEL,
        hub_strategy="every_save",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1.5e-4,
        max_length=1024,
        logging_steps=10,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        report_to="none",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    print("Initializing SFT trainer...")
    trainer = SFTTrainer(
        model=BASE_MODEL,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    print("Pushing to Hub...")
    trainer.push_to_hub()
    print(f"Done! Model: https://huggingface.co/{OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
