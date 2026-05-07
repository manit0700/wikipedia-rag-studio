#!/usr/bin/env python3
"""
Prepare Wikipedia RAG answer fine-tuning data.

This turns structured question/context/answer examples into SFT-ready JSONL
with the same `text` field expected by `train.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from transformers import AutoTokenizer

_tokenizer = None
_tokenizer_model = None


def get_tokenizer():
    global _tokenizer, _tokenizer_model
    model_name = os.environ.get("QMD_BASE_MODEL", "Qwen/Qwen3-1.7B")
    if _tokenizer is None or _tokenizer_model != model_name:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _tokenizer_model = model_name
    return _tokenizer


def load_source_examples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        examples = json.load(f)
    if not isinstance(examples, list):
        raise ValueError(f"{path}: top-level JSON value must be a list")
    return examples


def build_prompt(example: dict) -> str:
    question = str(example.get("question", "")).strip()
    answer = str(example.get("answer", "")).strip()
    sources = example.get("sources", [])
    if not question:
        raise ValueError("question is required")
    if not answer:
        raise ValueError(f"{question!r}: answer is required")
    if not isinstance(sources, list):
        raise ValueError(f"{question!r}: sources must be a list")

    source_lines = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        sid = source.get("id")
        title = str(source.get("title", "")).strip()
        url = str(source.get("url", "")).strip()
        excerpt = str(source.get("excerpt", "")).strip()
        if not title or not url or not excerpt:
            continue
        source_lines.append(f"[{sid}] {title} — {url}\nExcerpt: {excerpt}")

    user_prompt = [
        "/no_think Answer the question using only the provided sources.",
        "Cite every factual claim with bracketed citations like [1].",
        "End every answer with a Sources section listing the cited source titles and URLs.",
        "If the sources do not support the answer, say so clearly.",
        "",
        f"Question: {question}",
    ]
    if source_lines:
        user_prompt.extend(["", "Sources:", *source_lines])
    else:
        user_prompt.extend(["", "Sources:", "None provided."])

    messages = [
        {"role": "user", "content": "\n".join(user_prompt)},
        {"role": "assistant", "content": answer},
    ]
    tokenizer = get_tokenizer()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    text = text.replace("<think>\n\n</think>\n\n", "")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Wikipedia RAG answer data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/wiki_rag_answer_sources.json",
        help="Structured source JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wiki_rag_answer",
        help="Output directory for train/val JSONL",
    )
    parser.add_argument("--split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        input_path = Path(__file__).parent.parent / args.input
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_source_examples(input_path)
    formatted = []
    for example in examples:
        prompt = build_prompt(example)
        formatted.append(
            {
                "question": example["question"],
                "answer": example["answer"],
                "sources": example.get("sources", []),
                "text": prompt,
            }
        )

    random.seed(args.seed)
    random.shuffle(formatted)
    split_idx = int(len(formatted) * (1 - args.split))
    train_data = formatted[:split_idx]
    val_data = formatted[split_idx:]

    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data)]:
        with (output_dir / name).open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dataset_info = {
        "dataset_name": "qmd-wiki-rag-answer",
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "columns": ["text"],
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Prepared {len(formatted)} examples")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
