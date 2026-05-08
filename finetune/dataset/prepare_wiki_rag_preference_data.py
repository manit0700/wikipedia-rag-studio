#!/usr/bin/env python3
"""
Prepare Wikipedia RAG preference data for DPO.

Each example becomes:
  prompt   - question + retrieved sources
  chosen   - grounded answer with valid citations
  rejected - plausible bad answer with common failure modes
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


BAD_TEMPLATES = [
    "Here are some broader search query options for this topic:\n\n"
    "1. {question}\n2. history and background\n3. important facts\n\n"
    "Sources:\nNone provided.",
    "I think the answer is probably based on general knowledge, but the provided sources are not needed. "
    "This topic is important for many reasons and has had a large impact.\n\nSources:\nNone provided.",
    "{wrong_fact} [9]\n\nSources:\n[9] Unknown source — https://example.com",
]


def load_source_examples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        examples = json.load(f)
    if not isinstance(examples, list):
        raise ValueError(f"{path}: top-level JSON value must be a list")
    return examples


def build_prompt(example: dict) -> str:
    question = str(example.get("question", "")).strip()
    sources = example.get("sources", [])
    if not question:
        raise ValueError("question is required")
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
        source_lines.append(f"[{sid}] {title} - {url}\nExcerpt: {excerpt}")

    prompt = [
        "/no_think Answer the question using only the provided sources.",
        "Cite every factual claim with bracketed citations like [1].",
        "End every answer with a Sources section listing only cited sources.",
        "If the sources do not support the answer, say so clearly.",
        "",
        f"Question: {question}",
        "",
        "Sources:",
    ]
    prompt.extend(source_lines or ["None provided."])
    return "\n".join(prompt)


def build_rejected(example: dict, index: int) -> str:
    question = str(example.get("question", "")).strip()
    sources = example.get("sources", [])
    wrong_fact = "The provided sources prove a detailed answer that is not actually supported"
    if sources:
        title = str(sources[0].get("title", "This topic"))
        wrong_fact = f"{title} is mainly important because of facts not shown in the provided evidence"
    return BAD_TEMPLATES[index % len(BAD_TEMPLATES)].format(
        question=question,
        wrong_fact=wrong_fact,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Wikipedia RAG DPO data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/wiki_rag_answer_sources.json",
        help="Structured source JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/wiki_rag_preference",
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

    rows = []
    for index, example in enumerate(load_source_examples(input_path)):
        answer = str(example.get("answer", "")).strip()
        if not answer:
            continue
        rows.append(
            {
                "question": example["question"],
                "prompt": build_prompt(example),
                "chosen": answer,
                "rejected": build_rejected(example, index),
            }
        )

    random.seed(args.seed)
    random.shuffle(rows)
    split_idx = int(len(rows) * (1 - args.split))
    train_data = rows[:split_idx]
    val_data = rows[split_idx:]

    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data)]:
        with (output_dir / name).open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": "qmd-wiki-rag-preference",
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "columns": ["prompt", "chosen", "rejected"],
            },
            f,
            indent=2,
        )

    print(f"Prepared {len(rows)} preference pairs")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
