#!/usr/bin/env python3
"""
Evaluate the Wikipedia RAG cited-answer adapter.

This evaluator matches the answer-training prompt. Do not use eval.py for this
adapter; eval.py scores the query-expansion model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


DEFAULT_EXAMPLES = Path(__file__).parent / "data" / "wiki_rag_answer_sources.json"


def build_prompt(example: dict) -> str:
    question = str(example.get("question", "")).strip()
    sources = example.get("sources", [])

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
                source_lines.append(f"[{sid}] {title} - {url}\nExcerpt: {excerpt}")

    prompt = [
        "/no_think Answer the question using only the provided sources.",
        "Cite every factual claim with bracketed citations like [1].",
        "If the sources do not support the answer, say so clearly.",
        "",
        f"Question: {question}",
    ]
    if source_lines:
        prompt.extend(["", "Sources:", *source_lines])
    else:
        prompt.extend(["", "Sources:", "None provided."])
    return "\n".join(prompt)


def load_examples(path: Path, max_examples: int) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        examples = json.load(f)
    if not isinstance(examples, list):
        raise ValueError(f"{path} must contain a JSON list")
    if max_examples > 0:
        return examples[:max_examples]
    return examples


def load_model(model_path: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path_obj = Path(model_path)
    adapter_config = model_path_obj / "adapter_config.json"
    base_model = "Qwen/Qwen3-1.7B"
    if adapter_config.exists():
        with adapter_config.open(encoding="utf-8") as f:
            cfg = json.load(f)
        base_model = cfg.get("base_model_name_or_path", base_model)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = {"": 0} if torch.cuda.is_available() else None

    print(f"Loading base: {base_model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if adapter_config.exists():
        print(f"Loading adapter: {model_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, model_path)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    import torch

    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def score_output(example: dict, output: str) -> tuple[int, list[str]]:
    sources = example.get("sources", [])
    has_sources = isinstance(sources, list) and len(sources) > 0
    issues = []
    score = 100

    if has_sources and not re.search(r"\[\d+\]", output):
        score -= 40
        issues.append("missing bracket citation")
    if has_sources and "Sources:" not in output:
        score -= 20
        issues.append("missing Sources section")
    if not has_sources and not re.search(r"cannot|could not|not verify|sources do not", output, re.I):
        score -= 40
        issues.append("weak-evidence answer did not refuse clearly")
    if len(output.split()) < 8:
        score -= 20
        issues.append("answer is too short")

    return max(score, 0), issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Wikipedia RAG cited-answer adapter")
    parser.add_argument("model", help="Adapter path, merged model path, or HF model id")
    parser.add_argument("--examples", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    examples = load_examples(Path(args.examples), args.max_examples)
    model, tokenizer = load_model(args.model)

    scores = []
    for idx, example in enumerate(examples, 1):
        output = generate(model, tokenizer, build_prompt(example), args.max_new_tokens)
        score, issues = score_output(example, output)
        scores.append(score)
        print(f"\n[{idx}/{len(examples)}] {example['question']}")
        print("-" * 50)
        print(output[:800] + ("..." if len(output) > 800 else ""))
        print(f"Score: {score}%")
        if issues:
            print("Issues: " + ", ".join(issues))

    average = sum(scores) / len(scores) if scores else 0
    print(f"\n{'=' * 50}")
    print(f"Average: {average:.1f}% | Model: {args.model}")
    print(f"{'=' * 50}")
    return 0 if average >= 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
