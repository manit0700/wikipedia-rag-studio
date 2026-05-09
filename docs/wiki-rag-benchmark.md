# Wikipedia RAG Benchmark

This benchmark tracks whether the local Wikipedia RAG model follows the product contract:

- answer only from retrieved Wikipedia source snippets,
- cite factual claims with bracketed source IDs,
- include a final `Sources:` section,
- avoid fake source URLs,
- avoid reverting to query-expansion behavior,
- refuse when the provided sources do not support the answer.

## Current Results

These results come from the project fine-tuning workflow and the runtime smoke benchmark used by the portfolio demo.

| Model | Training stage | Score | Notes |
| --- | --- | ---: | --- |
| `wiki-rag-answer` | SFT | 92.0% | Learned the cited-answer format and generally follows the source contract. |
| `wiki-rag-answer-dpo` | SFT + DPO | 96.0% | Improved preference for grounded answers over generic or fake-citation answers. |

## Reproduce Locally

Start Ollama and make sure the models are installed:

```sh
ollama list
ollama create wiki-rag-answer -f finetune/Modelfile.wiki-rag-answer
ollama create wiki-rag-answer-dpo -f finetune/Modelfile.wiki-rag-answer-dpo
```

Run the portfolio benchmark:

```sh
npm run wiki-rag:benchmark -- --limit 10
```

Useful variants:

```sh
# Compare only the DPO model
npm run wiki-rag:benchmark -- --models wiki-rag-answer-dpo --limit 10

# Run the full curated evaluation set
npm run wiki-rag:benchmark -- --limit 50

# Use a different Ollama endpoint
npm run wiki-rag:benchmark -- --ollama-url http://127.0.0.1:11434
```

The benchmark reads `finetune/data/wiki_rag_answer_sources.json`, builds the same source-grounded prompt used by the app, calls Ollama, and reports an average score plus per-question failures.

## Training Evaluation

The adapter-level evaluator is still available for GPU environments:

```sh
cd finetune
uv run python eval_wiki_rag_answer.py outputs/wiki-rag-answer --max-examples 5
```

Use this when evaluating a PEFT adapter before exporting to GGUF. Use `npm run wiki-rag:benchmark` after the GGUF model has been imported into Ollama.
