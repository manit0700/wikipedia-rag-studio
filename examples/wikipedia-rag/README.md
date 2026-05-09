# Wikipedia RAG Demo

This example uses QMD to index Wikipedia article extracts and answer questions with citations.

## Commands

```sh
npm run wiki-rag:ingest -- "Ada Lovelace" --pages 5
npm run wiki-rag:ask -- "Ada Lovelace analytical engine"
npm run wiki-rag:demo -- "Ada Lovelace" --question "Why is she important?"
npm run wiki-rag:ui
```

## Output

The answer prints a short summary followed by a numbered `Sources` section.
Each citation includes the article title, canonical Wikipedia URL, and a supporting excerpt.

## Browser UI

`npm run wiki-rag:portfolio` starts the polished portfolio app at `http://127.0.0.1:4055`.
Use it to:

- build a corpus from a topic
- ask a grounded question
- compare SFT and DPO model output
- inspect the returned citations and local corpus state
- audit whether the answer used valid retrieved sources

## Ollama synthesis

The portfolio demo uses Ollama by default. Recommended model:

```sh
QMD_WIKI_RAG_OLLAMA_MODEL=wiki-rag-answer-dpo npm run wiki-rag:portfolio
```

The trained GGUF model should already be imported into Ollama with the Modelfile in `finetune/`.
