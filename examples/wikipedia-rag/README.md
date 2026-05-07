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

`npm run wiki-rag:ui` starts a local app at `http://127.0.0.1:4011`.
Use it to:

- build a corpus from a topic
- ask a grounded question
- run the full build-and-answer demo in one click
- inspect the returned citations and local corpus state

## Optional synthesis

If you want the bot to draft a richer paragraph from the retrieved evidence, set:

```sh
export QMD_WIKI_RAG_USE_LLM=1
```

QMD will then use its local generation model if one is available.
