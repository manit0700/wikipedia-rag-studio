# Wikipedia RAG Studio Demo Script

Use this flow when showing the project in an interview or portfolio review.

## 60-Second Overview

This project turns QMD into a local Wikipedia RAG product. It builds a topic-specific Wikipedia corpus, indexes it locally, retrieves evidence, and answers with numbered citations. I also fine-tuned a Qwen3-1.7B model with SFT and DPO so the local Ollama model follows a citation contract instead of giving generic answers.

## Demo Setup

```sh
cd /Users/manitdankhara/qmd
npm run wiki-rag:portfolio
```

Open:

```text
http://127.0.0.1:4055
```

Recommended demo inputs:

```text
Topic: Nike
Pages: 5
Question: tell me about Nike as a company
```

## What To Show

1. Build the corpus.
   Explain that the app fetches Wikipedia pages, writes local markdown documents, and stores a manifest.

2. Ask the question.
   Show that the answer uses retrieved evidence instead of only model memory.

3. Open the source audit.
   Point out citation IDs, source URLs, excerpts, and fake-URL checks.

4. Compare SFT vs DPO.
   Explain that SFT taught the answer format, while DPO improved preference for grounded answers over bad or generic outputs.

5. Run the benchmark.

```sh
npm run wiki-rag:benchmark -- --limit 10
```

## Talking Points

- The app is local-first: QMD handles retrieval and Ollama serves the tuned GGUF model.
- The model is not trusted blindly; output is checked against retrieved sources.
- DPO was used to reduce failure modes like query-expansion answers, missing citations, and fake sources.
- The repo includes training configs, Kaggle GPU workflow, Modelfiles, benchmark scripts, and screenshots.

## Short Technical Explanation

The user enters a topic and question. The app fetches Wikipedia pages for that topic and creates a local corpus. QMD indexes that corpus and retrieves relevant passages. The retrieved passages are inserted into a strict prompt for the tuned Ollama model. The UI then displays the answer, the cited sources, and audit checks that verify whether the answer follows the citation contract.

## If Asked About Results

Current project evaluation:

```text
SFT model: 92.0%
DPO model: 96.0%
```

The DPO model improved because it was trained to prefer answers that cite valid sources and reject unsupported claims.
