# Wikipedia RAG Fine-Tuning Plan

This document turns the RAG discussion into a concrete training path for the Wikipedia bot built on QMD.

## Goal

Improve grounded answers without trying to "teach" the model Wikipedia from scratch.

The model should learn to:
- rewrite questions into retrieval-friendly searches
- answer only from retrieved evidence
- cite every factual claim
- admit uncertainty when retrieval is weak

## What to Fine-Tune

### 1. Query rewriting

Input:
- one user question

Output:
- retrieval-oriented rewrites
- short keyword query
- natural-language semantic query

This is the highest-value target because better retrieval usually improves answer quality more than bigger generation.

### 2. Citation-first answer formatting

Input:
- question
- retrieved passages

Output:
- concise answer
- numbered citations
- source list with URLs

This is about discipline, not knowledge.

### 3. Weak-evidence behavior

Input:
- question
- poor or conflicting evidence

Output:
- "I don't know"
- or a short explanation that the corpus does not support a strong answer

This prevents confident hallucinations.

### 4. Domain adaptation

Teach the model patterns common in Wikipedia retrieval:
- proper nouns
- alternate titles
- disambiguation behavior
- section-aware references

## Recommended Training Sequence

### Stage A: Query rewrite SFT

Train the model on pairs like:

```json
{
  "query": "Why is Ada Lovelace important?",
  "output": [
    ["lex", "Ada Lovelace importance"],
    ["lex", "\"Ada Lovelace\" biography"],
    ["vec", "why is Ada Lovelace important in computing history"],
    ["vec", "what did Ada Lovelace contribute to early computers"]
  ]
}
```

Focus:
- keep named entities
- prefer short lex phrases
- produce 2-4 useful expansions
- avoid generic filler like "learn about"

### Stage B: Answering with citations

Train on retrieved context records:

```text
Question: Why is Ada Lovelace important?
Context:
[1] Ada Lovelace ... first algorithm ...
[2] Analytical Engine ...

Answer:
Ada Lovelace is important because she is widely credited with writing the first algorithm intended for a machine and with describing the Analytical Engine in a way that anticipated general-purpose computing. [1][2]

Sources:
[1] Ada Lovelace — https://en.wikipedia.org/wiki/Ada_Lovelace
[2] Analytical engine — https://en.wikipedia.org/wiki/Analytical_engine
```

Focus:
- short answer first
- every claim cited
- keep sources visible

### Stage C: Refusal and uncertainty

Train examples where retrieval is insufficient:

```text
Question: What was Ada Lovelace's exact favorite color?
Context: no source mentions this.

Answer:
I could not verify Ada Lovelace's favorite color from the retrieved Wikipedia sources.
```

Focus:
- do not invent facts
- do not fabricate citations

## Data Sources

Good training examples should come from:
- Wikipedia articles indexed through QMD
- query logs from real usage
- manually written edge cases
- hard examples with ambiguous entities

Avoid:
- synthetic factual claims without sources
- generic chat examples with no retrieval context
- examples that reward verbosity over grounding

## Metrics

Measure these separately:

1. **Retrieval quality**
- does the rewritten query pull the right pages?
- does the top result contain the answer?

2. **Citation quality**
- are citations present?
- do citations point to the right pages?
- are claims actually supported?

3. **Uncertainty quality**
- does the model refuse when context is weak?
- does it avoid guessing?

4. **Answer quality**
- is the answer short, direct, and readable?

## Practical Order of Work

1. Improve retrieval and reranking first.
2. Fine-tune query rewriting next.
3. Fine-tune citation-first answer formatting.
4. Add refusal behavior.
5. Only then consider a larger model change.

## What Not to Fine-Tune First

Do not start with:
- base factual knowledge
- end-to-end Wikipedia memorization
- large model scale-up

Those are expensive and usually lower ROI than better retrieval.

## Evaluation Checklist

Before shipping a checkpoint, verify:
- it preserves entity names
- it produces useful retrieval expansions
- it cites retrieved facts
- it refuses unsupported questions
- it improves top-k retrieval or answer faithfulness on a held-out set

## Minimal First Dataset

Start with three small buckets:
- query rewrite examples
- cited answer examples
- refusal examples

That gives you a useful model without overfitting to one narrow style.
