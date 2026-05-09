#!/usr/bin/env node

import { readFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const DEFAULT_DATASET = "finetune/data/wiki_rag_answer_sources.json";
const DEFAULT_MODELS = ["wiki-rag-answer", "wiki-rag-answer-dpo"];

function parseArgs(argv) {
  const args = {
    data: DEFAULT_DATASET,
    limit: 10,
    models: DEFAULT_MODELS,
    ollamaUrl: process.env.OLLAMA_HOST || "http://127.0.0.1:11434",
    maxTokens: Number(process.env.QMD_WIKI_RAG_MAX_TOKENS || 220),
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--data") args.data = argv[++i];
    else if (arg === "--limit") args.limit = Number(argv[++i]);
    else if (arg === "--models") {
      args.models = argv[++i]
        .split(",")
        .map((model) => model.trim())
        .filter(Boolean);
    } else if (arg === "--ollama-url") args.ollamaUrl = argv[++i];
    else if (arg === "--max-tokens") args.maxTokens = Number(argv[++i]);
    else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!Number.isFinite(args.limit) || args.limit < 1) {
    throw new Error("--limit must be a positive number");
  }
  if (!Number.isFinite(args.maxTokens) || args.maxTokens < 32) {
    throw new Error("--max-tokens must be at least 32");
  }
  if (args.models.length === 0) {
    throw new Error("--models must contain at least one model name");
  }
  return args;
}

function printHelp() {
  console.log(`Usage: npm run wiki-rag:benchmark -- [options]

Options:
  --models wiki-rag-answer,wiki-rag-answer-dpo
  --limit 10
  --data finetune/data/wiki_rag_answer_sources.json
  --ollama-url http://127.0.0.1:11434
  --max-tokens 220
`);
}

async function loadExamples(datasetPath, limit) {
  const absolutePath = path.resolve(process.cwd(), datasetPath);
  const raw = await readFile(absolutePath, "utf8");
  const examples = JSON.parse(raw);
  if (!Array.isArray(examples)) {
    throw new Error(`${datasetPath} must contain a JSON array`);
  }
  return examples.slice(0, limit);
}

function buildPrompt(example) {
  const sources = Array.isArray(example.sources) ? example.sources : [];
  const sourceLines = sources
    .map((source) => {
      const id = Number(source.id);
      const title = String(source.title || "").trim();
      const url = String(source.url || "").trim();
      const excerpt = String(source.excerpt || "").trim();
      if (!id || !title || !url || !excerpt) return "";
      return `[${id}] ${title} - ${url}\nExcerpt: ${excerpt}`;
    })
    .filter(Boolean);

  const prompt = [
    "/no_think Answer the question using only the provided sources.",
    "Cite every factual claim with bracketed citations like [1].",
    "End every answer with a Sources section listing the cited source titles and URLs.",
    "If the sources do not support the answer, say so clearly.",
    "",
    `Question: ${String(example.question || "").trim()}`,
    "",
    "Sources:",
    sourceLines.length > 0 ? sourceLines.join("\n") : "None provided.",
  ].join("\n");

  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
}

async function generate({ ollamaUrl, model, prompt, maxTokens }) {
  const response = await fetch(`${ollamaUrl.replace(/\/$/, "")}/api/generate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model,
      prompt,
      stream: false,
      options: {
        temperature: 0,
        num_predict: maxTokens,
      },
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Ollama request failed for ${model}: ${response.status} ${body}`);
  }

  const data = await response.json();
  return cleanOutput(String(data.response || ""));
}

function cleanOutput(output) {
  return output
    .replace(/<think>[\s\S]*?<\/think>/gi, "")
    .replace(/<\|im_(?:start|end)\|>/g, "")
    .trim();
}

function answerBeforeSources(output) {
  const match = output.match(/\n\s*Sources\s*:/i);
  if (!match || match.index === undefined) return output;
  return output.slice(0, match.index);
}

function citedIds(text) {
  return [...text.matchAll(/\[(\d+)\]/g)].map((match) => Number(match[1]));
}

function outputUrls(output) {
  return [...output.matchAll(/https?:\/\/[^\s)\]]+/g)].map((match) =>
    match[0].replace(/[.,;:]+$/, ""),
  );
}

function scoreOutput(example, output) {
  const sources = Array.isArray(example.sources) ? example.sources : [];
  const validIds = new Set(sources.map((source) => Number(source.id)).filter(Boolean));
  const sourceUrls = new Set(sources.map((source) => String(source.url || "").trim()).filter(Boolean));
  const body = answerBeforeSources(output);
  const citations = citedIds(body);
  const invalidIds = citations.filter((id) => !validIds.has(id));
  const urls = outputUrls(output);
  const fakeUrls = urls.filter((url) => !sourceUrls.has(url));
  const hasSourcesSection = /\n\s*Sources\s*:/i.test(output) || /^Sources\s*:/i.test(output);
  const looksLikeQueryExpansion =
    /expanded search query|search query options|refine the query|query variations/i.test(output);

  let score = 100;
  const issues = [];

  if (sources.length > 0 && citations.length === 0) {
    score -= 35;
    issues.push("missing inline citations");
  }
  if (invalidIds.length > 0) {
    score -= 20;
    issues.push(`invalid citation ids: ${[...new Set(invalidIds)].join(", ")}`);
  }
  if (sources.length > 0 && !hasSourcesSection) {
    score -= 20;
    issues.push("missing Sources section");
  }
  if (fakeUrls.length > 0) {
    score -= 25;
    issues.push("contains URL not in retrieved sources");
  }
  if (looksLikeQueryExpansion) {
    score -= 35;
    issues.push("answered as query expansion");
  }
  if (
    sources.length === 0 &&
    !/cannot|could not|not verify|sources do not|provided sources do not/i.test(output)
  ) {
    score -= 40;
    issues.push("did not refuse unsupported answer");
  }
  if (output.split(/\s+/).filter(Boolean).length < 12) {
    score -= 10;
    issues.push("answer is very short");
  }

  return {
    score: Math.max(0, score),
    issues,
  };
}

function formatRow(columns, widths) {
  return columns
    .map((column, index) => String(column).padEnd(widths[index]))
    .join("  ")
    .trimEnd();
}

async function benchmarkModel(args, model, examples) {
  const rows = [];
  for (let index = 0; index < examples.length; index += 1) {
    const example = examples[index];
    const output = await generate({
      ollamaUrl: args.ollamaUrl,
      model,
      prompt: buildPrompt(example),
      maxTokens: args.maxTokens,
    });
    const result = scoreOutput(example, output);
    rows.push({
      question: String(example.question || "").trim(),
      score: result.score,
      issues: result.issues,
    });
    console.log(
      `${model} [${index + 1}/${examples.length}] ${result.score}% - ${rows.at(-1).question}`,
    );
    if (result.issues.length > 0) {
      console.log(`  Issues: ${result.issues.join("; ")}`);
    }
  }
  return rows;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const examples = await loadExamples(args.data, args.limit);

  console.log(`Dataset: ${args.data}`);
  console.log(`Examples: ${examples.length}`);
  console.log(`Ollama: ${args.ollamaUrl}`);
  console.log("");

  const summaries = [];
  for (const model of args.models) {
    console.log(`Benchmarking ${model}`);
    const rows = await benchmarkModel(args, model, examples);
    const average = rows.reduce((sum, row) => sum + row.score, 0) / rows.length;
    const pass = rows.filter((row) => row.score >= 80).length;
    summaries.push({ model, average, pass, total: rows.length });
    console.log("");
  }

  const widths = [28, 9, 10, 9];
  console.log(formatRow(["Model", "Average", "Pass", "Examples"], widths));
  console.log(formatRow(["-".repeat(28), "-".repeat(7), "-".repeat(8), "-".repeat(8)], widths));
  for (const summary of summaries) {
    console.log(
      formatRow(
        [
          summary.model,
          `${summary.average.toFixed(1)}%`,
          `${summary.pass}/${summary.total}`,
          summary.total,
        ],
        widths,
      ),
    );
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
});
