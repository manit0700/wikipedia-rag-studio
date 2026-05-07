#!/usr/bin/env tsx

import { answerWikipediaQuestion, buildWikipediaCorpus, runWikipediaRagDemo } from "../src/wikipedia-rag.js";

type Command = "demo" | "ingest" | "ask";

function parseArgs(argv: string[]): { command: Command; args: string[]; question?: string; pages?: number } {
  const args: string[] = [];
  let question: string | undefined;
  let pages = 5;

  for (let i = 0; i < argv.length; i += 1) {
    const value = argv[i]!;
    if (value === "--question") {
      question = argv[i + 1];
      i += 1;
      continue;
    }

    if (value === "--pages") {
      pages = Number.parseInt(argv[i + 1] || "5", 10);
      i += 1;
      continue;
    }

    args.push(value);
  }

  const command = (args[0] ?? "demo") as Command;
  return { command, args: args.slice(1), question, pages: Number.isFinite(pages) && pages > 0 ? pages : 5 };
}

function printUsage(): void {
  console.log([
    "Wikipedia RAG demo built on QMD.",
    "",
    "Commands:",
    "  demo <topic> --question \"...\"   Build a Wikipedia corpus and answer a question",
    "  ingest <topic> --pages 5         Build the corpus only",
    "  ask \"...\"                       Ask against the last built corpus",
    "",
    "Examples:",
    "  tsx examples/wikipedia-rag.ts demo \"Ada Lovelace\" --question \"Why is she important?\"",
    "  tsx examples/wikipedia-rag.ts ingest \"Machine learning\" --pages 8",
    "  tsx examples/wikipedia-rag.ts ask \"What is a transformer model?\"",
  ].join("\n"));
}

async function main() {
  const { command, args, question, pages } = parseArgs(process.argv.slice(2));

  if (args.length === 0 && command !== "ask") {
    printUsage();
    process.exitCode = 1;
    return;
  }

  const topicOrQuestion = args.join(" ").trim();

  switch (command) {
    case "ingest": {
      const result = await buildWikipediaCorpus(topicOrQuestion, { maxPages: pages });
      console.log(`Indexed ${result.pages.length} Wikipedia pages for "${result.topic}".`);
      console.log(`Corpus: ${result.corpusDir}`);
      console.log(`DB: ${result.dbPath}`);
      break;
    }

    case "ask": {
      const answer = await answerWikipediaQuestion(topicOrQuestion || question || "", { maxResults: 5 });
      console.log(answer.answer);
      break;
    }

    case "demo":
    default: {
      const demoQuestion = question || topicOrQuestion;
      const demoTopic = topicOrQuestion;
      const result = await runWikipediaRagDemo(demoTopic, demoQuestion, { maxPages: pages });
      console.log(result.answer);
      break;
    }
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
