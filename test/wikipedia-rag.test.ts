import { beforeEach, afterEach, describe, expect, test, vi } from "vitest";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readFileSync, existsSync } from "node:fs";

import {
  answerWikipediaQuestion,
  buildWikipediaCorpus,
} from "../src/wikipedia-rag.js";

let testDir: string;

beforeEach(async () => {
  testDir = await mkdtemp(join(tmpdir(), "qmd-wikipedia-rag-"));
});

afterEach(async () => {
  vi.unstubAllGlobals();
  if (testDir) {
    await rm(testDir, { recursive: true, force: true });
  }
});

function wikipediaSearchResponse() {
  return {
    query: {
      search: [
        { pageid: 1, title: "Ada Lovelace", snippet: "Ada Lovelace was an English mathematician." },
        { pageid: 2, title: "Analytical engine", snippet: "The analytical engine is a proposed mechanical general-purpose computer." },
      ],
    },
  };
}

function wikipediaPageResponse() {
  return {
    query: {
      pages: {
        "1": {
          pageid: 1,
          title: "Ada Lovelace",
          fullurl: "https://en.wikipedia.org/wiki/Ada_Lovelace",
          extract: "Ada Lovelace was an English mathematician and writer, chiefly known for her work on Charles Babbage's proposed mechanical general-purpose computer, the Analytical Engine.",
        },
        "2": {
          pageid: 2,
          title: "Analytical engine",
          fullurl: "https://en.wikipedia.org/wiki/Analytical_engine",
          extract: "The Analytical Engine was a proposed mechanical general-purpose computer designed by Charles Babbage.",
        },
      },
    },
  };
}

function stubWikipediaFetch() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    const payload = url.includes("list=search") ? wikipediaSearchResponse() : wikipediaPageResponse();
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  });

  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("wikipedia rag demo", () => {
  test("builds a corpus and persists a manifest", async () => {
    stubWikipediaFetch();

    const result = await buildWikipediaCorpus("Ada Lovelace", {
      baseDir: testDir,
      maxPages: 2,
    });

    expect(result.pages).toHaveLength(2);
    expect(existsSync(join(result.corpusDir, "00000001-ada-lovelace.md"))).toBe(true);
    expect(existsSync(join(result.corpusDir, "00000002-analytical-engine.md"))).toBe(true);
    expect(existsSync(result.manifestPath)).toBe(true);

    const manifest = JSON.parse(readFileSync(result.manifestPath, "utf-8")) as {
      topic: string;
      documents: Array<{ displayPath: string; title: string; url: string }>;
    };

    expect(manifest.topic).toBe("Ada Lovelace");
    expect(manifest.documents.map((doc) => doc.title)).toContain("Ada Lovelace");
  });

  test("answers with numbered citations and urls", async () => {
    stubWikipediaFetch();

    await buildWikipediaCorpus("Ada Lovelace", {
      baseDir: testDir,
      maxPages: 2,
    });

    const answer = await answerWikipediaQuestion("Ada Lovelace Analytical Engine", {
      baseDir: testDir,
      maxResults: 2,
    });

    expect(answer.citations).toHaveLength(2);
    expect(answer.answer).toContain("[1]");
    expect(answer.answer).toContain("Sources");
    expect(answer.answer).toContain("https://en.wikipedia.org/wiki/Ada_Lovelace");
    expect(answer.answer).toContain("https://en.wikipedia.org/wiki/Analytical_engine");
  });
});
