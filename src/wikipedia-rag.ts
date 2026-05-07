import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  createStore,
  extractSnippet,
  type SearchResult,
  type HybridQueryResult,
} from "./index.js";

type WikiSearchHit = {
  pageid: number;
  title: string;
  snippet: string;
};

export type WikipediaPage = {
  pageid: number;
  title: string;
  url: string;
  extract: string;
};

export type WikipediaCitation = {
  id: number;
  title: string;
  url: string;
  displayPath: string;
  score: number;
  excerpt: string;
};

export type WikipediaRagPaths = {
  projectRoot: string;
  corpusDir: string;
  dbPath: string;
  manifestPath: string;
};

export type BuildCorpusResult = WikipediaRagPaths & {
  topic: string;
  pages: WikipediaPage[];
  documents: Array<{ displayPath: string; title: string; url: string }>;
};

export type AskResult = {
  question: string;
  answer: string;
  citations: WikipediaCitation[];
};

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");

export function resolveWikipediaRagPaths(baseDir = repoRoot): WikipediaRagPaths {
  const projectRoot = join(baseDir, "examples", "wikipedia-rag");

  return {
    projectRoot,
    corpusDir: join(projectRoot, "corpus"),
    dbPath: join(projectRoot, "wiki-rag.sqlite"),
    manifestPath: join(projectRoot, "manifest.json"),
  };
}

function ensureDir(path: string): void {
  if (!existsSync(path)) {
    mkdirSync(path, { recursive: true });
  }
}

function slugify(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-{2,}/g, "-")
    .toLowerCase();
}

function wikiPageUrl(title: string): string {
  return `https://en.wikipedia.org/wiki/${encodeURIComponent(title.replace(/ /g, "_"))}`;
}

async function fetchJson<T>(url: URL): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Wikipedia request failed with HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

async function searchWikipedia(topic: string, limit: number): Promise<WikiSearchHit[]> {
  const url = new URL("https://en.wikipedia.org/w/api.php");
  url.searchParams.set("action", "query");
  url.searchParams.set("format", "json");
  url.searchParams.set("utf8", "1");
  url.searchParams.set("list", "search");
  url.searchParams.set("srsearch", topic);
  url.searchParams.set("srlimit", String(limit));

  type Response = {
    query?: {
      search?: WikiSearchHit[];
    };
  };

  const payload = await fetchJson<Response>(url);
  return payload.query?.search ?? [];
}

async function fetchWikipediaPages(titles: string[]): Promise<WikipediaPage[]> {
  if (titles.length === 0) {
    return [];
  }

  const url = new URL("https://en.wikipedia.org/w/api.php");
  url.searchParams.set("action", "query");
  url.searchParams.set("format", "json");
  url.searchParams.set("utf8", "1");
  url.searchParams.set("redirects", "1");
  url.searchParams.set("prop", "extracts|info");
  url.searchParams.set("explaintext", "1");
  url.searchParams.set("exsectionformat", "plain");
  url.searchParams.set("inprop", "url");
  url.searchParams.set("titles", titles.join("|"));

  type Response = {
    query?: {
      pages?: Record<string, { pageid: number; title: string; fullurl?: string; extract?: string }>;
    };
  };

  const payload = await fetchJson<Response>(url);
  const pages = Object.values(payload.query?.pages ?? {}) as Array<{ pageid: number; title: string; fullurl?: string; extract?: string }>;

  return pages
    .map((page) => ({
      pageid: page.pageid,
      title: page.title,
      url: page.fullurl ?? wikiPageUrl(page.title),
      extract: page.extract?.trim() || "",
    }))
    .filter((page) => page.extract.length > 0);
}

function manifestFilePath(paths: WikipediaRagPaths): string {
  return paths.manifestPath;
}

function corpusFilePath(paths: WikipediaRagPaths, page: WikipediaPage): string {
  return join(paths.corpusDir, `${String(page.pageid).padStart(8, "0")}-${slugify(page.title)}.md`);
}

function writeWikipediaDocument(paths: WikipediaRagPaths, topic: string, page: WikipediaPage): { displayPath: string; title: string; url: string } {
  ensureDir(paths.projectRoot);
  ensureDir(paths.corpusDir);

  const displayPath = `${String(page.pageid).padStart(8, "0")}-${slugify(page.title)}.md`;
  const body = [
    `# ${page.title}`,
    "",
    `Source: ${page.url}`,
    `Topic: ${topic}`,
    "",
    page.extract,
    "",
  ].join("\n");

  writeFileSync(corpusFilePath(paths, page), body, "utf-8");
  return { displayPath, title: page.title, url: page.url };
}

function writeManifest(paths: WikipediaRagPaths, topic: string, documents: Array<{ displayPath: string; title: string; url: string }>): void {
  ensureDir(paths.projectRoot);
  writeFileSync(
    manifestFilePath(paths),
    JSON.stringify({ topic, documents }, null, 2),
    "utf-8"
  );
}

function readManifest(paths: WikipediaRagPaths): { topic: string | null; documents: Record<string, { title: string; url: string }> } {
  if (!existsSync(manifestFilePath(paths))) {
    return { topic: null, documents: {} };
  }

  try {
    const raw = JSON.parse(readFileSync(manifestFilePath(paths), "utf-8")) as {
      topic?: string;
      documents?: Array<{ displayPath: string; title: string; url: string }>;
    };

    return {
      topic: raw.topic ?? null,
      documents: Object.fromEntries((raw.documents ?? []).map((doc) => [doc.displayPath, { title: doc.title, url: doc.url }])),
    };
  } catch {
    return { topic: null, documents: {} };
  }
}

async function openStore(paths: WikipediaRagPaths) {
  ensureDir(paths.projectRoot);
  ensureDir(paths.corpusDir);

  return createStore({
    dbPath: paths.dbPath,
    config: {
      collections: {
        wikipedia: { path: paths.corpusDir, pattern: "**/*.md" },
      },
    },
  });
}

export async function buildWikipediaCorpus(
  topic: string,
  options: { maxPages?: number; baseDir?: string } = {}
): Promise<BuildCorpusResult> {
  const maxPages = options.maxPages ?? 5;
  const paths = resolveWikipediaRagPaths(options.baseDir ?? repoRoot);

  const hits = await searchWikipedia(topic, maxPages);
  const titles = hits.map((hit) => hit.title);
  const pages = await fetchWikipediaPages(titles);
  const documents = pages.map((page) => writeWikipediaDocument(paths, topic, page));

  writeManifest(paths, topic, documents);

  const store = await openStore(paths);
  try {
    await store.update({ collections: ["wikipedia"] });
  } finally {
    await store.close();
  }

  return { ...paths, topic, pages, documents };
}

function formatSnippetFromBody(body: string, question: string): string {
  const { snippet } = extractSnippet(body, question, 420);
  const cleaned = snippet
    .replace(/^@@.*\n/, "")
    .trim();

  return cleaned.length > 0 ? cleaned : body.slice(0, 420).trim();
}

async function maybeGenerateAnswer(
  question: string,
  citations: WikipediaCitation[],
  store: Awaited<ReturnType<typeof openStore>>
): Promise<string | null> {
  if (process.env.QMD_WIKI_RAG_USE_LLM !== "1") {
    return null;
  }

  const llm = (store.internal as { llm?: { generate: (prompt: string, options?: { maxTokens?: number; temperature?: number }) => Promise<{ text: string } | null> } }).llm;
  if (!llm) {
    return null;
  }

  const evidence = citations
    .map((citation) => `[${citation.id}] ${citation.title}\nURL: ${citation.url}\nExcerpt: ${citation.excerpt}`)
    .join("\n\n");

  const prompt = [
    "You are a Wikipedia RAG bot.",
    "Answer the user's question using only the evidence below.",
    "Cite every factual claim inline with bracketed citations like [1].",
    "If the evidence is insufficient, say so clearly.",
    "",
    `Question: ${question}`,
    "",
    "Evidence:",
    evidence,
    "",
    "Return a concise answer and then a short Sources section.",
  ].join("\n");

  try {
    const result = await llm.generate(prompt, { maxTokens: 500, temperature: 0.2 });
    return result?.text?.trim() || null;
  } catch {
    return null;
  }
}

function fallbackAnswer(question: string, citations: WikipediaCitation[]): string {
  if (citations.length === 0) {
    return `I could not find relevant Wikipedia pages for "${question}".`;
  }

  const lines = citations.slice(0, 3).map((citation) => {
    const sentence = citation.excerpt.split(/\.(?=\s|$)/).find(Boolean)?.trim() || citation.excerpt.trim();
    return `- [${citation.id}] ${sentence}`;
  });

  return [
    `Evidence-backed summary for "${question}":`,
    ...lines,
    "",
    "Sources are listed below with titles and URLs.",
  ].join("\n");
}

function renderSources(citations: WikipediaCitation[]): string {
  if (citations.length === 0) {
    return "";
  }

  return [
    "Sources",
    ...citations.map((citation) => `[${citation.id}] ${citation.title} — ${citation.url}\n    ${citation.excerpt}`),
  ].join("\n");
}

async function toCitation(
  index: number,
  result: SearchResult | HybridQueryResult,
  manifest: Record<string, { title: string; url: string }>,
  question: string,
  store: Awaited<ReturnType<typeof openStore>>
): WikipediaCitation {
  const source = manifest[result.displayPath] ?? { title: result.title, url: wikiPageUrl(result.title) };
  const filepath = "filepath" in result ? result.filepath : result.file;
  let body = "body" in result && result.body ? result.body : "";

  if (!body) {
    const loaded = await store.get(filepath, { includeBody: true });
    if (!("error" in loaded)) {
      body = loaded.body ?? "";
    }
  }

  if (!body && "bestChunk" in result) {
    body = result.bestChunk;
  }

  return {
    id: index + 1,
    title: source.title,
    url: source.url,
    displayPath: result.displayPath,
    score: result.score,
    excerpt: formatSnippetFromBody(body, question),
  };
}

export async function answerWikipediaQuestion(
  question: string,
  options: { baseDir?: string; maxResults?: number } = {}
): Promise<AskResult> {
  const paths = resolveWikipediaRagPaths(options.baseDir ?? repoRoot);
  const maxResults = options.maxResults ?? 5;
  const manifest = readManifest(paths);
  const store = await openStore(paths);

  try {
    const query = question.trim();
    const results = await store.searchLex(query, {
      collection: "wikipedia",
      limit: maxResults,
    });

    const fallbackQuery = [manifest.topic, ...Object.values(manifest.documents).map((doc) => doc.title), query]
      .filter((value): value is string => !!value && value.trim().length > 0)
      .join(" ");

    const selectedResults =
      results.length > 0
        ? results
        : await store.searchLex(fallbackQuery, {
            collection: "wikipedia",
            limit: maxResults,
          });

    const citations =
      selectedResults.length > 0
        ? await Promise.all(
            selectedResults.map((result, index) => toCitation(index, result, manifest.documents, question, store))
          )
        : await Promise.all(
            Object.entries(manifest.documents).map(async ([displayPath, source], index) => {
              const loaded = await store.get(`qmd://wikipedia/${displayPath}`, { includeBody: true });
              const body = !("error" in loaded) ? loaded.body ?? "" : "";
              return {
                id: index + 1,
                title: source.title,
                url: source.url,
                displayPath,
                score: 0,
                excerpt: formatSnippetFromBody(body || source.title, question),
              };
            })
          );
    const llmAnswer = await maybeGenerateAnswer(question, citations, store);
    const answer = llmAnswer || fallbackAnswer(question, citations);

    return {
      question,
      answer: `${answer}\n\n${renderSources(citations)}`.trim(),
      citations,
    };
  } finally {
    await store.close();
  }
}

export async function runWikipediaRagDemo(topic: string, question: string, options: { maxPages?: number; baseDir?: string } = {}): Promise<AskResult> {
  await buildWikipediaCorpus(topic, options);
  return answerWikipediaQuestion(question, { baseDir: options.baseDir, maxResults: 5 });
}
