#!/usr/bin/env tsx

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

import {
  answerWikipediaQuestion,
  buildWikipediaCorpus,
  resolveWikipediaRagPaths,
  runWikipediaRagDemo,
} from "../src/wikipedia-rag.js";

type UiStatus = {
  topic: string | null;
  corpusDir: string;
  dbPath: string;
  documentCount: number;
  manifestCount: number;
  recentHistory: UiEvent[];
};

type UiEvent = {
  at: string;
  kind: "ingest" | "ask" | "demo" | "error";
  title: string;
  detail: string;
};

type AppState = {
  busy: boolean;
  currentTopic: string | null;
  currentQuestion: string | null;
  lastAnswer: string | null;
  lastError: string | null;
  lastResult: Awaited<ReturnType<typeof answerWikipediaQuestion>> | null;
  history: UiEvent[];
};

const state: AppState = {
  busy: false,
  currentTopic: null,
  currentQuestion: null,
  lastAnswer: null,
  lastError: null,
  lastResult: null,
  history: [],
};

const DEFAULT_PORT = Number(process.env.PORT ?? "4011");

function paths() {
  return resolveWikipediaRagPaths();
}

function ensureDir(path: string): void {
  if (!existsSync(path)) {
    mkdirSync(path, { recursive: true });
  }
}

function readManifestSummary() {
  const p = paths();
  if (!existsSync(p.manifestPath)) {
    return { topic: null as string | null, documents: [] as Array<{ displayPath: string; title: string; url: string }> };
  }

  try {
    const raw = JSON.parse(readFileSync(p.manifestPath, "utf-8")) as {
      topic?: string;
      documents?: Array<{ displayPath: string; title: string; url: string }>;
    };

    return {
      topic: raw.topic ?? null,
      documents: raw.documents ?? [],
    };
  } catch {
    return { topic: null as string | null, documents: [] as Array<{ displayPath: string; title: string; url: string }> };
  }
}

function getStatus(): UiStatus {
  const p = paths();
  const manifest = readManifestSummary();
  const documentCount = existsSync(p.corpusDir)
    ? readdirSync(p.corpusDir).filter((entry) => entry.endsWith(".md")).length
    : 0;

  return {
    topic: manifest.topic,
    corpusDir: p.corpusDir,
    dbPath: p.dbPath,
    documentCount,
    manifestCount: manifest.documents.length,
    recentHistory: state.history.slice(0, 8),
  };
}

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function json(res: ServerResponse, statusCode: number, payload: unknown): void {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

async function readJsonBody(req: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    chunks.push(typeof chunk === "string" ? Buffer.from(chunk) : chunk);
  }

  if (chunks.length === 0) return {};
  const text = Buffer.concat(chunks).toString("utf-8").trim();
  return text ? JSON.parse(text) : {};
}

function addHistory(kind: UiEvent["kind"], title: string, detail: string): void {
  state.history.unshift({
    at: new Date().toISOString(),
    kind,
    title,
    detail,
  });
  state.history = state.history.slice(0, 20);
}

function splitAnswer(answer: string): { body: string; sources: string } {
  const marker = "\n\nSources";
  const idx = answer.indexOf(marker);
  if (idx < 0) {
    return { body: answer, sources: "" };
  }
  return {
    body: answer.slice(0, idx).trim(),
    sources: answer.slice(idx + 2).trim(),
  };
}

function renderPage(): string {
  const status = getStatus();
  const last = state.lastResult ? splitAnswer(state.lastResult.answer) : { body: "", sources: "" };
  const citations = state.lastResult?.citations ?? [];
  const recent = status.recentHistory;
  const hasCorpus = status.documentCount > 0;
  const answerProvider = state.lastResult?.answerProvider ?? "none";
  const ollamaEnabled = process.env.QMD_WIKI_RAG_PROVIDER === "ollama";

  return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Wikipedia RAG Studio</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f4f7fb;
        --panel: rgba(255, 255, 255, 0.88);
        --panel-strong: #ffffff;
        --text: #131722;
        --muted: #5c6577;
        --border: rgba(21, 28, 41, 0.08);
        --shadow: 0 14px 40px rgba(17, 24, 39, 0.08);
        --accent: #14532d;
        --accent-soft: #dff3e7;
        --warn: #7c2d12;
        --warn-soft: #feeadf;
        --radius: 18px;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(20, 83, 45, 0.08), transparent 30%),
          radial-gradient(circle at top right, rgba(37, 99, 235, 0.07), transparent 26%),
          var(--bg);
        color: var(--text);
      }
      .wrap {
        max-width: 1320px;
        margin: 0 auto;
        padding: 28px;
      }
      .hero {
        background: var(--panel);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: var(--shadow);
        padding: 28px;
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 20px;
      }
      .eyebrow {
        font-size: 12px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        font-weight: 700;
      }
      h1 {
        margin: 10px 0 8px;
        font-size: clamp(34px, 3vw, 56px);
        line-height: 1.02;
      }
      .lede {
        margin: 0;
        color: var(--muted);
        max-width: 70ch;
      }
      .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--panel-strong);
        font-weight: 700;
        white-space: nowrap;
      }
      .hero-actions {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 10px;
      }
      .chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
      }
      .chip {
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        padding: 5px 10px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--panel-strong);
        color: var(--muted);
        font-size: 12px;
        font-weight: 750;
      }
      .chip.strong {
        background: var(--accent-soft);
        color: var(--accent);
      }
      .dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: ${state.busy ? "#f59e0b" : hasCorpus ? "#14532d" : "#94a3b8"};
        box-shadow: 0 0 0 4px ${state.busy ? "rgba(245, 158, 11, 0.15)" : hasCorpus ? "rgba(20, 83, 45, 0.15)" : "rgba(148, 163, 184, 0.15)"};
      }
      .grid {
        display: grid;
        grid-template-columns: minmax(340px, 420px) minmax(0, 1fr);
        gap: 18px;
      }
      .stack {
        display: grid;
        gap: 18px;
      }
      .panel {
        background: var(--panel);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        padding: 18px;
      }
      .panel-header {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 10px;
      }
      .panel h2 {
        margin: 0 0 10px;
        font-size: 18px;
      }
      .panel-header h2 { margin: 0; }
      .sub {
        color: var(--muted);
        margin: 0 0 16px;
        font-size: 14px;
      }
      label {
        display: block;
        font-size: 13px;
        font-weight: 700;
        margin: 0 0 6px;
      }
      input, textarea, select {
        width: 100%;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.92);
        color: var(--text);
        padding: 12px 14px;
        font: inherit;
        outline: none;
      }
      textarea { min-height: 120px; resize: vertical; }
      input:focus, textarea:focus, select:focus { border-color: rgba(20, 83, 45, 0.35); box-shadow: 0 0 0 4px rgba(20, 83, 45, 0.08); }
      .row {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
      }
      .buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 14px;
      }
      button {
        appearance: none;
        border: 0;
        border-radius: 14px;
        padding: 12px 16px;
        font: inherit;
        font-weight: 700;
        cursor: pointer;
      }
      button.primary { background: #14532d; color: white; }
      button.secondary { background: #e5e7eb; color: #111827; }
      button.ghost { background: transparent; color: #111827; border: 1px solid var(--border); }
      button:hover:not(:disabled) { transform: translateY(-1px); }
      button:disabled { opacity: 0.5; cursor: not-allowed; }
      .metrics {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-top: 14px;
      }
      .metric {
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px;
      }
      .metric .k { color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }
      .metric .v { font-size: 22px; font-weight: 800; margin-top: 8px; }
      .metric .v.small { font-size: 15px; line-height: 1.25; word-break: break-word; }
      .answer {
        white-space: pre-wrap;
        line-height: 1.55;
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 16px;
        min-height: 180px;
      }
      .sources {
        display: grid;
        gap: 10px;
      }
      .source {
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 14px;
        display: grid;
        gap: 8px;
      }
      .source-title {
        display: flex;
        align-items: baseline;
        gap: 8px;
      }
      .source-id {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 28px;
        height: 28px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-weight: 800;
      }
      .source a {
        color: #14532d;
        text-decoration: none;
        word-break: break-word;
      }
      .source small, .muted {
        color: var(--muted);
      }
      .history {
        display: grid;
        gap: 10px;
      }
      .history-item {
        padding: 12px 14px;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: var(--panel-strong);
      }
      .history-item strong { display: block; margin-bottom: 4px; }
      .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        font-weight: 700;
        border-radius: 999px;
        padding: 6px 10px;
      }
      .badge.ok { background: var(--accent-soft); color: var(--accent); }
      .badge.warn { background: var(--warn-soft); color: var(--warn); }
      .columns {
        display: grid;
        grid-template-columns: 1.4fr 0.9fr;
        gap: 18px;
      }
      .list {
        display: grid;
        gap: 8px;
        margin: 0;
        padding: 0;
        list-style: none;
      }
      .list li {
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px 14px;
      }
      .footer-note { margin-top: 8px; color: var(--muted); font-size: 13px; }
      @media (max-width: 1000px) {
        .grid, .columns, .metrics, .row { grid-template-columns: 1fr; }
        .hero { flex-direction: column; }
        .hero-actions { align-items: flex-start; }
      }
    </style>
  </head>
  <body>
    <div class="wrap">
      <section class="hero">
        <div>
          <div class="eyebrow">QMD · Wikipedia RAG</div>
          <h1>Wikipedia RAG Studio</h1>
          <p class="lede">Build a topic corpus, ask grounded questions, and get numbered citations back from the retrieved Wikipedia pages.</p>
          <div class="chips">
            <span class="chip ${ollamaEnabled ? "strong" : ""}">Ollama ${ollamaEnabled ? "enabled" : "off"}</span>
            <span class="chip">Answer ${escapeHtml(answerProvider)}</span>
            <span class="chip">Local QMD index</span>
          </div>
        </div>
        <div class="hero-actions">
          <div class="status-pill"><span class="dot"></span>${state.busy ? "Working" : hasCorpus ? "Corpus ready" : "Idle"}</div>
        </div>
      </section>

      <div class="grid">
        <section class="panel">
          <h2>Control panel</h2>
          <p class="sub">Ingest a topic, ask a question against the corpus, or run both in one shot.</p>

          <div class="row">
            <div>
              <label for="topic">Topic</label>
              <input id="topic" name="topic" value="${escapeHtml(state.currentTopic ?? status.topic ?? "")}" placeholder="Ada Lovelace" />
            </div>
            <div>
              <label for="pages">Pages</label>
              <select id="pages" name="pages">
                ${[3, 5, 8, 10].map((n) => `<option value="${n}" ${n === 5 ? "selected" : ""}>${n}</option>`).join("")}
              </select>
            </div>
          </div>

          <div style="margin-top:12px">
            <label for="question">Question</label>
            <textarea id="question" name="question" placeholder="Why is Ada Lovelace important?">${escapeHtml(state.currentQuestion ?? "")}</textarea>
          </div>

          <div class="buttons">
            <button class="primary" id="demoBtn"${state.busy ? " disabled" : ""}>Build and ask</button>
            <button class="secondary" id="ingestBtn"${state.busy ? " disabled" : ""}>Build corpus</button>
            <button class="ghost" id="askBtn"${state.busy ? " disabled" : ""}>Ask only</button>
            <button class="ghost" id="refreshBtn">Refresh</button>
          </div>

          <div class="metrics">
            <div class="metric"><div class="k">Documents</div><div class="v">${status.documentCount}</div></div>
            <div class="metric"><div class="k">Manifest items</div><div class="v">${status.manifestCount}</div></div>
            <div class="metric"><div class="k">Answer</div><div class="v small">${escapeHtml(answerProvider)}</div></div>
            <div class="metric"><div class="k">Provider</div><div class="v small">${ollamaEnabled ? "Ollama" : "Fallback"}</div></div>
          </div>
          <div class="footer-note">Corpus path: <code>${escapeHtml(status.corpusDir)}</code></div>
        </section>

        <section class="stack">
          <section class="panel">
            <div class="panel-header">
              <div>
                <h2>Answer</h2>
                <p class="sub">The model answer is grounded in the retrieved pages.</p>
              </div>
              <span class="chip ${answerProvider === "ollama" ? "strong" : ""}">${escapeHtml(answerProvider)}</span>
            </div>
            <div class="answer">${escapeHtml(last.body || state.lastAnswer || "No answer yet. Build a corpus and ask a question.")}</div>
          </section>

          <section class="panel">
            <h2>Sources</h2>
            <p class="sub">Every cited page is listed with title, URL, and excerpt.</p>
            <div class="sources">
              ${citations.length > 0 ? citations.map((citation) => `
                <div class="source">
                  <div class="source-title"><span class="source-id">${citation.id}</span><strong>${escapeHtml(citation.title)}</strong></div>
                  <div><a href="${escapeHtml(citation.url)}" target="_blank" rel="noreferrer">${escapeHtml(citation.url)}</a></div>
                  <small>${escapeHtml(citation.excerpt)}</small>
                </div>
              `).join("") : `<div class="source"><span class="muted">No citations yet.</span></div>`}
            </div>
          </section>
        </section>
      </div>

      <div class="columns" style="margin-top:18px">
        <section class="panel">
          <h2>Corpus pages</h2>
          <p class="sub">What is actually on disk and available to QMD.</p>
          <ul class="list">
            ${status.documentCount > 0
              ? readManifestSummary().documents.slice(0, 8).map((doc) => `<li><strong>${escapeHtml(doc.title)}</strong><div class="muted">${escapeHtml(doc.url)}</div></li>`).join("")
              : `<li class="muted">No corpus has been built yet.</li>`}
          </ul>
        </section>

        <section class="panel">
          <h2>Recent activity</h2>
          <p class="sub">Latest local actions in this session.</p>
          <div class="history">
            ${recent.length > 0 ? recent.map((item) => `
              <div class="history-item">
                <strong>${escapeHtml(item.title)}</strong>
                <span class="badge ${item.kind === "error" ? "warn" : "ok"}">${escapeHtml(item.kind)}</span>
                <div class="muted" style="margin-top:8px">${escapeHtml(item.detail)}</div>
                <div class="muted" style="margin-top:6px">${escapeHtml(item.at)}</div>
              </div>
            `).join("") : `<div class="history-item muted">No actions yet.</div>`}
          </div>
        </section>
      </div>
    </div>

    <script>
      const topic = document.getElementById("topic");
      const question = document.getElementById("question");
      const pages = document.getElementById("pages");
      const demoBtn = document.getElementById("demoBtn");
      const ingestBtn = document.getElementById("ingestBtn");
      const askBtn = document.getElementById("askBtn");
      const refreshBtn = document.getElementById("refreshBtn");

      async function post(path, payload) {
        const response = await fetch(path, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data?.error || "Request failed");
        return data;
      }

      function lockButtons(locked) {
        [demoBtn, ingestBtn, askBtn].forEach((btn) => btn.disabled = locked);
      }

      demoBtn?.addEventListener("click", async () => {
        lockButtons(true);
        try {
          await post("/api/demo", { topic: topic.value.trim(), question: question.value.trim(), pages: Number(pages.value) });
          window.location.reload();
        } catch (error) {
          alert(error.message || String(error));
        } finally {
          lockButtons(false);
        }
      });

      ingestBtn?.addEventListener("click", async () => {
        lockButtons(true);
        try {
          await post("/api/ingest", { topic: topic.value.trim(), pages: Number(pages.value) });
          window.location.reload();
        } catch (error) {
          alert(error.message || String(error));
        } finally {
          lockButtons(false);
        }
      });

      askBtn?.addEventListener("click", async () => {
        lockButtons(true);
        try {
          await post("/api/ask", { question: question.value.trim() });
          window.location.reload();
        } catch (error) {
          alert(error.message || String(error));
        } finally {
          lockButtons(false);
        }
      });

      refreshBtn?.addEventListener("click", () => window.location.reload());
    </script>
  </body>
</html>`;
}

async function handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? "/", "http://localhost");

  if ((req.method === "GET" || req.method === "HEAD") && url.pathname === "/") {
    const html = renderPage();
    res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
    res.end(req.method === "HEAD" ? undefined : html);
    return;
  }

  if (req.method === "HEAD" && url.pathname === "/api/status") {
    const body = JSON.stringify({
      busy: state.busy,
      currentTopic: state.currentTopic,
      currentQuestion: state.currentQuestion,
      lastError: state.lastError,
      status: getStatus(),
      hasAnswer: !!state.lastResult,
    });
    res.writeHead(200, {
      "content-type": "application/json; charset=utf-8",
      "content-length": Buffer.byteLength(body),
    });
    res.end();
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/status") {
    json(res, 200, {
      busy: state.busy,
      currentTopic: state.currentTopic,
      currentQuestion: state.currentQuestion,
      lastError: state.lastError,
      status: getStatus(),
      hasAnswer: !!state.lastResult,
    });
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/ingest") {
    if (state.busy) {
      json(res, 409, { error: "The app is already processing another request." });
      return;
    }

    const body = await readJsonBody(req);
    const topic = String(body.topic ?? "").trim();
    const pages = Number.isFinite(Number(body.pages)) ? Math.max(1, Math.min(12, Number(body.pages))) : 5;

    if (!topic) {
      json(res, 400, { error: "Topic is required." });
      return;
    }

    state.busy = true;
    state.lastError = null;
    state.currentTopic = topic;
    state.currentQuestion = null;

    try {
      const result = await buildWikipediaCorpus(topic, { maxPages: pages });
      addHistory("ingest", `Built corpus for ${topic}`, `${result.pages.length} pages indexed into QMD.`);
      json(res, 200, {
        ok: true,
        topic,
        pages: result.pages.length,
        corpusDir: result.corpusDir,
        dbPath: result.dbPath,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      state.lastError = message;
      addHistory("error", `Ingest failed for ${topic}`, message);
      json(res, 500, { error: message });
    } finally {
      state.busy = false;
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/ask") {
    if (state.busy) {
      json(res, 409, { error: "The app is already processing another request." });
      return;
    }

    const body = await readJsonBody(req);
    const question = String(body.question ?? "").trim();
    if (!question) {
      json(res, 400, { error: "Question is required." });
      return;
    }

    state.busy = true;
    state.lastError = null;
    state.currentQuestion = question;

    try {
      const result = await answerWikipediaQuestion(question);
      state.lastResult = result;
      state.lastAnswer = result.answer;
      addHistory("ask", `Answered: ${question}`, `${result.citations.length} citations returned via ${result.answerProvider}.`);
      json(res, 200, {
        ok: true,
        question,
        answer: result.answer,
        citations: result.citations,
        answerProvider: result.answerProvider,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      state.lastError = message;
      addHistory("error", `Ask failed`, message);
      json(res, 500, { error: message });
    } finally {
      state.busy = false;
    }
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/demo") {
    if (state.busy) {
      json(res, 409, { error: "The app is already processing another request." });
      return;
    }

    const body = await readJsonBody(req);
    const topic = String(body.topic ?? "").trim();
    const question = String(body.question ?? "").trim();
    const pages = Number.isFinite(Number(body.pages)) ? Math.max(1, Math.min(12, Number(body.pages))) : 5;

    if (!topic) {
      json(res, 400, { error: "Topic is required." });
      return;
    }

    if (!question) {
      json(res, 400, { error: "Question is required." });
      return;
    }

    state.busy = true;
    state.lastError = null;
    state.currentTopic = topic;
    state.currentQuestion = question;

    try {
      const result = await runWikipediaRagDemo(topic, question, { maxPages: pages });
      state.lastResult = result;
      state.lastAnswer = result.answer;
      addHistory("demo", `Built and answered ${topic}`, `${result.citations.length} citations returned via ${result.answerProvider}.`);
      json(res, 200, {
        ok: true,
        topic,
        question,
        answer: result.answer,
        citations: result.citations,
        answerProvider: result.answerProvider,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      state.lastError = message;
      addHistory("error", `Demo failed`, message);
      json(res, 500, { error: message });
    } finally {
      state.busy = false;
    }
    return;
  }

  json(res, 404, { error: "Not found" });
}

async function main() {
  const p = paths();
  ensureDir(p.projectRoot);
  ensureDir(p.corpusDir);

  const server = createServer((req, res) => {
    void handleRequest(req, res).catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      if (!res.headersSent) {
        json(res, 500, { error: message });
      } else {
        res.end();
      }
    });
  });

  await new Promise<void>((resolve) => {
    server.listen(DEFAULT_PORT, "127.0.0.1", () => resolve());
  });

  const address = server.address();
  const port = typeof address === "object" && address ? address.port : DEFAULT_PORT;
  console.log(`Wikipedia RAG Studio running at http://127.0.0.1:${port}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
