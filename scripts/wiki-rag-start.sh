#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
EXAMPLE_ENV_FILE="${ROOT_DIR}/.env.example"

load_env_file() {
  local file="$1"
  local mode="${2:-override}"
  if [[ ! -f "$file" ]]; then
    return
  fi

  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
      local key="${line%%=*}"
      if [[ "$mode" == "default" && -n "${!key:-}" ]]; then
        continue
      fi
      export "$line"
    fi
  done < "$file"
}

load_env_file "$EXAMPLE_ENV_FILE" default
load_env_file "$ENV_FILE" override

PORT="${PORT:-4055}"
QMD_WIKI_RAG_PROVIDER="${QMD_WIKI_RAG_PROVIDER:-ollama}"
QMD_WIKI_RAG_OLLAMA_MODEL="${QMD_WIKI_RAG_OLLAMA_MODEL:-wiki-rag-answer-dpo}"
QMD_WIKI_RAG_OLLAMA_TIMEOUT_MS="${QMD_WIKI_RAG_OLLAMA_TIMEOUT_MS:-45000}"
QMD_WIKI_RAG_MAX_TOKENS="${QMD_WIKI_RAG_MAX_TOKENS:-220}"
export PORT QMD_WIKI_RAG_PROVIDER QMD_WIKI_RAG_OLLAMA_MODEL QMD_WIKI_RAG_OLLAMA_TIMEOUT_MS QMD_WIKI_RAG_MAX_TOKENS

if command -v lsof >/dev/null 2>&1 && lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use. Set PORT=4056 or stop the existing process."
  exit 1
fi

if [[ "$QMD_WIKI_RAG_PROVIDER" == "ollama" ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is required for QMD_WIKI_RAG_PROVIDER=ollama."
    echo "Install Ollama, then create the model from finetune/Modelfile.wiki-rag-answer-dpo."
    exit 1
  fi

  if ! ollama list | awk '{print $1}' | grep -qx "${QMD_WIKI_RAG_OLLAMA_MODEL}:latest"; then
    echo "Ollama model '${QMD_WIKI_RAG_OLLAMA_MODEL}' is not installed."
    echo "Expected setup:"
    echo "  ollama create wiki-rag-answer-dpo -f finetune/Modelfile.wiki-rag-answer-dpo"
    echo ""
    echo "If the GGUF file is missing, download/unzip it into:"
    echo "  finetune/outputs/wiki-rag-answer-dpo/gguf/"
    exit 1
  fi
fi

echo "Starting Wikipedia RAG Studio"
echo "  URL:      http://127.0.0.1:${PORT}"
echo "  Provider: ${QMD_WIKI_RAG_PROVIDER}"
echo "  Model:    ${QMD_WIKI_RAG_OLLAMA_MODEL}"
echo ""

cd "$ROOT_DIR"
exec npm run wiki-rag:ui
