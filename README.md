# Offline AI assistant

A small pipeline that mirrors a thesis-style architecture: **intent detection** → **local retrieval** (FAISS over optional text) → **on-device inference** via [Ollama](https://ollama.com/). General answers can come entirely from the small language model (SLM); indexed files only add context when they match the query.

Inference stays local. Optional **Google sync** (Gmail, Calendar, Photos) only runs when you ask it to, and writes text into your `data/` folder for retrieval—answers still run offline through Ollama.

## Requirements

- **Python** 3.9+ (3.10+ recommended)
- **Ollama** installed and a model pulled (default: `phi3`)

```bash
ollama pull phi3
```

Override the model with the `OLLAMA_MODEL` environment variable.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run will download the sentence-transformer embedding model (`all-MiniLM-L6-v2`) used for FAISS.

## Usage

Ask a single question:

```bash
python3 -m app.main "What is photosynthesis?"
```

Interactive prompt (omit the argument):

```bash
python3 -m app.main
```

The CLI prints detected intent, retrieved context snippets, the model response, and simple timing/memory metrics.

## Data you can add

| Location | Role |
|----------|------|
| `data/general_knowledge.txt` | Optional general RAG lines (one concept per line; `#` starts a comment line, skipped) |
| `data/general_knowledge_extra.txt` | Same format, optional second file |
| `data/general/*.txt` | Additional general `.txt` files |
| `data/personal_context.txt` | Personal notes for retrieval |
| `data/google_personal_sync.txt` | Created by Google sync (see below); merged with personal context if present |

If no general text files exist, the assistant still answers from the SLM alone.

## Optional: sync Google data for personal context

This step uses Google’s APIs and a browser login; it does **not** replace Ollama—it only refreshes local text for retrieval.

1. In [Google Cloud Console](https://console.cloud.google.com/), create a project and enable **Gmail API**, **Google Calendar API**, and **Photos Library API**.
2. Create **OAuth 2.0 Client ID** credentials of type **Desktop app**, download the JSON, and save it as `credentials.json` in the project root (or set `GOOGLE_OAUTH_CREDENTIALS` to that file’s path).
3. Run:

```bash
python3 -m app.main --sync-google
```

A token is stored under `data/google_token.json` (configurable via `GOOGLE_OAUTH_TOKEN`). Synced text defaults to `data/google_personal_sync.txt` (`GOOGLE_SYNC_OUTPUT` overrides this).

Then ask questions as usual; personal-style queries will prefer retrieved lines from your notes and synced file.

## How it works (short)

1. **Intent** (`app/intent.py`): Heuristic routing—general knowledge, personal context, or hybrid—so retrieval can emphasize the right corpus.
2. **Retrieval** (`app/retriever.py`): Sentence embeddings + FAISS over your line-oriented text files.
3. **LLM** (`app/llm.py`): `ollama run <model>` with a built prompt; no cloud inference during the answer step.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `OLLAMA_MODEL` | Ollama model name (default: `phi3`) |
| `GOOGLE_OAUTH_CREDENTIALS` | Path to OAuth client JSON |
| `GOOGLE_OAUTH_TOKEN` | Path to stored user token |
| `GOOGLE_SYNC_OUTPUT` | Path for synced Google text dump |

## Privacy and git

`credentials.json`, `data/google_token.json`, and `data/google_personal_sync.txt` are listed in `.gitignore`. Do not commit secrets or personal sync output.
