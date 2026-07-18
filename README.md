# Bolna × Loom — Intelligent Voice AI with Dynamic LLM Routing

A production-shaped AI **voice assistant** where the voice layer never knows which LLM is answering.
[**Bolna**](https://bolna.ai) handles the phone/voice interface, a **FastAPI** backend orchestrates the conversation, and [**Loom**](https://loom-weaves.vercel.app/docs) is the *single* gateway to every LLM provider — so swapping models or providers is a **config change, not a code change**.

> **Status:** 🚧 Early development. The architecture and Loom integration are defined; backend modules are being built one at a time. This README describes the target design — see [Roadmap](#roadmap) for what's actually implemented.

---

## Why this project

Most voice-AI stacks hardwire a single LLM SDK throughout the app. Changing model, provider, or routing strategy then means touching business logic everywhere. This project inverts that:

- **One interface to every provider** — the backend never imports `openai`, `anthropic`, or `google-generativeai` directly. All of it goes through Loom.
- **The voice agent is provider-blind** — Bolna only ever talks to our backend. It has no idea whether a turn was served by GPT, Claude, or Gemini.
- **Routing is a decision, not a hardcode** — small talk, complex reasoning, structured output, and long context can each be sent to the best-fit model, and that policy lives in one swappable module.

It's built as a portfolio piece to demonstrate voice AI, async backend engineering, LLM abstraction, and clean architecture.

## Architecture

```
User
  │  (speaks on a call)
  ▼
Bolna Voice Agent ──────────► FastAPI Backend
                               │   ├─ routing.py   (task → provider/model)
                               │   ├─ memory.py    (conversation state)
                               │   └─ services/loom.py  ── the only caller of Loom
                               ▼
                              Loom  ──►  OpenAI / Anthropic / Gemini / …
                               │
                               ▼   (normalized response: text, usage, cost)
Bolna Voice Agent ◄─────────── FastAPI Backend
  │
  ▼
User  (hears the reply)
```

The voice agent only ever communicates with our backend. Provider selection, normalization, and cost/usage accounting all happen behind Loom.

## Features

- **Voice agent (Bolna)** — answers calls, holds natural conversation, asks follow-ups, ends gracefully.
- **FastAPI orchestration** — async webhooks for Bolna, conversation state, structured logging.
- **Loom as the sole LLM interface** — provider-agnostic; no vendor SDKs in application code.
- **Dynamic model routing** — task-aware model selection isolated in one module:
  | Task | Routed to |
  | --- | --- |
  | Small talk | cheap / fast model |
  | Complex reasoning | stronger model |
  | Structured / JSON output | best structured-output model |
  | Long context | larger-context model |
- **Modular conversation memory** — remembers user name, prior answers, context, and pending questions.
- **Config-driven** — providers, models, and keys come entirely from environment variables.
- **Observability** — logs the selected provider, model, latency, token usage, and cost per turn.
- **Deliberate error handling** — timeouts, provider failures, malformed requests; designed so provider failover can be added without touching call sites.

## Tech stack

- **Python** · **FastAPI** (async-first)
- **[Loom](https://loom-weaves.vercel.app/docs)** (`loom-router`) — unified, provider-agnostic LLM interface
- **Pydantic** for typed request/response models and settings
- **Bolna** for the voice/telephony layer
- Redis (optional) for conversation memory

## Project structure

```
backend/
  app.py                 # FastAPI entrypoint
  routes/                # Bolna-facing webhook endpoints
  services/
    loom.py              # the ONLY module that calls Loom
    routing.py           # task → provider/model decision (isolated & swappable)
    memory.py            # conversation history (modular backend)
    logger.py            # structured, readable logging
  agents/                # conversation / agent behavior
  models/                # Pydantic request/response models
  config/                # env-driven settings
  utils/
about.md                 # full project brief
CLAUDE.md                # working notes & guardrails for AI-assisted dev
.env                     # secrets (gitignored)
```

## How Loom is used

The backend talks to Loom and nothing else for LLM calls. Async, provider-agnostic:

```python
from loom import AsyncLoom

client = AsyncLoom.from_env()  # reads keys from .env + environment

result = await client.generate(
    provider="openai",         # openai | anthropic | gemini | deepseek | …
    modality="text",
    model="gpt-4o-mini",       # chosen by the routing layer, not hardcoded
    prompt="…",
    params={"temperature": 0.7, "max_tokens": 512},
)

reply = result["text"]         # normalized across providers
# result["usage"] → input/output/cached tokens · result["cost"] → usd/local
```

Switching providers is just different arguments (driven by config/routing) — the calling code doesn't change. See the [Loom docs](https://loom-weaves.vercel.app/docs).

## Getting started

> These steps describe the intended setup; modules are still being implemented (see [Roadmap](#roadmap)).

**Prerequisites:** Python 3.10+, a Bolna account, and at least one LLM provider key.

```bash
# 1. Clone
git clone <repo-url> && cd bolna_x_loom

# 2. Environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install (loom-router + FastAPI stack)
pip install "loom-router[all]" fastapi uvicorn pydantic pydantic-settings

# 4. Configure — copy the example and fill in keys
cp .env.example .env

# 5. Run the backend
uvicorn backend.app:app --reload
```

Then point your Bolna agent's webhook at the backend's endpoint.

## Configuration

All configuration is via environment variables — **no secrets in code**.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI provider key |
| `ANTHROPIC_API_KEY` | Anthropic provider key |
| `GEMINI_API_KEY` | Google Gemini provider key |
| `LOOM_DEFAULT_PROVIDER` | Provider used when routing has no strong preference |
| `LOOM_DEFAULT_MODEL` | Default model id |

## Roadmap

- [x] Project scaffolding & config (`config/`, `.env.example`)
- [x] Loom service (`services/loom.py`) — verified end-to-end call
- [x] Dynamic routing layer (`services/routing.py`)
- [ ] Conversation memory (`services/memory.py`)
- [ ] Bolna webhook routes & conversation orchestration
- [ ] Structured logging (provider, model, latency, tokens, cost)
- [ ] Provider failover / retry
- [ ] Deployment notes

## License

TBD.
