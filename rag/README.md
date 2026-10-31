# RAG chatbot: ask questions about this project

A retrieval-augmented chatbot that answers questions about the dataset, the models (KNN, logistic regression, random forest), the metrics, and how to deploy and monitor the classifier. It answers **only** from the documents in `rag/knowledge/` and the repo `README.md`, cites its sources, and says "I don't know" instead of guessing.

```
question ─► input guardrails ─► BM25 retrieval (top-4 chunks) ─► Claude (structured JSON)
              │ injection,          rag/knowledge/*.md               │ answer, answerable, citations
              │ length, unicode     README.md                        ▼
              ▼                                                  output checks ─► answer + sources
           blocked                                               (citation ids, prompt-leak canary,
                                                                  refusal / truncation handling)
```

| File | What it does |
|---|---|
| `retriever.py` | Splits Markdown by heading and ranks chunks with BM25 (no vector DB, no embedding API, deterministic) |
| `guardrails.py` | Input checks, prompt-injection patterns, PII redaction, escaping of retrieved text |
| `chatbot.py` | The RAG loop: retrieve, prompt, call Claude, verify the output |
| `server.py` | FastAPI service: `/chat`, `/chat/feedback`, `/health`, `/metrics`, and the web UI at `/` |
| `evaluate.py` | Evaluation harness and quality gate |
| `eval/golden.jsonl` | 22 questions with expected sources, required facts and reference answers |
| `eval/redteam.jsonl` | 14 attacks: injection, jailbreaks, prompt extraction, secret exfiltration, medical advice, out-of-scope, hallucination bait |

**Adding knowledge:** drop more `.md` or `.txt` files into `rag/knowledge/` and restart the service. Then add a few questions about them to `eval/golden.jsonl` so the evaluation covers them.

## 1. Run it locally

```bash
pip install -r requirements-rag.txt
```

```bash
export ANTHROPIC_API_KEY=sk-ant-...      # your Claude API key
export RAG_API_KEYS=$(openssl rand -hex 24)   # the key clients must send; print it with: echo $RAG_API_KEYS
uvicorn rag.server:app --port 8100
```

Open http://localhost:8100, paste the `RAG_API_KEYS` value into the key box, and ask a question. Or call the API:

```bash
curl -X POST localhost:8100/chat -H "authorization: Bearer $RAG_API_KEYS" \
  -H 'content-type: application/json' -d '{"question":"Why does KNN need feature scaling?"}'
```

Example response:

```json
{"response_id":"…","answer":"KNN uses Euclidean distance, so …","answerable":true,
 "citations":[{"id":"models.md#3","source":"models.md","section":"K-nearest neighbors (KNN)","snippet":"…"}],
 "blocked":false,"disclaimer":"Educational demo. Not medical advice."}
```

Pass earlier turns in `history` (`[{"role":"user","content":…},{"role":"assistant","content":…}]`) for follow-up questions.

| Setting | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | (required) | Claude API key |
| `RAG_API_KEYS` | (required) | Comma-separated client keys. Without them `/chat` returns 503 (it fails closed) |
| `RAG_ALLOW_NO_AUTH` | off | Set to `1` to skip client keys. Local development only |
| `RAG_RATE_LIMIT` | 20 | Requests per minute per client key |
| `RAG_MODEL` | `claude-opus-5-5` | Claude model |
| `RAG_EFFORT` | `low` | Thinking effort. Raise to `medium` if answers to multi-part questions are thin |
| `RAG_TOP_K` | 4 | Chunks retrieved per question |
| `RAG_ENV` | | `production` hides `/docs` and `/openapi.json` |

## 2. Deploy

```bash
printf 'ANTHROPIC_API_KEY=sk-ant-...\nRAG_API_KEYS=%s\n' "$(openssl rand -hex 24)" > .env   # git-ignored
docker compose up --build rag            # http://localhost:8100
```

`docker compose up --build` starts it next to the classifier API, Prometheus and Grafana. Prometheus scrapes the chatbot's `/metrics`, and `monitoring/alerts.yml` has alerts for it.

The image (`Dockerfile.rag`) runs anywhere containers run. With Google Cloud Run, for example:

```bash
gcloud run deploy rag-chatbot --source . --port 8100 \
  --set-secrets ANTHROPIC_API_KEY=anthropic-key:latest,RAG_API_KEYS=rag-client-keys:latest
```

(Cloud Run builds from a file named `Dockerfile`, so either build `Dockerfile.rag` with Cloud Build first or deploy the pushed image with `--image`.) Keep both keys in the platform's secret manager, never in the image or in git. Put the service behind HTTPS (Cloud Run, App Runner and similar do this for you).

**Scaling note:** the rate limiter is in memory, so each replica counts on its own. With several replicas, move it to Redis or to your API gateway.

### Host it on Hugging Face Spaces (free, gives you a shareable link)

Run this on your own machine. It creates the Space, uploads the app and stores your keys as encrypted Space secrets:

```bash
pip install -U huggingface_hub anthropic
hf auth login                                   # paste a Hugging Face token with write access
export ANTHROPIC_API_KEY=sk-ant-...
python deploy/huggingface/deploy.py --public    # leave out --public to keep the Space private
```

It prints the chat link (`https://<user>-breast-cancer-docs-chatbot.hf.space/`), the dashboard link (`…/dashboard`), and the access key to type into the key box. Even a public Space requires that key for chat and metrics, so share it only with people you trust, and change it with `--access-key NEW_KEY`. On Spaces the usage and feedback logs reset when the Space restarts. To keep them, enable Persistent Storage and set `RAG_EVENTS_DB` and `RAG_FEEDBACK_LOG` to paths under `/data`.



```bash
python -m rag.evaluate --offline   # free and deterministic: retrieval + guardrails
python -m rag.evaluate             # full: calls Claude for answers, LLM judge, red team
```

The command prints every metric next to its threshold and **exits 1 if any metric fails**, so CI and deploy pipelines can use it as a gate. The full report, including each answer and the judge's reasoning, goes to `reports/rag_eval.json`.

| Layer | Metric | Threshold | How it's measured |
|---|---|---|---|
| Retrieval | hit rate @4 | ≥ 0.90 | An expected source file is in the top 4 chunks |
| Retrieval | MRR | ≥ 0.75 | How high the first relevant chunk ranks |
| Answers | answer rate | ≥ 0.90 | In-scope questions are answered, not refused |
| Answers | keyword recall | ≥ 0.85 | Required facts (e.g. "569", "0.25") appear in the answer |
| Answers | citation precision | ≥ 0.80 | Cited chunks come from the right source file |
| Answers | correctness | ≥ 4.0 / 5 | LLM judge compares the answer with a reference answer |
| Answers | faithfulness | ≥ 4.5 / 5 | LLM judge checks every claim against the retrieved text (hallucination check) |
| Security | input block rate | 100% | Direct injection, jailbreak and exfiltration prompts are stopped before the model |
| Security | false positives | 0% | No legitimate question is blocked |
| Security | red-team pass rate | 100% | Each attack is blocked, declined, or answered safely (judged) |
| Security | indirect injection | pass | A poisoned document in the corpus can't take over the answer |

Current offline results: retrieval hit rate 1.00, MRR 0.98, block rate 1.00, false positives 0.00. The full run needs an API key and makes about 60 Claude calls (roughly $1 per run with Opus 5.5).

**Keep it good after launch.**
- Every 👎 in the UI is written, with PII redacted, to `data/rag_feedback.jsonl`. Review it each week and turn real failures into new rows in `golden.jsonl` or `redteam.jsonl`.
- The `RagHighNoAnswerRate` alert means users are asking about things the docs don't cover. Add that content to `rag/knowledge/`.
- Re-run the full eval before changing the model, the prompt, `RAG_TOP_K` or the knowledge base.

## Dashboard: every metric in one place

Open **http://localhost:8100/dashboard** and enter one of your `RAG_API_KEYS` (the metrics endpoint needs a key, like `/chat`). Choose a window of 24 hours, 7, 30 or 90 days. The page refreshes every 30 seconds.

| Section | What it shows | Source |
|---|---|---|
| **Chatbot usage** | Requests, answer / "I don't know" / blocked / error rates, p50 and p95 latency, Claude tokens and estimated cost, cost per answer, citations per answer; charts of requests by outcome, latency and cost over time | `data/rag_events.db`: one row per request, holding no question text |
| **Quality & security** | Every evaluation metric next to its threshold with pass / fail / not run, a trend across eval runs, red-team results, and weak answers or retrieval misses | `reports/rag_eval*.json` and `reports/rag_eval_history.jsonl`, written by `python -m rag.evaluate` |
| **User feedback** | 👍 / 👎 counts, satisfaction rate, and the latest 👎 answers so you can fix them | `data/rag_feedback.jsonl` |
| **Classifier** | Model and version, prediction volume, confidence, offline vs. live accuracy / precision / recall / F1, predicted-class drift, per-feature PSI drift, and active alerts | The classifier API (`/model-info`, `/monitoring`) when `CLASSIFIER_URL` is set |

Under Docker Compose, `CLASSIFIER_URL` and the reports folder are already wired up. Locally, start the classifier API on port 8000 and then run:

```bash
CLASSIFIER_URL=http://localhost:8000 RAG_API_KEYS=change-me uvicorn rag.server:app --port 8100
```

Cost is estimated at `RAG_PRICE_IN` / `RAG_PRICE_OUT` USD per million tokens (default 4 / 20, the `claude-opus-5-5` list price). Change them if you switch models. The same request counters also go to Prometheus at `/metrics`, for Grafana and alerting.



| Threat | Defence |
|---|---|
| Unauthorised use, or someone running up your API bill | Bearer keys compared in constant time, per-key rate limit; the service refuses to run `/chat` without keys |
| Direct prompt injection and jailbreaks | NFKC unicode normalisation and high-confidence patterns block them before the model is called; the system prompt sets fixed rules |
| Indirect injection (instructions planted in documents) | Retrieved text is escaped and wrapped in `<document>` tags marked as untrusted data; tested with a poisoned document |
| System-prompt leakage | A random canary token sits in the system prompt; any answer containing it is suppressed |
| Hallucination | Answers must come from the retrieved documents; structured output with `answerable=false` when they don't; citations are checked against the chunks actually retrieved; no model call when retrieval finds nothing |
| Forged conversation history | History is re-validated: roles, alternation, length, injection patterns |
| Unsafe medical advice | The system prompt forbids diagnosis; the red team tests it; every response carries a disclaimer |
| Model safety refusals | `stop_reason == "refusal"` is handled; server-side `fallbacks: "default"` re-runs a declined request on Anthropic's recommended fallback model |
| XSS in the web UI | Model and document text are inserted with `textContent` only; strict CSP, `X-Frame-Options: DENY`, `nosniff` |
| Oversized or abusive input | 64 KB body cap, 2,000-character questions, 12 history turns, pydantic validation |
| PII and secrets in logs | Emails, phone numbers, SSNs, card numbers and `sk-ant-` keys are redacted; questions are truncated; raw answers aren't logged |
| Leaking internals | Errors return generic messages; `/docs` is hidden in production |
| Container hardening | Non-root user, read-only filesystem, all capabilities dropped, `no-new-privileges`, secrets injected at runtime |
| Vulnerable dependencies | `pip-audit` runs in CI |

**Not covered (add these before high-stakes or public use):** per-user identity (OAuth or SSO) instead of shared keys; a WAF or gateway in front of the service; a distributed rate limiter; the guardrails are pattern-based and can't catch every paraphrased injection, which is why the prompt and output layers exist; regular review of red-team results as new attack styles appear.
