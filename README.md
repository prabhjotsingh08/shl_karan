# SHL Conversational Assessment Recommender

Stateless conversational agent that helps hiring managers move from a vague intent (“I am hiring a Java developer”) to a **grounded shortlist of SHL Individual Test Solutions** from the [SHL product catalog](https://www.shl.com/solutions/products/product-catalog/).

Aligned with **SHL_AI_Intern_Assignment.pdf** (SHL Labs AI Intern take-home).

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 1) Build catalog (once, or when SHL site changes)
python scripts/crawl_shl_catalog.py

# 2) Build vector index
python scripts/build_vector_store.py --reset

# 3) API + UI (two terminals)
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py
```

Optional `.env`:

```bash
GEMINI_API_KEY=...           # better type extraction from queries
CHAT_USE_GEMINI_REPLIES=false
LOGFIRE_API_KEY=...
CHROMA_PATH=chroma_db
```

---

## Assignment compliance checklist

| Requirement | Implementation |
|-------------|----------------|
| `GET /health` → `{"status": "ok"}` | `backend/main.py` |
| `POST /chat` stateless, full `messages` history | `ChatAgent` + `ChatRequest` / `ChatResponse` |
| Response schema: `reply`, `recommendations`, `end_of_conversation` | `data_models.ChatResponse` |
| Recommendations: `name`, `url`, `test_type` (catalog codes e.g. `K`, `K,P`) | `ChatRecommendationItem` |
| 1–10 recommendations when committed; `[]` when clarifying/refusing | Orchestrator + analyzer |
| Clarify vague queries before recommending | `conversation_analyzer` + `clarification_engine` |
| Refine mid-conversation | Re-retrieve on merged user history |
| Compare assessments (catalog-grounded) | `comparison_engine` |
| Refuse off-topic / injection | `refusal_guard` |
| Max **8 turns (user + assistant messages)** | `len(messages) > 8` in `chat_agent.py` |
| Catalog-only URLs | `validate_catalog_recommendations` |
| Individual Test Solutions only | `crawler.py` filters pre-packaged job solutions |
| Target latency &lt; 30s | Rule-based routing; optional Gemini polish off by default |

Extra (not required by assignment): `POST /recommend` for batch Recall@10 workflows.

---

## How to run the scraper

The scraper lives in `src/shl_recommender/crawler.py` and is invoked via:

```bash
python scripts/crawl_shl_catalog.py
```

Optional paths:

```bash
python scripts/crawl_shl_catalog.py --output data/shl_individual_assessments.csv --json-output data/shl_individual_assessments.json
```

### What it does

1. Fetches all catalog listing pages from `https://www.shl.com/products/product-catalog/` (Individual Test Solutions table only).
2. For each row, opens the detail page and extracts description, job levels, languages, duration, remote/adaptive flags, and type codes (A, B, C, D, E, K, P, S).
3. Saves progress and raw HTML under `data_pages/page_XX.html` (cache for re-runs; gitignored).

### Where data is stored

| Output | Path | How to view |
|--------|------|-------------|
| **CSV** (used by app) | `data/shl_individual_assessments.csv` | Excel, `pandas`, any text editor |
| **JSON** (same records) | `data/shl_individual_assessments.json` | Editor or `python -m json.tool data/shl_individual_assessments.json` |
| **HTML cache** | `data_pages/page_01.html` … | Browser or editor (debugging only) |

Quick peek:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/shl_individual_assessments.csv'); print(df.shape); print(df[['name','url','assessment_types']].head())"
```

After crawling, rebuild embeddings:

```bash
python scripts/build_vector_store.py --reset
```

### If crawl fails with `getaddrinfo failed` / `Failed to resolve 'www.shl.com'`

That is a **DNS/network** problem on your machine (not a Python bug). Windows cannot resolve `www.shl.com` to an IP address.

**Fix connectivity first:**

```powershell
Resolve-DnsName www.shl.com
ping www.shl.com
```

Try: different network, disable VPN, flush DNS (`ipconfig /flushdns`), or set DNS to `8.8.8.8` / `1.1.1.1`.

**Work without live SHL access:**

```bash
# From cached listing HTML (if you have data_pages/page_*.html)
python scripts/crawl_shl_catalog.py --offline

# From an existing JSON export (no network)
python scripts/crawl_shl_catalog.py --from-json data/shl_individual_assessments.json
```

If you already have `data/shl_individual_assessments.csv`, skip crawling and run `python scripts/build_vector_store.py --reset` only.

---

## How retrieval works (RAG)

This project uses **retrieval-augmented ranking**, not a document-QA LLM chain.

### “Chunking”

There is **no text splitting**. Each SHL assessment is **one chunk** — a single embedding of `AssessmentMetadata.combined_text()`:

- name  
- description  
- job levels, languages, duration  
- type codes  

Roughly one vector per catalog row (~hundreds of assessments).

### Embedding model

`sentence-transformers/all-MiniLM-L6-v2` (384-dim) via `EmbeddingService` in `embedding.py`.

### Vector store

**ChromaDB** (persistent, cosine HNSW) at `chroma_db/`, collection `shl_assessments`.

### Matching (retrieval + ranking)

1. **Embed** the user query (or merged conversation text).
2. **Query Chroma** for top `candidate_pool_size` (default 20) by cosine similarity.
3. **Extract assessment types** from the query (Gemini if configured, else keyword heuristics in `type_extraction.py`).
4. **Rank**: type-matching candidates first, then by embedding similarity within each group.
5. **Count**: return 1–10 for `/chat`, 5–10 default for `/recommend`.

The LLM (Gemini) is **optional** and used for type codes only — not for generating assessment names. That keeps recommendations grounded in the catalog.

```
User messages → merged query text
       → embed query
       → Chroma similarity search
       → type-aware re-rank
       → validate URLs ⊆ catalog CSV
       → ChatRecommendationItem(name, url, test_type)
```

---

## Why 8 turns?

The assignment states:

> *The evaluator caps each conversation at **8 turns including user & assistant** and each call at a 30 second timeout.*

So **one turn = one message** (whether from the user or the assistant). A full exchange is typically 2 turns (user + assistant).

**Why it exists**

- Forces concise clarification instead of endless back-and-forth.  
- Keeps automated replay harness runs bounded.  
- Matches how SHL scores “turn cap” hard evals.

**Implementation**

- `config.max_conversation_turns = 8`  
- `chat_agent.py`: if `len(messages) > 8` → `end_of_conversation: true` and a closing message.  
- Streamlit UI shows `message_count / 8`.

---

## Project layout and how files interact

```
                    ┌─────────────────────────────────────────┐
                    │           scripts/ (CLI, offline)        │
                    │ crawl_shl_catalog → build_vector_store   │
                    │ generate_predictions, run_conversation_* │
                    └───────────────┬─────────────────────────┘
                                    │ writes CSV + Chroma
                                    ▼
┌──────────────┐    HTTP     ┌──────────────┐    imports    ┌────────────────────────────┐
│ frontend/    │ ──────────► │ backend/     │ ────────────► │ src/shl_recommender/       │
│ Streamlit UI │  /chat      │ main.py      │               │                            │
└──────────────┘  /health    └──────────────┘               │  agent/chat_agent.py ◄───┐ │
                                                              │    ├─ conversation_analyzer
                                                              │    ├─ clarification_engine
                                                              │    ├─ refusal_guard          │
                                                              │    ├─ comparison_engine      │
                                                              │    └─ recommendation_orchestrator
                                                              │           │                │
                                                              │           ▼                │
                                                              │  recommender.py ───────────┘
                                                              │    ├─ embedding.py → Chroma
                                                              │    └─ type_extraction.py
                                                              │  data_models.py (schemas)
                                                              │  crawler.py (scrape only)
                                                              └────────────────────────────┘
```

### File reference

| Path | Role |
|------|------|
| `backend/main.py` | FastAPI app: `/health`, `/chat`, optional `/recommend` |
| `frontend/app.py` | Chat UI; sends full `messages[]` to `/chat` each turn |
| `src/shl_recommender/crawler.py` | Scrape SHL catalog → metadata list |
| `src/shl_recommender/embedding.py` | Load CSV, embed, upsert Chroma; query client |
| `src/shl_recommender/recommender.py` | Vector retrieval + type ranking (core quality) |
| `src/shl_recommender/type_extraction.py` | Gemini/heuristic type codes from text |
| `src/shl_recommender/agent/chat_agent.py` | Dialogue router (clarify / recommend / compare / refuse) |
| `src/shl_recommender/agent/conversation_analyzer.py` | Intent + “enough context?” signals |
| `src/shl_recommender/agent/clarification_engine.py` | One clarifying question when vague |
| `src/shl_recommender/agent/recommendation_orchestrator.py` | Wraps engine; catalog validation; chat schema |
| `src/shl_recommender/agent/comparison_engine.py` | Name match + catalog-only compare text |
| `src/shl_recommender/agent/refusal_guard.py` | Off-topic / injection detection |
| `src/shl_recommender/agent/prompt_templates.py` | Reply templates |
| `src/shl_recommender/agent/llm_reply.py` | Optional Gemini reply polish |
| `src/shl_recommender/data_models.py` | Pydantic models for API and catalog |
| `src/shl_recommender/config.py` | Settings from env |
| `src/shl_recommender/evaluation/conversation_simulator.py` | Multi-turn test harness |
| `scripts/crawl_shl_catalog.py` | CLI entry for crawler |
| `scripts/build_vector_store.py` | CLI entry for Chroma build |
| `scripts/run_conversation_simulator.py` | Scripted chat scenarios |
| `tests/test_agent.py`, `tests/test_recommender.py` | Unit tests |
| `data/shl_individual_assessments.csv` | Catalog source of truth |
| `chroma_db/` | Persisted embeddings (gitignored) |
| `migration_plan.md` | v1→v2 migration notes (internal doc) |

### Runtime flow (one `/chat` call)

1. Client POSTs `{ "messages": [...] }`.  
2. `backend/main.py` → `ChatAgent.handle()`.  
3. Turn cap → refusal → end → compare → clarify → recommend/refine.  
4. On recommend: `RecommendationOrchestrator` merges user text → `RecommendationEngine.recommend()` → Chroma + rank.  
5. Map to `ChatRecommendationItem` → return `ChatResponse`.

---

## API examples

### Health

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Chat — clarification (empty recommendations)

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}]}"
```

### Chat — recommendations

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Mid-level Java developer working with stakeholders; need technical and personality screening\"}]}"
```

Example response shape:

```json
{
  "reply": "Based on your needs...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

---

## Tests and simulation

```bash
python -m pytest tests/ -q
python scripts/run_conversation_simulator.py --scenario clarify_then_recommend
```

---

## Deployment (Render)

See `render.yaml`. Before submitting:

1. Deploy backend; ensure `data/shl_individual_assessments.csv` and `chroma_db/` exist on the instance.  
2. Set `GEMINI_API_KEY` if using Gemini type extraction.  
3. Verify `GET /health` and `POST /chat` from the public URL.

---

## Files safe to delete locally

These are **not required** in git or at runtime if you already have CSV + Chroma:

| Item | Reason |
|------|--------|
| `data_pages/` | HTML scrape cache; regenerated on crawl |
| `scripts/data_pages/` | Duplicate cache (removed from repo) |
| `scripts/chroma_db/` | Stray copy (removed from repo) |
| `__pycache__/`, `.pytest_cache/` | Auto-regenerated |
| Old assignment PDFs | Keep only `SHL_AI_Intern_Assignment.pdf` |

**Do not delete:** `data/shl_individual_assessments.csv`, `chroma_db/` (active index), or `src/`.

---

## Remaining improvements

- [ ] Eval harness on SHL’s 10 public conversation traces (Recall@10 + behavior probes)  
- [ ] Embedding-based name resolution for “compare X and Y”  
- [ ] Production load test for p95 &lt; 30s on `/chat`  
- [ ] Optional `CHAT_USE_GEMINI_REPLIES=true` after latency check  

See also [`migration_plan.md`](migration_plan.md) for the original v1→v2 migration analysis.
