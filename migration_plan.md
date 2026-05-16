# SHL Conversational Agent — Migration Plan

This document maps the existing **single-shot recommendation system** (Assignment v1) to the new **stateless conversational agent** (Assignment v2) while maximizing reuse of proven retrieval and ranking logic.

---

## 1. Reusable Components

### Fully reusable (no or minimal changes)

| Module | Path | Role in new system |
|--------|------|-------------------|
| **Catalog scraper** | `src/shl_recommender/crawler.py`, `scripts/crawl_shl_catalog.py` | Unchanged data ingestion; still the source of truth for SHL individual assessments |
| **Catalog storage** | `data/shl_individual_assessments.csv`, `.json` | Loaded by `load_catalog()` for grounding and validation |
| **Embeddings** | `src/shl_recommender/embedding.py` (`EmbeddingService`, `build_vector_store`) | Same MiniLM embeddings for retrieval |
| **Vector DB (Chroma)** | `chroma_db/`, `get_chroma_client()`, collection helpers | Same persistent index; no migration |
| **Ranking logic** | `RecommendationEngine._rank_candidates`, `_determine_result_count`, type extraction | Core recall@10 behavior preserved |
| **Type extraction** | `src/shl_recommender/type_extraction.py` (Gemini + heuristics) | Reused inside recommendation path for multi-domain balance |
| **Config** | `src/shl_recommender/config.py` | Extended with chat/agent settings |
| **Logging** | `src/shl_recommender/logging_setup.py` (Logfire) | Reused across agent + API |
| **Health check pattern** | `backend/main.py` `GET /health` | Kept as-is |
| **Deployment** | `render.yaml`, `uvicorn` entry | Same backend service; frontend calls `/chat` |
| **Batch inference** | `scripts/generate_predictions.py` | Still uses `/recommend` or engine directly for eval sets |
| **Build script** | `scripts/build_vector_store.py` | Unchanged |

### Partially reusable (extend, do not replace)

| Module | Path | Reuse | Extension needed |
|--------|------|-------|------------------|
| **Recommendation engine** | `src/shl_recommender/recommender.py` | `recommend()`, retrieval, ranking, catalog index | Optional `min_results`/`max_results` (1–10 for chat; 5–10 default for `/recommend`) |
| **Data models** | `src/shl_recommender/data_models.py` | `AssessmentMetadata`, `RecommendationItem` | Add `ChatMessage`, `ChatRequest`, `ChatResponse` |
| **FastAPI backend** | `backend/main.py` | `/health`, `/recommend`, engine singleton | Add stateless `POST /chat` |
| **Frontend** | `frontend/app.py` | Streamlit chat UI shell | POST `/chat` with full `messages` history instead of one-shot `/recommend` |
| **Gemini integration** | `type_extraction.py` patterns | Retry/logging/coalesce helpers | Optional grounded reply generation in agent layer |
| **Tests** | `tests/test_recommender.py` | Helper unit tests | Add agent module tests |

### Modules needing redesign (new layer, not rewrite of core)

| Area | Current | Target |
|------|---------|--------|
| **API contract** | `POST /recommend` `{ query }` | `POST /chat` `{ messages[] }` → `{ reply, recommendations, end_of_conversation }` |
| **Orchestration** | Single call: query → recommend | Multi-turn: analyze → clarify \| retrieve \| compare \| refuse |
| **State** | None (stateless per request) | Stateless server; client sends full history each turn |
| **Response shape** | `recommended_assessments` + `extracted_categories` | Strict chat JSON; categories folded into `reply` text when useful |

### Completely missing (to build)

- `POST /chat` endpoint and Pydantic chat schemas
- Conversation analyzer (intent + context sufficiency)
- Clarification engine
- Recommendation orchestrator (wraps engine + history merge + refinement)
- Comparison engine (catalog-only diffs)
- Refusal / prompt-injection guard
- Prompt templates module
- `evaluation/` utilities + conversation simulator
- Agent-focused unit tests
- Frontend wired to `/chat` with turn limit UX

---

## 2. Architecture Changes

### From recommender → conversational agent

```
[Client] --messages[]--> POST /chat (stateless)
              |
              v
       +--------------+
       |  ChatAgent   |  dialogue orchestration
       +--------------+
         |    |    |    |
    refuse  clarify  compare  recommend/refine
         |    |    |    |
         v    v    v    v
    Refusal  Clarif.  Compare  RecommendationOrchestrator
    Guard    Engine    Engine         |
                                       v
                            RecommendationEngine (existing)
                                       |
                            Chroma + Embeddings + Ranking
```

**What stays the same:** vector retrieval, type-aware ranking, catalog-backed `RecommendationItem` construction.

**What changes:** when to call retrieval, how user text is aggregated across turns, and how natural-language `reply` is produced from grounded facts only.

### Dialogue orchestration design

`ChatAgent.handle(messages)` pipeline:

1. **Validate** — max 8 user turns; reject empty latest user message
2. **Refusal guard** — off-topic / injection → polite refusal, `recommendations=[]`
3. **Conversation analyzer** — intent: `clarify`, `recommend`, `refine`, `compare`, `end`, `off_topic`
4. **Route:**
   - `clarify` → clarification question, no retrieval
   - `recommend` / `refine` → orchestrator → engine → validate URLs against catalog
   - `compare` → comparison engine using catalog metadata only
   - `end` → closing reply, `end_of_conversation=true`
5. **Reply builder** — template-first (fast, testable); optional Gemini polish with catalog context only

### State handling strategy

- **Server:** fully stateless; no session DB
- **Client:** sends full `messages: [{role, content}, ...]` each request
- **Derived state per request:** merged search query from user turns, last recommendation names (from prior assistant replies or re-retrieval), turn count

### Prompt strategy

- **Retrieval query:** concatenate user messages (weighted: latest turn highest) — no LLM required for retrieval
- **Type extraction:** existing Gemini/heuristic path on merged query
- **Reply generation:** `prompt_templates.py` + structured facts (assessment names, types, durations); LLM only adds prose when `GEMINI_API_KEY` set
- **Grounding rule:** every fact in `reply` about an assessment must come from `catalog_index` or current `RecommendationItem` list

### Retrieval strategy

- Unchanged: embed merged query → Chroma top-K → type rank → slice to N (1–10 for chat)
- **Refinement:** append refinement cues ("more personality", "shorter", "remote only") to merged query; optionally boost types via keyword map
- **Clarification:** skip retrieval until role, skills, or assessment goal detected

### Comparison pipeline

1. Extract assessment names from user text (fuzzy match against catalog names)
2. If fewer than 2 names, ask which assessments to compare (from last recommendations)
3. Build side-by-side comparison from `AssessmentMetadata` fields only (duration, types, remote, adaptive, description excerpt)

### Clarification pipeline

`ConversationAnalyzer` scores context dimensions: role, skills/competencies, seniority, assessment goals (cognitive/personality/technical).

If score below threshold → `ClarificationEngine` asks **one** targeted question (role vs skills vs level), `recommendations=[]`.

### Refusal pipeline

Keyword/heuristic classifier for: prompt injection, unrelated domains (weather, coding homework, general chat), requests to ignore system rules.

Response: fixed polite template; no retrieval; no recommendations.

---

## 3. Missing Features

Required by the new assignment but not present in v1:

| # | Feature | Status before migration |
|---|---------|-------------------------|
| 1 | Stateless `POST /chat` | Missing |
| 2 | Client-passed conversation history | Missing (UI had local history only) |
| 3 | Response schema `{ reply, recommendations, end_of_conversation }` | Missing |
| 4 | Clarification questions for vague queries | Missing |
| 5 | Recommend only after sufficient context | Missing |
| 6 | Recommendation refinement across turns | Missing |
| 7 | Assessment comparison (catalog-grounded) | Missing |
| 8 | Off-topic / prompt-injection refusal | Missing |
| 9 | Max 8-turn conversation enforcement | Missing |
| 10 | Recommendation count 1–10 (chat) | Partial (engine fixed 5–10) |
| 11 | `end_of_conversation` flag | Missing |
| 12 | Catalog-only recommendation validation | Partial (engine uses catalog; no explicit guard) |
| 13 | Agent module package + unit tests | Missing |
| 14 | Evaluation utilities + conversation simulator | Missing |
| 15 | Structured agent logging / retries | Partial (Logfire in engine only) |

**Already satisfied:** `GET /health`, catalog crawl, Chroma pipeline, FastAPI, deployment blueprint, basic Streamlit UI.

---

## 4. Refactor Plan

### Phase 1 — Planning & schemas (this PR)

1. Add `migration_plan.md` (this file)
2. Add `src/shl_recommender/agent/` package with analyzer, clarification, orchestrator, comparison, refusal, templates, `chat_agent.py`
3. Extend `data_models.py` with `ChatMessage`, `ChatRequest`, `ChatResponse`
4. Extend `config.py` (`max_conversation_turns`, `chat_min_recommendations`, etc.)

### Phase 2 — API & engine hooks

5. Add optional `min_results` / `max_results` to `RecommendationEngine.recommend()`
6. Implement `POST /chat` in `backend/main.py`; keep `/recommend` backward compatible
7. Catalog URL validation helper in orchestrator

### Phase 3 — Quality & UX

8. Update `frontend/app.py` to call `/chat` with session history
9. Add `tests/test_agent.py` for analyzer, refusal, comparison, orchestrator
10. Add `src/shl_recommender/evaluation/conversation_simulator.py`
11. Update `README.md` with API examples, sample conversations, deployment, TODOs

### Phase 4 — Optional enhancements (TODO)

12. Gemini-grounded reply polish with JSON schema enforcement
13. Offline eval harness against labelled train set in conversational mode
14. Latency benchmarks and caching hot catalog lookups

---

## 5. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Hallucinated assessment details** | Wrong facts in `reply` | Template replies from `AssessmentMetadata`; validate recommendation URLs against catalog |
| **Hallucinated assessments** | Non-SHL items in list | Filter recommendations through `catalog_index` by URL/entity_id |
| **Exceeding 8-turn limit** | Spec violation | Count user messages; return `end_of_conversation=true` with explanation |
| **Irrelevant recommendations on vague queries** | Poor UX / low precision | Clarification path before first retrieval |
| **Schema violations** | API eval failure | Pydantic `ChatResponse` model; FastAPI `response_model` |
| **Latency > 30s** | Timeout | Rule-based routing; avoid mandatory LLM on hot path; reuse warm engine singleton |
| **Prompt injection** | Unsafe behavior | `refusal_guard` + system templates; never execute user instructions as system rules |
| **Breaking Recall@10 on `/recommend`** | Regression on v1 metric | Default `recommend()` params unchanged (5–10); chat uses explicit 1–10 |
| **Refinement drift** | Worse rankings | Merge full user history; re-run full retrieval+rank, don’t patch stale lists blindly |
| **Comparison without names** | Empty compare | Prompt user to name assessments or pick from last recommendations |

---

## 6. Target Folder Structure (after migration)

```
backend/
  main.py                 # /health, /recommend, /chat
frontend/
  app.py                  # Streamlit → /chat
scripts/
  crawl_shl_catalog.py
  build_vector_store.py
  generate_predictions.py
  check_gemini_agent.py
src/shl_recommender/
  agent/
    __init__.py
    chat_agent.py
    conversation_analyzer.py
    clarification_engine.py
    recommendation_orchestrator.py
    comparison_engine.py
    refusal_guard.py
    prompt_templates.py
    llm_reply.py          # optional Gemini replies
  evaluation/
    __init__.py
    conversation_simulator.py
  config.py
  crawler.py
  data_models.py
  embedding.py
  recommender.py
  type_extraction.py
  logging_setup.py
tests/
  test_recommender.py
  test_agent.py
migration_plan.md
```

---

## 7. API Contract Summary

**POST /chat**

Request:
```json
{
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Response:
```json
{
  "reply": "...",
  "recommendations": [ /* RecommendationItem[], 0–10 items */ ],
  "end_of_conversation": false
}
```

**POST /recommend** — unchanged for batch eval and backward compatibility.
