## SHL Assessment Recommendation System  Approach Summary

### 1. Problem Understanding
- Hiring teams struggle to map open role descriptions to relevant SHL individual assessments.
- Users provide either a natural-language query, full job description text, or a URL containing a JD.
- Deliverables include: a production-ready API, a lightweight UI, batch inference for the evaluation set, and supporting documentation.

### 2. Data Acquisition & Curation
- **Crawler (scripts/crawl_shl_catalog.py)** scrapes the SHL product catalog and filters to Individual Test Solutions, removing pre-packaged bundles and simulations (S type).
- Extracted fields: identifiers, name, URL, assessment type(s), job levels, languages, assessment length, remote/adaptive flags, and free-text descriptions.
- Saved to data/shl_individual_assessments.csv; this CSV acts as the single source for downstream embedding, retrieval, and offline experimentation.
- The crawler is idempotent and records defensive logs for observability.

### 3. Vector Store Construction
- scripts/build_vector_store.py loads the curated CSV and builds embeddings using sentence-transformers/all-MiniLM-L6-v2.
- Documents consist of the assessment name + rich metadata joined into a single text block (AssessmentMetadata.combined_text).
- Embeddings persisted in a ChromaDB collection (chroma_db/), enabling fast cosine similarity search across thousands of items.
- Metadata stored alongside each vector includes canonical URL, normalized assessment type codes, and entity identifier for quick access during recommendation.

### 4. Recommendation Engine
- Implemented in src/shl_recommender/recommender.py and instantiated by both FastAPI backend (backend/main.py) and CLI tooling.
- **Retrieval:** Query text is embedded with the same model and top candidates are pulled from Chroma using cosine similarity.
- **Ranking:** Candidates are ranked by embedding similarity scores (higher is better).
- **Balanced Mixing:** The engine can enforce a minimum count of both technical (K) and behavioral (P) assessments when enough candidates exist, before filling the rest by similarity score.

### 5. API Surface
- **Backend:** FastAPI app exposes:
  - GET /health  { "status": "ok" }
  - POST /recommend  RecommendationResponse containing up to 10 { name, url } pairs.
- **Frontend:** Streamlit (frontend/app.py) UI for quick interactive validation. An environment variable RECOMMENDER_API_URL allows pointing to remote deployments.
- **Batch Inference:** scripts/generate_predictions.py ingests CSV/Excel query lists and emits a submission-ready CSV (Query,Assessment_url) as required in Appendix 3.

### 6. Evaluation Strategy
- Primary metric: **Mean Recall@10** on labeled queries.
- Evaluation loop:
  1. Generate predictions for the labeled validation set.
  2. Compare retrieved URLs against ground-truth lists to compute Recall@K.
  3. Iterate on candidate pool size and balance threshold (min_category_mix) to optimize results.

### 7. Deployment Considerations
- **Backend** can be containerized via Uvicorn/Gunicorn and deployed on any managed platform (Railway, Render, Fly.io). Environment variables configure secrets and model overrides.
- **Frontend** deployable via Streamlit Community Cloud or as a static build served behind the API.
- Provide three URLs in the submission form: API endpoint, GitHub repo, and hosted UI.

### 8. Observability & Logging
- logging_setup.configure_logging centralizes Logfire integration; without credentials, logging falls back to standard output.
- Key actions (crawl events, embedding upserts, recommendation retrieval counts) are instrumented for traceability.
- ChromaDB persistence allows quick restarts without recomputing embeddings.

### 9. Reproducibility Checklist
1. python -m venv .venv && .venv\Scripts\activate
2. pip install -r requirements.txt
3. python scripts/crawl_shl_catalog.py
4. python scripts/build_vector_store.py --reset
5. uvicorn backend.main:app --host 0.0.0.0 --port 8000
6. (Optional) streamlit run frontend/app.py
7. python scripts/generate_predictions.py --input data/unlabeled_queries.xlsx --output data/predictions.csv

This document, together with the README, satisfies the two-page narrative requirement by outlining approach, infrastructure, and evaluation learnings for reviewers.
