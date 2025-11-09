# SHL Assessment Recommendation System

This repository implements a GenAI-powered recommendation system that maps free-form job descriptions to the most relevant SHL individual assessment tests.

## Features

- **Catalog crawler** – scrapes SHL's public catalog for individual assessment metadata and saves it to CSV.
- **Embedding pipeline** – builds a ChromaDB vector store using `all-MiniLM-L6-v2` embeddings.
- **Vector similarity ranking** – retrieves top candidates via vector similarity and ranks them by embedding similarity.
- **Balanced recommendation rule** – ensures technical (`K`) and behavioral (`P`) assessments are both represented when required.
- **FastAPI backend** – exposes `/health` and `/recommend` endpoints.
- **Streamlit frontend** – minimal UI to submit job descriptions and view recommendations.
- **Batch inference script** – generates CSV predictions for an unlabeled dataset.

## Project structure

- `backend/` – FastAPI application entry point.
- `frontend/` – Streamlit application.
- `scripts/` – CLI utilities for crawling, embedding, and batch inference.
- `src/shl_recommender/` – shared Python package with core logic.
- `data/` – default location for crawled catalog CSVs and generated predictions.
- `chroma_db/` – on-disk persistence for the Chroma vector store.

## Environment setup

1. **Create a virtual environment** (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables** by creating a `.env` file at the project root. Optional keys:

   ```bash
   LOGFIRE_API_KEY=your_logfire_api_key  # optional; omit or leave blank to disable remote logging
   ```

   Optional overrides:

   ```bash
   CHROMA_PATH=chroma_db
   EMBEDDINGS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
   ```

## Workflow

### 1. Crawl the catalog

```bash
python scripts/crawl_shl_catalog.py --output data/shl_individual_assessments.csv
```

This fetches individual assessment metadata, filters out pre-packaged job solutions, and stores the CSV locally.

### 2. Build embeddings and vector store

```bash
python scripts/build_vector_store.py --reset
```

This reads the CSV, generates embeddings with `all-MiniLM-L6-v2`, and upserts them into a persistent Chroma collection.

### 3. Run the FastAPI backend

```bash
uvicorn backend.main:app --reload --port 8000
```

Endpoints:

- `GET /health` → `{ "status": "ok" }`
- `POST /recommend` with payload `{ "query": "..." }` → returns top assessments.

### 4. Launch the Streamlit frontend

In a separate shell:

```bash
export RECOMMENDER_API_URL=http://localhost:8000  # optional
streamlit run frontend/app.py
```

Enter a job description and review the recommended assessments in the table output.

### 5. Batch predictions for test datasets

```bash
python scripts/generate_predictions.py --input data/unlabeled_queries.xlsx --column query --output data/predictions.csv
```

The script reads the specified column, runs recommendations for each row, and writes JSON-formatted assessment lists to the output CSV.

## Logs and monitoring

Logging is instrumented with [Logfire](https://logfire.pydantic.dev/). Supply `LOGFIRE_API_KEY` in `.env` to stream structured logs. Without a key, logging remains local.

## Notes

- The system uses vector similarity ranking based on embedding similarity scores.
- The balanced recommendation rule requires at least three `K` and three `P` assessments in the candidate pool to trigger; otherwise, the system returns the highest-ranking results.
- All scripts print progress to the terminal for traceability.

