# SHL Assessment Recommendation System: Solution Approach

## Problem Statement

Hiring teams struggle to navigate SHL's extensive catalog of individual assessments when matching candidates to job roles. The challenge is to build a GenAI-powered recommendation system that accepts free-form job descriptions and returns the most relevant SHL assessments, prioritizing quality matches while maintaining diversity across technical and behavioral evaluation dimensions.

## Solution Architecture

### Data Pipeline

**Catalog Acquisition:** A custom web crawler (`crawl_shl_catalog.py`) scrapes SHL's public catalog, extracting structured metadata for each individual assessment including name, URL, assessment type codes (A-P), job levels, languages, duration, and descriptive text. Pre-packaged bundles and simulation-only products are filtered out, resulting in a curated CSV of single assessments suitable for embedding.

**Embedding Generation:** The system uses `sentence-transformers/all-MiniLM-L6-v2`, a lightweight 80MB model that balances semantic understanding with inference speed. Each assessment is represented by concatenating its name, description, job levels, languages, and type codes into a single text document. These documents are embedded and stored in ChromaDB, a persistent vector database enabling sub-second cosine similarity searches across the catalog.

### Recommendation Engine

**Retrieval Strategy:** User queries undergo three-stage processing: (1) Semantic embedding using the same MiniLM model, (2) Vector similarity search retrieving the top-20 candidates from ChromaDB, (3) Re-ranking using a hybrid scoring approach.

**Hybrid Ranking Logic:** Initially, the system ranked candidates purely by embedding similarity. This baseline approach worked well for explicit queries but struggled with implicit requirements (e.g., "software engineer" should surface both technical knowledge tests and behavioral assessments). The optimized solution introduces LLM-powered type extraction where a Gemini Flash agent analyzes the query to identify relevant assessment categories (K for Knowledge & Skills, P for Personality & Behaviour, etc.). Candidates matching these extracted types are prioritized, with embedding similarity serving as the tiebreaker within each priority tier.

**Dynamic Result Count:** Rather than always returning 10 results, the engine adapts output size (5-10 recommendations) based on match confidence. Queries with strong type-category matches receive fuller result sets; ambiguous queries return smaller, higher-confidence sets.

---

## Performance Optimization Journey

### Initial Baseline (Pure Vector Search)

**Approach:** The first implementation used straightforward cosine similarity ranking. The top-20 candidates from ChromaDB were sorted by embedding distance, and the top-10 were returned directly.

**Limitations Identified:**
- **Category Imbalance:** Technical roles often returned predominantly Knowledge & Skills assessments while missing critical behavioral evaluations
- **Query Ambiguity:** Generic queries like "manager" yielded inconsistent results across runs
- **Fixed Output Size:** Always returning 10 results diluted precision when high-confidence matches were scarce

### Optimization Phase 1: Enhanced Metadata Embedding

**Changes:**
- Enriched the combined text representation to include job level descriptors and language support metadata
- Added assessment type labels explicitly to the embedding text rather than storing them only as metadata
- Expanded the candidate pool from 15 to 20 to improve the ranking stage's raw material

**Impact:** Improved semantic matching for queries mentioning specific seniority levels or language requirements. Queries like "entry-level French-speaking sales rep" began surfacing appropriate language-specific assessments.

### Optimization Phase 2: LLM-Powered Type Extraction

**Changes:**
- Integrated Pydantic AI with Gemini 2.5 Flash as a zero-shot classifier to extract assessment type codes from queries
- The LLM agent receives a system prompt describing SHL's eight assessment categories and returns relevant codes for each query
- Re-ranking algorithm prioritizes candidates whose type codes overlap with LLM-extracted types before applying similarity sorting

**Impact:** Dramatic improvement in category diversity. Queries for complex roles (e.g., "leadership position requiring analytical skills") now correctly surface balanced mixes of Knowledge (K), Personality (P), and Competency (C) assessments. The LLM acts as a query understanding layer that makes implicit requirements explicit.

### Optimization Phase 3: Adaptive Result Count

**Changes:**
- Introduced dynamic thresholding logic examining the number of candidates matching extracted types
- Result count scales from 5 (low-confidence queries) to 10 (high-confidence queries) based on type-match density in the top-ranked candidates
- Prevents "padding" with low-relevance results when the query is ambiguous

**Impact:** Improved precision by reducing noise in recommendation lists. Users see tighter, more relevant sets for niche roles while still receiving comprehensive coverage for broad queries.

### Optimization Phase 4: Model Selection & Infrastructure

**Changes:**
- Selected `all-MiniLM-L6-v2` over larger models like `all-mpnet-base-v2` for its 3x faster inference with minimal quality tradeoff
- Configured ChromaDB with persistent storage rather than in-memory mode, enabling instant cold-start initialization
- Implemented connection pooling and lazy initialization patterns to reduce API response latency

**Impact:** API response times dropped below 200ms for most queries. Deployment on free-tier cloud infrastructure (Render) became viable due to reduced memory footprint.

## Technical Highlights

**Pydantic AI Agent:** The type extraction agent uses structured output parsing, ensuring the LLM returns only valid assessment codes (A-P). Retry logic handles transient API failures gracefully.

**Scalability Considerations:** The vector store architecture separates embedding generation (offline batch process) from retrieval (online serving), allowing the catalog to scale to thousands of assessments without impacting API latency.

**Production Readiness:** FastAPI backend exposes health checks and structured logging via Logfire. Environment variables enable configuration overrides for different deployment contexts. A Streamlit frontend provides immediate visual feedback for development and stakeholder demos.

## Key Learnings

**Semantic search alone is insufficient for multi-dimensional matching problems.** Embedding similarity excels at capturing semantic relatedness but cannot enforce business rules like category diversity. The hybrid approach—LLM-driven intent classification feeding into vector ranking—combines the strengths of both paradigms.

**Query understanding is critical.** Users express requirements implicitly (e.g., "team lead" implies both technical expertise and interpersonal skills). Surfacing these latent dimensions through LLM analysis before retrieval significantly improved perceived recommendation quality.

**Model efficiency matters for real-world deployment.** While larger embedding models offered marginal quality gains during offline evaluation, the chosen lightweight model enabled deployment on resource-constrained infrastructure without sacrificing user experience.
