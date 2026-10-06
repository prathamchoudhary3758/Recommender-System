# Project Changelog

All notable changes and daily progress for the Two-Stage Recommender System & A/B Testing Framework will be documented in this file.

## [Day 2] - 2026-10-06

### Added
- **Synthetic Data Generator (`src/data/generator.py`):**
  - Generates realistic user cohorts with latent preference categories, activity tiers (casual, medium, heavy), and base CTRs.
  - Builds item catalog with Pareto power-law popularity distribution and intrinsic quality scores.
  - Simulates chronological interaction telemetry (clicks, impressions, dwell times, conversions).
- **Temporal Train/Validation/Test Splitter (`src/data/splitter.py`):**
  - Enforces strict temporal boundaries (Days 1-14 train, 15-21 val, 22-28 test) eliminating future lookahead data leakage.
  - Computes user and item coverage and cold-start diagnostics across temporal partitions.
- **Negative Sampling Engine (`src/data/sampler.py`):**
  - Implements uniform random negative sampling and popularity-biased (hard) negative sampling.
  - Provides data augmentation pipeline for candidate retrieval and ranker training.
- **Unit Test Suite (`tests/test_data.py`):**
  - Automated tests covering schema validity, strict temporal boundaries, negative disjointness, and dataset augmentation.
- **CLI Generation Script (`scripts/generate_data.py`):**
  - Command-line tool to generate and export processed Parquet datasets.

### Commits
- `feat(data): implement synthetic interaction dataset generator`
- `feat(data): implement temporal train-validation-test splitter`
- `feat(data): implement negative sampling strategy for training`
- `test(data): add automated unit test suite for data pipeline`
- `feat(cli): add dataset generation script with temporal split and negative sampling`

---

## [Day 1] - 2026-10-05

### Added
- **Project Repository Setup:** Initialized Git version control and established `.gitignore` rules for Python, virtual environments, unit tests, and large model/data artifacts.
- **Modular Directory Architecture:**
  - `src/data/`: Data synthesis, ingestion, and temporal splitting modules.
  - `src/features/`: Feature store, feature transformation, and affinity matrices.
  - `src/retrieval/`: Embedding generation and Faiss ANN index candidate retrieval.
  - `src/ranking/`: LightGBM / XGBoost learning-to-rank pipeline.
  - `src/evaluation/`: Offline ranking metrics (Recall@K, NDCG@K, MRR).
  - `src/ab_testing/`: Experimentation platform, deterministic hash splitting, SRM detection, and CUPED variance reduction.
  - `config/`: Centralized dataclass configurations and hyperparameters (`config/settings.py`).
  - `tests/`: Automated unit and integration test suite.
  - `notebooks/`: Exploratory analysis and visualization workspaces.
- **Dependency Management:** Specified locked core packages in `requirements.txt` (`faiss-cpu`, `lightgbm`, `xgboost`, `scipy`, `statsmodels`, `pytest`, `streamlit`).
- **Documentation Baseline:**
  - `implementation.txt`: Comprehensive 14-day execution plan and daily commit guidelines.
  - `project_explanation.txt`: Plain-language guide, real-world analogies, tech stack justification, glossary of terms, and interview elevator pitch.
- **Verification Tests:** Created `tests/test_scaffolding.py` validating package imports, directory health, and default configuration values.

### Commit
- `feat: initialize project scaffolding, dependencies, and architecture blueprint`
