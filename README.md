# Two-Stage Recommender System & A/B Testing Simulation Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Project Status: Active](https://img.shields.io/badge/status-active%20development-brightgreen.svg)](#)

A production-grade, two-stage recommender system combining **Candidate Retrieval (Faiss ANN vector search)** and **Re-Ranking (LightGBM/XGBoost)**, paired with an end-to-end **A/B Testing Simulation Engine** featuring **Sample Ratio Mismatch (SRM) detection** and **CUPED variance reduction**.

---

## 🎯 Architecture Overview

```
User Request
     │
     ▼
[Stage 1: Retrieval (Faiss)]       ──> Filters 100,000+ items to Top-100 candidates (<5ms)
     │
     ▼
[Stage 2: Ranking (LightGBM)]      ──> Scores candidates with rich features to pick Top-5
     │
     ▼
[A/B Experimentation Engine]       ──> Deterministic Salted Hash Routing (Control vs Treatment)
     │
     ├──> SRM Detector             ──> Chi-Square (χ²) goodness-of-fit to catch telemetry bugs
     ├──> CUPED Variance Reduction ──> 30-50% variance shrinkage using pre-experiment covariates
     └──> Statistical Power Suite  ──> Welch's t-test, Bootstrap CIs & MDE Power curves
```

---

## 📅 14-Day Implementation Roadmap

| Day | Focus Area | Status | Deliverable |
| :---: | :--- | :---: | :--- |
| **Day 1** | **Scaffolding & Architecture** | ✅ Completed | Modular package layout, config engine, test suite |
| **Day 2** | **Data Generation & Splitting** | ⏳ Up Next | Synthetic interaction generator, temporal splitter |
| **Day 3** | **Feature Store & Extraction** | 📅 Scheduled | User/Item engagement profiles & affinity metrics |
| **Day 4** | **Embedding Representation** | 📅 Scheduled | Matrix Factorization / Two-Tower user & item vectors |
| **Day 5** | **Faiss ANN Vector Index** | 📅 Scheduled | IndexFlatIP / IVFFlat retrieval & latency benchmark |
| **Day 6** | **LightGBM / XGBoost Ranker** | 📅 Scheduled | LambdaMART / Pointwise ranking model |
| **Day 7** | **Offline Evaluation Suite** | 📅 Scheduled | Recall@K, Precision@K, NDCG@K, MRR |
| **Day 8** | **Traffic Splitting Engine** | 📅 Scheduled | Deterministic hash routing & synthetic user simulator |
| **Day 9** | **Sample Ratio Mismatch (SRM)**| 📅 Scheduled | Chi-Square Goodness-of-Fit test & automated alerts |
| **Day 10**| **CUPED Variance Reduction** | 📅 Scheduled | Covariate adjustment ($\theta$) & variance shrinkage |
| **Day 11**| **Power & Hypothesis Testing** | 📅 Scheduled | Welch's t-test, Bootstrap CIs, MDE power curves |
| **Day 12**| **End-to-End Simulation** | 📅 Scheduled | 14-day A/B experiment simulation execution |
| **Day 13**| **Interactive Dashboard** | 📅 Scheduled | Streamlit analytics & recommendation visualizer |
| **Day 14**| **Release & Portfolio Wrap-up** | 📅 Scheduled | System benchmarks, full test suite & v1.0.0 release |

---

## 🛠️ Tech Stack

- **Core & Numerics:** Python 3.10+, NumPy, Pandas
- **Vector Retrieval:** Faiss (Facebook AI Similarity Search)
- **Machine Learning / Ranking:** LightGBM, XGBoost, Scikit-learn
- **Causal Inference & Statistics:** SciPy, Statsmodels
- **Testing & Quality Assurance:** Pytest
- **Dashboard & UI:** Streamlit

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/prathamchoudhary3758/Recommender-System.git
cd Recommender-System
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run tests
```bash
python -c "import sys; sys.path.insert(0, '.'); from tests.test_scaffolding import test_package_imports, test_default_config, test_project_directories; test_package_imports(); test_default_config(); test_project_directories(); print('ALL TESTS PASSED!')"
```

---

## 📖 Documentation
- [Roadmap Details](implementation.txt)
- [Conceptual Guide & Interview FAQ](project_explanation.txt)
- [Changelog](CHANGELOG.md)

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
