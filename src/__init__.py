"""
Two-Stage Recommender System & A/B Testing Framework.

Modules:
- data: Dataset generation, loading, and temporal splitting.
- features: Feature engineering and enrichment for users and items.
- retrieval: Embedding generation and Faiss ANN candidate search.
- ranking: LightGBM / XGBoost ranking models.
- evaluation: Offline ranking metrics (Recall@K, NDCG@K, MRR).
- ab_testing: Traffic splitting, SRM detection, CUPED variance reduction, and power analysis.
"""

__version__ = "0.1.0"
