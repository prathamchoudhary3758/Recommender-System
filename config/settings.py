"""
Central Configuration for the Two-Stage Recommender and A/B Testing Framework.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class DataConfig:
    """Settings for synthetic data generation and temporal splitting."""
    num_users: int = 10_000
    num_items: int = 2_000
    num_categories: int = 15
    simulation_days: int = 28
    interactions_per_user_mean: int = 35
    interactions_per_user_std: int = 10
    random_seed: int = 42
    
    # Temporal splitting: Days 1-14 train, 15-21 val, 22-28 test
    train_end_day: int = 14
    val_end_day: int = 21
    test_end_day: int = 28


@dataclass
class RetrievalConfig:
    """Settings for vector embedding generation and Faiss ANN search."""
    embedding_dim: int = 64
    candidate_k: int = 100  # Number of candidate items retrieved from catalog
    index_type: str = "IndexFlatIP"  # "IndexFlatIP", "IndexIVFFlat", or "IndexHNSWFlat"
    ivf_nlist: int = 50
    hnsw_m: int = 32
    metric: str = "inner_product"  # vectors are L2-normalized


@dataclass
class RankingConfig:
    """Settings for LightGBM / XGBoost ranking model."""
    top_k_recommendations: int = 5  # Final items displayed to user
    model_type: str = "lightgbm"  # "lightgbm" or "xgboost"
    learning_rate: float = 0.05
    n_estimators: int = 200
    max_depth: int = 6
    num_leaves: int = 31
    random_seed: int = 42


@dataclass
class ABTestingConfig:
    """Settings for experimentation, traffic routing, SRM, and CUPED."""
    experiment_name: str = "two_stage_recsys_v1"
    salt: str = "experiment_salt_2026_q1"
    traffic_split_control: float = 0.50
    traffic_split_treatment: float = 0.50
    alpha: float = 0.05  # Significance level
    target_power: float = 0.80  # Target statistical power (1 - beta)
    srm_threshold_p_value: float = 0.01  # Chi-square p-value threshold for SRM alert


@dataclass
class ProjectConfig:
    """Master project configuration combining all module settings."""
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    ab_testing: ABTestingConfig = field(default_factory=ABTestingConfig)

    @property
    def raw_data_dir(self) -> Path:
        return self.project_root / "data" / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.project_root / "data" / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"


def get_default_config() -> ProjectConfig:
    """Returns the default project configuration."""
    return ProjectConfig()
