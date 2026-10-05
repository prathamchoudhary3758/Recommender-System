"""
Day 1 Verification Test: Architecture & Scaffolding Smoke Tests.
"""

from pathlib import Path
from config.settings import get_default_config, ProjectConfig
import src
import src.data
import src.features
import src.retrieval
import src.ranking
import src.evaluation
import src.ab_testing


def test_package_imports():
    """Verify that all core packages import cleanly."""
    assert src.__version__ == "0.1.0"
    assert src.data is not None
    assert src.features is not None
    assert src.retrieval is not None
    assert src.ranking is not None
    assert src.evaluation is not None
    assert src.ab_testing is not None


def test_default_config():
    """Verify that configuration initializes with standard default parameters."""
    config = get_default_config()
    assert isinstance(config, ProjectConfig)
    
    # Check data defaults
    assert config.data.num_users == 10_000
    assert config.data.num_items == 2_000
    assert config.data.train_end_day == 14
    
    # Check retrieval defaults
    assert config.retrieval.embedding_dim == 64
    assert config.retrieval.candidate_k == 100
    
    # Check ranking defaults
    assert config.ranking.top_k_recommendations == 5
    assert config.ranking.model_type == "lightgbm"
    
    # Check A/B testing defaults
    assert config.ab_testing.traffic_split_control == 0.50
    assert config.ab_testing.traffic_split_treatment == 0.50
    assert config.ab_testing.alpha == 0.05
    assert config.ab_testing.target_power == 0.80


def test_project_directories():
    """Verify that core expected project directories exist."""
    config = get_default_config()
    root = config.project_root
    assert (root / "src").is_dir()
    assert (root / "config").is_dir()
    assert (root / "tests").is_dir()
    assert (root / "notebooks").is_dir()
    assert (root / "artifacts").is_dir()
