"""
Unit Test Suite for Data Generation, Temporal Splitting, and Negative Sampling.
Supports execution via pytest or standard python unittest.
"""

import unittest
import numpy as np
import pandas as pd

from config.settings import DataConfig
from src.data.generator import SyntheticDataGenerator
from src.data.splitter import TemporalSplitter
from src.data.sampler import NegativeSampler


class TestDataPipeline(unittest.TestCase):
    """Test suite covering the end-to-end data engine."""

    @classmethod
    def setUpClass(cls):
        cls.config = DataConfig(
            num_users=200,
            num_items=100,
            num_categories=5,
            simulation_days=28,
            train_end_day=14,
            val_end_day=21,
            test_end_day=28,
            random_seed=42,
        )
        cls.generator = SyntheticDataGenerator(cls.config)
        cls.users_df, cls.items_df, cls.interactions_df = cls.generator.generate_all()

    def test_generator_structure_and_types(self):
        """Verify that generated users, items, and interactions have the correct schemas."""
        users_df = self.users_df
        items_df = self.items_df
        interactions_df = self.interactions_df

        # Check users schema
        self.assertEqual(len(users_df), self.config.num_users)
        for col in ["user_id", "activity_tier", "age_group", "primary_category", "base_ctr"]:
            self.assertIn(col, users_df.columns)
        self.assertTrue(users_df["base_ctr"].between(0.0, 1.0).all())

        # Check items schema
        self.assertEqual(len(items_df), self.config.num_items)
        for col in ["item_id", "category_id", "popularity_score", "quality_score"]:
            self.assertIn(col, items_df.columns)
        self.assertTrue(items_df["popularity_score"].between(0.0, 1.0).all())

        # Check interactions schema
        self.assertGreater(len(interactions_df), 0)
        for col in [
            "interaction_id",
            "user_id",
            "item_id",
            "timestamp",
            "day",
            "category_match",
            "clicked",
            "dwell_time_secs",
            "conversion",
        ]:
            self.assertIn(col, interactions_df.columns)

        # Verify timestamps are strictly monotonically increasing
        diffs = interactions_df["timestamp"].diff().dropna()
        self.assertTrue((diffs >= 0).all())

    def test_temporal_splitting_strictness(self):
        """Verify that train, validation, and test splits strictly follow temporal day boundaries."""
        splitter = TemporalSplitter(self.config)
        train_df, val_df, test_df = splitter.split(self.interactions_df)

        # Verify non-empty and non-lossy
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(val_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertEqual(len(train_df) + len(val_df) + len(test_df), len(self.interactions_df))

        # Check boundary conditions (no future lookahead leakage)
        self.assertLessEqual(train_df["day"].max(), self.config.train_end_day)
        self.assertGreater(val_df["day"].min(), self.config.train_end_day)
        self.assertLessEqual(val_df["day"].max(), self.config.val_end_day)
        self.assertGreater(test_df["day"].min(), self.config.val_end_day)

        # Check summary metrics
        summary = splitter.get_split_summary(train_df, val_df, test_df)
        self.assertEqual(summary["total_events"], len(self.interactions_df))
        self.assertTrue(0.0 <= summary["val_user_coverage"] <= 1.0)
        self.assertTrue(0.0 <= summary["test_item_coverage"] <= 1.0)

    def test_negative_sampler_unobserved_constraint(self):
        """Verify negative sampler never returns items that the user already observed."""
        items = [f"i_{i:04d}" for i in range(50)]
        sampler = NegativeSampler(all_item_ids=items, random_seed=42)

        observed = {f"i_{i:04d}" for i in range(10)}  # User has seen first 10 items
        negs = sampler.sample_negatives_for_user(observed_items=observed, num_negatives=5)

        self.assertEqual(len(negs), 5)
        self.assertEqual(len(set(negs)), 5)  # No duplicates
        self.assertEqual(len(set(negs) & observed), 0)  # Completely disjoint from observed items

    def test_negative_sampler_popularity_strategy(self):
        """Verify popularity-weighted negative sampling works without error."""
        items = [f"i_{i:04d}" for i in range(30)]
        weights = np.linspace(0.1, 10.0, 30)
        sampler = NegativeSampler(all_item_ids=items, item_popularity_weights=weights, random_seed=42)

        observed = {"i_0000", "i_0001"}
        negs = sampler.sample_negatives_for_user(observed_items=observed, num_negatives=4, strategy="popularity")

        self.assertEqual(len(negs), 4)
        self.assertEqual(len(set(negs) & observed), 0)

    def test_negative_augmentation(self):
        """Verify augment_interactions_with_negatives expands dataset with clicked=0 records."""
        sampler = NegativeSampler(all_item_ids=self.items_df["item_id"].tolist(), random_seed=42)

        sample_interactions = self.interactions_df.head(100)
        positives_count = (sample_interactions["clicked"] == 1).sum()

        augmented = sampler.augment_interactions_with_negatives(
            sample_interactions, num_negatives_per_positive=2
        )

        if positives_count > 0:
            self.assertGreater(len(augmented), len(sample_interactions))
        diffs = augmented["timestamp"].diff().dropna()
        self.assertTrue((diffs >= 0).all())


if __name__ == "__main__":
    unittest.main()
